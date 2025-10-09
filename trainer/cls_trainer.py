# Trainer for Cls-to-Img MaskGIT
import copy

import os
import time
from datetime import datetime

from tqdm import tqdm
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from models import encoder_decoder
from sampler import confidence_sampler
from trainer.abstract_trainer import Trainer

from models.vq_model import VQ_models
from models.transformer import Transformer
from sampler.confidence_sampler_pgm import ConfidenceSampler
from sampler.halton_sampler import HaltonSampler
from sampler.halton_sampler_pgm import HaltonSamplerPGM
from dataset.dataloader import get_data
from utils.viz import reconstruction
from utils.masking_scheduler import get_mask_code
import omegaconf
from models import encoder_decoder
from huggingface_hub import hf_hub_download

class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialize model, optimizer, loss function, and data loaders."""
        super().__init__(args)
        if not args.disable_wandb:
            print(f"Init Cls-to-Img Maskgit on [GPU{args.global_rank}]")
        

        self.args = args                                                        # Main argument see main.py
        self.input_size = self.args.img_size // self.args.f_factor              # Define input size for transformer

        # Load transformer (Masked Bidirectional Transformer) and VQGAN models
        if not args.disable_wandb:
            print('loading model', args.model)
        self.vit = self.get_network(args.model['name'])                                      # Load Masked Bidirectional Transformer
        self.ae = self.get_network("vqgan-llama")                               # Load VQGAN

        # Initialize Exponential Moving Average (EMA) model if enabled
        if self.args.use_ema:
            self.ema = self.get_network("ema")

        # Define loss function and optimizer
        self.criterion = self.get_loss("cross_entropy", ignore_index=-100)      # Get cross-entropy loss
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.999), weight_decay=0.03, resume_checkpoint=self.args.resume_checkpoint)  # Get AdamW Optimizer

        # Set up automatic mixed precision for training efficiency
        # print('echo', print(self.args.device))
        if self.args.device != 'cpu' and self.args.dtype == "bfloat16":
            self.autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        # Load training and testing data if specified
        if args.data_folder != "":
            self.train_data, self.test_data = get_data(
                args.data, args.img_size, args.data_folder, args.bsize,
                args.num_workers, args.is_multi_gpus, args.seed, args=self.args
            )

        # Select the appropriate sampling method (Halton or Confidence-based)
        if args.sampler == "halton":
            if args.model['name'] == 'encoder-decoder':
                self.sampler = HaltonSamplerPGM(sm_temp_min=self.args.sm_temp_min, sm_temp_max=self.args.sm_temp,
                                            temp_pow=1, temp_warmup=self.args.temp_warmup, w=self.args.cfg_w,
                                            sched_pow=self.args.sched_pow, step=self.args.step,
                                            randomize=self.args.randomize, top_k=self.args.top_k)

            elif args.model['name'] == 'maskgit':
                self.sampler = HaltonSampler(sm_temp_min=self.args.sm_temp_min, sm_temp_max=self.args.sm_temp,
                                            temp_pow=1, temp_warmup=self.args.temp_warmup, w=self.args.cfg_w,
                                            sched_pow=self.args.sched_pow, step=self.args.step,
                                            randomize=self.args.randomize, top_k=self.args.top_k)
            else:
                raise ValueError
        elif args.sampler == "confidence":
            if args.model['name'] == "encoder-decoder":
                self.sampler = ConfidenceSampler(sm_temp=self.args.sm_temp, w=self.args.cfg_w, r_temp=self.args.r_temp,
                                                sched_mode=self.args.sched_mode, step=self.args.step)
            elif args.model['name'] == "maskgit":
                self.sampler = confidence_sampler.ConfidenceSampler(sm_temp=self.args.sm_temp, w=self.args.cfg_w, r_temp=self.args.r_temp,
                                                sched_mode=self.args.sched_mode, step=self.args.step)
        else:
            self.sampler = None
        if self.args.is_master:
            # self.trainner.logger.log_text({f"text/Parameters": args_str}, step=self.args.iter)
            if not self.args.disable_wandb:
                print("Parameters: ", self.args)
            args_str = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
           
            self.writer.log({"train/Parameters": args_str, 
                              'step': self.args.iter})
            self.log_add_txt("Parameters", args_str, self.args.iter)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder|ema, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "maskgit":
            if self.args.is_master:
                if not os.path.exists(self.args.saved_folder):
                    os.makedirs(self.args.saved_folder)
                    print(f"Folder created: {self.args.saved_folder}")

            # Define transformer architecture parameters
            hidden_dim, depth, heads = self.transformer_size(self.args.vit_size)
            model = Transformer(
                input_size=self.input_size, nclass=self.args.nb_class, c=hidden_dim,
                hidden_dim=hidden_dim, codebook_size=self.args.codebook_size,
                depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=self.args.dropout,
                register=self.args.register, proj=self.args.proj
            )

            # Load model checkpoint if resuming training
            if self.args.resume:
                if self.args.resume_checkpoint is None:
                    fname = 'current.pth'
                else:
                    fname = self.args.resume_checkpoint

                ckpt = os.path.join(self.args.saved_folder, fname)
                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=True)
                opt_state = checkpoint['optimizer_state_dict']
                self.args.iter = checkpoint['iter']
                self.args.global_epoch = checkpoint['global_epoch']
                if self.args.is_master:
                    print("Load ckpt from:", ckpt)
                    print("Number of iteration(s):", self.args.iter)
                    
        elif archi == "encoder-decoder":
            if self.args.is_master:
                if not os.path.exists(self.args.saved_folder):
                    os.makedirs(self.args.saved_folder)
                    print(f"Folder created: {self.args.saved_folder}")
            model = encoder_decoder.PartitionDIT(self.args, vocab_size=self.args.codebook_size + 2)
            
            # Load model checkpoint if resuming training
            if not self.args.resume_checkpoint:
                self.args.resume_checkpoint = "current.pth"
            if self.args.resume:
                possible_path = os.path.join(self.args.saved_folder, self.args.resume_checkpoint)
                if os.path.isfile(possible_path):
                    ckpt = possible_path
                else:
                    ckpt = self.args.saved_folder

                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=True)

                self.args.iter = checkpoint['iter']
                self.args.global_epoch = checkpoint['global_epoch']
                if self.args.is_master:
                    print("Load ckpt from:", ckpt)
                    print("Number of iteration(s):", self.args.iter)

        
        elif archi == "vqgan-llama":
            # Initialize and load VQGAN model
            model = VQ_models[f"VQ-{self.args.f_factor}"](codebook_size=16384, codebook_embed_dim=8)
            if not os.path.exists(self.args.vqgan_folder):
                #make dir parent
                os.makedirs(os.path.dirname(self.args.vqgan_folder), exist_ok=True)
                hf_hub_download(repo_id="FoundationVision/LlamaGen", 
                filename="vq_ds8_c2i.pt", 
                local_dir=os.path.dirname(self.args.vqgan_folder))
            checkpoint = torch.load(self.args.vqgan_folder, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model = model.eval()

        elif archi == "ema":
            # Load Exponential Moving Average (EMA) model
            model = self.get_ema(self.vit, device=self.args.device)
            if self.args.resume and os.path.isfile(self.args.saved_folder + "ema.pth"):
                ckpt = self.args.saved_folder + "ema.pth"
                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = "_orig_mod." + k if self.args.compile else k
                    k = "module." + k if self.args.is_multi_gpus else k
                    new_state_dict[k] = v
                model.module.load_state_dict(new_state_dict, strict=True)
            return model

        else:
            model = None

        model = model.to(self.args.device)

        if self.args.compile: # Enable model compilation if using PyTorch 2.0+
            model = torch.compile(model)

        if self.args.is_multi_gpus:  # Enable multi-GPU training if available
            model = DDP(model, device_ids=[self.args.device])
            if archi == "vqgan-llama":
                model = model.module

        if self.args.is_master:
            if not self.args.disable_wandb:
                print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    def train_one_epoch(self, log_iter=2_500, ema_update=1):
        """
        Train the model for one epoch
            :param
                log_iter -> int: Frequency of logging and saving the model.
                ema_update --> int: Frequency of updating the EMA model.
            :return
                (avg_loss, avg_acc) -> tuple: (average loss, average accuracy) over the train dataset.
        """
        log_iter=self.args.log_iter
        self.vit.train()
        cum_loss, cum_acc = 0., 0.
        # Deque to store loss and accuracy over a moving window
        window_loss = deque(maxlen=self.args.grad_cum)
        window_acc = deque(maxlen=self.args.grad_cum)

        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        for i_bar, data in enumerate(bar):
            # Determine whether to update gradients based on gradient accumulation steps
            update_grad = (i_bar % self.args.grad_cum) == self.args.grad_cum - 1
            # Adjust the learning rate with warmup and cosine decay for the 10% last iter
            self.adapt_learning_rate()

            code = data["code"].long().to(self.args.device)
            y = data["y"].to(self.args.device)
            # code = code.view(-1, self.input_size, self.input_size)  # Reshape code to match input size
            b, h, w = code.size()

            # Randomly drop conditions for conditional generation (CFG)
            drop_label = (torch.rand(b) < self.args.drop_label).to(self.args.device)
            # Apply masking to encoded codes
            mask_code, mask, loss_ign = get_mask_code(
                code, codebook_size=self.args.codebook_size,
                mode=self.args.sched_mode, sched_pow=self.args.sched_pow, 
                bos=self.args.bos_value, value=self.args.mask_value)
            with self.autocast:  # Perform forward pass using mixed precision (if available)
                if self.args.model['name'] == 'encoder-decoder':
                    # must feed clean data with PGM
                    mask_code[:, 1:] = code.flatten(1,2)
                    mask_code[:, 0] = self.args.bos_value
                    mask[:, 0] = 0  # bos always in group 0
                    pred = self.vit(
                        x=mask_code, y=y, drop_label=drop_label, 
                        group_idxs=mask.to(int), use_inference_mode=False)
                    pred[:, :, self.args.mask_value] = self.neg_infinity
                        
                    if self.args.bos_value:
                        pred = pred[:, 1:]
                    target_code = code.flatten(1, 2)
                    loss = self.criterion(pred.flatten(0, 1), 
                                        target_code.flatten())
                    loss = loss / self.args.grad_cum
                else:
                    target_code = torch.where(loss_ign, code.detach().clone(), -100)
                    pred = self.vit(x=mask_code, y=y, drop_label=drop_label)
                    loss = self.criterion(pred.reshape(b*h*w, self.args.codebook_size + 1), target_code.view(b*h*w))
            loss.backward()
            # Perform gradient update if gradient accumulation step is reached
            if update_grad:
                nn.utils.clip_grad_norm_(self.vit.parameters(), self.args.grad_clip)  # Clip gradient
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

                # Update EMA model periodically
                if self.args.use_ema and i_bar % ema_update == 0:
                    self.ema.update(self.vit)

            # Compute running loss and accuracy
            cum_loss += loss.cpu().item() * self.args.grad_cum
            if self.args.bos_value:
                acc = torch.max(pred.reshape(b, -1, self.args.codebook_size + 2).data, 2)[1]
                acc = (acc.view(b, -1) == target_code.view(b, -1)).float()
                acc = acc.reshape(-1)
            else:
                acc = torch.max(pred.reshape(b*h*w, self.args.codebook_size + 1).data, 1)[1]
                acc = (acc.view(b*h*w) == code.view(b*h*w)).float()[mask.view(b*h*w)]
            cum_acc += acc.mean().cpu().item()
            window_loss.append(loss.cpu().item() * self.args.grad_cum)
            window_acc.append(acc.mean().item())

            # Logging and visualization
            if update_grad:
                if self.args.is_multi_gpus: # Synchronize logs across multiple GPUs if applicable
                    mini_batch_loss = self.all_gather(torch.tensor(window_loss).mean())
                    mini_batch_acc = self.all_gather(torch.tensor(window_acc).mean())
                else:
                    mini_batch_loss = torch.tensor(window_loss).mean()
                    mini_batch_acc = torch.tensor(window_acc).mean()
                
                if self.args.is_master: # Master process logs metrics
                    self.writer.log({"train/MiniBatchLoss": mini_batch_loss.item(),
                                     'step': self.args.iter})
                    self.writer.log({"train/MiniBatchAcc": mini_batch_acc.item(),
                                     'step': self.args.iter})
                    self.writer.log({"train/LearningRate": self.optim.param_groups[0]['lr'],
                                     'step': self.args.iter})
                    self.log_add_scalar('Train/MiniBatchLoss', mini_batch_loss, self.args.iter)
                    self.log_add_scalar('Train/MiniBatchAcc', mini_batch_acc, self.args.iter)
                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.args.iter)

                if self.args.iter % log_iter == 0 and self.args.is_master:
                    with torch.no_grad():
                        nb_sample = min(b//2, 10)
                        gen_sample = self.sampler(self, nb_sample=nb_sample)[0]
                        unmasked_code = pred.argmax(dim=-1)[:10]
                        if self.args.bos_value:
                            reco_sample = reconstruction(self, code=code[:10], unmasked_code=unmasked_code, mask=mask[:10][:, 1:])
                        else:
                            reco_sample = reconstruction(self, code=code[:10], unmasked_code=unmasked_code, mask=mask[:10])
                        self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)
                        self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                        # Save the current model state
                        if not self.args.debug:
                            self.save_network(model=self.vit, path=os.path.join(self.args.saved_folder, "current.pth"),
                                            optimizer=self.optim, iter=self.args.iter, global_epoch=self.args.global_epoch)
                            if self.args.iter % 100_000 == 0:
                                self.save_network(model=self.vit, path=os.path.join(self.args.saved_folder, f"step_{self.args.iter//1000}K.pth"),
                                            optimizer=self.optim, iter=self.args.iter, global_epoch=self.args.global_epoch)
                            if self.args.use_ema:  # Save the EMA model if enabled
                                self.save_network(model=self.ema.module, path=os.path.join(self.args.saved_folder, "ema.pth"),
                                                iter=self.args.iter, global_epoch=self.args.global_epoch)

                # Increment global iteration counter
                self.args.iter += 1

            # Stop training if max iterations are reached
            if self.args.iter > self.args.max_iter:
                if self.args.is_master:
                    print("End of training: reached max iterations")
                return cum_loss / i_bar, cum_acc / i_bar

        # Return average loss for the epoch
        return cum_loss / len(self.train_data), cum_acc / len(self.train_data)

    @torch.no_grad()
    def eval_one_epoch(self):
        """
        Eval the model for one epoch
            :return
                (avg_loss, avg_acc) -> tuple: (average loss, average accuracy) over the test dataset.
        """
        # Use the Exponential Moving Average (EMA) model if enabled; otherwise, use the main model
        model = self.ema if self.args.use_ema else self.vit
        model.eval()
        cum_loss, cum_acc = 0., 0.

        bar = tqdm(self.test_data, leave=False) if self.args.is_master else self.test_data
        for i_bar, (data) in enumerate(bar):
            code = data["code"].long().to(self.args.device)
            y = data["y"].to(self.args.device)
            b, h, w = code.size()

            # Randomly drop conditions for conditional generation (CFG)
            drop_label = (torch.rand(b) < self.args.drop_label).to(self.args.device)

            # Apply masking to encoded codes
            mask_code, mask, loss_ign = get_mask_code(
                code, codebook_size=self.args.codebook_size,
                mode=self.args.sched_mode, sched_pow=self.args.sched_pow, bos=self.args.bos_value, value=-1, register=self.args.register)

            target_code = torch.where(loss_ign, code.detach().clone(), -100)
    
            with self.autocast:  # Perform forward pass using mixed precision (if available)
                if self.args.model['name'] == 'encoder-decoder':
                    # must feed clean data with PGM
                    mask_code[:, 1:] = code.flatten(1,2)
                    mask_code[:, 0] = self.args.bos_value
                    mask[:, 0] = 0  # bos always in group 0
                
                    pred = self.vit(
                        x=mask_code, y=y, drop_label=drop_label, 
                        group_idxs=mask.to(int), use_inference_mode=False)
                    pred[:, :, self.args.mask_value] = self.neg_infinity
                        
                    if self.args.bos_value:
                        pred = pred[:, 1:]
                    target_code = code.flatten(1, 2)
                    loss = self.criterion(pred.flatten(0, 1), 
                                        target_code.flatten())
                    loss = loss / self.args.grad_cum
                else:
                    target_code = torch.where(loss_ign, code.detach().clone(), -100)
                    pred = self.vit(x=mask_code, y=y, drop_label=drop_label)
                    loss = self.criterion(pred.reshape(b*h*w, self.args.codebook_size + 1), target_code.view(b*h*w))

            # Compute running loss and accuracy
            if self.args.bos_value:
                acc = torch.max(pred.reshape(b, -1, self.args.codebook_size + 2).data, 2)[1]
                acc = (acc.view(b, -1) == target_code.view(b, -1)).float()
                acc = acc.reshape(-1)
            else:
                acc = torch.max(pred.reshape(b*h*w, self.args.codebook_size + 1).data, 1)[1]
                acc = (acc.view(b*h*w) == code.view(b*h*w)).float()[mask.view(b*h*w)]
            cum_acc += acc.mean().cpu().item()
            cum_loss += loss.cpu().item()
        model.train()

        # Return average loss for the epoch
        return cum_loss / len(self.test_data), cum_acc / len(self.test_data)

    def fit(self, metrics_eval=20): # metrics_eval == 10
        """
        Train the model for multiple epochs.

        Args:
            metrics_eval (int): Frequency (in epochs) at which FID/IS is performed.

        This function handles:
            - Training and evaluation for each epoch
            - Synchronization across multiple GPUs if applicable
            - Model checkpointing at specific epochs
            - Logging training progress and metrics
        """

        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # This is the sampler used for evaluation (no for visualization)
        if self.args.sampler == "halton":
            eval_sampler = HaltonSampler(
                sm_temp_min=1., sm_temp_max=1., temp_warmup=0, w=0, sched_pow=self.args.sched_pow,
                step=self.args.step, randomize=self.args.randomize, top_k=-1
            )
        elif self.args.sampler == "confidence":
            eval_sampler = ConfidenceSampler(
                sm_temp=1., w=0, r_temp=self.args.r_temp, sched_mode=self.args.sched_mode, step=self.args.step
            )
        else:
            eval_sampler = None
        # Main training loop across epochs
        for e in range(self.args.global_epoch, self.args.epoch+1):
            # Synchronize dataset shuffling across multiple GPUs if applicable
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)
                self.test_data.sampler.set_epoch(e)

            # Train/Eval the model and get test loss and accuracy
            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.eval_one_epoch()

            # Synchronize loss and accuracy across GPUs if applicable
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss)
                train_acc = self.all_gather(train_acc)
                test_loss = self.all_gather(test_loss)
                test_acc = self.all_gather(test_acc)

            # Save model checkpoints at specific epochs
            if not self.args.debug and self.args.global_epoch % metrics_eval == 0 and self.args.global_epoch > 0 and self.args.is_master:
                self.save_network(model=self.vit,
                                  path=self.args.saved_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  optimizer=self.optim, iter=self.args.iter,
                                  global_epoch=self.args.global_epoch)

            # Logging and printing progress
            if self.args.is_master:
                clock_time = (time.time() - start)
                # self.log('Epoch/Loss', {"Train": train_loss, "Eval": test_loss}, on_step=False, on_epoch=True, sync_dist=True)
                # self.log('Epoch/Accuracy', {"Train": train_acc, "Eval": test_acc}, on_step=False, on_epoch=True, sync_dist=True)
                self.log_add_scalar('Epoch/Loss', {"Train": train_loss, "Eval": test_loss}, self.args.global_epoch)
                self.log_add_scalar('Epoch/Accuracy', {"Train": train_acc, "Eval": test_acc}, self.args.global_epoch)
                self.writer.log({"epoch/Loss": {"Train": train_loss, "Eval": test_loss},
                                 'epoch': self.args.global_epoch})
                self.writer.log({"epoch/Accuracy": {"Train": train_acc, "Eval": test_acc},
                                 'epoch': self.args.global_epoch})
                now = datetime.now()
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter},"
                      f" Train: {train_loss:.4f}, Eval: {test_loss:.4f},"
                      f" Time: {int(clock_time // 3600):.0f}:{int((clock_time % 3600) // 60):02d}:{int(clock_time % 60):02d},"
                      f" Date: {now.date()} {now.hour:02}:{now.minute:02}")

            # Perform FID/IS at specified intervals
            if e % metrics_eval == 0 and e > 0 and not self.args.debug and self.args.register == 0:
                m = self.eval(sampler=eval_sampler, num_images=10_000, save_exemple=False, compute_pr=False,
                              split="Test", mode="c2i", data=self.args.data.split("_")[0])

                if self.args.is_master:
                    self.log_add_scalar('Metrics/FID', m["FID"], self.args.global_epoch)
                    self.log_add_scalar('Metrics/IS', m["IS"], self.args.global_epoch)
                    self.writer.log({"metrics/FID": m["FID"], 'epoch': self.args.global_epoch})
                    self.writer.log({"metrics/IS": m["IS"], 'epoch': self.args.global_epoch})

            self.args.global_epoch += 1

            # Stop training if the maximum number of iterations is reached
            if self.args.iter > self.args.max_iter:
                break

        # Final model saving when training is completed
        if self.args.is_master:
            print(f"Training is done, saving last check point at {self.args.saved_folder}")
            self.save_network(model=self.vit, path=self.args.saved_folder + f"last.pth",
                              optimizer=self.optim, iter=self.args.iter,
                              global_epoch=self.args.global_epoch)
            if self.args.use_ema:
                self.save_network(model=self.ema.module, path=os.path.join(self.args.saved_folder, "last_ema.pth"),
                                  iter=self.args.iter, global_epoch=self.args.global_epoch)
        return
