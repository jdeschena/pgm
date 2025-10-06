# Abstract class for the trainer
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torchvision.utils as vutils
from torchvision.utils import save_image
import wandb
import sys
from dataset.dataloader import get_data
from models.ema import EMA
from metrics.sample_and_eval import SampleAndEval
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import json

def get_logger():
    # Info on rank 0, warning on all other ranks
    warning_only = torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
    # Check if running under SLURM
    if "SLURM_PROCID" in os.environ:
        # In SLURM environment, warning_only if not global rank 0
        warning_only = int(os.environ["SLURM_PROCID"]) != 0
    if warning_only:
        logger.remove()
        sink = sys.stderr
        level = "WARNING"
    else:
        logger.remove()
        sink = sys.stdout
        level = "INFO"

    # Add custom format for loguru logger for all messages:
    logger.remove()
    FMT_STRING = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                  "<level>{message}</level>")
    logger.add(sink, level=level, format=FMT_STRING)
    return logger


class FakeWriter:
    def log(self, *args, **kwargs):
        pass


class Trainer(object):
    """
    Abstract class for training a Vision Transformer (ViT) model.

    This class handles model initialization, logging, optimizer setup, and network configuration.
    """

    vit = None
    ae = None
    optim = None
    sampler = None
    writer = None
    sae = None


    def __init__(self, args):
        """ Initializes the Trainer class.
         :param
            args: Configuration object containing various hyperparameters and settings, see main.py
        """
        self.args = args
        logger = get_logger()
        # Initialize logging writer (TensorBoard)
        if self.args.is_master and self.args.writer_log != "":
            if not args.disable_wandb:
                logger.info(f"Logging to {self.args.writer_log}")
                wandb.login()  # This will pick up the token from the env var
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.run_name,    # Use writer_log as run name
                    dir=self.args.writer_log,     # Optional: working directory
                )
                wandb.define_metric("step")
                wandb.define_metric("train/*", step_metric="step")
                wandb.define_metric("epoch")
                wandb.define_metric("epoch/*", step_metric="epoch")
                wandb.define_metric("metrics/*", step_metric="epoch")
                
                self.tb_writer = SummaryWriter(log_dir=args.writer_log)
        
                self.writer = wandb
            else:
                self.writer = FakeWriter()
                self.tb_writer = None
        self.neg_infinity=-1000000.0
    def get_network(self, archi):
        """ Placeholder method to get a neural network architecture. """
        pass

    def transformer_size(self, size):
        """ Returns the transformer configuration based on the specified size.
            :param:
                size -> str: Transformer size (tiny, small, base, large, xlarge).
            :returns
                (hidden_dim, depth, heads) -> tuple: Transformer settings.
        """

        if size == "tiny":
            hidden_dim, depth, heads = 384, 6, 6
        elif size == "small":
            hidden_dim, depth, heads = 512, 12, 6
        elif size == "base":
            hidden_dim, depth, heads = 768, 12, 12
        elif size == "large":
            hidden_dim, depth, heads = 1024, 24, 16
        elif size == "xlarge":
            hidden_dim, depth, heads = 1152, 28, 16
        else:
            hidden_dim, depth, heads = 768, 12, 12
            if self.args.is_master:
                print("Size of the transformer not understood, initialize a Base VIT")

        return hidden_dim, depth, heads

    def log_add_img(self, names, img, iteration):
        """ Logs an image to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               img       -> Tensor: Image tensor to be logged.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.writer is None:
            return
        b, c, h, w = img.size()
        if w > 128 or h > 128:
            img = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=False)
        img = vutils.make_grid(img, nrow=min(10, len(img)), padding=2, normalize=False)
        img = (img + 1) / 2 
        img = torch.clip(img * 255, 0, 255).to(torch.uint8) 
        img_np = img.cpu().numpy().transpose(1, 2, 0)  
       
        self.tb_writer.add_image(tag=names, img_tensor=img, global_step=iteration)
        self.writer.log({names: wandb.Image(img_np), 'step': iteration})    
       

    def log_add_txt(self, names, txt, iteration):
        """ Logs a text to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               txt       -> str: Text content.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.tb_writer is None:
            return
        self.tb_writer.add_text(tag=names, text_string=txt, global_step=iteration)

    def log_add_scalar(self, names, scalar, iteration):
        """ Logs a scalar value to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               scalar    -> (float or dict): Scalar value(s) to log.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.tb_writer is None:
            return
        if isinstance(scalar, dict):
            self.tb_writer.add_scalars(main_tag=names, tag_scalar_dict=scalar, global_step=iteration)
        else:
            self.tb_writer.add_scalar(tag=names, scalar_value=scalar, global_step=iteration)

    def get_optim(self, net, lr, mode="AdamW", resume_checkpoint=None, **kwargs):
        """ Returns an optimizer for the given network.
            :param:
                net      -> (nn.Module or list): Model or list of models whose parameters will be optimized.
                lr       -> float: Learning rate.
                mode     -> str: Optimizer type ("AdamW", "Adam", "SGD").
                **kwargs -> Additional optimizer parameters.

            :return:
                Optimizer: Configured optimizer.
        """

        # Extract parameters from network(s)
        if isinstance(net, list):
            params = []
            for n in net:
                params += list(n.parameters())
        else:
            params = net.parameters()
        if resume_checkpoint is None:
            resume_checkpoint = "current.pth"
        # Choose an optimizer type
        if mode == "AdamW":
            optimizer = optim.AdamW(params, lr=lr, **kwargs)
        elif mode == "Adam":
            optimizer = optim.Adam(params, lr=lr, **kwargs)
        elif mode == "SGD":
            optimizer = optim.SGD(params, lr=lr, **kwargs)
        else:
            optimizer = None

        ckpt_path = os.path.join(self.args.saved_folder, resume_checkpoint)
        if self.args.resume and os.path.isfile(ckpt_path):
            ckpt = ckpt_path
            logger.info(f"Resuming training from {ckpt}")
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            if 'optimizer_state_dict' in checkpoint.keys() and checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer

    @staticmethod
    def get_loss(mode="cross_entropy", **kwargs):
        """Static method to return a loss function based on the specified mode. """

        if mode == "l1":
            return nn.L1Loss()
        elif mode == "l2":
            return nn.MSELoss()
        elif mode == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        return None

    def get_ema(self, model, device=None):
        """ Method to create an Exponential Moving Average (EMA) of the model parameters.
            :param:
               model  -> nn.Module: The model for which to create the EMA.
               device -> torch.device: The device on which to place the EMA.
            :returns:
                   EMA: An instance of the EMA class.
        """
        ema = EMA(model, device=device)
        return ema

    def train_one_epoch(self, epoch):
        """ Placeholder method for training the model for one epoch. """
        return

    def fit(self):
        """ Placeholder method for the main training loop. """
        pass

    @staticmethod
    def all_gather(obj, reduce="mean"):
        """ Static method to gather objects from all processes and reduce them.
            :param:
               obj -> any:The object to gather.
               reduce -> str: The reduction operation to apply. Options are "mean", "sum", or "none".
            :returns:
               obj -> torch.Tensor: The reduced object.
        """
        world_size = dist.get_world_size()
        tensor_list = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather_object(tensor_list, obj)
        obj = torch.FloatTensor(tensor_list)
        if reduce == "mean":
            obj = obj.mean()
        elif reduce == "sum":
            obj = obj.sum()
        elif reduce == "none":
            pass
        else:
            raise NameError("reduction not known")

        return obj

    def save_network(self, model, path, optimizer=None, iter=None, global_epoch=None):
        """ Save the state of the model, including the iteration,
            the optimizer state and the current epoch
            :params:
                model        -> nn.Module: The model to save.
                path         -> str: The path where the model state will be saved.
                optimizer    -> torch.optim.Optimizer: (optional) The optimizer whose state should be saved.
                iter         -> int: The current iteration number.
                global_epoch -> int: The current global epoch number.
        """
        state_dict = model.module.state_dict() if self.args.is_multi_gpus else model.state_dict()
        if self.args.compile:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            state_dict = new_state_dict

        torch.save({'iter': iter,
                    'global_epoch': global_epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': None if optimizer is None else optimizer.state_dict()
                    },
                   path)

    def adapt_learning_rate(self):
        """ Method to adapt the learning rate based on the current iteration.
            This method implements a linear warmup for the first few iterations and a cosine decay for the last 1
        """
        if self.args.iter < self.args.warm_up:  # linear warmup updates
            self.optim.param_groups[0]['lr'] = self.args.lr * (self.args.iter / self.args.warm_up)
        # cosine learning rate decay for the last 10% iterations
        elif self.args.iter > (self.args.max_iter - (.1 * self.args.max_iter)):
            decay_step = .1 * self.args.max_iter
            decay = (self.args.iter - (self.args.max_iter - decay_step)) / decay_step
            cosine_decay = (1 + np.cos(np.pi * decay))
            self.optim.param_groups[0]['lr'] = max(self.args.lr * 0.5 * cosine_decay, self.args.lr * 0.01)
        else: # basic lr for the rest of the time
            self.optim.param_groups[0]['lr'] = self.args.lr

    @torch.no_grad()
    def eval(self, sampler, num_images=50_000, save_exemple=True,
             compute_pr=True, split="train", mode="c2i", data="imagenet"):
        """ Method to evaluate the model.

        :params:
            eval_sampler -> Sampler: The sampler used for evaluation.
            num_images   -> int: The number of images to evaluate.
            save_exemple -> bool: Whether to save example images.
            compute_pr   -> bool: Whether to compute precision and recall.
            split        -> str: The dataset split to evaluate on ("Train" or "Test").
            mode         -> str: The mode of evaluation ("c2i" for code to image).
            data         -> str: The dataset on which to test
        Returns:
            The evaluation metrics.
        """

        # Initialize SampleAndEval
        self.sae = SampleAndEval(
            device=self.args.device, is_master=self.args.is_master, nb_gpus=self.args.nb_gpus,
            num_images=num_images, mode=mode, compute_manifold=compute_pr)

        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n" + str(sampler))
            if os.path.exists("saved_networks/ImageNet_256_train_stats.pt") and \
               self.args.data.startswith("imagenet"):
                print("Use pre-computed ImageNet stats")
            if save_exemple: # Save example images if master process
                x = self.sampler(self, nb_sample=20)[0]
                x = vutils.make_grid(x.float().cpu(), nrow=10, padding=0, normalize=True)
                x_name = str(self.sampler).replace(" ", "_").replace(",", "").replace(":", "") + ".jpg"
                #make dir
                if not os.path.exists("saved_images"):
                    os.makedirs("saved_images")
                save_image(x, f"saved_images/" + x_name)

        # Load the dataset for evaluation (containing the Images)
        train_loader, test_loader = get_data(
            data, 256, self.args.eval_folder,
            self.args.bsize, self.args.num_workers, 
            self.args.is_multi_gpus, -1, args=self.args)
        if split == 'train':
            data_loader = train_loader
        else:
            data_loader = test_loader
        print(f"Number of images in the dataset: {len(data_loader.dataset)}")

        # Evaluate the model
        scores = self.sae.compute_images_features_from_model(self, sampler, data_loader)
        
        if self.args.is_master:
            print(scores)
            if self.args.scores_save_path is not None:
                os.makedirs(os.path.dirname(self.args.scores_save_path), exist_ok=True)
                name = self.args.scores_save_path
                with open(name, 'w') as f:
                    json.dump(scores, f, indent=4)
        return scores
