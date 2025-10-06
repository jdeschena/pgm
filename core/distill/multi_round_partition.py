import torch
from transformers import AutoModelForMaskedLM
from loguru import logger
from models.loading_utils import get_backbone
import torch.nn.functional as F
import time
from pathlib import Path
import os
import sys
import copy

from core.sampling.ancestral import sample_categorical
from core.diffusion.absorbing import DiffusionCore
from core.sampling import AncestralSampler, AnalyticSampler
from tqdm import trange
from data.utils import params2key
from pathlib import Path
import os


from transformers import AutoTokenizer
from data import dataloader
from lightning.fabric import Fabric
import lightning as L

from tqdm import trange
import numpy as np
from einops import rearrange
import pandas as pd
from huggingface_hub import create_branch, create_repo, PyTorchModelHubMixin, hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file
from huggingface_hub import login

@torch.jit.script
def tv_dist(log_p: torch.Tensor, log_q: torch.Tensor):
    p = log_p.exp()
    q = log_q.exp()
    diff = (p - q).abs()
    loss = diff.sum(-1).mean()
    loss = loss / 2
    return loss


def load_scaling_student(size="sm", round=1):
    if not round in list(range(1, 8)):
        raise ValueError(f"Round value is too large: should be 1 <= round <= 7. Actual value: `{round}`")
    
    if size not in ("sm", "md", "large"):
        raise ValueError(f"Valid model sizes: sm, md, large. Actual value: `{size}`")
    
    revision = f"scaling_400k_{size}_step_{round * 10_000}"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model


def load_small_student(loss="kld", round=1):
    if not round in list(range(1, 8)):
        raise ValueError(f"Round value is too large: should be 1 <= round <= 7. Actual value: `{round}`")
    
    if loss not in ("kld", "mse", "tvd"):
        raise ValueError(f"Valid losses sizes: kld, mse, tvd. Actual value: `{loss}`")

    revision = f"baselines_{loss}_step_{round * 10_000}"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model


def load_mdlm_small():
    revision = "teacher_1M_sm"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model

    


def load_scaling_teacher(size="sm"):
    if size not in ("sm", "md", "large"):
        raise ValueError(f"Valid model sizes: sm, md, large. Actual value: `{size}`")
    
    revision = f"teacher_400k_{size}_step_400000"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model




class MultiRoundSDTT(DiffusionCore, PyTorchModelHubMixin, AncestralSampler, AnalyticSampler):
    def __init__(self, config, tokenizer, verbose=True):
        DiffusionCore.__init__(self, config, tokenizer)
        AncestralSampler.__init__(self, config)
        AnalyticSampler.__init__(self, config)
        self.neg_infinity = -1000000.0
        self.verbose = verbose

        self.teacher = None
        self.num_distill_steps = self.config.parameterization.num_distill_steps
        self.tot_num_sampl_steps = self.config.parameterization.orig_num_sampling_steps
        self.min_num_sampl_steps = self.config.parameterization.min_num_sampling_steps
        self.distill_mode = self.config.parameterization.distill_mode
        self.start_from_hf = self.config.parameterization.start_from_hf
        self.reset_optimizer_on_growth = (
            config.parameterization.reset_optimizer_on_growth
        )
                
        self.use_ema_on_growth = config.parameterization.use_ema_on_growth

        self.sampling_eps_tensor = torch.tensor(self.sampling_eps)
        self.sampling_mode = self.config.parameterization.sampling_mode
        assert self.sampling_mode in ("ancestral", "analytic")

        self.grow_dt_every = config.parameterization.grow_dt_every

        self.dt = (1 - self.sampling_eps) / self.tot_num_sampl_steps
        self.loss_precision = self.config.parameterization.loss_precision

        mode = self.distill_mode
        self._loss_fn = None  # fn to compare preds & targets
        if mode == "mse":
            self._loss_fn = self._mse
        elif mode == "tvd":
            self._loss_fn = self._tvd
        elif mode == "kl-fwd":
            self._loss_fn = self._fwd_kl
        elif mode == "kl-bwd":
            self._loss_fn = self._bwd_kl
        else:
            raise ValueError(mode)
        if verbose:
            logger.info(f"Distillation loss: {mode}")
        self.prepare_teacher_and_student()

    @classmethod
    def from_pretrained(cls, repo, revision):
        ckpt_path = hf_hub_download(repo_id=repo, revision=revision, filename="model.safetensors")
        config_path = hf_hub_download(repo_id=repo, revision=revision, filename="config.json")

        ckpt = load_file(ckpt_path)
        config = OmegaConf.load(config_path)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)

        model = cls(config, tokenizer, verbose=False)
        model.load_state_dict(ckpt)
        return model
     
    def push_to_hub(self, repo, revision="main", private=True):
        login(token=os.environ["HF_WRITE_KEY"])
        repo = create_repo(repo_id=repo, private=private, exist_ok=True).repo_id
        create_branch(repo, branch=revision, exist_ok=True)

        dict_config = OmegaConf.to_container(self.config)
        PyTorchModelHubMixin.push_to_hub(self, repo_id=repo, branch=revision, private=private, config=dict_config)

    def prepare_teacher_and_student(self):
        """
        If start from hf checkpoint:
            - Load the hf arch in student + teacher
        Else:
            - Init teacher as a copy of student

        if start checkpoint is not kuleshov-group/mdlm-owt -> load from disk

        """
        if self.verbose:
            logger.info("Loading teacher checkpoint...")
        ckpt_path = self.config.parameterization.checkpoint_path

        if self.start_from_hf and ckpt_path == "kuleshov-group/mdlm-owt":
            assert self.config.data_preprocess.legacy_start_end_bos
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                "kuleshov-group/mdlm-owt", trust_remote_code=True
            )

            # Hack so that teacher doesn't get registered as child
            self.teacher = [
                AutoModelForMaskedLM.from_pretrained(
                    "kuleshov-group/mdlm-owt", trust_remote_code=True
                ).eval()
            ]

            if self.config.compile:
                self.teacher[0] = torch.compile(self.teacher[0])

        else:
            self.teacher = [get_backbone(self.config, self.vocab_size)]

        if ckpt_path != "kuleshov-group/mdlm-owt":
            if self.verbose:
                logger.info(f"Loading checkpoint in teacher from `{ckpt_path}`.")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items()}
            self.teacher[0].load_state_dict(ckpt)
            self.backbone.load_state_dict(ckpt)

        self.teacher[0].eval()
        self.teacher[0].requires_grad_(False)
        # Reset EMA to use weights from checkpoint
        self.init_ema()

        if self.verbose:
            logger.info("Teacher checkpoint loaded.")

    def forward(self, xt, cond):
        if not self.time_conditioning:
            cond = torch.zeros_like(cond)

        group_idxs = (xt == self.mask_index).to(int)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            logits = self.backbone(xt, cond, 
                                   group_idxs=group_idxs)
        logits = self._subs_parameterization(logits, xt)
        return logits

    def forward_teacher(self, xt, cond):
        if not self.time_conditioning:
            cond = torch.zeros_like(cond)

        group_idxs = (xt == self.mask_index).to(int)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            logits = self.teacher[0](xt, cond, group_idxs=group_idxs)
        logits = self._subs_parameterization(logits, xt)
        return logits
    
    def _subs_parameterization(self, model_output, xt):
        neg_infinity = -10_000_000
        index = torch.full(
            size=(xt.shape[0], xt.shape[1], 1), 
            fill_value=self.mask_index, 
            device=xt.device)

        model_output = torch.scatter(
            model_output, dim=-1, 
            index=index, value=-float('inf'))
        
        unmasked_indices = (xt != self.mask_index)

        model_output = torch.where(
            unmasked_indices[..., None], 
            neg_infinity, model_output)
        model_output = torch.scatter(
            model_output, dim=-1, 
            index=xt[..., None], value=0.0)

        model_output = torch.log_softmax(model_output, dim=-1).clone()
        model_output[:, :, self.mask_index] = neg_infinity
        return model_output

    def to(self, device):
        DiffusionCore.to(self, device=device)
        self.teacher[0].to(device=device)

    @torch.no_grad
    def _teacher_logprobs_on_mask(self, xt, t_start):
        """
        Collect teacher predictions for ALL mask tokens
        """
        dt = self.dt

        space = torch.linspace(
            1, 0, self.num_distill_steps, device=t_start.device
        ).double()[:, None]
        t_start = t_start[None, :].double()
        t_end = t_start - dt * self.num_distill_steps
        # Evenly-spaced interpolation between t_start and t_end
        ts = t_start * space + (1 - t_start) * t_end
        # Ensure we don't feed the model values smaller than sampling_eps
        ts = torch.maximum(ts, self.sampling_eps_tensor)

        teacher_predictions = torch.zeros(
            (*xt.shape, self.vocab_size), device=xt.device
        )
        unmasked_tokens = torch.zeros(xt.shape, device=xt.device)
        curr_x = xt

        for idx in range(len(ts)):
            t = ts[idx].float()
            # TODO: add analytic sampler
            if self.loss_precision == 64:
                assert self.sampling_mode == "ancestral", "fp64 sampling not implemented for analytic"
            if self.sampling_mode == "ancestral":
                log_p_x0, q_xs = self._compute_ddpm_update(
                    curr_x, t, dt, forward=self.forward_teacher
                )
                if self.loss_precision == 64:
                    q_xs = q_xs.to(torch.float64)
                elif self.loss_precision == 32:
                    q_xs = q_xs.to(torch.float32)
                update = sample_categorical(q_xs)
                new_batch = self._ddpm_sample_update(curr_x, update)

            elif self.sampling_mode == "analytic":
                log_p_x0, new_batch = self._analytic_update(
                    curr_x,
                    t,
                    dt,
                    forward=self.forward_teacher,
                )
            else:
                raise ValueError(self.sampling_mode)

            updated = curr_x != new_batch
            # Extract predictions for denoised tokens
            teacher_predictions = teacher_predictions.to(log_p_x0.dtype)
            teacher_predictions[updated] = log_p_x0[updated]
            unmasked_tokens += updated
            curr_x = new_batch

        # Put predictions from model on last step for remaining MASK tokens
        last_preds_update_mask = (curr_x == self.mask_index) * torch.logical_not(
            unmasked_tokens
        )
        last_preds_update_mask = last_preds_update_mask[..., None].to(bool)
        teacher_predictions = torch.where(
            last_preds_update_mask, log_p_x0, teacher_predictions
        )
        return teacher_predictions

    def loss(self, x, t=None, attention_mask=None):
        if attention_mask is not None:
            assert (
                (attention_mask.to(int) == 1).all().item()
            ), "attention mask not supported"

        x0 = x
        if t is None:
            t = self._sample_t(x0.shape[0], x0.device)

        sigma, move_chance, dsigma = self._t_to_sigma(t)
        xt = self.q_xt(x0, move_chance)
        sigma = sigma.squeeze(-1)  # Original shape [bs, 1]
        # Loss on all masked tokens
        teacher_preds = self._teacher_logprobs_on_mask(xt, t)
        student_preds = self.forward(xt, sigma)
        is_mask = xt == self.mask_index

        target = teacher_preds[is_mask]
        preds = student_preds[is_mask]

        if self.loss_precision == "64":
            target = target.to(torch.float64)
            preds = preds.to(torch.float64)
        elif self.loss_precision == "32":
            target = target.to(torch.float32)
            preds = preds.to(torch.float32)

        loss = self._loss_fn(preds, target)
        return loss

    def _mse(self, preds, target):
        return F.mse_loss(preds, target)

    def _tvd(self, preds, target):
        return (preds - target).abs().sum(-1).mean()

    def _fwd_kl(self, preds, target):
        return F.kl_div(preds, target, log_target=True, reduction="batchmean")

    def _bwd_kl(self, preds, target):
        return F.kl_div(target, preds, log_target=True, reduction="batchmean")

    def training_step(self, batch):
        if self.ema is not None:
            assert (
                not self._using_ema_weights
            ), "SHOULD NOT USE EMA WEIGHTS DURING TRAINING!!!"
        x = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        step = self.trainer.global_step
        curr_round = step // self.grow_dt_every
        self.dt = (
            (1 - self.sampling_eps) / self.tot_num_sampl_steps * (self.num_distill_steps**curr_round)
        )
        if step > 0 and step % self.grow_dt_every == 0:
            effective_num_steps = round(1 / self.dt)
            if effective_num_steps < self.min_num_sampl_steps:
                logger.info(
                    f"Reached below the minimal effective number of sampling steps, stopping..."
                )
                sys.exit()
            else:
                logger.info(
                    f"Step {step}: Doubling `dt`! New effective number of steps: {effective_num_steps}."
                )
            self._student_to_teacher()
            if self.reset_optimizer_on_growth:
                logger.info("Resetting optimizers...")
                self.trainer.strategy.setup_optimizers(self.trainer)

        loss = self.loss(x, attention_mask=attention_mask)
        self.log(
            name="train/loss",
            value=loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def _student_to_teacher(
        self,
    ):
        start = time.perf_counter()
        if self.use_ema_on_growth:
            # Use EMA as teacher and student, and reset EMA for next round
            self.store_ema()
            student_ckpt = copy.deepcopy(self.backbone.state_dict())
            self.restore_ema()
            self.backbone.load_state_dict(student_ckpt)
            self.init_ema()
        else:
            student_ckpt = self.backbone.state_dict()

        self.teacher[0].load_state_dict(student_ckpt)
        end = time.perf_counter()

        logger.info(f"Swapped student into teacher in {end - start:.2f} seconds.")
        save_path = (
            Path(os.getcwd())
            / "student_checkpoints"
            / f"{self.trainer.global_step}.ckpt"
        )
        self.trainer.save_checkpoint(save_path)

    @torch.no_grad()
    def sample(
        self,
        n_samples=8,
        num_steps=256,
        seq_len=1024,
        sampler="ancestral",
        cache_preds=False,
        verbose=False,
        add_bos=False,
        add_eos=False,
        project_fn=lambda x: x,
    ):
        assert not cache_preds, "Not implemented"
        if cache_preds:
            assert (
                not self.config.time_conditioning
            ), "Cannot use caching with time-conditional network"

        assert sampler in ("ancestral", "analytic")
        if seq_len is None:
            seq_len = self.config.model.length

        batch = self._sample_prior(n_samples, seq_len)
        batch = project_fn(batch)

        if add_bos:
            batch[:, 0] = self.tokenizer.bos_token_id

        if add_eos:
            batch[:, -1] = self.tokenizer.eos_token_id

        # +1 because we use the last value for denoising
        ts = torch.linspace(1.0, self.sampling_eps, steps=num_steps + 1)
        dt = (1 - self.sampling_eps) / num_steps

        sampling_precision = self.config.parameterization.sampling.precision
        if sampling_precision == "fp64":
            dtype = torch.float64
        elif sampling_precision == "fp32":
            dtype = torch.float32
        elif sampling_precision == "bf16":
            dtype = torch.bfloat16
        else:
            assert sampling_precision is None
            dtype = None

        for i in trange(num_steps, desc="sampling...", disable=not verbose):
            t = ts[i] * torch.ones(n_samples, 1, device=self.device)
            if sampler == "ancestral":
                _, new_batch = self._ddpm_update(batch, t, dt, dtype)
            elif sampler == "analytic":
                _, new_batch = self._analytic_update(batch, t, dt, dtype)
            new_batch = project_fn(new_batch)
            # If no caching or an update was made, remove cache
            # if not cache_preds or not torch.allclose(new_batch, batch):
            #    cache = None
            batch = new_batch

        # Denoise
        if (batch == self.mask_index).any():
            t = ts[-1] * torch.ones(n_samples, 1, device=self.device)
            _, batch = self._ddpm_update(
                batch, t, dt, denoise=True, mask_idx=self.mask_index
            )
            batch = project_fn(batch)

        return batch


def sample_uncond(module):
    logger.info("Starting unconditional sampling.")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    uncond_cfg = sampling_cfg.uncond

    ckpt_name = config.checkpointing.resume_ckpt_path.split("/")[-1]
    metadata = dict(
        num_samples=uncond_cfg.num_samples,
        from_ema=uncond_cfg.from_ema,
        num_steps=uncond_cfg.num_steps,
        seq_len=uncond_cfg.seq_len,
        sampler=uncond_cfg.sampler,
        add_bos=uncond_cfg.add_bos,
        add_eos=uncond_cfg.add_eos,
        checkpoint_name=ckpt_name,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "uncond" / save_fname
    assert not save_path.exists(), save_fname

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(100 + fabric.global_rank)
    # Note: the next line creates a bug when calling functions from the module
    # pl_module = fabric.setup(module)
    pl_module = module
    fabric.to_device(pl_module)

    bs = uncond_cfg.batch_size
    num_steps = uncond_cfg.num_steps
    seq_len = uncond_cfg.seq_len
    target_num_samples = uncond_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert target_num_samples % (tot_num_device * bs) == 0
    n_sampling_rounds = target_num_samples // (tot_num_device * bs)

    if uncond_cfg.from_ema:
        pl_module.store_ema()

    all_samples = []
    for _ in trange(
        n_sampling_rounds,
        desc=f"Sampling with n_steps={num_steps}, seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        with fabric.autocast():
            out = pl_module.sample(
                n_samples=bs,
                num_steps=num_steps,
                seq_len=seq_len,
                sampler=uncond_cfg.sampler,
                add_bos=uncond_cfg.add_bos,
                add_eos=uncond_cfg.add_eos,
                cache_preds=uncond_cfg.cache_preds,
            )
        out = fabric.all_gather(data=out)
        if fabric.global_rank == 0:
            if out.ndim == 3:  # ndim == 2 when running on one device
                out = rearrange(out, "dev bs l -> (dev bs) l")
            all_samples.append(out.cpu())
        del out

    # Join and save to disk
    if fabric.global_rank == 0:
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_samples = all_samples[:target_num_samples]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(save_path, samples=all_samples, metadata=metadata)
        logger.info(f"Saved {len(all_samples)} samples in {save_path}")

    # Restore orig model weights
    if uncond_cfg.from_ema:
        pl_module.restore_ema()

    fabric.barrier()


def sample_cond_prefix(module):
    logger.info("Starting conditional sampling (cond on prefix).")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    cond_cfg = sampling_cfg.cond_prefix

    ckpt_name = config.checkpointing.resume_ckpt_path.split("/")[-1]
    metadata = dict(
        checkpoint_name=ckpt_name,
        num_samples=cond_cfg.num_samples,
        from_ema=cond_cfg.from_ema,
        dataset=cond_cfg.dataset,
        seq_len=cond_cfg.seq_len,
        prefix_len=cond_cfg.prefix_len,
        num_cont_per_prefix=cond_cfg.num_cont_per_prefix,
        min_seq_len=cond_cfg.min_seq_len,
        num_steps=cond_cfg.num_steps,
        sampler=cond_cfg.sampler,
        add_bos=cond_cfg.add_bos,
        add_eos=cond_cfg.add_eos,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "cond" / save_fname
    assert not save_path.exists(), save_fname
    # Extract args from cfg
    bs = cond_cfg.batch_size
    prefix_len = cond_cfg.prefix_len
    num_steps = cond_cfg.num_steps
    seq_len = cond_cfg.seq_len
    target_num_samples = cond_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert target_num_samples % (tot_num_device * bs) == 0
    # Load prefix dataset
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(200 + fabric.global_rank)

    if fabric.global_rank > 0:
        fabric.barrier()  # Make sure that only the first device does the preprocessing

    dataset = dataloader.get_dataset(
        cond_cfg.dataset,
        tokenizer,
        mode="valid",
        cache_dir=config.data_preprocess.data_cache,
        num_proc=config.trainer.devices * config.loader.num_workers,
        min_seq_len=cond_cfg.min_seq_len,
        seq_len=seq_len,
        group_text=False,
        remove_text=True,
        add_bos=cond_cfg.add_bos,
        add_eos=cond_cfg.add_eos,
    )

    if fabric.global_rank == 0:
        fabric.barrier()  # Make sure the data was preprocessed on one device before starting

    assert len(dataset) >= target_num_samples
    dataset = dataset.select(range(cond_cfg.num_samples))

    pl_module = module
    fabric.to_device(pl_module)

    if cond_cfg.from_ema:
        pl_module.store_ema()

    all_samples = []
    start = fabric.global_rank * bs
    stop = target_num_samples
    end = fabric.world_size * bs
    for idx in trange(
        start,
        stop,
        end,
        desc=f"Sampling with n_steps={num_steps}, seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        docs = dataset[idx : idx + bs]["input_ids"]
        prefixes = docs[:, :prefix_len]

        def project_fn(batch):
            batch[:, :prefix_len] = prefixes
            return batch

        # Generate potentially multiple continuations per prefix (typically 5)
        for _ in range(cond_cfg.num_cont_per_prefix):
            with fabric.autocast():
                out = pl_module.sample(
                    n_samples=bs,
                    num_steps=num_steps,
                    seq_len=seq_len,
                    sampler=cond_cfg.sampler,
                    add_bos=cond_cfg.add_bos,
                    add_eos=cond_cfg.add_eos,
                    cache_preds=cond_cfg.cache_preds,
                    project_fn=project_fn,
                )
            out = fabric.all_gather(data=out)
            if fabric.global_rank == 0:
                # unstack after all_gather
                if out.ndim == 3:
                    out = rearrange(out, "dev bs l -> (dev bs) l")
                all_samples.append(out.cpu())
            del out

    # Join and save to disk
    if fabric.global_rank == 0:
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_samples = all_samples[:target_num_samples * cond_cfg.num_cont_per_prefix]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        references = dataset[:target_num_samples]["input_ids"].numpy()
        np.savez(
            save_path, samples=all_samples, references=references, metadata=metadata
        )
        logger.info(f"Saved samples in {save_path}")

    if cond_cfg.from_ema:
        pl_module.restore_ema()

    fabric.barrier()


def _eval_suffix_nll_generators(module: MultiRoundSDTT, config, prefix: torch.Tensor, suffix):
    N = len(suffix)
    device = module.device
    batch_size = config.eval.lambada_openai.batch_size
    num_samples = config.eval.lambada_openai.num_samples
    add_eos = config.eval.lambada_openai.add_eos
    assert num_samples % batch_size == 0

    all_t = module._sample_t(num_samples, device=module.device)
    full_sentence = torch.cat([prefix, suffix], dim=-1, ).repeat(batch_size, 1).to(module.device)

    for idx in range(0, num_samples, batch_size):
        curr_t = all_t[idx: idx + batch_size]
        sigma, move_chance, dsigma = module._t_to_sigma(curr_t)
        sigma = sigma.squeeze(-1)

        xt = module.q_xt(full_sentence, move_chance)
        xt[:, :len(prefix)] = full_sentence[:, :len(prefix)]
        if add_eos:
            xt[:, -1] = full_sentence[:, -1]

        y = full_sentence
        scale = (dsigma / torch.expm1(sigma))[:, None]

        yield xt.to(device), y.to(device), scale.to(device), sigma.to(device), curr_t.to(device)


@torch.no_grad
def eval_suffix_nll(config, module: MultiRoundSDTT, prefix, suffix, sigma):
    """
    1. Generate all ways to mask the suffix.
    2. Evaluate the loss over all possible maskings
    3. Average over all possible masking
    """

    all_losses = []
    for xt, y, scale, sigma, t in _eval_suffix_nll_generators(module, config, prefix, suffix):
        preds = module(xt, sigma).log_softmax(-1)

        loss = - torch.gather(preds, dim=-1, index=y[..., None])[..., 0]
        is_masked = xt == module.mask_index
        loss = torch.where(is_masked.to(bool), loss, 0.0) * scale

        loss = loss.sum(-1)
        loss = loss.mean()
        all_losses.append(float(loss))

    return float(np.mean(all_losses))


@torch.no_grad
def eval_lambada(module: MultiRoundSDTT):
    logger.info("Starting eval acc/ppl on openai lambada")
    config = module.config
    lambada_cfg = config.eval.lambada_openai

    if config.eval.lambada_openai.from_ema:
        module.store_ema()

    tokenizer = module.tokenizer

    dataset = dataloader.get_dataset(
        "EleutherAI/lambada_openai",
        tokenizer,
        mode="test",
        cache_dir=config.data_preprocess.data_cache,
        num_proc=config.trainer.devices * config.loader.num_workers,
        group_text=False,
        remove_text=False,
        add_bos=lambada_cfg.add_bos,
        add_eos=lambada_cfg.add_eos,
    )

    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert tot_num_device == 1, "Code only works with one device"

    pl_module = module
    pl_module = pl_module.cuda()
    t = torch.tensor([pl_module.sampling_eps], device="cuda")
    sigma = pl_module._t_to_sigma(t)[0][0]

    all_losses = []
    all_last_correct = []
    add_eos = lambada_cfg.add_eos

    for idx in trange(
        len(dataset),
        desc="Evaluating lambada..."
    ):
        prefix = dataset[idx]["prefix_ids"]
        suffix = dataset[idx]["suffix_ids"]
        suffix_mask = suffix.clone()
        if add_eos:
            suffix_mask[:-1] = pl_module.mask_index
        else:
            suffix_mask[:] = pl_module.mask_index

        input_ids = torch.cat([prefix, suffix_mask]).cuda().reshape(1, -1)
        preds = pl_module(input_ids, sigma)

        assert pl_module.mask_index == preds.shape[-1] - 1
        greedy_tokens = preds[0, :, :-1].argmax(-1)
        suff_len = len(suffix)

        if add_eos:
            correct = greedy_tokens[-suff_len:-1].cpu() == suffix[:-1]
            correct = correct.all().item()

            loss = eval_suffix_nll(config, pl_module, prefix, suffix, sigma)

            all_losses.append(loss)
            all_last_correct.append(correct)

        else:
            raise NotImplementedError

    acc = np.mean(all_last_correct)
    avg_loss = np.mean(all_losses)

    from run_eval import CURR_DATETIME_STR
    csv_save_path = Path(os.getcwd()) / "csv" / CURR_DATETIME_STR / "lambada.csv"
    header = [
        "num_samples",
        "from_ema",
        "add_bos",
        "add_eos",
        "checkpoint_path",
        "acc",
        "ppl",
    ]

    row = [
        lambada_cfg.num_samples,
        lambada_cfg.from_ema,
        lambada_cfg.add_bos,
        lambada_cfg.add_eos,
        config.checkpointing.resume_ckpt_path,
        float(acc),
        float(np.exp(avg_loss)),
    ]

    df = pd.DataFrame([row], columns=header)
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_save_path)
    logger.info(f"Lambada results: \n{df}\n{'=' * 50}")

    if config.eval.lambada_openai.from_ema:
        module.restore_ema()

