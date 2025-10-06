import torch

from core import CoreLightning
from models.loading_utils import get_backbone
from tqdm import trange
from sampling.utils import top_k_top_p_filtering
from loguru import logger
from data.utils import params2key
from pathlib import Path
import os

from data import dataloader
from lightning.fabric import Fabric
from tqdm import trange
import numpy as np
from einops import rearrange
import lightning as L
from transformers import AutoTokenizer
import pandas as pd


class ARCore(CoreLightning):
    def __init__(self, config, tokenizer):
        CoreLightning.__init__(self, config)
        self.backbone = get_backbone(config, vocab_size=tokenizer.vocab_size)
        # self.loss_impl = config.parameterization.loss_implementation
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.bos_token_id = tokenizer.bos_token_id
        self.validate_config()

    def validate_config(self):
        assert self.config.time_conditioning is False
        assert self.config.parameterization.name == "ar"

    def forward(self, x):
        with torch.amp.autocast("cuda", dtype=torch.float32):
            logits = self.backbone(x)
        return logits

    def iter_params(self):
        return self.backbone.parameters()

    def loss(self, batch):
        x = batch[:, :-1]
        y = batch[:, 1:]

        output = self.backbone(x)
        logits = output.log_softmax(-1)
        loss = -torch.gather(logits, dim=-1, index=y[:, :, None])[:, :, 0]
        return loss

    def training_step(self, batch):
        x = batch["input_ids"]
        loss = self.loss(x)
        loss = loss.mean()

        self.log(
            name="train/loss",
            value=loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        x = batch["input_ids"]
        with torch.no_grad():
            loss = self.loss(x)
        loss = loss.mean()

        self.log(
            name="valid/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        batch_size,
        max_length=None,
        top_p=0.0,
        top_k=0,
        temperature=1.0,
        sample=True,
        random_start=True,
        prefix=None,
        verbose=False,
        return_logits=False,
    ):
        # TODO: add kv-caching to model
        n_new_toks = max_length

        if prefix is not None:
            assert not random_start, "cannot use random start with a prefix"
            input_ids = prefix.to(self.device)
            assert input_ids.shape[0] == batch_size, "dim 0 of input ids should match batch size"
            n_new_toks = max_length - prefix.shape[1]
        elif random_start:
            assert prefix is None, "prefix should be None if random start is True"
            input_ids = torch.randint(
                low=0, 
                high=self.vocab_size, 
                size=(batch_size, 1), 
                device=self.device,
            )
        else:
            # BOS token for gpt2
            n_new_toks -= 1
            input_ids = torch.tensor(
                [self.bos_token_id] * batch_size, device=self.device
            ).reshape(batch_size, 1)

        output_logits = []

        for _ in trange(n_new_toks, desc="Sampling...", disable=not verbose):
            logits = self(input_ids)[:, -1]
            if return_logits:
                output_logits.append(logits.clone())

            if sample:
                logits = top_k_top_p_filtering(logits, top_k, top_p)
                if temperature != 1.0:
                    logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)[..., None]

            input_ids = torch.concatenate([input_ids, next_token], dim=-1)

        # Do not return the first random token
        if random_start:
            input_ids = input_ids[:, 1:]

        if return_logits:
            output_logits = torch.stack(output_logits, dim=1)
            return input_ids, output_logits
        else:
            return input_ids


def sample_uncond(module):
    logger.info("Starting unconditional sampling.")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    uncond_cfg = sampling_cfg.uncond

    checkpoint_name = Path(config.checkpointing.resume_ckpt_path).name
    metadata = dict(
        num_samples=sampling_cfg.num_samples,
        seq_len=uncond_cfg.seq_len,
        random_start=sampling_cfg.random_start,
        checkpoint_name=checkpoint_name,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "uncond" / save_fname
    assert not save_path.exists(), save_path

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

    bs = sampling_cfg.batch_size
    seq_len = uncond_cfg.seq_len
    target_num_samples = sampling_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    # AR specific arguments
    top_p = sampling_cfg.top_p
    top_k = sampling_cfg.top_k
    temperature = sampling_cfg.temperature
    sample = sampling_cfg.sample
    random_start = sampling_cfg.random_start

    assert target_num_samples % (tot_num_device * bs) == 0
    n_sampling_rounds = target_num_samples // (tot_num_device * bs)

    all_samples = []
    for _ in trange(
        n_sampling_rounds,
        desc=f"Sampling with seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        with fabric.autocast():
            out = pl_module.sample(
                batch_size=bs,
                max_length=seq_len,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                sample=sample,
                random_start=random_start,
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

        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(save_path, samples=all_samples, metadata=metadata)
        logger.info(f"Saved {len(all_samples)} samples in {save_path}")

    
def sample_cond_prefix(module):
    logger.info("Starting conditional sampling (cond on prefix).")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    cond_cfg = sampling_cfg.cond_prefix
    checkpoint_name = Path(config.checkpointing.resume_ckpt_path).name

    metadata = dict(
        num_samples=sampling_cfg.num_samples,
        random_start=sampling_cfg.random_start,
        checkpoint_name=checkpoint_name,
        dataset=cond_cfg.dataset,
        seq_len=cond_cfg.seq_len,
        prefix_len=cond_cfg.prefix_len,
        num_cont_per_prefix=cond_cfg.num_cont_per_prefix,
        min_seq_len=cond_cfg.min_seq_len,
        add_bos=cond_cfg.add_bos,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "cond" / save_fname
    assert not save_path.exists(), save_path
    # Extract args from cfg
    bs = sampling_cfg.batch_size
    prefix_len = cond_cfg.prefix_len
    seq_len = cond_cfg.seq_len
    target_num_samples = sampling_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    # AR specific arguments
    top_p = sampling_cfg.top_p
    top_k = sampling_cfg.top_k
    temperature = sampling_cfg.temperature
    sample = sampling_cfg.sample
    random_start = sampling_cfg.random_start

    assert target_num_samples % (tot_num_device * bs) == 0
    # Load prefix dataset
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
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
        add_eos=False,
    )

    assert len(dataset) >= target_num_samples
    dataset = dataset.select(range(sampling_cfg.num_samples))

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(200 + fabric.global_rank)

    pl_module = module
    fabric.to_device(pl_module)

    all_samples = []
    start = fabric.global_rank * bs
    stop = target_num_samples
    end = fabric.world_size * bs
    for idx in trange(
        start,
        stop,
        end,
        desc=f"Sampling with seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        docs = dataset[idx : idx + bs]["input_ids"]
        prefixes = docs[:, :prefix_len]

        # Generate potentially multiple continuations per prefix (typically 5)
        for _ in range(cond_cfg.num_cont_per_prefix):
            with fabric.autocast():
                out = pl_module.sample(
                    batch_size=bs,
                    max_length=seq_len,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    sample=sample,
                    random_start=False,
                    prefix=prefixes
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
        all_samples = all_samples[:target_num_samples]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        references = dataset[:target_num_samples]["input_ids"].numpy()
        np.savez(
            save_path, samples=all_samples, references=references, metadata=metadata
        )
        logger.info(f"Saved samples in {save_path}")


@torch.no_grad
def eval_lambada(module: ARCore):
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

    all_losses = []
    all_last_correct = []
    add_eos = lambada_cfg.add_eos
    assert not add_eos

    for idx in trange(
        len(dataset),
        desc="Evaluating lambada..."
    ):
        prefix = dataset[idx]["prefix_ids"].to(module.device)
        suffix = dataset[idx]["suffix_ids"].to(module.device)
        # sample n tokens with argmax
        full_seq = torch.cat([prefix, suffix], dim=-1)
        full_gen = module.sample(batch_size=1, max_length=len(prefix) + len(suffix), sample=False, random_start=False, prefix=prefix[None], return_logits=False)
        logits = module(full_seq[None,])
        logits_target = suffix
        logits_preds = logits[0, len(prefix)-1: -1]

        loss = - torch.gather(logits_preds, dim=-1, index=logits_target[..., None]).sum()
        generated_suffix = full_gen[0, len(prefix):]

        correct = (generated_suffix == suffix).all().item()
        all_last_correct.append(correct)
        all_losses.append(float(loss))
        

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

