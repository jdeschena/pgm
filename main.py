import os

import hydra
import lightning as L
from omegaconf import OmegaConf
import torch

from pathlib import Path
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer

from data import utils as dutils
from data import dataloader
from utils import add_resolvers, prepare_logger, rm_null_values
from loading_utils import get_diffusion, get_diffusion_module
from lightning.pytorch.loggers import TensorBoardLogger


def train(config):
    logger.info("Starting training")

    if config.get("wandb", None):
        # remove entries with null keys
        wandb_args_dict = OmegaConf.to_object(config.wandb)
        wandb_args_dict = rm_null_values(wandb_args_dict)

        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=OmegaConf.to_object(config),
            **wandb_args_dict,
        )
    else:
        wandb_logger = None

    tb_logger = TensorBoardLogger("tb_logs", name="logs")
    loggers = tb_logger if wandb_logger is None else (wandb_logger, tb_logger)

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and dutils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
        logger.info(f"Training starting from checkpoint at {ckpt_path}")
    else:
        ckpt_path = None
        logger.info("Training starting from scratch (no checkpoint to reload)")

    # Load callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    datamodule = dataloader.TextDiffusionDataModule(config, tokenizer)

    model = get_diffusion(config, tokenizer)
    if config.compile:
        model.backbone = torch.compile(model.backbone)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=loggers,
    )

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


def sample(config):
    logger.info("Mode: sampling...")
    param_cfg = config.parameterization
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)

    module = get_diffusion_module(config)
    diffusion = get_diffusion(config, tokenizer)

    checkpoint_path = config.checkpointing.resume_ckpt_path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() or not checkpoint_path.name.endswith(".ckpt"):
        logger.warning(
            f"Path `{checkpoint_path.absolute()}` does not exist. Sampling with untrained/original checkpoint."
        )
    else:
        logger.info(f"Sampling with checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if config.compile:
            diffusion.backbone = torch.compile(diffusion.backbone)
            for k in ckpt.keys():
                assert "_orig_mod" in k, "Cannot use compile=True with this checkpoint"
        else:
            ckpt["state_dict"] = {
                k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()
            }

        diffusion.load_state_dict(ckpt["state_dict"])
        diffusion.load_ema_from_checkpoint(ckpt)

    run_uncond = param_cfg.sampling.uncond.run
    run_cond_prefix = param_cfg.sampling.cond_prefix.run
    assert (
        run_uncond or run_cond_prefix
    ), "config.parameterization.sampling.{cond_prefix|uncond}.run must be set"

    if run_uncond:
        module.sample_uncond(diffusion)

    if run_cond_prefix:
        module.sample_cond_prefix(diffusion)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    if hasattr(config, "seed"):
        L.seed_everything(config.seed)
    else:
        L.seed_everything(0)

    OmegaConf.save(config=config, f=Path(os.getcwd()) / "config.yaml")

    #from torch import multiprocessing as mp
    #mp.set_start_method("forkserver", force=True)

    logger.info(f"Arguments:\n{OmegaConf.to_yaml(config, resolve=True)}")
    mode = config.mode

    if mode == "train":
        logger.add(Path(os.getcwd()) / "logs_train.txt")
        train(config)
    elif mode == "sample":
        logger.add(Path(os.getcwd()) / "logs_sample.txt")
        sample(config)
    elif mode == "eval":
        logger.add(Path(os.getcwd()) / "logs_eval.txt")
        samples_eval(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    add_resolvers()
    prepare_logger()
    main()
