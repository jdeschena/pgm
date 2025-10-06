#!/bin/sh
#SBATCH --job-name=mdlm
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

export WANDB_API_KEY=<YOUR-API-KEY>
export TRITON_CACHE_DIR=/opt

srun \
    python -u -m main \
    parameterization.num_distill_steps=2 \
    parameterization.loss_precision=32 \
    model=dit-orig-small \
    tokenizer.name=gpt2-large \
    parameterization.start_from_hf=False \
    parameterization.checkpoint_path=<path-to-checkpoint-trained-on-the-text-pretrained-branch> \
    loader.global_batch_size=128 \
    loader.batch_size=16 \
    trainer.max_steps=50000 \
    trainer.devices=4 \
    trainer.num_nodes=2 \
    hydra.run.dir="./outputs/mdlm/" \
    loader.num_workers=12 \
    compile=False \
    trainer.val_check_interval=5000 \
    data_preprocess.data_cache=<path-to-cache-data> \
    wandb.project=partition-sdtt \
    wandb.name=mdlm_sdtt
