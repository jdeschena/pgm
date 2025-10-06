#!/bin/sh
#SBATCH --job-name=sdtt-8-8
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

export WANDB_API_KEY=<YOUR-API-KEY>
export TRITON_CACHE_DIR=/opt

srun \
    python -u -m sdtt.main \
    parameterization.num_distill_steps=2 \
    parameterization.loss_precision=32 \
    model=encoder-decoder-small \
    tokenizer.name=gpt2-large \
    model.swap.pre_query_mode=learn+freqs \
    model.swap.query_process_mode=linear \
    model.swap.normalize_mode=layernorm \
    model.decoder.self_attn_first=False \
    model.encoder.n_blocks=8 \
    model.decoder.n_blocks=8 \
    parameterization=partition-sdtt \
    parameterization.start_from_hf=False \
    parameterization.checkpoint_path=<ckpt-path> \
    loader.global_batch_size=128 \
    loader.batch_size=16 \
    trainer.max_steps=50000 \
    trainer.devices=4 \
    trainer.num_nodes=2 \
    hydra.run.dir="./outputs/encoder_decoder_8-8/" \
    loader.num_workers=8 \
    compile=False \
    trainer.val_check_interval=5000 \
    data_preprocess.data_cache=<path-to-cache-data> \
    wandb.project=partition-sdtt









