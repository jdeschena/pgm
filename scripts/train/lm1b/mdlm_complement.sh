python -u -m main \
    data=lm1b \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small \
    model.length=128 \
    algo=complement-mdlm \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=16 \
    trainer.num_nodes=2 \
    trainer.devices=4 \
    trainer.val_check_interval=20_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    wandb.project=pgm \
    wandb.notes=null \
    wandb.name="mdlm_complement_lm1b" \
    hydra.run.dir=./outputs/lm1b/mdlm_complement
