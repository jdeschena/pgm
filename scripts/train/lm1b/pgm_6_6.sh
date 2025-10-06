python -u -m main \
    neg_infinity_mode="true-inf" \
    data=lm1b \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small-encoder-decoder \
    model.length=128 \
    algo=partition-mdlm \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=16 \
    trainer.num_nodes=2 \
    trainer.devices=4 \
    trainer.val_check_interval=20_000 \
    trainer.precision="32-true" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    wandb.name="pgm_6_6_lm1b" \
    hydra.run.dir=./outputs/lm1b/pgm_6_6

