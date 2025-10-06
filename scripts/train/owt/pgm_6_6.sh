python -u -m main \
    neg_infinity_mode="true-inf" \
    data=openwebtext-split  \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small-encoder-decoder \
    algo=partition-mdlm \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    trainer.num_nodes=4 \
    trainer.devices=4 \
    trainer.val_check_interval=20_000 \
    trainer.precision="32-true" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    wandb.name="pgm_6_6_owt" \
    hydra.run.dir=./outputs/owt/pgm_6_6
