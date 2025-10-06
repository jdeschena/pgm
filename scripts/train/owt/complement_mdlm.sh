python -u -m main \
    data=openwebtext-split \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small \
    algo=complement-mdlm \
    loader.batch_size=16 \
    loader.eval_batch_size=32 \
    loader.num_workers=16 \
    trainer.num_nodes=4 \
    trainer.devices=4 \
    trainer.val_check_interval=20_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    eval.compute_generative_perplexity=True \
    wandb.name="complement_mdlm_owt" \
    hydra.run.dir=./outputs/owt/complement_mdlm
