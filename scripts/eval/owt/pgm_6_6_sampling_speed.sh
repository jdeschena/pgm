python -u -m main \
    mode=latency_sample \
    model=small-encoder-decoder \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.encoder.n_blocks=6 \
    model.decoder.n_blocks=6 \
    algo=partition-mdlm \
    sampling.num_sample_batches=20 \
    sampling.steps=128 \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/pgm_6_6_dim1024 \
