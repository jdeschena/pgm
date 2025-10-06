python -u -m main \
    mode=latency_train \
    model=small-encoder-decoder \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.encoder.n_blocks=6 \
    model.decoder.n_blocks=6 \
    algo=partition-mdlm \
    latency_train.num_timings=20 \
    latency_train.num_warmup=2 \
    loader.batch_size=32 \
    hydra.run.dir=./outputs/owt/pgm_6_6_dim1024 \
