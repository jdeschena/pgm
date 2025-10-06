python -u -m main \
    mode=latency_train \
    model=small-encoder-decoder \
    model.encoder.n_blocks=8 \
    model.decoder.n_blocks=8 \
    algo=partition-mdlm \
    latency_train.num_timings=20 \
    latency_train.num_warmup=2 \
    loader.batch_size=32 \
    hydra.run.dir=./outputs/owt/pgm_8_8 \
