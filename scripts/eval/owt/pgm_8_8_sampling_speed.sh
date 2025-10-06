python -u -m main \
    mode=latency_sample \
    model=small-encoder-decoder \
    model.encoder.n_blocks=8 \
    model.decoder.n_blocks=8 \
    algo=partition-mdlm \
    sampling.num_sample_batches=20 \
    sampling.steps=128 \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/pgm_8_8 \
