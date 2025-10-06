python -u -m main \
    mode=latency_sample \
    model=small \
    algo=mdlm \
    sampling.num_sample_batches=20 \
    sampling.steps=128 \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/mdlm
