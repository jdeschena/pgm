python -u -m main \
    mode=latency_train \
    model=small \
    algo=mdlm \
    latency_train.num_timings=20 \
    latency_train.num_warmup=2 \
    loader.batch_size=32 \
    hydra.run.dir=./outputs/owt/mdlm
