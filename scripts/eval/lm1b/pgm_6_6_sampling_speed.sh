# Important note: the json with the results will be saved in ./outputs/owt/mdlm/samples
python -u -m main \
    mode=latency_sample \
    model=small-encoder-decoder \
    model.length=128 \
    data=lm1b \
    algo=partition-mdlm \
    sampling.num_sample_batches=20 \
    sampling.steps=128 \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/lm1b/pgm_6_6 \
