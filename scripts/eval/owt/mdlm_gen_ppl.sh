# Important note: the json with the results will be saved in ./outputs/owt/mdlm/samples
python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path=PATH-TO-YOUR-CHECKPOINT \
    model=small \
    algo=mdlm \
    sampling.steps=128 \
    sampling.num_sample_batches=32 \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/mdlm \
