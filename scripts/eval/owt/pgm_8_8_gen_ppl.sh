# Important note: the json with the results will be saved in ./outputs/owt/pgm_8_8/samples
python -u -m main \
    mode=sample_eval \
    neg_infinity_mode="true-inf" \
    eval.checkpoint_path=PATH-TO-YOUR-CHECKPOINT \
    model=small-encoder-decoder \
    model.encoder.n_blocks=8 \
    model.decoder.n_blocks=8 \
    algo=partition-mdlm \
    sampling.steps=128 \
    sampling.num_sample_batches=32 \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/pgm_8_8 \
