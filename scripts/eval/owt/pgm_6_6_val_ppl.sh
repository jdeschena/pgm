# Important note: the json with the results will be saved in ./outputs/owt/pgm_6_6_dim1024/val_ppl/last_ckpt.json since hydra changes the cwd
python -u -m main \
    mode=ppl_eval \
    eval.checkpoint_path=PATH-TO-YOUR-CHECKPOINT \
    data=openwebtext-split  \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small-encoder-decoder \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.encoder.n_blocks=6 \
    model.decoder.n_blocks=6 \
    algo=partition-mdlm \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/pgm_6_6_dim1024 \
    eval.results_json_path="val_ppl/last_ckpt.json"
