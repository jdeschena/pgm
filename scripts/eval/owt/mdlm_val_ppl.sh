# Important note: the json with the results will be saved in ./outputs/owt/mdlm/val_ppl/last_ckpt.json since hydra changes the cwd
python -u -m main \
    mode=ppl_eval \
    eval.checkpoint_path=PATH-TO-YOUR-CHECKPOINT \
    data=openwebtext-split  \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small \
    algo=mdlm \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/owt/mdlm \
    eval.results_json_path="val_ppl/last_ckpt.json"
