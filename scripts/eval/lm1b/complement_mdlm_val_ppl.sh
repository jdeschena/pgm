# Important note: the json with the results will be saved in ./outputs/lm1b/complement_mdlm/val_ppl/last_ckpt.json since hydra changes the cwd
python -u -m main \
    mode=ppl_eval \
    eval.checkpoint_path=PATH-TO-YOUR-CHECKPOINT \
    data=lm1b  \
    data.cache_dir=YOUR-LOCAL-CACHE-PATH \
    model=small \
    model.length=128 \
    algo=mdlm \
    loader.eval_batch_size=32 \
    hydra.run.dir=./outputs/lm1b/complement_mdlm \
    eval.results_json_path="val_ppl/last_ckpt.json"
