mkdir -p outputs/maskgit

echo "START TIME: $(date)"

######################
### Set enviroment ###
######################
export WANDB_API_KEY=<YOUR-WANDB-API-KEY>


PYTHON_FILE=main.py
PYTHON_ARGS=" \
    global_bsize=256 \
    resume=False \
    compile=True \
    run_name=\"maskgit\" \
    bos_value=16385 \
    data_folder=./data_cache/imagenet_tokenized \
    vqgan_folder=./vqgan_cache/llamagen/vq_ds8_c2i.pt \
    saved_folder=\"outputs/maskgit/\" \
    test_only=False
    "

export CMD="$PYTHON_FILE $PYTHON_ARGS"

python -m torch.distributed.run $CMD

echo "END TIME: $(date)"