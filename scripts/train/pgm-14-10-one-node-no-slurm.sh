mkdir -p outputs/pgm_14_10

echo "START TIME: $(date)"

######################
### Set enviroment ###
######################
export WANDB_API_KEY=<YOUR-WANDB-API-KEY>


PYTHON_FILE=main.py
PYTHON_ARGS=" \
    sampler=halton \
    model.name=\"encoder-decoder\" \
    model.encoder.n_blocks=14 \
    model.decoder.n_blocks=10 \
    register=2 \
    global_bsize=256 \
    resume=False \
    compile=True \
    run_name=\"pgm_14_10\" \
    bos_value=16385 \
    data_folder=./data_cache/imagenet_tokenized \
    vqgan_folder=./vqgan_cache/llamagen/vq_ds8_c2i.pt \
    saved_folder=\"outputs/pgm_14_10/\" \
    "

export CMD="$PYTHON_FILE $PYTHON_ARGS"

python -m torch.distributed.run $CMD

echo "END TIME: $(date)"