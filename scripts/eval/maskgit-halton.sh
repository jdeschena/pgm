#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

mkdir -p outputs/maskgit

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x


export WANDB_API_KEY<your-wandb-key>

GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"
######################

######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((12535 + ${SLURM_JOBID} % 10000))

echo "Running on $SLURM_NNODES nodes"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
LAUNCHER=" --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "


for w in 0 1 4; do
    PYTHON_FILE=main.py
    PYTHON_ARGS=" \
        model.name=\"maskgit\" \
        register=1 \
        sampler=halton \
        cfg_w=$w \
        global_bsize=32 \
        test_only=True \
        resume=True \
        run_name=sample-maskgit-w-$w-e100 \
        data_folder=./data_cache/imagenet_tokenized \
        eval_folder=./data_cache/imagenet \
        vqgan_folder=./vqgan_cache/llamagen/vq_ds8_c2i.pt \
        saved_folder=outputs/maskgit \
        resume_checkpoint=\"epoch_100.pth\" \
        "


    export CMD="$LAUNCHER $PYTHON_FILE $PYTHON_ARGS"

    echo $CMD

    # srun error handling:
    # --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
    SRUN_ARGS=" \
        -ul \
        --jobid $SLURM_JOB_ID \
        --wait 60 \
        --environment=pgm_maskgit \
        --container-workdir=$PWD \
        "

    # bash -c is needed for the delayed interpolation of env vars to work
    srun $SRUN_ARGS bash -c " \
        python -m torch.distributed.run $CMD"

done

echo "END TIME: $(date)"
