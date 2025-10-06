# Partition Generative Modeling - Pre-training on text

## Structure of the repo
1. ```main.py```: Entrypoint, starts training and evaluation
2. ```trainer_base.py```: Boiler Plate stuff
3. ```algo.py```: Implementations of MDLM and PGM 
4. ```dataloader.py```: Data preparation
5. ```utils.py```: LR scheduler, logging, `fsspec` handling
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748), AR transformer, and our custom Partition Transformer, based on the DiT baseline.
7. ```configs/```: Hydra config files
8. ```scripts/```: Shell scripts for training/evaluation


## Setting up the project
* To install the requirements, please check the main branch.
* To **train** models on LM1B and OpenWebText, use the scripts in `scripts/train`. 
* To **evaluate** checkpoints, use the scripts in `scripts/eval`.
* To **distill** checkpoints, have a look at the `text_distill_sdtt` branch.
