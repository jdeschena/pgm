# Partition Generative Modeling - Pre-training on image
For our image modeling experiments, we use a modified version of the [Halton MaskGIT](valeoai/Halton-MaskGIT) codebase. For the requirements, see the main branch.


## Structure of the repo
1. ```main.py```: Entrypoint, starts training and evaluation
2. ```extract_vq_features.py```: Script for extracting VQ features from images as preparation for data-preprocessing
3. ```extract_train_fid.py```: Script for extracting FID features from validation set images as preparation for evaluation
4. ```trainer/```: Main class handles training and evaluation loop, `cls_trainer.py` is built upon `abstract_trainer.py`
5. ```configs/```: Hydra config files
6. ```dataset/```: Define dataclass and dataloader
7. ```metrics/```: Functions/class for computing the FID, IS score
8. ```models/```: Maskgit network architectures and VQGAN architectures. Supports [DiT](https://arxiv.org/abs/2212.09748), DIT Masked Transformer, and our custom Partition Transformer, based on the DiT baseline.
9. ```sampler/```: Sampling strategies for training and evaluation, including Halton and Confidence-based samplers.
10. ```utils/```: logging, masking scheduler, reconstruct images from VQGAN latent space
11. ```scripts/```: Shell scripts for training/evaluation

## Setting up the project
Before training, you must download and tokenize ImageNet. We use huggingface to download Imagenet, but it is a gated dataset, hence you must first visit [Huggingface's Imagenet page](https://huggingface.co/datasets/timm/imagenet-1k-wds), login, and sign the agreement. Furthermore, you must generate a [read access token here](https://huggingface.co/settings/tokens). Afterwards, you can download the data and tokenizer checkpoint with
```bash
python download_data_llamagen.py \
    --data_cache_dir ./data_cache/imagenet \
    --vqgan_cache_dir ./vqgan_cache/llamagen \
    --hf_token <your_hf_read_token> \
    --num_proc 1 # Number of parallel downloads; increase for faster processing
```
Make sure to replace the `hf_token` by your actual huggingface read token, and use a larger `--num_proc` to download the data faster. Using 8 processes, it takes <20' at ~40MiB/s download speed to download the data and tokenizer.

After downloading the data, you must tokenize it, which is done by
```bash
python extract_vq_features.py \
    --data-folder ./data_cache/imagenet \
    --dest-folder ./data_cache/imagenet_tokenized \
    --vqgan-folder ./vqgan_cache/llamagen/vq_ds8_c2i.pt \
    --bsize 128 \
    --img-size 256 \
    --num-workers 8 \
    --compile
```
On a single NVIDIA A100-80GB, the largest power-of-two that can be used as a batch size is 128. In that case, tokenization should take around 1h40 on a single GPU.

## Training
The training scripts are located in `scripts/train`. We use a Slurm cluster with Pyxis to submit jobs using a Docker container (more details can be found in the main branch). Since you likely have a different environment, we also provide scripts that can run in a plain Python environment (look for files with the suffix `-one-node-no-slurm.sh`) on a single node. For details on running in multi-node settings without Slurm, please refer to the [original repository](https://github.com/valeoai/Halton-MaskGIT).

## Sampling
We evaluate the Fréchet Inception Distance (FID) and Inception Score (IS) by comparing 50k generated samples against 50k samples from the validation set. The scripts to compute the FID and IS are located in `scripts/eval`.

## Acknowledgement
- This branch builds upon the [HaltonMaskGIT repo](https://github.com/valeoai/Halton-MaskGIT). We are grateful to their authors.
