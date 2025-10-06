# Partition Generative Models - Masked Modeling Without Masks
By [Justin Deschenaux](https://jdeschena.github.io), [Lan Tran](https://github.com/tranhuonglan), [Caglar Gulcehre](https://x.com/caglarml?lang=en)


[![arXiv](https://img.shields.io/badge/arXiv-2505.18883-red.svg)](https://arxiv.org/abs/2506.10892v1)
[![HuggingFace](https://img.shields.io/badge/🤗-Huggingface-blue)](hhttps://huggingface.co/jdeschena/pgm)

**TL;DR: Partition Generative Models (PGMs) speed up parallel generation by partitioning tokens and using sparse attention instead of masking.**

<div align="center">
  <img src="https://jdeschena.github.io/pgm/static/images/pgm_vs_mgm.jpg" width="60%">
</div>

## Getting Started
To get started, install the dependencies in `requirements.txt`. The requirements *do not* contain the `numpy` and `torch` dependencies, since these need to be set in combination. For us, we work in docker containers, built from `nvcr.io/nvidia/pytorch:25.02-py3`, which uses `torch==2.7.0` and `numpy==1.26.4`.

### Checkpoints
We release the raw checkpoints (distilled/undistilled) trained on OpenWebText (1M steps) and ImageNet (500k steps) on [Huggingface 🤗](https://huggingface.co/jdeschena/pgm). You need to download them from there if you want to run the evaluations.

### Trying the models
Once you have installed the dependencies, you should be able to sample from our models. You can try the PGMs trained on text in `notebooks/text.ipynb` and the PGMs trained on images in `notebooks/images.ipynb`. No need to download anything manually!

## Reproducing the Results
Our experiments for text and images are based on two main codebases. For text experiments, we build upon the [Duo](https://github.com/s-sahoo/duo) codebase. For image experiments, we adapt the [Halton MaskGIT](https://github.com/valeoai/Halton-MaskGIT) codebase. As a result, we maintain separate branches for text and image experiments:

- Text experiments (besides distillation) are on the `text_pretrain` branch.
- Image experiments are on the `image_pretrain` branch.

Additionally, we conduct experiments on distilled MDLM. The relevant code can be found on the `text_distill_sdtt` branch, which is a slight adaptation of our [SDTT](https://github.com/jdeschena/sdtt) codebase. Find further instructions on text/images in their respective branches

## Citation
```
@misc{deschenaux2025partitiongenerativemodelingmasked,
      title={Partition Generative Modeling: Masked Modeling Without Masks}, 
      author={Justin Deschenaux and Lan Tran and Caglar Gulcehre},
      year={2025},
      eprint={2505.18883},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.18883}, 
}
```