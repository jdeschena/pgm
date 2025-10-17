# Partition Generative Models - Masked Modeling Without Masks
By [Justin Deschenaux](https://jdeschena.github.io), [Lan Tran](https://github.com/tranhuonglan), [Caglar Gulcehre](https://x.com/caglarml?lang=en)


[![arXiv](https://img.shields.io/badge/arXiv-2505.18883-red.svg)](https://arxiv.org/abs/2506.10892v1)
[![HuggingFace](https://img.shields.io/badge/🤗-Huggingface-blue)](hhttps://huggingface.co/jdeschena/pgm)
[![Google Colab - Text](https://img.shields.io/badge/Google%20Colab%20(Text)-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/drive/1dhAh4hJ5s89PcQWlE7PoFxF2WBebM6g1)
[![Google Colab - Images](https://img.shields.io/badge/Google%20Colab%20(Images)-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/drive/1eRqnK3vasDFqxEq99LqboyoqQyfFI4AC)

**TL;DR: Partition Generative Models (PGMs) speed up parallel generation by partitioning tokens and using sparse attention instead of masking.**

<div align="center">
  <img src="https://jdeschena.github.io/pgm/static/images/pgm_vs_mgm.jpg" width="100%">
</div>

## Try Our Models
Try our models directly on Google Colab!  
- [Image modeling notebook](https://colab.research.google.com/drive/1eRqnK3vasDFqxEq99LqboyoqQyfFI4AC)  
- [Language modeling notebook](https://colab.research.google.com/drive/1dhAh4hJ5s89PcQWlE7PoFxF2WBebM6g1)  

## Getting started locally
To get started, install the dependencies in `requirements.txt`. The requirements *do not* contain the `numpy` and `torch` dependencies, since these need to be set in combination. For us, we work in docker containers, built from `nvcr.io/nvidia/pytorch:25.02-py3`, which uses `torch==2.7.0` and `numpy==1.26.4`.

### Reproducing the Results
Our experiments for text and images are based on two main codebases. For text experiments, we build upon the [Duo](https://github.com/s-sahoo/duo) codebase. For image experiments, we adapt the [Halton MaskGIT](https://github.com/valeoai/Halton-MaskGIT) codebase. As a result, we maintain separate branches for text and image experiments:

- Text experiments (besides distillation) are on the `text_pretrain` branch.
- Image experiments are on the `image_pretrain` branch.

Additionally, we distilled models using [SDTT](https://github.com/jdeschena/sdtt). The relevant code can be found on the `text_distill_sdtt` branch, which is a slight adaptation of the [SDTT](https://github.com/jdeschena/sdtt) codebase. You can find further instructions in the respective branches.

### Checkpoints
We release checkpoints trained on OpenWebText (1M steps, distilled and undistilled) and ImageNet (500k steps) on [🤗 Huggingface](https://huggingface.co/jdeschena/pgm). The checkpoints on HuggingFace are directly compatible with the code without conversion.

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
