from huggingface_hub import hf_hub_download
import torch
from default_config import get_default_config
from omegaconf import OmegaConf
import dataloader
import algo


def _patch_config_mdlm(config):
  config.data.tokenizer_name_or_path = 'gpt2'
  return config


def _patch_config_pgm(config, variant):
  assert variant in ('6-6-dim1024', '8-8')
  config.data.tokenizer_name_or_path = 'gpt2'
  config.neg_infinity_mode = 'true-inf'
  # update model
  model_args = OmegaConf.create(dict(
    name='small',
    type='encoder-decoder',
    hidden_size=768 if variant == '8-8' else 1024,
    cond_dim=128,
    length=1024,
    n_heads=12 if variant == '8-8' else 16,
    mlp_ratio=4,
    scale_by_sigma=True,
    dropout=0.1,
    tie_word_embeddings=False,
    encoder=dict(n_blocks=8 if variant == '8-8' else 6),
    swap=dict(
      pre_query_mode='learn+freqs',
      query_process_mode='linear',
      normalize_mode='layernorm'),
    decoder=dict(n_blocks=8 if variant == '8-8' else 6)
  ))
  # Update algo
  algo_args = OmegaConf.create(dict(
    name='partition-mdlm',
    backbone='encoder-decoder',
    parameterization='subs',
    time_conditioning=False,
    T=0,
    subs_masking=False,
    causal_attention=False,
    ignore_bos=False,
    loss_type='elbo',
    sampling_mode='efficient-uniform',
    post_process_mode='efficient'
  ))

  config.model = model_args
  config.algo = algo_args
  return config


class WrapperBase:
  def __init__(self, device='cuda', filename=None):
    self.config = get_default_config()
    self._patch_config()

    tokenizer = dataloader.get_tokenizer(self.config)
    if self.config.algo.name == 'mdlm':
      diffusion_model = algo.MDLM
    elif self.config.algo.name == 'partition-mdlm':
      diffusion_model = algo.PartitionMDLM
    else:
      raise ValueError(self.config.algo.name)

    self.model = diffusion_model(self.config, 
                                 tokenizer=tokenizer)
    self.model.to(device)
    ckpt_path = hf_hub_download(repo_id='jdeschena/pgm', 
                                filename=filename)
    content = torch.load(ckpt_path, weights_only=False, 
                         map_location='cpu')
    self.model.load_state_dict(content['state_dict'])
    self.model.ema.load_state_dict(content['ema'])
    self.model._eval_mode()

  def generate(self, num_samples=4, num_steps=32, p_nucleus=1.0,
               return_tokens=False):
    self.config.sampling.p_nucleus = p_nucleus
    tokens = self.model.generate_samples(num_samples, 
                                         num_steps)
    if return_tokens:
      return tokens
    else:
      text = self.model.tokenizer.batch_decode(tokens)
      return text


class MDLMWrapper(WrapperBase):
  def __init__(self, device='cuda', distilled=False):
    if distilled:
      filename = 'mdlm_sdtt_50k.ckpt'
    else:
      filename = 'mdlm.ckpt'
    WrapperBase.__init__(self, device, filename)

  def _patch_config(self):
    return _patch_config_mdlm(self.config)


class PGMWrapper(WrapperBase):
  def __init__(self, device='cuda', variant='6-6-dim1024', 
               distilled=False):
    self.variant = variant
    if variant == '6-6-dim1024':
      if distilled:
        filename = 'pgm_6_6_dim1024_sdtt_fp32_50k.ckpt'
      else:
        filename = 'pgm_6_6_dim1024.ckpt'
    elif variant == '8-8':
      if distilled:
        filename = 'pgm_8_8_sdtt_fp32_50k.ckpt'
      else:
        filename = 'pgm_8_8.ckpt'
    else:
      raise ValueError(variant)
    WrapperBase.__init__(self, device, filename)

  def _patch_config(self):
    out = _patch_config_pgm(self.config, self.variant)
    return out
