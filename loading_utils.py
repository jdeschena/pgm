from huggingface_hub import hf_hub_download
from trainer.cls_trainer import MaskGIT
import torch
from default_config import get_default_config
from torchvision.transforms import ToPILImage


def _patch_default_config(config):
  config.is_master = True
  config.global_rank = 0
  config.is_multi_gpus = False
  config.disable_wandb = True
  config.resume = False
  config.bsize = 1
  config.data_folder = ""
  config.eval_folder = ""
  config.iter = -1
  config.compile = False
  config.register = 1


class WrapperBase:
  def __init__(self, sampler='halton', device='cuda', 
               filename=None):
    self.config = get_default_config()
    self.config.sampler = sampler
    self._patch_config()
    
    self.model = MaskGIT(self.config)
    self.device = device
    ckpt_path = hf_hub_download(repo_id='jdeschena/pgm', 
                                filename=filename)
    content = torch.load(ckpt_path, weights_only=False, 
                         map_location='cpu')
    self.model.vit.load_state_dict(content['model_state_dict'])
    self.model.vit.to(device)
    self.to_pil = ToPILImage()

  def generate(self, num_samples=4, num_steps=32, cfg_w=0, 
               classes=None, verbose=True):
    if type(classes) is int:
      classes = torch.ones((num_samples,), dtype=torch.long, 
                           device=self.device) * classes
    else:
      assert len(classes) == num_samples

    self.model.sampler.w = cfg_w
    self.model.sampler.step = num_steps

    out, _, _ = self.model.sampler(self.model, 
      nb_sample=num_samples, labels=classes, verbose=verbose)
    out = (out + 1) / 2
    # Convert tensor to PIL images
    pil_images = []
    for i in range(out.shape[0]):
        img = out[i].cpu().clamp(0, 1)
        pil_images.append(self.to_pil(img))
    return pil_images

  def _patch_config(self):
    _patch_default_config(self.config)


class MaskGITWrapper(WrapperBase):
  def __init__(self, sampler='halton', device='cuda'):
    WrapperBase.__init__(self, sampler, device, 
                         filename='maskgit_100epochs.ckpt')


class PGMWrapper(WrapperBase):
  def __init__(self, sampler='halton', device='cuda', 
               variant='14-10'):
    self.variant = variant
    if variant == '12-12':
      filename = 'images_pgm_12_12_100epochs.ckpt'
    elif variant == '14-10':
      filename = 'images_pgm_14_10_100epochs.ckpt'
    else:
      raise ValueError(variant)
    WrapperBase.__init__(self, sampler, device, filename)

  def _patch_config(self):
    super()._patch_config()
    self.config.model.name = 'encoder-decoder'
    self.config.register = 2
    self.config.bos_value = 16385
    if self.variant == '12-12':
      self.config.model.encoder.n_blocks = 12
      self.config.model.decoder.n_blocks = 12
    elif self.variant == '14-10':
      self.config.model.encoder.n_blocks = 14
      self.config.model.decoder.n_blocks = 10
    else:
      raise ValueError(self.variant)

