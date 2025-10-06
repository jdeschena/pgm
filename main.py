import gc
import json
import os
import time

import fsspec
import hydra
import lightning as L
from lightning.fabric import Fabric
import numpy as np
import omegaconf
from pathlib import Path
import rich.syntax
import rich.tree
import torch
from tqdm import trange

import algo
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    try:
      print(f'First {k} tokens:', tokenizer.decode(first))
      print('ids:', first)
      print(f'Last {k} tokens:', tokenizer.decode(last))
      print('ids:', last)
    except:
      print('First tokens:', first)
      print('Last tokens:', last)

@torch.no_grad
def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  fabric = Fabric(accelerator=config.trainer.accelerator, 
                  devices=config.trainer.devices, 
                  num_nodes=config.trainer.num_nodes)
  fabric.launch()
  if config.eval.npz_path is None:
    model = _load_from_checkpoint(
      diffusion_model=diffusion_model,
      config=config,
      tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
      logger.info('Disabling EMA.')
      model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    assert num_strides == 1, stride_length == 1

    num_devices = config.trainer.num_nodes * config.trainer.devices
    if config.sampling.num_sample_batches % num_devices != 0:
      raise ValueError(
        f'Num. batches ({config.sampling.num_sample_batches}) '
        f'not divided by num. devices ({num_devices})')
    
    num_batch_per_device = (config.sampling.num_sample_batches 
                            // num_devices)
    total_num_samples = (config.sampling.num_sample_batches 
                        * config.loader.eval_batch_size)
    samples_fname = utils.vars_to_fname(
      predictor=config.sampling.predictor,
      noise_removal=config.sampling.noise_removal,
      use_float64=config.sampling.use_float64,
      p_nucleus=config.sampling.p_nucleus,
      num_samples=total_num_samples,
      inject_bos=config.sampling.inject_bos,
      num_sampling_steps=config.sampling.steps,
      ckpt_hash=utils.short_hash(config.eval.checkpoint_path)
    ) + ".json"

    model.to(fabric.device)
    if fabric.global_rank == 0:
      logger.info(f'Sampling {total_num_samples} samples.')
    all_text_samples = []
    all_np_samples = []
    for _ in trange(num_batch_per_device, desc=f'Sampling '
                    f'({config.sampling.steps} steps)', 
                    disable=fabric.global_rank != 0):
      if config.sampling.semi_ar:
        raise NotImplementedError
        _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
          stride_length=stride_length,
          num_strides=num_strides,
          dt=1 / config.sampling.steps)
        text_samples = intermediate_samples[-1]
        # Note: Samples generated using semi-ar method
        # need to to be processed before computing generative perplexity
        # since these samples contain numerous <|endoftext|> tokens
        # and diffusion.compute_generative_perplexity() discards
        # any text after the first EOS token.
      #else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      samples = fabric.all_gather(samples)
      if fabric.world_size > 1:
        samples = samples.flatten(0, 1)
      
      np_samples = samples.cpu().numpy()
      text_samples = model.tokenizer.batch_decode(samples)
      model.metrics.record_entropy(samples)
      all_text_samples.extend(text_samples)
      all_np_samples.append(np_samples)
  else:
    model = diffusion_model(config, tokenizer).to(fabric.device)
    content = np.load(config.eval.npz_path)
    all_np_samples = [content['samples']]
    all_text_samples = tokenizer.batch_decode(all_np_samples[0])
    assert config.eval.results_json_path is not None
    samples_fname = config.eval.results_json_path

  if fabric.global_rank == 0:
    logger.info("Evaluating generative perplexity...")
    all_np_samples = np.concatenate(all_np_samples)
    # Evaluate with retokenize and first chunk only (orig)
    model.metrics.record_generative_perplexity(
      all_text_samples, 
      config.model.length,
      retokenize=True,
      first_chunk_only=True,
      device=model.device)
    gen_ppl_first_chunk_retok = model.metrics.gen_ppl.compute().item()
    model.metrics.gen_ppl.reset()
    # Evaluate without retokenize and on the whole generation
    model.metrics.record_generative_perplexity(
      torch.tensor(all_np_samples).to(fabric.device), 
      config.model.length,
      retokenize=False,
      first_chunk_only=False,
      device=model.device)
    gen_ppl_all_no_retok = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()

    logger.info('Generative perplexity (retokenize, first '
               f'chunk only): {gen_ppl_first_chunk_retok:.5f}')
    logger.info('Generative perplexity (using ALL tokens '
               f'directly): {gen_ppl_all_no_retok:.5f}')
    logger.info(f'Sample entropy: {entropy:.5f}')
    samples_path = os.path.join(os.getcwd(), 'samples', 
                                samples_fname)
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)

    save_dict = dict(
      text=all_text_samples, 
      np_tokens_b64=utils.np_to_base64(all_np_samples), 
      gen_ppl_first_chunk_retok=gen_ppl_first_chunk_retok,
      gen_ppl_all_no_retok=gen_ppl_all_no_retok,
      entropy=entropy,
      ckpt_name=config.eval.checkpoint_path,
      config=omegaconf.OmegaConf.to_container(config, 
                                              resolve=True))
    with fsspec.open(samples_path, 'w') as f:
      json.dump(save_dict, f, indent=4)
    print('Samples saved at:', samples_path)
  fabric.barrier()


def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  results = trainer.validate(model, valid_ds)
  if config.eval.results_json_path is not None:
    save_path = Path(config.eval.results_json_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
      json.dump(results[0], f)
      print(f'Saved results to `{save_path}`')


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # Ensure dataset processing happens on rank 0 first
  fabric = L.Fabric(num_nodes=config.trainer.num_nodes, 
                  devices=config.trainer.devices, 
                  accelerator='cuda')
  fabric.launch()
  with fabric.rank_zero_first():
    train_ds, valid_ds = dataloader.get_dataloaders(
      config, tokenizer)
    
  _print_batch(train_ds, valid_ds, tokenizer)
  fabric.barrier()
  del fabric


  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, 
                            tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _sample_latency(diffusion_model, config, logger, tokenizer):
  logger.info('Starting to measure sampling latency and throughput.')
  model = diffusion_model(config, tokenizer).cuda()

  all_latencies = []
  all_throughputs = []
  # 2 warmup steps
  for _ in trange(config.sampling.num_sample_batches + 2, 
                  desc='Computing latency...'):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    samples = model.generate_samples(
      num_samples=config.loader.eval_batch_size,
      num_steps=config.sampling.steps,
      eps=1e-5)  # As per `restore_model_and_sample`
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    del samples
    gc.collect()
    throughput = (config.loader.eval_batch_size 
                  * config.model.length) / (end_time 
                                            - start_time)
    all_latencies.append(end_time - start_time)
    all_throughputs.append(throughput)
  # Skip warmup
  all_latencies = all_latencies[2:]
  all_throughputs = all_throughputs[2:]
  if config.eval.results_json_path is None:
    save_dir = Path(os.getcwd()) / 'sample_latency'
    save_fpath = os.path.join(save_dir, 
                    f'bs={config.loader.eval_batch_size}_'
                    f'num_steps={config.sampling.steps}.json')
  else:
    save_fpath = config.eval.results_json_path

  save_dir = os.path.dirname(save_fpath)
  os.makedirs(save_dir, exist_ok=True)
  mean_latency = float(np.mean(all_latencies))
  std_latency = float(np.std(all_latencies))
  mean_throughput = float(np.mean(all_throughputs))
  std_throughput = float(np.std(all_throughputs))
  
  with open(save_fpath, 'w') as f:
    json.dump({
      'mean_latency': mean_latency,
      'std_latency': std_latency,
      'mean_throughput': mean_throughput,
      'std_throughput': std_throughput
    }, f, indent=4)
  
  logger.info(
    f'Latency: {mean_latency:.4f} ± {std_latency:.4f} seconds')
  logger.info(
    f'Throughput: {mean_throughput:.4f} ± {std_throughput:.4f}'
    f'tokens/second')
  logger.info(f'Latency results saved at {save_fpath}')


def _latency_train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting to measure training latency and throughput.')
  model = diffusion_model(config, tokenizer).cuda()
  # FORWARD ONLY
  all_latencies = []
  all_throughputs = []
  cfg = config.latency_train
  for _ in trange(cfg.num_timings + cfg.num_warmup, 
                  desc='Timing forward (loss) computation...'):
    x0 = torch.randint(0, len(tokenizer), 
                       size=(config.loader.batch_size, 
                             config.model.length)).cuda()
    valid_tokens = torch.ones_like(x0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = model._loss(x0, valid_tokens, train_mode=True)
    torch.cuda.synchronize()
    end = time.perf_counter()

    latency = end - start
    throughput = (config.loader.batch_size 
                  * config.model.length) / latency
    all_latencies.append(latency)
    all_throughputs.append(throughput)

  all_latencies = all_latencies[cfg.num_warmup:]
  all_throughputs = all_throughputs[cfg.num_warmup:]
  forward_timings = dict(
    mean_latency=np.mean(all_latencies).item(),
    std_latency=np.std(all_latencies).item(),
    mean_throughput=np.mean(all_throughputs).item(),
    std_throughput=np.std(all_throughputs).item(),
  )

  # FORWARD + BACKWARD
  all_latencies = []
  all_throughputs = []

  cfg = config.latency_train
  for _ in trange(cfg.num_timings + cfg.num_warmup, 
    desc='Timing fwd+bwd (one gradient step) computation'):
    x0 = torch.randint(0, len(tokenizer), 
                       size=(config.loader.batch_size, 
                             config.model.length)).cuda()
    valid_tokens = torch.ones_like(x0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = model._loss(x0, valid_tokens, train_mode=True)
    loss.loss.backward()
    torch.cuda.synchronize()
    end = time.perf_counter()
    model.zero_grad()

    latency = end - start
    throughput = (config.loader.batch_size 
                  * config.model.length) / latency
    all_latencies.append(latency)
    all_throughputs.append(throughput)

  all_latencies = all_latencies[cfg.num_warmup:]
  all_throughputs = all_throughputs[cfg.num_warmup:]
  backward_timings = dict(
    mean_latency=np.mean(all_latencies).item(),
    std_latency=np.std(all_latencies).item(),
    mean_throughput=np.mean(all_throughputs).item(),
    std_throughput=np.std(all_throughputs).item(),
  )

  content = dict(
    fwd_timings=forward_timings,
    fwd_bwd_timings=backward_timings,
    model_config=omegaconf.OmegaConf.to_container(config.model, 
                                                  resolve=True)
  )
  save_dir = Path(os.getcwd()) / 'train_latency'
  save_dir.mkdir(parents=True, exist_ok=True)
  save_path = save_dir / cfg.save_name

  with open(save_path, 'w') as f:
    json.dump(content, f)
  logger.info(f'Saved results to `{save_path}`')


class FakeTokenizer:
  def __init__(self, vocab_length):
    self.vocab_size = vocab_length
    self.bos_token_id = 0
    self.mask_id = vocab_length - 1
  
  def __len__(self):
    return self.vocab_size


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'complement-mdlm':
    diffusion_model = algo.ComplementMDLM
  elif config.algo.name == 'partition-mdlm':
    diffusion_model = algo.PartitionMDLM
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  
  if hasattr(config.model, 'vocab_size'):
    kwargs['tokenizer'] = FakeTokenizer(config.model.vocab_size)
  
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'latency_train':
    _latency_train(**kwargs)
  elif config.mode == 'latency_sample':
    _sample_latency(**kwargs)
  elif config.mode == 'train':
    _train(**kwargs)
  else:
    raise ValueError(config.mode)


if __name__ == '__main__':
  main()