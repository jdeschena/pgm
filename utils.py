"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""
from contextlib import contextmanager
import logging
import pickle
import hashlib
import base64

import fsspec
import lightning
import numpy as np
import torch
from timm.scheduler import CosineLRScheduler


@contextmanager
def replace_context(obj, *key_value_pairs):
    original_values = []
    for source, new_value in key_value_pairs:
        source_value = getattr(obj, source)
        original_values.append(source_value)
        setattr(obj, source, new_value)
    yield
    for (key, _), orig_value in \
                        zip(key_value_pairs, original_values):
        setattr(obj, key, orig_value)


def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class LRHalveScheduler:
  def __init__(self, warmup_steps, n_halve_steps):
    self.warmup_steps = warmup_steps
    self.n_halve_steps = n_halve_steps
  
  def __call__(self, current_step):
    if current_step < self.warmup_steps:
      return current_step / self.warmup_steps
    return 0.5 ** ((current_step - self.warmup_steps)
                   // self.n_halve_steps)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=0.0,
    filter_value=-float("Inf"),
    dim=-1):
    """Filter a distribution of logits using top-k/top-p (nucleus) filtering.
    Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
      logits (Tensor): Tensor of logits
      top_k (int, optional): Number of top values to keep.
          Deactivated if k is 0. Defaults to 0.
      top_p (float, optional): Cumulative mass to retain.
          Deactivated if p = 0. Defaults to 0.0.
      filter_value (float, optional): Fill value to replace
          the entries removed by top-k/top-p filtering.
          Defaults to -float('Inf').
      dim (int, optional): Dimension of the filtering. Defaults to -1.

    Returns:
        logits: Tensor whose axis `dim` was filtered.
    """
    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
      # Remove all tokens with a probability less than
      # the last token of the top-k
      values, _ = torch.topk(logits, k=top_k, dim=-1)
      to_remove_mask = (
          logits < torch.min(values, dim=-1, keepdim=True)[0]
      )  # min returns a tuple (values, indices)
      logits[to_remove_mask] = filter_value

    if top_p > 0.0:
      sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1)
      cum_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1)

      sorted_indices_to_remove = cum_probs > top_p
      # Ensures at least one token is kept
      sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      mask_to_remove = torch.empty_like(sorted_indices_to_remove)
      mask_to_remove.scatter_(dim=-1,
                              index=sorted_indices,
                              src=sorted_indices_to_remove)
      logits[mask_to_remove] = filter_value

    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    return logits


def vars_to_fname(sep='#', **kwargs):
  bool_entries = []
  non_bool_entries = []

  for k, v in kwargs.items():
    if type(v) is bool:
      bool_entries.append((k, v))
    else:
      non_bool_entries.append((k, v))

  bool_entries.sort(key=lambda x: x[0])
  non_bool_entries.sort(key=lambda x: x[0])

  name = ''
  for k, v in non_bool_entries:
    name += str(k) + '=' + str(v)
    name += sep

  for i, (k, v) in enumerate(bool_entries):
    if v:
      name += k
    else:
      name += 'no_' + k

    if i != len(bool_entries) - 1:
      name += sep

  return name


def short_hash(value: str):
  digest = hashlib.md5(value.encode()).digest()
  h = base64.b64encode(digest).decode()
  h = h.replace('/', 'AA')
  assert h[-2:] == '=='
  return h[:-2]


def np_to_base64(arr: np.ndarray) -> str:
  arr_bytes = pickle.dumps(arr)
  base64_bytes = base64.b64encode(arr_bytes)
  return base64_bytes.decode('ascii')


def base64_to_np(b64_str: str) -> np.ndarray:
  base64_bytes = b64_str.encode('ascii')
  arr_bytes = base64.b64decode(base64_bytes)
  return pickle.loads(arr_bytes)

