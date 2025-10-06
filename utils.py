"""Utilities.

Some functions copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
"""

import torch
import os
from omegaconf import OmegaConf
import lightning as L
from loguru import logger
from timm.scheduler import CosineLRScheduler
import fsspec
import struct
import base64


def add_resolvers():
    OmegaConf.register_new_resolver("cwd", os.getcwd)
    OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def prepare_logger():
    fn_names = ["trace", "debug", "info", "success", "warning", "error", "critical"]
    for k in fn_names:
        fn = getattr(logger, k)
        fn = L.pytorch.utilities.rank_zero_only(fn)
        setattr(logger, k, fn)


def rm_null_values(in_dict):
    # Must copy since inplace modification during a loop is forbidden
    new_dict = dict()
    for k, v in in_dict.items():
        if v is not None:
            new_dict[k] = v

    return new_dict


def str_to_dtype(s):
    if s == "bf16-mixed":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype `{s}`")


def parse_str_int_list(s):
    l = s.split(",")
    l = [int(x) for x in l if x.strip() != ""]
    return l


class CosineDecayWarmupLRScheduler(
    CosineLRScheduler, torch.optim.lr_scheduler._LRScheduler
):
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


# From the original MDLM codebase; for debug now
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


def encode_numbers_to_base64(numbers):
    # Convert the list of numbers to a bytes object (using struct)
    byte_data = struct.pack(f'{len(numbers)}f', *numbers)  # Packing as floats ('f')
    
    # Encode the byte data into a base64 string
    base64_string = base64.b64encode(byte_data).decode('utf-8')
    
    return base64_string


def decode_base64_to_numbers(base64_string):
    # Decode the base64 string back into bytes
    byte_data = base64.b64decode(base64_string)
    
    # Unpack the bytes back into the original list of numbers
    number_count = len(byte_data) // 4  # Each float is 4 bytes
    numbers = struct.unpack(f'{number_count}f', byte_data)
    
    return list(numbers)
