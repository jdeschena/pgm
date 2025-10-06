import time
from lightning.pytorch.callbacks import Callback
import torch


class TextThroughputMonitor(Callback):
    def __init__(self):
        Callback.__init__(self)

        self._elapsed = dict(train=0.0, valid=0.0)
        self._starts = dict(train=0.0, valid=0.0)
        self._tot_num_example = dict(train=0, valid=0)
        self._tot_num_tokens = dict(train=0, valid=0)
        self._step_num_example = dict(train=0, valid=0)
        self._step_num_tokens = dict(train=0, valid=0)

    def state_dict(self):
        return dict(
            elapsed=self._elapsed,
            starts=self._starts,
            tot_num_example=self._tot_num_example,
            tot_num_tokens=self._tot_num_tokens,
            step_num_example=self._step_num_example,
            step_num_tokens=self._step_num_tokens
        )

    def load_state_dict(self, state):
        self._elapsed = state["elapsed"]
        self._starts = state["starts"]

        self._tot_num_example = state["tot_num_example"]
        self._tot_num_tokens = state["tot_num_tokens"]

        self._step_num_example = state["step_num_example"]
        self._step_num_tokens = state["step_num_tokens"]

    def _start(self, trainer, stage):
        self._starts[stage] = time.perf_counter()
        self._step_num_example[stage] = 0
        self._step_num_tokens[stage] = 0

    def _record_batch(self, batch, stage):
        self._step_num_example[stage] += batch["input_ids"].shape[0]
        self._step_num_tokens[stage] += batch["input_ids"].numel()

    def _end(self, trainer, pl_module, stage):
        if trainer.strategy.root_device.type == "cuda":
            # required or else perf_counter() won't be correct
            torch.cuda.synchronize()

        # Update global statistics
        elapsed = time.perf_counter() -  self._starts[stage]
        self._elapsed[stage] += elapsed

        self._tot_num_example[stage] += self._step_num_example[stage]
        self._tot_num_tokens[stage] += self._step_num_tokens[stage]

        device_batch_throughput = self._step_num_example[stage] / elapsed
        total_batch_throughput = device_batch_throughput * trainer.num_devices * trainer.num_nodes

        pl_module.log_dict({
            f"trainer/{stage}_device_batch_throughput": device_batch_throughput,
            f"trainer/{stage}_device_tok_per_sec": self._step_num_tokens[stage] / elapsed,
            f"trainer/{stage}_device_example_per_sec": self._step_num_example[stage] / elapsed,

            f"trainer/{stage}_total_batch_throughput": total_batch_throughput,
            f"trainer/{stage}_total_num_batches": self._tot_num_example[stage] * trainer.num_devices * trainer.num_nodes, 
            f"trainer/{stage}_total_num_tokens": self._tot_num_tokens[stage] * trainer.num_devices * trainer.num_nodes,

            f"trainer/{stage}_elapsed": self._elapsed[stage],
            }, 
            on_step=stage=="train",
            on_epoch=stage=="valid",
            sync_dist=False,
        )

    #@rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start(trainer, "train")

    #@rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._record_batch(batch, "train")
        # log only when gradient accumulation is over. this ensures that we only measure when the effective batch has
        # finished and the `optimizer.step()` time is included
        if not trainer.fit_loop._should_accumulate():
            self._end(trainer, pl_module, "train")

    #@rank_zero_only
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        self._start(trainer, "valid")

    #@rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        self._record_batch(batch, "valid")
        self._end(trainer, pl_module, "valid")
