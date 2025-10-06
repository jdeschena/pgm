import itertools
import torch
import abc
from .noise_schedule import get_noise
from core.core import CoreLightning
from models.loading_utils import get_backbone
from models.ema import ExponentialMovingAverage
from transformers import AutoModelForCausalLM
from loguru import logger

@torch.no_grad()
def gpt2_eval(
        config,
        samples,
        cache=[]
):
    if len(cache) == 0:
        model = AutoModelForCausalLM.from_pretrained("gpt2-large").eval()
        cache.append(model)

    model = cache[0].to(samples.device)
    bs = config.eval.ppl_with_ar.batch_size

    total_loss = 0
    num_examples = 0

    for idx in range(0, samples.shape[0], bs):
        batch = samples[idx: idx + bs]
        logits = model(batch).logits[:, :-1]
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.gather(logits, dim=-1, index=batch[:, 1:, None])[..., 0]
        total_loss += loss.mean(-1).sum().cpu().item()
        num_examples += logits.shape[0]

    return total_loss, num_examples


def loss_per_bucket(
    loss,
    ts,
    num_buckets=5,
    min_val=0.0,
    max_val=1.0,
):
    assert loss.ndim == 1
    assert ts.ndim == 1
    assert loss.shape[0] == ts.shape[0]

    splits = torch.linspace(min_val, max_val, num_buckets + 1)
    low = splits[:-1]
    high = splits[1:]

    results = torch.empty(size=(num_buckets,))
    counts = torch.empty(size=(num_buckets,))

    for idx, (lo, hi) in enumerate(zip(low, high)):
        mask = (ts >= lo) * (ts < hi)
        loss_in_bucket = loss[mask]
        counts[idx] = torch.sum(mask)

        if loss_in_bucket.numel() == 0:
            results[idx] = 0.0
        else:
            results[idx] = loss_in_bucket.sum()

    return results, counts


class DiffusionCore(CoreLightning, abc.ABC):
    def __init__(self, config, tokenizer):
        CoreLightning.__init__(self, config)

        self.tokenizer = tokenizer
        self.noise = get_noise(config, dtype=self.dtype)

        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        # If T == 0 -> continuous formulation
        # If T > 0 -> model trained on discrete
        self.T = self.config.T
        self.mask_index = self.tokenizer.vocab_size
        self.vocab_size = self.tokenizer.vocab_size + 1

        self.backbone = get_backbone(config, vocab_size=self.vocab_size)
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.init_ema()

        if hasattr(self.config.parameterization, "log_loss_buckets"):
            self.num_loss_buckets = self.config.parameterization.log_loss_buckets
        else:
            self.num_loss_buckets = -1

        # Note: this is a hack because on restart, lightning runs ONE eval step
        # Which messes up the statistics
        self.has_run_debug_valid = False

    def iter_params(self):
        return itertools.chain(self.backbone.parameters(), self.noise.parameters())

    def init_ema(self):
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                self.iter_params(),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None

    # Torch forward
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    # Compute ELBO of parametrization; used for loss + eval
    def diffusion_elbo(self, x0, t=None):
        raise NotImplementedError

    # Actual loss to optimize
    def loss(self, x, attention_mask):
        raise NotImplementedError

    # Helper to check that configuration is consistent
    def validate_config(self):
        pass

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    # def train(self, is_training=True):
    #     assert is_training in (True, False)
    #     if is_training:
    #         self.restore_ema()
    #         self.backbone.train()
    #         self.noise.train()
    #     else:
    #         self.store_ema()
    #         self.backbone.eval()
    #         self.noise.eval()

    # def eval(self):
    #     self.train(False)

    def _sample_t(self, n, device=None):
        if device is None:
            device = self.device

        if self.antithetic_sampling:
            _eps_t = torch.rand(1, device=device)
            # Otherwise we have twice the start point
            offset = torch.linspace(
                0,
                1,
                steps=n + 1,
                device=device,
            )[:-1]
            _eps_t = (_eps_t + offset) % 1
        else:
            _eps_t = torch.rand(n, device=device)

        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)

        return t

    def _t_to_sigma(self, t):
        assert t.ndim == 1
        dsigma = None
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        if self.change_of_variables:
            conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        return conditioning, move_chance, dsigma

    def q_xt(self, x0, move_chance):
        idxs = torch.rand(*x0.shape, device=x0.device) < move_chance
        xt = torch.where(idxs, self.mask_index, x0)
        return xt

    def training_step(self, batch):
        if self.ema is not None:
            assert not self._using_ema_weights, "SHOULD NOT USE EMA WEIGHTS DURING TRAINING!!!"
        x = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        loss = self.loss(x, attention_mask=attention_mask)
        self.log(
            name="train/loss",
            value=loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def _can_run_valid(self):
        if self.trainer is None:
            return True
        if self.trainer.fit_loop.restarting:
            val = self.has_run_debug_valid
            self.has_run_debug_valid = True
            return val
        else:
            return True
        

    def validation_step(self, batch):
        if self.ema is not None:
            assert self._using_ema_weights, "SHOULD BE USING EMA WEIGHTS DURING TRAINING"

        if not self._can_run_valid():
            logger.warning("Skipping the first validation step after reload (PL BUG)...")
            return

        x = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        loss = self.loss(x, attention_mask=attention_mask)
        self.log(
            name="valid/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        CoreLightning.on_validation_epoch_start(self)
        self.noise.eval()
        n_samples = self.config.eval.valid.n_samples

        if n_samples > 0:
            samples = self.sample(
                n_samples=self.config.eval.valid.n_samples,
                num_steps=self.config.eval.valid.num_steps,
            )
            total_loss, total_num_samples = gpt2_eval(self.config, samples)

            total_loss = self.all_gather(data=torch.tensor(total_loss)).sum()
            total_num_samples = self.all_gather(data=torch.tensor(total_num_samples)).sum()
            ppl = (total_loss / total_num_samples).exp()

            self.log(
                name="valid/gpt2_ppl",
                value=ppl,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def on_train_epoch_start(self):
        CoreLightning.on_train_epoch_start(self)
        self.noise.train()

    @torch.no_grad()
    def _log_buckets(self, loss, t, mode="train", key="loss"):
        if self.num_loss_buckets == -1:
            return
        results, counts = loss_per_bucket(loss, t, self.num_loss_buckets)

        results = self.all_gather(results).sum(0)
        counts = self.all_gather(counts).sum(0)

        mean = results / counts

        log_dict = dict()
        for idx in range(mean.shape[0]):
            if counts[idx] > 0:
                log_dict[f"{mode}/{key}_bucket_{idx}"] = float(mean[idx])

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.device
        )
    
    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

