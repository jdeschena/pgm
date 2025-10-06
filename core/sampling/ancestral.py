import abc
import torch


def sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


class AncestralSampler(abc.ABC):
    def __init__(self, config):
        self.config = config
        assert self.config.noise.type == 'loglinear'

    @abc.abstractmethod
    def forward(self, x, cond):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _t_to_sigma(self, t):
        raise NotImplementedError

    def _compute_ddpm_update(self, x, t, dt, cache=None, forward=None):
        if t.ndim > 1:
            t = t.squeeze(-1)

        if forward is None:
            forward = self.forward

        if cache is None:
            sigma_t, _ = self.noise(t)
            sigma_s, _ = self.noise(t - dt)

            assert sigma_t.ndim == 1, sigma_t.shape
            assert sigma_s.ndim == 1, sigma_s.shape

            move_chance_t = 1 - torch.exp(-sigma_t)[:, None, None]
            move_chance_s = 1 - torch.exp(-sigma_s)[:, None, None]
            # Because could be using `t` to condition the model
            sigma, _, _ = self._t_to_sigma(t)
            sigma = sigma.squeeze(-1)

            log_p_x0 = forward(x, sigma)
            assert move_chance_t.ndim == log_p_x0.ndim
            # Technically, this isn't q_xs since there's a division
            # term that is missing. This division term doesn't affect
            # the samples.
            q_xs = log_p_x0.exp() * (move_chance_t
                                    - move_chance_s)
            q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

            cache = q_xs

        return log_p_x0.detach(), cache.detach()
    
    def _ddpm_sample_update(self, xt, update):
        copy_flag = (xt != self.mask_index).to(xt.dtype)
        x_new = copy_flag * xt + (1 - copy_flag) * update
        return x_new
    
    def _ddpm_update(self, x, t, dt, dtype=None, cache=None, denoise=False, mask_idx=None, forward=None):
        _, cache = self._compute_ddpm_update(x, t, dt, cache, forward)
        if dtype is not None:
            cache = cache.to(dtype)
        if denoise:
            if mask_idx is not None:
                cache[:, :, mask_idx] = -float("inf")
            update = cache.argmax(-1)
        else:
            update = sample_categorical(cache)

        return cache, self._ddpm_sample_update(x, update)

