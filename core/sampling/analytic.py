import abc
import torch

def sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

@torch.jit.script
def _get_score(
    x: torch.Tensor, 
    sigma: torch.Tensor, 
    model_output: torch.Tensor,
    mask_index: int,
):
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, mask_index] = 0

    unmasked_score = -10_000_000 * torch.ones_like(
        masked_score)
    unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == mask_index).to(
        masked_score.dtype)[:, :, None]
    
    model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class AnalyticSampler(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def forward(self, x, cond):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _t_to_sigma(self, t):
        raise NotImplementedError

    def _transp_transition(self, i, sigma):
        sigma = sigma[..., None, None]
        edge = torch.zeros(size=(*i.shape, self.vocab_size), device=i.device)
        view_shape = i.shape[0] * i.shape[1]
        idx_arange = torch.arange(view_shape)
        # One-hot
        edge.view(view_shape, -1)[idx_arange, i.view(-1)] = 1
        edge *= torch.exp(-sigma)
        
        # Previous code that was using more memory
        #edge = torch.exp(-sigma) * F.one_hot(
        #i, num_classes=self.vocab_size)
        edge += torch.where(
            i == self.mask_index,
            1 - torch.exp(-sigma).squeeze(-1),
            0)[..., None]
        return edge

    def _analytic_update(self, x, t, step_size, dtype=None, forward=None):
        if forward is None:
            forward = self.forward

        if t.ndim > 1:
            t = t.squeeze(-1)
        if dtype is not None:
            t = t.to(dtype)
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma

        model_output = forward(x, torch.zeros_like(curr_sigma))
        if dtype is not None:
            model_output = model_output.to(dtype)
        score = self.get_score(x, curr_sigma, model_output)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return model_output.detach(), sample_categorical(probs)

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp())[:, None] * score.sum(dim=-1)
        score *= dsigma.exp()[:, None, None]
        score[..., self.mask_index] += extra_const
        return score
    
    def get_score(self, x, sigma, model_output):
        # jit to optimize memory usage
        return _get_score(x, sigma, model_output, self.mask_index)
    