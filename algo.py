import os

import torch

import trainer_base
import utils


@torch.jit.script
def _process_model_output_efficient(
  model_output, 
  xt, 
  sigma, 
  neg_infinity: float = -1000000.0,
  cast_fp64: bool = False, 
  mask_index: int=-1):
  del sigma
  if cast_fp64:
    model_output = model_output.to(torch.float64)

  index = torch.full(size=(xt.shape[0], xt.shape[1], 1), 
    fill_value=mask_index, device=xt.device)

  model_output = torch.scatter(model_output, dim=-1, 
                               index=index, value=neg_infinity)
  
  unmasked_indices = (xt != mask_index)

  model_output = torch.where(unmasked_indices[..., None], 
                             neg_infinity, model_output)
  model_output = torch.scatter(model_output, dim=-1, 
                               index=xt[..., None], value=0.0)

  model_output = torch.log_softmax(model_output, dim=-1)
  return model_output


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step):
    del current_accumulation_step
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.post_process_mode = config.algo.post_process_mode
    self._validate_configuration()

  def _process_model_output_orig(self, model_output, xt, sigma, 
                                 cast_fp64=False):
    del sigma
    if cast_fp64:
      model_output = model_output.to(torch.float64)
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.post_process_mode not in {
      'orig', 'orig+fp64', 'efficient', 'efficient+fp64'}:
      raise ValueError(self.post_process_mode)

  def _process_model_output(self, model_output, xt, sigma):
    cast_fp64 = 'fp64' in self.post_process_mode
    if 'orig' in self.post_process_mode:
      return self._process_model_output_orig(model_output, xt, 
        sigma, cast_fp64=cast_fp64)
    elif 'efficient' in self.post_process_mode:
      return _process_model_output_efficient(model_output, xt, 
        sigma, neg_infinity=self.neg_infinity, cast_fp64=cast_fp64, 
        mask_index=self.mask_index)
    else:
      raise ValueError(self.post_process_mode)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    if low_var:
      loss_coefficient = -1
    else:
      loss_coefficient = dalpha_t / (1 - alpha_t)
    return loss_coefficient * log_p_theta

  def _get_score(self, x, sigma, group_idxs=None):
    model_output = self.forward(x, sigma, group_idxs)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class ComplementMDLM(MDLM):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    if not train_mode:
      return MDLM.nll(self, x0, output_tokens, 
                      current_accumulation_step, train_mode)
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t)
    xt_complement = torch.where(xt == x0, self.mask_index, x0)
    dalpha_complement, alpha_complement = self.noise(1-t)
    alpha_complement = alpha_complement.unsqueeze(-1)
    sigma_complement = self._sigma_from_alphat(alpha_complement)

    full_noisy_batch = torch.cat([xt, xt_complement])
    full_sigma_batch = torch.cat([sigma, sigma_complement])

    log_x_theta = self.forward(full_noisy_batch, sigma=full_sigma_batch)
    utils.print_nans(log_x_theta, 'model_output')

    nll_orig = self.nll_per_token(
      log_x_theta=log_x_theta[:x0.shape[0]],
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')
    nll_complement = self.nll_per_token(
      log_x_theta=log_x_theta[x0.shape[0]:],
      xt=xt_complement,
      x0=x0,
      alpha_t=alpha_complement,
      dalpha_t=dalpha_complement,
      low_var=train_mode and self.loss_type == 'low_var')
    nll_full = torch.where(xt != x0, nll_orig, nll_complement)
    return nll_full / 2.0


class PartitionMDLM(MDLM):
  def __init__(self, config, tokenizer):
    self.sampling_mode = config.algo.sampling_mode
    super().__init__(config, tokenizer)

  def _validate_configuration(self):
    assert not self.time_conditioning, \
      "Partition MDLM cannot be time conditioned."

    assert self.post_process_mode in {'efficient', 
                  'efficient+fp64'}, self.post_process_mode
    assert self.sampling_mode in {'naive', 'efficient-uniform', 
                'efficient-non-uniform'}, self.sampling_mode
    return super()._validate_configuration()
  
  def _q_xt_partition(self, x, alpha_t):
    #  Probability of being in group 0 is alpha_t.
    #  -> masking probability is 1 - alpha_t
    group_idxs = torch.rand(
        *x.shape, device=x.device) < 1 - alpha_t
    return group_idxs.to(int)

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    del xt
    if self.post_process_mode == 'efficient+fp64':
      model_output = model_output.to(torch.float64)
    model_output[:, :, self.mask_index] = self.neg_infinity
    model_output = torch.log_softmax(model_output, dim=-1)
    return model_output

  def forward(self, xt, sigma, group_idxs=None, 
              clean_positions=None, noisy_positions=None, 
              concrete_lengths=None, use_inference_mode=False):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      model_output = self.backbone(xt, 
        sigma=sigma, 
        group_idxs=group_idxs,
        clean_positions=clean_positions,
        noisy_positions=noisy_positions,
        concrete_lengths=concrete_lengths,
        use_inference_mode=use_inference_mode)
      if use_inference_mode and self.config.sampling.p_nucleus < 1:
        model_output = utils.top_k_top_p_filtering(
          model_output, top_p=self.config.sampling.p_nucleus)
    return self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    t_complement = 1 - t

    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    dalpha_t_complement, alpha_t_complement = self.noise(t_complement)
    alpha_t_complement = alpha_t_complement.unsqueeze(-1)
    assert alpha_t_complement.ndim == 2

    group_idxs = self._q_xt_partition(x0, alpha_t)
    log_x_theta = self.forward(x0, torch.zeros_like(sigma),
                               group_idxs)
    utils.print_nans(log_x_theta, 'model_output')
    # For the group 0, the group 1 represents mask tokens.
    #  Hence, loss for tokens in the group 1 should be scaled 
    #  using alpha_t and dalpha_t.
    nll_alpha_t = torch.where(group_idxs != 0, alpha_t, 
                              alpha_t_complement)
    nll_dalpha_t = torch.where(group_idxs != 0, dalpha_t, 
                               dalpha_t_complement)
    tokens_nll = self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=None,  # is ignored in MDLM
      x0=x0,
      alpha_t=nll_alpha_t,
      dalpha_t=nll_dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

    if train_mode:
      return tokens_nll / 2
    else:
      # As if group 1 represents mask tokens
      return torch.where(group_idxs == 1, tokens_nll, 0.)

  @torch.no_grad
  def generate_samples(self, num_samples, num_steps=None, 
                       inject_bos=None, eps=1e-5, 
                       sampling_mode=None):
    if sampling_mode is None:
      sampling_mode = self.sampling_mode
    if sampling_mode == 'naive':
      return self.generate_samples_naive(
        num_samples, num_steps, inject_bos, eps)
    elif sampling_mode == 'efficient-uniform':
      return self.generate_samples_efficient_uniform(
        num_samples, num_steps, inject_bos, eps)
    elif sampling_mode == 'efficient-non-uniform':
      return self.generate_samples_efficient_non_uniform(
        num_samples, num_steps, inject_bos, eps)
    else:
      raise ValueError(sampling_mode)

  def generate_samples_efficient_uniform(
      self, num_samples, num_steps=None, inject_bos=None, 
      eps=1e-5):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    assert inject_bos, "Partition MDLM assumes BOS"
    assert self.sampler in ('ddpm', 'ddpm_cache')

    x = torch.full(size=(num_samples, 1), 
                   fill_value=self.tokenizer.bos_token_id, 
                   device=self.device)
    
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps

    clean_positions = torch.zeros(size=(num_samples, 1), 
                                  device=self.device, 
                                  dtype=torch.int64)
    noisy_positions = torch.arange(start=1, 
                                   end=self.config.model.length, 
                                   device=self.device, 
                                   dtype=torch.int64)[None
                                    ].repeat(num_samples, 1)
    # Compute a random permutation of the rows
    rand = torch.rand_like(noisy_positions, dtype=torch.float32)
    shuffled_indices = rand.argsort(dim=-1)
    noisy_positions = torch.gather(noisy_positions, dim=-1, 
                                   index=shuffled_indices)
    concrete_lengths = torch.ones(size=(num_samples,), 
                                  device=self.device, 
                                  dtype=torch.int64)
    assert self.config.model.length % num_steps == 0
    n_tok_per_normal_step =  self.config.model.length // num_steps
    all_n_tok_per_step = torch.full(size=(num_steps,), 
                                fill_value=n_tok_per_normal_step)
    # Last steps might need to predict more tokens
    all_n_tok_per_step[-1] += (self.config.model.length 
                           - num_steps * n_tok_per_normal_step)
    for t, n_tok_per_step in zip(timesteps[:-1], all_n_tok_per_step):
      t = t * torch.ones(x.shape[0], 1, device=self.device)
      # Select which tokens will be denoised
      noisy_pos_input = noisy_positions[:, :n_tok_per_step]
      # Denoise      
      denoised_token_values = self._ddpm_update_efficient(x=x, 
         t=t, dt=dt, p_x0=None, clean_positions=clean_positions, 
         noisy_positions=noisy_pos_input, 
         concrete_lengths=concrete_lengths)
      # Append newly denoised tokens
      x = torch.cat([x, denoised_token_values], dim=1)
      clean_positions = torch.cat([clean_positions, 
                                   noisy_pos_input], dim=1)
      noisy_positions = noisy_positions[:, n_tok_per_step:]
      concrete_lengths += n_tok_per_step
    # Reorder
    out = torch.empty_like(x).scatter_(dim=-1, 
                                       index=clean_positions, 
                                       src=x)
    return out

  def _gen_eff_non_unif_post_process(self, x, concrete_lengths, 
    n_denoise_per_seq, denoised_token_values, clean_positions, 
    noisy_positions, noisy_pos_input):
    new_concrete_lengths = concrete_lengths + n_denoise_per_seq
    n_tok_to_add = new_concrete_lengths.max() - x.shape[1]
    if n_tok_to_add > 0:
      pad = torch.zeros(size=(x.shape[0], n_tok_to_add), 
                        dtype=x.dtype, device=x.device)
      x = torch.cat([x, pad], dim=1)
      clean_positions = torch.cat([clean_positions, pad], dim=1)
    # TODO: maybe this can be vectorized...
    for i in range(x.shape[0]):
      if n_denoise_per_seq[i] == 0:
        continue
      x[i, concrete_lengths[i]: new_concrete_lengths[i]] \
            = denoised_token_values[i, :n_denoise_per_seq[i]]
      clean_positions[i, 
            concrete_lengths[i]:new_concrete_lengths[i]] = \
            noisy_pos_input[i, :n_denoise_per_seq[i]]
      noisy_positions[i, 
        :noisy_positions.shape[1] - n_denoise_per_seq[i]] = \
        noisy_positions[i, n_denoise_per_seq[i]:].clone()
    return x, clean_positions, new_concrete_lengths
  
  def generate_samples_efficient_non_uniform(self, 
    num_samples, num_steps=None, inject_bos=None, eps=1e-5):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    assert inject_bos, "Partition MDLM assumes BOS"
    assert self.sampler in ('ddpm', 'ddpm_cache')

    x = torch.full(size=(num_samples, 1), 
                   fill_value=self.tokenizer.bos_token_id, 
                   device=self.device)
    
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps

    clean_positions = torch.zeros(size=(num_samples, 1), 
                                  device=self.device, 
                                  dtype=torch.int64)
    noisy_positions = torch.arange(start=1, 
                                   end=self.config.model.length, 
                                   device=self.device, 
                                   dtype=torch.int64)[None
                                    ].repeat(num_samples, 1)
    # Compute a random permutation of the rows
    rand = torch.rand_like(noisy_positions, dtype=torch.float32)
    shuffled_indices = rand.argsort(dim=-1)
    noisy_positions = torch.gather(noisy_positions, dim=-1, 
                                   index=shuffled_indices)
    concrete_lengths = torch.ones(size=(num_samples,), 
                                  device=self.device, 
                                  dtype=torch.int64)
    if self.sampler not in ('ddpm', 'ddpm_cache'):
      raise ValueError(self.sampler)
    
    _, alpha_t = self.noise(timesteps[0])
    _, alpha_s = self.noise(timesteps[0] - dt)
    # We always denoise the same fraction of tokens
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
    
    for t in timesteps[:-1]:
      t = t * torch.ones(x.shape[0], 1, device=self.device)
      bin_count = torch.ones(size=(num_samples,), 
                             device=prob_denoise.device)
      bin_count *= self.config.model.length
      n_denoise_per_seq = torch.binomial(count=bin_count, 
                                         prob=prob_denoise).to(int)
      # Make sure that we don't denoise more than the sequence length
      n_denoise_per_seq = torch.min(n_denoise_per_seq, 
                self.config.model.length - concrete_lengths)
      denoise_seq_len = torch.max(n_denoise_per_seq).item()
      if denoise_seq_len == 0:
        continue
      # Some predictions will not be used, each sequence can 
      #  denoise a different number of tokens
      noisy_pos_input = noisy_positions[:, :denoise_seq_len]
      denoised_token_values = self._ddpm_update_efficient(x=x, 
         t=t, dt=dt, p_x0=None, clean_positions=clean_positions, 
         noisy_positions=noisy_pos_input, 
         concrete_lengths=concrete_lengths)
      # Update x based on the predictions. Make sure to denoise 
      #  the correct number of token per sequence
      (x, clean_positions, concrete_lengths) = \
        self._gen_eff_non_unif_post_process(x, concrete_lengths, 
        n_denoise_per_seq, denoised_token_values, clean_positions, 
        noisy_positions, noisy_pos_input)
      
    # If some tokens remain masked, finally denoise them
    if not torch.all(concrete_lengths == self.config.model.length):
      n_denoise_per_seq = self.config.model.length - concrete_lengths
      noisy_pos_input = noisy_positions[:, :self.config.model.length - concrete_lengths.min()]
      denoised_token_values = self._ddpm_update_efficient(x=x, 
         t=t, dt=dt, p_x0=None, clean_positions=clean_positions, 
         noisy_positions=noisy_pos_input, 
         concrete_lengths=concrete_lengths)
      (x, clean_positions, concrete_lengths) = \
        self._gen_eff_non_unif_post_process(x, concrete_lengths, 
        n_denoise_per_seq, denoised_token_values, clean_positions, 
        noisy_positions, noisy_pos_input)
    # Reorder
    out = torch.empty_like(x).scatter_(dim=-1, 
                                       index=clean_positions, 
                                       src=x)
    return out
  
  def generate_samples_naive(self, num_samples, num_steps=None, 
                             inject_bos=None, eps=1e-5):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens)
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    assert inject_bos, "Partition MDLM assumes BOS"
    x[:, 0] = self.tokenizer.bos_token_id
    
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    # Group 0: unmasked, group 1: masked
    group_idxs = torch.ones_like(x, dtype=int)
    group_idxs[:, 0] = 0
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, 
                                    device=self.device)
      if self.sampler == 'ddpm':
        _, x, group_idxs = self._ddpm_update(
          x=x, t=t, dt=dt, p_x0=None, group_idxs=group_idxs)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next, group_idxs = self._ddpm_update(
          x=x, t=t, dt=dt, p_x0=p_x0_cache, 
          group_idxs=group_idxs)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x, group_idxs = self._analytic_update(
          x=x,t=t, dt=dt, group_idxs=group_idxs)

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                    device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
      if self.sampler == 'analytic':
        x = self._denoiser_update(x=x, t=t0, 
                                  group_idxs=group_idxs)
      else:
        _, x, _ = self._ddpm_update(x=x, t=t0, dt=None,
                                    p_x0=p_x0_cache,
                                    noise_removal_step=True,
                                    group_idxs=group_idxs)
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      x = self.forward(xt=x, sigma=sigma, 
                       group_idxs=group_idxs).argmax(dim=-1)
    return x
