import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def make_group_self_attn_mask(group_idxs):
  # Return shape: N x L x L
  return group_idxs[:, None, :] == group_idxs[:, :, None]


# Training
def make_group_cross_attn_mask(group_idxs):
  # Return N x L x L
  return group_idxs[:, None, :] != group_idxs[:, :, None]


# Inference
def make_inference_self_attn_mask(seq_len, concrete_lengths):
  arrange = torch.arange(seq_len, device=concrete_lengths.device)
  mask = arrange[None, :] < concrete_lengths[:, None]
  mask = mask[:, None, :].expand(-1, seq_len, -1)
  return mask


def make_inference_cross_attn_mask(
    keys_tensor_length, 
    queries_tensor_length,
    concrete_lengths_keys,):
  """
  Queries positions == noisy positions
  Key positions == denoised positions
  Concrete length: number of denoised tokens in each 
                        element of the batch.
  """
  arrange = torch.arange(keys_tensor_length, device=concrete_lengths_keys.device)
  mask = arrange[None] < concrete_lengths_keys[:, None]  # BS x KV_LEN
  mask = mask[:, None, :]  # BS x 1 x KV_LEN
  mask = mask.expand(-1, queries_tensor_length, -1)  # BS x Q_LEN x KV_LEN
  return mask


def get_sinusoidal_embedding(idxs, dim, base=10_000):
  device = idxs.device
  denominator = base ** (2 * torch.arange(dim // 2, 
                                          device=device) / dim)
  arg = idxs[:, None] / denominator[None, :]
  cos = torch.cos(arg)
  sin = torch.sin(arg)
  out = torch.cat([cos, sin], dim=-1)
  return out


@torch.jit.script
def _partition_mean_train(x, group_idxs, group: int):
  mask = (group_idxs == group)  # BS x L
  group_div = mask.sum(-1)  # BS
  group_div = torch.where(group_div == 0, 1, group_div)
  out = (x * mask[..., None]).sum(1) / group_div[:, None]  # BS x H
  return mask, out  # (BS x L, BS x H)


@torch.jit.script
def _partition_logsumexp_train(x, group_idxs, group: int):
  mask = (group_idxs == group)  # BS x L
  out = torch.where(mask[..., None], x, -float('inf'))  # BS x L x H
  out = torch.logsumexp(out, dim=1)  # BS x H
  out = torch.where(out.isinf(), 0.0, out)
  return mask, out  # (BS x L, BS x H)


@torch.jit.script
def _partition_mean_inference(x, concrete_lengths):
  arrange = torch.arange(x.shape[1], device=x.device)[None, :]  # 1 x L
  mask = (arrange < concrete_lengths[:, None])  # BS x L
  group_div = mask.sum(1)  # BS
  group_div = torch.where(group_div == 0, 1, group_div)
  out = (x * mask[..., None]).sum(1) / group_div[:, None]  # BS x H
  return out  # BS x H


@torch.jit.script
def _partition_logsumexp_inference(x, concrete_lengths):
  arrange = torch.arange(x.shape[1], device=x.device)[None, :]  # 1 x L
  mask = arrange < concrete_lengths[:, None]  # BS x L
  out = torch.where(mask[..., None], x, -float('inf'))  # BS x L x H
  out = torch.logsumexp(out, dim=1)  # BS x H
  out = torch.where(out.isinf(), 0.0, out)  # BS x H
  return out  # BS x H


@torch.jit.script
def _index_rotary(source_tensor, index):
  # source shape: 1 x L x 3 x 1 x H/2
  # index shape: BS x L
  # out shape: BS x L x 3 x 1 x H/2
  index = index[..., None, None, None]  # BS x L x 1 x 1 x 1
  # BS x L x 3 x 1 x H/2
  index = index.expand(-1,-1, source_tensor.shape[2], 
                       source_tensor.shape[3], 
                       source_tensor.shape[4])
  # BS x L x 3 x 1 x H/2
  source_tensor = source_tensor.expand(index.shape[0],-1,-1,-1,-1)
  out = torch.gather(source_tensor, dim=1, index=index)
  return out


@torch.jit.script
def _index_freqs_swap(pos_freqs, positions):
  freqs = pos_freqs[None, positions]  # 1 x L x H
  freqs = pos_freqs[None].expand(positions.shape[0], -1, -1)
  positions = positions[..., None].expand(-1, -1, freqs.shape[-1])
  freqs = torch.gather(freqs, dim=1, index=positions)
  return freqs


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x=None, seq_len=None, device=None, seq_dim=1):
    if seq_len is not None and seq_len == self.seq_len_cached:
      return self.cos_cached, self.sin_cached

    if x is not None:
      seq_len = x.shape[seq_dim]
      device = x.device
    
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    if cos.shape[0] == 1:
      cos_in = cos[0,:,0,0,:cos.shape[-1]//2]
      sin_in = sin[0,:,0,0,:sin.shape[-1]//2]
    else:
      cos_in = cos[:, :,0,0,:cos.shape[-1]//2]
      sin_in = sin[:,:,0,0,:sin.shape[-1]//2]

    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos_in, sin_in)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos_in, sin_in)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_pos_emb_single(vec, cos, sin):
  # vec shape: b x s x h x d
  with torch.amp.autocast('cuda', enabled=False):
    cos = cos.to(vec.dtype)
    sin = sin.to(vec.dtype)
    if cos.shape[0] == 1:
      cos_in = cos[0,:,0,0,:cos.shape[-1]//2]
      sin_in = sin[0,:,0,0,:sin.shape[-1]//2]
    else:
      cos_in = cos[:,:,0,0,:cos.shape[-1]//2]
      sin_in = sin[:,:,0,0,:sin.shape[-1]//2]
    vec = flash_attn.layers.rotary.apply_rotary_emb_torch(vec, cos_in, sin_in)
  return vec


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


def apply_masked_mha(q, k, v, attn_mask):
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  attention_output = F.scaled_dot_product_attention(
    query=q, key=k, value=v, attn_mask=attn_mask[:, None], 
    dropout_p=0.0, is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]

def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, t_cond, rotary_cos_sin, self_attn_mask):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      # self.adaLN_modulation(c): (128, 1536)
      # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
      # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(t_cond
       ).chunk(6, dim=1)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
    x = apply_masked_mha(q, k, v, self_attn_mask)

    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x),
                                None,
                                gate_msa.unsqueeze(1),
                                x_skip,
                                self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp.unsqueeze(1), x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    if x.ndim == 2:
      return self.embedding[x]
    assert x.ndim == 3
    return torch.einsum(
      "blv,ve->ble",
      torch.nn.functional.softmax(x, dim=-1).float(),
      self.embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim,
               adaLN):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    x = self.norm_final(x)
    if self.adaLN:
      shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


class CrossAttnDDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN, cond_dim=None, 
               mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN

    self.q_norm1 = LayerNorm(dim)
    self.kv_norm1 = LayerNorm(dim)

    self.norm2 = LayerNorm(dim)
    self.attn_q = nn.Linear(dim, dim, bias=False)
    self.attn_kv = nn.Linear(dim, 2 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, q_x, kv_x, t_cond, rotary_cos_sin_queries, 
              rotary_cos_sin_keys, attn_mask):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = q_x
    q_x = self.q_norm1(q_x)
    kv_x = self.kv_norm1(kv_x)
    if self.adaLN:
      # self.adaLN_modulation(c): (128, 1536)
      # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
      # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
      (shift_msa, scale_msa, gate_msa, shift_mlp, 
        scale_mlp, gate_mlp
      ) = self.adaLN_modulation(t_cond).chunk(6, dim=1)
      
      q_x = modulate_fused(q_x, shift_msa, scale_msa)

    q = einops.rearrange(
      self.attn_q(q_x),
      'b s (h d) -> b s h d',
      h=self.n_heads)
    kv = einops.rearrange(
      self.attn_kv(kv_x),
      'b s (two h d) -> b s two h d',
      two=2,
      h=self.n_heads)

    k, v = torch.chunk(kv, chunks=2, dim=2)
    k = k[:, :, 0, :]
    v = v[:, :, 0, :]

    q = apply_rotary_pos_emb_single(q, *rotary_cos_sin_queries)
    k = apply_rotary_pos_emb_single(k, *rotary_cos_sin_keys)
    x = apply_masked_mha(q, k, v, attn_mask)

    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x),
                                None,
                                gate_msa.unsqueeze(1),
                                x_skip,
                                self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp.unsqueeze(1), x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class Encoder(nn.Module):
  def __init__(self, n_blocks, dim, n_heads, cond_dim, mlp_ratio, 
               dropout, adaLN):
    super().__init__()
    self.blocks = nn.ModuleList([DDiTBlock(dim, n_heads, adaLN, 
                                           cond_dim, mlp_ratio, 
                                           dropout) 
                                for _ in range(n_blocks)])

  def forward(self, x, t_cond, rotary_cos_sin, self_attn_mask):
    for layer in self.blocks:
      x = layer(x, t_cond, rotary_cos_sin, self_attn_mask)
    return x

  
class GroupSwapLayer(nn.Module):
  def __init__(
    self,
    hidden_dim,
    n_heads,
    pre_query_mode,
    query_process_mode,
    model_length,
    normalize_pre_queries,
  ):
    super().__init__()
    assert hidden_dim % n_heads == 0
    self.hidden_dim = hidden_dim
    self.n_heads = n_heads
    self.pre_query_mode = pre_query_mode
    self.query_process_mode = query_process_mode
    self.model_length = model_length
    self.normalize_pre_queries = normalize_pre_queries

    self._prepare_query_processing()
    self.hidden_to_key_value = nn.Linear(hidden_dim, 2 * hidden_dim)
    self.output_linear = nn.Linear(hidden_dim, hidden_dim)

  def _prepare_query_processing(self):
    if self.pre_query_mode not in {'learn', 'learn+freqs', 
                                   'learn+freqs+mean', 
                                   'learn+freqs+logsumexp'}:
      raise ValueError(self.pre_query_mode)

    if self.query_process_mode not in {'linear', 'mlp4'}:
      raise ValueError(self.pre_query_mode)

    if self.normalize_pre_queries == 'layernorm':
      self.pre_query_norm = nn.LayerNorm(self.hidden_dim)
    elif self.normalize_pre_queries == 'rmsnorm':
      self.pre_query_mode = nn.RMSNorm(self.hidden_dim)
    else:
      raise ValueError(self.normalize_pre_queries)

    self.base_embedding = nn.Embedding(1, 
      self.hidden_dim).weight

    if 'freqs' in self.pre_query_mode:
      seq_pos = torch.arange(self.model_length)
      pos_freqs = get_sinusoidal_embedding(seq_pos, 
                                           self.hidden_dim)
      self.register_buffer('pos_freqs', pos_freqs, 
                            persistent=False)

    ### SECOND, PREPARE THE PRE-QUERY PROCESSING
    if self.query_process_mode == 'linear':
      self.query_processor = nn.Linear(self.hidden_dim, 
                                       self.hidden_dim)
    elif self.query_process_mode == 'mlp4':
      self.query_processor = nn.Sequential(
      nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
      nn.SiLU(),
      nn.Linear(4 * self.hidden_dim, self.hidden_dim))
    else:
      raise ValueError(self.query_process_mode)
    
  def _compute_queries(self, x, positions, group_idxs, 
                       concrete_lengths, use_inference_mode):
    queries = self.base_embedding[None]  # 1 x 1 x H
    if 'freqs' in self.pre_query_mode:
      if not use_inference_mode:
        freqs = self.pos_freqs[None]  # 1 x L x H
      else:
        freqs = _index_freqs_swap(self.pos_freqs, positions)
      queries = queries + freqs

    if self.pre_query_mode.startswith('learn+freqs+'):
      # learn+freqs+mean, learn+freqs+logsumexp
      if not use_inference_mode:
        # Training: need to compute a value per group
        extract_fn = (_partition_logsumexp_train 
                      if 'logsumexp' in self.pre_query_mode 
                      else _partition_mean_train)
        _, val_grp_zero = extract_fn(x, group_idxs, 0)
        mask_one, val_grp_one = extract_fn(x, group_idxs, 1)
        queries = queries + torch.where(mask_one[..., None],  # (bs, l, dim)
                                        val_grp_zero[:, None],  
                                        val_grp_one[:, None])
      else:
        # Inference: all tokens are in the same group -> just broadcast
        extract_fn = (_partition_logsumexp_inference
                      if 'logsumexp' in self.pre_query_mode
                      else _partition_mean_inference)
        value = extract_fn(x, concrete_lengths)  # bs x dim
        queries = queries + value[:, None, :]

    queries = self.pre_query_norm(queries)
    out = self.query_processor(queries)
    return out
  
  def forward(
    self, 
    x, 
    rotary_cos_sin_queries, 
    rotary_cos_sin_keys, 
    # Training
    group_idxs, 
    # Inference
    position_queries,
    concrete_lengths, 
    cross_attn_mask,
    use_inference_mode):
    """
    Implementation:
      1: Compute query embeddings.
      2: Apply RoPE positional encoding
      3: Compute the cross-attention

    Arguments:
      x: tensor to process. Shape: (BS, L, H)
      group_idxs: binary tensor that describes the group to 
                  which each token belongs
      position_queries: position in the sequence of each 
                        element of the tensor. During training, 
                        a token on position k (e.g. 0 to 1023) 
                        will be aat position k in the final 
                        generated sample. During efficient 
                        inference, since we dont process mask 
                        tokens, we might have "holes", and for 
                        efficiency, we will pack the tokens to 
                        have sequences as short as possible. 
                        That means we could have for examples 
                        4 denoised tokens, but they could 
                        represent tokens at positions (0, 83, 2, 21) 
                        in the final sample. In this case, we
                        need to keep track of the position in 
                        the final sequence to apply the 
                        positional encoding correctly. Shape: (BS, L)
      position_keys: positions to ???
      rotary_cos_sin_queries: cos/sin frequencies to apply to 
                              the queries. 2-tuple.
      rotary_cos_sin_keys: cos/sin frequencies to apply to the 
                           keys. 2-tuple
      concrete_lengths: During efficient inference with a 
                        batch size > 1, some sequences in the 
                        batch might be more noisy than others. 
                        For example, sequence 0 might have 50% 
                        of masked tokens while sequence 1 
                        might have 55% of masked tokens. Since 
                        tensors need to have the same shape on 
                        all dimensions, we need to pad sequences 
                        to the longest sequence in the batch. 
                        This variable holds the number of actual 
                        tokens (non-pads). We assume that the 
                        pad tokens are all at the end (right) 
                        of the sequence. Shape: (BS, L)

    Which arguments are not None:
      - During training:
        - Receive group_idxs, rotary (both), position_queries is None
      - During inference:
        - Receives x, position_queries, rotary (both), group_ixs is None
    """
    group_queries = self._compute_queries(x, position_queries, 
                                          group_idxs, 
                                          concrete_lengths,
                                          use_inference_mode)
    keys, values = self.hidden_to_key_value(x).split(self.hidden_dim, 
                                                     dim=-1)
    pattern = 'bs l (n_heads head_dim) -> bs l n_heads head_dim'
    group_queries = einops.rearrange(group_queries, pattern, n_heads=self.n_heads)
    keys = einops.rearrange(keys, pattern, n_heads=self.n_heads)

    if self.pre_query_mode == 'learn':
      # When using learn pre_query_mode, the queries are shared for all positions
      #  but we need to repeat it to be able to apply RoPE
      if use_inference_mode:
        # Expand to number of queires
        expand_value = position_queries.shape[1]
      else:  # Training
        # Expand to number of keys (keys/queries have the same shape)
        expand_value = keys.shape[1]
      group_queries = group_queries.expand(-1, expand_value, 
                                           -1, -1)
      
    if self.pre_query_mode in ('learn', 'learn+freqs') \
                                and not use_inference_mode:
      group_queries = group_queries.expand(keys.shape[0], -1, 
                                           -1, -1)
    group_queries = apply_rotary_pos_emb_single(group_queries, 
                                                *rotary_cos_sin_queries)
    keys = apply_rotary_pos_emb_single(keys, *rotary_cos_sin_keys)
    values = einops.rearrange(values, 
      'bs l (n_heads head_dim) -> bs l n_heads head_dim', 
      n_heads=self.n_heads)

    out = apply_masked_mha(group_queries, keys, values, cross_attn_mask)
    out = self.output_linear(out)
    return out


class Decoder(nn.Module):
  def __init__(self, n_blocks, dim, n_heads, cond_dim, 
               mlp_ratio, dropout, adaLN, model_length, 
               swap_pre_query_mode, swap_query_process_mode, 
               swap_normalize_mode, decoder_mode, 
               self_attn_first):
    super().__init__()

    self.hidden_dim = dim
    self.n_heads = n_heads
    self.cond_dim = cond_dim
    self.mlp_ratio = mlp_ratio
    self.adaLN = adaLN
    self.model_length = model_length
    self.dropout = dropout
    self.n_blocks = n_blocks
    self.swap_pre_query_mode = swap_pre_query_mode
    self.decoder_mode = decoder_mode
    self.self_attn_first = self_attn_first
    self.group_swap = GroupSwapLayer(dim, n_heads, 
                swap_pre_query_mode, swap_query_process_mode, 
                model_length, swap_normalize_mode)

    if self.decoder_mode.startswith('interleave'):
      # interleave / interleave-k-cross / interleave-k-self
      self._prepare_interleave()
    elif self.decoder_mode.endswith('-cross-first'):
      # k-cross-first
      self._prepare_k_cross_first()
    elif self.decoder_mode == 'cross-only':
      self._prepare_cross_only()
    else:
      raise ValueError(self.decoder_mode)

  def _make_self_attn_block(self):
    return DDiTBlock(self.hidden_dim, self.n_heads, self.adaLN, 
                     self.cond_dim, self.mlp_ratio, self.dropout)
  
  def _make_cross_attn_block(self):
    return CrossAttnDDiTBlock(
      self.hidden_dim, self.n_heads, self.adaLN, self.cond_dim, 
      self.mlp_ratio, self.dropout)

  def _prepare_interleave(self):
    """
    Interleave self & cross attention layers.
    Optionally, use more self/cross layers

    """
    if self.decoder_mode == 'interleave':
      every = 2
      make_every = self._make_self_attn_block
      make_other = self._make_cross_attn_block
    else:
      # Format: interleave-k-cross / interleave-k-self
      chunks = self.decoder_mode.split('-')
      assert len(chunks) == 3, chunks
      assert chunks[0] == 'interleave', chunks
      assert chunks[-1] in ('cross', 'self'), chunks
      every = int(chunks[1]) + 1
      # Meaning we make k times more cross layers
      if chunks[-1] == 'cross':
        make_every = self._make_self_attn_block
        make_other = self._make_cross_attn_block
      else:
        make_every = self._make_cross_attn_block
        make_other = self._make_self_attn_block

    assert self.n_blocks % every == 0, f'{every=}, {self.n_blocks=}'

    layers = []
    start = 0 if self.self_attn_first else 1
    for i in range(start, self.n_blocks + start):
      if i % every == 0:
        layers.append(make_every())
      else:
        layers.append(make_other())
    self.layers = nn.ModuleList(layers)
  
  def _prepare_k_cross_first(self):
    """
    Use k cross attention first, then self-attention only
    """
    chunks = self.decoder_mode.split('-')
    assert len(chunks) == 3, chunks
    n_cross = int(chunks[0])
    assert n_cross < self.n_blocks, f'{n_cross=}, {self.n_blocks=}'
    layers = [self._make_cross_attn_block() 
                                    for _ in range(n_cross)]
    layers += [self._make_self_attn_block() 
                    for _ in range(self.n_blocks - n_cross)]
    self.layers = nn.ModuleList(layers)

  def _prepare_cross_only(self):
    """
    Use cross attention layers only
    """
    self.layers = nn.ModuleList([self._make_cross_attn_block() 
                                for _ in range(self.n_blocks)])

  def forward(
    self, 
    encoder_output,
    t_cond, 
    rotary_cos_sin_queries, 
    rotary_cos_sin_keys, 
    self_attn_mask,
    # Training
    group_idxs, 
    # Inference
    position_queries,
    concrete_lengths_keys, 
    use_inference_mode,
  ):
    """
    1. Apply GroupSwap -> prepare cross attention mask
    2. Apply layers
    """
    if not use_inference_mode:  # Training / Valid
      cross_attn_mask = make_group_cross_attn_mask(group_idxs)
      q_len = self.model_length
    else:  # Sampling
      # TODO: Make sure we don't attend to pad tokens
      q_len = position_queries.shape[1]
      kv_len = encoder_output.shape[1]
      cross_attn_mask = make_inference_cross_attn_mask(
        kv_len, q_len, concrete_lengths_keys)
      # IMPORTANT NOTE: during inference, the self attention 
      #  mask is different than during training, since the
      #  decoder input has a different shape than the encoder
      #  input.
      if self.decoder_mode != 'cross-only':
        concrete_lengths_queries = self.model_length - concrete_lengths_keys
        self_attn_mask = make_inference_self_attn_mask(
          q_len, concrete_lengths_queries)
      else:
        del self_attn_mask  # will not be used

    x = self.group_swap(encoder_output, rotary_cos_sin_queries,
      rotary_cos_sin_keys, group_idxs, position_queries, 
      concrete_lengths_keys, cross_attn_mask, use_inference_mode)

    for layer in self.layers:
      if isinstance(layer, DDiTBlock):  # self attention
        x = layer(x, t_cond, rotary_cos_sin_queries, 
                  self_attn_mask)
      else:  # cross attention
        x = layer(
          q_x=x,
          kv_x=encoder_output,
          t_cond=t_cond, 
          rotary_cos_sin_queries=rotary_cos_sin_queries,
          rotary_cos_sin_keys=rotary_cos_sin_keys,
          attn_mask=cross_attn_mask)
    return x
  

class PartitionDIT(nn.Module):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    assert not config.algo.causal_attention
    self.adaLN = True
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    self.hidden_dim = dim
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    self.cls_embed = LabelEmbedder(
      config.model.n_class, dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)
    self.n_heads = config.model.n_heads
    self.n_class = config.model.n_class
    self.model_length = config.model.length + config.register
    
    self.encoder = Encoder(
      config.model.encoder.n_blocks,
      dim,
      config.model.n_heads,
      dim,
      config.model.mlp_ratio,
      config.model.dropout,
      self.adaLN)
    
    self.decoder = Decoder(
      config.model.decoder.n_blocks,
      dim,
      config.model.n_heads,
      dim,
      config.model.mlp_ratio,
      config.model.dropout,
      self.adaLN,
      self.model_length,
      config.model.swap.pre_query_mode,
      config.model.swap.query_process_mode,
      config.model.swap.normalize_mode,
      config.model.decoder.mode,
      config.model.decoder.self_attn_first)
    
    self.output_layer = DDiTFinalLayer(dim, vocab_size,
                                       dim, 
                                       self.adaLN)
    
    self.prevent_full_partition = config.algo.prevent_full_partition
    self.register = config.register
    if self.register > 0:
      self.reg_tokens = nn.Embedding(self.register, dim)
      nn.init.normal_(self.reg_tokens.weight, std=0.02)

  def forward(
    self, 
    x, 
    # Training
    group_idxs=None, 
    # Inference
    clean_positions=None, 
    noisy_positions=None, 
    concrete_lengths=None,
    use_inference_mode=False,
    y=None,
    drop_label=False
  ):
    x = self.vocab_embed(x)
    if self.register > 0:
      reg_weights = self.reg_tokens.weight
      reg_weights = reg_weights.expand(
        x.shape[0], self.register, self.hidden_dim)
      if use_inference_mode:
        # registers for partition 0
        reg_weights = reg_weights[:, self.register // 2:]
        x = torch.cat([reg_weights, x], dim=1)
        clean_positions = clean_positions + self.register
        noisy_positions = noisy_positions + self.register
        reg_p0_positions = torch.arange(
          self.register // 2, self.register, device=x.device)
        reg_p0_positions = reg_p0_positions.repeat(x.shape[0], 
                                                   1)
        clean_positions = torch.cat(
          [reg_p0_positions, clean_positions], dim=1)

      else:
        # Structure: [reg. part1, reg. part0, bos, *other tokens]
        x = torch.cat([reg_weights, x], dim=1)
        zero = torch.zeros(size=(x.shape[0], self.register // 2), 
                           dtype=x.dtype, device=x.device)
        one = torch.ones(size=(x.shape[0], self.register // 2), 
                         dtype=x.dtype, device=x.device)
        group_idxs = torch.cat([one, zero, group_idxs], dim=1)

    y = torch.where(drop_label, self.n_class, y)
    t_cond = self.cls_embed(y)
    
    rotary_cos_sin = self.rotary_emb(seq_len=self.model_length, 
                                     device=x.device)

    if not use_inference_mode:  # Training
      assert group_idxs is not None
      assert clean_positions is None
      assert noisy_positions is None
      assert concrete_lengths is None
      self_attn_mask = make_group_self_attn_mask(group_idxs)
      rotary_cos_sin_queries = rotary_cos_sin
      rotary_cos_sin_keys = rotary_cos_sin
    else:  # Inference. NOTE: the self-attn mask is only for 
      #      the encoder during inference mode!!!
      assert group_idxs is None
      assert clean_positions is not None
      assert noisy_positions is not None
      assert concrete_lengths is not None
      self_attn_mask = make_inference_self_attn_mask(
        x.shape[1], concrete_lengths)
      rotary_cos_sin_queries = tuple(_index_rotary(vec, noisy_positions) 
                                     for vec in rotary_cos_sin)
      rotary_cos_sin_keys = tuple(_index_rotary(vec, clean_positions) 
                                  for vec in rotary_cos_sin)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      enc_out = self.encoder(x, t_cond, rotary_cos_sin_keys, 
                             self_attn_mask)
      dec_out = self.decoder(enc_out, t_cond, 
        rotary_cos_sin_queries, rotary_cos_sin_keys, 
        self_attn_mask, group_idxs, noisy_positions, 
        concrete_lengths, use_inference_mode)
      if self.register > 0 and not use_inference_mode:
        dec_out = dec_out[:, self.register:, :]
      # During sampling, we predict at the noisy positions 
      #  only, no need to discard tokens.
      out = self.output_layer(dec_out, t_cond)

    return out

  def _gen_eff_non_unif_post_process(self, x, concrete_lengths, 
                                     n_denoise_per_seq, denoised_token_values, clean_positions,
                                     noisy_positions, noisy_pos_input):
    new_concrete_lengths = concrete_lengths + n_denoise_per_seq
    n_tok_to_add = new_concrete_lengths.max() - x.shape[1]
    if n_tok_to_add > 0:
      pad = torch.zeros(size=(x.shape[0], n_tok_to_add), dtype=x.dtype, device=x.device)
      x = torch.cat([x, pad], dim=1)
      clean_positions = torch.cat([clean_positions, pad], dim=1)
    # TODO: vectorize this ugly loop. Doesnt seem to be a big
    #  bottleneck though.
    for i in range(x.shape[0]):
      if n_denoise_per_seq[i] == 0:
        continue
      x[i, concrete_lengths[i]: new_concrete_lengths[i]] = denoised_token_values[i, :n_denoise_per_seq[i]]
      clean_positions[i, concrete_lengths[i]:new_concrete_lengths[i]] = noisy_pos_input[i, :n_denoise_per_seq[i]]
      noisy_positions[i, :noisy_positions.shape[1] - n_denoise_per_seq[i]] = noisy_positions[i, n_denoise_per_seq[i]:].clone()
    return x, clean_positions, new_concrete_lengths
  
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
    
    n_tok_per_normal_step = int(self.config.model.length * dt)
    assert n_tok_per_normal_step > 0, f'{n_tok_per_normal_step=}'
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
  
  def generate_samples_efficient_non_uniform(self, num_samples, num_steps=None, 
                                 inject_bos=None, eps=1e-5):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    assert inject_bos, "Partition MDLM assumes BOS"

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
    
    _, alpha_t = self.noise(timesteps[0])
    _, alpha_s = self.noise(timesteps[0] - dt)
    # We always denoise the same fraction of tokens
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
    
    for t in timesteps[:-1]:
      t = t * torch.ones(x.shape[0], 1, device=self.device)
      n_denoise_per_seq = torch.binomial(count=torch.ones(size=(num_samples,), device=prob_denoise.device) * self.config.model.length, prob=prob_denoise).to(int)
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
