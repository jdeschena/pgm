import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn
from einops import rearrange

from .embeddings.vocab import EmbeddingLayer
from .embeddings.rotary import Rotary, apply_rotary_pos_emb
from .normalization import LayerNorm
from .scaling import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)
import omegaconf


class TransformerHybrid(nn.Module):
    def __init__(self, config, vocab_size: int, adaptive=True):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        assert not adaptive
        self.config = config
        self.adaptive = adaptive
        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)

        # choose layer times
        Block, FinalLayer = (
            (AdaTransformerBlock, AdaTransformerFinalLayer)
            if adaptive
            else (TransformerBlock, TransformerFinalLayer)
        )

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(
                Block(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = FinalLayer(
            config.model.hidden_size,
            vocab_size,
            config.model.cond_dim,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        cond_dim,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.norm1_ar = LayerNorm(dim)
        self.norm1_diff = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2_ar = LayerNorm(dim)
        self.norm2_diff = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def _flashattention(self, x, rotary_cos_sin, mode):
        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv,
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        x = flash_attn.flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=mode=="ar")
      
        x = rearrange(x, "b s h d -> b s (h d)")
        return x

    def forward(self, x, rotary_cos_sin, mode):
        x_skip = x
        if mode == "ar":
            x = self.norm1_ar(x)
        elif mode == "diff":
            x = self.norm1_diff(x)
        else:
            raise ValueError(mode)
        x = self._flashattention(x, rotary_cos_sin, mode)

        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        if mode == "ar":
            x_norm2 = self.norm2_ar(x)
        elif mode == "diff":
            x_norm2 = self.norm2_diff(x)
        else:
            raise ValueError(mode)
    
        x = bias_dropout_scale_fn(self.mlp(x_norm2), None, scale, x, self.dropout)
        return x


class AdaTransformerBlock(TransformerBlock):
    def __init__(
        self,
        dim,
        n_heads,
        cond_dim,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__(dim, n_heads, cond_dim, mlp_ratio, dropout)
        self.adaLN_modulation_ar = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation_ar.weight.data.zero_()
        self.adaLN_modulation_ar.bias.data.zero_()

    def _modulated_forward(self, x, rotary_cos_sin, c, seqlens=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        x = self._flashattention(x, rotary_cos_sin, seqlens)

        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x

    def forward(self, x, rotary_cos_sin, seqlens=None, c=None):
        if c is not None:
            return self._modulated_forward(x, rotary_cos_sin, c, seqlens)
        else:
            raise NotImplementedError()  # or call superclass


class TransformerFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final_ar = LayerNorm(hidden_size)
        self.norm_final_diff = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x, mode):
        if mode == "ar":
            x = self.norm_final_ar(x)
        elif mode == "diff":
            x = self.norm_final_diff(x)
        else:
            raise ValueError
        return self.linear(x)


class AdaTransformerFinalLayer(TransformerFinalLayer):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__(hidden_size, out_channels, cond_dim)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        return self.linear(x)
