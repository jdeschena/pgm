import torch
import torch.nn.functional as F
import huggingface_hub
from .modules.transformer import Transformer
from .modules.embeddings.time import TimestepEmbedder


class DiT(Transformer, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, adaptive=True):
        super().__init__(config, vocab_size, adaptive=adaptive)
        assert self.causal == False
        if self.adaptive:
            self.sigma_map = TimestepEmbedder(config.model.cond_dim)
            self.forward = self._forward_with_cond
        else:
            self.forward = self._forward_uncond

    def _forward_uncond(self, indices, sigma):
        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)
            x = self.output_layer(x)

        return x

    def _forward_with_cond(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None, c=c)
            x = self.output_layer(x, c)

        return x
