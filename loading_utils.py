from core import distill

def get_diffusion(config, tokenizer):
    mode = config.parameterization.name
    if mode == "multi-round-sdtt":
        return distill.multi_round_sdtt.MultiRoundSDTT(config, tokenizer)
    elif mode == "partition-sdtt":
        return distill.multi_round_partition.MultiRoundSDTT(config, tokenizer)
    else:
        raise ValueError(f"Unknown parameterization `{mode}`")


def get_diffusion_module(config):
    mode = config.parameterization.name
    if mode == "multi-round-sdtt":
        return distill.multi_round_sdtt
    elif mode == "partition-sdtt":
        return distill.multi_round_partition
    else:
        raise ValueError(f"Unknown parameterization `{mode}`")
