from .dit import DiT
from .dit_orig import DIT as DiTOrig
from .encoder_decoder_simple import PartitionDIT


def get_backbone(config, vocab_size):
    # set backbone
    mtype = config.model.type
    if mtype == "ddit":
        backbone = DiT(config, vocab_size=vocab_size, adaptive=config.time_conditioning)
    elif mtype == "ddit-orig":
        backbone = DiTOrig(config, vocab_size)
    elif mtype == "encoder-decoder":
        backbone = PartitionDIT(config, vocab_size)
    else:
        raise ValueError(f"Unknown backbone: {mtype}")

    return backbone
