import math
import torch


def get_mask_code(code, r=None, mode="arccos", value=None, bos=None, **kargs):
    """ Replace the code token by *value* according the the *mode* scheduler
       :param
        code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
        mode  -> str:                the rate of value to mask
        value -> int:                mask the code by the value
       :return
        masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
        mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
    """
    b, h, w = code.size()
    device = code.device
    if r is None:
        r = torch.rand(b)
    if mode == "root":      # root scheduler
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":  # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":  # square scheduler
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":  # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":  # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        raise ValueError
    
    masked_code = code.detach().clone()
    if bos is not None:
        masked_code = masked_code.reshape(b, -1) 
        bos_seq = torch.full((b, 1), bos, dtype=code.dtype, 
                             device=device)
        masked_code = torch.cat([bos_seq, masked_code], dim=-1)
        # print(masked_code.shape, code.shape)    
        rand = torch.rand(size=masked_code.size(), device=device)
        mask = rand < val_to_mask.view(b, 1).to(device)
    else:
        mask = torch.rand(size=masked_code.size()).to(device) < val_to_mask.view(b, 1, 1).to(device)
        if value > 0: 
            masked_code[mask] = torch.full_like(masked_code[mask], value)

    loss_mask = torch.ones_like(code).bool()
    return masked_code, mask, loss_mask

