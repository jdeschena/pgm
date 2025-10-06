import torch

def sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=0.0,
    filter_value=-float("Inf"),
    dim=-1,
):
    """Filter a distribution of logits using top-k/top-p (nucleus) filtering. Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
        logits (Tensor): Tensor of logits
        top_k (int, optional): Number of top values to keep. Deactivated if k is 0. Defaults to 0.
        top_p (float, optional): Cumulative mass to retain. Deactivated if p = 0. Defaults to 0.0.
        filter_value (float, optional): Fill value to replace the entries removed by top-k/top-p filtering. Defaults to -float('Inf').
        dim (int, optional): Dimension of the filtering. Defaults to -1.

    Returns:
        logits: Tensor whose axis `dim` was filtered.
    """
    if dim != -1:
        logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        values, idxs = torch.topk(logits, k=top_k, dim=-1)
        to_remove_mask = (
            logits < torch.min(values, dim=-1, keepdim=True)[0]
        )  # min returns a tuple (values, indices)
        logits[to_remove_mask] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cum_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        # This is done in the huggingface code, I imagine to make sure at least one token is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask_to_remove = torch.empty_like(sorted_indices_to_remove)
        mask_to_remove.scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[mask_to_remove] = filter_value

    if dim != -1:
        logits = torch.transpose(logits, dim, -1)

    return logits
