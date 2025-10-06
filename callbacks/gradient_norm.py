from lightning.pytorch.callbacks import Callback
import torch


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        pl_module.log(
            "trainer/grad_norm",
            gradient_norm(pl_module),
            on_step=True,
            on_epoch=False,
            sync_dist=False,  # Should not be needed since gradients are synced before step?
        )


@torch.no_grad()
def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq_norm = (p.grad.data**2).sum()
            total_norm += sq_norm
    total_norm = total_norm**0.5
    return total_norm
