import hydra.utils
import lightning as L
import torch
import abc


class CoreLightning(L.LightningModule, abc.ABC):
    def __init__(self, config):
        L.LightningModule.__init__(self)
        self.config = config
        try:
            self.save_hyperparameters()
        except KeyError:
            pass
        self.ema = None
        # Flag to ensure we don't overwrite the original weights
        # If loading the ema, but the ema is already loaded
        self._using_ema_weights = False

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def iter_params(self):
        raise NotImplementedError

    def configure_optimizers(self):
        assert self.config.optim.name == "adamw"
        optimizer = torch.optim.AdamW(
            self.iter_params(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def on_load_checkpoint(self, checkpoint):
        self.load_ema_from_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        self.save_ema_to_checkpoint(checkpoint)

    def on_train_start(self):
        self.move_ema_shadow_params_to_device()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.update_ema()

    def on_validation_epoch_start(self):
        self.store_ema()
        self.backbone.eval()

    def on_validation_epoch_end(self):
        self.restore_ema()

    def on_train_epoch_start(self):
        self.backbone.train()

    def store_ema(self):
        if self.ema and not self._using_ema_weights:
            self.ema.store(self.iter_params())
            self.ema.copy_to(self.iter_params())
            self._using_ema_weights = True

    def restore_ema(self):
        if self.ema and self._using_ema_weights:
            self.ema.restore(self.iter_params())
            self._using_ema_weights = False

    def update_ema(self):
        if self.ema:
            self.ema.update(self.iter_params())

    def load_ema_from_checkpoint(self, checkpoint):
        if self.ema:
            return self.ema.load_state_dict(checkpoint["ema"])

    def save_ema_to_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        return checkpoint

    def move_ema_shadow_params_to_device(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
