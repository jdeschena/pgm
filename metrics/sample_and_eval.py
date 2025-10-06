import os
import torch
from tqdm import tqdm

from torchmetrics.multimodal.clip_score import CLIPScore

from metrics.inception_metrics import MultiInceptionMetrics


class SampleAndEval:
    def __init__(self, device, is_master, nb_gpus, num_images=10_000, num_classes=1_000, compute_manifold=True, mode="c2i"):
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            device=device, compute_manifold=compute_manifold, num_classes=num_classes,
            num_inception_chunks=10, manifold_k=3, model="inception")

        self.num_images = num_images
        self.device = device
        self.is_master = is_master
        self.nb_gpus = nb_gpus
        self.mode = mode

        if mode == "t2i":
            self.clip_score = CLIPScore("openai/clip-vit-large-patch14").to(device)

    @torch.no_grad()
    def compute_images_features_from_model(self, trainer, sampler, data_loader):
        bar = tqdm(data_loader, leave=False, desc="Computing images features") if self.is_master else data_loader
        import math
        num_batches_per_device = self.num_images / self.nb_gpus / data_loader.batch_size
        num_batches_per_device = int(math.ceil(num_batches_per_device))
        print(f'{self.num_images=}')
        print(f'{self.nb_gpus=}')
        print(f'{data_loader.batch_size=}')
        print(f'Number of batches per device: {num_batches_per_device}')

        for i, (images, labels) in enumerate(bar):
            if i == num_batches_per_device:
                break
            labels = labels.to(self.device)
            if self.mode == "t2i":
                labels = labels[0]  # <- coco does have 5 captions for each img
                gen_images = sampler(trainer=trainer, txt_promt=labels)[0]
                self.clip_score.update(images, labels)
            elif self.mode == "c2i":
                gen_images = sampler(trainer=trainer, nb_sample=images.size(0), labels=labels, verbose=False)[0]
            elif self.mode == "vq":
                code = trainer.ae.encode(images.to(self.device)).to(self.device)
                code = code.view(code.size(0), trainer.input_size, trainer.input_size)
                # Decoding reel code
                gen_images = trainer.ae.decode_code(torch.clamp(code, 0, trainer.args.codebook_size - 1))
            
            assert gen_images.shape[0] == images.shape[0], f'{gen_images.shape=}, {images.shape=}'
            self.inception_metrics.update(gen_images, image_type="fake")
            self.inception_metrics.update(images, image_type="real")
        
        metrics = self.inception_metrics.compute()

        if self.mode == "t2i":
            metrics[f"clip_score"] = self.clip_score.compute().item()
        metrics = {f"{k}": round(v, 4) for k, v in metrics.items()}

        return metrics

