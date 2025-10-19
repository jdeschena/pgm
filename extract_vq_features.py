# Extract feature from the VQGAN
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from models.vq_model import VQ_models
from dataset.dataloader import get_data
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import os, numpy as np
torch.set_float32_matmul_precision('high')


def tensor2pil(image):
    """ Transform a tensor Image into """
    image = ((image + 1) / 2) * 255
    image = image.permute(1, 2, 0).clip(0, 255).cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


class Extractor:
    def __init__(self, args):
        self.args = args
        self.ae = self.get_network("vqgan-llama")                               # Load VQGAN
        self.patch_size = self.args.img_size // self.args.f_factor
        self.train_data, self.test_data = get_data(
            args.data, 
            img_size=args.img_size, 
            data_folder=args.data_folder, 
            bsize=args.bsize, 
            num_workers=args.num_workers, 
            is_multi_gpus=False, 
            seed=-1, 
            args=args, 
            drop_last=False)
    
    def get_network(self, archi):
        if archi == "vqgan-llama":
            model = VQ_models[f"VQ-{self.args.f_factor}"](
                codebook_size=16384, codebook_embed_dim=8)
            checkpoint = torch.load(self.args.vqgan_folder, 
                                    map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model = model.eval()
            model = model.to(self.args.device)

            if self.args.compile:
                model = torch.compile(model)
        else:
            model = None

        print(f"Size of model {archi}: "
                f"{sum(p.numel() for p in model.parameters() 
                       if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    @torch.no_grad()
    def extract_and_save(self, split):
        if split == "train":
            data = self.train_data
        elif split == "eval":
            data = self.test_data
        else:
            raise ValueError
        
        # create the folder is it does not exist
        root = self.args.dest_folder
        os.makedirs(root, exist_ok=True)
        bar = tqdm(data, leave=False)
        print(f"Extracting {split} dataset with {len(data)} batches...")
        dataset_entries = []
        for idx, (img, y) in enumerate(bar):
            bsize = img.size(0)
            img = img.to(self.args.device)
            _, _, [_, _, code] = self.ae.encode(img)
            code = code.reshape(bsize, -1)
            code = code.detach().cpu().numpy().astype(np.uint16)
            # append bos token to the code
            code = np.concatenate([
                np.full((bsize, 1), self.ae.bos_token_id, 
                        dtype=np.uint16), 
                code], axis=1)
            # save each code
            for i in range(bsize):
                output_dict = {
                    # 1D array of tokens
                    "input_ids": code[i].flatten(), 
                    # 1D array
                    "attention_mask": np.ones_like(code[i]).flatten(),
                    "label": int(y[i].item())
                }
                dataset_entries.append(output_dict)
           
        ds = DatasetDict({
            split: Dataset.from_list(dataset_entries)})
        ds.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "label"])
        ds.save_to_disk(root)
        print(f"Saved {split} dataset to disk at {root}.")


def main(args):
    extractor = Extractor(args)
    extractor.extract_and_save(split="train")
    extractor.extract_and_save(split="eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str, default="timm/imagenet-1k-wds", help="")
    parser.add_argument("--data-folder",  type=str, default=None,         help="data source")
    parser.add_argument("--dest-folder",  type=str, default="./outputs",         help="data destination")
    parser.add_argument("--vqgan-folder", type=str, default="saved_networks/vq_ds8_c2i.pt",         help="vqgan folder")

    parser.add_argument("--bsize",        type=int, default=128,        help="batch size")
    parser.add_argument("--img-size",     type=int, default=256,        help="image size")
    parser.add_argument("--f-factor",     type=int, default=8,          help="downsize factor for tokenizer")
    parser.add_argument("--num-workers",  type=int, default=8,          help="number of workers for loading")
    parser.add_argument("--compile",      action='store_true',          help="compile the network, pytorch 2.0")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Single GPU process
    args.is_master = True
    args.is_multi_gpus = False
    main(args)
