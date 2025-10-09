# Main file to launch training or evaluation
import os
import random
import numpy as np
import argparse
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from torch.distributed import init_process_group, destroy_process_group
import hydra
from omegaconf import DictConfig, OmegaConf
import time


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_train(args):
    """ Launch multi training"""

    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    args.is_master = args.global_rank == 0
    args.nb_gpus = torch.distributed.get_world_size()
    args.bsize = args.global_bsize // args.nb_gpus
    if args.is_master:
        print(f"{args.nb_gpus} GPU(s) found, launch DDP")
    print(f"Global rank {args.global_rank} on device {args.device} with batch size {args.bsize}")
    args.num_nodes = int(os.environ.get("SLURM_NNODES", 1))  # Get number of nodes from SLURM, default to 1
    train(args)
    destroy_process_group()

def sample_speed(args, maskgit):
    sampler = maskgit.sampler

    # sample with batch size = 1 for latency
    timings = []
    for _ in range(12):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = sampler(maskgit, nb_sample=1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append(end - start)
    
    timings = timings[2:]
    latency = np.mean(timings)

    # Sample with batch size = 32 for 
    timings = []
    for _ in range(12):
        torch.cuda.synchronize()
        start = time.perf_counter()
        sampler(maskgit, nb_sample=32)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append(end - start)
    
    timings = timings[2:]
    throughput = 32 / np.mean(timings)  # images/sec
    return dict(latency=latency, throughput=throughput)


def train(args):
    """ Main function: Train or eval MaskGIT """
    if args.mode == "cls-to-img":
        from trainer.cls_trainer import MaskGIT
    elif args.mode == "txt-to-img":
        from trainer.txt_trainer import MaskGIT
    else:
        raise "What is this mode ?????"
    maskgit = MaskGIT(args)

    if args.sample_speed:
        results = sample_speed(args, maskgit)
        print(results)
    elif args.test_only:
        eval_sampler = maskgit.sampler
        maskgit.eval(sampler=eval_sampler, num_images=50_000, save_exemple=True, compute_pr=False,
                     split="test", mode="c2i", data=args.data.split("_")[0])
    else:
        maskgit.fit()

    return 0

@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(cfg: DictConfig):
    args = OmegaConf.to_container(cfg, resolve=True)
    args = OmegaConf.create(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.iter = 0
    args.global_epoch = 0
 
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    world_size = torch.cuda.device_count()
    if world_size > 1:
        args.is_multi_gpus = True
        launch_multi_train(args)
    else:
        print(f"{world_size} GPU found")
        args.global_rank = 0
        args.num_nodes = 1
        args.is_master = True
        args.is_multi_gpus = False
        args.nb_gpus = 1
        args.bsize = args.global_bsize
        train(args)

if __name__ == "__main__":
    main()
