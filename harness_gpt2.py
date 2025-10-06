import torch

from tqdm import trange
from pathlib import Path

from loguru import logger

import torch

from pathlib import Path
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer

from data import dataloader
from tqdm import trange
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from transformers import GPT2LMHeadModel
from run_eval import CURR_DATETIME_STR


"""

Command for paper:
     python harness_gpt2.py --add_bos --add_eos

"""


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--data_cache",
        type=str,
    )
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--add_bos", action="store_true")
    parser.add_argument("--add_eos", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./gpt2_lambada")


def eval_loop(args, model, dataset, size):
    all_losses = []
    all_last_correct = []

    for idx in trange(len(dataset), desc=f"Evaluating lambada ({size})..."):
        prefix = dataset[idx]["prefix_ids"]
        suffix = dataset[idx]["suffix_ids"]

        full_seq = torch.cat([prefix, suffix], dim=-1).cuda()
        out = model(full_seq).logits.log_softmax(-1)

        if args.add_eos:
            preds = out[-len(suffix) - 1 : -2].cuda()
            targets = suffix[:-1].cuda()
        else:
            preds = out[-len(suffix) - 1 : -1].cuda()
            targets = suffix.cuda()

        loss = -torch.gather(preds, dim=-1, index=targets[..., None])[..., 0].sum()
        pred_tokens = preds.argmax(-1)
        correct = (pred_tokens == targets).all().item()

        all_losses.append(loss.cpu().item())
        all_last_correct.append(correct)

    acc = np.mean(all_last_correct)
    avg_loss = np.mean(all_losses)

    return acc, avg_loss


@torch.no_grad
def main(args):
    logger.info("Starting eval acc/ppl on openai lambada with GPT2")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = dataloader.get_dataset(
        "EleutherAI/lambada_openai",
        tokenizer,
        mode="test",
        cache_dir=args.data_cache,
        num_proc=args.num_proc,
        group_text=False,
        remove_text=False,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    sm_acc, sm_loss = eval_loop(args, model, dataset, "sm")
    del model

    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").cuda()
    md_acc, md_loss = eval_loop(args, model, dataset, "md")
    del model

    model = GPT2LMHeadModel.from_pretrained("gpt2-large").cuda()
    large_acc, large_loss = eval_loop(args, model, dataset, "large")
    del model

    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").cuda()
    xl_acc, xl_loss = eval_loop(args, model, dataset, "xlarge")
    del model

    csv_save_path = Path(args.save_dir) / CURR_DATETIME_STR / "lambada.csv"

    rows = [
        [args.add_bos, args.add_eos, "sm", float(sm_acc), float(np.exp(sm_loss))],
        [args.add_bos, args.add_eos, "md", float(md_acc), float(np.exp(md_loss))],
        [
            args.add_bos,
            args.add_eos,
            "large",
            float(large_acc),
            float(np.exp(large_loss)),
        ],
        [args.add_bos, args.add_eos, "xlarge", float(xl_acc), float(np.exp(xl_loss))],
    ]

    header = ["add_bos", "add_eos", "size", "acc", "ppl"]

    df = pd.DataFrame(rows, columns=header)
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_save_path)
    logger.info(f"Lambada results: \n{df}\n{'=' * 50}")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
