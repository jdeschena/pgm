from pathlib import Path
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoModel
from lightning.fabric import Fabric
from tqdm import trange
import torch
from datetime import datetime
import pandas as pd
from loguru import logger
#import mauve
from transformers import AutoTokenizer
from utils import encode_numbers_to_base64
from loading_utils import get_diffusion, get_diffusion_module
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


CURR_DATETIME_STR = datetime.now().strftime("%y.%m.%d-%H.%M.%S.%f")
datetime.now().strftime("")


def samples_eval(config):
    uncond_eval(config)
    mauve_eval(config)
    lambada_eval(config)


def mauve_eval(config):
    if not config.eval.mauve.run:
        return

    feature_extractor = AutoModel.from_pretrained("gpt2-large").eval().to("cuda:0")

    cond_parent_path = Path(os.getcwd()) / "samples" / "cond"
    files = list(cond_parent_path.rglob("*.npz"))

    all_result_rows = []
    all_metadata = []
    for fname in files:
        npz_file = np.load(fname, allow_pickle=True)

        metadata = npz_file["metadata"].item()
        references = npz_file["references"]
        samples = npz_file["samples"]

        num_steps = metadata.get("num_steps", "NA")
        seq_len = metadata.get("seq_len", "NA")

        logger.info(f"Computing MAUVE for num_steps={num_steps}, seq_len={seq_len}")
        # Eval on first k tokens
        samples = samples[:, : config.eval.mauve.max_num_tokens]
        references = references[:, : config.eval.mauve.max_num_tokens]

        q_features = mauve.utils.featurize_tokens_from_model(
            model=feature_extractor,
            tokenized_texts=torch.tensor(samples),
            batch_size=config.eval.mauve.batch_size,
            name="generated samples",
        ).numpy()

        p_features = mauve.utils.featurize_tokens_from_model(
            model=feature_extractor,
            tokenized_texts=torch.tensor(references),
            batch_size=config.eval.mauve.batch_size,
            name="references",
        ).numpy()

        mauve_results = []
        for run_idx in range(config.eval.mauve.num_rounds):
            res = mauve.compute_mauve(
                p_features=p_features,
                q_features=q_features,
                seed=1 + run_idx,
                device_id=0,
                verbose=False,
                batch_size=config.eval.mauve.batch_size,
                mauve_scaling_factor=config.eval.mauve.scaling_factor,
            ).mauve
            mauve_results.append(float(res))

        mauve_mean = np.mean(mauve_results)
        mauve_std = np.std(mauve_results)

        all_result_rows.append((mauve_mean, mauve_std))
        all_metadata.append(metadata)

    header_keys = set()
    for m in all_metadata:
       header_keys.update(set(m.keys()))

    header_keys = list(header_keys)
    header_keys.sort()
    csv_rows = []
    for res, meta in zip(all_result_rows, all_metadata):
        new_row = []
        for hkey in header_keys:
            new_row.append(meta.get(hkey, "NA"))
        new_row += res
        csv_rows.append(new_row)

    if len(csv_rows) > 0:
        header = header_keys + ["mauve (mean)", "mauve (std)"]
        mauve_res_save_path = (
            Path(os.getcwd()) / "csvs" / CURR_DATETIME_STR / f"mauve.csv"
        )
        mauve_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(csv_rows, columns=header)
        df.to_csv(mauve_res_save_path)
        logger.info(f"MAUVE results on conditional samples:\n{df}\n{'=' * 50}")


def uncond_eval_llama3(config):
    assert config.eval.ppl_with_ar.model == "llama3-8b"
    
    uncond_parent_path = Path(os.getcwd()) / "samples" / "uncond"

    files = list(uncond_parent_path.rglob("*.npz"))
    npz_files = [np.load(f, allow_pickle=True) for f in files]
    all_metadata = [f["metadata"].item() for f in npz_files]
    all_metadata_keys = set([k for d in all_metadata for k in d.keys()])
    all_metadata_keys = sorted(list(all_metadata_keys))

    header = all_metadata_keys + ["ar_ppl", "all_ar_ppl_b64"]
    bs = config.eval.ppl_with_ar.batch_size

    # Load model directly

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ar_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token=os.environ["HF_READ_KEY"]).eval()
    logger.info("Llama 3 loaded")

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision="32",
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    ar_model = fabric.to_device(ar_model)

    rows = []
    for f in npz_files:
        metadata = f["metadata"].item()
        samples = f["samples"]
        # detok
        all_text = gpt2_tokenizer.batch_decode(samples)
        all_losses = []

        for text in tqdm(all_text[::fabric.world_size], desc="Computing loss with llama3..."):
            text = text.replace("<|endoftext|>", "").strip()
            tokens = llama_tokenizer(text)["input_ids"]
            tokens = torch.tensor(tokens).to(ar_model.device)
            tokens = tokens[None,]

            with torch.inference_mode():
                logits = ar_model(tokens).logits.log_softmax(-1)

            loss = - torch.gather(logits[0, :-1], index=tokens[0, 1:, None], dim=-1)[..., 0].mean()
            all_losses.append(float(loss))

        # Communicate between devices
        all_losses = fabric.to_device(torch.tensor(all_losses))
        all_losses = fabric.all_gather(all_losses)
        all_losses = all_losses.flatten()

        avg_loss = all_losses.mean()
        all_losses_list = all_losses.cpu().numpy().tolist()
        all_losses_b64 = encode_numbers_to_base64(all_losses_list)
        
        ppl = avg_loss.exp()

        row = [metadata.get(k, "NA") for k in all_metadata_keys] + [float(ppl), all_losses_b64]
        rows.append(row)
        fabric.barrier()

    if fabric.global_rank == 0:
        uncond_res_save_path = (
            Path(os.getcwd())
            / "csvs"
            / CURR_DATETIME_STR
            / f"uncond_ppl_w_{config.eval.ppl_with_ar.model}.csv"
        )

        df = pd.DataFrame(rows, columns=header)
        uncond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(uncond_res_save_path)
        logger.info(f"AR perplexity of uncond samples:\n{df}\n{'=' * 50}")


def uncond_eval(config):
    if not config.eval.ppl_with_ar.run:
        return

    if config.eval.ppl_with_ar.model == "llama3-8b":
        return uncond_eval_llama3(config)
    
    uncond_parent_path = Path(os.getcwd()) / "samples" / "uncond"

    files = list(uncond_parent_path.rglob("*.npz"))
    npz_files = [np.load(f, allow_pickle=True) for f in files]
    all_metadata = [f["metadata"].item() for f in npz_files]
    all_metadata_keys = set([k for d in all_metadata for k in d.keys()])
    all_metadata_keys = sorted(list(all_metadata_keys))

    header = all_metadata_keys + ["ar_ppl", "all_ar_ppl_b64"]
    model_name = config.eval.ppl_with_ar.model
    ar_model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    bs = config.eval.ppl_with_ar.batch_size

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision="32",
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    ar_model = fabric.to_device(ar_model)

    rows = []
    for f in npz_files:
        metadata = f["metadata"].item()
        samples = f["samples"]
        all_losses = []

        def step_fn(idx):
            batch = samples[idx : idx + bs]
            batch = torch.tensor(batch)
            batch = fabric.to_device(batch)

            with torch.no_grad():
                logits = ar_model(batch).logits[:, :-1]

            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.gather(logits, dim=-1, index=batch[:, 1:, None])[..., 0]

            return loss.mean(-1)

        start = fabric.global_rank * bs
        stop = samples.shape[0]
        step = fabric.world_size * bs
        for idx in trange(
            start, stop, step, desc="Computing AR PPL", disable=fabric.global_rank > 0
        ):
            per_sample_loss = step_fn(idx)
            all_losses.extend(per_sample_loss.cpu().numpy().tolist())

        # Communicate between devices
        all_losses = fabric.to_device(torch.tensor(all_losses))
        all_losses = fabric.all_gather(all_losses)
        all_losses = all_losses.flatten()

        avg_loss = all_losses.mean()
        all_losses_list = all_losses.cpu().numpy().tolist()
        all_losses_b64 = encode_numbers_to_base64(all_losses_list)
        
        ppl = avg_loss.exp()

        row = [metadata.get(k, "NA") for k in all_metadata_keys] + [float(ppl), all_losses_b64]
        rows.append(row)
        fabric.barrier()

    if fabric.global_rank == 0:
        uncond_res_save_path = (
            Path(os.getcwd())
            / "csvs"
            / CURR_DATETIME_STR
            / f"uncond_ppl_w_{config.eval.ppl_with_ar.model}.csv"
        )

        df = pd.DataFrame(rows, columns=header)
        uncond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(uncond_res_save_path)
        logger.info(f"AR perplexity of uncond samples:\n{df}\n{'=' * 50}")


def lambada_eval(config):
    if not config.eval.lambada_openai.run:
        return
    param_cfg = config.parameterization
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)

    module = get_diffusion_module(config)
    diffusion = get_diffusion(config, tokenizer)

    checkpoint_path = config.checkpointing.resume_ckpt_path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() or not checkpoint_path.name.endswith(".ckpt"):
        logger.warning(
            f"Path `{checkpoint_path.absolute()}` does not exist. Evaluating lambada with untrained/original checkpoint."
        )
    else:
        logger.info(f"Evaluating lambada with checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if config.compile:
            diffusion.backbone = torch.compile(diffusion.backbone)
            for k in ckpt.keys():
                assert "_orig_mod" in k, "Cannot use compile=True with this checkpoint"
        else:
            ckpt["state_dict"] = {
                k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()
            }

        diffusion.load_state_dict(ckpt["state_dict"])
        diffusion.load_ema_from_checkpoint(ckpt)

    if not hasattr(module, "eval_lambada"):
        raise ValueError(f"Parameterization `{param_cfg.name}` has no `eval_lambada` implementation.")
    
    module.eval_lambada(diffusion)

