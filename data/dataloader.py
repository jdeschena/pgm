
import os
import datasets
import torch
import torch.distributed
import transformers


from loguru import logger
from lightning import LightningDataModule
from pathlib import Path

from . import utils as dutils

from .detokenizers import *
from .datasets import *
from itertools import chain
import lightning as L
from torchdata.stateful_dataloader import StatefulDataLoader
from loguru import logger


def group_texts(
        examples,
        seq_len,
        key_length="input_ids",
    ):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[key_length])
    # Ensure we only keep chunks of the same size
    total_length = (total_length // seq_len) * seq_len 
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    #total_length = (total_length // seq_len) * seq_len
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
        for k, t in concatenated_examples.items()
    }
    return result


def tokenize_dataset(
        dataset: datasets.Dataset,
        tokenizer: transformers.AutoTokenizer,
        text_key: str = "text",
        target_key: str = "input_ids",
        num_proc: int=1,
        min_seq_len: int=-1,
        seq_len: int =-1,
        group_text: bool =False,
        num_seqs: int=-1,
        add_bos: bool=True,
        add_eos: bool=True,
):
    """Tokenize a dataset into an iterable dataset. Add a key `input_ids` for the tokens

    Args:
        dataset (datasets.Dataset): Huggingface dataset (iterable or not)
        tokenizer_name (str): Name of the tokenizer to use.
        num_proc (int, optional): Number of processes to use to tokenize. Defaults to 1.
        min_seq_len (int, optional): Filter shorter documents. If -1, do not filter. Defaults to -1.
        seq_len (int, optional): If positive, truncate documents to this length. Defaults to -1.
        group_text (bool, optional): If true, will pack documents into chunks of seq_len tokens. Defaults to False.
        remove_text (bool, optional): Remove the "text" field from the dataset. Defaults to False.
        num_seqs (int, optional): Max number of elements in the dataset. If -1, do not truncate the dataset. Defaults to -1.
        add_bos (bool, optional): Whether to add a BOS token (picked from tokenizer). Defaults to True.
        add_eos (bool, optional): Whether to add an EOS token (picked from tokenizer). Defaults to True.

    Returns:
        datasets.Dataset: Processed dataset.
    """

    EOS = tokenizer.eos_token_id
    BOS = tokenizer.bos_token_id

    def tokenize(x):
        tokens = tokenizer(x[text_key], add_special_tokens=False)["input_ids"]
        if add_bos:
            tokens.insert(0, BOS)

        if add_eos:
            tokens.append(EOS)

        return {
            target_key: tokens,
        }

    dataset = dataset.map(tokenize, num_proc=num_proc)

    if min_seq_len > 0:
        dataset = dataset.filter(lambda x: len(x[target_key]) >= min_seq_len, num_proc=num_proc)

    if group_text:
        dataset = dataset.map(
        lambda x: group_texts(x, seq_len), 
        batched=True, 
        batch_size=1000,
        num_proc=num_proc
    )
        
    if seq_len > 0 and not group_text:  # Because group texts already trims
        def trunc(x):
            x[target_key] = x[target_key][:seq_len]
            return x
        dataset = dataset.map(trunc, num_proc=num_proc)

    if num_seqs > 0:
        dataset = dataset.select(range(num_seqs))

    return dataset


def get_dataset(
        dataset_name,
        tokenizer: transformers.AutoTokenizer,
        mode: str,  # train, valid, etc
        cache_dir: str,
        num_proc: int=len(os.sched_getaffinity(0)),
        min_seq_len: int=-1,
        seq_len: int =-1,
        group_text: bool =True,
        remove_text: bool=False,
        num_seqs: int=-1,
        # By default, GPT-2 should have EOS only
        add_bos: bool=False,
        add_eos: bool=True,
        legacy_start_end_bos=False,
        verbose=True,

):
    if legacy_start_end_bos:
        logger.warning("Using Legacy dataloader!!!")
        from . import legacy_dataloader as legacy_dl
        if dataset_name == "lm1b" and mode == "validation":
            logger.warning("Using LM1B test set as validation set")
            mode = "test"
        dataset = legacy_dl.get_dataset(
            dataset_name, 
            tokenizer, 
            wrap=group_text, 
            mode=mode, 
            cache_dir=cache_dir,
            block_size=seq_len,
            num_proc=num_proc
        )
        return dataset

    cache_name = dutils.vars_to_cache_name(
        dataset_name,
        tokenizer=tokenizer.name_or_path,
        mode=mode,
        group_text=group_text,
        seq_len=seq_len,
        min_seq_len=min_seq_len,
        num_seqs=num_seqs,
        add_bos=add_bos,
        add_eos=add_eos,
        remove_text=remove_text,
    )
    dataset_path = Path(cache_dir) / cache_name

    if dutils.fsspec_exists(dataset_path):
        if verbose:
            logger.info(f"Loading data from {dataset_path.name}")
        return datasets.load_from_disk(dataset_path).with_format("torch")
    
    # Actual data preprocessing
    if verbose:
        logger.info(f"Generating new data at: {dataset_path}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "webtext":
        dataset = get_webtext_dataset()
    elif dataset_name == "text8":
        assert group_text
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-crop":
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            #streaming=streaming,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:-25000]",
            cache_dir=cache_dir,
            #streaming=streaming,
        )
    elif dataset_name == "openwebtext-test":
        dataset = dataset.load_dataset(
            "openwebtext",
            split="train[-25000:]",
            cache_dir=cache_dir,
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            trust_remote_code=True,
            cache_dir=cache_dir,
            #streaming=streaming,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            trust_remote_code=True,
            cache_dir=cache_dir,
            #streaming=streaming,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news", 
            cache_dir=cache_dir, 
            #streaming=streaming,
        )
    elif dataset_name == "EleutherAI/lambada_openai":
        dataset = datasets.load_dataset("EleutherAI/lambada_openai", cache_dir=cache_dir)
        #load_dataset(
        #    "EleutherAI/lambada_openai",
        #    "en",
        #    cache_dir=cache_dir, 
        #)
    else:
        dataset = datasets.load_dataset(
            dataset_name, 
            cache_dir=cache_dir, 
            #streaming=streaming
        )


    if dataset_name in ["lambada", "openwebtext-train", "openwebtext-valid", "webtext"]:
        dataset = dataset
    else:
        dataset = dataset[mode]

    text_key = "text"
    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "ptb":
        text_key = "sentence"
        detokenizer = ptb_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        text_key = "article"
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def apply_detokenizer(x):
        x["text"] = detokenizer(x[text_key])

    if detokenizer is not None:
        dataset = dataset.map(apply_detokenizer, num_proc=num_proc)

    if dataset_name != "EleutherAI/lambada_openai":
        tokenized_data = tokenize_dataset(
            dataset,
            tokenizer,
            text_key=text_key,
            num_proc=num_proc,
            min_seq_len=min_seq_len,
            seq_len=seq_len,
            group_text=group_text,
            num_seqs=num_seqs,
            add_bos=add_bos,
            add_eos=add_eos,
        )
    else:
        # split prefix/suffix
        def hide_last_word(e):
            last_space_idx = e["text"].rfind(" ")
            prefix = e["text"][:last_space_idx]
            suffix = e["text"][last_space_idx:]
            e["prefix"] = prefix
            e["suffix"] = suffix
            return e
        
        dataset = dataset.map(hide_last_word)

        dataset = tokenize_dataset(
            dataset,
            tokenizer,
            text_key="prefix",
            target_key="prefix_ids",
            num_proc=num_proc,
            min_seq_len=-1,
            seq_len=-1,
            group_text=False,
            num_seqs=-1,
            add_bos=add_bos,
            add_eos=False,
        )

        tokenized_data = tokenize_dataset(
            dataset,
            tokenizer,
            text_key="suffix",
            target_key="suffix_ids",
            num_proc=num_proc,
            min_seq_len=-1,
            seq_len=-1,
            group_text=False,
            num_seqs=-1,
            add_bos=False,
            add_eos=add_eos,
        )

    # Remove text fields, keeping only tokens
    if remove_text:
        if dataset_name == "ptb":
            tokenized_data = tokenized_data.remove_columns("sentence")
        elif "scientific_papers" in dataset_name:
            tokenized_data = tokenized_data.remove_columns(
                ["article", "abstract", "section_names"]
            )
        elif dataset_name == "ag_news":
            tokenized_data = tokenized_data.remove_columns(["text, label"])
        else:
            tokenized_data = tokenized_data.remove_columns(["text"])

    tokenized_data.save_to_disk(dataset_path)
    tokenized_data = tokenized_data.with_format("torch")
    return tokenized_data


class TextDiffusionDataModule(LightningDataModule):
    def __init__(self, config, tokenizer):
        LightningDataModule.__init__(self)
        self.config = config
        self.tokenizer = tokenizer

        # datasets
        self.train_set = None
        self.valid_set = None
        # loaders
        self._train_loader = None
        self._valid_loader = None

    def debug_print_batch(self, k=64):
        train_ds = self.train_set
        valid_ds = self.valid_set
        #train_ds = self._get_dataset("train", verbose=False)
        #valid_ds = self._get_dataset("validation", verbose=False)

        batch_size = self.config.loader.batch_size
        for ds_type, ds in [("train", train_ds), ("valid", valid_ds)]:
            logger.info(f"Printing {ds_type} batch.")
            batch = ds[:batch_size]
            input_ids = batch["input_ids"]
            logger.info(f"Batch input_ids.shape: {input_ids.shape}")
            
            first = input_ids[0, :k]
            last = input_ids[0, -k:]
            
            logger.info(f"First {k} tokens: {self.tokenizer.decode(first)}")
            logger.info(f"ids: {first}")
            logger.info(f"Last {k} tokens: {self.tokenizer.decode(last)}")
            logger.info(f"ids: {last}")
            logger.info("=" * 50)


    def _get_dataset(self, mode, verbose=True):
        config = self.config
        if mode == "train":
            dataset_name = config.data.train
        elif mode == "validation":
            dataset_name = config.data.valid
        else:
            raise ValueError(f"Unknown mode: `{mode}`")

        ds = get_dataset(
            dataset_name,
            self.tokenizer,
            mode=mode,
            cache_dir=config.data_preprocess.data_cache,
            num_proc=config.loader.num_workers,
            min_seq_len=config.data_preprocess.min_seq_len,
            seq_len=config.data_preprocess.seq_len,
            group_text=config.data_preprocess.group_text,
            remove_text=config.data_preprocess.remove_text,
            num_seqs=config.data_preprocess.num_seqs,
            add_bos=config.data_preprocess.add_bos,
            add_eos=config.data_preprocess.add_eos,
            legacy_start_end_bos=config.data_preprocess.legacy_start_end_bos,
            verbose=verbose,
        )

        return ds

    def prepare_data(self):
        # This is executed on ONE process (eg download, tokenize)
        # Get download data + tokenize
        self.train_set = self._get_dataset("train")
        self.valid_set = self._get_dataset("validation")
        self.debug_print_batch()

        logger.info(f"Train set length: {len(self.train_set)}")
        logger.info(f"Valid set length: {len(self.valid_set)}")

    def setup(self, stage):
        # This is executed on EACH gpu process
        self.train_set = self._get_dataset("train", verbose=False)
        self.valid_set = self._get_dataset("validation", verbose=False)

    def train_dataloader(self):
        config = self.config

        loader = StatefulDataLoader(
            self.train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            persistent_workers=config.loader.persistent_workers,
            drop_last=True,
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        config = self.config

        loader = torch.utils.data.DataLoader(
            self.valid_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,
        )

        return loader
    
    def validate_config(self):
        num_gpus = torch.cuda.device_count()
        config = self.config
        if (
            config.loader.global_batch_size
            % (num_gpus * config.trainer.accumulate_grad_batches)
            != 0
        ):
            raise ValueError(
                f"Train Batch Size {config.training.batch_size}"
                f"not divisible by {num_gpus} gpus with accumulation "
                f"{config.trainer.accumulate_grad_batches}."
            )
        if config.loader.eval_global_batch_size % num_gpus != 0:
            raise ValueError(
                f"Eval Batch Size for {config.eval.batch_size} "
                f"not divisible by {num_gpus}."
            )
        
