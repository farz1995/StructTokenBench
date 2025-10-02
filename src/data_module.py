import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import *
from tokenizer import *
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from importlib import import_module

from tqdm import tqdm


def load_class(qualname: str):
    """Resolve both fully-qualified and bare class names."""
    if "." in qualname:
        mod, cls = qualname.rsplit(".", 1)
        return getattr(import_module(mod), cls)
    # bare name -> try src.tokenizers.<Name>
    import src.tokenizers as T
    return getattr(T, qualname)


def get_tokenizer_device(tokenizer_device) -> torch.device:
    """Get CPU or local GPU
    """
    if tokenizer_device == "cuda":
        gpu_idx = 0
        if torch.distributed.is_initialized():
            gpu_idx = torch.distributed.get_rank()
        device = torch.device(f"{tokenizer_device}:{gpu_idx}")
    elif tokenizer_device == "cpu":
        device = torch.device(tokenizer_device)
    return device

class ProteinDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer_name: str, tokenizer_device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only: bool, 
        precompute_tokens: bool, tokenizer_kwargs: dict,
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.tokenizer_device = tokenizer_device
        self.tokenizer_kwargs = tokenizer_kwargs
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only
        self.precompute_tokens = precompute_tokens

        if self.test_only:
            self.all_split_names = []
        else:
            self.all_split_names = ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]
        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {} 

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        pass

    # def _setup_tokenizer(self):
    #     """Initialize the tokenizer on appropriate device.
    #     """
    #     device = get_tokenizer_device(self.tokenizer_device)
    #     if self.tokenizer_name.startswith("Wrapped"):
    #         # all Wrapped tokenizers deal with loading logic inside __init__() when built up
    #         # assume only this type of tokenizer needs to be device aware
    #         tokenizer = eval(self.tokenizer_name)(device=device, **self.tokenizer_kwargs)
    #     elif self.tokenizer_name in ["WrappedMyRepTokenizer", "src.tokenizers.WrappedMyRepTokenizer"]:
    #         tokenizer = eval(self.tokenizer_name)(device=device, **self.tokenizer_kwargs)
    #     else:
    #         raise NotImplementedError
    #
    #     self.device_tokenizer_map[device] = tokenizer
    #
    #
    #     return tokenizer

    def _setup_tokenizer(self):
        # if util has a helper, use it; otherwise pass the string
        try:
            from util import get_tokenizer_device
            device = get_tokenizer_device(self.tokenizer_device)
        except Exception:
            device = self.tokenizer_device  # e.g., "cuda" or "cpu"

        kwargs = dict(self.tokenizer_kwargs or {})
        Tok = load_class(
            self.tokenizer_name)  # <-- now resolves both "WrappedMyRepTokenizer" and "src.tokenizers.WrappedMyRepTokenizer"
        tokenizer = Tok(device=device, **kwargs)
        return tokenizer


    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.tokenizer_device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer

    def get_codebook_embedding(self,):
        tokenizer = self.get_tokenizer()
        return tokenizer.get_codebook_embedding()

    def setup_hf_dataset(self, split: str = "train"):
        """Set up the HF-style dataset consumed by the dataloader.

        Continuous reps (get_num_tokens() is None):
          - DO NOT precompute/cache token ids
          - DO NOT filter by 'token_ids' at setup-time
          - DO NOT assert lengths using token_ids (they don't exist yet)
          - Optionally place a lightweight placeholder for 'real_seqs'

        Discrete tokenizers (get_num_tokens() is not None):
          - Keep the original behavior: optional cache + length asserts
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)

        # ------- detect tokenizer type -------
        tok = self.get_tokenizer()
        is_continuous = (getattr(tok, "get_num_tokens", lambda: None)() is None)

        # ------- IMPORTANT: remove earlier "my code" filter -------
        # Don't pre-filter samples by presence of 'token_ids' for continuous reps.
        # The dataset's __getitem__ will call the tokenizer to produce features on the fly.

        # ------- shard dataset -------
        assert torch.distributed.is_initialized()
        process_global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        dataset.shard(shard_idx=process_global_rank, num_shards=world_size)

        # ------- precompute path only for discrete tokenizers -------
        if not is_continuous and getattr(self, "precompute_tokens", False):
            self.py_logger.info(
                f"Precomputing tokenized ids on {process_global_rank} with world size {world_size}..."
            )
            dataset.cache_all_tokenized()

        # ------- post checks / placeholders -------
        from tqdm import tqdm  # ensure tqdm is available
        if dataset.data_name not in ["ConformationalSwitchDataset", "CASP14Dataset", "CAMEODataset"]:
            if not is_continuous:
                # Discrete: keep the original checks that rely on token_ids
                for i in tqdm(range(len(dataset.data))):
                    if "real_seqs" not in dataset.data[i] or dataset.data[i]["real_seqs"] is None:
                        L = len(dataset.data[i]["token_ids"])
                        dataset.data[i]["real_seqs"] = ["X"] * L
                    assert len(dataset.data[i]["real_seqs"]) == len(dataset.data[i]["token_ids"])
            else:
                # Continuous: do NOT touch token_ids. If some downstream code expects
                # 'real_seqs' to exist, give a cheap placeholder (empty list).
                for i in tqdm(range(len(dataset.data))):
                    if "real_seqs" not in dataset.data[i] or dataset.data[i]["real_seqs"] is None:
                        dataset.data[i]["real_seqs"] = []  # length will be validated at __getitem__ time

        # Mark for downstream if anyone cares
        dataset.is_continuous = bool(is_continuous)
        return dataset

    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None
        
        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders


class PretrainingDataModule(pl.LightningDataModule):

    def __init__(self, device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only,
    ):
        super().__init__()

        self.device = device
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only

        self.all_split_names = []
        if not self.test_only:
            self.all_split_names += ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]

        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {}
    
    def _setup_tokenizer(self):
        device = get_tokenizer_device(self.device)
        tokenizer = EsmSequenceTokenizer()
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer

    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer 
    
    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "seq_tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        # need to shard the dataset here:
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            dataset.shard(shard_idx=process_global_rank, num_shards=world_size)
        return dataset
    
    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None

        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders

    def setup(self, stage=None):
        pass
