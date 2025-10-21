import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import *
from tokenizer import *
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from importlib import import_module

from tqdm import tqdm

def _build_or_get_45class_map(self, dataset, split):
    """
    Build a 45-class mapping on TRAIN, reuse it for VALID/TEST.
    Produces:
      - self._label_keep_set: set of old fold labels kept
      - self._label_remap:    dict old_label -> new_label (0..44)
      - self.num_labels_for_model = 45
    """
    from collections import Counter
    import os, torch

    # if already built (from train), reuse
    if hasattr(self, "_label_keep_set") and hasattr(self, "_label_remap"):
        return self._label_keep_set, self._label_remap

    assert split == "train", "45-class map must be built on train split first."

    # 1) collect all fold labels in train
    train_labels = [int(s["fold_label"]) for s in dataset.data if "fold_label" in s]

    # 2) choose which 45 to keep
    # Option A: user-specified list via Hydra: +data.fold_allowlist=[...] (exactly 45 ints)
    allow = getattr(self, "fold_allowlist", None)
    if allow:
        keep = list(map(int, allow))
        assert len(keep) == 45, "fold_allowlist must contain exactly 45 labels"
    else:
        # Option B: pick the 45 most frequent folds in TRAIN
        counts = Counter(train_labels).most_common(45)
        keep = [lab for lab, _ in counts]

    # 3) make a stable mapping old -> new (sorted then enumerate)
    keep_sorted = sorted(keep)
    remap = {old: new for new, old in enumerate(keep_sorted)}

    # 4) stash for later splits and for logging
    self._label_keep_set = set(keep_sorted)
    self._label_remap = remap
    self.num_labels_for_model = 45

    # (optional) save mapping to disk so VALID/TEST workers can reuse
    if hasattr(self, "save_dir_path") and self.save_dir_path:
        os.makedirs(self.save_dir_path, exist_ok=True)
        torch.save(
            {"keep": keep_sorted, "map": remap},
            os.path.join(self.save_dir_path, "fold45_map.pt"),
        )

    return self._label_keep_set, self._label_remap


def _apply_45class_filter_and_remap(self, dataset, split):
    """
    Filter samples not in the kept folds and remap kept labels to 0..44.
    Returns the modified dataset.
    """
    import os, torch

    # load or build mapping
    if hasattr(self, "_label_keep_set") and hasattr(self, "_label_remap"):
        keep_set, remap = self._label_keep_set, self._label_remap
    else:
        # try to load saved mapping (if VALID/TEST is constructed first by any chance)
        mapping_path = getattr(self, "save_dir_path", None)
        if mapping_path:
            mapping_path = os.path.join(mapping_path, "fold45_map.pt")
        if mapping_path and os.path.isfile(mapping_path):
            obj = torch.load(mapping_path, map_location="cpu", weights_only=False)
            keep_set = set(obj["keep"])
            remap = obj["map"]
            self._label_keep_set, self._label_remap = keep_set, remap
            self.num_labels_for_model = 45
        else:
            # build now (will assert split == "train")
            keep_set, remap = self._build_or_get_45class_map(dataset, split)

    # filter + remap
    clean = []
    drop = 0
    for s in dataset.data:
        old = int(s["fold_label"])
        if old in keep_set:
            s["fold_label"] = remap[old]  # now 0..44
            clean.append(s)
        else:
            drop += 1
    dataset.data = clean
    msg = f"[45-class] split={split}: kept {len(clean)}, dropped {drop}"
    try:
        self.py_logger.info(msg)
    except Exception:
        print(msg)

    return dataset


def load_class(qualname: str):
    """Resolve both fully-qualified and bare class names."""
    if "." in qualname:
        mod, cls = qualname.rsplit(".", 1)
        # Backward-compat: accept alias "src.tokenizers" -> actual "src.stb_tokenizers"
        if mod == "src.tokenizers" or mod.startswith("src.tokenizers"):
            mod = mod.replace("src.tokenizers", "src.stb_tokenizers", 1)
        try:
            return getattr(import_module(mod), cls)
        except ModuleNotFoundError:
            # Fallback to local tokenizers package
            import src.stb_tokenizers as T
            return getattr(T, cls)
    # bare name -> try src.stb_tokenizers.<Name>
    import src.stb_tokenizers as T
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
    #     elif self.tokenizer_name in ["WrappedMyRepTokenizer", "src.stb_tokenizers.WrappedMyRepTokenizer"]:
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
            self.tokenizer_name)  # <-- now resolves both "WrappedMyRepTokenizer" and "src.stb_tokenizers.WrappedMyRepTokenizer"
        tokenizer = Tok(device=device, **kwargs)
        # Tok = load_class(self.cfg.tokenizer_module, self.cfg.tokenizer)
        # kwargs = dict(self.cfg.tokenizer_kwargs or {})
        # tokenizer = Tok(device=device, **kwargs)
        self.tokenizer = tokenizer  # <-- add this line
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

        # ------- Remote Homology: enforce 45-class mapping BEFORE sharding -------
        try:
            from dataset.tape_remote_homology import TapeRemoteHomologyDataset
            is_rh = isinstance(dataset, TapeRemoteHomologyDataset) and getattr(dataset, "target_field", "") == "fold_label"
        except Exception:
            is_rh = False
        if is_rh:
            # Build (on train) and/or load mapping, then filter + remap to 0..44
            if split == "train":
                _build_or_get_45class_map(self, dataset, split)
            dataset = _apply_45class_filter_and_remap(self, dataset, split)
            # downstream consumers may consult this
            setattr(self, "num_labels_for_model", 45)

        # ------- detect tokenizer type -------
        tok = self.get_tokenizer()
        is_continuous = (getattr(tok, "get_num_tokens", lambda: None)() is None)

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
