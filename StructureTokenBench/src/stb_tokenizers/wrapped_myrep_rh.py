# stb_tokenizers/wrapped_myrep_rh.py
import os
import h5py
import numpy as np
import torch

from src.protein_chain import WrappedProteinChain as PC

class WrappedMyRepRemoteHomologyTokenizer:
    """
    Continuous-embedding tokenizer for Remote Homology.
    - Reads per-residue embeddings from an H5 dataset (default: /embeddings_rh)
    - Keys are expected as "<pdb_id>_<chain_id>"
    - Returns a float array [L, D] for each chain
    """

    # match expectations in datasets/base.py
    pad_token_id = 0

    def __init__(
        self,
        h5_path: str | None = None,
        embeddings_dataset: str = "/embeddings_rh",
        key_template: str = "{pdb_id}_{chain_id}",
        fallback_to_any_chain: bool = False,
        device: str | None = None,
        **kwargs,
    ):
        # resolve h5 path
        if not h5_path:
            h5_path = os.environ.get("MYREP_H5")
        if not h5_path:
            raise ValueError(
                "WrappedMyRepRemoteHomologyTokenizer requires an H5. Set +tokenizer_kwargs.h5_path=... or MYREP_H5."
            )
        self.h5_path = os.path.abspath(os.path.expanduser(h5_path))
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"H5 not found: {self.h5_path}")

        self.h5 = h5py.File(self.h5_path, "r")
        if embeddings_dataset not in self.h5:
            raise KeyError(f"Dataset '{embeddings_dataset}' not found in {h5_path}")
        self.emb = self.h5[embeddings_dataset]
        self.key_template = key_template
        self.fallback = fallback_to_any_chain
        self.device = device or "cpu"

        # infer embedding dim from the first item
        first_key = next(iter(self.emb.keys()))
        sample = self.emb[first_key][()]
        if sample.ndim == 1:
            sample = sample[None, :]
        self.embed_dim = int(sample.shape[-1])

    # expected by the runner to detect continuous features
    @property
    def is_continuous(self) -> bool:
        return True

    def __call__(self, pdb_id: str, chain_id: str):
        key = self.key_template.format(pdb_id=pdb_id, chain_id=chain_id)
        if key in self.emb:
            arr = self.emb[key][()]
        elif self.fallback and len(self.emb.keys()) > 0:
            k = next(iter(self.emb.keys()))
            arr = self.emb[k][()]
        else:
            raise KeyError(f"Key '{key}' not found in embeddings dataset.")

        if arr.ndim == 1:
            arr = arr[None, :]
        return arr.astype(np.float32)

    # align with dataset expectations used in TapeRemoteHomologyDataset
    def get_num_tokens(self):
        return None

    @torch.no_grad()
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence: bool = False):
        """Return per-residue continuous features aligned to one chain.
        Outputs:
          token_ids:     (L, D) float32 tensor
          residue_index: (L,) int numpy array (author numbering if available, otherwise 0..L-1)
          seqs:          list[str] length L (dummy or real sequence)
        """
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        chain_up = (chain_id or "").strip().upper()

        # fetch features from H5
        arr = self(pdb_id, chain_up)
        if arr.ndim == 1:
            arr = arr[None, :]
        feats = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        L = int(feats.shape[0])

        # build residue index aligned to mmCIF author numbering if possible, else positions
        try:
            pc = PC.from_cif(pdb_path, chain_up or "detect", id=pdb_id)
            resid = np.asarray(pc.residue_index, dtype=int)
            if resid.shape[0] != L:
                resid = np.arange(L, dtype=int)
            seqs = list(pc.sequence) if use_sequence else ["X"] * L
        except Exception:
            resid = np.arange(L, dtype=int)
            seqs = ["X"] * L

        return feats, resid, seqs

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass
