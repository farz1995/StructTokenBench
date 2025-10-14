# StructTokenBench/src/stb_tokenizers/wrapped_myrep.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional
import h5py
import numpy as np
import torch
import torch.nn as nn

try:
    import faiss  # optional, faster than torch.cdist on CPU for large codebooks
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False


class WrappedMyRepTokenizer(nn.Module):
    """
    Drop-in tokenizer wrapper for StructTokenBench.
    It returns per-residue integer token ids for a given protein chain.

    Two modes:
      (A) H5 mode: read precomputed token indices from an HDF5 file.
          - pass h5_path=... and indices_dataset=/indices
      (B) Quantize mode: compute tokens from per-residue embeddings with a codebook.
          - pass model_ckpt=... (your model that outputs per-residue embeddings)
          - and either codebook_h5=/codebook or codebook_ndarray=<np.ndarray>

    Expected STB calls `tokenize(item)` where `item` contains a chain identifier.
    We try several common keys: item["chain_id"], item["pdb_chain_id"], item["id"].
    You can adapt `_resolve_chain_id` to your local data convention.
    """

    def __init__(
        self,
        h5_path: Optional[str] = None,
        indices_dataset: Optional[str] = "/indices",
        codebook_h5: Optional[str] = None,
        codebook_dataset: str = "/codebook",
        codebook_ndarray: Optional[np.ndarray] = None,
        model_ckpt: Optional[str] = None,
        device: str = "cpu",
        fallback_to_any_chain: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.pad_token_id = int(pad_token_id)
        self.fallback_to_any_chain = bool(fallback_to_any_chain)

        # (A) precomputed tokens (fastest & simplest for big evals)
        self.h5_path = h5_path
        self.indices_dataset = indices_dataset
        self._h5 = None  # lazy open

        # (B) quantize embeddings on the fly (optional)
        self.codebook = None
        if codebook_ndarray is not None:
            self.codebook = torch.from_numpy(codebook_ndarray.astype(np.float32))
        elif codebook_h5:
            with h5py.File(codebook_h5, "r") as f:
                self.codebook = torch.from_numpy(f[codebook_dataset][...].astype(np.float32))
        if self.codebook is not None:
            self.codebook = self.codebook.to(self.device)

        # Your embedding model (only needed for quantize mode)
        self.model = None
        if model_ckpt is not None:
            # TODO: replace this stub with your actual model loader
            # It must output a (L, D) float32 tensor of per-residue embeddings.
            self.model = self._load_my_model(model_ckpt).to(self.device).eval()

    # ====== Replace with your real model ======
    def _load_my_model(self, ckpt_path: str) -> nn.Module:
        class Dummy(nn.Module):
            def forward(self, item: Dict[str, Any]) -> torch.Tensor:
                # placeholder: produce random (L,D) for demonstration
                L = int(item.get("length", 128))
                D = 128
                return torch.randn(L, D)
        return Dummy()
    # ==========================================

    def _lazy_open_h5(self):
        if self._h5 is None:
            if not self.h5_path or not os.path.exists(self.h5_path):
                raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
            self._h5 = h5py.File(self.h5_path, "r")

    @staticmethod
    def _resolve_chain_id(item: Dict[str, Any]) -> Optional[str]:
        for k in ("chain_id", "pdb_chain_id", "uid", "id", "identifier"):
            if k in item and item[k]:
                return str(item[k])
        meta = item.get("meta") or {}
        for k in ("chain_id", "pdb_chain_id", "uid", "id"):
            if k in meta and meta[k]:
                return str(meta[k])
        return None

    def _tokens_from_h5(self, chain_id: str) -> Optional[np.ndarray]:
        self._lazy_open_h5()
        root = self._h5
        # Expected layout: /indices/<chain_id>  (variable-length int)
        if self.indices_dataset is None:
            # direct lookup at root[chain_id]
            if chain_id in root:
                return np.asarray(root[chain_id][...], dtype=np.int64)
            return None
        # dataset group
        if self.indices_dataset.lstrip("/") in root:
            grp = root[self.indices_dataset]
            if chain_id in grp:
                return np.asarray(grp[chain_id][...], dtype=np.int64)
            # fallback to any entry if asked (useful when chain suffixes differ)
            if self.fallback_to_any_chain and len(grp.keys()) > 0:
                return np.asarray(next(iter(grp.values()))[...], dtype=np.int64)
        return None

    @torch.inference_mode()
    def _tokens_from_quantize(self, item: Dict[str, Any]) -> torch.LongTensor:
        if self.model is None or self.codebook is None:
            raise RuntimeError("Quantize mode requires both model_ckpt and codebook.")
        emb = self.model(item)  # (L, D)
        if not torch.is_tensor(emb):
            emb = torch.from_numpy(np.asarray(emb)).float()
        emb = emb.to(self.device)
        # nearest codebook index for each residue
        if HAVE_FAISS and self.codebook.is_cuda is False:
            index = faiss.IndexFlatL2(self.codebook.shape[1])
            index.add(self.codebook.cpu().numpy())
            D, I = index.search(emb.cpu().numpy(), 1)
            idx = torch.from_numpy(I[:, 0].astype(np.int64))
        else:
            # torch.cdist fallback (works on GPU too)
            dists = torch.cdist(emb.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)  # (L, K)
            idx = torch.argmin(dists, dim=-1).long()
        return idx

    @torch.inference_mode()
    def tokenize(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Returns:
          {
            "input_ids": (L,) LongTensor   # token ids per residue
            "attention_mask": (L,) LongTensor of 1s
          }
        """
        chain_id = self._resolve_chain_id(item)
        tokens = None
        if self.h5_path:
            if chain_id is None:
                if not self.fallback_to_any_chain:
                    raise KeyError("Cannot resolve chain id for H5 lookup.")
            else:
                tokens = self._tokens_from_h5(chain_id)

        if tokens is None:
            # fall back to on-the-fly quantization
            idx = self._tokens_from_quantize(item)  # (L,)
        else:
            idx = torch.from_numpy(tokens.astype(np.int64))

        attn = torch.ones_like(idx, dtype=torch.long)
        return {"input_ids": idx.long(), "attention_mask": attn}

    @property
    def eos_token_id(self) -> int:
        return self.pad_token_id

    @property
    def pad_token_id_(self) -> int:
        return self.pad_token_id
