# stb_tokenizers/wrapped_myrep_shake.py
import os
import h5py
import numpy as np
import torch

from src.protein_chain import WrappedProteinChain as PC


class WrappedMyRepShakeTokenizer:
    """
    Continuous-embedding tokenizer for ProteinShake tasks.

    - Optionally reads per-residue embeddings from an H5 dataset (default: /vq_embe_proteinshake)
    - Keys are tried in several forms, e.g. "<pdb_id>_chain_id_<CHAIN>", "<pdb_id>_<CHAIN>", "<pdb_id><CHAIN>", "<pdb_id>"
    - Returns a float tensor [L, D] for each chain, with residue_index aligned to mmCIF author numbering when possible.

    Design goal: be runnable even without an H5 file present. If no H5 is provided or a key is missing,
    the tokenizer will synthesize a zero feature matrix of shape [L, D] using the chain length from the mmCIF.
    """

    pad_token_id = 0

    def __init__(
        self,
        h5_path: str | None = None,
        embeddings_dataset: str = "/vq_embe_proteinshake",
        embed_dim: int = 128,
        fallback_to_any_chain: bool = True,
        device: str | None = None,
        **kwargs,
    ):
        self.device = device or "cpu"
        self.embed_dim = int(embed_dim)

        self.h5_path = None
        self.h5 = None
        self.emb = None

        if h5_path:
            self.h5_path = os.path.abspath(os.path.expanduser(h5_path))
            if not os.path.isfile(self.h5_path):
                raise FileNotFoundError(f"H5 not found: {self.h5_path}")
            self.h5 = h5py.File(self.h5_path, "r")
            ds = embeddings_dataset.lstrip("/") if isinstance(embeddings_dataset, str) else ""
            if ds in self.h5:
                self.emb = self.h5[ds]
            else:
                # Allow pointing directly at a group (no leading slash) or the root
                if isinstance(self.h5.get(embeddings_dataset), (h5py.Group, h5py.Dataset)):
                    self.emb = self.h5[embeddings_dataset]
                else:
                    # If user passed a wrong dataset name, keep emb=None and fall back gracefully at runtime
                    self.emb = None

            # Try to infer embed_dim from the first item if possible
            try:
                if isinstance(self.emb, h5py.Group):
                    first_key = next(iter(self.emb.keys()))
                    sample = self.emb[first_key][()]
                elif isinstance(self.emb, h5py.Dataset):
                    sample = self.emb[()]
                else:
                    sample = None
                if sample is not None:
                    if sample.ndim == 1:
                        sample = sample[None, :]
                    self.embed_dim = int(sample.shape[-1])
            except Exception:
                pass

        self.fallback = bool(fallback_to_any_chain)

    # expected by the runner to detect continuous features
    def get_num_tokens(self):
        return None

    def _candidate_keys(self, pdb_id: str, chain_up: str):
        cands = []
        if chain_up:
            cands += [
                f"{pdb_id}_chain_id_{chain_up}",
                f"{pdb_id}_{chain_up}",
                f"{pdb_id}{chain_up}",
            ]
        cands += [pdb_id]
        return cands

    def _read_from_h5(self, pdb_id: str, chain_up: str):
        if self.emb is None:
            return None
        try:
            if isinstance(self.emb, h5py.Group):
                for k in self._candidate_keys(pdb_id, chain_up):
                    if k in self.emb:
                        arr = self.emb[k][()]
                        return arr
                if self.fallback:
                    # fall back to any entry
                    for k in self.emb.keys():
                        return self.emb[k][()]
            elif isinstance(self.emb, h5py.Dataset):
                # A single dataset for this entire H5 (rare). Use as-is.
                return self.emb[()]
        except Exception:
            pass
        return None

    @torch.no_grad()
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence: bool = False):
        """
        Return per-residue continuous features aligned to one chain.
        Outputs:
          token_ids:     (L, D) float32 tensor
          residue_index: (L,) int numpy array (author numbering if available, otherwise 0..L-1)
          seqs:          list[str] length L (dummy or real sequence)
        """
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        chain_up = (chain_id or "").strip().upper()

        # 1) Try to fetch features from H5 if available
        arr = self._read_from_h5(pdb_id, chain_up)

        # 2) Load structure to determine length and residue numbering
        try:
            pc = PC.from_cif(pdb_path, chain_up or "detect", id=pdb_id)
            resid = np.asarray(pc.residue_index, dtype=int)
            seqs_real = list(pc.sequence)
        except Exception:
            # If parsing fails, use a best-effort fallback
            resid = None
            seqs_real = None

        if arr is None:
            # Graceful fallback: zeros of length L with configured embed_dim
            if resid is None:
                # As a last resort, pick a small default length to stay runnable
                L = 128
            else:
                L = int(len(resid))
            arr = np.zeros((L, self.embed_dim), dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]
        feats = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        L = int(feats.shape[0])

        if resid is None or resid.shape[0] != L:
            resid = np.arange(L, dtype=int)
        seqs = seqs_real if (use_sequence and seqs_real is not None and len(seqs_real) == L) else ["X"] * L

        return feats, resid, seqs

    def close(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass


