# src/baselines/structsim_tokenizer.py
from dataclasses import dataclass
import h5py, numpy as np, torch
from torch import nn

@dataclass
class StructSimTokenizerCfg:
    h5_path: str
    key_mode: str = "prefix"   # keys like "1234_1abc"; we take the part after first '_'
    normalize: bool = True     # per-residue L2 normalize feats
    proj_dim: int | None = None  # if set, linearly project features to this dim

class StructSimTokenizer(nn.Module):
    """
    Drop-in tokenizer that returns per-residue embeddings from an HDF5 file.
    Expected H5: one dataset per protein, shape (L, D), float32.
    """
    def __init__(self, cfg: StructSimTokenizerCfg):
        super().__init__()
        self.cfg = cfg
        # Lazy-open in worker processes to be dataloader-safe
        self._h5 = None
        # Inspect one vector to build a projection layer if needed
        with h5py.File(cfg.h5_path, "r") as h5:
            any_key = next(iter(h5.keys()))
            in_dim = h5[any_key].shape[1]
        self.proj = nn.Identity() if (cfg.proj_dim is None or cfg.proj_dim == in_dim) \
                    else nn.Linear(in_dim, cfg.proj_dim, bias=False)

    @property
    def h5(self):
        if self._h5 is None:
            # SWMR/libver play nice with multi-process readers
            self._h5 = h5py.File(self.cfg.h5_path, "r", swmr=True, libver="latest")
        return self._h5

    def _map_id_to_key(self, prot_id: str) -> str:
        # Your screenshot shows keys like "103_1g51". If you pass "1g51",
        # we’ll search for "*_1g51". Otherwise use the key verbatim.
        if prot_id in self.h5: 
            return prot_id
        cand = [k for k in self.h5.keys() if k.split("_", 1)[-1] == prot_id]
        if not cand:
            raise KeyError(f"{prot_id} not found in H5")
        return cand[0]

    def encode_one(self, prot_id: str) -> torch.Tensor:
        arr = np.array(self.h5[self._map_id_to_key(prot_id)], dtype=np.float32)  # (L, D)
        x = torch.from_numpy(arr)
        if self.cfg.normalize:
            x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.proj(x)  # (L, D') if projected
        return x

    def batch(self, prot_ids: list[str]) -> dict:
        feats = [self.encode_one(pid) for pid in prot_ids]
        lens = [f.shape[0] for f in feats]
        maxlen = max(lens)
        D = feats[0].shape[1]
        X = torch.zeros(len(feats), maxlen, D, dtype=feats[0].dtype)
        mask = torch.zeros(len(feats), maxlen, dtype=torch.bool)
        for i, f in enumerate(feats):
            L = f.shape[0]
            X[i, :L] = f
            mask[i, :L] = True
        return {"tokens": X, "mask": mask, "lengths": torch.tensor(lens)}
