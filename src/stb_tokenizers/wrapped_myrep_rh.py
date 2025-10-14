# stb_tokenizers/wrapped_myrep_rh.py
import h5py
import numpy as np

class WrappedMyRepRemoteHomologyTokenizer:
    """
    Continuous-embedding tokenizer for Remote Homology.
    - Reads per-residue embeddings from an H5 dataset (default: /embeddings_rh)
    - Keys are expected as "<pdb_id>_<chain_id>"
    - Returns a float array [L, D] for each chain
    """

    def __init__(
        self,
        h5_path: str,
        embeddings_dataset: str = "/embeddings_rh",
        key_template: str = "{pdb_id}_{chain_id}",
        fallback_to_any_chain: bool = False,
        **kwargs,
    ):
        self.h5 = h5py.File(h5_path, "r")
        if embeddings_dataset not in self.h5:
            raise KeyError(f"Dataset '{embeddings_dataset}' not found in {h5_path}")
        self.emb = self.h5[embeddings_dataset]
        self.key_template = key_template
        self.fallback = fallback_to_any_chain

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

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass
