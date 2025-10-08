import os
import h5py
import torch
import numpy as np

class MissingRepresentation(Exception):
    """Signal to the dataset that this sample has no usable representation and should be skipped."""
    pass

def _load_alias_map(path):
    if not path: return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        warnings.warn(f"Could not read alias map at {path}")
        return {}

def _candidate_keys(pdb_id, chain_id):
    pid = pdb_id.lower()
    cid = (chain_id or "").upper()
    base = pid
    # order we will try
    cands = []
    if cid:
        cands += [f"{pid}_chain_id_{cid}", f"{pid}_chain_id_{cid.lower()}"]
    cands += [base]
    return cands

def _resolve_chain_alias(pdb_path, chain_id, available_for_pdb):
    """
    Try to map requested chain_id to one that exists in H5 for this pdb.
    Strategies:
      1) If only one chain exists in H5 for this pdb, use it.
      2) If requested 'C' missing but 'B' present and you KNOW B==C in your rep, map C->B.
      3) Try author-vs-label mapping via mmCIF if gemmi is available.
    """
    # only keys like '7m5f_chain_id_A' ... in this list
    up = [k.upper() for k in available_for_pdb]
    # 1) single option -> use it
    if len(available_for_pdb) == 1:
        return available_for_pdb[0]

    # 2) your specific consolidation rule (adjust as you like)
    # e.g., treat C as B if C missing but B exists
    if chain_id.upper() == "C":
        # look for *_B key
        for k in available_for_pdb:
            if k.upper().endswith("_CHAIN_ID_B"):
                return k

    # 3) author vs label mapping from mmCIF (best-effort)
    try:
        import gemmi
        st = gemmi.read_structure(pdb_path)
        # build a set of author chains present in file
        author_chains = set()
        for mdl in st:
            for ch in mdl:
                author_chains.add(ch.name)  # author asym id
        # if requested chain not in author set, but any available H5 key matches author chains, pick first
        for k in available_for_pdb:
            # extract suffix <X> from *_chain_id_<X>
            if "_chain_id_" in k:
                suffix = k.split("_chain_id_")[-1].upper()
                if suffix in author_chains:
                    return k
    except Exception:
        pass

    return None  # unresolved


def _load_codebook(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Codebook file not found: {path}")
    if path.endswith(".npz"):
        z = np.load(path)
        # allow 'codebook' or the first array key
        if "codebook" in z:
            C = z["codebook"]
        else:
            # take the first array-like
            key = [k for k in z.files if isinstance(z[k], np.ndarray)][0]
            C = z[key]
        return torch.as_tensor(C, dtype=torch.float32)
    if path.endswith(".pt") or path.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            # common names
            for k in ["codebook", "embeddings", "embedding.weight", "codebook.weight"]:
                if k in obj and isinstance(obj[k], (np.ndarray, torch.Tensor)):
                    C = obj[k]
                    return torch.as_tensor(C, dtype=torch.float32)
            # or if the whole object is the codebook tensor
            for v in obj.values():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    C = v
                    return torch.as_tensor(C, dtype=torch.float32)
            raise KeyError("Could not find 'codebook' tensor in checkpoint dict.")
        elif isinstance(obj, torch.Tensor):
            return obj.float()
        else:
            raise TypeError("Unsupported .pt content for codebook.")
    raise ValueError(f"Unsupported codebook file format: {path}")

@torch.no_grad()
def _assign_nearest_codes(feats: torch.Tensor, codebook: torch.Tensor, l2_normalize: bool=False) -> torch.LongTensor:
    """
    feats: [L, D] float32
    codebook: [K, D] float32
    returns: [L] long (indices)
    """
    X = feats
    C = codebook
    if l2_normalize:
        X = torch.nn.functional.normalize(X, dim=-1)
        C = torch.nn.functional.normalize(C, dim=-1)
        # cosine distance == 2 - 2*cos, argmin == argmax cos; we’ll just use matmul and argmax
        sims = X @ C.T  # [L, K]
        idx = torch.argmax(sims, dim=-1)
        return idx.long()
    # Euclidean: argmin ||x - c||^2 = argmin (||x||^2 + ||c||^2 - 2 x·c)
    x2 = (X * X).sum(dim=-1, keepdim=True)       # [L, 1]
    c2 = (C * C).sum(dim=-1).unsqueeze(0)        # [1, K]
    xc = X @ C.T                                  # [L, K]
    d2 = x2 + c2 - 2.0 * xc                       # [L, K]
    idx = torch.argmin(d2, dim=-1)
    return idx.long()




class WrappedMyRepTokenizer:
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = None
    mask_token_id = None
    vocab_size = None
    pad_value = 0.0

    _warned = set()

    def __init__(
            self,
            h5_path=None,
            d_model: int = 256,
            device: str = "cpu",
            allowlist_csvs=None,
            strict_allow: bool = True,
            prefer_exact_chain: bool = True,
            skip_on_missing: bool = True,
            missing_log_path: str | None = None,
            residue_index_mode: str = "auto",
            fallback_to_any_chain: bool = False,
            # codebook / indices knobs:
            codebook_path: str | None = None,
            l2_normalize: bool = False,
            quantize_continuous: bool = True,
            codebook_in_h5: bool = True,
            codebook_dataset: str = "/codebook",
            indices_dataset: str = "/indices",
    ):
        import os
        import h5py, torch

        # 0) Resolve h5_path FIRST (from arg or env), then validate
        if h5_path is None:
            h5_path = os.environ.get("MYREP_H5")

        if not h5_path:
            raise ValueError(
                "WrappedMyRepTokenizer requires an H5 file. "
                "Pass +tokenizer_kwargs.h5_path=/abs/path/to/your.h5 or set MYREP_H5."
            )

        self.h5_path = os.path.abspath(os.path.expanduser(h5_path))
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"H5 not found: {self.h5_path}")

        # 1) Store basic knobs (then you can safely open the H5 later)
        self.d_model = int(d_model)
        self.device = device
        self.allowlist_csvs = allowlist_csvs
        self.strict_allow = bool(strict_allow)
        self.prefer_exact_chain = bool(prefer_exact_chain)
        self.skip_on_missing = bool(skip_on_missing)
        self.missing_log_path = missing_log_path
        self.residue_index_mode = (residue_index_mode or "auto").lower()
        self.fallback_to_any_chain = bool(fallback_to_any_chain)

        # 2) H5 handle (lazy)
        self._h5lib = h5py
        self._h5 = None

        # 3) Codebook / indices config
        self.l2_normalize = bool(l2_normalize)
        self.quantize_continuous = bool(quantize_continuous)
        self.codebook_in_h5 = bool(codebook_in_h5)
        self.codebook_dataset = (codebook_dataset or "/codebook")
        self.indices_dataset = (indices_dataset or "/indices")
        self.codebook = None

        # 4) Load codebook from external file if provided
        if codebook_path:
            self.codebook = _load_codebook(codebook_path)

        # 5) Or load codebook from the SAME H5 (once)
        if self.codebook is None and self.codebook_in_h5:
            with h5py.File(self.h5_path, "r") as h5:
                ds = self.codebook_dataset.lstrip("/")  # "/codebook" -> "codebook"
                if ds in h5:
                    self.codebook = torch.tensor(h5[ds][()], dtype=torch.float32)
                    print(f"[MyRep] loaded codebook from /{ds}: {tuple(self.codebook.shape)}")
                else:
                    print(f"[MyRep] WARNING: '/{ds}' not found in H5. TOP={list(h5.keys())[:20]}")

    def _get_h5(self):
        if self._h5 is None:
            # open read-only per worker/process
            self._h5 = self._h5lib.File(self.h5_path, "r")
        return self._h5

    def to(self, device):
        self.device = device
        return self

    def get_num_tokens(self):
        return None  # continuous features ? no discrete vocab

    def get_codebook_embedding(self):
        return None  # continuous features

    def _make_residue_index(self, pdb_path: str, chain_for_tok: str, L: int) -> np.ndarray:
        """
        Build residue_index aligned with the mmCIF author numbering so that
        BaseDataset._get_selected_indices(...) can crop by residue_range.
        Fallback to simple 0..L-1 if anything goes wrong.
        """
        mode = (self.residue_index_mode or "auto").lower()

        # 1) Force positional indices
        if mode == "position":
            return np.arange(L, dtype=int)

        # 2) Try to use the repo's chain wrapper (preferred)
        try:
            try:
                from src.protein_chain import WrappedProteinChain as PC
            except Exception:
                from src.protein_chain import ProteinChain as PC  # name may differ; repo has one of these
            pc = PC(pdb_path, chain_for_tok)
            resid = np.asarray(pc.residue_index, dtype=int)  # repo exposes this attr in other paths
            # If lengths mismatch (e.g., missing residues), fall back gracefully
            if resid.shape[0] == L:
                return resid
        except Exception:
            pass

        # 3) Last resort: position indices
        return np.arange(L, dtype=int)

    def _load_alias_map(path):
        if not path: return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            warnings.warn(f"Could not read alias map at {path}")
            return {}

    def _candidate_keys(pdb_id, chain_id):
        pid = pdb_id.lower()
        cid = (chain_id or "").upper()
        base = pid
        # order we will try
        cands = []
        if cid:
            cands += [f"{pid}_chain_id_{cid}", f"{pid}_chain_id_{cid.lower()}"]
        cands += [base]
        return cands

    def _resolve_chain_alias(pdb_path, chain_id, available_for_pdb):
        """
        Try to map requested chain_id to one that exists in H5 for this pdb.
        Strategies:
          1) If only one chain exists in H5 for this pdb, use it.
          2) If requested 'C' missing but 'B' present and you KNOW B==C in your rep, map C->B.
          3) Try author-vs-label mapping via mmCIF if gemmi is available.
        """
        # only keys like '7m5f_chain_id_A' ... in this list
        up = [k.upper() for k in available_for_pdb]
        # 1) single option -> use it
        if len(available_for_pdb) == 1:
            return available_for_pdb[0]

        # 2) your specific consolidation rule (adjust as you like)
        # e.g., treat C as B if C missing but B exists
        if chain_id.upper() == "C":
            # look for *_B key
            for k in available_for_pdb:
                if k.upper().endswith("_CHAIN_ID_B"):
                    return k

        # 3) author vs label mapping from mmCIF (best-effort)
        try:
            import gemmi
            st = gemmi.read_structure(pdb_path)
            # build a set of author chains present in file
            author_chains = set()
            for mdl in st:
                for ch in mdl:
                    author_chains.add(ch.name)  # author asym id
            # if requested chain not in author set, but any available H5 key matches author chains, pick first
            for k in available_for_pdb:
                # extract suffix <X> from *_chain_id_<X>
                if "_chain_id_" in k:
                    suffix = k.split("_chain_id_")[-1].upper()
                    if suffix in author_chains:
                        return k
        except Exception:
            pass

        return None  # unresolved
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence: bool = False):
        """Return per-residue continuous features aligned to one chain.
        Outputs:
          token_ids:     (L, d_model) float32 tensor
          residue_index: (L,) int numpy array (0..L-1)
          seqs:          list[str] length L ('' or 'X')
        """
        h5 = self._get_h5()
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        chain_up = (chain_id or "").strip().upper()
        if pdb_id == "7m5f" and chain_up == "C":
            chain_up = "B"

        def _as_triplet(arr):
            import numpy as np
            import torch

            a = np.asarray(arr)
            # 1) already discrete
            if a.ndim == 1 or np.issubdtype(a.dtype, np.integer):
                tok = torch.as_tensor(a, dtype=torch.long)
                L = int(tok.numel())
                residue_index = np.arange(L, dtype=np.int32)
                seqs = None if not use_sequence else ["X"] * L
                return tok, residue_index, seqs

            # 2) continuous (L,d) -> quantize using self.codebook
            if a.ndim == 2 and np.issubdtype(a.dtype, np.floating):
                if self.codebook is None:
                    raise MissingRepresentation(
                        f"Utilization expects discrete token indices, but H5 has float features {a.shape} "
                        f"and NO codebook is loaded. Use +tokenizer_kwargs.codebook_in_h5=true and ensure "
                        f"'{self.codebook_dataset}' exists, or pass +tokenizer_kwargs.codebook_path=..."
                    )
                feats = torch.as_tensor(a, dtype=torch.float32)
                idx = _assign_nearest_codes(feats, self.codebook, l2_normalize=getattr(self, "l2_normalize", False))
                L = int(idx.numel())
                residue_index = np.arange(L, dtype=np.int32)
                seqs = None if not use_sequence else ["X"] * L
                return idx, residue_index, seqs

            raise MissingRepresentation(f"Unsupported array: shape={a.shape}, dtype={a.dtype}")

        idx_ds = self.indices_dataset.lstrip("/")
        if idx_ds in h5:
            cands = []
            if chain_up:
                cands += [
                    f"{idx_ds}/{pdb_id}_chain_id_{chain_up}",
                    f"{idx_ds}/{pdb_id}_chain_id_{chain_up.lower()}",
                ]
            cands += [f"{idx_ds}/{pdb_id}", idx_ds]  # bare pdb, or a flat /indices
            for k in cands:
                key = k.lstrip("/")
                if key in h5:
                    return _as_triplet(h5[key][()])

        tried = []
        # 1) exact chain key
        if chain_up:
            k_exact = f"{pdb_id}_chain_id_{chain_up}"
            tried.append(k_exact)
            if k_exact in h5:
                return _as_triplet(h5[k_exact][()])

            # also try lowercase chain (sometimes stored that way)
            k_low = f"{pdb_id}_chain_id_{chain_up.lower()}"
            tried.append(k_low)
            if k_low in h5:
                return _as_triplet(h5[k_low][()])

        # 2) bare PDB
        tried.append(pdb_id)
        if pdb_id in h5:
            import h5py, numpy as np, torch

            idx_ds = self.indices_dataset.lstrip("/")  # "/indices" -> "indices"
            obj = h5.get(idx_ds, None)

            # If /indices is a GROUP with per-chain datasets, try those.
            if isinstance(obj, h5py.Group):
                candidates = []
                if chain_up:
                    candidates += [
                        f"{idx_ds}/{pdb_id}_chain_id_{chain_up}",
                        f"{idx_ds}/{pdb_id}_chain_id_{chain_up.lower()}",
                    ]
                candidates += [f"{idx_ds}/{pdb_id}"]  # bare pdb if present

                tried_idx = []
                for k in candidates:
                    key = k.lstrip("/")
                    tried_idx.append("/" + key)
                    if key in h5:
                        arr = h5[key][()]  # expect 1D int indices
                        if arr.ndim == 1:
                            tok = torch.as_tensor(arr, dtype=torch.long)
                            L = int(tok.numel())
                            residue_index = np.arange(L, dtype=np.int32)
                            seqs = None if not use_sequence else ["X"] * L
                            return tok, residue_index, seqs
                # Optional debug:
                print(f"[MyRep] /{idx_ds} exists, but none of {tried_idx} found. Subkeys: {list(h5[idx_ds].keys())[:20]}")
            return _as_triplet(h5[pdb_id][()])

        # 3) fallback to first available chain for this pdb
        prefix = f"{pdb_id}_chain_id_"
        avail = sorted([k for k in h5.keys() if k.startswith(prefix)])
        if avail and self.fallback_to_any_chain:
            chosen = avail[0]
            if (pdb_id, chain_up) not in self._warned:
                preview = ", ".join(avail[:6]) + ("..." if len(avail) > 6 else "")
                print(f"[MyRep] WARNING: requested {pdb_id} chain '{chain_up}' not in H5; "
                      f"falling back to '{chosen}'. Available: [{preview}]")
                self._warned.add((pdb_id, chain_up))
            return _as_triplet(h5[chosen][()])

        # 4) no match → helpful error
        candidates = [k for k in h5.keys() if k == pdb_id or k.startswith(prefix)]
        raise KeyError(
            f"No H5 key found for pdb='{pdb_id}', chain='{chain_up}'. "
            f"Tried {tried}. Available for this pdb: {candidates}"
        )