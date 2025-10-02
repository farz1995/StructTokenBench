import os
import h5py
import torch
import numpy as np
from typing import Optional, Iterable

class MissingRepresentation(Exception):
    """Signal to the dataset that this sample has no usable representation and should be skipped."""
    pass


class WrappedMyRepTokenizer:
    # kept for interface compat
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = None
    mask_token_id = None
    vocab_size = None
    pad_value = 0.0

    _warned = set()

    def __init__(
        self,
        h5_path: Optional[str] = None,
        d_model: int = 256,
        device: str = "cpu",
        allowlist_csvs: Optional[Iterable[str]] = None,
        strict_allow: bool = True,
        prefer_exact_chain: bool = True,         # keep True for Remote Homology
        skip_on_missing: bool = True,            # skip samples w/o reps
        missing_log_path: Optional[str] = None,  # CSV to append missing pairs
        residue_index_mode: str = "auto",        # "mmcif" | "auto" | "position"
        use_bare_pdb_fallback: bool = True,      # <= enable PDB-level fallback
        fallback_to_any_chain: bool = False,     # usually False for benchmark
    ):
        # --- paths / state ---
        self.h5_path = h5_path or os.environ.get("MYREP_H5")
        assert self.h5_path, "Provide h5_path=... or set MYREP_H5"
        self.h5_path = os.path.abspath(os.path.expanduser(self.h5_path))
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"H5 not found: {self.h5_path}")

        self.d_model = int(d_model)
        self.device = device

        self._h5lib = h5py
        self._h5 = None

        # --- behavior knobs ---
        self.strict_allow = bool(strict_allow)
        self.prefer_exact_chain = bool(prefer_exact_chain)
        self.skip_on_missing = bool(skip_on_missing)
        self.residue_index_mode = (residue_index_mode or "auto").lower()
        self.use_bare_pdb_fallback = bool(use_bare_pdb_fallback)
        self.fallback_to_any_chain = bool(fallback_to_any_chain)

        # cache: (pdb, CHAIN) -> chosen key
        self._pairs_to_keys: dict[tuple[str, str], str] = {}

        # optional: allowlist
        self._allow_keys = None
        if allowlist_csvs:
            if isinstance(allowlist_csvs, str):
                allowlist_csvs = [allowlist_csvs]
            self.set_allowlist_from_csvs(list(allowlist_csvs))

        # optional: missing log
        self.missing_log_path = None
        if missing_log_path:
            self.missing_log_path = os.path.abspath(os.path.expanduser(missing_log_path))
            os.makedirs(os.path.dirname(self.missing_log_path), exist_ok=True)

    # ---------- helpers ----------

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = self._h5lib.File(self.h5_path, "r")
        return self._h5

    def to(self, device):
        self.device = device
        return self

    def get_num_tokens(self):
        return None  # continuous features

    def get_codebook_embedding(self):
        return None  # continuous features

    def set_allowlist_from_csvs(self, csv_paths):
        """CSV(s) with columns ['pdb','chain'] (chain optional)."""
        import csv
        allow = set()
        for p in csv_paths:
            p = os.path.abspath(os.path.expanduser(p))
            if not os.path.isfile(p):
                continue
            with open(p, "r") as f:
                r = csv.DictReader(f)
                cols = [c.lower() for c in (r.fieldnames or [])]
                has_chain = "chain" in cols
                for row in r:
                    pdb = (row.get("pdb") or row.get("pdb_id") or "").strip().lower()
                    if len(pdb) != 4:
                        continue
                    if has_chain:
                        ch = (row.get("chain") or row.get("chain_id") or "").strip()
                        if len(ch) == 1:
                            allow.add(f"{pdb}_chain_id_{ch}")
                    allow.add(pdb)  # also allow bare
        self._allow_keys = allow

    def _make_residue_index(self, pdb_path: str, chain_for_tok: str, L: int) -> np.ndarray:
        """Prefer mmCIF author indices; fallback to 0..L-1."""
        mode = (self.residue_index_mode or "auto").lower()
        if mode == "position":
            return np.arange(L, dtype=int)
        try:
            try:
                from src.protein_chain import WrappedProteinChain as PC
            except Exception:
                from src.protein_chain import ProteinChain as PC
            pc = PC(pdb_path, chain_for_tok)
            resid = np.asarray(pc.residue_index, dtype=int)
            if resid.shape[0] == L:
                return resid
        except Exception:
            pass
        return np.arange(L, dtype=int)

    # ---------- main API ----------

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence: bool = False):
        """
        Return:
          token_ids:     (L, d_model) float32 tensor
          residue_index: (L,) int numpy array
          seqs:          list[str] length L ('' or 'X')
        """
        h5 = self._get_h5()
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        chain_up = (chain_id or "").strip().upper()

        # allowlist check (if provided)
        if self._allow_keys is not None:
            k_chain = f"{pdb_id}_chain_id_{chain_up}" if chain_up else None
            if self.strict_allow and not (pdb_id in self._allow_keys or k_chain in self._allow_keys):
                if self.skip_on_missing:
                    raise MissingRepresentation(f"({pdb_id},{chain_up}) not in allowlist")
                raise KeyError(f"({pdb_id},{chain_up}) not in allowlist")

        # cached choice?
        if (pdb_id, chain_up) in self._pairs_to_keys:
            k = self._pairs_to_keys[(pdb_id, chain_up)]
            feats = torch.from_numpy(h5[k][()]).to(torch.float32)
            L = int(feats.shape[0])
            resid = self._make_residue_index(pdb_path, chain_up or "", L)
            seqs = (["X"] * L) if use_sequence else ([""] * L)
            return feats, resid, seqs

        def _as_triplet(arr: np.ndarray):
            feats = torch.from_numpy(arr).to(torch.float32)
            if feats.dim() == 2 and feats.shape[1] != self.d_model:
                if (pdb_id, "DIM") not in self._warned:
                    print(f"[MyRep] WARNING: d_model mismatch for {pdb_id}: "
                          f"H5 has {feats.shape[1]}, requested {self.d_model}. Using H5 width.")
                    self._warned.add((pdb_id, "DIM"))
            L = int(feats.shape[0])
            resid = self._make_residue_index(pdb_path, chain_up or "", L)
            seqs = (["X"] * L) if use_sequence else ([""] * L)
            return feats, resid, seqs

        tried = []

        # 1) exact chain key
        if chain_up:
            k = f"{pdb_id}_chain_id_{chain_up}"; tried.append(k)
            if k in h5:
                self._pairs_to_keys[(pdb_id, chain_up)] = k
                return _as_triplet(h5[k][()])
            k = f"{pdb_id}_chain_id_{chain_up.lower()}"; tried.append(k)
            if k in h5:
                self._pairs_to_keys[(pdb_id, chain_up)] = k
                return _as_triplet(h5[k][()])

        # 2) bare PDB fallback (enabled)
        if self.use_bare_pdb_fallback and pdb_id in h5:
            if (pdb_id, chain_up) not in self._warned:
                print(f"[MyRep] Fallback: using bare PDB key '{pdb_id}' for chain '{chain_up}'")
                self._warned.add((pdb_id, chain_up))
            self._pairs_to_keys[(pdb_id, chain_up)] = pdb_id
            return _as_triplet(h5[pdb_id][()])

        # 3) any-available-chain fallback (opt-in)
        if self.fallback_to_any_chain:
            prefix = f"{pdb_id}_chain_id_"
            avail = [k for k in h5.keys() if k.startswith(prefix)]
            if avail:
                chosen = sorted(avail)[0]
                if (pdb_id, chain_up) not in self._warned:
                    preview = ", ".join(sorted(avail)[:6]) + ("..." if len(avail) > 6 else "")
                    print(f"[MyRep] Fallback: using '{chosen}' for requested chain '{chain_up}'. "
                          f"Available: [{preview}]")
                    self._warned.add((pdb_id, chain_up))
                self._pairs_to_keys[(pdb_id, chain_up)] = chosen
                return _as_triplet(h5[chosen][()])

        # 4) nothing found → skip or error
        prefix = f"{pdb_id}_chain_id_"
        candidates = [k for k in h5.keys() if k == pdb_id or k.startswith(prefix)]
        msg = (f"No H5 key for pdb='{pdb_id}', chain='{chain_up}'. Tried {tried}. "
               f"Available: {candidates[:10]}{'...' if len(candidates)>10 else ''}")
        if self.skip_on_missing:
            if self.missing_log_path:
                try:
                    import csv
                    with open(self.missing_log_path, "a", newline="") as f:
                        csv.writer(f).writerow([pdb_id, chain_up])
                except Exception:
                    pass
            raise MissingRepresentation(msg)
        raise KeyError(msg)
