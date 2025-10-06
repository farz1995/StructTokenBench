import os
import h5py
import torch
import numpy as np

class MissingRepresentation(Exception):
    """Signal to the dataset that this sample has no usable representation and should be skipped."""
    pass


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
            prefer_exact_chain: bool = True,  # require exact chain if provided
            skip_on_missing: bool = True,  # raise MissingRepresentation (caught upstream)
            missing_log_path: str | None = None,  # CSV path to append missing pairs
            residue_index_mode: str = "auto",  # "mmcif" | "auto" | "position"
            fallback_to_any_chain: bool = False,  # keep False for remote homology
    ):
        import os
        import h5py

        # --- paths / basic state ---
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
        self.fallback_to_any_chain = bool(fallback_to_any_chain)

        # optional: file to log missing pairs
        self.missing_log_path = None
        if missing_log_path:
            self.missing_log_path = os.path.abspath(os.path.expanduser(missing_log_path))
            os.makedirs(os.path.dirname(self.missing_log_path), exist_ok=True)

        # allowlist (optional)
        self._allow_keys = None  # set[str] of allowed H5 keys
        self._pairs_to_keys = {}  # map (pdb, CHAIN) -> chosen H5 key

        if allowlist_csvs:
            if isinstance(allowlist_csvs, str):
                allowlist_csvs = [allowlist_csvs]
            self.set_allowlist_from_csvs(list(allowlist_csvs))

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

        def _as_triplet(arr):
            feats = torch.from_numpy(arr).to(torch.float32)
            # Optional: sanity-check width
            if feats.dim() == 2 and feats.shape[1] != self.d_model:
                # Don’t crash—just warn once per pdb_id
                if (pdb_id, "DIM") not in self._warned:
                    print(f"[MyRep] WARNING: d_model mismatch for {pdb_id}: "
                          f"H5 has {feats.shape[1]}, requested {self.d_model}. Using H5 width.")
                    self._warned.add((pdb_id, "DIM"))
                # adopt H5 width for this sample
                pass
            L = int(feats.shape[0])
            resid = np.arange(L, dtype=int)
            seqs = (["X"] * L) if use_sequence else ([""] * L)
            return feats, resid, seqs

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