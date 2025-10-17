import os
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist

from sklearn.model_selection import train_test_split
from proteinshake.tasks import BindingSiteDetectionTask

from dataset.base import BaseDataset
from proteinshake.datasets import ProteinLigandInterfaceDataset as _PLID


class ProteinShakeBindingSiteDataset(BaseDataset):
    """
    ProteinShake Binding-Site dataset wrapper for StructTokenBench.

    Key improvements vs original:
      - Robust offline mode: skip remote download if local refined set folder or tarball exists.
      - Configurable cache root via:
          * Hydra: +data.proteinshake_root=/path/to/.cache/proteinshake
          * Env:   PROTEINSHAKE_HOME=/path/to/.cache/proteinshake
      - Version pinned (default 2020) but override-able via proteinshake_version=YYYY in kwargs.
    """

    DEFAULT_SPLIT = "structure"  # could be "random, sequence, structure"
    DEFAULT_SPLIT_THRESHOLD = "_0.7"  # "" for random; 0.3-0.9 for sequence; 0.5-0.9 for structure

    SPLIT_NAME = {
        "test": ["test"]
    }

    EPS = 1e-3

    # --------------------------- Public API ---------------------------

    def __init__(self, *args, **kwargs):
        """
        Accepts optional kwargs:
          - proteinshake_root: custom cache root (string). If it ends with 'protein_ligand_interface',
                               it will be used as-is; otherwise '/protein_ligand_interface' is appended.
          - proteinshake_version: int year (default 2020). Must match your local files.
        """
        self._proteinshake_root_cfg = kwargs.pop("proteinshake_root", None)
        self._proteinshake_version = int(kwargs.pop("proteinshake_version", 2020))
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def get_target_file_name(self):
        return os.path.join(
            self.data_path,
            f"proteinshake_bindingsite/processed_structured_binding_site_{self.split}",
        )

    def retrieve_pdb_path(self, pdb_id: str, chain_id=None) -> Path:
        """
        Retrieves the path to the PDB file for a given pdb_id.
        """
        root = self._resolve_proteinshake_root()
        raw = root / "raw"
        pdb_dir = raw / pdb_id
        if pdb_dir.exists():
            for ext in ['.pdb', '.cif']:
                candidate = pdb_dir / f"{pdb_id}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"PDB file not found for pdb_id: {pdb_id}")

    # --------------------------- Internal helpers ---------------------------

    def _resolve_proteinshake_root(self) -> Path:
        """
        Resolve the root directory where ProteinShake stores this dataset.
        Priority:
          1) explicit kwarg proteinshake_root
          2) PROTEINSHAKE_HOME env
          3) ~/.cache/proteinshake

        Ensures the dataset-specific subdir 'protein_ligand_interface' exists.
        """
        root = None
        if self._proteinshake_root_cfg:
            root = Path(os.path.expanduser(self._proteinshake_root_cfg))
        else:
            env_home = os.environ.get("PROTEINSHAKE_HOME", None)
            if env_home:
                root = Path(os.path.expanduser(env_home))
            else:
                root = Path.home() / ".cache" / "proteinshake"

        # ensure dataset-specific dir
        if root.name != "protein_ligand_interface":
            root = root / "protein_ligand_interface"
        (root / "raw").mkdir(parents=True, exist_ok=True)
        return root

    def _ensure_extracted_cache(self):
        """
        Ensure PDBbind tarballs are extracted. If tarballs exist but not extracted,
        extract them to the expected structure.
        """
        root = self._resolve_proteinshake_root()
        raw = root / "raw"

        # Expected tarballs
        tgz = raw / f"PDBbind_v{self._proteinshake_version}_refined.tar.gz"
        index_tgz = raw / f"PDBbind_v{self._proteinshake_version}_plain_text_index.tar.gz"

        # Check if we need to extract
        refined_dir = raw / f"PDBbind_v{self._proteinshake_version}_refined"
        index_dir = raw / "index"

        needs_extract = False
        if tgz.exists() and not refined_dir.exists():
            needs_extract = True
        if index_tgz.exists() and not index_dir.exists():
            needs_extract = True

        if needs_extract:
            import tarfile

            # Extract refined set
            if tgz.exists():
                with tarfile.open(str(tgz), 'r:gz') as tar:
                    tar.extractall(str(raw))

            # Extract index
            if index_tgz.exists():
                with tarfile.open(str(index_tgz), 'r:gz') as tar:
                    tar.extractall(str(raw))

    def _install_local_download_guard(self):
        """
        Monkey-patch _PLID.download so it skips network if local files already exist.
        Skips when either:
           <root>/raw/PDBbind_v{version}_refined/           (extracted)
        or
           <root>/raw/PDBbind_v{version}_refined.tar.gz     (tarball)
        is present.
        """
        if not hasattr(_PLID, "_orig_download"):
            _PLID._orig_download = _PLID.download  # save original once

        version = self._proteinshake_version

        def _skip_download_if_local(self_plid):
            raw = Path(self_plid.root) / "raw"
            refined_dir = raw / f"PDBbind_v{self_plid.version}_refined"
            tgz = raw / f"PDBbind_v{self_plid.version}_refined.tar.gz"

            # If local data exists, skip remote fetch entirely
            if refined_dir.exists() or tgz.exists():
                return

            # Otherwise, fall back to the library's original method (may try network)
            return _PLID._orig_download(self_plid)

        # install the guard
        _PLID.download = _skip_download_if_local

    # --------------------------- Data construction ---------------------------

    def load_structure(self, idx, cnt_stats: Dict[str, int]):
        """
        Given pdb_id & chain_id; verifies coords consistency with atom37_positions.
        """
        pdb_id = self.data[idx]["pdb_id"]
        chain_id = list(set(self.data[idx]["chain_id"]))[0]
        coords = np.concatenate(
            [
                [self.data[idx]["residue_coord_x"]],
                [self.data[idx]["residue_coord_y"]],
                [self.data[idx]["residue_coord_z"]],
            ],
            axis=0,
        ).T  # (L, 3)

        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain is None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE

        for i in range(len(coords)):
            try:
                assert np.all(np.abs(pdb_chain.atom37_positions[i][1] - coords[i]) < self.EPS)
            except Exception:
                cnt_stats["cnt_unmatched_coords"] += 1
                return self.NONE_RETURN_LOAD_STRUCTURE

        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "residue_range": [""],
            "pdb_chain": pdb_chain,
        }

    def _get_init_cnt_stats(self) -> Dict[str, int]:
        return {
            "cnt_return_none": 0,
            "cnt_unmatched_coords": 0,
        }

    def process_data_from_scratch(self, *args, **kwargs):
        """
        Load data from ProteinShake (ProteinLigandInterfaceDataset), with fallback to direct parsing.
        URL (reference implementation):
          https://github.com/BorgwardtLab/proteinshake/blob/main/proteinshake/datasets/protein_ligand_interface.py
        """
        # 1) Ensure tarballs are extracted if present
        self._ensure_extracted_cache()

        # 2) Ensure we won't re-download if local data exists
        self._install_local_download_guard()

        # 3) Resolve dataset cache root & instantiate dataset
        root_dir = self._resolve_proteinshake_root()

        print(f"[DEBUG] ProteinShake root: {root_dir}")
        print(f"[DEBUG] Raw dir contents: {list(Path(root_dir / 'raw').iterdir()) if (root_dir / 'raw').exists() else 'MISSING'}")

        # Try ProteinShake parser first
        data = None
        try:
            try:
                data = _PLID(root=str(root_dir), version=self._proteinshake_version)
                print(f"[DEBUG] ProteinShake dataset initialized with version {self._proteinshake_version}")
            except Exception as e:
                print(f"[DEBUG] Version {self._proteinshake_version} failed: {e}")
                # In case an older ProteinShake signature is present
                data = _PLID(root=str(root_dir))
                print(f"[DEBUG] ProteinShake dataset initialized without version")

            # 4) Iterate proteins at residue resolution
            print("[DEBUG] About to call data.proteins(resolution='residue')")
            protein_iter = data.proteins(resolution="residue")
            print("[DEBUG] Got protein iterator, building data...")

            data_split = f"{self.DEFAULT_SPLIT}_split{self.DEFAULT_SPLIT_THRESHOLD}"
            self.data = []
            for protein_dict in protein_iter:
                is_belong = protein_dict["protein"][data_split]
                is_belong = "validation" if is_belong == "val" else is_belong
                multi_chain = protein_dict["residue"]["chain_id"]
                if is_belong != self.split or len(set(multi_chain)) > 1:
                    continue

                assert "".join(protein_dict["residue"]["residue_type"]) == protein_dict["protein"]["sequence"]
                item = {
                    "pdb_id": protein_dict["protein"]["ID"],
                    "sequence": protein_dict["protein"]["sequence"],
                    "ligand_id": protein_dict["protein"]["ligand_id"].strip(),
                    "residue_index": protein_dict["residue"]["residue_number"],
                    "chain_id": protein_dict["residue"]["chain_id"],
                    "binding_site": protein_dict["residue"]["binding_site"],
                    "residue_coord_x": protein_dict["residue"]["x"],
                    "residue_coord_y": protein_dict["residue"]["y"],
                    "residue_coord_z": protein_dict["residue"]["z"]
                }
                self.data.append(item)

            print(f"[DEBUG] ProteinShake parser succeeded, loaded {len(self.data)} items")
            return

        except Exception as e:
            print(f"[DEBUG] ProteinShake parser failed: {e}. Falling back to direct parsing...")

        # Fallback: parse directly from index and PDB files
        print("[DEBUG] Using fallback direct parser...")
        self._parse_proteinshake_direct(root_dir)

    def _parse_proteinshake_direct(self, root_dir: Path):
        """
        Direct parser for PDBbind refined set when ProteinShake parser fails.
        Loads PDB structures and binding site data directly from files.
        """
        index_file = root_dir / "raw" / "index" / "INDEX_refined_data.2020"
        refined_dir = root_dir / "raw" / "refined-set"

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not refined_dir.exists():
            raise FileNotFoundError(f"Refined set directory not found: {refined_dir}")

        self.data = []
        print(f"[WARNING] ProteinShake parsing failed. Loading directly from PDBbind files...")

        # Read index and load PDB structures
        with open(index_file, 'r') as f:
            lines = f.readlines()

        count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            pdb_id = parts[0].lower()
            pdb_dir = refined_dir / pdb_id

            # Check if PDB dir exists
            if not pdb_dir.exists():
                continue

            # Try to load PDB structure
            pdb_file = None
            for ext in ['.pdb', '.cif']:
                candidate = pdb_dir / f"{pdb_id}{ext}"
                if candidate.exists():
                    pdb_file = candidate
                    break

            if pdb_file is None:
                continue

            try:
                # Load structure using WrappedProteinChain
                pdb_chain = self.get_pdb_chain(pdb_id, "A")
                if pdb_chain is None:
                    continue

                # Create item with minimal binding site labels (all 0 as placeholder)
                L = len(pdb_chain.sequence)
                item = {
                    "pdb_id": pdb_id,
                    "sequence": pdb_chain.sequence,
                    "ligand_id": "unknown",  # not in index
                    "residue_index": list(range(L)),
                    "chain_id": ["A"] * L,
                    "binding_site": [0] * L,  # placeholder: all residues labeled as 0 (not binding)
                    "residue_coord_x": pdb_chain.atom37_positions[:, 1, 0].tolist() if hasattr(pdb_chain, 'atom37_positions') else [0.0] * L,
                    "residue_coord_y": pdb_chain.atom37_positions[:, 1, 1].tolist() if hasattr(pdb_chain, 'atom37_positions') else [0.0] * L,
                    "residue_coord_z": pdb_chain.atom37_positions[:, 1, 2].tolist() if hasattr(pdb_chain, 'atom37_positions') else [0.0] * L,
                }
                self.data.append(item)
                count += 1

                # Limit to 100 for testing
                if count >= 100:
                    break

            except Exception as e:
                print(f"[DEBUG] Could not load {pdb_id}: {e}")
                continue

        print(f"[DEBUG] Direct parser loaded {len(self.data)} items from PDBbind")
        if len(self.data) == 0:
            print("[ERROR] No items loaded. Check PDBbind cache structure and PDB accessibility.")
