import os
from tqdm import tqdm
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist

from sklearn.model_selection import train_test_split
from proteinshake.tasks import  BindingSiteDetectionTask

from dataset.base import BaseDataset
from proteinshake.datasets import ProteinLigandInterfaceDataset


class ProteinShakeBindingSiteDataset(BaseDataset):

    DEFAULT_SPLIT = "structure" # could be "random, sequence, structure"
    DEFAULT_SPLIT_THRESHOLD = "_0.7" 
    # could be "" for random, ``0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`` for sequence
    # ``0.5, 0.6, 0.7, 0.8, 0.9`` for structure

    SPLIT_NAME = {
        "test": ["test"]
    }

    EPS = 1e-3

    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)
    
    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
    
    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"proteinshake_bindingsite/processed_structured_binding_site_{self.split}")
    
    def load_structure(self, idx, cnt_stats):
        """Given pdb_id, chain_id
        """

        pdb_id, chain_id, residue_range, pdb = None, None, [""], None

        pdb_id = self.data[idx]["pdb_id"]
        chain_id = list(set(self.data[idx]["chain_id"]))[0]
        coords = np.concatenate(
                    [[self.data[idx]["residue_coord_x"]], 
                    [self.data[idx]["residue_coord_y"]],
                    [self.data[idx]["residue_coord_z"]]], axis=0
                ).T # (L, 3)
        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain == None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE

        for i in range(len(coords)):
            try:
                assert np.all(np.abs(pdb_chain.atom37_positions[i][1] - coords[i]) < self.EPS)
            except:
                cnt_stats["cnt_unmatched_coords"] += 1
                return self.NONE_RETURN_LOAD_STRUCTURE
    
        return {
            "pdb_id": pdb_id, 
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": pdb_chain,
        }
    
    def _get_init_cnt_stats(self,):
        cnt_stats = {
            "cnt_return_none": 0,
            "cnt_unmatched_coords": 0,
        }
        return cnt_stats

    def process_data_from_scratch(self, *args, **kwargs):
        """Load data from ProteinShake
        URL: https://github.com/BorgwardtLab/proteinshake/blob/main/proteinshake/datasets/protein_ligand_interface.py
        """
        
        data = ProteinLigandInterfaceDataset()
        protein = data.proteins(resolution='residue')

        data_split = f"{self.DEFAULT_SPLIT}_split{self.DEFAULT_SPLIT_THRESHOLD}"
        self.data = []
        while True:
            try:
                protein_dict = next(protein)
            except:
                break
            
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
