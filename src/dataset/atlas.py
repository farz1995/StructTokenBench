import os
from tqdm import tqdm

import pandas as pd
import torch
import torch.distributed as dist

from protein_chain import WrappedProteinChain
from dataset.base import BaseDataset
from dataset.cath import CATHLabelMappingDataset


class AtlasDataset(BaseDataset):
    
    """
    Dataset class for ATLAS residue-level properties (https://www.dsimb.inserm.fr/ATLAS/about.html)
    """

    ALLOWED_TARGET_FIELDS = set(["Bfactor", "RMSF", "Neq"])

    FULL_FIELD_MAPPING = {
        "bfactor_score": "Bfactor",
        "rmsf_score": "RMSF",
        "neq_score": "Neq",
    }

    SPLIT_NAME = {
        "test": ["fold_test", "superfamily_test"]
    }
    SAVE_SPLIT = ["train", "validation", "fold_test", "superfamily_test"]

    NORMALIZING_FACTOR = {
        "bfactor_score": 0.01,
        "rmsf_score": 1.0,
        "neq_score": 0.1,
    }
    
    def __init__(self, *args, **kwargs):
        assert kwargs.get("target_filed", set()).issubset(self.ALLOWED_TARGET_FIELDS)
        super().__init__(self, *args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer")
    
    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
    
    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"atlas/processed_structured_{self.split}")
    
    def extract_useful_features(self, ):
        # 1. read pdb list
        self.data = []
        pdb_list_file = os.path.join(self.data_path, 'atlas/2022_06_13_ATLAS_pdb.txt')
        with open(pdb_list_file, 'r') as f:
            for line in f:
                self.data.append({'pdb_id_chain_id': line.strip()})
        
        # 2. for each pdb, load per-residue labels from csv file
        for i in range(len(self.data)):
            item = self.data[i]
            pdb_id_chain_id = item["pdb_id_chain_id"]
            label_file_path = os.path.join(self.data_path, "atlas/analysis", 
                        pdb_id_chain_id, f"{pdb_id_chain_id}_per_residue_labels.csv")
            self.data[i]["annot_df"] = pd.read_csv(label_file_path)

    def process_data_from_scratch(self, *args, **kwargs):
        assert dist.get_world_size() == 1, "dataset not preprocessed and splitted, please not to use multi-GPU training"
        
        self.extract_useful_features()
        self.associate_with_CATH_labels()

        res = self.splitting_dataset()

        # save to disk
        for i, split in enumerate(self.SAVE_SPLIT):
            target_split_file = os.path.join(self.data_path, f"atlas/{split}")
            torch.save(res[i], target_split_file)
            if split == self.split:
                self.data = res[i]

        self.py_logger.info(f"Done preprocessing, splitting and saving.")

    def associate_with_CATH_labels(self, ):

        # associate with CATH labels
        cath_data_path = os.path.join(
            self.data_path[:self.data_path.rfind("/data/")],
            "./data/CATH"
        )
        self.cath_database = CATHLabelMappingDataset(data_path=cath_data_path)
        
        for i in tqdm(range(len(self.data))):
            pdb_id, chain_id = self.data[i]["pdb_id_chain_id"].split("_")
            ref_seq = "".join(self.data[i]["annot_df"]["seq"].values)
            res = self.cath_database.retrieve_labels(pdb_id, chain_id, ref_seq)
            # None: either cannot find PDB and its chain, 
            # or fail to do multi-sequence alignment
            if res is None:
                self.data[i] = None
            else:
                self.data[i]["fold_label"], self.data[i]["superfamily_label"], _ = res

        new_data = [x for x in self.data if x is not None]
        self.py_logger.info(f"After filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def retrieve_pdb_path(self, pdb_id, chain_id):
        # from PDB if called get_pdb_chain()
        # specifically defined if ATLAS
        pdb_id_chain_id = f"{pdb_id}_{chain_id}"
        pdb_path = os.path.join(self.data_path, "atlas/analysis", pdb_id_chain_id, f"{pdb_id_chain_id}.pdb")
        return pdb_path

    def load_structure(self, idx, cnt_stats):
        """
        `pdb_id_chain_id` contains both the pdb and chain id, like `2z6r_A`
        """
        pdb_id_chain_id = self.data[idx]["pdb_id_chain_id"]
        pdb_id, chain_id = pdb_id_chain_id.split("_")

        pdb_file_path = self.retrieve_pdb_path(pdb_id, chain_id)
        
        protein_chain = WrappedProteinChain.from_pdb(pdb_file_path) # NOTE: used ATLAS provided PDBs
        
        # `residue_range` default as [""] to indicate the whole protein
        # ATLAS data always annotate the whole protein from their processed pdb file
        residue_range = [""]
        assert len(self.data[idx]["annot_df"]) == len(protein_chain.sequence)
        assert "".join(self.data[idx]["annot_df"]["seq"].values) == protein_chain.sequence
        
        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": protein_chain,
        }
        
    def _get_item_structural_tokens(self, index):
        """Overwrite this method from BaseDataset to handle labels."""

        item = self.data[index]
        if "token_ids" in item and self.target_field in item:
            if self.is_global_or_local == "local":
                assert len(item["token_ids"]) == len(item[self.target_field])
            
            return item["token_ids"], item[self.target_field], item["real_seqs"]

        # put here because "get_target_file_name" is independent of self.target_field
        # assign labels to items before tokenizing
        data_target_field = self.FULL_FIELD_MAPPING[self.target_field]
        local_labels = self.data[index]["annot_df"][data_target_field].values # type: np.array
        local_labels = torch.from_numpy(local_labels) * self.NORMALIZING_FACTOR[self.target_field] # [n_residues, len(self.target_field)]
        self.data[index][self.target_field] = local_labels
        # it's already ensured that "annot_df" align with pdb_chain in "load_structure" function

        token_ids, assigned_labels, seqs = BaseDataset._get_item_structural_tokens(self, index, skip_check=True)
        assert len(token_ids) == len(assigned_labels)
        return token_ids, assigned_labels, seqs

