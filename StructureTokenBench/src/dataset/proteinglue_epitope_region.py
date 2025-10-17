import os
from tqdm import tqdm
from collections import Counter

import pandas as pd
import torch
import torch.distributed as dist

from sklearn.model_selection import train_test_split

from protein_chain import WrappedProteinChain
from dataset.base import BaseDataset
from dataset.cath import CATHLabelMappingDataset

from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.sequence import ProteinSequence


class ProteinGLUEEpitopeRegionDataset(BaseDataset):

    HOMODIMER_PROTEIN_FILE = "proteinglue_epitoperegion/Dset48.csv"
    HETEROMERIC_PROTEIN_FILE = "proteinglue_epitoperegion/HM_479_testing.csv"

    FULL_FIELD_MAPPING = {
        "epitope_label": "Interface",
    }

    SPLIT_NAME = {
        "test": ["fold_test", "superfamily_test"]
    }
    SAVE_SPLIT = ["train", "validation", "fold_test", "superfamily_test"]

    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)
    
    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
        
    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"proteinglue_epitoperegion/processed_structured_epitope_region_{self.split}")
    
    def extract_useful_features(self, ):

        data_target_field = self.FULL_FIELD_MAPPING[self.target_field]

        # process 34 homodimer proteins and 95 heteromeric proteins, no overlapped PDBs
        self.data = []
        for file in [self.HOMODIMER_PROTEIN_FILE, self.HETEROMERIC_PROTEIN_FILE]:
            df_data = pd.read_csv(os.path.join(self.data_path, file))

            pdb_id_list = list(set(df_data["name"].values.tolist()))
            for key in pdb_id_list:
                tmp_data = df_data[df_data["name"] == key]
                tmp_data = tmp_data[["Pos", "name", "length", "AliSeq"] 
                                    + [v for k,v in self.FULL_FIELD_MAPPING.items()]]
                
                if key.startswith("H_"):
                    pdb_id, chain_id = key[2:], None
                else:
                    pdb_id, chain_id = key.split("_")
                    # filter multi-chain proteins
                    if len(chain_id) > 1:
                        continue
                self.data.append({
                    "pdb_id": pdb_id.lower(), # ensure that pdb_id is lower-cased
                    "chain_id": chain_id,
                    "annot_seq": "".join(tmp_data["AliSeq"].values),
                    "annot_label": tmp_data[data_target_field].values.tolist()
                })
    
    def associate_with_CATH_labels(self, ):
        """Associate with CATH labels
        """

        cath_data_path = os.path.join(
            self.data_path[:self.data_path.rfind("/data/")],
            "./data/CATH"
        )
        self.cath_database = CATHLabelMappingDataset(data_path=cath_data_path)
        
        for i in range(len(self.data)):
            pdb_id = self.data[i]["pdb_id"]
            chain_id = str(self.data[i]["chain_id"])
            ref_seq = self.data[i]["annot_seq"]
            
            res = self.cath_database.retrieve_labels(pdb_id, chain_id, ref_seq)
            # None: either cannot find PDB and its chain, 
            # or fail to do multi-sequence alignment
            if res is None:
                self.data[i] = None
            else:
                self.data[i]["fold_label"], self.data[i]["superfamily_label"], chain_id = res
                # this dataset sometimes does not specify the chain ID
                # but we can update it once we find the matched CATH label
                if self.data[i]["chain_id"] is None:
                    self.data[i]["chain_id"] = chain_id

        new_data = [x for x in self.data if x is not None]
        self.py_logger.info(f"After filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def process_data_from_scratch(self, *args, **kwargs):
        """Load data from ProteinShake

        Original data columns:
            'Pos': the order of residues in the sequence
            'name': PDB ID
            'Interface': '0', non-interface positions; '1' interface residues
            'Interface1': 'NI', non-interface positions; 'I' interface residues
            'length': length of the protein sequence
            'AliPos': the order of residues in the alignment
            'AliSeq': amino acid of the protein
        """

        assert dist.get_world_size() == 1, "dataset not preprocessed and splitted, please not to use multi-GPU training"

        # extract fields
        self.extract_useful_features()
        # assign structural classification to proteins
        self.associate_with_CATH_labels()

        # no missing labels

        # split
        res = self.splitting_dataset(fold_split_ratio=0.5, fold_valid_ratio=0.2, 
                    superfamily_split_ratio=0.4, superfamily_valid_ratio=0.4,)

        # save to disk
        for i, split in enumerate(self.SAVE_SPLIT):
            target_split_file = os.path.join(self.data_path, f"proteinglue_epitoperegion/{self.target_field}_{split}")
            torch.save(res[i], target_split_file)
            self.py_logger.info(f"Saving to {target_split_file}")
            if split == self.split:
                self.data = res[i]

        self.py_logger.info(f"Done preprocessing, splitting and saving.")

    
    def _get_init_cnt_stats(self,):
        cnt_stats = {
            "cnt_return_none": 0,
            "cnt_inconsistent_alignment": 0,
        }
        return cnt_stats
        
    def load_structure(self, idx, cnt_stats):
        """Given pdb_id, chain_id
        """

        pdb_id, chain_id, residue_range, pdb = None, None, [""], None
        pdb_id = self.data[idx]["pdb_id"]
        chain_id = self.data[idx]["chain_id"]
        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain == None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE
        
        # get local labels for each residue
        
        # do not have residue local mapping information, used alignment to align annotated residues to PDB chains
        seq1 = ProteinSequence(self.data[idx]["annot_seq"])
        seq2 = ProteinSequence(pdb_chain.sequence)
        matrix = SubstitutionMatrix.std_protein_matrix()
        ali = align_optimal(seq1, seq2, matrix)[0]

        local_label = [-1] * len(pdb_chain)
        for idx1, idx2 in ali.trace:
            if idx2 == -1:
                continue
            if idx1 != -1:
                local_label[idx2] = self.data[idx]["annot_label"][idx1]
        
        # not every residue in pdb_chain can have an annotation
        cnt_stats["cnt_inconsistent_alignment"] += sum([1 for _ in local_label if _ == -1])
        local_label = [x if x != -1 else 0 for x in local_label]

        return {
            "pdb_id": pdb_id, 
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": pdb_chain, 
            self.target_field: local_label
        }

