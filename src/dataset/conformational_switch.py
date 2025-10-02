import os
from tqdm import tqdm
from collections import Counter
import time

import pandas as pd
import torch
import torch.distributed as dist
from src.baselines.wrapped_myrep import MissingRepresentation
from src.tokenizers import WrappedMyRepTokenizer
from sklearn.model_selection import train_test_split

from protein_chain import WrappedProteinChain
from tokenizer import *
import util
from dataset.base import BaseDataset


class ConformationalSwitchDataset(BaseDataset):

    NONE_RETURN_LOAD_STRUCTURE = {
        "prot1_pdb_id": None, 
        "prot1_chain_id": None,
        "prot1_residue_range": None,
        "prot1_pdb_chain": None,
        "prot2_pdb_id": None, 
        "prot2_chain_id": None,
        "prot2_residue_range": None,
        "prot2_pdb_chain": None,
    }

    FOLD_SWITCHING_FILE = "./conformational/codnas.csv"
    APO_HOLO_FILE = "./conformational/apo.csv"
    SPLIT_NAME = {
        "test": ["fold_switching_test", "apo_holo_test"]
    }


    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)

    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
    
    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"conformational/processed_structured_{self.target_field}_{self.split}")
    
    def collate_fn(self, batch):
        """passed to DataLoader as collate_fn argument"""
        batch = list(filter(lambda x: x is not None, batch))

        prot1_input_ids, prot2_input_ids, labels = tuple(zip(*batch))
        kwargs = {
            "constant_value": self.structure_pad_token_id, 
            "truncation_length": self.truncation_length
        }
        prot1_input_ids = util.pad_structures(prot1_input_ids, **kwargs)
        prot2_input_ids = util.pad_structures(prot2_input_ids, **kwargs)
        labels = torch.tensor(labels)  # type: ignore
        
        return {
            "input_list": (prot1_input_ids, prot2_input_ids),
            "targets": labels
        }

    def process_data_from_scratch(self, *args, **kwargs):
        
        file_name = getattr(self, f"{self.split[:-5].upper()}_FILE")
        df = pd.read_csv(os.path.join(self.data_path, file_name))
        # "other" for fold_switching, "apo" for apo_holo
        df.columns = df.columns.values.tolist()[:-1] + ["other"]

        self.data = []
        for i in range(len(df)):
            self.data.append({
                "prot1_pdb_id": df.iloc[i]["name"].split(".")[0],
                "prot1_chain_id": df.iloc[i]["name"].split(".")[1],
                "prot2_pdb_id": df.iloc[i]["other"].split(".")[0],
                "prot2_chain_id": df.iloc[i]["other"].split(".")[1],
                "seqlen": df.iloc[i]["seqlen"],
                "prot1_seq": df.iloc[i]["seq"],
                "prot2_seq": df.iloc[i]["seqres"]
            })
        self.py_logger.info(f"Done preprocessing.")
    
    def load_all_structures(self, ):
        """For each pdb_id in self.data[], load its pdb structures by
        calling self.load_structure()
        """
        process_global_rank = 0
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
        self.py_logger.info(f"Loading total {len(self.data)} structures on "
                            f"device {process_global_rank}")
        
        cnt_stats = self._get_init_cnt_stats()
        if self.fast_dev_run:
            self.data = self.data[:16]
        for i in tqdm(range(len(self.data))):
            res = self.load_structure(i, cnt_stats)
            
            for k in res.keys():
                self.data[i][k] = res[k]

            # special case: tm_align too slow, not sure why
            if self.split == "fold_switching_test" and i == 2:
                self.data[i]["prot1_pdb_id"] = None
                continue
            tmp = util.calculate_tm_rmsd_score(self.data[i]["prot1_pdb_chain"], self.data[i]["prot2_pdb_chain"])
            self.data[i]["tm_score"] = (tmp[0] + tmp[1]) / 2
            self.data[i]["negrmsd_score"] = -tmp[2]
            
        self.py_logger.info(f"Processing all structures results in count "
                            f"statistics: {cnt_stats}")
        
        bg_time = time.time()
        new_data = []
        for i in range(len(self.data)):
            if not self.data[i]["prot1_pdb_id"] is None:
                new_data.append(self.data[i])
        ed_time = time.time()
        print("Timing: ", (ed_time - bg_time))

        self.py_logger.info(f"After filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data

    def load_structure(self, index, cnt_stats):
        """Given pdb_id, chain_id
        """

        ret = {}
        for idx in [1, 2]:
            pdb_id = self.data[index][f"prot{idx}_pdb_id"]
            chain_id = self.data[index][f"prot{idx}_chain_id"]
            residue_range = [""]
            pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
            if pdb_chain == None:
                cnt_stats["cnt_return_none"] += 1
                return self.NONE_RETURN_LOAD_STRUCTURE
            ret[f"prot{idx}_pdb_id"] = pdb_id
            ret[f"prot{idx}_chain_id"] = chain_id
            ret[f"prot{idx}_residue_range"] = residue_range
            ret[f"prot{idx}_pdb_chain"] = pdb_chain
        
        return ret

    def sanity_check(self):
        """Currently necessary for TAPE Remote Homology, which have `residue_range`
        specified. Note that `residue_range` is used to extract useful residues for 
        global property prediction; while for local property prediction, the 
        `residue_range` is always [""] and `target_field` labels are annotated with binaries
        """
        new_data = []
        for item in self.data:
            flag = True
            for idx in [1, 2]:
                pdb_chain, residue_range = item[f"prot{idx}_pdb_chain"], item[f"prot{idx}_residue_range"]
                assert residue_range == [""]
                selected_indices = self._get_selected_indices(pdb_chain.residue_index, residue_range)
                
                if len(selected_indices) == 0:
                    flag = False
                # filter proteins that are too long
                if len(selected_indices) > self.filter_length:
                    flag = False
                
            if flag:
                new_data.append(item)
        self.data = new_data

        self.py_logger.info(f"After sanity check for selected residues, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
    
    
    def _get_item_structural_tokens(self, index):
        item = self.data[index]
        if "prot1_token_ids" in item:
            return item["prot1_token_ids"], item["prot2_token_ids"], item[self.target_field]

        token_ids_list = []
        residue_index_list = []
        for idx in [1, 2]:
            pdb_chain, residue_range = item[f"prot{idx}_pdb_chain"], item[f"prot{idx}_residue_range"]
            pdb_id, chain_id = item[f"prot{idx}_pdb_id"], item[f"prot{idx}_chain_id"]
            pdb_path = self.retrieve_pdb_path(pdb_id, chain_id)
            if isinstance(self.tokenizer, WrappedESM3Tokenizer):
                token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence)
            elif isinstance(self.tokenizer, (WrappedFoldSeekTokenizer, 
                                             WrappedProTokensTokenizer, WrappedAIDOTokenizer)):
                token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
            elif (isinstance(self.tokenizer, WrappedProteinMPNNTokenizer)
                or isinstance(self.tokenizer, WrappedMIFTokenizer)
                or isinstance(self.tokenizer, WrappedCheapS1D64Tokenizer)):
                reprs, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
                assert len(reprs) == len(residue_index)
                token_ids = reprs.detach() # code compatable if directly using as token_ids
            # elif isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
            #     token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
            #     assert len(token_ids) == len(residue_index)
            elif isinstance(self.tokenizer, WrappedMyRepTokenizer):
                # Try current index; if the wrapper signals a missing rep, advance to the next item.
                N = len(self.data)
                start_i = index
                i = index
                while True:
                    try:
                        # Recompute fields for the current i (since i may change when we skip)
                        item_i = self.data[i]
                        pdb_chain_i, residue_range_i = item_i["pdb_chain"], item_i["residue_range"]
                        pdb_id_i, chain_id_i = item_i["pdb_id"], item_i["chain_id"]
                        pdb_path_i = self.retrieve_pdb_path(pdb_id_i, chain_id_i)

                        # For non-Atlas datasets, normalize chain_id if needed
                        if self.data_name == "AtlasDataset":
                            chain_for_tok = " "
                        else:
                            chain_for_tok, _ = convert_chain_id(pdb_path_i, chain_id_i)

                        assigned_labels_i = item_i[self.target_field]

                        # Ask tokenizer for embeddings (may raise MissingRepresentation)
                        token_ids, residue_index, seqs = self.tokenizer.encode_structure(
                            pdb_path_i, chain_for_tok, self.use_sequence
                        )
                        # Success: use this sample and propagate the updated locals below
                        item = item_i
                        pdb_chain, residue_range = pdb_chain_i, residue_range_i
                        assigned_labels = assigned_labels_i
                        break

                    except MissingRepresentation:
                        i = (i + 1) % N
                        if i == start_i:
                            raise RuntimeError("All items are missing representations; nothing to iterate.")
                        # continue loop and try next item
            # select according to residue range constraints
            assert residue_range == [""]
            # filter proteins that are too long
            if len(token_ids) > self.filter_length:
                return None
            token_ids_list.append(token_ids)
            residue_index_list.append(residue_index)
        
        self.data[index]["prot1_token_ids"] = token_ids_list[0].to("cpu").detach().clone()
        self.data[index]["prot2_token_ids"] = token_ids_list[1].to("cpu").detach().clone()
        self.data[index]["prot1_residue_index"] = residue_index_list[0]
        self.data[index]["prot2_residue_index"] = residue_index_list[1]
        return token_ids_list[0], token_ids_list[1], item[self.target_field]
