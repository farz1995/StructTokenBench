import os
import time
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from Bio import PDB
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from protein_chain import WrappedProteinChain
from tokenizer import *
import util

def convert_chain_id(pdb_path, chain_id):

    if pdb_path.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
    
    structure = parser.get_structure("check", pdb_path)
    if chain_id in structure[0]:
        return chain_id, False

    atom_array = convert.get_structure(CIFFile.read(pdb_path), model=1, 
                        extra_fields=["b_factor"])
    new_atom_array = convert.get_structure(CIFFile.read(pdb_path), model=1, 
                        extra_fields=["b_factor"], use_author_fields=False)
    chain_id_mapping = [(x,y) for x,y in zip(atom_array.chain_id, new_atom_array.chain_id) if y == chain_id]
    
    assert len(set([x[0] for x in chain_id_mapping])) == 1
    
    new_chain_id = chain_id_mapping[0][0]
    return new_chain_id, True

class BaseDataset(Dataset):

    NONE_RETURN_LOAD_STRUCTURE = {
        "pdb_id": None, 
        "chain_id": None,
        "residue_range": None,
        "pdb_chain": None,
    }

    def __init__(self, *args, **kwargs):
        """
        in kwargs:
            data_path: data storage directory prefix
            target_field: target label name
            split: "train", "valid", or "test"
            py_logger: python logger
            tokenizer: sequence tokenizer or structural tokenzier
            in_memory: False
        """
        self.data_path = kwargs["data_path"]
        self.target_field = kwargs["target_field"]
        self.truncation_length = kwargs["truncation_length"]
        self.filter_length = kwargs["filter_length"]
        self.split = kwargs["split"]
        self.py_logger = kwargs["py_logger"]
        self.structure_pad_token_id = kwargs["tokenizer"].pad_token_id
        self.multi_label = kwargs["multi_label"]
        self.is_global_or_local = kwargs["is_global_or_local"]
        self.PDB_DATA_DIR = kwargs["pdb_data_dir"]
        self.fast_dev_run = kwargs.get("fast_dev_run", False)
        self.data_name = kwargs["data_name"]

        self.use_continuous = kwargs["use_continuous"]
        # `use_sequence`` for BaseDataset is always set to True to pass sequence
        # information to models, while `use_sequence` for the model itself is 
        # False by default to disable using sequence during tokenization
        self.use_sequence = True

        # try to load pre-processed data
        target_split_file = self.get_target_file_name()
        
        if os.path.exists(target_split_file):
            self.data = torch.load(target_split_file, weights_only=False)
            self.py_logger.info(f"Loading from processed file {target_split_file},"
                                f"structured data of {len(self.data)} entries.")
        else:
            self.py_logger.info(f"Cannot load from processed file {target_split_file} "
                                f"for structured data")
            if dist.is_initialized():
                assert dist.get_world_size() == 1
            # process data entries from raw data, different for every datasets
            self.process_data_from_scratch(*args, **kwargs)

            # preprocess index mappings before loading PDB structures, different for every datasets
            self.prepare_structure_loading()
                
            self.load_all_structures()

            self.sanity_check()
            # save to disk
            self.save_structured_data()
            
        # Dataset sharding will be done in LightningDataModule

        # assign tokenizer if haven't been assign in `process_data_from_scratch`
        if not hasattr(self, "tokenizer"):
            self.tokenizer = kwargs["tokenizer"]

        self.patch_due_to_protokens()

        self.patch_for_TAPE_homo()

    def patch_due_to_protokens(self,):
        """filter because ProTokens cannot proceed proteins longer than 1024
        """
        len_limit = 1024
        new_data = []
        if self.data_name == "ConformationalSwitchDataset":
            for i in range(len(self.data)):
                if (len(self.data[i]["prot1_pdb_chain"].sequence) <= len_limit 
                    and len(self.data[i]["prot2_pdb_chain"].sequence) <= len_limit):
                    new_data.append(self.data[i])
        else:
            for i in range(len(self.data)):
                if len(self.data[i]["pdb_chain"].sequence) <= len_limit:
                    new_data.append(self.data[i])
            
        if len(new_data) != len(self.data):
            self.data = new_data
            self.py_logger.info(f"reduce sequence lengths because of ProTokens from {len(self.data)} to {len(new_data)}")

    def patch_for_TAPE_homo(self,):
        """
        Filter proteins causing error in TAPE RH, which are indexed at 11220 (out of 12071) and 11958 (out of 12070)
        Error Example: 
            Bio.PDB.PDBExceptions.PDBConstructionException: Blank altlocs in duplicate residue SER (' ', 22, ' ') of chain 'A'
        Error Explanation: https://biopython.org/wiki/Reading_large_PDB_files
        """
        if self.data_name == "TapeRemoteHomologyDataset" and self.split == "train":
            skip_index = 11220
            self.data = self.data[:skip_index] + self.data[skip_index + 1:]
            skip_index = 11958
            self.data = self.data[:skip_index] + self.data[skip_index + 1:]
    
            self.py_logger.info(f"reduce sequence lengths for TAPE Homo to {len(self.data)}")
    
    def get_target_file_name(self,):
        assert NotImplementedError

    def save_structured_data(self, ):
        file = self.get_target_file_name()
        torch.save(self.data, file)
        self.py_logger.info(f"Save the processed, structured data to disk: {file}")
    
    def prepare_structure_loading(self):
        assert NotImplementedError

    def collate_fn(self, batch):
        """
        Robust collation for variable-length *continuous* features.

        Accepts:
          - {"token_ids": FloatTensor(L,D), "label": int, "residue_index": (L,) optional}
          - ({"token_ids": ...}, label)
          - (FloatTensor(L,D), label) or (ndarray(L,D), label)

        Returns (for model_module.py):
          - batch["input_list"]: a list with one dict containing padded tensors
          - batch["targets"]:    LongTensor (B,)
        Also returns friendly extras: token_ids, attention_mask, residue_index, labels, lengths
        """
        # remove Nones
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        # normalize each item to a dict with token_ids (Tensor), label (int), residue_index (optional Tensor)
        normed = []
        for s in batch:
            # tuple/list -> (data, label)
            if isinstance(s, (tuple, list)) and len(s) >= 2:
                data, label = s[0], s[1]
                if isinstance(data, dict):
                    item = dict(data)
                else:
                    item = {"token_ids": data}
                item["label"] = int(label)
            elif isinstance(s, dict):
                item = dict(s)
                if "label" not in item and "labels" in item:
                    item["label"] = int(item.pop("labels"))
            else:
                continue

            # map common alt keys to token_ids
            if "token_ids" not in item or item["token_ids"] is None:
                for k in ("features", "feats", "reps", "embeds", "inputs", "input_embeds"):
                    if k in item:
                        item["token_ids"] = item[k]
                        break
            if "token_ids" not in item or item["token_ids"] is None:
                continue  # skip unusable sample

            x = item["token_ids"]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32)
            else:
                x = x.to(torch.float32)
            item["token_ids"] = x

            if "label" not in item:
                item["label"] = 0  # default (should be overwritten upstream)

            ri = item.get("residue_index", None)
            if ri is not None and not torch.is_tensor(ri):
                ri = torch.as_tensor(ri, dtype=torch.int32)
                item["residue_index"] = ri

            normed.append(item)

        if len(normed) == 0:
            return None

        # pad
        lengths = [int(x["token_ids"].shape[0]) for x in normed]
        Lmax = max(lengths)
        D = int(normed[0]["token_ids"].shape[1])
        B = len(normed)

        feats = torch.zeros((B, Lmax, D), dtype=torch.float32)
        # attention mask semantics: True = padding, False = real token
        attn  = torch.ones((B, Lmax), dtype=torch.bool)
        resid = torch.zeros((B, Lmax), dtype=torch.int32)
        labels = torch.zeros((B,), dtype=torch.long)

        for i, it in enumerate(normed):
            x = it["token_ids"]; L = x.shape[0]
            feats[i, :L] = x
            attn[i, :L] = False  # mark real tokens as not-masked
            if "residue_index" in it and it["residue_index"] is not None:
                ri = it["residue_index"]
                if not torch.is_tensor(ri):
                    ri = torch.as_tensor(ri, dtype=torch.int32)
                resid[i, :L] = ri[:L]
            else:
                resid[i, :L] = torch.arange(L, dtype=torch.int32)
            labels[i] = int(it["label"])

        lengths = torch.as_tensor(lengths, dtype=torch.int32)

        #  Keys expected by your model:
        input_list = [{
            "token_ids": feats,            # (B, Lmax, D)
            "attention_mask": attn,        # (B, Lmax) bool, True=pad, False=token
            "residue_index": resid,        # (B, Lmax) int32
        }]
        out = {
            "input_list": input_list,
            "targets": labels,              # (B,)
            # extras (keep, can help debugging)
            "token_ids": feats,
            "attention_mask": attn,
            "residue_index": resid,
            "labels": labels,
            "lengths": lengths,
        }
        return out
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_pdb_chain(self, pdb_id, chain_id):
        try:
            file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
            protein_chain = WrappedProteinChain.from_cif(file, 
                                                chain_id=chain_id, id=pdb_id)
        except:
            self.py_logger.info(f"Cannot retrieve from local cluster, pdb_id: {pdb_id}, chain_id: {chain_id}")
            return None
        return protein_chain
    
    def _get_init_cnt_stats(self):
        return {}
    
    def load_structure(self, idx, cnt_stats):
        """
        Arguments:
            idx: index for self.data list
            cnt_stats: a dict to calculate statistics for unsable data entries
        Return:
            {
                "pdb_id": pdb_id, 
                "chain_id": chain_id,
                "residue_range": residue_range,
                "pdb_chain": pdb_chain, 
                "local_label": local_label # optional
            }
            # residue_range default as [""] to indicate the whole protein; 
            # e.g., ["6-100"] to indicate PDB residue_index ranging from 6 to 100
        """
        assert NotImplementedError
        
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
            assert "pdb_id" in res
            assert "chain_id" in res
            assert "residue_range" in res
            assert "pdb_chain" in res

        self.py_logger.info(f"Processing all structures results in count "
                            f"statistics: {cnt_stats}")
        
        bg_time = time.time()
        new_data = []
        for i in range(len(self.data)):
            if not self.data[i]["pdb_id"] is None:
                new_data.append(self.data[i])
        ed_time = time.time()
        print("Timing: ", (ed_time - bg_time))

        self.py_logger.info(f"After loading structure filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def sanity_check(self):
        """Filter according to length
        """

        new_data = []
        for item in self.data:
            pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
            selected_indices = self._get_selected_indices(pdb_chain.residue_index, residue_range)
            if len(selected_indices) == 0:
                continue
            # filter proteins that are too long
            if len(selected_indices) > self.filter_length:
                continue
            new_data.append(item)
        self.data = new_data

        self.py_logger.info(f"After sanity check for selected residues, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")

    def _get_selected_indices(self, residue_index, residue_range):
        """
        Arguments:
            residue_range: residue range with format like ["5-10", "20-300"] (default [""])
        """
        rr = residue_range
        if len(rr) == 1 and rr[0] == "":
            return np.arange(len(residue_index))
        
        left = [eval(sep.split("-")[0]) for sep in rr]
        right = [eval(sep.split("-")[1]) for sep in rr]
        rr_indices = [x for l, r in zip(left, right) for x in list(range(l, r+1))]

        selected_indices = []
        for i, ridx in enumerate(residue_index):
            if ridx in rr_indices:
                selected_indices.append(i)

        return selected_indices # a list
    
    def retrieve_pdb_path(self, pdb_id, chain_id):
        # specifically defined for ATLAS, PretrainPDB, CASP14 and CAMEO
        file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
        return file
    
    def _get_item_structural_tokens(self, index, skip_check=False):
        
        item = self.data[index]
        if not skip_check:
            if "token_ids" in item:
                if self.is_global_or_local == "local":
                    assert len(item["token_ids"]) == len(item[self.target_field])
                return item["token_ids"], item[self.target_field], item["real_seqs"]
    
        pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
        pdb_id, chain_id = item["pdb_id"], item["chain_id"]
        pdb_path = self.retrieve_pdb_path(pdb_id, chain_id)
        
        if self.data_name == "AtlasDataset":
            chain_id = " "
        else:
            # convert chain_id if necessary because some chain_id needs to 
            # use use_author_field (specified in biotite).
            # except atlas, other datasets' pdb_path is independent of chain_id; 
            # and for atlas, there is no need to transform chain_id
            chain_id, is_changed = convert_chain_id(pdb_path, chain_id)
        assigned_labels = item[self.target_field]
        assert pdb_chain is not None
        
        if self.is_global_or_local == "local":
            assert len(residue_range) == 1 and residue_range[0] == ""
        
            if self.data_name in "ProteinShakeBindingSiteDataset":
                label_residue_index = item["residue_index"]
            elif self.data_name in ["BioLIP2FunctionDataset", 
                "InterProFunctionDataset", "ProteinGLUEEpitopeRegionDataset", 
                "AtlasDataset"]:
                # all local labels already aligned to pdb_chain.residue_index
                label_residue_index = pdb_chain.residue_index
            else:
                raise NotImplementedError
            
            assert len(assigned_labels) == len(label_residue_index)


        # encode protein structure into token_ids
        if isinstance(self.tokenizer, WrappedESM3Tokenizer):
            # chain_id conversion is already automatically dealt with 
            # WrappedProteinChain, and produced pdb_chain
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedFoldSeekTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedProTokensTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedProteinMPNNTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedMIFTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedAIDOTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedCheapS1D64Tokenizer):
            # CheapS1D64 is continuous tokenizer
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        else:
            raise NotImplementedError
        
        assert len(token_ids) == len(residue_index)
        # code compatability in case token_ids store continuous reprs
        token_ids = token_ids.detach()
        assert len(residue_index) == len(seqs)
        
        if self.is_global_or_local == "local":
            # align residue_index and label_residue_index, so that token_ids align with assigned_labels
            org_len = len(token_ids)
            align_indices_1 = [i for i, x in enumerate(label_residue_index) if x in residue_index]
            label_residue_index = np.array(label_residue_index)[align_indices_1].tolist()
            assigned_labels = np.array(assigned_labels)[align_indices_1].tolist()

            align_indices_2 = [i for i, x in enumerate(residue_index) if x in label_residue_index]
            residue_index, token_ids = residue_index[align_indices_2], token_ids[align_indices_2]
            seqs = [x for i,x in enumerate(seqs) if i in set(align_indices_2)]

            try:
                assert (residue_index == np.array(label_residue_index)).all()
            except:
                # deal with repeated residue indices and achieve exact match with alignment
                idx_list = list(set(residue_index.tolist() + label_residue_index))
                
                alphabet = Alphabet(idx_list)
                sim_score = np.diag(np.ones(len(idx_list)))
                substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)
                seq1 = GeneralSequence(alphabet, label_residue_index)
                seq2 = GeneralSequence(alphabet, residue_index.tolist())
                alignment = align_optimal(seq1, seq2, substitution_matrix)
                
                alignment = alignment[0].trace
                align_indices_1, align_indices_2 = [], []
                for i in range(len(alignment)):
                    if (alignment[i] != -1).all():
                        align_indices_1.append(alignment[i][0])
                        align_indices_2.append(alignment[i][1])

                label_residue_index = np.array(label_residue_index)[align_indices_1].tolist()
                assigned_labels = np.array(assigned_labels)[align_indices_1].tolist()
                residue_index, token_ids = residue_index[align_indices_2], token_ids[align_indices_2]
                seqs = [x for i,x in enumerate(seqs) if i in set(align_indices_2)]


            if org_len - len(token_ids) != 0:
                print(">> residue reduced by : ", org_len - len(token_ids))

        # select according to residue range constraints for some global tasks
        selected_indices = self._get_selected_indices(residue_index, residue_range)
        assert len(selected_indices) != 0
        
        token_ids = token_ids[selected_indices]
        seqs = np.array(seqs)[selected_indices].tolist()
        if self.is_global_or_local == "local":
            assigned_labels = np.array(assigned_labels)[selected_indices].tolist()

        # cache the tokens
        self.data[index]["token_ids"] = token_ids.to("cpu").detach().clone()
        self.data[index][self.target_field] = assigned_labels
        self.data[index]["real_seqs"] = seqs
        if self.is_global_or_local == "local":
            assert len(token_ids) == len(assigned_labels)
        return token_ids, assigned_labels, seqs # torch.Tensor, List

    def __getitem__(self, index: int):
        return self._get_item_structural_tokens(index)

    def additional_label_filtering_for_TAPE_homo(self, tokenizer_name):

        if self.data_name == "TapeRemoteHomologyDataset":
            """
            The original TAPE dataset consists of 1195 labels.
            Filter label class that has less than 50 protein samples in the 
            training dataset, reducing from 1195 labels to 45 labels
            """

            labels_to_filter = set([
                22, 36, 47, 51, 73, 77, 78, 84, 88, 90, 126, 153, 176, 295, 
                0, 3, 21, 39, 45, 59, 70, 97, 179,
                26, 49, 60, 81, 95, 113, 124, 133, 143, 178,
                13, 14, 18, 42, 52, 56, 61, 91, 132, 135, 180, 246
            ])
            labels_mapping = {x: i for i, x in enumerate(sorted(list(labels_to_filter)))}

            assert self.target_field == "fold_label"
            new_data = []
            for x in self.data:
                if x[self.target_field] in labels_to_filter:
                    x[self.target_field] = labels_mapping[x[self.target_field]]
                    new_data.append(x)
            self.data = new_data

        if self.data_name == "TapeRemoteHomologyDataset" and tokenizer_name == "protokens":
            # filter 1ldt.cif
            new_data = []
            for i in range(len(self.data)):
                if self.data[i]["pdb_id"] != "1ldt":
                    new_data.append(self.data[i])
            self.data = new_data
        
    def additional_preprocessing_for_TAPE_homo(self, tokenizer_name):
        """
        Some proteins are skipped, because for all their residues, at least 
        one backbone coordinates are NaN
        """
        if ((tokenizer_name == "proteinmpnn" or tokenizer_name == "mif") 
            and self.data_name == "TapeRemoteHomologyDataset"):
            
            if self.split == "train":
                skip_index = set([793, 796, 894, 1119, 1200, 1303, 1315, 1686, 1966, 2359, 
                            2583, 3302, 4239, 4406, 4769, 4904, 7669, 9642, 9903, 9933, 
                            9937, 10517, 11832, 11836, 11958])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data

            if self.split == "valid":
                
                skip_index = set([499, 619, 630])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
            if self.split == "test_family_holdout":
                skip_index = set([41, 828, 1131])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
            if self.split == "test_superfamily_holdout":
                skip_index = set([97, 111, 115, 129, 350])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
    
    def cache_all_tokenized(self):
        """Precompute all tokenization results"""
        
        # flag_list, name_list, type_list = [], [], []
        # for tp in ALL_TOKENIZER_TYPE:
        #     for key in ALL_TOKENIZER_TYPE[tp]:
        #         flag_list.append(isinstance(self.tokenizer, eval(key)))
        #         name_list.append(key.replace("Wrapped", "").replace("Tokenizer", "").lower())
        #         type_list.append(tp)
        # flag = any(flag_list)
        #
        # if flag:
        #     index = np.nonzero(flag_list)[0].item()
        #     tokenizer_name = name_list[index]
        #     if isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
        #         tokenizer_name += f"_{self.tokenizer.ckpt_name}"
        #
        #     # use continous reprs
        #     continuous_flag = self.use_continuous
        #     if type_list[index] == "continuous":
        #         # continous flag only for discretized tokenizers (i.e., VQ-VAE-based PSTs)
        #         # set to False to avoid redundancy for continuous tokenizers
        #         continuous_flag = False
        #     continuous_flag = "" if not continuous_flag else "_continuous"
        #
        #     # use sequence ids
        #     sequence_flag = "" if not self.use_sequence else "_sequence"
        #
        #     # cache file to avoid redundant tokenizing for the same tokenizer
        #     # when tuning hyper-parameters
        #     cache_file_name = self.get_target_file_name() + f"_{tokenizer_name}_tokenized{continuous_flag}{sequence_flag}"
        #     if os.path.exists(cache_file_name):
        #         new_data = torch.load(cache_file_name, weights_only=False)
        #         self.data = new_data
        #         self.additional_label_filtering_for_TAPE_homo(tokenizer_name)
        #         self.py_logger.info(f"Loading cahced tokenized data from {cache_file_name}")
        #         return
        #     else:
        #         self.py_logger.info(f"Cannot load cahced tokenized data from {cache_file_name}, caching now")
        # else:
        #     raise NotImplementedError
        #
        #
        # self.additional_preprocessing_for_TAPE_homo(tokenizer_name)
        #
        # # pre-checking
        # for index in tqdm(range(len(self))):
        #     try:
        #         self[index]
        #     except:
        #         self.py_logger.info(f"[Error]: Something wrong for index {index} "
        #                             f"when using {tokenizer_name}\n[Warning]: if "
        #                             f"you're using your own PST, you can skip wrongly "
        #                             f"indexed samples for your PST. But please be aware that "
        #                             f"other PST benchmakred by the authors all used these samples")
        #         raise IndexError
        # if flag:
        #     torch.save(self.data, cache_file_name)
        try:
            tok = getattr(self, "tokenizer", None)
            if tok is None and hasattr(self, "datamodule"):
                tok = self.datamodule.get_tokenizer()
            if tok is not None and getattr(tok, "get_num_tokens", None):
                if tok.get_num_tokens() is None:
                    return  # no-op for continuous features
        except Exception:
            pass
        raise NotImplementedError

    def shard(self, shard_idx: int, num_shards: int):
        """Shard the dataset inplace by keeping the every `num_shards`"""
        self.py_logger.info(f"Loading shard {shard_idx} with world size {num_shards}")

        indices = range(len(self))[shard_idx::num_shards]
        self.data = [self.data[i] for i in indices]

        self.py_logger.info("Done sharded loading.")

    
    def splitting_dataset(self, fold_split_ratio=0.4, fold_valid_ratio=0.2, 
        superfamily_split_ratio=0.2, superfamily_valid_ratio=0.2, seed=42
    ):
        """
        Perform splitting:
        - step 1: for each fold, split superfamilies into two groups (60%, 40%) 
            for training and test, resulting in the fold-level datasets
        - step 2: among the fold-level training data, for each superfamily, 
            split the family into two groups (60%, 40%) for training and test, 
            resulting in the superfamily-level datasets
        - Step 3: from the test data above, randomly take out 20% proteins 
            to create a validation set
        """

        # for each fold, split superfamilies
        fold_list, superfamily_list = [], []
        for i in range(len(self.data)):
            fold_list.append(self.data[i]["fold_label"])
            superfamily_list.append(self.data[i]["superfamily_label"])
        fold_list, superfamily_list = np.array(fold_list), np.array(superfamily_list)

        fold_train_indices, fold_test_indices = [], []
        for fold_idx in set(fold_list):
            indices = (fold_list == fold_idx).nonzero()[0]
            superfamily_vocab = list(set(superfamily_list[indices]))
            if int(len(superfamily_vocab) * fold_split_ratio) > 0:
                sf_train, sf_test = train_test_split(superfamily_vocab, 
                                        test_size=fold_split_ratio, random_state=seed)
                sf_train = np.isin(superfamily_list[indices], sf_train)
                sf_test = np.isin(superfamily_list[indices], sf_test)
                fold_train_indices += (indices[sf_train]).tolist()
                fold_test_indices += (indices[sf_test]).tolist()
            else:
                fold_train_indices += indices.tolist()

        fold_test_indices, fold_valid_indices = train_test_split(fold_test_indices, 
                                    test_size=fold_valid_ratio, random_state=seed)

        # among the fold-level training data, for each superfamily, random split 
        fold_train_indices = np.array(fold_train_indices)
        sf_train_indices, sf_test_indices = [], []
        for sf_idx in set(superfamily_list[fold_train_indices].tolist()):
            indices = (superfamily_list[fold_train_indices] == sf_idx).nonzero()[0]
            if int(len(indices) * superfamily_split_ratio) > 0:
                train_indices, test_indices = train_test_split(indices, 
                                        test_size=superfamily_split_ratio, random_state=seed)
                sf_train_indices += fold_train_indices[train_indices].tolist()
                sf_test_indices += fold_train_indices[test_indices].tolist()
            else:
                sf_train_indices += fold_train_indices[indices].tolist()

        sf_test_indices, sf_valid_indices = train_test_split(sf_test_indices, 
                                    test_size=superfamily_valid_ratio, random_state=seed)
        
        train_indices = sf_train_indices
        valid_indices = fold_valid_indices + sf_valid_indices
        fold_test_indices = fold_test_indices
        superfamily_test_indices = sf_test_indices

        assert len(train_indices) == len(set(train_indices))
        assert len(valid_indices) == len(set(valid_indices))
        assert len(fold_test_indices) == len(set(fold_test_indices))
        assert len(superfamily_test_indices) == len(set(superfamily_test_indices))
        assert len(self.data) == (len(set(train_indices)) + len(set(valid_indices))
                            + len(set(fold_test_indices)) + len(set(superfamily_test_indices)))

        self.py_logger.info(f"After splitting, result in {len(train_indices)} training data, "
                            f"{len(valid_indices)} validation data, {len(fold_test_indices)} fold-level test data, "
                            f"{len(superfamily_test_indices)} superfamily-level test data")
        
        train_data = [self.data[idx] for idx in train_indices]
        valid_data = [self.data[idx] for idx in valid_indices]
        fold_test_data = [self.data[idx] for idx in fold_test_indices]
        superfamily_test_data = [self.data[idx] for idx in superfamily_test_indices]
        return train_data, valid_data, fold_test_data, superfamily_test_data
