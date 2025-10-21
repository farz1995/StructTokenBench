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
from biotite.structure.io.pdbx import CIFFile, convert
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from src.protein_chain import WrappedProteinChain
from src import util
from src.stb_tokenizers import WrappedMyRepShakeTokenizer, WrappedMyRepBioLIP2Tokenizer

def convert_chain_id(pdb_path, chain_id):

    # Ensure pdb_path is a string (Path objects on some platforms don't have
    # str methods like endswith), avoid AttributeError for PosixPath/WindowsPath
    pdb_path = str(pdb_path)

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
    
    def shard(self, shard_idx: int, num_shards: int):
        """
        Shard the dataset across multiple processes.
        Subsets self.data to only include samples for this shard.
        """
        if num_shards <= 1:
            return  # no sharding needed

        total_samples = len(self.data)
        samples_per_shard = total_samples // num_shards
        remainder = total_samples % num_shards

        # Calculate start and end indices for this shard
        start_idx = shard_idx * samples_per_shard + min(shard_idx, remainder)
        end_idx = start_idx + samples_per_shard + (1 if shard_idx < remainder else 0)

        self.data = self.data[start_idx:end_idx]

        # Log the sharding if possible
        try:
            self.py_logger.info(f"Sharded dataset: shard {shard_idx}/{num_shards}, "
                               f"samples {start_idx} to {end_idx-1} (total {len(self.data)})")
        except Exception:
            pass

    def get_target_file_name(self,):
        assert NotImplementedError

    def save_structured_data(self, ):
        file = self.get_target_file_name()
        torch.save(self.data, file)
        self.py_logger.info(f"Save the processed, structured data to disk: {file}")
    
    def prepare_structure_loading(self):
        # default no-op; datasets can override if they need special preprocessing
        return None

    def collate_fn(self, batch):
        """
        Robust collation for variable-length features.

        Inputs per sample can be:
          - {"token_ids": FloatTensor(L,D), "label": int|list[int], "residue_index": (L,) optional}
          - ({"token_ids": ...}, label)
          - (FloatTensor(L,D), label) or (ndarray(L,D), label)

        Outputs:
          - input_list: list with one dict of padded tensors
          - targets:
              global task  -> LongTensor (B,)
              local task   -> FloatTensor (B, Lmax) with -100 for pad
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
                # Accept per-residue labels as sequences
                if isinstance(label, (list, tuple, np.ndarray, torch.Tensor)):
                    item["label"] = label
                else:
                    try:
                        item["label"] = int(label)
                    except Exception:
                        item["label"] = label
                # Debug-print a few labels
                try:
                    if not hasattr(self, "_collate_debug_printed"):
                        self._collate_debug_printed = 0
                    if self._collate_debug_printed < 3:
                        head = label[:10] if isinstance(label, (list, tuple)) else (label.detach().cpu().numpy()[:10].tolist() if torch.is_tensor(label) and label.ndim > 0 else label)
                        print(f"[DEBUG][collate_fn] sample label type={type(label)}, len={len(label) if hasattr(label, '__len__') else 'NA'}, head={head}")
                        self._collate_debug_printed += 1
                except Exception:
                    pass
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
                item["label"] = 0  # default (overwritten upstream)

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

        # Detect local vs global labels by inspecting first item's label
        first_label = normed[0].get("label", 0)
        is_local = isinstance(first_label, (list, tuple, np.ndarray, torch.Tensor))
        # extra debug: print a summary of the first label
        try:
            head = first_label[:10] if isinstance(first_label, (list, tuple)) else (first_label.detach().cpu().numpy()[:10].tolist() if torch.is_tensor(first_label) and first_label.ndim > 0 else first_label)
            print(f"[DEBUG][collate_fn] detected is_local={is_local}, first_label_type={type(first_label)}, head={head}")
        except Exception:
            pass

        if is_local:
            targets = torch.full((B, Lmax), fill_value=-100, dtype=torch.float32)
        else:
            targets = torch.zeros((B,), dtype=torch.long)

        for i, it in enumerate(normed):
            x = it["token_ids"]; L = x.shape[0]
            feats[i, :L] = x
            attn[i, :L] = False
            if "residue_index" in it and it["residue_index"] is not None:
                ri = it["residue_index"]
                if not torch.is_tensor(ri):
                    ri = torch.as_tensor(ri, dtype=torch.int32)
                resid[i, :L] = ri[:L]
            else:
                resid[i, :L] = torch.arange(L, dtype=torch.int32)

            lab = it.get("label", 0)
            if is_local:
                lab_t = torch.as_tensor(lab, dtype=torch.float32)
                targets[i, :min(L, lab_t.shape[0])] = lab_t[:L]
            else:
                targets[i] = int(lab)

        lengths = torch.as_tensor(lengths, dtype=torch.int32)

        input_list = [{
            "token_ids": feats,
            "attention_mask": attn,
            "residue_index": resid,
        }]
        out = {
            "input_list": input_list,
            "targets": targets,
            # extras
            "token_ids": feats,
            "attention_mask": attn,
            "residue_index": resid,
            "labels": targets,
            "lengths": lengths,
        }
        return out
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_pdb_chain(self, pdb_id, chain_id):
        try:
            # Support multiple layouts:
            #  - PDB_DATA_DIR=/.../pdb_data/
            #  - PDB_DATA_DIR=/.../pdb_data/mmcif_files/
            #  - PDB_DATA_DIR=/.../pdb_data/mmcif_files/ (containing nested mmcif_files/)
            cand3 = os.path.join(self.PDB_DATA_DIR, "mmcif_files", "mmcif_files", f"{pdb_id}.cif")
            cand1 = os.path.join(self.PDB_DATA_DIR, "mmcif_files", f"{pdb_id}.cif")
            cand2 = os.path.join(self.PDB_DATA_DIR, f"{pdb_id}.cif")
            file = next((c for c in (cand3, cand1, cand2) if os.path.exists(c)), cand1)
            protein_chain = WrappedProteinChain.from_cif(
                file, chain_id=chain_id, id=pdb_id
            )
        except Exception:
            self.py_logger.info(
                f"Cannot retrieve from local cluster, pdb_id: {pdb_id}, chain_id: {chain_id}"
            )
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
        # Support base pointing to either the dataset root or the mmcif directory,
        # and handle a nested "mmcif_files/mmcif_files" layout.
        cand3 = os.path.join(self.PDB_DATA_DIR, "mmcif_files", "mmcif_files", f"{pdb_id}.cif")
        cand1 = os.path.join(self.PDB_DATA_DIR, "mmcif_files", f"{pdb_id}.cif")
        cand2 = os.path.join(self.PDB_DATA_DIR, f"{pdb_id}.cif")
        for c in (cand3, cand1, cand2):
            if os.path.exists(c):
                return c
        # Fallback to a reasonable default path for error messaging
        return cand1 if os.path.isdir(os.path.join(self.PDB_DATA_DIR, "mmcif_files")) else cand2
    
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
            try:
                chain_id, is_changed = convert_chain_id(pdb_path, chain_id)
            except Exception as e:
                self.py_logger.warning(f"Failed to convert chain_id for pdb_id: {pdb_id}, chain_id: {chain_id}. Skipping sample. Error: {e}")
                return None # Return None to indicate this sample should be filtered out
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


        # encode protein structure into token_ids without importing tokenizer classes here
        tok_name = getattr(self.tokenizer.__class__, "__name__", "")
        if tok_name in [
            "WrappedESM3Tokenizer",
            "WrappedOurPretrainedTokenizer",
        ]:
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(
                pdb_chain, self.use_continuous, self.use_sequence
            )
        elif isinstance(self.tokenizer, WrappedMyRepShakeTokenizer):
            # Continuous representation for ProteinShake tasks (or generic use)
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(
                pdb_path, chain_id, self.use_sequence
            )
        elif isinstance(self.tokenizer, WrappedMyRepBioLIP2Tokenizer):
            # Continuous representation for BioLIP2 tasks (same API as ProteinShake)
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(
                pdb_path, chain_id, self.use_sequence
            )
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

            # If alignment produced an empty sequence, skip this sample gracefully
            if len(token_ids) == 0:
                try:
                    self.py_logger.warning(f"Empty alignment for pdb_id={pdb_id}, chain_id={chain_id}; skipping sample")
                except Exception:
                    pass
                return None

            if org_len - len(token_ids) != 0:
                print(">> residue reduced by : ", org_len - len(token_ids))

        # select according to residue range constraints for some global tasks
        selected_indices = self._get_selected_indices(residue_index, residue_range)
        # If nothing falls within the selected range, skip this sample instead of asserting
        if len(selected_indices) == 0:
            try:
                self.py_logger.warning(f"No residues selected after range filtering for pdb_id={pdb_id}, chain_id={chain_id}, range={residue_range}; skipping sample")
            except Exception:
                pass
            return None

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
        pass

    def splitting_dataset(
        self,
        fold_split_ratio: float = 0.2,
        fold_valid_ratio: float = 0.1,
        superfamily_split_ratio: float = 0.2,
        superfamily_valid_ratio: float = 0.1,
    ):
        """
        Create four splits: train, validation, fold_test, superfamily_test.
        - fold_test holds out a fraction of entire fold_label classes
        - superfamily_test holds out a fraction of entire superfamily_label classes
        - train/validation come from the remaining pool (excluding both test sets)
          using a per-sample split (stratified by fold_label when possible).

        Notes:
        - superfamily_valid_ratio is accepted for API compatibility but not used
          separately (we keep a single validation set).
        - Deterministic behavior with a fixed seed for reproducibility.
        """
        rng = np.random.RandomState(0)
        data = self.data

        # Collect label arrays
        folds = [int(x["fold_label"]) for x in data if "fold_label" in x]
        superfams = [int(x["superfamily_label"]) for x in data if "superfamily_label" in x]
        assert len(folds) == len(data) and len(superfams) == len(data), (
            "Each sample must have 'fold_label' and 'superfamily_label' before splitting"
        )

        # Choose held-out folds
        unique_folds = np.array(sorted(set(folds)))
        rng.shuffle(unique_folds)
        n_hold_folds = max(1, int(round(fold_split_ratio * len(unique_folds)))) if len(unique_folds) > 0 else 0
        heldout_folds = set(unique_folds[:n_hold_folds])

        # Choose held-out superfams
        unique_sf = np.array(sorted(set(superfams)))
        rng.shuffle(unique_sf)
        n_hold_sf = max(1, int(round(superfamily_split_ratio * len(unique_sf)))) if len(unique_sf) > 0 else 0
        heldout_sf = set(unique_sf[:n_hold_sf])

        # Indices for each split
        idx_all = np.arange(len(data))
        idx_fold_test = [i for i in idx_all if folds[i] in heldout_folds]
        idx_sf_test = [i for i in idx_all if superfams[i] in heldout_sf]
        heldout_any = set(idx_fold_test) | set(idx_sf_test)
        pool = [i for i in idx_all if i not in heldout_any]

        # Train/Val split from pool (stratify by fold when possible)
        if len(pool) == 0:
            train_idx, val_idx = [], []
        elif len(pool) == 1 or fold_valid_ratio <= 0:
            train_idx, val_idx = pool, []
        else:
            y_fold = [folds[i] for i in pool]
            try:
                train_idx, val_idx = train_test_split(
                    pool,
                    test_size=min(max(fold_valid_ratio, 0.0), 0.9),
                    random_state=0,
                    stratify=y_fold if len(set(y_fold)) > 1 else None,
                )
            except Exception:
                # fallback: no stratification
                train_idx, val_idx = train_test_split(
                    pool, test_size=min(max(fold_valid_ratio, 0.0), 0.9), random_state=0
                )

        # Materialize splits
        train = [data[i] for i in train_idx]
        valid = [data[i] for i in val_idx]
        fold_test = [data[i] for i in idx_fold_test]
        superfamily_test = [data[i] for i in idx_sf_test]

        # Basic logging if possible
        try:
            self.py_logger.info(
                f"Split sizes -> train={len(train)}, valid={len(valid)}, "
                f"fold_test={len(fold_test)}, superfamily_test={len(superfamily_test)}"
            )
        except Exception:
            pass

        return [train, valid, fold_test, superfamily_test]
