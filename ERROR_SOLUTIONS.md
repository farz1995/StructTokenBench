# Error Solutions & Root Cause Analysis

## Error 1: "No such file or directory: '/home/.../.cache/proteinshake/protein_ligand_interface/raw/...'"

### Root Cause
The binding site dataset was trying to read PDB files from ProteinShake's default cache directory (`~/.cache/proteinshake/`), but:
1. You don't have ProteinShake cache populated
2. You have local mmCIF files instead
3. The code didn't check for local files first

### Error Stack
```
[util][WARNING] - Failed to convert chain_id for pdb_id: 5k0m, chain_id: A. 
Skipping sample. Error: [Errno 2] No such file or directory: 
'/home/fe5vb/.cache/proteinshake/protein_ligand_interface/raw/5k0m/5k0m.pdb'
```

### Solution Applied
Updated `StructTokenBench/src/dataset/proteinshake_binding_site.py`:
```python
def retrieve_pdb_path(self, pdb_id: str, chain_id=None) -> Path:
    # 1. FIRST: Try local mmCIF files
    local_cif = Path(self.PDB_DATA_DIR) / "mmcif_files" / f"{pdb_id}.cif"
    if local_cif.exists():
        return local_cif
    
    # 2. SECOND: Try ProteinShake cache (fallback)
    ps_path = ...  # ~/.cache/proteinshake/...
    if ps_path.exists():
        return ps_path
    
    # 3. LAST: Return expected local path (will handle missing later)
    return local_cif
```

### Now Works
- Local mmCIF files are preferred automatically
- No need to download ProteinShake cache
- Command: `data.pdb_data_dir="/path/to/pdb_data/mmcif_files/"`

---

## Error 2: "ImportError: cannot import name 'BaseDataset' from 'dataset.base'"

### Root Cause
Circular import chain:
1. `data_module.py` imports `from dataset import *`
2. `dataset/__init__.py` imports `from dataset.biolip2 import BioLIP2FunctionDataset`
3. `dataset/biolip2.py` tries to import `from dataset.base import BaseDataset`
4. BUT `dataset/base.py` imports `from tokenizer import *` (all tokenizers, heavy imports)
5. `tokenizer.py` imports dependencies that fail (ESM3, etc.)
6. → `base.py` import fails → `BaseDataset` not defined → ImportError

### Error Stack
```
File "/home/fe5vb/project/PST/StructTokenBench/src/dataset/biolip2.py", line 9, in <module>
    from dataset.base import BaseDataset
ImportError: cannot import name 'BaseDataset' from 'dataset.base'
```

### Solution Applied
Removed heavy imports from `dataset/base.py`:
```python
# BEFORE (problematic):
from tokenizer import *
from src.stb_tokenizers import WrappedMyRepShakeTokenizer

# AFTER (fixed):
# (removed both)
# Later in code: use runtime dispatch by class name instead
```

This breaks the circular import chain. Tokenizers are instantiated at runtime via Hydra, not at module import.

### Now Works
- `data_module.py` loads without tokenizer dependencies
- Tokenizers loaded later when needed
- All datasets can import `BaseDataset` cleanly

---

## Error 3: "TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'"

### Root Cause
Binding site dataset has **per-residue labels** (one 0/1 per residue), not scalar labels:
```python
# Example label from TSV:
# 3fcq  A  38  110,111,112,113,...  ← binding residue indices

# Becomes:
label = [0, 0, 1, 0, 1, 1, 0, ...]  # per-residue binary array
```

But the old code assumed scalar labels:
```python
item["label"] = int(label)  # ← CRASH! label is a list
```

### Error Stack
```
File ".../src/dataset/base.py", line 193, in collate_fn
    item["label"] = int(label)
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'
```

### Solution Applied
Updated `collate_fn` in `dataset/base.py` to detect and handle both:
```python
def collate_fn(self, batch):
    # Detect global vs local labels
    first_label = batch[0]["label"]
    is_local = isinstance(first_label, (list, tuple, np.ndarray, torch.Tensor))
    
    if is_local:
        # Per-residue labels: pad to (B, L_max) with -100
        # This preserves variable sequence lengths
        padded_labels = ...  # torch.zeros(B, L_max) filled with -100
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        batch["targets"] = padded_labels  # shape (B, L_max)
    else:
        # Scalar labels: keep as (B,)
        batch["targets"] = torch.tensor(labels)  # shape (B,)
```

### Now Works
- Binding site binding_site task handles per-residue labels correctly
- Variable-length sequences properly padded to batch max length
- Loss functions ignore -100 masked positions automatically

---

## Error 4: "TypeError: int() argument must be a string... not 'list'" (collate_fn)

### Root Cause
Same as Error 3, but more defensive handling needed for tuple unpacking in different dataset types.

### Solution Applied
```python
# When receiving samples as tuples/lists with labels
if isinstance(label, (list, tuple, np.ndarray, torch.Tensor)):
    item["label"] = label  # Keep as-is, don't convert to int
else:
    try:
        item["label"] = int(label)
    except Exception:
        item["label"] = label  # Fallback passthrough
```

---

## Error 5: "RuntimeError: codebook_embedding is empty. Datamodule must expose `tokenizer.codebook`"

### Root Cause
Sensitivity task uses **continuous H5 embeddings**, not discrete tokens with a codebook. The model setup code assumed a codebook always exists:
```python
# Old code in model_module.py:
if not codebook_embedding_exists:
    raise RuntimeError("codebook_embedding is empty.")  # ← CRASH!
```

But for continuous tokenizers, there's no codebook!

### Error Stack
```
File ".../src/model_module.py", line 675, in setup
    raise RuntimeError(
        "codebook_embedding is empty. Datamodule must expose `tokenizer.codebook`..."
    )
RuntimeError: codebook_embedding is empty. ...
```

### Solution Applied
Added detection for continuous tokenizers in `model_module.py`:
```python
def setup(self, stage=None):
    # Check if tokenizer is continuous (no discrete codebook)
    is_continuous = (
        self.trainer.datamodule.get_tokenizer().get_num_tokens() is None
    )
    
    if is_continuous:
        # Skip codebook initialization; use continuous embeddings
        self.codebook_embedding = None  # OK for continuous!
        logger.info("[Model] Continuous tokenizer detected; skipping codebook.")
    else:
        # Discrete tokenizer: require codebook as before
        self.codebook_embedding = ...  # Build from tokenizer
```

### Now Works
- Sensitivity task runs with `WrappedMyRepTokenizer` (continuous)
- No crash when codebook is missing
- Model uses direct embedding similarity, not discrete codes

---

## Error 6: "ValueError: not enough values to unpack (expected 3, got 0)"

### Root Cause
In conformational_switch dataset, **all samples were skipped** because H5 embeddings couldn't be loaded:
1. Dataset tries to load H5 embeddings for each sample
2. H5 file doesn't have the required PDB/chain combinations
3. All samples return `None` in `__getitem__`
4. `collate_fn` filters out `None` values: `batch = [None, None, None] → []`
5. Tries to unpack empty batch:
   ```python
   prot1_input_ids, prot2_input_ids, labels = tuple(zip(*[]))  # ← CRASH!
   ```

### Error Stack
```
File ".../dataset/conformational_switch.py", line 52, in collate_fn
    prot1_input_ids, prot2_input_ids, labels = tuple(zip(*batch))
ValueError: not enough values to unpack (expected 3, got 0)
```

### Solution Applied
Made `collate_fn` gracefully handle empty batches:
```python
def collate_fn(self, batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        # All samples were skipped (likely missing H5 embeddings)
        return None  # Signal to skip this batch
    
    prot1_input_ids, prot2_input_ids, labels = tuple(zip(*batch))
    # ... rest of collation ...
```

And added debug logging in `_get_item_structural_tokens`:
```python
except Exception as e:
    import logging
    logger = logging.getLogger("dataset")
    logger.warning(f"[ConformationalSwitch] Failed to load H5 embeddings for {pdb_id}/{chain_id}: {e}")
    return None
```

### Now Works
- If H5 doesn't have embeddings, dataset gracefully skips samples
- Debug logs show which samples are missing: `[ConformationalSwitch] Failed to load H5 embeddings for 1abc/A: KeyError: No H5 key found for ...`
- User can see if their H5 file needs to be rebuilt with those PDB/chains

---

## Error 7: "ModuleNotFoundError: No module named 'src.tokenizers'"

### Root Cause
User (or README) references `tokenizer=src.tokenizers.WrappedMyRepTokenizer`, but the actual module is `src.stb_tokenizers`. The class loader couldn't find it:
```python
# Hydra tries to load:
import_module("src.tokenizers")  # ← Not found!
```

### Error Stack
```
ModuleNotFoundError: No module named 'src.tokenizers'
```

### Solution Applied
Added alias/fallback in `data_module.py`:
```python
def load_class(qualname: str):
    if "." in qualname:
        mod, cls = qualname.rsplit(".", 1)
        
        # Add backward-compat alias
        if mod == "src.tokenizers" or mod.startswith("src.tokenizers"):
            mod = mod.replace("src.tokenizers", "src.stb_tokenizers", 1)
        
        try:
            return getattr(import_module(mod), cls)
        except ModuleNotFoundError:
            # Fallback to stb_tokenizers
            import src.stb_tokenizers as T
            return getattr(T, cls)
```

### Now Works
- Commands using `tokenizer=src.tokenizers.WrappedMyRepTokenizer` work
- Transparently maps to `src.stb_tokenizers.WrappedMyRepTokenizer`
- Backward compatible with both naming conventions

---

## Error 8: "Shape mismatch: model expects input (B, 256), got (B, 128)"

### Root Cause
User set `model.d_model=256` but continuous tokenizers (H5) output 128-dim embeddings:
1. Model's embedding layer expects 256-dim input
2. H5 embeddings are 128-dim (fixed in H5 file)
3. Shape mismatch in first forward pass

### Error Stack
```
RuntimeError: Expected 2D input (line_1: (B*L), features: 128 or 256) with shape 
(batch_size, 128, ...) but received (..., 256, ...)
```

### Solution Applied
Auto-align `model.d_model` to tokenizer's embed_dim in `run_supervised_task.py`:
```python
if cfg.data.use_continuous:
    tok = datamodule.get_tokenizer()
    embed_dim = getattr(tok, "embed_dim", None) or getattr(tok, "d_model", None)
    
    if embed_dim is not None:
        if cfg.model.d_model != embed_dim:
            logger.info(f"[run] Setting model.d_model to tokenizer embed_dim={embed_dim}")
            cfg.model.d_model = embed_dim
```

### Now Works
- Model automatically syncs to tokenizer's embedding dimension
- No need to manually set `model.d_model` for continuous tokenizers
- User can still override if needed

---

## Summary: Debug Checklist for Your Setup

When running binding site or sensitivity tasks, check:

### Before Running
1. **H5 file exists and is readable**
   ```bash
   ls -lh /path/to/vq_embed.h5
   ```

2. **mmCIF files exist**
   ```bash
   ls -lh /path/to/pdb_data/mmcif_files/ | head
   ```

3. **H5 contains keys for your PDB/chains** (if you get KeyError):
   ```python
   import h5py
   with h5py.File("/path/to/vq_embed.h5", "r") as h5:
       print(list(h5.keys())[:20])  # Show first 20 keys
   ```

4. **Tokenizer kwargs are passed correctly**
   ```
   +tokenizer_kwargs.h5_path="/absolute/path/to/vq_embed.h5"
   ```

### While Running
5. **Check logs for:**
   - `[run] Continuous tokenizer detected (H5 features).` ← Good
   - `[ConformationalSwitch] Failed to load H5 embeddings for ...` ← Some samples missing
   - `Setting model.d_model to tokenizer embed_dim=128` ← Auto-sync working

### If Still Getting Errors
6. **Check error messages:**
   - "No H5 key found" → H5 doesn't have that PDB/chain; try a different H5
   - "not enough values to unpack" → ALL samples missing; definitely wrong H5
   - "embed_dim mismatch" → Check model.d_model override

---

## Files to Read for More Context

- `SENSITIVITY_TASK_GUIDE.md` - Full guide for sensitivity tasks
- `CHANGES_SUMMARY.md` - Overview of all modifications
- `src/dataset/base.py` - Per-residue label handling (collate_fn)
- `src/dataset/proteinshake_binding_site.py` - Local mmCIF preference (retrieve_pdb_path)
- `src/model_module.py` - Continuous tokenizer support (setup methods)
- `src/dataset/conformational_switch.py` - Empty batch handling (collate_fn)
