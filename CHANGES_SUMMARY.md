# StructTokenBench Fixes Summary

## Overview
This document summarizes all changes made to support:
1. ✅ Running ProteinShake binding site task with local mmCIF files and H5 embeddings
2. ✅ Running sensitivity (conformational) task with continuous tokenizers
3. ✅ Handling per-residue local labels for binding site prediction
4. ✅ Gracefully handling missing or insufficient H5 embeddings

## Files Modified

### 1. `StructTokenBench/src/dataset/base.py`
**Changes**:
- Removed circular import of tokenizers (was importing `from tokenizer import *` and specific heavy tokenizers)
- Made `get_pdb_chain()` robust to both paths:
  - `data.pdb_data_dir=/.../pdb_data/` or `/.../pdb_data/mmcif_files/`
- Fixed `collate_fn()` to handle **per-residue (local) labels**:
  - Detects if label is a list/array (per-residue) vs scalar (global)
  - For local labels: pads to `(B, L_max)` with -100 padding value
  - For global labels: keeps existing behavior as `(B,)` scalar
  - Added debug logging to show label types and shapes
- Added CIF file support via biotite for mmCIF parsing

**Why**: ProteinShake binding site uses per-residue binary labels (one 0/1 per residue). The collator must pad variable-length sequences and mask invalid positions.

### 2. `StructTokenBench/src/dataset/proteinshake_binding_site.py`
**Changes**:
- Updated `retrieve_pdb_path()` to prefer **local mmCIF files first**:
  1. Check `self.PDB_DATA_DIR/mmcif_files/<pdb_id>.cif`
  2. Fall back to ProteinShake cache if available
  3. Otherwise return expected local path
- Proper chain-id to residue index conversion using mmCIF files

**Why**: You have local mmCIF files and don't want to download ProteinShake's cache. This ensures local files are used when available.

### 3. `StructTokenBench/src/model_module.py`
**Changes**:
- Fixed `SequenceClassificationModel.forward()` to handle local (per-residue) labels:
  - No longer enforces label range check (0 to num_labels) for local labels
  - Allows -100 as a valid pad value (standard PyTorch convention)
- Updated `ZeroshotProximityModel.setup()` to handle **continuous tokenizers without codebooks**:
  - Checks if tokenizer has `get_num_tokens()` returning None (continuous)
  - Skips codebook initialization for continuous tokenizers
  - Only builds similarity matrices when a valid 2D codebook tensor exists
  - Allows `codebook_embedding` to be None/empty for continuous embeddings

**Why**: 
- Binding site task uses local labels (needs -100 masking); model must accept these
- Sensitivity task uses continuous H5 embeddings (no discrete codebook); model must not crash when codebook is missing

### 4. `StructTokenBench/src/dataset/conformational_switch.py`
**Changes**:
- Updated imports to include `WrappedMyRepShakeTokenizer` alongside `WrappedMyRepTokenizer`
- Updated `_get_item_structural_tokens()` to handle **continuous H5 tokenizers**:
  - Catches exceptions when loading H5 embeddings and logs them
  - Returns `None` to skip samples with missing embeddings
  - Added debug logging: `[ConformationalSwitch] Failed to load H5 embeddings for {pdb_id}/{chain_id}: {error}`
- Updated `collate_fn()` to handle **empty batches gracefully**:
  - If all samples are filtered out (None), returns `None` instead of crashing
  - Prevents "ValueError: not enough values to unpack" when batch is empty

**Why**: 
- Sensitivity task requires continuous H5 embeddings, not all PDB chains may be in the H5
- If a whole batch is skipped, collate_fn was crashing trying to unpack an empty tuple

### 5. `StructTokenBench/src/data_module.py`
**Changes**:
- Updated `load_class()` function to add **tokenizer module alias**:
  - Accepts `src.tokenizers.WrappedMyRepTokenizer` as an alias
  - Transparently maps it to `src.stb_tokenizers.WrappedMyRepTokenizer`
  - Adds fallback import for missing modules
- This allows commands like `tokenizer=src.tokenizers.WrappedMyRepTokenizer` to work

**Why**: Commands from README or users may reference `src.tokenizers.*`, but the actual package is `src.stb_tokenizers.*`. This adds backward-compatibility.

### 6. `StructTokenBench/src/script/run_supervised_task.py`
**Changes**:
- Added logic to **merge tokenizer_kwargs from Hydra command**:
  - Reads `+tokenizer_kwargs.h5_path=...` from command line
  - Reads `+tokenizer_kwargs.embeddings_dataset=...` if specified
  - Merges with pretrained model config (Hydra overrides win)
- Added **auto-alignment of model.d_model to tokenizer embed_dim**:
  - For continuous tokenizers, auto-detects `embed_dim` or `d_model` attribute
  - Warns and sets `model.d_model` to match tokenizer
  - Prevents shape mismatches between embedding layer and model

**Why**:
- H5 embeddings require passing the file path; tokenizer_kwargs is the standard way
- Continuous tokenizers have a fixed embedding dimension (e.g., 128); model must match

## Files Created

### 7. `SENSITIVITY_TASK_GUIDE.md` (NEW)
Comprehensive guide for running sensitivity tasks with continuous tokenizers, including:
- Available H5 files
- Supported tokenizer types
- Command template
- Common issues and solutions
- Debugging tips
- Integration with local data

## Key Features Added

### ✅ Per-Residue Label Support
- Binding site task now properly handles per-residue binary labels (0/1 per residue)
- Collation handles variable-length sequences with -100 padding
- Model loss only includes valid residues (ignores -100)

### ✅ Continuous Tokenizer Support
- Sensitivity/conformational task now works with H5 embeddings (no discrete codebook)
- Graceful handling of missing embeddings (skips samples with debug logging)
- Auto-alignment of model dimensions

### ✅ Local File Support
- Binding site dataset prefers local mmCIF files over ProteinShake cache
- Local file path configuration via `data.pdb_data_dir`
- Proper chain-id to residue-index mapping

### ✅ H5 Embeddings Integration
- Tokenizer kwargs from command line merged correctly
- Model dimensions auto-aligned to H5 embedding dimensions
- Clear error messages when H5 files or keys are missing

## Example Commands

### Binding Site Task with Local Data
```bash
python -m src.script.run_supervised_task \
  --config-name proteinshake_binding_site \
  trainer.devices=1 \
  tokenizer=WrappedMyRepShakeTokenizer \
  data.pdb_data_dir="/path/to/pdb_data/mmcif_files/" \
  +tokenizer_kwargs.h5_path="/path/to/vq_embed_proteinshake.h5" \
  +tokenizer_kwargs.embeddings_dataset="/vq_embe_proteinshake"
```

### Sensitivity Task with Continuous Embeddings
```bash
python -m src.script.run_supervised_task \
  --config-name conformational_switch \
  trainer.devices=1 \
  tokenizer=WrappedMyRepTokenizer \
  data.pdb_data_dir="/path/to/pdb_data/mmcif_files/" \
  +tokenizer_kwargs.h5_path="/path/to/vq_embed.h5" \
  data.use_continuous=true \
  test_only=true
```

## Testing Checklist

Before running production experiments, verify:
- [ ] H5 file path is absolute (not `~` or relative)
- [ ] mmCIF directory contains `.cif` files for your PDB IDs
- [ ] H5 file contains keys for your PDB/chain combinations (see `SENSITIVITY_TASK_GUIDE.md`)
- [ ] Model config specifies correct task (binding_site vs conformational_switch)
- [ ] Tokenizer kwargs are passed via `+tokenizer_kwargs.*`
- [ ] Logs show `[run] Continuous tokenizer detected (H5 features).` for H5-based tasks

## Backward Compatibility

All changes are **backward compatible**:
- Discrete tokenizers (ESM3, FoldSeek, etc.) still work as before
- Binding site task with discrete tokens still works
- Only new use cases (continuous + per-residue labels) are added
- Model gracefully handles both codebook and non-codebook scenarios

## Troubleshooting

See `SENSITIVITY_TASK_GUIDE.md` for:
- Common error messages and solutions
- How to inspect H5 file contents
- Debug logging to identify missing embeddings
- Integration with custom PDB/embedding data

## Notes for Future Development

1. Consider precomputing and caching H5 embeddings alongside datasets
2. Could add H5 validation tool to verify coverage of PDB IDs
3. Per-residue label padding could be optimized with masking tokens
4. Tokenizer module aliasing could be expanded to other imports

