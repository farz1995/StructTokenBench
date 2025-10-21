# Quick Start Guide

## Setup Paths
```bash
# Set these to your actual paths
export PDB_DATA_DIR="/path/to/pdb_data/mmcif_files"
export DATA_DIR="/path/to/struct_token_bench_release_data/data"
export H5_EMBED="$(pwd)/StructTokenBench/src/vq_embed.h5"
export LOG_DIR="$(pwd)/struct_token_bench_logs"
```

## Task 1: ProteinShake Binding Site

### Train
```bash
cd StructTokenBench

python -m src.script.run_supervised_task \
  --config-name proteinshake_binding_site \
  trainer.devices=1 \
  trainer.default_root_dir="$LOG_DIR/proteinshake_runs" \
  run_name="bindshake_train" \
  tokenizer=WrappedMyRepShakeTokenizer \
  tokenizer_device=cuda \
  precompute_tokens=false \
  model.d_model=128 \
  data.use_sequence=false \
  data.filter_length=700 \
  data.truncation_length=700 \
  data.data_path="$DATA_DIR/functional/local/" \
  data.pdb_data_dir="$PDB_DATA_DIR" \
  data.target_field=binding_site \
  +data.proteinshake_root="/home/user/.cache/proteinshake" \
  +tokenizer_kwargs.h5_path="$H5_EMBED" \
  +tokenizer_kwargs.embeddings_dataset="/vq_embe_proteinshake"
```

### Validate
```bash
python -m src.script.run_supervised_task \
  --config-name proteinshake_binding_site \
  trainer.devices=1 \
  trainer.default_root_dir="$LOG_DIR/proteinshake_runs" \
  run_name="bindshake_val" \
  validate_only=true \
  tokenizer=WrappedMyRepShakeTokenizer \
  data.pdb_data_dir="$PDB_DATA_DIR" \
  +tokenizer_kwargs.h5_path="$H5_EMBED"
```

---

## Task 2: Sensitivity (Conformational Switch)

### Test Only (Zero-Shot)
```bash
cd StructTokenBench

python -m src.script.run_supervised_task \
  --config-name conformational_switch \
  trainer.devices=1 \
  tokenizer=WrappedMyRepTokenizer \
  tokenizer_device=cuda \
  precompute_tokens=false \
  data.pdb_data_dir="$PDB_DATA_DIR" \
  data.use_continuous=true \
  test_only=true \
  trainer.default_root_dir="$LOG_DIR/sensitivity_runs" \
  run_name="sensitivity_test" \
  +tokenizer_kwargs.h5_path="$H5_EMBED"
```

---

## Available H5 Files

For **binding site task**:
- `vq_embed_proteinshake.h5` (128-dim, ProteinShake-specific)
- Or use general: `vq_embed.h5` (128-dim)

For **sensitivity task**:
- `vq_embed.h5` (128-dim, default)
- `vq_embed_remote_homology_train_tst_valid.h5` (128-dim)

---

## Troubleshooting Quick Fixes

### Empty Batch Error
```
ValueError: not enough values to unpack (expected 3, got 0)
```
**Fix**: H5 file doesn't have your PDB/chains. Check:
```python
import h5py
with h5py.File("vq_embed.h5", "r") as h5:
    print("Keys:", list(h5.keys())[:10])
```

### Dimension Mismatch
```
Shape mismatch: expected 256, got 128
```
**Fix**: Remove `model.d_model=256` override; auto-sync handles it.

### Missing Embeddings
```
[ConformationalSwitch] Failed to load H5 embeddings for 1abc/A
```
**Fix**: This PDB/chain not in H5; dataset will skip it. Either:
- Use a different H5 file, or
- Build your own H5 with these PDBs

### Module Not Found
```
ModuleNotFoundError: No module named 'src.tokenizers'
```
**Fix**: Already handled! Transparently maps to `src.stb_tokenizers`.

---

## Check Your Setup

```bash
# Verify H5 exists
ls -lh StructTokenBench/src/vq_embed*.h5

# Verify mmCIF files
ls -lh $PDB_DATA_DIR | head

# Verify data directory
ls -lh $DATA_DIR/functional/local/proteinshake_bindingsite/out_ps/

# Test H5 content
python3 << 'EOF'
import h5py
with h5py.File("StructTokenBench/src/vq_embed.h5", "r") as h5:
    print(f"Keys in H5: {len(list(h5.keys()))} total")
    for k in list(h5.keys())[:5]:
        print(f"  {k}: {h5[k].shape}")
EOF
```

---

## Full Config Example

Create a file `my_config.yaml`:
```yaml
defaults:
  - conformational_switch

trainer:
  devices: 1
  max_epochs: 10
  default_root_dir: "./logs"

tokenizer: WrappedMyRepTokenizer

data:
  pdb_data_dir: "/home/user/pdb_data/mmcif_files"
  use_continuous: true

model:
  d_model: null  # Auto-detected from tokenizer

tokenizer_kwargs:
  h5_path: "/home/user/StructTokenBench/src/vq_embed.h5"
```

Then run:
```bash
python -m src.script.run_supervised_task --config-path . --config-name my_config
```

---

## Documentation Files

- **`ERROR_SOLUTIONS.md`** - Detailed error explanations with fixes
- **`SENSITIVITY_TASK_GUIDE.md`** - Full sensitivity task guide
- **`CHANGES_SUMMARY.md`** - All modifications explained
- **`QUICK_START.md`** - This file (copy-paste commands)
