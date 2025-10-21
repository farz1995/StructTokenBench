# Running Sensitivity (Conformational) Tasks with Continuous Tokenizers

## Overview
The sensitivity task (conformational switch detection) measures TM-Score between paired protein conformations. It **requires continuous representations** (not discrete tokens), which come from H5 files containing pre-computed embeddings.

## Available H5 Files

Located in `StructTokenBench/src/`:
- **`vq_embed.h5`**: Default continuous embeddings for general use (128-dim)
- **`vq_embed_proteinshake.h5`**: Specific to ProteinShake binding site representations
- **`vq_embed_casp_14.h5`**: CASP14 structure prediction task
- **`vq_embed_cameo.h5`**: CAMEO structure prediction task
- **`vq_embed_remote_homology_train_tst_valid.h5`**: Remote homology task

## Supported Tokenizers

### Continuous Tokenizers (for sensitivity task)
1. **`WrappedMyRepTokenizer`** (alias: `src.tokenizers.WrappedMyRepTokenizer`)
   - Reads float32 continuous embeddings from H5
   - Does NOT need a codebook (skips discretization)
   - `embed_dim=128` (default from vq_embed.h5)

2. **`WrappedMyRepShakeTokenizer`** (alias: `src.tokenizers.WrappedMyRepShakeTokenizer`)
   - Similar to MyRep but tailored for ProteinShake embeddings
   - Uses `vq_embed_proteinshake.h5`

### Discrete Tokenizers (NOT suitable for sensitivity)
- `WrappedESM3Tokenizer`
- `WrappedFoldSeekTokenizer`
- `WrappedProteinMPNNTokenizer`
- etc.

These will fail or produce poor results on conformational task because they don't use H5 embeddings.

## Command Template

```bash
cd StructTokenBench

python -m src.script.run_supervised_task \
  --config-name conformational_switch \
  trainer.devices=1 \
  tokenizer=WrappedMyRepTokenizer \
  tokenizer_device=cuda \
  precompute_tokens=false \
  +tokenizer_kwargs.h5_path="/absolute/path/to/vq_embed.h5" \
  data.pdb_data_dir="/absolute/path/to/pdb_data/mmcif_files/" \
  data.use_continuous=true \
  test_only=true \
  trainer.default_root_dir="struct_token_bench_logs/sensitivity_runs" \
  run_name="my_sensitivity_run"
```

## Common Issues & Solutions

### Issue 1: "ValueError: not enough values to unpack (expected 3, got 0)"
**Cause**: All samples were skipped because H5 embeddings not found.

**Solution**: 
1. Check H5 file path is absolute and exists
2. Verify `+tokenizer_kwargs.h5_path` is set correctly
3. Look for debug logs like `[ConformationalSwitch] Failed to load H5 embeddings for ...`
4. The H5 may not contain embeddings for conformational task PDBs; try a different H5 or check available keys in H5 with:
   ```python
   import h5py
   with h5py.File("/path/to/vq_embed.h5", "r") as h5:
       print(list(h5.keys())[:20])  # Show first 20 keys
   ```

### Issue 2: "No H5 key found for pdb=..., chain=..."
**Cause**: The requested PDB/chain combination doesn't exist in the H5 file.

**Solution**:
- This is expected if the H5 was built from a different protein subset
- The dataset will skip that sample (as designed)
- If ALL samples fail, use a different H5 file or build your own with your PDB set

### Issue 3: "KeyError: No H5 key found for..."
**Same as Issue 2** — the H5 doesn't have that PDB/chain. The code now logs it and skips.

### Issue 4: "embed_dim mismatch: expected 256, got 128"
**Cause**: You specified `model.d_model=256` but your tokenizer outputs 128-dim embeddings.

**Solution**:
- Either remove `model.d_model` override (auto-detection will set to 128)
- Or use an H5 with 256-dim embeddings
- The script automatically syncs `model.d_model` to tokenizer's `embed_dim`

### Issue 5: "codebook_embedding is empty"
**Cause**: Sensitivity task doesn't need a codebook, but the model tried to use one.

**Solution**: Already fixed! The code now detects continuous tokenizers and skips codebook initialization.

## Debugging Tips

1. **Enable debug logging**: Look for lines like:
   ```
   [ConformationalSwitch] Failed to load H5 embeddings for 3fcq/A: KeyError: ...
   ```
   These show which samples are being skipped.

2. **Check H5 structure**:
   ```python
   import h5py
   with h5py.File("/path/to/vq_embed.h5", "r") as h5:
       def print_structure(name, obj):
           print(name)
       h5.visititems(print_structure)
   ```

3. **Verify file paths are absolute**:
   - Don't use `~` or relative paths
   - Pass full `/home/user/project/...` paths

4. **Check model class**:
   - Default for conformational is `ZeroshotProximityModel` (from config)
   - This model computes embedding similarity scores, doesn't need codebook
   - Model setup now gracefully handles missing codebook for continuous tokenizers

## Example Run with Full Output

```bash
export CUDA_VISIBLE_DEVICES=0

python -m src.script.run_supervised_task \
  --config-name conformational_switch \
  trainer.devices=1 \
  tokenizer=WrappedMyRepTokenizer \
  data.pdb_data_dir="/home/user/project/pdb_data/mmcif_files/" \
  +tokenizer_kwargs.h5_path="/home/user/project/StructTokenBench/src/vq_embed.h5" \
  data.use_continuous=true \
  test_only=true \
  trainer.default_root_dir="/tmp/stb_sensitivity_logs" \
  run_name="test_continuous" 2>&1 | tee run.log
```

Then check logs for:
- `[run] Continuous tokenizer detected (H5 features).` ← Tokenizer recognized
- `[ConformationalSwitch] Failed to load H5 embeddings for ...` ← Sample skips
- Epoch progress and validation results

## Integration with Your Local Data

If you have your own PDB structures and want to use them:

1. **Prepare mmCIF files**: Place them in a directory, e.g., `/your/pdb_data/mmcif_files/`

2. **Set data path**:
   ```
   data.pdb_data_dir="/your/pdb_data/mmcif_files/"
   ```

3. **Choose or build H5 embeddings**:
   - Use one of the provided H5 files (if your PDBs are in them)
   - Or build your own with your tokenizer:
     ```python
     # Pseudocode: generate H5 from your PDB structures
     tokenizer = WrappedMyRepTokenizer(h5_path="...")
     # ... iterate your PDBs and save embeddings ...
     ```

4. **Run the task**:
   ```bash
   python -m src.script.run_supervised_task \
     --config-name conformational_switch \
     +tokenizer_kwargs.h5_path="/your/custom_embeddings.h5" \
     data.pdb_data_dir="/your/pdb_data/mmcif_files/" \
     ...
   ```

## Notes

- The conformational task measures **TM-Score** (global structure similarity)
- It's a **zero-shot test**—no training, just scoring pairs
- Uses embedding-based similarity (dot product or cosine in embedding space)
- Continuous embeddings preserve more structural information than discrete tokens
- The `ZeroshotProximityModel` doesn't train; it just scores pairs and computes correlation with true scores
