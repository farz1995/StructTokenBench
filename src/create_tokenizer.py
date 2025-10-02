import h5py, csv

h5_path = "vq_embed_remote_homology_train_tst_valid.h5"
csv_out = "h5_keys.csv"

def walk(hobj):
    """Yield rows of (kind, path, shape, dtype, size, n_attrs)."""
    for name, obj in hobj.items():
        if isinstance(obj, h5py.Group):
            yield ("group", obj.name, "", "", "", len(obj.attrs))
            # Recurse into the group
            yield from walk(obj)
        elif isinstance(obj, h5py.Dataset):
            shape = tuple(obj.shape) if hasattr(obj, "shape") else ""
            dtype = str(obj.dtype) if hasattr(obj, "dtype") else ""
            size  = getattr(obj, "size", "")
            yield ("dataset", obj.name, shape, dtype, size, len(obj.attrs))

with h5py.File(h5_path, "r") as f, open(csv_out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["kind", "path", "shape", "dtype", "size", "num_attrs"])
    # top-level groups/datasets printed like your snippet
    print(list(f.keys()))
    for row in walk(f):
        w.writerow(row)

print(f"Wrote: {csv_out}")
