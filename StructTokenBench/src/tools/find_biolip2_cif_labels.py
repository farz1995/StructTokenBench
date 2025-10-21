import os
import sys
import csv
import glob
import argparse

# Utility to find CIF paths for BioLIP2 TSV label rows and emit a mapping CSV.
# It scans binding/catalytic label TSVs and resolves CIF files under a pdb_data directory.


def default_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    data_dir = os.path.join(
        repo,
        "struct_token_bench_release_data",
        "data",
        "functional",
        "local",
        "biolip2",
        "out",
    )
    pdb_dir = os.path.join(repo, "pdb_data")
    return data_dir, pdb_dir


def find_cif(pdb_id: str, pdb_dir: str) -> str | None:
    # Support both pdb_data/mmcif_files/<id>.cif and pdb_data/mmcif_files/mmcif_files/<id>.cif
    # Try lower and upper case filenames.
    candidates = []
    for base in (
        os.path.join(pdb_dir, "mmcif_files"),
        os.path.join(pdb_dir, "mmcif_files", "mmcif_files"),
    ):
        for name in (f"{pdb_id}.cif", f"{pdb_id.upper()}.cif", f"{pdb_id.lower()}.cif"):
            candidates.append(os.path.join(base, name))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def collect_tsvs(data_dir: str) -> list[str]:
    patterns = [
        os.path.join(data_dir, "binding", "*", "processed_structured_*_pdb_ligand_label.tsv"),
        os.path.join(data_dir, "catalytic", "*", "processed_structured_*_pdb_ligand_label.tsv"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)


def parse_args():
    dflt_data, dflt_pdb = default_paths()
    ap = argparse.ArgumentParser(description="Map BioLIP2 CIF paths to labels from TSVs")
    ap.add_argument("--data_dir", default=dflt_data, help="BioLIP2 out dir (binding/catalytic subdirs inside)")
    ap.add_argument("--pdb_dir", default=dflt_pdb, help="pdb_data directory containing mmcif_files/")
    ap.add_argument("--out_csv", default="biolip2_cif_label_map.csv", help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows (0 = no limit)")
    return ap.parse_args()


def main():
    args = parse_args()
    tsv_files = collect_tsvs(args.data_dir)
    if not tsv_files:
        print(f"No TSV files found under {args.data_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    seen = set()
    n_out = 0
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        # header
        w.writerow(["pdb_id", "chain_id", "label", "tsv", "cif_path", "exists"])
        for tsv in tsv_files:
            try:
                with open(tsv, "r", encoding="utf-8") as fr:
                    header = fr.readline().strip().split("\t")
                    # Expect columns: pdb_id, ligand_chain, label
                    # Fall back to flexible index
                    try:
                        idx_pdb = header.index("pdb_id")
                    except ValueError:
                        idx_pdb = 0
                    try:
                        idx_chain = header.index("ligand_chain")
                    except ValueError:
                        idx_chain = 1
                    try:
                        idx_label = header.index("label")
                    except ValueError:
                        idx_label = 2

                    for line in fr:
                        if not line.strip():
                            continue
                        cols = line.rstrip("\n").split("\t")
                        if len(cols) < 3:
                            continue
                        pdb_id = cols[idx_pdb].strip()
                        chain = cols[idx_chain].strip()
                        label = cols[idx_label].strip()
                        key = (pdb_id, chain, label)
                        if key in seen:
                            continue
                        seen.add(key)

                        cif = find_cif(pdb_id, args.pdb_dir)
                        exists = bool(cif and os.path.isfile(cif))
                        w.writerow([pdb_id, chain, label, os.path.relpath(tsv, start=args.data_dir), cif or "", int(exists)])
                        n_out += 1
                        if args.limit and n_out >= args.limit:
                            break
            except Exception as e:
                print(f"Failed reading {tsv}: {e}")
            if args.limit and n_out >= args.limit:
                break

    print(f"Wrote {n_out} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


# python src/tools/find_biolip2_cif_labels.py --out_csv biolip2_cif_label_map.csv