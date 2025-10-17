import argparse
import os
import sys
import numpy as np
import torch

# Ensure we can import src.* when running from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.stb_tokenizers import WrappedMyRepTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Run WrappedMyRepTokenizer for distinctiveness/VQ modes")
    p.add_argument("--h5", required=False, default=os.environ.get("MYREP_H5"), help="Path to per-chain features H5 (or set MYREP_H5)")
    p.add_argument("--pdb", required=True, help="Path to mmCIF/PDB file for a structure (used for residue index mapping)")
    p.add_argument("--chain", required=False, default="", help="Chain ID (e.g., A)")

    p.add_argument("--mode", default="distinctiveness", choices=["identity", "vq_embed", "vq_index", "distinctiveness"], help="Feature output mode")
    p.add_argument("--metric", default="cosine", choices=["cosine", "l2"], help="VQ matching metric")

    p.add_argument("--vq", default=None, help="Path to vq_embed.h5 (codebook). Required for vq_* and distinctiveness modes")
    p.add_argument("--use-seq", action="store_true", help="Return dummy per-residue sequence placeholders ('X')")

    p.add_argument("--save-out", default=None, help="Optional path to save features as .npy")
    p.add_argument("--print-first", type=int, default=5, help="Preview first N rows of output")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.h5:
        raise SystemExit("--h5 is required (or set MYREP_H5)")
    if args.mode in {"vq_embed", "vq_index", "distinctiveness"} and not args.vq:
        raise SystemExit("--vq (path to vq_embed.h5) is required for this mode")

    tok = WrappedMyRepTokenizer(
        h5_path=args.h5,
        device="cpu",
        tokenizer_pretrained_ckpt_path=args.vq,
        feature_mode=args.mode,
        vq_metric=args.metric,
    )

    feats, resid, seqs = tok.encode_structure(args.pdb, args.chain, use_sequence=args.use_seq)

    # Summaries
    print(f"Mode: {args.mode} | Metric: {args.metric}")
    print(f"Features: shape={tuple(feats.shape)} dtype={feats.dtype}")
    print(f"Residue index: n={len(resid)} | min={int(np.min(resid)) if len(resid) else 'NA'} | max={int(np.max(resid)) if len(resid) else 'NA'}")
    if args.use_seq:
        print(f"Seqs: n={len(seqs)} | example={seqs[:min(len(seqs), args.print_first)]}")

    # Preview a few rows
    n = min(args.print_first, feats.shape[0])
    with torch.no_grad():
        arr = feats[:n].cpu().numpy()
    print(f"Preview first {n} rows (truncated):")
    # Print compactly
    np.set_printoptions(precision=4, suppress=True)
    print(arr)

    if args.save_out:
        out_path = os.path.abspath(args.save_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, feats.detach().cpu().numpy())
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

