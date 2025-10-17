#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot codebook "distinctiveness / pairwise similarity" like Fig. 3.

It loads cosine similarity matrices saved by the distinctiveness task, e.g.:
  tmp_simscore_dist/simscore_cos_<NAME>_casp14
  tmp_simscore_used_dist/simscore_cos_<NAME>_casp14

This version:
- does NOT require a suffix match; scans simscore_cos_* in the dirs you pass
- robustly loads tensors under PyTorch>=2.6 (weights_only fallback)
- maps a few known labels and keeps unknown ones
- produces violin of off-diagonal cos and CDF of nearest-neighbor cos
"""

import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional pretty names (unknowns will keep their raw label)
NAME_MAP = {
    "esm3": "ESM3",
    "foldseek": "FoldSeek",
    "protokens": "ProTokens",
    "ourpretrained_VanillaVQ": "VanillaVQ",
    "ourpretrained_AminoAseed": "AminoAseed",
    "stb_tokenizers.wrapped_myrep.myrep": "MyRep",
}

KNOWN_SUFFIXES = ("_casp14", "_cameo")  # stripped if present at the end


def parse_args():
    p = argparse.ArgumentParser(description="Plot codebook diversity (pairwise cosine).")
    p.add_argument("--simscore_dir", required=True, type=str,
                   help="Directory of cosine similarity matrices (unweighted).")
    p.add_argument("--used_simscore_dir", default=None, type=str,
                   help="Directory of cosine similarity matrices (usage-weighted).")
    p.add_argument("--out_dir", required=True, type=str,
                   help="Where to save figures.")
    p.add_argument("--dpi", default=200, type=int, help="Figure DPI.")
    return p.parse_args()


def debug_list_dir(path):
    try:
        print(f"[DEBUG] Listing {path}")
        for f in sorted(os.listdir(path)):
            print("       ", f)
    except Exception as e:
        print(f"[DEBUG] Could not list {path}: {e}")


def robust_load_tensor(path):
    import torch
    try:
        obj = torch.load(path, map_location="cpu")  # PyTorch>=2.6 default safe mode
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    # Accept a plain tensor
    if isinstance(obj, torch.Tensor):
        return obj

    # Accept a tuple/list where the first element is the cosine matrix
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        if isinstance(obj[0], torch.Tensor):
            return obj[0]

    # Accept dicts that hold the tensor under common keys
    if isinstance(obj, dict):
        for k in ("tensor", "S", "sim", "similarity", "data"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]

    raise TypeError(f"Unsupported object saved in {path}: {type(obj)}")



def label_from_filename(path):
    base = os.path.basename(path)
    # Expect "simscore_cos_<NAME>[optional suffix]"
    if not base.startswith("simscore_cos_"):
        return base
    label = base[len("simscore_cos_"):]
    # strip known suffixes if present
    for suf in KNOWN_SUFFIXES:
        if label.endswith(suf):
            label = label[: -len(suf)]
            break
    return NAME_MAP.get(label, label)


def load_dir(sim_dir):
    if not os.path.isdir(sim_dir):
        print(f"[WARN] Directory not found: {sim_dir}")
        return {}

    pattern = os.path.join(sim_dir, "simscore_cos_*")  # no suffix requirement
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No files matched pattern: {pattern}")
        debug_list_dir(sim_dir)
        return {}

    out = {}
    for f in files:
        try:
            S = robust_load_tensor(f).float().cpu().numpy()
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
            continue

        label = label_from_filename(f)

        if S.ndim == 2 and S.shape[0] == S.shape[1]:
            # --- full cosine matrix ---
            K = S.shape[0]
            np.fill_diagonal(S, np.nan)
            off = S[~np.eye(K, dtype=bool)].astype(np.float32)
            S_no_diag = np.where(np.eye(K, dtype=bool), -np.inf, S)
            nn = np.max(S_no_diag, axis=1).astype(np.float32)
            out[label] = (off, nn)

        elif S.ndim == 1:
            # --- weighted sample / per-code stat vector ---
            # Use it as the 'off-diagonal' sample for violin;
            # set nn=None so CDF step can skip it gracefully.
            off = S.astype(np.float32)
            out[label] = (off, None)

        else:
            print(f"[WARN] Skipping unsupported shape: {f} shape={S.shape}")
            continue


    return out


def violin_offdiag(models, out_png, dpi):
    if not models:
        print("[INFO] No models to plot (off-diagonal).")
        return
    labels = list(models.keys())
    data = [models[k][0] for k in labels]  # off-diag arrays
    med = [np.nanmedian(d) for d in data]
    order = np.argsort(med)
    labels = [labels[i] for i in order]
    data = [data[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cosine similarity (off-diagonal)")
    ax.set_title("Codebook distinctiveness: off-diagonal cosine distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("[SAVED]", out_png)


def cdf_nn(models, out_png, dpi):
    models = {k: v for k, v in models.items() if v[1] is not None}
    if not models:
        print("[INFO] No models with NN vectors to plot (NN-CDF).")
        return
    labels = list(models.keys())
    med = [np.nanmedian(models[k][1]) for k in labels]
    order = np.argsort(med)
    labels = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for lab in labels:
        nn = np.asarray(models[lab][1], dtype=np.float32)
        nn = nn[~np.isnan(nn)]
        x = np.sort(nn)
        y = np.arange(1, len(x) + 1) / max(1, len(x))
        ax.plot(x, y, label=lab)
    ax.set_xlabel("Nearest-neighbor cosine similarity")
    ax.set_ylabel("CDF")
    ax.set_title("Nearest-neighbor cosine CDF")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("[SAVED]", out_png)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] simscore_dir      :", args.simscore_dir)
    print("[INFO] used_simscore_dir :", args.used_simscore_dir)
    print("[INFO] out_dir           :", args.out_dir)

    models = load_dir(args.simscore_dir)
    if not models:
        print(f"[WARN] No usable unweighted cosine files in {args.simscore_dir}")
    else:
        violin_offdiag(models, os.path.join(args.out_dir, "offdiag_violin.png"), dpi=args.dpi)
        cdf_nn(models, os.path.join(args.out_dir, "nn_cosine_cdf.png"), dpi=args.dpi)

    if args.used_simscore_dir:
        used_models = load_dir(args.used_simscore_dir)
        if used_models:
            violin_offdiag(used_models, os.path.join(args.out_dir, "offdiag_violin_used.png"), dpi=args.dpi)
            cdf_nn(used_models, os.path.join(args.out_dir, "nn_cosine_cdf_used.png"), dpi=args.dpi)


if __name__ == "__main__":
    main()
