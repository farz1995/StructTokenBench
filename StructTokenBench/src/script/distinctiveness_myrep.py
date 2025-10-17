import os, numpy as np, torch, matplotlib.pyplot as plt

sim_path = "/home/fe5vb/project/PST/StructTokenBench/tmp_simscore_dist/simscore_cos_stb_tokenizers.wrapped_myrep.myrep_casp14"
out_dir  = "/home/fe5vb/project/PST/StructTokenBench/figures"
os.makedirs(out_dir, exist_ok=True)

S = torch.load(sim_path, map_location="cpu").float().cpu().numpy()
K = S.shape[0]
np.fill_diagonal(S, np.nan)

# Off-diagonal distribution
off = S[~np.eye(K, dtype=bool)]

plt.figure(figsize=(6,4))
plt.hist(off, bins=100, density=True)
plt.xlabel("Cosine similarity (off-diagonal)"); plt.ylabel("Density")
plt.title("MyRep codebook distinctiveness")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "myrep_offdiag_hist.png"), dpi=200); plt.close()

# Nearest-neighbor CDF
S_no_diag = np.where(np.eye(K, dtype=bool), -np.inf, S)
nn = np.max(S_no_diag, axis=1)
x = np.sort(nn); y = np.arange(1, len(x)+1) / len(x)

plt.figure(figsize=(6,4))
plt.plot(x, y, label="MyRep")
plt.xlabel("Nearest-neighbor cosine"); plt.ylabel("CDF"); plt.legend()
plt.title("MyRep NN cosine CDF")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "myrep_nn_cdf.png"), dpi=200); plt.close()

print("Saved:",
      os.path.join(out_dir, "myrep_offdiag_hist.png"),
      os.path.join(out_dir, "myrep_nn_cdf.png"))
