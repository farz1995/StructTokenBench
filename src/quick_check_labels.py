# quick_check_labels.py
import pandas as pd, sys
if len(sys.argv) < 2:
    print("Usage: python quick_check_labels.py <one or more CSV paths>")
    raise SystemExit(1)

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)
u = sorted(df["fold_label"].unique())
print("num_labels =", len(u), "min =", int(min(u)), "max =", int(max(u)))
