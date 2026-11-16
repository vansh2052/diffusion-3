"""Sample grid: one row per (run, sampler), the same starting noise in every column.

  python grid.py out.png run1:sampler run2:sampler ...
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("GRID_RUNS", "micro_runs"))
out, items = sys.argv[1], sys.argv[2:]
cols = 12
fig, ax = plt.subplots(len(items), cols, figsize=(cols * 1.1, len(items) * 1.25))
ax = np.atleast_2d(ax)
for r, it in enumerate(items):
    run, smp = it.split(":")
    imgs = np.load(os.path.join(RUNS, run, f"samples_{smp}.npy"))
    res = json.load(open(os.path.join(RUNS, run, "eval.json")))[smp]
    fid = res.get("kid75", res.get("fid"))
    for c in range(cols):
        ax[r, c].imshow(imgs[c]); ax[r, c].axis("off")
    ax[r, 0].set_title(f"{run} ({smp})  {'KIDx1000' if 'kid75' in res else 'FID'} {fid:.1f}", loc="left", fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=100)
print("saved", out)
