"""DSRL training-placement success vs env steps: fp32 SFT checkpoints vs earlier bf16 / step-0 runs (lamp)."""
import json, re, glob, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
LOGS = Path("/network/scratch/c/cloutiec/lerobot/slurm_logs")
OUT = Path(sys.argv[1])
def from_log(f):
    ev = {}
    for m in re.finditer(r"\[eval @ step (\d+)\].*?success_rate_train=([0-9.]+)", open(f).read()):
        ev[round(int(m.group(1)) / 25000)] = float(m.group(2))  # later (resumed) evals overwrite
    return [ev.get(i, np.nan) for i in range(1, 21)]
runs = {}
for st in (500, 1000, 4000, 8000):
    c = [from_log(f) for f in sorted(glob.glob(str(LOGS / f"dsrl_fp32sft{st}_lamp_s*-*.out")))]
    if c: runs[f"fp32 Expert-only {st//1000 if st>=1000 else st}{'k' if st>=1000 else ''}"] = c
runs["step 0 (base π0.5)"] = [from_log(f) for f in sorted(glob.glob(str(LOGS / "dsrl_step0_lamp_s?-*.out")))]
D = json.load(open("/network/scratch/c/cloutiec/dexrobot/results_export/dsrl.json"))["results"]["lamp"]
for arm, nm in (("arm0", "bf16 Expert-only"), ("arm1", "bf16 Full-FT")):
    for st in ("2000", "4000"):
        runs[f"{nm} {int(st)//1000}k"] = [v["success_rate"]["training_envs"] for v in D[arm]["checkpoints"][st]["seeds"].values()]
COL = {"fp32 Expert-only 500": "#86b6ef", "fp32 Expert-only 1k": "#2a78d6", "fp32 Expert-only 4k": "#184f95",
       "fp32 Expert-only 8k": "#0d366b", "bf16 Expert-only 2k": "#eb6834", "bf16 Expert-only 4k": "#a8461f",
       "bf16 Full-FT 2k": "#1baf7a", "bf16 Full-FT 4k": "#0f7552", "step 0 (base π0.5)": "#55544f"}
x = np.arange(1, 21) * 25
groups = [("fp32 Expert-only (new)", [k for k in runs if k.startswith("fp32")]),
          ("Earlier runs (bf16 SFT, step 0)", [k for k in runs if not k.startswith("fp32")])]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6b6a65",
                     "xtick.color": "#6b6a65", "ytick.color": "#6b6a65"})
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, (title, ks) in zip(axs, groups):
    for k in ks:
        a = np.array(runs[k], float); m = np.nanmean(a, 0); n = (~np.isnan(a)).sum(0)
        ax.fill_between(x, np.nanmin(a, 0), np.nanmax(a, 0), color=COL[k], alpha=0.12, lw=0)
        ls = "--" if k.startswith("step 0") else "-"
        ax.plot(x, m, color=COL[k], lw=2, ls=ls, label=f"{k}  (n={a.shape[0]})")
    ax.set_title(title, loc="left", fontsize=10); ax.grid(True, color="#e6e5e0"); ax.set_xlabel("DSRL env steps (k)")
    ax.set_ylim(-0.03, 1.03); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
axs[0].set_ylabel("training-placement success\n(mean over seeds; band = min–max)")
fig.tight_layout(); fig.savefig(OUT / "dsrl_curves.png", dpi=130)
print({k: [round(float(np.nanmean(np.array(v, float)[:, i])), 2) for i in (2, 3, 4, 5, 7, 19)] for k, v in runs.items()})
