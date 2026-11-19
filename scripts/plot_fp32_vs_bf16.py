"""fm_hand / action-diversity vs SFT step (fp32 vs bf16 vs base) and decoded-chunk plots. CPU only."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/network/scratch/c/cloutiec/dexrobot/lerobot")
OUT = R / "results_fp32/plots"
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"
SER = {"fp32 Expert-only": "#2a78d6", "bf16 Expert-only": "#eb6834", "bf16 Full-FT": "#1baf7a"}
MK = {"fp32 Expert-only": "o", "bf16 Expert-only": "s", "bf16 Full-FT": "^"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

bf = pd.read_csv(R / "results/offline_steerability_metrics.csv"); bf = bf[bf.task == "ScrewLightbulb"]
fp = pd.read_csv(R / "results_fp32/offline_steerability_metrics.csv")
base = pd.read_csv(R / "results_step0/offline_steerability_metrics.csv").query("task == 'ScrewLightbulb'").iloc[0]
series = {"fp32 Expert-only": fp[fp.recipe == "EXPERT-ONLY"], "bf16 Expert-only": bf[bf.recipe == "EXPERT-ONLY"],
          "bf16 Full-FT": bf[bf.recipe == "FULL-FT"]}
panels = [("fm_hand", "fm_hand (flow-matching loss, hand)"), ("fm_arm", "fm_arm (flow-matching loss, arm)"),
          ("trace_total", "trace_total (action spread)"), ("trace_hand", "trace_hand (hand action spread)"),
          ("erank_total", "erank_total (effective rank)"), ("lipschitz_local", "lipschitz_local (noise→action gain)")]

fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (m, title) in zip(axs.flat, panels):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(base[m], color=MUTED, ls="--", lw=1.5)
    ax.annotate("base π0.5 (step 0)", (1.0, base[m]), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                color=MUTED, fontsize=8)
    for name, d in series.items():
        d = d.sort_values("step")
        ax.plot(d.step, d[m], color=SER[name], lw=2, marker=MK[name], ms=6, mec="white", mew=1.5, label=name)
    ax.set_xscale("log"); ax.set_xticks([500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["500", "1k", "2k", "4k", "8k", "16k"]); ax.minorticks_off()
    ax.set_title(title, color=INK, fontsize=10, loc="left"); ax.set_xlabel("SFT step")
    if m.startswith("fm_"): ax.set_yscale("log")
axs[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("Lamp (ScrewLightbulb): offline metrics vs SFT step, fp32-master vs bf16 SFT", x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "metrics_vs_step.png", dpi=130); plt.close(fig)

# ---- decoded chunks
D = R / "results_fp32/decoded"
order = ["base_step0"] + [f"fp32_EXPERT-ONLY_{s}" for s in (500, 1000, 2000, 4000, 8000)] + \
        [f"bf16_EXPERT-ONLY_{s}" for s in (2000, 4000, 8000)] + [f"bf16_FULL-FT_{s}" for s in (2000, 4000, 8000)]
Z = {k: np.load(D / f"lamp_{k}.npz") for k in order if (D / f"lamp_{k}.npz").exists()}
names = list(Z)
def nice(k):
    return "base π0.5" if k == "base_step0" else k.replace("_EXPERT-ONLY_", " Expert-only ").replace("_FULL-FT_", " Full-FT ")
def col(k):
    return MUTED if k == "base_step0" else SER["fp32 Expert-only" if k.startswith("fp32") else
                                               "bf16 Full-FT" if "FULL" in k else "bf16 Expert-only"]

# spaghetti: obs 0, 64 noises; arm dim + 3 hand dims with largest demo-relative spread in base
A0 = {k: Z[k]["actions"][0] for k in names}; demo = Z[names[0]]["demo"][0]
v = A0["base_step0"].std(0).mean(0) if "base_step0" in A0 else A0[names[0]].std(0).mean(0)
dims = [int(np.argmax(v[:6]))] + list(6 + np.argsort(v[6:])[::-1][:3])
fig, axs = plt.subplots(len(names), len(dims), figsize=(3.2 * len(dims), 1.55 * len(names)), sharex=True, sharey="col")
for i, k in enumerate(names):
    for j, d in enumerate(dims):
        ax = axs[i, j]
        ax.plot(A0[k][:, :, d].T, color=col(k), lw=0.6, alpha=0.35)
        ax.plot(demo[:, d], color=INK, lw=1.8, ls="--")
        ax.grid(True, color=GRID, lw=0.6)
        if i == 0: ax.set_title(("arm" if d < 6 else "hand") + f" dim {d}", color=INK, fontsize=10)
        if j == 0: ax.set_ylabel(nice(k), rotation=0, ha="right", va="center", fontsize=9, color=INK)
for ax in axs[-1]: ax.set_xlabel("chunk step")
fig.suptitle(f"Decoded chunks for 64 DSRL noises, one demo observation (episode {Z[names[0]]['episodes'][0]}); "
             "dashed = demo action", x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "decoded_chunks_obs0.png", dpi=110); plt.close(fig)

# per-timestep spread (std over the 64 noises, averaged over obs and dims), arm vs hand
# one single-hue ramp per family, light -> dark with SFT step; base = dark gray dashed
from matplotlib.colors import to_rgb
def ramp(hex_, n):
    """n steps of one hue: tint (earliest step) -> base colour -> shade (latest step)."""
    c = np.array(to_rgb(hex_))
    return [tuple(c + (1 - c) * 0.5 * -t if t < 0 else c * (1 - 0.45 * t)) for t in np.linspace(-1, 1, n)]
fams = {"fp32 Expert-only": [k for k in names if k.startswith("fp32")],
        "bf16 Expert-only": [k for k in names if k.startswith("bf16_EXPERT")],
        "bf16 Full-FT": [k for k in names if k.startswith("bf16_FULL")]}
LC = {"base_step0": "#3a3a38"}
for fam, ks in fams.items():
    for k, c in zip(ks, ramp(SER[fam], len(ks))): LC[k] = c
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6))
for ax, (part, sl) in zip(axs, (("arm", slice(0, 6)), ("hand", slice(6, 22)))):
    for fam, ks in [("base", ["base_step0"])] + list(fams.items()):
        for k in ks:
            if k not in Z: continue
            s = Z[k]["actions"][..., sl].std(1).mean((0, 2))
            ax.plot(s, color=LC[k], lw=2.2 if k == "base_step0" else 2, ls="--" if k == "base_step0" else "-",
                    label=nice(k))
    ax.set_title(f"{part}: std across 64 noises per chunk step (mean over 4 obs, dims)", fontsize=10, loc="left", color=INK)
    ax.grid(True, color=GRID, lw=0.8); ax.set_xlabel("chunk step"); ax.set_ylim(bottom=0)
axs[0].set_ylabel("action std (normalized units)")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="center left", bbox_to_anchor=(0.80, 0.5), frameon=False, fontsize=9,
           title="darker = later SFT step", title_fontsize=9)
fig.tight_layout(rect=(0, 0, 0.80, 1)); fig.savefig(OUT / "spread_per_timestep.png", dpi=130); plt.close(fig)

# PCA of flattened chunks on a common basis, per obs
fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
for o, ax in enumerate(axs):
    X = {k: Z[k]["actions"][o].reshape(64, -1) for k in names}
    allX = np.concatenate(list(X.values())); mu = allX.mean(0)
    _, _, Vt = np.linalg.svd(allX - mu, full_matrices=False)
    for k in names:
        if k.startswith("bf16_FULL"): continue
        p = (X[k] - mu) @ Vt[:2].T
        ax.scatter(p[:, 0], p[:, 1], s=10, color=col(k), alpha=0.5, edgecolors="none", label=nice(k))
    dp = (Z[names[0]]["demo"][o].reshape(-1) - mu) @ Vt[:2].T
    ax.scatter(*dp, marker="*", s=180, color=INK, edgecolors="white", label="demo", zorder=5)
    ax.set_title(f"obs {o} (episode {Z[names[0]]['episodes'][o]})", fontsize=10, color=INK); ax.grid(True, color=GRID)
axs[0].legend(frameon=False, fontsize=7, markerscale=1.5)
fig.suptitle("Executed chunks (64 noises each) in a shared 2D PCA per observation — Full-FT omitted for clarity",
             x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "pca_chunks.png", dpi=120); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")), "| decoded:", names)
