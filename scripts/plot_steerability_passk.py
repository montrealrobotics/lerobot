#!/usr/bin/env python
"""Plot pass@k curves from the ``--steer_eval`` summaries, and coverage against DSRL outcome.

pass@k uses the unbiased estimator ``1 - C(n-c, k) / C(n, k)`` for every k, so curve shape is
visible below saturation; group curves are the mean over starts of the per-start estimator, and
bands are 90% bootstrap intervals over rollouts. DSRL outcomes annotated on the figures are
transcribed from the run logs into LAMP_OUTCOMES / COFFEE_OUTCOMES below.

Writes passk_lightbulb, passk_coffee and coverage_vs_dsrl (PNG + PDF) to --out_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette, slots 1-3 (validated all-pairs in light mode) + text/grid tokens.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e1", "#ffffff"

LAMP_ARMS = {"cat=True": ["cat_true_3k", "cat_true_7k", "cat_true_16k"],
             "cat=False": ["cat_false_2k", "cat_false_8k", "cat_false_20k"]}  # fmt: skip
COFFEE_ARMS = {"cat=True": ["cat_true_4k", "cat_true_8k"], "cat=False": ["cat_false_2k", "cat_false_8k"]}
MODES = [
    (1, "DSRL noise", "one randn vector held per chunk"),
    (32, "iid noise", "fresh per step (vanilla pi05)"),
]

# Final fixed-start DSRL success per start kitchen, from the run logs. 3k: the original run (RL seed 0)
# plus reruns with RL seeds 1-3 (dsrl_lightbulb_seed_reruns, 200k steps); "solved" = success >= 0.5.
LAMP_OUTCOMES = {
    "cat_true_3k": {0: [0.00, 0.00, 0.00, 0.00], 1: [0.05, 1.00, 1.00, 1.00]},
    "cat_true_7k": {0: [1.00], 1: [1.00]},
    "cat_true_16k": {0: [1.00], 1: [1.00]},
    "cat_false_2k": {0: [1.00], 1: [1.00]},
    "cat_false_8k": {0: [1.00], 1: [1.00]},
    "cat_false_20k": {0: [1.00], 1: [0.00]},
}
# Kitchen-bank DSRL, matched final evals (20 episodes/kitchen): held-out = 2 kitchens, reference = seeds 0,1.
# cat_true_8k / cat_false_2k are step_470014 / step_480022 (runs cancelled at ~95% of 500k).
COFFEE_OUTCOMES = {
    "cat_true_4k": {"heldout": 0.825, "reference": 0.950},
    "cat_true_8k": {"heldout": 0.500, "reference": 0.975},
    "cat_false_2k": {"heldout": 0.650, "reference": 0.850},
    "cat_false_8k": {"heldout": 0.625, "reference": 0.900},
}


def pass_at_k(n: int, c: int, k: np.ndarray) -> np.ndarray:
    return np.array([1.0 if n - c < kk else 1.0 - comb(n - c, kk) / comb(n, kk) for kk in k])


def curve_with_band(starts: list[dict], rng: np.random.Generator, n_boot: int) -> tuple:
    """Mean-over-starts pass@k for k = 1..n_min, with a 90% bootstrap band over rollouts."""
    n = min(len(s["rollouts"]) for s in starts)
    k = np.arange(1, n + 1)
    outcomes = [np.array([r["success"] for r in s["rollouts"][:n]], dtype=bool) for s in starts]
    point = np.mean([pass_at_k(n, int(o.sum()), k) for o in outcomes], axis=0)
    boots = np.empty((n_boot, n))
    for b in range(n_boot):
        boots[b] = np.mean([pass_at_k(n, int(rng.choice(o, n).sum()), k) for o in outcomes], axis=0)
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    return k, point, lo, hi


def load(base: Path, arm: str, nc: int) -> dict:
    with open(base / arm / f"steerability_nc{nc}.json") as f:
        return json.load(f)


def style_axis(ax, kmax: int):
    ax.set_xscale("log", base=2)
    ticks = [2**i for i in range(int(np.log2(kmax)) + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.9, kmax * 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def plot_curve(ax, k, point, lo, hi, idx, label):
    marks = [i for i, kk in enumerate(k) if (kk & (kk - 1)) == 0]  # markers at powers of two only
    ax.fill_between(k, lo, hi, color=SLOT[idx], alpha=0.12, linewidth=0)
    ax.plot(
        k,
        point,
        color=SLOT[idx],
        linewidth=1.6,
        label=label,
        marker=MARKERS[idx],
        markevery=marks,
        markersize=5.5,
        markeredgecolor=SURFACE,
        markeredgewidth=0.8,
    )


def corner_note(ax, lines: list[str]):
    ax.text(
        0.97,
        0.04,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=TEXT_SECONDARY,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": SURFACE, "edgecolor": GRID, "linewidth": 0.6},
    )


def outcome_str(vals: list[float]) -> str:
    solved = sum(v >= 0.5 for v in vals)
    return f"{vals[0]:.2f}" if len(vals) == 1 else f"{solved}/{len(vals)} RL seeds"


def fig_lightbulb(base: Path, out: Path, rng, n_boot):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), sharex=True, sharey=True)
    for col_group, (cat, arms) in enumerate(LAMP_ARMS.items()):
        for m, (nc, mode_name, mode_desc) in enumerate(MODES):
            col = col_group * 2 + m
            data = {arm: load(base, arm, nc) for arm in arms}
            for row, seed in enumerate(("0", "1")):
                ax = axes[row, col]
                for i, arm in enumerate(arms):
                    k, p, lo, hi = curve_with_band([data[arm]["starts"][seed]], rng, n_boot)
                    plot_curve(ax, k, p, lo, hi, i, arm.split("_")[-1])
                style_axis(ax, 64)
                if nc == 1:
                    corner_note(
                        ax,
                        ["final DSRL success"]
                        + [f"{a.split('_')[-1]}: {outcome_str(LAMP_OUTCOMES[a][int(seed)])}" for a in arms],
                    )
                if row == 0:
                    ax.set_title(
                        f"{cat} · {mode_name}\n{mode_desc}", fontsize=9, color=TEXT_PRIMARY, loc="left"
                    )
                if col == 0:
                    ax.set_ylabel(f"start kitchen seed {seed}\npass@k", fontsize=9, color=TEXT_PRIMARY)
                if row == 1:
                    ax.set_xlabel("k (rollouts)", fontsize=9, color=TEXT_SECONDARY)
    for cat_idx, (cat, arms) in enumerate(LAMP_ARMS.items()):
        handles = [
            plt.Line2D(
                [],
                [],
                color=SLOT[i],
                marker=MARKERS[i],
                linewidth=1.6,
                markersize=5.5,
                markeredgecolor=SURFACE,
                label=arms[i].split("_")[-1],
            )
            for i in range(len(arms))
        ]
        fig.legend(
            handles=handles,
            title=f"{cat} SFT checkpoint",
            loc="upper center",
            bbox_to_anchor=(0.30 + 0.43 * cat_idx, 1.02),
            ncol=3,
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )
    fig.suptitle(
        "ScrewLightbulb · pass@k of the frozen SFT policy on the fixed-start DSRL starts "
        "(64 rollouts/start, 90% bootstrap band)",
        fontsize=10,
        color=TEXT_PRIMARY,
        y=1.08,
    )
    fig.tight_layout()
    save(fig, out / "passk_lightbulb")


def fig_coffee(base: Path, out: Path, rng, n_boot):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), sharex=True, sharey=True)
    for col_group, (cat, arms) in enumerate(COFFEE_ARMS.items()):
        for m, (nc, mode_name, mode_desc) in enumerate(MODES):
            col = col_group * 2 + m
            data = {arm: load(base, arm, nc) for arm in arms}
            for row, group in enumerate(("train", "heldout")):
                ax = axes[row, col]
                for i, arm in enumerate(arms):
                    starts = [s for s in data[arm]["starts"].values() if s["group"] == group]
                    k, p, lo, hi = curve_with_band(starts, rng, n_boot)
                    plot_curve(ax, k, p, lo, hi, i, arm.split("_")[-1])
                style_axis(ax, 32)
                if nc == 1:
                    key, name = (
                        ("reference", "reference kitchens") if group == "train" else ("heldout", "held-out")
                    )
                    corner_note(
                        ax,
                        [f"kitchen-bank DSRL, {name}"]
                        + [f"{a.split('_')[-1]}: {COFFEE_OUTCOMES[a][key]:.2f}" for a in arms],
                    )
                if row == 0:
                    ax.set_title(
                        f"{cat} · {mode_name}\n{mode_desc}", fontsize=9, color=TEXT_PRIMARY, loc="left"
                    )
                if col == 0:
                    label = "10 training kitchens" if group == "train" else "2 held-out kitchens"
                    ax.set_ylabel(f"{label}\nmean pass@k", fontsize=9, color=TEXT_PRIMARY)
                if row == 1:
                    ax.set_xlabel("k (rollouts)", fontsize=9, color=TEXT_SECONDARY)
    for cat_idx, (cat, arms) in enumerate(COFFEE_ARMS.items()):
        handles = [
            plt.Line2D(
                [],
                [],
                color=SLOT[i],
                marker=MARKERS[i],
                linewidth=1.6,
                markersize=5.5,
                markeredgecolor=SURFACE,
                label=arms[i].split("_")[-1],
            )
            for i in range(len(arms))
        ]
        fig.legend(
            handles=handles,
            title=f"{cat} SFT checkpoint",
            loc="upper center",
            bbox_to_anchor=(0.30 + 0.43 * cat_idx, 1.02),
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )
    fig.suptitle(
        "CoffeePressButton · pass@k of the frozen SFT policy on the kitchen bank "
        "(32 rollouts/kitchen, start-pose rejection, 90% bootstrap band)",
        fontsize=10,
        color=TEXT_PRIMARY,
        y=1.08,
    )
    fig.tight_layout()
    save(fig, out / "passk_coffee")


def fig_coverage(lamp_base: Path, coffee_base: Path, out: Path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) lamp: DSRL-noise pass@k for every (checkpoint, start), solved vs failed
    k = np.arange(1, 65)
    failed_labels = []
    for arms in LAMP_ARMS.values():
        for arm in arms:
            d = load(lamp_base, arm, 1)
            for seed in (0, 1):
                s = d["starts"][str(seed)]
                vals = LAMP_OUTCOMES[arm][seed]
                solved = np.mean(vals) >= 0.5
                p = pass_at_k(s["n"], s["successes"], k)
                idx = 0 if solved else 1
                ax_a.plot(
                    k,
                    p,
                    color=SLOT[idx],
                    linewidth=1.2 if solved else 2.2,
                    alpha=0.55 if solved else 1.0,
                    zorder=2 if solved else 3,
                )
                if not solved:
                    failed_labels.append((arm, seed, p, vals))
    for j, (arm, seed, p, vals) in enumerate(failed_labels):
        x = 16
        ckpt = arm.split("_")[-1]
        cat = "cat=True" if arm.startswith("cat_true") else "cat=False"
        result = (
            f"DSRL success {vals[0]:.2f}"
            if len(vals) == 1
            else f"DSRL solved in {sum(v >= 0.5 for v in vals)}/{len(vals)} RL seeds"
        )
        ax_a.annotate(
            f"{cat} {ckpt} · start {seed}\n{result}",
            xy=(x, p[x - 1]),
            xytext=(22, 0.34 - 0.2 * j),
            textcoords="data",
            fontsize=7.5,
            color=TEXT_PRIMARY,
            arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.7},
        )
    style_axis(ax_a, 64)
    ax_a.set_xlabel("k (rollouts)", fontsize=9, color=TEXT_SECONDARY)
    ax_a.set_ylabel("pass@k (DSRL noise)", fontsize=9, color=TEXT_PRIMARY)
    ax_a.set_title(
        "(a) ScrewLightbulb · every checkpoint × start kitchen (12 curves)",
        fontsize=9,
        color=TEXT_PRIMARY,
        loc="left",
    )
    ax_a.legend(
        handles=[
            plt.Line2D([], [], color=SLOT[0], linewidth=1.2, label="DSRL solved this start"),
            plt.Line2D([], [], color=SLOT[1], linewidth=2.2, label="DSRL failed this start"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    # (b) coffee: held-out DSRL success vs held-out DSRL-noise pass@8
    kk = 8
    for i, arms in enumerate(COFFEE_ARMS.values()):
        for arm in arms:
            starts = [s for s in load(coffee_base, arm, 1)["starts"].values() if s["group"] == "heldout"]
            x = float(np.mean([pass_at_k(s["n"], s["successes"], np.array([kk]))[0] for s in starts]))
            y = COFFEE_OUTCOMES[arm]["heldout"]
            ax_b.scatter(
                x, y, s=64, color=SLOT[i], marker=MARKERS[i], edgecolor=SURFACE, linewidth=1.2, zorder=3
            )
            ax_b.annotate(
                arm.split("_")[-1],
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
    ax_b.set_xlim(0.5, 1.0)
    ax_b.set_ylim(0.4, 0.9)
    ax_b.grid(True, color=GRID, linewidth=0.6)
    ax_b.set_axisbelow(True)
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_b.spines[side].set_color(GRID)
    ax_b.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax_b.set_xlabel(f"held-out pass@{kk} of frozen SFT (DSRL noise)", fontsize=9, color=TEXT_SECONDARY)
    ax_b.set_ylabel("held-out success after kitchen-bank DSRL", fontsize=9, color=TEXT_PRIMARY)
    ax_b.set_title(
        "(b) CoffeePressButton · 4 checkpoints, 2 held-out kitchens",
        fontsize=9,
        color=TEXT_PRIMARY,
        loc="left",
    )
    ax_b.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=SLOT[i],
                marker=MARKERS[i],
                linestyle="",
                markersize=7,
                markeredgecolor=SURFACE,
                label=f"{cat} SFT",
            )
            for i, cat in enumerate(COFFEE_ARMS)
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Coverage does not predict DSRL outcome", fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, out / "coverage_vs_dsrl")


def save(fig, stem: Path):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = Path(os.path.expanduser("~/scratch/lerobot/outputs"))
    parser.add_argument("--lamp_dir", default=str(root / "steerability" / "lightbulb"))
    parser.add_argument("--coffee_dir", default=str(root / "steerability" / "coffee_kitchenbank"))
    parser.add_argument("--out_dir", default=str(root / "figures" / "steerability"))
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "figure.facecolor": SURFACE})
    fig_lightbulb(Path(args.lamp_dir), out, rng, args.n_boot)
    fig_coffee(Path(args.coffee_dir), out, rng, args.n_boot)
    fig_coverage(Path(args.lamp_dir), Path(args.coffee_dir), out)


if __name__ == "__main__":
    main()
