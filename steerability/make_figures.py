"""Recreate the figures from the JSON files in this folder.

    python make_figures.py            # writes figures/ next to this file, PNG + PDF

Per arm and task:
  dsrl_<task>_<arm>                  DSRL success over env steps; one line per SFT checkpoint, mean over 3 seeds
  before_rl_<task>_<arm>_iid         SFT checkpoint alone, IID noise
  before_rl_<task>_<arm>_repeated    SFT checkpoint alone, repeated noise (one vector per action chunk)
Per task, all arms together:
  overview_before_rl_<task>_<mode>

Needs matplotlib only.
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARM_STYLE = {"arm0": ("#2a78d6", "o"), "arm1": ("#eb6834", "s"), "arm2": ("#1baf7a", "^"),
             "arm3": ("#eda100", "D"), "arm4": ("#e87ba4", "v")}
CKPT_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
GROUPS = [("all_envs", "All environments"), ("training_envs", "Training environments"),
          ("held_out_envs", "Held-out environments")]
SPLIT_STYLE = {"training_envs": ("Training environments", "#2a78d6", "o"),
               "held_out_envs": ("Held-out environments", "#eb6834", "s")}
MODE_NAME = {"iid": "IID noise", "duplicated": "repeated noise"}
MODE_FILE = {"iid": "iid", "duplicated": "repeated"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_xlabel(xlabel, color=INK_2)


def shade_dsrl_checkpoints(ax, meta, task):
    for x in meta["tasks"][task]["dsrl_checkpoints"]:
        ax.axvspan(x / 1000 - 0.35, x / 1000 + 0.35, color=GRID, alpha=0.45, lw=0, zorder=0)


def header(fig, title, handles, labels):
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2, ncol=len(labels),
               loc="upper right", bbox_to_anchor=(0.995, 0.995))


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", facecolor=SURFACE, dpi=150)
    plt.close(fig)


def before_rl_series(ck, mode, grp):
    steps = sorted(ck, key=int)
    ps = [ck[s][mode][grp]["success_rate"] for s in steps]
    ns = [ck[s][mode][grp]["n_episodes"] for s in steps]
    ys = [100 * p for p in ps]
    se = [100 * math.sqrt(p * (1 - p) / n) for p, n in zip(ps, ns)]
    return [int(s) / 1000 for s in steps], ys, se, ns


def before_rl_arm(dn, meta, task, arm, mode, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(ax, "SFT step (k)")
    shade_dsrl_checkpoints(ax, meta, task)
    ck = dn[task][arm]["checkpoints"]
    ns_seen = {}
    for grp, (label, color, marker) in SPLIT_STYLE.items():
        xs, ys, se, ns = before_rl_series(ck, mode, grp)
        ax.fill_between(xs, [max(0, y - e) for y, e in zip(ys, se)], [min(100, y + e) for y, e in zip(ys, se)],
                        color=color, alpha=0.13, lw=0)
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5.5, markeredgecolor=SURFACE, label=label)
        ns_seen[label] = sorted(set(ns))
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlim(1, 21)
    ax.set_ylabel("Success rate (%)", color=INK_2)
    handles, labels = ax.get_legend_handles_labels()
    name = meta["arms"][arm]["name"]
    fig.suptitle(f"{meta['tasks'][task]['robocasa_task']} · {arm} · {name}\nBefore RL, {MODE_NAME[mode]}",
                 x=0.01, y=0.985, ha="left", va="top", color=INK)
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
               loc="upper left", bbox_to_anchor=(0.09, 0.85))
    eps = ", ".join(f"{k.split()[0].lower()} {'/'.join(map(str, v))}" for k, v in ns_seen.items())
    rungs = ", ".join(f"{c // 1000}k" for c in meta["tasks"][task]["dsrl_checkpoints"])
    fig.text(0.01, 0.015, f"Episodes per point: {eps}. Band: ±1 binomial SE.\nShaded: checkpoints DSRL was run from ({rungs}).",
             fontsize=8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.23)
    save(fig, out, f"before_rl_{task}_{arm}_{MODE_FILE[mode]}")


def before_rl_overview(dn, meta, task, mode, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (grp, title) in zip(axes, GROUPS):
        style(ax, "SFT step (k)")
        shade_dsrl_checkpoints(ax, meta, task)
        for arm, (color, marker) in ARM_STYLE.items():
            if arm not in dn[task]:
                continue
            xs, ys, se, _ = before_rl_series(dn[task][arm]["checkpoints"], mode, grp)
            ax.fill_between(xs, [max(0, y - e) for y, e in zip(ys, se)], [min(100, y + e) for y, e in zip(ys, se)],
                            color=color, alpha=0.12, lw=0)
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5, markeredgecolor=SURFACE, label=arm)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xticks(range(2, 21, 2))
    axes[0].set_ylabel("Success rate (%)", color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    header(fig, f"{meta['tasks'][task]['robocasa_task']} · before RL ({MODE_NAME[mode]})", handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, f"overview_before_rl_{task}_{MODE_FILE[mode]}")


def dsrl_curves(ds, meta, task, arm, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    cks = ds[task][arm]["checkpoints"]
    n_seeds = set()
    for ax, (grp, title) in zip(axes, GROUPS):
        style(ax, "Env steps (k)")
        for color, (ck, d) in zip(CKPT_BLUES, sorted(cks.items(), key=lambda kv: int(kv[0]))):
            by_step = {}
            for sd in d["seeds"].values():
                for s, v in zip(sd["env_steps"], sd["success_rate"][grp]):
                    if v is not None:
                        by_step.setdefault(s, []).append(100 * v)
            n_seeds.add(len(d["seeds"]))
            xs = sorted(by_step)
            mean = [st.mean(by_step[s]) for s in xs]
            std = [st.pstdev(by_step[s]) for s in xs]
            ax.fill_between([x / 1000 for x in xs], [max(0, m - e) for m, e in zip(mean, std)],
                            [min(100, m + e) for m, e in zip(mean, std)], color=color, alpha=0.15, lw=0)
            ax.plot([x / 1000 for x in xs], mean, color=color, lw=2, label=f"SFT {int(ck) // 1000}k")
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlim(0, 500)
    axes[0].set_ylabel("Success rate (%)", color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    name = meta["arms"][arm]["name"]
    header(fig, f"{meta['tasks'][task]['robocasa_task']} · {arm} · {name} · DSRL", handles, labels)
    fig.text(0.01, 0.015, f"Mean over {'/'.join(map(str, sorted(n_seeds)))} RL seeds; band = ±1 std across seeds. "
             "Deterministic noise actor, eval every 25k env steps.", fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, out, f"dsrl_{task}_{arm}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    dn_file = json.loads((HERE / "dual_noise.json").read_text())
    ds_file = json.loads((HERE / "dsrl.json").read_text())
    meta = dn_file["meta"]
    dn, ds = dn_file["results"], ds_file["results"]
    for task in meta["tasks"]:
        for mode in ("iid", "duplicated"):
            before_rl_overview(dn, meta, task, mode, args.out)
        for arm in meta["arms"]:
            dsrl_curves(ds, meta, task, arm, args.out)
            for mode in ("iid", "duplicated"):
                before_rl_arm(dn, meta, task, arm, mode, args.out)
    print(f"{len(list(args.out.glob('*.png')))} figures (PNG + PDF) -> {args.out}")


if __name__ == "__main__":
    main()
