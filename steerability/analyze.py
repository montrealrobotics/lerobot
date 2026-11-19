"""Join every diagnostic with the DSRL outcomes and ask which one explains what dup-noise doesn't.

Unit of analysis: (task, arm, sft_step, env); each env is a DSRL training or held-out cell
(`split`, from dsrl.json meta). Targets, from dsrl.json:
  final     mean over seeds of the env's success over the evals at >= 400k env steps (as summary.json)
  t80       mean over seeds of the first env step with env success >= 0.8 (never -> 525k)
  seed_std  std over seeds of `final`                       (split-seed checkpoints)
Baselines, from dual_noise.json: base.dup, base.iid (per env); base.dup_train, base.iid_train
(checkpoint level, training cells only); base.warmup from --baseline-csv (DSRL's own warmup success).
Diagnostics: per_env metrics of every <out>/<phase>/*.json, plus <out>/p0/ckpt_signals.csv
(checkpoint-level, broadcast to its envs).

For each task x split (train, heldout, all) x target x diagnostic:
  rho          Spearman, with a 95% bootstrap CI resampling (arm, ckpt) clusters
  partial      Spearman of rank residuals after regressing out the dup control  <- the headline:
               information beyond what dual-noise eval already gives. The control is base.dup
               for train/all and base.dup_train for heldout, so no held-out eval enters a predictor.
  partial_given_warmup   same, controlling for base.warmup instead
  loao_dR2     leave-one-arm-out out-of-sample R^2 of [dup + diag] minus [dup alone]; > 0 means
               the diagnostic predicts an unseen arm better, not just separates arms

  python analyze.py --out diag --dsrl-json dsrl.json --dual-noise-json dual_noise.json \
      --baseline-csv warmup_success.csv [--exclude lamp/arm4 ...]
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

CENSORED = 525_000.0
FINAL_FROM = 400_000
warnings.filterwarnings("ignore", message="An input array is constant")
KEYS = ["task", "arm", "sft_step", "env"]


def targets(dsrl, splits):
    rows = []
    for task, arms in dsrl.items():
        for arm, a in arms.items():
            for ckpt, c in a["checkpoints"].items():
                envs = next(iter(c["seeds"].values()))["per_env"].keys()
                for env in envs:
                    finals, t80s = [], []
                    for rec in c["seeds"].values():
                        v = np.asarray(rec["per_env"][env], float)
                        steps = np.asarray(rec["env_steps"], float)
                        finals.append(np.nanmean(v[steps >= FINAL_FROM]))
                        hit = np.nonzero(v >= 0.8)[0]
                        t80s.append(steps[hit[0]] if len(hit) else CENSORED)
                    rows.append(dict(task=task, arm=arm, sft_step=int(ckpt), env=env, split=splits[task][env],
                                     final=np.mean(finals),
                                     t80=np.mean(t80s), seed_std=np.std(finals)))
    return pd.DataFrame(rows)


def baselines(dn):
    rows = []
    for task, arms in dn.items():
        for arm, a in arms.items():
            for ckpt, c in a["checkpoints"].items():
                for env in c["duplicated"]["per_env"]:
                    rows.append(dict(task=task, arm=arm, sft_step=int(ckpt), env=env,
                                     **{"base.dup": c["duplicated"]["per_env"][env]["success_rate"],
                                        "base.iid": c["iid"]["per_env"][env]["success_rate"],
                                        "base.dup_train": c["duplicated"]["training_envs"]["success_rate"],
                                        "base.iid_train": c["iid"]["training_envs"]["success_rate"]}))
    return pd.DataFrame(rows)


def diagnostics(out: Path, extra_dirs=()):
    """Per-env rows of every <dir>/p[12]_*/*.json. A result file in a later dir (extra_dirs, in order) replaces
    the same (phase, task, arm, step) file from earlier ones entirely, e.g. a training-cells-only 1c rerun."""
    by_file = {}
    for root in [out, *map(Path, extra_dirs)]:
        for f in sorted(Path(root).expanduser().glob("p[12]_*/*.json")):
            d = json.loads(f.read_text())
            m = d["meta"]
            by_file[(m["phase"], m["task"], m["arm"], int(m["sft_step"]))] = d
    frames = []
    for (phase, *_), d in by_file.items():
        m = d["meta"]
        for env, met in d["per_env"].items():
            row = dict(task=m["task"], arm=m["arm"], sft_step=int(m["sft_step"]), env=env)
            row.update({f"{phase}.{k}": v for k, v in met.items() if isinstance(v, (int, float)) and v is not None})
            frames.append(row)
    df = pd.DataFrame(frames)
    if not df.empty:
        df = df.groupby(KEYS, as_index=False).first()  # merge phases into one row per unit
    p0 = out / "p0" / "ckpt_signals.csv"
    ck = pd.DataFrame()
    if p0.exists():
        ck = pd.read_csv(p0)
        ck = ck.rename(columns={k: f"p0.{k}" for k in ck.columns if k not in ("task", "arm", "sft_step")})
    return df, ck


def _resid(y, x):
    X = np.c_[np.ones(len(y)), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(x, y, z):
    """z: one control (n,) or several (n, k); each is rank-transformed."""
    z = np.asarray(z, float).reshape(len(x), -1)
    rz = np.column_stack([rankdata(c) for c in z.T])
    rx, ry = rankdata(x), rankdata(y)
    return spearmanr(_resid(rx, rz), _resid(ry, rz))[0]


def boot_ci(df, xcol, ycol, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(["arm", "sft_step"])]
    vals = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if s[xcol].nunique() > 1 and s[ycol].nunique() > 1:
            vals.append(spearmanr(s[xcol], s[ycol])[0])
    return (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)) if vals else (np.nan, np.nan)


def loao(df, xcol, ycol, base):
    """base: baseline column name or list of names; compares [base + xcol] with [base] on held-out arms."""
    base = [base] if isinstance(base, str) else list(base)
    if df.arm.nunique() < 3:
        return np.nan
    pred_b, pred_x, ys = [], [], []
    for arm in df.arm.unique():
        tr, te = df[df.arm != arm], df[df.arm == arm]
        if len(tr) < 6 or te.empty or tr[xcol].nunique() < 3 or tr.arm.nunique() < 2:
            continue  # need variation in x within the training arms, or the fit extrapolates
        mx, sx = tr[xcol].mean(), tr[xcol].std() + 1e-12
        for cols, store in ((base, pred_b), (base + [xcol], pred_x)):
            Xtr = np.c_[np.ones(len(tr)), *[(tr[c] - (mx if c == xcol else 0)) / (sx if c == xcol else 1) for c in cols]]
            Xte = np.c_[np.ones(len(te)), *[(te[c] - (mx if c == xcol else 0)) / (sx if c == xcol else 1) for c in cols]]
            beta, *_ = np.linalg.lstsq(Xtr, tr[ycol].to_numpy(), rcond=None)
            store.extend(Xte @ beta)
        ys.extend(te[ycol])
    if not ys:
        return np.nan
    ys = np.asarray(ys)
    sst = ((ys - ys.mean()) ** 2).sum() + 1e-12
    r2 = lambda p: 1 - ((ys - np.asarray(p)) ** 2).sum() / sst  # noqa: E731
    return float(r2(pred_x) - r2(pred_b))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--dsrl-json", required=True)
    p.add_argument("--dual-noise-json", required=True)
    p.add_argument("--min-units", type=int, default=8)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--baseline-csv", help="warmup_success.csv: task,arm,ckpt,warmup_success -> base.warmup")
    p.add_argument("--exclude", nargs="*", default=[], metavar="TASK/ARM", help="drop these ladders (default: none)")
    p.add_argument("--extra-diag-dirs", nargs="*", default=[],
                   help="more result roots; their files replace --out's for the same phase/task/arm/step")
    p.add_argument("--controls", nargs="*", default=[], metavar="COL",
                   help="covariates added to the dup control in partials and in the LOAO baseline, e.g. sft_step")
    p.add_argument("--analysis-name", default="analysis", help="subdir of --out for this run's outputs")
    args = p.parse_args()

    out = Path(args.out)
    dsrl = json.load(open(args.dsrl_json))
    splits = {t: {**{e: "train" for e in m["training_envs"]}, **{e: "heldout" for e in m["held_out_envs"]}}
              for t, m in dsrl["meta"]["tasks"].items()}
    T = targets(dsrl["results"], splits)
    df = T.merge(baselines(json.load(open(args.dual_noise_json))["results"]), on=KEYS, how="left")
    if args.baseline_csv:
        W = pd.read_csv(args.baseline_csv).rename(columns={"ckpt": "sft_step", "warmup_success": "base.warmup"})
        df = df.merge(W[["task", "arm", "sft_step", "base.warmup"]], on=["task", "arm", "sft_step"], how="left")
    for ta in args.exclude:
        task, arm = ta.split("/")
        df = df[~((df.task == task) & (df.arm == arm))]
    D, CK = diagnostics(out, args.extra_diag_dirs)
    if not D.empty:
        df = df.merge(D, on=KEYS, how="left")
    if not CK.empty:
        df = df.merge(CK, on=["task", "arm", "sft_step"], how="left")
    adir = out / args.analysis_name
    adir.mkdir(parents=True, exist_ok=True)
    df.to_csv(adir / "table.csv", index=False)

    diag_cols = [c for c in df.columns if c not in KEYS + ["split", "final", "t80", "seed_std"]
                 and pd.api.types.is_numeric_dtype(df[c]) and c not in args.controls
                 and not c.endswith((".n_episodes", ".n_states", ".n_rollouts"))]
    # Held-out targets get each checkpoint's TRAINING-cell mean of every diagnostic, so no held-out cell's
    # states enter a predictor (checkpoint-level metrics are only clean if computed on training cells: see
    # --extra-diag-dirs).
    phase_cols = [c for c in diag_cols if c.startswith(("p1_", "p2_"))]
    train_mean = df[df.split == "train"].groupby(["task", "arm", "sft_step"])[phase_cols].mean()
    rows = []
    groups = [(task, split, g[g.split == split] if split != "all" else g)
              for task, g in df.groupby("task") for split in ("train", "heldout", "all")]
    for task, split, g in groups:
        ctrl = "base.dup_train" if split == "heldout" else "base.dup"
        if split == "heldout":  # no held-out eval or held-out state may enter a predictor of held-out outcomes
            g = g.drop(columns=["base.dup", "base.iid"])
            idx = pd.MultiIndex.from_frame(g[["task", "arm", "sft_step"]])
            g = g.copy()
            g[phase_cols] = train_mean.reindex(idx)[phase_cols].to_numpy()
        for y in ("final", "t80", "seed_std"):
            for x in [c for c in diag_cols if c in g.columns]:
                ctrls = [ctrl, *args.controls]
                cols = [x, y, *ctrls, "arm", "sft_step"] + (["base.warmup"] if "base.warmup" in g else [])
                s = g[list(dict.fromkeys(cols))].replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) < args.min_units or s[x].nunique() < 3:
                    continue
                rho = spearmanr(s[x], s[y])[0]
                lo, hi = boot_ci(s, x, y, n=args.boot)
                part = partial_spearman(s[x].to_numpy(), s[y].to_numpy(), s[ctrls].to_numpy()) if x != ctrl else np.nan
                part_w = (partial_spearman(s[x].to_numpy(), s[y].to_numpy(), s["base.warmup"].to_numpy())
                          if "base.warmup" in s and x != "base.warmup" else np.nan)
                rows.append(dict(task=task, split=split, target=y, diagnostic=x, n=len(s), rho=rho, ci_lo=lo, ci_hi=hi,
                                 control="+".join(ctrls), partial_given_dup=part, partial_given_warmup=part_w,
                                 loao_dR2=loao(s, x, y, ctrls) if x != ctrl else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(adir / "correlations.csv", index=False)

    lines = ["# Which diagnostics explain DSRL outcomes beyond dual-noise success?", "",
             "Rows sorted by |partial_given_dup| (partial given the dup control plus any --controls). "
             "rho CI = 95% cluster bootstrap over (arm, ckpt).", "",
             "Held-out rows use each checkpoint's training-cell mean of every diagnostic. "
             "`seed_std` is the std over only 3 RL seeds per checkpoint, so its noise floor is high.", ""]
    if args.exclude:
        lines += [f"Excluded: {', '.join(args.exclude)}", ""]
    if args.controls or args.extra_diag_dirs:
        lines += [f"Extra controls: {args.controls}; extra diagnostic dirs: {args.extra_diag_dirs}", ""]
    for (task, split, y), g in R.groupby(["task", "split", "target"], sort=False):
        g = g.assign(_k=g.partial_given_dup.abs().fillna(-1)).sort_values("_k", ascending=False).drop(columns="_k")
        lines += [f"## {task} / {split} cells -- target `{y}` (control `{g.control.iloc[0]}`)", "",
                  g.drop(columns=["task", "split", "target", "control"]).round(3).to_markdown(index=False), ""]
    (adir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\nwrote {adir}/table.csv, correlations.csv, report.md")


if __name__ == "__main__":
    main()
