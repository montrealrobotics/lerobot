"""Phase 0 -- free signals from the DSRL runs you already have.

For every (task, arm, ckpt, seed) in dsrl.json, pulls the wandb history and computes:
  first_success_step     first env step with any collection success  (discovery)
  warmup_success         any success before updates start (min_buffer_size x n_action_steps)
  ma10_at_50k/100k/200k  on-policy success early in training
  alpha_final, alpha_min SAC temperature
  q_spread_mean          mean of the logged Q-ensemble spread (if logged)
  sat_final              actor tanh saturation (if logged)

Then, for every checkpoint whose seeds split (some solved, some not), prints the solved vs
failed seeds side by side: no success ever seen -> discovery; successes seen but not kept ->
learnability.

Key names differ between codebases, so they are regexes. Run `--list-keys` once on one run and
adjust --key-map if needed:

  python p0_wandb_signals.py --dsrl-json dsrl.json --out diag --list-keys lamp/arm1/4000/0
  python p0_wandb_signals.py --dsrl-json dsrl.json --out diag \
      --hosts "forge chacloutier-4B=https://forge.coreweave.com"

Each run's full history is fetched once (all keys, one pass, --workers runs in parallel) and
cached in <out>/p0/history/<run_id>.csv, so reruns and --key-map changes are instant.
Offline: --history-dir DIR reads DIR/<run_id>.csv (columns = keys) instead of the API.
"""
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

CENSORED = 525_000.0  # 'never' = one eval interval past the 500k budget

DEFAULT_KEYS = {
    "step": r"^_step$",                          
    "ma10": r"^episode/success_rate_ma10$",
    "ep_success": r"^episode/success$",
    "alpha": r"^train/temperature$",
    "entropy": r"^train/entropy$",
    "critic_loss": r"^train/loss_critic$",
    "actor_grad": r"^train/actor_grad_norm$",
    "critic_grad": r"^train/critic_grad_norm$",
    "ep_length": r"^episode/length$",
    "ep_return": r"^episode/raw_return_ma10$",
}


def md_table(df) -> str:
    """DataFrame -> markdown table without needing the optional `tabulate` package."""
    cols = [str(c) for c in df.columns]
    fmt = lambda v: "" if v is None or (isinstance(v, float) and v != v) else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def parse_project(s, host_overrides):
    """'real-lab/lerobot-dsrl' or 'forge chacloutier-4B/lerobot-dsrl' -> (base_url, entity, project)."""
    base, _, rest = s.rpartition(" ") if " " in s.split("/")[0] else ("", "", s)
    entity, project = rest.split("/", 1)
    url = host_overrides.get(f"{base} {entity}".strip()) or host_overrides.get(base) or None
    return url, entity, project


class History:
    """One full-history scan per run (all keys in a single pass), cached as CSV so reruns and
    key-map changes cost nothing. Thread-safe: one wandb.Api per thread."""

    def __init__(self, history_dir, hosts, cache_dir=None):
        self.dir, self.hosts = Path(history_dir) if history_dir else None, hosts
        self.cache = Path(cache_dir) if cache_dir else None
        self._local = threading.local()

    def _api(self, url):
        import wandb
        apis = getattr(self._local, "apis", None)
        if apis is None:
            apis = self._local.apis = {}
        if url not in apis:
            apis[url] = wandb.Api(overrides={"base_url": url}, timeout=60) if url else wandb.Api(timeout=60)
        return apis[url]

    def _run(self, project, run_id):
        url, ent, proj = parse_project(project, self.hosts)
        return self._api(url).run(f"{ent}/{proj}/{run_id}")

    def keys(self, project, run_id):
        if self.dir:
            return list(pd.read_csv(self.dir / f"{run_id}.csv", nrows=1).columns)
        return sorted(self._run(project, run_id).summary._json_dict.keys())

    def _one(self, project, rid):
        for src in (self.dir, self.cache):
            if src is not None and (src / f"{rid}.csv").exists():
                return pd.read_csv(src / f"{rid}.csv")
        recs = []
        for r in self._run(project, rid).scan_history(page_size=10000):
            recs.append({k: v for k, v in r.items() if isinstance(v, (int, float, bool)) or v is None})
        hist = pd.DataFrame(recs)
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
            hist.to_csv(self.cache / f"{rid}.csv", index=False)
        return hist

    def frame(self, project, run_ids):
        frames = [self._one(project, rid) for rid in run_ids]  # resumed jobs may have several ids
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve(all_keys, key_map):
    out = {}
    for name, rx in key_map.items():
        hits = [k for k in all_keys if re.search(rx, k)]
        out[name] = hits
    return out


def series(df, keys, step_col):
    if not keys or df.empty:
        return None
    k = keys[0]
    if k not in df:
        return None
    s = df[[step_col, k]].dropna().sort_values(step_col)
    return s[step_col].to_numpy(float), s[k].to_numpy(float)


def at(sx, step):
    if sx is None or len(sx[0]) == 0:
        return np.nan
    i = np.searchsorted(sx[0], step, side="right") - 1
    return float(sx[1][i]) if i >= 0 else np.nan


def signals(df, km, warmup_steps):
    step_col = km["step"][0] if km["step"] and km["step"][0] in df else "_step"
    ma10, eps = series(df, km["ma10"], step_col), series(df, km["ep_success"], step_col)
    succ = None
    for sx in (eps, ma10):
        if sx is not None:
            hit = np.nonzero(sx[1] > 0)[0]
            succ = float(sx[0][hit[0]]) if len(hit) else CENSORED
            break
    out = dict(first_success_step=succ if succ is not None else np.nan,
               never_succeeded=float(succ == CENSORED) if succ is not None else np.nan,
               warmup_success=float(succ <= warmup_steps) if succ is not None else np.nan,
               step_key=step_col)
    for s in (50_000, 100_000, 200_000):
        out[f"ma10_at_{s // 1000}k"] = at(ma10, s)
    for name in [n for n in km if n not in ("step", "ma10", "ep_success")]:  # incl. extras from --key-map
        sx = series(df, km[name], step_col)
        if sx is None:
            continue
        out[f"{name}_final"] = float(np.nanmean(sx[1][-max(1, len(sx[1]) // 10):]))
        out[f"{name}_early"] = float(np.nanmean(sx[1][: max(1, len(sx[1]) // 10)]))
        if name == "alpha":
            out["alpha_min"] = float(np.nanmin(sx[1]))
    return out


def final_success(seed_rec, last=5):
    return float(np.mean(seed_rec["success_rate"]["training_envs"][-last:]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsrl-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hosts", default="", help="'<project prefix>=<base_url>;...' for non-default wandb hosts")
    p.add_argument("--key-map", default=None, help="JSON file overriding DEFAULT_KEYS regexes")
    p.add_argument("--history-dir", default=None)
    p.add_argument("--warmup-steps", type=int, default=1000 * 16)
    p.add_argument("--list-keys", default=None, metavar="task/arm/ckpt/seed")
    p.add_argument("--workers", type=int, default=8, help="runs fetched in parallel")
    args = p.parse_args()

    dsrl = json.load(open(args.dsrl_json))["results"]
    hosts = dict(kv.split("=", 1) for kv in args.hosts.split(";") if "=" in kv)
    key_map = dict(DEFAULT_KEYS, **(json.load(open(args.key_map)) if args.key_map else {}))
    H = History(args.history_dir, hosts, cache_dir=Path(args.out) / "p0" / "history")

    if args.list_keys:
        t, a, c, s = args.list_keys.split("/")
        rec = dsrl[t][a]["checkpoints"][c]["seeds"][s]
        ks = H.keys(rec["wandb_project"], rec["wandb_runs"][0])
        print("\n".join(ks))
        print("\nresolved:", json.dumps(resolve(ks, key_map), indent=1))
        return

    jobs = [(task, arm, ckpt, seed, rec) for task, arms in dsrl.items() for arm, a in arms.items()
            for ckpt, c in a["checkpoints"].items() for seed, rec in c["seeds"].items()]

    def work(job):
        task, arm, ckpt, seed, rec = job
        row = dict(task=task, arm=arm, sft_step=int(ckpt), seed=int(seed), final_train=final_success(rec))
        try:
            hist = H.frame(rec["wandb_project"], rec["wandb_runs"])
            row.update(signals(hist, resolve(list(hist.columns), key_map), args.warmup_steps))
        except Exception as e:  # noqa: BLE001 -- keep going, record why
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            print(f"[{i}/{len(jobs)}]", row["task"], row["arm"], row["sft_step"], row["seed"],
                  {k: row.get(k) for k in ("final_train", "first_success_step", "error")}, flush=True)

    df = pd.DataFrame(rows).sort_values(["task", "arm", "sft_step", "seed"])
    out = Path(args.out) / "p0"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seed_signals.csv", index=False)

    # split checkpoints: solved vs failed seeds
    lines = ["# Split checkpoints: solved (final>=0.8) vs failed (final<=0.2) seeds", ""]
    for (t, a, c), g in df.groupby(["task", "arm", "sft_step"]):
        if (g.final_train >= 0.8).any() and (g.final_train <= 0.2).any():
            cols = [k for k in g.columns if k not in ("task", "arm", "sft_step", "step_key", "error")]
            lines += [f"## {t} {a} @ {c}", "", md_table(g[cols]), ""]
    (out / "split_checkpoints.md").write_text("\n".join(lines))

    # per-checkpoint means for analyze.py
    num = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    agg = num.groupby([df.task, df.arm, df.sft_step]).mean(numeric_only=True).drop(columns=["sft_step"], errors="ignore")
    agg.reset_index().to_csv(out / "ckpt_signals.csv", index=False)
    print(f"wrote {out}/seed_signals.csv, ckpt_signals.csv, split_checkpoints.md")


if __name__ == "__main__":
    main()
