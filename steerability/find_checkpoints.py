"""Find the SFT checkpoint behind every DSRL run, check it is readable here, write the manifest.

For each (task, arm, ckpt) in dsrl.json:
  1. read --policy_path from the DSRL run's wandb metadata (arm4: the CoreWeave host);
  2. if that path is not readable on this cluster (e.g. an Alliance /scratch/akur/... path),
     look for a copy with the same run-directory name under --search-roots;
  3. classify: ok / blocked (exists, no read permission) / missing (not on this cluster).

Writes
  <out>/checkpoints.tsv   task arm step status path         (one row per DSRL checkpoint)
  <out>/manifest.tsv      task arm run_dir                  (only (task, arm) with ALL rungs ok)
and prints what is blocked, so you know what to ask for.

  python steerability/find_checkpoints.py --dsrl-json steerability/dsrl.json --out steerability
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

DEFAULT_ROOTS = [
    "/network/scratch/a/artur.kuramshin/lerobot/outputs/train",
    "/network/scratch/c/cloutiec/lerobot/outputs/train",
]
HOSTS = {"forge": "https://forge.coreweave.com"}


def policy_path(api_for, project, run_id):
    base, _, rest = project.rpartition(" ") if " " in project.split("/")[0] else ("", "", project)
    api = api_for(HOSTS.get(base))
    meta = api.run(f"{rest}/{run_id}").metadata or {}
    args = meta.get("args") or []
    return next((args[i + 1] for i, a in enumerate(args) if a == "--policy_path" and i + 1 < len(args)), None)


def status(path):
    if path is None:
        return "unknown"
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg) and os.access(cfg, os.R_OK):
        return "ok"
    return "blocked" if os.path.exists(path) else "missing"  # exists but unreadable vs not on this cluster


def find_copy(path, roots):
    """Same run dir name + same checkpoints/<step>/pretrained_model under any search root."""
    m = re.search(r"([^/]+)/checkpoints/(\d{6})/pretrained_model/?$", path or "")
    if not m:
        return None
    run, step = m.groups()
    for root in roots:
        for cand in glob.glob(os.path.join(root, "*", run, "checkpoints", step, "pretrained_model")):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsrl-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search-roots", nargs="*", default=DEFAULT_ROOTS)
    args = ap.parse_args()

    import wandb
    apis = {}

    def api_for(url):
        if url not in apis:
            apis[url] = wandb.Api(overrides={"base_url": url}) if url else wandb.Api()
        return apis[url]

    d = json.load(open(args.dsrl_json))["results"]
    rows = []
    for task, arms in d.items():
        for arm, a in arms.items():
            for ck, c in sorted(a["checkpoints"].items(), key=lambda kv: int(kv[0])):
                rec = c["seeds"]["0"]
                try:
                    orig = policy_path(api_for, rec["wandb_project"], rec["wandb_runs"][0])
                except Exception as e:  # noqa: BLE001
                    rows.append((task, arm, ck, f"wandb-error:{type(e).__name__}", "")); continue
                st, path = status(orig), orig
                if st != "ok":
                    copy = find_copy(orig, args.search_roots)
                    if copy:
                        st, path = status(copy), copy
                rows.append((task, arm, ck, st, path or ""))
                print(f"{task:6s} {arm:5s} {int(ck):>6d}  {st:8s} {path}")

    out = Path(args.out)
    with open(out / "checkpoints.tsv", "w") as f:
        f.writelines("\t".join(r) + "\n" for r in rows)

    # manifest: one run dir per (task, arm); only if every rung is readable and they agree
    manifest, problems = [], []
    for task in d:
        for arm in d[task]:
            rs = [r for r in rows if r[0] == task and r[1] == arm]
            dirs = {str(Path(r[4]).parents[2]) for r in rs if r[4]}
            if all(r[3] == "ok" for r in rs) and len(dirs) == 1:
                manifest.append((task, arm, dirs.pop()))
            else:
                bad = [f"{r[2]}:{r[3]}" for r in rs if r[3] != "ok"]
                problems.append(f"{task} {arm}: " + (", ".join(bad) if bad else f"rungs come from {len(dirs)} different run dirs"))
    with open(out / "manifest.tsv", "w") as f:
        f.writelines("\t".join(m) + "\n" for m in manifest)

    print(f"\nmanifest.tsv: {len(manifest)} (task, arm) ready: " + ", ".join(f"{t}/{a}" for t, a, _ in manifest))
    if problems:
        print("not ready:\n  " + "\n  ".join(problems))
        print("\n'blocked' = on this cluster but unreadable (ask the owner to chmod); "
              "'missing' = only on the other cluster (copy it or run there).")


if __name__ == "__main__":
    main()