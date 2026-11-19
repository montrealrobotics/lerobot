"""Phase 1c -- does the checkpoint still look at (and distinguish) its inputs?

On the REFERENCE state set, with M fixed duplicated noises:

Observation reliance (per cell). For each perturbation, action change relative to the
noise-induced spread at that state (ratio > 1: the input moves the action more than noise does):
  blank_third       third-person camera zeroed
  blank_wrist       wrist camera zeroed
  shuffle_third     third-person image from a random other state
  swap_cell_third   third-person image from ANOTHER cell at the nearest macro step
  swap_cell_state   proprio from that other cell's state
A checkpoint that replays one memorized trajectory has swap_cell_* ~ 0.

Representation (checkpoint-level; repeated on every cell row and flagged ckpt_level):
  cka_<module>      linear CKA of pooled features vs --reference-checkpoint (default
                    lerobot/pi05_base) on the same preprocessed batches. Frozen modules -> 1.
  probe_r2_<module> 5-fold ridge R^2 from features to probe targets (EE pose + object positions)
  cell_acc_<module> 5-fold nearest-centroid accuracy at telling the cells apart

  python p1_obs_reliance.py --task lamp --arm arm3 --sft-step 16000 --checkpoint <ckpt> \
      --out diag --states diag/states/ref_lamp.pkl --obs-fn ... --reference-checkpoint lerobot/pi05_base
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from adapters import make_adapter
from common import (TASK_SPECS, TIMING, add_common_args, cells_for, dup, file_hash, load_policy, load_states, nanmean, timed,
                    result_path, sample_noise, write_result)

PERTURBATIONS = ["blank_third", "blank_wrist", "shuffle_third", "swap_cell_third", "swap_cell_state"]
# vision_tower: hooked on its first call per sample, i.e. the first camera (robot0_agentview_left).
# language_model.norm: pi05 runs the LM prefix pass as `language_model.forward(...)`, which bypasses forward
# hooks on language_model itself; its final norm is called normally, so it gives the LM's last-layer output.
DEFAULT_MODULES = "paligemma.model.vision_tower,paligemma.model.multi_modal_projector,paligemma.model.language_model.norm"


def short(module: str) -> str:
    """Metric-name suffix: last path component, except '<parent>.norm' -> '<parent>'."""
    parts = module.split(".")
    return parts[-2] if parts[-1] == "norm" and len(parts) > 1 else parts[-1]


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else np.nan


def _folds(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def ridge_cv_r2(X, Y, k=5, lams=(1e-2, 1e-1, 1, 10, 100), seed=0):
    n = len(X)
    if n < 2 * k:
        return np.nan
    rng = np.random.default_rng(seed)
    best = -np.inf
    for lam in lams:
        pred = np.zeros_like(Y, dtype=float)
        for te in _folds(n, k, rng):
            tr = np.setdiff1d(np.arange(n), te)
            mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
            my = Y[tr].mean(0)
            A = (X[tr] - mx) / sx
            # dual form: cheap when features >> samples
            alpha = np.linalg.solve(A @ A.T + lam * np.eye(len(tr)), Y[tr] - my)
            pred[te] = ((X[te] - mx) / sx) @ (A.T @ alpha) + my
        ss_res = ((Y - pred) ** 2).sum(0)
        ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
        best = max(best, float(np.mean(1 - ss_res / ss_tot)))
    return best


def centroid_cv_acc(X, labels, k=5, seed=0):
    labels = np.asarray(labels)
    if len(set(labels)) < 2 or len(X) < 2 * k:
        return np.nan
    rng = np.random.default_rng(seed)
    hits = 0
    for te in _folds(len(X), k, rng):
        tr = np.setdiff1d(np.arange(len(X)), te)
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
        cs = {c: ((X[tr][labels[tr] == c] - mx) / sx).mean(0) for c in set(labels[tr])}
        for i in te:
            x = (X[i] - mx) / sx
            hits += min(cs, key=lambda c: np.linalg.norm(x - cs[c])) == labels[i]
    return hits / len(X)


def extract(policy, states, modules):
    feats = {}
    for s in states:
        f = policy.features(policy.prepare(s["raw_obs"]), modules)
        for k, v in f.items():
            feats.setdefault(k, []).append(np.asarray(v).reshape(-1))
    return {k: np.stack(v) for k, v in feats.items()}


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--states", required=True)
    p.add_argument("--n-noise", type=int, default=8)
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--max-states-per-cell", type=int, default=30)
    p.add_argument("--feature-modules", default=DEFAULT_MODULES, help="comma list of module-name suffixes")
    p.add_argument("--reference-checkpoint", default="lerobot/pi05_base", help="'none' to skip CKA")
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    T, n_exec = spec["chunk_size"], spec["n_action_steps"]
    modules = [m for m in args.feature_modules.split(",") if m]
    adapter = make_adapter(args)
    rng = np.random.default_rng(args.noise_seed)
    Z = sample_noise("gaussian", args.n_noise, spec["noise_dim"], rng)

    S = load_states(args.states)["states"]
    if args.cells:  # e.g. training cells only: every pooled metric (CKA, probes, cell_acc, swap donors) then excludes the rest
        keep_cells = set(cells_for(args))
        S = [s for s in S if s["cell"] in keep_cells]
        assert S, f"no states in cells {sorted(keep_cells)}"
    by_cell = {}
    for s in S:
        by_cell.setdefault(s["cell"], []).append(s)
    for c, v in by_cell.items():
        if len(v) > args.max_states_per_cell:
            keep = np.linspace(0, len(v) - 1, args.max_states_per_cell).round().astype(int)
            by_cell[c] = [v[i] for i in keep]
    states = [s for v in by_cell.values() for s in v]

    # --- reference features first (cached), so only one 4B model is resident at a time
    ref_feats = None
    if args.reference_checkpoint != "none":
        # Keyed on task, state set, reference and module list. Valid across checkpoints of a task because
        # their preprocessors are byte-identical (checked 2026-10-03 for all 5 arms of lamp and coffee).
        mods_key = hashlib.sha1((",".join(modules) + "|" + ",".join(sorted(by_cell))).encode()).hexdigest()[:8]
        cache = Path(args.out) / "p1_obs_reliance" / "ref_cache" / \
            f"{args.task}_{file_hash(args.states)}_{args.reference_checkpoint.replace('/', '_').replace(':', '_')}_{mods_key}.npz"
        if cache.exists():
            ref_feats = dict(np.load(cache))
            TIMING["ref_cache_hit"] = 1.0
        else:
            TIMING["ref_cache_hit"] = 0.0
            ref = load_policy(args, adapter, path=args.reference_checkpoint,
                              processor_path=None if args.checkpoint.startswith("mock:") else args.checkpoint)
            with timed("ref_features_s"):
                ref_feats = extract(ref, states, modules)
            ref.release()
            del ref
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")  # atomic: concurrent jobs may race
            np.savez(tmp, **ref_feats)
            os.replace(tmp, cache)

    policy = load_policy(args, adapter)
    with timed("features_s"):
        cur_feats = extract(policy, states, modules)
    missing = [m for m in modules if m not in cur_feats or (ref_feats is not None and m not in ref_feats)]
    if missing:  # a hook that never fires would silently drop that module's CKA / probes
        raise RuntimeError(f"no features captured for {missing}")

    # --- observation reliance
    per_env = {}
    order = np.random.default_rng(args.seed)
    for cell, cs in by_cell.items():
        others = [s for c2, v in by_cell.items() if c2 != cell for s in v]
        rows = []
        for s in cs:
            a0 = policy.chunk(policy.prepare(s["raw_obs"]), dup(Z, T))[:, :n_exec].reshape(len(Z), -1)
            i, j = np.triu_indices(len(Z), 1)
            spread = float(np.linalg.norm(a0[i] - a0[j], axis=1).mean()) + 1e-8
            row = dict(noise_spread=spread)
            for kind in PERTURBATIONS:
                if kind.startswith("swap_cell"):
                    if not others:
                        continue
                    donor = min(others, key=lambda o: abs(o["macro"] - s["macro"]))["raw_obs"]
                elif kind == "shuffle_third":
                    donor = states[order.integers(len(states))]["raw_obs"]
                else:
                    donor = None
                o = adapter.perturb_obs(s["raw_obs"], kind, donor)
                ap = policy.chunk(policy.prepare(o), dup(Z, T))[:, :n_exec].reshape(len(Z), -1)
                row[f"rel_{kind}"] = float(np.linalg.norm(ap - a0, axis=1).mean() / spread)
            rows.append(row)
        per_env[cell] = {k: nanmean([r.get(k, np.nan) for r in rows]) for k in rows[0]}

    # --- representation (checkpoint-level)
    labels = [s["cell"] for s in states]
    probe = np.stack([np.asarray(s["probe"], float) for s in states])
    probe = probe[:, probe.std(0) > 1e-6]
    ckpt_level = {}
    for m, X in cur_feats.items():
        if ref_feats is not None and m in ref_feats:
            ckpt_level[f"cka_{short(m)}"] = linear_cka(X, ref_feats[m])
        ckpt_level[f"probe_r2_{short(m)}"] = ridge_cv_r2(X, probe) if probe.shape[1] else np.nan
        ckpt_level[f"cell_acc_{short(m)}"] = centroid_cv_acc(X, labels)
    for cell in per_env:
        per_env[cell].update(ckpt_level)
        print(cell, {k: round(v, 3) for k, v in per_env[cell].items()})

    write_result(result_path(args.out, "p1_obs_reliance", args), args, per_env,
                 dict(ckpt_level_keys=sorted(ckpt_level)), phase="p1_obs_reliance")


if __name__ == "__main__":
    main()
