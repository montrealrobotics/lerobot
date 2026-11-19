"""Phase 1a -- how much does the checkpoint's action depend on its noise input?

No env stepping: runs the policy on a fixed REFERENCE state set (collect_states.py) with the
same K duplicated noise vectors for every checkpoint, so differences are the checkpoint's.

Per state (executed part of the chunk, first n_action_steps, model space):
  total_var        mean over (t, dim) of the variance across K noises
  arm_var/hand_var same, split into the 6 arm and 16 hand dims
  part_ratio       participation ratio of the K chunks (effective # of directions noise moves)
  pair_gain        median ||a_i - a_j|| / ||z_i - z_j||  (global gain)
  local_gain       mean ||a(z + s*eps) - a(z)|| / (s*||eps||), s = --local-sigma  (local gain)
Per cell:
  noise_share      Var_noise / (Var_noise + Var_state) over the cell's states, at fixed noise:
                   ~0 means the action is set by the observation and noise is decorative.
Optional (--demo-npz or --demo-repo-id): memorization, in env space, standardized per dim
  nn_dist          mean distance of generated chunks to the nearest demo chunk
  nn_ep_entropy    entropy of WHICH demo episode is nearest, across noises, / log K.
                   ~0 = every noise replays the same demo.

  python p1_noise_sensitivity.py --task lamp --arm arm1 --sft-step 4000 --checkpoint <ckpt> \
      --out diag --states diag/states/ref_lamp.pkl --obs-fn mylab.envs:pi05_frame
"""
import argparse

import numpy as np

from adapters import make_adapter
from common import (TASK_SPECS, add_common_args, dup, load_policy, load_states, nanmean, result_path,
                    sample_noise, write_result)


def geometry(a, z, arm_dims):
    """a: (K, T, D) chunks, z: (K, dim) noises."""
    K = a.shape[0]
    var = a.var(axis=0)  # (T, D)
    X = a.reshape(K, -1)
    X = X - X.mean(0)
    lam = np.clip(np.linalg.eigvalsh(X @ X.T / max(K - 1, 1)), 0, None)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 1e-12 else 0.0
    i, j = np.triu_indices(K, 1)
    da = np.linalg.norm(a.reshape(K, -1)[i] - a.reshape(K, -1)[j], axis=1)
    dz = np.linalg.norm(z[i] - z[j], axis=1)
    return dict(total_var=float(var.mean()), arm_var=float(var[:, :arm_dims].mean()),
                hand_var=float(var[:, arm_dims:].mean()), part_ratio=pr,
                pair_gain=float(np.median(da / np.maximum(dz, 1e-8))),
                pair_spread=float(da.mean()))


def load_demo(args):
    if args.demo_npz:
        d = np.load(args.demo_npz)
        return d["action"].astype(np.float32), d["episode_index"].astype(int)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.demo_repo_id, root=args.demo_root, download_videos=False)
    hf = ds.hf_dataset.with_format("numpy", columns=["action", "episode_index"])
    return np.asarray(hf["action"], np.float32), np.asarray(hf["episode_index"]).astype(int)


def demo_chunks(actions, ep, n):
    starts = [i for i in range(len(actions) - n + 1) if ep[i] == ep[i + n - 1]]
    idx = np.asarray(starts)
    C = np.stack([actions[i:i + n] for i in idx])
    return C, ep[idx]


def nn_stats(gen, demo, demo_ep, mu, sd):
    g = ((gen - mu) / sd).reshape(gen.shape[0], -1)
    d = ((demo - mu) / sd).reshape(demo.shape[0], -1)
    best_d, best_i = np.full(len(g), np.inf), np.zeros(len(g), int)
    for s in range(0, len(d), 4096):
        blk = d[s:s + 4096]
        dist = np.sqrt(np.maximum((g ** 2).sum(1)[:, None] - 2 * g @ blk.T + (blk ** 2).sum(1)[None], 0))
        m = dist.argmin(1)
        better = dist[np.arange(len(g)), m] < best_d
        best_d[better], best_i[better] = dist[np.arange(len(g)), m][better], (s + m)[better]
    eps = demo_ep[best_i]
    _, cnt = np.unique(eps, return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(len(g))) if len(g) > 1 else 0.0
    return dict(nn_dist=float(best_d.mean() / np.sqrt(g.shape[1])), nn_ep_entropy=ent, nn_unique_eps=int(len(cnt)))


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--states", required=True, help="reference state set (.pkl from collect_states.py)")
    p.add_argument("--n-noise", type=int, default=64)
    p.add_argument("--n-fixed", type=int, default=8, help="noises used for noise_share")
    p.add_argument("--local-sigma", type=float, default=0.1)
    p.add_argument("--noise-seed", type=int, default=1234, help="keep fixed across checkpoints")
    p.add_argument("--max-states-per-cell", type=int, default=40)
    p.add_argument("--demo-npz", default=None, help="npz with 'action' (N,D) and 'episode_index' (N,)")
    p.add_argument("--demo-repo-id", default=None)
    p.add_argument("--demo-root", default=None)
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    T, n_exec, dim = spec["chunk_size"], spec["n_action_steps"], spec["noise_dim"]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    rng = np.random.default_rng(args.noise_seed)
    Z = sample_noise("gaussian", args.n_noise, dim, rng)
    E = sample_noise("gaussian", args.n_noise, dim, rng)
    E = E / np.linalg.norm(E, axis=1, keepdims=True) * np.sqrt(dim)

    demo = None
    if args.demo_npz or args.demo_repo_id:
        acts, ep = load_demo(args)
        demo, demo_ep = demo_chunks(acts[:, : spec["action_dim"]], ep, n_exec)
        mu, sd = acts[:, : spec["action_dim"]].mean(0), acts[:, : spec["action_dim"]].std(0) + 1e-6

    S = load_states(args.states)["states"]
    by_cell = {}
    for s in S:
        by_cell.setdefault(s["cell"], []).append(s)

    per_env, per_state = {}, []
    for cell, states in by_cell.items():
        if len(states) > args.max_states_per_cell:
            keep = np.linspace(0, len(states) - 1, args.max_states_per_cell).round().astype(int)
            states = [states[i] for i in keep]
        rows, fixed = [], []
        for s in states:
            batch = policy.prepare(s["raw_obs"])
            a_full = policy.chunk(batch, dup(Z, T))
            a = a_full[:, :n_exec]
            a_loc = policy.chunk(batch, dup(Z + args.local_sigma * E, T))[:, :n_exec]
            g = geometry(a, Z, spec["arm_dims"])
            dl = np.linalg.norm((a_loc - a).reshape(len(Z), -1), axis=1)
            g["local_gain"] = float(np.mean(dl / (args.local_sigma * np.linalg.norm(E, axis=1))))
            if demo is not None:
                g.update(nn_stats(policy.to_env(a_full)[:, :n_exec], demo, demo_ep, mu, sd))
            g.update(cell=cell, macro=s["macro"], rollout=s["rollout"])
            rows.append(g)
            fixed.append(a[: args.n_fixed])
        F = np.stack(fixed)  # (S, n_fixed, T, D)
        v_noise = F.var(axis=1).mean()
        v_state = F.var(axis=0).mean()
        agg = {k: nanmean([r[k] for r in rows]) for k in rows[0] if k not in ("cell", "macro", "rollout")}
        agg["noise_share"] = float(v_noise / max(v_noise + v_state, 1e-12))
        agg["n_states"] = len(rows)
        per_env[cell] = agg
        per_state += rows
        print(cell, {k: round(v, 4) for k, v in agg.items()})

    write_result(result_path(args.out, "p1_noise_sensitivity", args), args, per_env,
                 dict(per_state=per_state), phase="p1_noise_sensitivity")


if __name__ == "__main__":
    main()
