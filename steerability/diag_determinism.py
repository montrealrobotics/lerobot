"""Where does restored-branch divergence come from: the env restore or the policy?

From a saved state, run N steps twice per condition and report the max |diff| of the final MuJoCo state,
split by state component:
  env_replay    restore -> replay one fixed action sequence (the policy's first chunk, repeated): env only
  policy        restore -> policy with fixed noise (what p2_branching's check does)
  policy_twice_chunk  two chunk calls on the identical restored obs: is inference itself deterministic?
Conditions are run on a fresh env (never reset) and again after an env.reset().

  python diag_determinism.py --task lamp --arm arm4 --sft-step 4000 --checkpoint <ckpt> --out <dir> \
      --states <own states .pkl> --env-factory ... --obs-fn ... --cells s1_start_0
"""
import argparse
import json

import mujoco
import numpy as np

from adapters import _MJ_FULL, _unwrap, make_adapter
from common import TASK_SPECS, NoiseStream, add_common_args, cells_for, dup, load_policy, load_states, rollout, sample_noise

COMPONENTS = [("time", mujoco.mjtState.mjSTATE_TIME), ("qpos", mujoco.mjtState.mjSTATE_QPOS),
              ("qvel", mujoco.mjtState.mjSTATE_QVEL), ("act", mujoco.mjtState.mjSTATE_ACT),
              ("warmstart", mujoco.mjtState.mjSTATE_WARMSTART), ("ctrl", mujoco.mjtState.mjSTATE_CTRL),
              ("mocap_pos", mujoco.mjtState.mjSTATE_MOCAP_POS), ("mocap_quat", mujoco.mjtState.mjSTATE_MOCAP_QUAT)]


def components(env):
    raw = _unwrap(env)
    m, d = raw.sim.model._model, raw.sim.data._data
    out = {}
    for name, bit in COMPONENTS:
        v = np.empty(mujoco.mj_stateSize(m, bit))
        mujoco.mj_getState(m, d, v, bit)
        out[name] = v
    return out


def diff(a, b):
    return {k: float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0 for k in a}


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--states", required=True)
    p.add_argument("--n-steps", type=int, default=48)
    args = p.parse_args()
    spec = TASK_SPECS[args.task]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    S = load_states(args.states)["states"]
    cell = cells_for(args)[0]
    picks = [s for s in S if s["cell"] == cell]
    picks = [picks[0], picks[len(picks) // 3]]  # t = 0 and a mid-episode state
    z = sample_noise("gaussian", 1, spec["noise_dim"], np.random.default_rng(0))[0]
    report = {}
    for label, prep in (("fresh_env", False), ("after_reset", True)):
        env = adapter.make_env(cell, args.seed)
        if prep:
            adapter.reset(env, args.seed)
        for s in picks:
            key = f"{label}/t{s['t']}"
            # env only: replay the actions the policy produced in the first policy run
            obs = adapter.set_state(env, s["sim_state"])
            batch = policy.prepare(obs)
            chunk = policy.to_env(policy.chunk(batch, dup(z[None], spec["chunk_size"])))[0][: spec["n_action_steps"]]
            acts = np.concatenate([chunk] * (args.n_steps // spec["n_action_steps"] + 1))[: args.n_steps]
            finals = []
            for _ in range(2):
                adapter.set_state(env, s["sim_state"])
                for a in acts:
                    adapter.step(env, a)
                finals.append(components(env))
            report[f"{key}/env_replay"] = diff(*finals)
            finals = []
            for _ in range(2):
                obs = adapter.set_state(env, s["sim_state"])
                ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=0, first=z, after="fixed")
                rollout(policy, adapter, env, obs, ns, s["t"] + args.n_steps, spec["n_action_steps"], start_step=s["t"])
                finals.append(components(env))
            report[f"{key}/policy"] = diff(*finals)
            # same, but re-seed every global RNG at the restore: if this is exact, the divergence is global-RNG
            # randomness in stepping/observations (e.g. robosuite observable noise), not the restore
            import random, torch
            finals = []
            for _ in range(2):
                np.random.seed(0); random.seed(0); torch.manual_seed(0)
                obs = adapter.set_state(env, s["sim_state"])
                ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=0, first=z, after="fixed")
                rollout(policy, adapter, env, obs, ns, s["t"] + args.n_steps, spec["n_action_steps"], start_step=s["t"])
                finals.append(components(env))
            report[f"{key}/policy_reseeded"] = diff(*finals)
            # observations after one identical step from the identical restore
            obs_after = []
            for _ in range(2):
                adapter.set_state(env, s["sim_state"])
                o1, _, _ = adapter.step(env, acts[0])
                obs_after.append(o1)
            report[f"{key}/obs_after_1_step"] = {
                k: float(np.max(np.abs(np.asarray(obs_after[0][k], float) - np.asarray(obs_after[1][k], float))))
                for k in obs_after[0] if hasattr(obs_after[0][k], "shape")
                and np.asarray(obs_after[0][k]).shape == np.asarray(obs_after[1][k]).shape}
            obs = adapter.set_state(env, s["sim_state"])
            c1 = policy.chunk(policy.prepare(obs), dup(z[None], spec["chunk_size"]))
            obs = adapter.set_state(env, s["sim_state"])
            c2 = policy.chunk(policy.prepare(obs), dup(z[None], spec["chunk_size"]))
            report[f"{key}/policy_twice_chunk"] = {"chunk": float(np.max(np.abs(c1 - c2)))}
            for k in [k for k in report if k.startswith(key)]:
                print(k, {n: f"{v:.3g}" for n, v in report[k].items() if v})
    out = f"{args.out}/determinism_{args.task}_{args.arm}_{args.sft_step:06d}.json"
    json.dump(report, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
