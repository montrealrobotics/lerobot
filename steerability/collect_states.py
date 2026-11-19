"""Roll out a frozen checkpoint in every DSRL eval cell and save (sim state, obs, probe targets)
every few macro steps.

Two uses:
  * a REFERENCE state set (one per task, e.g. from arm0@16k) that every checkpoint is probed on
    in Phase 1 -- same inputs across checkpoints, so differences are the checkpoint's;
  * a checkpoint's OWN state set for Phase 2 branching (the states its DSRL run would visit).

Noise mode defaults to duplicated gaussian, i.e. what DSRL's gaussian warmup feeds the policy.

  python collect_states.py --task lamp --arm arm0 --sft-step 16000 --checkpoint <ckpt> \
      --out diag --env-factory mylab.envs:make_cell_env --obs-fn mylab.envs:pi05_frame \
      --n-rollouts 4 --record-every 2 --label ref
"""
import argparse
import pickle

from adapters import make_adapter
from common import NoiseStream, TASK_SPECS, add_common_args, bank_info, cells_for, load_policy, rollout, tag
from pathlib import Path


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--record-every", type=int, default=2, help="macro steps between saved states")
    p.add_argument("--noise-mode", choices=["dup", "iid"], default="dup")
    p.add_argument("--label", default=None, help="output name; default <task>_<arm>_<step>")
    args = p.parse_args()

    spec = TASK_SPECS[args.task]
    adapter = make_adapter(args)
    policy = load_policy(args, adapter)
    states, summary = [], {}
    for cell in cells_for(args):
        env = adapter.make_env(cell, args.seed)
        wins = 0
        for r in range(args.n_rollouts):
            seed = args.seed * 1000 + r
            obs = adapter.reset(env, seed)
            ns = NoiseStream(spec["noise_dim"], spec["chunk_size"], seed=seed + 7,
                             after="iid" if args.noise_mode == "iid" else "base")
            res = rollout(policy, adapter, env, obs, ns, args.max_steps or adapter.max_steps(env),
                          spec["n_action_steps"], record_every=args.record_every)
            wins += res.success
            for rec in res.records:
                rec.update(cell=cell, rollout=r, rollout_success=res.success, rollout_len=res.steps)
                states.append(rec)
            print(f"{cell} rollout {r}: success={res.success} steps={res.steps} states={len(res.records)}")
        summary[cell] = dict(success_rate=wins / args.n_rollouts, n_rollouts=args.n_rollouts)

    out = Path(args.out) / "states" / f"{args.label or tag(args)}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dict(meta=dict(task=args.task, arm=args.arm, sft_step=args.sft_step,
                                   checkpoint=args.checkpoint, noise_mode=args.noise_mode,
                                   record_every=args.record_every, summary=summary, bank=bank_info(args)),
                         states=states), f)
    print(f"wrote {out} ({len(states)} states)")


if __name__ == "__main__":
    main()
