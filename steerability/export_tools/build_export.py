"""Build the shareable results export: dual_noise.json, dsrl.json, summary.json, missing.json."""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "export_new"

ARMS = {
    "arm0": {"name": "expert only", "trains": "action expert, time MLPs, action projections", "trainable_params": "693M"},
    "arm1": {"name": "full FT", "trains": "everything", "trainable_params": "4.14B"},
    "arm2": {"name": "frozen LLM", "trains": "SigLIP, projector, action expert", "trainable_params": "1.11B"},
    "arm3": {"name": "LoRA r32", "trains": "LoRA on LM + expert; SigLIP, projector full-FT", "trainable_params": "468M"},
    "arm4": {"name": "LoRA LM, full expert", "trains": "LoRA on LM; action expert, SigLIP, projector full-FT", "trainable_params": "1.15B"},
}
SFT_RUN = {
    ("lamp", "arm0"): "2026-09-20/15-02-05_sft_arms_lamp_arm0_expert_only_lrx1_seed42",
    ("lamp", "arm1"): "2026-09-24/21-14-26_sft_arms_lamp_arm1_full_ft_lrx1_seed42",
    ("lamp", "arm2"): "2026-09-20/04-16-48_sft_arms_lamp_arm2_frozen_llm_lrx1_seed42",
    ("lamp", "arm3"): "2026-09-20/21-06-12_sft_arms_lamp_arm3_lora_r32_lrx1_seed42",
    ("coffee", "arm0"): "2026-09-20/19-38-27_sft_arms_coffee_arm0_expert_only_lrx1_seed42",
    ("coffee", "arm1"): "2026-09-24/21-11-32_sft_arms_coffee_arm1_full_ft_lrx1_seed42",
    ("coffee", "arm2"): "2026-09-20/04-16-11_sft_arms_coffee_arm2_frozen_llm_lrx1_seed42",
    ("coffee", "arm3"): "2026-09-20/18-11-00_sft_arms_coffee_arm3_lora_r32_lrx1_seed42",
    ("lamp", "arm4"): "2026-09-24/09-36-39_sft_arms_lamp_arm4_lora_vlm_full_expert_lrx1_seed42",
    ("coffee", "arm4"): "2026-09-23/22-49-49_sft_arms_coffee_arm4_lora_vlm_full_expert_lrx1_seed42",
}
TASKS = {
    "lamp": {
        "robocasa_task": "ScrewLightbulb",
        "environment": "lamp placement in kitchen seed 1 (layout 4, style 8)",
        "training_envs": ["s1_start_0", "s1_start_1"],
        "held_out_envs": ["s1_mid"],
        "dual_noise_training_group": "train",
        "dsrl_training_metric": "eval/success_rate_train",
        "dsrl_checkpoints": [2000, 4000, 8000, 16000],
        "sft_dataset_frames": 31842,
    },
    "coffee": {
        "robocasa_task": "CoffeePressButton",
        "environment": "kitchen (construction seed, layout, style)",
        "training_envs": ["seed0_L8S6", "seed1_L4S8"],
        "held_out_envs": ["seed11_L1S4", "seed14_L1S6"],
        "dual_noise_training_group": "reference",
        "dsrl_training_metric": "eval/success_rate_reference",
        "dsrl_checkpoints": [4000, 8000, 14000, 18000],
        "sft_dataset_frames": 7928,
        "note": "DSRL trains in 10 kitchens; training_envs are kitchens 0 and 1 of those 10, the only "
        "training kitchens scored at eval time.",
    },
}
SFT_BATCH = 32
EVAL_EVERY = 25_000
TOTAL = 500_000
FINAL_WINDOW = 100_000  # final performance = mean over evals with env_step > 400k


def r4(x):
    return None if x is None else round(float(x), 4)


def dual_noise():
    out, missing = {}, []
    for task, cfg in TASKS.items():
        g = cfg["dual_noise_training_group"]
        envs = cfg["training_envs"] + cfg["held_out_envs"]
        out[task] = {}
        for arm in ARMS:
            src = json.loads((HERE / f"dn/{task}_{arm}.json").read_text())
            rungs = {}
            for step in range(2000, 22000, 2000):
                row = src.get(str(step))
                if row is None:
                    missing.append(f"before-RL {task} {arm} {step}: checkpoint not scored")
                    continue
                modes = {}
                for mode in ("iid", "duplicated"):
                    m = row.get(mode)
                    if m is None:
                        missing.append(f"before-RL {task} {arm} {step}: {mode} noise not scored")
                        continue
                    per_env_n = m["n"] // len(envs)
                    per_env = {}
                    for label in envs:
                        key = next(k for k in m if k.startswith("pc_success_") and k.endswith(f"_{label}"))
                        per_env[label] = {"success_rate": r4(m[key] / 100), "n_episodes": per_env_n}
                    modes[mode] = {
                        "all_envs": {"success_rate": r4(m["pc_success"] / 100), "n_episodes": m["n"]},
                        "training_envs": {"success_rate": r4(m[f"pc_success_{g}"] / 100), "n_episodes": m[f"n_{g}"]},
                        "held_out_envs": {"success_rate": r4(m["pc_success_heldout"] / 100), "n_episodes": m["n_heldout"]},
                        "per_env": per_env,
                    }
                rungs[str(step)] = modes
            out[task][arm] = {"sft_run": SFT_RUN[(task, arm)], "checkpoints": rungs}
    return out, missing


def bucket(step):
    return int(round(step / EVAL_EVERY) * EVAL_EVERY)


def dsrl():
    runs = json.loads((HERE / "dsrl_wandb_evals.json").read_text())
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["arm"], r["checkpoint"], r["seed"])].append(r)
    out, missing, stitched = {}, [], []
    for task, cfg in TASKS.items():
        tm = cfg["dsrl_training_metric"]
        envs = cfg["training_envs"] + cfg["held_out_envs"]
        out[task] = {}
        for arm in ARMS:
            cks = {}
            for ck in cfg["dsrl_checkpoints"]:
                seeds = {}
                for seed in (0, 1, 2):
                    cand = sorted(groups.get((task, arm, ck, seed), []), key=lambda r: r["created"])
                    if not cand:
                        missing.append(f"DSRL {task} {arm} {ck // 1000}k seed {seed}: no run")
                        continue
                    by_step, source = {}, {}
                    for r in cand:  # later runs win where two runs logged the same eval step
                        for e in r["evals"]:
                            b = bucket(e["_step"])
                            if b == 0 or abs(e["_step"] - b) > 2_000:
                                continue  # off the 25k eval grid: a short test run, not the 500k run
                            by_step[b], source[b] = e, r["id"]
                    used = sorted(set(source.values()), key=[r["id"] for r in cand].index)
                    if len(used) > 1:
                        stitched.append(f"DSRL {task} {arm} {ck // 1000}k seed {seed}: evals come from runs {used}")
                    steps = sorted(by_step)
                    gaps = [s for s in range(EVAL_EVERY, TOTAL + 1, EVAL_EVERY) if s not in by_step]
                    if gaps:
                        missing.append(f"DSRL {task} {arm} {ck // 1000}k seed {seed}: no eval at "
                                       f"{', '.join(f'{g // 1000}k' for g in gaps)} env steps (wandb {used})")
                    per_env = {}
                    for label in envs:
                        key = next((k for k in cand[-1]["eval_keys"] if k.endswith(f"_{label}")), None)
                        per_env[label] = [r4(by_step[s].get(key)) if key else None for s in steps]
                    seeds[str(seed)] = {
                        "wandb_project": cand[-1].get("source", "real-lab/lerobot-dsrl"),
                        "wandb_runs": used,
                        "env_steps": steps,
                        "success_rate": {
                            "all_envs": [r4(by_step[s].get("eval/success_rate")) for s in steps],
                            "training_envs": [r4(by_step[s].get(tm)) for s in steps],
                            "held_out_envs": [r4(by_step[s].get("eval/success_rate_heldout")) for s in steps],
                        },
                        "per_env": per_env,
                    }
                cks[str(ck)] = {"seeds": seeds}
            out[task][arm] = {"sft_run": SFT_RUN[(task, arm)], "checkpoints": cks}
    return out, missing, stitched


def summary(dn, ds):
    rows = []
    for task, cfg in TASKS.items():
        for arm in ARMS:
            for ck in cfg["dsrl_checkpoints"]:
                before = dn[task][arm]["checkpoints"].get(str(ck), {})
                seeds = ds[task][arm]["checkpoints"][str(ck)]["seeds"]
                final = {}
                for grp in ("all_envs", "training_envs", "held_out_envs"):
                    per_seed = []
                    for sd in seeds.values():
                        vals = [v for s, v in zip(sd["env_steps"], sd["success_rate"][grp])
                                if s >= TOTAL - FINAL_WINDOW and v is not None]
                        if vals:
                            per_seed.append(st.mean(vals))
                    final[grp] = {
                        "mean": r4(st.mean(per_seed)) if per_seed else None,
                        "std_across_seeds": r4(st.stdev(per_seed)) if len(per_seed) > 1 else None,
                        "per_seed": [r4(v) for v in per_seed],
                    }
                rows.append({
                    "task": task,
                    "arm": arm,
                    "sft_checkpoint": ck,
                    "sft_epochs": round(ck * SFT_BATCH / cfg["sft_dataset_frames"], 1),
                    "before_rl": {
                        mode: {g: before[mode][g]["success_rate"] for g in ("all_envs", "training_envs", "held_out_envs")}
                        for mode in ("iid", "duplicated") if mode in before
                    },
                    "after_dsrl_final": final,
                    "n_seeds": len(seeds),
                })
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "arms": ARMS,
        "tasks": TASKS,
        "sft_batch_size": SFT_BATCH,
        "success_rate_units": "fraction in [0, 1]",
    }
    dn, dn_missing = dual_noise()
    ds, ds_missing, stitched = dsrl()
    json.dump({
        "meta": meta,
        "what": "Before RL: each SFT checkpoint on its own, scored on the same environments DSRL is evaluated in.",
        "noise_modes": {
            "iid": "fresh N(0,1) noise at every step of the action chunk (the SFT policy's own sampling)",
            "duplicated": "one N(0,1) vector held fixed across the action chunk (the input DSRL controls)",
        },
        "results": dn,
    }, open(OUT / "dual_noise.json", "w"), indent=1)
    json.dump({
        "meta": {**meta, "total_env_steps": TOTAL, "eval_every_env_steps": EVAL_EVERY,
                 "eval_episodes_per_env": 5, "actor_at_eval": "deterministic",
                 "source": "wandb real-lab/lerobot-dsrl (arms 0-3); forge.coreweave.com chacloutier-4B/lerobot-dsrl (coffee arm4); wandb.ai chacloutier-4B/lerobot-dsrl (lamp arm4, September-bank reruns `*_sepbank`, 2026-10-02)"},
        "what": "After DSRL: eval success over training, per RL seed. env_steps are rounded to the 25k eval grid.",
        "results": ds,
    }, open(OUT / "dsrl.json", "w"), indent=1)
    json.dump({
        "meta": meta,
        "final_performance": f"mean over the evals at {(TOTAL - FINAL_WINDOW) // 1000}k-{TOTAL // 1000}k env steps, per seed; then mean/std over seeds",
        "rows": summary(dn, ds),
    }, open(OUT / "summary.json", "w"), indent=1)
    json.dump({"missing": dn_missing + ds_missing, "stitched_from_multiple_wandb_runs": stitched},
              open(HERE / "missing.json", "w"), indent=1)
    print("before-RL missing:", len(dn_missing))
    for m in dn_missing:
        print("  ", m)
    print("DSRL missing:", len(ds_missing))
    for m in ds_missing:
        print("  ", m)
    print("stitched from several wandb runs:", len(stitched))
    for s in stitched:
        print("  ", s)
