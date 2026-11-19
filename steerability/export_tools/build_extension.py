"""Build the extension files of the results export from ext_raw/ (filled by fetch_ext.sh).

Two experiments, added next to the original dual_noise.json / dsrl.json / summary.json:

  lamp_extrap     the existing lamp DSRL runs, scored again on two EXTRAPOLATED lamp placements
                  (one pair-spacing past each training start). Before RL: every SFT rung on those
                  two cells. After DSRL: each run's final noise actor on all five cells.
  coffee_lateral  a NEW coffee setup: one kitchen, the coffee machine slid sideways. DSRL trains
                  on two offsets; held out = their midpoint; extrap = one spacing past each end.
                  Before RL: every SFT rung on the five cells. After DSRL: the in-loop eval curves.

Missing data is listed in missing_ext.json rather than failing, so this can run on partial results.

Our copy (Artur's original: build_extension.py.orig): builds ONLY arm4, from ext_raw_arm4/ (filled from
the arm4 drop folder by fetch_ext.sh), and splices it into his outputs in ../export_artur/, whose arms
0-3 entries are kept unchanged (our ext_raw/ lacks some of their raw files). Writes ../export_ext_rebuilt/.
Lamp before-RL training/held-out cells for the summary come from ../dual_noise.json (regenerated, sepbank).
"""

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "ext_raw_arm4"
OUT = HERE.parent / "export_ext_rebuilt"
BASE = HERE.parent / "export_artur"  # Artur's extension outputs; arms 0-3 taken from here verbatim
DUAL_NOISE = HERE.parent / "dual_noise.json"
ARMS = ("arm4",)  # arms built here; every other arm comes from BASE
RUNGS = list(range(2000, 22000, 2000))
TOTAL, EVAL_EVERY, FINAL_WINDOW = 500_000, 25_000, 100_000

LAMP = {
    "robocasa_task": "ScrewLightbulb",
    "bank": "placement_banks/screwlightbulb_xarm6_seed1_pair_extrap/bank.json",
    "environment": "lamp placement in kitchen seed 1 (layout 4, style 8)",
    "groups": {
        "training_envs": ["s1_start_0", "s1_start_1"],
        "held_out_envs": ["s1_mid"],
        "extrap_envs": ["s1_extrap_0_x1", "s1_extrap_1_x1"],
    },
    "extrap_note": "1x the training-pair separation (7.4 cm) past each training start, continuing the "
    "pair's (position, yaw) line. s1_extrap_0_x1: past s1_start_0, yaw -4.2 deg. s1_extrap_1_x1: past "
    "s1_start_1, yaw -175.6 deg; also harder for the frozen SFT policy itself.",
    "dsrl_checkpoints": [2000, 4000, 8000, 16000],
    "before_rl_cells_here": "extrap_envs only; training/held-out cells are in dual_noise.json",
}
COFFEE = {
    "robocasa_task": "CoffeePressButton",
    "bank": "placement_banks/coffeepressbutton_pandadex_s1_lateral/bank.json",
    "environment": "kitchen seed 1 (layout 4, style 8, nespresso); coffee machine (and mug) slid sideways",
    "axis": "lateral offset of the machine; + = the robot's left",
    "groups": {
        "training_envs": ["s1_lat-4cm", "s1_lat+0cm"],
        "held_out_envs": ["s1_lat-2cm"],
        "extrap_envs": ["s1_lat-8cm", "s1_lat+4cm"],
    },
    "dsrl_checkpoints": [4000, 8000, 14000, 18000],
    "start_rejection": "10 deg / 3 cm, up to 20 retries",
}
GROUPS = ("training_envs", "held_out_envs", "extrap_envs")


def r4(x):
    return None if x is None else round(float(x), 4)


def arm_of(name):
    m = re.search(r"arm\d", name)
    return m.group() if m else None


# ── Before RL ─────────────────────────────────────────────────────────────────────────

def load_dual_noise(suffix):
    """{arm: {step: {mode: row}}}, merged over clusters (later file wins per step)."""
    rows, runs = defaultdict(dict), {}
    for f in sorted((RAW / "dn").glob(f"*__dual_noise_{suffix}.json")):
        cluster, run, _ = f.name.split("__")
        arm = arm_of(run)
        for step, modes in json.loads(f.read_text()).items():
            rows[arm][int(step)] = modes
        runs[arm] = f"{cluster}:{run}"
    return rows, runs


def dn_block(src, cfg, groups):
    out, missing = {}, []
    for arm in ARMS:
        rungs = {}
        for step in RUNGS:
            row = src.get(arm, {}).get(step)
            if row is None:
                missing.append(f"before-RL {arm} {step}: not scored")
                continue
            modes = {}
            for mode in ("iid", "duplicated"):
                m = row.get(mode)
                if m is None:
                    missing.append(f"before-RL {arm} {step}: {mode} not scored")
                    continue
                cells = [k for k in m if k.startswith("pc_success_s")]
                n_cell = m["n"] // len(cells)
                per_env = {
                    lab: {"success_rate": r4(m[f"pc_success_{lab}"] / 100), "n_episodes": n_cell}
                    for g in groups for lab in cfg["groups"][g] if f"pc_success_{lab}" in m
                }
                block = {"per_env": per_env}
                for g in groups:  # recomputed from cells: a row may come from a bank with extra cells
                    labs = [lab for lab in cfg["groups"][g] if lab in per_env]
                    if len(labs) == len(cfg["groups"][g]):
                        block[g] = {
                            "success_rate": r4(st.mean(per_env[lab]["success_rate"] for lab in labs)),
                            "n_episodes": n_cell * len(labs),
                        }
                modes[mode] = block
            rungs[str(step)] = modes
        out[arm] = {"checkpoints": rungs}
    return out, missing


# ── After DSRL ────────────────────────────────────────────────────────────────────────

KV = re.compile(r"(success_rate\S*)\s*=\s*([\d.]+)")


def lamp_dsrl():
    out, missing = {}, []
    found = {}
    for f in (RAW / "lamp_dsrl").glob("*.log"):
        m = re.search(r"dsrl_(arm\d)_(\d+)_s(\d)\.log$", f.name)
        if not m:
            continue
        text = f.read_text(errors="replace")
        if "==== eval_only results ====" not in text:
            continue
        vals = {k.removeprefix("success_rate_"): float(v) for k, v in KV.findall(text.split("==== eval_only results ====")[-1])}
        found[(m[1], int(m[2]), int(m[3]))] = vals
    for arm in ARMS:
        cks = {}
        for ck in LAMP["dsrl_checkpoints"]:
            seeds = {}
            for seed in (0, 1, 2):
                v = found.get((arm, ck, seed))
                if v is None:
                    missing.append(f"DSRL lamp_extrap {arm} {ck // 1000}k seed {seed}: final actor not scored")
                    continue
                per_env = {lab: r4(v.get(lab)) for g in GROUPS for lab in LAMP["groups"][g]}
                seeds[str(seed)] = {
                    "per_env": per_env,
                    **{g: r4(st.mean(per_env[lab] for lab in LAMP["groups"][g])) for g in GROUPS},
                }
            cks[str(ck)] = {"seeds": seeds}
        out[arm] = {"checkpoints": cks}
    return out, missing


def coffee_dsrl():
    evals = defaultdict(dict)  # (arm, ck, seed) -> {env_step: {key: v}}
    for f in sorted((RAW / "coffee_dsrl").iterdir()):
        m = re.search(r"dsrl_(arm\d)_coffee_lat_0*(\d+)(?:_s(\d)|__step\d+_seed(\d))", f.name)
        if not m:
            continue
        key = (m[1], int(m[2]), int(m[3] if m[3] is not None else m[4]))
        for line in f.read_text(errors="replace").splitlines():
            e = re.match(r"\[eval @ step (\d+)\] (.*)", line.strip())
            if e:
                evals[key][int(e[1])] = {k: float(v) for k, v in re.findall(r"(\S+)=([\d.]+)", e[2])}
    out, missing = {}, []
    for arm in ARMS:
        cks = {}
        for ck in COFFEE["dsrl_checkpoints"]:
            seeds = {}
            for seed in (0, 1, 2):
                ev = evals.get((arm, ck, seed))
                if not ev:
                    missing.append(f"DSRL coffee_lateral {arm} {ck // 1000}k seed {seed}: no evals yet")
                    continue
                by = {}
                for s, row in ev.items():
                    b = int(round(s / EVAL_EVERY) * EVAL_EVERY)
                    if b and abs(s - b) <= 2_000:
                        by[b] = row
                steps = sorted(by)
                if not steps or steps[-1] < TOTAL:
                    missing.append(f"DSRL coffee_lateral {arm} {ck // 1000}k seed {seed}: last eval at "
                                   f"{(steps[-1] if steps else 0) // 1000}k of {TOTAL // 1000}k")
                labs = [lab for g in GROUPS for lab in COFFEE["groups"][g]]
                seeds[str(seed)] = {
                    "env_steps": steps,
                    "success_rate": {
                        "all_envs": [r4(by[s].get("success_rate")) for s in steps],
                        "training_envs": [r4(by[s].get("success_rate_train")) for s in steps],
                        "held_out_envs": [r4(by[s].get("success_rate_heldout")) for s in steps],
                        "extrap_envs": [r4(by[s].get("success_rate_extrap")) for s in steps],
                    },
                    "per_env": {lab: [r4(by[s].get(f"success_rate_{lab}")) for s in steps] for lab in labs},
                }
            cks[str(ck)] = {"seeds": seeds}
        out[arm] = {"checkpoints": cks}
    return out, missing


# ── Summary ───────────────────────────────────────────────────────────────────────────

def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    return {
        "mean": r4(st.mean(vals)) if vals else None,
        "std_across_seeds": r4(st.stdev(vals)) if len(vals) > 1 else None,
        "per_seed": [r4(v) for v in vals],
    }


def summary(lamp_dn, lamp_ds, cof_dn, cof_ds):
    base = json.loads(DUAL_NOISE.read_text())["results"]["lamp"]
    rows = []
    for arm in ARMS:
        for ck in LAMP["dsrl_checkpoints"]:
            before = {}
            for mode in ("iid", "duplicated"):
                b = base.get(arm, {}).get("checkpoints", {}).get(str(ck), {}).get(mode)
                x = lamp_dn[arm]["checkpoints"].get(str(ck), {}).get(mode, {}).get("extrap_envs")
                before[mode] = {
                    "training_envs": b["training_envs"]["success_rate"] if b else None,
                    "held_out_envs": b["held_out_envs"]["success_rate"] if b else None,
                    "extrap_envs": x["success_rate"] if x else None,
                }
            seeds = lamp_ds[arm]["checkpoints"][str(ck)]["seeds"]
            rows.append({
                "task": "lamp_extrap", "arm": arm, "sft_checkpoint": ck, "before_rl": before,
                "after_dsrl_final": {g: mean_sd([s[g] for s in seeds.values()]) for g in GROUPS},
                "after_dsrl_measure": "final noise actor, eval-only, 10 episodes per cell",
                "n_seeds": len(seeds),
            })
        for ck in COFFEE["dsrl_checkpoints"]:
            b = cof_dn[arm]["checkpoints"].get(str(ck), {})
            before = {mode: {g: b[mode][g]["success_rate"] if g in b.get(mode, {}) else None for g in GROUPS}
                      for mode in ("iid", "duplicated")}
            seeds = cof_ds[arm]["checkpoints"][str(ck)]["seeds"]
            final = {}
            for g in GROUPS:
                per_seed = []
                for sd in seeds.values():
                    v = [x for s, x in zip(sd["env_steps"], sd["success_rate"][g]) if s >= TOTAL - FINAL_WINDOW and x is not None]
                    per_seed.append(st.mean(v) if v else None)
                final[g] = mean_sd(per_seed)
            rows.append({
                "task": "coffee_lateral", "arm": arm, "sft_checkpoint": ck, "before_rl": before,
                "after_dsrl_final": final,
                "after_dsrl_measure": f"mean of in-loop evals at {(TOTAL - FINAL_WINDOW) // 1000}k-{TOTAL // 1000}k env steps",
                "n_seeds": len(seeds),
            })
    return rows


if __name__ == "__main__":
    lamp_src, lamp_runs = load_dual_noise("lamp_extrap")
    cof_src, cof_runs = load_dual_noise("coffee_lateral")
    lamp_dn, m1 = dn_block(lamp_src, LAMP, ("extrap_envs",))
    cof_dn, m2 = dn_block(cof_src, COFFEE, GROUPS)
    lamp_ds, m3 = lamp_dsrl()
    cof_ds, m4 = coffee_dsrl()
    modes = {
        "iid": "fresh N(0,1) noise at every step of the action chunk (the SFT policy's own sampling)",
        "duplicated": "one N(0,1) vector held fixed across the action chunk (the input DSRL controls)",
    }
    OUT.mkdir(exist_ok=True)

    def dump(name, obj):
        """Splice the arms built here into Artur's file; everything else stays as in BASE."""
        old = json.loads((BASE / name).read_text())
        assert old["meta"] == obj["meta"], f"{name}: meta differs from {BASE}"
        if "rows" in obj:
            new = {(r["task"], r["arm"], r["sft_checkpoint"]): r for r in obj["rows"]}
            old["rows"] = [new.pop((r["task"], r["arm"], r["sft_checkpoint"]), r) for r in old["rows"]]
            assert not new, f"{name}: rows with no slot in {BASE}: {list(new)}"
        else:
            for key in ("results", "sft_runs"):
                for arm, v in obj.get(key, {}).items():
                    old[key][arm] = v
        json.dump(old, open(OUT / name, "w"), indent=1)

    dump("dual_noise_lamp_extrap.json", {"meta": LAMP, "noise_modes": modes, "sft_runs": lamp_runs, "results": lamp_dn})
    dump("dsrl_lamp_extrap.json", {"meta": LAMP, "what": "Each existing lamp DSRL run's FINAL noise actor, "
         "eval-only, deterministic, 10 episodes per cell.", "results": lamp_ds})
    dump("dual_noise_coffee_lateral.json", {"meta": COFFEE, "noise_modes": modes, "sft_runs": cof_runs, "results": cof_dn})
    dump("dsrl_coffee_lateral.json", {"meta": {**COFFEE, "total_env_steps": TOTAL, "eval_every_env_steps": EVAL_EVERY,
         "eval_episodes_per_env": 5, "actor_at_eval": "deterministic"},
         "what": "New DSRL runs on the lateral coffee bank: in-loop eval success over training, per RL seed.",
         "results": cof_ds})
    dump("summary_ext.json", {"meta": {"lamp_extrap": LAMP, "coffee_lateral": COFFEE}, "rows": summary(lamp_dn, lamp_ds, cof_dn, cof_ds)})
    missing = {"lamp_extrap_before_rl": m1, "coffee_lateral_before_rl": m2, "lamp_extrap_dsrl": m3, "coffee_lateral_dsrl": m4}
    json.dump(missing, open(OUT / "missing_ext.json", "w"), indent=1)
    for k, v in missing.items():
        print(f"{k}: {len(v)} missing")
