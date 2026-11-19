import wandb
import numpy as np
import pandas as pd
import os


api = wandb.Api()


# Group your run IDs by checkpoint
# Format: {checkpoint_step: [seed0_run_id, seed1_run_id, seed2_run_id]}
COFFEE_RUNS = {
   4000:  ["z3f2ceym", "sjpo10p2", "u2tr4qda"],
   8000:  ["baqx1igq", "b39i7s0c", "k7cztkwd"],
   14000: ["xyk6qd8v", "g6kxlaui", "yb47d875"],
   18000: ["zyk2zrpb", "pmhau577", "ms0i0ilc"],
}


LAMP_RUNS = {
   2000:  ["w9wtw86b", "enrey7u1", "hpiu4584"],
   4000:  ["5l6gl74q", "8si7w1bx", "iqqqpqcy"],
   8000:  ["0gjtolju", "tegixvlk", "cwkwwcb1"],
   16000: ["am1nwt30", "30pedpgb", "zepe5bc6"],
}


ENTITY = "chacloutier-4B"
PROJECT = "lerobot-dsrl"


def get_last_100k_mean(run_id, train_key, heldout_key):
   run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
   df = run.history(keys=[train_key, heldout_key, "_step"])
   total_steps = df["_step"].max()
   last = df[df["_step"] >= total_steps - 100000]
   train = last[train_key].dropna().mean()
   heldout = last[heldout_key].dropna().mean()
   return train, heldout


def summarize(task_runs, task_name, train_key, heldout_key):
   print(f"\n=== {task_name} ===")
   print(f"{'checkpoint':>12}  {'train_mean':>12}  {'heldout_mean':>14}  {'n_seeds':>8}")
   for step, run_ids in sorted(task_runs.items()):
       trains, heldouts = [], []
       for run_id in run_ids:
           try:
               t, h = get_last_100k_mean(run_id, train_key, heldout_key)
               trains.append(t)
               heldouts.append(h)
           except Exception as e:
               print(f"  step {step} run {run_id} failed: {e}")
       if trains:
           print(f"{step:>12}  {np.mean(trains)*100:>11.1f}%  {np.mean(heldouts)*100:>13.1f}%  {len(trains):>8}")


summarize(COFFEE_RUNS, "CoffeePressButton arm4", "eval/success_rate_reference", "eval/success_rate_heldout")
summarize(LAMP_RUNS, "ScrewLightbulb arm4", "eval/success_rate_train", "eval/success_rate_heldout")


COFFEE_KEYS = ["_step", "eval/success_rate", "eval/success_rate_reference", "eval/success_rate_heldout"]
LAMP_KEYS   = ["_step", "eval/success_rate", "eval/success_rate_train", "eval/success_rate_heldout"]


os.makedirs("dsrl_history", exist_ok=True)


def fetch_history(task_runs, task_name, keys, train_key, heldout_key):
   print(f"\n=== {task_name} ===")
   print(f"{'checkpoint':>12}  {'train_mean':>12}  {'heldout_mean':>14}  {'n_seeds':>8}")


   for step, run_ids in sorted(task_runs.items()):
       trains, heldouts = [], []
       for i, run_id in enumerate(run_ids):
           try:
               run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
               df = run.history(keys=keys)
               df["sft_step"] = step
               df["seed"] = i
               df["run_id"] = run_id


               # Save full history to CSV
               fname = f"dsrl_history/{task_name.replace(' ', '_')}_{step}_s{i}_{run_id}.csv"
               df.to_csv(fname, index=False)


               # Also compute summary
               total_steps = df["_step"].max()
               last = df[df["_step"] >= total_steps - 100000]
               trains.append(last[train_key].dropna().mean())
               heldouts.append(last[heldout_key].dropna().mean())


           except Exception as e:
               print(f"  step {step} run {run_id} failed: {e}")


       if trains:
           print(f"{step:>12}  {np.mean(trains)*100:>11.1f}%  {np.mean(heldouts)*100:>13.1f}%  {len(trains):>8}")


fetch_history(COFFEE_RUNS, "coffee", COFFEE_KEYS, "eval/success_rate_reference", "eval/success_rate_heldout")
fetch_history(LAMP_RUNS,   "lamp",   LAMP_KEYS,   "eval/success_rate_train",     "eval/success_rate_heldout")


print("\nCSVs saved to dsrl_history/")
