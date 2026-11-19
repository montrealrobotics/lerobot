# SFT arms × DSRL: results export

Five ways of fine-tuning pi05 (the "arms"), on two RoboCasa tasks, each measured
**before RL** (the SFT checkpoint on its own) and **after DSRL** (steered with RL).
All success rates are fractions in [0, 1]. Every file has a `meta` block with the arm
and task definitions.

| file | what's in it |
|---|---|
| `dual_noise.json` | Before RL, every SFT checkpoint (2k–20k), under both noise modes |
| `dsrl.json` | After DSRL, eval curves over 500k env steps, per RL seed |
| `summary.json` | One row per DSRL run checkpoint: before-RL and final after-DSRL success side by side |
| `make_figures.py` | Redraws everything in `figures/` from these files (`python make_figures.py`) |
| `figures/` | PNG + PDF. Per arm and task: `dsrl_<task>_<arm>` (DSRL curves, one line per SFT checkpoint, mean over 3 seeds), `before_rl_<task>_<arm>_iid` and `_repeated` (SFT alone, across the checkpoint ladder). Per task: `overview_before_rl_<task>_<mode>` (all arms) |

## Terms

- **Arms:** arm0 expert only, arm1 full fine-tune, arm2 frozen LLM, arm3 LoRA r32,
  arm4 LoRA on the language model with the action expert fully fine-tuned.
- **Environments.** ScrewLightbulb (`lamp`): one kitchen, and an environment is a lamp
  placement. Two training environments and one held-out environment between them.
  CoffeePressButton (`coffee`): an environment is a kitchen. DSRL trains in ten
  kitchens; the training-environment numbers use kitchens 0 and 1 of those ten, and
  kitchens 11 and 14 are held out.
- **Noise modes (before RL).** `duplicated`: one random noise vector held fixed across
  each action chunk, the input DSRL learns to control. `iid`: fresh noise at every step,
  the SFT policy's normal sampling.
- **Final performance (after DSRL):** mean success over the evals at 400k–500k env
  steps, per seed, then mean and std over the 3 seeds.

## Shapes

```
dual_noise.json  results[task][arm].checkpoints["2000"][mode]
                   -> {all_envs, training_envs, held_out_envs, per_env[env]}
                      each {success_rate, n_episodes}

dsrl.json        results[task][arm].checkpoints["2000"].seeds["0"]
                   -> env_steps: [25000, 50000, ..., 500000]
                      success_rate: {all_envs, training_envs, held_out_envs}: [one value per env_step]
                      per_env[env]: [one value per env_step]
                      wandb_runs: run ids in real-lab/lerobot-dsrl

summary.json     rows[]: {task, arm, sft_checkpoint, sft_epochs,
                          before_rl[mode][group], after_dsrl_final[group]{mean, std_across_seeds, per_seed},
                          n_seeds}
```

DSRL was run from checkpoints 2k, 4k, 8k, 16k (lamp) and 4k, 8k, 14k, 18k (coffee).
Before-RL checkpoints have 25 episodes per environment (20 for arm4; `n_episodes` has
the exact count); DSRL evals have 5 per environment with a deterministic policy.
Arms 0–3 were logged to wandb `real-lab`, arm4 to `chacloutier-4B` on forge.coreweave.com
(`wandb_project` on each DSRL seed says which).

Four coffee arm0 seeds have no eval at 500k (4k seeds 0–2, 8k seed 0); their final
performance averages the evals at 400k–475k.
