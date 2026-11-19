# Steerability diagnostics for the SFT-arms × DSRL study

These scripts test *why* one SFT checkpoint steers better than another under DSRL. DSRL success is broken into four parts, each measured separately:

- **Existence**: does any noise the actor can emit lead to success?
- **Discovery**: does exploration find it?
- **Learnability**: is the noise→outcome map smooth enough to climb?
- **Transfer**: does the steering carry over to other cells?

`analyze.py` then checks which measurement explains the DSRL outcomes *beyond* what dual-noise success already explains.

| phase | script | question | needs | rough cost / checkpoint |
|---|---|---|---|---|
| 0 | `p0_wandb_signals.py` | failed seeds: never saw success (discovery) or saw it and lost it (learnability)? | wandb | minutes, all runs at once |
| — | `collect_states.py` | saves sim states + obs along base rollouts (input to 1 and 2) | env | ~4 rollouts × cells |
| 1a | `p1_noise_sensitivity.py` | does the action still depend on the noise? | GPU only | ~5–10 min |
| 1b | `p1_box_reachability.py` | does success survive noise restricted to [-1,1]? | env | 4 kinds × 20 ep × cells |
| 1c | `p1_obs_reliance.py` | does it still look at / tell apart its inputs? drift vs pi05_base | GPU only | ~5 min + one cached base pass |
| 2a | `p2_branching.py` | how much does choosing the noise help, state by state? | env + restore | cells × states × K × R partial rollouts |
| 2b | `p2_smoothness.py` | is the good-noise region a basin or isolated pockets? | 2a output | ~1/3 of 2a |
| 2c | `p2_transfer.py` | do good noises from one cell work in another? | 2a output | small |
| — | `analyze.py` | join everything, partial correlations, leave-one-arm-out | JSONs | seconds |

Phase 2a defaults (5 states × 16 noises × 2 repeats × 3 lamp cells) mean 480 partial rollouts. At ~43 s per full episode that can approach 5 h per checkpoint, so for a first pass use `--k 8 --states-per-cell 3`.

## 1. Wiring (once)

The RoboCasa adapter needs two hooks from your repo. See `wiring_example.py`.

- `--env-factory module:fn`: `fn(task, cell, seed)` returns an env that always resets into that bank cell, with the same construction as `eval_sft_on_banks.py` and the coffee start-pose rejection. Its `step`/`reset` must return robosuite obs dicts.
- `--obs-fn module:fn`: `fn(task, obs)` returns the batch the frozen pi05 sees in `DSRLEnvWrapper`: 2 cameras, state, and prompt.

Before using anything real, run the test with the mock backend. It needs no GPU, MuJoCo or lerobot:

```bash
printf "lamp\tarm0\tmock:steer=0.9\nlamp\tarm2\tmock:steer=0.2,mem=0.8\n" > /tmp/m.tsv
python collect_states.py --task lamp --arm ref --sft-step 0 --checkpoint mock:steer=0.8 --out /tmp/d --adapter mock --label ref_lamp
for ph in states noise obs box branching smoothness transfer; do ./run_ladder.sh $ph /tmp/m.tsv /tmp/d; done
python analyze.py --out /tmp/d --dsrl-json dsrl.json --dual-noise-json dual_noise.json
```

(For `obs` with the mock backend, add `--reference-checkpoint mock:steer=1`.)

Then check that state restore is deterministic on the real env before trusting Phase 2:

```bash
python p2_branching.py ... --states-per-cell 1 --k 2 --check-determinism
# must print max_state_diff == 0 (or ~1e-9). If not, fix RobocasaAdapter._reset_controllers.
```

## 2. Run order

```bash
export ENV_FACTORY=scripts.steerability_wiring:make_cell_env OBS_FN=scripts.steerability_wiring:pi05_frame
# manifest.tsv:  task<TAB>arm<TAB>sft_run_dir   (one line per arm and task)

# Phase 0 -- free. Run --list-keys first and fix --key-map if a signal resolves to [].
python p0_wandb_signals.py --dsrl-json dsrl.json --out diag --list-keys lamp/arm1/4000/0
python p0_wandb_signals.py --dsrl-json dsrl.json --out diag --hosts "forge=https://forge.coreweave.com"

# Reference state sets (one per task, shared by every checkpoint in Phase 1).
python collect_states.py --task lamp   --arm arm0 --sft-step 16000 --checkpoint <arm0 lamp>/checkpoints/016000/pretrained_model   --out diag --label ref_lamp   --env-factory $ENV_FACTORY --obs-fn $OBS_FN
python collect_states.py --task coffee --arm arm0 --sft-step 14000 --checkpoint <arm0 coffee>/checkpoints/014000/pretrained_model --out diag --label ref_coffee --env-factory $ENV_FACTORY --obs-fn $OBS_FN

# Phase 1 over the DSRL ladders (LAUNCH="sbatch ... --wrap" to fan out jobs)
./run_ladder.sh noise manifest.tsv diag --demo-repo-id <lamp/coffee dataset>   # memorization is optional
./run_ladder.sh obs   manifest.tsv diag
./run_ladder.sh box   manifest.tsv diag

# Phase 2: each checkpoint's own states, then branching -> smoothness -> transfer
./run_ladder.sh states    manifest.tsv diag
./run_ladder.sh branching manifest.tsv diag --k 8 --states-per-cell 3
./run_ladder.sh smoothness manifest.tsv diag
./run_ladder.sh transfer   manifest.tsv diag

python analyze.py --out diag --dsrl-json dsrl.json --dual-noise-json dual_noise.json
```

`run_ladder.sh` runs only the DSRL checkpoints (lamp 2k/4k/8k/16k, coffee 4k/8k/14k/18k), because those are the ones with outcomes to explain. To restrict a run, put the arms you want in their own manifest. The `--demo-repo-id` above is task-specific, so for the memorization metrics run `noise` with a manifest containing one task at a time.

## 3. Outputs

```
diag/states/<tag>.pkl                    sim states + obs + probe targets (pickle)
diag/p0/seed_signals.csv                 one row per DSRL seed
diag/p0/ckpt_signals.csv                 per-checkpoint means (joined by analyze.py)
diag/p0/split_checkpoints.md             solved vs failed seeds, side by side
diag/<phase>/<task>_<arm>_<step>.json    {meta, per_env: {cell: {metric: value}}, ...details}
diag/analysis/table.csv                  one row per (task, arm, ckpt, env): targets + every metric
diag/analysis/correlations.csv, report.md
```

In `report.md`, read **`partial_given_dup`** first: it measures what a diagnostic adds beyond dual-noise success. Then check **`loao_dR2`** (> 0 means it predicts an *unseen arm* better than dup alone). Treat a large raw `rho` that disappears in the partial as dual-noise success relabeled.

As a reference point, the existing data already gives one result. For coffee time-to-0.8, iid success has ρ = −0.65 against −0.42 for dup, with partial −0.57 and positive `loao_dR2`. For lamp `final`, dup alone has ρ = 0.84.

## 4. Caveats built into the design

- **Model space.** Geometry metrics (1a, 1c) are computed on `predict_action_chunk` output before the postprocessor, so dimensions are comparable. Memorization uses env-space actions standardized per dimension by the demo statistics.
- **Common random numbers everywhere.** The noise sets are seeded identically across checkpoints. Box reachability uses the same reset seeds for every noise kind. Branching uses the same continuation seed for every z at a given state. As a result, differences come from the checkpoint or the noise, not from sampling luck.
- **`best_split`** chooses the best noise on one repeat and scores it on the other. The plain max over K noises is biased upward, more so for noisy checkpoints.
- **CKA for frozen modules is 1 by construction.** For example, the projector is frozen in arm0. That is expected, not a finding.
- **`pi05_base` has its own chunk size.** `features()` handles this by using the reference model's own chunk size. The base model receives batches built by the evaluated checkpoint's preprocessor.
- **Probe R²** selects the ridge λ by cross-validation on the same folds. It is mildly optimistic, but the bias is the same for every checkpoint.
- **Coffee transfer** matches states by macro step across kitchens with different layouts. Read it as "same task phase," not "same pose."
- **Phase 0 key names are regexes.** `--list-keys` shows what was resolved, and a signal that resolves to `[]` is simply absent from the output.
