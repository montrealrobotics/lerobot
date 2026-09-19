#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=dsrl_3k_seed

# ─────────────────────────────────────────────────────────────────────────────────────
# Is the cat=True 3k DSRL failure an unlucky RL seed, or does DSRL fail on this checkpoint?
#
# All four historical 3k runs used the default --seed 0 (two with an identical config) and all
# show the same shape: seed-0 kitchen never solved at any eval, seed-1 kitchen oscillating
# 0.9 -> 0.05 -> 1.0 -> 0.05 (the 10k value is the near-zero-noise initial actor, which the frozen
# policy already solves on seed 1). Yet random DSRL-parameterized noise succeeds on seed 0 6/64 and
# on seed 1 25/64 (steerability eval), so the exploration signal exists.
#
# Reruns differ from the final 3k run (wandb qbzjxs7o) ONLY in --seed (network init, warmup noise,
# SAC sampling; construction-seed kitchens are unaffected), total_steps (200k covers the region
# where every previous run had already failed/oscillated), and a cheaper eval (every 10k, 5
# deterministic episodes per kitchen). Per-kitchen evals + per-episode collection success + SAC
# losses are logged for diagnosing seed-1 instability vs seed-0 never-learned.
#
# Usage: sbatch dsrl_seed_reruns_lightbulb_3k.sbatch <rl_seed>
# ─────────────────────────────────────────────────────────────────────────────────────

set -uo pipefail
RL_SEED="${1:?rl seed (previous runs all used 0)}"

source ~/.bashrc
cd ~/lerobot
source .venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
if ! ls /cvmfs/ai.mila.quebec >/dev/null 2>&1; then
    echo "WARNING: /cvmfs unreadable on $(hostname); removing it from PATH"
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v cvmfs | paste -sd:)
fi
uv pip install --quiet numpy==2.2.5

SFT=~/scratch/lerobot/outputs/train/2026-08-28/15-26-13_robocasa_pi05/checkpoints/003000/pretrained_model
[ -f "$SFT/model.safetensors" ] || { echo "missing checkpoint $SFT"; exit 1; }
CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")

OUT=~/scratch/lerobot/outputs/dsrl_lightbulb_seed_reruns/cat_true_3k_seed${RL_SEED}
if compgen -G "$OUT/step_*" > /dev/null; then
    echo "REFUSING: $OUT already contains checkpoints."; exit 1
fi
mkdir -p "$OUT"
NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
    --policy_path "$SFT" \
    --task ScrewLightbulb \
    --robot XArm6DexLeapRHOmron \
    --controller "$CONTROLLER" \
    --fps 30 \
    --prompt "screw the light bulb into the lamp" \
    --total_steps 200000 \
    --min_buffer_size 500 \
    --reward_mode goal \
    --collect_envs 2 \
    --utd_ratio 40 \
    --eval_seeds 0,1 \
    --eval_episodes 5 \
    --eval_freq 10000 \
    --save_freq 10000 \
    --log_freq 100 \
    --seed "$RL_SEED" \
    --output_dir "$OUT" \
    --wandb_enable \
    --wandb_project lerobot-dsrl-robocasa \
    --wandb_name "dsrl_pi05_lightbulb_3k_seed${RL_SEED}" \
    2>&1 | tee "$OUT/run.log"
