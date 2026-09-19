#!/bin/bash
#SBATCH --partition long
#SBATCH -c 12
#SBATCH --mem=64G                 # 16 resident kitchens (8 parallel rollouts x 2 starts)
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=steer_lightbulb

# ─────────────────────────────────────────────────────────────────────────────────────
# Best-of-K steerability of FROZEN ScrewLightbulb SFT checkpoints (inference only).
#
# Measured on the starts the fixed-start DSRL runs trained in (construction seeds 0,1), where the
# standard SFT metric is anti-predictive of the DSRL outcome. Each rollout draws fresh randn noise
# per action chunk (DSRL's warmup sampler) through the DSRL harness itself:
#   MODE=dsrl  --noise_chunk_size 1   one vector held across the chunk (the space DSRL steers in)
#   MODE=iid   --noise_chunk_size 32  fresh noise per timestep (vanilla pi05; control)
# 64 rollouts per start -> unbiased pass@k for k=1..64 plus mean success.
#
# Checkpoints: the frozen ones DSRL used, plus a later best-eval checkpoint per approach
# (odd-step SFT checkpoints have been pruned from both runs, so 16k/20k stand in for 11k/15k).
#
# Usage: sbatch steerability_eval_lightbulb.sbatch <cat_true_3k|..._7k|..._16k|cat_false_2k|..._8k|..._20k> <dsrl|iid>
# ─────────────────────────────────────────────────────────────────────────────────────

set -uo pipefail
ARM="${1:?arm}"
MODE="${2:?dsrl | iid}"

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

TRAIN=~/scratch/lerobot/outputs/train
TRUE_RUN="$TRAIN/2026-08-28/15-26-13_robocasa_pi05/checkpoints"
FALSE_RUN="$TRAIN/2026-08-31/18-26-31_robocasa_pi05/checkpoints"
case "$ARM" in
    cat_true_3k)   SFT="$TRUE_RUN/003000" ;;
    cat_true_7k)   SFT="$TRUE_RUN/007000" ;;
    cat_true_16k)  SFT="$TRUE_RUN/016000" ;;
    cat_false_2k)  SFT="$FALSE_RUN/002000" ;;
    cat_false_8k)  SFT="$FALSE_RUN/008000" ;;
    cat_false_20k) SFT="$FALSE_RUN/020000" ;;
    *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
case "$MODE" in
    dsrl) NC=1 ;;
    iid)  NC=32 ;;
    *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
[ -f "$SFT/pretrained_model/model.safetensors" ] || { echo "missing checkpoint $SFT"; exit 1; }
CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")

OUT=~/scratch/lerobot/outputs/steerability/lightbulb/${ARM}
mkdir -p "$OUT"
NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
python examples/tutorial/rl/dsrl_pi05_robocasa_example.py --steer_eval \
    --policy_path "$SFT/pretrained_model" \
    --task ScrewLightbulb \
    --robot XArm6DexLeapRHOmron \
    --controller "$CONTROLLER" \
    --fps 30 \
    --prompt "screw the light bulb into the lamp" \
    --reward_mode goal \
    --eval_seeds 0,1 \
    --noise_chunk_size "$NC" \
    --steer_rollouts_per_start 64 \
    --steer_parallel_per_start 8 \
    --seed 0 \
    --output_dir "$OUT" \
    --wandb_enable \
    --wandb_project lerobot-dex \
    --wandb_name "steer_lightbulb_${ARM}_${MODE}" \
    2>&1 | tee "$OUT/steer_${MODE}.log"
