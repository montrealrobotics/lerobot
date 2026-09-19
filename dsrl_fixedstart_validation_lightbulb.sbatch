#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=dsrl_validation_lightbulb

# ─────────────────────────────────────────────────────────────────────────────────────
# Fixed-start DSRL on a LATER "best eval" SFT checkpoint, to validate the steerability metric
# (steerability_eval_lightbulb.sbatch) against an actual DSRL outcome, and to test "select by
# steerability" against "take the best-eval / final checkpoint". Flags are copied verbatim from
# the most recent old fixed-start lamp runs (wandb argv of dsrl_lightbulb_parallel/embodiment_false/8k),
# so these are directly comparable to the 3k/7k/2k/8k DSRL outcomes.
#
# Usage: sbatch dsrl_fixedstart_validation_lightbulb.sbatch <cat_true_16k|cat_false_20k>
# ─────────────────────────────────────────────────────────────────────────────────────

set -uo pipefail
ARM="${1:?cat_true_16k | cat_false_20k}"

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
case "$ARM" in
    cat_true_16k)  SFT="$TRAIN/2026-08-28/15-26-13_robocasa_pi05/checkpoints/016000/pretrained_model" ;;
    cat_false_20k) SFT="$TRAIN/2026-08-31/18-26-31_robocasa_pi05/checkpoints/020000/pretrained_model" ;;
    *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
[ -f "$SFT/model.safetensors" ] || { echo "missing checkpoint $SFT"; exit 1; }
CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")

OUT=~/scratch/lerobot/outputs/dsrl_lightbulb_fixedstart_validation/${ARM}
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
    --total_steps 500000 \
    --min_buffer_size 500 \
    --reward_mode goal \
    --collect_envs 2 \
    --utd_ratio 40 \
    --eval_seeds 0,1 \
    --eval_episodes 20 \
    --eval_freq 25000 \
    --save_freq 25000 \
    --log_freq 1000 \
    --output_dir "$OUT" \
    --wandb_enable \
    --wandb_project lerobot-dsrl-robocasa \
    --wandb_name "dsrl_pi05_lightbulb_validation_${ARM}" \
    2>&1 | tee "$OUT/run.log"
