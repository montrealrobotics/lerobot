#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=dsrl_bank_lightbulb

# ─────────────────────────────────────────────────────────────────────────────────────
# Experiment 2 (ScrewLightbulb): DSRL with object-placement randomization.
#
# The kitchens and robot base stay as in the fixed-placement runs, but every reset moves the lamp
# to the next of 10 feasibility-checked training placements per kitchen (reshuffled each pass).
# Everything else mirrors the 7k run's argv (wandb ye3t0xrs), so placement is the only change.
# Bank: scripts/build_lamp_placement_bank.py; 'reference' = the original fixed placement, itself
# held out of the bank. Eval cells: heldout (2x2) + reference (2) + train:2 (2x2).
#
# Usage:
#   sbatch dsrl_placement_bank_lightbulb.sbatch <cat_true_7k|cat_true_3k|cat_false_2k|cat_false_8k> train
#   sbatch dsrl_placement_bank_lightbulb.sbatch <arm> eval [NOISE_ACTOR_DIR]
# 'eval' scores a noise actor on the same cells (default: the fixed-placement run's final/).
#
# NOTE: never put a comment line inside a backslash-continued command -- bash ends the
# command at the '#', silently dropping every later flag and the redirect.
# ─────────────────────────────────────────────────────────────────────────────────────

set -u
ARM="${1:?arm: cat_true_7k | cat_true_3k | cat_false_2k | cat_false_8k}"
MODE="${2:?mode: train | eval}"
ACTOR_OVERRIDE="${3:-}"

source ~/.bashrc
cd ~/lerobot
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
# torch.compile execs nvcc via PATH; on a node whose /cvmfs mount is broken that raises
# 'OSError: Transport endpoint is not connected' mid-training. Strip /cvmfs only if unreadable.
if ! ls /cvmfs/ai.mila.quebec >/dev/null 2>&1; then
    echo "WARNING: /cvmfs unreadable on $(hostname); removing it from PATH"
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v cvmfs | paste -sd:)
fi
uv pip install --quiet numpy==2.2.5

TRAIN=~/scratch/lerobot/outputs/train
OLD=~/scratch/lerobot/outputs/dsrl_lightbulb_parallel
BANK=~/scratch/lerobot/placement_banks/screwlightbulb_xarm6_seed01/bank.json
case "$ARM" in
    cat_true_7k)  SFT="$TRAIN/2026-08-28/15-26-13_robocasa_pi05/checkpoints/007000/pretrained_model"; OLD_RUN="$OLD/7k" ;;
    cat_false_2k) SFT="$TRAIN/2026-08-31/18-26-31_robocasa_pi05/checkpoints/002000/pretrained_model"; OLD_RUN="$OLD/embodiment_false/2k" ;;
    cat_true_3k)  SFT="$TRAIN/2026-08-28/15-26-13_robocasa_pi05/checkpoints/003000/pretrained_model"; OLD_RUN="$OLD/3k" ;;
    cat_false_8k) SFT="$TRAIN/2026-08-31/18-26-31_robocasa_pi05/checkpoints/008000/pretrained_model"; OLD_RUN="$OLD/embodiment_false/8k" ;;
    *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")

COMMON=(
    --policy_path "$SFT"
    --task ScrewLightbulb
    --robot XArm6DexLeapRHOmron
    --controller "$CONTROLLER"
    --fps 30
    --prompt "screw the light bulb into the lamp"
    --reward_mode goal
    --collect_envs 2
    --placement_bank "$BANK"
    --eval_placement_sets "heldout,reference,train:2"
    --dsrl_image_resize 64
    --dsrl_hidden_dim 128
    --dsrl_num_layers 3
    --dsrl_num_q 10
    --wandb_enable
)

if [ "$MODE" = "train" ]; then
    OUT=~/scratch/lerobot/outputs/dsrl_lightbulb_placementbank/${ARM}
    # Refuse to write into a directory that already holds a run: two runs sharing one output
    # dir is exactly how the coffee cat_true 8k checkpoints were overwritten.
    if compgen -G "$OUT/step_*" > /dev/null; then
        echo "REFUSING: $OUT already contains checkpoints. Move it aside or pick a new ARM tag."
        exit 1
    fi
    mkdir -p "$OUT"
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py "${COMMON[@]}" \
        --total_steps 500000 \
        --min_buffer_size 500 \
        --utd_ratio 40 \
        --eval_episodes 2 \
        --eval_freq 25000 \
        --save_freq 25000 \
        --log_freq 100 \
        --eval_videos 4 \
        --output_dir "$OUT" \
        --wandb_project lerobot-dsrl-robocasa \
        --wandb_name "dsrl_pi05_lightbulb_bank_${ARM}" \
        2>&1 | tee "$OUT/run.log"
elif [ "$MODE" = "eval" ]; then
    if [ -n "$ACTOR_OVERRIDE" ]; then
        ACTOR="$ACTOR_OVERRIDE"
    elif [ -d "$OLD_RUN/final" ]; then
        ACTOR="$OLD_RUN/final"
    else
        ACTOR=$(ls -d "$OLD_RUN"/step_* | sort -t_ -k2 -n | tail -1)
        echo "WARNING: $OLD_RUN has no final/, using latest $ACTOR"
    fi
    TAG=$(echo "$ACTOR" | sed "s|$HOME/scratch/lerobot/outputs/||; s|/|__|g")
    OUT=~/scratch/lerobot/outputs/dsrl_lightbulb_placementbank/eval/${ARM}/${TAG}
    mkdir -p "$OUT"
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py "${COMMON[@]}" \
        --eval_only \
        --noise_actor_path "$ACTOR" \
        --eval_episodes 5 \
        --eval_videos 10 \
        --output_dir "$OUT" \
        --wandb_project lerobot-dex \
        --wandb_name "bankeval_lightbulb_${ARM}_${TAG}" \
        2>&1 | tee "$OUT/eval.log"
else
    echo "unknown MODE=$MODE"; exit 1
fi
