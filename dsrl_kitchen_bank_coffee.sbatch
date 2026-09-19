#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8
#SBATCH --mem=64G                  # 10 resident training kitchens (~3.4 GB each) + eval kitchens + pi05
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=dsrl_kitchens_coffee

# ─────────────────────────────────────────────────────────────────────────────────────
# Experiment 2 (CoffeePressButton): DSRL with KITCHEN randomization.
#
# Placement randomization does not transfer here: RoboCasa centres the robot on the coffee machine
# in every kitchen, so moving the machine moves the robot with it, and the mug does not affect the
# button press. Each of the 2 collect slots is instead a POOL of kitchens, switching kitchen on
# every reset; env steps per SAC update stay identical to the fixed-kitchen runs, whose argv this
# otherwise mirrors. Bank: scripts/build_coffee_kitchen_bank.py (10 training kitchens including
# seeds 0,1, balanced across the 3 machine models, + 2 held-out kitchens).
#
# Start-pose rejection is on for training AND eval, old and new checkpoints alike, so held-out
# numbers measure kitchen novelty rather than starts where the arm reset noise jams the hand into
# a wall cabinet. L8S4 (seed 2) is excluded from the bank; seed 10 took its place.
#
# Usage:
#   sbatch dsrl_kitchen_bank_coffee.sbatch <cat_true_4k|cat_true_8k|cat_false_2k|cat_false_8k> train
#   sbatch dsrl_kitchen_bank_coffee.sbatch <arm> eval [NOISE_ACTOR_DIR]
# 'eval' defaults to the fixed-kitchen run's checkpoint: final/ for cat_true_4k and cat_false_8k,
# and the last surviving step (320k) for the two whose runs never reached 500k.
#
# NOTE: never put a comment line inside a backslash-continued command -- bash ends the
# command at the '#', silently dropping every later flag and the redirect.
# ─────────────────────────────────────────────────────────────────────────────────────

set -u
ARM="${1:?arm: cat_true_4k | cat_true_8k | cat_false_2k | cat_false_8k}"
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
OLD=~/scratch/lerobot/outputs/dsrl_coffee_parallel
BANK=~/scratch/lerobot/placement_banks/coffeepressbutton_pandadex_kitchens/bank.json
case "$ARM" in
    # final/ provenance checked via wandb metadata: both written by the Sep 7-8 runs of these configs.
    cat_true_4k)  SFT="$TRAIN/2026-09-04/08-14-09_robocasa_pi05/checkpoints/004000/pretrained_model"; OLD_ACTOR="$OLD/4k/final" ;;
    cat_false_8k) SFT="$TRAIN/2026-09-03/21-19-54_robocasa_pi05/checkpoints/008000/pretrained_model"; OLD_ACTOR="$OLD/8k/final" ;;
    # The old cat=False 2k run stopped at 320k steps (no final/); all its checkpoints are 21-19-54/002000.
    cat_false_2k) SFT="$TRAIN/2026-09-03/21-19-54_robocasa_pi05/checkpoints/002000/pretrained_model"; OLD_ACTOR="$OLD/2k/step_320020" ;;
    # The old cat=True 8k run shares its directory with a cat=False run that overwrote it; final/ there
    # is cat=False. step_320026 is the last save logged by the cat=True run (wandb 3sr5baor, argv
    # 08-14-09/008000) and still carries that run's mtime (09-06 20:26), so it was not overwritten.
    cat_true_8k)  SFT="$TRAIN/2026-09-04/08-14-09_robocasa_pi05/checkpoints/008000/pretrained_model"; OLD_ACTOR="$OLD/8k/step_320026" ;;
    *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

COMMON=(
    --policy_path "$SFT"
    --task CoffeePressButton
    --robot PandaDexLeapRHOmron
    --reward_mode goal
    --collect_envs 2
    --kitchen_bank "$BANK"
    --eval_kitchen_sets "heldout,reference"
    --reject_start_rot_deg 10
    --reject_start_pos_cm 3
    --wandb_enable
)

if [ "$MODE" = "train" ]; then
    OUT=~/scratch/lerobot/outputs/dsrl_coffee_kitchenbank/${ARM}
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
        --eval_episodes 10 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --log_freq 100 \
        --eval_videos 4 \
        --output_dir "$OUT" \
        --wandb_project lerobot-dsrl-robocasa \
        --wandb_name "dsrl_pi05_coffee_kitchenbank_${ARM}" \
        2>&1 | tee "$OUT/run.log"
elif [ "$MODE" = "eval" ]; then
    if [ -n "$ACTOR_OVERRIDE" ]; then
        ACTOR="$ACTOR_OVERRIDE"
    else
        ACTOR="$OLD_ACTOR"
    fi
    [ -f "$ACTOR/model.safetensors" ] || { echo "no noise actor at $ACTOR"; exit 1; }
    TAG=$(echo "$ACTOR" | sed "s|$HOME/scratch/lerobot/outputs/||; s|/|__|g")
    OUT=~/scratch/lerobot/outputs/dsrl_coffee_kitchenbank/eval/${ARM}/${TAG}
    mkdir -p "$OUT"
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py "${COMMON[@]}" \
        --collect_envs 1 \
        --eval_only \
        --noise_actor_path "$ACTOR" \
        --eval_episodes 20 \
        --eval_videos 4 \
        --output_dir "$OUT" \
        --wandb_project lerobot-dex \
        --wandb_name "kitcheneval_coffee_${ARM}_${TAG}" \
        2>&1 | tee "$OUT/eval.log"
else
    echo "unknown MODE=$MODE"; exit 1
fi
