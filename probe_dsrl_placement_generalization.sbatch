#!/bin/bash
#SBATCH --partition long
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=probe_dsrl_placement
#SBATCH --time=4:00:00

# ─────────────────────────────────────────────────────────────────────────────────────
# Sparse probe: does DSRL's 1.000 survive a change of object placement inside its OWN
# training kitchens?
#
# Each DSRL collect env is one construction seed, which fixes the kitchen AND the object
# placement, so `7k` reached success_rate=1.000 on exactly two lamp poses. This probe asks
# whether that transfers to other poses in the same two kitchens.
#
# CELL A (control, unpinned --eval_seeds 0,1) reproduces the *exact* training/eval
# condition. It matters for two reasons: it is the baseline the test cell is compared
# against, and it validates the new --eval_only code path end-to-end. If A does not come
# back at ~1.000, the harness is wrong, not the policy -- so read A before B.
#
# CELL B (pinned 8:6 and 4:8, placement seeds 0,1,2) is the same two kitchens with
# held-out lamp poses. Pinning changes rng consumption (choice over a 1-entry pool vs 120),
# so NONE of these placements is the training one -- that is the point, but it is also why
# cell A cannot be folded into cell B.
#
# ~22 rollouts total, ~1 GPU-h. NOTE: never put a comment inside the backslash-continued
# command -- bash ends the command at the '#' and silently drops the rest.
# ─────────────────────────────────────────────────────────────────────────────────────

source ~/.bashrc
cd ~/lerobot
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

uv pip install --quiet numpy==2.2.5

SFT_TRUE=~/scratch/lerobot/outputs/train/2026-08-28/15-26-13_robocasa_pi05/checkpoints
DSRL=~/scratch/lerobot/outputs/dsrl_lightbulb_parallel
CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")
OUTBASE=~/scratch/lerobot/outputs/probe_dsrl_placement/lightbulb_cat_true_7k
mkdir -p "$OUTBASE"

# --dsrl_* MUST match the training run: dsrl_config is not saved in the checkpoint, so the
# compact encoder is rebuilt from these and a mismatch fails the state-dict load.
run_cell () {
    local tag=$1 scenes=$2 seeds=$3 episodes=$4
    mkdir -p "$OUTBASE/$tag"
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${tag} \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}_${tag} \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
        --eval_only \
        --noise_actor_path "$DSRL/7k/final" \
        --policy_path "$SFT_TRUE/007000/pretrained_model" \
        --task ScrewLightbulb \
        --robot XArm6DexLeapRHOmron \
        --controller "$CONTROLLER" \
        --fps 30 \
        --prompt "screw the light bulb into the lamp" \
        --reward_mode goal \
        --collect_envs 1 \
        --eval_scenes "$scenes" \
        --eval_seeds "$seeds" \
        --eval_episodes "$episodes" \
        --eval_videos 4 \
        --dsrl_image_resize 64 \
        --dsrl_hidden_dim 128 \
        --dsrl_num_layers 3 \
        --dsrl_num_q 10 \
        --output_dir "$OUTBASE/$tag" \
        --wandb_enable \
        --wandb_project lerobot-dex \
        --wandb_name "probe_placement_7k_${tag}" \
        > "$OUTBASE/$tag/eval.log" 2>&1 &
    echo "launched cell $tag | PID $! | log: $OUTBASE/$tag/eval.log"
}

run_cell control_training_placements "" "0,1" 5
sleep 90
run_cell test_heldout_placements "8:6,4:8" "0,1,2" 2

wait
echo "placement-generalization probe exited"
grep -h -A 20 "==== eval_only results ====" "$OUTBASE"/*/eval.log || true
