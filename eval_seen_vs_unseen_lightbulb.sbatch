#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8                     # RoboCasa stepping is CPU-bound and we run several at once
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1         # ONE GPU shared by all cells (native bf16 for pi05 inference)
#SBATCH --job-name=eval_seen_unseen_lightbulb
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────────────────────────────
# Seen-vs-unseen-environment table for ScrewLightbulb: SFT-only vs SFT+DSRL.
#
# WHY THESE SCENES. Ground truth from regeneration_episodes.json: the 37 kept SFT demos
# cover 33 distinct (layout, style) pairs. (1,4) is one of the 14 pairs GUARANTEED to
# survive the "filtered" step, so it is in-distribution. (8,6) and (4,8) are the two
# kitchens the DSRL runs collected in (construction seeds 0 and 1) and NEITHER pair
# appears in the SFT data -- layout 8 only appears as (8,0)/(8,8), layout 4 only as
# (4,1)/(4,3)/(4,7)/(4,11). So both DSRL training kitchens are already unseen.
#
# WHY PLACEMENTS, NOT EPISODES. RoboCasa freezes the scene AND the object placement per
# env instance (hard_reset=False), so repeated episodes in one cell differ only by the
# robot's arm-joint jitter (robocasa passes initialization_noise magnitude 0.02 rad,
# resampled from the env rng every reset). Object pose is the axis that actually matters
# for reachability, so the budget is spread over 5 placement seeds x 4 episodes rather
# than 1 placement x 20 episodes.
#
# WHY THE UNSEEN CELLS ARE RE-MEASURED. Pinning the scene changes the rng consumption
# (rng.choice over a 1-entry pool vs 120), so pinned-(8,6)-placement-0 is NOT the same
# initial condition as the old unpinned seed-0 eval. The existing training-curve numbers
# are therefore not protocol-comparable to the seen-scene numbers; all three scenes are
# measured here under one protocol.
# ─────────────────────────────────────────────────────────────────────────────────────

source ~/.bashrc
cd ~/lerobot
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

# RoboCasa hard-asserts numpy==2.2.5; guard against a re-sync to the lockfile's 2.2.6.
uv pip install --quiet numpy==2.2.5

SFT_TRUE=~/scratch/lerobot/outputs/train/2026-08-28/15-26-13_robocasa_pi05/checkpoints
SFT_FALSE=~/scratch/lerobot/outputs/train/2026-08-31/18-26-31_robocasa_pi05/checkpoints
DSRL=~/scratch/lerobot/outputs/dsrl_lightbulb_parallel

CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")
OUTBASE=~/scratch/lerobot/outputs/eval_seen_unseen/lightbulb
mkdir -p "$OUTBASE"

SCENES="1:4,8:6,4:8"        # seen, unseen(=DSRL kitchen 0), unseen(=DSRL kitchen 1)
PLACEMENTS="0,1,2,3,4"      # placement seeds within each pinned scene
EPISODES=4                  # per (scene, placement) cell -> 20 rollouts per scene

# ── Arm 1: SFT + DSRL ────────────────────────────────────────────────────────────────
# The --dsrl_* flags MUST match the training run: dsrl_config is not saved in the
# checkpoint, so the compact encoder is rebuilt from these and a mismatch fails the load.
# NOTE: never put a comment line inside the backslash-continued command -- bash ends the
# command at the '#', silently dropping every later flag, the redirect, and the '&'.
run_dsrl_eval () {
    local tag=$1 actor=$2 frozen=$3
    mkdir -p "$OUTBASE/dsrl_${tag}"
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_d${tag} \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}_d${tag} \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
        --eval_only \
        --noise_actor_path "$actor" \
        --policy_path "$frozen" \
        --task ScrewLightbulb \
        --robot XArm6DexLeapRHOmron \
        --controller "$CONTROLLER" \
        --fps 30 \
        --prompt "screw the light bulb into the lamp" \
        --reward_mode goal \
        --eval_scenes "$SCENES" \
        --eval_seeds "$PLACEMENTS" \
        --eval_episodes $EPISODES \
        --eval_videos 3 \
        --dsrl_image_resize 64 \
        --dsrl_hidden_dim 128 \
        --dsrl_num_layers 3 \
        --dsrl_num_q 10 \
        --output_dir "$OUTBASE/dsrl_${tag}" \
        --wandb_enable \
        --wandb_project lerobot-dsrl-robocasa-eval \
        --wandb_name "seenunseen_dsrl_lightbulb_${tag}" \
        > "$OUTBASE/dsrl_${tag}/eval.log" 2>&1 &
    echo "launched DSRL cell $tag | PID $! | log: $OUTBASE/dsrl_${tag}/eval.log"
}

# ── Arm 2: SFT only (frozen pi05, i.i.d. Gaussian flow noise) ────────────────────────
# Routed through lerobot-eval on purpose: inside the DSRL harness --noise_chunk_size 1
# repeats one noise vector across all 32 chunk steps, whereas vanilla pi05 draws i.i.d.
# noise per step -- so the DSRL harness cannot produce a faithful SFT-only baseline.
# One invocation per scene, since env.layout_id/style_id take a single value.
run_sft_eval () {
    local tag=$1 ckpt=$2 layout=$3 style=$4
    local out="$OUTBASE/sft_${tag}/L${layout}S${style}"
    mkdir -p "$out"
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_s${tag}_${layout}_${style} \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}_s${tag}_${layout}_${style} \
    lerobot-eval \
        --policy.path "$ckpt" \
        --env.type robocasa \
        --env.task ScrewLightbulb \
        --env.robot XArm6DexLeapRHOmron \
        --env.controller "$CONTROLLER" \
        --env.fps 30 \
        --env.layout_id "$layout" \
        --env.style_id "$style" \
        --env.camera_name "robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand,robot0_agentview_center" \
        --eval.batch_size 5 \
        --eval.n_episodes 20 \
        --eval.n_videos 3 \
        --eval.use_async_envs false \
        --output_dir "$out" \
        > "$out/eval.log" 2>&1 &
    echo "launched SFT cell $tag L${layout}S${style} | PID $! | log: $out/eval.log"
}

# ── P0: the two complete cat-specific DSRL runs + their SFT-only counterparts ────────
# Stagger so the pi05 loads / torch.compile passes do not thrash the GPU at once.
run_dsrl_eval cat_true_3k "$DSRL/3k/final"  "$SFT_TRUE/003000/pretrained_model"
sleep 90
run_dsrl_eval cat_true_7k "$DSRL/7k/final"  "$SFT_TRUE/007000/pretrained_model"
sleep 90

for LS in "1 4" "8 6" "4 8"; do
    set -- $LS
    run_sft_eval cat_true_3k "$SFT_TRUE/003000/pretrained_model" "$1" "$2"
    sleep 60
done

wait
echo "all lightbulb seen-vs-unseen cells exited"
