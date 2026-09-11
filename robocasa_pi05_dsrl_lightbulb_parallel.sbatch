#!/bin/bash
#SBATCH --partition long
#SBATCH -c 8                    # more cores: RoboCasa stepping is CPU-bound and we run 3 at once
#SBATCH --mem=48G                # 3 RoboCasa processes share this; bump if you see OOM kills
#SBATCH --gres=gpu:l40s:1        # ONE GPU, shared by all runs (l40s has native bf16 — much faster
                                 # than rtx8000's emulated bf16 for pi05 inference)
#SBATCH --job-name=dsrl_lightbulb_parallel

source ~/.bashrc
cd ~/lerobot  # repo-relative paths (.venv, examples/) must resolve regardless of submit dir
source .venv/bin/activate

# Headless GPU rendering. NOTE: no shared NUMBA_CACHE_DIR export here — each run sets its own
# below, since concurrent processes sharing one numba cache dir can corrupt it ("EOFError:
# Ran out of input"). PYTHONUNBUFFERED so each run's progress lands in its log live.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

# RoboCasa hard-asserts numpy==2.2.5; guard against an accidental re-sync to the lockfile's 2.2.6.
uv pip install --quiet numpy==2.2.5

# Frozen pi05 checkpoints (XArm6DexLeapRHOmron, ScrewLightbulb, 22-DoF, joint-position control).
# ROOT=~/scratch/lerobot/outputs/train/2026-08-31/18-26-31_robocasa_pi05/checkpoints
ROOT=~/scratch/lerobot/outputs/train/2026-08-28/15-26-13_robocasa_pi05/checkpoints
# Joint-position controller: the ScrewLightbulb data is absolute joint targets, so the env MUST
# use this controller (not the default OSC_POSE) for the policy's actions to make sense. Resolve
# it from the installed robosuite, exactly as the training script does.
CONTROLLER=$(python -c "import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), 'controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json'))")
echo "controller = $CONTROLLER"

# Absolute output base (scratch has the space). Each run gets its own subdir holding its
# checkpoints, eval videos, AND its run.log — one obvious place to look per run. The slurm-%j.out
# only carries the launch echoes below + the GPU epilogue, since each run's stdout is redirected.
OUTBASE=~/scratch/lerobot/outputs/dsrl_lightbulb_parallel/embodiment_false/

# One DSRL run per checkpoint, backgrounded onto the shared GPU. Each gets its own numba +
# torch-inductor cache (isolation), output dir, wandb name, and log file. fps=30 and the
# joint-pos controller match how these checkpoints were trained.
run_dsrl () {
    local tag=$1 step=$2
    mkdir -p "$OUTBASE/${tag}"
    # Eval is serial (one env, 2 seeds x eval_episodes per eval) while training collects in
    # parallel, so it dominated wall clock: measured at 64% of the lightbulb job's 52 h (0.179 s
    # per eval env step x ~16k eval steps, vs 0.255 s x 10k training steps). Fewer, rarer evals
    # cut total time ~2x. NOTE: never put a comment line inside the backslash-continued command
    # below — bash ends the command at the `#`, silently dropping every later flag, the log
    # redirect, and the trailing `&`.
    NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${tag} \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}_${tag} \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
        --policy_path "$ROOT/$step/pretrained_model" \
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
        --output_dir "$OUTBASE/${tag}" \
        --wandb_enable \
        --wandb_project lerobot-dsrl-robocasa \
        --wandb_name "dsrl_pi05_lightbulb_${tag}" \
        > "$OUTBASE/${tag}/run.log" 2>&1 &
    echo "launched $tag (step $step) | PID $! | log: $OUTBASE/${tag}/run.log"
}

# Stagger starts so the three pi05 loads + torch.compile passes don't thrash the CPU/GPU at once.
run_dsrl 3k  003000
sleep 120
run_dsrl 7k  007000
# sleep 120
# run_dsrl 15k 016000

# Block until all three finish (or the job hits its time limit).
wait
echo "all runs exited"
