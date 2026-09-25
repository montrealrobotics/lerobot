#!/bin/bash
# THREE DSRL runs sharing one H100 -- ALLIANCE/DRAC. Single-run sibling: dsrl_drac.sbatch.
# Design: SFT_ARMS_DSRL.md. Per-task invariants: dsrl_task_args.sh (shared, so the two
# launchers cannot drift).
#
#   sbatch --job-name=dsrl_arm2_lamp_s16000 dsrl_drac_pack.sbatch lamp <sft_run_dir> \
#       16000:0 16000:1 16000:2
#
# Each spec is <sft_step>:<rl_seed>. Args after the specs are forwarded verbatim to every
# run in the pack. Any number of specs works; 3 is what the resources below are sized for.
#
# WHY PACK. Measured at collect_envs=1 (SFT_ARMS_DSRL.md): a macro step is 1.136 s, of
# which ~0.64 s is GPU (frozen policy 0.208 + SAC 0.434) and ~0.49 s is CPU (env stepping).
# So one run leaves the GPU ~44% idle and uses 8.8 GiB of 80. Raising --collect_envs does
# NOT help -- at constant effective UTD throughput is flat -- and the GPU cannot be
# subdivided instead, because MIG slices cannot do EGL and DSRL renders every step. Packing
# independent runs is therefore the only lever. Three runs saturate the GPU, so each is
# slower than it would be alone, but aggregate throughput is ~2.5-3x.
#
#SBATCH --account=def-kdarvish
#SBATCH --gpus=h100:1
# GPU MEMORY DECIDES THE PACK SIZE, and it differs by task (measured on H100, steady state
# with SAC updates running): lamp ~13 GB/run -> 3 per GPU; coffee ~27.5 GB/run -> 2 per GPU.
# Coffee holds renderers for its 10-kitchen collection pool plus 4 eval kitchens, where lamp
# has one scene. Three coffee runs exceed 80 GB, and the pack dies at the FIRST SAC update
# (buffer ~1000) with "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR ... cusolverDnCreate"
# -- cuSOLVER's handle is simply the first allocation to fail. The launcher refuses more than
# MAX_PER_GPU specs per task. For a 2-run coffee pack pass
#   --cpus-per-task=16 --mem=120000M
# HOST memory matters as much as GPU memory: at --mem=64000M every 2-run coffee pack sat at
# the cgroup ceiling (thousands of memory.max events, one oom_kill), and packs aborted with a
# silent SIGABRT inside env.step at startup -- native code failing an allocation. A rorqual
# H100 node is 512 GB / 4 GPUs, so ~120 GB is one GPU's fair share.
# Sized below for THREE runs: ~3 CPU each for MuJoCo stepping + rendering, and ~28 GiB RSS each
# once the 100k replay buffer of 64x64 float32 observations is full (18.1 GiB measured at a
# 2k buffer, plus ~10 GiB for the rest of it).
#SBATCH --cpus-per-task=12
#SBATCH --mem=96000M
# Packed runs are individually slower than solo ones, so a pack will usually hit the wall
# clock before all three finish. That is handled, not hoped for: --signal below reaches the
# batch script 300 s early, the trap forwards SIGTERM to every python, each writes
# resume_state.pt and exits 99, and the job requeues. Runs that already finished are skipped
# on the requeue via their .dsrl_complete sentinel -- without that they would restart from
# scratch, because a completed run deletes its own resume state.
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/akur/scratch/lerobot/slurm_logs/%x-%j.out

set -euo pipefail

TASK=${1:?usage: sbatch dsrl_drac_pack.sbatch <lamp|coffee> <sft_run_dir> <step:seed>...}
RUN_DIR=${2:?usage: sbatch dsrl_drac_pack.sbatch <lamp|coffee> <sft_run_dir> <step:seed>...}
shift 2

SPECS=()
while [ $# -gt 0 ]; do
    case "$1" in
        [0-9]*:[0-9]*) SPECS+=("$1"); shift ;;
        *) break ;;
    esac
done
[ ${#SPECS[@]} -gt 0 ] || { echo "no <step:seed> specs given" >&2; exit 2; }
EXTRA_ARGS=("$@")

# Cluster plumbing -- the ONLY thing that differs between clusters. #SBATCH directives
# cannot be conditional, so on an offline cluster override them on the command line
# (it takes precedence over the file), e.g. for rorqual:
#   sbatch --account=<best FairShare from `sshare -U -u $USER`> ... dsrl_drac_pack.sbatch ...
case "${SLURM_CLUSTER_NAME:-$(hostname)}" in
  nibi*)
    REPO=/project/6068036/akur/lerobot
    SCRATCH=/home/akur/scratch
    HF_HOME_DIR=/scratch/akur/hf_cache
    OFFLINE=0 ;;
  # narval and rorqual compute nodes have NO internet: HF must come from a pre-warmed
  # cache and wandb logs offline, synced afterwards from a login node with
  # scripts/sync_offline_wandb.sh. There is no ~/scratch symlink on rorqual, so the
  # #SBATCH --output above must be overridden at submit time there too.
  rorqual*)
    REPO=/lustre09/project/6068036/akur/lerobot   # = ~/links/projects/def-gberseth/akur/lerobot
    SCRATCH=/scratch/$USER
    HF_HOME_DIR=/scratch/$USER/hf_cache
    OFFLINE=1 ;;
  narval*)
    REPO=$HOME/projects/rrg-gberseth/akur/lerobot
    SCRATCH=/scratch/$USER
    HF_HOME_DIR=/scratch/$USER/hf_cache
    OFFLINE=1 ;;
  *) echo "unknown cluster '${SLURM_CLUSTER_NAME:-$(hostname)}' -- add it to the case above" >&2; exit 2 ;;
esac
[ -d "$REPO" ] || { echo "no repo at $REPO" >&2; exit 2; }
BANKS=$SCRATCH/lerobot/placement_banks
cd "$REPO"

module load StdEnv/2023 python/3.12.4
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=$HF_HOME_DIR
if [ "$OFFLINE" = "1" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export WANDB_MODE=offline
    echo "offline cluster: HF from cache at $HF_HOME, wandb logging offline"
fi
export TOKENIZERS_PARALLELISM=false
# A native abort (SIGABRT) otherwise leaves no trace at all in run.log; this makes Python
# dump every thread's stack when it happens.
export PYTHONFAULTHANDLER=1
# The NVIDIA driver caches compiled shaders under ~/.cache/nvidia/GLCache -- in $HOME, a
# filesystem shared by every job on every node. Concurrent EGL processes (three per DSRL
# pack, several eval workers per SFT job) race on those files, and a reader that catches a
# half-written shader makes the driver abort: SIGABRT, exit 134, no Python traceback. That
# matches the dsrl_arm0_coffee_04000 crashes on nibi and rorqual, whose cache files were
# being written inside the crash window. MuJoCo compiles a handful of shaders, so disabling
# the disk cache costs nothing measurable.
export __GL_SHADER_DISK_CACHE=0
# Three processes share the allocation; leave each a couple of BLAS threads rather than
# letting all three think they own 12 cores.
export OMP_NUM_THREADS=2
mkdir -p "$SCRATCH/lerobot/slurm_logs"

JOB=${SLURM_JOB_NAME:-dsrl_${TASK}_pack}
echo "=== $(date) job=${SLURM_JOB_ID:-none} node=$(hostname) task=$TASK"
echo "run_dir=$RUN_DIR"
echo "specs=${SPECS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Which physical GPU this job got -- to correlate the startup SIGABRTs with node/GPU placement.
echo "gpu: $(nvidia-smi --query-gpu=index,pci.bus_id,uuid --format=csv,noheader | tr '\n' ' ') CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
[ -d "$RUN_DIR" ] || { echo "no such SFT run dir: $RUN_DIR" >&2; exit 2; }

# DSRL renders every collection step, so there is no eval-free fallback. MIG cannot do EGL.
if [ -n "${DSRL_MUJOCO_GL:-}" ]; then
    export MUJOCO_GL="$DSRL_MUJOCO_GL"
    export PYOPENGL_PLATFORM="$DSRL_MUJOCO_GL"
    echo "GL backend forced by DSRL_MUJOCO_GL=$DSRL_MUJOCO_GL"
elif GL_EXPORTS=$(python scripts/check_gl_backend.py --emit-exports 2>/dev/null) && [ -n "$GL_EXPORTS" ]; then
    eval "$GL_EXPORTS"
    echo "GL backend: MUJOCO_GL=$MUJOCO_GL MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-n/a}"
else
    echo "PREFLIGHT FAILED: no working headless GL backend on this node." >&2
    echo "  Run 'python scripts/check_gl_backend.py' here; check you got a FULL GPU." >&2
    exit 1
fi

python - <<'PREFLIGHT'
import os, sys
fail = []
import numpy
if numpy.__version__ != "2.2.5":
    fail.append(f"numpy is {numpy.__version__}, RoboCasa asserts 2.2.5")
from huggingface_hub import get_token
if not get_token():
    fail.append(f"no HF token visible (HF_HOME={os.environ.get('HF_HOME')}; export HF_TOKEN)")
import netrc
has_netrc = False
try:
    has_netrc = "api.wandb.ai" in netrc.netrc(os.path.expanduser("~/.netrc")).hosts
except Exception:
    pass
if not os.environ.get("WANDB_API_KEY") and not has_netrc:
    fail.append("no wandb credentials (set WANDB_API_KEY or run `wandb login`)")
import torch
if not torch.cuda.is_available():
    fail.append("CUDA not available")
else:
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
if fail:
    print("\nPREFLIGHT FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("preflight OK")
PREFLIGHT

# Per-task invariants (robot, fps, controller, bank, eval sets, prompt).
source "$REPO/dsrl_task_args.sh"

case "$TASK" in coffee) MAX_PER_GPU=2 ;; *) MAX_PER_GPU=3 ;; esac
if [ ${#SPECS[@]} -gt "$MAX_PER_GPU" ]; then
    echo "refusing ${#SPECS[@]} $TASK runs on one GPU (max $MAX_PER_GPU): they exceed 80 GB and" >&2
    echo "  die at the first SAC update with CUSOLVER_STATUS_INTERNAL_ERROR. See the header." >&2
    exit 2
fi

# Output dir and wandb name come from the SFT RUN + STEP, not from the job name, so a job may
# hold runs of different steps (2 per GPU does not divide 3 seeds per step) and every run
# still lands in dsrl/dsrl_<arm>_<task>_<step>/step<step>_seed<n> -- the layout of all earlier
# runs, which is also where a requeued run finds its resume_state.pt. A non-default SFT
# lr-scale or seed is appended so a sweep cannot collide with the default run.
SFT_NAME=$(basename "$RUN_DIR")
ARMTAG=$(grep -oE 'arm[0-9]+' <<< "$SFT_NAME" | head -1)
[ -n "$ARMTAG" ] || { echo "cannot find an armN tag in $SFT_NAME" >&2; exit 2; }
SFT_SUFFIX=$(grep -oE 'lrx[^_]+_seed[0-9]+$' <<< "$SFT_NAME" || true)
[ -n "$SFT_SUFFIX" ] && [ "$SFT_SUFFIX" != "lrx1_seed42" ] && ARMTAG="${ARMTAG}_${SFT_SUFFIX}"
[ -f "$BANK" ] || { echo "no bank at $BANK -- rsync it from Mila" >&2; exit 2; }

PIDS=(); TAGS=(); OUTS=()

# One run in a background subshell, retried if it ABORTS AT STARTUP.
#
# Intermittently, every process of a pack dies with SIGABRT (exit 134, no Python traceback)
# within the first minutes, while render contexts are being created -- seen 3 times, always
# at startup, never once a run is past it (lamp packs ran 12 h without one). The cause is
# not established; it looks GPU/driver-level. A run that dies that way before it has ever
# written resume_state.pt has lost nothing, so it is simply retried (at most MAX_ATTEMPTS).
# Any other failure (e.g. the rc=1 OOM) still fails loudly. SIGTERM is forwarded to the
# current child, so preemption and the wall-clock requeue behave exactly as before.
MAX_ATTEMPTS=3
run_one () {
    local out=$1; shift
    local attempt=1 rc child="" termed=""
    # `|| true`: under set -e a failed kill (child already gone, e.g. TERM during the retry
    # pause) would otherwise exit the subshell from inside the trap with rc=1.
    trap 'termed=1; if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi' TERM
    : > "$out/run.log"
    while :; do
        "$@" >> "$out/run.log" 2>&1 &
        child=$!
        rc=0; wait "$child" || rc=$?
        # If the TERM trap interrupted `wait`, the child is still alive: wait for its real
        # status. (Testing rc > 128 would confuse that with a genuine exit 134.)
        if kill -0 "$child" 2>/dev/null; then rc=0; wait "$child" || rc=$?; fi
        if [ "$rc" = "134" ] && [ -z "$termed" ] && [ ! -f "$out/resume_state.pt" ] \
                && [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
            attempt=$((attempt + 1))
            # Nothing of value exists without resume state; drop the aborted attempt's wandb
            # dirs so they are not later uploaded as a stub run.
            rm -rf "$out"/wandb/run-* "$out"/wandb/offline-run-*
            echo "[retry] SIGABRT at startup; attempt $attempt/$MAX_ATTEMPTS at $(date)" >> "$out/run.log"
            sleep 60
            # A TERM that lands during the pause must not start a fresh attempt: report it the
            # way an interrupted run would, so the pack still requeues.
            [ -n "$termed" ] && return 143
            continue
        fi
        return "$rc"
    done
}

launch_one () {
    local step=$1 seed=$2
    local tag="step${step}_seed${seed}"
    local policy="$RUN_DIR/checkpoints/$(printf %06d "$step")/pretrained_model"
    local group="dsrl_${ARMTAG}_${TASK}_$(printf %05d "$step")"
    local out="$SCRATCH/lerobot/outputs/dsrl/$group/$tag"

    if [ ! -d "$policy" ]; then
        echo "[$tag] no such checkpoint: $policy" >&2
        return 2
    fi
    if [ -f "$out/.dsrl_complete" ]; then
        echo "[$tag] already complete, skipping (requeue-safe)"
        return 0
    fi
    mkdir -p "$out"

    # Per-run caches: concurrent processes sharing one numba cache dir corrupt it, which
    # surfaces as "EOFError: Ran out of input" at env creation.
    ( run_one "$out" env \
    NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_$tag" \
    TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}_$tag" \
    python examples/tutorial/rl/dsrl_pi05_robocasa_example.py \
        --policy_path "$policy" \
        "${TASK_ARGS[@]}" \
        --utd_ratio 20 \
        --total_steps 500000 \
        --eval_freq 25000 --eval_episodes 5 \
        --seed "$seed" \
        --output_dir "$out" \
        --resume \
        --wandb_enable --wandb_name "${group}_${tag}" \
        "${EXTRA_ARGS[@]}" ) &

    PIDS+=($!); TAGS+=("$tag"); OUTS+=("$out")
    echo "[$tag] pid $! | ckpt step $step | seed $seed | log $out/run.log"
}

for spec in "${SPECS[@]}"; do
    launch_one "${spec%%:*}" "${spec##*:}"
    # Stagger: three simultaneous pi05 loads and EGL context creations thrash the node.
    sleep 45
done
[ ${#PIDS[@]} -gt 0 ] || { echo "nothing to run (all complete?)"; exit 0; }

# Slurm's SIGTERM on preemption hits every process in the cgroup; this trap covers the
# --signal=B:TERM@300 wall-clock warning, which only reaches the batch script.
forward_term () {
    echo "[pack] signal received, forwarding SIGTERM to ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap forward_term TERM

NEED_REQUEUE=0; WORST=0
for i in "${!PIDS[@]}"; do
    rc=0
    # `wait` returns 128+signal when the trap interrupts it rather than when the child
    # exits, so wait again for the real status.
    wait "${PIDS[$i]}" || rc=$?
    if [ "$rc" -gt 128 ]; then rc=0; wait "${PIDS[$i]}" || rc=$?; fi

    if [ "$rc" = "99" ]; then
        echo "[${TAGS[$i]}] checkpointed at the signal -> will requeue"
        NEED_REQUEUE=1
    elif [ "$rc" = "0" ]; then
        touch "${OUTS[$i]}/.dsrl_complete"
        echo "[${TAGS[$i]}] complete"
    else
        echo "[${TAGS[$i]}] FAILED rc=$rc -- see ${OUTS[$i]}/run.log" >&2
        tail -20 "${OUTS[$i]}/run.log" >&2 || true
        WORST=$rc
    fi
done

if [ "$NEED_REQUEUE" = "1" ]; then
    echo "[pack] requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
echo "=== done rc=$WORST ($(date)) ==="
exit $WORST
