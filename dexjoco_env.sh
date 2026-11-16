# DexJoCo pins mujoco 3.4.0 while robosuite needs the 3.2.6 in .venv, hence a separate venv.
# Node-local because imports from a venv on the networked scratch filesystem have failed
# intermittently in spawned workers.

# Every cluster keeps the inputs in $SCRATCH/dexjoco-data: dexjoco/ (the repo), datasets/, and hf/ (pi05_base at
# the revision the Mila runs used, and the paligemma tokenizer). On Mila these are symlinks.
DEXJOCO_DATA=$SCRATCH/dexjoco-data
DEXJOCO=$DEXJOCO_DATA/dexjoco

source ~/.bashrc          # HF_TOKEN lives here on Mila
export HF_HOME=$DEXJOCO_DATA/hf
export LEROBOT_DEXJOCO_DATASETS=$DEXJOCO_DATA/datasets
export UV_CACHE_DIR=$SCRATCH/uv_cache
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/venv_${SLURM_JOB_ID:-local}"

# Jobs starting together race on the shared uv cache (a file one job is still unpacking is
# missing for the other), so retry after a random delay into a fresh venv: a failed sync can
# leave packages without metadata that the next sync does not repair.
for attempt in 1 2 3; do
    ~/.local/bin/uv sync --frozen --quiet --extra pi --extra peft --extra training --extra dataset \
        && "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import importlib.metadata as m; [m.version(p) for p in ('torch', 'accelerate', 'transformers', 'lerobot')]" \
        && break
    [ "$attempt" = 3 ] && { echo "uv sync failed 3 times" >&2; exit 1; }
    rm -rf "$UV_PROJECT_ENVIRONMENT"
    sleep $((30 + RANDOM % 60))
done
~/.local/bin/uv pip install --quiet --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    mujoco==3.4.0 "dm-robotics-transformations>=0.10.0" "tyro>=1.0.16"
# --no-deps: the plugin pins lerobot==0.6.1, which would replace this fork.
~/.local/bin/uv pip install --quiet --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$DEXJOCO"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"

# Drop AdamW state from every checkpoint but the newest: resume only ever reads `last`,
# and the pruned rungs keep pretrained_model/ so the checkpoint ladder stays usable for
# eval and for dual-noise selection.
prune_loop () {
    local root="$1"
    while true; do
        sleep 300
        [ -d "$root" ] || continue
        local last_target
        last_target=$(basename "$(readlink -f "$root/last")" 2>/dev/null)
        [ -n "$last_target" ] && [ "$last_target" != last ] || continue
        for d in "$root"/[0-9]*; do
            [ -d "$d" ] || continue
            [ "$(basename "$d")" = "$last_target" ] && continue
            if [ -d "$d/training_state" ]; then
                rm -rf "$d/training_state"
                echo "[prune] dropped optimizer state from $(basename "$d") (keeping $last_target)"
            fi
        done
    done
}
