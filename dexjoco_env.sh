# DexJoCo pins mujoco 3.4.0 while robosuite needs the 3.2.6 in .venv, hence a separate venv.
# Node-local because imports from a venv on the networked scratch filesystem have failed
# intermittently in spawned workers.

DEXJOCO=${DEXJOCO:-/network/scratch/a/artur.kuramshin/dexjoco}

source ~/.bashrc          # HF_HOME / HF_TOKEN live here on Mila
export UV_CACHE_DIR=${UV_CACHE_DIR:-/network/scratch/a/artur.kuramshin/uv_cache}
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/venv_${SLURM_JOB_ID:-local}"

# Jobs starting together race on the shared uv cache (a file one job is still unpacking is
# missing for the other), so retry after a random delay.
for attempt in 1 2 3; do
    ~/.local/bin/uv sync --frozen --quiet --extra pi --extra peft --extra training --extra dataset && break
    [ "$attempt" = 3 ] && { echo "uv sync failed 3 times" >&2; exit 1; }
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
