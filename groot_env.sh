#!/bin/bash
# Run a command inside the GR00T N1.7 environment, regardless of what the submitting shell had active.
#
#   sbatch ... --wrap "~/lerobot-groot/groot_env.sh python examples/training/groot_phase0_smoke.py ..."
#
# sbatch exports the caller's environment by default, so a shell with the pi05 venv active leaks its
# VIRTUAL_ENV/PATH into the job and Python then imports ~/lerobot (GR00T N1.5) instead of this
# worktree. Everything below resets that first. The venv lives on scratch: HOME's 100G quota cannot
# hold it.
set -euo pipefail
GROOT_REPO=/home/mila/a/artur.kuramshin/lerobot-groot
GROOT_VENV=/network/scratch/a/artur.kuramshin/venvs/lerobot-groot

unset VIRTUAL_ENV PYTHONPATH PYTHONHOME CONDA_PREFIX
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -e '/lerobot/.venv/bin' -e '/envs/' | paste -sd:)
export PATH="$GROOT_VENV/bin:$PATH" VIRTUAL_ENV="$GROOT_VENV"
source ~/.bashrc >/dev/null 2>&1 || true   # HF_HOME / HF_TOKEN
export PATH="$GROOT_VENV/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-$$}"
cd "$GROOT_REPO"

python - <<'PY'
import lerobot, transformers, sys
assert lerobot.__file__.startswith("/home/mila/a/artur.kuramshin/lerobot-groot/"), lerobot.__file__
assert transformers.__version__ >= "5.4", transformers.__version__
print(f"[groot_env] python {sys.executable} | lerobot {lerobot.__file__} | transformers {transformers.__version__}")
PY
exec "$@"
