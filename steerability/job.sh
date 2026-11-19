#!/usr/bin/env bash
# Run any command with the steerability environment set, regardless of the submitting shell.
set -euo pipefail
cd ~/scratch/dexrobot/lerobot
source ~/scratch/envs/lerobot_robocasa/bin/activate
export PYTHONPATH=$PWD/steerability:$PWD:${PYTHONPATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export ENV_FACTORY=steerability_wiring:make_cell_env OBS_FN=steerability_wiring:pi05_frame
exec "$@"