#!/bin/bash
# CPU-only check that a cluster can run the dexrobot jobs, before spending GPU hours there.
#
#   cluv submit narval scripts/cluv/check.sh
#
# 1. data: checksums of the banks and HF files against Mila's manifest; pi05_base size
# 2. imports, numpy pin, ScrewLightbulb / CoffeePressButton registered
# 3. offline Hub cache: pi05_base config, PaliGemma tokenizer, both datasets' metadata
# 4. the lamp scene builds and resets; numbers must match Mila
#    (nbody 276, ngeom 1450, nmesh 131, qpos sum 1.934530)
# 5. SFT dry run, Expert-only lamp fp32 (expect ~693M trainable params, assertions pass)
#
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00

export FORCE_MUJOCO_GL=disable   # CPU nodes have no EGL
source scripts/cluv/common.sh
export MUJOCO_GL=disable; unset PYOPENGL_PLATFORM
FAIL=0
step () { echo; echo "### $*"; }

step "1. data"
( cd "$DEXROBOT_DATA" && md5sum --quiet -c MANIFEST.md5 ) && echo "checksums OK" || FAIL=1
while read -r size path; do
    got=$(stat -L -c %s "$DEXROBOT_DATA/$path")
    [ "$got" = "$size" ] && echo "size OK: $path" || { echo "SIZE MISMATCH: $path $got != $size"; FAIL=1; }
done < "$DEXROBOT_DATA/MANIFEST.sizes"

step "2-4. imports, envs, offline cache, scene"
python -X faulthandler - <<'PY' || FAIL=1
import os, sys
import numpy as np
assert np.__version__ == "2.2.5", np.__version__
import torch, transformers, accelerate, peft, lerobot, mujoco
print("torch", torch.__version__, "| transformers", transformers.__version__, "| mujoco", mujoco.__version__)
import robosuite, robocasa
from robosuite.environments.base import REGISTERED_ENVS
for name in ("ScrewLightbulb", "CoffeePressButton"):
    assert name in REGISTERED_ENVS, f"{name} not registered"
print("envs registered OK")

from huggingface_hub import hf_hub_download
cfg = hf_hub_download("lerobot/pi05_base", "config.json")
st = hf_hub_download("lerobot/pi05_base", "model.safetensors")
print("pi05_base offline OK:", os.path.getsize(st) // 2**20, "MiB")
tok = transformers.AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("paligemma tokenizer offline OK, vocab", len(tok))
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
for repo in ("akuramshin/robocasa_lightbulbscrew_dex_filtered",
             "akuramshin/robocasa_coffeepressbutton_dex_augstyle"):
    meta = LeRobotDatasetMetadata(repo)
    print(f"{repo} offline OK: {meta.total_episodes} episodes, {meta.total_frames} frames")

from robosuite.controllers import load_composite_controller_config
cc = os.path.join(robosuite.__path__[0], "controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json")
ccfg = load_composite_controller_config(controller=cc, robot="XArm6DexLeapRHOmron")
env = robosuite.make("ScrewLightbulb", robots="XArm6DexLeapRHOmron", controller_configs=ccfg,
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, seed=1)
env.reset()
m = env.sim.model
got = (m.nbody, m.ngeom, m.nmesh, round(float(np.sum(env.sim.data.qpos)), 6))
print("scene:", got)
assert got == (276, 1450, 131, 1.93453), f"scene differs from Mila: {got}"
print("scene matches Mila OK")
PY

step "5. SFT dry run"
python examples/training/train_pi05_casa_experiments.py --experiment arm0_expert_only --task lamp \
    --dtype float32 --save-steps 500 1000 --stop-step 1000 --dry-run 2>&1 \
    | grep -vE "^(WARNING|INFO)" | tail -15
[ "${PIPESTATUS[0]}" = "0" ] || FAIL=1

echo
[ "$FAIL" = "0" ] && echo "=== ALL CHECKS PASSED ===" || echo "=== CHECKS FAILED ==="
exit $FAIL
