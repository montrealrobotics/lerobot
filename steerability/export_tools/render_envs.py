"""Render the six environments for the task figure, built exactly as the eval builds them."""

import os
import sys
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="numba_"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "/home/mila/a/artur.kuramshin/lerobot/examples/training")

from pathlib import Path

from PIL import Image
from train_pi05_casa_experiments import TASKS, resolve_controller

from lerobot.envs.robocasa_env import RoboCasaEnv as RoboCasaGymEnv
from lerobot.envs.robocasa_placement_bank import (
    PlacementBankWrapper,
    StartPoseRejectionWrapper,
    find_entry,
    load_placement_bank,
)
from lerobot.utils.import_utils import register_third_party_plugins

RES = int(os.environ.get("RES", 1024))
CAM = os.environ.get("CAM", "robot0_agentview_left")
OUT = Path(sys.argv[1])
BANKS = Path("/network/scratch/a/artur.kuramshin/lerobot/placement_banks")
FPS = {"lamp": 30, "coffee": 20}
# (task, file stem, construction seed, lamp placement id)
CELLS = [
    ("lamp", "lamp_training_1_s1_start_0", 1, "s1_start_0"),
    ("lamp", "lamp_training_2_s1_start_1", 1, "s1_start_1"),
    ("lamp", "lamp_held_out_s1_mid", 1, "s1_mid"),
    ("coffee", "coffee_training_1_kitchen0_L8S6", 0, None),
    ("coffee", "coffee_training_2_kitchen1_L4S8", 1, None),
    ("coffee", "coffee_held_out_kitchen11_L1S4", 11, None),
]

register_third_party_plugins()
lamp_bank = load_placement_bank(BANKS / "screwlightbulb_xarm6_seed1_pair" / "bank.json")
OUT.mkdir(parents=True, exist_ok=True)
for task, stem, seed, placement in CELLS:
    spec = TASKS[task]
    env = RoboCasaGymEnv(
        task_name=spec.robocasa_task,
        robot=spec.robot,
        controller=resolve_controller(spec.controller_filename),
        control_freq=FPS[task],
        camera_name=CAM,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=RES,
        observation_height=RES,
        seed=seed,
    )
    if task == "coffee":  # same start-pose rejection as the coffee DSRL and before-RL evals
        env = StartPoseRejectionWrapper(env, max_rot_deg=10.0, max_pos_m=0.03, max_retries=10)
    if placement is not None:
        env = PlacementBankWrapper(env, lamp_bank, seed, [find_entry(lamp_bank, seed, placement)], shuffle=False)
    obs, _ = env.reset()
    img = next(iter(obs["pixels"].values()))
    Image.fromarray(img).save(OUT / f"{stem}.png")
    print(f"{stem}: {img.shape}", flush=True)
    env.close()
