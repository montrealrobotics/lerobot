"""The two hooks the RoboCasa adapter needs. Pass

    --env-factory steerability_wiring:make_cell_env
    --obs-fn      steerability_wiring:pi05_frame

Both reproduce what the DSRL eval does (examples/tutorial/rl/dsrl_pi05_robocasa_example.py
--eval_only, task invariants from dsrl_task_args.sh), or the diagnostics measure a different setup.
"""
import json
import os

import gymnasium as gym
import numpy as np

# September DSRL bank (arms 0-3), regenerated 2026-10-02 from the seed-1 calib bank; md5 919c6488.
# The 09-27 rebuild at ..._seed1_pair/ (md5 c9a65d35, used by lamp arm4 DSRL) has different starts.
LAMP_BANK = "/network/projects/real-g-grp/dex_vla_checkpoints/placement_banks/screwlightbulb_xarm6_seed1_pair/bank.json"  # Artur's original; pose-identical to _pair_dsrl
COFFEE_BANK = "/network/projects/real-g-grp/dex_vla_checkpoints/placement_banks/coffeepressbutton_pandadex_kitchens/bank.json"  # Artur's original; equivalent to my copy
# Fallback only: like DSRL's --prompt, used when the env exposes no ep_meta['lang'].
PROMPTS = {
    "lamp": "screw the light bulb into the lamp base",
    "coffee": "press the button on the coffee machine to serve coffee",
}
CAMERAS = "robot0_agentview_left,robot0_eye_in_hand"
THIRD_PERSON, WRIST = "robot0_agentview_left", "robot0_eye_in_hand"


def _joint_pos_controller() -> str:
    import robosuite

    return os.path.join(
        os.path.dirname(robosuite.__file__),
        "controllers/config/robots/default_xarm6dexleaprhomron_joint_pos.json",
    )


# Controller and fps must match the checkpoint: a wrong controller silently falls back to OSC_POSE.
ENV_SPECS = {
    "lamp": dict(task="ScrewLightbulb", robot="XArm6DexLeapRHOmron", fps=30, controller=_joint_pos_controller,
                 bank=LAMP_BANK, rejection=None),
    "coffee": dict(task="CoffeePressButton", robot="PandaDexLeapRHOmron", fps=20, controller=None,
                   bank=COFFEE_BANK, rejection=dict(max_rot_deg=10.0, max_pos_m=0.03, max_retries=10)),
}

# Per-task language description seen at the last reset (DSRL prefers it over --prompt).
_LANG: dict[str, str] = {}


class _CellEnv(gym.Wrapper):
    """Outermost wrapper: reseeds robosuite's reset-noise RNG (``env.rng``, otherwise one stream from
    the construction seed) from reset(seed=...) and records the episode's language description for
    `pi05_frame`. Seeding happens once per reset, outside the start-pose rejection, so its retries
    still draw fresh noise."""

    def __init__(self, env, task):
        super().__init__(env)
        self.task = task

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.unwrapped._env.rng = np.random.default_rng(seed)
        out = self.env.reset(seed=seed, options=options)
        lang = self.env.unwrapped._env.get_ep_meta().get("lang")
        if isinstance(lang, str) and lang:
            _LANG[self.task] = lang
        return out


def _cell_seed_and_entry(task: str, cell: str, bank: dict):
    """Lamp cells are placement ids in scene 1 ('s1_start_0'); coffee cells are kitchens
    ('seed0_L8S6', no placement entry)."""
    if task == "lamp":
        from lerobot.envs.robocasa_placement_bank import find_entry

        for seed in bank["_scenes_by_seed"]:
            try:
                return seed, find_entry(bank, seed, cell)
            except KeyError:
                continue
        raise KeyError(f"cell {cell!r} is not a placement in {LAMP_BANK}")
    seed = int(cell.split("_", 1)[0].removeprefix("seed"))
    known = {k["scene_seed"]: k for k in bank["train"] + bank["heldout"]}
    if seed not in known:
        raise KeyError(f"cell {cell!r}: kitchen seed {seed} is not in {COFFEE_BANK}")
    k = known[seed]
    if cell != f"seed{seed}_L{k['layout_id']}S{k['style_id']}":
        raise ValueError(f"cell {cell!r} does not match bank kitchen seed{seed}_L{k['layout_id']}S{k['style_id']}")
    return seed, None


def make_cell_env(task: str, cell: str, seed: int):
    """robosuite/robocasa env whose every reset() lands in `cell`.

    Same construction as scripts/eval_sft_on_banks.py / DSRL --eval_only: robot, controller,
    fps, the two DSRL cameras at the default 256x256, start-pose rejection inside the
    PlacementBankWrapper pinned to ONE entry. Only difference: return_raw_obs=True, so step/reset
    return robosuite obs dicts (images unflipped; `pi05_frame` applies the flip).
    `seed` only seeds reset noise; the kitchen is fixed by the cell.
    """
    from lerobot.envs.robocasa_env import RoboCasaEnv as RoboCasaGymEnv
    from lerobot.envs.robocasa_placement_bank import (
        PlacementBankWrapper,
        StartPoseRejectionWrapper,
        load_placement_bank,
    )

    spec = ENV_SPECS[task]
    if task == "lamp":
        bank = load_placement_bank(spec["bank"])
    else:
        with open(os.path.expanduser(spec["bank"])) as f:
            bank = json.load(f)
    if bank["task"] != spec["task"] or bank["robot"] != spec["robot"]:
        raise ValueError(f"bank is for {bank['task']}/{bank['robot']}, wiring expects {spec['task']}/{spec['robot']}")
    scene_seed, entry = _cell_seed_and_entry(task, cell, bank)

    controller = spec["controller"]() if callable(spec["controller"]) else spec["controller"]
    env = RoboCasaGymEnv(
        task_name=spec["task"],
        robot=spec["robot"],
        controller=controller,
        control_freq=spec["fps"],
        camera_name=CAMERAS,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=256,
        observation_height=256,
        seed=scene_seed,
        return_raw_obs=True,
    )
    # The adapter finds the robosuite env by walking .env / .unwrapped; the gym env keeps it in ._env.
    env.env = env._env
    if spec["rejection"] is not None:
        env = StartPoseRejectionWrapper(env, **spec["rejection"])
    if entry is not None:
        env = PlacementBankWrapper(env, bank, scene_seed, [entry], shuffle=False)
    return _CellEnv(env, task)


def pi05_frame(task: str, obs: dict) -> dict:
    """robosuite obs dict -> batch dict (leading dim 1) for the checkpoint's preprocessor.

    Same frame the frozen pi05 sees in DSRL: RoboCasaEnv._format_raw_obs (vertical flip,
    state = joint_pos + hand qpos) then preprocess_observation (uint8 HWC -> float CHW in [0,1]),
    plus the episode's language description. No embodiment_id, as in DSRL's default (the model
    falls back to config.default_embodiment_id).
    """
    import torch

    from lerobot.envs.robocasa_env import ROBOT_CONFIGS

    def img(cam):
        x = np.ascontiguousarray(obs[f"{cam}_image"][::-1])
        return torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255

    state = np.concatenate(
        [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][ROBOT_CONFIGS[ENV_SPECS[task]["robot"]]["gripper_qpos_slice"]]]
    )
    return {
        f"observation.images.{THIRD_PERSON}": img(THIRD_PERSON),
        f"observation.images.{WRIST}": img(WRIST),
        "observation.state": torch.from_numpy(state)[None].float(),
        "task": [_LANG.get(task, PROMPTS[task])],
    }
