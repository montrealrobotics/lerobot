#!/usr/bin/env python
"""3-Cube Stacking environment for LeRobot.

A Sawyer robot arm must stack three cubes (A/red, B/green, C/blue) into a
vertical tower: B on A, then C on top of B.

Control: 4-DOF end-effector delta (dx, dy, dz, gripper), clipped to [-1, 1].
Observation: RGB pixels (96×96) + agent_pos (7-dim state vector).
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

import metaworld

# ── Constants ────────────────────────────────────────────────────────────────
CUBE_HALF_SIZE = 0.030       # 6 cm cube, half-size (matches MetaWorld cylinder resting height)
CUBE_SIZE      = 2 * CUBE_HALF_SIZE   # full edge length = 0.060 m

ACTION_SCALE   = 0.01        # metres per action unit (matches MetaWorld)
MOCAP_LOW      = np.array([-0.5,  0.4,  0.07])
MOCAP_HIGH     = np.array([ 0.5,  1.0,  0.50])
MOCAP_INIT_POS = np.array([ 0.0,  0.6,  0.2 ])   # starting position of end-effector (matches MetaWorld home)

# Initial Sawyer arm joint angles matching MetaWorld's reset configuration.
# These put the hand near [0, 0.6, 0.2] so the weld constraint converges quickly.
INIT_QPOS_ARM = np.array([
     1.88928818e+00, -5.75768906e-01, -9.76659350e-01,
     1.64199149e+00,  9.42859626e-01,  1.04369624e+00,
     2.29283331e+00,  0.0, 0.0,  # gripper joints
])
FRAME_SKIP     = 5           # physics sub-steps per action (80 Hz control)

# Vertical offset between mocap body and the cube centre when gripper is
# at grasp height.  Empirically: MetaWorld expert grasps obj at z≈0.03
# with mocap_z = 0.07  →  offset ≈ 0.047.  Use 0.050 for safety margin.
GRIPPER_CUBE_OFFSET = 0.050

# Table workspace for random cube initialisation
CUBE_X_RANGE  = (-0.25, 0.25)
CUBE_Y_RANGE  = ( 0.45, 0.75)
MIN_CUBE_SEP  = 0.10         # minimum distance between any two cube centres

# Success tolerances
STACK_XY_TOL  = 0.03         # metres — horizontal tolerance for "on top of"
STACK_Z_TOL   = 0.015        # metres — vertical tolerance from ideal stacked z
SUCCESS_STEPS = 5            # consecutive steps the stack must hold

# XML path (written to MetaWorld's asset directory so relative includes resolve)
_METAWORLD_ASSETS = Path(metaworld.__file__).parent / "assets"
_XML_PATH = _METAWORLD_ASSETS / "sawyer_xyz" / "sawyer_cube_stacking.xml"

# Camera name matching MetaWorld bin-picking setup
CAMERA_NAME = "corner2"
CAMERA_POS  = np.array([0.75, 0.075, 0.7])   # override: same as MetaWorld bin-picking


# ── Oracle phases ─────────────────────────────────────────────────────────────
(
    PH_ABOVE_B, PH_DOWN_B, PH_GRASP_B, PH_LIFT_B,
    PH_CARRY_B, PH_PLACE_B, PH_RELEASE_B, PH_RETREAT1,
    PH_ABOVE_C, PH_DOWN_C, PH_GRASP_C, PH_LIFT_C,
    PH_CARRY_C, PH_PLACE_C, PH_RELEASE_C, PH_DONE,
) = range(16)

_LIFT_Z    = 0.22          # carry height
_APPROACH_GAIN  = 20.0     # P-gain for approach moves
_FINE_GAIN      = 12.0     # P-gain for precision moves (lowering)


class StackingOraclePolicy:
    """State-machine oracle that stacks B on A, then C on the stack.

    Usage::
        oracle = StackingOraclePolicy()
        obs, info = env.reset()
        oracle.reset(env)
        for _ in range(max_steps):
            action = oracle.get_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def reset(self, env: "StackingEnv") -> None:
        self.phase = PH_ABOVE_B
        self.phase_steps = 0

    def get_action(self, env: "StackingEnv") -> np.ndarray:
        mocap  = env.data.mocap_pos[0].copy()
        pos_A  = env.data.body("cube_A").xpos.copy()
        pos_B  = env.data.body("cube_B").xpos.copy()
        pos_C  = env.data.body("cube_C").xpos.copy()
        self.phase_steps += 1

        # ── Helpers ──────────────────────────────────────────────────────────
        def delta(target, gain=_APPROACH_GAIN):
            return np.clip((target - mocap) * gain, -1.0, 1.0)

        def dist(target):
            return float(np.linalg.norm(target - mocap))

        def xy_dist(target):
            return float(np.linalg.norm(target[:2] - mocap[:2]))

        def advance():
            self.phase += 1
            self.phase_steps = 0

        # ── Phase logic ───────────────────────────────────────────────────────
        gripper = -1.0   # open by default

        if self.phase == PH_ABOVE_B:
            # Move above cube_B at carry height (open gripper)
            target = np.array([pos_B[0], pos_B[1], _LIFT_Z])
            if dist(target) < 0.025:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_DOWN_B:
            # Lower to grasp height above cube_B.
            # The mocap can't go below MOCAP_LOW[2]=0.07, so we target that
            # floor; the gripper fingers extend ~0.05 m below the mocap body.
            grasp_z = MOCAP_LOW[2]
            target = np.array([pos_B[0], pos_B[1], grasp_z])
            xy_ok = np.linalg.norm(target[:2] - mocap[:2]) < 0.020
            z_ok  = mocap[2] <= grasp_z + 0.005
            if xy_ok and z_ok and self.phase_steps > 5:
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH_GRASP_B:
            # Close gripper (hold for 20 steps)
            if self.phase_steps >= 20:
                advance()
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.phase == PH_LIFT_B:
            # Lift straight up
            gripper = 1.0
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_CARRY_B:
            # Move horizontally above cube_A
            gripper = 1.0
            target = np.array([pos_A[0], pos_A[1], _LIFT_Z])
            if xy_dist(target) < 0.020:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_PLACE_B:
            # Lower until cube_B rests on cube_A.
            # target mocap_z = cube_A.z + CUBE_SIZE + GRIPPER_CUBE_OFFSET
            gripper = 1.0
            place_z = pos_A[2] + CUBE_SIZE + GRIPPER_CUBE_OFFSET
            place_z = max(place_z, MOCAP_LOW[2])
            target = np.array([pos_A[0], pos_A[1], place_z])
            stacked_z = pos_A[2] + CUBE_SIZE
            b_placed = (abs(pos_B[2] - stacked_z) < 0.015 and
                        np.linalg.norm(pos_B[:2] - pos_A[:2]) < 0.04)
            if b_placed or (dist(target) < 0.012 and self.phase_steps > 15):
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH_RELEASE_B:
            # Open gripper, stay put (10 steps)
            if self.phase_steps >= 15:
                advance()
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == PH_RETREAT1:
            # Lift away from the stack before going for cube_C
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_ABOVE_C:
            # Move above cube_C
            target = np.array([pos_C[0], pos_C[1], _LIFT_Z])
            if dist(target) < 0.025:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_DOWN_C:
            grasp_z = MOCAP_LOW[2]
            target = np.array([pos_C[0], pos_C[1], grasp_z])
            xy_ok = np.linalg.norm(target[:2] - mocap[:2]) < 0.020
            z_ok  = mocap[2] <= grasp_z + 0.005
            if xy_ok and z_ok and self.phase_steps > 5:
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH_GRASP_C:
            if self.phase_steps >= 20:
                advance()
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.phase == PH_LIFT_C:
            gripper = 1.0
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_CARRY_C:
            # Move above stack (cube_A's xy)
            gripper = 1.0
            target = np.array([pos_A[0], pos_A[1], _LIFT_Z])
            if xy_dist(target) < 0.020:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH_PLACE_C:
            gripper = 1.0
            place_z = pos_A[2] + 2 * CUBE_SIZE + GRIPPER_CUBE_OFFSET
            place_z = max(place_z, MOCAP_LOW[2])
            target = np.array([pos_A[0], pos_A[1], place_z])
            stacked_z = pos_A[2] + 2 * CUBE_SIZE
            c_placed = (abs(pos_C[2] - stacked_z) < 0.015 and
                        np.linalg.norm(pos_C[:2] - pos_A[:2]) < 0.04)
            if c_placed or (dist(target) < 0.012 and self.phase_steps > 15):
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        else:  # PH_RELEASE_C or PH_DONE
            return np.array([0.0, 0.0, 0.0, -1.0])


# ── Environment ───────────────────────────────────────────────────────────────

class StackingEnv(gym.Env):
    """3-cube stacking environment using MetaWorld's Sawyer arm assets.

    Observation (pixels_agent_pos):
        - pixels: (H, W, 3) uint8 RGB
        - agent_pos: (7,) float32 = [mocap_x, mocap_y, mocap_z, gripper,
                                      cube_A_x, cube_A_y, cube_A_z]
          (cube_B and C visible only through the image)

    Action: (4,) float32 = [dx, dy, dz, gripper], clipped to [-1, 1]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 96,
        observation_height: int = 96,
        visualization_width: int = 480,
        visualization_height: int = 480,
        camera_name: str = CAMERA_NAME,
    ):
        super().__init__()
        self.obs_type         = obs_type
        self.render_mode      = render_mode
        self.observation_width  = observation_width
        self.observation_height = observation_height
        self.visualization_width  = visualization_width
        self.visualization_height = visualization_height
        self.camera_name      = camera_name

        # Task metadata (required by LeRobot utils)
        self.task = "cube_stacking"
        self.task_description = (
            "Stack the three cubes into a vertical tower: "
            "place the green cube on the red cube, then place the blue cube on top."
        )
        self._max_episode_steps = 600

        # Load MuJoCo model
        if not _XML_PATH.exists():
            raise FileNotFoundError(
                f"Could not find stacking XML at {_XML_PATH}. "
                "The file should have been created by the LeRobot install."
            )
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data  = mujoco.MjData(self.model)

        # Adjust camera position (same as MetaWorld bin-picking corner2 override)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self.model.cam_pos[cam_id] = CAMERA_POS

        # Renderer (lazy-init in render())
        self._renderer = None

        # Oracle policy (available for data collection scripts)
        self.expert_policy = StackingOraclePolicy()

        # Success counter
        self._success_steps = 0

        # Observation space
        if obs_type == "state":
            raise NotImplementedError("State-only obs_type not supported.")
        elif obs_type in ("pixels", "pixels_agent_pos"):
            obs_dict: dict[str, spaces.Space] = {
                "pixels": spaces.Box(
                    low=0, high=255,
                    shape=(observation_height, observation_width, 3),
                    dtype=np.uint8,
                )
            }
            if obs_type == "pixels_agent_pos":
                obs_dict["agent_pos"] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                )
            self.observation_space = spaces.Dict(obs_dict)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type!r}")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply a 4-DOF action: [dx, dy, dz, gripper]."""
        action = np.clip(action, -1.0, 1.0)
        # End-effector delta (same as MetaWorld's set_xyz_action)
        new_pos = self.data.mocap_pos[0] + action[:3] * ACTION_SCALE
        self.data.mocap_pos[0] = np.clip(new_pos, MOCAP_LOW, MOCAP_HIGH)
        self.data.mocap_quat[0] = [1.0, 0.0, 1.0, 0.0]  # fixed orientation
        # Gripper: +1 = close, -1 = open
        self.data.ctrl[0] =  action[3]
        self.data.ctrl[1] = -action[3]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Return observation dict."""
        image = self._render_frame()
        if self.obs_type == "pixels":
            return {"pixels": image}
        # pixels_agent_pos: 4-dim proprioceptive state — robot only, no cube positions.
        # Cube locations are perceived via the image. Identical dim to TwoCubeStackingEnv
        # so 2-cube checkpoints finetune to 3-cube with zero architecture changes.
        mocap   = self.data.mocap_pos[0].astype(np.float64)
        gripper = float(self.data.ctrl[0])
        agent_pos = np.array([*mocap, gripper], dtype=np.float64)
        return {"pixels": image, "agent_pos": agent_pos}

    def _render_frame(self) -> np.ndarray:
        """Render a frame at the observation resolution (96×96 for policy input)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.visualization_height,
                width=self.visualization_width,
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        img = self._renderer.render()          # HWC uint8
        # MetaWorld's corner2 camera outputs a flipped image; match that
        img = np.flip(img, (0, 1)).copy()
        if img.shape[0] != self.observation_height or img.shape[1] != self.observation_width:
            img = cv2.resize(
                img, (self.observation_width, self.observation_height),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def _render_visualization(self) -> np.ndarray:
        """Render a frame at the full visualization resolution (480×480 for video)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.visualization_height,
                width=self.visualization_width,
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        img = self._renderer.render()          # HWC uint8 at visualization size
        return np.flip(img, (0, 1)).copy()

    def _is_stacked(self) -> tuple[bool, bool]:
        """Return (b_on_a, c_on_b): whether each stacking step is achieved."""
        pos_A = self.data.body("cube_A").xpos
        pos_B = self.data.body("cube_B").xpos
        pos_C = self.data.body("cube_C").xpos

        b_on_a = (
            np.linalg.norm(pos_B[:2] - pos_A[:2]) < STACK_XY_TOL and
            abs(pos_B[2] - (pos_A[2] + CUBE_SIZE)) < STACK_Z_TOL
        )
        c_on_b = (
            np.linalg.norm(pos_C[:2] - pos_B[:2]) < STACK_XY_TOL and
            abs(pos_C[2] - (pos_B[2] + CUBE_SIZE)) < STACK_Z_TOL
        )
        return b_on_a, c_on_b

    def _compute_reward(self) -> float:
        """Dense staged reward."""
        pos_A = self.data.body("cube_A").xpos
        pos_B = self.data.body("cube_B").xpos
        pos_C = self.data.body("cube_C").xpos
        mocap = self.data.mocap_pos[0]

        b_on_a, c_on_b = self._is_stacked()

        if not b_on_a:
            # Stage 1: pick B and place on A
            r_reach = float(np.exp(-10 * np.linalg.norm(mocap - pos_B))) * 0.3
            r_place = float(np.exp(-10 * np.linalg.norm(pos_B[:2] - pos_A[:2]))) * 0.5
            return r_reach + r_place
        else:
            # Stage 2: pick C and place on stack
            r_reach = float(np.exp(-10 * np.linalg.norm(mocap - pos_C))) * 0.3
            r_place = float(np.exp(-10 * np.linalg.norm(pos_C[:2] - pos_B[:2]))) * 0.5
            r_bonus = 1.0  # bonus for completing stage 1
            r_full  = 2.0 if c_on_b else 0.0
            return r_reach + r_place + r_bonus + r_full

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # CRITICAL: reset the mocap weld constraint to zero relative offset.
        # Without this, MuJoCo bakes in the zero-qpos relative pose (hand far
        # from mocap) as the weld target → huge constraint forces → arm diverges.
        # This mirrors MetaWorld's SawyerMocapBase.reset_mocap_welds().
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.model.eq_data[i] = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                )

        # Initialise arm to MetaWorld home configuration so the weld converges fast
        self.data.qpos[:len(INIT_QPOS_ARM)] = INIT_QPOS_ARM

        # Randomise cube positions with minimum separation
        positions = []
        for _ in range(3):
            for attempt in range(500):
                x = rng.uniform(*CUBE_X_RANGE)
                y = rng.uniform(*CUBE_Y_RANGE)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - p) >= MIN_CUBE_SEP for p in positions):
                    positions.append(pos)
                    break
            else:
                # fallback: fixed offset grid
                positions.append(np.array([-0.10 + 0.15 * len(positions), 0.60]))

        cube_names = ["cube_A", "cube_B", "cube_C"]
        for name, (x, y) in zip(cube_names, positions):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr : qpos_addr + 3] = [x, y, CUBE_HALF_SIZE + 0.001]  # slight offset above table
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]  # no rotation

        # Initialise end-effector
        self.data.mocap_pos[0]  = MOCAP_INIT_POS.copy()
        self.data.mocap_quat[0] = [1.0, 0.0, 1.0, 0.0]
        self.data.ctrl[:] = [-1.0, 1.0]   # open gripper

        # Warm-up simulation
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._success_steps = 0
        self.expert_policy.reset(self)
        obs = self._get_obs()
        return obs, {"is_success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got shape {action.shape}")

        self._apply_action(action)
        b_on_a, c_on_b = self._is_stacked()
        is_success = bool(b_on_a and c_on_b)

        if is_success:
            self._success_steps += 1
        else:
            self._success_steps = 0

        stable_success = self._success_steps >= SUCCESS_STEPS
        reward = self._compute_reward()
        terminated = stable_success
        obs = self._get_obs()

        info = {
            "is_success":    stable_success,
            "b_on_a":        bool(b_on_a),
            "c_on_b":        bool(c_on_b),
            "partial_stack": int(b_on_a) + int(c_on_b),  # 0, 1, or 2
        }
        if terminated:
            info["final_info"] = {
                "is_success": True,
                "b_on_a": True,
                "c_on_b": True,
            }
            self.reset()

        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray:
        return self._render_visualization()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── Two-cube oracle ───────────────────────────────────────────────────────────

(
    PH2_ABOVE_B, PH2_DOWN_B, PH2_GRASP_B, PH2_LIFT_B,
    PH2_CARRY_B, PH2_PLACE_B, PH2_RELEASE_B, PH2_RETREAT, PH2_DONE,
) = range(9)


class TwoCubeOraclePolicy:
    """Oracle that places cube_B (green) on cube_A (red).

    Executes the same first 8 phases as StackingOraclePolicy, then holds.
    """

    def reset(self, env: "TwoCubeStackingEnv") -> None:
        self.phase = PH2_ABOVE_B
        self.phase_steps = 0

    def get_action(self, env: "TwoCubeStackingEnv") -> np.ndarray:
        mocap  = env.data.mocap_pos[0].copy()
        pos_A  = env.data.body("cube_A").xpos.copy()
        pos_B  = env.data.body("cube_B").xpos.copy()
        self.phase_steps += 1

        def delta(target, gain=_APPROACH_GAIN):
            return np.clip((target - mocap) * gain, -1.0, 1.0)

        def dist(target):
            return float(np.linalg.norm(target - mocap))

        def advance():
            self.phase += 1
            self.phase_steps = 0

        gripper = -1.0  # open by default

        if self.phase == PH2_ABOVE_B:
            target = np.array([pos_B[0], pos_B[1], _LIFT_Z])
            if dist(target) < 0.025:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH2_DOWN_B:
            grasp_z = MOCAP_LOW[2]
            target = np.array([pos_B[0], pos_B[1], grasp_z])
            xy_ok = np.linalg.norm(target[:2] - mocap[:2]) < 0.020
            z_ok  = mocap[2] <= grasp_z + 0.005
            if xy_ok and z_ok and self.phase_steps > 5:
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH2_GRASP_B:
            if self.phase_steps >= 20:
                advance()
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.phase == PH2_LIFT_B:
            gripper = 1.0
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH2_CARRY_B:
            gripper = 1.0
            target = np.array([pos_A[0], pos_A[1], _LIFT_Z])
            if float(np.linalg.norm(target[:2] - mocap[:2])) < 0.020:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH2_PLACE_B:
            gripper = 1.0
            place_z = pos_A[2] + CUBE_SIZE + GRIPPER_CUBE_OFFSET
            place_z = max(place_z, MOCAP_LOW[2])
            target = np.array([pos_A[0], pos_A[1], place_z])
            stacked_z = pos_A[2] + CUBE_SIZE
            b_placed = (abs(pos_B[2] - stacked_z) < 0.015 and
                        np.linalg.norm(pos_B[:2] - pos_A[:2]) < 0.04)
            if b_placed or (dist(target) < 0.012 and self.phase_steps > 15):
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH2_RELEASE_B:
            if self.phase_steps >= 15:
                advance()
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == PH2_RETREAT:
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        else:  # PH2_DONE — hold position
            return np.array([0.0, 0.0, 0.0, -1.0])


# ── Two-cube environment ───────────────────────────────────────────────────────

class TwoCubeStackingEnv(gym.Env):
    """2-cube stacking environment: place cube_B (green) on cube_A (red).

    Observation (pixels_agent_pos):
        - pixels: (H, W, 3) uint8 RGB
        - agent_pos: (10,) float32 = [mocap_x, mocap_y, mocap_z, gripper,
                                       cube_A_x, cube_A_y, cube_A_z,
                                       cube_B_x, cube_B_y, cube_B_z]

    Action: (4,) float32 = [dx, dy, dz, gripper], clipped to [-1, 1]

    Cube_C is present in the scene as an inert distractor — this ensures visual
    features transfer cleanly when finetuning for 3-cube stacking.

    State dim (10) intentionally matches the 3-cube finetune dataset so that
    checkpoint warm-start requires no architecture changes.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 96,
        observation_height: int = 96,
        visualization_width: int = 480,
        visualization_height: int = 480,
        camera_name: str = CAMERA_NAME,
    ):
        super().__init__()
        self.obs_type             = obs_type
        self.render_mode          = render_mode
        self.observation_width    = observation_width
        self.observation_height   = observation_height
        self.visualization_width  = visualization_width
        self.visualization_height = visualization_height
        self.camera_name          = camera_name

        self.task = "two_cube_stacking"
        self.task_description = (
            "Stack the cubes: place the green cube on top of the red cube."
        )
        self._max_episode_steps = 350

        if not _XML_PATH.exists():
            raise FileNotFoundError(
                f"Could not find stacking XML at {_XML_PATH}."
            )
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data  = mujoco.MjData(self.model)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self.model.cam_pos[cam_id] = CAMERA_POS

        self._renderer = None
        self.expert_policy = TwoCubeOraclePolicy()
        self._success_steps = 0

        if obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unknown obs_type: {obs_type!r}")

        obs_dict: dict[str, spaces.Space] = {
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(observation_height, observation_width, 3),
                dtype=np.uint8,
            )
        }
        if obs_type == "pixels_agent_pos":
            obs_dict["agent_pos"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
            )
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        new_pos = self.data.mocap_pos[0] + action[:3] * ACTION_SCALE
        self.data.mocap_pos[0] = np.clip(new_pos, MOCAP_LOW, MOCAP_HIGH)
        self.data.mocap_quat[0] = [1.0, 0.0, 1.0, 0.0]
        self.data.ctrl[0] =  action[3]
        self.data.ctrl[1] = -action[3]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self) -> dict[str, np.ndarray]:
        image = self._render_frame()
        if self.obs_type == "pixels":
            return {"pixels": image}
        # 4-dim proprioceptive state: robot only. Cube positions come from the image.
        mocap   = self.data.mocap_pos[0].astype(np.float64)
        gripper = float(self.data.ctrl[0])
        agent_pos = np.array([*mocap, gripper], dtype=np.float64)
        return {"pixels": image, "agent_pos": agent_pos}

    def _render_frame(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.visualization_height,
                width=self.visualization_width,
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        img = self._renderer.render()
        img = np.flip(img, (0, 1)).copy()
        if img.shape[0] != self.observation_height or img.shape[1] != self.observation_width:
            img = cv2.resize(
                img, (self.observation_width, self.observation_height),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def _render_visualization(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.visualization_height,
                width=self.visualization_width,
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        img = self._renderer.render()
        return np.flip(img, (0, 1)).copy()

    def _is_b_on_a(self) -> bool:
        pos_A = self.data.body("cube_A").xpos
        pos_B = self.data.body("cube_B").xpos
        return (
            np.linalg.norm(pos_B[:2] - pos_A[:2]) < STACK_XY_TOL and
            abs(pos_B[2] - (pos_A[2] + CUBE_SIZE)) < STACK_Z_TOL
        )

    def _compute_reward(self) -> float:
        pos_A  = self.data.body("cube_A").xpos
        pos_B  = self.data.body("cube_B").xpos
        mocap  = self.data.mocap_pos[0]
        b_on_a = self._is_b_on_a()
        if not b_on_a:
            r_reach = float(np.exp(-10 * np.linalg.norm(mocap - pos_B))) * 0.3
            r_place = float(np.exp(-10 * np.linalg.norm(pos_B[:2] - pos_A[:2]))) * 0.5
            return r_reach + r_place
        else:
            return 1.0 + float(np.exp(-10 * np.linalg.norm(pos_B[:2] - pos_A[:2]))) * 0.5

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.model.eq_data[i] = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                )

        self.data.qpos[:len(INIT_QPOS_ARM)] = INIT_QPOS_ARM

        # Randomise all 3 cube positions (cube_C is an inert distractor)
        positions = []
        for _ in range(3):
            for _ in range(500):
                x = rng.uniform(*CUBE_X_RANGE)
                y = rng.uniform(*CUBE_Y_RANGE)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - p) >= MIN_CUBE_SEP for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array([-0.10 + 0.15 * len(positions), 0.60]))

        for name, (x, y) in zip(["cube_A", "cube_B", "cube_C"], positions):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr : qpos_addr + 3] = [x, y, CUBE_HALF_SIZE + 0.001]
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.mocap_pos[0]  = MOCAP_INIT_POS.copy()
        self.data.mocap_quat[0] = [1.0, 0.0, 1.0, 0.0]
        self.data.ctrl[:] = [-1.0, 1.0]

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._success_steps = 0
        self.expert_policy.reset(self)
        obs = self._get_obs()
        return obs, {"is_success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got shape {action.shape}")

        self._apply_action(action)
        b_on_a = self._is_b_on_a()

        if b_on_a:
            self._success_steps += 1
        else:
            self._success_steps = 0

        stable_success = self._success_steps >= SUCCESS_STEPS
        reward = self._compute_reward()
        terminated = stable_success
        obs = self._get_obs()

        info = {
            "is_success": stable_success,
            "b_on_a": bool(b_on_a),
        }
        if terminated:
            info["final_info"] = {"is_success": True, "b_on_a": True}
            self.reset()

        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray:
        return self._render_visualization()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── LeRobot factory function ──────────────────────────────────────────────────

def create_stacking_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls=None,
) -> dict[str, dict[int, Any]]:
    """Create vectorised stacking environments (mirrors create_metaworld_envs API)."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable.")
    gym_kwargs = dict(gym_kwargs or {})
    fns = [lambda: StackingEnv(**gym_kwargs) for _ in range(n_envs)]
    return {"cube_stacking": {0: env_cls(fns)}}


def create_two_cube_stacking_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls=None,
) -> dict[str, dict[int, Any]]:
    """Create vectorised 2-cube stacking environments."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable.")
    gym_kwargs = dict(gym_kwargs or {})
    fns = [lambda: TwoCubeStackingEnv(**gym_kwargs) for _ in range(n_envs)]
    return {"two_cube_stacking": {0: env_cls(fns)}}


# ── Four-cube oracle ──────────────────────────────────────────────────────────

_XML_PATH_4 = _METAWORLD_ASSETS / "sawyer_xyz" / "sawyer_cube_stacking_4.xml"

(
    PH4_ABOVE_B, PH4_DOWN_B, PH4_GRASP_B, PH4_LIFT_B,
    PH4_CARRY_B, PH4_PLACE_B, PH4_RELEASE_B, PH4_RETREAT1,
    PH4_ABOVE_C, PH4_DOWN_C, PH4_GRASP_C, PH4_LIFT_C,
    PH4_CARRY_C, PH4_PLACE_C, PH4_RELEASE_C, PH4_RETREAT2,
    PH4_ABOVE_D, PH4_DOWN_D, PH4_GRASP_D, PH4_LIFT_D,
    PH4_CARRY_D, PH4_PLACE_D, PH4_RELEASE_D, PH4_DONE,
) = range(24)


_LIFT_Z_4HIGH = 0.32   # higher carry height for phases after C is placed — clears 3-high stack


class FourCubeOraclePolicy:
    """Oracle that stacks B on A, C on B, D on C (bottom-to-top: A/red, B/green, C/blue, D/yellow)."""

    def reset(self, env: "FourCubeStackingEnv") -> None:
        self.phase = PH4_ABOVE_B
        self.phase_steps = 0

    def get_action(self, env: "FourCubeStackingEnv") -> np.ndarray:
        mocap = env.data.mocap_pos[0].copy()
        pos_A = env.data.body("cube_A").xpos.copy()
        pos_B = env.data.body("cube_B").xpos.copy()
        pos_C = env.data.body("cube_C").xpos.copy()
        pos_D = env.data.body("cube_D").xpos.copy()
        self.phase_steps += 1

        def delta(target, gain=_APPROACH_GAIN):
            return np.clip((target - mocap) * gain, -1.0, 1.0)

        def dist(target):
            return float(np.linalg.norm(target - mocap))

        def xy_dist(target):
            return float(np.linalg.norm(target[:2] - mocap[:2]))

        def advance():
            self.phase += 1
            self.phase_steps = 0

        gripper = -1.0  # open by default

        # ── Place B on A ──────────────────────────────────────────────────────
        if self.phase == PH4_ABOVE_B:
            target = np.array([pos_B[0], pos_B[1], _LIFT_Z])
            if dist(target) < 0.025:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_DOWN_B:
            grasp_z = MOCAP_LOW[2]
            target = np.array([pos_B[0], pos_B[1], grasp_z])
            if np.linalg.norm(target[:2] - mocap[:2]) < 0.020 and mocap[2] <= grasp_z + 0.005 and self.phase_steps > 5:
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH4_GRASP_B:
            if self.phase_steps >= 20:
                advance()
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.phase == PH4_LIFT_B:
            gripper = 1.0
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_CARRY_B:
            gripper = 1.0
            target = np.array([pos_A[0], pos_A[1], _LIFT_Z])
            if xy_dist(target) < 0.020:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_PLACE_B:
            gripper = 1.0
            place_z = max(pos_A[2] + CUBE_SIZE + GRIPPER_CUBE_OFFSET, MOCAP_LOW[2])
            target = np.array([pos_A[0], pos_A[1], place_z])
            b_placed = abs(pos_B[2] - (pos_A[2] + CUBE_SIZE)) < 0.015 and np.linalg.norm(pos_B[:2] - pos_A[:2]) < 0.04
            if b_placed or (dist(target) < 0.012 and self.phase_steps > 15):
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH4_RELEASE_B:
            if self.phase_steps >= 15:
                advance()
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == PH4_RETREAT1:
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        # ── Place C on B ──────────────────────────────────────────────────────
        elif self.phase == PH4_ABOVE_C:
            target = np.array([pos_C[0], pos_C[1], _LIFT_Z])
            if dist(target) < 0.025:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_DOWN_C:
            grasp_z = MOCAP_LOW[2]
            target = np.array([pos_C[0], pos_C[1], grasp_z])
            if np.linalg.norm(target[:2] - mocap[:2]) < 0.020 and mocap[2] <= grasp_z + 0.005 and self.phase_steps > 5:
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH4_GRASP_C:
            if self.phase_steps >= 20:
                advance()
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.phase == PH4_LIFT_C:
            gripper = 1.0
            target = np.array([mocap[0], mocap[1], _LIFT_Z])
            if mocap[2] >= _LIFT_Z - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_CARRY_C:
            gripper = 1.0
            target = np.array([pos_A[0], pos_A[1], _LIFT_Z])
            if xy_dist(target) < 0.020:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_PLACE_C:
            gripper = 1.0
            place_z = max(pos_A[2] + 2 * CUBE_SIZE + GRIPPER_CUBE_OFFSET, MOCAP_LOW[2])
            target = np.array([pos_A[0], pos_A[1], place_z])
            c_placed = abs(pos_C[2] - (pos_A[2] + 2 * CUBE_SIZE)) < 0.015 and np.linalg.norm(pos_C[:2] - pos_A[:2]) < 0.04
            if c_placed or (dist(target) < 0.012 and self.phase_steps > 15):
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH4_RELEASE_C:
            # Hold longer to let the 3-high stack stabilise before retreating
            if self.phase_steps >= 30:
                advance()
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == PH4_RETREAT2:
            # Must rise to _LIFT_Z_4HIGH to clear the 3-high stack (C top ≈ 0.18 m)
            target = np.array([mocap[0], mocap[1], _LIFT_Z_4HIGH])
            if mocap[2] >= _LIFT_Z_4HIGH - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        # ── Place D on C — use higher lift height to clear 3-high stack ───────
        elif self.phase == PH4_ABOVE_D:
            target = np.array([pos_D[0], pos_D[1], _LIFT_Z_4HIGH])
            if dist(target) < 0.025:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_DOWN_D:
            grasp_z = MOCAP_LOW[2]
            target = np.array([pos_D[0], pos_D[1], grasp_z])
            if np.linalg.norm(target[:2] - mocap[:2]) < 0.020 and mocap[2] <= grasp_z + 0.005 and self.phase_steps > 5:
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        elif self.phase == PH4_GRASP_D:
            if self.phase_steps >= 20:
                advance()
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.phase == PH4_LIFT_D:
            gripper = 1.0
            target = np.array([mocap[0], mocap[1], _LIFT_Z_4HIGH])
            if mocap[2] >= _LIFT_Z_4HIGH - 0.01:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_CARRY_D:
            gripper = 1.0
            target = np.array([pos_A[0], pos_A[1], _LIFT_Z_4HIGH])
            if xy_dist(target) < 0.020:
                advance()
            return np.array([*delta(target), gripper])

        elif self.phase == PH4_PLACE_D:
            gripper = 1.0
            place_z = max(pos_A[2] + 3 * CUBE_SIZE + GRIPPER_CUBE_OFFSET, MOCAP_LOW[2])
            target = np.array([pos_A[0], pos_A[1], place_z])
            d_placed = abs(pos_D[2] - (pos_A[2] + 3 * CUBE_SIZE)) < 0.015 and np.linalg.norm(pos_D[:2] - pos_A[:2]) < 0.04
            if d_placed or (dist(target) < 0.012 and self.phase_steps > 15):
                advance()
            return np.array([*delta(target, _FINE_GAIN), gripper])

        else:  # PH4_RELEASE_D or PH4_DONE
            return np.array([0.0, 0.0, 0.0, -1.0])


# ── Four-cube environment ─────────────────────────────────────────────────────

class FourCubeStackingEnv(gym.Env):
    """4-cube stacking environment: stack B/green on A/red, C/blue on B, D/yellow on C.

    Observation (pixels_agent_pos):
        - pixels: (H, W, 3) uint8 RGB
        - agent_pos: (4,) float32 = [mocap_x, mocap_y, mocap_z, gripper]
          All cube positions come from the image only.

    Action: (4,) float32 = [dx, dy, dz, gripper], clipped to [-1, 1]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 224,
        observation_height: int = 224,
        visualization_width: int = 480,
        visualization_height: int = 480,
        camera_name: str = CAMERA_NAME,
    ):
        super().__init__()
        self.obs_type             = obs_type
        self.render_mode          = render_mode
        self.observation_width    = observation_width
        self.observation_height   = observation_height
        self.visualization_width  = visualization_width
        self.visualization_height = visualization_height
        self.camera_name          = camera_name

        self.task = "four_cube_stacking"
        self.task_description = (
            "Stack the four cubes into a vertical tower: "
            "green on red, blue on green, yellow on blue."
        )
        self._max_episode_steps = 900

        if not _XML_PATH_4.exists():
            raise FileNotFoundError(f"Could not find 4-cube XML at {_XML_PATH_4}.")

        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH_4))
        self.data  = mujoco.MjData(self.model)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self.model.cam_pos[cam_id] = CAMERA_POS

        self._renderer = None
        self.expert_policy = FourCubeOraclePolicy()
        self._success_steps = 0

        if obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unknown obs_type: {obs_type!r}")

        obs_dict: dict[str, spaces.Space] = {
            "pixels": spaces.Box(low=0, high=255,
                                 shape=(observation_height, observation_width, 3),
                                 dtype=np.uint8)
        }
        if obs_type == "pixels_agent_pos":
            obs_dict["agent_pos"] = spaces.Box(low=-np.inf, high=np.inf,
                                               shape=(4,), dtype=np.float64)
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        new_pos = self.data.mocap_pos[0] + action[:3] * ACTION_SCALE
        self.data.mocap_pos[0] = np.clip(new_pos, MOCAP_LOW, MOCAP_HIGH)
        self.data.mocap_quat[0] = [1.0, 0.0, 1.0, 0.0]
        self.data.ctrl[0] =  action[3]
        self.data.ctrl[1] = -action[3]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self) -> dict[str, np.ndarray]:
        image = self._render_frame()
        if self.obs_type == "pixels":
            return {"pixels": image}
        mocap = self.data.mocap_pos[0].astype(np.float64)
        gripper = float(self.data.ctrl[0])
        return {"pixels": image, "agent_pos": np.array([*mocap, gripper], dtype=np.float64)}

    def _render_frame(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model,
                                             height=self.visualization_height,
                                             width=self.visualization_width)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        img = np.flip(self._renderer.render(), (0, 1)).copy()
        if img.shape[0] != self.observation_height or img.shape[1] != self.observation_width:
            img = cv2.resize(img, (self.observation_width, self.observation_height),
                             interpolation=cv2.INTER_AREA)
        return img

    def _render_visualization(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model,
                                             height=self.visualization_height,
                                             width=self.visualization_width)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return np.flip(self._renderer.render(), (0, 1)).copy()

    def _is_stacked(self) -> tuple[bool, bool, bool]:
        """Return (b_on_a, c_on_b, d_on_c)."""
        pos_A = self.data.body("cube_A").xpos
        pos_B = self.data.body("cube_B").xpos
        pos_C = self.data.body("cube_C").xpos
        pos_D = self.data.body("cube_D").xpos
        b_on_a = (np.linalg.norm(pos_B[:2] - pos_A[:2]) < STACK_XY_TOL and
                  abs(pos_B[2] - (pos_A[2] + CUBE_SIZE)) < STACK_Z_TOL)
        c_on_b = (np.linalg.norm(pos_C[:2] - pos_B[:2]) < STACK_XY_TOL and
                  abs(pos_C[2] - (pos_B[2] + CUBE_SIZE)) < STACK_Z_TOL)
        d_on_c = (np.linalg.norm(pos_D[:2] - pos_C[:2]) < STACK_XY_TOL and
                  abs(pos_D[2] - (pos_C[2] + CUBE_SIZE)) < STACK_Z_TOL)
        return b_on_a, c_on_b, d_on_c

    def _compute_reward(self) -> float:
        pos_A = self.data.body("cube_A").xpos
        pos_B = self.data.body("cube_B").xpos
        pos_C = self.data.body("cube_C").xpos
        pos_D = self.data.body("cube_D").xpos
        mocap = self.data.mocap_pos[0]
        b_on_a, c_on_b, d_on_c = self._is_stacked()

        if not b_on_a:
            return (float(np.exp(-10 * np.linalg.norm(mocap - pos_B))) * 0.3 +
                    float(np.exp(-10 * np.linalg.norm(pos_B[:2] - pos_A[:2]))) * 0.5)
        elif not c_on_b:
            return (1.0 +
                    float(np.exp(-10 * np.linalg.norm(mocap - pos_C))) * 0.3 +
                    float(np.exp(-10 * np.linalg.norm(pos_C[:2] - pos_B[:2]))) * 0.5)
        elif not d_on_c:
            return (2.0 +
                    float(np.exp(-10 * np.linalg.norm(mocap - pos_D))) * 0.3 +
                    float(np.exp(-10 * np.linalg.norm(pos_D[:2] - pos_C[:2]))) * 0.5)
        else:
            return 3.0

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.model.eq_data[i] = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                )

        self.data.qpos[:len(INIT_QPOS_ARM)] = INIT_QPOS_ARM

        # Randomise all 4 cube positions with minimum separation
        positions = []
        for _ in range(4):
            for _ in range(500):
                x = rng.uniform(*CUBE_X_RANGE)
                y = rng.uniform(*CUBE_Y_RANGE)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - p) >= MIN_CUBE_SEP for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array([-0.15 + 0.10 * len(positions), 0.60]))

        for name, (x, y) in zip(["cube_A", "cube_B", "cube_C", "cube_D"], positions):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr : qpos_addr + 3] = [x, y, CUBE_HALF_SIZE + 0.001]
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.mocap_pos[0]  = MOCAP_INIT_POS.copy()
        self.data.mocap_quat[0] = [1.0, 0.0, 1.0, 0.0]
        self.data.ctrl[:] = [-1.0, 1.0]

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._success_steps = 0
        self.expert_policy.reset(self)
        return self._get_obs(), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got shape {action.shape}")

        self._apply_action(action)
        b_on_a, c_on_b, d_on_c = self._is_stacked()
        is_success = bool(b_on_a and c_on_b and d_on_c)

        if is_success:
            self._success_steps += 1
        else:
            self._success_steps = 0

        stable_success = self._success_steps >= SUCCESS_STEPS
        terminated = stable_success
        obs = self._get_obs()

        info = {
            "is_success":    stable_success,
            "b_on_a":        bool(b_on_a),
            "c_on_b":        bool(c_on_b),
            "d_on_c":        bool(d_on_c),
            "partial_stack": int(b_on_a) + int(c_on_b) + int(d_on_c),
        }
        if terminated:
            info["final_info"] = {"is_success": True}
            self.reset()

        return obs, self._compute_reward(), terminated, False, info

    def render(self) -> np.ndarray:
        return self._render_visualization()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def create_four_cube_stacking_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls=None,
) -> dict[str, dict[int, Any]]:
    """Create vectorised 4-cube stacking environments."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable.")
    gym_kwargs = dict(gym_kwargs or {})
    fns = [lambda: FourCubeStackingEnv(**gym_kwargs) for _ in range(n_envs)]
    return {"four_cube_stacking": {0: env_cls(fns)}}
