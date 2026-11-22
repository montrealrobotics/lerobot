#!/usr/bin/env python

import gymnasium as gym
import numpy as np
import torch

from lerobot.rl.dsrl.dsrl_env_wrapper import (
    BASE_ACTION_KEY,
    DSRL_ACTION_CHUNK_INTERRUPTED,
    DSRL_ACTION_CHUNK_RAW_REWARD,
    DSRL_ACTION_CHUNK_SHAPED_REWARD,
    DSRL_ACTION_CHUNK_STEPS,
    DSRLEnvWrapper,
)


class _ChunkTestEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)}
        )
        self.actions = []
        self.count = 0

    def reset(self, **kwargs):
        self.actions.clear()
        self.count = 0
        return {"state": np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        self.actions.append(float(action[0]))
        self.count += 1
        obs = {"state": np.array([float(self.count)], dtype=np.float32)}
        reward = float(action[0])
        terminated = self.count >= 2
        return obs, reward, terminated, False, {"inner_count": self.count}


def _prepare_obs(obs):
    # SyncVectorEnv batches the dict obs along axis 0 → (num_envs, 1).
    return {"observation.state": torch.from_numpy(np.asarray(obs["state"], dtype=np.float32))}


def _make_wrapper(generate_action_chunk_fn, *, n_obs_steps=2, reward_fn=None):
    env = gym.vector.SyncVectorEnv([_ChunkTestEnv])
    return env, DSRLEnvWrapper(
        env=env,
        noise_dim=3,
        prepare_obs_fn=_prepare_obs,
        generate_action_chunk_fn=generate_action_chunk_fn,
        device="cpu",
        n_obs_steps=n_obs_steps,
        reward_fn=reward_fn,
    )


def test_wrapper_executes_action_chunk_as_one_macro_step():
    generate_calls = []

    def generate_action_chunk_fn(stacked_obs, noise):
        generate_calls.append((stacked_obs, noise))
        return torch.tensor([[[1.0], [2.0], [3.0]]])  # (num_envs=1, T=3, action_dim=1)

    env, wrapped = _make_wrapper(generate_action_chunk_fn)

    policy_obs, _ = wrapped.reset()
    assert policy_obs["observation.state"].shape == (1, 1)
    assert policy_obs["observation.state"].item() == 0.0

    next_obs, reward, terminated, truncated, info = wrapped.step(np.zeros((1, 3), dtype=np.float32))

    # The env terminates after 2 inner steps, so the 3-action chunk is interrupted.
    assert next_obs["observation.state"].item() == 2.0
    assert reward[0] == 3.0
    assert bool(terminated[0]) is True
    assert bool(truncated[0]) is False
    # SyncVectorEnv autoresets after termination, so the inner env's action log is cleared.
    assert info[DSRL_ACTION_CHUNK_STEPS][0] == 2
    assert bool(info[DSRL_ACTION_CHUNK_INTERRUPTED][0]) is True
    assert info[DSRL_ACTION_CHUNK_RAW_REWARD][0] == 3.0
    assert info[DSRL_ACTION_CHUNK_SHAPED_REWARD][0] == 3.0
    # Frozen policy conditions on the stacked n_obs_steps history.
    assert len(generate_calls) == 1
    assert generate_calls[0][0]["observation.state"].shape == (1, 2, 1)


def test_wrapper_shapes_terminal_reward():
    def generate_action_chunk_fn(stacked_obs, noise):
        return torch.tensor([[[1.0], [2.0], [3.0]]])

    def reward_fn(reward, terminated, truncated, info):
        del truncated, info
        return reward + (10.0 if terminated else 0.0)

    _env, wrapped = _make_wrapper(generate_action_chunk_fn, reward_fn=reward_fn)
    wrapped.reset()

    _obs, reward, _terminated, _truncated, info = wrapped.step(np.zeros((1, 3), dtype=np.float32))

    assert reward[0] == 13.0
    assert info[DSRL_ACTION_CHUNK_RAW_REWARD][0] == 3.0
    assert info[DSRL_ACTION_CHUNK_SHAPED_REWARD][0] == 13.0


def test_wrapper_requires_vector_env():
    import pytest

    with pytest.raises(TypeError):
        DSRLEnvWrapper(
            env=_ChunkTestEnv(),
            noise_dim=3,
            prepare_obs_fn=_prepare_obs,
            generate_action_chunk_fn=lambda obs, noise: torch.zeros((1, 3, 1)),
            device="cpu",
        )


# ── Residual mode ─────────────────────────────────────────────────────────────────────


def _make_residual_wrapper(base_action_fn, residual_fn, *, num_envs=1):
    env = gym.vector.SyncVectorEnv([_ChunkTestEnv for _ in range(num_envs)])
    return env, DSRLEnvWrapper(
        env=env,
        noise_dim=2,
        prepare_obs_fn=_prepare_obs,
        generate_action_chunk_fn=lambda *_: (_ for _ in ()).throw(AssertionError("unused in residual mode")),
        device="cpu",
        n_obs_steps=1,
        base_action_fn=base_action_fn,
        residual_fn=residual_fn,
    )


def test_residual_mode_exposes_base_chunk_and_executes_base_plus_residual():
    seen_states = []

    def base_action_fn(stacked_obs):
        state = stacked_obs["observation.state"][:, 0]  # (B, 1)
        seen_states.append(state.clone())
        # Base chunk depends on the observation it was sampled at: 10*state + [1, 2].
        return (10 * state + torch.tensor([1.0, 2.0])).unsqueeze(-1)  # (B, T=2, A=1)

    def residual_fn(stacked_obs, base, residual):
        return base + torch.from_numpy(residual).view(base.shape)

    _, wrapped = _make_residual_wrapper(base_action_fn, residual_fn)
    obs, _ = wrapped.reset()
    assert torch.equal(obs[BASE_ACTION_KEY], torch.tensor([[1.0, 2.0]]))

    _, reward, terminated, _, info = wrapped.step(np.array([[0.5, -0.5]], dtype=np.float32))
    # Executed actions are base + residual = [1.5, 1.5]; the env sums them as its reward.
    assert reward[0] == 3.0
    assert info[DSRL_ACTION_CHUNK_STEPS][0] == 2
    # A fresh base chunk was sampled at the post-chunk observation (one call per macro step).
    assert len(seen_states) == 2


def test_residual_partial_reset_keeps_in_flight_base_chunks():
    calls = {"n": 0}

    def base_action_fn(stacked_obs):
        calls["n"] += 1
        batch = stacked_obs["observation.state"].shape[0]
        return torch.full((batch, 2, 1), float(calls["n"]))

    def residual_fn(stacked_obs, base, residual):
        return base

    _, wrapped = _make_residual_wrapper(base_action_fn, residual_fn, num_envs=2)
    obs, _ = wrapped.reset()
    assert torch.equal(obs[BASE_ACTION_KEY], torch.full((2, 2), 1.0))
    obs, *_ = wrapped.step(np.zeros((2, 2), dtype=np.float32))
    assert torch.equal(obs[BASE_ACTION_KEY], torch.full((2, 2), 2.0))

    obs, _ = wrapped.reset(env_mask=np.array([True, False]))
    # Env 0 was reset and gets the new base chunk; env 1 keeps the one its observation carried.
    assert torch.equal(obs[BASE_ACTION_KEY][0], torch.full((2,), 3.0))
    assert torch.equal(obs[BASE_ACTION_KEY][1], torch.full((2,), 2.0))


def test_residual_mode_requires_both_functions():
    import pytest

    env = gym.vector.SyncVectorEnv([_ChunkTestEnv])
    with pytest.raises(ValueError, match="both"):
        DSRLEnvWrapper(
            env=env,
            noise_dim=2,
            prepare_obs_fn=_prepare_obs,
            generate_action_chunk_fn=None,
            device="cpu",
            base_action_fn=lambda obs: None,
        )
