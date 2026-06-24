from __future__ import annotations

from lerobot.scripts import lerobot_eval


class DummyVecEnv:
    num_envs = 1

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _patch_run_one(monkeypatch):
    def fake_run_one(task_group, task_id, env, **kwargs):
        return task_group, task_id, {
            "sum_rewards": [1.0],
            "max_rewards": [1.0],
            "successes": [True],
            "video_paths": [],
        }

    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)


def test_eval_policy_all_keeps_env_open_by_default(monkeypatch):
    _patch_run_one(monkeypatch)
    env = DummyVecEnv()

    lerobot_eval.eval_policy_all(
        envs={"dummy": {0: env}},
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
    )

    assert env.close_calls == 0


def test_eval_policy_all_can_close_envs_after_eval(monkeypatch):
    _patch_run_one(monkeypatch)
    env = DummyVecEnv()

    lerobot_eval.eval_policy_all(
        envs={"dummy": {0: env}},
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        close_envs_after_eval=True,
    )

    assert env.close_calls == 1
