"""SB3 VecEnv on mjmlx's batched physics, replicating Gymnasium HalfCheetah-v4's exact reward/obs/reset-noise/episode-length defaults for comparability with published baselines (e.g. SB3 zoo PPO)."""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

FRAME_SKIP = 5
DT = 0.05  # frame_skip * option timestep (0.01)
MAX_EPISODE_STEPS = 1000
FORWARD_REWARD_WEIGHT = 1.0
CTRL_COST_WEIGHT = 0.1
RESET_NOISE_SCALE = 0.1


def _half_cheetah_xml_path() -> str:
    import gymnasium.envs.mujoco as gm

    return os.path.join(os.path.dirname(gm.__file__), "assets", "half_cheetah.xml")


class MjmlxHalfCheetahVecEnv(VecEnv):
    def __init__(self, native_module, num_envs: int, use_gpu: bool = True, seed: int | None = None):
        self._native = native_module
        self._model = native_module.load_model(_half_cheetah_xml_path())
        self._sim = native_module.create_batched(self._model, num_envs, frame_skip=1, use_gpu=use_gpu, solver_iterations=0)
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self._model.nu

        obs_dim = (self.nq - 1) + self.nv  # exclude root x per HalfCheetah-v4 default
        observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (self.nu,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self._rng = np.random.default_rng(seed)
        self._elapsed = np.zeros(num_envs, dtype=np.int64)
        self._actions: np.ndarray | None = None
        self._prev_x = np.zeros(num_envs, dtype=np.float32)

        all_mask = np.ones(num_envs, dtype=np.int32)
        self._reset_envs(all_mask)

    def _reset_envs(self, mask: np.ndarray) -> None:
        self._sim.reset(mask)
        base_qpos, base_qvel = self._sim.state()
        idx = np.nonzero(mask)[0]
        for i in idx:
            qpos = base_qpos[i] + self._rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE, self.nq).astype(np.float32)
            qvel = base_qvel[i] + (self._rng.standard_normal(self.nv).astype(np.float32) * RESET_NOISE_SCALE)
            self._sim.set_env_qpos(int(i), qpos)
            self._sim.set_env_qvel(int(i), qvel)
        self._elapsed[idx] = 0
        qpos_after, _ = self._sim.state()
        self._prev_x[idx] = qpos_after[idx, 0]

    def _obs(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        return np.concatenate([qpos[:, 1:], qvel], axis=1).astype(np.float32)

    # ── VecEnv API ────────────────────────────────────────────────

    def reset(self):
        mask = np.ones(self.num_envs, dtype=np.int32)
        self._reset_envs(mask)
        qpos, qvel = self._sim.state()
        return self._obs(qpos, qvel)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

    def step_wait(self):
        assert self._actions is not None
        x_before = self._prev_x.copy()
        for _ in range(FRAME_SKIP):
            self._sim.step(self._actions)
        qpos, qvel = self._sim.state()

        x_after = qpos[:, 0]
        forward_vel = (x_after - x_before) / DT
        ctrl_cost = CTRL_COST_WEIGHT * np.sum(np.square(self._actions), axis=1)
        rewards = (FORWARD_REWARD_WEIGHT * forward_vel - ctrl_cost).astype(np.float32)

        self._elapsed += 1
        truncated = self._elapsed >= MAX_EPISODE_STEPS
        not_finite = ~np.isfinite(qpos).all(axis=1) | ~np.isfinite(qvel).all(axis=1)
        dones = truncated | not_finite

        obs = self._obs(qpos, qvel)
        infos = [{} for _ in range(self.num_envs)]
        done_idx = np.nonzero(dones)[0]
        for i in done_idx:
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not not_finite[i])
            infos[i]["terminal_observation"] = obs[i].copy()

        self._prev_x = x_after.copy()
        if len(done_idx) > 0:
            mask = dones.astype(np.int32)
            self._reset_envs(mask)
            reset_qpos, reset_qvel = self._sim.state()
            reset_obs = self._obs(reset_qpos, reset_qvel)
            obs = obs.copy()
            obs[done_idx] = reset_obs[done_idx]

        self._actions = None
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
