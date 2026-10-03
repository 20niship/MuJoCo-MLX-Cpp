"""Go2の階段地形cmd_vel追従タスク。観測/報酬/ドメインランダマイズはIsaacLabのUnitreeGo2RoughEnvCfgを踏襲(height_scanと per-env物理パラメータDRは本エンジンの制約でスキップ、詳細はtrain_go2_ppo.py末尾コメント参照)。"""

from __future__ import annotations

import os
import re

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from go2_terrain import StairsTerrain

PHYSICS_DT = 0.02  # go2_mjx.xmlの<option timestep>。IsaacLab/MJX流に物理step=制御stepの50Hzへ
DECIMATION = 1
CONTROL_DT = PHYSICS_DT * DECIMATION
EPISODE_LENGTH_S = 20.0
MAX_EPISODE_STEPS = int(EPISODE_LENGTH_S / CONTROL_DT)

ACTION_SCALE = 0.25  # rough_env_cfgでのgo2上書き値(base既定は0.5)
# IsaacLabのUNITREE_GO2_CFG.init_state.joint_pos(FL,FR,RL,RRの順): hip=左0.1/右-0.1、thigh=前0.8/後1.0、calf=-1.5
DEFAULT_JOINT_POS = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32
)
CTRL_LOW = np.array([-0.9472, -1.4, -2.6227] * 4, dtype=np.float32)
CTRL_HIGH = np.array([0.9472, 2.5, -0.84776] * 4, dtype=np.float32)
FOOT_BODY_IDX = np.array([4, 7, 10, 13])  # FL,FR,RL,RR の calf(=foot)ボディ
BASE_BODY_IDX = 1

# reward weights (velocity_env_cfg.RewardsCfg + go2 rough_env_cfg の上書き)
W_TRACK_LIN = 1.5
W_TRACK_ANG = 0.75
W_LIN_VEL_Z = -2.0
W_ANG_VEL_XY = -0.05
W_TORQUE = -0.0002
W_DOF_ACC = -2.5e-7
W_ACTION_RATE = -0.01
W_FEET_AIR_TIME = 0.01
FEET_AIR_TIME_TARGET = 0.5
TRACK_STD_SQ = 0.25

BASE_CONTACT_FORCE_THRESHOLD = 1.0
PUSH_INTERVAL_STEPS = (int(10.0 / CONTROL_DT), int(15.0 / CONTROL_DT))


def _quat_rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ワールド座標系ベクトルvをbody座標系へ回転(qの逆回転)。quat_wxyz: (N,4), v: (N,3) or (3,)。"""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    v = np.broadcast_to(v, (quat_wxyz.shape[0], 3))
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # 逆回転 = 共役クォータニオン(-x,-y,-z)での回転
    qw, qx, qy, qz = w, -x, -y, -z
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return np.stack([rx, ry, rz], axis=1).astype(np.float32)


def _yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    half = yaw / 2.0
    return np.stack([np.cos(half), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(half)], axis=1).astype(np.float32)


def _go2_xml_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "benchmarks",
        "models",
        "external",
        "mujoco_menagerie",
        "unitree_go2",
        "go2_mjx.xml",
    )


def build_terrain_model_xml(terrain: StairsTerrain, out_path: str) -> None:
    src_path = os.path.abspath(_go2_xml_path())
    assets_dir = os.path.join(os.path.dirname(src_path), "assets")
    with open(src_path) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets"', f'meshdir="{assets_dir}"')
    # go2_mjx.xmlはgitignore対象の外部アセット(再fetchでtimestep指定が消える)なのでここで必ず注入する
    if 'timestep="' in xml:
        xml = re.sub(r'timestep="[^"]*"', f'timestep="{PHYSICS_DT}"', xml, count=1)
    else:
        xml = xml.replace('<option ', f'<option timestep="{PHYSICS_DT}" ', 1)
    terrain_geoms = f"<body name=\"terrain\">\n    {terrain.to_mjcf_bodies()}\n  </body>\n</worldbody>"
    xml = xml.replace("</worldbody>", terrain_geoms)
    with open(out_path, "w") as f:
        f.write(xml)


class MjmlxGo2Env(VecEnv):
    def __init__(
        self,
        native_module,
        num_envs: int,
        cmd_vel_min: float,
        cmd_vel_max: float,
        use_gpu: bool = True,
        seed: int | None = None,
        terrain_seed: int | None = None,
        generated_xml_path: str = "/tmp/go2_stairs_generated.xml",
    ):
        self._native = native_module
        self.terrain = StairsTerrain(seed=terrain_seed)
        build_terrain_model_xml(self.terrain, generated_xml_path)

        self._model = native_module.load_model(generated_xml_path)
        self._sim = native_module.create_batched(self._model, num_envs, frame_skip=1, use_gpu=use_gpu, solver_iterations=0)
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self._model.nu
        assert self.nu == 12

        self.num_envs = num_envs
        self.num_actions = self.nu
        self.max_episode_length = MAX_EPISODE_STEPS
        self.device = "cpu"  # 方策ネットワークはCPU、物理はmjmlx側で別途GPU実行
        self.cfg = {
            "task": "go2_stairs_cmd_vel",
            "cmd_vel_min": cmd_vel_min,
            "cmd_vel_max": cmd_vel_max,
            "control_dt": CONTROL_DT,
            "max_episode_steps": MAX_EPISODE_STEPS,
        }

        self.cmd_vel_min = cmd_vel_min
        self.cmd_vel_max = cmd_vel_max
        self._rng = np.random.default_rng(seed)
        self._elapsed = np.zeros(num_envs, dtype=np.int64)
        self._commands = np.zeros((num_envs, 3), dtype=np.float32)
        self._last_action = np.zeros((num_envs, self.nu), dtype=np.float32)
        self._prev_joint_vel = np.zeros((num_envs, self.nu), dtype=np.float32)
        self._feet_air_time = np.zeros((num_envs, 4), dtype=np.float32)
        self._last_foot_contact = np.zeros((num_envs, 4), dtype=bool)
        self._next_push_step = self._rng.integers(*PUSH_INTERVAL_STEPS, size=num_envs)

        all_mask = np.ones(num_envs, dtype=np.int32)
        self._reset_envs(all_mask)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.from_numpy(self._elapsed)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._elapsed = value.detach().cpu().numpy().astype(np.int64)

    def _sample_commands(self, n: int) -> np.ndarray:
        cmd = np.zeros((n, 3), dtype=np.float32)
        for _ in range(8):  # 棄却サンプリング。数回で大半収束、残りはクリップ
            need = np.linalg.norm(cmd, axis=1) == 0
            if not need.any():
                break
            cand = self._rng.uniform(-1.0, 1.0, size=(need.sum(), 3)).astype(np.float32)
            norm = np.linalg.norm(cand, axis=1)
            ok = (norm >= self.cmd_vel_min) & (norm <= self.cmd_vel_max)
            idx = np.nonzero(need)[0][ok]
            cmd[idx] = cand[ok]
        norm = np.linalg.norm(cmd, axis=1)
        zero = norm == 0
        if zero.any():
            cand = self._rng.uniform(-1.0, 1.0, size=(zero.sum(), 3)).astype(np.float32)
            n2 = np.linalg.norm(cand, axis=1, keepdims=True) + 1e-6
            target = self._rng.uniform(self.cmd_vel_min, self.cmd_vel_max, size=(zero.sum(), 1))
            cmd[zero] = cand / n2 * target
        return cmd

    def _reset_envs(self, mask: np.ndarray) -> None:
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return
        self._sim.reset(mask)
        qpos, qvel = self._sim.state()
        qpos = qpos.copy()
        qvel = qvel.copy()

        x, y = self.terrain.sample_spawn_xy(self._rng, n)
        z = self.terrain.height_at(x) + 0.40  # IsaacLabのUNITREE_GO2_CFG.init_state.pos[2]
        yaw = self._rng.uniform(-np.pi, np.pi, size=n)
        qpos[idx, 0] = x
        qpos[idx, 1] = y
        qpos[idx, 2] = z
        qpos[idx, 3:7] = _yaw_to_quat(yaw)

        joint_scale = self._rng.uniform(0.5, 1.5, size=(n, 12)).astype(np.float32)
        qpos[idx, 7:19] = DEFAULT_JOINT_POS * joint_scale

        qvel[idx, 0:2] = self._rng.uniform(-0.5, 0.5, size=(n, 2))
        qvel[idx, 2] = self._rng.uniform(-0.5, 0.5, size=n)
        qvel[idx, 3:6] = self._rng.uniform(-0.5, 0.5, size=(n, 3))
        qvel[idx, 6:18] = 0.0

        self._sim.set_state(qpos, qvel)
        self._elapsed[idx] = 0
        self._commands[idx] = self._sample_commands(n)
        self._last_action[idx] = 0.0
        self._prev_joint_vel[idx] = 0.0
        self._feet_air_time[idx] = 0.0
        self._last_foot_contact[idx] = False
        self._next_push_step[idx] = self._elapsed[idx] + self._rng.integers(*PUSH_INTERVAL_STEPS, size=n)

    def _obs(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        quat = qpos[:, 3:7]
        base_lin_vel = _quat_rotate_inverse(quat, qvel[:, 0:3])
        base_ang_vel = qvel[:, 3:6]
        gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (qpos.shape[0], 1))
        projected_gravity = _quat_rotate_inverse(quat, gravity)
        joint_pos_rel = qpos[:, 7:19] - DEFAULT_JOINT_POS
        joint_vel = qvel[:, 6:18]
        return np.concatenate(
            [base_lin_vel, base_ang_vel, projected_gravity, self._commands, joint_pos_rel, joint_vel, self._last_action],
            axis=1,
        ).astype(np.float32)

    def _obs_tensordict(self, obs: np.ndarray) -> TensorDict:
        return TensorDict({"policy": torch.from_numpy(obs)}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        qpos, qvel = self._sim.state()
        return self._obs_tensordict(self._obs(qpos, qvel))

    def reset(self) -> tuple[TensorDict, dict]:
        mask = np.ones(self.num_envs, dtype=np.int32)
        self._reset_envs(mask)
        qpos, qvel = self._sim.state()
        return self._obs_tensordict(self._obs(qpos, qvel)), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        action = np.clip(actions.detach().cpu().numpy(), -1.0, 1.0).astype(np.float32)
        ctrl_target = DEFAULT_JOINT_POS + ACTION_SCALE * action
        ctrl = np.clip(ctrl_target, CTRL_LOW, CTRL_HIGH)

        for _ in range(DECIMATION):
            self._sim.step(ctrl)
        qpos, qvel = self._sim.state()
        cfrc_ext = self._sim.cfrc_ext()  # (N, nbody*6): [torque(3),force(3)]/body

        not_finite = ~np.isfinite(qpos).all(axis=1) | ~np.isfinite(qvel).all(axis=1)
        finite_qpos = np.where(np.isfinite(qpos), qpos, 0.0)
        finite_qvel = np.where(np.isfinite(qvel), qvel, 0.0)
        blew_up = not_finite | (np.abs(finite_qpos).max(axis=1) > 1000.0) | (np.abs(finite_qvel).max(axis=1) > 1000.0)

        quat = qpos[:, 3:7]
        base_lin_vel = _quat_rotate_inverse(quat, qvel[:, 0:3])
        base_ang_vel = qvel[:, 3:6]
        joint_vel = qvel[:, 6:18]

        cfrc = cfrc_ext.reshape(self.num_envs, -1, 6)
        base_force_norm = np.linalg.norm(cfrc[:, BASE_BODY_IDX, 3:6], axis=1)
        base_contact = base_force_norm > BASE_CONTACT_FORCE_THRESHOLD
        foot_force_norm = np.linalg.norm(cfrc[:, FOOT_BODY_IDX, 3:6], axis=2)
        foot_contact = foot_force_norm > BASE_CONTACT_FORCE_THRESHOLD

        cmd = self._commands
        lin_err = np.sum((cmd[:, 0:2] - base_lin_vel[:, 0:2]) ** 2, axis=1)
        ang_err = (cmd[:, 2] - base_ang_vel[:, 2]) ** 2
        r_track_lin = W_TRACK_LIN * np.exp(-lin_err / TRACK_STD_SQ)
        r_track_ang = W_TRACK_ANG * np.exp(-ang_err / TRACK_STD_SQ)
        # 接触衝撃の瞬間は速度が物理的に大きく跳ねうるので、二乗前にクリップしてreward発散を防ぐ
        r_lin_vel_z = W_LIN_VEL_Z * np.clip(base_lin_vel[:, 2], -20.0, 20.0) ** 2
        r_ang_vel_xy = W_ANG_VEL_XY * np.sum(np.clip(base_ang_vel[:, 0:2], -50.0, 50.0) ** 2, axis=1)
        joint_acc = np.clip((joint_vel - self._prev_joint_vel) / CONTROL_DT, -500.0, 500.0)  # 接触衝撃時の差分近似スパイクでreward/value発散するのを防ぐ
        r_dof_acc = W_DOF_ACC * np.sum(joint_acc**2, axis=1)
        r_action_rate = W_ACTION_RATE * np.sum((action - self._last_action) ** 2, axis=1)

        contact_filt = foot_contact | self._last_foot_contact
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += CONTROL_DT
        r_air_time = W_FEET_AIR_TIME * np.sum((self._feet_air_time - FEET_AIR_TIME_TARGET) * first_contact, axis=1)
        r_air_time *= np.linalg.norm(cmd[:, 0:2], axis=1) > 0.1
        self._feet_air_time *= ~contact_filt
        self._last_foot_contact = foot_contact

        try:
            qfrc_actuator = self._sim.qfrc_actuator()
            joint_torque = qfrc_actuator[:, 6:18]
            r_torque = W_TORQUE * np.sum(joint_torque**2, axis=1)
        except AttributeError:
            r_torque = np.zeros(self.num_envs, dtype=np.float32)  # ponytail: qfrc_actuatorバインディング未ビルド時のフォールバック

        rewards = (
            r_track_lin + r_track_ang + r_lin_vel_z + r_ang_vel_xy + r_dof_acc + r_action_rate + r_air_time + r_torque
        )
        # 未知の接触イベント等でどれかの項が想定外に暴れてもvalue関数の発散を防ぐ最終防衛ライン
        rewards = np.where(blew_up, 0.0, np.clip(rewards, -50.0, 50.0)).astype(np.float32)

        self._elapsed += 1
        truncated = self._elapsed >= MAX_EPISODE_STEPS
        terminated = base_contact | blew_up
        dones = truncated | terminated

        obs = self._obs(qpos, qvel)
        self._last_action = action.copy()
        self._prev_joint_vel = joint_vel.copy()

        push_due = self._elapsed >= self._next_push_step
        if push_due.any():
            qpos2, qvel2 = qpos.copy(), qvel.copy()
            pidx = np.nonzero(push_due)[0]
            qvel2[pidx, 0:2] += self._rng.uniform(-0.5, 0.5, size=(len(pidx), 2))
            self._sim.set_state(qpos2, qvel2)
            self._next_push_step[pidx] = self._elapsed[pidx] + self._rng.integers(*PUSH_INTERVAL_STEPS, size=len(pidx))

        done_idx = np.nonzero(dones)[0]
        if len(done_idx) > 0:
            mask = dones.astype(np.int32)
            self._reset_envs(mask)
            reset_qpos, reset_qvel = self._sim.state()
            reset_obs = self._obs(reset_qpos, reset_qvel)
            obs = obs.copy()
            obs[done_idx] = reset_obs[done_idx]

        extras = {"time_outs": torch.from_numpy(truncated)}
        return (
            self._obs_tensordict(obs),
            torch.from_numpy(rewards),
            torch.from_numpy(dones).long(),
            extras,
        )

    def close(self) -> None:
        pass
