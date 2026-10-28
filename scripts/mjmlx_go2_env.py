"""Go2の階段地形cmd_vel追従タスク。観測/報酬/ドメインランダマイズはIsaacLabのUnitreeGo2RoughEnvCfgを踏襲(height_scanと per-env物理パラメータDRは本エンジンの制約でスキップ、詳細はtrain_go2_ppo.py末尾コメント参照)。"""

from __future__ import annotations

import os
import re

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from go2_terrain import CORRIDOR_WIDTH, StairsTerrain
from mjmlx_sensors import ContactSensor, Imu, SensorHub

PHYSICS_DT = 0.005  # Isaac LabのGo2と同じdt(0.005s x decimation 4)。0.01以上ではPD(kp=50,armature=0.01)が陽的積分で不安定
DECIMATION = 4  # 4*0.005 = 0.02s -> 50Hz制御
CONTROL_DT = PHYSICS_DT * DECIMATION
INTEGRATOR = os.environ.get("MJMLX_GO2_INTEGRATOR", "")  # 空ならXMLのまま(euler、eulerdamp無効=PDは陽的)。"implicitfast"でPDを陰的に扱う(Isaac LabのPDアクチュエータ相当、CPU/GPU差は同程度)
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
# 報酬重みとreset乱数のプリセット。relaxed=罰則を弱めた従来設定、isaac_flat=IsaacLabのUnitreeGo2FlatEnvCfg準拠
PRESETS = {
    "relaxed": dict(lin_vel_z=-0.4, ang_vel_xy=-0.02, torque=-5e-6, dof_acc=-1e-7, action_rate=-0.003, feet_air_time=0.01,
                    flat_orientation=0.0, joint_reset_scale=(0.5, 1.5), reset_vel=0.5),
    "isaac_flat": dict(lin_vel_z=-2.0, ang_vel_xy=-0.05, torque=-2e-4, dof_acc=-2.5e-7, action_rate=-0.01, feet_air_time=0.25,
                       flat_orientation=-2.5, joint_reset_scale=(1.0, 1.0), reset_vel=0.0),
}
FEET_AIR_TIME_TARGET = 0.5
TRACK_STD_SQ = 0.25

# go2_mjx.xmlのiterations=1はCPU(Newton)向け。GPUのソルバは接触が重なると収束が遅く、1〜3回だと一部のenvで速度が発散するため8にする(CPUでも同値を使い挙動を揃える)
SOLVER_ITERATIONS = 16  # CPU(MuJoCo C)との差を小さくするため十分に反復する(収束した環は早期終了)
BASE_CONTACT_FORCE_THRESHOLD = 1.0
SPAWN_FOOTPRINT_RADIUS = 0.5  # 胴体中心から足先までの最大水平距離(関節角ランダム化で脚が開く分を含む)
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


def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """重力(0,0,-1)をbody座標へ。_quat_rotate_inverse(quat, [0,0,-1])の閉形式(行列の3行目)で、配列確保が少ない。"""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return np.stack([2.0 * (w * y - x * z), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0], axis=1).astype(np.float32)


def _rowdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _quat_rotate_inverse_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ワールド座標系のvをbody座標系へ(qの逆回転)。q: (N,4)wxyz, v: (N,3)。"""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(-xyz, v, dim=1)
    return v + w * t + torch.cross(-xyz, t, dim=1)


def _projected_gravity_t(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([2.0 * (w * y - x * z), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0], dim=1)


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


def build_terrain_model_xml(terrain: StairsTerrain, out_path: str, self_collision: bool = True, foot_condim: int = 6) -> None:
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
    if not self_collision:  # 脚・胴体同士の衝突を切って地形との接触だけにする(Isaac LabのUNITREE_GO2_CFGはenabled_self_collisions=False)。衝突ペアが約7分の1になる
        xml = xml.replace('<default class="collision">\n        <geom group="3"/>', '<default class="collision">\n        <geom group="3" conaffinity="0"/>', 1)
    if foot_condim != 6:  # 足の摩擦円錐の次元(6=並進+ねじり+転がり、3=並進のみ)。3だとピラミッドの拘束行が足1接触あたり10行->4行になりsolverが約1.2倍速い(Isaac Lab/Genesisは並進摩擦のみ)
        xml = xml.replace('condim="6"', f'condim="{foot_condim}"')
    for kv in filter(None, os.environ.get("MJMLX_GO2_XML_SUBST", "").split(";")):  # 切り分け用: 'old->new;old2->new2' をXMLに文字列置換する
        old, new = kv.split("->", 1)
        xml = xml.replace(old, new)
    if INTEGRATOR:
        if 'integrator="' in xml:
            xml = re.sub(r'integrator="[^"]*"', f'integrator="{INTEGRATOR}"', xml, count=1)
        else:
            xml = xml.replace('<option ', f'<option integrator="{INTEGRATOR}" ', 1)
    terrain_geoms = f"<body name=\"terrain\">\n    {terrain.to_mjcf_bodies()}\n  </body>\n</worldbody>"
    xml = xml.replace("</worldbody>", terrain_geoms)
    with open(out_path, "w") as f:
        f.write(xml)


TERM_NAMES = ["track_lin", "track_ang", "lin_vel_z", "ang_vel_xy", "dof_acc", "action_rate", "feet_air_time", "torque", "flat_orientation"]
METRIC_NAMES = [f"Reward/{k}" for k in TERM_NAMES] + [
    "Metrics/lin_vel_error", "Metrics/ang_vel_error", "Metrics/cmd_speed", "Metrics/base_height", "Metrics/action_abs", "Metrics/action_rate_abs",
    "Metrics/ctrl_saturated_frac", "Metrics/joint_vel_abs", "Metrics/joint_pos_dev", "Metrics/base_tilt", "Metrics/feet_air_time",
    "Metrics/joint_torque_abs", "Metrics/foot_contact_frac", "Termination/base_contact", "Termination/blew_up", "Termination/out_of_terrain", "Termination/time_out",
]


def _make_step_fns(w: dict, terrain_length: float, default_pos: torch.Tensor, ctrl_low: torch.Tensor, ctrl_high: torch.Tensor):
    """1 stepの観測・報酬・終了判定をまとめた純粋関数を作る。小さいtorch演算が数百個に分かれると起動オーバーヘッドが支配的なので、cudaではtorch.compileで融合する。"""

    def ctrl_fn(action):
        ctrl_target = default_pos + ACTION_SCALE * action
        ctrl = torch.minimum(torch.maximum(ctrl_target, ctrl_low), ctrl_high)
        return ctrl, (ctrl_target != ctrl).float().mean()

    def core(qpos, qvel, contact_force, torque_sq, torque_abs_mean, ctrl_sat, action, last_action, prev_joint_vel, feet_air, last_foot_contact, commands, truncated, ov_base, ov_foot, ov_lin, ov_ang, ov_grav):
        blew_up = ~(torch.cat([qpos, qvel], dim=1).abs().amax(dim=1) <= 1000.0)  # NaN/infはamaxがNaN/infで比較がFalseになる
        base_contact = (contact_force[:, 0] > BASE_CONTACT_FORCE_THRESHOLD) if ov_base is None else ov_base
        foot_contact = (contact_force[:, 1:] > BASE_CONTACT_FORCE_THRESHOLD) if ov_foot is None else ov_foot
        quat = qpos[:, 3:7]
        base_lin_vel = _quat_rotate_inverse_t(quat, qvel[:, 0:3]) if ov_lin is None else ov_lin
        base_ang_vel = qvel[:, 3:6] if ov_ang is None else ov_ang
        joint_vel = qvel[:, 6:18]
        projected_gravity = _projected_gravity_t(quat) if ov_grav is None else ov_grav
        cmd = commands
        d_lin = cmd[:, 0:2] - base_lin_vel[:, 0:2]
        lin_err = (d_lin * d_lin).sum(dim=1)
        ang_err = (cmd[:, 2] - base_ang_vel[:, 2]) ** 2
        r_track_lin = W_TRACK_LIN * torch.exp(-lin_err / TRACK_STD_SQ)
        r_track_ang = W_TRACK_ANG * torch.exp(-ang_err / TRACK_STD_SQ)
        # 接触衝撃の瞬間は速度が物理的に大きく跳ねうるので、二乗前にクリップしてreward発散を防ぐ
        r_lin_vel_z = w["lin_vel_z"] * base_lin_vel[:, 2].clamp(-20.0, 20.0) ** 2
        r_ang_vel_xy = w["ang_vel_xy"] * (base_ang_vel[:, 0:2].clamp(-50.0, 50.0) ** 2).sum(dim=1)
        joint_acc = ((joint_vel - prev_joint_vel) / CONTROL_DT).clamp(-500.0, 500.0)  # 接触衝撃時の差分近似スパイクでreward/value発散するのを防ぐ
        r_dof_acc = w["dof_acc"] * (joint_acc**2).sum(dim=1)
        d_act = action - last_action
        r_action_rate = w["action_rate"] * (d_act * d_act).sum(dim=1)
        contact_filt = foot_contact | last_foot_contact
        first_contact = (feet_air > 0.0) & contact_filt
        feet_air = feet_air + CONTROL_DT
        r_air_time = w["feet_air_time"] * ((feet_air - FEET_AIR_TIME_TARGET) * first_contact).sum(dim=1)
        r_air_time = r_air_time * (cmd[:, 0:2].norm(dim=1) > 0.1)
        feet_air_new = feet_air * ~contact_filt
        tilt2 = (projected_gravity[:, 0:2] ** 2).sum(dim=1)
        r_flat_orient = w["flat_orientation"] * tilt2
        r_torque = w["torque"] * torque_sq
        terms = torch.stack([r_track_lin, r_track_ang, r_lin_vel_z, r_ang_vel_xy, r_dof_acc, r_action_rate, r_air_time, r_torque, r_flat_orient])
        rewards = terms.sum(dim=0)
        # 未知の接触イベント等でどれかの項が想定外に暴れてもvalue関数の発散を防ぐ最終防衛ライン
        rewards = torch.where(blew_up, torch.zeros_like(rewards), rewards.clamp(-50.0, 50.0))
        # 通路の外へ出たら地形が無いので落下して学習に使えない: 終了にする
        out_of_terrain = (qpos[:, 1].abs() > CORRIDOR_WIDTH / 2) | (qpos[:, 0] < 0.0) | (qpos[:, 0] > terrain_length)
        dones = truncated | base_contact | blew_up | out_of_terrain
        obs = torch.cat([base_lin_vel, base_ang_vel, projected_gravity, cmd, qpos[:, 7:19] - default_pos, qvel[:, 6:18], last_action], dim=1)
        f = lambda x: x.float().mean()
        metrics = torch.stack(list(terms.mean(dim=1)) + [
            lin_err.sqrt().mean(), ang_err.sqrt().mean(), cmd.norm(dim=1).mean(), qpos[:, 2].mean(), action.abs().mean(), d_act.abs().mean(),
            ctrl_sat, joint_vel.abs().mean(), (qpos[:, 7:19] - default_pos).abs().mean(), tilt2.sqrt().mean(), feet_air.mean(),
            torch.as_tensor(torque_abs_mean, dtype=torch.float32, device=qpos.device), f(foot_contact), f(base_contact), f(blew_up), f(out_of_terrain), f(truncated),
        ])
        return obs, rewards, dones, terms, feet_air_new, foot_contact, metrics, joint_vel.clone(), action.clone()

    if os.environ.get("MJMLX_ENV_COMPILE", "1") != "0" and default_pos.is_cuda:
        try:
            return torch.compile(ctrl_fn), torch.compile(core)
        except Exception:  # noqa: BLE001  # ponytail: compile不可の環境ではeagerで動かす
            pass
    return ctrl_fn, core


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
        flat_terrain: bool = False,
        reward_preset: str = "relaxed",
        device: str = "cpu",
        generated_xml_path: str = "/tmp/go2_stairs_generated.xml",
        use_sensors: bool = False,
        self_collision: bool = True,
        foot_condim: int = 6,
    ):
        self._native = native_module
        self._w = PRESETS[reward_preset]
        self.terrain = StairsTerrain(seed=terrain_seed, flat=flat_terrain)
        build_terrain_model_xml(self.terrain, generated_xml_path, self_collision=self_collision, foot_condim=foot_condim)

        self._model = native_module.load_model(generated_xml_path)
        self._sim = native_module.create_batched(self._model, num_envs, frame_skip=1, use_gpu=use_gpu, solver_iterations=SOLVER_ITERATIONS)
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nu = self._model.nu
        assert self.nu == 12

        self.num_envs = num_envs
        # use_sensors: 観測のbase速度・重力をMJCFの<sensor>(gyro/orientation/velocimeter相当)、足/胴体の接触をContactSensorから取る(Isaac Lab形式)。センサ値は1物理step遅れ
        self._hub = SensorHub(self._sim, self._model) if use_sensors else None
        self._imu = Imu(self._hub) if use_sensors else None
        self._feet_contact_sensor = ContactSensor(self._sim, num_envs, FOOT_BODY_IDX, BASE_CONTACT_FORCE_THRESHOLD) if use_sensors else None
        self._base_contact_sensor = ContactSensor(self._sim, num_envs, [BASE_BODY_IDX], BASE_CONTACT_FORCE_THRESHOLD) if use_sensors else None
        # 毎step読む値(qpos/qvel/足と胴体の接触力ノルム/関節トルク)をGPUで1つに詰めて1回で転送する。個別に読み戻すと配列ごとに遅い。センサ経路やCPU実行では使わない
        self._pack = None
        if self._hub is None:
            try:
                self._pack = self._sim.set_pack_spec(np.array([BASE_BODY_IDX, *FOOT_BODY_IDX], dtype=np.int32), 6, 12)
            except RuntimeError:
                pass
        if self._pack is not None:
            self._pk_pin = torch.empty((num_envs, self._pack), dtype=torch.float32, pin_memory=torch.cuda.is_available())
            self._pk_np = self._pk_pin.numpy()
        self.num_actions = self.nu
        self.max_episode_length = MAX_EPISODE_STEPS
        self.device = device  # 方策ネットワーク/PPOバッファの置き場。物理はmjmlx側で別途GPU実行
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
        dev = self.device
        self._default_pos = torch.from_numpy(DEFAULT_JOINT_POS).to(dev)
        self._ctrl_low = torch.from_numpy(CTRL_LOW).to(dev)
        self._ctrl_high = torch.from_numpy(CTRL_HIGH).to(dev)
        self._commands = torch.zeros((num_envs, 3), device=dev)
        self._last_action = torch.zeros((num_envs, self.nu), device=dev)
        self._prev_joint_vel = torch.zeros((num_envs, self.nu), device=dev)
        self._feet_air_time = torch.zeros((num_envs, 4), device=dev)
        self._last_foot_contact = torch.zeros((num_envs, 4), dtype=torch.bool, device=dev)
        self._ctrl_fn, self._core = _make_step_fns(self._w, float(self.terrain.total_length), self._default_pos, self._ctrl_low, self._ctrl_high)
        self._next_push_step = self._rng.integers(*PUSH_INTERVAL_STEPS, size=num_envs)

        all_mask = np.ones(num_envs, dtype=np.int32)
        qpos, qvel = self._sim.state()
        qpos, qvel = qpos.copy(), qvel.copy()
        self._reset_envs(all_mask, qpos, qvel)
        self._sim.set_state(qpos, qvel)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.from_numpy(self._elapsed).to(self.device)

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

    def _reset_envs(self, mask: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """maskのenvの状態をqpos/qvel(host配列、書き換える)に初期化する。simへの反映は呼び出し側がset_stateでまとめて行う(読み戻し・書き込みを減らすため)。"""
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return
        if self._hub is not None:
            self._sim.reset(mask)  # センサ側の「リセット直後」フラグを立てる

        x, y = self.terrain.sample_spawn_xy(self._rng, n)
        z = self.terrain.max_height_within(x, SPAWN_FOOTPRINT_RADIUS) + 0.40  # 脚が張り出す範囲の最高地形の上に置く。 IsaacLabのUNITREE_GO2_CFG.init_state.pos[2]
        yaw = self._rng.uniform(-np.pi, np.pi, size=n)
        qpos[idx, 0] = x
        qpos[idx, 1] = y
        qpos[idx, 2] = z
        qpos[idx, 3:7] = _yaw_to_quat(yaw)

        joint_scale = self._rng.uniform(*self._w["joint_reset_scale"], size=(n, 12)).astype(np.float32)
        qpos[idx, 7:19] = DEFAULT_JOINT_POS * joint_scale

        rv = self._w["reset_vel"]
        qvel[idx, 0:3] = self._rng.uniform(-rv, rv, size=(n, 3))
        qvel[idx, 3:6] = self._rng.uniform(-rv, rv, size=(n, 3))
        qvel[idx, 6:18] = 0.0

        self._elapsed[idx] = 0
        idx_t = torch.from_numpy(idx).to(self.device)
        self._commands[idx_t] = torch.from_numpy(self._sample_commands(n)).to(self.device)
        self._last_action[idx_t] = 0.0
        self._prev_joint_vel[idx_t] = 0.0
        self._feet_air_time[idx_t] = 0.0
        self._last_foot_contact[idx_t] = False
        self._next_push_step[idx] = self._elapsed[idx] + self._rng.integers(*PUSH_INTERVAL_STEPS, size=n)

    def _obs(self, qpos: torch.Tensor, qvel: torch.Tensor, rows: torch.Tensor | None = None, from_sensors: bool = False) -> torch.Tensor:
        """観測(N,48)。rowsを渡すとそのenvだけ(qpos/qvelもその行分を渡す)。"""
        cmd = self._commands if rows is None else self._commands[rows]
        last_action = self._last_action if rows is None else self._last_action[rows]
        if from_sensors:
            base_lin_vel = self._imu.lin_vel_b.to(self.device)
            base_ang_vel = self._imu.ang_vel_b.to(self.device)
            projected_gravity = self._imu.projected_gravity_b.to(self.device)
        else:
            quat = qpos[:, 3:7]
            base_lin_vel = _quat_rotate_inverse_t(quat, qvel[:, 0:3])
            base_ang_vel = qvel[:, 3:6]
            projected_gravity = _projected_gravity_t(quat)
        return torch.cat([base_lin_vel, base_ang_vel, projected_gravity, cmd, qpos[:, 7:19] - self._default_pos, qvel[:, 6:18], last_action], dim=1)

    def _obs_tensordict(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def _to_dev(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device)

    def get_observations(self) -> TensorDict:
        qpos, qvel = self._sim.state()
        return self._obs_tensordict(self._obs(self._to_dev(qpos), self._to_dev(qvel)))

    def reset(self) -> tuple[TensorDict, dict]:
        mask = np.ones(self.num_envs, dtype=np.int32)
        qpos, qvel = self._sim.state()
        qpos, qvel = qpos.copy(), qvel.copy()
        self._reset_envs(mask, qpos, qvel)
        self._sim.set_state(qpos, qvel)
        return self._obs_tensordict(self._obs(self._to_dev(qpos), self._to_dev(qvel))), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # 観測・報酬・終了判定はself.device(cudaならGPU)上のtorchで計算する。ホストに残るのは物理への入出力と、リセット/押しの乱数処理だけ
        N, nq, nv = self.num_envs, self.nq, self.nv
        action = actions.detach().to(self.device, dtype=torch.float32)  # IsaacLab同様clipしない(関節目標のみ制御範囲へclip)
        ctrl, ctrl_sat = self._ctrl_fn(action)
        self._sim.step(np.ascontiguousarray(ctrl.cpu().numpy()), DECIMATION)
        ov_base = ov_foot = ov_lin = ov_ang = ov_grav = None
        if self._pack is not None:
            self._sim.read_pack_into(self._pk_np)  # ピン留めバッファへ直接読む
            pk = self._pk_np
            pk_t = self._pk_pin.to(self.device, non_blocking=True)
            qpos_h, qvel_h = pk[:, :nq], pk[:, nq:nq + nv]
            qpos, qvel = pk_t[:, :nq], pk_t[:, nq:nq + nv]
            contact_force = pk_t[:, nq + nv:nq + nv + 5]  # [base, FL, FR, RL, RR]の力ノルム
            torque_sq, torque_abs_mean = pk_t[:, nq + nv + 5], pk_t[:, nq + nv + 6].sum() / (N * 12)
        else:
            qpos_h, qvel_h = self._sim.state()
            qpos, qvel = self._to_dev(qpos_h), self._to_dev(qvel_h)
            cfrc = self._to_dev(self._sim.cfrc_ext()).reshape(N, -1, 6)
            contact_force = cfrc[:, [BASE_BODY_IDX, *FOOT_BODY_IDX], 3:6].norm(dim=2)
            try:
                joint_torque = self._to_dev(self._sim.qfrc_actuator())[:, 6:18]
            except AttributeError:
                joint_torque = torch.zeros((N, 12), device=self.device)  # ponytail: qfrc_actuatorバインディング未ビルド時のフォールバック
            torque_sq, torque_abs_mean = (joint_torque**2).sum(dim=1), joint_torque.abs().mean()
        if self._hub is not None:
            self._hub.read()
            self._feet_contact_sensor.update(CONTROL_DT)
            self._base_contact_sensor.update(CONTROL_DT)
            ov_base = self._base_contact_sensor.is_contact[:, 0].to(self.device)
            ov_foot = self._feet_contact_sensor.is_contact.to(self.device)
            ov_lin, ov_ang, ov_grav = (t.to(self.device) for t in (self._imu.lin_vel_b, self._imu.ang_vel_b, self._imu.projected_gravity_b))

        self._elapsed += 1
        truncated = self._to_dev(self._elapsed >= MAX_EPISODE_STEPS)
        obs, rewards, dones, terms, feet_air, foot_contact, metrics, joint_vel, last_action = self._core(
            qpos, qvel, contact_force, torque_sq, torque_abs_mean, ctrl_sat, action, self._last_action, self._prev_joint_vel, self._feet_air_time,
            self._last_foot_contact, self._commands, truncated, ov_base, ov_foot, ov_lin, ov_ang, ov_grav)
        self.last_reward_terms = dict(zip(TERM_NAMES, terms))
        self._feet_air_time, self._last_foot_contact = feet_air, foot_contact
        self._last_action, self._prev_joint_vel = last_action, joint_vel

        dones_h = dones.cpu().numpy()  # リセット対象の確定(ホストの乱数処理に必要)
        push_due = self._elapsed >= self._next_push_step
        done_idx = np.nonzero(dones_h)[0]
        if push_due.any() or len(done_idx) > 0:
            qpos2, qvel2 = qpos_h.copy(), qvel_h.copy()  # 以降の指標は押す前・リセット前の値を使うので別配列に書く
            if push_due.any():
                pidx = np.nonzero(push_due)[0]
                qvel2[pidx, 0:2] += self._rng.uniform(-0.5, 0.5, size=(len(pidx), 2))
                self._next_push_step[pidx] = self._elapsed[pidx] + self._rng.integers(*PUSH_INTERVAL_STEPS, size=len(pidx))
            if len(done_idx) > 0:
                self._reset_envs(dones_h.astype(np.int32), qpos2, qvel2)
                if self._hub is not None:
                    self._feet_contact_sensor.reset(torch.from_numpy(dones_h))
                    self._base_contact_sensor.reset(torch.from_numpy(dones_h))
                idx_t = torch.from_numpy(done_idx).to(self.device)
                obs[idx_t] = self._obs(self._to_dev(qpos2[done_idx]), self._to_dev(qvel2[done_idx]), rows=idx_t)
            self._sim.set_state(qpos2, qvel2)

        # rsl_rl が iteration ごとに平均して tensorboard/標準出力へ書く。報酬は項ごとのstep平均(重み込み)、Metricsは追従誤差など
        log = dict(zip(METRIC_NAMES, metrics.tolist()))
        extras = {"time_outs": truncated, "log": log}
        return self._obs_tensordict(obs), rewards, dones.long(), extras

    def close(self) -> None:
        pass
