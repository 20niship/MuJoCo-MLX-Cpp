"""Unitree G1の平地速度追従タスク(Isaac Lab v2.3.2 `Isaac-Velocity-Flat-G1-v0` の再現。出典cfgとIsaac Labとの差分はREADMEの「G1の学習」)。"""

from __future__ import annotations

import math
import os
import re

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjmlx_go2_env import _projected_gravity_t, _quat_rotate_inverse_t, _yaw_to_quat

PHYSICS_DT = 0.005  # velocity_env_cfg: sim.dt=0.005, decimation=4
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
EPISODE_LENGTH_S = 20.0
MAX_EPISODE_STEPS = int(EPISODE_LENGTH_S / CONTROL_DT)
SOLVER_ITERATIONS = 16  # Go2と同じ(GPUソルバの収束のため)
ACTION_SCALE = 0.5  # ActionsCfg: JointPositionActionCfg(scale=0.5, use_default_offset=True)。G1は上書きしない
CONTACT_FORCE_THRESHOLD = 1.0  # ContactSensorの既定閾値
CMD_RESAMPLE_STEPS = int(10.0 / CONTROL_DT)  # resampling_time_range=(10, 10)
REL_STANDING_ENVS = 0.02
HEADING_STIFFNESS = 0.5
CMD_RANGES = dict(lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))  # G1FlatEnvCfg
# ObservationsCfg.PolicyCfgのUnoise(enable_corruption=True)。base_lin_vel, base_ang_vel, projected_gravity, [commands: 無し], joint_pos, joint_vel, [actions: 無し]
OBS_NOISE = dict(lin_vel=0.1, ang_vel=0.2, gravity=0.05, joint_pos=0.01, joint_vel=1.5)

# G1FlatEnvCfg/G1Rewardsの重み。Isaac LabのRewardManagerは weight*func*step_dt を返すので、下のstepでCONTROL_DTを掛ける
W = dict(
    termination=-200.0, track_lin=1.0, track_ang=1.0, lin_vel_z=-0.2, ang_vel_xy=-0.05, dof_torques=-2.0e-6, dof_acc=-1.0e-7,
    action_rate=-0.005, feet_air_time=0.75, feet_slide=-0.1, flat_orientation=-1.0, dof_pos_limits=-1.0,
    joint_deviation_hip=-0.1, joint_deviation_arms=-0.1, joint_deviation_torso=-0.1,
)
TRACK_STD = 0.5  # exp(-err/std^2)
FEET_AIR_THRESHOLD = 0.4
SOFT_LIMIT_FACTOR = 0.9  # G1_CFG.soft_joint_pos_limit_factor

# menagerieのg1.xml(29DoF)の関節順。Isaac LabのG1_CFG(torso_joint=腰のyaw 1軸 + 腕5軸 + 指7軸)との対応は関節名の正規表現で取る
JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
NJ = len(JOINTS)


def _gains(j: str) -> tuple[float, float]:
    """G1_CFG.actuators(legs/feet/arms)の(stiffness, damping)。waist_roll/pitchはIsaac側に無いのでtorso_jointと同じにする"""
    if "ankle" in j:
        return 20.0, 2.0
    if "hip_pitch" in j or "knee" in j or j.startswith("waist"):
        return 200.0, 5.0
    if "hip" in j:
        return 150.0, 5.0
    return 40.0, 10.0  # shoulder/elbow/wrist(arms)


KP = np.array([_gains(j)[0] for j in JOINTS], dtype=np.float32)
KV = np.array([_gains(j)[1] for j in JOINTS], dtype=np.float32)
# G1_CFG.init_state.joint_pos(正規表現で他は0)。elbow_pitch=menagerieのelbow
_DEFAULT = {"hip_pitch": -0.20, "knee": 0.42, "ankle_pitch": -0.23, "elbow": 0.87, "left_shoulder_roll": 0.16, "left_shoulder_pitch": 0.35,
            "right_shoulder_roll": -0.16, "right_shoulder_pitch": 0.35}
DEFAULT_JOINT_POS = np.array([next((v for k, v in _DEFAULT.items() if j == k or j.endswith("_" + k)), 0.0) for j in JOINTS], dtype=np.float32)

# 報酬の関節グループ(Isaac Lab G1のSceneEntityCfgの正規表現に対応)
_idx = lambda pred: np.array([i for i, j in enumerate(JOINTS) if pred(j)], dtype=np.int64)
G_TORQUE = _idx(lambda j: "hip" in j or "knee" in j)  # flat: hip_.* + knee
G_ACC = _idx(lambda j: "hip" in j or "knee" in j)  # rough継承: hip_.* + knee
G_LIMIT = _idx(lambda j: "ankle" in j)
G_HIP = _idx(lambda j: "hip_yaw" in j or "hip_roll" in j)
G_ARMS = _idx(lambda j: any(k in j for k in ("shoulder", "elbow", "wrist")))  # Isaacはshoulder x3/elbow_pitch/elbow_roll。menagerieはwrist pitch/yawも含める
G_TORSO = _idx(lambda j: j.startswith("waist"))  # Isaacはtorso_joint(=waist_yaw)のみ。menagerieのwaist roll/pitchも含める

# 接触を残すgeom(ほかのコリジョンメッシュはcontype=conaffinity=0にして衝突ペアを減らす。IsaacのG1_MINIMAL_CFG相当。足の球は常に残る)
KEEP_COLLISION_MESHES = {"pelvis_contour_link", "torso_link", "left_knee_link", "right_knee_link"}
TERMINATION_BODY = "torso_link"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _g1_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "models", "external", "mujoco_menagerie", "unitree_g1"))


def build_g1_xml(out_path: str, floor_friction: float = 0.8) -> None:
    """menagerieのg1.xmlを学習用に書き換えて保存する(元ファイルは変えない)。"""
    # PDゲインとfrictionloss=0はIsaac LabのG1_CFG、床摩擦0.8はphysics_material(静摩擦0.8)に合わせる。
    # actuatorfrcrange(実機トルク)はGPUに実装が無くCPU/GPUで挙動が分かれるため外す(Isaac Labのeffort_limit_sim=300/20も未適用)。
    d = _g1_dir()
    with open(os.path.join(d, "g1.xml")) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets"', f'meshdir="{os.path.join(d, "assets")}"')
    xml = xml.replace('<option integrator="implicitfast"/>', f'<option timestep="{PHYSICS_DT}" integrator="implicitfast"/>')
    xml = xml.replace('<joint armature="0.01" frictionloss="0.3"/>', '<joint armature="0.01" frictionloss="0"/>')
    xml = xml.replace('<position kp="500" dampratio="1" inheritrange="1"/>', '<position inheritrange="1"/>')
    xml = re.sub(r'\s*actuatorfrcrange="[^"]*"', "", xml)

    def collision(m: re.Match) -> str:
        g = m.group(0)
        mesh = re.search(r'mesh="([^"]+)"', g)
        if mesh and mesh.group(1) in KEEP_COLLISION_MESHES:
            return g
        return g.replace("<geom ", '<geom contype="0" conaffinity="0" ', 1)

    xml = re.sub(r'<geom\b[^>]*class="collision"[^>]*/>', collision, xml)
    act = "\n".join(f'    <position class="g1" name="{j}_joint" joint="{j}_joint" kp="{KP[i]:g}" kv="{KV[i]:g}"/>' for i, j in enumerate(JOINTS))
    xml = re.sub(r"<actuator>.*?</actuator>", f"<actuator>\n{act}\n  </actuator>", xml, flags=re.S)
    sens = "\n".join(f'    <framelinvel name="{b}_linvel" objtype="body" objname="{b}"/>' for b in FOOT_BODIES)  # feet_slideの足速度
    xml = re.sub(r"<sensor>.*?</sensor>", f"<sensor>\n{sens}\n  </sensor>", xml, flags=re.S)
    xml = xml.replace("</worldbody>", f'  <geom name="floor" type="plane" size="0 0 0.05" friction="{floor_friction} 0.005 0.0001"/>\n  </worldbody>', 1)
    with open(out_path, "w") as f:
        f.write(xml)


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_frame_vel(q: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    """世界座標のvをヨーのみの回転の逆で回す(track_lin_vel_xy_yaw_frame_expの`quat_apply_inverse(yaw_quat(q), v)`)"""
    yaw = _yaw_from_quat(q)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1], v_w[:, 2]], dim=1)


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """body座標のvを世界座標へ(qの回転)。q: (N,4)wxyz"""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


TERM_NAMES = [
    "termination", "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "action_rate_l2",
    "feet_air_time", "feet_slide", "flat_orientation_l2", "dof_pos_limits", "joint_deviation_hip", "joint_deviation_arms", "joint_deviation_torso",
]
METRIC_NAMES = [f"Reward/{k}" for k in TERM_NAMES] + [
    "Metrics/lin_vel_error", "Metrics/ang_vel_error", "Metrics/cmd_speed", "Metrics/base_height", "Metrics/base_tilt", "Metrics/joint_pos_dev",
    "Metrics/foot_contact_frac", "Termination/base_contact", "Termination/blew_up", "Termination/time_out",
]


class MjmlxG1Env(VecEnv):
    def __init__(
        self,
        native_module,
        num_envs: int,
        use_gpu: bool = True,
        seed: int | None = None,
        device: str = "cpu",
        generated_xml_path: str = "/tmp/g1_flat_generated.xml",
        self_collision: bool = False,  # G1_CFG: enabled_self_collisions=False
        max_condim: int = 0,
        obs_noise: bool = True,
        solver_iterations: int = SOLVER_ITERATIONS,
    ):
        import mujoco

        self._native = native_module
        build_g1_xml(generated_xml_path)
        mjm = mujoco.MjModel.from_xml_path(generated_xml_path)
        assert mjm.nu == NJ and mjm.nq == 7 + NJ
        assert [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, 1 + i) for i in range(NJ)] == [f"{j}_joint" for j in JOINTS]
        rng_ = mjm.jnt_range[1:1 + NJ].astype(np.float32)
        self._joint_lo, self._joint_hi = rng_[:, 0], rng_[:, 1]
        bid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
        self._torso_id, self._foot_ids = bid(TERMINATION_BODY), [bid(b) for b in FOOT_BODIES]
        # 初期の高さ: 既定姿勢で足底の球が床に触れる高さ + 余裕(Isaac LabのG1_CFGはz=0.74。menagerieのメッシュ寸法が違うので求め直す)
        d = mujoco.MjData(mjm)
        d.qpos[2], d.qpos[3] = 0.0, 1.0
        d.qpos[7:7 + NJ] = DEFAULT_JOINT_POS
        mujoco.mj_forward(mjm, d)
        sole = d.geom_xpos[:, 2] - np.where(mjm.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE, mjm.geom_size[:, 0], 0.0)
        foot_geoms = [g for g in range(mjm.ngeom) if mjm.geom_bodyid[g] in self._foot_ids and mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.init_height = float(-sole[foot_geoms].min() + 0.02)

        self._model = native_module.load_model(generated_xml_path, self_collision=self_collision, max_condim=max_condim)
        self._sim = native_module.create_batched(self._model, num_envs, frame_skip=1, use_gpu=use_gpu, solver_iterations=solver_iterations)
        self.nq, self.nv, self.nu = self._model.nq, self._model.nv, self._model.nu
        assert (self.nq, self.nv, self.nu) == (7 + NJ, 6 + NJ, NJ)
        self.num_envs, self.num_actions, self.max_episode_length = num_envs, NJ, MAX_EPISODE_STEPS
        self.device = device
        self.obs_noise = obs_noise
        self.cfg = {"task": "g1_flat_velocity", "control_dt": CONTROL_DT, "max_episode_steps": MAX_EPISODE_STEPS}

        self._pack = None
        try:  # 毎step読む値(qpos/qvel/torso・足の接触力ノルム)をGPUで1つに詰めて1回で転送する(CPU実行では使えない)
            self._pack = self._sim.set_pack_spec(np.array([self._torso_id, *self._foot_ids], dtype=np.int32), 6, NJ)
        except RuntimeError:
            pass
        if self._pack is not None:
            self._pk_pin = torch.empty((num_envs, self._pack), dtype=torch.float32, pin_memory=torch.cuda.is_available())
            self._pk_np = self._pk_pin.numpy()

        self._rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(0 if seed is None else seed)
        dev, f32 = device, torch.float32
        t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)
        self._default_pos, self._kp, self._kv = t(DEFAULT_JOINT_POS), t(KP), t(KV)
        self._lo, self._hi = t(self._joint_lo), t(self._joint_hi)
        mid, half = 0.5 * (self._lo + self._hi), 0.5 * (self._hi - self._lo) * SOFT_LIMIT_FACTOR
        self._soft_lo, self._soft_hi = mid - half, mid + half
        self._elapsed = np.zeros(num_envs, dtype=np.int64)
        z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
        self._commands = z((num_envs, 3))
        self._heading_target = z(num_envs)
        self._is_heading = torch.ones(num_envs, dtype=torch.bool, device=dev)
        self._cmd_timer = np.zeros(num_envs, dtype=np.int64)
        self._last_action, self._prev_joint_vel = z((num_envs, NJ)), z((num_envs, NJ))
        self._contact_time, self._air_time = z((num_envs, 2)), z((num_envs, 2))
        self.last_reward_terms: dict = {}
        qpos, qvel = self._sim.state()
        qpos, qvel = qpos.copy(), qvel.copy()
        self._reset_envs(np.ones(num_envs, dtype=np.int32), qpos, qvel)
        self._sim.set_state(qpos, qvel)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.from_numpy(self._elapsed).to(self.device)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._elapsed = value.detach().cpu().numpy().astype(np.int64)

    def _to_dev(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device)

    def _resample_commands(self, idx: np.ndarray) -> None:
        """UniformVelocityCommand._resample_command: 一様サンプリング + 2%は静止(速度0)、全envがheading指令"""
        n = len(idx)
        it = torch.from_numpy(idx).to(self.device)
        u = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi, generator=self._gen)
        cmd = torch.stack([u(*CMD_RANGES["lin_vel_x"]), u(*CMD_RANGES["lin_vel_y"]), u(*CMD_RANGES["ang_vel_z"])], dim=1)
        standing = torch.rand(n, device=self.device, generator=self._gen) <= REL_STANDING_ENVS
        cmd[standing] = 0.0
        self._commands[it] = cmd
        self._heading_target[it] = u(-math.pi, math.pi)
        self._is_heading[it] = ~standing  # standing envはheadingで更新しない(`is_heading_env & ~is_standing_env`)
        self._cmd_timer[idx] = 0

    def _reset_envs(self, mask: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return
        # reset_base: x,y in +-0.5, yaw in +-pi、速度0(G1は上書きして0)。reset_robot_joints: position_range=(1,1)=既定姿勢、速度0
        qpos[idx, 0] = self._rng.uniform(-0.5, 0.5, size=n)
        qpos[idx, 1] = self._rng.uniform(-0.5, 0.5, size=n)
        qpos[idx, 2] = self.init_height
        qpos[idx, 3:7] = _yaw_to_quat(self._rng.uniform(-math.pi, math.pi, size=n))
        qpos[idx, 7:] = DEFAULT_JOINT_POS
        qvel[idx] = 0.0
        self._elapsed[idx] = 0
        it = torch.from_numpy(idx).to(self.device)
        self._last_action[it] = 0.0
        self._prev_joint_vel[it] = 0.0
        self._contact_time[it] = 0.0
        self._air_time[it] = 0.0
        self._resample_commands(idx)

    def _update_heading_commands(self, quat: torch.Tensor) -> None:
        """heading_command: ang_vel_z = clip(0.5 * wrap(heading_target - yaw), range)"""
        err = _wrap_to_pi(self._heading_target - _yaw_from_quat(quat))
        w = torch.clamp(HEADING_STIFFNESS * err, *CMD_RANGES["ang_vel_z"])
        self._commands[:, 2] = torch.where(self._is_heading, w, self._commands[:, 2])

    def _obs(self, quat: torch.Tensor, qvel: torch.Tensor, qpos_j: torch.Tensor) -> torch.Tensor:
        lin, ang, grav = _quat_rotate_inverse_t(quat, qvel[:, 0:3]), qvel[:, 3:6], _projected_gravity_t(quat)
        jp, jv = qpos_j - self._default_pos, qvel[:, 6:]
        if self.obs_noise:
            nz = lambda x, s: x + torch.empty_like(x).uniform_(-s, s, generator=self._gen)
            lin, ang, grav = nz(lin, OBS_NOISE["lin_vel"]), nz(ang, OBS_NOISE["ang_vel"]), nz(grav, OBS_NOISE["gravity"])
            jp, jv = nz(jp, OBS_NOISE["joint_pos"]), nz(jv, OBS_NOISE["joint_vel"])
        return torch.cat([lin, ang, grav, self._commands, jp, jv, self._last_action], dim=1)

    def _obs_tensordict(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        qpos, qvel = (self._to_dev(a) for a in self._sim.state())
        return self._obs_tensordict(self._obs(qpos[:, 3:7], qvel, qpos[:, 7:]))

    def reset(self) -> tuple[TensorDict, dict]:
        qpos, qvel = self._sim.state()
        qpos, qvel = qpos.copy(), qvel.copy()
        self._reset_envs(np.ones(self.num_envs, dtype=np.int32), qpos, qvel)
        self._sim.set_state(qpos, qvel)
        qp, qv = self._to_dev(qpos), self._to_dev(qvel)
        self._update_heading_commands(qp[:, 3:7])
        return self._obs_tensordict(self._obs(qp[:, 3:7], qv, qp[:, 7:])), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        N, nq, nv = self.num_envs, self.nq, self.nv
        action = actions.detach().to(self.device, dtype=torch.float32)
        target = torch.minimum(torch.maximum(self._default_pos + ACTION_SCALE * action, self._lo), self._hi)  # 関節範囲へclip
        self._sim.step(np.ascontiguousarray(target.cpu().numpy()), DECIMATION)
        if self._pack is not None:
            self._sim.read_pack_into(self._pk_np)
            pk = self._pk_pin.to(self.device, non_blocking=True)
            qpos_h, qvel_h = self._pk_np[:, :nq], self._pk_np[:, nq:nq + nv]
            qpos, qvel, force = pk[:, :nq], pk[:, nq:nq + nv], pk[:, nq + nv:nq + nv + 3]  # [torso, left foot, right foot]の接触力ノルム
        else:  # CPU実行
            qpos_h, qvel_h = self._sim.state()
            qpos, qvel = self._to_dev(qpos_h), self._to_dev(qvel_h)
            cfrc = self._to_dev(self._sim.cfrc_ext()).reshape(N, -1, 6)
            force = cfrc[:, [self._torso_id, *self._foot_ids], 3:6].norm(dim=2)
        foot_vel = self._to_dev(self._sim.sensordata())[:, :6].reshape(N, 2, 3)[:, :, :2]  # 足(ankle_roll_link)の世界座標xy速度

        quat, jpos, jvel = qpos[:, 3:7], qpos[:, 7:], qvel[:, 6:]
        base_lin, base_ang_b, grav = _quat_rotate_inverse_t(quat, qvel[:, 0:3]), qvel[:, 3:6], _projected_gravity_t(quat)
        yaw_rate_w = _rotate(quat, base_ang_b)[:, 2]  # 世界座標の角速度z
        self._elapsed += 1
        self._cmd_timer += 1
        cmd = self._commands

        # --- 接触(ContactSensor相当): 足のcontact/air time ---
        feet_contact = force[:, 1:] > CONTACT_FORCE_THRESHOLD
        self._contact_time = (self._contact_time + CONTROL_DT) * feet_contact
        self._air_time = (self._air_time + CONTROL_DT) * ~feet_contact

        # --- 報酬(Isaac Lab flat G1)。Isaac LabはweightにCONTROL_DTを掛けて返す ---
        blew_up = ~(torch.cat([qpos, qvel], dim=1).abs().amax(dim=1) <= 1000.0)
        base_contact = force[:, 0] > CONTACT_FORCE_THRESHOLD
        truncated = self._to_dev(self._elapsed >= MAX_EPISODE_STEPS)
        terminated = (base_contact | blew_up) & ~truncated
        vel_yaw = _yaw_frame_vel(quat, qvel[:, 0:3])
        in_contact = self._contact_time > 0.0
        in_mode = torch.where(in_contact, self._contact_time, self._air_time)
        single = in_contact.int().sum(dim=1) == 1
        air_r = torch.clamp(torch.where(single.unsqueeze(-1), in_mode, torch.zeros_like(in_mode)).amin(dim=1), max=FEET_AIR_THRESHOLD)
        air_r = air_r * (cmd[:, :2].norm(dim=1) > 0.1)
        joint_acc = ((jvel - self._prev_joint_vel) / CONTROL_DT).clamp(-500.0, 500.0)  # 接触衝撃の差分スパイク対策(Go2と同じ)
        torque = self._kp * (target - jpos) - self._kv * jvel  # 位置アクチュエータ(implicit)の出力トルク。effort limitは未適用
        lim = (-(jpos - self._soft_lo).clamp(max=0.0) + (jpos - self._soft_hi).clamp(min=0.0))[:, G_LIMIT].sum(dim=1)
        dev = (jpos - self._default_pos).abs()
        ang_err, lin_err = (cmd[:, 2] - yaw_rate_w) ** 2, ((cmd[:, :2] - vel_yaw[:, :2]) ** 2).sum(dim=1)
        raw = {
            "termination": terminated.float(),
            "track_lin_vel_xy_exp": torch.exp(-lin_err / TRACK_STD**2),
            "track_ang_vel_z_exp": torch.exp(-ang_err / TRACK_STD**2),
            "lin_vel_z_l2": base_lin[:, 2].clamp(-20.0, 20.0) ** 2,
            "ang_vel_xy_l2": (base_ang_b[:, :2].clamp(-50.0, 50.0) ** 2).sum(dim=1),
            "dof_torques_l2": (torque[:, G_TORQUE] ** 2).sum(dim=1),
            "dof_acc_l2": (joint_acc[:, G_ACC] ** 2).sum(dim=1),
            "action_rate_l2": ((action - self._last_action) ** 2).sum(dim=1),
            "feet_air_time": air_r,
            "feet_slide": (foot_vel.norm(dim=2) * feet_contact).sum(dim=1),
            "flat_orientation_l2": (grav[:, :2] ** 2).sum(dim=1),
            "dof_pos_limits": lim,
            "joint_deviation_hip": dev[:, G_HIP].sum(dim=1),
            "joint_deviation_arms": dev[:, G_ARMS].sum(dim=1),
            "joint_deviation_torso": dev[:, G_TORSO].sum(dim=1),
        }
        wmap = dict(zip(TERM_NAMES, [W["termination"], W["track_lin"], W["track_ang"], W["lin_vel_z"], W["ang_vel_xy"], W["dof_torques"], W["dof_acc"],
                                     W["action_rate"], W["feet_air_time"], W["feet_slide"], W["flat_orientation"], W["dof_pos_limits"],
                                     W["joint_deviation_hip"], W["joint_deviation_arms"], W["joint_deviation_torso"]]))
        terms = torch.stack([wmap[k] * raw[k] * CONTROL_DT for k in TERM_NAMES])
        rewards = torch.where(blew_up, torch.zeros_like(terms[0]), terms.sum(dim=0).clamp(-50.0, 50.0))  # 発散時のvalue爆発防止(Go2と同じ最終防衛線)
        self.last_reward_terms = dict(zip(TERM_NAMES, terms))
        self._prev_joint_vel, self._last_action = jvel.clone(), action.clone()

        dones = truncated | base_contact | blew_up
        f = lambda x: x.float().mean()
        metrics = torch.stack(list(terms.mean(dim=1)) + [
            lin_err.sqrt().mean(), ang_err.sqrt().mean(), cmd.norm(dim=1).mean(), qpos[:, 2].mean(), grav[:, :2].norm(dim=1).mean(), dev.mean(),
            f(feet_contact), f(base_contact), f(blew_up), f(truncated),
        ])

        # コマンドの再サンプル(10s毎)とheading指令の更新、リセット
        due = np.nonzero(self._cmd_timer >= CMD_RESAMPLE_STEPS)[0]
        if len(due):
            self._resample_commands(due)
        self._update_heading_commands(quat)
        obs = self._obs(quat, qvel, jpos)
        done_idx = np.nonzero(dones.cpu().numpy())[0]
        if len(done_idx) > 0:
            qpos2, qvel2 = qpos_h.copy(), qvel_h.copy()
            mask = np.zeros(N, dtype=np.int32)
            mask[done_idx] = 1
            self._reset_envs(mask, qpos2, qvel2)
            self._sim.set_state(qpos2, qvel2)
            it = torch.from_numpy(done_idx).to(self.device)
            q2, v2 = self._to_dev(qpos2[done_idx]), self._to_dev(qvel2[done_idx])
            self._update_heading_commands(self._to_dev(qpos2[:, 3:7]))  # リセットしたenvのheading指令を新しい向きで更新
            obs[it] = self._obs_rows(q2, v2, it)
        log = dict(zip(METRIC_NAMES, metrics.tolist()))
        return self._obs_tensordict(obs), rewards, dones.long(), {"time_outs": truncated, "log": log}

    def _obs_rows(self, qpos: torch.Tensor, qvel: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:  # リセットしたenvの観測(その行のcommands/last_actionを使う)
        saved_cmd, saved_act = self._commands, self._last_action
        self._commands, self._last_action = saved_cmd[rows], saved_act[rows]
        try:
            return self._obs(qpos[:, 3:7], qvel, qpos[:, 7:])
        finally:
            self._commands, self._last_action = saved_cmd, saved_act

    def close(self) -> None:
        pass
