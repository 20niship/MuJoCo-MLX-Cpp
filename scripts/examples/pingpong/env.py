"""Keep-up on the GPU (vkmujoco) for rsl-rl: keep a ball bouncing on an SO-101 arm's paddle.

`KeepUpVecEnv` steps every environment together. By default the policy observes a 33-d state including the ball.
With `camera=True` the ball is removed from the actor's observation and replaced by two fixed cameras (cam_a behind the
arm, cam_side beside the table, now and one step ago); the critic still sees the true ball (asymmetric actor-critic).

Contacts are inferred from the ball's velocity change over `chunks` short calls per control step (not read from
MjData); one control rate (30 Hz). Ball mass, drag and servo gain are randomised per environment. The scene
(scene/*.xml, meshes fetched as scene/README.md says) and the reference pose/footprint (scene/ref.*) are baked by robo-ping-pong's
scripts/bake_vkmujoco_example.py, so this example needs no IK or hardware code.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import vkmujoco
from rsl_rl.env import VecEnv
from tensordict import TensorDict

SCENE = Path(__file__).resolve().parent / "scene"

CAUSES = ("", "dropped", "hit the arm", "flew off", "stalled")
CAUSE_DROPPED, CAUSE_ARM, CAUSE_FLEW, CAUSE_STALLED = 1, 2, 3, 4
CONTACT_DV = 0.4  # m/s: a velocity change this far beyond free flight is a contact
CURRICULUM_RING, CURRICULUM_STAGES = 0.06, 9
BALL_SLICE = slice(15, 21)  # ball position (3) and velocity (3) within the 33-d observation
BALL_SLICE = slice(15, 21)  # ball position (3) and velocity (3) within the 33-d observation


@dataclass(frozen=True)
class Randomization:
    """What changes from one episode to the next, beyond where the ball starts.

    Drawn once per episode at reset, except where noted. The reward and the
    termination checks always use the TRUE state; only what the policy observes
    is corrupted. The point is a policy that does not depend on any one number
    we are unsure of, because on hardware every one of these is wrong by some
    amount we will only measure later.
    """

    # Physics, as multipliers on (or ranges for) the nominal values.
    ball_mass: tuple[float, float] = (0.9, 1.1)
    drag_coefficient: tuple[float, float] = (0.40, 0.55)  # the literature's range
    # Every takeoff direction is turned by this much, one-sigma, about a random horizontal axis.
    takeoff_scatter_deg: float = 2.0
    # Servo position gain, as a multiplier.
    servo_kp: tuple[float, float] = (0.8, 1.2)

    # Sensing, one-sigma. Ball position at the 2.0 mm the camera study
    # measured for the fused pair; joints at about one encoder tick (4096/rev).
    ball_pos_noise: float = 0.002
    ball_vel_noise: float = 0.05
    joint_pos_noise: float = 0.0015
    joint_vel_noise: float = 0.02
    # Camera pipeline lag: the ball is observed this many decisions late.
    ball_latency_steps: tuple[int, ...] = (0, 1)

    # Disturbances: a sideways nudge to the ball in flight, standing in for
    # spin, air currents and everything else the v1 ball model leaves out.
    #
    # A guess, and the one that decides how long a rally lasts. At first it was
    # 0.5/s at 0.15 m/s; with it v6 lasted a median 24.5 s, without it 113 s,
    # and in an otherwise calm world over half its rallies reached 5 minutes.
    # That is stronger than still indoor air plausibly pushes a ball, so it is
    # 3x weaker and rarer until filmed real bounces say how far balls wander.
    push_rate_hz: float = 0.2
    push_speed: float = 0.05  # one-sigma, m/s, horizontal


@dataclass(frozen=True)
class KeepUpConfig:
    episode_s: float = 20.0
    # How fast the policy may move each joint's target, rad/s. About the
    # STS3215's no-load top speed, so nothing asked for here is beyond the
    # hardware - and enough to cross the footprint while a dropped ball falls.
    max_joint_speed: float = 4.5
    # Time a servo takes to start following a moved target. With the speed
    # above, it sets how long the arm needs to get under a ball - see
    # `reach_time` and `face_up_footprint`.
    servo_lag_s: float = 0.10

    # A bounce only scores if the ball rose at least this far above the face
    # since the last one that scored. Without it, the cheapest policy is to
    # rest the ball on the blade and buzz it: dozens of contacts a second,
    # none of them a bounce.
    min_rise: float = 0.04

    # Getting under the dropped ball at all. The first contact of an episode
    # ends a flight the policy did not launch, so it earns this flat bonus
    # instead of a flight score - which matters most for far drops, where the
    # catch is the hard part.
    catch: float = 1.0

    # What a good bounce is. Scored once per completed flight - takeoff from the
    # paddle to the next scoring contact - as
    #
    #     flight_weight * flight_time * height * straight * home
    #
    # Paying per SECOND of flight rather than per bounce is the point. v1 was
    # paid +1 a bounce inside a 20 s cap, so the best policy was the fastest
    # legal dribble: ~80 bounces at an ~8 cm apex, barely over `min_rise`. Per
    # second, a low bounce earns nothing for being frequent; it has to be good.
    #
    # Height: full marks for an apex 0.20-0.35 m above the face it left. 0.20 m
    # is 0.40 s in the air, twelve decisions at 30 Hz - time to correct a bad
    # bounce. 0.35 m keeps impact under 2.7 m/s and the drift from a small tilt
    # error inside the blade. Outside the band the score falls off smoothly, and
    # never below `low_credit`, so the first clumsy bounces still teach.
    flight_weight: float = 2.0  # reward per second of flight at full quality
    height_band: tuple[float, float] = (0.20, 0.35)
    height_sigma: float = 0.05
    low_credit: float = 0.25
    # Straight: scored on the takeoff velocity against the IDEAL direction,
    # which is not always vertical. Away from the centre, the ideal bounce
    # carries the ball `homing_gain` of the way back towards it over one
    # flight; at the centre that is straight up. Half credit is kept for any
    # angle so height still pays alone.
    straight_sigma_deg: float = 10.0
    homing_gain: float = 0.5
    # Home: the flight score is scaled by how near the centre it came down,
    # from half at a distance to full within a few centimetres.
    home_sigma: float = 0.06
    # A small bonus per contact for striking near the blade's centre.
    centre: float = 0.2
    centre_sigma: float = 0.025  # metres off the blade centre
    # Staying in the game, per second the ball is alive. Without it every
    # step only costs (the two terms below), and the first v3 run learned the
    # obvious answer: take the catch bonus and knock the ball away, ending the
    # episode in 0.45 s - 97% of episodes, return pinned at the catch bonus.
    # 0.5/s is small next to a good bounce (up to 2/s) but outweighs what a
    # still-clumsy policy spends moving, so leaving early never pays. A ball
    # resting on the blade cannot farm it: `stall_s` ends that episode.
    alive: float = 0.5
    # Costs that protect the hardware rather than shape the task. Kept small:
    # with velocity actions an exploring policy moves a lot, and at the old
    # 0.02 / 0.001 they summed to ~-25 an episode.
    action_rate: float = 0.005
    joint_velocity: float = 0.0002

    # Failure conditions.
    stall_s: float = 2.0  # no scoring bounce for this long: the ball is dead
    max_reach: float = 0.50  # the ball has gone further than this from the base

    # Where the ball starts. `spawn_radius` 0 drops it over the centre,
    # `drop_offset` off; above 0, anywhere in the face-up footprint within
    # `spawn_radius * spawn_scale` of the centre. `spawn_scale` belongs to the
    # env, not the config, so a curriculum can raise it as the policy earns it.
    # Even at full spread `centre_fraction` of drops stay at the centre, so the
    # policy keeps practising the rally while it learns to chase.
    spawn_radius: float = 0.0
    centre_fraction: float = 0.25
    # While a curriculum stage is being learned, this fraction of the
    # off-centre drops come from its newest ring - between `spawn_scale_prev`
    # and `spawn_scale` of the radius - so the stage is actually practised
    # rather than diluted by every distance already mastered.
    frontier_fraction: float = 0.5
    drop_height: tuple[float, float] = (0.08, 0.25)  # above the nominal face
    drop_offset: float = 0.015
    drop_speed: float = 0.05  # horizontal, m/s
    drop_vz: tuple[float, float] = (0.0, 0.0)  # m/s, thrown down (-) or up (+)
    joint_noise: float = 0.03

    # None: a fixed world. See `randomized()`.
    rand: Randomization | None = None

    @classmethod
    def randomized(cls, **overrides) -> "KeepUpConfig":
        """Drops from anywhere in the footprint, plus every `Randomization` term.

        Drop heights start at 0.25 m so a ball at the edge of the footprint
        falls for long enough (~0.25 s) for the arm to get there.
        """
        base = dict(spawn_radius=0.60, drop_height=(0.25, 0.60), drop_speed=0.10, drop_vz=(-0.3, 0.2), joint_noise=0.06, rand=Randomization())
        base.update(overrides)
        return cls(**base)


class ArmKinematics:
    """Torch forward kinematics of the arm, built from the scene's MjModel."""

    def __init__(self, m: mujoco.MjModel, ref: dict, device: str = "cpu", points_per_mesh: int = 48):
        self.device = torch.device(device)
        self.joint_ids = [int(j) for j in ref["joint_ids"]]
        # Bodies from the root down to the paddle, in order. Each joint's body is on the chain.
        self.face_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, ref["face_site"])
        chain, b = [], int(m.site_bodyid[self.face_site])
        while b != 0:
            chain.append(b)
            b = int(m.body_parentid[b])
        self.chain = chain[::-1]
        self._jidx = {int(m.jnt_bodyid[j]): k for k, j in enumerate(self.joint_ids)}

        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.body_pos = {b: t(m.body_pos[b]) for b in self.chain}
        self.body_rot = {b: t(_quat_to_mat(m.body_quat[b])) for b in self.chain}
        self.jaxis = [t(m.jnt_axis[j]) for j in self.joint_ids]
        self.jpos = [t(m.jnt_pos[j]) for j in self.joint_ids]
        self.site_pos = t(m.site_pos[self.face_site])
        self.site_rot = t(_quat_to_mat(m.site_quat[self.face_site]))

        # Points on every arm geom, in its body frame, with the geom's radius.
        ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        pts, rad, bod = [], [], []
        rng = np.random.default_rng(0)
        for g in range(m.ngeom):
            if g in (ball, table) or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            body = int(m.geom_bodyid[g])
            if m.body_rootid[body] == 0:
                continue
            p, r = _geom_points(m, g, rng, points_per_mesh)
            pts.append(p), rad.append(np.full(len(p), r)), bod.append(np.full(len(p), body))
        self.pts_local = t(np.concatenate(pts))
        self.pts_rad = t(np.concatenate(rad))
        pts_body = np.concatenate(bod)
        # Body index within the chain, for each point (bodies off the chain are mapped to their chain ancestor).
        self.pts_chain_idx = torch.as_tensor([self._chain_index(int(b), m) for b in pts_body], device=self.device)
        # The table is a finite box: only points over it can collide with it.
        tb = ref["table"]
        self.table_x = (-tb["arm_inset"], -tb["arm_inset"] + tb["length"])
        self.table_half_width = tb["width"] / 2

    def _chain_index(self, body: int, m) -> int:
        while body not in self.chain:
            body = int(m.body_parentid[body])
        return self.chain.index(body)

    def _frames(self, q: torch.Tensor):
        """World rotation (N,3,3) and position (N,3) of every body on the chain."""
        N = q.shape[0]
        R = torch.eye(3, device=q.device).expand(N, 3, 3)
        p = torch.zeros(N, 3, device=q.device)
        out = []
        for b in self.chain:
            p = p + R @ self.body_pos[b]
            R = R @ self.body_rot[b]
            k = self._jidx.get(b)
            if k is not None:
                Rj = _axis_angle(self.jaxis[k], q[:, k])
                # rotate about the joint axis through jpos (in the body frame)
                shift = self.jpos[k] - (Rj @ self.jpos[k].unsqueeze(-1)).squeeze(-1)
                p = p + (R @ shift.unsqueeze(-1)).squeeze(-1)
                R = R @ Rj
            out.append((R, p))
        return out

    def face(self, q: torch.Tensor):
        """(centre (N,3), outward normal (N,3)) of the hitting face for joint angles q (N,5)."""
        R, p = self._frames(q)[-1]
        centre = p + (R @ self.site_pos.unsqueeze(-1)).squeeze(-1)
        normal = (R @ self.site_rot)[:, :, 2]
        return centre, normal

    def points(self, q: torch.Tensor) -> torch.Tensor:
        """World positions (N,P,3) of the sampled arm points."""
        fr = self._frames(q)
        Rs = torch.stack([f[0] for f in fr], dim=1)  # (N,C,3,3)
        ps = torch.stack([f[1] for f in fr], dim=1)  # (N,C,3)
        R = Rs[:, self.pts_chain_idx]  # (N,P,3,3)
        p = ps[:, self.pts_chain_idx]
        return p + (R @ self.pts_local.unsqueeze(-1)).squeeze(-1)

    def clearance(self, q: torch.Tensor) -> torch.Tensor:
        """Smallest height of any arm point above the table top, less its radius (N,). Points off the table do not count."""
        P = self.points(q)
        over = (P[..., 0] > self.table_x[0]) & (P[..., 0] < self.table_x[1]) & (P[..., 1].abs() < self.table_half_width)
        h = P[..., 2] - self.pts_rad
        return torch.where(over, h, torch.full_like(h, 1e3)).amin(dim=1)


class TorchTableGuard:
    """the furthest point along current -> target that keeps the arm clear of the table."""

    def __init__(self, kin: ArmKinematics, margin: float = 0.004, bisect_steps: int = 7):
        self.kin, self.margin, self.steps = kin, margin, bisect_steps

    def is_safe(self, q: torch.Tensor) -> torch.Tensor:
        return self.kin.clearance(q) >= self.margin

    def clamp(self, target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        ok_t = self.is_safe(target)
        ok_c = self.is_safe(current)
        lo = torch.zeros(target.shape[0], device=target.device)
        hi = torch.ones_like(lo)
        for _ in range(self.steps):
            mid = 0.5 * (lo + hi)
            ok = self.is_safe(current + (target - current) * mid[:, None])
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        out = current + (target - current) * lo[:, None]
        out = torch.where(ok_c[:, None], out, current)  # already in violation: do not make it worse
        return torch.where(ok_t[:, None], target, out)


# -- helpers ------------------------------------------------------------------


def _quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (N,3,3) about `axis` (3,) by `angle` (N,) - Rodrigues."""
    ax = axis / axis.norm()
    K = torch.zeros(3, 3, device=axis.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -ax[2], ax[1], ax[2], -ax[0], -ax[1], ax[0]
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, device=axis.device) + s * K + (1 - c) * (K @ K)


def _geom_points(m: mujoco.MjModel, g: int, rng, n_mesh: int) -> tuple[np.ndarray, float]:
    """Points (in the body frame) that bound geom g, and the radius to subtract."""
    gt, size = int(m.geom_type[g]), m.geom_size[g]
    R, p0 = _quat_to_mat(m.geom_quat[g]), m.geom_pos[g]
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        a, c = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.asarray(m.mesh_vert[a : a + c])
        # Farthest-point sampling keeps the extremes, which is what a clearance test needs.
        take = [0]
        d = np.linalg.norm(v - v[0], axis=1)
        for _ in range(min(n_mesh, len(v)) - 1):
            i = int(d.argmax())
            take.append(i)
            d = np.minimum(d, np.linalg.norm(v - v[i], axis=1))
        loc, r = v[take], 0.0
    elif gt == mujoco.mjtGeom.mjGEOM_BOX:
        loc = np.array([[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        r = 0.0
    elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        loc, r = np.zeros((1, 3)), float(size[0])
    elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
        loc, r = np.array([[0, 0, -size[1]], [0, 0, size[1]]]), float(size[0])
    elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ring = np.stack([size[0] * np.cos(a), size[0] * np.sin(a)], axis=1)
        loc = np.concatenate([np.c_[ring, np.full(8, h)] for h in (-size[1], size[1])])
        r = 0.0
    else:
        loc, r = np.zeros((1, 3)), float(size[0])
    return loc @ R.T + p0, r


class KeepUpVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        cfg: KeepUpConfig | None = None,
        device: str = "cpu",
        seed: int = 0,
        use_gpu: bool = True,
        chunks: int = 4,
        start_stage: int = 0,
        curriculum: bool = True,
        camera: bool = False,
        size: tuple[int, int] = (96, 64),
    ):
        self.cfg = cfg or KeepUpConfig.randomized()
        c, R = self.cfg, self.cfg.rand
        self.num_envs, self.device = num_envs, torch.device(device)
        self.camera = camera
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=self.device)  # _reset runs inside __init__
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # Baked reference: nominal pose, reachable footprint, indices and constants (see scene/ref.json).
        ref = json.loads((SCENE / "ref.json").read_text())
        arrs = np.load(SCENE / "ref.npz")
        ref.update({k: arrs[k] for k in arrs.files})
        xml = SCENE / ("keepup_cam.xml" if camera else "keepup.xml")
        m = mujoco.MjModel.from_xml_path(str(xml))
        self.r = ref["ball_radius"]
        self.blade_r = ref["blade_radius"]
        self.gravity = ref["gravity"]
        self.q0_np = ref["q0"]
        self.home = np.asarray(ref["nominal_face"][:2], float)
        self.footprint_step = ref["footprint_step"]
        self.kin = ArmKinematics(m, ref, device=device)
        self.guard = TorchTableGuard(self.kin)
        self.lower = torch.as_tensor(ref["lower"], dtype=torch.float32, device=self.device)
        self.upper = torch.as_tensor(ref["upper"], dtype=torch.float32, device=self.device)
        self.q0 = torch.as_tensor(self.q0_np, dtype=torch.float32, device=self.device)
        self.face0_z = ref["face0_z"]
        self.qpos_adr, self.qvel_adr = ref["qpos_adr"], ref["qvel_adr"]
        self.ball_q, self.ball_v = ref["ball_qpos_adr"], ref["ball_dof_adr"]
        self.ball_body, self.nbody, self.nu = ref["ball_body"], ref["nbody"], ref["nu"]
        self.act_order = ref["act_order"]  # ctrl is in actuator order; ours is joint order
        self.base_drag_k, self.base_cd = ref["ball_drag_k"], ref["ball_drag_coefficient"]
        self.footprint, self.reach_time = ref["footprint"], ref["reach_time"]

        # The GPU simulator.
        self._model = vkmujoco.load_model(str(xml))
        assert self._model.nq == m.nq and self._model.nu == m.nu
        self._sim = vkmujoco.create_batched(self._model, num_envs, use_gpu=use_gpu, domain_rand=True)
        self.timestep = ref["timestep"]
        self.substeps = ref["steps_per_command"]
        self.dt = self.substeps * self.timestep
        sizes = [self.substeps // chunks + (i < self.substeps % chunks) for i in range(chunks)]
        self.chunk_steps = [s for s in sizes if s > 0]
        if camera:
            self.width, self.height = size
            self._cam = vkmujoco.create_camera(self._sim, self.width, self.height)
            assert self._cam.num_cameras == 2, "expected cam_a and cam_side in the scene"
            self._prev = torch.zeros(num_envs, 6, self.height, self.width, device=self.device, dtype=torch.uint8)

        self.num_actions = 5
        self.max_episode_length = int(round(c.episode_s / self.dt))
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.cfg_dict = {"task": "keepup", "control_dt": self.dt, "max_episode_steps": self.max_episode_length}

        # Curriculum.
        self.curriculum = curriculum
        self.stage = start_stage
        self.recent: deque = deque(maxlen=200)
        self._apply_stage()
        if not curriculum:
            self.spawn_scale = self.spawn_scale_prev = 1.0

        # Per-environment episode state.
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.device)
        N = num_envs
        self.target, self.prev_action = z(N, 5), z(N, 5)
        self.armed, self.has_takeoff = z(N, dt=torch.bool), z(N, dt=torch.bool)
        self.peak, self.apex_z, self.since_bounce = z(N), z(N), z(N)
        self.to_t0, self.to_facez, self.to_off_ideal, self.to_off_vert = z(N), z(N), z(N), z(N)
        self.bounces, self.latency, self.cause = z(N, dt=torch.long), z(N, dt=torch.long), z(N, dt=torch.long)
        self.spawn_dist, self.sim_time = z(N), z(N)
        self.seen_pos, self.seen_vel = z(N, 3), z(N, 3)
        self.flight_apex, self.flight_n, self.last_home = z(N), z(N), z(N)
        self.finished_bounces, self.finished_cause = np.zeros(0), np.zeros(0, dtype=int)

        qpos, qvel = self._sim.state()
        qpos, qvel = qpos.copy(), qvel.copy()
        self._reset(np.ones(N, bool), qpos, qvel)
        self._sim.set_state(qpos, qvel)
        self._q, self._qd = self._to_t(qpos), self._to_t(qvel)

    # -- curriculum -----------------------------------------------------------

    def _apply_stage(self) -> None:
        self.spawn_scale = self.stage * CURRICULUM_RING / self.cfg.spawn_radius if self.cfg.spawn_radius else 0.0
        self.spawn_scale_prev = max(0, self.stage - 1) * CURRICULUM_RING / self.cfg.spawn_radius if self.cfg.spawn_radius else 0.0

    def _update_curriculum(self, bounces: np.ndarray, spawn_dist: np.ndarray) -> None:
        inner = (self.stage - 1) * CURRICULUM_RING
        for b, d in zip(bounces, spawn_dist):
            if self.stage == 0 or d > inner:
                self.recent.append(float(b))
        need, bar = (200, 15.0) if self.stage == 0 else (100, 12.0)
        if self.stage < CURRICULUM_STAGES and len(self.recent) >= need and np.mean(self.recent) >= bar:
            self.stage += 1
            self.recent = deque(maxlen=100)
            self._apply_stage()
            print(f"curriculum: stage {self.stage}/{CURRICULUM_STAGES}", flush=True)

    # -- resets ---------------------------------------------------------------

    def _spawn(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Ball drop points (n,3) and the distance of each from the centre (n,)."""
        c, rng = self.cfg, self.rng
        z = self.face0_z + self.r + rng.uniform(*c.drop_height, n)
        reach = c.spawn_radius * self.spawn_scale
        fall = np.sqrt(2.0 * (z - self.r - self.face0_z) / self.gravity)
        catchable = self.reach_time[None, :] <= fall[:, None]  # (n, P)
        dist = np.hypot(*(self.footprint - self.home).T)
        near = catchable & (dist[None, :] <= reach)
        inner = c.spawn_radius * self.spawn_scale_prev
        ring = catchable & (dist[None, :] > inner) & (dist[None, :] <= reach)
        use_ring = ring.any(1) & (inner < reach) & (rng.random(n) < c.frontier_fraction)
        pool = np.where(use_ring[:, None], ring, near)
        pool = np.where(pool.any(1, keepdims=True), pool, catchable)
        centre = (reach < self.footprint_step) | ~pool.any(1) | (rng.random(n) < c.centre_fraction)
        # a uniformly random True entry of each row
        pick = (np.cumsum(pool, 1) > (rng.random(n)[:, None] * pool.sum(1, keepdims=True))).argmax(1)
        xy = self.footprint[pick] + rng.uniform(-0.5, 0.5, (n, 2)) * self.footprint_step
        off = rng.uniform(-c.drop_offset, c.drop_offset, (n, 2))
        xy = np.where(centre[:, None], self.home + off, xy)
        return np.c_[xy, z], np.hypot(*(xy - self.home).T)

    def _reset(self, mask: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Re-initialise the environments in `mask`: writes qpos/qvel (host arrays) and the per-env state and physics."""
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return
        c, R, rng = self.cfg, self.cfg.rand, self.rng
        q = self.q0_np + rng.uniform(-c.joint_noise, c.joint_noise, (n, 5))
        q = np.clip(q, self.lower.cpu().numpy(), self.upper.cpu().numpy())
        qpos[idx[:, None], self.qpos_adr] = q
        qvel[idx[:, None], self.qvel_adr] = 0.0
        spawn, dist = self._spawn(n)
        v = np.c_[rng.uniform(-c.drop_speed, c.drop_speed, (n, 2)), rng.uniform(*c.drop_vz, n)]
        qpos[idx, self.ball_q : self.ball_q + 3] = spawn
        qpos[idx, self.ball_q + 3 : self.ball_q + 7] = (1.0, 0.0, 0.0, 0.0)
        qvel[idx, self.ball_v : self.ball_v + 6] = 0.0
        qvel[idx, self.ball_v : self.ball_v + 3] = v

        if R is not None:  # per-environment physics, drawn once per episode
            mass = np.ones((n, self.nbody), np.float32)
            mass[:, self.ball_body] = rng.uniform(*R.ball_mass, n)
            cd = rng.uniform(*R.drag_coefficient, n)
            gain = rng.uniform(*R.servo_kp, (n, self.nu)).astype(np.float32)
            ids = idx.astype(np.int32)
            self._sim.set_env_params(env_idx=ids, mass_scale=mass, gain_scale=gain)
            self._sim.set_free_drag(env_idx=ids, drag_k=(self.base_drag_k * cd / self.base_cd)[:, None].astype(np.float32))
            self.latency[torch.as_tensor(idx, device=self.device)] = torch.as_tensor(rng.choice(R.ball_latency_steps, n), device=self.device)

        t = torch.as_tensor(idx, device=self.device)
        tq = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        self.target[t] = tq
        self.prev_action[t] = 0.0
        self.armed[t], self.has_takeoff[t] = True, False
        self.peak[t], self.apex_z[t] = -1e3, -1e3
        self.since_bounce[t] = 0.0
        self.bounces[t], self.cause[t] = 0, 0
        self.spawn_dist[t] = torch.as_tensor(dist, dtype=torch.float32, device=self.device)
        self.sim_time[t] = 0.0
        self.seen_pos[t] = torch.as_tensor(spawn, dtype=torch.float32, device=self.device)
        self.seen_vel[t] = torch.as_tensor(v, dtype=torch.float32, device=self.device)
        self.flight_apex[t], self.flight_n[t], self.last_home[t] = 0.0, 0.0, 0.0
        self.episode_length_buf[t] = 0
        self._fresh[t] = True  # the camera has no previous frame

    # -- helpers --------------------------------------------------------------

    def _to_t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32).to(self.device)

    def _arm(self, qpos):
        return qpos[:, self.qpos_adr]

    def _ball(self, qpos, qvel):
        return qpos[:, self.ball_q : self.ball_q + 3], qvel[:, self.ball_v : self.ball_v + 3]

    def _scatter(self, v: torch.Tensor, sigma_deg: float) -> torch.Tensor:
        """Turn each velocity by a small random angle about a random horizontal axis."""
        N = v.shape[0]
        speed = v.norm(dim=1, keepdim=True).clamp_min(1e-9)
        phi = torch.rand(N, device=self.device, generator=self.gen) * 2 * np.pi
        horiz = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], dim=1)
        k = torch.cross(v / speed, horiz, dim=1)
        k = k / k.norm(dim=1, keepdim=True).clamp_min(1e-9)
        th = torch.randn(N, device=self.device, generator=self.gen)[:, None] * np.radians(sigma_deg)
        return v * torch.cos(th) + torch.cross(k, v, dim=1) * torch.sin(th) + k * (k * v).sum(1, keepdim=True) * (1 - torch.cos(th))

    def _takeoff_angles(self, pos: torch.Tensor, vel: torch.Tensor):
        """(degrees off the ideal direction, degrees off vertical) for a takeoff at `pos` with `vel`."""
        up = vel[:, 2] > 0.0
        flight = 2.0 * vel[:, 2].clamp_min(1e-6) / self.gravity
        home = torch.as_tensor(self.home, dtype=torch.float32, device=self.device)
        ideal = torch.cat([self.cfg.homing_gain * (home - pos[:, :2]) / flight[:, None], vel[:, 2:3]], dim=1)
        cos = (ideal * vel).sum(1) / (ideal.norm(dim=1) * vel.norm(dim=1)).clamp_min(1e-9)
        off_ideal = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
        off_vert = torch.rad2deg(torch.atan2(vel[:, :2].norm(dim=1), vel[:, 2]))
        ninety = torch.full_like(off_ideal, 90.0)
        return torch.where(up, off_ideal, ninety), torch.where(up, off_vert, ninety)

    def _score_flight(self, now, landed_xy):
        c = self.cfg
        rise = self.apex_z - self.r - self.to_facez
        lo, hi = c.height_band
        gap = torch.maximum(torch.maximum(lo - rise, rise - hi), torch.zeros_like(rise))
        height = c.low_credit + (1 - c.low_credit) * torch.exp(-((gap / c.height_sigma) ** 2))
        straight = 0.5 + 0.5 * torch.exp(-((self.to_off_ideal / c.straight_sigma_deg) ** 2))
        home = torch.as_tensor(self.home, dtype=torch.float32, device=self.device)
        dist = (landed_xy - home).norm(dim=1)
        homing = 0.5 + 0.5 * torch.exp(-((dist / c.home_sigma) ** 2))
        flight = c.flight_weight * (now - self.to_t0) * height * straight * homing
        return torch.where(self.has_takeoff, flight, torch.full_like(flight, c.catch)), rise

    # -- VecEnv API -----------------------------------------------------------

    def _pack(self, obs: torch.Tensor) -> TensorDict:
        """Observation groups for rsl-rl from the 33-d vector."""
        if not self.camera:
            return TensorDict({"policy": obs}, batch_size=[self.num_envs])
        cur = self._frame()
        prev = torch.where(self._fresh[:, None, None, None], cur, self._prev)
        self._prev, self._fresh[:] = cur, False
        image = torch.cat([cur, prev], dim=1).float() / 255.0  # (N, 12, H, W): two cameras, now and one step ago
        proprio = torch.cat([obs[:, : BALL_SLICE.start], obs[:, BALL_SLICE.stop :]], dim=1)
        return TensorDict({"policy": proprio, "camera": image, "critic": obs}, batch_size=[self.num_envs])

    def _frame(self) -> torch.Tensor:
        """The two cameras' RGB for every environment, (N, 6, H, W) uint8 on the policy device."""
        self._cam.render()
        rgb = torch.as_tensor(self._cam.rgb()[..., :3]).to(self.device)  # (N, 2, H, W, 3)
        return rgb.permute(0, 1, 4, 2, 3).reshape(self.num_envs, 6, self.height, self.width)

    def get_observations(self) -> TensorDict:
        return self._pack(self._obs(self._q, self._qd))

    def reset(self):
        qpos, qvel = self._sim.state()
        qpos, qvel = qpos.copy(), qvel.copy()
        self._reset(np.ones(self.num_envs, bool), qpos, qvel)
        self._sim.set_state(qpos, qvel)
        self._q, self._qd = self._to_t(qpos), self._to_t(qvel)
        return self.get_observations(), {}

    def _obs(self, qpos, qvel) -> torch.Tensor:
        R = self.cfg.rand
        q, qd = self._arm(qpos), qvel[:, self.qvel_adr]
        bp, bv = self._ball(qpos, qvel)
        face, normal = self.kin.face(q)
        late = (self.latency > 0)[:, None]
        sp, sv = torch.where(late, self.seen_pos, bp), torch.where(late, self.seen_vel, bv)
        self.seen_pos, self.seen_vel = bp.clone(), bv.clone()
        q_obs, qd_obs = q - self.q0, qd
        if R:
            n = lambda s, sig: torch.randn(s, device=self.device, generator=self.gen) * sig
            sp, sv = sp + n(sp.shape, R.ball_pos_noise), sv + n(sv.shape, R.ball_vel_noise)
            q_obs, qd_obs = q_obs + n(q.shape, R.joint_pos_noise), qd_obs + n(q.shape, R.joint_vel_noise)
        home = torch.tensor([self.home[0], self.home[1], self.face0_z], dtype=torch.float32, device=self.device)
        period = torch.full((self.num_envs, 1), self.dt * 30.0, device=self.device)
        return torch.cat([q_obs, self.target - self.q0, qd_obs, sp - face, sv, normal, face - home, self.prev_action, period], dim=1)

    def step(self, actions: torch.Tensor):
        c, R, dev, N = self.cfg, self.cfg.rand, self.device, self.num_envs
        a = actions.detach().to(dev, torch.float32).clamp(-1.0, 1.0)
        q_now = self._arm(self._q)
        want = (self.target + a * c.max_joint_speed * self.dt).clamp(self.lower, self.upper)
        self.target = self.guard.clamp(want, q_now)
        ctrl = np.ascontiguousarray(self.target.cpu().numpy()[:, self.act_order].astype(np.float32))

        qpos, qvel = self._q, self._qd
        # Sideways nudges to the ball in flight.
        if R and R.push_rate_hz > 0:
            bp, _ = self._ball(qpos, qvel)
            face, _ = self.kin.face(self._arm(qpos))
            free = (bp - face).norm(dim=1) > 0.06
            push = (torch.rand(N, device=dev, generator=self.gen) < R.push_rate_hz * self.dt) & free
            if push.any():
                qvel = qvel.clone()
                qvel[push, self.ball_v : self.ball_v + 2] += torch.randn(int(push.sum()), 2, device=dev, generator=self.gen) * R.push_speed
                self._sim.set_state(qpos.cpu().numpy(), qvel.cpu().numpy())

        reward = torch.zeros(N, device=dev)
        active = self.cause == 0
        g = torch.tensor([0.0, 0.0, -self.gravity], device=dev)
        t_chunk = 0.0
        for n in self.chunk_steps:
            _, v0 = self._ball(qpos, qvel)
            self._sim.step(ctrl, n)
            qpos_np, qvel_np = self._sim.state()
            qpos, qvel = self._to_t(qpos_np), self._to_t(qvel_np)
            dtc = n * self.timestep
            t_chunk += dtc
            now = self.sim_time + t_chunk
            bp, v1 = self._ball(qpos, qvel)
            face, normal = self.kin.face(self._arm(qpos))
            dv = v1 - (v0 + g * dtc)
            contact = (dv.norm(dim=1) > CONTACT_DV) & active
            rel = bp - face
            h = (rel * normal).sum(1)
            lateral = (rel - h[:, None] * normal).norm(dim=1)
            on_paddle = contact & (lateral < self.blade_r + 0.015) & (h.abs() < self.r + 0.04)
            on_table = contact & ~on_paddle & (bp[:, 2] < self.r + 0.05)
            on_arm = contact & ~on_paddle & ~on_table
            self.cause = torch.where(on_table & (self.cause == 0), CAUSE_DROPPED, self.cause)
            self.cause = torch.where(on_arm & (self.cause == 0), CAUSE_ARM, self.cause)

            nc = dv / dv.norm(dim=1, keepdim=True).clamp_min(1e-9)
            align = (nc * normal).sum(1).abs()
            quality = ((align - 0.707) / (1.0 - 0.707)).clamp(0, 1)
            scores = on_paddle & self.armed & (quality > 0)
            flight, rise = self._score_flight(now, bp[:, :2])
            reward = reward + torch.where(scores, c.centre * torch.exp(-((lateral / c.centre_sigma) ** 2)) + flight, torch.zeros_like(flight))
            took = scores & self.has_takeoff
            self.flight_apex += torch.where(took, rise, torch.zeros_like(rise))
            self.flight_n += took.float()
            self.last_home = torch.where(scores, (bp[:, :2] - torch.as_tensor(self.home, dtype=torch.float32, device=dev)).norm(dim=1), self.last_home)
            self.bounces += scores.long()
            self.since_bounce = torch.where(scores, torch.zeros_like(self.since_bounce), self.since_bounce)
            self.armed = self.armed & ~scores

            if on_paddle.any():  # takeoff: scatter the direction, start a new flight
                if R and R.takeoff_scatter_deg > 0:
                    v1 = torch.where(on_paddle[:, None], self._scatter(v1, R.takeoff_scatter_deg), v1)
                    qvel = qvel.clone()
                    qvel[:, self.ball_v : self.ball_v + 3] = v1
                    self._sim.set_state(qpos.cpu().numpy(), qvel.cpu().numpy())
                off_i, off_v = self._takeoff_angles(bp, v1)
                self.to_t0 = torch.where(on_paddle, now, self.to_t0)
                self.to_facez = torch.where(on_paddle, face[:, 2], self.to_facez)
                self.to_off_ideal = torch.where(on_paddle, off_i, self.to_off_ideal)
                self.to_off_vert = torch.where(on_paddle, off_v, self.to_off_vert)
                self.has_takeoff = self.has_takeoff | on_paddle
                self.peak = torch.where(on_paddle, torch.full_like(self.peak, -1e3), self.peak)
                self.apex_z = torch.where(on_paddle, torch.full_like(self.apex_z, -1e3), self.apex_z)
            self.peak = torch.maximum(self.peak, bp[:, 2] - self.r - face[:, 2])
            self.apex_z = torch.maximum(self.apex_z, bp[:, 2])
            self.armed = self.armed | ((self.peak > c.min_rise) & ~on_paddle)
            active = active & (self.cause == 0)

        self.sim_time += t_chunk
        qd_arm = qvel[:, self.qvel_adr]
        reward = reward + torch.where(self.cause == 0, torch.full_like(reward, c.alive * self.dt), torch.zeros_like(reward))
        reward = reward - c.action_rate * ((a - self.prev_action) ** 2).sum(1) - c.joint_velocity * (qd_arm**2).sum(1)
        self.prev_action = a
        self.episode_length_buf += 1
        self.since_bounce += self.dt

        bp, _ = self._ball(qpos, qvel)
        self.cause = torch.where((self.cause == 0) & (bp[:, :2].norm(dim=1) > c.max_reach), CAUSE_FLEW, self.cause)
        self.cause = torch.where((self.cause == 0) & (bp[:, 2] < self.r + 0.02), CAUSE_DROPPED, self.cause)
        self.cause = torch.where((self.cause == 0) & (self.since_bounce > c.stall_s), CAUSE_STALLED, self.cause)
        terminated = self.cause != 0
        truncated = ~terminated & (self.episode_length_buf >= self.max_episode_length)
        dones = terminated | truncated
        reward = torch.nan_to_num(reward, nan=0.0).clamp(-50.0, 50.0)

        self._q, self._qd = qpos, qvel
        log = self._log(dones, terminated)
        done_np = dones.cpu().numpy()
        self.finished_bounces = self.bounces[dones].cpu().numpy()
        self.finished_cause = self.cause[dones].cpu().numpy()
        if done_np.any():
            if self.curriculum:
                self._update_curriculum(self.bounces[dones].cpu().numpy(), self.spawn_dist[dones].cpu().numpy())
            qpos_np, qvel_np = qpos.cpu().numpy().copy(), qvel.cpu().numpy().copy()
            self._reset(done_np, qpos_np, qvel_np)
            self._sim.set_state(qpos_np, qvel_np)
            qpos, qvel = self._to_t(qpos_np), self._to_t(qvel_np)
            self._q, self._qd = qpos, qvel
        obs = self._obs(qpos, qvel)
        extras = {"time_outs": truncated, "log": log}
        return self._pack(obs), reward, dones.long(), extras

    def _log(self, dones: torch.Tensor, terminated: torch.Tensor) -> dict:
        if not dones.any():
            return {"stage": float(self.stage)}
        d = dones
        out = {
            "stage": float(self.stage),
            "bounces": float(self.bounces[d].float().mean()),
            "episode_s": float(self.episode_length_buf[d].float().mean() * self.dt),
        }
        for k, name in enumerate(CAUSES[1:], start=1):
            out["end/" + name.replace(" ", "_")] = float(((self.cause == k) & d).float().sum() / d.sum())
        out["end/timeout"] = float((~terminated & d).float().sum() / d.sum())
        has = d & (self.flight_n > 0)
        if has.any():
            out["apex_m"] = float((self.flight_apex[has] / self.flight_n[has]).mean())
        got = d & (self.bounces > 0)
        if got.any():
            out["home_dist_m"] = float(self.last_home[got].mean())
        return out

    def close(self) -> None:
        pass
