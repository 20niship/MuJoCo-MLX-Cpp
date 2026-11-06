"""MJCFの<sensor>とbody接触力を、Isaac Labのセンサデータ(ImuData / ContactSensorData)と同じ名前・形・座標系で返す薄いラッパ。

quatはwxyz、`_w`は世界座標、`_b`はセンサ(site/body)座標。テンソルはtorch(deviceは指定可)。センサ値はMuJoCoと同じくstep開始時点の状態に対応する(1物理step遅れ)。
"""

from __future__ import annotations

import numpy as np
import torch


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """wxyzクォータニオンqの逆回転でvを座標変換する(世界->body)。"""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


class SensorHub:
    """モデルの<sensor>を名前で引く。read()で全センサを1回で取得して使い回す。"""

    def __init__(self, sim, model, device: str = "cpu"):
        self._sim = sim
        self.device = device
        self._slices = {name: (adr, adr + dim) for name, _type, adr, dim in model.sensors() if name}
        self._data: torch.Tensor | None = None

    @property
    def names(self) -> list[str]:
        return list(self._slices)

    def read(self) -> torch.Tensor:
        """(num_envs, nsensordata)。stepごとに1回呼ぶ。"""
        self._data = torch.from_numpy(self._sim.sensordata()).to(self.device)
        return self._data

    def __getitem__(self, name: str) -> torch.Tensor:
        if self._data is None:
            self.read()
        lo, hi = self._slices[name]
        return self._data[:, lo:hi]

    def __contains__(self, name: str) -> bool:
        return name in self._slices


class Imu:
    """Isaac LabのImuData相当。MJCFのsensor名を指定する(既定はmujoco_menagerie Go2の名前)。

    lin_acc_bは重力込みの固有加速度(MuJoCoのaccelerometer)。Isaac LabのImuと同じく速度の1step差分で求めるため、接触の衝撃では厳密値からずれる。
    """

    def __init__(self, hub: SensorHub, gyro="gyro", accelerometer="accelerometer", quat="orientation", pos="global_position",
                 lin_vel_w="global_linvel", velocimeter: str | None = None):
        self._hub = hub
        self._names = dict(gyro=gyro, acc=accelerometer, quat=quat, pos=pos, lin_vel_w=lin_vel_w, velocimeter=velocimeter)
        self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=hub.device)

    def _get(self, key: str) -> torch.Tensor:
        return self._hub[self._names[key]]

    @property
    def ang_vel_b(self) -> torch.Tensor:
        return self._get("gyro")

    @property
    def lin_acc_b(self) -> torch.Tensor:
        return self._get("acc")

    @property
    def quat_w(self) -> torch.Tensor:
        return self._get("quat")

    @property
    def pos_w(self) -> torch.Tensor:
        return self._get("pos")

    @property
    def lin_vel_b(self) -> torch.Tensor:
        if self._names["velocimeter"] is not None:
            return self._get("velocimeter")
        return quat_rotate_inverse(self.quat_w, self._get("lin_vel_w"))

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        return quat_rotate_inverse(self.quat_w, self._gravity_w.expand(self.quat_w.shape[0], 3))


class ContactSensor:
    """Isaac LabのContactSensor相当(bodyごとの正味接触力と接地/空中時間)。

    net_forces_w[N, num_bodies, 3]はMuJoCoのcfrc_ext(そのbodyが接触から受ける力)の力成分。update(dt)を制御stepごとに1回呼ぶ。
    """

    def __init__(self, sim, num_envs: int, body_ids, force_threshold: float = 1.0, device: str = "cpu"):
        self._sim = sim
        self.device = device
        self._body_ids = np.asarray(body_ids, dtype=np.int64)
        self.force_threshold = force_threshold
        n = len(self._body_ids)
        z = lambda: torch.zeros(num_envs, n, device=device)  # noqa: E731
        self.net_forces_w = torch.zeros(num_envs, n, 3, device=device)
        self.current_contact_time, self.current_air_time = z(), z()
        self.last_contact_time, self.last_air_time = z(), z()
        self.is_contact = torch.zeros(num_envs, n, dtype=torch.bool, device=device)

    def update(self, dt: float) -> None:
        cfrc = self._sim.cfrc_ext()
        cfrc = cfrc.reshape(cfrc.shape[0], -1, 6)
        self.net_forces_w = torch.from_numpy(np.ascontiguousarray(cfrc[:, self._body_ids, 3:6])).to(self.device)
        contact = self.net_forces_w.norm(dim=-1) > self.force_threshold
        first_contact = (self.current_air_time > 0) & contact
        first_air = (self.current_contact_time > 0) & ~contact
        self.last_air_time = torch.where(first_contact, self.current_air_time + dt, self.last_air_time)
        self.current_air_time = torch.where(~contact, self.current_air_time + dt, torch.zeros_like(self.current_air_time))
        self.last_contact_time = torch.where(first_air, self.current_contact_time + dt, self.last_contact_time)
        self.current_contact_time = torch.where(contact, self.current_contact_time + dt, torch.zeros_like(self.current_contact_time))
        self.is_contact = contact
        self._dt = dt

    def compute_first_contact(self, dt: float) -> torch.Tensor:
        """直近の制御stepで接地した瞬間のbody。"""
        return (self.current_contact_time > 0) & (self.current_contact_time < dt + 1e-6)

    def compute_first_air(self, dt: float) -> torch.Tensor:
        """直近の制御stepで離地した瞬間のbody。"""
        return (self.current_air_time > 0) & (self.current_air_time < dt + 1e-6)

    def reset(self, env_mask: torch.Tensor) -> None:
        m = env_mask.to(self.device)
        for t in (self.current_contact_time, self.current_air_time, self.last_contact_time, self.last_air_time):
            t[m] = 0.0
        self.net_forces_w[m] = 0.0
        self.is_contact[m] = False


def grid_pattern(size=(1.6, 1.0), resolution: float = 0.1) -> np.ndarray:
    """Isaac Labの GridPatternCfg 相当。センサ座標のxy格子(N, 2)。端を含む。"""
    xs = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
    ys = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """wxyzクォータニオンからyaw成分だけのクォータニオンを作る。"""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """wxyzクォータニオンqでvを回転する(body->世界)。vは(..., n, 3)、qは(..., 4)。"""
    w, xyz = q[..., :1], q[..., 1:]
    xyz, w = xyz.unsqueeze(-2), w.unsqueeze(-2)
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


class RayCaster:
    """Isaac Labの RayCaster / height scanner 相当。bodyに追従する格子状のレイをGPUで一括して撃つ。

    alignment: "yaw"(胴体のyawだけに追従、Isaac Labの既定。height scanner向け)/ "base"(姿勢に完全追従)/ "world"。
    sim, nativeは_mjmlx_rl_native.create_batched()の戻り値とモジュール。body_idは追従するbody(name2idで取得)。
    update()は現在の状態でレイを撃つ。pos/quatはstep時点のbody姿勢(1物理step遅れ)、geom側は常に最新のqposから再計算する。
    miss(max_distを超える/何も無い)は distance=inf、geomid=-1。geomgroup_maskは地形のgeom groupだけを撃つのに使う(例: 1<<0)。
    """

    def __init__(self, native, sim, body_id: int, pattern_xy: np.ndarray | None = None, offset_z: float = 0.0,
                 ray_dir=(0.0, 0.0, -1.0), alignment: str = "yaw", max_dist: float = 0.0, geomgroup_mask: int = -1,
                 bodyexclude: int = -1, flg_static: bool = True, device: str = "cpu"):
        if alignment not in ("yaw", "base", "world"):
            raise ValueError("alignment must be 'yaw', 'base' or 'world'")
        self._sim, self._body_id, self.alignment, self.device = sim, int(body_id), alignment, device
        self._rc = native.create_raycaster(sim)
        pat = grid_pattern() if pattern_xy is None else np.asarray(pattern_xy, np.float32)
        self.num_rays = len(pat)
        self._local = torch.from_numpy(np.concatenate([pat, np.full((self.num_rays, 1), offset_z, np.float32)], axis=1)).to(device)
        d = torch.tensor(ray_dir, dtype=torch.float32, device=device)
        self._dir_local = (d / d.norm()).expand(self.num_rays, 3).contiguous()
        self._kw = dict(max_dist=float(max_dist), bodyexclude=int(bodyexclude), flg_static=bool(flg_static), geomgroup_mask=int(geomgroup_mask))
        self.pos_w = self.ray_origins_w = self.ray_hits_w = self.distance = None
        self.geomid = None

    def update(self) -> None:
        n = self._sim.xpos().shape[0]
        pos = torch.from_numpy(self._sim.xpos().reshape(n, -1, 3)[:, self._body_id]).to(self.device)
        quat = torch.from_numpy(self._sim.xquat().reshape(n, -1, 4)[:, self._body_id]).to(self.device)
        if self.alignment == "yaw":
            q = _yaw_quat(quat)
        elif self.alignment == "base":
            q = quat
        else:
            q = torch.tensor([1.0, 0, 0, 0], device=self.device).expand(n, 4)
        local = self._local.expand(n, -1, -1)
        origins = pos.unsqueeze(1) + _quat_rotate(q, local)
        dirs = _quat_rotate(q, self._dir_local.expand(n, -1, -1)) if self.alignment == "base" else self._dir_local.expand(n, -1, -1)
        dist, gid = self._rc.cast(np.ascontiguousarray(origins.cpu().numpy(), np.float32), np.ascontiguousarray(dirs.cpu().numpy(), np.float32), **self._kw)
        dist = torch.from_numpy(dist).to(self.device)
        miss = dist < 0
        self.distance = torch.where(miss, torch.full_like(dist, float("inf")), dist)
        self.geomid = torch.from_numpy(gid).to(self.device)
        self.pos_w, self.ray_origins_w = pos, origins
        hits = origins + dirs * dist.clamp(min=0).unsqueeze(-1)
        self.ray_hits_w = torch.where(miss.unsqueeze(-1), torch.full_like(hits, float("inf")), hits)

    def height_scan(self, offset: float = 0.5) -> torch.Tensor:
        """Isaac Labの height_scan 観測: センサ高さ - ヒット点の高さ - offset (N, num_rays)。missはinfになるので呼び出し側でclipする。"""
        return self.pos_w[:, 2:3] - self.ray_hits_w[..., 2] - offset
