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
