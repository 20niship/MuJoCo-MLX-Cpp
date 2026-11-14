"""batchedのレイキャスト(sim.raycast)がmj_rayと一致するか。usage: test_raycast.py <build_dir>"""

import os
import sys

sys.path.insert(0, sys.argv[1])
import mujoco
import numpy as np
import vkmujoco

X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "raycast.xml")
m = mujoco.MjModel.from_xml_path(X)
d = mujoco.MjData(m)
B, R = 4, 400
rng = np.random.default_rng(0)
sim = vkmujoco.create_batched(vkmujoco.load_model(X), B, use_gpu=True)

# 各envで物体の位置と向きをばらつかせる(rayの始点は物体より十分高く、内側から始まるrayは作らない)
q = np.tile(m.qpos0, (B, 1)).astype(np.float32)
for e in range(B):
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        q[e, a : a + 3] += rng.uniform(-0.15, 0.15, 3) * [1, 1, 0.5]
        quat = rng.normal(size=4)
        q[e, a + 3 : a + 7] = quat / np.linalg.norm(quat)
sim.set_state(q, np.zeros((B, m.nv), np.float32))

pnt = np.stack([np.c_[rng.uniform(-1.8, 1.8, R), rng.uniform(-0.9, 0.9, R), rng.uniform(1.0, 1.5, R)] for _ in range(B)]).astype(np.float32)
target = np.stack([np.c_[rng.uniform(-1.5, 1.5, R), rng.uniform(-0.8, 0.8, R), rng.uniform(-0.1, 0.5, R)] for _ in range(B)]).astype(np.float32)
vec = target - pnt
vec /= np.linalg.norm(vec, axis=2, keepdims=True)


def reference(e, bodyexclude=-1, geomgroup=None, flg_static=1):
    d.qpos[:] = q[e]
    mujoco.mj_forward(m, d)
    dist, gid = np.full(R, -1.0), np.full(R, -1)
    for r in range(R):
        g = np.zeros(1, np.int32)
        t = mujoco.mj_ray(m, d, pnt[e, r].astype(np.float64), vec[e, r].astype(np.float64), geomgroup, flg_static, bodyexclude, g)
        dist[r], gid[r] = t, g[0]
    return dist, gid


def check(name, **kw):
    dist, gid = sim.raycast(pnt, vec, **({"bodyexclude": kw.get("bodyexclude", -1)} | ({"geomgroup": kw["geomgroup"]} if "geomgroup" in kw else {}) | {"flg_static": kw.get("flg_static", True)}))
    bad = 0
    for e in range(B):
        rd, rg = reference(e, kw.get("bodyexclude", -1), kw.get("geomgroup"), int(kw.get("flg_static", True)))
        hit = rd >= 0
        assert np.array_equal(dist[e] >= 0, hit), f"{name}: hit/miss differs in env {e}: {np.sum((dist[e] >= 0) != hit)} rays"
        assert np.abs(dist[e][hit] - rd[hit]).max() < 1e-3, f"{name}: distance differs in env {e}: {np.abs(dist[e][hit] - rd[hit]).max()}"
        assert np.array_equal(gid[e][hit], rg[hit]), f"{name}: geom id differs in env {e}"
        bad += int(np.sum(hit))
    print(f"{name}: {bad}/{B * R} rays hit, all match mj_ray")
    return bad


assert check("all geoms") > B * R // 4
# 形状ごとに当たっていること(テストが何も当てずに通らないように)
_, gid = sim.raycast(pnt, vec)
for name in ("floor", "sphere", "capsule", "cylinder", "ellipsoid", "box", "wall"):
    assert np.any(gid == m.geom(name).id), f"no ray hit {name}"
check("bodyexclude=sphere", bodyexclude=m.body("sphere").id)
check("geomgroup 0 only", geomgroup=np.array([1, 0, 0, 0, 0, 0], np.uint8))
check("no static", flg_static=False)
# (nray, 3)の共通rayでも同じ
d0, g0 = sim.raycast(pnt[0], vec[0])
assert d0.shape == (B, R) and np.array_equal(d0[0] >= 0, sim.raycast(pnt, vec)[0][0] >= 0)
print("ok")
