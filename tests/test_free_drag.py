"""自由関節ボールの二乗空気抵抗(set_free_drag)がMuJoCo(xfrc_applied)と一致するか。usage: test_free_drag.py <build_dir>"""

import os
import sys

sys.path.insert(0, sys.argv[1])
import mujoco
import numpy as np
import vkmujoco

X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ball_after_arm.xml")
m = mujoco.MjModel.from_xml_path(X)
d = mujoco.MjData(m)
sim = vkmujoco.create_batched(vkmujoco.load_model(X), 3, use_gpu=True, domain_rand=True)
assert sim.num_free_joints == 1

k = np.array([0.0, 3.54e-4, 3.54e-3], np.float32)  # 抵抗なし / 実ボール相当 / 10倍
sim.set_free_drag(drag_k=k[:, None])
ba, bv, bid = m.jnt_qposadr[m.njnt - 1], m.jnt_dofadr[m.njnt - 1], m.jnt_bodyid[m.njnt - 1]
q0 = m.qpos0.copy()
q0[ba : ba + 3] = [0.2, 0.2, 2.0]
v0 = np.zeros(m.nv)
v0[bv : bv + 3] = [8.0, 0.0, 6.0]
for i in range(3):
    sim.set_env_qpos(i, q0.astype(np.float32))
    sim.set_env_qvel(i, v0.astype(np.float32))
sim.step(np.zeros((3, m.nu), np.float32), 1000)
q, v = sim.qpos(), sim.qvel()

for i, kk in enumerate(k):
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_forward(m, d)
    for _ in range(1000):
        vel = d.qvel[bv : bv + 3]
        d.xfrc_applied[bid, :3] = -kk * np.linalg.norm(vel) * vel
        mujoco.mj_step(m, d)
    dp = np.abs(q[i, ba : ba + 3] - d.qpos[ba : ba + 3]).max()
    dv = np.abs(v[i, bv : bv + 3] - d.qvel[bv : bv + 3]).max()
    print(f"k={kk:.2e}: max|dpos|={dp:.2e} max|dvel|={dv:.2e}")
    assert dp < 5e-3 and dv < 5e-3, f"drag mismatch for k={kk}"
print("free_drag OK")
