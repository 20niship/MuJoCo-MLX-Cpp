#!/usr/bin/env python3
"""MuJoCo Warp(NVIDIA GPU上のMuJoCo)でGo2(平地)を並列envで動かし、速度を測る。JSONを1行(MJWARP_RESULT ...)出力する。

別venvで実行する(mujoco-warpとwarp-langだけ入れる。ドライバがCUDA 12.xならwarp-lang==1.17.0に固定):
  uv venv --python 3.12 ~/mjwarp/.venv && VIRTUAL_ENV=~/mjwarp/.venv uv pip install mujoco-warp "warp-lang==1.17.0"
  ~/mjwarp/.venv/bin/python scripts/mjwarp_bench.py --num_envs 4096

物理だけ(観測・報酬なし)。制御1step = dt 0.005 x 4 substeps + ランダムaction(位置制御)。genesis_bench.py と同じ条件。
--iterations でソルバの反復数を指定(省略はgo2_mjx.xml既定の1)。同じモデル(menagerie go2_mjx + 平地)をmjmlxも読む。
"""

import argparse
import json
import os
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(HERE, "..", "benchmarks", "models", "external", "mujoco_menagerie", "unitree_go2")
DT, DECIMATION = 0.005, 4

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iterations", type=int, default=None, help="ソルバ反復数(省略: xml既定)")
p.add_argument("--ls_iterations", type=int, default=None)
p.add_argument("--solver", choices=["newton", "cg"], default=None)
p.add_argument("--nconmax", type=int, default=16, help="envあたりの接触数の確保量")
p.add_argument("--njmax", type=int, default=64, help="envあたりの拘束数の確保量")
args = p.parse_args()

mjm = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, "scene_mjx.xml"))
mjm.opt.timestep = DT
if args.iterations is not None:
    mjm.opt.iterations = args.iterations
if args.ls_iterations is not None:
    mjm.opt.ls_iterations = args.ls_iterations
if args.solver:
    mjm.opt.solver = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG}[args.solver]
mjd = mujoco.MjData(mjm)
mujoco.mj_resetDataKeyframe(mjm, mjd, 0)  # home
mujoco.mj_forward(mjm, mjd)

m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=args.num_envs, nconmax=args.nconmax, njmax=args.njmax)
default = mjm.key_ctrl[0].astype(np.float32)

N, nu = args.num_envs, mjm.nu
rng = np.random.default_rng(0)
lo, hi = mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1]


POOL = [np.clip(default + 0.5 * rng.standard_normal((N, nu)), lo, hi).astype(np.float32) for _ in range(4)]  # mjmlxのベンチと同じく入力を使い回す


def new_ctrl(i=0):
    return POOL[i % 4]


ctrl_wp = d.ctrl


def control_step(c):
    ctrl_wp.assign(c)
    for _ in range(DECIMATION):
        mjw.step(m, d)


for i in range(args.warmup):
    control_step(new_ctrl(i))
wp.synchronize()
t = []
for i in range(args.iters):
    c = new_ctrl(i)
    wp.synchronize()
    t0 = time.perf_counter()
    control_step(c)
    wp.synchronize()
    t.append(time.perf_counter() - t0)
qpos = d.qpos.numpy()
res = {
    "num_envs": N,
    "physics_step_s": sum(t) / len(t),
    "physics_step_median_s": sorted(t)[len(t) // 2],
    "solver": mujoco.mjtSolver(mjm.opt.solver).name,
    "iterations": int(mjm.opt.iterations),
    "ls_iterations": int(mjm.opt.ls_iterations),
    "finite": bool(np.isfinite(qpos).all()),
    "nefc_max": int(d.nefc.numpy().max()),
    "nacon": int(d.nacon.numpy().max()),
    "base_z_mean": float(qpos[:, 2].mean()),
    "mujoco": mujoco.__version__,
    "warp": wp.config.version,
    "mujoco_warp": getattr(mjw, "__version__", "?"),
}
print("MJWARP_RESULT " + json.dumps(res), flush=True)
