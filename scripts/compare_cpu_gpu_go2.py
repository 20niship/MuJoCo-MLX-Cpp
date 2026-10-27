#!/usr/bin/env python3
"""Go2のCPU(MuJoCo C)とGPU(Vulkan)物理の1制御step後の差を、接触を含む多様な状態で測る。

GPUのenvをランダムactionで走らせて集めた状態を、CPU/GPU両方に同じに設定し、同じactionで1制御step(decimation分のsubstep)進めて比較する。
接触解の差は長い軌道ではカオス的に増幅されるので、1 stepの差で評価する。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--warm", type=int, default=60, help="状態を集めるためにGPU envを走らせるstep数")
    ap.add_argument("--samples", type=int, default=5, help="比較する状態の数(warmの後、10stepごと)")
    ap.add_argument("--preset", default="isaac_flat")
    ap.add_argument("--solver-iters", type=int, default=None)
    ap.add_argument("--lift", type=float, default=0.0, help="胴体を持ち上げて接触を無くした状態で比べる(物理のうち接触以外の差を見る)[m]")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    import mjmlx_go2_env as e

    if args.solver_iters is not None:
        e.SOLVER_ITERATIONS = args.solver_iters
    kw = dict(cmd_vel_min=0.2, cmd_vel_max=1.0, seed=0, terrain_seed=0, flat_terrain=True, reward_preset=args.preset)
    gpu = e.MjmlxGo2Env(native, args.envs, use_gpu=True, generated_xml_path="/tmp/go2_cmp_gpu.xml", **kw)
    cpu = e.MjmlxGo2Env(native, args.envs, use_gpu=False, generated_xml_path="/tmp/go2_cmp_cpu.xml", **kw)
    rng = np.random.default_rng(1)
    gpu.reset()
    cpu.reset()

    def act():
        return torch.from_numpy(rng.normal(0, 0.5, (args.envs, 12)).astype(np.float32))

    for _ in range(args.warm):
        gpu.step(act())
    dq, dv, ncont, dvl, dva = [], [], [], [], []
    for s in range(args.samples):
        q, v = gpu._sim.state()
        q, v = q.copy(), v.copy()
        a = act()
        q[:, 2] += args.lift
        cpu._sim.set_state(q, v)
        gpu._sim.set_state(q, v)
        ctrl = np.clip(e.DEFAULT_JOINT_POS + e.ACTION_SCALE * a.numpy(), e.CTRL_LOW, e.CTRL_HIGH)
        for _ in range(e.DECIMATION):
            cpu._sim.step(ctrl)
            gpu._sim.step(ctrl)
        qc, vc = cpu._sim.state()
        qg, vg = gpu._sim.state()
        dq.append(np.abs(qc - qg)[:, 7:19].max(axis=1))   # 関節角の最大差
        dv.append(np.abs(vc - vg)[:, :6].max(axis=1))     # 胴体速度の最大差
        dvl.append(np.abs(vc - vg)[:, :3].max(axis=1)); dva.append(np.abs(vc - vg)[:, 3:6].max(axis=1))
        for _ in range(10):
            gpu.step(act())
    dq, dv, dvl, dva = np.concatenate(dq), np.concatenate(dv), np.concatenate(dvl), np.concatenate(dva)
    print(f"envs x samples={dq.size}  joint-pos diff[rad]: mean={dq.mean():.2e} p95={np.percentile(dq, 95):.2e} max={dq.max():.2e}")
    print(f"                       base lin-vel diff[m/s]: mean={dvl.mean():.2e} p95={np.percentile(dvl, 95):.2e}   ang-vel diff[rad/s]: mean={dva.mean():.2e} p95={np.percentile(dva, 95):.2e}")
    print(f"                       base-vel diff[m/s,rad/s]: mean={dv.mean():.2e} p95={np.percentile(dv, 95):.2e} max={dv.max():.2e}")


if __name__ == "__main__":
    main()
