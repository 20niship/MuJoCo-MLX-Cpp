#!/usr/bin/env python3
"""G1のCPU(MuJoCo C)とGPU(Vulkan)物理の1制御step後の差を、接触を含む多様な状態で測る(compare_cpu_gpu_go2.pyのG1版)。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--warm", type=int, default=40, help="状態を集めるためにGPU envを走らせるstep数")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--action-std", type=float, default=0.5)
    ap.add_argument("--lift", type=float, default=0.0, help="胴体を持ち上げて接触を無くした状態で比べる[m]")
    ap.add_argument("--max-condim", type=int, default=0)
    ap.add_argument("--self-collision", action="store_true")
    ap.add_argument("--solver-iters", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    import mjmlx_g1_env as e

    kw = dict(seed=0, obs_noise=False, max_condim=args.max_condim, self_collision=args.self_collision)
    if args.solver_iters is not None:
        kw["solver_iterations"] = args.solver_iters
    gpu = e.MjmlxG1Env(native, args.envs, use_gpu=True, generated_xml_path="/tmp/g1_cmp_gpu.xml", **kw)
    cpu = e.MjmlxG1Env(native, args.envs, use_gpu=False, generated_xml_path="/tmp/g1_cmp_cpu.xml", **kw)
    rng = np.random.default_rng(1)
    gpu.reset()
    cpu.reset()
    lo, hi = gpu._joint_lo, gpu._joint_hi

    def act():
        return torch.from_numpy(rng.normal(0, args.action_std, (args.envs, e.NJ)).astype(np.float32))

    for _ in range(args.warm):
        gpu.step(act())
    dq, dvl, dva, ncon = [], [], [], []
    for _ in range(args.samples):
        q, v = gpu._sim.state()
        q, v = q.copy(), v.copy()
        a = act()
        q[:, 2] += args.lift
        cpu._sim.set_state(q, v)
        gpu._sim.set_state(q, v)
        ctrl = np.clip(e.DEFAULT_JOINT_POS + e.ACTION_SCALE * a.numpy(), lo, hi)
        for _ in range(e.DECIMATION):
            cpu._sim.step(ctrl)
            gpu._sim.step(ctrl)
        qc, vc = cpu._sim.state()
        qg, vg = gpu._sim.state()
        dq.append(np.abs(qc - qg)[:, 7:].max(axis=1))
        dvl.append(np.abs(vc - vg)[:, :3].max(axis=1))
        dva.append(np.abs(vc - vg)[:, 3:6].max(axis=1))
        for _ in range(10):
            gpu.step(act())
    dq, dvl, dva = np.concatenate(dq), np.concatenate(dvl), np.concatenate(dva)
    print(f"envs x samples={dq.size}  joint-pos diff[rad]: mean={dq.mean():.2e} p95={np.percentile(dq, 95):.2e} max={dq.max():.2e}")
    print(f"                       base lin-vel diff[m/s]: mean={dvl.mean():.2e} p95={np.percentile(dvl, 95):.2e}   ang-vel diff[rad/s]: mean={dva.mean():.2e} p95={np.percentile(dva, 95):.2e}")


if __name__ == "__main__":
    main()
