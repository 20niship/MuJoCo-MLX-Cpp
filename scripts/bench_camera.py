#!/usr/bin/env python3
"""Go2のカメラ描画時間(解像度 x env数)と物理stepに対する相対コストを計測する。

usage: bench_camera.py [--build-dir build] [--envs 256 1024 4096] [--sizes 64 128]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MJMLX_GO2_XML_SUBST"] = (
    '<camera name="track"->'
    '<camera name="orbit" mode="targetbody" target="base" pos="-1.4 -1.4 0.7" fovy="45"/><camera name="track";'
    "<worldbody>-><worldbody>"
    '<light pos="0 0 4" dir="0.3 0.2 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>'
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from mjmlx_go2_env import MjmlxGo2Env

    print("envs size | step ms | render ms (GPU) | readback ms | render+readback ms | env-frames/s | render/step")
    for n in args.envs:
        env = MjmlxGo2Env(native, n, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
        env.reset()
        rng = np.random.default_rng(0)
        act = lambda: torch.from_numpy(rng.normal(0, 0.5, (n, 12)).astype(np.float32))
        for _ in range(3):
            env.step(act())
        ts = []
        for _ in range(args.iters):
            a = act()
            t0 = time.perf_counter()
            env.step(a)
            ts.append(time.perf_counter() - t0)
        step_ms = np.mean(ts) * 1e3
        for size in args.sizes:
            cam = native.create_camera(env._sim, size, size)
            cam.render()
            cam.rgb(), cam.depth()
            tr, tb = [], []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                cam.render()
                t1 = time.perf_counter()
                cam.rgb(), cam.depth()
                t2 = time.perf_counter()
                tr.append(t1 - t0)
                tb.append(t2 - t1)
            r_ms, b_ms = np.mean(tr) * 1e3, np.mean(tb) * 1e3
            print(f"{n:5d} {size:4d} | {step_ms:7.1f} | {r_ms:8.1f} | {b_ms:8.1f} | {r_ms + b_ms:8.1f} | {n / ((r_ms + b_ms) / 1e3):9.0f} | {r_ms / step_ms:.2f}")
            del cam


if __name__ == "__main__":
    main()
