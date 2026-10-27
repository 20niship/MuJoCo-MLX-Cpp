#!/usr/bin/env python3
"""Go2学習envと同じsimの上で、全envのカメラRGB/depthを1ドローコールで取得して保存する例。

usage: example_go2_camera.py [--build-dir build] [--envs 16] [--size 128] [--out runs/go2_camera]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Go2のMJCFに前方カメラ(base搭載)と平行光源を足す(学習envは環境変数のXML置換を読む)。
os.environ["MJMLX_GO2_XML_SUBST"] = (
    '<camera name="track"->'
    '<camera name="front" pos="0.4 0 0.15" xyaxes="0 -1 0 0.3 0 1" fovy="90"/>'
    '<camera name="orbit" mode="targetbody" target="base" pos="-1.4 -1.4 0.7" fovy="45"/><camera name="track";'
    "<worldbody>-><worldbody>"
    '<light pos="0 0 4" dir="0.3 0.2 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>'
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="runs/go2_camera")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from PIL import Image
    from mjmlx_go2_env import MjmlxGo2Env

    env = MjmlxGo2Env(native, args.envs, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(args.steps):
        env.step(torch.from_numpy(rng.normal(0, 0.5, (args.envs, 12)).astype(np.float32)))

    cam = native.create_camera(env._sim, args.size, args.size)
    t0 = time.perf_counter()
    cam.render()
    rgb, depth = cam.rgb(), cam.depth()
    dt = time.perf_counter() - t0
    print(f"envs={args.envs} cameras={cam.num_cameras} draw_calls={cam.draw_calls} rgb={rgb.shape} depth={depth.shape} {dt * 1e3:.1f}ms")

    os.makedirs(args.out, exist_ok=True)
    n = min(args.envs, 8)
    for c in range(cam.num_cameras):
        Image.fromarray(np.concatenate(list(rgb[:n, c]), axis=1)).save(f"{args.out}/rgb_cam{c}.png")
        d = depth[:n, c]
        d8 = (255 * np.clip(d / max(float(d[d < 99].max(initial=1.0)), 1e-3), 0, 1)).astype(np.uint8)
        Image.fromarray(np.concatenate(list(d8), axis=1)).save(f"{args.out}/depth_cam{c}.png")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
