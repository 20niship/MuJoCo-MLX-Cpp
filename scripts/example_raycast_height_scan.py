#!/usr/bin/env python3
"""Go2の階段地形で、胴体に追従する格子状のレイ(height scanner)を全envで撃って地形の高さと比較する例。"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from mjmlx_go2_env import MjmlxGo2Env
    from mjmlx_sensors import RayCaster, grid_pattern

    env = MjmlxGo2Env(native, args.envs, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(args.steps):
        env.step(torch.from_numpy(rng.normal(0, 0.3, (args.envs, 12)).astype(np.float32)))

    # Isaac Labのheight scanner既定: 1.6m x 1.0m、0.1m間隔、胴体の上20cmから真下へ、yawだけ追従。地形(world body、group 0)だけを撃つ
    base = env._model.name2id(1, "base")  # mjOBJ_BODY = 1
    scanner = RayCaster(native, env._sim, base, grid_pattern((1.6, 1.0), 0.1), offset_z=0.2, alignment="yaw", geomgroup_mask=1 << 0)
    scanner.update()
    t0 = time.perf_counter()
    scanner.update()
    dt = time.perf_counter() - t0

    hs = scanner.height_scan(offset=0.5).clamp(-1.0, 1.0)
    print(f"envs={args.envs} rays/env={scanner.num_rays} update={dt * 1e3:.2f}ms (kinematics + cast + host copies)")
    print(f"height_scan obs {tuple(hs.shape)}: min={hs.min():.3f} max={hs.max():.3f} (NaN: {torch.isnan(hs).any().item()})")

    # 地形の天面高さ(StairsTerrain)と、ヒット点のz・xを突き合わせる
    hits = scanner.ray_hits_w
    expect = env.terrain.max_height_within(hits[..., 0].flatten().numpy().astype(np.float64), 0.0).reshape(hits.shape[:2])
    ok = torch.isfinite(hits[..., 2])
    err = (hits[..., 2].numpy() - expect)[ok.numpy()]
    print(f"hit z vs terrain top: mean|err|={np.abs(err).mean():.4f} m, max|err|={np.abs(err).max():.4f} m over {ok.sum().item()} hits")


if __name__ == "__main__":
    main()
