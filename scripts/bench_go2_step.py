#!/usr/bin/env python3
"""Go2 env.step(物理5substep+報酬)の1stepあたり時間を計測する。MJMLX_PROF=1 を付けると段ごとの内訳も出る。

高速化前(PR #33 のb0a2367より前)の実測: 4096env, 1.68s/step。before/afterの比較は
`--build-dir` に別コミットのビルドを指定して同じ引数で実行する。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--solver-iters", type=int, default=None, help="Newton反復数の上書き(既定はenvの値)")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from mjmlx_go2_env import MjmlxGo2Env

    if args.solver_iters is not None:
        import mjmlx_go2_env
        mjmlx_go2_env.SOLVER_ITERATIONS = args.solver_iters
    env = MjmlxGo2Env(native, args.envs, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
    env.reset()
    rng = np.random.default_rng(0)
    act = lambda: torch.from_numpy(rng.normal(0, 0.5, (args.envs, 12)).astype(np.float32))
    for _ in range(args.warmup):
        env.step(act())
    t = []
    for _ in range(args.steps):
        a = act()
        t0 = time.perf_counter()
        env.step(a)
        t.append(time.perf_counter() - t0)
    t = np.array(t)
    print(f"{args.build_dir}: envs={args.envs} step mean={t.mean()*1e3:.0f}ms min={t.min()*1e3:.0f}ms "
          f"-> {args.envs / t.mean():.0f} env-steps/s")


if __name__ == "__main__":
    main()
