#!/usr/bin/env python3
"""前進専用G1方策の評価: 観測ノイズありで全envを20秒走らせ、転倒率・前進速度(yaw座標系)・横ずれ・yaw速度を出す。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--solver-iters", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from rsl_rl.runners import OnPolicyRunner

    import mjmlx_g1_env as e
    from train_g1_ppo import make_g1_train_cfg

    env = e.MjmlxG1Env(native, args.envs, use_gpu=True, seed=1, forward_only=True, solver_iterations=args.solver_iters)
    runner = OnPolicyRunner(env, make_g1_train_cfg(0, 1, "g1_forward"), log_dir=None, device="cpu")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cpu")
    obs, _ = env.reset()
    alive = np.ones(args.envs, bool)
    fell_at = np.full(args.envs, np.nan)
    vx, vy, wz, cmd = [], [], [], []
    with torch.no_grad():
        for t in range(e.MAX_EPISODE_STEPS):
            obs, _, dones, extras = env.step(policy(obs))
            q, v = (torch.from_numpy(a) for a in env._sim.state())
            vel = e._yaw_frame_vel(q[:, 3:7], v[:, 0:3])
            vx.append(vel[:, 0].numpy()), vy.append(vel[:, 1].numpy()), wz.append(e._rotate(q[:, 3:7], v[:, 3:6])[:, 2].numpy())
            cmd.append(env._commands[:, 0].numpy().copy())
            d = dones.numpy().astype(bool)
            fell_at[d & alive & (extras["time_outs"].numpy() == 0)] = t * e.CONTROL_DT
            alive &= ~d
    warm = int(2.0 / e.CONTROL_DT)
    vx, vy, wz = (np.array(a)[warm:] for a in (vx, vy, wz))
    print(f"fell within 20 s: {np.isfinite(fell_at).mean():.3f}  (median fall time {np.nanmedian(fell_at) if np.isfinite(fell_at).any() else 'none'})")
    print(f"mean vx {vx.mean():.3f} (mean cmd {np.mean(cmd):.3f})  mean vy {vy.mean():.3f}  mean wz {wz.mean():.3f}  mean |wz| {np.abs(wz).mean():.3f}")


if __name__ == "__main__":
    main()
