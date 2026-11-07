#!/usr/bin/env python3
"""前進専用G1方策の評価: 観測ノイズありで全envを20秒走らせ、転倒率・前進速度(yaw座標系)・横ずれ・yaw速度を出す。"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))]  # このディレクトリ、examples/(common.py)、scripts/(mjmlx_sensors.py等)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--solver-iters", type=int, default=4)
    ap.add_argument("--fwd-weights", type=str, default="")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from rsl_rl.runners import OnPolicyRunner

    import mjmlx_g1_env as e
    from train_g1_ppo import make_g1_train_cfg

    env = e.MjmlxG1Env(native, args.envs, use_gpu=True, seed=1, forward_only=True, solver_iterations=args.solver_iters,
                       fwd_weights={k: float(v) for k, v in (kv.split("=") for kv in args.fwd_weights.split(",") if kv)} or None)
    runner = OnPolicyRunner(env, make_g1_train_cfg(0, 1, "g1_forward"), log_dir=None, device="cpu")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cpu")
    obs, _ = env.reset()
    q0, _ = env._sim.state()
    pos0, yaw0 = q0[:, :2].copy(), e._yaw_from_quat(torch.from_numpy(q0[:, 3:7].copy())).numpy()
    alive = np.ones(args.envs, bool)
    fell_at = np.full(args.envs, np.nan)
    vx, vy, wz, cmd, yaws = [], [], [], [], []
    with torch.no_grad():
        for t in range(e.MAX_EPISODE_STEPS):
            obs, _, dones, extras = env.step(policy(obs))
            q, v = (torch.from_numpy(a) for a in env._sim.state())
            vel = e._yaw_frame_vel(q[:, 3:7], v[:, 0:3])
            if t == e.MAX_EPISODE_STEPS - 2:
                q_end = q.numpy().copy()  # 最後のstepは時間切れでリセットされるのでその1つ前の状態を終了時とする
            yaws.append(e._yaw_from_quat(q[:, 3:7]).numpy())
            vx.append(vel[:, 0].numpy()), vy.append(vel[:, 1].numpy()), wz.append(e._rotate(q[:, 3:7], v[:, 3:6])[:, 2].numpy())
            cmd.append(env._commands[:, 0].numpy().copy())
            d = dones.numpy().astype(bool)
            fell_at[d & alive & (extras["time_outs"].numpy() == 0)] = t * e.CONTROL_DT
            alive &= ~d
    q1, ok = q_end, ~np.isfinite(fell_at)  # 転倒しなかったenvの終了時の位置・向き
    yaw1 = e._yaw_from_quat(torch.from_numpy(q1[:, 3:7].copy())).numpy()
    yaw_path = np.unwrap(np.concatenate([yaw0[None], np.array(yaws[:-1])]), axis=0)  # 巻き戻しを展開して累積の向きの変化を取る(周回すると360度を超える)
    dyaw = np.degrees(yaw_path[-1] - yaw_path[0])[ok]
    disp = (q1[:, :2] - pos0)[ok]
    fwd = np.stack([np.cos(yaw0), np.sin(yaw0)], axis=1)[ok]
    along = (disp * fwd).sum(1)
    cross = disp[:, 0] * fwd[:, 1] * -1 + disp[:, 1] * fwd[:, 0]  # 開始時の向きに対する横方向の変位
    course = np.degrees(np.arctan2(cross, along))  # 進行方向(始点→終点)と開始時の向きのなす角
    warm = int(2.0 / e.CONTROL_DT)
    vx, vy, wz = (np.array(a)[warm:] for a in (vx, vy, wz))
    print(f"fell within 20 s: {np.isfinite(fell_at).mean():.3f}  (median fall time {np.nanmedian(fell_at) if np.isfinite(fell_at).any() else 'none'})")
    print(f"20 s accumulated heading change [deg]: mean {dyaw.mean():.1f}, mean |.| {np.abs(dyaw).mean():.1f}; displacement along {along.mean():.2f} m, cross-track mean |.| {np.abs(cross).mean():.2f} m, course angle mean |.| {np.abs(course).mean():.1f} deg")
    print(f"mean vx {vx.mean():.3f} (mean cmd {np.mean(cmd):.3f})  mean vy {vy.mean():.3f}  mean wz {wz.mean():.3f}  mean |wz| {np.abs(wz).mean():.3f}")


if __name__ == "__main__":
    main()
