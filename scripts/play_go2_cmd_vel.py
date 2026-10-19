#!/usr/bin/env python3
"""学習済みGo2方策に平地で前後左右のcmd_velを与え、指令と実測の胴体速度をcaptionに載せた2x2タイル動画を書き出す。"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TILE_W, TILE_H = 480, 360
HOME_Z = 0.40  # 平地でのスポーン高さ(mjmlx_go2_env._reset_envsと同じ)
# (ラベル, 胴体座標系の指令[vx,vy,wz]の方向, 開始位置[x,y]): 通路は全幅3m・長さ約11.8mなので、移動方向の手前から出発する
CASES = [
    ("forward", (1, 0, 0), (3.0, 0.0)),
    ("backward", (-1, 0, 0), (8.8, 0.0)),
    ("left", (0, 1, 0), (5.9, -1.25)),
    ("right", (0, -1, 0), (5.9, 1.25)),
]


def latest_checkpoint(logdir: Path) -> Path:
    cands = [(int(m.group(1)), p) for p in logdir.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if not cands:
        sys.exit(f"{logdir} にmodel_*.ptが無い")
    return max(cands)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs/go2_stairs")
    ap.add_argument("--checkpoint", default=None, help="省略時はlogdir内の最新model_N.pt")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--speed", type=float, default=0.5, help="指令の大きさ[m/s]")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default="runs/video/cmd_vel.mp4")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import _mjmlx_rl_native as native
    from rsl_rl.runners import OnPolicyRunner

    import mjmlx_go2_env as e
    from train_go2_ppo import make_train_cfg

    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.logdir))
    xml_path = "/tmp/go2_play_generated.xml"
    env = e.MjmlxGo2Env(native, len(CASES), cmd_vel_min=0.0, cmd_vel_max=1.0, use_gpu=not args.cpu, seed=0, terrain_seed=0,
                        flat_terrain=True, generated_xml_path=xml_path)
    runner = OnPolicyRunner(env, make_train_cfg(0, 1), log_dir=None, device="cpu")
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device="cpu")

    env.reset()
    n = len(CASES)
    qpos, qvel = env._sim.state()
    qpos, qvel = qpos.copy(), qvel.copy()
    qvel[:] = 0.0
    qpos[:, 3:7] = [1, 0, 0, 0]
    qpos[:, 7:19] = e.DEFAULT_JOINT_POS
    qpos[:, 2] = HOME_Z
    cmd = np.zeros((n, 3), np.float32)
    for i, (_, d, (x, y)) in enumerate(CASES):
        qpos[i, 0], qpos[i, 1] = x, y
        cmd[i, : len(d)] = np.array(d, np.float32) * args.speed
    env._sim.set_state(qpos, qvel)
    env._commands[:] = cmd
    env._elapsed[:] = 0
    env._last_action[:] = 0.0
    env._prev_joint_vel[:] = 0.0
    env._next_push_step[:] = 10**9  # 評価ではランダム押しを止める
    obs = env.get_observations()

    steps = int(args.seconds / e.CONTROL_DT)
    fps = 1.0 / e.CONTROL_DT
    frames_q = np.zeros((steps, n, qpos.shape[1]), np.float32)
    meas = np.zeros((steps, n, 3), np.float32)  # 胴体座標系の実測 [vx, vy, wz]
    first_done = np.full(n, -1)
    with torch.no_grad():
        for t in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            q, v = env._sim.state()
            frames_q[t] = q
            meas[t, :, :2] = e._quat_rotate_inverse(q[:, 3:7], v[:, 0:3])[:, :2]
            meas[t, :, 2] = v[:, 5]
            d = dones.numpy().astype(bool)
            first_done[(first_done < 0) & d] = t
            env._commands[:] = cmd  # 終了して再サンプルされても指令は固定する

    warm = int(1.0 / e.CONTROL_DT)  # 立ち上がり1sを除いて評価
    summary = {"checkpoint": str(ckpt), "speed": args.speed, "cases": {}}
    for i, (name, _, _) in enumerate(CASES):
        m = meas[warm:, i].mean(axis=0)
        err = float(np.linalg.norm(m[:2] - cmd[i, :2]))
        summary["cases"][name] = {"cmd": cmd[i].tolist(), "mean_vel": m.tolist(), "vel_error": err, "fell_or_reset_at_step": int(first_done[i])}
        print(f"{name:9s} cmd=({cmd[i,0]:+.2f},{cmd[i,1]:+.2f}) mean(vx,vy,wz)=({m[0]:+.2f},{m[1]:+.2f},{m[2]:+.2f}) |err|={err:.2f}"
              + (f"  終了/リセット step={first_done[i]}" if first_done[i] >= 0 else ""))

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, TILE_H, TILE_W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.2, 135, -20
    font = ImageFont.load_default(size=18)
    win = int(0.5 / e.CONTROL_DT)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * TILE_W}x{2 * TILE_H}",
         "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    last = None
    for t in range(steps):
        tiles = []
        for i, (name, _, _) in enumerate(CASES):
            data.qpos[:] = frames_q[t, i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, cam)
            im = Image.fromarray(renderer.render())
            mv = meas[max(0, t - win + 1) : t + 1, i].mean(axis=0)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, TILE_W, 50], fill=(0, 0, 0))
            d.text((8, 4), f"{name}  cmd vx={cmd[i,0]:+.2f} vy={cmd[i,1]:+.2f}  t={t / fps:.1f}s", fill=(255, 255, 255), font=font)
            d.text((8, 27), f"meas vx={mv[0]:+.2f} vy={mv[1]:+.2f} wz={mv[2]:+.2f}", fill=(120, 255, 120), font=font)
            tiles.append(np.asarray(im))
        last = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        ff.stdin.write(last.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    Image.fromarray(last).save(out.with_suffix(".png"))
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out} ({steps} frames, {fps:g} fps, 2x2 tiles)")


if __name__ == "__main__":
    main()
