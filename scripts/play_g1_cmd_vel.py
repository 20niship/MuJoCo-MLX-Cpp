#!/usr/bin/env python3
"""学習済みG1方策に平地で前進/横/旋回のcmd_velを与え、指令と実測の胴体速度をcaptionに載せた2x2タイル動画を書き出す(play_go2_cmd_vel.pyのG1版)。"""

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
# (ラベル, 指令[vx, vy, wz])。学習時の指令範囲は vx 0..1, vy -0.5..0.5, wz -1..1
ALL_CASES = [("forward 0.5", (0.5, 0.0, 0.0)), ("forward 1.0", (1.0, 0.0, 0.0)), ("left 0.4", (0.0, 0.4, 0.0)), ("turn 0.8", (0.0, 0.0, 0.8))]
FWD_CASES = [("forward 0.4", (0.4, 0.0, 0.0)), ("forward 0.5", (0.5, 0.0, 0.0)), ("forward 0.6", (0.6, 0.0, 0.0)), ("forward 0.8", (0.8, 0.0, 0.0))]


def latest_checkpoint(logdir: Path) -> Path:
    cands = [(int(m.group(1)), p) for p in logdir.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if not cands:
        sys.exit(f"{logdir} にmodel_*.ptが無い")
    return max(cands)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs/g1_flat")
    ap.add_argument("--checkpoint", default=None, help="省略時はlogdir内の最新model_N.pt")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="runs/video/g1_cmd_vel.mp4")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--forward-only", action="store_true", help="前進専用タスクの方策を再生(指令 vx 0.4/0.5/0.6/0.8)")
    ap.add_argument("--topdown", action="store_true", help="上から見た固定カメラ + 胴体軌跡と基準線(直進性の確認用)")
    ap.add_argument("--plain-floor", action="store_true", help="地面を市松模様にしない")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import _mjmlx_rl_native as native
    from rsl_rl.runners import OnPolicyRunner

    import mjmlx_g1_env as e
    from train_g1_ppo import make_g1_train_cfg

    e.CMD_RESAMPLE_STEPS = 10**9  # 評価では指令を固定する
    xml_path = "/tmp/g1_play_generated.xml"
    CASES = FWD_CASES if args.forward_only else ALL_CASES
    n = len(CASES)
    env = e.MjmlxG1Env(native, n, use_gpu=not args.cpu, seed=0, generated_xml_path=xml_path, obs_noise=False, forward_only=args.forward_only)
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.logdir))
    runner = OnPolicyRunner(env, make_g1_train_cfg(0, 1, "g1_forward" if args.forward_only else "g1_flat"), log_dir=None, device="cpu")
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device="cpu")

    env.reset()
    qpos, qvel = env._sim.state()
    qpos, qvel = qpos.copy(), qvel.copy()
    qvel[:] = 0.0
    qpos[:, 0:2], qpos[:, 3:7], qpos[:, 2], qpos[:, 7:] = 0.0, [1, 0, 0, 0], env.init_height, e.DEFAULT_JOINT_POS
    env._sim.set_state(qpos, qvel)
    cmd = torch.tensor([c for _, c in CASES], dtype=torch.float32)
    if args.forward_only:  # 向きは開始時(yaw=0)を保つheading指令。vxだけ固定する
        env._yaw0[:], env._heading_target[:] = 0.0, 0.0
        env._commands[:, 0] = cmd[:, 0]
    else:
        env._commands[:] = cmd
        env._is_heading[:] = False
    env._elapsed[:] = 0
    env._last_action[:] = 0.0
    env._prev_joint_vel[:] = 0.0
    obs = env.get_observations()

    steps = int(args.seconds / e.CONTROL_DT)
    fps = 1.0 / e.CONTROL_DT
    frames_q = np.zeros((steps, n, qpos.shape[1]), np.float32)
    meas = np.zeros((steps, n, 3), np.float32)  # yaw座標系の実測 [vx, vy, wz(世界)]
    first_done = np.full(n, -1)
    with torch.no_grad():
        for t in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            q, v = env._sim.state()
            qt, vt = torch.from_numpy(q), torch.from_numpy(v)
            meas[t, :, :2] = e._yaw_frame_vel(qt[:, 3:7], vt[:, 0:3])[:, :2].numpy()
            meas[t, :, 2] = e._rotate(qt[:, 3:7], vt[:, 3:6])[:, 2].numpy()
            frames_q[t] = q
            d = dones.numpy().astype(bool)
            first_done[(first_done < 0) & d] = t
            if args.forward_only:
                env._commands[:, 0] = cmd[:, 0]
            else:
                env._commands[:] = cmd
                env._is_heading[:] = False

    warm = int(2.0 / e.CONTROL_DT)  # 立ち上がり2sを除いて評価
    summary = {"checkpoint": str(ckpt), "cases": {}}
    for i, (name, c) in enumerate(CASES):
        m = meas[warm:, i].mean(axis=0)
        err = float(np.linalg.norm(m - np.array(c)))
        summary["cases"][name] = {"cmd": list(c), "mean_vel": m.tolist(), "vel_error": err, "fell_or_reset_at_step": int(first_done[i])}
        print(f"{name:12s} cmd=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) mean(vx,vy,wz)=({m[0]:+.2f},{m[1]:+.2f},{m[2]:+.2f}) |err|={err:.2f}"
              + (f"  終了/リセット step={first_done[i]}" if first_done[i] >= 0 else ""))

    xml = Path(xml_path).read_text()
    if not args.plain_floor:  # 描画専用: 床に市松模様(物理は変えない)
        tex = ('<texture name="floor_checker" type="2d" builtin="checker" rgb1="0.50 0.62 0.74" rgb2="0.92 0.94 0.97" width="512" height="512" mark="edge" markrgb="0.9 0.9 0.9"/>'
               '<material name="floor_checker" texture="floor_checker" texuniform="true" texrepeat="1 1" reflectance="0.05"/>')
        xml = xml.replace("<asset>", "<asset>" + tex, 1).replace('<geom name="floor" ', '<geom name="floor" material="floor_checker" ')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, TILE_H, TILE_W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135, -15
    if args.topdown:  # 真上から、開始点の前方を中央にした固定カメラ(+xが画面の右)
        cam.distance, cam.azimuth, cam.elevation = 15.0, 90, -89
        cam.lookat[:] = [5.5, 0.0, 0.0]
    trail = [[] for _ in CASES]
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
        for i, (name, c) in enumerate(CASES):
            data.qpos[:] = frames_q[t, i]
            mujoco.mj_forward(model, data)
            if not args.topdown:
                cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, cam)
            if args.topdown:
                trail[i].append(data.qpos[:2].copy())
                sc = renderer.scene
                marks = [np.array([x, 0.0]) for x in np.arange(0.0, 12.01, 1.0)]  # 基準線(+x方向、0.5 m刻み)
                for p_, rgba, r_ in [(m_, (1, 1, 1, 1), 0.04) for m_ in marks] + [(p_, (1, 0.2, 0.2, 1), 0.05) for p_ in trail[i][::5]]:
                    if sc.ngeom >= sc.maxgeom:
                        break
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r_, 0, 0]), np.array([p_[0], p_[1], 0.02]), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
                    sc.ngeom += 1
            im = Image.fromarray(renderer.render())
            mv = meas[max(0, t - win + 1) : t + 1, i].mean(axis=0)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, TILE_W, 50], fill=(0, 0, 0))
            d.text((8, 4), f"{name}  cmd vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f}  t={t / fps:.1f}s", fill=(255, 255, 255), font=font)
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
