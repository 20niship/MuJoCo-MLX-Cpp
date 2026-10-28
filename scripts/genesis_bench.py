#!/usr/bin/env python3
"""Genesis(genesis-world)でGo2を並列envで動かし、速度を測る。JSONを1行(GENESIS_RESULT ...)出力する。

~/genesis/.venv のPythonで実行する(bench_camera_vs_isaac.py / bench_genesis_physics から呼ぶ想定):
  ~/genesis/.venv/bin/python scripts/genesis_bench.py --mode physics --num_envs 4096
  ~/genesis/.venv/bin/python scripts/genesis_bench.py --mode camera --num_envs 256 --size 64   # BatchRenderer(Madrona, rasterizer)

physics: 制御1step = dt 0.005 x decimation 4 の物理step + ランダムaction(位置制御)。camera: 同じ物理+1カメラ/env(RGB+depth)の描画。
"""

import argparse
import json
import time

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["physics", "camera"], default="physics")
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--size", type=int, default=64)
p.add_argument("--dump", default=None, help="env 0..3のRGBを横に並べてPNG保存(画像の健全性確認用)")
p.add_argument("--iters", type=int, default=20)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--renderer", choices=["batch_rasterizer", "batch_raytracer", "rasterizer"], default="batch_rasterizer")
args = p.parse_args()

import numpy as np
import torch
import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")
DT, DECIMATION = 0.005, 4

kw = {}
if args.mode == "camera":
    if args.renderer == "rasterizer":
        kw["renderer"] = gs.renderers.Rasterizer()
    else:
        kw["renderer"] = gs.options.renderers.BatchRenderer(use_rasterizer=args.renderer == "batch_rasterizer")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT),
    rigid_options=gs.options.RigidOptions(enable_self_collision=False),
    show_viewer=False,
    **kw,
)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.42)))
cam = None
if args.mode == "camera":
    scene.add_light(pos=(0, 0, 4), dir=(0.3, 0.2, -1), color=(1, 1, 1), directional=True, castshadow=False, cutoff=45.0, intensity=1.0)
    cam = scene.add_camera(res=(args.size, args.size), pos=(0.4, 0, 0.6), lookat=(2.0, 0, 0.4), fov=47.5, GUI=False)
scene.build(n_envs=args.num_envs)
if cam is not None:
    T = np.eye(4)
    T[:3, 3] = (0.4, 0.0, 0.15)
    T[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])  # ベース前方(+x)を見る(OpenGL: -z前方)
    cam.attach(robot.base_link, T)

names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
default = np.array([0.0] * 4 + [0.8, 0.8, 1.0, 1.0] + [-1.5] * 4, dtype=np.float32)
dofs = [robot.get_joint(n).dofs_idx_local[0] for n in names]
robot.set_dofs_kp(np.full(12, 20.0), dofs)
robot.set_dofs_kv(np.full(12, 0.5), dofs)
robot.set_dofs_position(torch.tensor(default, device=gs.device).repeat(args.num_envs, 1), dofs)
default_t = torch.tensor(default, device=gs.device)


def control_step():
    act = default_t + 0.5 * torch.randn(args.num_envs, 12, device=gs.device)
    robot.control_dofs_position(act, dofs)
    for _ in range(DECIMATION):
        scene.step()


def sync():
    import quadrants as qd

    qd.sync()  # Genesisの物理カーネルはquadrantsのstream上で動くので両方待つ
    torch.cuda.synchronize()


def render():
    cam.move_to_attach()
    out = cam.render(rgb=True, depth=True, force_render=True)
    return out


for _ in range(args.warmup):
    control_step()
    if cam is not None:
        render()
sync()
res = {"mode": args.mode, "num_envs": args.num_envs, "size": args.size, "renderer": args.renderer if cam is not None else None}
t = []
for _ in range(args.iters):
    t0 = time.perf_counter()
    control_step()
    sync()
    t.append(time.perf_counter() - t0)
res["physics_step_s"] = sum(t) / len(t)
if cam is not None:
    t = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = render()
        sync()
        t.append(time.perf_counter() - t0)
    res["render_only_s"] = sum(t) / len(t)
    t = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        control_step()
        out = render()
        sync()
        t.append(time.perf_counter() - t0)
    res["step_plus_render_s"] = sum(t) / len(t)
    res["out_types"] = [type(o).__name__ for o in out[:2]]
    res["rgb_shape"] = tuple(out[0].shape)
if cam is not None and args.dump:
    from PIL import Image

    rgb = out[0].cpu().numpy()
    d = out[1].cpu().numpy()
    res["depth_range"] = [float(d.min()), float(d.max())]
    Image.fromarray(np.concatenate(list(rgb[:4]), axis=1).astype(np.uint8)).save(args.dump)
print("GENESIS_RESULT " + json.dumps(res), flush=True)
