#!/usr/bin/env python3
"""Isaac Lab側: Go2(平地)にTiledCamera(RGB+distance_to_camera)を1台/envで付け、描画のみと物理+描画の時間を測ってJSONを1行出力する。

~/isaac/.venv のPythonで実行する(bench_camera_vs_isaac.py から呼ばれる):
  OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/isaac_camera_bench.py --num_envs 256 --size 64 --headless
"""

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--size", type=int, default=64)
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--warmup", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

cfg = UnitreeGo2FlatEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 2.5
# 描画を最小にする(RTXの重い機能を全て切る)。
cfg.sim.render.enable_translucency = False
cfg.sim.render.enable_reflections = False
cfg.sim.render.enable_global_illumination = False
cfg.sim.render.antialiasing_mode = "Off"
cfg.sim.render.enable_dlssg = False
cfg.scene.tiled_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.15), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),  # baseの前方(+x)を向く
    data_types=["rgb", "distance_to_camera"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
    width=args.size,
    height=args.size,
)
import traceback

try:
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    cam = env.scene["tiled_camera"]
    act = torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=env.device)


    def sync():
        torch.cuda.synchronize()


    for _ in range(args.warmup):
        env.step(act)
    sync()
    t = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        env.step(act)  # 物理(decimation) + レンダリング + センサ更新
        sync()
        t.append(time.perf_counter() - t0)
    step_render = sum(t) / len(t)

    # 描画のみ: レンダー + カメラのバッファ更新(出力はGPUテンソルのまま)
    t = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        env.sim.render()
        cam.update(dt=0.0, force_recompute=True)
        out = cam.data.output
        sync()
        t.append(time.perf_counter() - t0)
    render_only = sum(t) / len(t)
    rgb_shape = tuple(out["rgb"].shape)
    print("ISAAC_RESULT " + json.dumps({"num_envs": args.num_envs, "size": args.size, "step_plus_render_s": step_render, "render_only_s": render_only, "rgb_shape": rgb_shape}))
    env.close()
    app.close()
except BaseException:
    print('ISAAC_ERROR ' + traceback.format_exc(), flush=True)
    raise
