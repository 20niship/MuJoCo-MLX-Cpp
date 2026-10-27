#!/usr/bin/env python3
"""Go2のカメラ画像をMuJoCo標準レンダラ(mujoco.Renderer)と比較する: depth誤差とロボットシルエットIoU、side-by-side画像を出力。

usage: compare_camera_mujoco.py [--build-dir build] [--envs 8] [--size 128] [--cam orbit] [--out runs/camera_compare]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")
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
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cam", default="orbit")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="runs/camera_compare")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import mujoco
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
    cam.render()
    rgb, depth = cam.rgb(), cam.depth()
    qpos = env._sim.qpos()

    m = mujoco.MjModel.from_xml_path("/tmp/go2_stairs_generated.xml")
    d = mujoco.MjData(m)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    ours_idx = cid  # mjmlxはMJCFのカメラ順
    r = mujoco.Renderer(m, args.size, args.size)
    r.enable_depth_rendering()

    def mj_depth(q, hide_robot=False):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        opt = mujoco.MjvOption()
        if hide_robot:
            opt.geomgroup[2] = 0  # ロボットのvisual geomはgroup 2
        r.update_scene(d, camera=cid, scene_option=opt)
        r.enable_depth_rendering()
        return r.render().copy()

    def mj_rgb(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cid)
        r.disable_depth_rendering()
        return r.render().copy()

    zfar = 100.0
    rows, errs, ious, ious_all = [], [], [], []
    for e in range(args.envs):
        q = qpos[e].astype(np.float64)
        dm = mj_depth(q)
        dfloor = mj_depth(q, hide_robot=True)  # 床のみのdepth
        do = depth[e, ours_idx]
        hit_m, hit_o = dm < 0.5 * m.vis.map.zfar * m.stat.extent, do < 0.5 * zfar
        both = hit_m & hit_o
        errs.append(np.abs(dm - do)[both])
        ious_all.append((hit_m & hit_o).sum() / max((hit_m | hit_o).sum(), 1))
        rm, ro = hit_m & (dm < dfloor - 0.1), hit_o & (do < dfloor - 0.1)  # ロボットのシルエット(床より手前)
        ious.append((rm & ro).sum() / max((rm | ro).sum(), 1))
        rows.append(np.concatenate([rgb[e, ours_idx], mj_rgb(q)], axis=0))
    err = np.concatenate(errs)
    print(f"cam={args.cam} size={args.size} envs={args.envs}")
    print(f"depth |ours-mujoco| (両方ヒット画素): mean={err.mean():.4f} median={np.median(err):.4f} p95={np.percentile(err, 95):.4f} m")
    print(f"ロボットシルエットIoU: mean={np.mean(ious):.3f} min={np.min(ious):.3f}")
    print(f"ヒット(空以外)マスクIoU: mean={np.mean(ious_all):.4f}")
    os.makedirs(args.out, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=1)).save(f"{args.out}/{args.cam}_ours_top_mujoco_bottom.png")


if __name__ == "__main__":
    main()
