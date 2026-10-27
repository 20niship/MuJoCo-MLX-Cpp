#!/usr/bin/env python3
"""Go2のカメラ速度をIsaac Lab(TiledCamera)と比較する。1カメラ/env、RGB+depth、64px/128px x env数。

こちら(mjmlx)はこのプロセス内で、Isaac Labは ~/isaac/.venv のPythonを子プロセスで起動して測る(1構成1プロセス)。
指標: 描画のみ env-frames/s、物理+描画 env-steps/s(制御1step=物理4substep+描画1回)。JSONと比較表を出力する。

usage: bench_camera_vs_isaac.py [--build-dir build] [--envs 64 256 1024 4096] [--sizes 64 128] [--out runs/camera_vs_isaac.json]
       [--only mjmlx|isaac] [--isaac-python ~/isaac/.venv/bin/python] [--isaac-lab-dir ~/isaac/IsaacLab]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Isaacの水平画角 2*atan(20.955/(2*24)) = 47.5度に合わせた前方カメラ1台(baseの+x方向)。
os.environ["MJMLX_GO2_XML_SUBST"] = (
    '<camera name="track"->'
    '<camera name="front" pos="0.4 0 0.15" xyaxes="0 -1 0 0 0 1" fovy="47.5"/><camera name="track";'
    "<worldbody>-><worldbody>"
    '<light pos="0 0 4" dir="0.3 0.2 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>'
)


def bench_mjmlx(args, n, size):
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from mjmlx_go2_env import MjmlxGo2Env

    env = MjmlxGo2Env(native, n, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
    env.reset()
    cam = native.create_camera(env._sim, size, size)
    rng = np.random.default_rng(0)
    act = lambda: torch.from_numpy(rng.normal(0, 0.5, (n, 12)).astype(np.float32))
    for _ in range(3):
        env.step(act())
        cam.render()
        cam.rgb(), cam.depth()

    def timeit(fn, iters=args.iters):
        t = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t.append(time.perf_counter() - t0)
        return sum(t) / len(t)

    render_gpu = timeit(cam.render)
    render_rb = timeit(lambda: (cam.render(), cam.rgb(), cam.depth()))

    def step_render():
        env.step(act())
        cam.render()
        cam.rgb(), cam.depth()

    both = timeit(step_render)
    return {"render_gpu_s": render_gpu, "render_readback_s": render_rb, "step_plus_render_s": both}


def bench_isaac(args, n, size):
    cmd = ["env", "OMNI_KIT_ACCEPT_EULA=YES", args.isaac_python, "-u", os.path.join(HERE, "isaac_camera_bench.py"), "--num_envs", str(n), "--size", str(size), "--headless", "--iters", str(args.iters)]
    try:
        p = subprocess.run(cmd, cwd=args.isaac_lab_dir, capture_output=True, text=True, timeout=args.isaac_timeout)
    except subprocess.TimeoutExpired:
        return {"skip": f"timeout {args.isaac_timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith("ISAAC_RESULT "):
            return json.loads(line[len("ISAAC_RESULT "):])
    err = [l for l in (p.stdout + p.stderr).splitlines() if "out of memory" in l.lower() or "ISAAC_ERROR" in l or "Error" in l]
    return {"skip": (err[-1][:200] if err else f"no result (exit={p.returncode})")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="runs/camera_vs_isaac.json")
    ap.add_argument("--only", choices=["mjmlx", "isaac"], default=None)
    ap.add_argument("--isaac-python", default=os.path.expanduser("~/isaac/.venv/bin/python"))
    ap.add_argument("--isaac-lab-dir", default=os.path.expanduser("~/isaac/IsaacLab"))
    ap.add_argument("--isaac-timeout", type=int, default=1800)
    args = ap.parse_args()
    sys.path.insert(0, args.build_dir)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for n in args.envs:
        for size in args.sizes:
            key = f"{n}x{size}"
            r = results.setdefault(key, {"num_envs": n, "size": size})
            if args.only != "isaac":
                r["mjmlx"] = bench_mjmlx(args, n, size)
            if args.only != "mjmlx":
                r["isaac"] = bench_isaac(args, n, size)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(results, open(args.out, "w"), indent=1)
            print(key, json.dumps(r), flush=True)

    print("\n| envs | px | mjmlx 描画のみ GPU (env-frames/s) | mjmlx 描画+読出 (env-frames/s) | Isaac 描画のみ (env-frames/s) | mjmlx 物理+描画 (env-steps/s) | Isaac 物理+描画 (env-steps/s) |")
    print("|---|---|---|---|---|---|---|")
    f = lambda n, s: "-" if not s else f"{n / s:,.0f}"
    for r in results.values():
        m, i = r.get("mjmlx", {}), r.get("isaac", {})
        iv = (lambda k: f(r["num_envs"], i.get(k))) if "skip" not in i else (lambda k: "skip: " + i["skip"][:40])
        print(f"| {r['num_envs']} | {r['size']} | {f(r['num_envs'], m.get('render_gpu_s'))} | {f(r['num_envs'], m.get('render_readback_s'))} | "
              f"{iv('render_only_s') if i else '-'} | {f(r['num_envs'], m.get('step_plus_render_s'))} | {iv('step_plus_render_s') if i else '-'} |")


if __name__ == "__main__":
    main()
