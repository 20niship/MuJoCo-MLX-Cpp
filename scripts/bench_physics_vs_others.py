#!/usr/bin/env python3
"""カメラ無しのGo2(平地)物理stepをmjmlx / Isaac Lab / Genesisで比較する。制御1step=dt 0.005 x decimation 4、ランダムaction。

usage: bench_physics_vs_others.py [--build-dir build] [--envs 1024 4096] [--out runs/physics_vs_others.json] [--only mjmlx|isaac|genesis]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def bench_mjmlx(n, iters):
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from mjmlx_go2_env import MjmlxGo2Env

    env = MjmlxGo2Env(native, n, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
    env.reset()
    rng = np.random.default_rng(0)
    act = lambda: torch.from_numpy(rng.normal(0, 0.5, (n, 12)).astype(np.float32))
    for _ in range(3):
        env.step(act())
    t = []
    for _ in range(iters):
        a = act()
        t0 = time.perf_counter()
        env.step(a)
        t.append(time.perf_counter() - t0)
    return {"physics_step_s": sum(t) / len(t)}


def run_sub(cmd, tag, cwd=None, timeout=3600):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"skip": f"timeout {timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    err = [l for l in (p.stdout + p.stderr).splitlines() if "rror" in l or "out of memory" in l.lower()]
    return {"skip": (err[-1][:200] if err else f"no result (exit={p.returncode})")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="runs/physics_vs_others.json")
    ap.add_argument("--only", choices=["mjmlx", "isaac", "genesis"], default=None)
    ap.add_argument("--isaac-python", default=os.path.expanduser("~/isaac/.venv/bin/python"))
    ap.add_argument("--isaac-lab-dir", default=os.path.expanduser("~/isaac/IsaacLab"))
    ap.add_argument("--genesis-python", default=os.path.expanduser("~/genesis/.venv/bin/python"))
    args = ap.parse_args()
    sys.path.insert(0, args.build_dir)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for n in args.envs:
        r = results.setdefault(str(n), {"num_envs": n})
        if args.only in (None, "mjmlx"):
            r["mjmlx"] = bench_mjmlx(n, args.iters)
        if args.only in (None, "isaac"):
            r["isaac"] = run_sub(["env", "OMNI_KIT_ACCEPT_EULA=YES", args.isaac_python, "-u", os.path.join(HERE, "isaac_camera_bench.py"), "--no_camera", "--num_envs", str(n), "--iters", str(args.iters), "--headless"], "ISAAC_RESULT", cwd=args.isaac_lab_dir)
        if args.only in (None, "genesis"):
            r["genesis"] = run_sub([args.genesis_python, "-u", os.path.join(HERE, "genesis_bench.py"), "--mode", "physics", "--num_envs", str(n), "--iters", str(args.iters)], "GENESIS_RESULT")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(n, json.dumps(r), flush=True)

    print("\n| envs | mjmlx (env-steps/s) | Isaac Lab | Genesis |\n|---|---|---|---|")
    for r in results.values():
        cells = [("skip" if "skip" in r[w] else f"{r['num_envs'] / r[w]['physics_step_s']:,.0f}") if w in r else "-" for w in ("mjmlx", "isaac", "genesis")]
        print(f"| {r['num_envs']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
