#!/usr/bin/env python3
"""カメラ無しのGo2(平地)物理stepをmjmlx / MuJoCo Warp / Isaac Lab / Genesisで比較する。制御1step=dt 0.005 x decimation 4、ランダムaction。

mjmlx は2通り測る: `mjmlx`=環境のenv.step(観測・報酬まで含む、学習で実際に使う速さ)と `mjmlx_bare`=物理のsim.stepだけ(観測・報酬なし。MuJoCo Warp / Genesisの測り方と同じ)。

usage: bench_physics_vs_others.py [--envs 1024 4096] [--out runs/physics_vs_others.json] [--only mjmlx|mjmlx_bare|mjwarp|isaac|genesis]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "examples", "go2"))  # mjmlx_go2_env


def bench_mjmlx(n, iters, self_collision=True):
    import numpy as np
    import torch
    from mjmlx_go2_env import MjmlxGo2Env

    env = MjmlxGo2Env(n, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True, self_collision=self_collision)
    env.reset()
    rng = np.random.default_rng(0)
    act = lambda: torch.from_numpy(rng.normal(0, 0.5, (n, 12)).astype(np.float32))
    for _ in range(25):  # GPUのクロックが上がるまで待つ
        env.step(act())
    t = []
    for _ in range(iters):
        a = act()
        t0 = time.perf_counter()
        env.step(a)
        t.append(time.perf_counter() - t0)
    return {"physics_step_s": sum(t) / len(t), "physics_step_median_s": sorted(t)[len(t) // 2]}


def bench_mjmlx_bare(n, iters, self_collision=True):
    """物理のstepだけ(観測・報酬なし)。MuJoCo Warp / Genesisと同じ範囲を測る。"""
    import numpy as np
    from mjmlx_go2_env import MjmlxGo2Env

    env = MjmlxGo2Env(n, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True, self_collision=self_collision)
    env.reset()
    sim = env._sim
    rng = np.random.default_rng(0)
    default = np.array([0.0, 0.9, -1.8] * 4, np.float32)
    pool = [
        (default + 0.5 * rng.standard_normal((n, 12))).astype(np.float32) for _ in range(4)
    ]  # 学習ループと同じく、毎step新しい配列を確保せず入力を使い回す(確保のたびにGPUバッファの確保と転送が走り、16384 envで物理より遅くなる)
    ctrl = lambda i: pool[i % 4]
    for i in range(25):
        sim.step(ctrl(i), 4)
    sim.sync()
    t = []
    for i in range(iters):
        c = ctrl(i)
        t0 = time.perf_counter()
        sim.step(c, 4)
        sim.sync()  # vkmujocoのstepは非同期でtorch.cuda.synchronizeでは待てない。ホストへ転送せずGPUの完了だけを待つ
        t.append(time.perf_counter() - t0)
    return {"physics_step_s": sum(t) / len(t), "physics_step_median_s": sorted(t)[len(t) // 2]}


def run_sub(cmd, tag, cwd=None, timeout=3600):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"skip": f"timeout {timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1 :])
    err = [l for l in (p.stdout + p.stderr).splitlines() if "rror" in l or "out of memory" in l.lower()]
    return {"skip": (err[-1][:200] if err else f"no result (exit={p.returncode})")}


def child():
    """1つの(kind, envs)を測って結果を1行出す。サイズごとに別プロセスで測る(同じプロセスで環境を作り直すとVulkanのメモリが解放されず、大きいサイズで遅くなる)。"""
    _, _, kind, n, iters, self_collision = sys.argv
    fn = bench_mjmlx if kind == "mjmlx" else bench_mjmlx_bare
    print("MJMLX_RESULT " + json.dumps(fn(int(n), int(iters), self_collision == "1")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="runs/physics_vs_others.json")
    ap.add_argument("--no-self-collision", action="store_true", help="mjmlxの脚/胴体同士の衝突を切る(Isaac Lab/Genesisのベンチ設定に合わせる)")
    ap.add_argument("--only", choices=["mjmlx", "mjmlx_bare", "mjwarp", "isaac", "genesis"], default=None)
    ap.add_argument("--isaac-python", default=os.path.expanduser("~/isaac/.venv/bin/python"))
    ap.add_argument("--isaac-lab-dir", default=os.path.expanduser("~/isaac/IsaacLab"))
    ap.add_argument("--mjwarp-python", default=os.path.expanduser("~/mjwarp/.venv/bin/python"))
    ap.add_argument("--mjwarp-iterations", type=int, nargs="*", default=[], help="MuJoCo Warpのソルバ反復数を変えた行も測る(省略時はgo2_mjx.xml既定の1だけ)")
    ap.add_argument("--genesis-python", default=os.path.expanduser("~/genesis/.venv/bin/python"))
    args = ap.parse_args()

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for n in args.envs:
        r = results.setdefault(str(n), {"num_envs": n})
        if args.only in (None, "mjmlx"):
            r["mjmlx"] = run_sub([sys.executable, os.path.abspath(__file__), "--child", "mjmlx", str(n), str(args.iters), "0" if args.no_self_collision else "1"], "MJMLX_RESULT")
        if args.only in (None, "mjmlx_bare"):
            r["mjmlx_bare"] = run_sub([sys.executable, os.path.abspath(__file__), "--child", "mjmlx_bare", str(n), str(args.iters), "0" if args.no_self_collision else "1"], "MJMLX_RESULT")
        if args.only in (None, "mjwarp"):
            for it in [None] + list(args.mjwarp_iterations):
                cmd = [args.mjwarp_python, "-u", os.path.join(HERE, "mjwarp_bench.py"), "--num_envs", str(n), "--iters", str(args.iters)] + ([] if it is None else ["--iterations", str(it)])
                r["mjwarp" if it is None else f"mjwarp_it{it}"] = run_sub(cmd, "MJWARP_RESULT")
        if args.only in (None, "isaac"):
            r["isaac"] = run_sub(
                [
                    "env",
                    "OMNI_KIT_ACCEPT_EULA=YES",
                    args.isaac_python,
                    "-u",
                    os.path.join(HERE, "isaac_camera_bench.py"),
                    "--no_camera",
                    "--num_envs",
                    str(n),
                    "--iters",
                    str(args.iters),
                    "--headless",
                ],
                "ISAAC_RESULT",
                cwd=args.isaac_lab_dir,
            )
        if args.only in (None, "genesis"):
            r["genesis"] = run_sub([args.genesis_python, "-u", os.path.join(HERE, "genesis_bench.py"), "--mode", "physics", "--num_envs", str(n), "--iters", str(args.iters)], "GENESIS_RESULT")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(n, json.dumps(r), flush=True)

    cols = ["mjmlx", "mjmlx_bare", "mjwarp"] + sorted({k for r in results.values() for k in r if k.startswith("mjwarp_it")}) + ["isaac", "genesis"]
    print("\n| envs | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for r in results.values():
        cells = [("skip" if "skip" in r[w] else f"{r['num_envs'] / r[w].get('physics_step_median_s', r[w]['physics_step_s']):,.0f}") if w in r else "-" for w in cols]
        print(f"| {r['num_envs']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    child() if len(sys.argv) > 1 and sys.argv[1] == "--child" else main()
