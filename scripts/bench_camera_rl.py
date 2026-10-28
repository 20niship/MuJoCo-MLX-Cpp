#!/usr/bin/env python3
"""カメラあり強化学習の1 step(物理4substep + 描画 + RGB/depthがGPU上のtorchテンソルとして学習側で使える状態まで)の env-steps/s を測る。mjmlxのみ。

構成ごとに子プロセスを起こす(VRAM・Vulkan状態を持ち越さない)。torchは必ずcudaで動かす(GPUクロックが学習時と同条件になる)。
  - phys:        MjmlxGo2Env.step のみ
  - render:      cam.render() のみ(GPU上まで、読み出しなし)
  - step_render: step + render(読み出しなし)
  - rl_cpu:      step + render + CPU読み出し(numpy) + torch.cuda へアップロード(従来経路)
  - rl_gpu:      step + render + Vulkan->CUDA interopのゼロコピーテンソル(CudaCamera.read()が使える場合)
usage: PYTHONPATH=/home/owner/torch-cu126 bench_camera_rl.py [--build-dir build] [--envs 256 1024 4096] [--sizes 64 128] [--iters 20] [--out runs/camera_rl.json]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# モデル同梱のtrackカメラを置き換えて、全envで丁度1カメラにする。
# front: Isaac比較と同じ前方視点(ロボット自身はほぼ写らず軽い) / orbit: ロボット全体が写る三人称視点(約40万三角形/envが写り重い)
_FRONT = '<camera name="front" pos="0.4 0 0.15" xyaxes="0 -1 0 0 0 1" fovy="47.5"/>'
_ORBIT = '<camera name="front" mode="targetbody" target="base" pos="-1.4 -1.4 0.7" fovy="45"/>'
_orbit = "--camera" in sys.argv and sys.argv[sys.argv.index("--camera") + 1] == "orbit"
os.environ["MJMLX_GO2_XML_SUBST"] = (
    '<camera name="track" pos="0.846 -1.465 0.916" xyaxes="0.866 0.500 0.000 -0.171 0.296 0.940" mode="trackcom"/>->' + (_ORBIT if _orbit else _FRONT) + ";"
    "<worldbody>-><worldbody>"
    '<light pos="0 0 4" dir="0.3 0.2 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>'
)


def sm_clock():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], text=True).split()[0])
    except Exception:
        return None


def run_one(args, n, size):
    import numpy as np
    import torch
    import _mjmlx_rl_native as native
    from mjmlx_go2_env import MjmlxGo2Env

    dev = torch.device("cuda")
    torch.zeros(1, device=dev)  # CUDAコンテキストを先に作る(学習と同条件)
    env = MjmlxGo2Env(native, n, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
    env.reset()
    cam = native.create_camera(env._sim, size, size)
    assert cam.num_cameras == 1, cam.num_cameras
    rng = np.random.default_rng(0)
    act = torch.from_numpy(rng.normal(0, 0.5, (n, 12)).astype(np.float32))
    from mjmlx_camera_cuda import CudaCamera

    gcam = CudaCamera.try_create(cam) if args.mode == "gpu" else None  # gpuモードを有効化するとrender()にpack passが入るので、cpuモードとは別プロセスで測る

    def cpu_obs():
        return torch.from_numpy(cam.rgb()).to(dev), torch.from_numpy(cam.depth()).to(dev)

    def gpu_obs():
        return gcam.read()

    if gcam:
        cases = {"render_pack": lambda: cam.render(), "rl_gpu": lambda: (env.step(act), cam.render(), gpu_obs())}
        cam.render()
        gpu_obs()
    else:
        cases = {
            "phys": lambda: env.step(act),
            "render": lambda: cam.render(),
            "step_render": lambda: (env.step(act), cam.render()),
            "rl_cpu": lambda: (env.step(act), cam.render(), cpu_obs()),
        }
    for fn in cases.values():
        for _ in range(3):
            fn()
    # 構成間の時間変動(GPUクロック・熱)を均すため、ケースを交互に rounds 回測り、ケースごとの中央値を取る。
    samples = {name: [] for name in cases}
    for _ in range(args.rounds):
        for name, fn in cases.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t0) / args.iters)
    out = {name + "_s": sorted(v)[len(v) // 2] for name, v in samples.items()}
    out["sm_mhz"] = sm_clock()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--camera", choices=["front", "orbit"], default="front")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="runs/camera_rl.json")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="cpu", help="(内部用)")
    ap.add_argument("--reps", type=int, default=3, help="cpu/gpuプロセスを交互に起こす回数(中央値を採用)")
    ap.add_argument("--one", type=int, nargs=2, default=None, help="(内部用) n size を1構成だけ測りJSONを標準出力へ")
    args = ap.parse_args()
    sys.path.insert(0, args.build_dir)
    if args.one:
        print("RL_RESULT " + json.dumps(run_one(args, *args.one)))
        return
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print("| envs | px | phys | render | step+render | RL step (CPU読出+upload) | RL step (CUDA interop) | SM MHz |")
    print("|---|---|---|---|---|---|---|---|")
    for size in args.sizes:
        for n in args.envs:
            runs = {}
            for _ in range(args.reps):
                for mode in ("cpu", "gpu"):
                    cmd = [sys.executable, os.path.abspath(__file__), "--build-dir", os.path.abspath(args.build_dir), "--iters", str(args.iters), "--rounds", str(args.rounds),
                           "--camera", args.camera, "--mode", mode, "--one", str(n), str(size)]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    line = [l for l in p.stdout.splitlines() if l.startswith("RL_RESULT ")]
                    if not line:
                        print(f"| {n} | {size} | skip({mode}): {(p.stderr.strip().splitlines() or ['?'])[-1][:120]} |")
                        continue
                    for k, v in json.loads(line[0][len("RL_RESULT "):]).items():
                        runs.setdefault(k, []).append(v)
            if not runs:
                continue
            r = {k: sorted(v)[len(v) // 2] for k, v in runs.items()}
            results[f"{n}x{size}"] = r
            f = lambda k: f"{n / r[k]:,.0f}" if k in r else "-"
            print(f"| {n} | {size} | {f('phys_s')} | {f('render_s')} | {f('step_render_s')} | {f('rl_cpu_s')} | {f('rl_gpu_s')} | {r['sm_mhz']} |", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
