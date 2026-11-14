#!/usr/bin/env python3
"""README記載のIsaac Lab/Genesisとの比較値をPNGにプロットする(画像はrepoに入れずPRへ添付する)。

usage: plot_perf_comparison.py [--out runs/media]
数値はREADMEの計測値(RTX 3060、Go2、env-steps/s)。Isaac Lab/Genesisは過去の単発計測で、同一セッションの再計測ではない。
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OURS, FAST, ISAAC, GEN = "#7c5cff", "#ff7a59", "#76b900", "#3aa6ff"
WARP, WARP2 = "#b5d97a", "#4f7d0a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def bars(ax, labels, series, fmt=lambda v: f"{v / 1000:.0f}K"):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for i, (name, vals, col) in enumerate(series):
        for j, v in enumerate(vals):
            if v is None:
                continue
            ax.bar(x[j] + (i - (n - 1) / 2) * w, v, w * 0.92, color=col, label=name if j == min(k for k, u in enumerate(vals) if u is not None) else None)
            ax.text(x[j] + (i - (n - 1) / 2) * w, v, fmt(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("env-steps/s (higher is better)")
    ax.legend(frameon=False)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/media")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    foot = "RTX 3060 12GB, Unitree Go2. Isaac Lab / Genesis: earlier single measurements (not re-run in the same session)."

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars(
        ax,
        ["256", "1,024", "4,096", "16,384"],
        [
            ("Isaac Lab", [16615, 57738, 120091, None], ISAAC),
            ("Genesis", [59705, 154430, 252518, 313000], GEN),
            ("This repo", [63297, 192823, 409549, 526102], OURS),
            ("This repo (fast mode)", [None, None, 480278, 649196], FAST),
        ],
    )
    ax.set_xlabel("parallel envs")
    ax.set_title("Go2 env.step throughput vs Isaac Lab / Genesis")
    fig.text(0.01, 0.005, foot + " Ours include obs+reward; Genesis is bare scene.step().", fontsize=7.5, color="#666")
    save(fig, out, "physics_vs_isaac_genesis.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars(
        ax,
        ["1,024", "4,096", "16,384"],
        [
            ("MuJoCo Warp (xml default, 1 iter)", [45000, 83000, 46000], WARP),
            ("MuJoCo Warp (16 iters)", [42000, 139000, 318000], WARP2),
            ("This repo env.step", [225000, 406000, 466000], OURS),
            ("This repo physics only", [291000, 501000, 503000], "#b9a8ff"),
            ("This repo physics only, fast mode", [309000, 523000, 517000], FAST),
        ],
    )
    ax.set_xlabel("parallel envs")
    ax.set_title("Go2 throughput vs MuJoCo Warp (same session, median of 30 steps)")
    fig.text(0.01, 0.005, "RTX 3060 12GB, Go2, dt 0.005 x4. MuJoCo Warp 4.0.1 + warp-lang 1.17.0 (driver CUDA 12.7). Speed only.", fontsize=7.5, color="#666")
    save(fig, out, "physics_vs_mujoco_warp.png")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    envs = ["64", "256", "1,024", "4,096"]
    gen = [10400, 32100, 64400, 82900]
    for ax, title, ours in [(axs[0], "front camera (robot barely visible)", [24273, 76307, 199027, 337618]), (axs[1], "third-person orbit camera (~400K tris/env)", [21250, 53839, 98292, 124415])]:
        bars(ax, envs, [("Genesis BatchRenderer (past value)", gen, GEN), ("This repo (Vulkan->CUDA interop)", ours, OURS)])
        ax.set_title(title)
        ax.set_xlabel("parallel envs (1 camera/env, 64 px)")
    fig.suptitle("Camera-in-the-loop RL step: physics x4 + render + RGB/depth as CUDA tensors", y=0.99)
    fig.text(0.01, 0.005, foot, fontsize=7.5, color="#666")
    save(fig, out, "camera_rl_vs_genesis.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars(ax, ["64", "256", "1,024"], [("Isaac Lab TiledCamera", [2238, 4133, 7278], ISAAC), ("This repo", [4180, 10972, 17604], OURS)], fmt=lambda v: f"{v / 1000:.1f}K")
    ax.set_xlabel("parallel envs (1 camera/env, 64 px)")
    ax.set_title("Physics + render step vs Isaac Lab (4096 envs: Isaac timed out; ours 19.6K)")
    fig.text(0.01, 0.005, foot + " Shading / clip planes differ.", fontsize=7.5, color="#666")
    save(fig, out, "camera_vs_isaac.png")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    steps = [
        ("PR #33 start", 7200),
        ("dt=0.005 x4", 22500),
        ("shared-mem solver", 32000),
        ("local Jacobian", 51000),
        ("VRAM buffers", 100000),
        ("packed readback + Cholesky", 150000),
        ("solver tuning", 149338),
        ("host numpy cleanup", 192412),
        ("torch obs/reward\n(device=cuda)", 307812),
        ("host overhead cut", 409549),
        ("fast mode", 480278),
    ]
    ax.bar(range(len(steps)), [v for _, v in steps], color=[OURS] * (len(steps) - 1) + [FAST])
    for i, (_, v) in enumerate(steps):
        ax.text(i, v, f"{v / 1000:.0f}K", ha="center", va="bottom", fontsize=9)
    ax.axhline(120091, color=ISAAC, ls="--", lw=1.2, label="Isaac Lab (4096 envs)")
    ax.axhline(252518, color=GEN, ls="--", lw=1.2, label="Genesis (4096 envs)")
    ax.set_xticks(range(len(steps)), [s for s, _ in steps], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("env-steps/s")
    ax.set_title("Go2 4096 envs: optimization history")
    ax.legend(frameon=False)
    fig.text(0.01, 0.005, "Milestones from different sessions / GPU clock states (device=cpu until 'torch obs/reward'): indicative, not a strict series.", fontsize=7.5, color="#666")
    save(fig, out, "optimization_history.png")


if __name__ == "__main__":
    main()
