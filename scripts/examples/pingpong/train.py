"""Keep-up on the GPU: train, evaluate and render, with vkmujoco (thousands of SO-101 arms at once) + rsl-rl PPO.

    uv run --group pingpong python scripts/examples/pingpong/train.py train --envs 4096 --run state
    uv run --group pingpong python scripts/examples/pingpong/train.py train --camera --envs 512 --run cam
    uv run --group pingpong python scripts/examples/pingpong/train.py eval runs/cam --camera --every 250
    uv run --group pingpong python scripts/examples/pingpong/train.py render runs/cam/best.pt out.mp4 --camera

State version: the actor sees the ball's position and velocity. `--camera`: the actor sees two fixed cameras (96x64 RGB, now and
one step ago) and its own joints; the critic keeps the true ball state. Checkpoints and tensorboard logs go to
scripts/examples/pingpong/runs/<run>/. `eval` ranks the checkpoints by real bounces and copies the best one to best.pt.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def train_cfg(seed: int, iterations: int, device: str, steps_per_env: int, camera: bool = False) -> dict:
    mlp = dict(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    cfg = {
        "seed": seed,
        "device": device,
        "num_steps_per_env": steps_per_env,
        "init_at_random_ep_len": True,
        "max_iterations": iterations,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "check_for_nan": True,
        "save_interval": 50,
        "experiment_name": "keepup",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "class_name": "OnPolicyRunner",
        "actor": {**mlp, "class_name": "MLPModel", "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}},
        "critic": {**mlp, "class_name": "MLPModel", "distribution_cfg": None},
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_gpu": None,
    }
    if camera:
        cfg["obs_groups"] = {"actor": ["policy", "camera"], "critic": ["critic"]}
        cnn = dict(output_channels=[32, 64, 64], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="elu", flatten=True)
        cfg["actor"] = {**cfg["actor"], "class_name": "CNNModel", "cnn_cfg": cnn, "obs_normalization": False}
    return cfg


def make_env(args, cfg, num_envs: int, curriculum: bool = True, **kwargs):
    from env import KeepUpVecEnv

    return KeepUpVecEnv(num_envs, cfg=cfg, device=args.device, seed=args.seed, curriculum=curriculum, camera=args.camera, size=tuple(args.size), **kwargs)


def load_policy(args, env, checkpoint):
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, train_cfg(0, 1, args.device, 24, args.camera), log_dir=None, device=args.device)
    runner.load(str(checkpoint))
    return runner.get_inference_policy(device=args.device)


def set_stage(env, stage: int) -> None:
    env.stage = stage
    env._apply_stage()


# -- train -----------------------------------------------------------------------------------------------


def cmd_train(args) -> int:
    from env import KeepUpConfig
    from rsl_rl.runners import OnPolicyRunner

    args.envs = args.envs or (512 if args.camera else 4096)
    env = make_env(args, KeepUpConfig.randomized(), args.envs, curriculum=not args.no_curriculum)
    runner = OnPolicyRunner(env, train_cfg(args.seed, args.iterations, args.device, args.steps_per_env, args.camera), log_dir=str(HERE / "runs" / args.run), device=args.device)
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    return 0


# -- eval ------------------------------------------------------------------------------------------------


def evaluate(ckpt: Path, args) -> tuple[float, float, float]:
    """Mean and median bounces, and the share of episodes that ran to the time limit.

    Every checkpoint plays the same drops for a fixed horizon and ALL episodes that finish count. Stopping at the first N finished
    episodes would keep only the short failures and make a working policy look broken.
    """
    from env import KeepUpConfig

    env = make_env(args, KeepUpConfig.randomized(episode_s=args.episode_s), args.envs, curriculum=False)
    set_stage(env, args.stage)
    policy = load_policy(args, env, ckpt)
    obs = env.get_observations()
    bounces: list[float] = []
    timeouts = 0
    for _ in range(int(args.horizon * env.max_episode_length)):
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs, stochastic_output=args.stochastic))
        bounces.extend(env.finished_bounces.tolist())
        timeouts += int((env.finished_cause == 0).sum())
    del env
    torch.cuda.empty_cache()
    return float(np.mean(bounces)), float(np.median(bounces)), timeouts / max(len(bounces), 1)


def cmd_eval(args) -> int:
    ckpts = {int(m.group(1)): c for c in args.run_dir.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", c.name))}
    pick = [args.only] if args.only else [i for i in sorted(ckpts) if i % args.every == 0 and i > 0] + [max(ckpts)]
    results = {}
    for it in dict.fromkeys(pick):
        results[it] = evaluate(ckpts[it], args)
        print(f"model_{it}.pt: bounces mean {results[it][0]:.1f} median {results[it][1]:.0f}, full episodes {results[it][2]:.0%}", flush=True)
    if args.only:
        return 0
    best = max(results, key=lambda i: results[i][0])  # Train/mean_reward falls whenever the curriculum widens the drops: not used
    shutil.copy(ckpts[best], args.run_dir / "best.pt")
    print(f"best.pt <- model_{best}.pt (mean bounces {results[best][0]:.1f})")
    return 0


# -- render ----------------------------------------------------------------------------------------------


def cmd_render(args) -> int:
    """Replay a policy on the GPU and draw env 0 with MuJoCo's renderer (the policy's two camera views in the corner)."""
    import cv2
    import mujoco
    from env import SCENE, KeepUpConfig

    W, H = args.video_size
    env = make_env(args, KeepUpConfig.randomized(), 8, curriculum=False)
    set_stage(env, args.stage)
    policy = load_policy(args, env, args.checkpoint)
    model = mujoco.MjModel.from_xml_path(str(SCENE / ("keepup_cam.xml" if args.camera else "keepup.xml")))
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    view = mujoco.MjvCamera()
    view.lookat[:], view.distance, view.azimuth, view.elevation = (0.3, 0.0, 0.2), 1.3, 150, -20
    obs = env.get_observations()
    frames = []
    for _ in range(int(args.seconds / env.dt)):
        data.qpos[:] = env._q[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=view)
        img = np.ascontiguousarray(renderer.render())
        if args.camera:
            frame = obs["camera"][0, :6].cpu().numpy()  # current frame of both cameras
            for k in range(2):
                tile = (frame[3 * k : 3 * k + 3].transpose(1, 2, 0) * 255).astype(np.uint8)
                img[H - 170 : H - 10, 10 + 250 * k : 250 + 250 * k] = cv2.resize(tile, (240, 160), interpolation=cv2.INTER_NEAREST)
        text = f"{'camera' if args.camera else 'state'} policy, stage {args.stage}   bounces this episode: {int(env.bounces[0])}"
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs, stochastic_output=args.stochastic))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "26",
            "-movflags",
            "+faststart",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out} ({len(frames)} frames)")
    return 0


# -- cli -------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--camera", action="store_true", help="actor sees camera images instead of the ball state")
        q.add_argument("--size", type=int, nargs=2, default=(96, 64), metavar=("W", "H"), help="camera image size")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--device", default="cuda", help="where the policy lives (physics is always on the Vulkan GPU)")

    t = sub.add_parser("train", help="PPO training")
    common(t)
    t.add_argument("--envs", type=int, default=None, help="default 4096 (512 with --camera)")
    t.add_argument("--iterations", type=int, default=1500)
    t.add_argument("--steps-per-env", type=int, default=24)
    t.add_argument("--run", default="default")
    t.add_argument("--no-curriculum", action="store_true", help="drops from the whole footprint from the start")
    t.add_argument("--resume", default=None, help="checkpoint to continue from")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="rank checkpoints by real bounces, copy the best to best.pt")
    common(e)
    e.add_argument("run_dir", type=Path)
    e.add_argument("--only", type=int, default=None, help="evaluate just this iteration")
    e.add_argument("--every", type=int, default=100, help="evaluate every N iterations (and the last checkpoint)")
    e.add_argument("--envs", type=int, default=128)
    e.add_argument("--episode-s", type=float, default=10.0)
    e.add_argument("--horizon", type=float, default=5.0, help="steps to run, in episode lengths")
    e.add_argument("--stage", type=int, default=3, help="curriculum stage of the drops (a 0.06 m ring per stage, 9 = whole footprint)")
    e.add_argument("--stochastic", action="store_true", help="sample actions like training does (default: the mean action)")
    e.set_defaults(fn=cmd_eval, seed=123)

    r = sub.add_parser("render", help="write an mp4 of a checkpoint")
    common(r)
    r.add_argument("checkpoint", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--seconds", type=float, default=20.0)
    r.add_argument("--stage", type=int, default=3)
    r.add_argument("--video-size", type=int, nargs=2, default=(800, 600), metavar=("W", "H"))
    r.add_argument("--stochastic", action="store_true")
    r.set_defaults(fn=cmd_render, seed=7)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
