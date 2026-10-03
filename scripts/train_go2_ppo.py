"""Go2階段地形cmd_vel追従PPO学習(rsl_rl版。事前に pip install rsl-rl-lib tensordict gitpython tensorboard torch)。"""

import argparse
import os
import sys

from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--logdir", type=str, default="runs/go2_stairs")
    parser.add_argument("--build-dir", type=str, default="build")
    parser.add_argument("--cpu", action="store_true", help="use CPU physics instead of GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terrain-seed", type=int, default=0)
    parser.add_argument("--cmd-vel-min", type=float, default=0.2, help="||cmd_vel||の下限[m/s or rad/s]")
    parser.add_argument("--cmd-vel-max", type=float, default=1.0, help="||cmd_vel||の上限[m/s or rad/s]")
    args = parser.parse_args()

    sys.path.insert(0, args.build_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _mjmlx_rl_native as native

    from mjmlx_go2_env import MjmlxGo2Env

    env = MjmlxGo2Env(
        native,
        args.envs,
        cmd_vel_min=args.cmd_vel_min,
        cmd_vel_max=args.cmd_vel_max,
        use_gpu=not args.cpu,
        seed=args.seed,
        terrain_seed=args.terrain_seed,
    )

    mlp_cfg = dict(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    train_cfg = {
        "seed": args.seed,
        "device": "cpu",
        "num_steps_per_env": 24,
        "init_at_random_ep_len": True,
        "max_iterations": args.max_iterations,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "check_for_nan": True,
        "save_interval": 50,
        "experiment_name": "go2_stairs",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "class_name": "OnPolicyRunner",
        "actor": {
            **mlp_cfg,
            "class_name": "MLPModel",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {**mlp_cfg, "class_name": "MLPModel", "distribution_cfg": None},
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
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

    runner = OnPolicyRunner(env, train_cfg, log_dir=args.logdir, device=train_cfg["device"])
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=train_cfg["init_at_random_ep_len"])


if __name__ == "__main__":
    main()

# ponytail: height_scan・per-env物理DR(質量/摩擦/COM)は batched simが全envで単一modelを共有するため未実装。per-envモデルAPIを追加すれば拡張可。
