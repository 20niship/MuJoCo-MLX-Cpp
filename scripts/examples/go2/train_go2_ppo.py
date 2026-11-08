"""Go2階段地形cmd_vel追従PPO学習(rsl_rl版。事前に pip install rsl-rl-lib tensordict gitpython tensorboard torch)。"""

import argparse
import os
import sys

from rsl_rl.runners import OnPolicyRunner

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))]  # このディレクトリ、examples/(common.py)、scripts/(mjmlx_sensors.py等)

from common import make_train_cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--logdir", type=str, default="runs/go2_stairs")
    parser.add_argument("--build-dir", type=str, default="build")
    parser.add_argument("--cpu", action="store_true", help="use CPU physics instead of GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PPO更新のデバイス(cuda可)")
    parser.add_argument("--flat", action="store_true", help="階段なしの平地で学習する")
    parser.add_argument("--reward-preset", default="relaxed", choices=["relaxed", "isaac_flat"])
    parser.add_argument("--terrain-seed", type=int, default=0)
    parser.add_argument("--no-self-collision", action="store_true", help="簡略化オプション: 脚/胴体同士の衝突を切る(Isaac Labと同じ。既定は自己衝突あり)")
    parser.add_argument("--max-condim", type=int, default=0, help="簡略化オプション: ロボットのgeomのcondim上限。3で足のねじり・転がり摩擦が無くなる(既定0=モデルのまま)")
    parser.add_argument("--domain-rand", action="store_true", help="env別の物理ランダム化(摩擦・base質量・base COM・アクチュエータ出力)を起動時に1回適用する")
    parser.add_argument("--use-sensors", action="store_true", help="観測のbase速度・重力と接触判定をMJCFの<sensor>/ContactSensor(Isaac Lab形式)から取る")
    parser.add_argument("--cmd-vel-min", type=float, default=0.2, help="||cmd_vel||の下限[m/s or rad/s]")
    parser.add_argument("--cmd-vel-max", type=float, default=1.0, help="||cmd_vel||の上限[m/s or rad/s]")
    args = parser.parse_args()

    sys.path.insert(0, args.build_dir)
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
        flat_terrain=args.flat,
        reward_preset=args.reward_preset,
        device=args.device,
        use_sensors=args.use_sensors,
        self_collision=not args.no_self_collision,
        max_condim=args.max_condim,
        domain_rand=args.domain_rand,
    )

    train_cfg = make_train_cfg(args.seed, args.max_iterations)
    train_cfg["device"] = args.device

    runner = OnPolicyRunner(env, train_cfg, log_dir=args.logdir, device=train_cfg["device"])
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=train_cfg["init_at_random_ep_len"])


if __name__ == "__main__":
    main()

# ponytail: height_scan・per-env物理DR(質量/摩擦/COM)は batched simが全envで単一modelを共有するため未実装。per-envモデルAPIを追加すれば拡張可。
