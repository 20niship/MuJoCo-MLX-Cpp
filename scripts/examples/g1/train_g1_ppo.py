"""G1平地速度追従のPPO学習(rsl_rl)。設定はIsaac Lab v2.3.2のG1FlatPPORunnerCfg(config/g1/agents/rsl_rl_ppo_cfg.py)。"""

import argparse
import os
import sys

from rsl_rl.runners import OnPolicyRunner

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))]  # このディレクトリ、examples/(common.py)、scripts/(mjmlx_sensors.py等)
from common import make_train_cfg  # noqa: E402


def make_g1_train_cfg(seed: int, max_iterations: int, experiment_name: str = "g1_flat") -> dict:
    """Go2と共通のPPO設定から、G1FlatPPORunnerCfgとの差(ネットワーク[256,128,128]、entropy_coef 0.008)だけ変える。"""
    cfg = make_train_cfg(seed, max_iterations, experiment_name)
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [256, 128, 128]
    cfg["algorithm"]["entropy_coef"] = 0.008
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--logdir", type=str, default="runs/g1_flat")
    parser.add_argument("--build-dir", type=str, default="build")
    parser.add_argument("--cpu", action="store_true", help="use CPU physics instead of GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PPO更新のデバイス(cuda可)")
    parser.add_argument("--self-collision", action="store_true", help="自己衝突を入れる(Isaac LabのG1_CFGは無効。既定は無効)")
    parser.add_argument("--max-condim", type=int, default=0, help="簡略化オプション: ロボットのgeomのcondim上限")
    parser.add_argument("--forward-only", action="store_true", help="前進専用タスク(指令 vx 0.4..0.8、足首PD 200/5、報酬はW_FWD)。既定は全方向のIsaac Lab flat G1")
    parser.add_argument("--ankle-kp", type=float, default=None)
    parser.add_argument("--ankle-kd", type=float, default=None)
    parser.add_argument("--fwd-weights", type=str, default="", help="前進専用の報酬重みの上書き 例: yaw_rate=-0.5,heading_hold=-1,lateral_vel=-1")
    parser.add_argument("--solver-iters", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None, help="チェックポイントから学習を続ける(報酬を変えた後の追加学習用)")
    parser.add_argument("--no-obs-noise", action="store_true", help="観測ノイズ(Isaac LabのUnoise)を切る")
    args = parser.parse_args()

    sys.path.insert(0, args.build_dir)
    import _mjmlx_rl_native as native

    from mjmlx_g1_env import MjmlxG1Env

    env = MjmlxG1Env(native, args.envs, use_gpu=not args.cpu, seed=args.seed, device=args.device, self_collision=args.self_collision,
                     max_condim=args.max_condim, obs_noise=not args.no_obs_noise, forward_only=args.forward_only,
                     ankle_gains=(args.ankle_kp, args.ankle_kd) if args.ankle_kp is not None else None, solver_iterations=args.solver_iters,
                     fwd_weights={k: float(v) for k, v in (kv.split("=") for kv in args.fwd_weights.split(",") if kv)} or None)
    train_cfg = make_g1_train_cfg(args.seed, args.max_iterations, "g1_forward" if args.forward_only else "g1_flat")
    train_cfg["device"] = args.device
    runner = OnPolicyRunner(env, train_cfg, log_dir=args.logdir, device=train_cfg["device"])
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=train_cfg["init_at_random_ep_len"])


if __name__ == "__main__":
    main()
