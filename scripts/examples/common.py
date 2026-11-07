"""Go2/G1の学習環境で共有する、クォータニオンのtorchヘルパーとrsl_rl PPOの共通設定。"""

import numpy as np
import torch


def _quat_rotate_inverse_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ワールド座標系のvをbody座標系へ(qの逆回転)。q: (N,4)wxyz, v: (N,3)。"""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(-xyz, v, dim=1)
    return v + w * t + torch.cross(-xyz, t, dim=1)


def _projected_gravity_t(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([2.0 * (w * y - x * z), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0], dim=1)


def _yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    half = yaw / 2.0
    return np.stack([np.cos(half), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(half)], axis=1).astype(np.float32)


def make_train_cfg(seed: int, max_iterations: int, experiment_name: str = "go2_stairs") -> dict:
    """学習/再生で共有するrsl_rlの設定(Isaac LabのRoughPPORunnerCfg系)。G1は呼び出し側でネットワークとentropy_coefを上書きする。"""
    mlp_cfg = dict(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    return {
        "seed": seed,
        "device": "cpu",
        "num_steps_per_env": 24,
        "init_at_random_ep_len": True,
        "max_iterations": max_iterations,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "check_for_nan": True,
        "save_interval": 50,
        "experiment_name": experiment_name,
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
