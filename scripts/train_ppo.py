"""Train PPO on HalfCheetah via mjmlx's batched GPU physics. Requires the native extension built first: cmake --build build --target _mjmlx_rl_native"""

import argparse
import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--logdir", type=str, default="runs/halfcheetah")
    parser.add_argument("--build-dir", type=str, default="build")
    parser.add_argument("--cpu", action="store_true", help="use CPU physics instead of GPU")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.build_dir)
    import _mjmlx_rl_native as native

    from mjmlx_halfcheetah_vec_env import MjmlxHalfCheetahVecEnv

    raw_env = MjmlxHalfCheetahVecEnv(native, args.envs, use_gpu=not args.cpu, seed=args.seed)
    env = VecMonitor(raw_env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=64,
        batch_size=4096,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        seed=args.seed,
    )

    class ThroughputCallback(BaseCallback):
        def _on_training_start(self) -> None:
            self._t0 = time.time()
            self._n0 = self.num_timesteps

        def _on_step(self) -> bool:
            if self.num_timesteps % (self.model.n_steps * self.model.n_envs) == 0:
                elapsed = time.time() - self._t0
                sps = (self.num_timesteps - self._n0) / max(elapsed, 1e-9)
                print(f"[{self.num_timesteps} steps] {sps:.0f} env-steps/sec")
            return True

    model.learn(total_timesteps=args.timesteps, callback=ThroughputCallback(), progress_bar=False)

    os.makedirs(args.logdir, exist_ok=True)
    out_path = os.path.join(args.logdir, "ppo_halfcheetah.zip")
    model.save(out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
