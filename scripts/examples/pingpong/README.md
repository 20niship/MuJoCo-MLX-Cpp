# SO-101 keep-up (卓球のラリー練習) on the GPU

An SO-101 arm with a paddle keeps a ping-pong ball bouncing. Thousands of arms are simulated at once with `vkmujoco` and the policy
is trained with rsl-rl PPO. Two variants share one environment (`env.py`) and one script (`train.py`):

| | actor sees | critic sees |
|---|---|---|
| state (default) | joints, face pose, **ball position and velocity** | same |
| `--camera` | joints, face pose, **two camera images** (now and one step ago), never the ball | the true ball state (asymmetric actor-critic) |

The cameras are `cam_a` (behind and above the arm) and `cam_side` (beside the table), 96x64 RGB, rendered for every environment by
`vkmujoco.create_camera`.

The task is a port of the keep-up task of [mariapatni/robo-ping-pong](https://github.com/mariapatni/robo-ping-pong): reward per
second of flight (high, straight, coming back to the centre), a 9-stage curriculum on where the ball is dropped, per-environment
randomisation of ball mass, drag and servo gain, sensor noise, ball latency and random pushes.

## Files

| file | |
|---|---|
| `env.py` | `KeepUpVecEnv` (rsl-rl `VecEnv`; `camera=True` for the image variant), `KeepUpConfig`, torch FK of the arm and the table guard |
| `train.py` | `train`, `eval` and `render` sub-commands |
| `scene/keepup.xml`, `scene/keepup_cam.xml` | the physics scene (arm, table, net, paddle, ball; the second adds the two cameras) |
| `scene/ref.json`, `scene/ref.npz` | nominal pose, reachable footprint of the face and constants, baked once from the CPU reference world |
| `scene/assets/` | the SO-101 meshes, **not in git**: fetch them as below |

## 1. Fetch the SO-101 meshes

The robot description is the Robot Studio SO-101 from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
(Apache-2.0); its meshes are not copied into this repository.

```bash
cd scripts/examples/pingpong/scene/assets
for f in $(grep -o 'assets/[^"]*\.stl' ../keepup.xml | sed 's|assets/||' | sort -u); do
  curl -fsSLO https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotstudio_so101/assets/$f
done
```

## 2. Train, evaluate, render

```bash
uv sync --group pingpong     # torch, rsl-rl, tensordict, opencv
cd scripts/examples/pingpong

uv run python train.py train --envs 4096 --run state             # ~25 min on a 12 GB GPU for 1500 iterations
uv run python train.py train --camera --envs 512 --run cam        # slower: images for every environment, every step

uv run python train.py eval runs/cam --camera --every 250         # ranks checkpoints, copies the best to runs/cam/best.pt
MUJOCO_GL=egl uv run python train.py render runs/cam/best.pt cam.mp4 --camera
```

Checkpoints and tensorboard logs (`Loss/*`, `Train/mean_reward`, `Episode/bounces`, ...) go to `runs/<run>/` (git-ignored).

`eval` plays every checkpoint on the same drops for a fixed horizon and averages **all** episodes that finish in it. Do not stop at
the first N finished episodes: those are the short failures, and a working policy then looks broken. `Train/mean_reward` is not
used to pick the best checkpoint because it falls whenever the curriculum widens the drops. The `Episode/*` values in the training
log are averaged per step, which over-states long episodes; trust `eval`.

## Results

128 environments, 10 s episodes, balls dropped in the stage-3 ring (0.18 m), mean action:

| policy | mean bounces | median | episodes that ran the full 10 s |
|---|---|---|---|
| arm does not move | 0.4 | | |
| state, 1500 iterations | 24.0 | 26 | 92% |
| camera, 2250 iterations (resumed once) | 8.6 | 0 | 26% |

The camera policy works but is far behind the state policy: half of its episodes still end in the first second, the rest run long.
Higher resolution, a crop around the ball, or a ball detector in front of the state policy are the obvious next steps.

## Notes and limits

- vkmujoco does not yet handle a sphere against a cylinder correctly (the ball is kicked sideways), so the round blade is a box of
  equal area in this scene.
- Contacts are inferred from the ball's velocity change over four short steps per control step, not read from the simulator.
  Restitution is model-wide: there is no per-hit restitution jitter or rim loss as in the CPU original.
- One control rate (30 Hz) for all environments.
- `scene/` was generated from robo-ping-pong's CPU world by `scripts/bake_vkmujoco_example.py` in that repository; the example itself
  has no IK or hardware code.
