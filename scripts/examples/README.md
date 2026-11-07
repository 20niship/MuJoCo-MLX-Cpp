# Training and verification examples

Robot environments, PPO training, playback and CPU/GPU checks that run on top of the engine. Everything here is example code (not part of `libmjmlx`); engine-level benchmarks and tools stay in [`scripts/`](../README.md).

```
scripts/examples/
  common.py        helpers shared by Go2 and G1 (quaternion functions, rsl_rl PPO base config)
  go2/             Unitree Go2: mjmlx_go2_env.py, go2_terrain.py, train_go2_ppo.py, play_go2_cmd_vel.py,
                   compare_cpu_gpu_go2.py, example_go2_camera.py, example_go2_sensors.py
  g1/              Unitree G1: mjmlx_g1_env.py, train_g1_ppo.py, play_g1_cmd_vel.py, eval_g1_forward.py, compare_cpu_gpu_g1.py
  halfcheetah/     train_ppo.py, mjmlx_halfcheetah_vec_env.py (stable-baselines3 example)
```

Run from the repository root, e.g. `PYTHONPATH=<torch-cu126 etc.> uv run python scripts/examples/go2/train_go2_ppo.py ...`. Each script puts its own directory, `scripts/examples/` and `scripts/` on `sys.path`, so cross-imports (`common`, `mjmlx_sensors`, `mjmlx_camera_cuda`) work from any location. Build the Python extension first (see the [top README](../../README.md#build)); `--build-dir` points at it.

Videos referenced here are attached on GitHub and are not stored in the repository.

<table>
<tr>
<td align="center"><video src="https://github.com/user-attachments/assets/6b5d210c-da5b-401b-8277-287d5a3b85c3" controls muted width="400"></video><br><b>Go2</b> flat ground cmd_vel (work in progress)</td>
<td align="center"><video src="https://github.com/user-attachments/assets/1c79aae8-1332-4f13-b42b-4f6cc9de6615" controls muted width="400"></video><br><b>G1</b> forward-only, side view</td>
<td align="center"><video src="https://github.com/user-attachments/assets/539b7676-baa8-4600-8a9c-ddae2274d4c8" controls muted width="400"></video><br><b>G1</b> forward-only, top view</td>
</tr>
</table>

## Train Go2 (PPO, rsl_rl)

```bash
uv sync --group build --group go2

# flat ground, Isaac Lab (UnitreeGo2FlatEnvCfg)-style rewards, PPO update on CUDA
PYTHONPATH=<torch-cu126 etc.> uv run python scripts/examples/go2/train_go2_ppo.py --build-dir build \
  --flat --reward-preset isaac_flat --device cuda --envs 4096 --max-iterations 1500 --logdir runs/go2_flat

# stairs terrain (default), checkpoints every 50 iterations under --logdir
uv run python scripts/examples/go2/train_go2_ppo.py --build-dir build --envs 2048 --logdir runs/go2_stairs

# roll out the trained policy with forward/back/left/right commands -> tiled mp4 in runs/video/
uv run python scripts/examples/go2/play_go2_cmd_vel.py --logdir runs/go2_flat --build-dir build --reward-preset isaac_flat --speed 0.5

tensorboard --logdir runs/
```

Useful flags: `--reward-preset {relaxed,isaac_flat}`, `--device cuda`, `--no-self-collision --max-condim 3` (fast mode), `--use-sensors` (observations via the batched IMU / contact sensors), `--cpu` (MuJoCo C instead of the GPU engine). `MJMLX_GO2_INTEGRATOR=implicitfast` integrates PD actuators implicitly.

Simplifications versus Isaac Lab: no height scanner (blind policy), no per-env physical domain randomization (one model shared by all envs), no terrain curriculum, `cmd_vel` sampled directly in a norm range instead of heading-command mode.

Trained policy rollout (Go2, flat ground, `isaac_flat` preset, 500 iterations, commands forward/back/left/right at 0.5 m/s; the policy tracks back/left/right but stays still for forward, so this is a work in progress):

A self-contained HalfCheetah-v4 PPO example (stable-baselines3) is in `scripts/examples/halfcheetah/train_ppo.py` (`just train-ppo`).

## Train G1 (PPO, rsl_rl; reproduces Isaac Lab `Isaac-Velocity-Flat-G1-v0`)

```bash
uv sync --group build --group go2 && just fetch-models        # menagerie unitree_g1 (29 DoF)
PYTHONPATH=<torch-cu126 etc.> uv run python scripts/examples/g1/train_g1_ppo.py --build-dir build --device cuda --envs 4096 --max-iterations 1500 --logdir runs/g1_flat
uv run python scripts/examples/g1/play_g1_cmd_vel.py --logdir runs/g1_flat --build-dir build           # forward / forward 1.0 / lateral / turn tiles -> runs/video/g1_cmd_vel.mp4
uv run python scripts/examples/g1/compare_cpu_gpu_g1.py --build-dir build                              # 1-step difference vs MuJoCo C
```

**Reference chosen and why.** We reproduce Isaac Lab v2.3.2 `Isaac-Velocity-Flat-G1-v0`: [`flat_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py), [`rough_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py) (rewards inherited), [`rsl_rl_ppo_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py) (`G1FlatPPORunnerCfg`), [`velocity_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py), [`mdp/rewards.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py) and the `G1_CFG` actuator/default-pose table in [`isaaclab_assets/robots/unitree.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_assets/isaaclab_assets/robots/unitree.py). Reasons: the code exists and every number is readable; it uses the same training library (rsl_rl PPO) as our Go2 environment; the *flat* config needs no height scanner, no mass/COM randomisation and no pushes (friction is a fixed 0.8), so this engine's gaps (no ray casting, one model per batch, no terrain curriculum) do not matter; and Isaac Lab 2.3.2 can be run locally under the same conditions for a curve comparison.

Considered and not used:
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (`legged_gym/envs/g1/g1_config.py`): 12-DoF G1 URDF, a recurrent actor-critic with hidden size 32 and a gait-phase style reward set (`feet_swing_height`, `contact`, ...) on the legged_gym stack; different library and robot definition, so curves would not be comparable with our rsl_rl setup.
- MuJoCo Playground G1 joystick ([`rss2025/.../locomotion/g1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/rss2025/mujoco_playground/_src/locomotion/g1/joystick.py)): MJX/Warp + Brax PPO, 0.02 s control with 0.002 s physics, `feet_phase`/`pose` rewards, pushes and per-env domain randomisation (`randomize.py`) that needs per-env model parameters we do not have.
- HumanoidVerse / humanoid-gym / Genesis locomotion examples: not investigated (the Genesis example is Go2).

**What matches Isaac Lab:** dt 0.005 x decimation 4, 20 s episodes, action = default + 0.5 x action, observation (99 = base lin/ang vel, projected gravity, command, joint pos/vel, last action) with the same uniform observation noise, heading-mode yaw command with 2% standing envs and 10 s resampling (vx 0..1, vy -0.5..0.5), the full flat reward set (termination -200, track_lin 1.0, track_ang 1.0, lin_vel_z -0.2, ang_vel_xy -0.05, torques/acc on hip+knee, action_rate -0.005, biped feet_air_time 0.75 / 0.4 s, feet_slide -0.1, flat_orientation -1.0, ankle limits, hip/arm/waist deviation) each multiplied by the control step like Isaac's `RewardManager`, termination on torso contact, PD gains from `G1_CFG` (legs 150-200/5, ankles 20/2, arms 40/10), default pose, no self-collision, reset to the default pose with random yaw, and the PPO runner (actor/critic [256, 128, 128] ELU, entropy 0.008, lr 1e-3 adaptive, 24 steps/env).

**Differences (be aware when comparing):**
- Robot: menagerie `g1.xml` has 29 DoF (3 waist axes, 7 arm axes, no fingers) while Isaac's G1 has a 1-axis waist, 5 arm axes and 14 finger joints. Extra waist/wrist joints use the same stiff gains and are covered by the existing deviation penalties. The pelvis start height is computed so the feet touch the floor (0.80 m vs 0.74 m).
- Collision: only foot spheres, pelvis, torso and knees collide with the floor (like Isaac's `G1_MINIMAL_CFG`, exact set differs). `actuatorfrcrange` is removed and Isaac's `effort_limit_sim` (300/20) is not applied: the engine does not clamp actuator force. Joint `frictionloss` is set to 0 (as in Isaac), floor friction 0.8.
- Torque and joint acceleration for the rewards are computed from the PD law and finite differences per control step; contact/air time and foot speed are sampled once per control step (Isaac samples per physics step); foot speed uses a MuJoCo `framelinvel` sensor (one substep lag).
- Integrator: `implicitfast` handles the PD velocity term (PhysX implicit actuators in Isaac).

**Measured (RTX 3060, 1024 envs, seed 42, 24 steps/env/iteration, 300 iterations each; mean episode reward / mean episode length in steps, max 1000):**

| iteration | this repo | Isaac Lab 2.3.2 (same task, same PPO settings) |
|---|---|---|
| 10 | -7.27 / 40.2 | -5.96 / 40.6 |
| 50 | -5.45 / 42.1 | -5.40 / 47.1 |
| 100 | -5.29 / 61.9 | -5.12 / 54.9 |
| 150 | -4.99 / 69.0 | -5.24 / 72.8 |
| 200 | -4.86 / 71.3 | -6.87 / 179.2 |
| 250 | -4.92 / 81.6 | -11.76 / 802.9 |
| 299 | -4.95 / 86.8 | -8.64 / 967.8 |

**The curves track each other for the first ~150 iterations, then Isaac Lab's G1 learns to stay up (episodes reach ~970 of 1000 steps) while ours does not (~87 steps, i.e. it still falls after ~1.7 s). Isaac Lab's behaviour is therefore not reproduced yet.** The cause is not identified. What we know: with zero action (default pose, the PD gains above) the menagerie G1 tips over within ~1.5 s under both MuJoCo C and this engine, so it is not specific to the GPU engine; CPU/GPU agree in contact-free steps (2.5e-6 rad mean) and with foot contact when standing (~5e-5 rad), and are looser in random-action states with impacts (5.8e-4 rad mean, 2.3e-3 p95; Go2: 3.6e-5; not caused by solver iterations or condim). Candidate causes we have not tested: mass/inertia and foot-geometry differences between the menagerie model and Isaac's G1 USD (hands, 4-sphere feet), contact softness versus PhysX, missing effort limits, the 1-per-control-step contact sampling. Playback of the 300-iteration policy (`scripts/examples/g1/play_g1_cmd_vel.py`) shows it falling after ~1.5 s for all four commands.

G1 physics is also currently **slow in this engine**: about 4-5K env-steps/s (nv=35: the solver spends ~50 ms per substep at 1024 envs and stops getting slower beyond about 8 Newton iterations; Go2 is ~100x faster per env), i.e. ~5 s per iteration here versus ~1 s for Isaac Lab at 1024 envs on an idle GPU, so the default 1500 x 4096 run would take many hours. Raising the shared-memory solver limit from nv<=32 to 40 was tried and did not help, so it was reverted. H1 is not done yet.

### Forward-only task (`--forward-only`): the G1 walks

The omnidirectional task above did not converge in our runs, so a forward-only task was added on top (the omnidirectional mode is unchanged): command `vx` in [0.4, 0.8] m/s fixed, `vy = wz = 0`, no heading mode and no standing envs.

```bash
PYTHONPATH=<torch-cu126 etc.> uv run python scripts/examples/g1/train_g1_ppo.py --build-dir build --forward-only --solver-iters 4 --envs 1024 --max-iterations 800 --device cuda --logdir runs/g1_forward_v1
# then continue with the final reward weights (see below), e.g. 400 more iterations; the straight-walking v3 adds --fwd-weights yaw_rate=-0.5,heading_hold=-1,lateral_vel=-1 and resumes from it
uv run python scripts/examples/g1/eval_g1_forward.py runs/g1_forward_v3/model_1746.pt --build-dir build   # 256 envs x 20 s: falls, speed, yaw rate, heading change
uv run python scripts/examples/g1/train_g1_ppo.py --build-dir build --forward-only --solver-iters 4 --envs 1024 --max-iterations 400 --device cuda --logdir runs/g1_forward --resume runs/g1_forward_v1/model_799.pt
uv run python scripts/examples/g1/play_g1_cmd_vel.py --forward-only --logdir runs/g1_forward --build-dir build --seconds 12   # vx 0.4 / 0.5 / 0.6 / 0.8 tiles, checkerboard floor
```

**Root cause of the earlier fall after ~1.5 s (isolated with MuJoCo C, no GPU engine involved):** with Isaac Lab's ankle gains (kp 20, kd 2) the default pose cannot stand even with zero action. A statically balanced inverted pendulum needs ankle stiffness above m*g*h_com (about 35 kg x 9.81 x 0.69 m = 240 Nm/rad) and the ankles have little authority (action scale 0.5 rad x kp 20 = 10 Nm). Scan with MuJoCo C, default pose, 5 s: ankle kp 100 falls, kp 150 falls or leans (depends on kd), kp >= 200 stands (tilt 0.05-0.08 rad). The forward-only task therefore uses ankle PD **200 / 5** (the omnidirectional default stays Isaac's 20 / 2). Zero action then stands for 6+ s on the GPU engine. The fall also happens in MuJoCo C, so the GPU engine is not the cause. Not tested: effort limits, contact sampling frequency, foot geometry. The final policy was evaluated with `scripts/examples/g1/eval_g1_forward.py`.

**Reward (forward-only):** Isaac Lab's flat G1 terms (termination -200, lin_vel_z, ang_vel_xy, hip/knee torque and acceleration, action_rate, biped feet_air_time 0.75, feet_slide, flat_orientation, ankle limits, hip/arm/waist deviation, all scaled by the control step) with changes: `track_lin_vel_xy_exp` 1.0 -> **2.0** (forward speed is the main task), `track_ang_vel_z_exp` 1.0 -> **2.0** and a new **alive +0.5** term. First attempt with `track_ang` 0.5 converged to a policy that walks in a circle (yaw rate ~0.65 rad/s, forward speed only 0.2-0.3 m/s for commands 0.4-0.8), so it was raised to 2.0 and training was continued from that checkpoint.

**Result (RTX 3060, 1024 envs, `--solver-iters 4` for speed, 800 + 400 iterations, about 70 min total):** training mean episode length 965 / 1000 at iteration 799 (reward 16.2) and 954 / 1000 at the end (reward 30.0; max episode length is 1000 steps = 20 s). Evaluation of the final checkpoint with observation noise on (256 envs x 20 s, `deterministic`): **0 % of envs fell within 20 s**, mean forward speed **0.54 m/s for a mean command of 0.60 m/s** (commands 0.4 / 0.5 / 0.6 / 0.8 -> 0.45 / 0.49 / 0.53 / 0.60 m/s in the 12 s playback, no falls), lateral drift 0.08 m/s. **This v2 policy still turned while walking** (mean yaw rate 0.45 rad/s), walking along an arc; this is fixed in v3 below. Speed still saturates below the command at 0.8. Isaac Lab was not run for the forward-only task (it has no such configuration).

#### Going straight (v3): rewards added and results

Added (forward-only mode only; the omnidirectional rewards are unchanged), all scaled by the control step like the other terms:
- `yaw_rate_l2` (weight -0.5): world-frame yaw rate squared.
- `heading_hold` (-1.0): squared deviation of the yaw from the heading at episode start.
- `lateral_vel_l2` (-1.0): squared lateral body velocity.
- The command now includes Isaac Lab's heading mode: `wz` command = clip(0.5 x (start heading - yaw)), so the heading error is observable by the policy without changing the 99-dim observation (checkpoints stay compatible); `track_ang` (2.0) tracks it.

Weights were chosen with short runs (512 envs, 150 iterations each, resumed from v2). Evaluation: `scripts/examples/g1/eval_g1_forward.py` (256 envs x 20 s, observation noise on; heading change is accumulated, so circling shows as >360 deg):

| model | fell < 20 s | mean vx (cmd 0.60) | mean vy | mean yaw rate | mean abs yaw rate | 20 s heading change | along / cross-track displacement | course angle |
|---|---|---|---|---|---|---|---|---|
| v1: `track_ang` 0.5 (800 it) | 0 % | 0.31 | 0.14 | 0.75 | 1.31 | +817 deg | 0.34 m / 0.40 m | 51 deg |
| v2: `track_ang` 2.0 (+400 it) | 0 % | 0.54 | 0.08 | 0.45 | 0.58 | +399 deg | 1.23 m / 0.94 m | 43 deg |
| A: yaw -0.5, heading -1, lateral -1 (+150 it) | 0 % | 0.54 | 0.04 | 0.11 | 0.40 | +7.9 deg (abs 11.4) | 10.5 m / 1.52 m | 8.5 deg |
| B: yaw -2, heading -3, lateral -2 (+150 it) | 0.4 % | 0.50 | 0.07 | 0.11 | 0.42 | +23.8 deg | 9.1 m / 3.23 m | 20 deg |
| **v3: A weights, +400 it (final)** | 0.4 % (1/256, at 7 s) | 0.51 | -0.00 | **0.06** | 0.29 | **-2.6 deg (abs 4.0)** | 10.2 m / 0.74 m | **4.1 deg** |

(Yaw rate in rad/s, speeds in m/s. Cross-track = lateral displacement from the start heading line at 20 s, mean absolute value.) The stronger weights (B) were worse after the same 150 iterations. The final policy walks in an essentially straight line (heading changes by about 4 deg in 20 s; the playback tiles show straight trails with a slight drift to one side, up to ~0.6 m at 9.5 m for vx 0.8). The mean signed yaw rate is 0.06 rad/s (target ~0.1); the mean absolute yaw rate (0.29) is the left-right sway of the gait, not a turn. Forward speed still saturates below the command (0.51 vs 0.60 m/s, 0.60 vs 0.80 at the highest command) and one of 256 envs fell. Videos (local, not committed): `runs/video/g1_forward_straight.mp4` (side view, checkerboard floor, 20 s) and `runs/video/g1_forward_straight_topdown.mp4` (top-down with the pelvis trail in red and a 1 m reference line).
