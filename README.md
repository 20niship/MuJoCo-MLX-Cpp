# MuJoCo-MLX-Cpp

**English** | [日本語](README_JA.md)

**Project page: https://20niship.github.io/MuJoCo-MLX-Cpp/**

GPU-batched MuJoCo physics and camera rendering on Vulkan, for reinforcement learning with thousands of parallel environments.

- **Physics**: MuJoCo-compatible pipeline (kinematics, collision, constraint solver, integration) as GPU compute kernels, validated against MuJoCo C.
- **Camera**: every env x every camera rendered in one indirect draw; RGB+depth handed to PyTorch as CUDA tensors without a CPU copy (Linux).
- **RL**: Unitree Go2 environment with Isaac Lab-style observations/rewards, trained with [rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO.
- **Libraries**: `libmjmlx` (GPU physics, C API `mjmlx.h`) and `libmjb` (dual-backend C API `mjb.h`: MuJoCo C on CPU or the GPU engine). Python via nanobind.

The tensor backend is [`extern/mlx_vulkan`](https://github.com/20niship/mlx_vulkan) (MKX, an MLX-style array library on Vulkan compute), vendored as a git submodule. The Metal backend was removed; Vulkan is the only GPU backend. (The `MJB_BACKEND_MLX` enum and `mlx` names in the API are historical and mean "the GPU engine".)

## Highlights (RTX 3060 12 GB unless noted)

Unitree Go2, flat ground, dt 0.005 x 4 substeps, `env.step` with observations and rewards, `device=cuda`:

| Envs | This repo (default) | This repo (fast mode) | Isaac Lab | Genesis |
|---|---|---|---|---|
| 4,096 | 409K | **480K** | ~120K | ~252K |
| 16,384 | 526K | **649K** | - | ~313K |

env-steps/s. Fast mode is `self_collision=False, max_condim=3` (what Isaac Lab/Genesis use); the default keeps the exact model. Isaac Lab / Genesis values are earlier single measurements and not re-run in the same session; Genesis measures a bare `scene.step()` while ours includes observation and reward. Details and caveats: [docs/PERFORMANCE.md](docs/PERFORMANCE.md) (Japanese).

Camera RL step (physics x4 substeps + render + RGB/depth as CUDA tensors, 1 camera/env, 64 px, 4096 envs, front view): **337,618 env-steps/s** (Genesis past value: 82,900). Third-person orbit view (whole robot visible): 124,415.

Media for this README and the project page (training videos, CPU vs GPU comparison, camera views) is **not stored in the repository**; it is attached on GitHub and referenced by URL.

https://github.com/user-attachments/assets/349e410a-1bc7-4451-830a-485e4a42be4b

*CPU (MuJoCo C, left) vs GPU (right) for piper / openarm / go2 / g1, produced by `just check-video-export`. Arms agree; legged robots diverge after landing because contact solutions are chaotic, so we compare single-step error instead (below).*

## Build

Requirements: Linux with a Vulkan 1.3 GPU driver (NVIDIA tested; lavapipe is used in CI), CMake >= 3.25, a C++20 compiler, [just](https://github.com/casey/just), [uv](https://github.com/astral-sh/uv), and MuJoCo (`pip install mujoco`, found automatically, or `-DMUJOCO_ROOT=...`). macOS via MoltenVK is untested.

```bash
git clone --recursive https://github.com/20niship/MuJoCo-MLX-Cpp
cd MuJoCo-MLX-Cpp
just build                      # libmjmlx + libmjb + tests/benchmarks (Release) into build/
just build-python               # also builds the nanobind extension (_mjmlx_rl_native)
```

Equivalent raw CMake: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DMJMLX_BUILD_PYTHON=ON && cmake --build build -j`.

CMake options: `MJMLX_BUILD_TESTS`, `MJMLX_BUILD_BENCHMARKS`, `MJMLX_BUILD_EXAMPLES`, `MJMLX_BUILD_PYTHON` (default OFF), `MJMLX_BUILD_CAMERA` (default ON; OFF turns `mjmlx_camera_*` into stubs).

Fetch robot models (mesh MJCFs for piper/openarm/go2/g1/h1, git-ignored):

```bash
just fetch-models
```

## Train Go2 (PPO, rsl_rl)

```bash
uv sync --group build --group go2

# flat ground, Isaac Lab (UnitreeGo2FlatEnvCfg)-style rewards, PPO update on CUDA
PYTHONPATH=<torch-cu126 etc.> uv run python scripts/train_go2_ppo.py --build-dir build \
  --flat --reward-preset isaac_flat --device cuda --envs 4096 --max-iterations 1500 --logdir runs/go2_flat

# stairs terrain (default), checkpoints every 50 iterations under --logdir
uv run python scripts/train_go2_ppo.py --build-dir build --envs 2048 --logdir runs/go2_stairs

# roll out the trained policy with forward/back/left/right commands -> tiled mp4 in runs/video/
uv run python scripts/play_go2_cmd_vel.py --logdir runs/go2_flat --build-dir build --reward-preset isaac_flat --speed 0.5

tensorboard --logdir runs/
```

Useful flags: `--reward-preset {relaxed,isaac_flat}`, `--device cuda`, `--no-self-collision --max-condim 3` (fast mode), `--use-sensors` (observations via the batched IMU / contact sensors), `--cpu` (MuJoCo C instead of the GPU engine). `MJMLX_GO2_INTEGRATOR=implicitfast` integrates PD actuators implicitly.

Simplifications versus Isaac Lab: no height scanner (blind policy), no per-env physical domain randomization (one model shared by all envs), no terrain curriculum, `cmd_vel` sampled directly in a norm range instead of heading-command mode.

A self-contained HalfCheetah-v4 PPO example (stable-baselines3) is in `scripts/train_ppo.py` (`just train-ppo`).

## Simulation simplification options (load time)

The MJCF is not modified; the in-memory model is overridden before compilation, only for articulated bodies (terrain is untouched). Defaults are exact MuJoCo behaviour.

| Option | Effect | Cost |
|---|---|---|
| `self_collision=False` | robot geoms do not collide with each other (`conaffinity=0`); Go2 pairs 233 -> 23 | self-contact disappears |
| `max_condim=3` | drops torsional/rolling friction of the feet (10 -> 4 constraint rows per contact) | slight change in foot slipping/spinning |

```c
MjmlxModelOptions opt = mjmlx_default_model_options();   // {self_collision=1, max_condim=0}
opt.self_collision = 0; opt.max_condim = 3;
MjmlxModel* m = mjmlx_load_model_ex("go2.xml", &opt);
```
```python
model = native.load_model(path, self_collision=False, max_condim=3)
env = MjmlxGo2Env(native, 4096, ..., self_collision=False, max_condim=3)
```

Go2 4096 / 16384 envs, env-steps/s: default 409K / 528K; `max_condim=3` 468K / 639K; both options 481K / 648K.

## Camera sensor (Vulkan, batched GPU sim only)

`<camera>` / `<light>` / geom rgba from the MJCF; Phong shading, RGB + depth, hfield and 2D textures (plane/box/hfield), spatial tendons drawn as cylinders. GPU-side frustum culling and LOD (mesh vertex clustering, `MJMLX_CAM_LOD_K`), one `vkCmdDrawIndexedIndirect` for all envs and cameras.

```python
cam = native.create_camera(sim, 128, 128)       # sim from create_batched(use_gpu=True)
cam.render()
rgb, depth = cam.rgb(), cam.depth()             # (num_envs, ncam, H, W, 3) uint8 / (..., H, W) float32 [m]

from mjmlx_camera_cuda import CudaCamera        # Vulkan -> CUDA external-memory interop (Linux)
gcam = CudaCamera.try_create(cam)               # None if unsupported -> fall back to the CPU readback above
cam.render(); rgba, depth = gcam.read()         # CUDA tensors, no CPU copy
```

Not supported: shadows, translucency, lights attached to moving bodies, trackcom (approximated by the body origin). Interop needs Linux + `VK_KHR_external_memory_fd` + CUDA torch; set `MKX_VULKAN_DEVICE_INDEX` to match the CUDA device on multi-GPU hosts. Camera needs a graphics+compute queue, `multiDrawIndirect`, `drawIndirectFirstInstance`, `shaderClipDistance` (only `create_camera` fails otherwise). Examples: `scripts/example_go2_camera.py`, comparison with MuJoCo's renderer: `scripts/compare_camera_mujoco.py`, speed: `scripts/bench_camera*.py`.

## Benchmarks and comparisons

Check that nothing else (ollama, desktop compositing, other GPU jobs) is using the GPU; it changes timings by 2x. Compare commits/settings **alternately within one session**, not across days.

```bash
uv run python scripts/bench_go2_step.py --build-dir build --envs 4096        # env.step speed
MJMLX_PROF=1 uv run python scripts/bench_go2_step.py --build-dir build       # per-stage GPU/host time
uv run python scripts/bench_physics_vs_others.py ...                         # this repo / Isaac Lab / Genesis
uv run python scripts/bench_camera_vs_isaac.py --build-dir build             # camera vs Isaac Lab TiledCamera
PYTHONPATH=<torch-cu126> uv run python scripts/bench_camera_rl.py --camera front --envs 64 256 1024 4096 --sizes 64 128
uv run python scripts/compare_cpu_gpu_go2.py --build-dir build               # 1-step difference vs MuJoCo C
just check-video-export                                                      # CPU|GPU tiled comparison video
just bench                                                                   # regression check against benchmarks/baseline.csv
```

Go2 1-control-step difference from MuJoCo C (joint angles): mean 3.6e-5 rad, p95 8e-7 rad (default); mean 1.4e-5 rad with `--max-condim 3` applied to both sides. Isaac Lab setup instructions and the full measurement log (what each optimization bought, regressions, GPU clock effects): [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Tests

```bash
just check-ci          # what CI runs (lavapipe-compatible subset)
just check-models      # real-robot MJCFs (piper/openarm/go2/g1/h1), local only, needs a real GPU
cd build && ctest -L correctness   # sensors, SDF, ...
cd build && ctest -L robust        # GPU batch sanity sweep (representative); robust_full = full nv x constraint grid; known_issue = tracked failures
```

Tests run the GPU engine and MuJoCo C side by side and compare trajectories: `test_gpu_vs_cpu` (humanoid, high-DOF tree, contact stress, tendons, real robots), `test_batched_collision_primitives`, `test_robustness` (nv x {limit, contact, tendon, frictionloss}: divergence, env-to-env mismatch, dependence on batch size/position, repeatability), `test_sensors`, `test_sdf_*`, `test_camera*`, `test_math_full`, `test_linalg_full`. CI runs everything in `scripts/Dockerfile` (Debian + lavapipe), identical to `just docker-test`. Environment variables for the robustness sweep: `MJMLX_SOLVER_SHARED=0`, `MJMLX_SOLVER_CG=1`, `LP_NUM_THREADS=1`, `ROBUST_VERBOSE=1`.

## C API

```c
#include <mjmlx/mjb.h>

MjbBackend* gpu = mjb_create_backend(MJB_BACKEND_MLX);   // the GPU engine; MJB_BACKEND_CPU = MuJoCo C (double, threads)
MjbModel* model = mjb_load_model(gpu, "humanoid.xml");
MjbBatchedConfig cfg = { .num_envs = 8192 };
MjbBatchedSim* sim = mjb_batched_create(model, &cfg);
mjb_batched_step(sim, controls);
const float* qpos = mjb_batched_get_qpos(sim, &n);
```

`mjb.h` works with either backend; `mjmlx.h` is the lower-level GPU-only API (`mjmlx_batched_*`, sensors, camera, packed read-back `mjmlx_batched_set_pack_spec/get_pack`). **The single-env pipeline (`mjmlx_step`, `mjmlx_forward`, ...) was removed and throws; use the batched API** (a batch of 1 is fine). `mjmlx_grad_step` is a stub; differentiable physics is not implemented. See the headers in [`include/mjmlx/`](include/mjmlx/).

## What is supported

Batched GPU path, checked against MuJoCo C. Full matrix and known gaps: [CONFORMANCE.md](CONFORMANCE.md). Design: [ARCHITECTURE.md](ARCHITECTURE.md).

- Joints: free, ball, hinge, slide. Integrators: Euler; implicit/implicitfast damping handled through the damping term (no RK4).
- Contacts: pyramidal friction, condim 3/4/6, margin/gap; plane/sphere/capsule/box/cylinder/ellipsoid/mesh (GJK/EPA), analytic sphere/capsule-box, SDF-SDF and SDF-mesh.
- Constraints: joint limits, DOF friction loss, fixed and spatial tendons (limits, friction loss, wrap).
- Gravity compensation, TENDON/SITE transmission, contact forces (`cfrc_ext`), 19 sensor types (IMU, frame, joint, actuator, subtreecom).
- Not supported in the batched path: equality constraints, actuator activation dynamics (FILTER/INTEGRATOR/MUSCLE), heightfield contacts (cameras do render hfields), RK4, inverse dynamics, islands/noslip/sleep, differentiable physics.

## Layout

```
include/mjmlx/     public C headers (mjmlx.h, mjb.h)
src/               batched.cpp (pipeline), *_vmap.cpp (smooth/constraint/solver graph), compat/mkx_kernels (GLSL kernels), camera.cpp, sensor.cpp, io.cpp, mjb.cpp
python/            nanobind bindings
scripts/           Go2 env/training, benchmarks, comparisons
tests/             C++ tests, fixture MJCFs
docs/              index.html (GitHub Pages), PERFORMANCE.md
extern/mlx_vulkan  Vulkan tensor backend (submodule)
```

## Contributing rules

Images and videos must never be committed (they are git-ignored). Attach them on GitHub (PR/issue) and reference the `user-attachments` URL from the README or `docs/index.html`.

## License

Apache 2.0
