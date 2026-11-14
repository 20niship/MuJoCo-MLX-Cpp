# MuJoCo-MLX-Cpp

**English** | [日本語](README_JA.md)

[![CI](https://github.com/20niship/MuJoCo-MLX-Cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/20niship/MuJoCo-MLX-Cpp/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://20niship.github.io/MuJoCo-MLX-Cpp/)
[![License](https://img.shields.io/github/license/20niship/MuJoCo-MLX-Cpp)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/20niship/MuJoCo-MLX-Cpp)](https://github.com/20niship/MuJoCo-MLX-Cpp/commits/main)
[![Stars](https://img.shields.io/github/stars/20niship/MuJoCo-MLX-Cpp?style=flat)](https://github.com/20niship/MuJoCo-MLX-Cpp/stargazers)
![Vulkan](https://img.shields.io/badge/Vulkan-compute-AC162C?logo=vulkan&logoColor=white)
![No NVIDIA required](https://img.shields.io/badge/NVIDIA%20GPU-not%20required-76B900?logo=nvidia&logoColor=white)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.5-orange)
![Python](https://img.shields.io/badge/Python-nanobind-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20(tested)-lightgrey)

**Project page: https://20niship.github.io/MuJoCo-MLX-Cpp/**

GPU-batched MuJoCo physics and camera rendering on Vulkan, for reinforcement learning with thousands of parallel environments.

- **Physics**: MuJoCo-compatible pipeline (kinematics, collision, constraint solver, integration) as GPU compute kernels, validated against MuJoCo C.
- **Camera**: every env x every camera rendered in one indirect draw; RGB+depth handed to PyTorch as CUDA tensors without a CPU copy (Linux).
- **RL**: Unitree Go2 environment with Isaac Lab-style observations/rewards, trained with [rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO.
- **Libraries**: `libmjmlx` (GPU physics, C API `mjmlx.h`) and `libmjb` (dual-backend C API `mjb.h`: MuJoCo C on CPU or the GPU engine). Python via nanobind.

The tensor backend is [`extern/mlx_vulkan`](https://github.com/20niship/mlx_vulkan) (MKX, an MLX-style array library on Vulkan compute), vendored as a git submodule. The Metal backend was removed; Vulkan is the only GPU backend. (The `MJB_BACKEND_MLX` enum and `mlx` names in the API are historical and mean "the GPU engine".)

## Why this library

- **Same models as MuJoCo.** It loads the same MJCF XML files as MuJoCo (menagerie Go2/G1/Piper, Gymnasium Humanoid/HalfCheetah, ...) through MuJoCo's own parser, and every feature is checked against MuJoCo C in CI. Switching an existing MuJoCo model to the GPU engine is a `load_model(path)` call; no model conversion, no new asset format. `libmjb` additionally lets the same C code run on MuJoCo C (CPU) or the GPU engine by changing one enum.
- **Vendor-neutral: no NVIDIA GPU required.** Compute and rendering use only Vulkan (compute shaders written in GLSL). Any Vulkan 1.2+ device works: NVIDIA, AMD, Intel, or even a CPU through Mesa lavapipe (this is what CI uses). Only the zero-copy camera -> PyTorch path (Vulkan -> CUDA interop) needs an NVIDIA GPU + CUDA; without it you get the same images through a CPU read-back.
- **Cross-platform by construction.** Linux is tested on NVIDIA and lavapipe. The Vulkan layer ([`mlx_vulkan`](https://github.com/20niship/mlx_vulkan)) also targets macOS through MoltenVK, but **macOS has not been verified for this repository yet** (camera features there are also unverified).
- **Almost no dependencies.** The C++ core needs only the Vulkan loader/headers, `glslangValidator` (shaders are compiled at first use), and MuJoCo (for MJCF parsing and the CPU reference). No CUDA toolkit, no Python required for the C/C++ libraries, no ML framework. Python bindings add nanobind; training adds PyTorch + rsl_rl. Everything else (VMA, doctest) is fetched by CMake.
- **Faster than Isaac Lab and Genesis.** Go2 `env.step` on an RTX 3060: 3.4x Isaac Lab and 1.6x Genesis at 4096 envs (409K vs 120K / 252K env-steps/s), 1.7x Genesis at 16384 envs, ahead of Isaac Lab at every measured size; with a camera on every env 4.1x Genesis and up to 2.7x Isaac Lab TiledCamera (see Highlights for conditions). Caveats: Isaac Lab / Genesis are earlier single measurements (not re-run in the same session); Genesis measures a bare `scene.step()`; at 256 envs we are only on par with Genesis (63K vs 60K).
- **Built for RL throughput.** Hundreds of thousands of env-steps/s on a mid-range GPU, batched cameras, observations/rewards computed on the training device, one device-to-host transfer per step.

## Highlights (RTX 3060 12 GB unless noted)

Unitree Go2, flat ground, dt 0.005 x 4 substeps, `env.step` with observations and rewards, `device=cuda`:

| Envs | This repo (default) | This repo (fast mode) | Isaac Lab | Genesis |
|---|---|---|---|---|
| 4,096 | 409K | **480K** | ~120K | ~252K |
| 16,384 | 526K | **649K** | - | ~313K |

env-steps/s. Fast mode is `self_collision=False, max_condim=3` (what Isaac Lab/Genesis use); the default keeps the exact model. Isaac Lab / Genesis values are earlier single measurements and not re-run in the same session; Genesis measures a bare `scene.step()` while ours includes observation and reward. Details and caveats: [Detailed performance log](#detailed-performance-log).

### vs MuJoCo Warp (NVIDIA's GPU port of MuJoCo), measured in the same session

Same model (menagerie `go2_mjx` on a flat floor), dt 0.005 x 4 substeps, random position-control actions, median of 30 control steps, env-steps/s, RTX 3060, each size in a fresh process. "Physics only" is `sim.step` without observations and rewards, which is what MuJoCo Warp (and Genesis) measure; "env.step" adds observations and rewards.

| Envs | This repo, physics only | This repo, `env.step` | MuJoCo Warp (xml default: Newton, 1 iteration) | MuJoCo Warp (16 iterations, as ours) |
|---|---|---|---|---|
| 1,024 | **291K** | 225K | 45K | 42K |
| 4,096 | **501K** | 406K | 83K | 139K |
| 16,384 | **503K** | 466K | 46K | 318K |

`--no-self-collision` (the Isaac Lab / Genesis setting) raises ours to 309K / 523K / 517K (physics only). Caveats:

- MuJoCo Warp 4.0.1 targets CUDA 13; this machine's driver is CUDA 12.7, so `warp-lang` is pinned to 1.17.0. Allocation is `nconmax=16`, `njmax=64` per world (largest use seen: 56 constraints per world). The MuJoCo Warp numbers are untuned and not all that stable across iteration settings: at 16,384 envs 1 iteration is 7x slower than 16 (reproducible, 46K vs 310-320K; the robot sinks lower with 1 iteration and has more contacts). MJX (JAX) was not measured.
- Speed only. No accuracy comparison against MuJoCo C was made for MuJoCo Warp; ours is in the table below.
- The GPU is shared with other jobs on this machine, so values move by about 10%. Isaac Lab and Genesis were not re-run (their environments are no longer installed); their numbers above are the earlier single measurements.

```bash
uv venv --python 3.12 ~/mjwarp/.venv && VIRTUAL_ENV=~/mjwarp/.venv uv pip install mujoco-warp "warp-lang==1.17.0"
uv run python scripts/bench_physics_vs_others.py --envs 1024 4096 16384 --only mjmlx_bare   # likewise mjmlx, mjwarp (--mjwarp-iterations 16)
```

Camera RL step (physics x4 substeps + render + RGB/depth as CUDA tensors, 1 camera/env, 64 px, 4096 envs, front view): **337,618 env-steps/s** (Genesis past value: 82,900). Third-person orbit view (whole robot visible): 124,415.

**Performance charts**

![physics](https://github.com/user-attachments/assets/05cc46e9-9091-4535-a864-6a232b6297a8)

![vs MuJoCo Warp](https://github.com/user-attachments/assets/2c8e598d-588f-4934-899e-55143ed920a9)

![camera vs Genesis](https://github.com/user-attachments/assets/6c4644cc-8e79-4f4e-a3f0-14c5ec7eca8e)

![camera vs Isaac](https://github.com/user-attachments/assets/cac72f15-a227-49ae-a040-4c669b0d8ed4)

Media for this README and the project page (training videos, CPU vs GPU comparison, camera views) is **not stored in the repository**; it is attached on GitHub and referenced by URL.

https://github.com/user-attachments/assets/2575874e-0fe2-4981-a96e-c1e924d22542

*CPU (MuJoCo C, left) vs GPU (right) for piper / openarm / go2 / g1, produced by `just check-video-export`. Arms agree; legged robots diverge after landing because contact solutions are chaotic, so we compare single-step error instead (below).*

## Build

Requirements: a Vulkan driver (NVIDIA tested; lavapipe `mesa-vulkan-drivers` is used in CI and needs no GPU), the Vulkan headers/loader and `glslangValidator` (`libvulkan-dev glslang-tools` on Debian/Ubuntu), CMake >= 3.25, a C++20 compiler, and MuJoCo (`pip install mujoco`, found automatically, or `-DMUJOCO_ROOT=...`). [just](https://github.com/casey/just) and [uv](https://github.com/astral-sh/uv) are optional conveniences used by the recipes below. Runtime link dependencies of `libmjmlx`: `libvulkan` and `libmujoco` only. macOS via MoltenVK is untested.

```bash
git clone --recursive https://github.com/20niship/MuJoCo-MLX-Cpp
cd MuJoCo-MLX-Cpp
just build                      # libmjmlx + libmjb + tests/benchmarks (Release) into build/
just build-python               # also installs the Python package `vkmujoco` (nanobind + scikit-build-core)
```

Equivalent raw CMake: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DMJMLX_BUILD_PYTHON=ON && cmake --build build -j`.

CMake options: `MJMLX_BUILD_TESTS`, `MJMLX_BUILD_BENCHMARKS`, `MJMLX_BUILD_EXAMPLES`, `MJMLX_BUILD_PYTHON` (default OFF), `MJMLX_BUILD_CAMERA` (default ON; OFF turns `mjmlx_camera_*` into stubs).

Fetch robot models (mesh MJCFs for piper/openarm/go2/g1/h1, git-ignored):

```bash
just fetch-models
```

## Trained policies

Training, evaluation and comparison code lives in [`scripts/examples/`](scripts/examples/) (Go2, G1, HalfCheetah); how to run it, the choice of reference implementations, differences from Isaac Lab and all training results are in **[scripts/examples/README.md](scripts/examples/README.md)** ([日本語](scripts/examples/README_JA.md)).

<table>
<tr>
<td align="center"><video src="https://github.com/user-attachments/assets/6b5d210c-da5b-401b-8277-287d5a3b85c3" controls muted width="400"></video><br><b>Go2</b>, flat ground, cmd_vel tracking (work in progress: tracks backward / left / right, stays still for forward)</td>
<td align="center"><video src="https://github.com/user-attachments/assets/1c79aae8-1332-4f13-b42b-4f6cc9de6615" controls muted width="400"></video><br><b>G1</b>, forward-only task, 20 s, side view (walks straight; speed stays below the command)</td>
</tr>
</table>

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
model = vkmujoco.load_model(path, self_collision=False, max_condim=3)
env = MjmlxGo2Env(4096, ..., self_collision=False, max_condim=3)
```

Go2 4096 / 16384 envs, env-steps/s: default 409K / 528K; `max_condim=3` 468K / 639K; both options 481K / 648K.

## Camera sensor (Vulkan, batched GPU sim only)

`<camera>` / `<light>` / geom rgba from the MJCF; Phong shading, RGB + depth, hfield and 2D textures (plane/box/hfield), spatial tendons drawn as cylinders. GPU-side frustum culling and LOD (mesh vertex clustering, `MJMLX_CAM_LOD_K`), one `vkCmdDrawIndexedIndirect` for all envs and cameras.

```python
cam = vkmujoco.create_camera(sim, 128, 128)  # sim from create_batched(use_gpu=True)
cam.render()
rgb, depth = cam.rgb(), cam.depth()  # (num_envs, ncam, H, W, 3) uint8 / (..., H, W) float32 [m]

from mjmlx_camera_cuda import CudaCamera  # Vulkan -> CUDA external-memory interop (Linux)

gcam = CudaCamera.try_create(cam)  # None if unsupported -> fall back to the CPU readback above
cam.render()
rgba, depth = gcam.read()  # CUDA tensors, no CPU copy
```

Not supported: shadows, translucency, lights attached to moving bodies, trackcom (approximated by the body origin). Interop needs Linux + `VK_KHR_external_memory_fd` + CUDA torch; set `MKX_VULKAN_DEVICE_INDEX` to match the CUDA device on multi-GPU hosts. Camera needs a graphics+compute queue, `multiDrawIndirect`, `drawIndirectFirstInstance`, `shaderClipDistance` (only `create_camera` fails otherwise). Examples: `scripts/examples/go2/example_go2_camera.py`, comparison with MuJoCo's renderer: `scripts/compare_camera_mujoco.py`, speed: `scripts/bench_camera*.py`.

## Benchmarks and comparisons

Check that nothing else (ollama, desktop compositing, other GPU jobs) is using the GPU; it changes timings by 2x. Compare commits/settings **alternately within one session**, not across days.

```bash
uv run python scripts/bench_go2_step.py --envs 4096        # env.step speed
MJMLX_PROF=1 uv run python scripts/bench_go2_step.py       # per-stage GPU/host time
uv run python scripts/bench_physics_vs_others.py ...                         # this repo / MuJoCo Warp / Isaac Lab / Genesis
uv run python scripts/bench_camera_vs_isaac.py             # camera vs Isaac Lab TiledCamera
PYTHONPATH=<torch-cu126> uv run python scripts/bench_camera_rl.py --camera front --envs 64 256 1024 4096 --sizes 64 128
uv run python scripts/examples/go2/compare_cpu_gpu_go2.py               # 1-step difference vs MuJoCo C
just check-video-export                                                      # CPU|GPU tiled comparison video
just bench                                                                   # regression check against benchmarks/baseline.csv
```

Go2 1-control-step difference from MuJoCo C (joint angles): mean 3.6e-5 rad, p95 8e-7 rad (default); mean 1.4e-5 rad with `--max-condim 3` applied to both sides. Isaac Lab setup, camera tables and GPU clock effects: [Detailed performance log](#detailed-performance-log).

## Detailed performance log

Provisional values, RTX 3060 12 GB unless noted. The full Japanese log is in [README_JA.md](README_JA.md#性能測定ログ詳細). Compare runs **within one session, alternating**; GPU clock state changes results by up to 2x.

### Isaac Lab setup (for the comparison)

```bash
uv venv --python 3.11 ~/isaac/.venv && source ~/isaac/.venv/bin/activate
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/isaac/IsaacLab && cd ~/isaac/IsaacLab
uv pip install "setuptools<81" wheel && uv pip install --no-build-isolation flatdict==4.0.1   # install first: isaaclab.sh --install fails building flatdict
./isaaclab.sh --install rsl_rl
OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless --max_iterations 20
```

Training iteration (4096 envs x 24 steps): this repo ~1.3 s vs Isaac Lab ~2.2 s.

### GPU clock

Without a CUDA context the SM clock stays near 550 MHz under pure Vulkan compute; with `device=cuda` (as in training) torch activity raises it to ~1140-1840 MHz, so the same code is 1.5-1.6x faster. Always benchmark with `device=cuda`, and compare against Genesis (which runs on CUDA) only that way.

### Where the time goes (Go2, 4096 envs)

The solver is ~60-75% of GPU time, bound by shared-memory instruction count (18 threads/env). Kinematics, forward, collision and Euler together take ~0.9 ms per substep. The per-step read-back is one packed transfer.

### Camera-in-the-loop RL (Go2 flat, 1 camera/env, RGB+depth as CUDA tensors, `device=cuda`, median of 3 alternating runs)

Front view, env-steps/s (physics x4 + render + Vulkan->CUDA interop):

| envs | px | physics only | render only (env-frames/s) | this repo | Genesis (past) | ratio |
|---|---|---|---|---|---|---|
| 64 | 64 | 29,993 | 296,503 | **24,273** | 10,400 | 2.3x |
| 256 | 64 | 102,601 | 849,761 | **76,307** | 32,100 | 2.4x |
| 1024 | 64 | 269,492 | 1,698,282 | **199,027** | 64,400 | 3.1x |
| 4096 | 64 | 456,442 | 2,112,591 | **337,618** | 82,900 | 4.1x |
| 4096 | 128 | 473,909 | 1,285,141 | **273,812** | 66,100 | 4.1x |

Third-person orbit view (whole robot visible, ~400K triangles/env): 64 px / 4096 envs 124,415 (Genesis 82,900, 1.5x); 128 px / 4096 envs 113,679 (Genesis 66,100, 1.7x).

Against Isaac Lab TiledCamera (physics x4 + render, 64 px): 64 envs 4,180 vs 2,238; 256 envs 10,972 vs 4,133; 1024 envs 17,604 vs 7,278; 4096 envs 19,608 (Isaac timed out for 1024x128 and 4096). Camera placement and FOV match; shading and clip planes differ (Phong vs minimal RTX).

Per step at 4096 envs (ms): physics 9.0; render 2.0 (front 64 px) / 23-25 (orbit), about half of that vertex processing; pack pass 0.5-1.8; tensor handoff 0.01.

LOD default: `MJMLX_CAM_LOD_K` = 16 below 128 px height, 4 at >= 128 px. Versus MuJoCo's renderer (third person, 16 envs): 64 px K=16 IoU 0.960, depth error 28.2/75.2 mm (mean/p95); 128 px K=4 IoU 0.965, 14.9/37.9 mm.

Interop is Linux + `VK_KHR_external_memory_fd` + CUDA only (sync via `vkQueueWaitIdle`); elsewhere RGB/depth are read back through the CPU. Next ideas: meshlet culling, back-face culling, Hi-Z occlusion.

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
scripts/           benchmarks, camera/sensor helpers, tools
scripts/examples/  Go2 / G1 / HalfCheetah environments, training, playback, CPU-GPU checks (see scripts/examples/README.md)
tests/             C++ tests, fixture MJCFs
docs/              index.html (GitHub Pages)
extern/mlx_vulkan  Vulkan tensor backend (submodule)
```

## Contributing rules

Images and videos must never be committed (they are git-ignored). Attach them on GitHub (PR/issue) and reference the `user-attachments` URL from the README or `docs/index.html`.

## License

Apache 2.0
