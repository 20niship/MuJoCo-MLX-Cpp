# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Two C++ shared libraries:

- **`libmjmlx.dylib`** — MuJoCo physics pipeline reimplemented on Metal GPU using MLX C++
- **`libmjb.dylib`** — Unified dual-backend C API that dispatches to either MuJoCo C (CPU) or MuJoCo-MLX (GPU)

Designed for consumption from Python (via nanobind), Unity/C# (via P/Invoke), or any language with C FFI.

## Performance

Humanoid benchmarks on Apple M4 Max (40-core GPU, 64 GB unified memory):

| Envs | Steps/sec | Notes |
|------|-----------|-------|
| 256 | 15,027 | |
| 1,024 | 68,753 | |
| 4,096 | 255,362 | |
| 8,192 | **331,207** | Peak throughput |

Architecture: `Metal kinematics -> compile(vmap(forward)) -> Metal Euler`

Validated training: **5,678 reward** on Gymnasium Humanoid-v5 (12.6x MuJoCo C baseline), 70K SPS at 8192 envs.

## Architecture

```
libmjb.dylib (unified dual-backend API)
    |
    +-- mjb_* C API (47 functions)
    |     Backend selection: MJB_BACKEND_CPU or MJB_BACKEND_MLX
    |     float* interface everywhere (CPU does double->float conversion)
    |
    +-- CPU backend: wraps MuJoCo C (mj_step, mj_forward, ...)
    |     Batched: GCD thread pool over N independent mjData*
    |
    +-- MLX backend: wraps libmjmlx (mjmlx_step, mjmlx_forward, ...)
          Batched: compiled vmap Metal GPU pipeline

libmjmlx.dylib (GPU physics core)
    |
    +-- Physics pipeline (C++ / MLX)
    |     io, math, smooth, collision, constraint, solver, forward,
    |     passive, support, scan
    |
    +-- Metal kernels
    |     Kinematics FK, fused Euler (source-generated MSL)
    |
    +-- Batched simulation (331K SPS)
    |     Metal kin -> compile(vmap(forward)) -> Metal euler
    |
    +-- Differentiable step (planned)
    |     grad(step) for empowerment / model-based RL
    |
    +-- C API (mjmlx.h, 48 functions)
          Opaque handles, float* data exchange (unified memory)
```

## Building

Requirements: macOS 14+, Apple Silicon, CMake >= 3.25, MLX and MuJoCo (via pip).

```bash
pip install mlx mujoco

cmake -B build
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

This builds both `libmjmlx.dylib` and `libmjb.dylib`. The build auto-detects MLX and MuJoCo from pip.

To override paths:

```bash
cmake -B build -DMLX_ROOT=/path/to/mlx -DMUJOCO_ROOT=/path/to/mujoco
```

### Python bindings (nanobind)

```bash
cmake -B build -DMJMLX_BUILD_PYTHON=ON
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

### Running tests

```bash
# Full suite (162 tests across 16 suites)
./run_tests.sh /path/to/humanoid.xml

# Or via CTest
cd build && cmake .. -DMJMLX_TEST_MODEL=/path/to/humanoid.xml && ctest --output-on-failure

# Conformance tests only (Phase 1 + Phase 2)
cd build && ctest -L correctness --output-on-failure
```

## Unified C API (mjb.h)

The `mjb_*` API lets you pick a backend at creation time:

```c
#include <mjmlx/mjb.h>

// GPU physics (Metal, float32, 70K+ SPS)
MjbBackend* gpu = mjb_create_backend(MJB_BACKEND_MLX);
MjbModel* model = mjb_load_model(gpu, "humanoid.xml");
MjbData* data = mjb_make_data(model);
mjb_step(model, data);

// CPU physics (MuJoCo C, double, thread-safe)
MjbBackend* cpu = mjb_create_backend(MJB_BACKEND_CPU);
MjbModel* cpu_model = mjb_load_model(cpu, "humanoid.xml");

// Batched simulation (both backends)
MjbBatchedConfig config = { .num_envs = 8192 };
MjbBatchedSim* sim = mjb_batched_create(model, &config);
mjb_batched_step(sim, controls);
const float* qpos = mjb_batched_get_qpos(sim, &n);
```

See [`include/mjmlx/mjb.h`](include/mjmlx/mjb.h) for the full API.

## MLX Physics API (mjmlx.h)

Lower-level API for direct GPU physics access:

```c
#include <mjmlx/mjmlx.h>

MjmlxModel* model = mjmlx_load_model("humanoid.xml");
MjmlxData* data = mjmlx_make_data(model);
mjmlx_step(model, data);
const float* qpos = mjmlx_get_qpos(data, &n);  // zero-copy unified memory
```

See [`include/mjmlx/mjmlx.h`](include/mjmlx/mjmlx.h) for the full API.

## Conformance

### Phase 1: Synth Physics Foundation
- **Gravity compensation** (`body_gravcomp` / `qfrc_gravcomp`) — validated against MuJoCo C
- **Per-body contact forces** (`cfrc_ext` via `rne_post_constraint`) — integrated into forward pipeline
- **Exclude signature** collision filtering (`nexclude`, `exclude_signature`) — matches MuJoCo C encoding
- **Model validation warnings** at load time for unsupported features (mesh, box, tendons, etc.)
- **High-DOF verification** (nv=67) — perfect match with MuJoCo C, stable through 100+ steps

### Phase 2: Contact Friction
- **Pyramidal friction (condim=3)** — 4 pyramid edge rows per contact, D/aref match MuJoCo C within 0.001%
- **Pyramidal friction (condim=4,6)** — torsion + rolling friction, 6/10 rows per contact, D values identical

47 conformance tests across 7 test suites, all validated against MuJoCo C reference.

See [CONFORMANCE.md](CONFORMANCE.md) for the full gap analysis and feature matrix.

## Consumers

- **[MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX)** (Python) — uses nanobind extension from this repo
- **MuJoCo-MLX-Unity** (C#, planned) — ships `libmjb.dylib` + P/Invoke bindings

## License

Apache 2.0
