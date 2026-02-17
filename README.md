# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Two C++ shared libraries:

- **`libmjmlx.dylib`** -- MuJoCo physics pipeline reimplemented on Metal GPU using MLX C++
- **`libmjb.dylib`** -- Unified dual-backend C API that dispatches to either MuJoCo C (CPU) or MuJoCo-MLX (GPU)

Designed for consumption from Python (via nanobind), Unity/C# (via P/Invoke), or any language with C FFI.

## Performance

Humanoid benchmarks on Apple M4 Max (40-core GPU, 64 GB unified memory):

| Envs | Steps/sec | Notes |
|------|-----------|-------|
| 256 | 15,027 | |
| 1,024 | 68,753 | |
| 4,096 | 255,362 | |
| 8,192 | **331,207** | Peak throughput |

Architecture: `Metal kinematics -> compile(vmap(forward)) -> Metal integration` (Euler in batched, Euler/RK4 in scalar)

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
    +-- Tendon system
    |     Fixed (joint-based) tendons: ten_length, ten_velocity, ten_J
    |     Tendon passive forces, limits, friction (scalar path)
    |     TENDON + SITE actuator transmission
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
# Full suite (237 tests across 28 suites)
./run_tests.sh /path/to/humanoid.xml

# Or via CTest
cd build && cmake .. -DMJMLX_TEST_MODEL=/path/to/humanoid.xml && ctest --output-on-failure

# Conformance tests only
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

All features validated against MuJoCo C reference implementation. **237 tests across 28 test suites, all passing.**

### Phase 1: Synth Physics Foundation
- **Gravity compensation** (`body_gravcomp` / `qfrc_gravcomp`) -- validated against MuJoCo C
- **Per-body contact forces** (`cfrc_ext` via `rne_post_constraint`) -- integrated into forward pipeline
- **Exclude signature** collision filtering (`nexclude`, `exclude_signature`) -- matches MuJoCo C encoding
- **Model validation warnings** at load time for unsupported features
- **High-DOF verification** (nv=67) -- perfect match with MuJoCo C, stable through 100+ steps

### Phase 2: Contact Friction
- **Pyramidal friction (condim=3)** -- 4 pyramid edge rows per contact, D/aref match MuJoCo C within 0.001%
- **Pyramidal friction (condim=4,6)** -- torsion + rolling friction, 6/10 rows per contact, D values identical

### Phase 3: Collision Geometry
- **plane-box** -- multi-contact (up to 4 face vertices), ncon/nefc match MuJoCo C exactly
- **sphere-box** -- closest-point on OBB, with interior fallback
- **capsule-box** -- segment-to-OBB closest approach with iterative refinement
- **box-box** -- SAT (15 axes) with support point contact generation
- **plane-cylinder** -- multi-contact (face center + rim points)
- **sphere-cylinder** -- cylinder-local closest-point with barrel/cap/rim handling
- **capsule-cylinder** -- segment-to-cylinder iterative projection refinement
- **GJK/EPA** -- full convex collision via Gilbert-Johnson-Keerthi + Expanding Polytope Algorithm (scalar: 64-iter GJK + 64-iter EPA; vmap: 32-iter GJK + support-based depth estimation)
- **plane-mesh** -- multi-contact (all penetrating vertices), near-exact MuJoCo C match
- **mesh-mesh** -- GJK/EPA with proper mesh support functions (vertex argmax), not sphere approximation
- **plane-hfield** -- heightfield terrain collision against all geom types, grid-cell triangle tests
- **plane-ellipsoid** -- analytic ellipsoid collision against planes, spheres, capsules
- All collision pair types work in both scalar and vmap (batched) pipelines with proper support functions

### Phase 4: Equality Constraints + DOF Friction
- **Equality constraints (CONNECT/WELD/JOINT)** -- 12 tests, qacc diff ~0 vs MuJoCo C
- **DOF friction loss** -- solver friction clamping with linear zone cost, qacc diff ~1e-6
- Correct constraint ordering: equality -> friction -> limits -> contacts

### Phase 5: Tendon System + Transmission
- **Fixed (joint-based) tendons** -- `ten_length`, `ten_velocity`, `ten_J` match MuJoCo C within 1e-5
- **Tendon passive forces** -- spring + damping via `ten_J^T` projection
- **TENDON transmission** -- actuators through tendons, moment = gear * ten_J
- **SITE transmission** -- actuators at sites, full 6-DOF Jacobian projection
- **Tendon limits** -- constraint rows for tendon length bounds, qacc diff ~1e-3
- **Tendon friction loss** -- friction constraints through tendons, exact match
- **Spatial tendons (wrapping geometry)** -- DEFERRED: MJX supports it but requires ~400 lines of geodesic path computation around spheres/cylinders; most RL models use fixed tendons only
- Fixed tendons + passive forces + limits + friction: both scalar and vmap. Transmission: scalar path

### Phase 6: Actuator Dynamics (scalar + vmap)
- **FILTER dynamics** -- first-order low-pass `da/dt = (ctrl - act) / tau`, Euler integration
- **FILTEREXACT dynamics** -- exact exponential integration of the same ODE
- **INTEGRATOR dynamics** -- pure integration `da/dt = ctrl`
- **Activation clamping** -- `actuator_actlimited` / `actuator_actrange`
- Mixed stateless (NONE) + stateful actuators in the same model
- **MUSCLE dynamics** -- DEFERRED (biomechanical models only)

### Phase 7: Advanced Integrators (scalar path)
- **RK4 (4th-order Runge-Kutta)** -- 4 forward evaluations per step, weighted average; qpos diff ~3e-8 vs MuJoCo C
- **ImplicitFast** -- implicit velocity integration via velocity derivative (`deriv_smooth_vel`): modified mass matrix M' = M - dt * dqfrc/dqvel, Cholesky solve; qvel diff ~1e-6 vs MuJoCo C
- Euler, RK4, Implicit, ImplicitFast all supported in scalar path
- Batched (Metal) pipeline uses Euler only; vmap RK4/Implicit DEFERRED (needs Metal kernel rewrite)

### Deferred Phases
- **Sensors** (Phase 8) -- ~50 sensor types, large surface area; RL training reads joint/body state directly, not XML sensors
- **Differentiable physics** (Phase 9) -- `grad(step)` via `mx::grad`; requires making full pipeline autodiff-compatible, separate project-scale effort
- **Inverse dynamics / constraint islands / noslip / sleep** (Phase 10) -- niche features, not needed for common RL workflows

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Test Suite Summary

| Suite | Tests | What it validates |
|-------|-------|-------------------|
| test_io_full | 9 | Model loading, array shapes, initial state |
| test_forward_full | 7 | Forward dynamics pipeline, step correctness |
| test_physics_full | 20 | Physics accuracy vs MuJoCo C |
| test_solver_full | 11 | Constraint solver (Newton + CG) |
| test_collision_full | 7 | Core collision detection |
| test_math_full | 17 | Math utilities (quaternion, rotation, etc.) |
| test_linalg_full | 11 | Linear algebra (Cholesky, solve, etc.) |
| test_vmap_smooth | 12 | Vmap-compatible smooth dynamics |
| test_gravcomp | 7 | Gravity compensation |
| test_cfrc_ext | 6 | Contact force computation |
| test_exclude | 6 | Collision exclusion filtering |
| test_validation | 8 | Model validation warnings |
| test_high_dof | 6 | High-DOF models (67 DOF) |
| test_friction_condim3 | 7 | Pyramidal friction (condim=3) |
| test_friction_condim46 | 7 | Torsion/rolling friction (condim=4,6) |
| test_collision_box | 6 | Box collision pairs |
| test_collision_cylinder | 6 | Cylinder collision pairs |
| test_collision_mesh | 7 | Mesh/GJK/EPA collisions |
| test_collision_hfield | 7 | Heightfield terrain collisions |
| test_collision_ellipsoid | 7 | Ellipsoid collisions |
| test_equality | 12 | Equality constraints (CONNECT/WELD/JOINT) |
| test_dof_friction | 9 | DOF friction loss + solver clamping |
| test_fixed_tendon | 9 | Fixed tendon system |
| test_transmission | 7 | TENDON + SITE transmission |
| test_tendon_constraint | 6 | Tendon limits + friction loss |
| test_rk4 | 6 | RK4 integrator |
| test_filter_dynamics | 7 | Activation dynamics (FILTER/INTEGRATOR) |
| test_implicit | 7 | ImplicitFast integrator |
| **TOTAL** | **237** | |

## Consumers

- **[MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX)** (Python) -- uses nanobind extension from this repo
- **MuJoCo-MLX-Unity** (C#, planned) -- ships `libmjb.dylib` + P/Invoke bindings

## License

Apache 2.0
