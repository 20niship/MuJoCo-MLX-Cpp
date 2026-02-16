# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

A C++ shared library (`libmjmlx.dylib`) that implements the MuJoCo physics pipeline on Metal GPU using MLX's C++ API. Usable from any language via its C API -- designed for both Python (via nanobind) and Unity/C# (via P/Invoke).

## Status

**Phase 1d complete -- GPU batched simulation with full contact physics.**

### Performance (humanoid, Apple Silicon GPU)

| Envs | Steps/sec | Notes |
|------|-----------|-------|
| 1 | 46 | Single-env scalar pipeline |
| 64 | 2,634 | |
| 256 | 11,226 | |
| 1,024 | 44,989 | |
| 4,096 | 164,140 | |
| 8,192 | **198,263** | Full contact physics |

Architecture: `Metal kinematics -> compile(vmap(forward)) -> Metal Euler`

### Pipeline Architecture

The batched simulation uses a 3-phase hybrid pipeline optimized for Apple Silicon:

```
Phase 1: Metal FK kernel      (single GPU dispatch, all B envs)
    - Forward kinematics: qpos -> xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat
    - One fused Metal kernel replaces ~130 separate dispatches

Phase 2: compile(vmap(forward_dynamics))
    - COM, CRB, mass matrix, collision, constraints, solver, RNE, actuation
    - Pure MLX array ops (required for vmap compatibility)
    - mx::compile fuses the computation graph for efficient execution
    - Newton solver uses GPU-native Cholesky (cholesky_gpu) for direct solves
    - CG solver uses Cholesky-preconditioned Polak-Ribiere

Phase 3: Metal Euler kernel   (single GPU dispatch, all B envs)
    - Built-in Cholesky + solve + velocity update + position integration
    - One fused Metal kernel for the entire integration step
```

This matches the approach used by Google's MJX (JAX-based MuJoCo), which uses
`jax.vmap + jax.jit` over pure JAX ops. Our implementation goes further by using
custom Metal kernels for the two hottest phases (kinematics and integration),
while keeping Phase 2 as pure array ops for vmap compatibility.

### Modules

| Module | Status | Description |
|--------|--------|-------------|
| `io.cpp` | Done | Model loading (MuJoCo C -> MLX arrays), data initialization, ModelCache |
| `math.cpp` | Done | Quaternion ops, spatial algebra, collision geometry helpers |
| `smooth_vmap.cpp` | Done | Vmap-compatible COM, CRB, GPU-native Cholesky factorization, COM vel, RNE |
| `forward_vmap.cpp` | Done | Vmap-compatible forward pipeline with vectorized transmissions/passive |
| `constraint_vmap.cpp` | Done | Vmap-compatible collision detection + constraint generation |
| `solver_vmap.cpp` | Done | Newton + CG constraint solver with GPU Cholesky and warmstart |
| `batched.cpp` | Done | Hybrid pipeline: Metal kernels + compile(vmap(forward)) + C API |
| `smooth.cpp` | Done | Scalar fallback: kinematics, Cholesky/LDL, RNE, transmission |
| `collision.cpp` | Done | 5 primitive pairs (plane/sphere/capsule) with broadphase |
| `constraint.cpp` | Done | Joint limits + contact constraints with KBI impedance |
| `solver.cpp` | Done | Scalar fallback: CG with Polak-Ribiere + Newton linesearch |
| `forward.cpp` | Done | Scalar fallback: full pipeline + Euler integration |
| `nn.cpp` | Stub | Actor-critic neural network (MLX C++) |
| `ppo.cpp` | Stub | PPO training loop |

### Key Optimizations

- **GPU-native Cholesky** (`cholesky_gpu`): Column-vectorized Cholesky decomposition using pure MLX ops (no CPU sync). Works inside `mx::vmap` for batched Newton solver and mass matrix factorization. Matches Python `gpu_cholesky()` from gpu_linalg.py.
- **Metal kernel fusion**: Kinematics and Euler integration run as single Metal dispatches, replacing hundreds of individual GPU kernel launches.
- **mx::compile graph fusion**: Phase 2 (forward dynamics) is wrapped in `mx::compile`, which fuses the MLX computation graph for efficient Metal execution.
- **Vectorized tree traversals**: Scatter-matrix accumulation for COM/CRB replaces per-body loops.
- **Precomputed ModelCache**: Batched actuator moment matrix, passive stiffness arrays, tree masks eliminate per-DOF loops.
- **Newton + CG solver**: Newton mode (default for humanoid) constructs H = M + J^T D J and Cholesky-solves for exact search direction in 1-2 iterations. CG mode uses M^{-1} preconditioning with Polak-Ribiere updates for 4-10 iterations.
- **Warmstart**: Solver compares warmstart vs smooth acceleration costs, picking whichever converges faster.

### Validated

- Humanoid model loads correctly (nq=28, nv=27, nu=21, nbody=17, ngeom=20)
- Forward kinematics produces correct body positions (torso at z=1.282)
- Mass matrix is symmetric (symmetry error = 0.0, verified per step)
- GPU Cholesky factorization: max|L@L^T - M| < 1e-5 for humanoid mass matrix
- GPU vs CPU batched pipeline: max|qvel_diff| < 0.1 (1 step), stable over 20+ steps
- Free-fall simulation stable over 100+ steps (z drops 1.282 -> 0.068)
- Full contact simulation stable over 100+ steps with bounded velocities
- Physics deterministic across all batch sizes (1 to 8192 envs)
- Per-env reset while others continue, correct state isolation
- All 64 envs produce identical output with no control input
- Metal kernel source generators: kinematics FK + fused Euler (same MSL as Python)
- 99 tests across 8 test suites: math (17), io (9), forward (7), physics (20), collision (7), solver (20), linalg (12), batched (7)

### Phase 0 spike results

- `compile(vmap(step))`: **204M SPS** on 8192 envs (double pendulum)
- Custom Metal kernels dispatch from C++
- `grad(vmap(step))`: differentiable 10-step batched rollout
- C API shared library: 25M SPS, clean symbol exports
- Neural network in C++: 14.4M inferences/sec (batch 4096)

## Architecture

```
libmjmlx.dylib (this repo)
    |
    +-- Physics pipeline (C++ / MLX) [DONE]
    |     io, math, smooth, collision, constraint, solver, forward,
    |     passive, support, scan
    |
    +-- Metal kernels [DONE]
    |     kinematics FK, fused Euler (source-generated MSL)
    |
    +-- Batched simulation [DONE - 198K SPS]
    |     Metal kin -> compile(vmap(forward)) -> Metal euler
    |     Full C API: create, step, reset, get_state
    |
    +-- Differentiable step [PLANNED]
    |     grad(step) for empowerment / model-based RL
    |
    +-- Actor-critic neural network (MLX C++) [PLANNED]
    |     forward, backward, PPO update
    |
    +-- C API (mjmlx.h) [DONE - 43 functions]
    |     opaque handles, float* data exchange (unified memory)
    |
    +-- nanobind Python extension [PLANNED]
          Exposes C++ as Python mx.array API
```

## Building

Requirements: macOS 14+, Apple Silicon, CMake >= 3.25, MLX (via pip), MuJoCo >= 3.0 (via pip).

```bash
pip install mlx mujoco

cmake -B build
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

The build auto-detects MLX and MuJoCo from pip installations. To override:

```bash
cmake -B build \
  -DMLX_ROOT=/path/to/mlx \
  -DMUJOCO_ROOT=/path/to/mujoco
```

### Running tests

Run the full test suite:

```bash
# All tests (pass model path for model-dependent tests)
./run_tests.sh /path/to/humanoid.xml

# Or use CTest
cd build
cmake .. -DMJMLX_TEST_MODEL=/path/to/humanoid.xml
ctest --output-on-failure

# Individual tests
./build/test_math_full                      # 17 math tests
./build/test_linalg_full /path/to/humanoid.xml  # 12 Cholesky/solve tests
./build/test_solver_full /path/to/humanoid.xml  # 20 solver tests
./build/test_batched_diag /path/to/humanoid.xml # 7 GPU accuracy tests
```

Test suite: 99 tests across 8 suites covering math, I/O, kinematics, physics,
collisions, solver (Newton + CG), linear algebra (Cholesky + solve), and
batched GPU pipeline accuracy.

### Benchmarking

```bash
./build/test_batched /path/to/humanoid.xml 8192 100   # 8192 envs, 100 steps
```

## C API

See [`include/mjmlx/mjmlx.h`](include/mjmlx/mjmlx.h) for the full API. Key functions:

```c
// Load model
MjmlxModel* model = mjmlx_load_model("humanoid.xml");
MjmlxModelInfo info = mjmlx_model_info(model);

// Single-env simulation
MjmlxData* data = mjmlx_make_data(model);
mjmlx_forward(model, data);                     // position + velocity + acceleration
mjmlx_step(model, data);                        // forward + integrate
const float* qpos = mjmlx_get_qpos(data, &n);   // zero-copy (unified memory)
const float* xpos = mjmlx_get_xpos(data, &n);   // body positions

// Batched simulation (Metal GPU) -- 198K SPS on humanoid
MjmlxBatchedConfig config = { .num_envs = 8192, .use_gpu = 1 };
MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
mjmlx_batched_step(sim, controls);

// Neural network + PPO training -- coming in Phase 2
MjmlxActorCritic* nn = mjmlx_nn_create(&nn_config);
MjmlxPPOTrainer* trainer = mjmlx_ppo_create(sim, nn, &ppo_config);
float avg_reward = mjmlx_ppo_iterate(trainer);
```

## Consumers

- **[MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX)** (Python) -- will use nanobind extension from this repo
- **MuJoCo-MLX-Unity** (C#, planned) -- will ship `libmjmlx.dylib` in Plugins/ and use P/Invoke

## License

Apache 2.0
