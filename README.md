# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

A C++ shared library (`libmjmlx.dylib`) that implements the MuJoCo physics pipeline on Metal GPU using MLX's C++ API. Usable from any language via its C API -- designed for both Python (via nanobind) and Unity/C# (via P/Invoke).

## Status

**Phase 1d complete -- GPU throughput parity achieved (2x target).**

### Performance (humanoid, Apple Silicon GPU)

| Envs | Steps/sec | vs Python 334K target |
|------|-----------|-----------------------|
| 1 | 159 | -- |
| 64 | 10,078 | -- |
| 256 | 40,121 | 12% |
| 1,024 | 151,385 | 45% |
| 4,096 | 540,246 | **162%** |
| 8,192 | **689,255** | **206%** |

Architecture: `Metal kinematics -> compile(vmap(forward)) -> Metal Euler`

### Modules

| Module | Status | Description |
|--------|--------|-------------|
| `io.cpp` | Done | Model loading (MuJoCo C -> MLX arrays), data initialization, ModelCache |
| `math.cpp` | Done | Quaternion ops, spatial algebra, collision geometry helpers |
| `smooth_vmap.cpp` | Done | Vmap-compatible COM, CRB, GPU-native M^{-1}, COM vel, RNE |
| `forward_vmap.cpp` | Done | Vmap-compatible forward pipeline with vectorized transmissions/passive |
| `constraint_vmap.cpp` | Done | Vmap-compatible collision detection + constraint generation |
| `solver_vmap.cpp` | Done | CG solver with GPU M^{-1} preconditioner (no CPU linalg) |
| `batched.cpp` | Done | Hybrid pipeline: Metal kernels + compile(vmap(forward)) + C API |
| `smooth.cpp` | Done | Scalar fallback: kinematics, Cholesky/LDL, RNE, transmission |
| `collision.cpp` | Done | 5 primitive pairs (plane/sphere/capsule) with broadphase |
| `constraint.cpp` | Done | Joint limits + contact constraints with KBI impedance |
| `solver.cpp` | Done | Scalar fallback: CG with Polak-Ribiere + Newton linesearch |
| `forward.cpp` | Done | Scalar fallback: full pipeline + Euler integration |
| `nn.cpp` | Stub | Actor-critic neural network (MLX C++) |
| `ppo.cpp` | Stub | PPO training loop |

### Validated

- Humanoid model loads correctly (nq=28, nv=27, nu=21, nbody=17, ngeom=20)
- Forward kinematics produces correct body positions (torso at z=1.282)
- Free-fall simulation stable over 100+ steps (z drops 1.282 -> 0.068)
- Physics deterministic across all batch sizes (1 to 8192 envs)
- Per-env reset while others continue, correct state isolation
- Metal kernel source generators: kinematics FK + fused Euler (same MSL as Python)

### Key optimizations (Phase 1d)

- **GPU-native M^{-1}**: Neumann series `D^{-1}(I + N + N^2 + N^3)` replaces CPU-only `cholesky_inv`, keeping entire graph on GPU
- **mx::compile fusion**: With no CPU sync points, the full pipeline compiles into fused Metal dispatches
- **Vectorized tree traversals**: Scatter-matrix accumulation for COM/CRB replaces per-body loops
- **Precomputed ModelCache**: Batched actuator moment matrix, passive stiffness arrays eliminate per-DOF loops

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
    +-- Batched simulation [DONE - 689K SPS]
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

```bash
./build/test_io /path/to/humanoid.xml
./build/test_smooth /path/to/humanoid.xml
./build/test_forward /path/to/humanoid.xml 100
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

// Batched simulation (Metal GPU) -- 689K SPS on humanoid
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
