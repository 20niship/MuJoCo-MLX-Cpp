# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

A C++ shared library (`libmjmlx.dylib`) that implements the MuJoCo physics pipeline on Metal GPU using MLX's C++ API. Usable from any language via its C API -- designed for both Python (via nanobind) and Unity/C# (via P/Invoke).

## Status

**Phase 1d complete -- full physics pipeline + batched simulation.**

| Module | Status | Description |
|--------|--------|-------------|
| `io.cpp` | Done | Model loading (MuJoCo C -> MLX arrays), data initialization |
| `math.cpp` | Done | Quaternion ops, spatial algebra, collision geometry helpers |
| `passive.cpp` | Done | Spring/damper forces |
| `support.cpp` | Done | Mass matrix ops (dense/sparse), Jacobians, xfrc accumulation |
| `smooth.cpp` | Done | Kinematics, COM, CRB, Cholesky/LDL factorization, RNE, transmission |
| `collision.cpp` | Done | 5 primitive pairs (plane/sphere/capsule) with broadphase |
| `constraint.cpp` | Done | Joint limits + contact constraints with KBI impedance |
| `solver.cpp` | Done | CG with Polak-Ribiere + Newton linesearch |
| `forward.cpp` | Done | Full pipeline orchestration + Euler integration |
| `batched.cpp` | Done | Batched sim C API + Metal kernel source generators |
| `nn.cpp` | Stub | Actor-critic neural network (MLX C++) |
| `ppo.cpp` | Stub | PPO training loop |

### Validated:

- Humanoid model loads correctly (nq=28, nv=27, nu=21, nbody=17, ngeom=20)
- Forward kinematics produces correct body positions (torso at z=1.282)
- Gravity forces computed correctly (max |qfrc_bias| = 400.68)
- Free-fall simulation stable over 100+ steps (z drops 1.282 -> 0.068)
- Gravity acceleration matches expected value (0.049 m/s per step)
- Dense Cholesky (CPU) and sparse LDL factorization both implemented
- Batched simulation: 4 envs x 10 steps, per-env reset, state isolation verified
- Metal kernel source generators: kinematics FK + fused Euler (same MSL as Python)

### Phase 0 spike results:

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
    +-- Metal kernels [DONE - source generators]
    |     kinematics, linalg, euler (ported from Python inline MSL)
    |
    +-- Batched simulation [DONE - per-env loop, Metal hybrid pending]
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

// Batched simulation (Metal GPU) -- coming in Phase 1d
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
