# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

A C++ shared library (`libmjmlx.dylib`) that implements the MuJoCo physics pipeline on Metal GPU using MLX's C++ API. Usable from any language via its C API -- designed for both Python (via nanobind) and Unity/C# (via P/Invoke).

## Status

**Phase 1a: Repo scaffolding complete.** All 43 C API functions declared, library compiles and exports clean symbols. Physics modules are stubs pending port from [MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX) Python.

### Validated in Phase 0 spike:

- `mx::vmap` vectorizes correctly over batch dimension
- `mx::compile` gives 3.4x JIT speedup
- `mx::grad` computes correct autodiff
- `compile(vmap(step))`: **204M SPS** on 8192 envs (double pendulum)
- Custom Metal kernels dispatch from C++
- `grad(vmap(step))`: differentiable 10-step batched rollout works
- C API shared library: 25M SPS through C API, clean symbol exports

## Architecture

```
libmjmlx.dylib (this repo)
    |
    +-- Physics pipeline (C++ / MLX)
    |     types, io, math, smooth, collision, constraint, solver, forward
    |
    +-- Metal kernels
    |     kinematics, linalg, euler (ported from Python inline MSL)
    |
    +-- Batched simulation
    |     compile(vmap(step)) for N parallel environments
    |
    +-- Differentiable step
    |     grad(step) for empowerment / model-based RL
    |
    +-- Actor-critic neural network (MLX C++)
    |     forward, backward, PPO update
    |
    +-- C API (mjmlx.h)
    |     43 functions, opaque handles, float* data exchange
    |
    +-- nanobind Python extension (planned)
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

## C API

See [`include/mjmlx/mjmlx.h`](include/mjmlx/mjmlx.h) for the full API. Key functions:

```c
// Load model
MjmlxModel* model = mjmlx_load_model("humanoid.xml");
MjmlxModelInfo info = mjmlx_model_info(model);

// Single-env simulation
MjmlxData* data = mjmlx_make_data(model);
mjmlx_step(model, data);
const float* qpos = mjmlx_get_qpos(data, &n);

// Batched simulation (Metal GPU)
MjmlxBatchedConfig config = { .num_envs = 8192, .use_gpu = 1 };
MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
mjmlx_batched_step(sim, controls);

// Neural network + PPO training
MjmlxActorCritic* nn = mjmlx_nn_create(&nn_config);
MjmlxPPOTrainer* trainer = mjmlx_ppo_create(sim, nn, &ppo_config);
float avg_reward = mjmlx_ppo_iterate(trainer);
```

## Consumers

- **[MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX)** (Python) -- will use nanobind extension from this repo
- **MuJoCo-MLX-Unity** (C#) -- will ship `libmjmlx.dylib` in Plugins/ and use P/Invoke

## License

Apache 2.0
