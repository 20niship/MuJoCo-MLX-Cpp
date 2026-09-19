mlx_root := env_var_or_default("MLX_ROOT", "/opt/homebrew/Cellar/mlx/0.31.2")
mujoco_root := env_var_or_default("MUJOCO_ROOT", "")

# Configure + build
build:
    mkdir -p build
    cmake -S . -B build -DMLX_ROOT={{mlx_root}} \
        {{ if mujoco_root != "" { "-DMUJOCO_ROOT=" + mujoco_root } else { "" } }}
    cmake --build build -j

# Build then run the test suite
check: build
    ./scripts/run_tests.sh

# Build then run only the tests that don't depend on the removed single-env pipeline (for CI)
check-ci: build
    cd build && ./test_math_full && ./test_linalg_full && ./test_batched_collision_primitives

# Configure + build the MKX (Vulkan) backend into build-mkx/
build-mkx:
    mkdir -p build-mkx
    cmake -S . -B build-mkx -DMJMLX_TENSOR_BACKEND=MKX \
        {{ if mujoco_root != "" { "-DMUJOCO_ROOT=" + mujoco_root } else { "" } }}
    cmake --build build-mkx -j

# Build the MKX backend then run its tests (requires a working Vulkan device)
check-ci-mkx: build-mkx
    cd build-mkx && ./test_math_full && ./test_linalg_full && ./test_batched_collision_primitives

# Build then run the benchmark regression check
bench: build
    ./scripts/bench_check.sh

# Build the MKX backend then run the same benchmarks (separate history file, for MLX-vs-MKX comparison)
bench-mkx: build-mkx
    BUILD_DIR=build-mkx HISTORY_CSV=benchmarks/history-mkx.csv ./scripts/bench_check.sh

# Build then run each benchmark 5x, report min/max/avg (smooths out occasional slow runs)
bench-repeat *ARGS: build
    ./scripts/bench_repeat.sh {{ARGS}}

# Same as bench-repeat but for the MKX backend
bench-repeat-mkx *ARGS: build-mkx
    BUILD_DIR=build-mkx HISTORY_CSV=benchmarks/history-mkx.csv ./scripts/bench_repeat.sh {{ARGS}}

# Download external benchmark robot models (Go2, H1); bench/bench-mkx already do this automatically
fetch-models:
    ./scripts/fetch_models.sh

# Build the RL Python extension (MLX backend) into build/ using the uv-managed venv
build-python: _venv
    mkdir -p build
    cmake -S . -B build \
        -DMLX_ROOT=.venv/lib/python3.10/site-packages/mlx \
        -DMUJOCO_ROOT=.venv/lib/python3.10/site-packages/mujoco \
        -DMJMLX_BUILD_PYTHON=ON -DPython_EXECUTABLE=.venv/bin/python
    cmake --build build --target _mjmlx_rl_native -j

# Build the RL Python extension (MKX/Vulkan backend) into build-mkx/
build-python-mkx: _venv
    mkdir -p build-mkx
    cmake -S . -B build-mkx -DMJMLX_TENSOR_BACKEND=MKX \
        -DMUJOCO_ROOT=.venv/lib/python3.10/site-packages/mujoco \
        -DMJMLX_BUILD_PYTHON=ON -DPython_EXECUTABLE=.venv/bin/python
    cmake --build build-mkx --target _mjmlx_rl_native -j

# Train PPO on HalfCheetah via the batched GPU physics. Run build-python (MLX) or build-python-mkx first; pass --build-dir build-mkx to use the latter.
train-ppo *ARGS: _venv
    .venv/bin/python scripts/train_ppo.py {{ARGS}}

_venv:
    uv sync --group build

# Remove build directory
clean:
    /bin/rm -rf build
