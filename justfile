mujoco_root := env_var_or_default("MUJOCO_ROOT", "")

# Configure + build (MKX/Vulkan backend)
build:
    mkdir -p build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        {{ if mujoco_root != "" { "-DMUJOCO_ROOT=" + mujoco_root } else { "" } }}
    cmake --build build -j

# Build then run the test suite
check: build
    ./scripts/run_tests.sh

# Build then run only the tests that don't depend on the removed single-env pipeline (for CI; requires a working Vulkan device)
check-ci: build
    cd build && ./test_math_full && ./test_linalg_full && ./test_batched_collision_primitives && ctest -R gpu_vs_cpu --output-on-failure

# Build then run the benchmark regression check (5 repeats, min/max/avg)
bench: build
    BUILD_DIR=build HISTORY_CSV=benchmarks/history.csv ./scripts/bench_check.py

# Build the RL Python extension into build/ using the uv-managed venv
build-python: _venv
    mkdir -p build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_ROOT=.venv/lib/python3.10/site-packages/mujoco \
        -DMJMLX_BUILD_PYTHON=ON -DPython_EXECUTABLE=.venv/bin/python
    cmake --build build --target _mjmlx_rl_native -j

# Train PPO on HalfCheetah via the batched GPU physics. Run build-python first.
train-ppo *ARGS: _venv
    .venv/bin/python scripts/train_ppo.py {{ARGS}}

_venv:
    uv sync --group build

# Remove build directory
clean:
    /bin/rm -rf build
