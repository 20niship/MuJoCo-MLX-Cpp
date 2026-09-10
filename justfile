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
    cd build && ./test_math_full && ./test_linalg_full

# Configure + build the MKX (Vulkan) backend into build-mkx/
build-mkx:
    mkdir -p build-mkx
    cmake -S . -B build-mkx -DMJMLX_TENSOR_BACKEND=MKX \
        {{ if mujoco_root != "" { "-DMUJOCO_ROOT=" + mujoco_root } else { "" } }}
    cmake --build build-mkx -j

# Build the MKX backend then run its tests (requires a working Vulkan device)
check-ci-mkx: build-mkx
    cd build-mkx && ./test_math_full && ./test_linalg_full

# Build then run the benchmark regression check
bench: build
    ./scripts/bench_check.sh

# Build the MKX backend then run the same benchmarks (separate history file, for MLX-vs-MKX comparison)
bench-mkx: build-mkx
    BUILD_DIR=build-mkx HISTORY_CSV=benchmarks/history-mkx.csv ./scripts/bench_check.sh

# Download external benchmark robot models (Go2, H1); bench/bench-mkx already do this automatically
fetch-models:
    ./scripts/fetch_models.sh

# Remove build directory
clean:
    /bin/rm -rf build
