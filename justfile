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
    ./run_tests.sh

# Build then run only the tests that don't depend on the removed single-env pipeline (for CI)
check-ci: build
    cd build && ./test_math_full && ./test_linalg_full

# Remove build directory
clean:
    /bin/rm -rf build
