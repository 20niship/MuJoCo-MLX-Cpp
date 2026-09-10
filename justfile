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

# Remove build directory
clean:
    /bin/rm -rf build
