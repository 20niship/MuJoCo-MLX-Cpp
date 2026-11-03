mujoco_root := env_var_or_default("MUJOCO_ROOT", "")
build_dir := env_var_or_default("BUILD_DIR", "build")

# Configure + build (MKX/Vulkan backend)
build:
    mkdir -p {{build_dir}}
    cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE=Release \
        {{ if mujoco_root != "" { "-DMUJOCO_ROOT=" + mujoco_root } else { "" } }}
    cmake --build {{build_dir}} -j

# Build then run the test suite
check: build
    ./scripts/run_tests.sh

# Build then run only the tests that don't depend on the removed single-env pipeline (for CI; requires a working Vulkan device)
# contact_stress(nv=300)はlavapipeで約23分かかりstep1でGPU状態が発散するため、実GPUでは `ctest` 全体を回して確認する。
check-ci: build
    cd {{build_dir}} && ./test_math_full && ./test_linalg_full && ./test_batched_collision_primitives && (! [ -x ./test_camera ] || ./test_camera || [ $? -eq 77 ]) && (! [ -x ./test_camera_tendon ] || ./test_camera_tendon || [ $? -eq 77 ]) && ctest -R gpu_vs_cpu -E "contact_stress|model_" --output-on-failure

# CIと同じDockerイメージ(lavapipe)でcheck-ciを実行する。GPUで回す場合は `just docker-test --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`
docker-test *ARGS:
    docker build -f scripts/Dockerfile -t mjmlx-ci .
    docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/work {{ARGS}} mjmlx-ci

# mesh付き実ロボットMJCF(piper/openarm/go2/g1/h1)をbenchmarks/models/external/へ取得する(git管理外)
fetch-models:
    ./scripts/fetch_models.py

# 実ロボットモデルのCPU/GPU軌道比較test(関節角・body位置姿勢の差)。ローカル専用、CIには含まれない
check-models: fetch-models build
    cd {{build_dir}} && ctest -L models --output-on-failure

# 実ロボットモデルのCPU/GPU挙動をレンダリングし、N行x2列(CPU|GPU)のcaption付き動画を runs/video/compare.mp4 に書き出す
check-video-export: fetch-models build
    #!/usr/bin/env bash
    set -euo pipefail
    E=benchmarks/models/external; M=$E/mujoco_menagerie; out=runs/video; mkdir -p $out
    specs=()
    add() { # name model seconds extra-args...
        name=$1; model=$2; sec=$3; shift 3
        {{build_dir}}/test_gpu_vs_cpu $model --seconds $sec --fps 30 --dump $out/$name.bin "$@" --tol-joint 99 --tol-pos 99 --tol-rot 99 > $out/$name.log
        specs+=("$model:$out/$name.bin")
    }
    add piper $M/agilex_piper/scene.xml 3 --ctrl sin --bodies base_link,link8 
    add openarm $E/openarm_mujoco/v1/scene.xml 3 --ctrl sin --bodies openarm_left_hand_tcp 
    add go2 $M/unitree_go2/scene.xml 3 --lift 0.3 --ctrl sin --bodies base 
    add g1 $M/unitree_g1/scene.xml 3 --lift 0.1 --ctrl sin --bodies pelvis 
    # 既存の.venvを書き換えないよう使い捨て環境で実行する
    uv run --no-project --with mujoco==3.5.0 --with numpy --with pillow python scripts/render_compare.py $out "${specs[@]}"

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
