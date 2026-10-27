# MuJoCo-MLX-Cpp

GPU-accelerated MuJoCo physics on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Two C++ shared libraries:

- **`libmjmlx.dylib`** -- MuJoCo physics pipeline reimplemented on GPU using MLX C++
- **`libmjb.dylib`** -- Unified dual-backend C API that dispatches to either MuJoCo C (CPU) or MuJoCo-MLX (GPU)

Designed for consumption from Python (via nanobind), Unity/C# (via P/Invoke), or any language with C FFI.

## Performance

Humanoid stepping throughput on Apple M4 Max (40-core GPU, 64 GB unified memory):

| Envs | Steps/sec | Notes |
|------|-----------|-------|
| 256 | 15,027 | |
| 1,024 | 68,753 | |
| 4,096 | 255,362 | |
| 8,192 | **331,207** | Peak throughput (pure stepping) |

Architecture: `GPU kinematics -> compile(vmap(forward)) -> GPU integration` (Euler in batched, Euler/RK4/ImplicitFast in scalar)

### Training benchmarks (Gymnasium Humanoid-v5, 8192 envs, 32M steps)

| Backend state | Best AvgR | Avg SPS | Wall time | Notes |
|---------------|-----------|---------|-----------|-------|
| Phase 1e (pre-conformance) | 5,678 | 70K | 454s | Tendons silently ignored; unrealistic leg dynamics |
| **Full conformance** | **1,102** | **73K** | **439s** | Hip-knee tendon coupling active; physically correct |

The Phase 1e result was inflated by missing tendon constraints -- the gymnasium humanoid XML has 2 fixed tendons coupling hip and knee joints, which were not implemented in the vmap path at Phase 1e. With tendons active, the humanoid must learn a more realistic gait. The conformance result (1,102) is still 2.4x the MuJoCo C baseline (451 @ 8M steps). Reward was still climbing at 32M steps.

Reproduce benchmarks (requires [MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX) Python package):

```bash
python examples/train_humanoid.py --gpu-ceiling   # ~73K SPS (GPU+vmap, 8192 envs)
python examples/train_humanoid.py --cpu-ceiling   # ~29K SPS (dispatch_apply thread pool, 252 envs)
```

The CPU ceiling uses `dispatch_apply` over N `mjData*` instances (the same C MuJoCo physics as float64). This is ~2x faster than Python multiprocessing (`AsyncVectorEnv`) by eliminating IPC overhead.

SPS parity maintained via comprehensive zero-eval caching across two optimization passes:
1. **Tendon cache** -- constant Jacobian, scatter matrix, actuator masks precomputed at load time
2. **Full vmap cache** -- tree scatter matrices (smooth_vmap), DOF friction/joint limit/tendon constraint/contact invweight/equality constraint data (constraint_vmap) all precomputed in `ModelCache`

The entire vmap pipeline (smooth, constraint, solver, forward) is zero `mx::eval()`, zero `.data<>()`, zero CPU-GPU sync.

### Self-contained PPO example (HalfCheetah-v4, this repo only)

`scripts/train_ppo.py` trains PPO (stable-baselines3) on Gymnasium's `HalfCheetah-v4` MJCF (`half_cheetah.xml`) using this repo's own batched physics as the vector env (`scripts/mjmlx_halfcheetah_vec_env.py` replicates HalfCheetah-v4's reward/observation/reset-noise/episode-length exactly, for comparability with published baselines), via a small numpy-facing nanobind module (`python/rl_bindings.cpp`) rather than the `mlx.core`-sharing bindings in `python/bindings.cpp` (see that file's header comment for why).

Measured on Apple M-series (2048 envs, PPO throughput including gradient updates):

| Backend | env-steps/sec |
|---|---|
| MKX (Vulkan) | 55,000 - 150,000 |
| MKX (Vulkan) | 18,000 - 26,000 |

```bash
just build-python && just train-ppo --envs 2048 --timesteps 2000000                          # MLX
just build-python-mkx && just train-ppo --envs 2048 --timesteps 2000000 --build-dir build-mkx  # MKX
```

TensorBoard: `uv run tensorboard --logdir runs/`

### Go2階段地形cmd_vel追従学習(`scripts/train_go2_ppo.py`)

Unitree Go2を階段状のBox地形(高さ3〜5cm、奥行き20〜50cmの段)の上でcmd_vel(vx, vy, vyaw)追従させるPPO学習。観測・報酬・PPOハイパーパラメータ・ネットワーク構成・ドメインランダマイズは[IsaacLabのUnitreeGo2RoughEnvCfg / UnitreeGo2RoughPPORunnerCfg](https://github.com/isaac-sim/IsaacLab)を踏襲し、学習ライブラリもIsaacLabと同じ[rsl_rl](https://github.com/leggedrobotics/rsl_rl)を使う(SB3は使わない)。

依存パッケージ(`go2`/`build` dependency groupに定義済み):

```bash
uv sync --group build --group go2
```

ビルド(Python拡張、Go2モデルのメッシュ取得込み):

```bash
# macOS (MLX/Metal backend)
cmake -B build -DMJMLX_BUILD_PYTHON=ON
cmake --build build --target _mjmlx_rl_native -j

# Linux (MKX/Vulkan backend, 例: NVIDIA GPU機)
cmake -B build-mkx -DMJMLX_TENSOR_BACKEND=MKX -DCMAKE_BUILD_TYPE=Release -DMJMLX_BUILD_PYTHON=ON
cmake --build build-mkx --target _mjmlx_rl_native -j

# unitree_go2のメッシュ/XML(gitignore対象、初回のみ)を取得
uv run python3 -c "from pathlib import Path; import sys; sys.path.insert(0,'scripts'); from bench_check import fetch_models; fetch_models(Path('benchmarks/models/external'))"
```

学習実行:

```bash
uv run python scripts/train_go2_ppo.py --build-dir build-mkx --envs 2048 --max-iterations 1500 --logdir runs/go2_stairs
# macOSでMLX backendを使う場合は --build-dir build (デフォルト)
# GPU無し/デバッグ用途なら --cpu を付ける
```

チェックポイントは`--logdir`配下に`save_interval`(既定50)イテレーションごとに`model_{iteration}.pt`として自動保存される(rsl_rlの`OnPolicyRunner`が管理、学習終了時にも最終モデルを保存)。

学習ログはTensorBoardで確認する:

```bash
tensorboard --logdir runs/go2_stairs --host 0.0.0.0 --port 6006
```

#### 平地での学習・推論・Isaac Lab準拠の報酬(Linux / MKX Vulkan)

```bash
# 階段なしの平地 + Isaac Lab(UnitreeGo2FlatEnvCfg)準拠の報酬、PPO更新もGPU(CUDA)で回す
PYTHONPATH=<cu126等のtorch> uv run python scripts/train_go2_ppo.py --build-dir build --flat \
  --reward-preset isaac_flat --device cuda --envs 4096 --max-iterations 1500 --logdir runs/go2_flat

# 学習済みモデルで前後左右のcmd_velを与え、2x2タイル動画(runs/video/*.mp4)と追従誤差を出す
uv run python scripts/play_go2_cmd_vel.py --logdir runs/go2_flat --build-dir build --reward-preset isaac_flat --speed 0.5
```

- `--reward-preset`: `relaxed`(罰則を弱めた従来設定)/ `isaac_flat`(Isaac Lab準拠: `flat_orientation_l2`・`feet_air_time`など)
- `--device cuda`: ドライバに合うCUDA版torchが要る(例: ドライバ12.7なら `torch+cu126`)。CPUのままでも動くがPPO更新が約17秒/iter遅い
- `MJMLX_GO2_INTEGRATOR=implicitfast`: PDアクチュエータを陰的に扱う(既定はXMLのまま=陽的)

エンジンの制約により、IsaacLab版から以下を省略・簡略化している:
- **height_scanner観測**: 本エンジンにレイキャストAPIが無いため、盲目(proprioception only)方策になる
- **per-envの物理DR(質量/摩擦/COM)**: batched simが全envで単一モデルを共有するため、env別に質量・摩擦を変えるAPIが無い
- **地形カリキュラム**: 実装対象外(固定難度の階段地形のみ)
- **heading_command方式のcmd_vel生成**: `||cmd_vel|| ∈ [--cmd-vel-min, --cmd-vel-max]`の直接サンプリングに簡略化

## Performance measurement(学習速度・CPU/GPU差・他ツールとの比較)

事前に別プロセス(ollama等)がGPU/CPUを使っていないか確認する(各段の時間が2〜3倍ぶれる)。以降のスクリプトは `--build-dir`(既定 `build`)でPython拡張のビルド先を指す。

### 1. Go2のenv.step速度(4096 env)
```bash
uv run python scripts/bench_go2_step.py --build-dir build --envs 4096     # 平均/最小[ms]とenv-steps/s
MJMLX_PROF=1 uv run python scripts/bench_go2_step.py --build-dir build --steps 40   # 段(kin/forward/collision/solver/cfrc/euler)ごとの.build/.eval[ms/substep]
scripts/prof_min.sh build 5                                              # 上を複数回回し、段ごとの最小値を出す(ぶれ対策)
```
`--build-dir` を別コミットのビルドにすれば前後比較できる。`MJMLX_TG_<KERNEL>`(KIN/FWD/COLL)でスレッド数、`MJMLX_SOLVER_SHARED=0`・`MKX_DEVICE_LOCAL=0`で最適化を無効にして効果を切り分けられる。

### 2. 学習1 iterationの時間(Isaac Labと比較)
`train_go2_ppo.py` のログの `Iteration time` / `Collection time` が1 iteration(4096 env × 24 step)の時間。Isaac Labは別環境(Python 3.11)に入れて同じ条件で測る:
```bash
uv venv --python 3.11 ~/isaac/.venv && source ~/isaac/.venv/bin/activate   # python は別名の場合があるので .venv/bin/python を直接使う
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/isaac/IsaacLab && cd ~/isaac/IsaacLab
uv pip install "setuptools<81" wheel && uv pip install --no-build-isolation flatdict==4.0.1   # isaaclab.sh --install がflatdictのビルドで失敗するため先に入れる
./isaaclab.sh --install rsl_rl
OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless --max_iterations 20   # ログの Computation / collection / learning を見る
```
RTX 3060(12 GB)での実測は `docs/wip_isaac_comparison.md`(本リポジトリ約1.3 s/iter、Isaac Lab約2.2 s/iter)。

### 3. CPU(MuJoCo C)とGPUの物理の差
```bash
uv run python scripts/compare_cpu_gpu_go2.py --build-dir build              # 接触を含む状態で、同じ状態から1制御step進めた関節角・胴体速度の差
uv run python scripts/compare_cpu_gpu_go2.py --build-dir build --lift 1.0   # 接触なし(接触以外の物理の差)
just check-video-export   # CPU|GPUを並べた複数ロボットのタイル動画(runs/video/compare.mp4)
```
長い軌道の比較は接触でカオス的に増幅されるので、1 stepの差で評価する。

### 4. 経路探索(vamp)との比較の測り方
本リポジトリには経路探索(motion planning)の実装は無い。比較対象のベースラインとして [vamp](https://github.com/KavrakiLab/vamp)(CPU SIMDのRRT-Connect等)の計画時間は次で測れる:
```bash
uv venv --python 3.12 .venv-vamp && uv pip install --python .venv-vamp/bin/python vamp-planner numpy
```
```python
import time, numpy as np, vamp
a = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]); b = np.array([2.35, 1.0, 0.0, -0.8, 0.0, 2.5, 0.785])
env = vamp.Environment(); env.add_sphere(vamp.Sphere([0.55, 0.0, 0.25], 0.2))   # 障害物を足して難しくする
rng = vamp.panda.xorshift(); ts = []
for _ in range(20):
    t = time.perf_counter(); r = vamp.panda.rrtc(a, b, env, vamp.RRTCSettings(), rng); ts.append((time.perf_counter() - t) * 1e3)
print(f"solved={r.solved} median={np.median(ts):.3f} ms")   # 球1個の単純な場面で約0.03 ms(AVX2のCPU)
```
難しい問題セットはvampが配布するMotionBenchMaker(bookshelf / cage / table など)を使う。

## Architecture

```
libmjb.dylib (unified dual-backend API)
    |
    +-- mjb_* C API (47 functions)
    |     Backend selection: MJB_BACKEND_CPU or MJB_BACKEND_MLX
    |     float* interface everywhere (CPU does double->float conversion)
    |
    +-- CPU backend: wraps MuJoCo C (mj_step, mj_forward, ...)
    |     Batched: GCD thread pool over N independent mjData*
    |
    +-- MLX backend: wraps libmjmlx (mjmlx_step, mjmlx_forward, ...)
          Batched: compiled vmap GPU pipeline

libmjmlx.dylib (GPU physics core)
    |
    +-- Physics pipeline (C++ / MLX)
    |     io, math, smooth, collision, constraint, solver, forward,
    |     passive, support, scan
    |
    +-- Tendon system
    |     Fixed (joint-based) tendons: ten_length, ten_velocity, ten_J
    |     Tendon passive forces, limits, friction
    |     TENDON + SITE actuator transmission (scalar + vmap)
    |
    +-- GPU kernels
    |     Kinematics FK, fused Euler (GLSL (mkx_kernels))
    |
    +-- Batched simulation (331K SPS)
    |     GPU kin -> compile(vmap(forward)) -> GPU euler
    |
    +-- Differentiable step (planned)
    |     grad(step) for empowerment / model-based RL
    |
    +-- C API (mjmlx.h, 48 functions)
          Opaque handles, float* data exchange (unified memory)
```

## Building

Requirements: macOS 14+, Apple Silicon, CMake >= 3.25, MLX and MuJoCo (via pip).

```bash
pip install mlx mujoco

cmake -B build
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

This builds both `libmjmlx.dylib` and `libmjb.dylib`. The build auto-detects MLX and MuJoCo from pip.

To override paths:

```bash
cmake -B build -DMUJOCO_ROOT=/path/to/mujoco
```

### Python bindings (nanobind)

```bash
cmake -B build -DMJMLX_BUILD_PYTHON=ON
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

### Running tests

```bash
# Full suite (237 tests across 28 suites)
./scripts/run_tests.sh /path/to/humanoid.xml

# Or via CTest
cd build && cmake .. -DMJMLX_TEST_MODEL=/path/to/humanoid.xml && ctest --output-on-failure

# Conformance tests only
cd build && ctest -L correctness --output-on-failure
```

## Unified C API (mjb.h)

The `mjb_*` API lets you pick a backend at creation time:

```c
#include <mjmlx/mjb.h>

// GPU physics (GPU, float32, 73K+ SPS training)
MjbBackend* gpu = mjb_create_backend(MJB_BACKEND_MLX);
MjbModel* model = mjb_load_model(gpu, "humanoid.xml");
MjbData* data = mjb_make_data(model);
mjb_step(model, data);

// CPU physics (MuJoCo C, double, thread-safe)
MjbBackend* cpu = mjb_create_backend(MJB_BACKEND_CPU);
MjbModel* cpu_model = mjb_load_model(cpu, "humanoid.xml");

// Batched simulation (both backends)
MjbBatchedConfig config = { .num_envs = 8192 };
MjbBatchedSim* sim = mjb_batched_create(model, &config);
mjb_batched_step(sim, controls);
const float* qpos = mjb_batched_get_qpos(sim, &n);
```

See [`include/mjmlx/mjb.h`](include/mjmlx/mjb.h) for the full API.

## MLX Physics API (mjmlx.h)

Lower-level API for direct GPU physics access:

```c
#include <mjmlx/mjmlx.h>

MjmlxModel* model = mjmlx_load_model("humanoid.xml");
MjmlxData* data = mjmlx_make_data(model);
mjmlx_step(model, data);
const float* qpos = mjmlx_get_qpos(data, &n);  // zero-copy unified memory
```

See [`include/mjmlx/mjmlx.h`](include/mjmlx/mjmlx.h) for the full API.

## Conformance

All features validated against MuJoCo C reference implementation. **237 tests across 28 test suites, all passing.**

### Phase 1: Synth Physics Foundation
- **Gravity compensation** (`body_gravcomp` / `qfrc_gravcomp`) -- validated against MuJoCo C
- **Per-body contact forces** (`cfrc_ext` via `rne_post_constraint`) -- integrated into forward pipeline
- **Exclude signature** collision filtering (`nexclude`, `exclude_signature`) -- matches MuJoCo C encoding
- **Model validation warnings** at load time for unsupported features
- **High-DOF verification** (nv=67) -- perfect match with MuJoCo C, stable through 100+ steps

### Phase 2: Contact Friction
- **Pyramidal friction (condim=3)** -- 4 pyramid edge rows per contact, D/aref match MuJoCo C within 0.001%
- **Pyramidal friction (condim=4,6)** -- torsion + rolling friction, 6/10 rows per contact, D values identical

### Phase 3: Collision Geometry
- **plane-box** -- multi-contact (up to 4 face vertices), ncon/nefc match MuJoCo C exactly
- **sphere-box** -- closest-point on OBB, with interior fallback
- **capsule-box** -- segment-to-OBB closest approach with iterative refinement
- **box-box** -- SAT (15 axes) with support point contact generation
- **plane-cylinder** -- multi-contact (face center + rim points)
- **sphere-cylinder** -- cylinder-local closest-point with barrel/cap/rim handling
- **capsule-cylinder** -- segment-to-cylinder iterative projection refinement
- **GJK/EPA** -- full convex collision via Gilbert-Johnson-Keerthi + Expanding Polytope Algorithm (scalar: 64-iter GJK + 64-iter EPA; vmap: 32-iter GJK + support-based depth estimation). Both use proper mesh support functions (vertex argmax), not sphere approximation
- **plane-mesh** -- multi-contact (all penetrating vertices), near-exact MuJoCo C match
- **mesh-mesh** -- GJK/EPA with proper mesh support functions
- **plane-hfield** -- heightfield terrain collision against all geom types, grid-cell triangle tests
- **plane-ellipsoid, sphere-ellipsoid, capsule-ellipsoid** -- analytic collision via ellipsoid-to-sphere transform
- All collision pair types work in both scalar and vmap (batched) pipelines

### Phase 4: Equality Constraints + DOF Friction
- **Equality constraints (CONNECT/WELD/JOINT)** -- 12 tests, qacc diff ~0 vs MuJoCo C
- **DOF friction loss** -- solver friction clamping with linear zone cost, qacc diff ~1e-6
- Correct constraint ordering: equality -> friction -> limits -> contacts

### Phase 5: Tendon System + Transmission
- **Fixed (joint-based) tendons** -- `ten_length`, `ten_velocity`, `ten_J` match MuJoCo C within 1e-5
- **Tendon passive forces** -- spring + damping via `ten_J^T` projection
- **TENDON transmission** -- actuators through tendons, moment = gear * ten_J
- **SITE transmission** -- actuators at sites, full 6-DOF Jacobian projection
- **Tendon limits** -- constraint rows for tendon length bounds, qacc diff ~1e-3
- **Tendon friction loss** -- friction constraints through tendons, exact match
- **Spatial tendons (wrapping geometry)** -- DEFERRED: MJX supports it but requires ~400 lines of geodesic path computation around spheres/cylinders; most RL models use fixed tendons only
- Fixed tendons + passive forces + limits + friction: both scalar and vmap. Transmission: scalar path

### Phase 6: Actuator Dynamics (scalar + vmap)
- **FILTER dynamics** -- first-order low-pass `da/dt = (ctrl - act) / tau`, Euler integration
- **FILTEREXACT dynamics** -- exact exponential integration of the same ODE
- **INTEGRATOR dynamics** -- pure integration `da/dt = ctrl`
- **Activation clamping** -- `actuator_actlimited` / `actuator_actrange`
- Mixed stateless (NONE) + stateful actuators in the same model
- **MUSCLE dynamics** -- DEFERRED (biomechanical models only)

### Phase 7: Advanced Integrators (scalar path)
- **RK4 (4th-order Runge-Kutta)** -- 4 forward evaluations per step, weighted average; qpos diff ~3e-8 vs MuJoCo C
- **ImplicitFast** -- implicit velocity integration via velocity derivative (`deriv_smooth_vel`): modified mass matrix M' = M - dt * dqfrc/dqvel, Cholesky solve; qvel diff ~1e-6 vs MuJoCo C
- Euler, RK4, Implicit, ImplicitFast all supported in scalar path
- Batched (GPU) pipeline uses Euler only; vmap RK4/Implicit DEFERRED (needs GPU kernel rewrite)

### Deferred Phases
- **Sensors** (Phase 8) -- ~50 sensor types, large surface area; RL training reads joint/body state directly, not XML sensors
- **Differentiable physics** (Phase 9) -- `grad(step)` via `mx::grad`; requires making full pipeline autodiff-compatible, separate project-scale effort
- **Inverse dynamics / constraint islands / noslip / sleep** (Phase 10) -- niche features, not needed for common RL workflows

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Test Suite Summary

| Suite | Tests | What it validates |
|-------|-------|-------------------|
| test_io_full | 9 | Model loading, array shapes, initial state |
| test_forward_full | 7 | Forward dynamics pipeline, step correctness |
| test_physics_full | 20 | Physics accuracy vs MuJoCo C |
| test_solver_full | 11 | Constraint solver (Newton + CG) |
| test_collision_full | 7 | Core collision detection |
| test_math_full | 17 | Math utilities (quaternion, rotation, etc.) |
| test_linalg_full | 11 | Linear algebra (Cholesky, solve, etc.) |
| test_vmap_smooth | 12 | Vmap-compatible smooth dynamics |
| test_gravcomp | 7 | Gravity compensation |
| test_cfrc_ext | 6 | Contact force computation |
| test_exclude | 6 | Collision exclusion filtering |
| test_validation | 8 | Model validation warnings |
| test_high_dof | 6 | High-DOF models (67 DOF) |
| test_friction_condim3 | 7 | Pyramidal friction (condim=3) |
| test_friction_condim46 | 7 | Torsion/rolling friction (condim=4,6) |
| test_collision_box | 6 | Box collision pairs |
| test_collision_cylinder | 6 | Cylinder collision pairs |
| test_collision_mesh | 7 | Mesh/GJK/EPA collisions |
| test_collision_hfield | 7 | Heightfield terrain collisions |
| test_collision_ellipsoid | 7 | Ellipsoid collisions |
| test_equality | 12 | Equality constraints (CONNECT/WELD/JOINT) |
| test_dof_friction | 9 | DOF friction loss + solver clamping |
| test_fixed_tendon | 9 | Fixed tendon system |
| test_transmission | 7 | TENDON + SITE transmission |
| test_tendon_constraint | 6 | Tendon limits + friction loss |
| test_rk4 | 6 | RK4 integrator |
| test_filter_dynamics | 7 | Activation dynamics (FILTER/INTEGRATOR) |
| test_implicit | 7 | ImplicitFast integrator |
| **TOTAL** | **237** | |

## Consumers

- **[MuJoCo-MLX](https://github.com/arghyasur1991/MuJoCo-MLX)** (Python) -- uses nanobind extension from this repo
- **MuJoCo-MLX-Unity** (C#, planned) -- ships `libmjb.dylib` + P/Invoke bindings

## License

Apache 2.0
