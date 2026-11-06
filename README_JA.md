# MuJoCo-MLX-Cpp

[English](README.md) | **日本語**

**プロジェクトページ: https://20niship.github.io/MuJoCo-MLX-Cpp/**

Vulkan上でMuJoCoの物理とカメラ描画をGPUバッチ実行する、数千env並列の強化学習向けライブラリ。

- **物理**: MuJoCo互換のパイプライン(運動学・衝突・拘束solver・積分)をGPU compute kernelで実装。MuJoCo Cと突き合わせて検証している。
- **カメラ**: 全env x 全カメラを1回のindirect drawで描画。RGB+depthをCPUコピーなしでPyTorchのCUDAテンソルとして渡せる(Linux)。
- **強化学習**: Unitree Go2環境(Isaac Lab準拠の観測・報酬)を[rsl_rl](https://github.com/leggedrobotics/rsl_rl)のPPOで学習。
- **ライブラリ**: `libmjmlx`(GPU物理、C API `mjmlx.h`)と`libmjb`(CPU=MuJoCo C / GPUエンジンを切り替えられるC API `mjb.h`)。Pythonはnanobind。

テンソルバックエンドは[`extern/mlx_vulkan`](https://github.com/20niship/mlx_vulkan)(MKX。Vulkan compute上のMLX風配列ライブラリ)をgit submoduleで取り込んでいる。Metalバックエンドは削除済みで、GPUバックエンドはVulkanのみ。(API中の`MJB_BACKEND_MLX`や`mlx`という名前は歴史的なもので「GPUエンジン」を指す。)

## ハイライト(特記なき限りRTX 3060 12 GB)

Unitree Go2、平地、dt 0.005 x 4 substep、観測・報酬込みの`env.step`、`device=cuda`:

| env数 | 本リポジトリ(既定) | 本リポジトリ(高速モード) | Isaac Lab | Genesis |
|---|---|---|---|---|
| 4,096 | 409K | **480K** | 約120K | 約252K |
| 16,384 | 526K | **649K** | - | 約313K |

単位はenv-steps/s。高速モードは`self_collision=False, max_condim=3`(Isaac Lab/Genesisと同じ設定)で、既定は正確なモデルのまま。Isaac Lab/Genesisは過去の単発計測で同一セッションでは再計測していない。Genesisは素の`scene.step()`、こちらは観測・報酬込み。条件と注意点は[docs/PERFORMANCE.md](docs/PERFORMANCE.md)。

カメラ付きRLの1 step(物理4 substep + 描画 + RGB/depthがCUDAテンソルになるまで、1カメラ/env、64px、4096 env、前方視点): **337,618 env-steps/s**(Genesis過去値 82,900)。三人称視点(ロボット全体が写る): 124,415。

READMEとプロジェクトページで使う動画・画像(学習動画、CPU比較、カメラ映像)は**リポジトリに含めない**。GitHub上で添付し、URLで参照する。

https://github.com/user-attachments/assets/349e410a-1bc7-4451-830a-485e4a42be4b

*左がCPU(MuJoCo C)、右がGPU。piper / openarm / go2 / g1。`just check-video-export`で生成。アームは一致し、脚ロボットは着地後の接触解の差で軌道が分かれる(接触はカオス的に増幅されるので、1 stepの差で評価する。下記)。*

## ビルド

必要なもの: Vulkan 1.3対応GPUドライバのLinux(NVIDIAで確認、CIはlavapipe)、CMake >= 3.25、C++20コンパイラ、[just](https://github.com/casey/just)、[uv](https://github.com/astral-sh/uv)、MuJoCo(`pip install mujoco`で自動検出、または`-DMUJOCO_ROOT=...`)。macOS(MoltenVK)は未検証。

```bash
git clone --recursive https://github.com/20niship/MuJoCo-MLX-Cpp
cd MuJoCo-MLX-Cpp
just build                      # libmjmlx + libmjb + テスト/ベンチ (Release) を build/ へ
just build-python               # nanobind拡張(_mjmlx_rl_native)も作る
```

素のCMake: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DMJMLX_BUILD_PYTHON=ON && cmake --build build -j`。

CMakeオプション: `MJMLX_BUILD_TESTS`、`MJMLX_BUILD_BENCHMARKS`、`MJMLX_BUILD_EXAMPLES`、`MJMLX_BUILD_PYTHON`(既定OFF)、`MJMLX_BUILD_CAMERA`(既定ON。OFFで`mjmlx_camera_*`はスタブ)。

実ロボットのモデル(piper/openarm/go2/g1/h1のメッシュ付きMJCF、git管理外)の取得:

```bash
just fetch-models
```

## Go2の学習(PPO、rsl_rl)

```bash
uv sync --group build --group go2

# 平地 + Isaac Lab(UnitreeGo2FlatEnvCfg)準拠の報酬、PPO更新はCUDA
PYTHONPATH=<cu126等のtorch> uv run python scripts/train_go2_ppo.py --build-dir build \
  --flat --reward-preset isaac_flat --device cuda --envs 4096 --max-iterations 1500 --logdir runs/go2_flat

# 階段地形(既定)。チェックポイントは--logdir配下に50 iterationごと
uv run python scripts/train_go2_ppo.py --build-dir build --envs 2048 --logdir runs/go2_stairs

# 学習済み方策で前後左右のcmd_velを与え、タイル動画(runs/video/)と追従誤差を出す
uv run python scripts/play_go2_cmd_vel.py --logdir runs/go2_flat --build-dir build --reward-preset isaac_flat --speed 0.5

tensorboard --logdir runs/
```

主なオプション: `--reward-preset {relaxed,isaac_flat}`、`--device cuda`、`--no-self-collision --max-condim 3`(高速モード)、`--use-sensors`(batchedのIMU/接触センサ経由の観測)、`--cpu`(GPUエンジンの代わりにMuJoCo C)。`MJMLX_GO2_INTEGRATOR=implicitfast`でPDアクチュエータを陰的に積分する。

Isaac Lab版からの簡略化: height_scannerなし(盲目方策)、env別の物理ドメインランダマイズなし(全envで単一モデル)、地形カリキュラムなし、`cmd_vel`はheading方式でなく`||cmd_vel||`範囲の直接サンプリング。

HalfCheetah-v4のPPOサンプル(stable-baselines3)は`scripts/train_ppo.py`(`just train-ppo`)。

## シミュレーション簡略化オプション(読み込み時)

MJCFは変更せず、読み込み時にメモリ上のモデルを上書きしてからコンパイルする。対象は関節を持つbody以下のみ(地形は触らない)。既定はMuJoCoと同じ正確な挙動。

| オプション | 内容 | 影響 |
|---|---|---|
| `self_collision=False` | ロボットのgeom同士の衝突を切る(`conaffinity=0`)。Go2は衝突ペア233→23 | 自己接触が起きない |
| `max_condim=3` | 足のねじり・転がり摩擦をなくす(接触あたりの拘束行 10→4) | 足先の滑り・回りがわずかに変わる |

```c
MjmlxModelOptions opt = mjmlx_default_model_options();   // {self_collision=1, max_condim=0}
opt.self_collision = 0; opt.max_condim = 3;
MjmlxModel* m = mjmlx_load_model_ex("go2.xml", &opt);
```
```python
model = native.load_model(path, self_collision=False, max_condim=3)
env = MjmlxGo2Env(native, 4096, ..., self_collision=False, max_condim=3)
```

Go2 4096 / 16384 envのenv-steps/s: 既定 409K / 528K、`max_condim=3` 468K / 639K、両方 481K / 648K。

## カメラセンサ(Vulkan、GPUバッチsim専用)

MJCFの`<camera>`/`<light>`/geomのrgbaを使う。Phong、RGB+depth、hfieldと2Dテクスチャ(plane/box/hfield)、空間tendon(円柱として描画)に対応。GPU上のフラスタムカリングとLOD(メッシュ頂点クラスタリング、`MJMLX_CAM_LOD_K`)で、全env x 全カメラを`vkCmdDrawIndexedIndirect` 1回で描く。

```python
cam = native.create_camera(sim, 128, 128)       # simは create_batched(use_gpu=True) の戻り値
cam.render()
rgb, depth = cam.rgb(), cam.depth()             # (num_envs, ncam, H, W, 3) uint8 / (..., H, W) float32 [m]

from mjmlx_camera_cuda import CudaCamera        # Vulkan -> CUDA external memory interop (Linux)
gcam = CudaCamera.try_create(cam)               # 非対応環境はNone -> 上のCPU読み出しを使う
cam.render(); rgba, depth = gcam.read()         # CUDAテンソル、CPUコピーなし
```

未対応: 影、半透明、可動bodyに付いたlight、trackcom(body原点で近似)。interopはLinux + `VK_KHR_external_memory_fd` + CUDA版torchが必要で、複数GPUでは`MKX_VULKAN_DEVICE_INDEX`をCUDAのデバイスと揃える。カメラにはgraphics+compute queue、`multiDrawIndirect`、`drawIndirectFirstInstance`、`shaderClipDistance`が必要(満たさない時は`create_camera`だけが失敗)。例: `scripts/example_go2_camera.py`、MuJoCo描画との比較: `scripts/compare_camera_mujoco.py`、速度: `scripts/bench_camera*.py`。

## ベンチマークと比較

計測前に他のプロセス(ollama・デスクトップ描画・他のGPUジョブ)がGPUを使っていないか確認する(2倍ぶれる)。コミットや設定の比較は日をまたがず**同一セッションで交互に**行う。

```bash
uv run python scripts/bench_go2_step.py --build-dir build --envs 4096        # env.stepの速度
MJMLX_PROF=1 uv run python scripts/bench_go2_step.py --build-dir build       # 段ごとのGPU/ホスト時間
uv run python scripts/bench_physics_vs_others.py ...                         # 本リポジトリ / Isaac Lab / Genesis
uv run python scripts/bench_camera_vs_isaac.py --build-dir build             # Isaac Lab TiledCameraとの比較
PYTHONPATH=<cu126等のtorch> uv run python scripts/bench_camera_rl.py --camera front --envs 64 256 1024 4096 --sizes 64 128
uv run python scripts/compare_cpu_gpu_go2.py --build-dir build               # MuJoCo Cとの1 step差
just check-video-export                                                      # CPU|GPU並べたタイル動画
just bench                                                                   # benchmarks/baseline.csvとの退行チェック
```

Go2の1制御stepのMuJoCo Cとの差(関節角): 既定は平均3.6e-5 rad・p95 8e-7 rad。`--max-condim 3`を両側に適用すると平均1.4e-5 rad。Isaac Labのセットアップ手順と測定履歴(最適化ごとの効果・退行の原因・GPUクロックの影響)は[docs/PERFORMANCE.md](docs/PERFORMANCE.md)。

## テスト

```bash
just check-ci          # CIと同じ(lavapipeで動く範囲)
just check-models      # 実ロボットMJCF(piper/openarm/go2/g1/h1)。ローカル専用、実GPUが必要
cd build && ctest -L correctness   # センサ、SDFなど
cd build && ctest -L robust        # GPUバッチの健全性(代表のみ)。robust_full=nv x 拘束構成の総当たり、known_issue=既知の失敗
```

GPUエンジンとMuJoCo Cを並べて軌道を比較する: `test_gpu_vs_cpu`(humanoid・高DOF木・接触ストレス・tendon・実ロボット)、`test_batched_collision_primitives`、`test_robustness`(nv x {limit, contact, tendon, frictionloss}で発散・env間不一致・バッチサイズ/位置への依存・反復不一致を見る)、`test_sensors`、`test_sdf_*`、`test_camera*`、`test_math_full`、`test_linalg_full`。CIは`scripts/Dockerfile`(Debian + lavapipe)で実行し、`just docker-test`と同一。健全性テストは環境変数(`MJMLX_SOLVER_SHARED=0`、`MJMLX_SOLVER_CG=1`、`LP_NUM_THREADS=1`、`ROBUST_VERBOSE=1`)で条件を振れる。

## C API

```c
#include <mjmlx/mjb.h>

MjbBackend* gpu = mjb_create_backend(MJB_BACKEND_MLX);   // GPUエンジン。MJB_BACKEND_CPU = MuJoCo C (double, スレッド並列)
MjbModel* model = mjb_load_model(gpu, "humanoid.xml");
MjbBatchedConfig cfg = { .num_envs = 8192 };
MjbBatchedSim* sim = mjb_batched_create(model, &cfg);
mjb_batched_step(sim, controls);
const float* qpos = mjb_batched_get_qpos(sim, &n);
```

`mjb.h`は両バックエンドで動く。`mjmlx.h`はGPU専用の低レベルAPI(`mjmlx_batched_*`、センサ、カメラ、まとめ読み出し`mjmlx_batched_set_pack_spec/get_pack`)。**単体env用のpipeline(`mjmlx_step`、`mjmlx_forward`など)は削除済みで例外を投げる。batched APIを使うこと**(env数1でも可)。`mjmlx_grad_step`はスタブで、微分可能物理は未実装。ヘッダは[`include/mjmlx/`](include/mjmlx/)。

## 対応状況

batched GPU経路。MuJoCo Cとの照合結果。全機能表と既知の欠落は[CONFORMANCE.md](CONFORMANCE.md)、設計は[ARCHITECTURE.md](ARCHITECTURE.md)。

- 関節: free, ball, hinge, slide。積分: Euler(implicit/implicitfastはdampingの項で陰的に扱う。RK4は無し)。
- 接触: ピラミッド摩擦、condim 3/4/6、margin/gap。plane/sphere/capsule/box/cylinder/ellipsoid/mesh(GJK/EPA)、sphere/capsule-boxの解析接触、SDF-SDF・SDF-mesh。
- 拘束: 関節リミット、DOF摩擦損失、固定・空間tendon(リミット・摩擦損失・wrap)。
- 重力補償、TENDON/SITE transmission、接触力(`cfrc_ext`)、センサ19種(IMU・フレーム・関節・アクチュエータ・subtreecom)。
- batched経路で未対応: 等式拘束、アクチュエータのactivation動力学(FILTER/INTEGRATOR/MUSCLE)、hfieldの接触(カメラ描画は対応)、RK4、逆動力学、island/noslip/sleep、微分可能物理。

## ディレクトリ構成

```
include/mjmlx/     公開Cヘッダ (mjmlx.h, mjb.h)
src/               batched.cpp (パイプライン), *_vmap.cpp (smooth/constraint/solverのグラフ), compat/mkx_kernels (GLSLカーネル), camera.cpp, sensor.cpp, io.cpp, mjb.cpp
python/            nanobindバインディング
scripts/           Go2環境・学習、ベンチ、比較
tests/             C++テスト、MJCFフィクスチャ
docs/              index.html (GitHub Pages), PERFORMANCE.md
extern/mlx_vulkan  Vulkanテンソルバックエンド (submodule)
```

## 開発ルール

画像・動画は絶対にcommit/pushしない(`.gitignore`済み)。PR/issueへGitHub上で添付し、READMEや`docs/index.html`からは`user-attachments`のURLで参照する。

## ライセンス

Apache 2.0
