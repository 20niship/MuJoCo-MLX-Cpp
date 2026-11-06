# MuJoCo-MLX-Cpp

[English](README.md) | **日本語**

[![CI](https://github.com/20niship/MuJoCo-MLX-Cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/20niship/MuJoCo-MLX-Cpp/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://20niship.github.io/MuJoCo-MLX-Cpp/)
[![License](https://img.shields.io/github/license/20niship/MuJoCo-MLX-Cpp)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/20niship/MuJoCo-MLX-Cpp)](https://github.com/20niship/MuJoCo-MLX-Cpp/commits/main)
[![Stars](https://img.shields.io/github/stars/20niship/MuJoCo-MLX-Cpp?style=flat)](https://github.com/20niship/MuJoCo-MLX-Cpp/stargazers)
![Vulkan](https://img.shields.io/badge/Vulkan-compute-AC162C?logo=vulkan&logoColor=white)
![No NVIDIA required](https://img.shields.io/badge/NVIDIA%20GPU-not%20required-76B900?logo=nvidia&logoColor=white)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.5-orange)
![Python](https://img.shields.io/badge/Python-nanobind-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20(tested)-lightgrey)

**プロジェクトページ: https://20niship.github.io/MuJoCo-MLX-Cpp/**

Vulkan上でMuJoCoの物理とカメラ描画をGPUバッチ実行する、数千env並列の強化学習向けライブラリ。

- **物理**: MuJoCo互換のパイプライン(運動学・衝突・拘束solver・積分)をGPU compute kernelで実装。MuJoCo Cと突き合わせて検証している。
- **カメラ**: 全env x 全カメラを1回のindirect drawで描画。RGB+depthをCPUコピーなしでPyTorchのCUDAテンソルとして渡せる(Linux)。
- **強化学習**: Unitree Go2環境(Isaac Lab準拠の観測・報酬)を[rsl_rl](https://github.com/leggedrobotics/rsl_rl)のPPOで学習。
- **ライブラリ**: `libmjmlx`(GPU物理、C API `mjmlx.h`)と`libmjb`(CPU=MuJoCo C / GPUエンジンを切り替えられるC API `mjb.h`)。Pythonはnanobind。

テンソルバックエンドは[`extern/mlx_vulkan`](https://github.com/20niship/mlx_vulkan)(MKX。Vulkan compute上のMLX風配列ライブラリ)をgit submoduleで取り込んでいる。Metalバックエンドは削除済みで、GPUバックエンドはVulkanのみ。(API中の`MJB_BACKEND_MLX`や`mlx`という名前は歴史的なもので「GPUエンジン」を指す。)

## 特長

- **MuJoCoと同じモデルをそのまま使える。** MuJoCoのパーサでMJCF(XML)を読むので、menagerieのGo2/G1/Piper、GymnasiumのHumanoid/HalfCheetahなどがそのまま動く。全機能をCIでMuJoCo Cと突き合わせている。既存のMuJoCoモデルをGPUエンジンに切り替えるのは`load_model(path)`だけで、モデル変換も新しいアセット形式も不要。`libmjb`ならenumを1つ変えるだけで同じCコードをMuJoCo C(CPU)とGPUエンジンのどちらでも動かせる。
- **NVIDIA GPU不要(ベンダー非依存)。** 計算も描画もVulkan(GLSLのcompute shader)だけを使う。Vulkan 1.2以上のデバイスならNVIDIA/AMD/Intelのどれでも、Mesa lavapipe経由でCPUのみでも動く(CIはこれ)。NVIDIA + CUDAが必要なのは、カメラ画像をコピーなしでPyTorchに渡す経路(Vulkan→CUDA interop)だけで、無い環境ではCPU読み出しで同じ画像が得られる。
- **クロスプラットフォーム設計。** Linux(NVIDIAとlavapipe)で検証済み。Vulkan層([`mlx_vulkan`](https://github.com/20niship/mlx_vulkan))はMoltenVK経由でmacOSも対象にしているが、**このリポジトリでのmacOS動作は未検証**(カメラ機能も未検証)。
- **依存がほとんど無い。** C++コアが必要とするのはVulkan(ローダーとヘッダ)、`glslangValidator`(シェーダを初回使用時にコンパイル)、MuJoCo(MJCFのパースとCPU参照)だけ。CUDA Toolkit、PyTorch等のMLフレームワーク、C/C++ライブラリ利用時のPythonは不要。Pythonバインディングはnanobind、学習にはPyTorch + rsl_rlを追加で使う。その他(VMA、doctest)はCMakeが取得する。
- **Isaac LabやGenesisより速い。** RTX 3060のGo2 `env.step`で、4096 envならIsaac Labの3.4倍・Genesisの1.6倍(409K vs 120K / 252K env-steps/s)、16384 envsでGenesisの1.7倍、計測した全サイズでIsaac Labより速い。全envにカメラを付けた場合はGenesisの4.1倍、Isaac Lab TiledCameraの最大2.7倍(条件はハイライト参照)。注意: Isaac Lab/Genesisは過去の単発計測(同一セッションでの再計測ではない)、Genesisは素の`scene.step()`、256 envsではGenesisとほぼ同等(63K vs 60K)。
- **RLのスループット重視。** 中級GPUで数十万env-steps/s、バッチカメラ、観測・報酬は学習デバイス上で計算、1 stepあたりデバイス→ホスト転送は1回。

## ハイライト(特記なき限りRTX 3060 12 GB)

Unitree Go2、平地、dt 0.005 x 4 substep、観測・報酬込みの`env.step`、`device=cuda`:

| env数 | 本リポジトリ(既定) | 本リポジトリ(高速モード) | Isaac Lab | Genesis |
|---|---|---|---|---|
| 4,096 | 409K | **480K** | 約120K | 約252K |
| 16,384 | 526K | **649K** | - | 約313K |

単位はenv-steps/s。高速モードは`self_collision=False, max_condim=3`(Isaac Lab/Genesisと同じ設定)で、既定は正確なモデルのまま。Isaac Lab/Genesisは過去の単発計測で同一セッションでは再計測していない。Genesisは素の`scene.step()`、こちらは観測・報酬込み。条件と注意点は「性能測定ログ(詳細)」。

カメラ付きRLの1 step(物理4 substep + 描画 + RGB/depthがCUDAテンソルになるまで、1カメラ/env、64px、4096 env、前方視点): **337,618 env-steps/s**(Genesis過去値 82,900)。三人称視点(ロボット全体が写る): 124,415。

**性能グラフ**

![physics](https://github.com/user-attachments/assets/05cc46e9-9091-4535-a864-6a232b6297a8)

![camera vs Genesis](https://github.com/user-attachments/assets/6c4644cc-8e79-4f4e-a3f0-14c5ec7eca8e)

![camera vs Isaac](https://github.com/user-attachments/assets/cac72f15-a227-49ae-a040-4c669b0d8ed4)

![history](https://github.com/user-attachments/assets/e5522b76-834a-4d85-85ac-344f9079f0be)

READMEとプロジェクトページで使う動画・画像(学習動画、CPU比較、カメラ映像)は**リポジトリに含めない**。GitHub上で添付し、URLで参照する。

https://github.com/user-attachments/assets/2575874e-0fe2-4981-a96e-c1e924d22542

*左がCPU(MuJoCo C)、右がGPU。piper / openarm / go2 / g1。`just check-video-export`で生成。アームは一致し、脚ロボットは着地後の接触解の差で軌道が分かれる(接触はカオス的に増幅されるので、1 stepの差で評価する。下記)。*

## ビルド

必要なもの: Vulkanドライバ(NVIDIAで確認。CIはGPU不要のlavapipe=`mesa-vulkan-drivers`)、Vulkanのヘッダ/ローダーと`glslangValidator`(Debian/Ubuntuなら`libvulkan-dev glslang-tools`)、CMake >= 3.25、C++20コンパイラ、MuJoCo(`pip install mujoco`で自動検出、または`-DMUJOCO_ROOT=...`)。[just](https://github.com/casey/just)と[uv](https://github.com/astral-sh/uv)は下のレシピ用の任意ツール。`libmjmlx`の実行時リンク依存は`libvulkan`と`libmujoco`のみ。macOS(MoltenVK)は未検証。

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

Isaac Lab版からの簡略化: Go2環境はまだheight_scannerを使わない(盲目方策。[レイキャストAPI](#レイキャストheight-scanner--lidarbatched-gpu-sim専用)はあるが観測には未接続)、env別の物理ドメインランダマイズなし(全envで単一モデル)、地形カリキュラムなし、`cmd_vel`はheading方式でなく`||cmd_vel||`範囲の直接サンプリング。

学習済み方策の再生(Go2、平地、`isaac_flat`、500 iteration、前後左右に0.5 m/sで指令。後退・左・右は追従するが前進だけ静止するので途中経過):

https://github.com/user-attachments/assets/6b5d210c-da5b-401b-8277-287d5a3b85c3

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

## レイキャスト(height scanner / LiDAR、GPUバッチsim専用)

全envのレイを1回のGPU dispatch(1スレッド=1レイ)で撃つ`mj_ray`相当。plane / sphere / capsule / ellipsoid / cylinder / box / meshに対応し、`bodyexclude`・`flg_static`・`geomgroup`・`max_dist`もMuJoCoと同様に使える。結果はMuJoCo Cの`mj_ray`と一致(`ctest -R raycast`、各ケース2400本のランダムなレイで距離誤差は最大約1e-5、当たり/外れ・geom idの不一致なし)。hfieldとSDFは当たらない。メッシュは面を総当たり(AABBで枝刈りのみ)なので、大きなvisual meshはgeom groupのマスクで除外する。

```python
rc = native.create_raycaster(sim)
dist, geomid = rc.cast(origins, dirs, max_dist=0.0, bodyexclude=-1, flg_static=True, geomgroup_mask=-1)
# origins, dirs: float32 (num_envs, nray, 3) 世界座標。dist: float32 (num_envs, nray)、-1が外れ(mj_rayと同じ)。geomid: int32、-1が外れ

from mjmlx_sensors import RayCaster, grid_pattern          # Isaac LabのRayCaster / height scanner相当
scanner = RayCaster(native, sim, base_body_id, grid_pattern((1.6, 1.0), 0.1), offset_z=0.2, alignment="yaw", geomgroup_mask=1 << 0)
scanner.update()
obs = scanner.height_scan(offset=0.5).clamp(-1.0, 1.0)     # (num_envs, num_rays)
```

C APIは`mjmlx.h`の`mjmlx_raycaster_create/cast/free`。Go2の階段地形での例(ヒット点の高さを地形と比較、平均誤差1.5 mm): `scripts/example_raycast_height_scan.py`。速度(RTX 3060、Go2、1 envあたり187本、地形のgroupのみ): キャスト自体は約60 M rays/s(4096 envsで12 ms)。Pythonの`RayCaster.update()`はレイと結果がホストメモリを経由するためnumpy/torchのコピーが乗り、4096 envsで約38 ms。

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

Go2の1制御stepのMuJoCo Cとの差(関節角): 既定は平均3.6e-5 rad・p95 8e-7 rad。`--max-condim 3`を両側に適用すると平均1.4e-5 rad。Isaac Labのセットアップ手順と測定履歴(最適化ごとの効果・退行の原因・GPUクロックの影響)は下の「性能測定ログ(詳細)」。

## 性能測定ログ(詳細)

以下の値は暫定値で、特記なき限りRTX 3060。

### Performance measurement(学習速度・CPU/GPU差・Isaac Labとの比較)

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

#### CPU(MuJoCo C)とGPU(MKX/Vulkan)の比較動画

`just check-video-export` が生成する、piper / openarm / go2 / g1 の4ロボットを左CPU・右GPUで並べたタイル動画(3秒)。

https://github.com/user-attachments/assets/349e410a-1bc7-4451-830a-485e4a42be4b

動画はリポジトリには含めない(サイズが大きいため)。上のコマンドで `runs/video/compare.mp4` に生成される。arm2種はCPUとGPUが一致し、go2/g1は着地後に接触解の差で軌道が分かれる。
長い軌道の比較は接触でカオス的に増幅されるので、1 stepの差で評価する。

### 4. カメラ描画速度(Isaac Lab TiledCameraと比較)
```bash
uv run python scripts/bench_camera_vs_isaac.py --build-dir build            # 両方測ってJSON(runs/camera_vs_isaac.json)と表を出す
uv run python scripts/bench_camera_vs_isaac.py --only mjmlx --envs 256 --sizes 64   # こちらだけ
```
Isaac側は `scripts/isaac_camera_bench.py`(`~/isaac/.venv`、1構成1プロセス、RTX機能を全て切り、Go2平地env、TiledCamera 1台/env、rgb+distance_to_camera)を子プロセスで呼ぶ。RTX 3060 12GB、Go2、1カメラ/env、env-frames/s(描画のみ)と env-steps/s(物理4substep+描画1回):

| envs | px | mjmlx 描画のみ(GPU) 旧→新 | mjmlx 描画+CPU読出 旧→新 | Isaac 描画のみ | mjmlx 物理+描画 旧→新 | Isaac 物理+描画 |
|---|---|---|---|---|---|---|
| 64 | 64 | 1,718 → **40,630** | 1,531 → **27,520** | 3,298 | 1,161 → **4,180** | 2,238 |
| 64 | 128 | 1,708 → **17,206** | 1,190 → **10,595** | 1,557 | 960 → **3,563** | 1,179 |
| 256 | 64 | 1,960 → **50,956** | 1,762 → **34,094** | 5,490 | 1,390 → **10,972** | 4,133 |
| 256 | 128 | 1,898 → **18,567** | 1,090 → **9,367** | 2,427 | 1,010 → **6,196** | 2,240 |
| 1024 | 64 | 2,763 → **53,466** | 1,737 → **31,618** | 7,924 | 1,499 → **17,604** | 7,278 |
| 1024 | 128 | 2,794 → **18,832** | 1,214 → **8,816** | skip | 1,138 → **7,363** | skip |
| 4096 | 64 | 3,882 → **55,908** | 1,949 → **29,512** | skip | 1,797 → **19,608** | skip |
| 4096 | 128 | 3,867 → **22,578** | 1,397 → **8,631** | skip | 1,319 → **7,423** | skip |

注記(条件差): Isaacの出力はGPUテンソルのまま(CPU読出なし)、こちらの「描画のみ」もGPU上まで、「描画+読出」はrgb+depthをnumpyへ。Isaacの物理+描画は`env.step`全体(観測・報酬・イベントを含む)、こちらは`MjmlxGo2Env.step`(報酬含む)。Isaacは物理dt 0.005 x4、こちらも同じ。カメラ位置/画角(水平47.5度)は揃えたが、近遠面・照明・シェーディングは異なる(こちらはPhongのみ、Isaacは最小設定のRTX)。Isaacの1024x128と4096は1800sのタイムアウトで未完(原因は未調査。VRAM不足と断定はできない)。旧はGPU上のフラスタム/LODカリング導入前(Go2のvisual meshが約40万三角形/envありジオメトリ律速で、64pxでIsaacの0.35〜0.52倍)、新は導入後(1ドローのまま、GPU上のcompute passで(env,cam,geom)毎にフラスタム球カリングとLOD選択、メッシュは頂点クラスタリングで1/4,1/16,1/64に簡略化)。この比較のカメラはIsaacと同じ前方視点でロボット自身はほぼ写らない。ロボット全体が写る三人称カメラ(`--camera orbit`)では 1024env 64px で 描画のみ 30,072、物理+描画 14,421 env-frames(steps)/s。LODの強さは環境変数 `MJMLX_CAM_LOD_K`(既定は高さ128px未満で16、128px以上で4、0でLOD無効)で変えられる。


#### カメラあり強化学習の1 step(物理4substep + 描画 + RGB/depthがGPU上のtorchテンソルとして使える状態まで)

```bash
PYTHONPATH=/path/to/torch-cu126 uv run python scripts/bench_camera_rl.py --camera front --envs 64 256 1024 4096 --sizes 64 128   # 前方視点(ロボットはほぼ写らない)
PYTHONPATH=/path/to/torch-cu126 uv run python scripts/bench_camera_rl.py --camera orbit ...                                       # 三人称視点(ロボット全体が写る、約40万三角形/env)
```

RTX 3060、Go2平地、**全envで丁度1カメラ**(モデル同梱の`track`を置換。上の`bench_camera_vs_isaac.py`の置換は`track`を残すので、現在のmenagerieのモデルでは mjmlx だけ2カメラ/envを描いており、上の表のmjmlx列は同条件ではない)、RGB+depth、torchは`device=cuda`、SMクロックは全計測で1837MHz、計測前に他のGPUジョブが無いことを確認(util 0%)。cpu/gpu経路を別プロセスで**交互に3回**走らせて中央値(プロセス内でも2ラウンドの中央値)。env-steps/s。mjmlxのみ計測(Genesisは過去値)。

- 旧: 描画後にRGB/depthをCPUへ読み出し(numpy)、`torch.from_numpy(...).to('cuda')`でアップロード。
- 新: Vulkan->CUDA external memory interop(`scripts/mjmlx_camera_cuda.py`)で、描画結果をGPU上のまま`cuda`テンソルとして使う。

前方視点(`--camera front`):

| envs | px | 物理のみ | 描画のみ(env-frames/s) | 旧: 物理+描画+CPU読出+upload | **新: interop** | 新/旧 | Genesis(過去値) | 新/Genesis |
|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 29,993 | 296,503 | 21,091 | **24,273** | 1.15x | 10,400 | 2.3x |
| 256 | 64 | 102,601 | 849,761 | 55,566 | **76,307** | 1.37x | 32,100 | 2.4x |
| 1024 | 64 | 269,492 | 1,698,282 | 91,594 | **199,027** | 2.17x | 64,400 | 3.1x |
| 4096 | 64 | 456,442 | 2,112,591 | 76,654 | **337,618** | 4.40x | 82,900 | 4.1x |
| 64 | 128 | 30,658 | 257,009 | 16,241 | **23,082** | 1.42x | 10,300 | 2.2x |
| 256 | 128 | 95,593 | 654,376 | 27,764 | **69,929** | 2.52x | 29,100 | 2.4x |
| 1024 | 128 | 258,027 | 1,145,990 | 22,431 | **174,913** | 7.80x | 52,400 | 3.3x |
| 4096 | 128 | 473,909 | 1,285,141 | 21,983 | **273,812** | 12.5x | 66,100 | 4.1x |

三人称視点(`--camera orbit`、ロボット全体が写る重い場合)。128pxの「旧」はLOD係数も旧既定(K=16)、「新」は解像度別の既定(後述):

| envs | px | 物理のみ | 描画のみ(env-frames/s) | 旧: CPU読出+upload | **新: interop(+128pxはLOD K=4)** | 新/旧 | Genesis(過去値) | 新/Genesis |
|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 29,719 | 110,413 | 17,715 | **21,250** | 1.20x | 10,400 | 2.0x |
| 256 | 64 | 99,535 | 157,113 | 43,489 | **53,839** | 1.24x | 32,100 | 1.7x |
| 1024 | 64 | 269,572 | 175,600 | 61,517 | **98,292** | 1.60x | 64,400 | 1.5x |
| 4096 | 64 | 468,667 | 179,121 | 55,447 | **124,415** | 2.24x | 82,900 | 1.5x |
| 64 | 128 | 30,264 | 108,247 | 12,169 | **20,873** | 1.71x | 10,300 | 2.0x |
| 256 | 128 | 99,612 | 149,680 | 19,085 | **51,695** | 2.71x | 29,100 | 1.8x |
| 1024 | 128 | 267,239 | 166,360 | 16,132 | **91,333** | 5.66x | 52,400 | 1.7x |
| 4096 | 128 | 463,801 | 169,719 | 16,239 | **113,679** | 7.00x | 66,100 | 1.7x |

施策ごとの寄与(三人称128px、4096env、env-steps/s):

| 構成 | env-steps/s |
|---|---|
| 旧(CPU読出+upload、LOD K=16) | 16,239 |
| + LOD既定をK=4に(CPU読出のまま) | 20,093 |
| + CUDA interop(LOD K=16のまま) | 50,218 |
| + interop + LOD K=4(新) | **113,679** |

Genesisの値は過去の計測(BatchRenderer(Madrona)、GPUテンソル出力)で、今回は再計測していない。GPU・カメラ位置・シーンの複雑さ・シェーディング・`env.step`に含まれる処理(mjmlxは`MjmlxGo2Env.step`の報酬込み)が揃っている保証はない。

ボトルネック(4096env、1 step、ms):

| | 前方64px | 三人称64px | 三人称128px |
|---|---|---|---|
| 物理step | 9.0 | 8.7 | 9.0 |
| 描画(GPU) | 2.0 | 23.0 | 24.8 |
| (うち頂点処理のみ。`MJMLX_CAM_EXP=discard`) | - | 12.7 | 14.0 |
| 旧: RGB読出(numpy) | 13.9 | 13.8 | 51.5 |
| 旧: depth読出(numpy) | 8.0 | 7.8 | 46.1 |
| 旧: torchへアップロード(rgb+depth) | 12.1 | 12.7 | 50.8 |
| 新: pack passの追加(render+pack - render) | 0.5 | 0.6 | 1.8 |
| 新: テンソル取得 | 0.01 | 0.01 | 0.01 |

- 旧経路は読み出し+アップロード(34〜148ms)が物理+描画(11〜35ms)を大きく上回り、そこが支配的だった。interopで消える。
- 描画前の運動学の再実行(`s.kinematics(qpos)`)は4096envでも0.2〜0.3ms(初回のみ約94ms)。**施策(1)(物理出力の再利用)は効果が無いため実装しない**(物理stepの最終substepの運動学は1 substep前の姿勢なので、再利用すると姿勢も1 substepずれる)。
- 物理と描画は同じ`VkQueue`で直列に走り、どちらもGPU律速(描画のGPU時間 + 物理のGPU時間 = 計測した1 step時間と一致)。**施策(3)(非同期パイプライン)** は同一キューでは重ならず、別キューにしても同じSMを取り合うので、重なっても短縮は描画分(前方64pxで1 step 11msのうち2.0ms)が上限で、観測遅延を生む割に見合わないため実装しない。
- **施策(4)(ホスト側コスト)**: 描画は1 submit。ホスト側の固定費は小さく(上の運動学0.2〜0.3ms + submit)、64envは物理側が1 step 2.1ms(30k steps/s)で頭打ち。GPU計算量でなく固定オーバーヘッドと思われるが、原因は未調査。
- 並べ替え(タイル画像->`(env,cam,H,W)`)は、初版の`vkCmdCopyImageToBuffer`(タイル数個の領域)が4096envで約4ms(描画より遅い)だったので、compute shader(`record_pack`)に置換して`render()`と同じsubmitに入れた。前方64px・4096envで 275,787 -> 337,618(本ブランチ内の単発計測と上の交互3回の中央値で、同一手順の比較ではない)。
- **施策(5)(LOD)**: 解像度が高いほど同じLODでもシルエット誤差が小さいので、`height >= 128`の既定を`MJMLX_CAM_LOD_K=4`にした(64pxは従来の16)。MuJoCo標準レンダラ(`scripts/compare_camera_mujoco.py`、三人称、16env)との比較:

| px | K | ロボットシルエットIoU | depth誤差 平均 / p95 | 描画のみ(4096env) |
|---|---|---|---|---|
| 64 | 16(既定) | 0.960 | 28.2 / 75.2 mm | 181k |
| 64 | 4 | 0.899 | 30.2 / 77.4 mm | 424k |
| 128 | 16(旧既定) | 0.989 | 13.9 / 37.2 mm | 60k |
| 128 | 4(新既定) | 0.965 | 14.9 / 37.9 mm | 178k |

128pxのK=4は64pxのK=16と同等のIoU(0.965 vs 0.960)なので採用し、64pxのK=4はIoUが落ちる(0.899)ので採用しなかった。品質を優先するなら`MJMLX_CAM_LOD_K=16`で従来に戻せる。

未達・残り: Genesisの過去値は全構成で上回ったが、三人称視点では描画(約23ms/4096env)が1 stepの約7割を占める。頂点処理が約半分で、次の手はメッシュレットによる細かい単位のカリング・LODの更新、背面カリング、Hi-Zによるオクルージョンカリング。interopはLinux+VK_KHR_external_memory_fd+CUDAのみ(同期は`vkQueueWaitIdle`で、GPU間セマフォではない)。Vulkanのデバイスは`MKX_VULKAN_DEVICE_INDEX`、CUDAは既定のデバイス0を使うので、複数GPUでは揃える必要がある。mac(MoltenVK)では`VK_KHR_external_memory_fd`が無いため従来のCPU読み出しになる(実機未検証)。


### 5. Go2物理stepの速度: 履歴の再検証・退行の原因・最適化(RTX 3060、Go2平地、dt 0.005 x decimation 4、ランダムaction、env-steps/s)

測り方: `scripts/bench_physics_vs_others.py`(mjmlx/Isaac Lab/Genesis)、コミット間・設定間の比較は同一条件で**交互に複数回**走らせて平均/最小/最大を見る。GPUを他プロセス(デスクトップ描画など)が使うと全段が約2倍に遅くなり、実行ごとに50k〜110kとばらつく(旧記録の「約53k」はこの状態)ので、**絶対値ではなく同一セッション内の比**で比べる。

**コミット別(4096 env、3〜6回の平均。各コミットを実際にビルドして `dt=0.005 x 4` に揃えて計測)**

| コミット | 内容 | env-steps/s |
|---|---|---|
| b0a2367 | PR #33 開始 | 7,200 |
| 087ef1c | dtをIsaacに合わせ4 substep | 22,500 |
| a829b7c | solverの共有メモリ化 | 32,000 |
| 15faba3 | 接触ヤコビをlocal配列(solver約2倍) | 51,000 |
| 4d35636 | VRAM化(「Isaacの約1.7倍速い」の時点) | 93,000〜107,000 |
| bb3b606 | 87bfbf9の直前 | 89,000〜97,000 |
| 87bfbf9 | solver収束を厳しく(SI=16, GTOL=1e-10, CG=30) | 77,000〜83,000 |
| 91965d9 / af6bd6c | sensor PR #38 | 87,000〜99,000 |
| 8da6ee2 | 現main(PR #40) | 86,000〜89,000 |

- 当時の高速化の記述は再現できた(4d35636は当時のIsaac Labより速い水準)。
- 退行の主因は **87bfbf9 のsolver設定**(精度のための収束厳格化)。同一バイナリでSI/GTOL/CGだけ戻すと現main 68k → 80k(SI=8のみ)、GTOLのみ戻すと89k。約10〜15%。
- sensor PR・camera PRの影響は計測のばらつき(±10%)内で、有意な退行は検出できなかった。
- 以前の「57k→53k」のばらつきの多くはGPU競合・クロック状態による。

**ボトルネック分析(4096 env、1制御step=4 substep)**

- Vulkanのタイムスタンプで1 dispatchごとのGPU時間を測った(`extern/mlx_vulkan` に一時パッチ。案はPR本文)。GPU時間の約78%が solver 1本、残り(kin/forward/collision/euler)は合計でsubstepあたり約0.9ms。
- 読み戻し: mlx_vulkanのステージングが非キャッシュのホストメモリで、readbackが約200MB/sしか出ない。`cfrc_ext`(1.4MB)の取得だけで約8ms、1 stepで読む3配列の合計は約11ms。`HOST_CACHED` を選ぶと cfrc 8.5→0.85ms、env.step 48→34ms(実測。サブモジュール側の修正案)。
- ホスト側: グラフ構築は約0.6ms/substep、Pythonのnumpyは約3.5ms/step。
- solverのCGは多くの状態で上限(30反復)まで回っていた。Newton反復は平均約3回。

**最適化と効果(4096 env、交互5回平均。baseは現main)**

| 変更 | 効果 |
|---|---|
| 読み戻しをGPUで1バッファに詰めて1回転送(`set_pack_spec/read_pack`)、リセット経路の読み戻し/書き込みを集約(出力はビット一致) | 約82k → 112k |
| solverのNewton方向をCGから共有メモリ上のコレスキーへ(`MJMLX_SOLVER_CG=1`で従来) | 112k → 150k(solver 1回 6.9→3.4ms) |
| 地形の同じ高さの区間を1 boxに統合(平地は23→1個、衝突ペア739→233) | collision約4分の1 |
| `step(ctrl, n)`でctrl転送をstepあたり1回に | 転送待ち6回/step削減 |
| forwardの作業領域をローカル配列に | forward 527→300us |
| Newton減少量による早期終了(`MJMLX_NTOL`、既定1e-4) | solver 3.47→2.25ms。CPUとの1step差は不変(関節角 平均3.7e-5→4.3e-5 rad、p95 4.9e-6) |
| (任意)`self_collision=False`(Isaac Lab/Genesisと同じ設定) | さらに約4% |

**Before/After(同一セッション、GPU競合下の値)**

| env数 | 変更前 | 変更後 | 変更後(self衝突なし) | Isaac Lab | Genesis |
|---|---|---|---|---|---|
| 256 | 16,455 | 35,712 | 38,382 | 16,615 | 59,705 |
| 1024 | 38,141 | 78,914 | 80,956 | 57,738 | 154,430 |
| 4096 | 53,413 | 112,293 | 117,336 | 120,091 | 252,518 |

条件差: Genesisは観測・報酬なしの `scene.step()` x4、mjmlx/Isaacは観測・報酬込みの `env.step`。mjmlxはMuJoCo系の接触モデル(摩擦ピラミッドcondim 6)でGenesisより重い。
**Genesisには未達**(256/1024/4096でそれぞれ約0.6/0.5/0.45倍)。残り: solverが依然GPU時間の大半(共有メモリ命令数が律速)、小envではホストのグラフ構築とsubstep内の約22 dispatchの直列レイテンシが支配的。次の手はsolverの行列組み立て(H=M+J^T D J)の対称利用とvec4共有メモリ読み、frictionloss/limit行を密Jから外す、kin/forward/collisionの融合、mlx_vulkanのHOST_CACHEDステージング。


### 6. solver高速化(PR #42 の続き。値はすべて暫定値)

Go2 4096 env の `env.step`(ランダムaction、dt 0.005 x 4)を、PR #42 時点(Before)と本ブランチ(After)で**同一セッションで交互に3回**測定(GPU使用率は計測前後で約10%、デスクトップ描画あり。実行間のばらつきは約±3%):

| | 平均 | 最小 | 最大 |
|---|---|---|---|
| Before(#42) | 112,559 env-steps/s | 111,300 | 114,633 |
| After(本ブランチ) | 149,338 env-steps/s | 143,902 | 152,159 |

**平均 112.6k → 149.3k env-steps/s(1.33倍)**。同条件の追加7回交互計測(別セッション、GPU使用率8〜14%)でも After/Before = 1.30倍(256 env 1.22倍、1024 env 1.26倍)。Isaac Lab/Genesisは再計測しておらず、前節の過去値(Genesis 4096 env 約252k、Isaac Lab 約120k)のまま。Genesisには未達。

最適化ごとの効果(Vulkanタイムスタンプで測った4096 envの1 dispatch時間。GPU競合下の値で、同一セッション内の比較):

| 変更 | 効果 |
|---|---|
| 直線探索の二分法を導関数が十分小さければ打ち切る(`MJMLX_LSTOL`) | solver 2300 → 2060 us |
| Hの対称組み立て・非アクティブ行の分岐除去 | → 1961 us |
| J^T fを4本の独立累積に、グローバル読み込みをまとめて発行 | → 1866 us |
| 共有メモリ削減(拘束行96→72、作業領域の重ね合わせ)で同時実行数を増やす | → 1620 us |
| Eulerをsolverのqaccから直接積分(`MJMLX_EULER_EXACT=1`で従来) | euler 390 → 49 us |
| collisionのペア表をシェーダ定数に焼き込み | collision 111 → 70 us |
| packのトルクを二乗和/絶対値和に縮約、発散判定を簡略化 | 約+4% |

分析: solverカーネルは18スレッド(1ワープ)/envのレイテンシ律速で、共有メモリを増やして同時実行数を下げると実行時間がほぼ比例して悪化した(12KB増で1.9倍)。そのため共有メモリの削減が効く。
正しさ: `ctest -L correctness`(10件)・`-L models`(8件)全pass。`compare_cpu_gpu_go2.py` のCPUとの1step差は 平均3.6e-5 rad・p95 8e-7 rad(PR #42: 平均3.8e-5・p95 5e-6)で悪化なし。精度に影響する変更は、拘束行上限72(超過分の接触は落ちる。Go2は問題なし)、Euler直接積分(solverの収束許容 `MJMLX_NTOL` に依存)、直線探索の打ち切り(1e-3)。
`gpu_vs_cpu_contact_stress` が外部のGPU高負荷時に1回だけ失敗(224秒)したが、再実行7回はすべて通過。
残り: solverがGPU時間の約6割、forward/kinematicsが各1割。ホスト側(グラフ構築・numpy・readback)が小〜中env数では支配的で、mlx_vulkanのHOST_CACHEDステージング(mlx_vulkan PR #12)とグラフキャッシュが次の手。


### 7. ホスト側の追加高速化とGenesisとの差(暫定値)

mlx_vulkan PR #12(HOST_CACHEDステージング、未マージ)の参照への更新と、`env.step` のnumpy整理(行ごとの内積をeinsumに、重力投影を閉形式に、発散判定を1回に)。#44 に対して、同一セッション・交互3回(GPU使用率8〜31%のデスクトップ描画下、ランダムaction):

| env数 | Before(#44)平均 | After平均(最小〜最大) | 倍率 | Genesis(同env数) |
|---|---|---|---|---|
| 4096 | 155,993 | 192,412(177,453〜202,668) | 1.23倍 | 約252k |
| 16384 | 175,549 | 209,792(200,653〜215,589) | 1.19〜1.24倍 | 約313k |

Genesisの8192/16384 envは各1回の測定(8192: 約288k、16384: 約313k)で、4096以下は前節の過去値。**Genesisには未達**(4096で約0.76倍、16384で約0.67倍)。Isaac Lab(4096で約120k)は上回る。
Vulkanタイムスタンプによる16384 envの内訳: GPU 約36ms/step(solver 約72%、forward 約14%、kinematics 約6%、collision 約5%)、残りはホスト(readback・numpy)。大env数ではGPU時間そのもの(solverの共有メモリ命令数、forwardのローカルメモリ)が律速で、次の手はsolverのJから単位行(frictionloss/limit)を外す疎化、forward/kinematics/collisionの融合、env.stepの観測・報酬のGPU(torch)化。
**この参照更新はmlx_vulkan #12 のマージ後に、mainのマージコミットへ更新するまでmainにマージしない。**


### 8. 観測・報酬のtorch(GPU)化とGPUクロックの影響(暫定値)

`env.step` の観測・報酬・終了判定を `device` 上のtorchで計算するようにした(ホストに残るのは物理への入出力とリセット/押しの乱数処理のみ。cpu deviceでは従来とobs 1.5e-7、reward 5e-7まで一致)。ランダムaction、同一セッション・交互3回、`device=cuda`(`PYTHONPATH=~/torch-cu126`、学習時と同じ条件)での平均 env-steps/s:

| env数 | Before(main) | After | 倍率 | Genesis(過去値) |
|---|---|---|---|---|
| 256 | 67,515 | 63,297 | 0.94倍 | 約60k |
| 1,024 | 173,874 | 192,823 | 1.11倍 | 約154k |
| 4,096 | 307,812 | 311,661 | 1.01倍 | 約252k |
| 16,384 | 348,304 | 426,645 | 1.23倍 | 約313k |

小env数ではtorchの細かいカーネル起動が増えて僅かに遅く(256 envで0.94倍)、大env数ほど効く。Isaac Lab/Genesisは再計測しておらず過去値。

**重要: 計測条件でGPUクロックが約2倍変わる。** CUDAコンテキストを持たないcpu deviceのベンチでは、Vulkan計算だけのGPU使用率が低くSMクロックが上がらない(計測中の平均SMクロック 約550MHz)。`device=cuda`(学習と同じ)ではtorchの活動でクロックが上がり(約1140MHz)、同じコードが約1.5〜1.6倍速い(4096 envで cpu device 195k → cuda device 308k)。GenesisはCUDA上で動くため、学習時の条件(cuda device)で比べるのが公平で、その条件ではGenesisの過去値を全env数で上回った(256 envはほぼ同等)。cpu deviceのベンチ値(前節まで)は低クロック状態の値。
未実施: solverの単位行の疎化、カーネル融合(ts計測ではdispatch間の隙間はほぼ無く効果は小さい見込み)、mlx_vulkanのグラフキャッシュ(PR #6。入力・ctrlを永続バッファに置く設計変更が必要で、削れるのはホストのグラフ構築 約1ms/stepのみ)。


### 9. ホスト側のオーバーヘッド削減と高速モード(暫定値、すべて `device=cuda`・SMクロック約1840MHz)

#47 に対する同一セッション・交互3回(平均 env-steps/s):

| env数 | Before(#47) | After(既定設定) | 倍率 | 高速モード | 倍率 |
|---|---|---|---|---|---|
| 4096 | 318,967 | 409,549 | 1.28倍 | 480,278 | 1.50倍 |
| 16384 | 432,840 | 526,102 | 1.22倍 | 649,196 | 1.50倍 |

高速モード = 読み込み時オプション `self_collision=False, max_condim=3`(下記「シミュレーション簡略化オプション」。既定は従来どおり正確なモデル)。

最適化ごとの効果(`env.step` の内訳、4096 env / 16384 env の ms):

| 変更 | 4096 env | 16384 env |
|---|---|---|
| (起点 #47) | 12.16 | 37.92 |
| `env.step` の観測・報酬を `torch.compile` で融合(「その他」3.48→1.65ms) | 10.28 | 36.25 |
| 接触無し経路用の `cfrc_ext` ゼロ配列(毎substep 1.4MB)の確保をやめる(グラフ構築 1.34→0.40ms) | 9.38 | 31.33 |
| `read_pack_into`(ピン留めバッファへ直接読む、H2Dを非同期に) | 9.26 | 30.65 |

分析: ブースト状態ではGPU時間は4096 envで約6.1ms/step(solver 約75%)、ホスト側が約3.2msだった。`mx::zeros` による毎substepのホストでのゼロ埋めがグラフ構築の大半(約250us/substep)で、これを除くとmlx_vulkanのグラフキャッシュ(replay)で削れるものは残らない。dispatch間の隙間はほぼ無く、カーネル融合の見込み効果も小さい。ブロック幅・スレッドグループ・NTOL/共有メモリ上限の再調整は効果なし。
既定設定で目標(4096 env 470k)に届かなかった理由はsolverの共有メモリ命令数が律速のためで、残る手は18スレッド/envから32スレッド/envへの変更や単位行の疎化(いずれも10%前後の見込み)。


### 10. シミュレーション簡略化オプション(読み込み時、既定は正確なモデルのまま)

速度のために接触モデルを簡略化するオプション。**MJCFファイルは変更せず**、読み込み時にメモリ上のモデルを上書きしてからコンパイルする(関節を持つbodyとその子孫=ロボットのgeomだけが対象で、地形などの静的bodyのgeomは触らない)。既定(`self_collision=True, max_condim=0`)は従来と完全に同じ挙動。

| オプション | 内容 | 精度への影響 |
|---|---|---|
| `self_collision=False` | ロボットのgeom同士の衝突を切る(geomの `conaffinity` を0に)。Isaac Labの `enabled_self_collisions=False` と同じ。ロボットのgeomは `contype!=0`(MuJoCo既定は1)が前提 | 脚と胴体などの自己接触が起きなくなる(通常の歩行では稀)。衝突ペアが約1/10(Go2: 233→23) |
| `max_condim=3` | ロボットのgeomの `condim` を3以下にする。足(condim 6)の**ねじり摩擦・転がり摩擦が無くなり**並進摩擦のみになる(Isaac Lab/Genesisと同じ) | 足の接触の拘束行が10→4行でsolverが速い。足先で滑る・回る挙動がわずかに変わる。CPU(MuJoCo C)との比較は同じオプションを両方に渡して行う |

使い方:
```c
MjmlxModelOptions opt = mjmlx_default_model_options();   // {self_collision=1, max_condim=0}
opt.self_collision = 0; opt.max_condim = 3;
MjmlxModel* m = mjmlx_load_model_ex("go2.xml", &opt);      // mjmlx_load_model_from_string_ex もある。既存のmjmlx_load_modelは従来どおり
```
```python
model = native.load_model(path, self_collision=False, max_condim=3)           # 低レベル
env = MjmlxGo2Env(native, 4096, ..., self_collision=False, max_condim=3)       # Go2環境
```
CLI: `bench_go2_step.py --no-self-collision --max-condim 3`、`train_go2_ppo.py --no-self-collision --max-condim 3`、`compare_cpu_gpu_go2.py --max-condim 3`。

速度(4096 env / 16384 env、device=cuda、SMクロック1837MHz、同一セッション交互3回の平均 env-steps/s):

| | 4096 env | 16384 env |
|---|---|---|
| 既定(正確) | 409,103 | 527,604 |
| `max_condim=3` | 467,660(1.15倍) | 638,882(1.21倍) |
| `self_collision=False, max_condim=3` | 481,351(1.17倍) | 648,499(1.23倍) |

CPUとの1step差(関節角、`compare_cpu_gpu_go2.py`): 既定は 平均3.6e-5 rad・p95 8e-7 rad、`--max-condim 3`(CPU・GPUの両方に同じオプション)は 平均1.4e-5 rad・p95 5e-7 rad。GPUとCPUが同じ簡略モデルを解いているかの確認であり、簡略化自体による本来の物理との差(足のねじり・転がり摩擦の欠落)は含まない。

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
docs/              index.html (GitHub Pages)
extern/mlx_vulkan  Vulkanテンソルバックエンド (submodule)
```

## 開発ルール

画像・動画は絶対にcommit/pushしない(`.gitignore`済み)。PR/issueへGitHub上で添付し、READMEや`docs/index.html`からは`user-attachments`のURLで参照する。

## ライセンス

Apache 2.0
