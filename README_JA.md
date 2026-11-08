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

## 学習した方策

学習・評価・比較のコードは[`scripts/examples/`](scripts/examples/)(Go2、G1、HalfCheetah)にある。実行方法、再現元の選定、Isaac Labとの差分、学習結果は**[scripts/examples/README_JA.md](scripts/examples/README_JA.md)**([English](scripts/examples/README.md))。

<table>
<tr>
<td align="center"><video src="https://github.com/user-attachments/assets/6b5d210c-da5b-401b-8277-287d5a3b85c3" controls muted width="400"></video><br><b>Go2</b>、平地のcmd_vel追従(途中経過: 後退・左・右は追従、前進だけ静止)</td>
<td align="center"><video src="https://github.com/user-attachments/assets/1c79aae8-1332-4f13-b42b-4f6cc9de6615" controls muted width="400"></video><br><b>G1</b>、前進専用タスク、20秒、横から(直進する。速度は指令に届かない)</td>
</tr>
</table>

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

未対応: 影、半透明、可動bodyに付いたlight、trackcom(body原点で近似)。interopはLinux + `VK_KHR_external_memory_fd` + CUDA版torchが必要で、複数GPUでは`MKX_VULKAN_DEVICE_INDEX`をCUDAのデバイスと揃える。カメラにはgraphics+compute queue、`multiDrawIndirect`、`drawIndirectFirstInstance`、`shaderClipDistance`が必要(満たさない時は`create_camera`だけが失敗)。例: `scripts/examples/go2/example_go2_camera.py`、MuJoCo描画との比較: `scripts/compare_camera_mujoco.py`、速度: `scripts/bench_camera*.py`。

## ベンチマークと比較

計測前に他のプロセス(ollama・デスクトップ描画・他のGPUジョブ)がGPUを使っていないか確認する(2倍ぶれる)。コミットや設定の比較は日をまたがず**同一セッションで交互に**行う。

```bash
uv run python scripts/bench_go2_step.py --build-dir build --envs 4096        # env.stepの速度
MJMLX_PROF=1 uv run python scripts/bench_go2_step.py --build-dir build       # 段ごとのGPU/ホスト時間
uv run python scripts/bench_physics_vs_others.py ...                         # 本リポジトリ / Isaac Lab / Genesis
uv run python scripts/bench_camera_vs_isaac.py --build-dir build             # Isaac Lab TiledCameraとの比較
PYTHONPATH=<cu126等のtorch> uv run python scripts/bench_camera_rl.py --camera front --envs 64 256 1024 4096 --sizes 64 128
uv run python scripts/examples/go2/compare_cpu_gpu_go2.py --build-dir build               # MuJoCo Cとの1 step差
just check-video-export                                                      # CPU|GPU並べたタイル動画
just bench                                                                   # benchmarks/baseline.csvとの退行チェック
```

Go2の1制御stepのMuJoCo Cとの差(関節角): 既定は平均3.6e-5 rad・p95 8e-7 rad。`--max-condim 3`を両側に適用すると平均1.4e-5 rad。Isaac Labのセットアップ手順、カメラの測定表、GPUクロックの影響は下の「性能測定ログ(詳細)」。

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
RTX 3060(12 GB)での実測は、本リポジトリ約1.3 s/iter、Isaac Lab約2.2 s/iter。

### 3. CPU(MuJoCo C)とGPUの物理の差
```bash
uv run python scripts/examples/go2/compare_cpu_gpu_go2.py --build-dir build              # 接触を含む状態で、同じ状態から1制御step進めた関節角・胴体速度の差
uv run python scripts/examples/go2/compare_cpu_gpu_go2.py --build-dir build --lift 1.0   # 接触なし(接触以外の物理の差)
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

| envs | px | mjmlx 描画のみ(GPU) | mjmlx 描画+CPU読出 | Isaac 描画のみ | mjmlx 物理+描画 | Isaac 物理+描画 |
|---|---|---|---|---|---|---|
| 64 | 64 | **40,630** | **27,520** | 3,298 | **4,180** | 2,238 |
| 64 | 128 | **17,206** | **10,595** | 1,557 | **3,563** | 1,179 |
| 256 | 64 | **50,956** | **34,094** | 5,490 | **10,972** | 4,133 |
| 256 | 128 | **18,567** | **9,367** | 2,427 | **6,196** | 2,240 |
| 1024 | 64 | **53,466** | **31,618** | 7,924 | **17,604** | 7,278 |
| 1024 | 128 | **18,832** | **8,816** | skip | **7,363** | skip |
| 4096 | 64 | **55,908** | **29,512** | skip | **19,608** | skip |
| 4096 | 128 | **22,578** | **8,631** | skip | **7,423** | skip |

注記(条件差): Isaacの出力はGPUテンソルのまま(CPU読出なし)、こちらの「描画のみ」もGPU上まで、「描画+読出」はrgb+depthをnumpyへ。Isaacの物理+描画は`env.step`全体(観測・報酬・イベントを含む)、こちらは`MjmlxGo2Env.step`(報酬含む)。Isaacは物理dt 0.005 x4、こちらも同じ。カメラ位置/画角(水平47.5度)は揃えたが、近遠面・照明・シェーディングは異なる(こちらはPhongのみ、Isaacは最小設定のRTX)。Isaacの1024x128と4096は1800sのタイムアウトで未完(原因は未調査。VRAM不足と断定はできない)。GPU上のcompute passで(env,cam,geom)毎にフラスタム球カリングとLOD選択を行い(1ドロー)、メッシュは頂点クラスタリングで1/4,1/16,1/64に簡略化している。この比較のカメラはIsaacと同じ前方視点でロボット自身はほぼ写らない。ロボット全体が写る三人称カメラ(`--camera orbit`)では 1024env 64px で 描画のみ 30,072、物理+描画 14,421 env-frames(steps)/s。LODの強さは環境変数 `MJMLX_CAM_LOD_K`(既定は高さ128px未満で16、128px以上で4、0でLOD無効)で変えられる。


#### カメラあり強化学習の1 step(物理4substep + 描画 + RGB/depthがGPU上のtorchテンソルとして使える状態まで)

```bash
PYTHONPATH=/path/to/torch-cu126 uv run python scripts/bench_camera_rl.py --camera front --envs 64 256 1024 4096 --sizes 64 128   # 前方視点(ロボットはほぼ写らない)
PYTHONPATH=/path/to/torch-cu126 uv run python scripts/bench_camera_rl.py --camera orbit ...                                       # 三人称視点(ロボット全体が写る、約40万三角形/env)
```

RTX 3060、Go2平地、**全envで丁度1カメラ**(モデル同梱の`track`を置換。上の`bench_camera_vs_isaac.py`の置換は`track`を残すので、現在のmenagerieのモデルでは mjmlx だけ2カメラ/envを描いており、上の表のmjmlx列は同条件ではない)、RGB+depth、torchは`device=cuda`、SMクロックは全計測で1837MHz、計測前に他のGPUジョブが無いことを確認(util 0%)。cpu/gpu経路を別プロセスで**交互に3回**走らせて中央値(プロセス内でも2ラウンドの中央値)。env-steps/s。mjmlxのみ計測(Genesisは過去値)。

- Vulkan->CUDA external memory interop(`scripts/mjmlx_camera_cuda.py`)で、描画結果をGPU上のまま`cuda`テンソルとして使う。

前方視点(`--camera front`):

| envs | px | 物理のみ | 描画のみ(env-frames/s) | **interop** | Genesis(過去値) | 倍率 |
|---|---|---|---|---|---|---|
| 64 | 64 | 29,993 | 296,503 | **24,273** | 10,400 | 2.3x |
| 256 | 64 | 102,601 | 849,761 | **76,307** | 32,100 | 2.4x |
| 1024 | 64 | 269,492 | 1,698,282 | **199,027** | 64,400 | 3.1x |
| 4096 | 64 | 456,442 | 2,112,591 | **337,618** | 82,900 | 4.1x |
| 64 | 128 | 30,658 | 257,009 | **23,082** | 10,300 | 2.2x |
| 256 | 128 | 95,593 | 654,376 | **69,929** | 29,100 | 2.4x |
| 1024 | 128 | 258,027 | 1,145,990 | **174,913** | 52,400 | 3.3x |
| 4096 | 128 | 473,909 | 1,285,141 | **273,812** | 66,100 | 4.1x |

三人称視点(`--camera orbit`、ロボット全体が写る重い場合)。LODは解像度別の既定(後述):

| envs | px | 物理のみ | 描画のみ(env-frames/s) | **interop(+128pxはLOD K=4)** | Genesis(過去値) | 倍率 |
|---|---|---|---|---|---|---|
| 64 | 64 | 29,719 | 110,413 | **21,250** | 10,400 | 2.0x |
| 256 | 64 | 99,535 | 157,113 | **53,839** | 32,100 | 1.7x |
| 1024 | 64 | 269,572 | 175,600 | **98,292** | 64,400 | 1.5x |
| 4096 | 64 | 468,667 | 179,121 | **124,415** | 82,900 | 1.5x |
| 64 | 128 | 30,264 | 108,247 | **20,873** | 10,300 | 2.0x |
| 256 | 128 | 99,612 | 149,680 | **51,695** | 29,100 | 1.8x |
| 1024 | 128 | 267,239 | 166,360 | **91,333** | 52,400 | 1.7x |
| 4096 | 128 | 463,801 | 169,719 | **113,679** | 66,100 | 1.7x |

Genesisの値は過去の計測(BatchRenderer(Madrona)、GPUテンソル出力)で、今回は再計測していない。GPU・カメラ位置・シーンの複雑さ・シェーディング・`env.step`に含まれる処理(mjmlxは`MjmlxGo2Env.step`の報酬込み)が揃っている保証はない。

ボトルネック(4096env、1 step、ms):

| | 前方64px | 三人称64px | 三人称128px |
|---|---|---|---|
| 物理step | 9.0 | 8.7 | 9.0 |
| 描画(GPU) | 2.0 | 23.0 | 24.8 |
| (うち頂点処理のみ。`MJMLX_CAM_EXP=discard`) | - | 12.7 | 14.0 |
| pack passの追加(render+pack - render) | 0.5 | 0.6 | 1.8 |
| テンソル取得 | 0.01 | 0.01 | 0.01 |

- LOD: 解像度が高いほど同じLODでもシルエット誤差が小さいので、`height >= 128`の既定は`MJMLX_CAM_LOD_K=4`、64pxは16。MuJoCo標準レンダラ(`scripts/compare_camera_mujoco.py`、三人称、16env)との比較:

| px | K | ロボットシルエットIoU | depth誤差 平均 / p95 | 描画のみ(4096env) |
|---|---|---|---|---|
| 64 | 16 (64pxの既定) | 0.960 | 28.2 / 75.2 mm | 181k |
| 64 | 4 | 0.899 | 30.2 / 77.4 mm | 424k |
| 128 | 16 | 0.989 | 13.9 / 37.2 mm | 60k |
| 128 | 4 (128pxの既定) | 0.965 | 14.9 / 37.9 mm | 178k |

128pxのK=4は64pxのK=16と同等のIoU(0.965 vs 0.960)。64pxのK=4はIoUが落ちる(0.899)ので既定にしていない。品質優先なら`MJMLX_CAM_LOD_K=16`にする。

制約・次の手: 三人称視点では描画(約23ms/4096env)が1 stepの約7割を占める。頂点処理が約半分で、次の手はメッシュレットによる細かい単位のカリング・LODの更新、背面カリング、Hi-Zによるオクルージョンカリング。interopはLinux+VK_KHR_external_memory_fd+CUDAのみ(同期は`vkQueueWaitIdle`で、GPU間セマフォではない)。Vulkanのデバイスは`MKX_VULKAN_DEVICE_INDEX`、CUDAは既定のデバイス0を使うので、複数GPUでは揃える必要がある。mac(MoltenVK)では`VK_KHR_external_memory_fd`が無いためCPU経由の読み出しになる(実機未検証)。


### 5. シミュレーション簡略化オプション(読み込み時、既定は正確なモデルのまま)

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
scripts/           ベンチ、カメラ/センサのヘルパー、ツール
scripts/examples/  Go2 / G1 / HalfCheetahの環境・学習・再生・CPU/GPU比較(scripts/examples/README_JA.mdを参照)
tests/             C++テスト、MJCFフィクスチャ
docs/              index.html (GitHub Pages)
extern/mlx_vulkan  Vulkanテンソルバックエンド (submodule)
```

## 開発ルール

画像・動画は絶対にcommit/pushしない(`.gitignore`済み)。PR/issueへGitHub上で添付し、READMEや`docs/index.html`からは`user-attachments`のURLで参照する。

## ライセンス

Apache 2.0
