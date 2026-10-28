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

## Performance measurement(学習速度・CPU/GPU差・Isaac Labとの比較)

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

注記(条件差): Isaacの出力はGPUテンソルのまま(CPU読出なし)、こちらの「描画のみ」もGPU上まで、「描画+読出」はrgb+depthをnumpyへ。Isaacの物理+描画は`env.step`全体(観測・報酬・イベントを含む)、こちらは`MjmlxGo2Env.step`(報酬含む)。Isaacは物理dt 0.005 x4、こちらも同じ。カメラ位置/画角(水平47.5度)は揃えたが、近遠面・照明・シェーディングは異なる(こちらはPhongのみ、Isaacは最小設定のRTX)。Isaacの1024x128と4096は1800sのタイムアウトで未完(原因は未調査。VRAM不足と断定はできない)。旧はGPU上のフラスタム/LODカリング導入前(Go2のvisual meshが約40万三角形/envありジオメトリ律速で、64pxでIsaacの0.35〜0.52倍)、新は導入後(1ドローのまま、GPU上のcompute passで(env,cam,geom)毎にフラスタム球カリングとLOD選択、メッシュは頂点クラスタリングで1/4,1/16,1/64に簡略化)。この比較のカメラはIsaacと同じ前方視点でロボット自身はほぼ写らない。ロボット全体が写る三人称カメラ(`--camera orbit`)では 1024env 64px で 描画のみ 30,072、物理+描画 14,421 env-frames(steps)/s。LODの強さは環境変数 `MJMLX_CAM_LOD_K`(既定16、0でLOD無効)で変えられる。


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

### カメラセンサ(Vulkan, GPUバッチsim専用)

MJCFの`<camera>`/`<light>`/geomのrgbaを使い、全env x 全カメラを`vkCmdDrawIndexedIndirect` 1回でレンダリングする(Phong、RGB+depth)。物理stepとは別パスで、`create_camera`した時のみパイプラインを構築する。

```python
cam = native.create_camera(sim, 128, 128)  # sim: create_batched(use_gpu=True)の戻り値
cam.render()
rgb, depth = cam.rgb(), cam.depth()        # (num_envs, ncam, H, W, 3) uint8 / (num_envs, ncam, H, W) float32(m)
```

例: `scripts/example_go2_camera.py`、MuJoCo比較: `scripts/compare_camera_mujoco.py`、速度: `scripts/bench_camera.py`。カメラはCMakeオプション `MJMLX_BUILD_CAMERA`(既定ON)で分離されている。OFFにすると `mjmlx_camera_*` はスタブ(`create_camera` が明確なエラー)になり、カメラ無しの学習・ビルドには影響しない。ONでも、graphics+compute queue family・multiDrawIndirect・drawIndirectFirstInstance・shaderClipDistanceのいずれかを持たない/mkxが先に初期化された場合は `create_camera` だけが失敗する(理由をstderrに出す)。MoltenVK(mac)でのこれらのfeatureの有無は未検証(実機が無い)。カメラ用デバイスは `mjmlx_load_model` 時にmkxへ注入される(`VulkanBackend::use_external_device`)。環境変数 `MJMLX_CAMERA_DEVICE=0` で注入を無効化できる。

未対応: 影・テクスチャ・hfield・半透明・可動bodyに付いたlight・trackcom(body原点で近似)。

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
- **Spatial tendons (wrapping geometry)** -- batched GPU path only (site/pulley/sphere・cylinder wrap/sidesite, matches MuJoCo C); the scalar/vmap paths still support fixed tendons only
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
