# Go2 cmd_vel追従学習: 現状と次にやること

最終更新: 2026-10-24。ブランチ `feature/go2-flat-training`(一部未コミット)。

## 現状サマリ

- 階段地形・平地とも、まだ「立って歩く」歩容は出ていない。
- `isaac_flat` プリセット(Isaac Lab準拠の報酬)で最も改善した(`runs/go2_flat_isaacpreset`、500 iteration)。
  - 後退 -0.46(指令-0.5)、左 +0.38(+0.5)、右 -0.31(-0.5)は追従。**前進だけ静止(0.00)**。
  - 姿勢の傾き `base_tilt` は 0.09 rad に収まった(旧設定は 0.3〜0.8)。胴体高さは約 0.20 m と立位(約 0.3〜0.4)より低い。
- 学習速度はIsaac Labより約19倍遅い(下記)。

## 学習ログからわかったこと

| run | 設定 | 結果 |
|---|---|---|
| `go2_stairs_noclip`(階段) | action clipなし・torque 1/4 | 横倒しのまま静止。`base_tilt` 0.13→0.79、胴体接地でしか終了しないので倒れても終了しない |
| `go2_flat`(平地) | 同上 | iter 190で報酬ピーク後に崩壊。標準偏差 0.76→1.11、高さ 0.20 |
| `go2_flat_fast`(平地、1500 iter) | 罰則を弱めた(`relaxed`) | 標準偏差 2.2、`ctrl_saturated_frac` 0.68、高さ 0.16。前進は追従したが体を引きずる這い方で約3.8秒でリセット |
| `go2_flat_isaacpreset`(平地、500 iter) | `isaac_flat`(Isaac Lab準拠) | 標準偏差 0.54で安定、傾き 0.09、`lin_vel_error` 0.33(最良)。前進だけ静止 |

## Isaac Lab設定との差(原因の最有力)

`UnitreeGo2FlatEnvCfg` を読んだところ、旧設定は次の点で食い違っていた。

| 項目 | Isaac Lab(平地) | 旧設定 |
|---|---|---|
| `flat_orientation_l2` | -2.5 | 0(項なし) → 低姿勢の這い方を許容 |
| `feet_air_time` 重み | 0.25 | 0.01 |
| reset関節角スケール | 1.0 | 0.5〜1.5 |
| reset初速度 | 0 | ±0.5 |
| 罰則重み | `torque` -2e-4 ほか標準 | 弱めた値 |

`--reward-preset isaac_flat` で上の値になる(`scripts/mjmlx_go2_env.py` の `PRESETS`)。

## 速度比較(RTX 3060、4096 env、24 step/iteration)

| | Isaac Lab | こちら(MKX GPU) |
|---|---|---|
| iteration | 約2.1 s | 約41 s |
| 収集(物理+報酬) | 1.90 s | 約27〜40 s |
| PPO更新 | 0.28 s(GPU) | 約14 s(CPU) |
| step/s | 約45,000 | 約2,400 |

- 高速化PR #33(forward SoA化・TG調整・solver早期終了)で `env.step` は 1.61 s → 1.17 s(約1.38倍)。`scripts/bench_go2_step.py --build-dir <dir>` で前後比較できる。
- 1 substepの内訳(4096 env): solver 約110 ms、forward 38 ms、kinematics 30 ms、collision 11 ms。
- dt/decimationが違う(こちら 0.004×5、Isaac 0.005×4)ので完全に同条件ではない。
- Isaac Lab公式設定の300 iterationは完走(エピソード長 1000、標準偏差 0.35)。

## 調査ログ(速度, 2026-10-24)

### 実験1: env.stepの時間配分
- 4096 env、random action。`env.step` 約0.8 s(最小0.52 s)のうち、`sim.step` はGPUへ投げるだけで約7 ms、`sim.state()`(GPU同期) 約770 ms、`cfrc_ext` 約24 ms。python/numpy側(観測・報酬)は約10〜30 msで、全体の数%。
- 結論: 収集時間はほぼGPU物理。Python側の最適化は効かない。

### 実験2: ステージ内訳(`MJMLX_PROF=1`、build-new、substepあたり)
- cfrc 7 / collision 23 / euler 6 / forward 44 / kin 40 / solver 92 ms(合計約214 ms)。ただしprofは段ごとに同期するので1段あたり約10 msの同期コストが乗る(通常実行は約105〜156 ms/substep)。
- kinematicsは `MJMLX_TG_KIN` を 1/4/32 にしても 45/39/70 ms で、スレッド数を増やすと悪化する。
- kinematicsのアブレーション(`MJMLX_ABL`): 出力ループ3つ(xmat/ximat/geom)を全部消しても 43→20 ms、本体チェーンも消した空に近いカーネルでも約20 ms残る。つまりkin 40 msの約半分は「カーネル1発あたりの固定的なコスト(同期・バッファ確保など)」で、計算そのものではない。段ごとのprofだけでは真の内訳が分からない。
- `solver_iterations` を 8→1 にしても `env.step` の最小値は 553→528 ms でほぼ変わらない。solverの反復数は支配的ではない。

### 実験3: 他プロセスによるGPU占有(重要)
- `ollama runner`(別プロセス、pid 51333)がGPUメモリ約10.6 GBとCPU 70%、`ollama serve` がCPU 1000%超を使っている。
- `env.step` の平均(740〜1080 ms)は最小(520〜550 ms)の約1.5〜2倍にぶれる。これまでの速度比較(41 s/iter など)の一部は、この競合を含んでいる。**クリーンな条件では `env.step` は約520 ms**(24 stepで約12.5 s)で、Isaac Lab(1.9 s)との差は約6.6倍。
- 残りGPUメモリが約1 GBしか無く、CUDA版torchでのPPO更新(下記)も `OutOfMemoryError` で試せなかった。

### 実験5: env.stepが「接触あり」で2.5倍遅くなる(重要)
- GPUが空いた状態(他プロセス無し)で、zero actionの `env.step` を16回連続で測った(ms): `[479, 504, 506, 919, 1350, 1538, 1457, 1280, 1348, ..., 1113, 1218]`。
- 最初の3 stepは約500 ms(空中〜接地前)、接地して接触が増えると約1100〜1500 msに上がる。前節の「最小値 約520 ms」は接触が少ない状態の値で、学習中の実態は約1.2 s/step。
- つまり接触あり時の追加コスト約0.7 s/step(約140 ms/substep)がcollision+solverの接触処理。ここが最大のボトルネック。Isaac Labは約80 ms/step。

### 実験6: PPO更新をcuda化(クリーン条件、isaac_flat、4096 env、24 step)
- iteration時間: CPU PPO 約44 s → **CUDA PPO 約27 s**(約17 s短縮)。PPO更新は約17 s → 0.3 s未満に。
- 残りの約27 sは物理収集(約1.1 s × 24 step)。Isaac Lab(約2.1 s/iter)との差は約13倍に縮まった(旧: 約19倍)。
- 以降の学習は `PYTHONPATH=/home/owner/torch-cu126 ... --device cuda` で回す。

### 実験4(参考): PPO更新のGPU化の実装メモ
- 実装: `train_go2_ppo.py --device cuda`、envの出力tensorを `.to(device)`。
- 現行の `.venv` のtorch(2.14.1+cu130)はドライバ565(CUDA 12.7)に対して新しすぎてcudaが使えない。`/home/owner/torch-cu126` に torch 2.8.0+cu126 を入れた(`PYTHONPATH=/home/owner/torch-cu126` で有効。`.venv` は変更していない)。cuda自体は使えることを確認済み。
- 検証済み(実験6)。ollamaがGPUメモリを開けるまで待つ必要があった。

### 実験7: solverの拘束行列などを共有メモリへ(効果大)
- 接触あり状態(4096 env、`MJMLX_PROF=1`、substepあたり)のsolverは約195 ms。Newton反復数を変えると `solver ≒ 70 ms(固定) + 19 ms × 反復数`(`MJMLX_GTOL=0`、SI=1/2/4/8 で 91/123/147/211 ms)。
- `MAX_EFC` を 256→48 に下げても変わらない(メモリ量・ゼロ埋めは原因ではない)。
- 原因の見立て: `tid==0` の1スレッドが `nefc×NV` の行列ベクトル積や10回の二分法をglobalメモリ(`solver_scratch`)上で逐次に回す。ループ長が動的で展開されず、1回ごとにglobalメモリのレイテンシ(数百cycle)を待つ。
- 対策: nv<=32のモデルは J/D/aref/force/Jaref/act/floss/Jv/grad/search/qacc/Ma/Mv/H を `shared` に置く(`solver_use_shared`)。共有メモリに載せるため拘束行数上限を96に下げる(超過分の接触は落ちる)。`MJMLX_SOLVER_SHARED=0` で従来動作。
- 結果(接触あり、substepあたり): solver 195 → **49 ms**。`env.step`(4096 env) 平均 1434 → **755 ms**。
- 検証: `gpu_vs_cpu_humanoid / high_dof`、`model_gpu_vs_cpu_piper / go2_freefall / go2_landing / g1_freefall / g1_landing / h1_freefall` 全通過。

### 実験8: kinematicsの出力をSoA化(kin 34→3.4 ms)
- `MJMLX_PROF` を「ホスト側のグラフ構築(.build)」と「GPU評価(.eval)」に分離。構築は kin 前で約11〜14 ms/substep、他は1 ms前後。
- kinematicsのアブレーション(`MJMLX_ABL`)で、チェーン計算は約2.7 ms、**出力書き込み3ループが約30 ms**と判明。1スレッド=1envでAoS(env*S+k)に書くと、ワープ内の各レーンが別キャッシュラインに散らばり、書き込みトランザクションが約290万回になる。
- 対策: 出力を SoA(成分*B+env)にし、`MJMLX_TG_KIN` の既定を1→64に。隣接envが連続アドレスに書くのでコアレスする。forwardは入力をSoAで読む。vmap経路(GPU forwardを使わない経路)とstate出力、collision入力は転置してAoSに戻す。
- ポイント: **SoAはTG>=32のときだけ効く**(TG=1だと1ワープ1レーンなのでコアレスしない。SoA化だけ入れるとkinは34→44 msに悪化)。
- 結果(接触あり、substepあたり): kin.eval 34 → 3.4 ms、`env.step` 平均 755 → 665 ms。
- 検証: gpu_vs_cpu_humanoid / high_dof、piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験9: forwardの出力(作業領域)もSoA化(forward 37→6 ms)
- forwardカーネルは出力バッファ(qM/cinert/cvel/subtree_com/qfrc/cdof)を**作業メモリとして読み書き**しており、AoSのためスレッドごとに散らばった大量のglobalアクセスが発生していた。
- 対策: これらの出力を全部SoAにし(`X_out[off+e]` → `X_out[e*B+env]` を機械的に置換)、forward TGの既定を8→64に。solver/eulerもSoAで読む(solverはworkgroup=1env なので `NBATCH=gl_NumWorkGroups.x`)。state出力(subtree_com/cinert/cvel/qfrc_actuator)とvmap経路は転置でAoSに戻す。
- 結果(接触あり、substepあたり): forward.eval 37 → **6 ms**。`env.step`(4096 env) 平均 665 → **533 ms**。
- 検証: gpu_vs_cpu_humanoid / high_dof、piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験10: 不要なglobal書き込みの削除(collision 20→10 ms、solver 49→31 ms)
- collisionは毎回 `contact_data` の全域(1env 128接触×8 = 1024 float)を1スレッドでゼロ埋めしていた。solverとcfrcは `contact_count` 個しか読まないので削除(散らばった書き込み約420万回/substep)。
- solverは終わりに `contact_fn_out` を128個全部、thread 0が逐次に書いていた。`contact_count` 個だけに変更。
- 注意: 計測中にollama(別プロセス)がCPU/GPUを使うと `env.step` が2〜3倍にぶれる。比較は最小値(min)か、静かな時の複数回で見る。
- 結果(4096 env、静かな時): `env.step` 平均 533 → **約330 ms**(最小約245 ms)、約12,300 env-steps/s。接触あり段階の内訳(substep): collision 10 / solver 31 / forward 6 / kin 3 / euler 2。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験11: 現在地と、solverの内訳が測りにくい理由
- **学習1 iteration(isaac_flat、cuda PPO、4096 env × 24 step): 約6.9 s**(旧: CPU PPO 44 s → cuda PPO 27 s → 今回の物理最適化後 6.9 s)。Isaac Lab 約2.2 s に対して約3.1倍遅い(旧: 約19倍)。
- `env.step` 内訳(非prof、ゼロaction、4096 env、280 ms): GPU物理(`sim.state()`で同期) 227 ms、`cfrc_ext()` 取得 27 ms、`qfrc_actuator()` 6 ms、Python側 約15 ms。
- 計測中のGPUは SMクロック 780 MHz(最大2100)・消費電力 約31 W(TDP 170 W)・utilization 15〜100%で、飽和していない(throttle reason=idle)。rootが無いのでクロック固定(`nvidia-smi -lgc`)は不可。
- solverを段ごとに切り出す計測(`MJMLX_SABL=1..4`): 各点の状態を出力に書いて早期returnする(依存を切るとコンパイラが前段を消してしまうため)。拘束構築まで 11 ms、初期勾配まで 18 ms、CG/直線探索はほぼ0(`MJMLX_CGITERS`を10→1にしても変わらない)、全体 29 ms。反復数(SI 1〜8)・CG反復・`MAX_EFC`(共有メモリ量)を変えてもほぼ変わらない → 反復部分ではなく、固定的な前処理と、Newton反復の更新部分が効いている。
- 残る想定: `tid==0` が `nefc×NV` の行列ベクトル積などを逐次に回す部分の命令数が、見積もりより大きい。次はこれを全スレッド並列にする。
- 共有メモリ版はglobalメモリバリア(`memoryBarrierBuffer`)を外した(31→27.5 ms)。`MJMLX_KEEP_MBB=1` で従来動作。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験12: solverの前処理(ゼロ埋め・初期勾配)を削る
- `MJMLX_SABL=5`(ゼロ埋め直後)で測ると、solverの前処理のゼロ埋めだけで約6 ms/substep。共有メモリ版はJ/D/aref/floss等を行を作る時に全要素書くようにし、global scratchと共有メモリの一括ゼロ埋めを廃止(→0.8 ms)。
- 初期勾配は `qacc=0` なので `Ma=0`、`Jaref=-aref`。1スレッドで回していた行列ベクトル積を省き、行ごとの更新を全スレッドで並列化。
- **計測のぶれ対策**: 別プロセスのollamaが動くと各段が2〜3倍にぶれる。`scripts/prof_min.sh [build-dir] [回数]` で複数回回して段ごとの最小値を取る。
- 静かな時の最小値(substepあたり、接触あり): solver 18.4 / collision 4.7 / forward 3.2 / cfrc 2.5 / kin 1.3 / euler 1.0 ms(合計 約31 ms)。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験13: CPUとGPUの差(1 step比較)とeulerdampの修正
- `scripts/compare_cpu_gpu_go2.py`: GPU envで集めた接触を含む多様な状態を、CPU(MuJoCo C)とGPUに同じに設定し、同じactionで1制御step進めて関節角・胴体速度の差を測る(`--lift` で接触なしの状態)。長い軌道の比較はカオス的に増幅されるので1 stepで見る。
- 解像度の確認: Newton打ち切り(`MJMLX_GTOL`)・CG反復を変えても1 stepの差はほぼ変わらない(平均 約1.4e-2 rad) → 原因はソルバの精度ではない。
- **接触なし**(胴体を1 m持ち上げ)でも差が大きい(関節角 平均 2.0e-2 rad)ので、接触以外の物理の差と判明。原因: GPUのEulerカーネルが常に関節dampingを陰的に扱い、さらにactuatorの速度項(biasprm の kd)も陰的に畳み込んでいた。Go2のXMLは `eulerdamp="disable"` なのでCPUは全て陽的。
- 修正: 積分器がEulerで `eulerdamp` が無効のときは陰的dampingを入れない。actuatorの速度項の陰的処理は implicit/implicitfast のときだけ行う。
- 結果(1 step、関節角): 接触なし 平均2.0e-2 → 5.7e-4 rad(p95 3e-2 → 3e-6)、接触あり 平均1.4e-2 → 5.3e-3 rad(p95 2.3e-2 → 9.5e-3)。
- 残る差: 接触ありで平均5e-3 rad、最大9e-2 rad。接触まわり(foot の condim=6 のねじり/転がり摩擦行をGPUが持たない、接触面のsolimp/solrefの混合など)の可能性。
- 検証: humanoid / high_dof / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験14: solverの反復内のglobal読み出しを共有メモリへ(solver 45→12 ms)
- 反復あたり約5 ms(SI=1,2,4,8 で 11/15.6/27/45 ms)かかっていたが、CG・直線探索・Newton打ち切りを変えてもほぼ効かなかった。反復内で毎回、質量行列 `qM_in`(SoA配置で成分ごとに別キャッシュライン)をglobalメモリから読んでいたのが原因。
- 対策: 共有メモリ版は `qM` と `qfrc_smooth` を最初に共有メモリへ1回だけ読み込む(`QMV` / `QFS` マクロ)。さらに直線探索(Jv・合計・二分法・候補コスト)を全スレッドで並列化(結果はほぼ変わらず、効いたのは上記のqM)。
- 結果(接触あり、4096 env、substepあたり): solver 45 → **12 ms**(SI=8)。1 step後のCPU/GPU差は変わらず(関節角 平均5.1e-3 rad)。
- **学習1 iteration(isaac_flat、cuda PPO): 約2.9 s**(Isaac Lab 約2.2 s の1.3倍。収集 2.6 s / 更新 0.27 s)。
- Newton反復数: SI=4にしても1 stepのCPU差はほぼ同じ(5.6e-3 rad)で速くなる(旧solver時 45→27 ms)ので、速度が必要なら下げられる。現在はSI=8のまま。
- 検証: humanoid / high_dof / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験15: collisionのSoA読み出し+TG=32、cdof/subtree_comの共有メモリ化
- collisionが読むgeomの位置・姿勢をkinematicsのSoA出力のまま読み(転置をやめる)、`MJMLX_TG_COLL` の既定を1→32に。実験10で `contact_data` の全消去を無くしたので、TG>1でもコアレスの利点が出るようになった(TG=8/16/32/64 で 13.1/6.9/4.7/5.7 ms、TG=1 は7.3 ms)。
- solverは `cdof` / `subtree_com` も共有メモリに読み込む(効果は小さい)。
- 結果(substepあたり、最小値): collision 8.4 → 4.7 ms。合計 forward 3.3 / cfrc 2.5 / kin 1.8 / solver 11.9 / euler 1.2 / collision 4.7 = 約25 ms。
- 学習1 iteration は約2.9〜3.0 s(クロックのぶれ込みで実測は2.86〜3.05 s)。Isaac Lab 約2.2 s の約1.3倍。
- env.stepの内訳(非prof、約132 ms): GPU物理 約113 ms、`cfrc_ext` 取得 約7 ms、`qfrc_actuator` 約2 ms、Python側 約8 ms。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験16: solverの接触ヤコビ作業領域をlocal配列に(solver 12→5.7 ms)
- 反復数(SI)・limit/frictionloss行・`MAX_EFC`(共有メモリ量)を変えても残る約10 msの固定コストが、接触ごとの `jacp_tmp`(スレッド専用の54 float)をglobalの `solver_scratch` に置いていたことだった(接触スレッドが約300回の散らばったglobal読み書き)。共有メモリ版ではスレッドのlocal配列に変更。
- 結果(substepあたり、最小値): solver 11.9 → **5.7 ms**。合計 forward 3.5 / cfrc 2.7 / kin 1.8 / solver 5.7 / euler 1.1 / collision 4.9 = 約20 ms。1 step後のCPU/GPU差は不変(関節角 平均5.1e-3 rad)。
- **学習1 iteration(isaac_flat、cuda PPO、4096 env): 約2.3 s(収集 2.05 s / 更新 0.27 s)**。Isaac Lab 約2.2 s(収集 1.9 s)の約1.07倍。旧: 約44 s(CPU PPO・旧ビルド)。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験17: 計算バッファをVRAM(デバイスローカル)に置く(最大の効果、Isaac Labを上回る)
- 原因: `extern/mlx_vulkan` のバッファ確保は `HOST_VISIBLE|HOST_COHERENT` だけを要求していた。dGPU(RTX 3060、ReBAR無し: BAR1は256 MiB)ではこれがシステムメモリになり、**全カーネルのload/storeがPCIe越し**だった。これまでの「globalメモリの散らばったアクセスが極端に遅い」は、この上に乗った現象。
- 対策(サブモジュール `20niship/mlx_vulkan` の `perf/device-local-buffers` ブランチ、5e658dd): デバイスローカルかつホスト不可視のメモリ型があればそれを使い、CPUとのupload/downloadはステージングバッファ+`vkCmdCopyBuffer`経由に。型が無い/確保失敗ならホストメモリにフォールバック(統合GPU・lavapipeは従来通り)。`MKX_DEVICE_LOCAL=0` で従来動作。
- 結果(substepあたり、最小値): forward 3.5→1.0 / cfrc 2.8→0.3 / kin 1.8→0.6 / euler 1.1→0.5 / collision 4.8→1.6 / solver 5.8→5.6 ms。`env.step`(4096 env) 平均 92 → **52 ms**(最小 32 ms)、約79,000 env-steps/s(Isaac Lab 約45,000)。
- **学習1 iteration(isaac_flat、cuda PPO、4096 env、24 step): 約1.3 s(収集 1.0 s / 更新 0.27 s)**。Isaac Lab 約2.2 s(収集 1.9 s)の約1.7倍速い。旧: 約44 s。
- 1 step後のCPU/GPU差は不変(関節角 平均5.1e-3 rad)。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。
- 注意: 親リポジトリのサブモジュールポインタを `5e658dd` に更新。CIは `perf/device-local-buffers` のコミットをfetchできる必要がある。

### 実験18: GPU solverにcondim 4/6(ねじり・転がり摩擦)を追加(CPU/GPU差が約半分)
- 切り分け: Go2のXMLの足geom `condim="6"` を `3` にすると、1 step後のCPU/GPU差が約半分になる(関節角 5.1e-3 → 2.7e-3 rad、胴体速度 6.3e-2 → 2.4e-2)。GPU solverはcondim>=3を一律に4行(接線2方向×±)として扱っており、ねじり・転がり摩擦行を持たなかった。
- 実装(共有メモリ版・ピラミッド錐): 接触あたり `2*(condim-1)` 行。方向0,1=接線(並進)、2=ねじり(法線まわりの回転ヤコビアン)、3,4=転がり(接線1,2まわり)。各方向で `J_normal ± mu_k*J_k`。全行で同じD(mu0基準)を使う。CPUのスカラー実装(`src/constraint.cpp`)と同じ定式化。拘束行の上限(96)に合わせて接触の打ち切りを `MAX_EFC-10` に。
- 結果(1制御step後、接触あり): 関節角 平均 5.1e-3 → **3.0e-3 rad**(p95 9.5e-3 → 4.8e-3)、胴体速度 平均 6.3e-2 → **3.1e-2**(p95 1.45e-1 → 5.3e-2)。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall 通過。

### 実験19: 接触の margin/gap の扱いをMuJoCoに合わせる(CPU/GPU差が約8分の1)
- 切り分け: XMLの `margin="0.001"` を `0` にすると1 step後の差が 3.0e-3 → 4.3e-4 rad に下がる → marginの扱いがCPUと違う。他の候補(`impratio`、foot摩擦の `0.02 0.01`、solimp、CPUのNewton反復数)は影響が小さいか変わらなかった。
- 原因(3点): (1) 接触のmargin/gapを2geomの**和**にしていた(MuJoCoは `mju_max`)。(2) solverの接触の有効判定のしきい値を `margin + gap` にしていた(MuJoCoは `includemargin = margin - gap`)。(3) 拘束の `pos` が `dist` のままだった(MuJoCoは `pos = dist - includemargin`)。ここの(3)が大きく、marginが0.001でも接触の反発の基準位置が1 mmずれていた。
- 修正: `io.cpp` でmargin/gapを最大値に、`batched.cpp` で検出しきい値は `margin`・solver側は `includemargin = margin - gap`、solverで `pos = dist - includemargin`。
- 結果(1制御step後、接触あり): 関節角 平均 3.0e-3 → **3.8e-4 rad**(p95 4.7e-3 → 4.7e-4)、胴体速度 平均 3.1e-2 → 1.25e-2(p95 5.3e-2 → 1.6e-2)。実験13(eulerdamp)と実験18(condim 6)と合わせ、着手時の平均 1.4e-2 rad から約1/37。
- 検証: humanoid / high_dof / contact_stress / piper / go2_freefall / go2_landing / g1_landing / h1_freefall / batched_collision_primitives 通過。

## 次にやること(優先順)

1. **前進が静止する原因を調べる**: 前進cmdだけ0になる。`flat_isaacpreset` の最新ckptで、cmd方向ごとの学習中の報酬・脚の接地(`foot_contact_frac`)を見る。前進時のスポーン位置(向き)の偏りも確認。
2. **胴体高さ 0.20 m を上げる**: Isaac Labの立位は約0.3 m。関節の `DEFAULT_JOINT_POS`、`ACTION_SCALE`、PD(kp=50, armature=0.01)とIsaacのDCモータ(stiffness 25 / damping 0.5 / effort 23.5)の差を確認する。
3. ~~PPO更新をGPUに~~: 完了(実験6)。iteration 約44 s → 約27 s。
4. **物理の高速化(さらに)**: 実験5より、接触あり時のcollision+solver(約140 ms/substep)が本丸。接触数の上限(`MAX_CONTACTS_PER_ENV`)・solverのワークグループ構成・行列構築の見直し。: solverの厳密Cholesky化、kinematics+forward統合、substepを5→4(dt 0.005)。見込みは物理で2〜3倍。10倍には届かない。
5. 長めの学習(isaac_flat、1500 iteration)で収束するか見る。必要なら `entropy_coef` / `init_std` を調整。
6. 階段地形への展開は平地で歩けてから。傾き閾値での終了条件の追加も検討(横倒しのまま報酬を稼ぐ局所解を防ぐ)。

## 環境メモ

- Isaac Sim 5.1.0 / Isaac Lab v2.3.2: `/home/owner/isaac`(Python 3.11の専用venv、約16 GB)。リポジトリ外。
  - 実行: `OMNI_KIT_ACCEPT_EULA=YES /home/owner/isaac/.venv/bin/python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless`(`/home/owner/isaac/IsaacLab` で)。
  - `isaaclab.sh --install` は `flatdict` のビルドに失敗するので、`uv pip install "setuptools<81" wheel` のあと `--no-build-isolation` で入れ直した。
  - シェルの `python` は別のalias(3.13)になっているので、venvのpythonを直接指定する。
- 速度計測は他の学習を `kill -STOP` で止めてから行う(GPU共有でばらつく)。
- 本repoのビルド: `build-new`(PR #33取り込み後)、`build`(高速化前、比較用)。
- 公開済みの学習済みGo2モデルで、このobs/actionに合うものは見つからなかった(`unitree_rl_mjlab` はG1のみ)。

## 未コミット

- `scripts/mjmlx_go2_env.py`: 報酬プリセット(`relaxed` / `isaac_flat`)と `flat_orientation` 項
- `scripts/train_go2_ppo.py`: `--reward-preset`
- `scripts/play_go2_cmd_vel.py`: `--reward-preset`
