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
