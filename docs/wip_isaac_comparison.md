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

## 次にやること(優先順)

1. **前進が静止する原因を調べる**: 前進cmdだけ0になる。`flat_isaacpreset` の最新ckptで、cmd方向ごとの学習中の報酬・脚の接地(`foot_contact_frac`)を見る。前進時のスポーン位置(向き)の偏りも確認。
2. **胴体高さ 0.20 m を上げる**: Isaac Labの立位は約0.3 m。関節の `DEFAULT_JOINT_POS`、`ACTION_SCALE`、PD(kp=50, armature=0.01)とIsaacのDCモータ(stiffness 25 / damping 0.5 / effort 23.5)の差を確認する。
3. **PPO更新をGPUに**: 14 s → 約0.3 sで、iterationが約41 s → 約27 sになる見込み(torchのデバイスをcudaにするだけの可能性)。以前は「不要」との指示。再確認して決める。
4. **物理の高速化(さらに)**: solverの厳密Cholesky化、kinematics+forward統合、substepを5→4(dt 0.005)。見込みは物理で2〜3倍。10倍には届かない。
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
