# 学習・検証のサンプル

エンジンの上で動くロボット環境、PPO学習、再生、CPU/GPUの比較。ここにあるのはサンプルコード(`libmjmlx`の一部ではない)で、エンジン直結のベンチ/ツールは[`scripts/`](../README_JA.md)に残している。

```
scripts/examples/
  common.py        Go2とG1の共通部分(クォータニオン関数、rsl_rl PPOの基本設定)
  go2/             Unitree Go2: mjmlx_go2_env.py, go2_terrain.py, train_go2_ppo.py, play_go2_cmd_vel.py,
                   compare_cpu_gpu_go2.py, example_go2_camera.py, example_go2_sensors.py
  g1/              Unitree G1: mjmlx_g1_env.py, train_g1_ppo.py, play_g1_cmd_vel.py, eval_g1_forward.py, compare_cpu_gpu_g1.py
  halfcheetah/     train_ppo.py, mjmlx_halfcheetah_vec_env.py(stable-baselines3のサンプル)
```

リポジトリのルートから実行する(例: `PYTHONPATH=<cu126等のtorch> uv run python scripts/examples/go2/train_go2_ppo.py ...`)。各スクリプトは自分のディレクトリ・`scripts/examples/`・`scripts/`を`sys.path`に入れるので、`common`・`mjmlx_sensors`・`mjmlx_camera_cuda`のimportはどこから実行しても通る。先にPython拡張をビルドしておく([トップREADME](../../README_JA.md#ビルド))。`--build-dir`でそのビルド先を指す。

ここで参照する動画はGitHubに添付したもので、リポジトリには含めない。

<table>
<tr>
<td align="center"><video src="https://github.com/user-attachments/assets/6b5d210c-da5b-401b-8277-287d5a3b85c3" controls muted width="400"></video><br><b>Go2</b> 平地cmd_vel(途中経過)</td>
<td align="center"><video src="https://github.com/user-attachments/assets/1c79aae8-1332-4f13-b42b-4f6cc9de6615" controls muted width="400"></video><br><b>G1</b> 前進専用、横から</td>
<td align="center"><video src="https://github.com/user-attachments/assets/539b7676-baa8-4600-8a9c-ddae2274d4c8" controls muted width="400"></video><br><b>G1</b> 前進専用、真上から</td>
</tr>
</table>

## Go2の学習(PPO、rsl_rl)

```bash
uv sync --group build --group go2

# 平地 + Isaac Lab(UnitreeGo2FlatEnvCfg)準拠の報酬、PPO更新はCUDA
PYTHONPATH=<cu126等のtorch> uv run python scripts/examples/go2/train_go2_ppo.py --build-dir build \
  --flat --reward-preset isaac_flat --device cuda --envs 4096 --max-iterations 1500 --logdir runs/go2_flat

# 階段地形(既定)。チェックポイントは--logdir配下に50 iterationごと
uv run python scripts/examples/go2/train_go2_ppo.py --build-dir build --envs 2048 --logdir runs/go2_stairs

# 学習済み方策で前後左右のcmd_velを与え、タイル動画(runs/video/)と追従誤差を出す
uv run python scripts/examples/go2/play_go2_cmd_vel.py --logdir runs/go2_flat --build-dir build --reward-preset isaac_flat --speed 0.5

tensorboard --logdir runs/
```

主なオプション: `--reward-preset {relaxed,isaac_flat}`、`--device cuda`、`--no-self-collision --max-condim 3`(高速モード)、`--use-sensors`(batchedのIMU/接触センサ経由の観測)、`--cpu`(GPUエンジンの代わりにMuJoCo C)。`MJMLX_GO2_INTEGRATOR=implicitfast`でPDアクチュエータを陰的に積分する。

Isaac Lab版からの簡略化: height_scannerなし(盲目方策)、env別の物理ドメインランダマイズなし(全envで単一モデル)、地形カリキュラムなし、`cmd_vel`はheading方式でなく`||cmd_vel||`範囲の直接サンプリング。

学習済み方策の再生(Go2、平地、`isaac_flat`、500 iteration、前後左右に0.5 m/sで指令。後退・左・右は追従するが前進だけ静止するので途中経過):

HalfCheetah-v4のPPOサンプル(stable-baselines3)は`scripts/examples/halfcheetah/train_ppo.py`(`just train-ppo`)。

## G1の学習(PPO、rsl_rl。Isaac Lab `Isaac-Velocity-Flat-G1-v0` の再現)

```bash
uv sync --group build --group go2 && just fetch-models        # menagerieのunitree_g1(29DoF)
PYTHONPATH=<cu126等のtorch> uv run python scripts/examples/g1/train_g1_ppo.py --build-dir build --device cuda --envs 4096 --max-iterations 1500 --logdir runs/g1_flat
uv run python scripts/examples/g1/play_g1_cmd_vel.py --logdir runs/g1_flat --build-dir build           # 前進/前進1.0/横/旋回のタイル動画 -> runs/video/g1_cmd_vel.mp4
uv run python scripts/examples/g1/compare_cpu_gpu_g1.py --build-dir build                              # MuJoCo Cとの1 step差
```

**再現元とその理由。** Isaac Lab v2.3.2の`Isaac-Velocity-Flat-G1-v0`を再現する: [`flat_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py)、[`rough_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py)(報酬を継承)、[`rsl_rl_ppo_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py)(`G1FlatPPORunnerCfg`)、[`velocity_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)、[`mdp/rewards.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py)、[`isaaclab_assets/robots/unitree.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_assets/isaaclab_assets/robots/unitree.py)の`G1_CFG`(アクチュエータ・初期姿勢)。理由: コードが実在し数値を全て読める、学習ライブラリ(rsl_rl PPO)がGo2環境と同じ、**平地(flat)cfgはheight scanner・質量/COMのランダム化・押しを使わない**(摩擦は0.8固定)ので本エンジンの制約(レイキャスト無し・1バッチ1モデル・地形カリキュラム無し)が影響しない、ローカルのIsaac Lab 2.3.2で同条件の学習曲線を比較できる。

検討して採用しなかったもの:
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)(`legged_gym/envs/g1/g1_config.py`): 12DoFのG1 URDF、隠れ層32のリカレントactor-critic、歩容位相寄りの報酬(`feet_swing_height`、`contact`など)でlegged_gym系。ライブラリもロボット定義も異なり、rsl_rlの学習と曲線を比較できない。
- MuJoCo Playground G1 joystick([`rss2025/.../locomotion/g1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/rss2025/mujoco_playground/_src/locomotion/g1/joystick.py)): MJX/Warp + Brax PPO、制御0.02s・物理0.002s、`feet_phase`/`pose`報酬、押しとenv別のドメインランダマイズ(`randomize.py`、env別モデルパラメータが必要で本エンジンには無い)。
- HumanoidVerse / humanoid-gym / Genesisのlocomotion例: 未調査(Genesisの例はGo2)。

**Isaac Labと一致させたもの:** dt 0.005 x decimation 4、20秒エピソード、action=既定姿勢+0.5 x action、観測(99次元=胴体の線/角速度・重力投影・指令・関節角/速度・前回action)とIsaacと同じ一様観測ノイズ、heading方式のyaw指令・静止env 2%・10秒ごとの再サンプル(vx 0..1、vy -0.5..0.5)、平地の報酬一式(termination -200、track_lin 1.0、track_ang 1.0、lin_vel_z -0.2、ang_vel_xy -0.05、hip+kneeのtorque/acc、action_rate -0.005、二足のfeet_air_time 0.75 / 0.4s、feet_slide -0.1、flat_orientation -1.0、足首のリミット、hip/腕/腰の偏差。Isaacの`RewardManager`と同じく制御stepを掛ける)、胴体接触で終了、`G1_CFG`のPDゲイン(脚150-200/5、足首20/2、腕40/10)、既定姿勢、自己衝突なし、既定姿勢とランダムyawでリセット、PPO設定(actor/critic [256, 128, 128] ELU、entropy 0.008、lr 1e-3 adaptive、24 step/env)。

**差分(比較時の注意):**
- ロボット: menagerieの`g1.xml`は29DoF(腰3軸、腕7軸、指なし)。IsaacのG1は腰1軸・腕5軸・指14関節。余分な腰/手首の関節は同じ高い剛性にし、既存の偏差ペナルティの対象にした。骨盤の初期高さは足が床に付くように計算(0.80m、Isaacは0.74m)。
- 衝突: 床と衝突するのは足の球・骨盤・胴体・膝のみ(Isaacの`G1_MINIMAL_CFG`に相当、厳密な集合は異なる)。`actuatorfrcrange`は外し、Isaacの`effort_limit_sim`(300/20)は未適用(エンジンがアクチュエータ力をクランプしない)。関節の`frictionloss`は0(Isaacと同じ)、床摩擦0.8。
- 報酬のトルクと関節加速度はPD則と制御step間の差分で計算。接触/滞空時間と足速度は制御stepに1回のサンプル(Isaacは物理stepごと)。足速度はMuJoCoの`framelinvel`センサ(1 substep遅れ)。
- 積分: `implicitfast`でPDの速度項を陰的に扱う(IsaacはPhysXの陰的アクチュエータ)。

**実測(RTX 3060、1024 env、seed 42、24 step/env/iteration、各300 iteration。エピソード平均報酬 / 平均エピソード長[step、最大1000]):**

| iteration | 本リポジトリ | Isaac Lab 2.3.2(同じタスク・同じPPO設定) |
|---|---|---|
| 10 | -7.27 / 40.2 | -5.96 / 40.6 |
| 50 | -5.45 / 42.1 | -5.40 / 47.1 |
| 100 | -5.29 / 61.9 | -5.12 / 54.9 |
| 150 | -4.99 / 69.0 | -5.24 / 72.8 |
| 200 | -4.86 / 71.3 | -6.87 / 179.2 |
| 250 | -4.92 / 81.6 | -11.76 / 802.9 |
| 299 | -4.95 / 86.8 | -8.64 / 967.8 |

**最初の約150 iterationは曲線が近いが、その後Isaac LabのG1は倒れずに立ち続ける(エピソード長が1000のうち約970)ようになるのに対し、本リポジトリは学習が進まない(約87 step=約1.7秒で転倒)。Isaac Labの挙動は再現できていない。** 原因は未特定。分かっていること: 零action(既定姿勢・上記PDゲイン)だと、menagerieのG1はMuJoCo Cでも本エンジンでも約1.5秒で倒れるので、GPUエンジン固有の問題ではない。CPU/GPUの一致は、接触なしの1 stepで平均2.5e-6 rad、足の接触で立っている状態で約5e-5 radで、衝撃を含むランダムaction状態では平均5.8e-4 / p95 2.3e-3 rad(Go2は3.6e-5。solver反復数・condimが原因ではない)。未検証の候補: menagerieモデルとIsaacのG1 USDの質量・慣性・足形状の違い(手、4球の足)、PhysXとの接触の柔らかさの違い、effort limit未適用、接触サンプリングが制御stepに1回であること。300 iterationの方策を`scripts/examples/g1/play_g1_cmd_vel.py`で再生すると、4つの指令すべてで約1.5秒後に転倒する。

G1の物理は現状このエンジンで**遅い**: 約4〜5K env-steps/s(nv=35。1024 envsでsolverが1 substepあたり約50ms、Newton約8反復を超えても遅くならない。Go2はenvあたり約100倍速い)。1024 envで1 iterationあたり本環境は約5秒、Isaac Labは空いたGPUで約1秒で、既定の1500 iteration x 4096 envは数時間以上かかる。共有メモリsolverの上限をnv<=32から40に上げる試行は効果が無く戻した。H1は未対応。

### 前進専用タスク(`--forward-only`): G1が歩く

上の全方向タスクは我々の学習では収束しなかったので、前進専用のタスクを追加した(全方向モードはそのまま): 指令は`vx`を[0.4, 0.8] m/sの範囲、`vy = wz = 0`、heading方式と静止envなし。

```bash
PYTHONPATH=<cu126等のtorch> uv run python scripts/examples/g1/train_g1_ppo.py --build-dir build --forward-only --solver-iters 4 --envs 1024 --max-iterations 800 --device cuda --logdir runs/g1_forward_v1
# 最終の報酬重み(下記)で追加学習(例: 400 iteration)。直進版v3は --fwd-weights yaw_rate=-0.5,heading_hold=-1,lateral_vel=-1 を付けてこれから再開
uv run python scripts/examples/g1/eval_g1_forward.py runs/g1_forward_v3/model_1746.pt --build-dir build   # 256 env x 20秒: 転倒・速度・yaw速度・向きの変化
uv run python scripts/examples/g1/train_g1_ppo.py --build-dir build --forward-only --solver-iters 4 --envs 1024 --max-iterations 400 --device cuda --logdir runs/g1_forward --resume runs/g1_forward_v1/model_799.pt
uv run python scripts/examples/g1/play_g1_cmd_vel.py --forward-only --logdir runs/g1_forward --build-dir build --seconds 12   # vx 0.4 / 0.5 / 0.6 / 0.8 のタイル、市松模様の床
```

**以前「約1.5秒で転倒」した原因(MuJoCo Cで切り分け。GPUエンジンは無関係):** Isaac Labの足首ゲイン(kp 20、kd 2)では、既定姿勢は零actionでも立てない。静的に釣り合う倒立振子には足首の剛性がm*g*h_com(約35kg x 9.81 x 0.69m = 240 Nm/rad)より大きいことが必要で、足首のauthorityも小さい(action scale 0.5 rad x kp 20 = 10 Nm)。MuJoCo Cで既定姿勢を5秒走らせて足首kpを走査: kp 100は倒れ、kp 150は倒れるか傾く(kdによる)、kp 200以上は立つ(傾き0.05〜0.08 rad)。そこで前進専用タスクは足首PDを**200 / 5**にする(全方向の既定はIsaacの20 / 2のまま)。これでGPUエンジンでも零actionで6秒以上立っていられる。MuJoCo Cでも同じように倒れるので、GPUエンジンが原因ではない。未検証: effort limit、接触サンプリング頻度、足の形状。最終方策は`scripts/examples/g1/eval_g1_forward.py`で評価した。

**報酬(前進専用):** Isaac Labのflat G1の項(termination -200、lin_vel_z、ang_vel_xy、hip/kneeのtorque・acc、action_rate、二足のfeet_air_time 0.75、feet_slide、flat_orientation、足首のリミット、hip/腕/腰の偏差。すべて制御stepを掛ける)に次の変更を加えた: `track_lin_vel_xy_exp` 1.0→**2.0**(前進速度が主タスク)、`track_ang_vel_z_exp` 1.0→**2.0**、新しく**alive +0.5**。最初に`track_ang`を0.5にした学習は、円を描いて歩く方策(yaw速度約0.65 rad/s、指令0.4〜0.8に対して前進速度は0.2〜0.3 m/sのみ)に収束したため、2.0に上げてそのチェックポイントから追加学習した。

**結果(RTX 3060、1024 env、速度のため`--solver-iters 4`、800 + 400 iteration、合計約70分):** 学習時の平均エピソード長は799 iterationで965 / 1000(報酬16.2)、最後は954 / 1000(報酬30.0。最大エピソード長は1000 step=20秒)。最終チェックポイントを観測ノイズありで評価(256 env x 20秒、決定論的): **20秒以内に転倒したenvは0 %**、平均前進速度**0.54 m/s(平均指令0.60 m/s)**(12秒の再生では指令0.4 / 0.5 / 0.6 / 0.8に対して0.45 / 0.49 / 0.53 / 0.60 m/s、転倒なし)、横方向のずれ0.08 m/s。**このv2の方策は歩きながらまだ旋回していた**(平均yaw速度0.45 rad/s)ので弧を描いて進んだ。下のv3で直した。指令0.8では速度が指令に届かないのは残っている。Isaac Labには前進専用の設定が無いので、このタスクではIsaac Labの比較は行っていない。

#### 直進させる(v3): 追加した報酬と結果

追加(前進専用モードのみ。全方向の報酬は不変。他の項と同じく制御stepを掛ける):
- `yaw_rate_l2`(重み -0.5): 世界座標のyaw速度の2乗。
- `heading_hold`(-1.0): エピソード開始時の向きからのyawのずれの2乗。
- `lateral_vel_l2`(-1.0): 胴体の横方向速度の2乗。
- 指令にIsaac Labのheading方式を入れた: `wz`指令 = clip(0.5 x (開始時の向き - yaw))。観測(99次元)を変えずに向きのずれを方策が観測できる(チェックポイントは互換)。`track_ang`(2.0)がこれを追従する。

重みは短い試験(512 env、各150 iteration、v2から再開)で決めた。評価は`scripts/examples/g1/eval_g1_forward.py`(256 env x 20秒、観測ノイズあり。向きの変化は累積なので、旋回すると360度を超える):

| モデル | 20秒以内の転倒 | 平均vx(指令0.60) | 平均vy | yaw速度の平均 | yaw速度の絶対平均 | 20秒の向きの変化 | 進行方向/横方向の変位 | 進行角 |
|---|---|---|---|---|---|---|---|---|
| v1: `track_ang` 0.5(800 it) | 0 % | 0.31 | 0.14 | 0.75 | 1.31 | +817度 | 0.34 m / 0.40 m | 51度 |
| v2: `track_ang` 2.0(+400 it) | 0 % | 0.54 | 0.08 | 0.45 | 0.58 | +399度 | 1.23 m / 0.94 m | 43度 |
| A: yaw -0.5、heading -1、lateral -1(+150 it) | 0 % | 0.54 | 0.04 | 0.11 | 0.40 | +7.9度(絶対値11.4) | 10.5 m / 1.52 m | 8.5度 |
| B: yaw -2、heading -3、lateral -2(+150 it) | 0.4 % | 0.50 | 0.07 | 0.11 | 0.42 | +23.8度 | 9.1 m / 3.23 m | 20度 |
| **v3: Aの重みで+400 it(最終)** | 0.4 %(256中1、7秒で転倒) | 0.51 | -0.00 | **0.06** | 0.29 | **-2.6度(絶対値4.0)** | 10.2 m / 0.74 m | **4.1度** |

(yaw速度はrad/s、速度はm/s。横方向の変位=20秒時点の開始時の向きの直線からの横ずれの絶対値の平均。)同じ150 iterationでは強い重み(B)の方が悪かった。最終方策はほぼ直線で歩く(20秒で向きは約4度しか変わらない。再生のタイルでも軌跡はほぼ直線で、指令0.8では9.5 mで最大約0.6 mの片寄りがある)。yaw速度の平均は0.06 rad/s(目標は約0.1)。絶対値の平均(0.29)は歩容の左右の揺れであり、旋回ではない。前進速度は指令に届かない(0.51 / 0.60 m/s、最大指令では0.60 / 0.80)ことと、256 envのうち1つが転倒したことは未解決。動画(ローカル、commitしない): `runs/video/g1_forward_straight.mp4`(横から、市松模様の床、20秒)と`runs/video/g1_forward_straight_topdown.mp4`(真上から、胴体の軌跡を赤、1 m刻みの基準線)。
