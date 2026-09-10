# [P0] `nv > 80` ゲートが実robotモデル(Go2/H1)を既存の高速batchedカーネル経路から締め出している

## 背景・経緯

`just bench` でGo2(四足, Unitree)・H1(ヒューマノイド, Unitree)を`mujoco_menagerie`から取得してベンチマークに追加したところ、以下の性能が観測された(MLX/Metalバックエンド、実機M系Mac、コミット`94d2da2`時点)。

| ベンチマーク | 条件 | steps/sec |
|---|---|---|
| `scalar/go2` | 1 env, CPU | 1,106 |
| `scalar/h1` | 1 env, CPU | 1,444 |
| `batched/pendulum` | 64 envs, GPU | 27,190〜28,149 |
| `batched/go2` | 32 envs, GPU | **40** |
| `batched/h1` | 32 envs, GPU | **23** |

batched(GPU)経路でGo2/H1が scalar の**1/40〜1/60**まで落ち込む。batched/pendulumは64 envsで27,000+ steps/secに達しているため、GPUバッチング自体が壊れているわけではない。Go2/H1固有の何かがボトルネックになっている。

## FACT(このセッション中にコードを読んで確認した事実)

1. `src/batched.cpp` にはnv(自由度数)が80を超える大規模モデル向けに、kinematics/forward/collision/solverの全フェーズをMetalカスタムカーネル(MSL)のみで実行する「all-Metal高速経路」が実装済み。該当箇所:
   - `make_forward_source(m)` (`src/batched.cpp` 内、行番号は今回のセッション中の編集でズレている可能性があるため関数名で検索すること)
   - `make_collision_source(m)`
   - `make_solver_source(m, solver_iters, cg_iters)`
   - 構築ガード: `if (m.nv > 80) { ... ctx->forward_kernel = ...; ctx->uses_metal_forward = true; }` および同様のガードがcollision/solverカーネル構築の直前に3箇所ある(`if (m.nv > 80 && m.cache.collision_pairs.size() > 0)`)。
2. **この高速経路は既にmesh衝突(GJK)をサポートしている。** `make_collision_source`のMSL本体を`grep -n "mesh"`で確認したところ、以下が実装済み:
   - plane-mesh collision(平面とメッシュの衝突、コメント`// Plane-mesh collision`)
   - mesh-mesh GJK collision(コメント`// Mesh-mesh GJK collision`、`mesh_vertadr_buf`/`mesh_vertnum_buf`を使った頂点ベースのGJK)
   - つまり「メッシュ衝突だからMetalカーネル化できない」という制約は存在しない。
3. `python3 -c "import mujoco; ..."` でGo2/H1の実モデルを読み込み、実測したDOF数:
   - **Go2: nq=19, nv=18, nbody=14, ngeom=57, nmesh=16, mesh形状のgeom数=33**
   - **H1: nq=26, nv=25, nbody=21, ngeom=54, nmesh=21, mesh形状のgeom数=21**
4. 上記より、Go2(nv=18)・H1(nv=25)は**いずれも`nv > 80`のゲートを満たさない**。したがって現在の`batched/go2`・`batched/h1`ベンチマークは、MLXバックエンドであっても高速Metalカーネル経路を一切使わず、`src/constraint_vmap.cpp`の汎用vmapパス(`vmap_gjk`/`vmap_gjk_depth`/`vmap_gjk_collision`、fixed-32-iteration GJK + サポート点ベースの深度推定)を通っている。
5. `CONFORMANCE.md`のPhase 3.3にはmesh/GJK/EPAの精度検証記録があるが、これはscalarおよびvmap経路の検証であり、batched Metalカーネル経路でのmesh衝突精度検証記録は見当たらない(要確認、下記HYPOTHESIS参照)。

## HYPOTHESIS(未検証の仮説、上記FACTから導かれる推測)

- **H1**: `batched/go2`・`batched/h1`が遅い主因は、GJK/EPAの反復計算(32〜64回のサポート点計算、条件分岐)がMLXの`mx::vmap`+`mx::compile`によって効率よくバッチ化・融合されず、env数×衝突ペア数×GJK反復回数に比例した非常に多いdispatch/グラフノード数を生んでいるため。これはMLX自体の一般的な制約というより、**このリポジトリのvmap GJK実装がMLXのグラフフュージョンと相性が悪い**可能性が高い(検証されていない)。
- **`nv > 80`という閾値は、大規模モデル向けMetalカーネルのスタック/レジスタ容量制約(コメント「Metal kernel stack memory is limited (~24KB per thread)」参照)から来ていると推測されるが、下限側の閾値ではなく単に「大きいモデル専用」という設計意図だった可能性がある。** つまりGo2/H1のような中規模(nv=18〜25)でmesh衝突を伴うモデルを高速経路に載せることは、開発時点で想定されていなかった可能性が高い。
- ゲート条件を`nv > 80`から「meshジオム数が一定以上」または「常にMetalカーネル経路を試み、構築失敗時のみvmapへフォールバック」に変更するだけで、Go2/H1のbatched性能が大幅に改善する可能性がある(未検証)。

## 対応方針(提案)

1. `src/batched.cpp`の3箇所のゲート条件(`if (m.nv > 80)` / `if (m.nv > 80 && m.cache.collision_pairs.size() > 0)`)を、Go2/H1のような中規模mesh衝突モデルでも高速経路に入れるよう緩和する。具体案:
   - 単純に閾値を下げる(例: `nv > 10`)、または
   - `m.cache.collision_pairs.size() > 0`(衝突ペアが存在する)を主条件にし、nv条件を撤廃、または
   - まず高速経路の構築を試み、失敗時(スタック容量超過など)のみ既存vmapへフォールバックする設計に変更。
2. ゲート変更後、`just bench`で`batched/go2`・`batched/h1`を再計測。
3. 変更が精度に影響しないことを確認するため、`tests/test_collision_mesh.cpp`相当のGo2/H1スケール(頂点数16〜60超)での数値検証テストを追加(→ Issue 06参照、テストの追加自体は別issue)。

## 定量目標

- **`batched/go2`(32 envs, MLX)を現状の40 steps/secから少なくとも10,000 steps/sec以上に改善する。**(pendulum batchedが27,000+ steps/secであることから、mesh衝突のオーバーヘッドを差し引いても同じ桁に乗ることを目指す)
- **`batched/h1`も同様に現状の23 steps/secから10,000 steps/sec以上を目指す。**
- 変更前後で`qpos`/`qacc`の数値差がscalar(CPU MuJoCo)基準比で既存の許容誤差(`CONFORMANCE.md`記載の基準、概ね1e-3〜1e-4オーダー)を超えないこと。

## 関連

- Issue 02(MKXへの同カーネル移植)はこのIssueの対応方針が確定してから着手する方が手戻りが少ない(MLX側でゲート条件・grid/threadgroupの妥当な値が固まってから、GLSL版に同じ値を反映できるため)。
- Issue 06(mesh衝突カーネルの精度検証テスト追加)はこのIssueの受け入れ条件でもある。
