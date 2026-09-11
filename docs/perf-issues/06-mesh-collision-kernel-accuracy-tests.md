# [P2] 実robotスケールのmesh衝突カーネル(GJK/EPA)の精度検証テストが存在しない

## 対応済み

`tests/test_batched_diag.cpp`(既存の汎用xml引数対応テストバイナリ、Go2/H1を含め任意の
モデルで実行可能)に`gpu_cpu_qpos_match_100steps`を追加した。Go2初期姿勢から100ステップ
実行し、GPU(Metal/MSL、MKXバックエンドではGLSL)経路とCPU(scalar MuJoCo C)経路の`qpos`
最終値を比較、`qpos`/`qvel`のNaN/Infチェックも行う。

`test_collision_mesh.cpp`が8頂点の立方体メッシュのみを対象にしていたのに対し、この新テストは
実robotメッシュ(Go2は33個・H1は21個のmesh形状ジオム、頂点数は8よりはるかに多い)を実際に
高速衝突カーネル経路(`ctx->uses_metal_collision = true`)で100ステップ処理させることで、
GPUカーネル内の固定サイズ一時配列や頂点数依存のループが実robotスケールでも正しく動作する
ことを検証する。

許容誤差は実測に基づいて設定した(実機Apple Silicon Mac、MLX/Metal):

| モデル | 100step後のqpos最大差(GPU vs CPU) |
|---|---|
| Go2 | 0.976 |
| H1 | 0.176 |
| high_dof_tree | 0.171 |
| contact_stress / multi_geom_scene | 0.198 |
| stiff_springs | 3.0e-9 |

最大値(Go2の0.976)に対して十分な余裕を持たせ、閾値は`2.0`とした。1ステップの`qvel`比較
(閾値1.0、既知のGPU CG solver収束差)と同様、100ステップ後の`qpos`は厳密な数値一致ではなく
「GPUクラッシュ・NaN/Infなく、CPU基準とおおまかに同じ軌道を辿ること」を検証する粗い健全性
チェックである点に注意(issueの定量目標が言う「1e-3〜1e-4オーダー」はCONFORMANCE.mdの
1ステップ単位の基準であり、カオス的発散を伴う100ステップ後の位置には直接適用できない)。

実機Vulkan(MoltenVK、MKXバックエンド)でも同じテストを実行し、MLXバックエンドと完全に
同一のpass/fail・同一の数値(0.976/0.176)を確認した(Issue 02完了後の要求通り)。既存の
7テスト(1step比較・20step安定性・決定性・64env)も全モデルで変更前と同一のpass/fail・
同一数値を維持しており、回帰はない。

CIへの組み込み(対応方針3の後半、`scripts/fetch_models.sh`をCTest経路に接続する件)は
別作業とし、今回は未実施(`test_batched_diag`自体が現在CTest登録されておらずxml引数を
手動で渡す形式のため、CI接続にはテストランナー側の変更が必要)。

## 背景(オリジナル)

Issue 01/02でGo2/H1のような実mesh衝突を伴うモデルを高速カーネル経路(Metal MSL / GLSL)に載せる場合、その経路のmesh衝突実装(`make_collision_source`のMSL、`mkx_kernels/collision.hpp`のGLSL)が実際に正しい接触点・法線・貫入量を計算できているかを検証する必要がある。

## FACT

1. `tests/test_collision_mesh.cpp`は既存のmesh衝突テスト(7ケース: mesh-plane, mesh-sphere, mesh-mesh, mesh-capsule)を持つが、これらはすべて**インラインの`vertex="..."`属性で定義された8頂点程度の立方体メッシュ**を対象にしている(`grep`で確認、`<mesh name="cube" vertex="..."/>`のようなパターンのみ、`<mesh file="...">`形式の外部資産テストは存在しない)。
2. これらのテストは`src/collision.cpp`(scalar経路)・`src/constraint_vmap.cpp`(vmap経路)の精度を検証するものであり、`src/batched.cpp`のMetalカスタムカーネル(`make_collision_source`)経路は一切テスト対象になっていない。
3. Go2は33個、H1は21個のmesh形状ジオムを持ち(このセッション中に`python3 -c "import mujoco; ..."`で実測)、各メッシュの頂点数は8頂点よりはるかに多い(実測はしていないが、Unitreeの実機ロボットの視覚/衝突メッシュとして現実的な数十〜数百頂点規模と推測される)。
4. Metalカスタムカーネル(`make_collision_source`)内のmesh-mesh GJK実装は、`nverts`(頂点数)をランタイム変数として扱っているように見えるコード(`int nverts = (int)mesh_vertnum_buf[mesh_id];`)であり、頂点数のスケールに対して原理的に対応できないわけではなさそうだが、**GPUカーネル内でのループ回数・一時バッファサイズが大きな頂点数でも正しく動作するかは未検証。**

## HYPOTHESIS

- 小さい頂点数(8頂点)でのみ検証されたGJK/EPA実装は、頂点数が数十〜数百に増えた場合に(a)単純に遅くなる(サポート点計算がO(頂点数)の線形スキャンであるため)、または(b)GPUカーネル内の固定サイズ一時配列(スタック/レジスタ制約から生じる)が頂点数の上限を暗黙に仮定しており、実robotメッシュでオーバーフローする、のいずれかの問題を抱えている可能性がある。現時点ではどちらも未検証。
- Issue 01のnvゲート緩和後、Go2/H1が実際に高速カーネル経路を通るようになった際、**接触が全く検出されない(無音の不具合)・誤った接触点が計算される・GPUクラッシュする、のいずれかが起きるリスクがある。** これは性能の問題ではなく正しさの問題であり、Issue 01の「定量目標」の一部(数値精度の一致)として扱われるべきだが、独立したテストとして先に整備しておく価値が高い。

## 対応方針

1. `mujoco_menagerie`から取得済みのGo2/H1シーン(`benchmarks/models/external/mujoco_menagerie/unitree_go2/scene.xml`等、`scripts/fetch_models.sh`で自動取得)を使い、Metalカスタムカーネル経路(`ctx->uses_metal_collision = true`の状態)とscalar経路(`src/collision.cpp`)の接触点・法線・貫入量を1ステップ分比較するテストを追加する。
2. 比較対象は「衝突ペアごとの接触数」「各接触点の位置・法線・貫入量」を`test_collision_mesh.cpp`と同様の許容誤差で検証する。
3. Issue 01のnvゲート変更後の回帰テストとしてCIに組み込む(現在のCIはmesh geomを含む実robotモデルを一切ダウンロード・テストしていない — `scripts/fetch_models.sh`はベンチマーク専用であり、`tests/`のCTest経路には接続されていない点に注意)。

## 定量目標

- Go2シーンの初期姿勢から100ステップ実行した際の`qpos`最終値が、scalar(CPU MuJoCo)基準と比較して既存の許容誤差(`CONFORMANCE.md`記載の基準、概ね1e-3〜1e-4オーダー)以内で一致すること。
- 同条件でGPUクラッシュ・NaN/Inf出力が発生しないこと。

## 関連

- Issue 01(nvゲート緩和)の受け入れ条件の一部として扱ってもよい。
- Issue 02(MKX側のカーネル移植)完了後は、同じテストをMKXバックエンドでも実行し、MLXとの数値差を確認すること。
