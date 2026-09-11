# [P0] MKXバックエンドのbatched(vmap)経路がbroadcast不整合でクラッシュし、現状のベースライン計測すらできない

## 解決済み(Issue 01 + 02の副作用)

Issue 01(PR #4)でforward Metalカーネルの構築を`nv > 80`ゲートから無条件化したことで、
`src/batched.cpp`のbatchedパイプラインは`use_metal_fwd`(=`ctx->uses_metal_forward`)が
常にtrueになり、本Issueが問題にしていた`vmapped_fwd`(`vmap_forward`/`vmap_gjk_collision`
経由の汎用vmapホストループ)が実行時に一切呼ばれなくなった。Issue 02(PR #5)でMKX側にも
forward/collision/solver GLSLカーネルを接続したことで、この経路はMKXバックエンドでも
同様に無条件で有効になっている。

`vmapped_fwd`のクロージャ自体は`build_context`内で今も構築されるが(未使用のまま)、
呼び出されることがないため、本Issueが報告していた`mx_compat_mkx: incompatible shapes
for broadcast: [2,3,3,] vs [2,1,]`は発生し得ない状態になった。

**検証**: 実機Vulkan(MoltenVK)、コミット`6135453`(Issue 01+02適用後)で`just bench-mkx`を
実行し、`batched/t_shape`(本Issueがクラッシュを報告していたモデル)・`batched/go2`・
`batched/h1`・`batched/pendulum`・`batched/high_dof_tree`の全batchedベンチマークが
クラッシュせず完走することを確認した:

| ベンチマーク | steps/sec |
|---|---|
| batched/pendulum | 22,436 |
| batched/t_shape | 11,807 |
| batched/go2 | 2,939 |
| batched/h1 | 1,674 |
| batched/high_dof_tree | 897 |

定量目標「`just bench-mkx`が`batched/t_shape`・`batched/go2`・`batched/h1`を含め、
クラッシュせず完走すること」は達成済み。精度についても`test_batched_diag`でMLX
バックエンドと同一の数値一致を確認済み(PR #5参照)。

根本原因(mkx shimのbroadcastチェックの厳密さとこのリポジトリのvmapコードのどちらに
問題があったか)自体は未特定のままだが、実行時にそのコードパスへ到達しなくなったため、
実害はない。将来nv<=80かつMKXで新たにvmap経路を使うモデル/設定(例: forwardカーネルの
構築に失敗した場合のフォールバック等)を追加する際は、このIssueの記録を参照して
broadcast不整合を再検証すること。

## 背景(オリジナル、Issue 01/02着手前の状況)

MLX→mkx(Vulkan)バックエンド移植後、`just bench-mkx`でGo2/H1を含むベンチマークを実行したところ、batched(GPU)経路が例外またはセグメンテーションフォルトで停止し、**現状(Issue 01/02の改善前)のMKXバッチ性能を一切計測できていない。**

## FACT

1. `just bench-mkx`実行時(実機Vulkan/MoltenVK、macOS)、以下のエラーで停止することを確認した:
   ```
   mx_compat_mkx: incompatible shapes for broadcast: [2,3,3,] vs [2,1,]
   ```
   これは`src/compat/mx_compat_mkx.h`の`mx::where`/`mx::multiply`等がbroadcast時に発生させる例外で、rank 3([2,3,3])とrank 2([2,1])の形状が数値的に整合しない(numpy broadcastルール上、2番目の次元が3 vs 2で不一致)場合に投げられる。
   - 発生源はおそらく`src/constraint_vmap.cpp`の`vmap_gjk_collision`関数内、`mx::where(state.overlap, overlap_result.frame, gap_result.frame)`周辺(このセッション中に`grep`で特定したが、`state.overlap`と`frame`の実際の shape がなぜこの組み合わせになるかまでは未特定)。
   - このエラーは、独立した別の実行では`batched/t_shape`ベンチマーク実行時に発生し、さらに別の実行(Go2/H1込みの完全なベンチマーク実行)では発生箇所が変わり、代わりに`Segmentation fault`で停止した(再現性はあるが、発生タイミング・具体的なクラッシュ内容が実行ごとに変わる)。
2. `test_math_full`(17/17 PASS)・`test_linalg_full`(11/11 PASS)は実機Vulkan上で問題なく成功しており、mkx shim自体の基本的な演算(スカラー算術・線形代数)は正しく動作している。**問題はbatched vmap経路(複数env・複数関節・複数contactにまたがる複雑なbroadcastパターン)に限定される。**
3. `src/batched.cpp`には、mkxの`vmap`がMLXと違い単一入出力しかサポートしないため、複数入出力(12入力/9出力)をホストループで処理する`mkx_vmap_batch0`という独自ヘルパーを実装した。このヘルパー自体は「1 envずつslice→forward_fn呼び出し→stack」という素朴な実装で、それ自体はMLXのvmapと数学的に等価なはずだが、**forward_fn内部で呼ばれる`vmap_forward`(smooth_vmap.cpp/constraint_vmap.cpp/solver_vmap.cpp/forward_vmap.cppにまたがる大きな関数群)が内部で行うbroadcastの一部が、mkx shimの厳密なnumpy broadcastチェックに引っかかっている。**

## HYPOTHESIS

- MLX本体の実際のbroadcastルールと、このセッションで書いたmkx shim(`src/compat/mx_compat_mkx.h`の`broadcast_shapes`関数)のルールが、何らかのエッジケースで食い違っている可能性がある。具体的には:
  - MLXは`mx::where(cond, x, y)`の3引数を独立にbroadcastするが、shimの実装もそれを踏襲しているはず(確認済み、コード上は正しく3引数とも`broadcast_shapes`を通している)。
  - `state.overlap`(GJKの重なり判定フラグ)が単一スカラーであるべき箇所で、実際には複数要素を持つ配列になっている可能性がある(例えば複数の衝突ペアや複数のシンプレックス頂点に対応する配列になっていて、期待されるscalar/(1,)ではなく(2,1)のような中途半端な形状になっている)。これはこのリポジトリのMLXコード自体の設計(1関数=1衝突ペア処理、という前提)と、実際に渡されるデータの形状が食い違っている可能性を示唆するが、**MLX実行時にはこの不整合が(たまたま)エラーにならずに動いてしまっている可能性がある**(MLXの内部実装がnumpy broadcastよりも緩い、または暗黙のsqueeze/reshapeを行っている可能性がある)。
- 上記が正しい場合、根本原因はこのリポジトリのvmap衝突判定コード側にある「shapeの不整合(ただしMLX上ではたまたま無害)」であり、mkx shim側だけを緩めても本質的な修正にはならない可能性がある。

## 対応方針

1. まず「発生源のarrayの実際のshapeをログ出力する」デバッグを行い、`state.overlap`・`overlap_result.frame`・`gap_result.frame`の正確な形状をクラッシュ時に特定する(`mx_compat_mkx.h`の例外メッセージは既にshapeを含めるよう改善済みだが、呼び出し元のどの変数がその形状を持つかまでは紐付けられていない)。
2. MLXバックエンドで同じ箇所に一時的なshapeロギングを入れ、MLX上での実際の形状と比較する。
3. 不整合がこのリポジトリのvmapコード側のバグであれば、そちらを修正する(mkx shimではなく)。mkx shim側の厳密チェックが「たまたま今まで隠れていたバグ」を顕在化させた可能性が高いため、shim側を緩めるのではなく元コードを直すことを優先する。
4. Issue 01/02が先に完了しGo2/H1がvmap経路を経由しなくなれば、このクラッシュは(Go2/H1に関しては)回避される可能性がある。ただしT-shapeタスクや将来追加されるnv<=80の他モデルは引き続きvmap経路を使うため、根本修正は必要。

## 定量目標

- `just bench-mkx`が`batched/t_shape`・`batched/go2`・`batched/h1`を含め、クラッシュせず完走すること(steps/secの値自体は低くても構わない、まずは「測れる」状態にすることが目標)。
- MLXバックエンドでの同一モデル・同一ステップ数実行結果(`qpos`/`qacc`)と、MKXバックエンドでの実行結果が、既存の許容誤差(`CONFORMANCE.md`基準)内で一致すること。

## 関連

- Issue 01/02が完了すればGo2/H1はこのクラッシュの影響を受けなくなる可能性が高いが、根本原因の特定・修正は独立した価値がある(他の小規模モデルのMKXサポートに影響するため)。
