# [P1] MKX(mkx/Vulkan)のCPU側グラフ処理・ディスパッチオーバーヘッドを計測・削減する

## 追記: 根本原因を特定、mlx_vulkan本体(upstream)側を修正

batched/go2・pendulum・t_shapeがMLXに対して著しく遅い(15〜25%程度)原因を`VulkanBackend::dispatch`
(`mlx_vulkan`本体、`src/mkx/vulkan/vulkan_backend.cpp`)のソースを直接確認して特定した。

**原因**: `dispatch()`が呼び出しごとに独立してcommand buffer/descriptor poolのalloc+begin、
`vkQueueSubmit`、**`vkQueueWaitIdle`**、free/destroyを行っていた。`mkx::eval<Backend,
Arrays...>`は複数ノードをトポロジカルソートして順にdispatchする設計だが、dispatchのたびに
`vkQueueWaitIdle`でGPU完了を待ってしまうため、1回のeval()内にdispatchがN個あれば同期がN回
発生し、CPU-GPUパイプライニングが完全に阻害されていた(CPU側は次のcommand buffer構築すら
GPU完了を待ってからしか始められない)。`bench_utils.h`のベンチ実装は複数ステップを走らせた
後に1回だけ`get_xpos`でevalする設計だが、compile()がno-op(Issue 05)なため各ステップの
custom kernel(kin/forward/collision/solver/euler、最大5個)がそのままdispatch数として
積み上がる。1ステップあたりの実計算が小さいモデル(pendulum、go2のnv=18等)ほど固定オーバー
ヘッドの比率が大きく、逆に計算量が大きいhigh_dof_tree(nv≈69)はMLXと同等以上だったことも
この説明と整合する。

**修正**: `mlx_vulkan`(20niship/mlx_vulkan)本体を修正し、同一eval()内の複数dispatchを1つの
command bufferに積み、次の`wait_idle()`でまとめて1回`vkQueueSubmit`+`vkQueueWaitIdle`する形に
変更した(PR: https://github.com/20niship/mlx_vulkan/pull/1 )。dispatch間はchained op向けの
保守的なshader write→readバリアで正しさを保っている。`eval<Backend, Arrays...>`/
`ComputeBackend` conceptのシグネチャは変更していないため、このリポジトリ側の呼び出しコードの
変更は不要で、`CMakeLists.txt`の`MKX_GIT_TAG`を更新するのみで取り込める。

**検証**: `mkx_tests`(mlx_vulkan本体の53 test cases/2600 assertions)全pass。このリポジトリの
`test_batched_diag`(Go2/H1)・`test_math_full`・`test_linalg_full`も変更前と完全に同一の
pass/fail・同一の数値を維持(回帰なし)。

**ベンチマーク(実機Vulkan/MoltenVK、64 envs)**:

| ベンチマーク | mx::eval統合後(このIssueの対応方針1) | command buffer統合後 | 倍率 |
|---|---|---|---|
| batched/pendulum | 16,460 steps/sec | 48,353 steps/sec | ×2.9 |
| batched/t_shape | 9,203 | 24,517 | ×2.7 |
| batched/go2 | 3,128 | 8,449 | ×2.7 |
| batched/h1 | 3,378 | 5,640 | ×1.7 |
| batched/high_dof_tree | 905 | 1,050 | ×1.2 |

MLX比も大幅に改善した(pendulum 16%→46%、t_shape 24%→65%、go2 22%→59%、h1 43%→72%)。
定量目標「env数に依存しないCPU側オーバーヘッドを500us未満」の直接検証は依然未実施だが、
支配的要因(dispatchごとの同期待ち)は解消したため、残るギャップは主にGLSL側のGJK/EPA
カーネル自体の実行時間(MSLとの実装差・コンパイラ差)と推測される(未計測)。

## 対応状況(対応方針1のみ実施、2・3は未着手)

`src/compat/mx_compat_mkx.h`の`mx::eval(Arrays&... arrs)`(複数配列を1回の呼び出しで渡す形)を、
配列ごとに個別`mkx::eval<Backend>()`を呼ぶループから、`mkx::eval<Backend>(proxies...)`を
1回だけ呼ぶ形に変更した(`std::apply`でtupleからlvalue参照パックへ展開)。`eval({a,b,c})`
(initializer_list形式)も同様に、内部の`mkx::detail::topo_sort`/`eval_node`を直接呼んで
1回の`Backend::wait_idle()`にまとめた。

これにより`mx::eval(a, b, c)`という1回の呼び出し構文を使っている箇所(`src/batched.cpp`内に
複数箇所存在、例: `mx::eval(ctx->solver_pair_props, ctx->solver_body_dof_masks,
ctx->solver_body_rootid)`)は、GPU同期(`wait_idle`)がN回→1回に減った。ただし、
`mx::eval(a); mx::eval(b); mx::eval(c);`のように**別々の文として書かれている**箇所(FACT 3
に記載の初期化コードなど)はこの変更の対象外(呼び出し自体が複数回発生しているため、shim側の
修正だけでは統合できない)。

### 検証結果(実機Vulkan/MoltenVK)

`test_math_full`(17/17)・`test_linalg_full`(11/11)・`test_batched_diag`(Go2/H1/
contact_stress/stiff_springs/multi_geom_scene/high_dof_tree)は変更前と完全に同一の
pass/fail・同一の数値を維持(回帰なし)。

`just bench-mkx`のsteps/sec比較(64 envs、変更前は本セッションのIssue 02 PR #5時点の
実測値):

| ベンチマーク | before | after | 変化 |
|---|---|---|---|
| scalar/pendulum | 13,024us/step | 11,214us/step | 約14%高速化 |
| scalar/go2 | 18,439us/step | 16,810us/step | 約9%高速化 |
| scalar/h1 | 17,280us/step | 14,754us/step | 約15%高速化 |
| batched/go2 | 2,939 steps/sec | 3,785 steps/sec | 約29%向上 |
| batched/h1 | 1,674 steps/sec | 2,392 steps/sec | 約43%向上 |
| batched/pendulum, t_shape | - | - | stddevが大きく(±30〜50%)有意差を判定できず |

scalar(単一env、GPU同期回数が相対的に多い)・batched go2/h1で一貫した高速化が確認できた。
pendulum/t_shapeは測定ノイズが大きく本変更単独の寄与を切り分けられなかった(複数回再実行して
中央値を取るなどの統計的な検証は未実施)。

### 未着手(対応方針2・3、定量目標の「CPU側固定オーバーヘッドを500us未満」の直接検証)

- トポロジカルソート・ハッシュ計算のCPU時間とGPU実行時間を切り分けるマイクロベンチマークは
  書いていない。
- `mx::eval(a); mx::eval(b);`のように分割呼び出しになっている初期化コード(FACT 3)の統合は
  未実施(construction時の1回限りのコストであり優先度を下げた)。
- 上記の理由により「env数に依存しないCPU側オーバーヘッドを500us未満に抑える」という定量目標
  そのものは未検証。今回の変更で改善したのは事実だが、目標達成の可否は追加の計測が必要。

## 背景(オリジナル)

Issue 02(高速カーネル経路のMKX移植)が完了しても、mkx自体の実行モデル(グラフ構築・トポロジカルソート・Vulkanディスパッチ)にCPU側の固定オーバーヘッドがあり、これがMLX(Metal)と同等の数万steps/secに到達する上での残存ボトルネックになる可能性が高い。この項目は「測ってから直す」ための調査タスクであり、具体的な修正内容はプロファイリング結果次第で変わる。

## FACT

1. `mkx::eval<Backend>(arrs...)`(mlx_vulkan / `mkx/core/eval.hpp`)は、呼び出しごとに:
   - 未評価ノードの集合に対してトポロジカルソートを行う(`std::unordered_set<OpNode*>`, `std::vector<NodePtr>`を使ったC++側の処理)。
   - 各ノードについて`shader_source_for(node.type)`でGLSLソース文字列を生成し、`std::hash<std::string>`でハッシュ化してパイプラインキャッシュを引く。
   - 全ノードのdispatchが終わった後、`Backend::wait_idle()`を**同期的に**呼び出す。
   - これらはすべてノード数に比例したCPU側処理であり、Vulkan API呼び出し(`vkCmdDispatch`等)そのものとは別に、C++オブジェクト生成・ハッシュ計算・マップ検索のオーバーヘッドが発生する。
2. `mx::eval`(このリポジトリの`src/compat/mx_compat_mkx.h`shim)は、複数配列を渡された場合(`eval(a,b,c,...)`)、**1つずつ個別に`mkx::eval<Backend>(proxy)`を呼ぶ**実装になっている(`for (auto& p : proxies) mkx::eval<Backend>(p);`)。これは配列ごとに独立したトポロジカルソート+`wait_idle()`を発生させるため、MLXの「複数配列をまとめて1回のトポロジカルソートで評価する」挙動と比べて**GPU同期回数が意図せず増えている可能性がある**(未計測)。
3. batched.cpp内、`ctx`構造体の各モデル定数配列(`body_parentid`, `body_pos`等)の初期化時に`mx::eval(...)`が個別に多数回呼ばれている箇所がある(コード中に`mx::eval(ctx->body_parentid); mx::eval(ctx->body_pos); mx::eval(ctx->body_quat);`のように複数行に分けて書かれている箇所が複数存在)。これは初期化時の1回限りのコストであり、毎ステップのボトルネックではないが、上記2の実装により本来1回で済むはずの同期が複数回に分割されている典型例。

## HYPOTHESIS

- 現状の`mkx_vmap_batch0`(Issue 02で置き換え予定)によるホストループでは、env数B×forward_fn内のノード数(MLXのvmap相当グラフで「~945ノード/27自由度モデル」という記述が`ARCHITECTURE.md`にあり、これがおおよその目安になる)個のOpNodeがCPU側で生成される。B=32(現状のbatched/go2設定)だけでも数万ノード規模になり、これがトポロジカルソート・ハッシュ計算のC++側コストとして無視できない可能性が高い(未計測、Issue 02完了後はこの経路自体が使われなくなるため優先度は下がるが、nv<=80のvmapフォールバック経路には残り続ける)。
- `Backend::wait_idle()`が呼び出しごとに同期的にGPUの完了を待つ設計は、CPU-GPUのパイプライニング(次のステップの準備をGPU実行中に並行して行う)を妨げている可能性がある。MLXは非同期実行・遅延評価を前提にした設計だが、mkxの`wait_idle()`同期モデルはより素朴(かつ低性能)である可能性が高い。

## 対応方針(調査ベース、結論はプロファイリング後に確定)

1. `mx::eval`のshim実装を、複数配列をまとめて1回の`mkx::eval<Backend>`呼び出しに統合できないか検討する(mkxの`eval<Backend, class... Arrays>`テンプレートは可変長引数を受け取れる設計になっているため、shim側の`Raw<float>`プロキシをパラメータパックとして展開して1回で呼べる可能性がある)。
2. 簡単なマイクロベンチマーク(例: 100個の小さな配列を作成→加算→eval、を1回のeval呼び出しと100回のeval呼び出しで比較)を書き、`wait_idle()`同期回数がスループットに与える影響を定量化する。
3. トポロジカルソート・ハッシュ計算にかかる時間を、GPU実行時間と切り分けて計測する(例えば`Backend::compile`のみを事前に全ノード分呼び出しておき、2回目以降の実行時間を計測することでパイプラインキャッシュのウォームアップ後の純粋なディスパッチコストを見る)。

## 定量目標

- Go2/H1(32 envs)の1ステップあたりのCPU側固定オーバーヘッド(GPU実行時間を除いた、グラフ構築+ディスパッチ発行+同期待ちのCPU時間)を計測し、**env数に依存しない部分を500マイクロ秒未満に抑える**ことを目標とする(数値の妥当性はIssue 02完了後の実測値で再検討する)。
- `mx::eval(a,b,c,...)`の複数配列呼び出しを1回の`wait_idle()`に統合し、個別呼び出し(現状)との性能差を計測・記録する。

## 関連

- Issue 02(高速カーネル経路への移行)が完了すれば、この項目の影響範囲はnv<=80のvmapフォールバック経路のみに限定される。
