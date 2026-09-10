# [P1] mkxの`compile()`は恒等パススルーで、演算融合(fusion)を一切行わない

## 背景

MLXの`mx::compile(fn)`は複数の演算ノードを1つ以上のより大きなGPUカーネルに融合し、ディスパッチ回数を削減する最適化を行う。`src/batched.cpp`のコメント(`DECISION: Using mx::compile around the full pipeline ... enables MLX to fuse the computation graph. This is critical for performance: without compile, each MLX op dispatches a separate Metal kernel.`)にある通り、このリポジトリの設計はMLXの`compile`によるフュージョンに強く依存している。

## FACT

1. mlx_vulkan(mkx)の`mkx::compile(Fn fn)`実装(`mkx/ops/transforms.hpp`)は、ソースコード上のコメント(日本語、mlx_vulkan側の実装者によるもの)によれば「eval()がshaderソースのハッシュでpipelineを既にキャッシュしているためcompileは今のところpassthrough、複数ノードを1shaderに融合するfuse最適化は必要になったら追加する」という設計であり、**実装は単に`return fn;`(恒等関数)を返すだけで、何の最適化も行わない。**
2. `src/compat/mx_compat_mkx.h`のshim実装(`mx::compile`)も、mkx本体の設計に合わせて同様に恒等パススルーとして実装した(`template <class Fn> auto compile(Fn fn) { return fn; }`)。これはMKXバックエンド固有の制約であり、MLXバックエンドでは本物の`mx::compile`(実際に融合を行う)がそのまま使われている。
3. Issue 02(高速カーネル経路)が完了すれば、forward/collision/solverフェーズは個別のGLSLカスタムカーネル(1フェーズ=1ディスパッチ)になり、`compile`によるフュージョンの恩恵を受ける汎用op列そのものが消える。したがって**この項目の影響範囲は、高速カーネル経路でカバーされない残りの処理(nv<=80のvmapフォールバック、または将来追加される新しい物理演算)に限定される。**

## HYPOTHESIS

- Issue 01/02が完了した後の世界では、この項目の優先度はP1からP2程度まで下がる可能性が高い(影響範囲が限定的になるため)。ただし、任意の新しいvmap経路(例えば将来サポートするRK4/ImplicitFast積分器など、`CONFORMANCE.md`に「DEFERRED」と記載されている項目)がMKXで実装される場合、compileの欠如が同様のボトルネックを再発させる可能性がある。
- mkx側にfusion実装を追加する場合、GLSLでは複数の演算を1つのシェーダにインライン展開する形になると推測されるが、mkxの現在のグラフ表現(`OpNode`ごとに独立したShaderGroup分類)がそのような融合をどこまで素直にサポートできる設計になっているかは未調査。

## 対応方針

1. Issue 01/02の完了後、実際にどの程度の汎用op列がMKXの物理演算パイプラインに残るかを棚卸しする。
2. 残存範囲が小さければ、compile融合の実装よりも「残った処理を専用カーネルに移す」方針(Issue 02と同じアプローチの横展開)の方が費用対効果が高い可能性がある。
3. 残存範囲が無視できない場合のみ、mlx_vulkan本体(mkx)側にfusion実装を追加する(このリポジトリ側からの修正ではなく、upstream albo mlx_vulkan repoへの貢献になる可能性が高い)。

## 定量目標

- Issue 01/02完了後に残る汎用vmap経路について、代表的な小規模モデル(nv<=80、T-shapeタスク相当)のforward dynamics 1ステップ分のグラフを対象に、現状のノード数(ディスパッチ数)を計測し、fusion実装によって**ディスパッチ数を5分の1以下に削減する**ことを目標とする(具体的なノード数は未計測、まず計測することが前提)。

## 関連

- Issue 02(高速カーネル経路)の進捗次第でこの項目の必要性・優先度が変わるため、Issue 02完了後に再評価すること。
