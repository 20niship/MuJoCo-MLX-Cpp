# Vulkan(MKX)でmesh衝突込みbatched envを数万steps/secにするための課題一覧

目標: mesh衝突(GJK/EPA)を含むbatched環境(Go2/H1等)で、Vulkan(mkx)バックエンドがMLX(Metal)と同等の数万steps/sec規模のスループットを出せるようにする。

GitHub Issuesがこのrepoで無効化されているため、1見出し=1ファイルとしてここに詳細を記載した(後日Issue化する場合はこのファイル1つを1 issueとして貼り付けること)。優先度順(早く解決すると後の検証がやりやすい順)に並んでいる。

1. [`01-nv-gate-excludes-mesh-robots-from-fast-kernel-path.md`](01-nv-gate-excludes-mesh-robots-from-fast-kernel-path.md) — **最優先・着手コスト低。** MLX側の既存高速カーネル経路(mesh衝突対応済み)が`nv > 80`条件でGo2(nv=18)/H1(nv=25)を締め出している。ゲート緩和だけで大幅改善する可能性が高い。
2. [`02-wire-up-mkx-fast-kernels.md`](02-wire-up-mkx-fast-kernels.md) — vendor済みだが未接続のMKX用GLSLカーネル(forward/collision/solver)を`src/batched.cpp`に配線する。Issue 01完了後に着手。
3. [`03-mkx-vmap-broadcast-crash.md`](03-mkx-vmap-broadcast-crash.md) — MKXのbatched vmap経路がbroadcast不整合でクラッシュし、現状のベースライン計測自体ができない。
4. [`04-mkx-dispatch-overhead-profiling.md`](04-mkx-dispatch-overhead-profiling.md) — mkxのCPU側グラフ処理・ディスパッチ同期オーバーヘッドの計測・削減。
5. [`05-mkx-compile-is-noop.md`](05-mkx-compile-is-noop.md) — mkxの`compile()`が演算融合を一切行わない(恒等パススルー)。Issue 02完了後は影響範囲が縮小する見込み。
6. [`06-mesh-collision-kernel-accuracy-tests.md`](06-mesh-collision-kernel-accuracy-tests.md) — 実robotスケール(数十頂点)のmesh衝突カーネルの精度検証テストが存在しない。Issue 01の受け入れ条件でもある。

各ファイル内でFACT(このセッション中に確認した事実)とHYPOTHESIS(未検証の推測)を明確に分離して記載している。定量目標も各ファイルに記載済み。
