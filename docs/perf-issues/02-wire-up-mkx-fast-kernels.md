# [P0] MKX(Vulkan)向けにvendor済みのforward/collision/solver GLSLカーネルを接続する

## 背景

MLX→mkx(Vulkan)バックエンド移植の際、mlx_vulkan(旧mlx-cross-platform)側で既にGLSLへ移植済みだった6種のMetalカーネル(kinematics/euler/euler_devmem/forward/collision/solver)を`src/compat/mkx_kernels/*.hpp`としてこのリポジトリへvendorした。このうちkinematics/euler/euler_devmemの3つは`src/batched.cpp`のMKX分岐(`#if defined(MJMLX_BACKEND_MKX)`)から実際に呼び出すよう配線したが、**forward/collision/solverの3つはvendorしただけで`src/batched.cpp`から一切呼ばれていない。**

## FACT

1. `src/compat/mkx_kernels/forward.hpp`・`collision.hpp`・`solver.hpp`は存在し、それぞれ`mkx::mujoco::make_forward_kernel(...)` 相当のファクトリ関数(namespaceは`mjmlx::mkx_kernels`にリネーム済み)を提供する。ビルドは通る(includeされているだけで未使用関数として存在)。
2. `src/batched.cpp`内、`if (m.nv > 80) { ... }` のMLX分岐の中でのみ`make_forward_source`/`make_collision_source`/`make_solver_source`(MSL版)を呼びており、対応する`#if defined(MJMLX_BACKEND_MKX)`分岐が存在しない。つまり**MKXバックエンドでは、nvの大小に関わらずforward/collision/solverの独自カーネル経路が一切使われず、常に`src/constraint_vmap.cpp`のvmap経路(かつMKX用ホストループ`mkx_vmap_batch0`経由)にフォールバックする。**
3. `make_forward_kernel`のGLSL側シグネチャは、MLX側の`make_forward_source(m)`(Modelを丸ごと受け取り関数内部で全フィールドを抽出)と異なり、`(nb, nv, nq, nu, njnt, dt, gravity, body_parentid, body_rootid, body_mass, body_inertia, dof_bodyid, dof_parentid, dof_damping, dof_armature, dof_stiffness, dof_qposadr, qpos_spring, act_gain0, act_bias0, dof_jtype, dof_rotaxis, dof_jid, body_dofs, jnt_dofadr0)` という20個超の個別引数を要求する形になっている(mlx_vulkan側の実装者が独自に構造化し直したため)。**これらの引数(特に`dof_jtype`/`dof_rotaxis`/`dof_jid`/`body_dofs`)がMuJoCoの標準モデルフィールドのどれに対応するかは、このセッションでは特定できておらず未検証。**
4. `make_collision_kernel(ng, npairs, max_contacts_per_env=128)` および `make_solver_kernel(nb, nv, timestep, use_pyramidal, refsafe, impratio, solver_iters, cg_iters)` はMLX側の`make_collision_source(m)`/`make_solver_source(m, si, cgi)`より引数が単純化されている。

## HYPOTHESIS

- forward.hppの複雑な引数マッピングは、mlx_vulkan側の実装者が「わかりやすい形」に独自再構成した結果であり、MLX側の`make_forward_source`内部で行っている抽出処理(`bpar_ptr`, `bmass_ptr`, `jda_ptr`, `dof_stiffness`計算など、`src/batched.cpp`内に実装済み)をそのまま踏襲すれば埋められる可能性が高いが、**`dof_jtype`/`dof_rotaxis`/`dof_jid`/`body_dofs`が具体的に何を指すかはGLSLソース本体(`forward.hpp`内のシェーダ文字列)を読み込んで初めて確定できる**(このセッションでは実施していない)。
- Issue 01の対応(ゲート緩和)を先にMLX側で行い、実際にGo2/H1で使われるgrid/threadgroup・入出力配列の順序・shapeが固まってから、このIssueに着手する方が手戻りが少ない。

## 対応方針

1. Issue 01の対応(MLX側のnvゲート緩和)を先に完了し、MLX側でGo2/H1に対する`make_forward_source`/`make_collision_source`/`make_solver_source`呼び出しが実際に発火する状態を作る。
2. `src/compat/mkx_kernels/forward.hpp`のGLSL本体を読み、20個超の引数それぞれがMuJoCoのどのモデルフィールドに対応するかを特定する(コメントで記録する)。
3. `src/batched.cpp`のforward/collision/solverカーネル構築・ディスパッチ箇所に、既存のkinematics/eulerと同じパターンで`#if defined(MJMLX_BACKEND_MKX) ... #else ... #endif`分岐を追加する。
4. `mx::fast::kernel_inputs`/`kernel_shapes`/`kernel_outputs`(`src/compat/mx_compat_mkx.h`に既存)を使い、mkxの4引数`operator()`形式でディスパッチする。
5. 入出力の対応順序(位置バインドのため名前一致ではなく順序一致が必須)をkinematics/eulerで行ったのと同様に1つずつ突き合わせる。

## 定量目標

- `just bench-mkx`実行時、`batched/go2`・`batched/h1`がクラッシュせず完走し、steps/secが計測できること(具体的な数値目標はIssue 01でMLX側の実測値が確定してから設定する。当面の目安として **MLX実測値の30%以上**(例: MLXが15,000 steps/secなら4,500 steps/sec以上)を暫定目標とする)。
- Vulkanディスパッチ回数が、env数Bに対してO(1)であること(kinematics/eulerと同様、grid=(B,1,1)の単一ディスパッチで全env分を処理する)。現状のホストループ版vmap(`mkx_vmap_batch0`)はO(B×グラフノード数)のディスパッチを生成しており、これがボトルネックである可能性が高い(Issue 04参照)。

## 関連

- Issue 01(MLX側ゲート緩和、先行して完了させること)
- Issue 04(残存する汎用vmap経路のCPU/GPUオーバーヘッド調査)
- Issue 06(mesh衝突カーネルの精度検証テスト — GLSL版にも同様の検証が必要)
