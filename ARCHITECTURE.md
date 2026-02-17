# Architecture

Comprehensive design document for MuJoCo-MLX-Cpp. This documents key architectural decisions, trade-offs, and lessons learned during development. **Read this before making changes to avoid undoing intentional design choices.**

## Table of Contents

- [High-Level Pipeline](#high-level-pipeline)
- [Forward Pipeline Detail](#forward-pipeline-detail)
- [Why Metal + Vmap Hybrid](#why-metal--vmap-hybrid)
- [Computation Graph Rules](#computation-graph-rules)
- [Key Data Structures](#key-data-structures)
- [Cholesky Factorization](#cholesky-factorization)
- [Per-Body Loops vs Scatter Matrices](#per-body-loops-vs-scatter-matrices)
- [Mass Matrix Construction](#mass-matrix-construction)
- [Constraint Solver](#constraint-solver)
- [Constraint Construction](#constraint-construction)
- [DOF Friction Loss](#dof-friction-loss)
- [Tendon System](#tendon-system)
- [Collision System](#collision-system)
- [Metal Kernels](#metal-kernels)
- [Performance Characteristics](#performance-characteristics)
- [Comparison with Python mujoco-mlx and MJX](#comparison-with-python-mujoco-mlx-and-mjx)
- [Known Issues and Future Work](#known-issues-and-future-work)

---

## High-Level Pipeline

The batched simulation runs a 3-phase hybrid pipeline for each timestep:

```
 Phase 1: Metal Kinematics Kernel
 ---------------------------------
 Single GPU dispatch for ALL N environments.
 qpos -> xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat
 Replaces ~130 separate MLX dispatches with one fused Metal kernel.

         |
         v

 Phase 2: compile(vmap(forward_dynamics))
 -----------------------------------------
 Pure MLX array operations, traced by vmap across N environments.
 COM position, CRB mass matrix, Cholesky factorization,
 tendon computation, collision detection, constraint generation,
 solver, transmission, COM velocity, passive forces, RNE,
 actuation (incl. activation dynamics), acceleration,
 rne_post_constraint (cfrc_ext).
 mx::compile fuses the computation graph into fewer GPU dispatches.

         |
         v

 Phase 3: Metal Integration Kernel
 -----------------------------------
 Single GPU dispatch for ALL N environments.
 Built-in Cholesky + solve + velocity update + position integration.
 Currently Euler only in Metal; RK4 and ImplicitFast are supported
 in the scalar pipeline but not yet in the batched Metal kernel.
```

The pipeline is orchestrated in `batched.cpp:make_batched_step()`.

---

## Forward Pipeline Detail

The scalar `forward()` pipeline calls sub-stages in this order:

```
fwd_position:
  kinematics -> com_pos -> crb -> factor_m -> tendon -> collision -> make_constraint -> transmission

fwd_velocity:
  actuator_velocity -> com_vel -> passive (incl. tendon spring/damping) -> rne

fwd_actuation:
  actuator force computation

fwd_acceleration:
  qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_actuator + qfrc_applied
  qacc_smooth = M^{-1} * qfrc_smooth

constraint solve:
  if nefc > 0: solve(m, d)     [CG or Newton]
  else: qacc = qacc_smooth

post-constraint:
  rne_post_constraint -> cfrc_ext
```

The vmap pipeline mirrors this with `vmap_*` versions of each function. All vmap functions use pure MLX array ops (no eval, no data<>).

**Constraint ordering** within `efc_J`:
1. Equality constraints (`d.ne` rows)
2. DOF friction loss + Tendon friction loss (`d.nf` rows)
3. Joint limits + Tendon limits (`d.nl` rows)
4. Contact constraints (remaining rows)

---

## Why Metal + Vmap Hybrid

Metal GPU kernels (written in MSL) operate on raw flat buffers with explicit per-environment indexing via `thread_position_in_grid`. They are **not compatible with `mx::vmap`**, which traces a function over a batch dimension by intercepting MLX array operations.

Therefore:
- **Metal kernels** (Phases 1 and 3) run **outside** the vmap boundary. They process all N environments in a single dispatch by using thread IDs to index into the flat batch arrays.
- **Phase 2** (forward dynamics) runs **inside** `mx::vmap`. It must use only pure MLX array ops -- no `eval()`, no `data<>()`, no CPU sync.

This is the same architecture used by:
- **Python mujoco-mlx**: Custom Metal kernels for kinematics and Euler; pure MLX ops for the vmap'd forward dynamics.
- **Google MJX**: Uses `jax.vmap + jax.jit` over pure JAX ops (no custom GPU kernels).

Our advantage over MJX: custom Metal kernels for the two hottest phases (kinematics and integration) reduce dispatch overhead.

---

## Computation Graph Rules

**All functions called inside `mx::vmap` must follow these rules:**

1. **No `mx::eval()`** -- Forces synchronous GPU execution, breaking the lazy graph.
2. **No `.data<T>()`** -- Requires evaluated data, which forces eval.
3. **No CPU-side conditionals on array values** -- Array values aren't available until eval.
4. **Use `mx::where()` for conditional logic** -- Branch-free selection on GPU.
5. **Fixed output shapes** -- vmap requires uniform shapes across the batch dimension.

Functions that follow these rules: `vmap_com_pos`, `vmap_crb`, `vmap_factor_m`, `vmap_com_vel`, `vmap_rne`, `vmap_collision`, `vmap_make_constraint`, `vmap_solve`, `vmap_transmission`, `vmap_tendon`, `vmap_passive`, `vmap_fwd_actuation`, `vmap_fwd_acceleration`.

**Model constants** (topology arrays, scatter matrices, masks) are accessed inside vmap functions but are never mutated. They are precomputed in `ModelCache` during `init_cache()` using `eval()` + `data<>()`, which is safe because `init_cache()` runs at model load time, not inside vmap.

---

## Key Data Structures

### Model

Loaded from MuJoCo XML via `io.cpp`. Contains all model parameters as `mx::array`:
- Joint/body/geom topology (parentid, bodyid, types)
- Physical properties (mass, inertia, damping, stiffness, frictionloss)
- Actuator configuration (gear, bias, limits)
- Tendon properties (adr, num, stiffness, damping, wrap objects)
- Equality constraint properties (type, obj1id, obj2id, data, solref, solimp)
- DOF solver parameters (dof_solref, dof_solimp for friction loss)
- Solver options (solver type, iterations, timestep)

### ModelCache

Precomputed at load time by `init_cache()`. Contains:
- **Tree topology**: `tree_levels` (bodies grouped by depth), `body_dofs` (DOF indices per body), `body_parentid_vec` (C++ vector for fast indexing)
- **Scatter matrices**: For level-parallel backward accumulation in `vmap_com_pos` and `vmap_crb`
- **Mass matrix mask**: Lower-triangular DOF ancestor mask (`make_m_mask`)
- **Collision pairs**: Pre-filtered geometry pairs for broadphase (includes hfield, ellipsoid, mesh data)
- **Joint plans**: Integration plans (simple/free/ball joints), actuator moment matrices
- **CDoF plan**: Precomputed indices for vectorized cdof computation
- **Body DOF masks**: For Jacobian computation in constraints
- **max_nefc**: Pre-computed maximum constraint rows (equality + friction + limits + contacts)

### Data

Per-environment state. In batched mode, vmap adds a leading batch dimension:
- **State**: qpos, qvel, qacc, ctrl, act
- **Derived**: xpos, xquat, xmat (from kinematics), cinert, cvel, cdof, cdof_dot (from smooth dynamics), qfrc_bias, qfrc_smooth, qfrc_constraint (from RNE/solver)
- **Tendon**: ten_length, ten_velocity, ten_J (from tendon computation)
- **Constraint**: efc_J, efc_D, efc_aref, efc_force, efc_frictionloss, contact info
- **Counts**: nefc, ne (equality), nf (friction), nl (limits), ncon (contacts)

---

## Cholesky Factorization

**File:** `smooth_vmap.cpp:cholesky_gpu()`

The GPU-native Cholesky decomposes A = L*L^T using pure MLX array ops (no `eval`). It is used for:
1. Mass matrix factorization (`vmap_factor_m`)
2. Newton solver's H = M + J^T*D*J factorization

### Why column-vectorized with one-hot scatter

MLX arrays are **immutable** -- there is no in-place index assignment (`L[i,j] = val`) in the computation graph. To build L column-by-column:

1. Compute the diagonal element: `L[j,j] = sqrt(A[j,j] - sum(L[j,0:j]^2))`
2. Compute the off-diagonal column: `L[j+1:n,j] = (A[j+1:n,j] - L[j+1:n,0:j] @ L[j,0:j]^T) / L[j,j]`
3. Build a full `(n,1)` column vector
4. Create a `(1,n)` one-hot row selector (1 at position j)
5. Scatter via: `L = L + full_col @ one_hot` (outer product adds column j)

This creates ~35 graph nodes per column, ~945 total for humanoid (n=27). The `mx::compile` step fuses these into efficient GPU operations.

### Numerical guard

A `1e-6` floor on the diagonal prevents negative sqrt from numerical noise. This matches Python's `gpu_cholesky` implementation in `gpu_linalg.py`.

### Field naming

`d.qM_inv` stores the Cholesky factor L, **not** the actual inverse M^{-1}. The name is legacy. It is consumed by `vmap_solve_m()` which calls `cholesky_solve_gpu(L, rhs, n)`.

---

## Per-Body Loops vs Scatter Matrices

**This is a critical architectural decision. Do not change without benchmarking.**

### When scatter matrices work (backward accumulation)

`vmap_com_pos` and the backward pass of `vmap_rne` use scatter matrices for level-parallel accumulation:

```cpp
// For each tree level (bottom to top):
auto child_vals = mx::take(arr, child_ids, 0);
auto scatter_mat = ...; // (nb, nc) with 1 at (parent_id, child_idx)
arr = mx::add(arr, mx::matmul(scatter_mat, child_vals));
```

This works because **scatter-add is commutative** -- each child's contribution to its parent is independent of other children. The order doesn't matter.

### When per-body loops are correct (forward propagation)

`vmap_com_vel` and the forward pass of `vmap_rne` use per-body sequential loops:

```cpp
for (int bid : c.tree_levels[lvl]) {
    auto cv = cvel[pid];  // parent's velocity
    for (int di : c.body_dofs[bid]) {
        cdof_dot[di] = motion_cross(cv, cdof[di]);
        cv = cv + cdof[di] * qvel[di];  // accumulate DOF contributions
    }
    cvel[bid] = cv;
}
d.cvel = mx::stack(cvel);
```

**An attempt was made to vectorize this using scatter matrices (batching all bodies per level into a single matmul). This REGRESSED performance from 198K to 47K SPS because:**

1. **Data dependency chains**: Scatter-add through the full `(nb, 6)` tensor creates a long chain where each level depends on ALL previous levels through the entire array. The compiler cannot parallelize across levels.

2. **Independent graph branches**: The per-body approach stores results in a `std::vector<mx::array>` where each body's computation is an independent graph branch. The final `mx::stack()` is the only convergence point, giving `mx::compile` maximum freedom to fuse and schedule ops in parallel.

3. **Small ops are efficient**: Per-body operations on `(6,)` vectors are tiny and GPU-friendly after fusion. The matmul overhead of scatter matrices is not offset by parallelism gains.

**Rule of thumb**: Use scatter matrices for backward/commutative accumulation. Use per-body loops for forward/sequential propagation.

---

## Mass Matrix Construction

**Files:** `io.cpp` (mask), `smooth_vmap.cpp` (vmap_crb)

### Lower-triangular mask (CRITICAL)

`make_m_mask` in `init_cache()` is **strictly lower-triangular**. For each DOF i, it walks from i toward the root, setting `mask[i][j]=1` for each ancestor j (always j <= i):

```cpp
for (int i = 0; i < nv; i++) {
    int j = i;
    while (j > -1) {
        mask_data[i * nv + j] = 1.0f;
        j = dof_par_ptr[j];
    }
}
```

The full symmetric mass matrix is recovered in `vmap_crb` via: `qm = qm + tril(qm, -1)^T`.

**An earlier bug** set `mask[j][i]=1` too (making it symmetric), which caused the mass matrix to have a symmetry error of 30.34, breaking Cholesky factorization and causing simulation divergence. This was one of the hardest bugs to diagnose because the Cholesky code was correct -- the input was wrong.

The Python reference `_get_mass_matrix_mask` in `support.py` is also strictly lower-triangular.

---

## Constraint Solver

**Files:** `solver.cpp` (scalar), `solver_vmap.cpp` (vmap/GPU)

### Newton vs CG

- **Newton** (default for humanoid, `iterations="1"`): Forms the Hessian `H = M + J^T*D*J` and Cholesky-solves for the exact search direction. Converges in 1-2 iterations.
- **CG**: Uses M^{-1} preconditioned gradient descent with Polak-Ribiere updates. Needs 4-10 iterations. Used when direct solve is too expensive.

### Solver iteration count

The iteration count comes from the model XML (e.g., `humanoid.xml` specifies `iterations="1"` for Newton). **Do not artificially inflate the count.** A previous workaround forced a minimum of 3 iterations to compensate for mass matrix bugs. That bug (see [Mass Matrix Construction](#mass-matrix-construction)) has been fixed.

### Warmstart

The solver compares warmstart cost (from previous timestep's qacc) against smooth cost (unconstrained qacc). It picks whichever has lower constraint violation cost. This prevents divergence when the previous solution is stale (e.g., after contact loss).

### 5-alpha line search

Instead of binary/backtracking search (which needs conditional branches, incompatible with vmap), the solver evaluates 5 candidate step sizes in parallel:
1. Newton/CG optimal alpha
2. 50% of optimal
3. 10% of optimal
4. 0.01 (fixed safety net)
5. 0.001 (fixed safety net)

All 5 costs are computed in one batched operation and the minimum is selected via `mx::argmin`. Branch-free and fully vectorizable.

### Friction force clamping (linear zone)

DOF friction loss constraints use a special "linear zone" cost function. When `|Jaref| >= R * frictionloss`, the constraint enters the linear zone:

- The constraint is removed from the quadratic active set (`active_mask = 0`)
- The force is clamped to `+/-frictionloss`
- The cost becomes linear (not quadratic)

**This is critical for the `alpha_n` calculation.** The Newton step denominator must only include quadratic terms, while the numerator must include both quadratic and linear derivatives:

```
alpha_n = -(linear_gauss + linear_con + linear_floss) / max(2 * (quad_gauss + quad_con), eps)
```

If `linear_floss` (the derivative of the linear friction cost) is omitted, the solver diverges because `alpha_n` collapses when friction rows dominate. Both `solver.cpp` and `solver_vmap.cpp` implement this via `apply_friction_clamp` / `vmap_friction_clamp` helper functions.

### prev_grad correctness (CG)

In the Polak-Ribiere beta computation, `prev_grad` must be captured **before** updating `grad`. An earlier bug used the already-updated gradient in the denominator, causing CG to diverge.

---

## Constraint Construction

**Files:** `constraint.cpp` (scalar), `constraint_vmap.cpp` (vmap/GPU)

### Constraint types supported

| Type | Source | Rows per instance | Notes |
|------|--------|-------------------|-------|
| Equality (CONNECT) | `eq_type=1` | 3 | Positional error in 3D |
| Equality (WELD) | `eq_type=0` | 6 | Position (3) + orientation (3) |
| Equality (JOINT) | `eq_type=2` | 1 | Joint position error |
| DOF friction loss | `dof_frictionloss > 0` | 1 per DOF | Identity Jacobian, K=0 |
| Tendon friction loss | `tendon_frictionloss > 0` | 1 per tendon | `ten_J` Jacobian, K=0 |
| Joint limit | `jnt_limited` | 1 per active limit | Upper or lower bound |
| Tendon limit | `tendon_limited` | 1 per active limit | `ten_J` Jacobian, signed |
| Contact (condim=1) | collision | 1 | Normal only |
| Contact (condim=3) | collision | 4 | Pyramidal friction (2 tangent dirs) |
| Contact (condim=4) | collision | 6 | + torsion friction |
| Contact (condim=6) | collision | 10 | + rolling friction |

### Fixed-size outputs

All pre-computed collision pairs are always evaluated, even when not in contact. Inactive constraints get `D=0`, which makes them no-ops in the solver. This ensures uniform output shapes across environments, which is required for vmap.

### KBI (Stiffness, Damping, Impedance) computation

Constraint parameters are computed from `solref` (reference) and `solimp` (impedance):
- **K** (stiffness): `1 / (max(solref[0], eps) * h)^2` (set to 0 for friction loss)
- **B** (damping): `2 / (max(solref[0], eps) * h)` (scaled by solref[1])
- **Imp** (impedance): interpolated from solimp range using positional error

### Degenerate normal fallback

`vmap_make_frame` computes tangent frames from contact normals using cross products. When the normal is near-parallel to `[0,0,1]` (cross product magnitude < 1e-6), it falls back to `[0,1,0]` as the reference axis. This uses `mx::where` for branch-free execution.

---

## DOF Friction Loss

**Files:** `constraint.cpp`, `constraint_vmap.cpp`, `solver.cpp`, `solver_vmap.cpp`

DOF friction loss is a joint-level friction mechanism where each DOF with `frictionloss > 0` generates a constraint row. The constraint applies a force opposing velocity, bounded by `+/-frictionloss`.

### Constraint generation

- **Jacobian**: Identity row `e_i` (1 at DOF index, 0 elsewhere)
- **Position**: Always 0 (no positional error for friction)
- **Stiffness (K)**: Explicitly set to 0 (matching MuJoCo C `engine_core_constraint.c`)
- **Damping (B)**: Computed from `dof_solref` / `dof_solimp`
- **aref**: `-B * qvel[i]`

### Solver integration

The friction clamping logic is the most intricate part:

1. **Linear zone detection**: If `|Jaref| >= R * frictionloss`, the row enters the linear zone
2. **Active set update**: Linear zone rows are removed from the quadratic active set
3. **Force clamping**: Force is saturated at `+/-frictionloss` (sign determined by Jaref direction)
4. **Cost**: Quadratic cost for quadratic rows + linear cost for friction rows in the linear zone
5. **Line search**: `alpha_n` numerator includes `linear_floss` derivative term

This is re-evaluated at every solver iteration since `Jaref` changes as the solution evolves.

---

## Tendon System

**Files:** `smooth.cpp` (`tendon()`), `smooth_vmap.cpp` (`vmap_tendon()`), `passive.cpp` (`tendon_passive()`)

### Fixed (joint-based) tendons

Fixed tendons compute a linear combination of joint values:

```
ten_length[i] = sum(wrap_prm[j] * qpos[jnt_qposadr[wrap_objid[j]]])
```

where the sum is over all wrap objects of type `mjWRAP_JOINT` in tendon `i`.

The tendon Jacobian is sparse and constant (does not depend on state):

```
ten_J[tendon_id, jnt_dofadr[wrap_objid[j]]] = wrap_prm[j]
```

Tendon velocity is computed as `ten_velocity = ten_J @ qvel`.

### Model fields

| Field | Shape | Description |
|-------|-------|-------------|
| `tendon_adr` | (ntendon,) | Start index in wrap arrays |
| `tendon_num` | (ntendon,) | Number of wrap objects per tendon |
| `tendon_stiffness` | (ntendon,) | Spring stiffness |
| `tendon_damping` | (ntendon,) | Damping coefficient |
| `tendon_lengthspring` | (ntendon, 2) | Spring rest length range |
| `wrap_type` | (nwrap,) | Wrap object type (1 = JOINT) |
| `wrap_objid` | (nwrap,) | Joint index |
| `wrap_prm` | (nwrap,) | Coefficient / moment arm |

### Pipeline position

Tendon computation is placed in `fwd_position` after `factor_m` and before `collision`, matching MuJoCo C's ordering. This ensures tendon data (`ten_length`, `ten_J`) is available for:
- Tendon passive forces (in `fwd_velocity -> passive()`)
- Tendon limit constraints (in `make_constraint`)
- Tendon friction loss constraints (in `make_constraint`)
- Tendon transmission actuators (in `transmission`)

### Passive forces

Tendon spring and damping forces are computed in `passive.cpp`:

```
force[t] = -stiffness * (ten_length - rest_length) - damping * ten_velocity
qfrc_passive += ten_J^T * force
```

The rest length uses `tendon_lengthspring` range clamping: `rest = clamp(ten_length, lo, hi)`.

### Vmap compatibility

The vmap path pre-computes the constant Jacobian from model data (safe to use `eval`/`data<>` since it's model-time, not trace-time). The `ten_length` computation uses a gather-matmul pattern: gather `qpos` at relevant indices, multiply by coefficients, then scatter-sum into tendon lengths via a pre-built one-hot matrix. This avoids CPU-side loops during vmap tracing.

### Spatial tendons (DEFERRED)

Spatial tendons wrap around geometry surfaces (spheres, cylinders) and require computing shortest paths (~400 lines of dense geometric code in MJX). **DEFERRED**: Most RL models use fixed (joint-based) tendons only. Spatial wrapping is needed primarily for anatomical hand models. The model fields (`wrap_type` values 3-5 for SITE/SPHERE/CYLINDER) are loaded but non-JOINT wrap types emit a warning and are skipped.

---

## Collision System

**Files:** `collision.cpp` (scalar), `constraint_vmap.cpp` (vmap)

### Supported collision pairs

| Pair | Algorithm | Multi-contact | Vmap |
|------|-----------|---------------|------|
| plane-sphere | Analytic | 1 | Yes |
| plane-capsule | Analytic | 1 | Yes |
| plane-box | Face projection | Up to 4 | Yes (1) |
| plane-cylinder | Rim + center | Up to 6 | Yes (1) |
| plane-mesh | All vertices | Variable | Yes (1) |
| plane-hfield | Grid triangles | Variable | Yes (1) |
| plane-ellipsoid | Analytic | 1 | Yes |
| sphere-sphere | Analytic | 1 | Yes |
| sphere-capsule | Segment closest | 1 | Yes |
| sphere-box | OBB closest | 1 | Yes |
| sphere-cylinder | Region-based | 1 | Yes |
| sphere-ellipsoid | Analytic | 1 | Yes |
| capsule-capsule | Segment-segment | 1 | Yes |
| capsule-box | Iterative | 1 | Yes |
| capsule-cylinder | Iterative | 1 | Yes |
| capsule-ellipsoid | Analytic | 1 | Yes |
| box-box | SAT (15 axes) | 1 | Yes |
| mesh-* | GJK/EPA | 1 | GJK + depth est. |
| hfield-* | Grid cells | Variable | 1 |

### GJK/EPA (convex collision)

64-iteration GJK with evolving simplex (point -> line -> triangle -> tetrahedron), followed by 64-iteration EPA for penetration depth. The vmap path uses 32-iteration GJK with support-based depth estimation (~18 sample directions: 6 axis-aligned + 8 diagonal + simplex face normals) instead of EPA, for GPU compatibility. Both scalar and vmap paths use proper mesh support functions (vertex argmax), not sphere approximation.

### Heightfield (hfield)

Grid-cell based collision: identifies the grid cell containing the query point, tests against both triangles of the cell, returns the deepest penetrating contact. Supports all geom types against hfield.

### Ellipsoid

Analytic collision using the ellipsoid-to-sphere transform: scale the world so the ellipsoid becomes a unit sphere, compute the contact in that space, then transform back. Supports plane-ellipsoid, sphere-ellipsoid, and capsule-ellipsoid.

---

## Metal Kernels

### Kinematics Kernel (Phase 1)

Source-generated MSL in `batched.cpp:make_kinematics_source()`. Embeds model topology directly in the shader:
- Joint types, parent IDs, qpos/qvel addresses are compile-time constants
- Forward kinematics loop unrolled over joint chain
- Stack memory: ~7 floats per body (pos, quat). Skip if `nbody * 7 * 4 > 24KB`

### Integration Kernel (Phase 3)

Source-generated MSL in `batched.cpp:make_euler_source()`. Currently Euler-only in Metal; RK4 and ImplicitFast are supported in the scalar path. Embeds Cholesky factorization and solve inline:
- Stack memory: `2*nv*nv + 4*nv + nq` floats. Limited to `nv <= 80`
- Performs: Cholesky(M) -> solve for qacc -> velocity update -> position integration
- For humanoid (nv=27): ~5KB stack per thread, well within limits

---

## Performance Characteristics

### Benchmark (humanoid, Apple M-series GPU, Release build)

| Envs | Steps/sec | Notes |
|------|-----------|-------|
| 256 | 15,027 | GPU underutilized |
| 512 | 34,222 | |
| 1,024 | 68,753 | |
| 2,048 | 128,366 | |
| 4,096 | 255,362 | Exceeds Python mujoco-mlx |
| 8,192 | **331,207** | Peak throughput |

### Where time goes

The forward dynamics computation graph has ~6,200 nodes (with 1 solver iteration):
- Cholesky factorization: ~945 nodes (27x27 mass matrix)
- COM velocity + RNE: ~1,800 nodes (per-body sequential)
- Collision + constraints: ~1,200 nodes
- Solver (1 Newton iteration): ~800 nodes
- COM position + CRB: ~1,400 nodes (level-parallel scatter)

### Scaling behavior

- Below 1024 envs: GPU underutilized, compilation overhead dominates
- 1024-4096 envs: Linear scaling, GPU filling up
- 4096+ envs: Near peak utilization, memory bandwidth becomes the limit
- 8192 envs: 331K SPS, close to memory bandwidth ceiling

---

## Comparison with Python mujoco-mlx and MJX

| Aspect | MuJoCo-MLX-Cpp | Python mujoco-mlx | MJX (JAX) |
|--------|---------------|-------------------|-----------|
| Language | C++ | Python | Python/JAX |
| GPU backend | Metal (MLX) | Metal (MLX) | XLA (CUDA/TPU) |
| Custom kernels | Yes (kin + Euler) | Yes (kin + Euler) | No |
| Batching | mx::vmap | mx.vmap | jax.vmap |
| Compilation | mx::compile | mx.compile | jax.jit |
| Cholesky (vmap) | MLX array ops | MLX array ops | JAX ops |
| Peak SPS (humanoid) | 331K | 214K (post-hardening) | >1M (CUDA A100) |

Key differences from Python mujoco-mlx:
- Same architecture (Metal + vmap hybrid)
- C++ eliminates Python interpreter overhead
- Mass matrix mask bug fix (see above) was found during C++ port
- Solver iteration count now matches XML spec (Python had the same)

---

## Conformance Phases

### Phase 1: Synth Physics Foundation

**rne_post_constraint (cfrc_ext)**: Called automatically at the end of `forward()`, after the constraint solver. Computes `cfrc_ext` using a `cdof`-based projection from `qfrc_constraint`. This is an approximation compared to MuJoCo C's geometry-based approach using `efc_force`, but produces matching results for typical contact scenarios (validated within 0.001 tolerance).

**Gravity Compensation**: `passive()` includes `gravcomp()` for bodies with `body_gravcomp != 0`. Force `-(mass * gravcomp * gravity)` projected to joint space via translational Jacobian at body COM.

**Exclude Signature**: Collision filtering via `exclude_signature` in both `init_cache()` (vmap pair pre-filtering) and scalar `collision()` narrowphase. Encoding: `(min_body_id << 16) | max_body_id`.

**Model Validation**: `validate_model()` scans `mjModel` at load time and emits warnings for unsupported features.

### Phase 2: Contact Friction

**Pyramidal Friction (condim=3)**: 4 constraint rows per contact (opposing pyramid edges per tangent direction). Friction-scaled impedance: `invw_py = invw * (1 + mu^2)`, `R_py = 2*mu^2 * invw_py * (1-imp)/imp / impratio`.

**Pyramidal Friction (condim=4,6)**: Torsion (condim=4, +2 rows) and rolling (condim=6, +4 more rows) friction using 3rd/4th/5th geom_friction parameters.

### Phase 3: Collision Geometry

See [Collision System](#collision-system) above for the full collision pair matrix.

Key algorithms: SAT for box-box, GJK/EPA for mesh/convex (64-iter scalar, 32-iter + depth estimation vmap), grid-cell for hfield, ellipsoid-to-sphere transform for ellipsoid. All pairs work in both scalar and vmap pipelines with proper support functions (no sphere approximation).

**Not yet implemented**: cylinder-cylinder collision (uncommon in RL models).

### Phase 4: Equality Constraints + DOF Friction

See [Constraint Construction](#constraint-construction) and [DOF Friction Loss](#dof-friction-loss) above.

### Phase 5: Tendon System + Transmission

See [Tendon System](#tendon-system) above.
- **5.1 Fixed tendons**: Complete. `ten_length`, `ten_velocity`, `ten_J`, passive spring/damping forces.
- **5.2 Spatial tendons**: **Deferred**. MJX supports wrapping geometry (sphere/cylinder) but requires ~400 lines of geodesic path code. Most RL models use fixed tendons only.
- **5.3 TENDON + SITE transmission**: Complete. TENDON transmission (`moment = gear * ten_J`), SITE transmission (full 6-DOF Jacobian at site, gear wrench projection). Both scalar and vmap paths.
- **5.4 Tendon friction loss**: Complete. Friction constraints through tendons using `ten_J` as Jacobian, same `compute_kbi` as DOF friction.
- **5.5 Tendon limits**: Complete. Limit constraints on tendon length, using `tendon_limited`/`tendon_range`/`tendon_margin` with `ten_J` Jacobian.

### Phase 6: Actuator Dynamics

- **6.1 FILTER + FILTEREXACT + INTEGRATOR**: Complete (scalar + vmap). Activation state `act` with `act_dot` computation, Euler and exact exponential integration, activation clamping. Force uses `act` for stateful actuators, `ctrl` for stateless. Vmap path uses vectorized scatter-matmul for act_dot.
- **6.3 MUSCLE**: **Deferred**. Biomechanical muscle model with piece-wise linear dynamics. Rarely used in standard RL models.

### Phase 7: Advanced Integrators

- **7.1 RK4**: Complete (scalar path only). Classic 4th-order Runge-Kutta with 4 forward evaluations per step. Weighted-average qacc for velocity update and weighted-average qvel for position update. Refactored `integrate_pos()` and `integrate_act()` as shared helpers used by both Euler and RK4.
- **7.2 ImplicitFast**: Complete (scalar path). Implicit velocity integration via `deriv_smooth_vel()` computing analytical ∂qfrc_smooth/∂qvel (joint damping + tendon damping + affine actuator velocity terms). Modified mass matrix M' = M - dt * qderiv, Cholesky factor and solve for qacc. Both `IMPLICIT` and `IMPLICITFAST` integrator types supported (difference: IMPLICIT would include RNE derivative, which is left as TODO matching MJX). MuJoCo C stores the standard qacc in `d->qacc` while using implicit qacc internally for velocity update; we match this behavior.

---

## Scalar vs Vmap Feature Parity

Not all features are implemented in both the scalar (CPU) and vmap (GPU/batched) paths. The scalar path is the reference implementation; the vmap path lags in these areas:

| Feature | Scalar | Vmap | Notes |
|---------|--------|------|-------|
| Euler integration | Yes | Yes (Metal kernel) | |
| RK4 integration | Yes | No | Metal kernel is Euler-only; deferred |
| ImplicitFast integration | Yes | No | Needs dense Cholesky in vmap; deferred |
| Activation dynamics (act_dot) | Yes | Yes | Vectorized via scatter-matmul |
| Tendon passive forces | Yes | Yes | Vectorized via ten_J^T @ force |
| Gravity compensation | Yes | No | Requires vmap-compatible Jacobian; deferred |
| Tendon limit constraints | Yes | Yes | Precomputed tenJ_row in cache |
| Tendon friction loss | Yes | Yes | Precomputed tenJ_row in cache |
| mesh-* collision | GJK/EPA (64 iter) | GJK + depth est. (32 iter) | Both use proper mesh support |

Remaining gaps: **RK4/ImplicitFast** in batched (needs Metal kernel rewrite or dense Cholesky in vmap), and **gravity compensation** in vmap (needs vmap-compatible body Jacobian computation). All deferred since they affect few RL models.

---

## Known Issues and Future Work

1. **Graph size**: The computation graph (~6,200 nodes) is large. Reducing it would speed up both compilation and execution. Main targets:
   - Cholesky (945 nodes for 27x27): Could benefit from a custom Metal kernel for batched factorization
   - Per-body loops in com_vel/rne (~1,800 nodes): Vectorization was attempted and failed (see above), but a custom Metal kernel approach might work

2. **First-step compilation**: The first batched step triggers `mx::compile`, which takes ~200ms. Subsequent steps are fast.

3. **Small batch sizes**: Below 1024 envs, GPU utilization is low and per-step overhead dominates.

4. **Memory**: Each environment uses ~50KB of state. At 8192 envs, total GPU memory is ~400MB.

5. **Spatial tendons**: Wrapping geometry (sphere/cylinder) for tendon paths is **deferred** (Phase 5.2). MJX supports it but it requires ~400 lines of geodesic computation; most RL models don't need it.

6. **MUSCLE actuators**: MUSCLE gain/bias/dynamics **deferred** (Phase 6.3). Biomechanical models only.

7. **Sensors**: Not implemented (Phase 8). ~50 sensor types; RL training reads joint/body state directly. **DEFERRED.**

8. **Differentiable physics**: `grad(step)` via `mx::grad` (Phase 9). Requires full pipeline to be autodiff-compatible. Separate project-scale effort. **DEFERRED.**

9. **Inverse dynamics / constraint islands / noslip / sleep** (Phase 10): Niche features not needed for common RL. **DEFERRED.**
