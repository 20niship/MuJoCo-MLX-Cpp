# Architecture

Comprehensive design document for MuJoCo-MLX-Cpp. This documents key architectural decisions, trade-offs, and lessons learned during development. **Read this before making changes to avoid undoing intentional design choices.**

## Table of Contents

- [High-Level Pipeline](#high-level-pipeline)
- [Why Metal + Vmap Hybrid](#why-metal--vmap-hybrid)
- [Computation Graph Rules](#computation-graph-rules)
- [Key Data Structures](#key-data-structures)
- [Cholesky Factorization](#cholesky-factorization)
- [Per-Body Loops vs Scatter Matrices](#per-body-loops-vs-scatter-matrices)
- [Mass Matrix Construction](#mass-matrix-construction)
- [Constraint Solver](#constraint-solver)
- [Constraint Construction](#constraint-construction)
- [Metal Kernels](#metal-kernels)
- [Performance Characteristics](#performance-characteristics)
- [Comparison with Python mujoco-mlx and MJX](#comparison-with-python-mujoco-mlx-and-mjx)
- [Known Issues and Future Work](#known-issues-and-future-work)

---

## High-Level Pipeline

The batched simulation runs a 3-phase hybrid pipeline for each timestep:

```
 Phase 1: Metal Kinematics Kernel
 ─────────────────────────────────
 Single GPU dispatch for ALL N environments.
 qpos -> xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos, geom_xmat
 Replaces ~130 separate MLX dispatches with one fused Metal kernel.

         │
         ▼

 Phase 2: compile(vmap(forward_dynamics))
 ─────────────────────────────────────────
 Pure MLX array operations, traced by vmap across N environments.
 COM position, CRB mass matrix, Cholesky factorization,
 collision detection, constraint generation, solver,
 COM velocity, RNE, actuation, acceleration,
 rne_post_constraint (cfrc_ext).
 mx::compile fuses the computation graph into fewer GPU dispatches.

         │
         ▼

 Phase 3: Metal Euler Kernel
 ────────────────────────────
 Single GPU dispatch for ALL N environments.
 Built-in Cholesky + solve + velocity update + position integration.
 Replaces the entire integration step with one fused Metal kernel.
```

The pipeline is orchestrated in `batched.cpp:make_batched_step()`.

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

Functions that follow these rules: `vmap_com_pos`, `vmap_crb`, `vmap_factor_m`, `vmap_com_vel`, `vmap_rne`, `vmap_collision`, `vmap_make_constraint`, `vmap_solve`, `vmap_transmission`, `vmap_passive`, `vmap_fwd_actuation`, `vmap_fwd_acceleration`.

**Model constants** (topology arrays, scatter matrices, masks) are accessed inside vmap functions but are never mutated. They are precomputed in `ModelCache` during `init_cache()` using `eval()` + `data<>()`, which is safe because `init_cache()` runs at model load time, not inside vmap.

---

## Key Data Structures

### Model

Loaded from MuJoCo XML via `io.cpp`. Contains all model parameters as `mx::array`:
- Joint/body/geom topology (parentid, bodyid, types)
- Physical properties (mass, inertia, damping, stiffness)
- Actuator configuration (gear, bias, limits)
- Solver options (solver type, iterations, timestep)

### ModelCache

Precomputed at load time by `init_cache()`. Contains:
- **Tree topology**: `tree_levels` (bodies grouped by depth), `body_dofs` (DOF indices per body), `body_parentid_vec` (C++ vector for fast indexing)
- **Scatter matrices**: For level-parallel backward accumulation in `vmap_com_pos` and `vmap_crb`
- **Mass matrix mask**: Lower-triangular DOF ancestor mask (`make_m_mask`)
- **Collision pairs**: Pre-filtered geometry pairs for broadphase
- **Joint plans**: Integration plans (simple/free/ball joints), actuator moment matrices
- **CDoF plan**: Precomputed indices for vectorized cdof computation
- **Body DOF masks**: For Jacobian computation in constraints

### Data

Per-environment state. In batched mode, vmap adds a leading batch dimension:
- **State**: qpos, qvel, qacc, ctrl, act
- **Derived**: xpos, xquat, xmat (from kinematics), cinert, cvel, cdof, cdof_dot (from smooth dynamics), qfrc_bias, qfrc_smooth, qfrc_constraint (from RNE/solver)
- **Constraint**: efc_J, efc_D, efc_aref, contact info

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

**File:** `solver_vmap.cpp`

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

### prev_grad correctness (CG)

In the Polak-Ribiere beta computation, `prev_grad` must be captured **before** updating `grad`. An earlier bug used the already-updated gradient in the denominator, causing CG to diverge.

---

## Constraint Construction

**File:** `constraint_vmap.cpp`

### Fixed-size outputs

All pre-computed collision pairs are always evaluated, even when not in contact. Inactive constraints get `D=0`, which makes them no-ops in the solver. This ensures uniform output shapes across environments, which is required for vmap.

### Degenerate normal fallback

`vmap_make_frame` computes tangent frames from contact normals using cross products. When the normal is near-parallel to `[0,0,1]` (cross product magnitude < 1e-6), it falls back to `[0,1,0]` as the reference axis. This uses `mx::where` for branch-free execution.

---

## Metal Kernels

### Kinematics Kernel (Phase 1)

Source-generated MSL in `batched.cpp:make_kinematics_source()`. Embeds model topology directly in the shader:
- Joint types, parent IDs, qpos/qvel addresses are compile-time constants
- Forward kinematics loop unrolled over joint chain
- Stack memory: ~7 floats per body (pos, quat). Skip if `nbody * 7 * 4 > 24KB`

### Euler Kernel (Phase 3)

Source-generated MSL in `batched.cpp:make_euler_source()`. Embeds Cholesky factorization and solve inline:
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

## Phase 1 Conformance Additions

### rne_post_constraint (cfrc_ext)

`rne_post_constraint()` is now called automatically at the end of the `forward()` pipeline (in `forward.cpp`), after the constraint solver. It computes `cfrc_ext` — the per-body sum of external contact and constraint forces — using a `cdof`-based projection from `qfrc_constraint`. This is an approximation compared to MuJoCo C's geometry-based approach using `efc_force`, but produces matching results for typical contact scenarios (validated within 0.001 tolerance against MuJoCo C).

### Gravity Compensation

`passive()` in `passive.cpp` now includes a `gravcomp()` function that computes `qfrc_gravcomp` for bodies with `body_gravcomp != 0`. The force is computed as `-(mass * gravcomp * gravity)` and projected to joint space via the translational Jacobian at the body's COM. The result is accumulated into `qfrc_passive`.

### Exclude Signature

Collision filtering via `exclude_signature` is implemented in both `init_cache()` (for vmap pair pre-filtering) and the scalar `collision()` narrowphase loop. The encoding matches MuJoCo C: `(min_body_id << 16) | max_body_id`.

### Model Validation

`validate_model()` in `io.cpp` scans the `mjModel` at load time and emits `[mjmlx WARNING]` messages to stderr for unsupported features (mesh/box/hfield/ellipsoid/cylinder geoms, tendon/site transmission, muscle actuators, equality constraints, RK4/implicit integrators, sensors). Models still load and simulate with the supported subset.

---

## Phase 2 Conformance Additions

### Pyramidal Friction (condim=3)

`constraint.cpp` and `constraint_vmap.cpp` now generate pyramidal friction constraint rows when `condim >= 3` and `cone == PYRAMIDAL` (the default). For condim=3, each contact produces 4 constraint rows — two opposing pyramid edges per tangent direction:

```
J_edge[2k]   = J_normal + μ[k] * J_tangent[k]
J_edge[2k+1] = J_normal - μ[k] * J_tangent[k]    for k = 0, 1
```

The impedance for pyramidal rows uses a friction-scaled formula:
- `invw_py = invw * (1 + μ²)` — diagonal approximation correction
- `R_py = 2μ² * invw_py * (1-imp)/imp / impratio` — pyramidal R

All 4 rows are simple unilateral inequalities (force >= 0), so the existing CG/Newton solver handles them without modification. The `max_nefc` computation in `io.cpp` now accounts for `2*(condim-1)` rows per friction contact pair.

D values and aref match MuJoCo C within 0.001% (validated in `test_friction_condim3.cpp`).

---

## Known Issues and Future Work

1. **Graph size**: The computation graph (~6,200 nodes) is large. Reducing it would speed up both compilation and execution. Main targets:
   - Cholesky (945 nodes for 27x27): Could benefit from a custom Metal kernel for batched factorization
   - Per-body loops in com_vel/rne (~1,800 nodes): Vectorization was attempted and failed (see above), but a custom Metal kernel approach might work

2. **First-step compilation**: The first batched step triggers `mx::compile`, which takes ~200ms. Subsequent steps are fast.

3. **Small batch sizes**: Below 1024 envs, GPU utilization is low and per-step overhead dominates.

4. **Memory**: Each environment uses ~50KB of state. At 8192 envs, total GPU memory is ~400MB.
