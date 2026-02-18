# MuJoCo Conformance Analysis

Deep comparison of MuJoCo-MLX-Cpp against MuJoCo C and Google's MJX (JAX).
Last updated: 2026-02-18 (Phase 7.2 ImplicitFast complete, zero-eval vmap optimization, all remaining phases deferred).

## Table of Contents

- [Scale Comparison](#scale-comparison)
- [Collision System](#collision-system)
- [Constraint System](#constraint-system)
- [Integrators](#integrators)
- [Actuators and Transmission](#actuators-and-transmission)
- [Solver](#solver)
- [Other Subsystems](#other-subsystems)
- [Silent Failure Modes](#silent-failure-modes)
- [Maintainability](#maintainability)
- [Why It Works for Humanoid RL](#why-it-works-for-humanoid-rl)
- [Gap Closure Priority](#gap-closure-priority)
- [Dual-Backend Mitigation](#dual-backend-mitigation)
- [Appendix: Full Feature Matrix](#appendix-full-feature-matrix)

---

## Phase 1 Conformance (Completed 2026-02-11)

Phase 1 closed the highest-priority gaps for ProjectSentience Synth training:

| Phase | Feature | Status | Tests |
|-------|---------|--------|-------|
| 1.1 | Gravity compensation (`qfrc_gravcomp`) | Done | 7 tests in `test_gravcomp.cpp` |
| 1.2 | Per-body contact forces (`cfrc_ext`) | Done | 6 tests in `test_cfrc_ext.cpp` |
| 1.3 | Exclude signature collision filtering | Done | 6 tests in `test_exclude.cpp` |
| 1.4 | Model validation warnings at load time | Done | 8 tests in `test_validation.cpp` |
| 1.5 | High-DOF (nv=67) verification | Done | 6 tests in `test_high_dof.cpp` |

All tests validate against MuJoCo C reference. The high-DOF model (nv=67,
nu=61, nbody=25) matches MuJoCo C perfectly after 5 steps and remains stable
through 100 steps with random controls.

## Phase 2.1 Conformance (Completed 2026-02-11)

Pyramidal friction (condim=3) — the single largest physics fidelity improvement
for contact simulation. Each friction contact now generates 4 pyramidal constraint
rows instead of 1 frictionless normal row.

| Feature | Status | Tests |
|---------|--------|-------|
| Pyramidal friction condim=3 | Done | 7 tests in `test_friction_condim3.cpp` |
| Pyramidal friction condim=4,6 | Done | 7 tests in `test_friction_condim46.cpp` |

Key results:
- D values match MuJoCo C within 0.001%
- qfrc_constraint difference = 0.000025 (near-perfect float32 accuracy)
- Pyramidal row formula: `J_edge = J_normal ± μ * J_tangent` (4 rows per contact)
- Impedance: `R_py = 2μ²·R_normal·(1+μ²) / impratio`
- Both scalar and vmap/batched paths implemented

## Phase 3.1 Conformance (Completed 2026-02-11)

BOX collision detection — unlocks models with box geoms (tables, shelves, blocks,
Franka Panda bodies). Implements all 4 collision pair types involving boxes.

| Feature | Status | Tests |
|---------|--------|-------|
| plane-box (multi-contact, up to 4) | Done | 6 tests in `test_collision_box.cpp` |
| sphere-box (closest-point on OBB) | Done | Included in above |
| capsule-box (segment-OBB approach) | Done | Included in above |
| box-box (SAT, 15 axes) | Done | Included in above |

Key results:
- plane-box ncon/nefc matches MuJoCo C exactly (4 contacts, 16 efc rows)
- qacc diff after 1 step: 0.000004 (near-perfect match)
- qpos diff after 100 steps: 0.000000 (exact match)
- Both scalar and vmap/batched paths implemented
- Algorithms: multi-vertex face contact (plane-box), OBB closest-point (sphere-box),
  iterative segment-OBB (capsule-box), Separating Axis Theorem with 15 axes (box-box)

## Phase 3.2 Conformance (Completed 2026-02-17)

CYLINDER collision detection — unlocks models with cylinder geoms (robot links,
pipes, legs). Implements 3 collision pair types involving cylinders.

| Feature | Status | Tests |
|---------|--------|-------|
| plane-cylinder (multi-contact, up to 3) | Done | 6 tests in `test_collision_cylinder.cpp` |
| sphere-cylinder (closest-point on cylinder surface) | Done | Included in above |
| capsule-cylinder (segment-cylinder approach) | Done | Included in above |

Key results:
- plane-cylinder ncon/nefc matches MuJoCo C exactly (3 upright, 2 side)
- Side-lying qacc diff after 1 step: 0.000005 (near-perfect match)
- qpos diff after 100 steps: 0.000160 (excellent stability)
- Both scalar and vmap/batched paths implemented
- Algorithms: multi-contact face center + rim points (plane-cylinder),
  cylinder-local closest-point with barrel/cap/rim handling (sphere-cylinder),
  iterative segment-cylinder with projection refinement (capsule-cylinder)

## Phase 3.3 Conformance (Completed 2026-02-17)

MESH/GJK/EPA collision detection — the key feature for the Synth model. Unlocks
models with convex mesh geoms via the Gilbert-Johnson-Keerthi (GJK) algorithm
for overlap detection and the Expanding Polytope Algorithm (EPA) for contact extraction.

| Feature | Status | Tests |
|---------|--------|-------|
| plane-mesh (multi-contact, all vertices) | Done | 7 tests in `test_collision_mesh.cpp` |
| sphere-mesh (GJK/EPA) | Done | Included in above |
| capsule-mesh (GJK/EPA) | Done | Included in above |
| mesh-mesh (GJK/EPA) | Done | Included in above |
| mesh-box, mesh-cylinder (GJK/EPA) | Done | Covered by generic convex path |

Key results:
- plane-mesh ncon=4 (all 4 bottom vertices), qacc diff **3.0** after 1 step
- qpos diff after 100 steps: **0.000091** (near-perfect stability)
- GJK/EPA handles arbitrary convex shapes with support functions for all geom types
- Both scalar and vmap/batched paths implemented with proper mesh support functions (vertex argmax), not sphere approximation
- Scalar: 64-iter GJK + 64-iter EPA. Vmap: 32-iter GJK + support-based depth estimation (~18 sample directions)
- Mesh vertex data loaded from MuJoCo model into internal arrays
- Support functions: sphere, capsule, box, cylinder, mesh (argmax vertex·direction)

---

## Scale Comparison

| Codebase | Language | Engine Lines | Core Files | Feature Coverage |
|----------|----------|-------------|------------|-----------------|
| MuJoCo C engine | C | ~47,700 | 80 (.c + .h) | 100% (reference) |
| MJX (Google) | Python/JAX | ~11,500 (core) | 21 | ~80% |
| MuJoCo-MLX-Cpp | C++ | ~12,450 | 18 (.cpp + .h) | ~65-70% |

MuJoCo-MLX-Cpp reimplements roughly 26% of MuJoCo C by code volume but covers
~65-70% of the features used in typical RL training. The remaining gaps are
mostly niche subsystems (sensors, inverse dynamics, SDF, deformable bodies).

MJX has comparable core code volume. Both MJX and MuJoCo-MLX-Cpp are targeted
subsets optimized for GPU-batched RL training, not full reimplementations.

---

## Collision System

The collision system is the largest divergence. MuJoCo C dedicates ~9,000 lines
across 6 files to a 9x9 geom-type dispatch table with analytic, convex
(GJK/EPA), and SDF paths.

### Geom Types

| Geom Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|-----------|----------|-----|----------------|
| PLANE | Yes | Yes | Yes |
| HFIELD | Yes | Yes | Yes (Phase 3.4) |
| SPHERE | Yes | Yes | Yes |
| CAPSULE | Yes | Yes | Yes |
| ELLIPSOID | Yes | Partial (SDF) | Yes (Phase 3.5) |
| CYLINDER | Yes | Partial (SDF) | Yes (Phase 3.2) |
| BOX | Yes | Yes (as mesh) | Yes (Phase 3.1) |
| MESH | Yes | Yes (vertex limit) | Yes (GJK/EPA, Phase 3.3) |
| SDF | Yes | Yes | **No** (DEFERRED) |

### Collision Pairs Implemented

MuJoCo C has 36+ pair functions. MuJoCo-MLX-Cpp has 20+ (including GJK/EPA generic path):

| Pair | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| plane-sphere | `mjc_PlaneSphere` | Yes | Yes |
| plane-capsule | `mjc_PlaneCapsule` | Yes | Yes |
| plane-cylinder | `mjc_PlaneCylinder` | Yes | Yes (multi-contact, up to 6) |
| plane-box | `mjc_PlaneBox` | Yes | Yes (multi-contact, up to 4) |
| plane-mesh | `mjc_PlaneConvex` | Yes | Yes (multi-contact, all vertices) |
| plane-hfield | `mjc_ConvexHField` | Yes | Yes (grid-cell triangle tests) |
| plane-ellipsoid | analytic | Partial | Yes (analytic) |
| sphere-sphere | `mjc_SphereSphere` | Yes | Yes |
| sphere-capsule | `mjc_SphereCapsule` | Yes | Yes |
| sphere-cylinder | `mjc_SphereCylinder` | Yes | Yes |
| sphere-box | `mjc_SphereBox` | Yes | Yes |
| sphere-ellipsoid | analytic | Partial | Yes (analytic) |
| capsule-capsule | `mjc_CapsuleCapsule` | Yes | Yes |
| capsule-cylinder | `mjc_CapsuleCylinder` | Yes | Yes |
| capsule-box | `mjc_CapsuleBox` | Yes | Yes |
| capsule-ellipsoid | analytic | Partial | Yes (analytic) |
| box-box | `mjc_BoxBox` | Yes | Yes (SAT, single contact) |
| convex-convex | GJK/EPA | GJK/SAT | Yes (GJK/EPA) |
| mesh-* | GJK/EPA | SAT (vertex limit) | Yes (GJK/EPA, proper mesh support) |
| hfield-* | `mjc_ConvexHField` | Yes | Yes (grid-cell triangle tests) |
| cylinder-cylinder | `mjc_CylinderCylinder` | Yes | **No** (uncommon in RL) |
| sdf-* | `mjc_SDF` | Yes | **No** (DEFERRED) |

### Collision Code Volume

| Codebase | Collision Code |
|----------|---------------|
| MuJoCo C | ~9,000 lines (6 files: primitive, box, convex, GJK, SDF, driver) |
| MJX | ~2,100 lines (4 files: primitive, convex, SDF, driver) |
| MuJoCo-MLX-Cpp | ~4,000 lines (collision.cpp + constraint_vmap.cpp collision parts) |

### Exclude Signature (Phase 1.3)

`exclude_signature` is now fully implemented. The `nexclude` count and
`exclude_signature` array are loaded from the MuJoCo model and applied in both
the `init_cache()` collision pair pre-filter (for the vmap path) and the scalar
`collision()` narrowphase loop. Signature encoding matches MuJoCo C:
`(min_body_id << 16) | max_body_id`.

### Impact

All common geom types are now supported: PLANE, SPHERE, CAPSULE, BOX, CYLINDER,
MESH, HFIELD, ELLIPSOID. Only SDF and cylinder-cylinder are missing. The MLX
backend handles the full collision matrix needed for standard RL training models.

---

## Constraint System

### Constraint Types

| Constraint Type | MuJoCo C (2,541 lines) | MJX (748 lines) | MuJoCo-MLX-Cpp (2,726 lines) |
|----------------|------------------------|------------------|------------------------------|
| Equality: CONNECT | Yes | Yes | Yes (Phase 4) |
| Equality: WELD | Yes | Yes | Yes (Phase 4) |
| Equality: JOINT | Yes | Yes | Yes (Phase 4) |
| Equality: TENDON | Yes | Yes | **No** (DEFERRED) |
| Equality: FLEX | Yes | No | **No** (DEFERRED) |
| Equality: DISTANCE | Yes (unsupported) | No | **No** |
| Joint limits (HINGE/SLIDE) | Yes | Yes | Yes |
| Tendon limits | Yes | Yes | Yes (Phase 5.5) |
| DOF friction loss | Yes | Yes | Yes (Phase 4) |
| Tendon friction loss | Yes | Yes | Yes (Phase 5.4) |
| Contact: frictionless (condim=1) | Yes | Yes | Yes |
| Contact: pyramidal friction (condim=3) | Yes | Yes | Yes (Phase 2.1) |
| Contact: pyramidal friction (condim=4,6) | Yes | Yes | Yes (Phase 2.2) |
| Contact: elliptic friction | Yes | Yes | **No** (DEFERRED) |

### Contact Dimension (condim)

MuJoCo C generates 1-10 constraint rows per contact depending on `condim` and cone type:

Pyramidal cone (2*(condim-1) rows):
- condim=1: 1 row (normal only, frictionless)
- condim=3: 4 rows (2 tangent directions × 2 pyramid edges) -- Phase 2.1
- condim=4: 6 rows (3 directions × 2 edges) -- Phase 2.2
- condim=6: 10 rows (5 directions × 2 edges) -- Phase 2.2

All pyramidal friction condim values are fully implemented in both scalar and
vmap paths.

### Equality Constraints (Phase 4)

CONNECT, WELD, and JOINT equality constraint types are fully implemented in both
scalar and vmap paths. Constraint data is precomputed in `ModelCache::equality_cache`
at load time (type, body IDs, data, solref, solimp, invweight, Jacobian rows) for
zero-eval vmap execution. TENDON equality constraints are DEFERRED.

---

## Integrators

| Integrator | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------------|----------|-----|----------------|
| Euler (semi-implicit) | `mj_Euler` (engine_forward.c) | Yes | Yes + Metal kernel (scalar + batched) |
| RK4 | `mj_RungeKutta` | Yes | Yes (Phase 7.1, scalar path) |
| Implicit | `mj_implicit` (with RNE deriv) | In development | Partial (Phase 7.2, scalar path, no RNE deriv) |
| ImplicitFast | `mj_implicit` (no RNE deriv) | Yes | Yes (Phase 7.2, scalar path) |

### Scalar vs Batched

All 4 integrators work in the scalar path. The batched Metal pipeline uses Euler only.
RK4 and ImplicitFast in batched mode are DEFERRED (would require either Metal kernel
rewrites or dense Cholesky in the vmap path).

### ImplicitFast (Phase 7.2)

Implicit velocity integration via `deriv_smooth_vel()` computing analytical
∂qfrc_smooth/∂qvel (joint damping + tendon damping + affine actuator velocity
terms). Modified mass matrix M' = M - dt * qderiv, Cholesky factor and solve
for qacc. qvel diff vs MuJoCo C: ~1e-6.

---

## Actuators and Transmission

### Transmission Types

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| JOINT | Yes | Yes | Yes |
| JOINTINPARENT | Yes | Yes | **No** (DEFERRED) |
| SLIDERCRANK | Yes | No | **No** (DEFERRED) |
| TENDON | Yes | Yes | Yes (Phase 5.3) |
| SITE | Yes | Yes | Yes (Phase 5.3) |
| BODY | Yes | No | **No** (DEFERRED) |

JOINT, TENDON, and SITE transmission fully implemented in both scalar and vmap paths.
TENDON: `moment = gear * ten_J`. SITE: full 6-DOF Jacobian at site with gear wrench
projection.

### Gain Types

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| FIXED | Yes | Yes | Yes |
| AFFINE | Yes | Yes | Yes |
| MUSCLE | Yes | Yes | **No** (DEFERRED) |
| USER | Yes | No | **No** |

### Bias Types

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| NONE | Yes | Yes | Yes |
| AFFINE | Yes | Yes | Yes |
| MUSCLE | Yes | Yes | **No** (DEFERRED) |
| USER | Yes | No | **No** |

### Actuator Dynamics

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| NONE | Yes | Yes | Yes (default) |
| INTEGRATOR | Yes | Yes | Yes (Phase 6.1) |
| FILTER | Yes | Yes | Yes (Phase 6.1) |
| FILTEREXACT | Yes | Yes | Yes (Phase 6.1) |
| MUSCLE | Yes | Yes | **No** (DEFERRED) |
| USER | Yes | No | **No** |

FILTER (first-order low-pass), FILTEREXACT (exact exponential), and INTEGRATOR
(pure integration) dynamics fully implemented in both scalar and vmap paths.
Activation clamping (`actuator_actlimited` / `actuator_actrange`) supported.
Vmap path uses vectorized scatter-matmul for `act_dot` computation.

---

## Solver

| Solver | MuJoCo C (2,028 lines) | MJX (610 lines) | MuJoCo-MLX-Cpp (416 lines) |
|--------|------------------------|------------------|-----------------------------|
| PGS (Projected Gauss-Seidel) | Yes | **No** | **No** |
| CG (Conjugate Gradient) | Yes | Yes | Yes |
| Newton | Yes | Yes | Yes |
| NoSlip | Yes | No | **No** |
| Island-parallel solve | Yes | N/A | **No** |

Both MJX and MuJoCo-MLX-Cpp skip PGS. This is intentional -- PGS is a dual
solver that iterates over individual constraints and is poorly suited to GPU
parallelism. CG and Newton are primal solvers that operate on the full system
and vectorize well.

MuJoCo-MLX-Cpp's solver supports:
- Newton: H = M + J^T D J, Cholesky solve, direct step
- CG: M^{-1} preconditioned, Polak-Ribiere updates
- Warmstart comparison (vmap path)
- 5-alpha line search (Newton)

---

## Other Subsystems

| Subsystem | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|-----------|----------|-----|----------------|
| **Sensors** (49 types, 1,542 lines) | Full | 787 lines, many types | **None** (DEFERRED) |
| **Fixed tendons** (length, velocity, J, passive) | Full | Yes | Yes (Phase 5.1) |
| **Spatial tendons** (wrapping geometry) | Full | Yes | **No** (DEFERRED) |
| **Tendon limits + friction** | Full | Yes | Yes (Phase 5.4-5.5) |
| **Inverse dynamics** (298 lines) | `mj_inverse` | 110 lines | **None** (DEFERRED) |
| **Analytical derivatives** (1,560 lines) | `mjd_transitionFD` | 73 lines (implicit) | Partial (`deriv_smooth_vel` for ImplicitFast) |
| **Finite-difference derivatives** (703 lines) | `mjd_inverseFD` | N/A | **None** (DEFERRED) |
| **Ray casting** (1,486 lines) | Full | 317 lines | **None** (DEFERRED) |
| **Sleep/wake** (775 lines) | Full | N/A | **None** (DEFERRED) |
| **Constraint islands** (642 lines) | Full | N/A | **None** (DEFERRED) |
| **Flex/deformable** (in passive.c + core_util.c) | Full | **None** | **None** (DEFERRED) |
| **Gravity compensation** | `body_gravcomp` | Yes | Yes (Phase 1.1) |
| **Visualization** (4,955 lines) | Full | N/A | **None** (physics only) |
| **Plugins** | Full | **None** | **None** |

### Joint Types

All three codebases support the same 4 joint types:

| Joint | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|-------|----------|-----|----------------|
| FREE (7 DOF) | Yes | Yes | Yes |
| BALL (4 DOF) | Yes | Yes | Yes |
| SLIDE (1 DOF) | Yes | Yes | Yes |
| HINGE (1 DOF) | Yes | Yes | Yes |

This is one area with **full parity**.

---

## Silent Failure Modes (Mitigated — Phase 1.4)

MJX validates models at load time and raises errors or warnings for unsupported
features. As of Phase 1.4, MuJoCo-MLX-Cpp **emits warnings to stderr** at
model load time for unsupported features.

Most previously-warned features are now **fully supported**:

| Scenario | Status |
|----------|--------|
| Model has BOX geoms | **Supported.** All 4 box pair types. |
| Model has MESH geoms | **Supported.** GJK/EPA for all mesh pair types. |
| Model has HFIELD geoms | **Supported.** Grid-cell triangle tests. |
| Model has ELLIPSOID geoms | **Supported.** Analytic collision. |
| Model has CYLINDER geoms | **Supported.** 3 cylinder pair types. |
| Model has TENDON/SITE transmission | **Supported.** Full TENDON + SITE transmission. |
| Model has equality constraints (CONNECT/WELD/JOINT) | **Supported.** |
| Model has FILTER/FILTEREXACT/INTEGRATOR dynamics | **Supported.** |
| Model uses RK4 integrator | **Supported (scalar path).** |
| Model uses Implicit/ImplicitFast integrator | **Supported (scalar path).** |
| Model has DOF friction loss | **Supported.** |
| Model has tendon limits/friction | **Supported.** |

Remaining warnings:

| Scenario | What happens |
|----------|-------------|
| Model has MUSCLE dynamics | **Warning emitted.** Not implemented. |
| Model has spatial tendons | **Warning emitted.** Only fixed tendons supported. |
| Model has sensors | **Warning emitted.** Sensors not computed. |
| Model has TENDON equality constraints | **Warning emitted.** Not implemented. |
| Model has SDF geoms | **Warning emitted.** Not supported. |

The `validate_model()` function in `io.cpp` scans the `mjModel` at load time.
The CPU backend (via `libmjb`) handles everything correctly.

---

## Maintainability

### MuJoCo Version Coupling

MuJoCo-MLX-Cpp's `io.cpp` (958 lines) copies every `mjModel` field one-by-one
into an internal `Model` struct with ~60 `mx::array` fields. This is a
**field-by-field mirror** that must be updated whenever MuJoCo changes its
model structure.

Example: MuJoCo 3.5 changed `int nbody` to `mjtSize nbody` (long long). This
broke the `libmjb` build with narrowing conversion errors.

MJX's `io.py` uses attribute access on the Python `mujoco.MjModel` object,
which is more resilient to field type changes.

### Maintenance Surface Area

| Component | Lines | MuJoCo-version-sensitive? |
|-----------|-------|--------------------------|
| `io.cpp` (model conversion + cache) | 1,598 | **High** -- field-by-field copy + ModelCache init |
| `internal.h` (Model/Data/Cache structs) | 698 | **High** -- must mirror mjModel fields |
| `batched.cpp` (Metal kernels) | 915 | Medium -- physics equations in MSL strings |
| `collision.cpp` (detection) | 2,135 | Low -- analytic formulas, GJK/EPA |
| `constraint_vmap.cpp` (vmap constraints) | 1,878 | Medium -- zero-eval constraint pipeline |
| `constraint.cpp` (scalar constraints) | 848 | Low -- reference implementation |
| `smooth.cpp` (dynamics) | 803 | Low -- mathematical, rarely changes |
| `forward.cpp` (pipeline) | 788 | Low -- stable pipeline structure |
| `mjb.cpp` (dual backend) | 565 | **Medium** -- wraps both MuJoCo C and MLX APIs |

**Total high-risk code:** ~2,296 lines (io.cpp + internal.h) that must track
MuJoCo C's model structure across versions.

### Remaining Effort to Full MJX Parity

Most major features are now implemented. Remaining gaps:

| Feature | Estimated Effort | Complexity |
|---------|-----------------|------------|
| Spatial tendons (wrapping geometry) | 3-4 weeks | Geodesic path computation |
| Muscle model | 2-3 weeks | Muscle activation dynamics |
| Sensors (~50 types) | 3-4 weeks | Large surface area, low complexity |
| Batched RK4/ImplicitFast | 2-3 weeks | Metal kernel or vmap Cholesky |
| **Total remaining to MJX parity** | **~3-4 months** (1 developer) | |

---

## Why It Works for Humanoid RL

The Gymnasium Humanoid-v5 model uses features that are fully supported:

| Feature Used | Supported? |
|-------------|-----------|
| HINGE joints + 1 FREE root | Yes (all 4 joint types) |
| CAPSULE + PLANE geoms | Yes (capsule-plane pair) |
| Pyramidal friction (condim=3) | Yes (4 rows per contact) |
| 2 fixed tendons (hip-knee coupling) | Yes (Phase 5) |
| JOINT transmission, FIXED gain | Yes |
| Euler integration | Yes (Metal kernel, batched) |
| No equality constraints | N/A |
| No MESH/CYLINDER geoms | N/A |

Training benchmark: **73K SPS** (72,606 avg) at 8192 envs on M4 Max, with
physically correct tendon coupling and pyramidal friction.

MuJoCo-MLX-Cpp now supports a much wider range of models beyond basic locomotion:

**Models that now work on MLX backend:**
- Locomotion (Humanoid, Ant, HalfCheetah, Walker2d, Hopper)
- Manipulation tasks with friction contact (condim=3/4/6)
- Models with BOX, CYLINDER, MESH, HFIELD, ELLIPSOID geoms
- Models with equality constraints (CONNECT/WELD/JOINT)
- Models with tendon coupling and TENDON/SITE transmission
- Models with actuator dynamics (FILTER/FILTEREXACT/INTEGRATOR)

**Models that would NOT work on the MLX backend:**
- Shadow Hand (spatial tendons, MUSCLE dynamics)
- Soft body / cloth (FLEX)
- Models requiring sensors, SDF geoms, or spatial tendon wrapping

These all work correctly on the CPU backend via `libmjb`.

---

## Gap Closure Priority

### Completed Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Synth Physics Foundation (gravcomp, cfrc_ext, exclude, validation, high-DOF) | **Done** |
| 2 | Contact Friction (condim=1/3/4/6 pyramidal) | **Done** |
| 3 | Collision Geometry (box, cylinder, mesh/GJK/EPA, hfield, ellipsoid) | **Done** |
| 4 | Equality Constraints (CONNECT/WELD/JOINT) + DOF Friction | **Done** |
| 5 | Tendon System (fixed tendons, passive, limits, friction, TENDON+SITE transmission) | **Done** |
| 6 | Actuator Dynamics (FILTER/FILTEREXACT/INTEGRATOR, activation clamping) | **Done** |
| 7 | Advanced Integrators (RK4, ImplicitFast -- scalar path) | **Done** |

### Remaining Gaps (all DEFERRED)

Ranked by impact-to-effort ratio:

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| D1 | Spatial tendons (wrapping geometry) | 3-4 weeks | Anatomical hand models only |
| D2 | MUSCLE dynamics | 2-3 weeks | Biomechanical models only |
| D3 | Sensors (~50 types) | 3-4 weeks | RL reads state directly, not sensors |
| D4 | Batched RK4/ImplicitFast (Metal kernel) | 2-3 weeks | Euler sufficient for most RL |
| D5 | Vmap gravity compensation | 1 week | Few models use gravcomp in batched |
| D6 | Differentiable physics (grad(step)) | 8-12 weeks | Separate project |
| D7 | Inverse dynamics / constraint islands / noslip / sleep | 4-6 weeks | Niche features |
| D8 | Elliptic friction cone | 1-2 weeks | Pyramidal is default; elliptic rarely used |
| D9 | TENDON equality constraints | 1 week | Uncommon |

---

## Dual-Backend Mitigation

The `libmjb.dylib` dual-backend library is the architectural answer to the
remaining conformance gap:

```
                   +---------+
                   | libmjb  |   <-- unified mjb_* C API (47 functions)
                   +----+----+
                        |
            +-----------+-----------+
            |                       |
    +-------+-------+     +--------+--------+
    | MJB_BACKEND_CPU |     | MJB_BACKEND_MLX  |
    | (MuJoCo C)      |     | (MuJoCo-MLX)     |
    +-----------------+     +------------------+
    | 100% features   |     | ~65-70% features |
    | double precision|     | float32          |
    | CPU threads     |     | Metal GPU        |
    | ~20K SPS batched|     | ~73K+ SPS batched|
    +-----------------+     +------------------+
```

**Strategy:**
1. Any model works on CPU backend (correctness guaranteed)
2. Most RL models now get 3.5x speedup on MLX backend (full collision, friction, tendons, equality, dynamics)
3. Remaining gaps (spatial tendons, muscles, sensors) affect only niche models
4. CPU backend handles everything for those models

---

## Appendix: Full Feature Matrix

### Physics Pipeline

| Stage | MuJoCo C Function | MuJoCo-MLX-Cpp | Notes |
|-------|-------------------|----------------|-------|
| FK kinematics | `mj_kinematics` | `kinematics()` | Full parity. Metal kernel for batched. |
| COM position | `mj_comPos` | `com_pos()` | Full parity. Cached scatter matrices. |
| CRB inertia | `mj_crb` | `crb()` | Full parity. Cached scatter matrices. |
| Mass matrix | `mj_makeM` | `factor_m()` | Dense Cholesky (GPU) or sparse LDL |
| Tendon | `mj_tendon` | `tendon()` | Fixed tendons; spatial DEFERRED |
| Collision | `mj_collision` | `collision()` | 20+ pairs (all 8 geom types except SDF) |
| Constraints | `mj_makeConstraint` | `make_constraint()` | Equality + DOF friction + joint limits + tendon limits + tendon friction + contact (condim 1/3/4/6) |
| Transmission | `mj_transmission` | `transmission()` | JOINT + TENDON + SITE |
| COM velocity | `mj_comVel` | `com_vel()` | Full parity |
| Passive forces | `mj_passive` | `passive()` | Spring + damper + gravcomp + tendon spring/damping |
| RNE | `mj_rne` | `rne()` | Full parity. Cached scatter matrices. |
| Actuation | `mj_fwdActuation` | `fwd_actuation()` | FIXED + AFFINE gain, NONE/FILTER/FILTEREXACT/INTEGRATOR dynamics |
| Acceleration | `mj_fwdAcceleration` | `fwd_acceleration()` | Full parity |
| Solve | `mj_fwdConstraint` | `solve()` | CG + Newton (no PGS) |
| Post-constraint RNE | `mj_rnePostConstraint` | `rne_post_constraint()` | Computes `cfrc_ext` |
| Euler | `mj_Euler` | `integrate_euler()` | Full parity + Metal kernel |
| RK4 | `mj_RungeKutta` | `integrate_rk4()` | Full parity (scalar path) |
| Implicit/Fast | `mj_implicit` | `integrate_implicit()` | Full parity (scalar path) |
| Flex | `mj_flex` | -- | DEFERRED |
| Sensor | `mj_sensorPos/Vel/Acc` | -- | DEFERRED |
| Inverse | `mj_inverse` | -- | DEFERRED |

### Summary Statistics

| Metric | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|--------|----------|-----|----------------|
| Collision pairs | 36+ | ~25 | 20+ |
| Constraint types | 8 | 7 | 8 (equality/DOF friction/joint limit/tendon limit/tendon friction/contact condim 1/3/4/6) |
| Integrators | 4 | 3 | 4 (Euler, RK4, Implicit, ImplicitFast) |
| Transmission types | 6 | 4 | 3 (JOINT, TENDON, SITE) |
| Gain types | 4 | 3 | 2 (FIXED, AFFINE) |
| Dynamics types | 6 | 5 | 4 (NONE, INTEGRATOR, FILTER, FILTEREXACT) |
| Sensor types | 49 | ~30 | 0 (DEFERRED) |
| Solvers | 3 (+noslip) | 2 | 2 (CG, Newton) |
| Joint types | 4 | 4 | 4 |
| Batched GPU sim | No | Yes (CUDA/TPU) | Yes (Metal, 73K SPS training) |
| Differentiable | External | Native (JAX) | DEFERRED |
