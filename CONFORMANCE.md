# MuJoCo Conformance Analysis

Deep comparison of MuJoCo-MLX-Cpp against MuJoCo C and Google's MJX (JAX).
Last updated: 2026-02-17 (Phase 3.3 MESH/GJK/EPA collisions complete).

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
- Both scalar and vmap/batched paths implemented (vmap uses sphere approximation)
- Mesh vertex data loaded from MuJoCo model into internal arrays
- Support functions: sphere, capsule, box, cylinder, mesh (argmax vertex·direction)

---

## Scale Comparison

| Codebase | Language | Engine Lines | Core Files | Feature Coverage |
|----------|----------|-------------|------------|-----------------|
| MuJoCo C engine | C | ~47,700 | 80 (.c + .h) | 100% (reference) |
| MJX (Google) | Python/JAX | ~11,500 (core) | 21 | ~80% |
| MuJoCo-MLX-Cpp | C++ | ~7,562 | 22 (.cpp + .h) | ~25-30% |

MuJoCo-MLX-Cpp reimplements roughly 16% of MuJoCo C by code volume. The feature
gap is wider than the code gap because the missing features (collision geometry,
constraint friction, tendons, muscles) are among the most complex subsystems.

MJX has 50% more core code and covers ~3x more features. Both MJX and
MuJoCo-MLX-Cpp are targeted subsets, not full reimplementations.

---

## Collision System

The collision system is the largest divergence. MuJoCo C dedicates ~9,000 lines
across 6 files to a 9x9 geom-type dispatch table with analytic, convex
(GJK/EPA), and SDF paths.

### Geom Types

| Geom Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|-----------|----------|-----|----------------|
| PLANE | Yes | Yes | Yes |
| HFIELD | Yes | Yes | **No** |
| SPHERE | Yes | Yes | Yes |
| CAPSULE | Yes | Yes | Yes |
| ELLIPSOID | Yes | Partial (SDF) | **No** |
| CYLINDER | Yes | Partial (SDF) | Yes |
| BOX | Yes | Yes (as mesh) | Yes |
| MESH | Yes | Yes (vertex limit) | Yes (GJK/EPA) |
| SDF | Yes | Yes | **No** |

### Collision Pairs Implemented

MuJoCo C has 36+ pair functions. MuJoCo-MLX-Cpp has 12+ (including GJK/EPA generic path):

| Pair | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| plane-sphere | `mjc_PlaneSphere` | Yes | Yes |
| plane-capsule | `mjc_PlaneCapsule` | Yes | Yes |
| plane-cylinder | `mjc_PlaneCylinder` | Yes | Yes (multi-contact, up to 3) |
| plane-box | `mjc_PlaneBox` | Yes | Yes (multi-contact, up to 4) |
| plane-convex | `mjc_PlaneConvex` | Yes | **No** |
| sphere-sphere | `mjc_SphereSphere` | Yes | Yes |
| sphere-capsule | `mjc_SphereCapsule` | Yes | Yes |
| sphere-cylinder | `mjc_SphereCylinder` | Yes | Yes |
| sphere-box | `mjc_SphereBox` | Yes | Yes |
| capsule-capsule | `mjc_CapsuleCapsule` | Yes | Yes |
| capsule-cylinder | `mjc_CapsuleCylinder` | Yes | Yes |
| capsule-box | `mjc_CapsuleBox` | Yes | Yes |
| box-box | `mjc_BoxBox` | Yes | Yes (SAT, single contact) |
| convex-convex | GJK/EPA | GJK/SAT | Yes (GJK/EPA) |
| mesh-* | GJK/EPA | SAT (vertex limit) | Yes (GJK/EPA, multi-contact for plane) |
| hfield-* | `mjc_ConvexHField` | Yes | **No** |
| sdf-* | `mjc_SDF` | Yes | **No** |

### Collision Code Volume

| Codebase | Collision Code |
|----------|---------------|
| MuJoCo C | ~9,000 lines (6 files: primitive, box, convex, GJK, SDF, driver) |
| MJX | ~2,100 lines (4 files: primitive, convex, SDF, driver) |
| MuJoCo-MLX-Cpp | ~840 lines (collision.cpp + constraint_vmap.cpp collision parts) |

### Exclude Signature (Phase 1.3)

`exclude_signature` is now fully implemented. The `nexclude` count and
`exclude_signature` array are loaded from the MuJoCo model and applied in both
the `init_cache()` collision pair pre-filter (for the vmap path) and the scalar
`collision()` narrowphase loop. Signature encoding matches MuJoCo C:
`(min_body_id << 16) | max_body_id`.

### Impact

Any model with CYLINDER, MESH, or HFIELD geoms will have **no collisions
at all** on the MLX backend for those geom types. BOX collision is now supported
(Phase 3.1). The humanoid works because `foot_contacts_only` filters down to
capsule-plane pairs.

---

## Constraint System

### Constraint Types

| Constraint Type | MuJoCo C (2,541 lines) | MJX (748 lines) | MuJoCo-MLX-Cpp (699 lines) |
|----------------|------------------------|------------------|-----------------------------|
| Equality: CONNECT | Yes | Yes | **No** |
| Equality: WELD | Yes | Yes | **No** |
| Equality: JOINT | Yes | Yes | **No** |
| Equality: TENDON | Yes | Yes | **No** |
| Equality: FLEX | Yes | No | **No** |
| Equality: DISTANCE | Yes (unsupported) | No | **No** |
| Joint limits (HINGE/SLIDE) | Yes | Yes | Yes |
| Tendon limits | Yes | Yes | **No** |
| DOF friction loss | Yes | Yes | **No** |
| Tendon friction loss | Yes | Yes | **No** |
| Contact: frictionless (condim=1) | Yes | Yes | Yes |
| Contact: pyramidal friction (condim=3) | Yes | Yes | **Yes (Phase 2.1)** |
| Contact: pyramidal friction (condim=4,6) | Yes | Yes | **Yes (Phase 2.2)** |
| Contact: elliptic friction | Yes | Yes | **No** |

### Contact Dimension (condim)

MuJoCo C generates 1-10 constraint rows per contact depending on `condim` and cone type:

Pyramidal cone (2*(condim-1) rows):
- condim=1: 1 row (normal only, frictionless)
- condim=3: 4 rows (2 tangent directions × 2 pyramid edges) — **Implemented (Phase 2.1)**
- condim=4: 6 rows (3 directions × 2 edges)
- condim=6: 10 rows (5 directions × 2 edges)

Elliptic cone (condim rows):
- condim=1: 1 row
- condim=3: 3 rows (normal + 2 tangent)
- condim=4: 4 rows
- condim=6: 6 rows

MuJoCo-MLX-Cpp generates **1 row per contact (normal only)**, regardless of
the model's `condim` setting. This means objects cannot grip, resist sliding,
or produce torsional friction.

### Equality Constraints

MuJoCo-MLX-Cpp loads `eq_type`, `eq_obj1id`, `eq_obj2id`, `eq_data`, `eq_solref`,
`eq_solimp` from the MuJoCo model in `io.cpp`, but these arrays are **never
used** in the constraint pipeline. Models with closed kinematic chains, fixed
attachments, or joint coupling will produce incorrect physics.

---

## Integrators

| Integrator | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------------|----------|-----|----------------|
| Euler (semi-implicit) | `mj_Euler` (engine_forward.c) | Yes | Yes + Metal kernel |
| RK4 | `mj_RungeKutta` | Yes | **Declared in enum, not implemented** |
| Implicit | `mj_implicit` (with RNE deriv) | In development | **No** |
| ImplicitFast | `mj_implicit` (no RNE deriv) | Yes | **No** |

### Note on Implicit Euler Damping

MuJoCo-MLX-Cpp's Metal Euler kernel adds `dof_damping * dt` to the mass matrix
diagonal before Cholesky factorization. This is a practical stabilization hack
that provides some of the benefit of implicit integration for damped systems,
but it is **not** the full implicit integrator. MuJoCo C's implicit integrator
additionally accounts for the Jacobian of bias forces (RNE derivative).

### Impact

Euler integration is sufficient for most RL training scenarios. The implicit
integrator matters most for stiff systems (high-gain PD controllers, tendons,
muscle dynamics) where Euler requires very small timesteps.

---

## Actuators and Transmission

### Transmission Types

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| JOINT | Yes | Yes | Yes |
| JOINTINPARENT | Yes | Yes | **No** |
| SLIDERCRANK | Yes | No | **No** |
| TENDON | Yes | Yes | **No** |
| SITE | Yes | Yes | **No** |
| BODY | Yes | No | **No** |

MuJoCo-MLX-Cpp's `io.cpp` line 814: `if (trntype != 0) continue` -- any
non-JOINT transmission is silently skipped.

### Gain Types

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| FIXED | Yes | Yes | Yes |
| AFFINE | Yes | Yes | Yes |
| MUSCLE | Yes | Yes | **No** |
| USER | Yes | No | **No** |

### Bias Types

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| NONE | Yes | Yes | Yes |
| AFFINE | Yes | Yes | Yes (partial -- no velocity term in scalar path) |
| MUSCLE | Yes | Yes | **No** |
| USER | Yes | No | **No** |

### Actuator Dynamics

| Type | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|------|----------|-----|----------------|
| NONE | Yes | Yes | Yes (default) |
| INTEGRATOR | Yes | Yes | **No** |
| FILTER | Yes | Yes | **No** |
| FILTEREXACT | Yes | Yes | **No** |
| MUSCLE | Yes | Yes | **No** |
| USER | Yes | No | **No** |

`act_dot` is allocated in `make_data()` but never computed. Models with
position-controlled actuators (PD servos via FILTER dynamics) will produce
**zero actuator activation** -- the control signal passes through with FIXED
gain but the filter state never integrates.

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
| **Sensors** (49 types, 1,542 lines) | Full | 787 lines, many types | **None** |
| **Tendons** (wrapping, limits, forces) | Full | Yes (smooth.py) | **None** |
| **Inverse dynamics** (298 lines) | `mj_inverse` | 110 lines | **None** |
| **Analytical derivatives** (1,560 lines) | `mjd_transitionFD` | 73 lines (implicit) | **Stub only** |
| **Finite-difference derivatives** (703 lines) | `mjd_inverseFD` | N/A | **None** |
| **Ray casting** (1,486 lines) | Full | 317 lines | **None** |
| **Sleep/wake** (775 lines) | Full | N/A | **None** |
| **Constraint islands** (642 lines) | Full | N/A | **None** |
| **Flex/deformable** (in passive.c + core_util.c) | Full | **None** | **None** |
| **Gravity compensation** | `body_gravcomp` | Yes | **Yes (Phase 1.1)** — `qfrc_gravcomp` computed and added to `qfrc_passive` |
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
features. As of Phase 1.4, MuJoCo-MLX-Cpp now **emits warnings to stderr** at
model load time for unsupported features:

| Scenario | What happens |
|----------|-------------|
| Model has BOX geoms | **Supported.** All 4 box pair types work (plane/sphere/capsule/box). |
| Model has MESH geoms | **Warning emitted.** Same -- no collisions. |
| Model has HFIELD/ELLIPSOID/CYLINDER geoms | **Warning emitted.** No collisions for those types. |
| Model has TENDON/SITE transmission | **Warning emitted.** Actuator produces zero force. |
| Model has MUSCLE dynamics | **Warning emitted.** `act_dot` never computed. |
| Model has equality constraints | **Warning emitted.** Constraints not enforced. |
| Model uses RK4 integrator | **Warning emitted.** Integration is always Euler. |
| Model uses implicit integrator | **Warning emitted.** Integration is always Euler. |
| Model has sensors | **Warning emitted.** Sensors not computed. |

The `validate_model()` function in `io.cpp` scans the `mjModel` at load time
and prints `[mjmlx WARNING]` messages for each unsupported feature detected.
Models still load and simulate (with the supported subset), but the user is
informed about what won't work. The CPU backend (via `libmjb`) handles
everything correctly.

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
| `io.cpp` (model conversion) | 958 | **High** -- field-by-field copy |
| `internal.h` (Model struct) | 220 | **High** -- must mirror mjModel fields |
| `batched.cpp` (Metal kernels) | 915 | Medium -- physics equations in MSL strings |
| `forward.cpp` (pipeline) | 451 | Low -- stable pipeline structure |
| `smooth.cpp` (dynamics) | 655 | Low -- mathematical, rarely changes |
| `collision.cpp` (detection) | 398 | Low -- analytic formulas |
| `mjb.cpp` (dual backend) | 565 | **Medium** -- wraps both MuJoCo C and MLX APIs |

**Total high-risk code:** ~1,180 lines (io.cpp + internal.h Model struct) that
must track MuJoCo C's model structure across versions.

### Effort to Reach MJX Parity

| Feature | Estimated Effort | Complexity |
|---------|-----------------|------------|
| Contact friction (pyramidal cone) | 2-3 weeks | Constraint gen + solver changes |
| BOX collisions (4 pairs) | 2-4 weeks | Analytic geometry |
| MESH collisions (GJK/EPA) | 4-8 weeks | Algorithm complexity |
| Equality constraints | 2-3 weeks | Jacobian construction |
| Actuator dynamics (filter/integrator) | 1-2 weeks | ODE integration |
| Muscle model | 2-3 weeks | Muscle activation dynamics |
| RK4 integrator | 1-2 weeks | 4-stage evaluation |
| ImplicitFast integrator | 2-4 weeks | RNE derivative |
| Tendon system | 3-4 weeks | Wrapping geometry |
| HFIELD collision | 2-3 weeks | Height field sampling |
| **Total to MJX parity** | **~4-7 months** (1 developer) | |

Each feature also requires:
- Vmap-compatible variant (for batched GPU path)
- Metal kernel updates (kinematics/integration if affected)
- Test suite additions (validated against MuJoCo C reference)
- Performance regression testing

---

## Why It Works for Humanoid RL

The Gymnasium Humanoid-v5 model fits perfectly in the supported subset:

| Feature Used | Supported? |
|-------------|-----------|
| HINGE joints + 1 FREE root | Yes (all 4 joint types) |
| CAPSULE + PLANE geoms | Yes (capsule-plane pair) |
| `foot_contacts_only` filtering | Yes (reduces to 2 contacts) |
| JOINT transmission, FIXED gain | Yes |
| Euler integration | Yes |
| No equality constraints | N/A |
| No tendons or muscles | N/A |
| No MESH/CYLINDER geoms | N/A |

This makes MuJoCo-MLX-Cpp a viable accelerator for the most common RL
locomotion benchmarks (Humanoid, Ant, HalfCheetah, Walker2d, Hopper -- all
use capsule/sphere + plane geoms with simple actuators).

**Models that would NOT work on the MLX backend:**
- Shadow Hand (MESH geoms, TENDON transmission, equality constraints)
- Franka Panda (SITE transmission, equality constraints)
- Any manipulation task (requires friction for grasping)
- Soft body / cloth (FLEX)

These all work correctly on the CPU backend via `libmjb`.

---

## Gap Closure Priority

Ranked by impact-to-effort ratio for expanding the MLX backend:

### Priority 1: Contact Friction (pyramidal cone)
- **Impact:** Unlocks physically realistic contact for all existing geom pairs
- **Effort:** 2-3 weeks
- **Changes:** Constraint generation (3-5 rows per contact instead of 1), solver
  handling of friction constraints, vmap constraint pipeline
- **Why first:** Even with only plane/sphere/capsule, friction makes the physics
  dramatically more realistic. Humanoid training would benefit from foot grip.

### Priority 2: BOX Collisions — DONE (Phase 3.1)
- **Status:** Complete. All 4 box pair types implemented (plane-box with multi-contact,
  sphere-box, capsule-box, box-box with SAT). Both scalar and vmap paths.
- **Tests:** 6 tests in `test_collision_box.cpp`, qacc match within 0.000004 of MuJoCo C.

### Priority 3: Equality Constraints
- **Impact:** Unlocks closed kinematic chains, weld joints, joint coupling
- **Effort:** 2-3 weeks
- **Changes:** Jacobian construction for CONNECT/WELD/JOINT types, always-active
  in solver (not inequality)
- **Why third:** Required for many articulated robot models.

### Priority 4: RK4 + ImplicitFast Integrators
- **Impact:** Better stability for stiff systems
- **Effort:** 3-6 weeks
- **Changes:** Multi-stage evaluation (RK4), RNE derivative (ImplicitFast),
  Metal kernel updates
- **Why fourth:** Euler works for most RL training; stiff systems are niche.

### Priority 5: Actuator Dynamics + TENDON/SITE Transmission
- **Impact:** Unlocks PD-controlled robots, muscle models, tendon-driven hands
- **Effort:** 4-6 weeks
- **Changes:** ODE integration for act_dot, tendon wrapping geometry, SITE
  Jacobians
- **Why fifth:** Complex systems that need this are already better served by the
  CPU backend.

---

## Dual-Backend Mitigation

The `libmjb.dylib` dual-backend library (introduced in Phase 2a) is the
architectural answer to the conformance gap:

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
    | 100% features   |     | ~25-30% features |
    | double precision|     | float32          |
    | CPU threads     |     | Metal GPU        |
    | ~20K SPS batched|     | ~70K+ SPS batched|
    +-----------------+     +------------------+
```

**Strategy:**
1. Any model works on CPU backend (correctness guaranteed)
2. Simple locomotion models get 3.5x speedup on MLX backend
3. Gaps only matter for GPU-batched training on complex models
4. Close gaps incrementally (friction done, BOX done, next: CYLINDER, equality)

---

## Appendix: Full Feature Matrix

### Physics Pipeline

| Stage | MuJoCo C Function | MuJoCo-MLX-Cpp | Notes |
|-------|-------------------|----------------|-------|
| FK kinematics | `mj_kinematics` | `kinematics()` | Full parity. Metal kernel for batched. |
| COM position | `mj_comPos` | `com_pos()` | Full parity |
| CRB inertia | `mj_crb` | `crb()` | Full parity |
| Mass matrix | `mj_makeM` | `factor_m()` | Dense Cholesky (GPU) or sparse LDL |
| Collision | `mj_collision` | `collision()` | 5 of 36+ pairs |
| Constraints | `mj_makeConstraint` | `make_constraint()` | Limits + normal contact only |
| Transmission | `mj_transmission` | `transmission()` | JOINT only |
| COM velocity | `mj_comVel` | `com_vel()` | Full parity |
| Passive forces | `mj_passive` | `passive()` | Spring + damper + **gravcomp (Phase 1.1)** |
| RNE | `mj_rne` | `rne()` | Full parity |
| Actuation | `mj_fwdActuation` | `fwd_actuation()` | FIXED + AFFINE gain only |
| Acceleration | `mj_fwdAcceleration` | `fwd_acceleration()` | Full parity |
| Solve | `mj_fwdConstraint` | `solve()` | CG + Newton (no PGS) |
| Post-constraint RNE | `mj_rnePostConstraint` | `rne_post_constraint()` | **Yes (Phase 1.2)** — computes `cfrc_ext` |
| Euler | `mj_Euler` | `integrate_euler()` | Full parity + Metal kernel |
| RK4 | `mj_RungeKutta` | -- | Not implemented |
| Implicit | `mj_implicit` | -- | Not implemented |
| Flex | `mj_flex` | -- | Not implemented |
| Tendon | `mj_tendon` | -- | Not implemented |
| Sensor | `mj_sensorPos/Vel/Acc` | -- | Not implemented |
| Inverse | `mj_inverse` | -- | Not implemented |

### Summary Statistics

| Metric | MuJoCo C | MJX | MuJoCo-MLX-Cpp |
|--------|----------|-----|----------------|
| Collision pairs | 36+ | ~25 | 5 |
| Constraint types | 8 | 7 | 2 |
| Integrators | 4 | 3 | 1 |
| Transmission types | 6 | 4 | 1 |
| Gain types | 4 | 3 | 2 |
| Dynamics types | 6 | 5 | 1 |
| Sensor types | 49 | ~30 | 0 |
| Solvers | 3 (+noslip) | 2 | 2 |
| Joint types | 4 | 4 | 4 |
| Batched GPU sim | No | Yes (CUDA/TPU) | Yes (Metal) |
| Differentiable | External | Native (JAX) | Planned |
