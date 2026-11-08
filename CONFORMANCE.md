# MuJoCo Conformance

What the batched GPU engine supports relative to MuJoCo C, and how each item is checked. Scope: the batched GPU path only; the single-env pipeline was removed (see [ARCHITECTURE.md](ARCHITECTURE.md)). Numbers below are float32 GPU vs float64 MuJoCo C unless noted.

Legend: **Yes** supported and tested against MuJoCo C. **Partial** see note. **No** not implemented (the model loads; the feature is ignored or approximated as noted, with a warning on stderr where `validate_model` detects it).

## Dynamics

| Feature | Status | Notes |
|---|---|---|
| Joints: free, ball, hinge, slide | Yes | `gpu_vs_cpu_*`, robustness sweep |
| Kinematics, CRB mass matrix, COM velocity, RNE, passive forces | Yes | `test_gpu_vs_cpu` (humanoid, high-DOF tree nv=67-class, real robots) |
| Joint damping, stiffness, armature | Yes | |
| Gravity compensation (`body_gravcomp`) | Yes | |
| Euler integrator | Yes | fused with solver output (`qacc`) |
| implicit / implicitfast | Partial | damping folded into the explicit kernel the way MuJoCo's actuator velocity derivative does; no full dense implicit solve |
| RK4 | No | |
| Actuators: position/velocity/motor (stateless), gear, ctrl/force limits | Yes | |
| Actuator activation dynamics (FILTER/FILTEREXACT/INTEGRATOR), act limits | No | `act` is not integrated |
| MUSCLE | No | warns at load |
| Transmission: JOINT, TENDON, SITE | Yes | `gpu_vs_cpu_tendon_*` |

## Contacts

| Feature | Status | Notes |
|---|---|---|
| Pyramidal friction, condim 3 / 4 / 6 | Yes | torsion and rolling rows |
| margin / gap, solref/solimp, impratio, refsafe | Yes | |
| contype/conaffinity, exclude, parent-child filtering | Yes | |
| plane vs sphere / capsule / box / cylinder / ellipsoid / mesh | Yes | generic GJK/EPA support functions |
| sphere/capsule vs box | Yes | analytic closest point |
| convex pairs (box, cylinder, ellipsoid, mesh) | Yes | GJK/EPA |
| SDF-SDF, SDF-mesh | Yes | `test_sdf_gear`, `test_sdf_mesh`; SDF vs non-SDF is ignored |
| Heightfield contacts | No | cameras do render hfields |
| Contacts per env | Partial | solver rows are capped (72 for `nv <= 32`, else 256); surplus contacts are dropped |
| External contact force `cfrc_ext` | Partial | normal force aggregated per body, torque 0 |

## Constraints

| Feature | Status | Notes |
|---|---|---|
| Joint limits | Yes | |
| DOF friction loss | Yes | |
| Fixed tendons (length, Jacobian, spring/damper, limits, friction loss) | Yes | |
| Spatial tendons (sites, pulleys, sphere/cylinder wrap, sidesites) | Yes | `gpu_vs_cpu_tendon_sphere/_pulley` |
| Equality constraints (connect/weld/joint/tendon) | No | |
| Newton solver with exact line search | Yes | direction by shared-memory Cholesky |
| CG solver | Partial | `MJMLX_SOLVER_CG=1` uses CG for the Newton direction; no separate PGS/CG solver types |
| noslip, islands, sleep | No | |

## Sensors (batched only)

IMU (gyro/accelerometer/velocimeter/magnetometer), frame pos/quat/vel/acc, joint and ball pos/vel, actuator pos/vel, subtreecom: 19 types, `test_sensors`, G1's built-in MJCF sensors in `model_sensors_g1`. Not supported: force, torque, touch, rangefinder, camera sensors.

## Robots in the RL environments

| Robot | Status |
|---|---|
| Unitree Go2 | training env (`mjmlx_go2_env.py`), flat + stairs |
| Unitree G1 (menagerie, 29 DoF) | training env (`mjmlx_g1_env.py`, flat, Isaac Lab `Isaac-Velocity-Flat-G1-v0`); runs on the non-shared solver path (nv=35), about 4-5K env-steps/s |
| Unitree H1 | CPU/GPU trajectory test only |

## Other

| Feature | Status |
|---|---|
| Camera rendering (RGB + depth), compared against MuJoCo's renderer | Yes (`camera_vs_mujoco`; shadows/translucency/moving lights unsupported) |
| Differentiable physics (`grad(step)`) | No (`mjmlx_grad_step` is a stub) |
| Inverse dynamics, `mj_forward`-style single-env API | No (removed) |
| Per-env models | Partial | `domain_rand=True` + `set_env_params`: per-env body mass/inertia scale, COM offset, friction scale, actuator output scale; validated against MuJoCo C with the same `mjModel` edits (`test_env_rand`). Not randomised: damping/armature/stiffness, geom size, solref/solimp, gravity; `invweight0` is not recomputed |

## Measured agreement with MuJoCo C

- Go2, one control step from the same contact state, joint angles: mean 3.6e-5 rad, p95 8e-7 rad. With `max_condim=3` applied to both sides: mean 1.4e-5 rad, p95 5e-7 rad (`scripts/examples/go2/compare_cpu_gpu_go2.py`).
- Long trajectories with contact diverge chaotically; tests bound them per model rather than expecting bitwise agreement. Arms (piper, openarm) match over the whole video; go2/g1 diverge after landing.
- Camera vs MuJoCo's renderer (third person, 16 envs, 128 px): robot silhouette IoU 0.965, depth error 14.9 mm mean / 37.9 mm p95 at the default LOD.
- `ctest -L correctness` (sensors, SDF, tendons, humanoid/high-DOF/contact stress), `-L robust` / `robust_full` (nv x constraint sweep), `-L models` (real robots, local only), `-L known_issue` (tracked failures).

## Silent-failure checks

`validate_model()` prints `[mjmlx WARNING]` for loaded features that the batched path does not honour (MUSCLE actuators, SITE transmission without sites, SDF geoms against non-SDF). Treat any such warning as "results are not MuJoCo-equivalent for this feature".
