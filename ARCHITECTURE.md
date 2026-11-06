# Architecture

Design notes for MuJoCo-MLX-Cpp: how the batched GPU simulation is structured and why. Read this before changing the pipeline. Benchmark numbers live in [README](README.md#detailed-performance-log) and the [README](README.md); feature coverage in [CONFORMANCE.md](CONFORMANCE.md).

## Scope

- **GPU backend: Vulkan only.** Tensors and kernels run on [`extern/mlx_vulkan`](https://github.com/20niship/mlx_vulkan) (MKX, an MLX-style API on Vulkan compute; GLSL kernels via `mkx::fast`). The earlier Metal/MLX backend was removed (#30).
- **Batched only.** The single-env pipeline (`mjmlx_step`, `mjmlx_forward`, scalar RK4/implicit, ...) was removed; those entry points throw ("Single-env MLX pipeline removed"). Everything runs through `mjmlx_batched_*` / `mjb_batched_*`. A batch of 1 is valid.
- **float32 on the GPU**, compared against MuJoCo C (float64) in tests.
- `src/smooth.cpp`, `solver.cpp`, `constraint.cpp`, `collision.cpp`, `passive.cpp`, `scan.cpp` are leftovers of the removed single-env pipeline and are **not built** (see `MJMLX_SOURCES` in `CMakeLists.txt`).

## Components

```
libmjb  (src/mjb.cpp)      unified C API: MJB_BACKEND_CPU -> MuJoCo C with a thread pool over mjData*
   |                                      MJB_BACKEND_MLX -> libmjmlx (the GPU engine; name is historical)
libmjmlx
   io.cpp                  MJCF (via MuJoCo C mjModel) -> internal Model + ModelCache, load-time options
   batched.cpp             builds the step: kernels, buffers, per-step dispatch order, packed read-back
   compat/mkx_kernels/     GLSL kernel generators: kinematics, forward, collision, solver, tendon, euler
   smooth_vmap / constraint_vmap / solver_vmap / forward_vmap.cpp
                           array-op (non-kernel) versions of the same stages: fallback path and model-cache construction
   sensor.cpp              <sensor> evaluation from step outputs
   camera.cpp              Vulkan renderer (graphics queue), CUDA interop
python/                    nanobind: rl_bindings.cpp (numpy-facing, used by training), bindings.cpp
```

## One simulation substep

All N environments are processed by each dispatch; a kernel thread (or workgroup) is assigned to an env. There is no host synchronisation inside a substep; the host only reads results once per control step.

```
qpos, qvel
  |  kinematics kernel      xpos/xquat/xmat/geom poses (topology baked into the shader as constants)
  |  forward kernel         com_pos, CRB mass matrix, com_vel, passive, actuation, RNE -> qfrc_smooth   (SoA scratch)
  |  tendon kernel          fixed + spatial tendon length / Jacobian (when the model has tendons)
  |  collision kernel       broadphase-free fixed pair table baked into the shader -> contacts
  |  solver kernel          constraint rows + Newton solve, one workgroup per env
  |  cfrc_ext pack          contact normal force -> per-body external force
  |  euler kernel           integrate qvel/qpos directly from the solver's qacc
  v
qpos', qvel'  (+ pack kernel: gathers what the env reads into one buffer, one transfer)
```

`env.step(ctrl, n)` runs `n` substeps per control step (Go2: dt 0.005 x 4) with a single ctrl upload.

### Kernels

- **Kinematics / forward**: one thread per env; scratch outputs are SoA (`array[k * NBATCH + env]`) so neighbouring threads touch neighbouring addresses. This alone was a 6-10x win. Thread-group widths are tuned per kernel (`MJMLX_TG_KIN/FWD/COLL`).
- **Collision**: the geom-pair table (after `contype/conaffinity/exclude/parent-child` filtering) is baked into the shader source. Pair types: analytic plane contacts, analytic sphere/capsule-box, generic GJK/EPA over a per-type support function (sphere, capsule, ellipsoid, cylinder, box, mesh), SDF-SDF/SDF-mesh via a voxel grid and gradient descent. Heightfield contacts are not implemented in the batched kernel.
- **Solver** (`solver.hpp`, ~60-75% of GPU time): one workgroup per env (18 threads for small models). Contact rows are built in parallel (thread = contact) into pyramidal friction rows (condim 3/4/6); then Newton with an exact line search. The Newton direction is a Cholesky solve in shared memory (`MJMLX_SOLVER_CG=1` restores the old CG). Rows: equalities are not supported; DOF friction loss, joint limits, tendon limits/friction loss, contacts. Early exits: gradient tolerance (`MJMLX_GTOL`), Newton decrease (`MJMLX_NTOL`), line-search derivative tolerance (`MJMLX_LSTOL`).
- **Euler**: integrates directly from the solver's `qacc` (`MJMLX_EULER_EXACT=1` re-solves the mass matrix as before). Implicit/implicitfast damping is folded into the same mechanism as joint damping; there is no RK4.
- **Camera**: separate Vulkan graphics pipeline, built only on `create_camera`.

### Shared memory and its limits

For `nv <= 32` the solver keeps its constraint matrices in workgroup shared memory (`MJMLX_SOLVER_SHARED=0` disables). To fit, the number of constraint rows per env is capped at **72** (`MJMLX_MAX_EFC`, default 256 for larger models); contacts beyond the cap are dropped. Go2 is fine. Occupancy is the dominant factor: adding shared memory reduces concurrent workgroups and runtime grows almost proportionally, so shrinking the footprint is the lever that works.

Subtle pitfall: per-thread early-exit decisions must be uniform across the workgroup or barriers diverge. Models with tendon constraint rows therefore skip the per-thread line-search early exit (it diverged on lavapipe and blew up `qvel`).

## Host side

- **Zero host evaluation inside the substep graph**: no `eval()` / `.data<>()` in the per-step path; model constants live in `ModelCache`, built once at load time (`init_cache`, where eval is allowed). Avoid `mx::zeros` for large per-substep buffers: host-side zero fill was most of the graph build time (~250 us/substep); the contact-free path no longer allocates a `cfrc_ext` zero array.
- **Packed read-back**: the env declares what it reads (`mjmlx_batched_set_pack_spec`); a kernel gathers qpos/qvel/contact force/torque statistics into one buffer, so each control step costs one device-to-host transfer (`read_pack_into` writes straight into a pinned buffer). `mlx_vulkan` allocates VRAM (device-local) and uses `HOST_CACHED` staging for read-back; non-cached staging reads ran at ~200 MB/s.
- **Observation / reward on the training device**: `MjmlxGo2Env.step` computes observation, reward and termination in torch on `device` (fused with `torch.compile` on CUDA; `MJMLX_ENV_COMPILE=0` disables). Only physics I/O and random resets stay on the host.
- **GPU clocks matter**: with no CUDA context the GPU idles near 550 MHz under pure Vulkan compute, while torch activity pulls it to 1140-1840 MHz. Always benchmark with `device=cuda` as in training.

## Load-time model options

`MjmlxModelOptions { self_collision, max_condim }` rewrite the in-memory `mjModel` before compilation, only for geoms of articulated bodies (static terrain is untouched). Default is exact MuJoCo behaviour. `self_collision=0` sets robot geom `conaffinity` to 0 (so robot geoms must have `contype != 0`); `max_condim=3` removes torsional/rolling friction. Pass the same options to the CPU reference when comparing.

## Camera pipeline

- One `vkCmdDrawIndexedIndirect` draws all envs x cameras x geoms. A compute pass performs frustum-sphere culling and LOD selection per (env, cam, geom); LOD meshes are vertex-clustered to 1/4, 1/16, 1/64. `MJMLX_CAM_LOD_K` scales the LOD threshold (default 16 below 128 px height, 4 at >= 128 px; 0 disables).
- A compute `pack` pass reorders the tile atlas into `(env, cam, H, W)` (a `vkCmdCopyImageToBuffer` version cost ~4 ms at 4096 envs).
- On Linux with `VK_KHR_external_memory_fd` the buffers are exported and imported by CUDA (`scripts/mjmlx_camera_cuda.py`), so RGB/depth are CUDA tensors with no CPU copy. Sync is `vkQueueWaitIdle`, not a cross-API semaphore.
- Physics and rendering share one `VkQueue` and both are GPU-bound, so an async pipeline would only hide the render share (2 of 11 ms for front 64 px) at the cost of observation lag; not implemented. Re-running kinematics before rendering costs 0.2-0.3 ms at 4096 envs, so reusing physics outputs was also dropped.
- The camera Vulkan device is injected into mkx at model load (`MJMLX_CAMERA_DEVICE=0` disables). `MJMLX_BUILD_CAMERA=OFF` swaps in `camera_stub.cpp`.
- Spatial tendons are drawn as cylinders from per-(env, tendon) paths computed on the GPU.

## Sensors

`src/sensor.cpp`, batched only. Computed on demand from the step outputs (xpos/xquat/cvel/subtree_com/qpos/qvel) outside the compiled step, so models without `<sensor>` pay nothing. Values correspond to the state at the start of the last step (same as `d->sensordata` after `mj_step`). Supported: gyro/accelerometer/velocimeter/magnetometer, frame*, joint*/ball*, actuatorpos/vel, subtreecom (19 types; see `src/sensor.cpp`). Not yet: force/torque/touch, rangefinder, camera. `scripts/mjmlx_sensors.py` mirrors Isaac Lab's Imu / ContactSensor layout.

## Raycast

`src/raycast.cpp` + `compat/mkx_kernels/raycast.hpp`: a standalone compute kernel, one thread per ray over all envs x rays in one dispatch. Geom poses come from re-running the kinematics kernel on the current `qpos` (0.2-0.3 ms at 4096 envs, SoA `[(g*3+c)*B+env]` read directly); a per-geom table (type, body, mesh id, group, static flag, alpha 0 flag, local AABB, size) and the mesh vertex/face buffers are copied from `mjModel` at creation. Semantics follow `mj_ray` (plane only hit from the +z side, ray inside a primitive returns the exit hit, geoms with alpha 0 are skipped, `bodyexclude` / `flg_static` / `geomgroup`); verified against `mj_ray` in `tests/test_raycast.cpp`. Meshes are tested face by face with a local AABB prefilter (no BVH): mask large visual meshes out with `geomgroup`. hfield and SDF geoms are never hit. Inputs and outputs go through host memory (`mjmlx_raycaster_cast`), which dominates the cost for small ray counts.

## Testing strategy

The GPU engine is checked against MuJoCo C by running both on the same model and comparing trajectories (`test_gpu_vs_cpu`, tolerances per model), plus targeted primitives (`test_batched_collision_primitives`), sensors, SDF, camera vs MuJoCo's renderer, and a robustness sweep (`test_robustness`) over `nv` x {free, limit, contact, tendon, frictionloss}: divergence, env-to-env mismatch, dependence on batch size/position, repeatability. CI uses lavapipe (software Vulkan), so tests there are a subset; real-GPU-only tests carry labels (`models`, `known_issue`). `ctest -L correctness|robust|robust_full|models|known_issue`.

## Known issues and future work

- Raycast does not cover hfield/SDF geoms, has no BVH for meshes and no GPU-resident (zero-copy) I/O yet.
- Not in the batched path: equality constraints, actuator activation dynamics (FILTER/INTEGRATOR/MUSCLE), RK4, heightfield contacts, inverse dynamics, islands/noslip/sleep, differentiable physics (`mjmlx_grad_step` is a stub).
- Constraint rows capped at 72/env on the shared-memory solver path.
- `robust_known_issue_*` tests track configurations that still disagree with MuJoCo C.
- Performance headroom (see [README](README.md#detailed-performance-log)): solver shared-memory instruction count (32 threads/env instead of 18, sparse unit rows for friction loss/limits), host graph build, third-person camera vertex stage (meshlet culling, back-face culling, Hi-Z).
- Camera interop is Linux/CUDA only; macOS (MoltenVK) is unverified.
