// Copyright 2026 Arghya Sur
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MuJoCo-MLX public C API.
//
// GPU-accelerated MuJoCo physics via Apple MLX.
// Provides single-env and batched (compile+vmap) simulation,
// and differentiable physics.
//
// All handles are opaque. Data exchange uses float* pointers
// into MLX unified memory (zero-copy on Apple Silicon).

#ifndef MJMLX_H
#define MJMLX_H

#include "mjmlx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Library info
// ============================================================

MJMLX_API const char* mjmlx_version(void);

// ============================================================
// Model I/O
// ============================================================

// Load model from MJCF XML file. Returns NULL on error.
MJMLX_API MjmlxModel* mjmlx_load_model(const char* xml_path);

// Load model from MJCF XML string. Returns NULL on error.
MJMLX_API MjmlxModel* mjmlx_load_model_from_string(const char* xml_string);

// Load model with contact filtering applied. foot_contacts_only=1 keeps only
// foot-floor contacts (reduces collision pairs from ~126 to 2 for humanoid).
MJMLX_API MjmlxModel* mjmlx_load_model_filtered(const char* xml_path, int foot_contacts_only);

// Free model and all associated memory.
MJMLX_API void mjmlx_free_model(MjmlxModel* model);

// Get model dimensions.
MJMLX_API MjmlxModelInfo mjmlx_model_info(const MjmlxModel* model);

// ============================================================
// Model field accessors (MuJoCo C conformance)
// ============================================================

// Get simulation timestep (model.opt.timestep).
MJMLX_API float mjmlx_model_opt_timestep(const MjmlxModel* model);

// Set simulation timestep.
MJMLX_API void mjmlx_model_set_opt_timestep(MjmlxModel* model, float dt);

// Get body mass for a specific body ID.
MJMLX_API float mjmlx_model_body_mass(const MjmlxModel* model, int body_id);

// Name-to-ID lookup (delegates to MuJoCo C mj_name2id).
// obj_type: MuJoCo object type (e.g., mjOBJ_BODY=1, mjOBJ_JOINT=2, mjOBJ_GEOM=5).
// Returns -1 if not found.
MJMLX_API int mjmlx_name2id(const MjmlxModel* model, int obj_type, const char* name);

// Access the underlying MuJoCo C mjModel pointer (for model field accessors).
// Returns a const mjModel* cast to void*. Valid for the lifetime of the MjmlxModel.
MJMLX_API const void* mjmlx_get_mj_model(const MjmlxModel* model);

// ============================================================
// Single-environment simulation
// ============================================================

// Create simulation data for a single environment.
MJMLX_API MjmlxData* mjmlx_make_data(const MjmlxModel* model);

// Free simulation data.
MJMLX_API void mjmlx_free_data(MjmlxData* data);

// Reset data to initial state (qpos0, zero velocities). Analogous to mj_resetData.
MJMLX_API void mjmlx_reset_data(const MjmlxModel* model, MjmlxData* data);

// Run full forward kinematics + dynamics.
MJMLX_API void mjmlx_forward(const MjmlxModel* model, MjmlxData* data);

// Advance simulation by one timestep.
MJMLX_API void mjmlx_step(const MjmlxModel* model, MjmlxData* data);

// Split step: compute position, velocity, actuation (allows ctrl modification between).
// Analogous to mj_step1 in MuJoCo C.
MJMLX_API void mjmlx_step1(const MjmlxModel* model, MjmlxData* data);

// Split step: compute acceleration, solve constraints, integrate.
// Analogous to mj_step2 in MuJoCo C.
MJMLX_API void mjmlx_step2(const MjmlxModel* model, MjmlxData* data);

// Compute forward kinematics only (position-dependent quantities).
// Analogous to mj_kinematics in MuJoCo C.
MJMLX_API void mjmlx_kinematics(const MjmlxModel* model, MjmlxData* data);

// Set generalized coordinates. qpos must have model.nq elements.
MJMLX_API void mjmlx_set_qpos(MjmlxData* data, const float* qpos, int n);

// Set generalized velocities. qvel must have model.nv elements.
MJMLX_API void mjmlx_set_qvel(MjmlxData* data, const float* qvel, int n);

// Set actuator controls. ctrl must have model.nu elements.
MJMLX_API void mjmlx_set_ctrl(MjmlxData* data, const float* ctrl, int n);

// Get state pointers (zero-copy unified memory on Apple Silicon).
// These point into MLX array storage -- valid until next step/forward.
MJMLX_API const float* mjmlx_get_qpos(const MjmlxData* data, int* n_out);
MJMLX_API const float* mjmlx_get_qvel(const MjmlxData* data, int* n_out);
MJMLX_API const float* mjmlx_get_ctrl(const MjmlxData* data, int* n_out);

// Get derived quantities (computed by forward/step).
MJMLX_API const float* mjmlx_get_xpos(const MjmlxData* data, int* n_out);   // body positions [nbody*3]
MJMLX_API const float* mjmlx_get_xquat(const MjmlxData* data, int* n_out);  // body quaternions [nbody*4]
MJMLX_API const float* mjmlx_get_xipos(const MjmlxData* data, int* n_out);  // body COM positions [nbody*3]
MJMLX_API const float* mjmlx_get_cvel(const MjmlxData* data, int* n_out);   // body COM velocities [nbody*6]
MJMLX_API const float* mjmlx_get_qfrc_bias(const MjmlxData* data, int* n_out); // Coriolis+gravity [nv]
MJMLX_API const float* mjmlx_get_qacc(const MjmlxData* data, int* n_out);     // acceleration [nv]
MJMLX_API const float* mjmlx_get_qfrc_constraint(const MjmlxData* data, int* n_out); // constraint forces [nv]
MJMLX_API const float* mjmlx_get_qfrc_actuator(const MjmlxData* data, int* n_out);   // actuator forces [nv]
MJMLX_API const float* mjmlx_get_qfrc_passive(const MjmlxData* data, int* n_out);    // passive forces [nv]
MJMLX_API const float* mjmlx_get_qfrc_gravcomp(const MjmlxData* data, int* n_out);  // gravity compensation [nv]
MJMLX_API const float* mjmlx_get_qfrc_smooth(const MjmlxData* data, int* n_out);     // smooth forces [nv]
MJMLX_API const float* mjmlx_get_qacc_smooth(const MjmlxData* data, int* n_out);     // smooth acceleration [nv]
MJMLX_API const float* mjmlx_get_subtree_com(const MjmlxData* data, int* n_out);     // subtree COM [nbody*3]
MJMLX_API const float* mjmlx_get_cinert(const MjmlxData* data, int* n_out);          // body inertias [nbody*10]
MJMLX_API const float* mjmlx_get_cfrc_ext(const MjmlxData* data, int* n_out);        // external contact forces [nbody*6]
MJMLX_API int mjmlx_get_ncon(const MjmlxData* data);                                 // number of contacts
MJMLX_API int mjmlx_get_nefc(const MjmlxData* data);                                 // constraint rows

// Compute cfrc_ext (per-body contact forces) from constraint forces.
// Analogous to mj_rnePostConstraint in MuJoCo C.
MJMLX_API void mjmlx_rne_post_constraint(const MjmlxModel* model, MjmlxData* data);

// ============================================================
// Batched simulation (compile + vmap -- Metal GPU)
// ============================================================

// Create batched simulation. Compiles and vmaps the step function.
MJMLX_API MjmlxBatchedSim* mjmlx_batched_create(
    const MjmlxModel* model,
    const MjmlxBatchedConfig* config);

// Free batched simulation.
MJMLX_API void mjmlx_batched_free(MjmlxBatchedSim* sim);

// Step all environments.
// ctrl: float[num_envs * nu] -- actuator controls for all envs
// After stepping, use mjmlx_batched_get_* to read state.
MJMLX_API void mjmlx_batched_step(
    MjmlxBatchedSim* sim,
    const float* ctrl);

// Reset environments where mask[i] != 0.
// mask: int[num_envs] -- nonzero = reset that env to initial state
MJMLX_API void mjmlx_batched_reset(
    MjmlxBatchedSim* sim,
    const int* reset_mask);

// Copies qpos+qvel out in one eval()/sync; cheaper than get_qpos+get_qvel since each mjmlx_batched_get_* call is its own Vulkan submit+wait on MKX.
MJMLX_API void mjmlx_batched_get_state(
    const MjmlxBatchedSim* sim, float* qpos_out, float* qvel_out,
    int* nq_out, int* nv_out);

// Overwrites the full qpos/qvel batch in one array rebuild each; prefer over looping mjmlx_batched_set_env_qpos/qvel per-env, which rebuilds+re-evals the whole B*nq/B*nv array on every call.
MJMLX_API void mjmlx_batched_set_state(
    MjmlxBatchedSim* sim, const float* qpos, const float* qvel);

// Get batched state pointers (zero-copy into MLX unified memory).
// Returns float[num_envs * dim] for each quantity.
MJMLX_API const float* mjmlx_batched_get_qpos(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_qvel(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_xpos(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_subtree_com(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_cinert(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_cvel(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_qfrc_actuator(const MjmlxBatchedSim* sim, int* n_out);
MJMLX_API const float* mjmlx_batched_get_cfrc_ext(const MjmlxBatchedSim* sim, int* n_out);

// Per-env state setters (for custom RSI resets with noise).
MJMLX_API void mjmlx_batched_set_env_qpos(MjmlxBatchedSim* sim, int env_idx, const float* qpos, int nq);
MJMLX_API void mjmlx_batched_set_env_qvel(MjmlxBatchedSim* sim, int env_idx, const float* qvel, int nv);

// Evaluate qpos + qvel in a single GPU fence. Call before get_qpos/get_qvel to
// avoid two sequential GPU syncs.
MJMLX_API void mjmlx_batched_eval_state(const MjmlxBatchedSim* sim);

// ============================================================
// Differentiable simulation (for empowerment, model-based RL)
// ============================================================

// Compute d(next_state)/d(ctrl) for the current state.
// Gradient is with respect to actuator controls.
// grad_out: float[nv * nu] -- Jacobian matrix (row-major)
MJMLX_API void mjmlx_grad_step(
    const MjmlxModel* model,
    const MjmlxData* data,
    float* grad_out);

// Batched gradient: compute Jacobians for all environments.
// grad_out: float[num_envs * nv * nu]
MJMLX_API void mjmlx_batched_grad_step(
    MjmlxBatchedSim* sim,
    float* grad_out);

#ifdef __cplusplus
}
#endif

#endif // MJMLX_H
