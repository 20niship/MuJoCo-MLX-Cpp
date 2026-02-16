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

// MuJoCo Backend (mjb) unified C API.
//
// Provides a single physics API that dispatches to either:
//   MJB_BACKEND_CPU  -- MuJoCo C (double-precision, CPU thread pool)
//   MJB_BACKEND_MLX  -- MuJoCo-MLX (float32, Metal GPU via MLX)
//
// All state exchange uses float* pointers. The CPU backend converts
// double <-> float transparently. The MLX backend returns zero-copy
// pointers into unified memory.

#ifndef MJB_H
#define MJB_H

#include "mjb_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Backend lifecycle
// ============================================================

MJB_API MjbBackend* mjb_create_backend(MjbBackendType type);
MJB_API void mjb_free_backend(MjbBackend* backend);
MJB_API MjbBackendType mjb_backend_type(const MjbBackend* backend);

// ============================================================
// Model I/O
// ============================================================

MJB_API MjbModel* mjb_load_model(MjbBackend* b, const char* xml_path);
MJB_API MjbModel* mjb_load_model_filtered(MjbBackend* b, const char* xml_path, int foot_contacts_only);
MJB_API void mjb_free_model(MjbModel* model);

// ============================================================
// Model accessors
// ============================================================

MJB_API MjbModelInfo mjb_model_info(const MjbModel* model);
MJB_API float mjb_model_opt_timestep(const MjbModel* model);
MJB_API void mjb_model_set_opt_timestep(MjbModel* model, float dt);
MJB_API float mjb_model_body_mass(const MjbModel* model, int body_id);
MJB_API int mjb_name2id(const MjbModel* model, int obj_type, const char* name);

// ============================================================
// Data lifecycle
// ============================================================

MJB_API MjbData* mjb_make_data(MjbModel* model);
MJB_API void mjb_free_data(MjbData* data);
MJB_API void mjb_reset_data(MjbModel* model, MjbData* data);

// ============================================================
// Simulation
// ============================================================

MJB_API void mjb_step(MjbModel* model, MjbData* data);
MJB_API void mjb_forward(MjbModel* model, MjbData* data);
MJB_API void mjb_step1(MjbModel* model, MjbData* data);
MJB_API void mjb_step2(MjbModel* model, MjbData* data);
MJB_API void mjb_kinematics(MjbModel* model, MjbData* data);

// ============================================================
// State access (always float*)
// ============================================================

MJB_API void mjb_set_qpos(MjbData* data, const float* qpos, int n);
MJB_API void mjb_set_qvel(MjbData* data, const float* qvel, int n);
MJB_API void mjb_set_ctrl(MjbData* data, const float* ctrl, int n);

MJB_API const float* mjb_get_qpos(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_qvel(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_ctrl(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_xpos(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_xquat(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_xipos(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_cvel(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_qfrc_actuator(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_subtree_com(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_cinert(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_cfrc_ext(const MjbData* data, int* n_out);

// ============================================================
// Batched simulation
// ============================================================

// Create batched simulation.
// CPU backend: internally thread-pools over N mjData* instances.
// MLX backend: uses compiled+vmapped Metal GPU step.
MJB_API MjbBatchedSim* mjb_batched_create(
    MjbModel* model,
    const MjbBatchedConfig* config);

MJB_API void mjb_batched_free(MjbBatchedSim* sim);

// Step all environments. ctrl: float[num_envs * nu].
MJB_API void mjb_batched_step(MjbBatchedSim* sim, const float* ctrl);

// Reset environments where mask[i] != 0.
MJB_API void mjb_batched_reset(MjbBatchedSim* sim, const int* reset_mask);

// Batched state access: float[num_envs * dim].
MJB_API const float* mjb_batched_get_qpos(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_qvel(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_xpos(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_subtree_com(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_cinert(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_cvel(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_qfrc_actuator(const MjbBatchedSim* sim, int* n_out);
MJB_API const float* mjb_batched_get_cfrc_ext(const MjbBatchedSim* sim, int* n_out);

// ============================================================
// Differentiable simulation (MLX backend only)
// ============================================================

// Compute d(next_state)/d(ctrl). Returns -1 if backend doesn't support it.
MJB_API int mjb_grad_step(MjbModel* model, MjbData* data, float* grad_out);

#ifdef __cplusplus
}
#endif

#endif // MJB_H
