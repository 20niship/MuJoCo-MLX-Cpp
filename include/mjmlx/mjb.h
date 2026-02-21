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
MJB_API MjbModel* mjb_load_model_from_string(MjbBackend* b, const char* xml_string);
MJB_API void mjb_free_model(MjbModel* model);

// ============================================================
// Model accessors
// ============================================================

MJB_API MjbModelInfo mjb_model_info(const MjbModel* model);
MJB_API float mjb_model_opt_timestep(const MjbModel* model);
MJB_API void mjb_model_set_opt_timestep(MjbModel* model, float dt);
MJB_API float mjb_model_body_mass(const MjbModel* model, int body_id);
MJB_API int mjb_name2id(const MjbModel* model, int obj_type, const char* name);
MJB_API const char* mjb_id2name(const MjbModel* model, int obj_type, int id);
MJB_API int mjb_model_jnt_qposadr(const MjbModel* model, int jnt_id);
MJB_API int mjb_model_jnt_dofadr(const MjbModel* model, int jnt_id);
MJB_API int mjb_model_jnt_type(const MjbModel* model, int jnt_id);
MJB_API int mjb_model_nconmax(const MjbModel* model);
MJB_API int mjb_model_geom_type(const MjbModel* model, int geom_id);

// Per-element model accessors (sensors, mocap, tendons, equality, hfield)
MJB_API int mjb_model_sensor_adr(const MjbModel* model, int sensor_id);
MJB_API int mjb_model_body_mocapid(const MjbModel* model, int body_id);
MJB_API float mjb_model_tendon_width(const MjbModel* model, int tendon_id);
MJB_API int mjb_model_hfield_adr(const MjbModel* model, int hfield_id);

// Bulk model array accessors
MJB_API const float* mjb_model_eq_data(const MjbModel* model, int* n_out);
MJB_API const float* mjb_model_hfield_data(const MjbModel* model, int* n_out);

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
MJB_API void mjb_rne_post_constraint(MjbModel* model, MjbData* data);

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
MJB_API const float* mjb_get_geom_xpos(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_geom_xmat(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_sensordata(const MjbData* data, int* n_out);

// Additional data getters for component binding
MJB_API const float* mjb_get_xaxis(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_site_xpos(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_site_xmat(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_actuator_length(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_actuator_velocity(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_actuator_force(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_mocap_pos(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_mocap_quat(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_ten_length(const MjbData* data, int* n_out);
MJB_API const float* mjb_get_wrap_xpos(const MjbData* data, int* n_out);

// Int data getters (tendon wrapping)
MJB_API const int* mjb_get_ten_wrapadr(const MjbData* data, int* n_out);
MJB_API const int* mjb_get_ten_wrapnum(const MjbData* data, int* n_out);
MJB_API const int* mjb_get_wrap_obj(const MjbData* data, int* n_out);

// Mocap setters
MJB_API void mjb_set_mocap_pos(MjbData* data, const float* pos, int n);
MJB_API void mjb_set_mocap_quat(MjbData* data, const float* quat, int n);

// Per-geom model data (static, for component binding)
MJB_API const float* mjb_model_geom_pos(const MjbModel* model, int* n_out);
MJB_API const float* mjb_model_geom_quat(const MjbModel* model, int* n_out);

// ============================================================
// Per-index state setters (float→double on CPU)
// ============================================================

MJB_API void mjb_set_qpos_at(MjbData* data, int index, float value);
MJB_API void mjb_set_qvel_at(MjbData* data, int index, float value);
MJB_API void mjb_set_ctrl_at(MjbData* data, int index, float value);

// ============================================================
// xfrc_applied (external force/torque per body, 6 floats each)
// ============================================================

MJB_API const float* mjb_get_xfrc_applied(const MjbData* data, int* n_out);
MJB_API void mjb_set_xfrc_applied(MjbData* data, const float* values, int n);

// ============================================================
// Warnings / diagnostics
// ============================================================

// Returns data->warning[index].number (0..mjNWARNING-1).
MJB_API int mjb_get_warning_count(const MjbData* data, int index);

// ============================================================
// Model I/O (save)
// ============================================================

// Save the last compiled model to XML. Returns 0 on success, -1 on error.
MJB_API int mjb_save_last_xml(const MjbModel* model, const char* path,
                              char* error_buf, int error_buf_size);

// ============================================================
// Utility wrappers
// ============================================================

// Compute 6D object velocity (3 rotational + 3 translational).
// result must point to 6 floats. flg_local: 0=global, 1=local frame.
MJB_API void mjb_object_velocity(const MjbModel* model, const MjbData* data,
                                 int objtype, int objid, int flg_local,
                                 float* result6);

// Load a MuJoCo plugin library (.so/.dylib).
MJB_API void mjb_load_plugin_library(const char* path);

// Write to model hfield_data array (already float in MuJoCo).
MJB_API void mjb_model_set_hfield_data(MjbModel* model, int offset,
                                       const float* values, int n);

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

// Returns 1 if the batched sim is actually running Metal GPU kernels,
// 0 if it fell back to CPU thread pool (e.g. nv > 80).
MJB_API int mjb_batched_is_gpu(const MjbBatchedSim* sim);

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

// Per-env state setters (for custom RSI resets with noise).
// CPU backend: writes directly to env's mjData.
// MLX backend: eval + copy + modify row + create new mx::array.
MJB_API void mjb_batched_set_env_qpos(MjbBatchedSim* sim, int env_idx, const float* qpos, int nq);
MJB_API void mjb_batched_set_env_qvel(MjbBatchedSim* sim, int env_idx, const float* qvel, int nv);

// Evaluate qpos + qvel in a single GPU fence before reading state. No-op for CPU backend.
MJB_API void mjb_batched_eval_state(const MjbBatchedSim* sim);

// ============================================================
// Differentiable simulation (MLX backend only)
// ============================================================

// Compute d(next_state)/d(ctrl). Returns -1 if backend doesn't support it.
MJB_API int mjb_grad_step(MjbModel* model, MjbData* data, float* grad_out);

#ifdef __cplusplus
}
#endif

#endif // MJB_H
