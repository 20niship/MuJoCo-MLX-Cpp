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
// differentiable physics, and actor-critic neural networks.
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

// Free model and all associated memory.
MJMLX_API void mjmlx_free_model(MjmlxModel* model);

// Get model dimensions.
MJMLX_API MjmlxModelInfo mjmlx_model_info(const MjmlxModel* model);

// ============================================================
// Single-environment simulation
// ============================================================

// Create simulation data for a single environment.
MJMLX_API MjmlxData* mjmlx_make_data(const MjmlxModel* model);

// Free simulation data.
MJMLX_API void mjmlx_free_data(MjmlxData* data);

// Run full forward kinematics + dynamics.
MJMLX_API void mjmlx_forward(const MjmlxModel* model, MjmlxData* data);

// Advance simulation by one timestep.
MJMLX_API void mjmlx_step(const MjmlxModel* model, MjmlxData* data);

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
MJMLX_API const float* mjmlx_get_qfrc_smooth(const MjmlxData* data, int* n_out);     // smooth forces [nv]
MJMLX_API const float* mjmlx_get_qacc_smooth(const MjmlxData* data, int* n_out);     // smooth acceleration [nv]
MJMLX_API const float* mjmlx_get_subtree_com(const MjmlxData* data, int* n_out);     // subtree COM [nbody*3]
MJMLX_API const float* mjmlx_get_cinert(const MjmlxData* data, int* n_out);          // body inertias [nbody*10]
MJMLX_API int mjmlx_get_ncon(const MjmlxData* data);                                 // number of contacts
MJMLX_API int mjmlx_get_nefc(const MjmlxData* data);                                 // constraint rows

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

// ============================================================
// Neural network (actor-critic via MLX C++)
// ============================================================

// Create actor-critic network with given architecture.
MJMLX_API MjmlxActorCritic* mjmlx_nn_create(const MjmlxActorCriticConfig* config);

// Free actor-critic network.
MJMLX_API void mjmlx_nn_free(MjmlxActorCritic* nn);

// Forward pass: observations -> actions (sampled) + log_probs + values.
// obs:      float[batch * obs_dim]
// actions:  float[batch * act_dim]  (output, sampled from policy)
// logprobs: float[batch]            (output)
// values:   float[batch]            (output)
MJMLX_API void mjmlx_nn_get_action_and_value(
    MjmlxActorCritic* nn,
    const float* obs, int batch,
    float* actions, float* logprobs, float* values);

// Deterministic forward: observations -> mean actions.
MJMLX_API void mjmlx_nn_get_deterministic_action(
    MjmlxActorCritic* nn,
    const float* obs, int batch,
    float* actions);

// Get value only.
MJMLX_API void mjmlx_nn_get_value(
    MjmlxActorCritic* nn,
    const float* obs, int batch,
    float* values);

// Save/load network weights.
MJMLX_API int mjmlx_nn_save(const MjmlxActorCritic* nn, const char* path);
MJMLX_API int mjmlx_nn_load(MjmlxActorCritic* nn, const char* path);

// ============================================================
// PPO Training (full pipeline in C++)
// ============================================================

// Create PPO trainer with batched sim + actor-critic.
MJMLX_API MjmlxPPOTrainer* mjmlx_ppo_create(
    MjmlxBatchedSim* sim,
    MjmlxActorCritic* nn,
    const MjmlxPPOConfig* config);

// Free PPO trainer (does NOT free sim or nn).
MJMLX_API void mjmlx_ppo_free(MjmlxPPOTrainer* trainer);

// Run one rollout iteration: collect numSteps of experience, then PPO update.
// Returns average reward over the rollout.
MJMLX_API float mjmlx_ppo_iterate(MjmlxPPOTrainer* trainer);

// Get training statistics from the last iteration.
MJMLX_API float mjmlx_ppo_policy_loss(const MjmlxPPOTrainer* trainer);
MJMLX_API float mjmlx_ppo_value_loss(const MjmlxPPOTrainer* trainer);
MJMLX_API float mjmlx_ppo_entropy(const MjmlxPPOTrainer* trainer);
MJMLX_API float mjmlx_ppo_approx_kl(const MjmlxPPOTrainer* trainer);
MJMLX_API int mjmlx_ppo_global_step(const MjmlxPPOTrainer* trainer);

#ifdef __cplusplus
}
#endif

#endif // MJMLX_H
