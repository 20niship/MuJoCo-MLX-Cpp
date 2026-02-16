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

// Public type definitions for MuJoCo-MLX C API.
// All types are opaque handles -- internal state lives in C++.

#ifndef MJMLX_TYPES_H
#define MJMLX_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define MJMLX_API __declspec(dllexport)
#else
#define MJMLX_API __attribute__((visibility("default")))
#endif

// Opaque handles
typedef struct MjmlxModel MjmlxModel;
typedef struct MjmlxData MjmlxData;
typedef struct MjmlxBatchedSim MjmlxBatchedSim;
typedef struct MjmlxActorCritic MjmlxActorCritic;
typedef struct MjmlxPPOTrainer MjmlxPPOTrainer;

// Integrator types (matches MuJoCo mjtIntegrator)
typedef enum {
    MJMLX_INTEGRATOR_EULER = 0,
    MJMLX_INTEGRATOR_RK4 = 1,
    MJMLX_INTEGRATOR_IMPLICIT = 2,
    MJMLX_INTEGRATOR_IMPLICITFAST = 3,
} MjmlxIntegrator;

// Solver types (matches MuJoCo mjtSolver)
typedef enum {
    MJMLX_SOLVER_PGS = 0,
    MJMLX_SOLVER_CG = 1,
    MJMLX_SOLVER_NEWTON = 2,
} MjmlxSolver;

// Disable flags (matches MuJoCo mjtDisableBit)
typedef enum {
    MJMLX_DISABLE_CONSTRAINT = 1 << 0,
    MJMLX_DISABLE_EQUALITY = 1 << 1,
    MJMLX_DISABLE_FRICTIONLOSS = 1 << 2,
    MJMLX_DISABLE_LIMIT = 1 << 3,
    MJMLX_DISABLE_CONTACT = 1 << 4,
    MJMLX_DISABLE_PASSIVE = 1 << 5,
    MJMLX_DISABLE_GRAVITY = 1 << 6,
    MJMLX_DISABLE_CLAMPCTRL = 1 << 7,
    MJMLX_DISABLE_WARMSTART = 1 << 8,
    MJMLX_DISABLE_FILTERPARENT = 1 << 9,
    MJMLX_DISABLE_ACTUATION = 1 << 10,
} MjmlxDisableFlag;

// Model dimensions (returned by mjmlx_model_info)
typedef struct {
    int nq;       // generalized coordinates
    int nv;       // degrees of freedom
    int nu;       // actuators
    int na;       // activations
    int nbody;    // bodies
    int njnt;     // joints
    int ngeom;    // geoms
    int nsite;    // sites
    int ncon;     // max contacts
} MjmlxModelInfo;

// Batched simulation config
typedef struct {
    int num_envs;                     // number of parallel environments
    int foot_contacts_only;           // 1 = filter to foot contacts (faster)
    MjmlxIntegrator integrator;       // integration method
    int use_gpu;                      // 1 = use Metal GPU, 0 = CPU
    int solver_iterations;            // 0 = use model default, >0 = override solver iterations
} MjmlxBatchedConfig;

// Actor-critic network config
typedef struct {
    int obs_dim;
    int act_dim;
    int hidden_sizes[4];    // up to 4 hidden layers (0-terminated)
    float init_log_std;     // initial log standard deviation
} MjmlxActorCriticConfig;

// PPO hyperparameters
typedef struct {
    float learning_rate;
    float gamma;
    float gae_lambda;
    float clip_coef;
    float vf_coef;
    float ent_coef;
    float max_grad_norm;
    float target_kl;
    int num_steps;          // rollout length
    int num_minibatches;
    int update_epochs;
    int anneal_lr;          // 1 = anneal learning rate
} MjmlxPPOConfig;

// Rollout step result (per environment)
typedef struct {
    float reward;
    int terminated;
    int truncated;
} MjmlxStepResult;

#ifdef __cplusplus
}
#endif

#endif // MJMLX_TYPES_H
