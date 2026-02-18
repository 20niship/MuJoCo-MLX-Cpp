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

// Public type definitions for the MuJoCo Backend (mjb) unified C API.
// This API provides a single interface that dispatches to either:
//   - MuJoCo C (CPU, double-precision)
//   - MuJoCo-MLX (Metal GPU, float32)

#ifndef MJB_TYPES_H
#define MJB_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define MJB_API __declspec(dllexport)
#else
#define MJB_API __attribute__((visibility("default")))
#endif

// Backend types
typedef enum {
    MJB_BACKEND_CPU = 0,  // MuJoCo C (double, CPU threads)
    MJB_BACKEND_MLX = 1,  // MuJoCo-MLX (float32, Metal GPU)
} MjbBackendType;

// Opaque handles
typedef struct MjbBackend MjbBackend;
typedef struct MjbModel MjbModel;
typedef struct MjbData MjbData;
typedef struct MjbBatchedSim MjbBatchedSim;

// Model dimensions
typedef struct {
    int nq;           // generalized coordinates
    int nv;           // degrees of freedom
    int nu;           // actuators
    int nbody;        // bodies
    int njnt;         // joints
    int ngeom;        // geoms
    int nsite;        // sites
    int nmocap;       // mocap bodies
    int ntendon;      // tendons
    int nsensor;      // sensors
    int nsensordata;  // sensor data values
    int neq;          // equality constraints
} MjbModelInfo;

// Batched simulation config
typedef struct {
    int num_envs;
    int foot_contacts_only;   // 1 = filter to foot-floor contacts
    int solver_iterations;    // 0 = model default, >0 = override
} MjbBatchedConfig;

#ifdef __cplusplus
}
#endif

#endif // MJB_TYPES_H
