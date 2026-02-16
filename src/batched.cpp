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

#include "internal.h"
#include "mjmlx/mjmlx.h"

#include <stdexcept>

namespace mjmlx {

// TODO: Phase 1b - port from Python mjmlx._src.batched
std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu) {
  (void)m;
  (void)num_envs;
  (void)use_gpu;
  throw std::runtime_error("not implemented");
}

}  // namespace mjmlx

extern "C" {

MJMLX_API MjmlxBatchedSim* mjmlx_batched_create(
    const MjmlxModel* model,
    const MjmlxBatchedConfig* config) {
  if (!model || !config) return nullptr;
  MjmlxBatchedSim* sim = new MjmlxBatchedSim{};
  sim->sim.model = &model->model;
  sim->sim.num_envs = config->num_envs;
  sim->sim.config = *config;
  // TODO: Phase 1b - call make_batched_step and set compiled_step
  return sim;
}

MJMLX_API void mjmlx_batched_free(MjmlxBatchedSim* sim) {
  delete sim;
}

MJMLX_API void mjmlx_batched_step(MjmlxBatchedSim* sim, const float* ctrl) {
  (void)sim;
  (void)ctrl;
  // TODO: Phase 1b - run compiled step
}

MJMLX_API void mjmlx_batched_reset(MjmlxBatchedSim* sim,
                                    const int* reset_mask) {
  (void)sim;
  (void)reset_mask;
  // TODO: Phase 1b - reset envs where mask != 0
}

MJMLX_API const float* mjmlx_batched_get_qpos(const MjmlxBatchedSim* sim,
                                              int* n_out) {
  if (n_out) *n_out = 0;
  return nullptr;  // TODO: Phase 1b - return pointer into MLX array
}

MJMLX_API const float* mjmlx_batched_get_qvel(const MjmlxBatchedSim* sim,
                                               int* n_out) {
  if (n_out) *n_out = 0;
  return nullptr;  // TODO: Phase 1b - return pointer into MLX array
}

MJMLX_API const float* mjmlx_batched_get_xpos(const MjmlxBatchedSim* sim,
                                               int* n_out) {
  if (n_out) *n_out = 0;
  return nullptr;  // TODO: Phase 1b - return pointer into MLX array
}

MJMLX_API void mjmlx_grad_step(
    const MjmlxModel* model,
    const MjmlxData* data,
    float* grad_out) {
  (void)model;
  (void)data;
  (void)grad_out;
  // TODO: Phase 1b - compute d(next_state)/d(ctrl)
}

MJMLX_API void mjmlx_batched_grad_step(
    MjmlxBatchedSim* sim,
    float* grad_out) {
  (void)sim;
  (void)grad_out;
  // TODO: Phase 1b - batched gradient computation
}

}  // extern "C"
