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

extern "C" {

MJMLX_API MjmlxActorCritic* mjmlx_nn_create(
    const MjmlxActorCriticConfig* config) {
  if (!config) return nullptr;
  MjmlxActorCritic* nn = new MjmlxActorCritic{};
  nn->nn.config = *config;
  // TODO: Phase 1b - initialize network weights
  return nn;
}

MJMLX_API void mjmlx_nn_free(MjmlxActorCritic* nn) {
  delete nn;
}

MJMLX_API void mjmlx_nn_get_action_and_value(
    MjmlxActorCritic* nn,
    const float* obs, int batch,
    float* actions, float* logprobs, float* values) {
  (void)nn;
  (void)obs;
  (void)batch;
  (void)actions;
  (void)logprobs;
  (void)values;
  // TODO: Phase 1b - forward pass with sampling
}

MJMLX_API void mjmlx_nn_get_deterministic_action(
    MjmlxActorCritic* nn,
    const float* obs, int batch,
    float* actions) {
  (void)nn;
  (void)obs;
  (void)batch;
  (void)actions;
  // TODO: Phase 1b - deterministic forward
}

MJMLX_API void mjmlx_nn_get_value(
    MjmlxActorCritic* nn,
    const float* obs, int batch,
    float* values) {
  (void)nn;
  (void)obs;
  (void)batch;
  (void)values;
  // TODO: Phase 1b - value head only
}

MJMLX_API int mjmlx_nn_save(const MjmlxActorCritic* nn, const char* path) {
  (void)nn;
  (void)path;
  // TODO: Phase 1b - serialize weights to path
  return -1;  // stub: failure
}

MJMLX_API int mjmlx_nn_load(MjmlxActorCritic* nn, const char* path) {
  (void)nn;
  (void)path;
  // TODO: Phase 1b - load weights from path
  return -1;  // stub: failure
}

}  // extern "C"
