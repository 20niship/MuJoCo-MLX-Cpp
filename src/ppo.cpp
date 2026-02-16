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

MJMLX_API MjmlxPPOTrainer* mjmlx_ppo_create(
    MjmlxBatchedSim* sim,
    MjmlxActorCritic* nn,
    const MjmlxPPOConfig* config) {
  if (!sim || !nn || !config) return nullptr;
  MjmlxPPOTrainer* trainer = new MjmlxPPOTrainer{};
  trainer->trainer.sim = &sim->sim;
  trainer->trainer.nn = &nn->nn;
  trainer->trainer.config = *config;
  // TODO: Phase 1b - initialize PPO trainer
  return trainer;
}

MJMLX_API void mjmlx_ppo_free(MjmlxPPOTrainer* trainer) {
  delete trainer;
}

MJMLX_API float mjmlx_ppo_iterate(MjmlxPPOTrainer* trainer) {
  (void)trainer;
  // TODO: Phase 1b - rollout + PPO update
  return 0.0f;  // stub: average reward
}

MJMLX_API float mjmlx_ppo_policy_loss(const MjmlxPPOTrainer* trainer) {
  if (!trainer) return 0.0f;
  return trainer->trainer.last_policy_loss;
}

MJMLX_API float mjmlx_ppo_value_loss(const MjmlxPPOTrainer* trainer) {
  if (!trainer) return 0.0f;
  return trainer->trainer.last_value_loss;
}

MJMLX_API float mjmlx_ppo_entropy(const MjmlxPPOTrainer* trainer) {
  if (!trainer) return 0.0f;
  return trainer->trainer.last_entropy;
}

MJMLX_API float mjmlx_ppo_approx_kl(const MjmlxPPOTrainer* trainer) {
  if (!trainer) return 0.0f;
  return trainer->trainer.last_approx_kl;
}

MJMLX_API int mjmlx_ppo_global_step(const MjmlxPPOTrainer* trainer) {
  if (!trainer) return 0;
  return trainer->trainer.global_step;
}

}  // extern "C"
