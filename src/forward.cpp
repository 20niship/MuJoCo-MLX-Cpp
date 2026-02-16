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

#include <mlx/mlx.h>
#include <stdexcept>

namespace mx = mlx::core;

namespace mjmlx {

// TODO: Phase 1b - port from Python mjmlx._src.forward
Data forward(const Model& m, Data d) {
  (void)m;
  (void)d;
  throw std::runtime_error("not implemented");
}

// TODO: Phase 1b - port from Python mjmlx._src.forward
Data step(const Model& m, Data d) {
  (void)m;
  (void)d;
  throw std::runtime_error("not implemented");
}

}  // namespace mjmlx

extern "C" {

MJMLX_API void mjmlx_forward(const MjmlxModel* model, MjmlxData* data) {
  if (!model || !data) return;
  try {
    data->data = mjmlx::forward(model->model, data->data);
  } catch (...) {
    throw;
  }
}

MJMLX_API void mjmlx_step(const MjmlxModel* model, MjmlxData* data) {
  if (!model || !data) return;
  try {
    data->data = mjmlx::step(model->model, data->data);
  } catch (...) {
    throw;
  }
}

MJMLX_API void mjmlx_set_qpos(MjmlxData* data, const float* qpos, int n) {
  if (!data || !qpos || n <= 0) return;
  data->data.qpos = mx::array(qpos, {n}, mx::float32);
}

MJMLX_API void mjmlx_set_qvel(MjmlxData* data, const float* qvel, int n) {
  if (!data || !qvel || n <= 0) return;
  data->data.qvel = mx::array(qvel, {n}, mx::float32);
}

MJMLX_API void mjmlx_set_ctrl(MjmlxData* data, const float* ctrl, int n) {
  if (!data || !ctrl || n <= 0) return;
  data->data.ctrl = mx::array(ctrl, {n}, mx::float32);
}

// Helper: return data pointer from MLX array (unified memory, zero-copy)
static const float* get_array_ptr(const mx::array& arr, int* n_out) {
  mx::eval(arr);
  if (n_out) *n_out = arr.size();
  return (arr.size() > 0) ? arr.data<float>() : nullptr;
}

MJMLX_API const float* mjmlx_get_qpos(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qpos, n_out);
}

MJMLX_API const float* mjmlx_get_qvel(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qvel, n_out);
}

MJMLX_API const float* mjmlx_get_ctrl(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.ctrl, n_out);
}

MJMLX_API const float* mjmlx_get_xpos(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.xpos, n_out);
}

MJMLX_API const float* mjmlx_get_xquat(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.xquat, n_out);
}

MJMLX_API const float* mjmlx_get_xipos(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.xipos, n_out);
}

MJMLX_API const float* mjmlx_get_cvel(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.cvel, n_out);
}

MJMLX_API const float* mjmlx_get_qfrc_bias(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qfrc_bias, n_out);
}

}  // extern "C"
