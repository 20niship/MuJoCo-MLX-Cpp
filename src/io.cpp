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

// TODO: Phase 1b - port from Python mjmlx._src.io
Model load_model(const char* xml_path) {
  (void)xml_path;
  throw std::runtime_error("not implemented");
}

// TODO: Phase 1b - port from Python mjmlx._src.io
Model load_model_from_string(const char* xml_string) {
  (void)xml_string;
  throw std::runtime_error("not implemented");
}

// TODO: Phase 1b - port from Python mjmlx._src.io
Data make_data(const Model& model) {
  (void)model;
  throw std::runtime_error("not implemented");
}

}  // namespace mjmlx

extern "C" {

MJMLX_API MjmlxModel* mjmlx_load_model(const char* xml_path) {
  try {
    mjmlx::Model m = mjmlx::load_model(xml_path);
    MjmlxModel* out = new MjmlxModel{};
    out->model = std::move(m);
    return out;
  } catch (...) {
    return nullptr;
  }
}

MJMLX_API MjmlxModel* mjmlx_load_model_from_string(const char* xml_string) {
  try {
    mjmlx::Model m = mjmlx::load_model_from_string(xml_string);
    MjmlxModel* out = new MjmlxModel{};
    out->model = std::move(m);
    return out;
  } catch (...) {
    return nullptr;
  }
}

MJMLX_API void mjmlx_free_model(MjmlxModel* model) {
  delete model;
}

MJMLX_API MjmlxModelInfo mjmlx_model_info(const MjmlxModel* model) {
  if (!model) {
    return MjmlxModelInfo{0, 0, 0, 0, 0, 0, 0, 0, 0};
  }
  // TODO: Phase 1b - populate from model
  MjmlxModelInfo info{};
  info.nq = model->model.nq;
  info.nv = model->model.nv;
  info.nu = model->model.nu;
  info.na = model->model.na;
  info.nbody = model->model.nbody;
  info.njnt = model->model.njnt;
  info.ngeom = model->model.ngeom;
  info.nsite = model->model.nsite;
  info.ncon = model->model.ncon;
  return info;
}

MJMLX_API MjmlxData* mjmlx_make_data(const MjmlxModel* model) {
  if (!model) return nullptr;
  try {
    MjmlxData* d = new MjmlxData{};
    d->model_ref = &model->model;
    // TODO: Phase 1b - call mjmlx::make_data(model->model)
    return d;
  } catch (...) {
    return nullptr;
  }
}

MJMLX_API void mjmlx_free_data(MjmlxData* data) {
  delete data;
}

}  // extern "C"
