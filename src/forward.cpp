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

// ── Forward sub-pipelines ─────────────────────────────────────────────────────

static Data fwd_position(const Model& m, Data d) {
  d = kinematics(m, d);
  d = com_pos(m, d);
  d = crb(m, d);
  d = factor_m(m, d);
  d = collision(m, d);
  d = make_constraint(m, d);
  d = transmission(m, d);
  return d;
}

static Data fwd_velocity(const Model& m, Data d) {
  if (m.nu > 0 && d.actuator_moment.size() > 0) {
    d.actuator_velocity = mx::flatten(mx::matmul(d.actuator_moment, mx::reshape(d.qvel, {m.nv, 1})));
  }
  d = com_vel(m, d);
  d = passive(m, d);
  d = rne(m, d);
  return d;
}

static Data fwd_actuation(const Model& m, Data d) {
  if (m.nu == 0 || (m.opt.disableflags & DisableBit::ACTUATION)) {
    d.qfrc_actuator = mx::zeros({m.nv});
    return d;
  }

  auto ctrl = d.ctrl;

  // Clamp control
  if (!(m.opt.disableflags & DisableBit::CLAMPCTRL) && m.actuator_ctrllimited.size() > 0) {
    mx::eval(m.actuator_ctrllimited); mx::eval(m.actuator_ctrlrange); mx::eval(ctrl);
    auto lim_ptr = m.actuator_ctrllimited.data<int>();
    for (int i = 0; i < m.nu; i++) {
      if (!lim_ptr[i]) continue;
      auto lo = mx::slice(m.actuator_ctrlrange, {i, 0}, {i + 1, 1});
      auto hi = mx::slice(m.actuator_ctrlrange, {i, 1}, {i + 1, 2});
      auto val = mx::clip(mx::slice(ctrl, {i}, {i + 1}), mx::flatten(lo), mx::flatten(hi));
      auto before = mx::slice(ctrl, {0}, {i});
      auto after = mx::slice(ctrl, {i + 1}, {m.nu});
      ctrl = mx::concatenate({before, val, after}, 0);
    }
  }

  // Compute actuator force (simplified: FIXED gain with AFFINE bias)
  mx::eval(m.actuator_gaintype); mx::eval(m.actuator_gainprm);
  mx::eval(m.actuator_biastype); mx::eval(m.actuator_biasprm);
  auto gaintype_ptr = m.actuator_gaintype.data<int>();
  auto biastype_ptr = m.actuator_biastype.data<int>();

  std::vector<float> force_data(m.nu, 0.0f);
  mx::eval(ctrl); mx::eval(d.actuator_length);
  auto ctrl_ptr = ctrl.data<float>();

  for (int i = 0; i < m.nu; i++) {
    float gain = 0.0f;
    mx::eval(m.actuator_gainprm);
    auto gp = m.actuator_gainprm.data<float>();
    if (gaintype_ptr[i] == static_cast<int>(GainType::FIXED)) {
      gain = gp[i * 10];
    } else {
      gain = gp[i * 10];  // fallback
    }

    float bias = 0.0f;
    if (biastype_ptr[i] == static_cast<int>(BiasType::AFFINE)) {
      mx::eval(m.actuator_biasprm);
      auto bp = m.actuator_biasprm.data<float>();
      mx::eval(d.actuator_length);
      auto len_ptr = d.actuator_length.data<float>();
      bias = bp[i * 10] + bp[i * 10 + 1] * len_ptr[i];
    }

    force_data[i] = gain * ctrl_ptr[i] + bias;
  }

  // Clamp force
  if (m.actuator_forcelimited.size() > 0) {
    mx::eval(m.actuator_forcelimited); mx::eval(m.actuator_forcerange);
    auto flim = m.actuator_forcelimited.data<int>();
    auto frange = m.actuator_forcerange.data<float>();
    for (int i = 0; i < m.nu; i++) {
      if (flim[i]) {
        float lo = frange[i * 2], hi = frange[i * 2 + 1];
        force_data[i] = std::max(lo, std::min(hi, force_data[i]));
      }
    }
  }

  auto force = mx::array(force_data.data(), {m.nu}, mx::float32);
  d.actuator_force = force;

  // Map to joint space: qfrc = moment^T @ force
  if (d.actuator_moment.size() > 0) {
    d.qfrc_actuator = mx::flatten(mx::matmul(mx::transpose(d.actuator_moment),
                                              mx::reshape(force, {m.nu, 1})));
  } else {
    d.qfrc_actuator = mx::zeros({m.nv});
  }

  return d;
}

static Data fwd_acceleration(const Model& m, Data d) {
  auto qfrc_applied = d.qfrc_applied;
  if (d.xfrc_applied.size() > 0) {
    qfrc_applied = mx::add(qfrc_applied, xfrc_accumulate(m, d));
  }
  auto qfrc_smooth = mx::add(mx::subtract(d.qfrc_passive, d.qfrc_bias),
                              mx::add(d.qfrc_actuator, qfrc_applied));
  auto qacc_smooth = solve_m(m, d, qfrc_smooth);
  d.qfrc_smooth = qfrc_smooth;
  d.qacc_smooth = qacc_smooth;
  return d;
}

static Data integrate_euler(const Model& m, Data d) {
  float dt = m.opt.timestep;
  auto qacc = d.qacc;

  // Advance velocity
  auto new_qvel = mx::add(d.qvel, mx::multiply(qacc, mx::array(dt)));

  // Integrate position (simplified: handles free joints with quaternion integration)
  mx::eval(m.jnt_type); mx::eval(m.jnt_qposadr); mx::eval(m.jnt_dofadr);
  auto jnt_type_ptr = m.jnt_type.data<int>();
  auto jnt_qposadr_ptr = m.jnt_qposadr.data<int>();
  auto jnt_dofadr_ptr = m.jnt_dofadr.data<int>();

  std::vector<mx::array> parts;
  for (int j = 0; j < m.njnt; j++) {
    int jt = jnt_type_ptr[j];
    int qa = jnt_qposadr_ptr[j];
    int da = jnt_dofadr_ptr[j];

    if (jt == static_cast<int>(JointType::FREE)) {
      auto pos = mx::add(mx::slice(d.qpos, {qa}, {qa + 3}),
                         mx::multiply(mx::array(dt), mx::slice(new_qvel, {da}, {da + 3})));
      auto quat_new = quat_integrate(mx::slice(d.qpos, {qa + 3}, {qa + 7}),
                                      mx::slice(new_qvel, {da + 3}, {da + 6}), dt);
      parts.push_back(mx::concatenate({pos, quat_new}, 0));
    } else if (jt == static_cast<int>(JointType::BALL)) {
      auto quat_new = quat_integrate(mx::slice(d.qpos, {qa}, {qa + 4}),
                                      mx::slice(new_qvel, {da}, {da + 3}), dt);
      parts.push_back(quat_new);
    } else {
      // HINGE or SLIDE
      auto val = mx::add(mx::slice(d.qpos, {qa}, {qa + 1}),
                         mx::multiply(mx::array(dt), mx::slice(new_qvel, {da}, {da + 1})));
      parts.push_back(val);
    }
  }

  d.qpos = mx::concatenate(parts, 0);
  d.qvel = new_qvel;
  d.qacc_warmstart = d.qacc;
  return d;
}

Data forward(const Model& m, Data d) {
  d = fwd_position(m, d);
  d = fwd_velocity(m, d);
  d = fwd_actuation(m, d);
  d = fwd_acceleration(m, d);

  // Solve constraints
  int nefc_count = d.efc_J.shape(0);
  if (nefc_count == 0) {
    d.qacc = d.qacc_smooth;
    d.qfrc_constraint = mx::zeros({m.nv});
  } else {
    d = solve(m, d);
  }

  return d;
}

Data step(const Model& m, Data d) {
  d = forward(m, d);
  d = integrate_euler(m, d);
  return d;
}

// step1: forward position + velocity + actuation (user can modify ctrl before step2)
Data step1(const Model& m, Data d) {
  d = fwd_position(m, d);
  d = fwd_velocity(m, d);
  d = fwd_actuation(m, d);
  return d;
}

// step2: acceleration + constraint solve + integration
Data step2(const Model& m, Data d) {
  d = fwd_acceleration(m, d);
  int nefc_count = d.efc_J.shape(0);
  if (nefc_count == 0) {
    d.qacc = d.qacc_smooth;
    d.qfrc_constraint = mx::zeros({m.nv});
  } else {
    d = solve(m, d);
  }
  d = integrate_euler(m, d);
  return d;
}

// rne_post_constraint: compute per-body contact forces from constraint forces.
Data rne_post_constraint(const Model& m, Data d) {
  int nbody = m.nbody;
  int nv = m.nv;
  d.cfrc_ext = mx::zeros({nbody, 6});

  if (d.nefc == 0 || d.efc_J.size() == 0 || d.efc_force.size() == 0) {
    return d;
  }

  m.init_cache();
  mx::eval(d.qfrc_constraint);
  mx::eval(d.cdof);
  auto qfrc_ptr = d.qfrc_constraint.data<float>();
  auto cdof_ptr = d.cdof.data<float>();

  std::vector<float> cfrc_data(nbody * 6, 0.0f);
  for (int di = 0; di < nv; di++) {
    float f = qfrc_ptr[di];
    if (f == 0.0f) continue;
    int bid = m.cache.dof_bodyid_vec[di];
    for (int k = 0; k < 6; k++) {
      cfrc_data[bid * 6 + k] += cdof_ptr[di * 6 + k] * f;
    }
  }
  d.cfrc_ext = mx::array(cfrc_data.data(), {nbody, 6}, mx::float32);
  return d;
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

MJMLX_API const float* mjmlx_get_qacc(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qacc, n_out);
}

MJMLX_API const float* mjmlx_get_qfrc_constraint(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qfrc_constraint, n_out);
}

MJMLX_API const float* mjmlx_get_qfrc_actuator(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qfrc_actuator, n_out);
}

MJMLX_API const float* mjmlx_get_qfrc_passive(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qfrc_passive, n_out);
}

MJMLX_API const float* mjmlx_get_qfrc_smooth(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qfrc_smooth, n_out);
}

MJMLX_API const float* mjmlx_get_qacc_smooth(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.qacc_smooth, n_out);
}

MJMLX_API const float* mjmlx_get_subtree_com(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.subtree_com, n_out);
}

MJMLX_API const float* mjmlx_get_cinert(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.cinert, n_out);
}

MJMLX_API const float* mjmlx_get_cfrc_ext(const MjmlxData* data, int* n_out) {
  if (!data) { if (n_out) *n_out = 0; return nullptr; }
  return get_array_ptr(data->data.cfrc_ext, n_out);
}

MJMLX_API int mjmlx_get_ncon(const MjmlxData* data) {
  if (!data) return 0;
  return data->data.ncon;
}

MJMLX_API int mjmlx_get_nefc(const MjmlxData* data) {
  if (!data) return 0;
  return data->data.nefc;
}

MJMLX_API void mjmlx_step1(const MjmlxModel* model, MjmlxData* data) {
  if (!model || !data) return;
  try {
    data->data = mjmlx::step1(model->model, data->data);
  } catch (...) {
    throw;
  }
}

MJMLX_API void mjmlx_step2(const MjmlxModel* model, MjmlxData* data) {
  if (!model || !data) return;
  try {
    data->data = mjmlx::step2(model->model, data->data);
  } catch (...) {
    throw;
  }
}

MJMLX_API void mjmlx_kinematics(const MjmlxModel* model, MjmlxData* data) {
  if (!model || !data) return;
  try {
    data->data = mjmlx::kinematics(model->model, data->data);
  } catch (...) {
    throw;
  }
}

MJMLX_API void mjmlx_rne_post_constraint(const MjmlxModel* model, MjmlxData* data) {
  if (!model || !data) return;
  try {
    data->data = mjmlx::rne_post_constraint(model->model, data->data);
  } catch (...) {
    throw;
  }
}

}  // extern "C"
