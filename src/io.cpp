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

// Model I/O: load MuJoCo C models and convert to MLX arrays.
// Port of Python mjmlx._src.io (put_model, make_data, put_data).

#include "internal.h"
#include <mujoco/mujoco.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mjmlx {

// Helper: convert a C double array to mx::array (float32)
static mx::array to_mx_f(const double* data, int n) {
    std::vector<float> buf(n);
    for (int i = 0; i < n; i++) buf[i] = static_cast<float>(data[i]);
    return mx::array(buf.data(), {n});
}

// Helper: convert a C double array to mx::array with 2D shape
static mx::array to_mx_f2(const double* data, int rows, int cols) {
    int n = rows * cols;
    std::vector<float> buf(n);
    for (int i = 0; i < n; i++) buf[i] = static_cast<float>(data[i]);
    return mx::array(buf.data(), {rows, cols});
}

// Helper: convert a C int array to mx::array (int32)
static mx::array to_mx_i(const int* data, int n) {
    return mx::array(data, {n}, mx::int32);
}

// Helper: convert a C uint8/byte array to mx::array (int32)
static mx::array to_mx_byte(const unsigned char* data, int n) {
    std::vector<int> buf(n);
    for (int i = 0; i < n; i++) buf[i] = static_cast<int>(data[i]);
    return mx::array(buf.data(), {n}, mx::int32);
}

Model load_model(const char* xml_path) {
    char error[1000] = "";
    mjModel* m = mj_loadXML(xml_path, nullptr, error, sizeof(error));
    if (!m) {
        throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    }

    Model model;

    // Counts
    model.nq = m->nq;
    model.nv = m->nv;
    model.nu = m->nu;
    model.na = m->na;
    model.nbody = m->nbody;
    model.njnt = m->njnt;
    model.ngeom = m->ngeom;
    model.nsite = m->nsite;
    model.ncam = m->ncam;
    model.nmocap = m->nmocap;
    model.ntendon = m->ntendon;
    model.neq = m->neq;
    model.npair = m->npair;

    // Options
    model.opt.timestep = static_cast<float>(m->opt.timestep);
    model.opt.tolerance = static_cast<float>(m->opt.tolerance);
    model.opt.ls_tolerance = static_cast<float>(m->opt.ls_tolerance);
    model.opt.iterations = m->opt.iterations;
    model.opt.ls_iterations = m->opt.ls_iterations;
    model.opt.gravity = to_mx_f(m->opt.gravity, 3);
    model.opt.integrator = static_cast<IntegratorType>(m->opt.integrator);
    model.opt.solver = static_cast<SolverType>(m->opt.solver);
    model.opt.disableflags = m->opt.disableflags;
    model.opt.impratio = static_cast<float>(m->opt.impratio);
    model.opt.wind = to_mx_f(m->opt.wind, 3);
    model.opt.viscosity = static_cast<float>(m->opt.viscosity);
    model.opt.density = static_cast<float>(m->opt.density);

    // Statistics
    model.stat.meaninertia = static_cast<float>(m->stat.meaninertia);
    model.stat.meanmass = static_cast<float>(m->stat.meanmass);
    model.stat.meansize = static_cast<float>(m->stat.meansize);
    model.stat.extent = static_cast<float>(m->stat.extent);
    model.stat.center = to_mx_f(m->stat.center, 3);

    // Reference configuration
    model.qpos0 = to_mx_f(m->qpos0, (int)m->nq);
    model.qpos_spring = to_mx_f(m->qpos_spring, (int)m->nq);

    // Body properties
    model.body_parentid = to_mx_i(m->body_parentid, (int)m->nbody);
    model.body_weldid = to_mx_i(m->body_weldid, (int)m->nbody);
    model.body_jntadr = to_mx_i(m->body_jntadr, (int)m->nbody);
    model.body_jntnum = to_mx_i(m->body_jntnum, (int)m->nbody);
    model.body_dofadr = to_mx_i(m->body_dofadr, (int)m->nbody);
    model.body_dofnum = to_mx_i(m->body_dofnum, (int)m->nbody);
    model.body_geomadr = to_mx_i(m->body_geomadr, (int)m->nbody);
    model.body_geomnum = to_mx_i(m->body_geomnum, (int)m->nbody);
    model.body_pos = to_mx_f2(m->body_pos, (int)m->nbody, 3);
    model.body_quat = to_mx_f2(m->body_quat, (int)m->nbody, 4);
    model.body_mass = to_mx_f(m->body_mass, (int)m->nbody);
    model.body_subtreemass = to_mx_f(m->body_subtreemass, (int)m->nbody);
    model.body_inertia = to_mx_f2(m->body_inertia, (int)m->nbody, 3);
    model.body_ipos = to_mx_f2(m->body_ipos, (int)m->nbody, 3);
    model.body_iquat = to_mx_f2(m->body_iquat, (int)m->nbody, 4);
    model.body_invweight0 = to_mx_f2(m->body_invweight0, (int)m->nbody, 2);
    model.body_mocapid = to_mx_i(m->body_mocapid, (int)m->nbody);

    // Compute body root IDs (matching Python logic)
    {
        std::vector<int> rootid(m->nbody, 0);
        for (int i = 1; i < m->nbody; i++) {
            int pid = m->body_parentid[i];
            rootid[i] = (pid == 0) ? i : rootid[pid];
        }
        model.body_rootid = mx::array(rootid.data(), {(int)m->nbody}, mx::int32);
    }

    // Joint properties
    if (m->njnt > 0) {
        model.jnt_type = to_mx_i(m->jnt_type, (int)m->njnt);
        model.jnt_bodyid = to_mx_i(m->jnt_bodyid, (int)m->njnt);
        model.jnt_qposadr = to_mx_i(m->jnt_qposadr, (int)m->njnt);
        model.jnt_dofadr = to_mx_i(m->jnt_dofadr, (int)m->njnt);
        model.jnt_range = to_mx_f2(m->jnt_range, (int)m->njnt, 2);
        model.jnt_limited = to_mx_byte(m->jnt_limited, (int)m->njnt);
        model.jnt_axis = to_mx_f2(m->jnt_axis, (int)m->njnt, 3);
        model.jnt_pos = to_mx_f2(m->jnt_pos, (int)m->njnt, 3);
        model.jnt_stiffness = to_mx_f(m->jnt_stiffness, (int)m->njnt);
        model.jnt_margin = to_mx_f(m->jnt_margin, (int)m->njnt);
        model.jnt_solref = to_mx_f2(m->jnt_solref, (int)m->njnt, 2);
        model.jnt_solimp = to_mx_f2(m->jnt_solimp, (int)m->njnt, 5);
    }

    // DOF properties
    if (m->nv > 0) {
        model.dof_bodyid = to_mx_i(m->dof_bodyid, (int)m->nv);
        model.dof_jntid = to_mx_i(m->dof_jntid, (int)m->nv);
        model.dof_parentid = to_mx_i(m->dof_parentid, (int)m->nv);
        model.dof_Madr = to_mx_i(m->dof_Madr, (int)m->nv);
        model.dof_armature = to_mx_f(m->dof_armature, (int)m->nv);
        model.dof_damping = to_mx_f(m->dof_damping, (int)m->nv);
        model.dof_invweight0 = to_mx_f(m->dof_invweight0, (int)m->nv);
        model.dof_frictionloss = to_mx_f(m->dof_frictionloss, (int)m->nv);
    }

    // Site properties
    if (m->nsite > 0) {
        model.site_bodyid = to_mx_i(m->site_bodyid, (int)m->nsite);
        model.site_pos = to_mx_f2(m->site_pos, (int)m->nsite, 3);
        model.site_quat = to_mx_f2(m->site_quat, (int)m->nsite, 4);
    }

    // Actuator properties
    if (m->nu > 0) {
        model.actuator_trntype = to_mx_i(m->actuator_trntype, (int)m->nu);
        model.actuator_trnid = to_mx_i((const int*)m->actuator_trnid, (int)(m->nu * 2));
        model.actuator_trnid = mx::reshape(model.actuator_trnid, {(int)m->nu, 2});
        model.actuator_gaintype = to_mx_i(m->actuator_gaintype, (int)m->nu);
        model.actuator_gainprm = to_mx_f2(m->actuator_gainprm, (int)m->nu, 10);
        model.actuator_biastype = to_mx_i(m->actuator_biastype, (int)m->nu);
        model.actuator_biasprm = to_mx_f2(m->actuator_biasprm, (int)m->nu, 10);
        model.actuator_dyntype = to_mx_i(m->actuator_dyntype, (int)m->nu);
        model.actuator_dynprm = to_mx_f2(m->actuator_dynprm, (int)m->nu, 10);
        model.actuator_gear = to_mx_f2(m->actuator_gear, (int)m->nu, 6);
        model.actuator_ctrllimited = to_mx_byte(m->actuator_ctrllimited, (int)m->nu);
        model.actuator_ctrlrange = to_mx_f2(m->actuator_ctrlrange, (int)m->nu, 2);
        model.actuator_forcelimited = to_mx_byte(m->actuator_forcelimited, (int)m->nu);
        model.actuator_forcerange = to_mx_f2(m->actuator_forcerange, (int)m->nu, 2);
    }

    // Geom properties (for collision)
    if (m->ngeom > 0) {
        model.geom_type = to_mx_i(m->geom_type, (int)m->ngeom);
        model.geom_bodyid = to_mx_i(m->geom_bodyid, (int)m->ngeom);
        model.geom_pos = to_mx_f2(m->geom_pos, (int)m->ngeom, 3);
        model.geom_quat = to_mx_f2(m->geom_quat, (int)m->ngeom, 4);
        model.geom_size = to_mx_f2(m->geom_size, (int)m->ngeom, 3);
        model.geom_friction = to_mx_f2(m->geom_friction, (int)m->ngeom, 3);
        model.geom_solmix = to_mx_f(m->geom_solmix, (int)m->ngeom);
        model.geom_solref = to_mx_f2(m->geom_solref, (int)m->ngeom, 2);
        model.geom_solimp = to_mx_f2(m->geom_solimp, (int)m->ngeom, 5);
        model.geom_margin = to_mx_f(m->geom_margin, (int)m->ngeom);
        model.geom_gap = to_mx_f(m->geom_gap, (int)m->ngeom);
        model.geom_contype = to_mx_i(m->geom_contype, (int)m->ngeom);
        model.geom_conaffinity = to_mx_i(m->geom_conaffinity, (int)m->ngeom);
        model.geom_condim = to_mx_i(m->geom_condim, (int)m->ngeom);
    }

    // Explicit contact pairs
    if (m->npair > 0) {
        model.pair_geom1 = to_mx_i(m->pair_geom1, (int)m->npair);
        model.pair_geom2 = to_mx_i(m->pair_geom2, (int)m->npair);
        model.pair_dim = to_mx_i(m->pair_dim, (int)m->npair);
        model.pair_margin = to_mx_f(m->pair_margin, (int)m->npair);
        model.pair_gap = to_mx_f(m->pair_gap, (int)m->npair);
        model.pair_solref = to_mx_f2(m->pair_solref, (int)m->npair, 2);
        model.pair_solimp = to_mx_f2(m->pair_solimp, (int)m->npair, 5);
        model.pair_friction = to_mx_f2(m->pair_friction, (int)m->npair, 5);
    }

    // Equality constraints
    if (m->neq > 0) {
        model.eq_type = to_mx_i(m->eq_type, (int)m->neq);
        model.eq_obj1id = to_mx_i(m->eq_obj1id, (int)m->neq);
        model.eq_obj2id = to_mx_i(m->eq_obj2id, (int)m->neq);
        model.eq_data = to_mx_f2(m->eq_data, (int)m->neq, 11);
        model.eq_solref = to_mx_f2(m->eq_solref, (int)m->neq, 2);
        model.eq_solimp = to_mx_f2(m->eq_solimp, (int)m->neq, 5);
    }

    // Eval all arrays to materialize them
    // (MLX is lazy -- this ensures everything is in GPU memory)
    // We skip this for now since model arrays are typically small

    mj_deleteModel(m);
    return model;
}

Model load_model_from_string(const char* xml_string) {
    char error[1000] = "";
    mjModel* m = mj_loadXML(nullptr, nullptr, error, sizeof(error));
    // mj_loadXML doesn't support string loading directly.
    // We need to use mj_loadXML with a VFS or write to a temp file.
    // For now, throw not implemented.
    throw std::runtime_error("load_model_from_string not yet implemented -- use load_model with a file path");
}

Data make_data(const Model& model) {
    Data d;

    int nq = model.nq, nv = model.nv, nu = model.nu, na = model.na;
    int nbody = model.nbody;

    // State: copy qpos0 as initial position
    d.qpos = (model.qpos0.size() > 0) ? mx::array(model.qpos0) : mx::zeros({nq});
    d.qvel = mx::zeros({nv});
    d.qacc = mx::zeros({nv});
    d.ctrl = mx::zeros({nu > 0 ? nu : 1});
    if (nu == 0) d.ctrl = mx::array({});
    d.act = (na > 0) ? mx::zeros({na}) : mx::array({});

    // Kinematics
    d.xpos = mx::zeros({nbody, 3});
    auto ident_quat = mx::array({1.0f, 0.0f, 0.0f, 0.0f});
    d.xquat = mx::tile(ident_quat, {nbody, 1});
    d.xmat = mx::tile(mx::eye(3), {nbody, 1, 1});
    d.xipos = mx::zeros({nbody, 3});
    d.ximat = mx::tile(mx::eye(3), {nbody, 1, 1});
    if (model.ngeom > 0) {
        d.geom_xpos = mx::zeros({model.ngeom, 3});
        d.geom_xmat = mx::tile(mx::eye(3), {model.ngeom, 1, 1});
    }
    if (model.nsite > 0) {
        d.site_xpos = mx::zeros({model.nsite, 3});
        d.site_xmat = mx::tile(mx::eye(3), {model.nsite, 1, 1});
    }

    // Dynamics
    d.subtree_com = mx::zeros({nbody, 3});
    d.cinert = mx::zeros({nbody, 10});
    d.crb = mx::zeros({nbody, 10});
    d.cvel = mx::zeros({nbody, 6});
    d.cdof = mx::zeros({nv, 6});
    d.cdof_dot = mx::zeros({nv, 6});
    d.qM = (nv > 0) ? mx::eye(nv) : mx::zeros({0, 0});
    d.qfrc_bias = mx::zeros({nv});
    d.qfrc_passive = mx::zeros({nv});
    d.qfrc_actuator = mx::zeros({nv});
    d.qfrc_gravcomp = mx::zeros({nv});
    d.qfrc_smooth = mx::zeros({nv});
    d.qacc_smooth = mx::zeros({nv});

    // Applied forces
    d.xfrc_applied = mx::zeros({nbody, 6});
    d.qfrc_applied = mx::zeros({nv});

    // Actuator
    d.actuator_length = mx::zeros({nu});
    d.actuator_moment = mx::zeros({nu, nv});
    d.actuator_velocity = mx::zeros({nu});
    d.actuator_force = mx::zeros({nu});
    d.act_dot = mx::zeros({model.na});

    // Solver
    d.qfrc_constraint = mx::zeros({nv});
    d.qacc_warmstart = mx::zeros({nv});

    // Constraint
    d.efc_J = mx::zeros({0, nv});
    d.efc_D = mx::zeros({0});
    d.efc_aref = mx::zeros({0});
    d.efc_force = mx::zeros({0});
    d.efc_frictionloss = mx::zeros({0});
    d.nefc = 0;
    d.ne = 0; d.nf = 0; d.nl = 0;
    d.ncon = 0;
    d.time = mx::array(0.0f);

    return d;
}

} // namespace mjmlx

// ── C API wrappers ───────────────────────────────────────────

extern "C" {

MJMLX_API MjmlxModel* mjmlx_load_model(const char* xml_path) {
    try {
        auto* handle = new MjmlxModel();
        handle->model = mjmlx::load_model(xml_path);
        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_load_model error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API MjmlxModel* mjmlx_load_model_from_string(const char* xml_string) {
    try {
        auto* handle = new MjmlxModel();
        handle->model = mjmlx::load_model_from_string(xml_string);
        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_load_model_from_string error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API void mjmlx_free_model(MjmlxModel* model) {
    delete model;
}

MJMLX_API MjmlxModelInfo mjmlx_model_info(const MjmlxModel* model) {
    if (!model) return {};
    auto& m = model->model;
    return MjmlxModelInfo{
        m.nq, m.nv, m.nu, m.na, m.nbody, m.njnt, m.ngeom, m.nsite, m.ncon
    };
}

MJMLX_API MjmlxData* mjmlx_make_data(const MjmlxModel* model) {
    if (!model) return nullptr;
    try {
        auto* handle = new MjmlxData();
        handle->data = mjmlx::make_data(model->model);
        handle->model_ref = &model->model;
        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_make_data error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API void mjmlx_free_data(MjmlxData* data) {
    delete data;
}

} // extern "C"
