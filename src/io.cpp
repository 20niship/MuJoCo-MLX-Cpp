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
#include <unistd.h>

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

// Helper: convert a C float array to mx::array with 2D shape (for mesh_vert etc.)
static mx::array to_mx_f2(const float* data, int rows, int cols) {
    return mx::array(data, {rows, cols});
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

// Apply foot-contacts-only filtering on a MuJoCo C model.
// Sets contype=0, conaffinity=0 on all geoms except feet and floor,
// reducing collision pairs from ~126 to 2 for the humanoid.
static void apply_foot_contacts_only(mjModel* m) {
    int floor_id = mj_name2id(m, mjOBJ_GEOM, "floor");
    int rfoot_id = mj_name2id(m, mjOBJ_GEOM, "right_foot");
    int lfoot_id = mj_name2id(m, mjOBJ_GEOM, "left_foot");

    for (int i = 0; i < m->ngeom; i++) {
        if (i != floor_id && i != rfoot_id && i != lfoot_id) {
            m->geom_contype[i] = 0;
            m->geom_conaffinity[i] = 0;
        }
    }
}

// Convert an mjModel* to our internal Model struct.
// The mjModel* is NOT freed here -- caller manages its lifetime.
static Model convert_model(mjModel* m) {
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
    model.body_gravcomp = to_mx_f(m->body_gravcomp, (int)m->nbody);
    model.ngravcomp = m->ngravcomp;

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

        // geom_dataid: maps geom to its mesh id (-1 for non-mesh)
        model.geom_dataid = to_mx_i(m->geom_dataid, (int)m->ngeom);
    }

    // Mesh data (for GJK/EPA convex collision)
    if (m->nmesh > 0) {
        model.nmesh = (int)m->nmesh;
        model.mesh_vertadr = to_mx_i(m->mesh_vertadr, (int)m->nmesh);
        model.mesh_vertnum = to_mx_i(m->mesh_vertnum, (int)m->nmesh);
        int total_verts = 0;
        for (int i = 0; i < m->nmesh; i++)
            total_verts += m->mesh_vertnum[i];
        if (total_verts > 0) {
            model.mesh_vert = to_mx_f2(m->mesh_vert, total_verts, 3);
        }
    }

    // Hfield data (for height field collision)
    if (m->nhfield > 0) {
        model.nhfield = (int)m->nhfield;
        model.hfield_nrow = to_mx_i(m->hfield_nrow, (int)m->nhfield);
        model.hfield_ncol = to_mx_i(m->hfield_ncol, (int)m->nhfield);
        model.hfield_size = to_mx_f2(m->hfield_size, (int)m->nhfield, 4);
        model.hfield_adr = to_mx_i(m->hfield_adr, (int)m->nhfield);
        int total_data = 0;
        for (int i = 0; i < m->nhfield; i++)
            total_data += m->hfield_nrow[i] * m->hfield_ncol[i];
        if (total_data > 0)
            model.hfield_data = to_mx_f2(m->hfield_data, total_data, 1);
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

    // Exclude body pairs
    model.nexclude = m->nexclude;
    if (m->nexclude > 0) {
        model.exclude_signature = to_mx_i(m->exclude_signature, (int)m->nexclude);
    }

    return model;
}

// Validate model for unsupported features and emit warnings.
// This prevents silent failure modes where models load but simulate incorrectly.
static void validate_model(const mjModel* m) {
    // Check for unsupported geom types (mesh, hfield, ellipsoid, cylinder, box)
    bool has_mesh = false, has_box = false, has_hfield = false;
    bool has_ellipsoid = false, has_cylinder = false;
    for (int i = 0; i < m->ngeom; i++) {
        switch (m->geom_type[i]) {
            case mjGEOM_MESH:     has_mesh = true; break;
            case mjGEOM_BOX:      has_box = true; break;
            case mjGEOM_HFIELD:   has_hfield = true; break;
            case mjGEOM_ELLIPSOID: has_ellipsoid = true; break;
            case mjGEOM_CYLINDER: has_cylinder = true; break;
            default: break;
        }
    }
    // MESH geom collision is now supported via GJK/EPA (Phase 3.3)
    // BOX geom collision is now supported (Phase 3.1)
    // HFIELD geom collision is now supported (Phase 3.4)
    // ELLIPSOID geom collision is now supported via GJK/EPA (Phase 3.5)
    // CYLINDER geom collision is now supported (Phase 3.2)

    // Check for unsupported actuator types
    bool has_tendon_trn = false, has_muscle = false, has_site_trn = false;
    bool has_filter_dyn = false, has_integrator_dyn = false;
    for (int i = 0; i < m->nu; i++) {
        if (m->actuator_trntype[i] == mjTRN_TENDON) has_tendon_trn = true;
        if (m->actuator_trntype[i] == mjTRN_SITE) has_site_trn = true;
        if (m->actuator_gaintype[i] == mjGAIN_MUSCLE) has_muscle = true;
        if (m->actuator_dyntype[i] == mjDYN_FILTEREXACT ||
            m->actuator_dyntype[i] == mjDYN_FILTER) has_filter_dyn = true;
        if (m->actuator_dyntype[i] == mjDYN_INTEGRATOR) has_integrator_dyn = true;
    }
    if (has_tendon_trn)
        fprintf(stderr, "[mjmlx WARNING] Model has TENDON transmission -- actuator will produce zero force.\n");
    if (has_site_trn)
        fprintf(stderr, "[mjmlx WARNING] Model has SITE transmission -- actuator will produce zero force.\n");
    if (has_muscle)
        fprintf(stderr, "[mjmlx WARNING] Model has MUSCLE actuators -- act_dot not computed, activation stays at zero.\n");
    if (has_filter_dyn)
        fprintf(stderr, "[mjmlx WARNING] Model has FILTER actuator dynamics -- not implemented, dynamics ignored.\n");
    if (has_integrator_dyn)
        fprintf(stderr, "[mjmlx WARNING] Model has INTEGRATOR actuator dynamics -- not implemented, dynamics ignored.\n");

    // Check for tendons
    if (m->ntendon > 0)
        fprintf(stderr, "[mjmlx WARNING] Model has %lld tendons -- not supported, ignored.\n", (long long)m->ntendon);

    // Check for equality constraints
    if (m->neq > 0)
        fprintf(stderr, "[mjmlx WARNING] Model has %lld equality constraints -- not enforced in MLX backend.\n", (long long)m->neq);

    // Check integrator type
    if (m->opt.integrator == mjINT_RK4)
        fprintf(stderr, "[mjmlx WARNING] Model uses RK4 integrator -- only Euler supported, using Euler.\n");
    if (m->opt.integrator == mjINT_IMPLICIT || m->opt.integrator == mjINT_IMPLICITFAST)
        fprintf(stderr, "[mjmlx WARNING] Model uses implicit integrator -- only Euler supported, using Euler.\n");

    // Check for sensors (not supported)
    if (m->nsensor > 0)
        fprintf(stderr, "[mjmlx WARNING] Model has %lld sensors -- not supported, ignored.\n", (long long)m->nsensor);
}

// Load a MuJoCo C model and convert to internal Model.
// Returns {Model, mjModel*} pair -- caller owns the mjModel*.
std::pair<Model, mjModel*> load_model_pair(const char* xml_path) {
    char error[1000] = "";
    mjModel* m = mj_loadXML(xml_path, nullptr, error, sizeof(error));
    if (!m) {
        throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    }
    validate_model(m);
    Model model = convert_model(m);
    return {std::move(model), m};
}

std::pair<Model, mjModel*> load_model_filtered_pair(const char* xml_path, bool foot_contacts_only) {
    char error[1000] = "";
    mjModel* m = mj_loadXML(xml_path, nullptr, error, sizeof(error));
    if (!m) {
        throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    }
    if (foot_contacts_only) {
        apply_foot_contacts_only(m);
    }
    Model model = convert_model(m);
    return {std::move(model), m};
}

std::pair<Model, mjModel*> load_model_from_string_pair(const char* xml_string) {
    char tmppath[] = "/tmp/mjmlx_model_XXXXXX.xml";
    int fd = mkstemps(tmppath, 4);
    if (fd < 0) throw std::runtime_error("Failed to create temp file for XML");
    size_t len = strlen(xml_string);
    ssize_t written = write(fd, xml_string, len);
    close(fd);
    if (written != (ssize_t)len) {
        unlink(tmppath);
        throw std::runtime_error("Failed to write XML to temp file");
    }
    try {
        auto result = load_model_pair(tmppath);
        unlink(tmppath);
        return result;
    } catch (...) {
        unlink(tmppath);
        throw;
    }
}

// Legacy load functions (for internal C++ use where mjModel* isn't needed)
Model load_model(const char* xml_path) {
    auto [model, mj] = load_model_pair(xml_path);
    mj_deleteModel(mj);
    return model;
}

Model load_model_filtered(const char* xml_path, bool foot_contacts_only) {
    auto [model, mj] = load_model_filtered_pair(xml_path, foot_contacts_only);
    mj_deleteModel(mj);
    return model;
}

Model load_model_from_string(const char* xml_string) {
    auto [model, mj] = load_model_from_string_pair(xml_string);
    mj_deleteModel(mj);
    return model;
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

// ── Model cache initialization ───────────────────────────────

void Model::init_cache() const {
    if (cache.initialized) return;

    // DECISION: Eagerly evaluate topology arrays here so their data<>() pointers are
    // safe to use for building C++ vectors. These are small model constants (O(nbody))
    // that are accessed repeatedly during graph construction. Eager eval here is fine
    // because init_cache() runs once at load time, NOT inside vmap.
    mx::eval(body_parentid); mx::eval(body_rootid);
    mx::eval(dof_bodyid);
    if (njnt > 0) {
        mx::eval(jnt_bodyid); mx::eval(jnt_type);
        mx::eval(jnt_dofadr); mx::eval(jnt_qposadr);
        mx::eval(jnt_limited);
    }

    // Plain C++ vectors for loop indexing
    auto parent_ptr = body_parentid.data<int>();
    auto rootid_ptr = body_rootid.data<int>();
    cache.body_parentid_vec.assign(parent_ptr, parent_ptr + nbody);
    cache.body_rootid_vec.assign(rootid_ptr, rootid_ptr + nbody);
    if (nv > 0) {
        auto dof_bid_ptr = dof_bodyid.data<int>();
        cache.dof_bodyid_vec.assign(dof_bid_ptr, dof_bid_ptr + nv);
    }

    // ── Tree levels (BFS from root) ──
    cache.tree_levels.push_back({0});
    std::vector<bool> visited(nbody, false);
    visited[0] = true;
    while (true) {
        auto& prev = cache.tree_levels.back();
        std::vector<int> next;
        for (int pid : prev) {
            for (int bid = 1; bid < nbody; bid++) {
                if (parent_ptr[bid] == pid && !visited[bid]) {
                    next.push_back(bid);
                    visited[bid] = true;
                }
            }
        }
        if (next.empty()) break;
        cache.tree_levels.push_back(next);
    }

    // ── Body-DOF mapping ──
    cache.body_dofs.resize(nbody);
    if (njnt > 0) {
        auto jnt_bid_ptr = jnt_bodyid.data<int>();
        auto jnt_type_ptr = jnt_type.data<int>();
        auto jnt_da_ptr = jnt_dofadr.data<int>();
        auto jnt_qa_ptr = jnt_qposadr.data<int>();

        for (int di = 0; di < nv; di++) {
            int bid = cache.dof_bodyid_vec[di];
            cache.body_dofs[bid].push_back(di);

            for (int ji = 0; ji < njnt; ji++) {
                if (jnt_bid_ptr[ji] != bid) continue;
                int jt = jnt_type_ptr[ji];
                int dw = (jt == 0) ? 6 : (jt == 1) ? 3 : 1;
                int da = jnt_da_ptr[ji];
                if (da <= di && di < da + dw) {
                    cache.dof_info.push_back({di, bid, jt, ji, jnt_qa_ptr[ji]});
                    break;
                }
            }
        }
    }

    // ── CDoF plan (vectorized cdof masks) ──
    if (nv > 0) {
        std::vector<int> bids(nv), jidxs(nv), root_bids(nv);
        std::vector<float> is_h(nv,0), is_s(nv,0), is_ft(nv,0), is_fr(nv,0), is_b(nv,0);
        std::vector<float> ftu(nv*3, 0);
        std::vector<float> c0(nv,0), c1(nv,0), c2(nv,0);

        for (auto& di : cache.dof_info) {
            int idx = di.dof_idx;
            bids[idx] = di.body_id;
            jidxs[idx] = di.jnt_idx;
            root_bids[idx] = rootid_ptr[di.body_id];
            int local = idx - (njnt > 0 ?
                (int)(mx::eval(jnt_dofadr), jnt_dofadr.data<int>()[di.jnt_idx]) : 0);

            if (di.jnt_type == (int)JointType::HINGE) {
                is_h[idx] = 1.0f;
            } else if (di.jnt_type == (int)JointType::SLIDE) {
                is_s[idx] = 1.0f;
            } else if (di.jnt_type == (int)JointType::FREE) {
                auto da = jnt_dofadr.data<int>()[di.jnt_idx];
                int ld = idx - da;
                if (ld < 3) {
                    is_ft[idx] = 1.0f;
                    ftu[idx*3 + ld] = 1.0f;
                } else {
                    is_fr[idx] = 1.0f;
                    int col = ld - 3;
                    if (col == 0) c0[idx] = 1.0f;
                    else if (col == 1) c1[idx] = 1.0f;
                    else c2[idx] = 1.0f;
                }
            } else if (di.jnt_type == (int)JointType::BALL) {
                is_b[idx] = 1.0f;
                auto da = jnt_dofadr.data<int>()[di.jnt_idx];
                int ld = idx - da;
                if (ld == 0) c0[idx] = 1.0f;
                else if (ld == 1) c1[idx] = 1.0f;
                else c2[idx] = 1.0f;
            }
        }

        cache.cdof_plan.bids = mx::array(bids.data(), {nv}, mx::int32);
        cache.cdof_plan.jidxs = mx::array(jidxs.data(), {nv}, mx::int32);
        cache.cdof_plan.root_bids = mx::array(root_bids.data(), {nv}, mx::int32);
        cache.cdof_plan.is_hinge = mx::reshape(mx::array(is_h.data(), {nv}), {nv, 1});
        cache.cdof_plan.is_slide = mx::reshape(mx::array(is_s.data(), {nv}), {nv, 1});
        cache.cdof_plan.is_free_trans = mx::reshape(mx::array(is_ft.data(), {nv}), {nv, 1});
        cache.cdof_plan.is_free_rot = mx::reshape(mx::array(is_fr.data(), {nv}), {nv, 1});
        cache.cdof_plan.is_ball = mx::reshape(mx::array(is_b.data(), {nv}), {nv, 1});
        cache.cdof_plan.free_trans_unit = mx::array(ftu.data(), {nv, 3});
        cache.cdof_plan.rot_col0_mask = mx::reshape(mx::array(c0.data(), {nv}), {nv, 1});
        cache.cdof_plan.rot_col1_mask = mx::reshape(mx::array(c1.data(), {nv}), {nv, 1});
        cache.cdof_plan.rot_col2_mask = mx::reshape(mx::array(c2.data(), {nv}), {nv, 1});
    }

    // ── Body DOF ancestor masks (for Jacobian computation) ──
    cache.body_dof_masks.assign(nbody, mx::array(0.0f));
    for (int body_id = 0; body_id < nbody; body_id++) {
        std::set<int> ancestors;
        int bid = body_id;
        while (bid >= 0) {
            ancestors.insert(bid);
            bid = (bid > 0) ? parent_ptr[bid] : -1;
        }
        std::vector<float> mask(nv, 0.0f);
        for (int di = 0; di < nv; di++) {
            if (ancestors.count(cache.dof_bodyid_vec[di])) {
                mask[di] = 1.0f;
            }
        }
        cache.body_dof_masks[body_id] = mx::array(mask.data(), {nv}, mx::float32);
    }

    // ── Dense mass matrix tree mask ──
    // DECISION: make_m_mask is strictly lower-triangular. The mask encodes the DOF
    // parent-chain topology: for each DOF i, walk from i toward the root setting
    // mask[i][j]=1 for each ancestor j (which always satisfies j <= i). The full
    // symmetric mass matrix is recovered later in vmap_crb via: qm = qm + tril(qm,-1)^T.
    // An earlier bug had mask[j][i]=1 too (making it symmetric), which broke the
    // Cholesky factorization (error=30.34). The Python reference _get_mass_matrix_mask
    // in support.py is also strictly lower-triangular.
    if (nv > 0 && !is_sparse(*this)) {
        mx::eval(dof_parentid);
        auto dof_par_ptr = dof_parentid.data<int>();
        std::vector<float> mask_data(nv * nv, 0.0f);
        for (int i = 0; i < nv; i++) {
            int j = i;
            while (j > -1) {
                mask_data[i * nv + j] = 1.0f;
                j = dof_par_ptr[j];
            }
        }
        cache.make_m_mask = mx::array(mask_data.data(), {nv, nv}, mx::float32);
    }

    // ── Build exclude set for fast lookup ──
    std::set<int> exclude_set;
    if (nexclude > 0 && exclude_signature.size() > 0) {
        mx::eval(exclude_signature);
        auto ex_ptr = exclude_signature.data<int>();
        for (int i = 0; i < nexclude; i++) {
            exclude_set.insert(ex_ptr[i]);
        }
    }

    // ── Collision pairs (pre-computed from model topology) ──
    if (ngeom > 0) {
        mx::eval(geom_type); mx::eval(geom_bodyid);
        mx::eval(geom_contype); mx::eval(geom_conaffinity);
        mx::eval(body_weldid); mx::eval(geom_margin);
        mx::eval(geom_size);

        auto gtype = geom_type.data<int>();
        auto gbid = geom_bodyid.data<int>();
        auto gcon = geom_contype.data<int>();
        auto gaff = geom_conaffinity.data<int>();
        auto bwid_ptr = body_weldid.data<int>();
        auto gmargin_ptr = geom_margin.data<float>();
        auto gsize_ptr = geom_size.data<float>();

        for (int g1 = 0; g1 < ngeom; g1++) {
            for (int g2 = g1 + 1; g2 < ngeom; g2++) {
                int t1 = gtype[g1], t2 = gtype[g2];
                int g1_ = g1, g2_ = g2, t1_ = t1, t2_ = t2;
                if (t1 > t2) { std::swap(g1_, g2_); std::swap(t1_, t2_); }

                int mask = (gcon[g1_] & gaff[g2_]) | (gcon[g2_] & gaff[g1_]);
                if (!mask) continue;

                int b1 = gbid[g1_], b2 = gbid[g2_];
                int w1 = bwid_ptr[b1], w2 = bwid_ptr[b2];
                if (w1 == w2) continue;
                if (!(opt.disableflags & DisableBit::FILTERPARENT)) {
                    int w1p = (w1 > 0) ? bwid_ptr[parent_ptr[w1]] : 0;
                    int w2p = (w2 > 0) ? bwid_ptr[parent_ptr[w2]] : 0;
                    if (w1 != 0 && w2 != 0 && (w1 == w2p || w2 == w1p)) continue;
                }

                // Check exclude_signature: skip excluded body pairs
                // MuJoCo C signature format: (min(b1,b2) << 16) | max(b1,b2)
                if (!exclude_set.empty()) {
                    int bmin = std::min(b1, b2);
                    int bmax = std::max(b1, b2);
                    int sig = (bmin << 16) | bmax;
                    if (exclude_set.count(sig)) continue;
                }

                ModelCache::CollisionPair cp;
                cp.g1 = g1_; cp.g2 = g2_;
                cp.type1 = t1_; cp.type2 = t2_;
                cp.body1 = b1; cp.body2 = b2;
                cp.margin = gmargin_ptr[g1_] + gmargin_ptr[g2_];
                float gap = 0.0f;
                if (geom_gap.size() > 0) {
                    mx::eval(geom_gap);
                    auto gp = geom_gap.data<float>();
                    gap = gp[g1_] + gp[g2_];
                }
                cp.gap = gap;
                for (int k = 0; k < 3; k++) {
                    cp.size1[k] = gsize_ptr[g1_ * 3 + k];
                    cp.size2[k] = gsize_ptr[g2_ * 3 + k];
                }
                // Mesh data ids
                if (geom_dataid.size() > 0) {
                    mx::eval(geom_dataid);
                    auto gdid = geom_dataid.data<int>();
                    cp.dataid1 = gdid[g1_];
                    cp.dataid2 = gdid[g2_];
                }
                // Friction: max of both geoms
                if (geom_friction.size() > 0) {
                    mx::eval(geom_friction);
                    auto gf = geom_friction.data<float>();
                    float f0 = std::max(gf[g1_*3], gf[g2_*3]);
                    cp.friction[0] = f0; cp.friction[1] = f0;
                    cp.friction[2] = std::max(gf[g1_*3+1], gf[g2_*3+1]);
                    cp.friction[3] = std::max(gf[g1_*3+2], gf[g2_*3+2]);
                    cp.friction[4] = cp.friction[3];
                }
                // Solref: average
                if (geom_solref.size() > 0) {
                    mx::eval(geom_solref);
                    auto sr = geom_solref.data<float>();
                    cp.solref[0] = 0.5f * (sr[g1_*2] + sr[g2_*2]);
                    cp.solref[1] = 0.5f * (sr[g1_*2+1] + sr[g2_*2+1]);
                } else {
                    cp.solref[0] = 0.02f; cp.solref[1] = 1.0f;
                }
                // Solimp: average
                if (geom_solimp.size() > 0) {
                    mx::eval(geom_solimp);
                    auto si = geom_solimp.data<float>();
                    for (int k = 0; k < 5; k++)
                        cp.solimp[k] = 0.5f * (si[g1_*5+k] + si[g2_*5+k]);
                } else {
                    float def[] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
                    for (int k = 0; k < 5; k++) cp.solimp[k] = def[k];
                }
                // Condim
                cp.condim = 1;
                if (geom_condim.size() > 0) {
                    mx::eval(geom_condim);
                    auto cdp = geom_condim.data<int>();
                    cp.condim = std::max(cdp[g1_], cdp[g2_]);
                }

                cache.collision_pairs.push_back(cp);
            }
        }

        // Add explicit <pair> directives (even if contype/conaffinity mask is 0)
        if (npair > 0 && pair_geom1.size() > 0) {
            mx::eval(pair_geom1); mx::eval(pair_geom2);
            auto pg1 = pair_geom1.data<int>();
            auto pg2 = pair_geom2.data<int>();

            // Optional pair properties
            bool has_dim = (pair_dim.size() > 0);
            bool has_margin = (pair_margin.size() > 0);
            bool has_gap = (pair_gap.size() > 0);
            bool has_solref = (pair_solref.size() > 0);
            bool has_solimp = (pair_solimp.size() > 0);
            bool has_friction = (pair_friction.size() > 0);
            if (has_dim) mx::eval(pair_dim);
            if (has_margin) mx::eval(pair_margin);
            if (has_gap) mx::eval(pair_gap);
            if (has_solref) mx::eval(pair_solref);
            if (has_solimp) mx::eval(pair_solimp);
            if (has_friction) mx::eval(pair_friction);

            for (int pi = 0; pi < npair; pi++) {
                int g1_ = pg1[pi], g2_ = pg2[pi];
                int t1_ = gtype[g1_], t2_ = gtype[g2_];
                // Canonical order: lower type first
                if (t1_ > t2_) { std::swap(g1_, g2_); std::swap(t1_, t2_); }

                // Check if this pair is already in the list (dedup)
                bool dup = false;
                for (auto& existing : cache.collision_pairs) {
                    if ((existing.g1 == g1_ && existing.g2 == g2_) ||
                        (existing.g1 == g2_ && existing.g2 == g1_)) {
                        dup = true; break;
                    }
                }
                if (dup) continue;

                ModelCache::CollisionPair cp;
                cp.g1 = g1_; cp.g2 = g2_;
                cp.type1 = t1_; cp.type2 = t2_;
                cp.body1 = gbid[g1_]; cp.body2 = gbid[g2_];
                cp.margin = has_margin ? pair_margin.data<float>()[pi] :
                            (gmargin_ptr[g1_] + gmargin_ptr[g2_]);
                cp.gap = has_gap ? pair_gap.data<float>()[pi] : 0.0f;
                for (int k = 0; k < 3; k++) {
                    cp.size1[k] = gsize_ptr[g1_ * 3 + k];
                    cp.size2[k] = gsize_ptr[g2_ * 3 + k];
                }
                if (geom_dataid.size() > 0) {
                    mx::eval(geom_dataid);
                    auto gdid = geom_dataid.data<int>();
                    cp.dataid1 = gdid[g1_];
                    cp.dataid2 = gdid[g2_];
                }
                if (has_friction) {
                    auto fp = pair_friction.data<float>();
                    for (int k = 0; k < 5; k++) cp.friction[k] = fp[pi * 5 + k];
                } else if (geom_friction.size() > 0) {
                    auto gf = geom_friction.data<float>();
                    float f0 = std::max(gf[g1_*3], gf[g2_*3]);
                    cp.friction[0] = f0; cp.friction[1] = f0;
                    cp.friction[2] = std::max(gf[g1_*3+1], gf[g2_*3+1]);
                    cp.friction[3] = std::max(gf[g1_*3+2], gf[g2_*3+2]);
                    cp.friction[4] = cp.friction[3];
                }
                if (has_solref) {
                    auto sr = pair_solref.data<float>();
                    cp.solref[0] = sr[pi * 2]; cp.solref[1] = sr[pi * 2 + 1];
                } else {
                    cp.solref[0] = 0.02f; cp.solref[1] = 1.0f;
                }
                if (has_solimp) {
                    auto si = pair_solimp.data<float>();
                    for (int k = 0; k < 5; k++) cp.solimp[k] = si[pi * 5 + k];
                } else {
                    float def[] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
                    for (int k = 0; k < 5; k++) cp.solimp[k] = def[k];
                }
                cp.condim = has_dim ? pair_dim.data<int>()[pi] : 3;

                cache.collision_pairs.push_back(cp);
            }
        }

        cache.max_ncon = (int)cache.collision_pairs.size();

        // Pre-bake mesh vertex slices for vmap collision path
        if (mesh_vert.size() > 0 && mesh_vertadr.size() > 0 && mesh_vertnum.size() > 0) {
            mx::eval(mesh_vertadr); mx::eval(mesh_vertnum); mx::eval(mesh_vert);
            auto vadr = mesh_vertadr.data<int>();
            auto vnum = mesh_vertnum.data<int>();
            for (auto& cp : cache.collision_pairs) {
                if (cp.type1 == (int)GeomType::MESH && cp.dataid1 >= 0) {
                    int start = vadr[cp.dataid1];
                    int count = vnum[cp.dataid1];
                    if (count > 0)
                        cp.mesh_verts1 = mx::slice(mesh_vert, {start, 0}, {start + count, 3});
                }
                if (cp.type2 == (int)GeomType::MESH && cp.dataid2 >= 0) {
                    int start = vadr[cp.dataid2];
                    int count = vnum[cp.dataid2];
                    if (count > 0)
                        cp.mesh_verts2 = mx::slice(mesh_vert, {start, 0}, {start + count, 3});
                }
            }
        }

        // Pre-bake hfield data per collision pair for vmap path
        if (hfield_data.size() > 0 && hfield_adr.size() > 0) {
            mx::eval(hfield_adr); mx::eval(hfield_nrow); mx::eval(hfield_ncol);
            mx::eval(hfield_size); mx::eval(hfield_data);
            auto hadr = hfield_adr.data<int>();
            auto hnrow = hfield_nrow.data<int>();
            auto hncol = hfield_ncol.data<int>();
            auto hsz = hfield_size.data<float>();
            for (auto& cp : cache.collision_pairs) {
                int hf_type = -1, hf_dataid = -1;
                if (cp.type1 == (int)GeomType::HFIELD && cp.dataid1 >= 0) {
                    hf_type = 1; hf_dataid = cp.dataid1;
                } else if (cp.type2 == (int)GeomType::HFIELD && cp.dataid2 >= 0) {
                    hf_type = 2; hf_dataid = cp.dataid2;
                }
                if (hf_dataid >= 0) {
                    cp.hf_nrow = hnrow[hf_dataid];
                    cp.hf_ncol = hncol[hf_dataid];
                    for (int k = 0; k < 4; k++) cp.hf_size[k] = hsz[hf_dataid * 4 + k];
                    int start = hadr[hf_dataid];
                    int count = cp.hf_nrow * cp.hf_ncol;
                    if (count > 0) {
                        cp.hf_data = mx::reshape(
                            mx::slice(hfield_data, {start, 0}, {start + count, 1}),
                            {cp.hf_nrow, cp.hf_ncol});
                    }
                }
            }
        }
    }

    // ── Joint limits ──
    if (njnt > 0 && jnt_limited.size() > 0) {
        auto limited = jnt_limited.data<int>();
        auto jt_ptr = jnt_type.data<int>();
        auto jda_ptr = jnt_dofadr.data<int>();

        for (int j = 0; j < njnt; j++) {
            if (!limited[j]) continue;
            int jt = jt_ptr[j];
            if (jt != (int)JointType::SLIDE && jt != (int)JointType::HINGE) continue;

            ModelCache::LimitInfo li;
            li.jnt_idx = j;
            li.dof_adr = jda_ptr[j];
            mx::eval(jnt_range);
            auto jr = jnt_range.data<float>();
            li.range_low = jr[j * 2];
            li.range_high = jr[j * 2 + 1];
            if (jnt_solref.size() > 0) {
                mx::eval(jnt_solref);
                auto sp = jnt_solref.data<float>();
                li.solref[0] = sp[j*2]; li.solref[1] = sp[j*2+1];
            } else {
                li.solref[0] = 0.02f; li.solref[1] = 1.0f;
            }
            if (jnt_solimp.size() > 0) {
                mx::eval(jnt_solimp);
                auto sp = jnt_solimp.data<float>();
                for (int k = 0; k < 5; k++) li.solimp[k] = sp[j*5+k];
            } else {
                float def[] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
                for (int k = 0; k < 5; k++) li.solimp[k] = def[k];
            }
            li.margin = 0.0f;
            if (jnt_margin.size() > 0) {
                mx::eval(jnt_margin);
                li.margin = jnt_margin.data<float>()[j];
            }
            cache.limits.push_back(li);
        }
    }
    cache.max_nl = (int)cache.limits.size();
    // Compute max contact constraint rows accounting for condim and multi-contact pairs:
    // plane-box: up to 4 contacts, plane-cylinder: up to 6, capsule-box: up to 2, others: 1
    // condim=1: 1 row/contact, condim=3 pyramidal: 4 rows/contact, condim=4: 6, condim=6: 10
    int max_contact_rows = 0;
    for (auto& cp : cache.collision_pairs) {
        int rows_per_contact = (cp.condim <= 1) ? 1 : 2 * (cp.condim - 1);
        int max_contacts = 1;
        if (cp.type1 == (int)GeomType::PLANE && cp.type2 == (int)GeomType::BOX)
            max_contacts = 4;
        else if (cp.type1 == (int)GeomType::PLANE && cp.type2 == (int)GeomType::CYLINDER)
            max_contacts = 6; // 2 faces × (center + 2 rim points)
        else if (cp.type1 == (int)GeomType::CAPSULE && cp.type2 == (int)GeomType::BOX)
            max_contacts = 2;
        else if (cp.type1 == (int)GeomType::BOX && cp.type2 == (int)GeomType::BOX)
            max_contacts = 8;
        else if (cp.type1 == (int)GeomType::PLANE && cp.type2 == (int)GeomType::MESH) {
            // plane-mesh: up to all vertices can penetrate (use dataid to get count)
            int dataid = cp.dataid2;
            if (dataid >= 0 && mesh_vertnum.size() > 0) {
                mx::eval(mesh_vertnum);
                max_contacts = mesh_vertnum.data<int>()[dataid];
            } else {
                max_contacts = 8; // conservative fallback
            }
        }
        else if (cp.type2 == (int)GeomType::MESH || cp.type1 == (int)GeomType::MESH)
            max_contacts = 1; // GJK/EPA returns 1 contact
        else if (cp.type1 == (int)GeomType::HFIELD || cp.type2 == (int)GeomType::HFIELD)
            max_contacts = 50; // MuJoCo C limit per hfield pair (mjMAXCONPAIR)
        max_contact_rows += max_contacts * rows_per_contact;
    }
    cache.max_nefc = cache.max_nl + max_contact_rows;

    // ── Joint integration plan ──
    if (njnt > 0) {
        auto jt_ptr = jnt_type.data<int>();
        auto jqa_ptr = jnt_qposadr.data<int>();
        auto jda_ptr = jnt_dofadr.data<int>();
        for (int j = 0; j < njnt; j++) {
            int jt = jt_ptr[j];
            if (jt == (int)JointType::HINGE || jt == (int)JointType::SLIDE) {
                cache.simple_qa.push_back(jqa_ptr[j]);
                cache.simple_da.push_back(jda_ptr[j]);
            } else if (jt == (int)JointType::FREE) {
                cache.free_joints.push_back({jqa_ptr[j], jda_ptr[j]});
            } else if (jt == (int)JointType::BALL) {
                cache.ball_joints.push_back({jqa_ptr[j], jda_ptr[j]});
            }
        }
    }

    // ── DOF damping ──
    if (nv > 0) {
        mx::eval(dof_damping);
        auto dp = dof_damping.data<float>();
        cache.dof_damping_vals.assign(dp, dp + nv);
    }

    // ── Gravity 6D ──
    if (!(opt.disableflags & DisableBit::GRAVITY)) {
        cache.gravity_6d = mx::concatenate({mx::zeros({3}), mx::negative(opt.gravity)}, 0);
    } else {
        cache.gravity_6d = mx::zeros({6});
    }

    // ── Precomputed actuator matrices ──
    if (nu > 0 && njnt > 0) {
        std::vector<float> moment_data(nu * nv, 0.0f);
        std::vector<int> act_qpos_idx(nu, 0);
        std::vector<float> act_gear_vals(nu, 0.0f);

        mx::eval(actuator_trntype);
        mx::eval(actuator_trnid);
        mx::eval(actuator_gear);
        auto trn_type_ptr = actuator_trntype.data<int>();
        auto trn_id_ptr = actuator_trnid.data<int>();
        auto gear_ptr = actuator_gear.data<float>();

        for (int ai = 0; ai < nu; ai++) {
            int trnt = trn_type_ptr[ai];
            if (trnt != 0) continue; // Only JOINT transmission for now
            int ji = trn_id_ptr[ai * 2];
            float g0 = gear_ptr[ai * 6];

            // Find the DOF address for this joint
            int da = -1, qa = -1;
            for (auto& di : cache.dof_info) {
                if (di.jnt_idx == ji) {
                    da = di.dof_idx;
                    qa = di.qpos_adr;
                    break;
                }
            }
            if (da >= 0 && da < nv) {
                moment_data[ai * nv + da] = g0;
                act_qpos_idx[ai] = qa;
                act_gear_vals[ai] = g0;
                cache.actuator_info.push_back({ai, ji, da, qa, g0});
            }
        }
        cache.act_moment_const = mx::array(moment_data.data(), mx::Shape{nu, nv}, mx::float32);
        cache.act_qpos_idxs = mx::array(act_qpos_idx.data(), mx::Shape{nu}, mx::int32);
        cache.act_gear = mx::array(act_gear_vals.data(), mx::Shape{nu}, mx::float32);
    }

    // ── Precomputed passive force arrays ──
    if (nv > 0 && njnt > 0) {
        std::vector<float> stiff_per_dof(nv, 0.0f);
        std::vector<int> qpos_per_dof(nv, 0);

        mx::eval(jnt_stiffness);
        auto stiff_ptr = jnt_stiffness.data<float>();
        for (auto& di : cache.dof_info) {
            stiff_per_dof[di.dof_idx] = stiff_ptr[di.jnt_idx];
            qpos_per_dof[di.dof_idx] = di.qpos_adr;
        }
        cache.passive_stiffness = mx::array(stiff_per_dof.data(), mx::Shape{nv}, mx::float32);
        cache.passive_qpos_idxs = mx::array(qpos_per_dof.data(), mx::Shape{nv}, mx::int32);
    }

    cache.initialized = true;
}

} // namespace mjmlx

// ── C API wrappers ───────────────────────────────────────────

extern "C" {

MJMLX_API MjmlxModel* mjmlx_load_model(const char* xml_path) {
    try {
        auto* handle = new MjmlxModel();
        auto [model, mj] = mjmlx::load_model_pair(xml_path);
        handle->model = std::move(model);
        handle->mj_model = mj;
        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_load_model error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API MjmlxModel* mjmlx_load_model_filtered(const char* xml_path, int foot_contacts_only) {
    try {
        auto* handle = new MjmlxModel();
        auto [model, mj] = mjmlx::load_model_filtered_pair(xml_path, foot_contacts_only != 0);
        handle->model = std::move(model);
        handle->mj_model = mj;
        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_load_model_filtered error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API MjmlxModel* mjmlx_load_model_from_string(const char* xml_string) {
    try {
        auto* handle = new MjmlxModel();
        auto [model, mj] = mjmlx::load_model_from_string_pair(xml_string);
        handle->model = std::move(model);
        handle->mj_model = mj;
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

MJMLX_API void mjmlx_reset_data(const MjmlxModel* model, MjmlxData* data) {
    if (!model || !data) return;
    try {
        data->data = mjmlx::make_data(model->model);
        data->model_ref = &model->model;
    } catch (...) {}
}

MJMLX_API float mjmlx_model_opt_timestep(const MjmlxModel* model) {
    if (!model) return 0.0f;
    return model->model.opt.timestep;
}

MJMLX_API void mjmlx_model_set_opt_timestep(MjmlxModel* model, float dt) {
    if (!model) return;
    model->model.opt.timestep = dt;
}

MJMLX_API float mjmlx_model_body_mass(const MjmlxModel* model, int body_id) {
    if (!model || body_id < 0 || body_id >= model->model.nbody) return 0.0f;
    mx::eval(model->model.body_mass);
    return model->model.body_mass.data<float>()[body_id];
}

MJMLX_API int mjmlx_name2id(const MjmlxModel* model, int obj_type, const char* name) {
    if (!model || !model->mj_model || !name) return -1;
    return mj_name2id(model->mj_model, obj_type, name);
}

} // extern "C"
