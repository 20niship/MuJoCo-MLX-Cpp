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

// Unified dual-backend implementation.
// Dispatches to MuJoCo C (CPU) or MuJoCo-MLX (Metal GPU).

#include "mjmlx/mjb.h"
#include "mjmlx/mjmlx.h"
#include <mujoco/mujoco.h>
#include <dispatch/dispatch.h>
#include <cstring>
#include <vector>
#include <cstdio>

// ── Opaque handle definitions ────────────────────────────────────────────

struct MjbBackend {
    MjbBackendType type;
};

struct MjbModel {
    MjbBackendType type;
    // CPU backend
    mjModel* mj = nullptr;
    // MLX backend
    MjmlxModel* mlx = nullptr;

    mutable std::vector<float> fbuf;  // double->float conversion buffer

    ~MjbModel() {
        if (mj) mj_deleteModel(mj);
        if (mlx) mjmlx_free_model(mlx);
    }
};

struct MjbData {
    MjbBackendType type;
    MjbModel* model_ref = nullptr;

    // CPU backend
    mjData* mj = nullptr;
    mutable std::vector<float> fbuf;  // double->float conversion buffer

    // MLX backend
    MjmlxData* mlx = nullptr;

    ~MjbData() {
        if (mj) mj_deleteData(mj);
        if (mlx) mjmlx_free_data(mlx);
    }
};

struct MjbBatchedSim {
    MjbBackendType type;
    MjbModel* model_ref = nullptr;
    int num_envs = 0;
    bool gpu_active = false;  // true only if Metal GPU kernels are actually running

    // CPU backend: N independent mjData* + contiguous float buffers
    std::vector<mjData*> cpu_datas;
    mutable std::vector<float> qpos_buf, qvel_buf, xpos_buf;
    mutable std::vector<float> subtree_com_buf, cinert_buf, cvel_buf;
    mutable std::vector<float> qfrc_actuator_buf, cfrc_ext_buf;

    // MLX backend
    MjmlxBatchedSim* mlx_sim = nullptr;

    ~MjbBatchedSim() {
        for (auto* d : cpu_datas) if (d) mj_deleteData(d);
        if (mlx_sim) mjmlx_batched_free(mlx_sim);
    }
};

// ── Internal helpers ────────────────────────────────────────────────────

// Get the underlying C mjModel* regardless of backend.
// MLX models keep a copy for name/field lookups.
static const mjModel* get_mj_model(const MjbModel* model) {
    if (!model) return nullptr;
    if (model->type == MJB_BACKEND_CPU) return model->mj;
    return static_cast<const mjModel*>(mjmlx_get_mj_model(model->mlx));
}

// Convert double array to float buffer, return pointer
static const float* d2f(const double* src, int n, std::vector<float>& buf) {
    buf.resize(n);
    for (int i = 0; i < n; i++) buf[i] = static_cast<float>(src[i]);
    return buf.data();
}

// Fill float buffer from double array, return size
static const float* cpu_get(const double* src, int n, int* n_out, std::vector<float>& buf) {
    if (n_out) *n_out = n;
    if (n == 0) return nullptr;
    return d2f(src, n, buf);
}

// ── Backend lifecycle ───────────────────────────────────────────────────

extern "C" {

MJB_API MjbBackend* mjb_create_backend(MjbBackendType type) {
    auto* b = new MjbBackend();
    b->type = type;
    return b;
}

MJB_API void mjb_free_backend(MjbBackend* backend) {
    delete backend;
}

MJB_API MjbBackendType mjb_backend_type(const MjbBackend* backend) {
    return backend ? backend->type : MJB_BACKEND_CPU;
}

// ── Model I/O ───────────────────────────────────────────────────────────

MJB_API MjbModel* mjb_load_model(MjbBackend* b, const char* xml_path) {
    if (!b || !xml_path) return nullptr;
    auto* m = new MjbModel();
    m->type = b->type;
    try {
        if (b->type == MJB_BACKEND_CPU) {
            char error[1000] = "";
            m->mj = mj_loadXML(xml_path, nullptr, error, sizeof(error));
            if (!m->mj) {
                fprintf(stderr, "mjb_load_model CPU error: %s\n", error);
                delete m;
                return nullptr;
            }
        } else {
            m->mlx = mjmlx_load_model(xml_path);
            if (!m->mlx) { delete m; return nullptr; }
        }
    } catch (...) {
        delete m;
        return nullptr;
    }
    return m;
}

MJB_API MjbModel* mjb_load_model_filtered(MjbBackend* b, const char* xml_path, int foot_contacts_only) {
    if (!b || !xml_path) return nullptr;
    auto* m = new MjbModel();
    m->type = b->type;
    try {
        if (b->type == MJB_BACKEND_CPU) {
            char error[1000] = "";
            m->mj = mj_loadXML(xml_path, nullptr, error, sizeof(error));
            if (!m->mj) {
                fprintf(stderr, "mjb_load_model_filtered CPU error: %s\n", error);
                delete m;
                return nullptr;
            }
            if (foot_contacts_only) {
                // Apply foot-contacts-only filter (same as mjmlx)
                int floor_id = mj_name2id(m->mj, mjOBJ_GEOM, "floor");
                int rfoot_id = mj_name2id(m->mj, mjOBJ_GEOM, "right_foot");
                int lfoot_id = mj_name2id(m->mj, mjOBJ_GEOM, "left_foot");
                for (int i = 0; i < m->mj->ngeom; i++) {
                    if (i != floor_id && i != rfoot_id && i != lfoot_id) {
                        m->mj->geom_contype[i] = 0;
                        m->mj->geom_conaffinity[i] = 0;
                    }
                }
            }
        } else {
            m->mlx = mjmlx_load_model_filtered(xml_path, foot_contacts_only);
            if (!m->mlx) { delete m; return nullptr; }
        }
    } catch (...) {
        delete m;
        return nullptr;
    }
    return m;
}

MJB_API MjbModel* mjb_load_model_from_string(MjbBackend* b, const char* xml_string) {
    if (!b || !xml_string) return nullptr;
    auto* m = new MjbModel();
    m->type = b->type;
    try {
        if (b->type == MJB_BACKEND_CPU) {
            char error[1000] = "";
            mjVFS vfs;
            mj_defaultVFS(&vfs);
            int len = (int)strlen(xml_string);
            mj_addBufferVFS(&vfs, "model.xml", xml_string, len);
            m->mj = mj_loadXML("model.xml", &vfs, error, sizeof(error));
            mj_deleteVFS(&vfs);
            if (!m->mj) {
                fprintf(stderr, "mjb_load_model_from_string CPU error: %s\n", error);
                delete m;
                return nullptr;
            }
        } else {
            m->mlx = mjmlx_load_model_from_string(xml_string);
            if (!m->mlx) { delete m; return nullptr; }
        }
    } catch (...) {
        delete m;
        return nullptr;
    }
    return m;
}

MJB_API void mjb_free_model(MjbModel* model) {
    delete model;
}

// ── Model accessors ─────────────────────────────────────────────────────

MJB_API MjbModelInfo mjb_model_info(const MjbModel* model) {
    if (!model) return {};
    const mjModel* m = get_mj_model(model);
    if (!m) return {};
    MjbModelInfo info = {};
    info.nq = (int)m->nq;
    info.nv = (int)m->nv;
    info.nu = (int)m->nu;
    info.nbody = (int)m->nbody;
    info.njnt = (int)m->njnt;
    info.ngeom = (int)m->ngeom;
    info.nsite = (int)m->nsite;
    info.nmocap = (int)m->nmocap;
    info.ntendon = (int)m->ntendon;
    info.nsensor = (int)m->nsensor;
    info.nsensordata = (int)m->nsensordata;
    info.neq = (int)m->neq;
    return info;
}

MJB_API float mjb_model_opt_timestep(const MjbModel* model) {
    if (!model) return 0.0f;
    if (model->type == MJB_BACKEND_CPU) return (float)model->mj->opt.timestep;
    return mjmlx_model_opt_timestep(model->mlx);
}

MJB_API void mjb_model_set_opt_timestep(MjbModel* model, float dt) {
    if (!model) return;
    if (model->type == MJB_BACKEND_CPU) model->mj->opt.timestep = dt;
    else mjmlx_model_set_opt_timestep(model->mlx, dt);
}

MJB_API float mjb_model_body_mass(const MjbModel* model, int body_id) {
    if (!model) return 0.0f;
    if (model->type == MJB_BACKEND_CPU) {
        if (body_id < 0 || body_id >= model->mj->nbody) return 0.0f;
        return (float)model->mj->body_mass[body_id];
    }
    return mjmlx_model_body_mass(model->mlx, body_id);
}

MJB_API int mjb_name2id(const MjbModel* model, int obj_type, const char* name) {
    if (!model || !name) return -1;
    if (model->type == MJB_BACKEND_CPU) return mj_name2id(model->mj, obj_type, name);
    return mjmlx_name2id(model->mlx, obj_type, name);
}

MJB_API const char* mjb_id2name(const MjbModel* model, int obj_type, int id) {
    const mjModel* m = get_mj_model(model);
    if (!m) return nullptr;
    return mj_id2name(m, obj_type, id);
}

MJB_API int mjb_model_jnt_qposadr(const MjbModel* model, int jnt_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || jnt_id < 0 || jnt_id >= m->njnt) return -1;
    return m->jnt_qposadr[jnt_id];
}

MJB_API int mjb_model_jnt_dofadr(const MjbModel* model, int jnt_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || jnt_id < 0 || jnt_id >= m->njnt) return -1;
    return m->jnt_dofadr[jnt_id];
}

MJB_API int mjb_model_jnt_type(const MjbModel* model, int jnt_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || jnt_id < 0 || jnt_id >= m->njnt) return -1;
    return m->jnt_type[jnt_id];
}

MJB_API int mjb_model_nconmax(const MjbModel* model) {
    const mjModel* m = get_mj_model(model);
    return m ? (int)m->nconmax : 0;
}

MJB_API int mjb_model_geom_type(const MjbModel* model, int geom_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || geom_id < 0 || geom_id >= m->ngeom) return -1;
    return m->geom_type[geom_id];
}

MJB_API int mjb_model_sensor_adr(const MjbModel* model, int sensor_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || sensor_id < 0 || sensor_id >= m->nsensor) return -1;
    return m->sensor_adr[sensor_id];
}

MJB_API int mjb_model_body_mocapid(const MjbModel* model, int body_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || body_id < 0 || body_id >= m->nbody) return -1;
    return m->body_mocapid[body_id];
}

MJB_API float mjb_model_tendon_width(const MjbModel* model, int tendon_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || tendon_id < 0 || tendon_id >= m->ntendon) return 0.0f;
    return (float)m->tendon_width[tendon_id];
}

MJB_API int mjb_model_hfield_adr(const MjbModel* model, int hfield_id) {
    const mjModel* m = get_mj_model(model);
    if (!m || hfield_id < 0 || hfield_id >= m->nhfield) return -1;
    return m->hfield_adr[hfield_id];
}

MJB_API const float* mjb_model_eq_data(const MjbModel* model, int* n_out) {
    const mjModel* m = get_mj_model(model);
    if (!m || m->neq == 0) { if (n_out) *n_out = 0; return nullptr; }
    int total = m->neq * mjNEQDATA;
    return cpu_get(m->eq_data, total, n_out, model->fbuf);
}

MJB_API const float* mjb_model_hfield_data(const MjbModel* model, int* n_out) {
    const mjModel* m = get_mj_model(model);
    if (!m || m->nhfield == 0) { if (n_out) *n_out = 0; return nullptr; }
    int total = (int)m->nhfielddata;
    if (n_out) *n_out = total;
    return m->hfield_data;
}

// ── Data lifecycle ──────────────────────────────────────────────────────

MJB_API MjbData* mjb_make_data(MjbModel* model) {
    if (!model) return nullptr;
    auto* d = new MjbData();
    d->type = model->type;
    d->model_ref = model;
    try {
        if (model->type == MJB_BACKEND_CPU) {
            d->mj = mj_makeData(model->mj);
            if (!d->mj) { delete d; return nullptr; }
        } else {
            d->mlx = mjmlx_make_data(model->mlx);
            if (!d->mlx) { delete d; return nullptr; }
        }
    } catch (...) {
        delete d;
        return nullptr;
    }
    return d;
}

MJB_API void mjb_free_data(MjbData* data) {
    delete data;
}

MJB_API void mjb_reset_data(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) {
        mj_resetData(model->mj, data->mj);
    } else {
        mjmlx_reset_data(model->mlx, data->mlx);
    }
}

// ── Simulation ──────────────────────────────────────────────────────────

MJB_API void mjb_step(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) mj_step(model->mj, data->mj);
    else mjmlx_step(model->mlx, data->mlx);
}

MJB_API void mjb_forward(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) mj_forward(model->mj, data->mj);
    else mjmlx_forward(model->mlx, data->mlx);
}

MJB_API void mjb_step1(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) mj_step1(model->mj, data->mj);
    else mjmlx_step1(model->mlx, data->mlx);
}

MJB_API void mjb_step2(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) mj_step2(model->mj, data->mj);
    else mjmlx_step2(model->mlx, data->mlx);
}

MJB_API void mjb_kinematics(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) mj_kinematics(model->mj, data->mj);
    else mjmlx_kinematics(model->mlx, data->mlx);
}

MJB_API void mjb_rne_post_constraint(MjbModel* model, MjbData* data) {
    if (!model || !data) return;
    if (data->type == MJB_BACKEND_CPU) {
        mj_rnePostConstraint(model->mj, data->mj);
    } else {
        mjmlx_rne_post_constraint(model->mlx, data->mlx);
    }
}

// ── State access ────────────────────────────────────────────────────────

MJB_API void mjb_set_qpos(MjbData* data, const float* qpos, int n) {
    if (!data || !qpos || n <= 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        for (int i = 0; i < n; i++) data->mj->qpos[i] = qpos[i];
    } else {
        mjmlx_set_qpos(data->mlx, qpos, n);
    }
}

MJB_API void mjb_set_qvel(MjbData* data, const float* qvel, int n) {
    if (!data || !qvel || n <= 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        for (int i = 0; i < n; i++) data->mj->qvel[i] = qvel[i];
    } else {
        mjmlx_set_qvel(data->mlx, qvel, n);
    }
}

MJB_API void mjb_set_ctrl(MjbData* data, const float* ctrl, int n) {
    if (!data || !ctrl || n <= 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        for (int i = 0; i < n; i++) data->mj->ctrl[i] = ctrl[i];
    } else {
        mjmlx_set_ctrl(data->mlx, ctrl, n);
    }
}

MJB_API const float* mjb_get_qpos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->qpos, data->model_ref->mj->nq, n_out, data->fbuf);
    return mjmlx_get_qpos(data->mlx, n_out);
}

MJB_API const float* mjb_get_qvel(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->qvel, data->model_ref->mj->nv, n_out, data->fbuf);
    return mjmlx_get_qvel(data->mlx, n_out);
}

MJB_API const float* mjb_get_ctrl(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->ctrl, data->model_ref->mj->nu, n_out, data->fbuf);
    return mjmlx_get_ctrl(data->mlx, n_out);
}

MJB_API const float* mjb_get_xpos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->xpos, data->model_ref->mj->nbody * 3, n_out, data->fbuf);
    return mjmlx_get_xpos(data->mlx, n_out);
}

MJB_API const float* mjb_get_xquat(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->xquat, data->model_ref->mj->nbody * 4, n_out, data->fbuf);
    return mjmlx_get_xquat(data->mlx, n_out);
}

MJB_API const float* mjb_get_xipos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->xipos, data->model_ref->mj->nbody * 3, n_out, data->fbuf);
    return mjmlx_get_xipos(data->mlx, n_out);
}

MJB_API const float* mjb_get_cvel(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->cvel, data->model_ref->mj->nbody * 6, n_out, data->fbuf);
    return mjmlx_get_cvel(data->mlx, n_out);
}

MJB_API const float* mjb_get_qfrc_actuator(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->qfrc_actuator, data->model_ref->mj->nv, n_out, data->fbuf);
    return mjmlx_get_qfrc_actuator(data->mlx, n_out);
}

MJB_API const float* mjb_get_subtree_com(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->subtree_com, data->model_ref->mj->nbody * 3, n_out, data->fbuf);
    return mjmlx_get_subtree_com(data->mlx, n_out);
}

MJB_API const float* mjb_get_cinert(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->cinert, data->model_ref->mj->nbody * 10, n_out, data->fbuf);
    return mjmlx_get_cinert(data->mlx, n_out);
}

MJB_API const float* mjb_get_cfrc_ext(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->cfrc_ext, data->model_ref->mj->nbody * 6, n_out, data->fbuf);
    return mjmlx_get_cfrc_ext(data->mlx, n_out);
}

MJB_API const float* mjb_get_geom_xpos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->geom_xpos, data->model_ref->mj->ngeom * 3, n_out, data->fbuf);
    // MLX: fall back to C model's data (not computed by MLX)
    return nullptr;
}

MJB_API const float* mjb_get_geom_xmat(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->geom_xmat, data->model_ref->mj->ngeom * 9, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_sensordata(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->sensordata, data->model_ref->mj->nsensordata, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_model_geom_pos(const MjbModel* model, int* n_out) {
    const mjModel* m = get_mj_model(model);
    if (!m) { if (n_out) *n_out = 0; return nullptr; }
    return cpu_get(m->geom_pos, m->ngeom * 3, n_out, model->fbuf);
}

MJB_API const float* mjb_model_geom_quat(const MjbModel* model, int* n_out) {
    const mjModel* m = get_mj_model(model);
    if (!m) { if (n_out) *n_out = 0; return nullptr; }
    return cpu_get(m->geom_quat, m->ngeom * 4, n_out, model->fbuf);
}

// ── Additional data getters for component binding ───────────────────────

MJB_API const float* mjb_get_xaxis(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->xaxis, data->model_ref->mj->njnt * 3, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_site_xpos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->site_xpos, data->model_ref->mj->nsite * 3, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_site_xmat(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->site_xmat, data->model_ref->mj->nsite * 9, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_actuator_length(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->actuator_length, data->model_ref->mj->nu, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_actuator_velocity(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->actuator_velocity, data->model_ref->mj->nu, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_actuator_force(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->actuator_force, data->model_ref->mj->nu, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_mocap_pos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->mocap_pos, data->model_ref->mj->nmocap * 3, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_mocap_quat(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->mocap_quat, data->model_ref->mj->nmocap * 4, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_ten_length(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->ten_length, data->model_ref->mj->ntendon, n_out, data->fbuf);
    return nullptr;
}

MJB_API const float* mjb_get_wrap_xpos(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU) {
        int nwrap = (int)data->model_ref->mj->nwrap;
        return cpu_get(data->mj->wrap_xpos, nwrap * 6, n_out, data->fbuf);
    }
    return nullptr;
}

MJB_API const int* mjb_get_ten_wrapadr(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU) {
        int nt = (int)data->model_ref->mj->ntendon;
        if (n_out) *n_out = nt;
        return data->mj->ten_wrapadr;
    }
    return nullptr;
}

MJB_API const int* mjb_get_ten_wrapnum(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU) {
        int nt = (int)data->model_ref->mj->ntendon;
        if (n_out) *n_out = nt;
        return data->mj->ten_wrapnum;
    }
    return nullptr;
}

MJB_API const int* mjb_get_wrap_obj(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU) {
        int nwrap = (int)data->model_ref->mj->nwrap;
        if (n_out) *n_out = nwrap;
        return data->mj->wrap_obj;
    }
    return nullptr;
}

MJB_API void mjb_set_mocap_pos(MjbData* data, const float* pos, int n) {
    if (!data || !pos || n <= 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        for (int i = 0; i < n; i++) data->mj->mocap_pos[i] = pos[i];
    }
}

MJB_API void mjb_set_mocap_quat(MjbData* data, const float* quat, int n) {
    if (!data || !quat || n <= 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        for (int i = 0; i < n; i++) data->mj->mocap_quat[i] = quat[i];
    }
}

// ── Per-index state setters ─────────────────────────────────────────────

MJB_API void mjb_set_qpos_at(MjbData* data, int index, float value) {
    if (!data || index < 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        if (index < data->model_ref->mj->nq)
            data->mj->qpos[index] = (double)value;
    }
}

MJB_API void mjb_set_qvel_at(MjbData* data, int index, float value) {
    if (!data || index < 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        if (index < data->model_ref->mj->nv)
            data->mj->qvel[index] = (double)value;
    }
}

MJB_API void mjb_set_ctrl_at(MjbData* data, int index, float value) {
    if (!data || index < 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        if (index < data->model_ref->mj->nu)
            data->mj->ctrl[index] = (double)value;
    }
}

// ── xfrc_applied ────────────────────────────────────────────────────────

MJB_API const float* mjb_get_xfrc_applied(const MjbData* data, int* n_out) {
    if (!data) { if (n_out) *n_out = 0; return nullptr; }
    if (data->type == MJB_BACKEND_CPU)
        return cpu_get(data->mj->xfrc_applied, data->model_ref->mj->nbody * 6, n_out, data->fbuf);
    return nullptr;
}

MJB_API void mjb_set_xfrc_applied(MjbData* data, const float* values, int n) {
    if (!data || !values || n <= 0) return;
    if (data->type == MJB_BACKEND_CPU) {
        int max_n = data->model_ref->mj->nbody * 6;
        int count = n < max_n ? n : max_n;
        for (int i = 0; i < count; i++) data->mj->xfrc_applied[i] = (double)values[i];
    }
}

// ── Warnings / diagnostics ──────────────────────────────────────────────

MJB_API int mjb_get_warning_count(const MjbData* data, int index) {
    if (!data || index < 0 || index >= mjNWARNING) return 0;
    if (data->type == MJB_BACKEND_CPU) return data->mj->warning[index].number;
    return 0;
}

// ── Model I/O (save) ────────────────────────────────────────────────────

MJB_API int mjb_save_last_xml(const MjbModel* model, const char* path,
                              char* error_buf, int error_buf_size) {
    const mjModel* m = get_mj_model(model);
    if (!m || !path) return -1;
    mj_saveLastXML(path, const_cast<mjModel*>(m), error_buf, error_buf_size);
    if (error_buf && error_buf[0] != '\0') return -1;
    return 0;
}

// ── Utility wrappers ────────────────────────────────────────────────────

MJB_API void mjb_object_velocity(const MjbModel* model, const MjbData* data,
                                 int objtype, int objid, int flg_local,
                                 float* result6) {
    if (!model || !data || !result6) return;
    if (data->type == MJB_BACKEND_CPU) {
        double res[6];
        mj_objectVelocity(model->mj, data->mj, objtype, objid, res, flg_local);
        for (int i = 0; i < 6; i++) result6[i] = (float)res[i];
    }
}

MJB_API void mjb_load_plugin_library(const char* path) {
    if (path) mj_loadPluginLibrary(path);
}

MJB_API void mjb_model_set_hfield_data(MjbModel* model, int offset,
                                       const float* values, int n) {
    if (!model || !values || n <= 0 || offset < 0) return;
    mjModel* m = nullptr;
    if (model->type == MJB_BACKEND_CPU) m = model->mj;
    else m = const_cast<mjModel*>(static_cast<const mjModel*>(mjmlx_get_mj_model(model->mlx)));
    if (!m) return;
    int max_n = (int)m->nhfielddata - offset;
    if (max_n <= 0) return;
    int count = n < max_n ? n : max_n;
    for (int i = 0; i < count; i++) m->hfield_data[offset + i] = values[i];
}

// ── Batched simulation ──────────────────────────────────────────────────

MJB_API MjbBatchedSim* mjb_batched_create(MjbModel* model, const MjbBatchedConfig* config) {
    if (!model || !config || config->num_envs <= 0) return nullptr;
    auto* sim = new MjbBatchedSim();
    sim->type = model->type;
    sim->model_ref = model;
    sim->num_envs = config->num_envs;

    try {
        if (model->type == MJB_BACKEND_CPU) {
            sim->cpu_datas.resize(config->num_envs);
            for (int i = 0; i < config->num_envs; i++) {
                sim->cpu_datas[i] = mj_makeData(model->mj);
                if (!sim->cpu_datas[i]) { delete sim; return nullptr; }
            }
            if (config->solver_iterations > 0) {
                model->mj->opt.iterations = config->solver_iterations;
            }
        } else {
            MjmlxBatchedConfig mlx_config = {};
            mlx_config.num_envs = config->num_envs;
            mlx_config.foot_contacts_only = config->foot_contacts_only;
            mlx_config.integrator = MJMLX_INTEGRATOR_EULER;
            mlx_config.use_gpu = 1;
            mlx_config.solver_iterations = config->solver_iterations;

            sim->mlx_sim = mjmlx_batched_create(model->mlx, &mlx_config);
            sim->gpu_active = (sim->mlx_sim != nullptr);

            if (!sim->mlx_sim) {
                fprintf(stderr, "mjb_batched_create: GPU batched sim failed, "
                        "falling back to CPU batched mode.\n");
                sim->type = MJB_BACKEND_CPU;
                sim->model_ref = model;
                const mjModel* cmj = get_mj_model(model);
                if (!cmj) { delete sim; return nullptr; }
                mjModel* mj = const_cast<mjModel*>(cmj);
                sim->cpu_datas.resize(config->num_envs);
                for (int i = 0; i < config->num_envs; i++) {
                    sim->cpu_datas[i] = mj_makeData(mj);
                    if (!sim->cpu_datas[i]) { delete sim; return nullptr; }
                }
                if (config->solver_iterations > 0)
                    mj->opt.iterations = config->solver_iterations;
            }
        }
    } catch (...) {
        delete sim;
        return nullptr;
    }
    return sim;
}

MJB_API int mjb_batched_is_gpu(const MjbBatchedSim* sim) {
    if (!sim) return 0;
    return sim->gpu_active ? 1 : 0;
}

MJB_API void mjb_batched_free(MjbBatchedSim* sim) {
    delete sim;
}

MJB_API void mjb_batched_step(MjbBatchedSim* sim, const float* ctrl) {
    if (!sim || !ctrl) return;

    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* cm = get_mj_model(sim->model_ref);
        mjModel* m = const_cast<mjModel*>(cm);
        int ne = sim->num_envs;
        int nu = cm->nu;

        dispatch_apply((size_t)ne,
            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
            ^(size_t i) {
                mjData* d = sim->cpu_datas[i];
                for (int j = 0; j < nu; j++)
                    d->ctrl[j] = (double)ctrl[i * nu + j];
                mj_step(m, d);
            }
        );
    } else {
        mjmlx_batched_step(sim->mlx_sim, ctrl);
    }
}

MJB_API void mjb_batched_reset(MjbBatchedSim* sim, const int* reset_mask) {
    if (!sim || !reset_mask) return;

    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* cm = get_mj_model(sim->model_ref);
        mjModel* m = const_cast<mjModel*>(cm);
        for (int i = 0; i < sim->num_envs; i++) {
            if (reset_mask[i]) {
                mj_resetData(m, sim->cpu_datas[i]);
            }
        }
    } else {
        mjmlx_batched_reset(sim->mlx_sim, reset_mask);
    }
}

// CPU batched state: gather from N mjData* into contiguous float buffer
static const float* cpu_batched_gather(
    const MjbBatchedSim* sim,
    const double* (getter)(const mjData*),
    int per_env, int* n_out,
    std::vector<float>& buf)
{
    int ne = sim->num_envs;
    int total = ne * per_env;
    buf.resize(total);
    for (int i = 0; i < ne; i++) {
        const double* src = getter(sim->cpu_datas[i]);
        for (int j = 0; j < per_env; j++)
            buf[i * per_env + j] = (float)src[j];
    }
    if (n_out) *n_out = total;
    return buf.data();
}

// Macros for CPU batched getters
#define CPU_BATCHED_GET(field, per_env) \
    cpu_batched_gather(sim, [](const mjData* d) -> const double* { return d->field; }, \
                       per_env, n_out, sim->field##_buf)

MJB_API const float* mjb_batched_get_qpos(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(qpos, m->nq);
    }
    return mjmlx_batched_get_qpos(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_qvel(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(qvel, m->nv);
    }
    return mjmlx_batched_get_qvel(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_xpos(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(xpos, m->nbody * 3);
    }
    return mjmlx_batched_get_xpos(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_subtree_com(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(subtree_com, m->nbody * 3);
    }
    return mjmlx_batched_get_subtree_com(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_cinert(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(cinert, m->nbody * 10);
    }
    return mjmlx_batched_get_cinert(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_cvel(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(cvel, m->nbody * 6);
    }
    return mjmlx_batched_get_cvel(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_qfrc_actuator(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(qfrc_actuator, m->nv);
    }
    return mjmlx_batched_get_qfrc_actuator(sim->mlx_sim, n_out);
}

MJB_API const float* mjb_batched_get_cfrc_ext(const MjbBatchedSim* sim, int* n_out) {
    if (!sim) { if (n_out) *n_out = 0; return nullptr; }
    if (sim->type == MJB_BACKEND_CPU) {
        const mjModel* m = get_mj_model(sim->model_ref);
        return CPU_BATCHED_GET(cfrc_ext, m->nbody * 6);
    }
    return mjmlx_batched_get_cfrc_ext(sim->mlx_sim, n_out);
}

#undef CPU_BATCHED_GET

MJB_API void mjb_batched_set_env_qpos(MjbBatchedSim* sim, int env_idx,
                                      const float* qpos, int nq) {
    if (!sim || !qpos || env_idx < 0 || env_idx >= sim->num_envs) return;

    if (sim->type == MJB_BACKEND_CPU) {
        mjData* d = sim->cpu_datas[env_idx];
        for (int i = 0; i < nq; i++) d->qpos[i] = (double)qpos[i];
    } else {
        mjmlx_batched_set_env_qpos(sim->mlx_sim, env_idx, qpos, nq);
    }
}

MJB_API void mjb_batched_set_env_qvel(MjbBatchedSim* sim, int env_idx,
                                      const float* qvel, int nv) {
    if (!sim || !qvel || env_idx < 0 || env_idx >= sim->num_envs) return;

    if (sim->type == MJB_BACKEND_CPU) {
        mjData* d = sim->cpu_datas[env_idx];
        for (int i = 0; i < nv; i++) d->qvel[i] = (double)qvel[i];
    } else {
        mjmlx_batched_set_env_qvel(sim->mlx_sim, env_idx, qvel, nv);
    }
}

MJB_API void mjb_batched_eval_state(const MjbBatchedSim* sim) {
    if (!sim) return;
    if (sim->type != MJB_BACKEND_CPU && sim->mlx_sim)
        mjmlx_batched_eval_state(sim->mlx_sim);
    // CPU backend: state arrays are always materialized — no-op
}

// ── Differentiable simulation ───────────────────────────────────────────

MJB_API int mjb_grad_step(MjbModel* model, MjbData* data, float* grad_out) {
    if (!model || !data || !grad_out) return -1;
    if (model->type == MJB_BACKEND_CPU) return -1;  // CPU doesn't support grad
    mjmlx_grad_step(model->mlx, data->mlx, grad_out);
    return 0;
}

}  // extern "C"
