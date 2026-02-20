// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// nanobind Python extension for MuJoCo-MLX-Cpp.
// Exposes Model, Data, BatchedSim as Python classes with mx.array properties.
//
// Uses NB_DOMAIN "mlx" so mlx::core::array <-> Python mx.array conversion
// is handled automatically by MLX's registered type casters.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "internal.h"
#include "mjmlx/mjmlx.h"
#include "mjmlx/mjmlx_types.h"

namespace nb = nanobind;
namespace mx = mlx::core;

// ── Python wrapper classes ───────────────────────────────────────────────────

struct PyModel {
    std::shared_ptr<MjmlxModel> handle;
    PyModel(std::shared_ptr<MjmlxModel> h) : handle(std::move(h)) {}
    const mjmlx::Model& m() const { return handle->model; }
    mjmlx::Model& m_mut() { return handle->model; }
};

struct PyData {
    std::shared_ptr<MjmlxData> handle;
    PyData(std::shared_ptr<MjmlxData> h) : handle(std::move(h)) {}
    const mjmlx::Data& d() const { return handle->data; }
    mjmlx::Data& d_mut() { return handle->data; }
};

struct PyBatchedSim {
    std::shared_ptr<MjmlxBatchedSim> handle;
    std::shared_ptr<MjmlxModel> model_ref;
    int num_envs;
    int frame_skip;

    PyBatchedSim(std::shared_ptr<MjmlxBatchedSim> h,
                 std::shared_ptr<MjmlxModel> m, int n, int fs)
        : handle(std::move(h)), model_ref(std::move(m)), num_envs(n), frame_skip(fs) {}

    mjmlx::BatchedSim& sim() { return handle->sim; }
    const mjmlx::BatchedSim& sim() const { return handle->sim; }
};

// ── Module definition ────────────────────────────────────────────────────────

NB_MODULE(_mjmlx_native, m) {
    m.doc() = "MuJoCo-MLX C++ physics engine (nanobind extension)";

    // ── Model ────────────────────────────────────────────────────────────────

    nb::class_<PyModel>(m, "Model")
        // Dimensions
        .def_prop_ro("nq", [](const PyModel& m) { return m.m().nq; })
        .def_prop_ro("nv", [](const PyModel& m) { return m.m().nv; })
        .def_prop_ro("nu", [](const PyModel& m) { return m.m().nu; })
        .def_prop_ro("na", [](const PyModel& m) { return m.m().na; })
        .def_prop_ro("nbody", [](const PyModel& m) { return m.m().nbody; })
        .def_prop_ro("njnt", [](const PyModel& m) { return m.m().njnt; })
        .def_prop_ro("ngeom", [](const PyModel& m) { return m.m().ngeom; })
        .def_prop_ro("nsite", [](const PyModel& m) { return m.m().nsite; })
        .def_prop_ro("npair", [](const PyModel& m) { return m.m().npair; })

        // Option scalars (read-write for config overrides)
        .def_prop_rw("opt_timestep",
            [](const PyModel& m) { return m.m().opt.timestep; },
            [](PyModel& m, float v) { m.m_mut().opt.timestep = v; })
        .def_prop_ro("opt_gravity", [](const PyModel& m) { return m.m().opt.gravity; })
        .def_prop_rw("opt_iterations",
            [](const PyModel& m) { return m.m().opt.iterations; },
            [](PyModel& m, int v) { m.m_mut().opt.iterations = v; })
        .def_prop_rw("opt_integrator",
            [](const PyModel& m) { return static_cast<int>(m.m().opt.integrator); },
            [](PyModel& m, int v) { m.m_mut().opt.integrator = static_cast<mjmlx::IntegratorType>(v); })
        .def_prop_rw("opt_solver",
            [](const PyModel& m) { return static_cast<int>(m.m().opt.solver); },
            [](PyModel& m, int v) { m.m_mut().opt.solver = static_cast<mjmlx::SolverType>(v); })
        .def_prop_rw("opt_disableflags",
            [](const PyModel& m) { return m.m().opt.disableflags; },
            [](PyModel& m, int v) { m.m_mut().opt.disableflags = v; })

        // Statistics
        .def_prop_ro("stat_meaninertia", [](const PyModel& m) { return m.m().stat.meaninertia; })
        .def_prop_ro("stat_meanmass", [](const PyModel& m) { return m.m().stat.meanmass; })
        .def_prop_ro("stat_extent", [](const PyModel& m) { return m.m().stat.extent; })

        // Reference configuration
        .def_prop_ro("qpos0", [](const PyModel& m) { return m.m().qpos0; })

        // Key body arrays
        .def_prop_ro("body_parentid", [](const PyModel& m) { return m.m().body_parentid; })
        .def_prop_ro("body_pos", [](const PyModel& m) { return m.m().body_pos; })
        .def_prop_ro("body_quat", [](const PyModel& m) { return m.m().body_quat; })
        .def_prop_ro("body_mass", [](const PyModel& m) { return m.m().body_mass; })
        .def_prop_ro("body_subtreemass", [](const PyModel& m) { return m.m().body_subtreemass; })
        .def_prop_ro("body_inertia", [](const PyModel& m) { return m.m().body_inertia; })
        .def_prop_ro("body_ipos", [](const PyModel& m) { return m.m().body_ipos; })
        .def_prop_ro("body_iquat", [](const PyModel& m) { return m.m().body_iquat; })
        .def_prop_ro("body_jntadr", [](const PyModel& m) { return m.m().body_jntadr; })
        .def_prop_ro("body_jntnum", [](const PyModel& m) { return m.m().body_jntnum; })
        .def_prop_ro("body_dofadr", [](const PyModel& m) { return m.m().body_dofadr; })
        .def_prop_ro("body_dofnum", [](const PyModel& m) { return m.m().body_dofnum; })
        .def_prop_ro("body_invweight0", [](const PyModel& m) { return m.m().body_invweight0; })

        // Joint arrays
        .def_prop_ro("jnt_type", [](const PyModel& m) { return m.m().jnt_type; })
        .def_prop_ro("jnt_bodyid", [](const PyModel& m) { return m.m().jnt_bodyid; })
        .def_prop_ro("jnt_qposadr", [](const PyModel& m) { return m.m().jnt_qposadr; })
        .def_prop_ro("jnt_dofadr", [](const PyModel& m) { return m.m().jnt_dofadr; })
        .def_prop_ro("jnt_range", [](const PyModel& m) { return m.m().jnt_range; })
        .def_prop_ro("jnt_limited", [](const PyModel& m) { return m.m().jnt_limited; })
        .def_prop_ro("jnt_axis", [](const PyModel& m) { return m.m().jnt_axis; })
        .def_prop_ro("jnt_stiffness", [](const PyModel& m) { return m.m().jnt_stiffness; })

        // DOF arrays
        .def_prop_ro("dof_bodyid", [](const PyModel& m) { return m.m().dof_bodyid; })
        .def_prop_ro("dof_armature", [](const PyModel& m) { return m.m().dof_armature; })
        .def_prop_ro("dof_damping", [](const PyModel& m) { return m.m().dof_damping; })
        .def_prop_ro("dof_invweight0", [](const PyModel& m) { return m.m().dof_invweight0; })

        // Actuator arrays
        .def_prop_ro("actuator_ctrlrange", [](const PyModel& m) { return m.m().actuator_ctrlrange; })
        .def_prop_ro("actuator_gear", [](const PyModel& m) { return m.m().actuator_gear; })

        // Geom arrays
        .def_prop_ro("geom_type", [](const PyModel& m) { return m.m().geom_type; })
        .def_prop_ro("geom_bodyid", [](const PyModel& m) { return m.m().geom_bodyid; })
        .def_prop_ro("geom_size", [](const PyModel& m) { return m.m().geom_size; })
    ;

    // ── Data ─────────────────────────────────────────────────────────────────

    nb::class_<PyData>(m, "Data")
        // State (read-write)
        .def_prop_rw("qpos",
            [](const PyData& d) { return d.d().qpos; },
            [](PyData& d, mx::array v) { d.d_mut().qpos = v; })
        .def_prop_rw("qvel",
            [](const PyData& d) { return d.d().qvel; },
            [](PyData& d, mx::array v) { d.d_mut().qvel = v; })
        .def_prop_rw("ctrl",
            [](const PyData& d) { return d.d().ctrl; },
            [](PyData& d, mx::array v) { d.d_mut().ctrl = v; })
        .def_prop_ro("qacc", [](const PyData& d) { return d.d().qacc; })
        .def_prop_ro("time", [](const PyData& d) { return d.d().time; })

        // Kinematics
        .def_prop_ro("xpos", [](const PyData& d) { return d.d().xpos; })
        .def_prop_ro("xquat", [](const PyData& d) { return d.d().xquat; })
        .def_prop_ro("xmat", [](const PyData& d) { return d.d().xmat; })
        .def_prop_ro("xipos", [](const PyData& d) { return d.d().xipos; })
        .def_prop_ro("geom_xpos", [](const PyData& d) { return d.d().geom_xpos; })
        .def_prop_ro("site_xpos", [](const PyData& d) { return d.d().site_xpos; })

        // Dynamics
        .def_prop_ro("subtree_com", [](const PyData& d) { return d.d().subtree_com; })
        .def_prop_ro("cinert", [](const PyData& d) { return d.d().cinert; })
        .def_prop_ro("cvel", [](const PyData& d) { return d.d().cvel; })
        .def_prop_ro("cdof", [](const PyData& d) { return d.d().cdof; })
        .def_prop_ro("qM", [](const PyData& d) { return d.d().qM; })
        .def_prop_ro("qfrc_bias", [](const PyData& d) { return d.d().qfrc_bias; })
        .def_prop_ro("qfrc_passive", [](const PyData& d) { return d.d().qfrc_passive; })
        .def_prop_ro("qfrc_actuator", [](const PyData& d) { return d.d().qfrc_actuator; })
        .def_prop_ro("qfrc_smooth", [](const PyData& d) { return d.d().qfrc_smooth; })
        .def_prop_ro("qacc_smooth", [](const PyData& d) { return d.d().qacc_smooth; })
        .def_prop_ro("qfrc_constraint", [](const PyData& d) { return d.d().qfrc_constraint; })

        // Applied forces (read-write)
        .def_prop_rw("xfrc_applied",
            [](const PyData& d) { return d.d().xfrc_applied; },
            [](PyData& d, mx::array v) { d.d_mut().xfrc_applied = v; })

        // Constraint info
        .def_prop_ro("ncon", [](const PyData& d) { return d.d().ncon; })
        .def_prop_ro("nefc", [](const PyData& d) { return d.d().nefc; })
        .def_prop_ro("efc_force", [](const PyData& d) { return d.d().efc_force; })
    ;

    // ── BatchedSim ───────────────────────────────────────────────────────────

    nb::class_<PyBatchedSim>(m, "BatchedSim")
        .def_prop_ro("num_envs", [](const PyBatchedSim& s) { return s.num_envs; })
        .def_prop_ro("frame_skip", [](const PyBatchedSim& s) { return s.frame_skip; })
        .def_prop_ro("cpu_mode", [](const PyBatchedSim& s) { return s.handle->cpu_mode; })

        // Step: advance all environments by frame_skip physics substeps
        .def("step", [](PyBatchedSim& s, nb::object ctrl_obj) {
            auto* handle = &*s.handle;

            if (handle->cpu_mode) {
                // CPU path: dispatch_apply over N mjData*
                auto& sim = s.sim();
                int B = s.num_envs;
                int nu = handle->cpu_model->nu;

                // Sync qpos/qvel to mjData (in case set_qpos/set_qvel were called)
                cpu_sync_state(handle);

                // Extract ctrl as flat float* for the C stepping function
                const float* ctrl_ptr = nullptr;
                std::vector<float> ctrl_buf;
                if (!ctrl_obj.is_none() && nu > 0) {
                    mx::array ctrl = nb::cast<mx::array>(ctrl_obj);
                    mx::eval(ctrl);
                    const float* src = ctrl.data<float>();
                    ctrl_buf.assign(src, src + B * nu);
                    ctrl_ptr = ctrl_buf.data();
                }

                cpu_batched_step(handle, ctrl_ptr, s.frame_skip);
                cpu_gather_state(handle);
                return;
            }

            // GPU path: compiled + vmapped MLX step
            auto& sim = s.sim();
            int B = s.num_envs;
            int nu = sim.model->nu;
            int ctrl_dim = std::max(1, nu);

            mx::array ctrl = (ctrl_obj.is_none() || nu == 0)
                ? mx::zeros({B, ctrl_dim})
                : nb::cast<mx::array>(ctrl_obj);

            for (int i = 0; i < s.frame_skip; i++) {
                auto results = sim.compiled_step({sim.qpos, sim.qvel, ctrl});
                sim.qpos = results[0];
                sim.qvel = results[1];
                if (results.size() > 2) sim.xpos = results[2];
                if (results.size() > 3) sim.subtree_com = results[3];
                if (results.size() > 4) sim.cinert = results[4];
                if (results.size() > 5) sim.cvel = results[5];
                if (results.size() > 6) sim.qfrc_actuator = results[6];
                if (results.size() > 7) sim.cfrc_ext = results[7];
            }
        }, nb::arg("ctrl") = nb::none())

        // Reset environments where mask[i] != 0
        .def("reset", [](PyBatchedSim& s, mx::array mask) {
            auto* handle = &*s.handle;
            auto& sim = s.sim();
            int B = s.num_envs;

            mx::eval(mask);
            auto mask_data = mask.data<int32_t>();

            if (handle->cpu_mode) {
                mjModel* m = handle->cpu_model;
                for (int i = 0; i < B; i++) {
                    if (mask_data[i])
                        mj_resetData(m, handle->cpu_datas[i]);
                }
                cpu_gather_state(handle);
                return;
            }

            int nq = sim.model->nq, nv = sim.model->nv;
            mx::eval(sim.qpos);
            mx::eval(sim.qvel);

            auto qp = std::vector<float>(sim.qpos.data<float>(),
                                          sim.qpos.data<float>() + B * nq);
            auto qv = std::vector<float>(sim.qvel.data<float>(),
                                          sim.qvel.data<float>() + B * nv);
            mx::eval(sim.model->qpos0);
            auto q0 = sim.model->qpos0.data<float>();

            for (int i = 0; i < B; i++) {
                if (mask_data[i]) {
                    std::memcpy(&qp[i * nq], q0, nq * sizeof(float));
                    std::memset(&qv[i * nv], 0, nv * sizeof(float));
                }
            }
            sim.qpos = mx::reshape(mx::array(qp.data(), {B * nq}, mx::float32), {B, nq});
            sim.qvel = mx::reshape(mx::array(qv.data(), {B * nv}, mx::float32), {B, nv});
        })

        // State accessors (zero-copy mx.array)
        .def_prop_ro("qpos", [](const PyBatchedSim& s) { return s.sim().qpos; })
        .def_prop_ro("qvel", [](const PyBatchedSim& s) { return s.sim().qvel; })
        .def_prop_ro("xpos", [](const PyBatchedSim& s) { return s.sim().xpos; })
        .def_prop_ro("subtree_com", [](const PyBatchedSim& s) { return s.sim().subtree_com; })
        .def_prop_ro("cinert", [](const PyBatchedSim& s) { return s.sim().cinert; })
        .def_prop_ro("cvel", [](const PyBatchedSim& s) { return s.sim().cvel; })
        .def_prop_ro("qfrc_actuator", [](const PyBatchedSim& s) { return s.sim().qfrc_actuator; })
        .def_prop_ro("cfrc_ext", [](const PyBatchedSim& s) { return s.sim().cfrc_ext; })

        // Direct state setters (for initialization with noise)
        .def("set_qpos", [](PyBatchedSim& s, mx::array qpos) {
            s.sim().qpos = qpos;
            if (s.handle->cpu_mode) {
                mx::eval(qpos);
                const float* p = qpos.data<float>();
                int nq = s.handle->cpu_model->nq;
                for (int i = 0; i < s.num_envs; i++) {
                    mjData* d = s.handle->cpu_datas[i];
                    for (int j = 0; j < nq; j++) d->qpos[j] = (double)p[i*nq + j];
                }
            }
        })
        .def("set_qvel", [](PyBatchedSim& s, mx::array qvel) {
            s.sim().qvel = qvel;
            if (s.handle->cpu_mode) {
                mx::eval(qvel);
                const float* v = qvel.data<float>();
                int nv = s.handle->cpu_model->nv;
                for (int i = 0; i < s.num_envs; i++) {
                    mjData* d = s.handle->cpu_datas[i];
                    for (int j = 0; j < nv; j++) d->qvel[j] = (double)v[i*nv + j];
                }
            }
        })
    ;

    // ── Free functions ───────────────────────────────────────────────────────

    m.def("load_model", [](const std::string& xml_path, bool foot_contacts_only) -> PyModel {
        MjmlxModel* raw = foot_contacts_only
            ? mjmlx_load_model_filtered(xml_path.c_str(), 1)
            : mjmlx_load_model(xml_path.c_str());
        if (!raw) throw std::runtime_error("Failed to load model: " + xml_path);
        return PyModel(std::shared_ptr<MjmlxModel>(raw, mjmlx_free_model));
    }, nb::arg("xml_path"), nb::arg("foot_contacts_only") = false,
       "Load a MuJoCo model from an MJCF XML file.");

    m.def("load_model_from_string", [](const std::string& xml_string) -> PyModel {
        MjmlxModel* raw = mjmlx_load_model_from_string(xml_string.c_str());
        if (!raw) throw std::runtime_error("Failed to load model from string");
        return PyModel(std::shared_ptr<MjmlxModel>(raw, mjmlx_free_model));
    }, nb::arg("xml_string"),
       "Load a MuJoCo model from an MJCF XML string.");

    m.def("make_data", [](PyModel& model) -> PyData {
        MjmlxData* raw = mjmlx_make_data(&*model.handle);
        if (!raw) throw std::runtime_error("Failed to create data");
        return PyData(std::shared_ptr<MjmlxData>(raw, mjmlx_free_data));
    }, nb::arg("model"),
       "Create simulation data for a model.");

    m.def("forward", [](PyModel& model, PyData& data) {
        mjmlx_forward(&*model.handle, &*data.handle);
    }, nb::arg("model"), nb::arg("data"),
       "Run full forward kinematics + dynamics.");

    m.def("step", [](PyModel& model, PyData& data) {
        mjmlx_step(&*model.handle, &*data.handle);
    }, nb::arg("model"), nb::arg("data"),
       "Advance simulation by one timestep.");

    m.def("create_batched", [](PyModel& model, int num_envs, int frame_skip,
                                bool use_gpu, int solver_iterations) -> PyBatchedSim {
        MjmlxBatchedConfig config;
        config.num_envs = num_envs;
        config.use_gpu = use_gpu ? 1 : 0;
        config.foot_contacts_only = 0;
        config.integrator = static_cast<MjmlxIntegrator>(model.m().opt.integrator);
        config.solver_iterations = solver_iterations;

        auto* raw = mjmlx_batched_create(&*model.handle, &config);
        if (!raw) throw std::runtime_error("Failed to create batched sim");

        return PyBatchedSim(
            std::shared_ptr<MjmlxBatchedSim>(raw, mjmlx_batched_free),
            model.handle,
            num_envs,
            frame_skip);
    }, nb::arg("model"), nb::arg("num_envs"),
       nb::arg("frame_skip") = 1, nb::arg("use_gpu") = true,
       nb::arg("solver_iterations") = 0,
       "Create a batched simulation with N parallel environments.");

    m.def("version", []() -> std::string {
        return mjmlx_version();
    }, "Get library version string.");

    // ── Constants ────────────────────────────────────────────────────────────

    m.attr("INTEGRATOR_EULER") = 0;
    m.attr("INTEGRATOR_RK4") = 1;
    m.attr("SOLVER_PGS") = 0;
    m.attr("SOLVER_CG") = 1;
    m.attr("SOLVER_NEWTON") = 2;
    m.attr("DISABLE_CONTACT") = (1 << 4);
    m.attr("DISABLE_EULERDAMP") = (1 << 15);
}
