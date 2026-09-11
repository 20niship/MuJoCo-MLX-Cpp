// Minimal nanobind module for RL training: wraps mjmlx's plain C API (mjmlx.h) with numpy ndarrays instead of mlx::core::array, sidestepping bindings.cpp's NB_DOMAIN "mlx" cross-extension type sharing (requires the exact nanobind build mlx's own wheel used, which a locally pip-installed nanobind doesn't match in practice).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "mjmlx/mjmlx.h"
#include "mjmlx/mjmlx_types.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

using FloatArray = nb::ndarray<float, nb::numpy, nb::c_contig>;
using IntArray = nb::ndarray<int32_t, nb::numpy, nb::c_contig>;

namespace {

// Copies an mjmlx C-API float* (num_envs * dim) into a freshly owned numpy array.
FloatArray copy_out(const float* src, int n, int num_envs) {
    if (num_envs <= 0 || n <= 0) throw std::runtime_error("mjmlx_rl: empty state array");
    int dim = n / num_envs;
    float* buf = new float[static_cast<size_t>(n)];
    std::copy(src, src + n, buf);
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    size_t shape[2] = {static_cast<size_t>(num_envs), static_cast<size_t>(dim)};
    return FloatArray(buf, 2, shape, owner);
}

} // namespace

struct RlModel {
    MjmlxModel* handle = nullptr;
    ~RlModel() { if (handle) mjmlx_free_model(handle); }
};

struct RlBatchedSim {
    MjmlxBatchedSim* handle = nullptr;
    int num_envs = 0;
    int nq = 0, nv = 0, nu = 0;
    ~RlBatchedSim() { if (handle) mjmlx_batched_free(handle); }

    void step(FloatArray ctrl) {
        if (nu > 0) {
            if (ctrl.ndim() != 2 || static_cast<int>(ctrl.shape(0)) != num_envs || static_cast<int>(ctrl.shape(1)) != nu) {
                throw std::runtime_error("mjmlx_rl: ctrl must have shape (num_envs, nu)");
            }
            mjmlx_batched_step(handle, ctrl.data());
        } else {
            mjmlx_batched_step(handle, nullptr);
        }
    }

    void reset(IntArray mask) {
        if (mask.ndim() != 1 || static_cast<int>(mask.shape(0)) != num_envs) {
            throw std::runtime_error("mjmlx_rl: mask must have shape (num_envs,)");
        }
        mjmlx_batched_reset(handle, mask.data());
    }

    void set_env_qpos(int env_idx, FloatArray qpos_row) { mjmlx_batched_set_env_qpos(handle, env_idx, qpos_row.data(), static_cast<int>(qpos_row.shape(0))); }
    void set_env_qvel(int env_idx, FloatArray qvel_row) { mjmlx_batched_set_env_qvel(handle, env_idx, qvel_row.data(), static_cast<int>(qvel_row.shape(0))); }

    FloatArray qpos() { int n; auto* p = mjmlx_batched_get_qpos(handle, &n); return copy_out(p, n, num_envs); }
    FloatArray qvel() { int n; auto* p = mjmlx_batched_get_qvel(handle, &n); return copy_out(p, n, num_envs); }
    FloatArray xpos() { int n; auto* p = mjmlx_batched_get_xpos(handle, &n); return copy_out(p, n, num_envs); }
    FloatArray cfrc_ext() { int n; auto* p = mjmlx_batched_get_cfrc_ext(handle, &n); return copy_out(p, n, num_envs); }
};

NB_MODULE(_mjmlx_rl_native, m) {
    m.doc() = "Minimal numpy-facing mjmlx bindings for RL training (no mlx.core dependency at the Python boundary)";

    nb::class_<RlModel>(m, "Model")
        .def_prop_ro("nq", [](const RlModel& m) { return mjmlx_model_info(m.handle).nq; })
        .def_prop_ro("nv", [](const RlModel& m) { return mjmlx_model_info(m.handle).nv; })
        .def_prop_ro("nu", [](const RlModel& m) { return mjmlx_model_info(m.handle).nu; })
        .def_prop_ro("nbody", [](const RlModel& m) { return mjmlx_model_info(m.handle).nbody; });

    nb::class_<RlBatchedSim>(m, "BatchedSim")
        .def_prop_ro("num_envs", [](const RlBatchedSim& s) { return s.num_envs; })
        .def_prop_ro("nq", [](const RlBatchedSim& s) { return s.nq; })
        .def_prop_ro("nv", [](const RlBatchedSim& s) { return s.nv; })
        .def_prop_ro("nu", [](const RlBatchedSim& s) { return s.nu; })
        .def("step", &RlBatchedSim::step, nb::arg("ctrl"))
        .def("reset", &RlBatchedSim::reset, nb::arg("mask"))
        .def("set_env_qpos", &RlBatchedSim::set_env_qpos, nb::arg("env_idx"), nb::arg("qpos"))
        .def("set_env_qvel", &RlBatchedSim::set_env_qvel, nb::arg("env_idx"), nb::arg("qvel"))
        .def("qpos", &RlBatchedSim::qpos)
        .def("qvel", &RlBatchedSim::qvel)
        .def("xpos", &RlBatchedSim::xpos)
        .def("cfrc_ext", &RlBatchedSim::cfrc_ext);

    m.def("load_model", [](const std::string& xml_path) -> std::shared_ptr<RlModel> {
        auto model = std::make_shared<RlModel>();
        model->handle = mjmlx_load_model(xml_path.c_str());
        if (!model->handle) throw std::runtime_error("mjmlx_rl: failed to load model: " + xml_path);
        return model;
    }, nb::arg("xml_path"));

    m.def("create_batched", [](std::shared_ptr<RlModel> model, int num_envs, int frame_skip,
                                bool use_gpu, int solver_iterations) -> std::shared_ptr<RlBatchedSim> {
        MjmlxBatchedConfig cfg{};
        cfg.num_envs           = num_envs;
        cfg.foot_contacts_only = 0;
        cfg.integrator         = MJMLX_INTEGRATOR_EULER;
        cfg.use_gpu            = use_gpu ? 1 : 0;
        cfg.solver_iterations  = solver_iterations;
        (void)frame_skip; // frame skipping is done by calling step() repeatedly from Python

        auto sim = std::make_shared<RlBatchedSim>();
        sim->handle = mjmlx_batched_create(model->handle, &cfg);
        if (!sim->handle) throw std::runtime_error("mjmlx_rl: failed to create batched sim");
        sim->num_envs = num_envs;
        auto info = mjmlx_model_info(model->handle);
        sim->nq = info.nq; sim->nv = info.nv; sim->nu = info.nu;
        return sim;
    }, nb::arg("model"), nb::arg("num_envs"), nb::arg("frame_skip") = 1,
       nb::arg("use_gpu") = true, nb::arg("solver_iterations") = 0);

    m.def("version", []() -> std::string { return mjmlx_version(); });
}
