// バッチ版 mj_ray(GPU)。全env x 全レイを1回のdispatchで処理する。高さスキャナ/LiDAR/rangefinderの基盤。

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "compat/mkx_kernels/raycast.hpp"
#include "internal.h"
#include "mjmlx/mjmlx.h"

// mjModelの必要分は作成時にバッファへ写すので、作成後にモデルが解放されても使える(レンダラと同じ)。
struct MjmlxRaycaster {
    MjmlxBatchedSim* sim = nullptr;
    int B = 0, ngeom = 0;
    std::optional<mkx::fast::Kernel<>> kernel;
    std::optional<mx::array> ginfo, mesh_vert, mesh_face, mesh_adr;
};

extern "C" {

MJMLX_API MjmlxRaycaster* mjmlx_raycaster_create(MjmlxBatchedSim* sim) {
    if (!sim || sim->cpu_mode || !sim->sim.kinematics || !sim->mj_model) {
        fprintf(stderr, "mjmlx_raycaster_create: GPU batched simのみ対応\n");
        return nullptr;
    }
    try {
        const mjModel* m = sim->mj_model;
        auto r = std::make_unique<MjmlxRaycaster>();
        r->sim = sim;
        r->B = sim->sim.num_envs;
        r->ngeom = (int)m->ngeom;
        if (r->ngeom <= 0) throw std::runtime_error("model has no geoms");

        constexpr int GI = mjmlx::mkx_kernels::RAYCAST_GINFO;
        std::vector<float> gi((size_t)r->ngeom * GI, 0.0f);
        for (int g = 0; g < r->ngeom; g++) {
            float* p = &gi[(size_t)g * GI];
            int body = m->geom_bodyid[g];
            int mat = m->geom_matid[g];
            float alpha = mat >= 0 ? m->mat_rgba[4 * mat + 3] : m->geom_rgba[4 * g + 3];
            p[0] = (float)m->geom_type[g];
            p[1] = (float)body;
            p[2] = (float)m->geom_dataid[g];
            p[3] = (float)m->geom_group[g];
            p[4] = m->body_weldid[body] == 0 ? 1.0f : 0.0f;
            p[5] = alpha == 0.0f ? 1.0f : 0.0f;
            for (int k = 0; k < 6; k++) p[6 + k] = (float)m->geom_aabb[6 * g + k];
            for (int k = 0; k < 3; k++) p[13 + k] = (float)m->geom_size[3 * g + k];
        }
        r->ginfo = mx::array(gi.data(), {(int)gi.size()}, mx::float32);

        int nv = (int)m->nmeshvert, nf = (int)m->nmeshface, nm = (int)m->nmesh;
        std::vector<float> mv(std::max(nv * 3, 1), 0.0f), mf(std::max(nf * 3, 1), 0.0f), ma(std::max(nm * 3, 1), 0.0f);
        for (int i = 0; i < nv * 3; i++) mv[i] = m->mesh_vert[i];
        for (int i = 0; i < nf * 3; i++) mf[i] = (float)m->mesh_face[i];
        for (int i = 0; i < nm; i++) {
            ma[3 * i] = (float)m->mesh_vertadr[i];
            ma[3 * i + 1] = (float)m->mesh_faceadr[i];
            ma[3 * i + 2] = (float)m->mesh_facenum[i];
        }
        r->mesh_vert = mx::array(mv.data(), {(int)mv.size()}, mx::float32);
        r->mesh_face = mx::array(mf.data(), {(int)mf.size()}, mx::float32);
        r->mesh_adr = mx::array(ma.data(), {(int)ma.size()}, mx::float32);
        mx::eval(*r->ginfo, *r->mesh_vert, *r->mesh_face, *r->mesh_adr);
        r->kernel = mjmlx::mkx_kernels::make_raycast_kernel(r->ngeom, r->B);
        return r.release();
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_raycaster_create error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API void mjmlx_raycaster_free(MjmlxRaycaster* r) { delete r; }

MJMLX_API int mjmlx_raycaster_cast(MjmlxRaycaster* r, const float* origins, const float* dirs, int nray, float max_dist,
                                   int bodyexclude, int flg_static, int geomgroup_mask, float* dist_out, int* geomid_out) {
    if (!r || !origins || !dirs || nray <= 0 || !dist_out) return -1;
    try {
        auto& s = r->sim->sim;
        const int B = r->B;
        const size_t n = (size_t)B * nray;

        // 現在のqposで運動学を再計算してgeom姿勢(SoA)を得る
        auto kin = s.kinematics(s.qpos);
        std::vector<float> params = {(float)nray, max_dist, (float)bodyexclude, flg_static ? 1.0f : 0.0f, (float)geomgroup_mask};
        auto o = mx::array(origins, {(int)(n * 3)}, mx::float32);
        auto d = mx::array(dirs, {(int)(n * 3)}, mx::float32);
        auto p = mx::array(params.data(), {(int)params.size()}, mx::float32);

        const uint32_t tg = 64;
        const uint32_t grid = (uint32_t)((n + tg - 1) / tg * tg);
        auto raw = (*r->kernel)(
            mx::fast::kernel_inputs({kin[7], kin[8], *r->ginfo, *r->mesh_vert, *r->mesh_face, *r->mesh_adr, o, d, p}),
            mx::fast::kernel_shapes({{(int)n}, {(int)n}}),
            std::array<uint32_t, 3>{grid, 1, 1}, std::array<uint32_t, 3>{tg, 1, 1});
        auto outs = mx::fast::kernel_outputs(raw);
        mx::eval(outs[0], outs[1]);
        std::memcpy(dist_out, outs[0].data<float>(), n * sizeof(float));
        if (geomid_out) {
            const float* g = outs[1].data<float>();
            for (size_t i = 0; i < n; i++) geomid_out[i] = (int)g[i];
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_raycaster_cast error: %s\n", e.what());
        return -1;
    }
}

} // extern "C"
