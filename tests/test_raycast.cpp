// XMLを埋め込んだ自己完結テストなので、モデルファイル無しでlavapipe(CI)でも動く。

#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mjmlx/mjmlx.h"

namespace {

// 静的geom(各型1つ)+動くbody(free joint, box+sphere)+group 3のgeom+透明geom。メッシュは凸な四面体を頂点直書きで与える。
const char* XML = R"(
<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81" timestep="0.005"/>
<asset><mesh name="tet" vertex="0 0 0  0.4 0 0  0 0.4 0  0 0 0.4"/></asset>
<worldbody>
  <geom name="floor" type="plane" size="6 6 0.1"/>
  <geom name="sph" type="sphere" size="0.3" pos="1.2 0.0 0.3"/>
  <geom name="cap" type="capsule" size="0.12 0.3" pos="-1.0 0.8 0.5" euler="0.6 0.4 0.0"/>
  <geom name="ell" type="ellipsoid" size="0.35 0.2 0.25" pos="0.0 -1.3 0.4" euler="0.3 0.0 0.8"/>
  <geom name="cyl" type="cylinder" size="0.25 0.3" pos="-1.4 -1.2 0.35" euler="0.0 0.5 0.0"/>
  <geom name="box" type="box" size="0.3 0.2 0.25" pos="1.3 1.3 0.3" euler="0.0 0.0 0.7"/>
  <geom name="mesh" type="mesh" mesh="tet" pos="0.2 1.5 0.0" euler="0.0 0.0 0.5"/>
  <geom name="grp3" type="sphere" size="0.25" pos="0.0 0.0 0.25" group="3"/>
  <geom name="invisible" type="sphere" size="0.2" pos="-0.6 -0.2 0.2" rgba="1 0 0 0"/>
  <body name="mover" pos="0 0 1"><freejoint/>
    <geom name="m_box" type="box" size="0.15 0.1 0.1"/>
    <geom name="m_sph" type="sphere" size="0.1" pos="0.3 0 0"/>
  </body>
</worldbody></mujoco>)";

struct Stats { int hit_miss_mismatch = 0, id_mismatch = 0, both_hit = 0, total = 0; double max_err = 0.0; };

} // namespace

int main() {
    printf("=== test_raycast ===\n");
    MjmlxModel* model = mjmlx_load_model_from_string(XML);
    if (!model) { fprintf(stderr, "FAIL: load model\n"); return 1; }
    const mjModel* m = static_cast<const mjModel*>(mjmlx_get_mj_model(model));
    const int nq = m->nq, B = 6, R = 400;
    const int mover = mj_name2id(m, mjOBJ_BODY, "mover");

    MjmlxBatchedConfig cfg = {};
    cfg.num_envs = B; cfg.use_gpu = 1; cfg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &cfg);
    MjmlxRaycaster* rc = sim ? mjmlx_raycaster_create(sim) : nullptr;
    if (!rc) { fprintf(stderr, "FAIL: create raycaster\n"); return 1; }

    // envごとに別の姿勢(位置・向き)
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<float> qpos(B * nq), qvel(B * m->nv, 0.0f);
    for (int e = 0; e < B; e++) {
        float* q = &qpos[e * nq];
        q[0] = 0.8f * U(rng); q[1] = 0.8f * U(rng); q[2] = 0.4f + 0.3f * (U(rng) + 1.0f);
        float w[4] = {1.0f + U(rng), U(rng), U(rng), U(rng)};
        float n = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2] + w[3]*w[3]);
        for (int k = 0; k < 4; k++) q[3 + k] = w[k] / n;
    }
    mjmlx_batched_set_state(sim, qpos.data(), qvel.data());

    // 参照側: 各envのmjDataでmj_kinematics
    std::vector<mjData*> ds(B);
    for (int e = 0; e < B; e++) {
        ds[e] = mj_makeData(m);
        for (int i = 0; i < nq; i++) ds[e]->qpos[i] = qpos[e * nq + i];
        mj_kinematics(m, ds[e]);
    }

    // レイ: 上空/横から全方向。原点は箱の外側(-2.5..2.5)
    std::vector<float> org(B * R * 3), dir(B * R * 3);
    std::uniform_real_distribution<float> UP(0.0f, 1.0f);
    for (int i = 0; i < B * R; i++) {
        org[3*i] = 2.5f * U(rng); org[3*i+1] = 2.5f * U(rng); org[3*i+2] = 0.05f + 2.0f * UP(rng);
        float t = 6.2831853f * UP(rng), z = -1.0f + 1.6f * UP(rng), s = std::sqrt(1.0f - z * z);
        float len = 0.5f + 2.0f * UP(rng);  // 正規化されない向きも通す
        dir[3*i] = len * s * std::cos(t); dir[3*i+1] = len * s * std::sin(t); dir[3*i+2] = len * z;
    }

    struct Case { const char* name; int bodyex; int flg_static; int mask; float maxd; };
    const Case cases[] = {
        {"default", -1, 1, -1, 0.0f},
        {"bodyexclude(mover)", mover, 1, -1, 0.0f},
        {"flg_static=0", -1, 0, -1, 0.0f},
        {"geomgroup mask w/o group3", -1, 1, ~(1 << 3), 0.0f},
        {"max_dist=1.5", -1, 1, -1, 1.5f},
    };

    int failed = 0;
    std::vector<float> dist(B * R);
    std::vector<int> gid(B * R);
    for (const auto& c : cases) {
        if (mjmlx_raycaster_cast(rc, org.data(), dir.data(), R, c.maxd, c.bodyex, c.flg_static, c.mask, dist.data(), gid.data()) != 0) {
            fprintf(stderr, "FAIL: %s: cast failed\n", c.name); failed++; continue;
        }
        mjtByte grp[6]; for (int g = 0; g < 6; g++) grp[g] = (c.mask >> g) & 1;
        Stats st; int hits = 0;
        for (int e = 0; e < B; e++) for (int r = 0; r < R; r++) {
            int i = e * R + r;
            double p[3] = {org[3*i], org[3*i+1], org[3*i+2]}, v[3] = {dir[3*i], dir[3*i+1], dir[3*i+2]};
            double vn = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            for (int k = 0; k < 3; k++) v[k] /= vn;
            int ref_g = -1;
            double ref = mj_ray(m, ds[e], p, v, c.mask == -1 ? nullptr : grp, c.flg_static, c.bodyex, &ref_g, nullptr);
            if (c.maxd > 0 && ref > c.maxd) { ref = -1; ref_g = -1; }
            st.total++;
            bool rh = ref >= 0, gh = dist[i] >= 0;
            if (rh) hits++;
            if (rh != gh) { st.hit_miss_mismatch++; continue; }
            if (!rh) continue;
            st.both_hit++;
            st.max_err = std::max(st.max_err, std::fabs(ref - dist[i]));
            if (ref_g != gid[i]) st.id_mismatch++;
        }
        // 接線すれすれのレイはfloat誤差で当たり外れが割れうるので0.3%まで許す
        bool ok = hits >= 10 && st.hit_miss_mismatch <= st.total * 3 / 1000 && st.max_err < 2e-3 && st.id_mismatch <= st.total / 200;
        printf("%-28s rays=%d hits=%d  hit/miss mismatch=%d  geomid mismatch=%d  max|dist err|=%.2e  %s\n", c.name, st.total, hits, st.hit_miss_mismatch, st.id_mismatch, st.max_err, ok ? "OK" : "FAIL");
        if (!ok) failed++;
        if (c.maxd == 0.0f && c.bodyex < 0 && c.flg_static && c.mask == -1) {
            // 全geom型が実際に当たっていること(透明geomは参照のmj_rayも無視する)
            std::vector<int> cnt(m->ngeom, 0);
            for (int i = 0; i < B * R; i++) if (gid[i] >= 0) cnt[gid[i]]++;
            for (int g = 0; g < m->ngeom; g++) {
                const char* gn = mj_id2name(m, mjOBJ_GEOM, g);
                bool transparent = gn && std::string(gn) == "invisible";
                bool good = transparent ? cnt[g] == 0 : cnt[g] > 0;
                printf("  geom %-10s hits=%d %s\n", gn, cnt[g], good ? "" : "<-- unexpected");
                if (!good) failed++;
            }
        }
    }

    // 下向きグリッド(height scanner用途): 真下は床(geom 0)、一様な高さ
    {
        std::vector<float> o(B * 9 * 3), d(B * 9 * 3);
        for (int e = 0; e < B; e++) for (int k = 0; k < 9; k++) {
            int i = e * 9 + k;
            o[3*i] = -2.0f + 0.5f * (k % 3); o[3*i+1] = 4.0f + 0.5f * (k / 3); o[3*i+2] = 1.0f;  // 物体の無い領域
            d[3*i] = 0; d[3*i+1] = 0; d[3*i+2] = -1;
        }
        std::vector<float> dd(B * 9); std::vector<int> gg(B * 9);
        bool ok = mjmlx_raycaster_cast(rc, o.data(), d.data(), 9, 0.0f, -1, 1, -1, dd.data(), gg.data()) == 0;
        for (int i = 0; i < B * 9 && ok; i++) ok = std::fabs(dd[i] - 1.0f) < 1e-4f && gg[i] == 0;
        printf("%-28s %s\n", "downward grid hits floor", ok ? "OK" : "FAIL");
        if (!ok) failed++;
    }

    // alpha=0のgeomの真上から撃つ: MuJoCoと同じ扱い(無視して床に当たる)になること
    {
        std::vector<float> o(B * 3), d(B * 3);
        for (int e = 0; e < B; e++) { o[3*e] = -0.6f; o[3*e+1] = -0.2f; o[3*e+2] = 1.0f; d[3*e] = 0; d[3*e+1] = 0; d[3*e+2] = -1; }
        std::vector<float> dd(B); std::vector<int> gg(B);
        bool ok = mjmlx_raycaster_cast(rc, o.data(), d.data(), 1, 0.0f, mover, 1, -1, dd.data(), gg.data()) == 0;
        for (int e = 0; e < B && ok; e++) {
            double p[3] = {-0.6, -0.2, 1.0}, v[3] = {0, 0, -1}; int rg = -1;
            double ref = mj_ray(m, ds[e], p, v, nullptr, 1, mover, &rg, nullptr);
            ok = std::fabs(ref - dd[e]) < 1e-4 && rg == gg[e];
        }
        printf("%-28s %s\n", "transparent geom like mj_ray", ok ? "OK" : "FAIL");
        if (!ok) failed++;
    }

    // 原始geomの内側から撃つ(出口側のヒットを返すこと)
    {
        const char* names[] = {"sph", "cap", "ell", "cyl", "box"};
        const int nd = 8, nr = 5 * nd;
        std::vector<float> o(B * nr * 3), d(B * nr * 3);
        std::vector<int> expect_geom(nr);
        std::mt19937 r2(11);
        for (int k = 0; k < 5; k++) {
            int g = mj_name2id(m, mjOBJ_GEOM, names[k]);
            for (int j = 0; j < nd; j++) {
                float t = 6.2831853f * UP(r2), z = -1.0f + 2.0f * UP(r2), sn = std::sqrt(1.0f - z * z);
                for (int e = 0; e < B; e++) {
                    int i = e * nr + k * nd + j;
                    for (int c = 0; c < 3; c++) o[3*i+c] = (float)ds[e]->geom_xpos[3*g+c];
                    d[3*i] = sn * std::cos(t); d[3*i+1] = sn * std::sin(t); d[3*i+2] = z;
                }
            }
        }
        std::vector<float> dd(B * nr); std::vector<int> gg(B * nr);
        bool ok = mjmlx_raycaster_cast(rc, o.data(), d.data(), nr, 0.0f, mover, 1, -1, dd.data(), gg.data()) == 0;
        double worst = 0.0;
        for (int e = 0; e < B && ok; e++) for (int r = 0; r < nr; r++) {
            int i = e * nr + r;
            double p[3] = {o[3*i], o[3*i+1], o[3*i+2]}, v[3] = {d[3*i], d[3*i+1], d[3*i+2]}; int rg = -1;
            double ref = mj_ray(m, ds[e], p, v, nullptr, 1, mover, &rg, nullptr);
            worst = std::max(worst, std::fabs(ref - dd[i]));
            if (std::fabs(ref - dd[i]) > 1e-3 || rg != gg[i]) { ok = false; printf("  inside mismatch env %d ray %d: ref %.5f (geom %d) gpu %.5f (geom %d)\n", e, r, ref, rg, dd[i], gg[i]); break; }
        }
        printf("%-28s %s (max err %.2e)\n", "rays starting inside geoms", ok ? "OK" : "FAIL", worst);
        if (!ok) failed++;
    }

    for (auto* d : ds) mj_deleteData(d);
    mjmlx_raycaster_free(rc);
    mjmlx_batched_free(sim);
    mjmlx_free_model(model);
    printf(failed ? "FAILED (%d)\n" : "PASSED\n", failed);
    return failed ? 1 : 0;
}
