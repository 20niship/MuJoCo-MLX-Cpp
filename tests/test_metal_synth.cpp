// Metal kernel validation for Synth humanoid model (nv=237)
// Tests against MuJoCo C reference (double precision).
// Phase 0: Euler devmem accuracy
// Phase 1+: Metal forward, collision, solver kernels (added incrementally)

#include "test_utils.h"
#include "internal.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

static float max_abs_diff(const float* a, const double* b, int n) {
    float maxd = 0;
    for (int i = 0; i < n; i++) {
        float d = std::abs(a[i] - (float)b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

static float max_abs_diff_ff(const float* a, const float* b, int n) {
    float maxd = 0;
    for (int i = 0; i < n; i++) {
        float d = std::abs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_metal_synth <synth_humanoid.xml>\n");
        return 1;
    }
    const char* xml_path = argv[1];

    printf("=== test_metal_synth ===\n");
    printf("  XML: %s\n", xml_path);

    // Load with MuJoCo C
    char error[1024] = {0};
    mjModel* mj_m = mj_loadXML(xml_path, nullptr, error, sizeof(error));
    if (!mj_m) {
        fprintf(stderr, "MuJoCo C load failed: %s\n", error);
        return 1;
    }
    mjData* mj_d = mj_makeData(mj_m);

    printf("  MuJoCo C: nq=%d nv=%d nu=%d nbody=%d ngeom=%d\n",
           mj_m->nq, mj_m->nv, mj_m->nu, mj_m->nbody, mj_m->ngeom);

    // Load with mjmlx
    auto* model = mjmlx_load_model(xml_path);
    if (!model) {
        fprintf(stderr, "mjmlx load failed\n");
        mj_deleteData(mj_d); mj_deleteModel(mj_m);
        return 1;
    }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nb = info.nbody;

    printf("  mjmlx:    nq=%d nv=%d nu=%d nbody=%d\n\n",
           info.nq, info.nv, info.nu, info.nbody);

    // ═══════════════════════════════════════════════════════════════
    // Phase 0: Diagnose euler_devmem accuracy
    // ═══════════════════════════════════════════════════════════════
    TEST_SECTION("Phase 0: Euler Devmem Accuracy");

    // Test 0a: Diagnose vmap_forward intermediate values (qM, qfrc_smooth)
    // Uses MuJoCo C kinematics as input to isolate forward dynamics from both
    // the Metal kinematics kernel and the Euler kernel.
    TEST_BEGIN("vmap_forward_qM_qfrc_smooth_vs_mjc");
    {
        mj_resetData(mj_m, mj_d);
        mj_forward(mj_m, mj_d);

        std::vector<double> mj_qM_dense(nv * nv, 0.0);
        mj_fullM(mj_m, mj_qM_dense.data(), mj_d->qM);

        // Set up mjmlx Data with MuJoCo C kinematics
        mjmlx::Model& m_ref = model->model;
        m_ref.init_cache();
        mjmlx::Data d;
        {
            std::vector<float> qp(nq);
            for (int i = 0; i < nq; i++) qp[i] = (float)mj_m->qpos0[i];
            d.qpos = mx::array(qp.data(), {nq}, mx::float32);
        }
        d.qvel = mx::zeros({nv});
        d.ctrl = mx::zeros({std::max(info.nu, 1)});
        d.qfrc_applied = mx::zeros({nv});
        d.xfrc_applied = mx::zeros({nb, 6});

        {
            int nj = m_ref.njnt, ng = m_ref.ngeom;
            auto cvt = [](const double* src, int n) {
                std::vector<float> v(n);
                for (int i = 0; i < n; i++) v[i] = (float)src[i];
                return v;
            };
            auto xp = cvt(mj_d->xpos, nb*3);
            auto xq = cvt(mj_d->xquat, nb*4);
            auto xm = cvt(mj_d->xmat, nb*9);
            auto xip = cvt(mj_d->xipos, nb*3);
            auto xim = cvt(mj_d->ximat, nb*9);
            auto xa = cvt(mj_d->xanchor, nj*3);
            auto xax = cvt(mj_d->xaxis, nj*3);
            auto gxp = cvt(mj_d->geom_xpos, ng*3);
            auto gxm = cvt(mj_d->geom_xmat, ng*9);
            d.xpos = mx::reshape(mx::array(xp.data(), {nb*3}, mx::float32), {nb, 3});
            d.xquat = mx::reshape(mx::array(xq.data(), {nb*4}, mx::float32), {nb, 4});
            d.xmat = mx::reshape(mx::array(xm.data(), {nb*9}, mx::float32), {nb, 3, 3});
            d.xipos = mx::reshape(mx::array(xip.data(), {nb*3}, mx::float32), {nb, 3});
            d.ximat = mx::reshape(mx::array(xim.data(), {nb*9}, mx::float32), {nb, 3, 3});
            d.xanchor = mx::reshape(mx::array(xa.data(), {nj*3}, mx::float32), {nj, 3});
            d.xaxis = mx::reshape(mx::array(xax.data(), {nj*3}, mx::float32), {nj, 3});
            d.geom_xpos = mx::reshape(mx::array(gxp.data(), {ng*3}, mx::float32), {ng, 3});
            d.geom_xmat = mx::reshape(mx::array(gxm.data(), {ng*9}, mx::float32), {ng, 3, 3});
        }

        d = mjmlx::vmap_forward(m_ref, d, true);
        mx::eval(d.qM, d.qfrc_smooth, d.qacc_smooth, d.qfrc_bias, d.qfrc_passive, d.qfrc_actuator);

        auto qM_flat = mx::flatten(d.qM);
        mx::eval(qM_flat);
        const float* mlx_qM = qM_flat.data<float>();

        // Compare qM
        float maxd_qM = 0;
        int wi = -1, wj = -1;
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < nv; j++) {
                float diff = std::abs(mlx_qM[i*nv+j] - (float)mj_qM_dense[i*nv+j]);
                if (diff > maxd_qM) { maxd_qM = diff; wi = i; wj = j; }
            }
        printf("    qM max diff: %.6e at [%d,%d]\n", maxd_qM, wi, wj);
        if (wi >= 0) printf("    qM[%d,%d] mlx=%.6f, MjC=%.6f\n", wi, wj,
                            mlx_qM[wi*nv+wj], (float)mj_qM_dense[wi*nv+wj]);

        float min_diag = 1e30f, max_diag = 0;
        for (int i = 0; i < nv; i++) {
            float v = std::abs(mlx_qM[i*nv+i]);
            if (v < min_diag) min_diag = v;
            if (v > max_diag) max_diag = v;
        }
        printf("    mlx qM diag range: [%.6e, %.6e]\n", min_diag, max_diag);

        // qfrc_smooth
        const float* mlx_qfrc = d.qfrc_smooth.data<float>();
        float maxd_qfrc = 0;
        int wf = -1;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qfrc[i] - (float)mj_d->qfrc_smooth[i]);
            if (diff > maxd_qfrc) { maxd_qfrc = diff; wf = i; }
        }
        printf("    qfrc_smooth max diff: %.6e at [%d]\n", maxd_qfrc, wf);
        if (wf >= 0) printf("    qfrc_smooth[%d] mlx=%.6f, MjC=%.6f\n", wf,
                            mlx_qfrc[wf], (float)mj_d->qfrc_smooth[wf]);

        // qfrc_bias
        const float* mlx_bias = d.qfrc_bias.data<float>();
        float maxd_bias = 0;
        int wb = -1;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_bias[i] - (float)mj_d->qfrc_bias[i]);
            if (diff > maxd_bias) { maxd_bias = diff; wb = i; }
        }
        printf("    qfrc_bias max diff: %.6e at [%d]\n", maxd_bias, wb);
        if (wb >= 0) printf("    qfrc_bias[%d] mlx=%.6f, MjC=%.6f\n", wb,
                            mlx_bias[wb], (float)mj_d->qfrc_bias[wb]);

        // qacc_smooth
        const float* mlx_qacc = d.qacc_smooth.data<float>();
        float maxd_qacc = 0;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qacc[i] - (float)mj_d->qacc_smooth[i]);
            if (diff > maxd_qacc) maxd_qacc = diff;
        }
        printf("    qacc_smooth max diff: %.6e\n", maxd_qacc);
        printf("    qacc_smooth[0:5] mlx: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", mlx_qacc[i]);
        printf("\n    qacc_smooth[0:5] MjC: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", (float)mj_d->qacc_smooth[i]);
        printf("\n");

        CHECK_LT(maxd_qM, 1.0f, "qM matches MuJoCo C within 1.0");
        CHECK_LT(maxd_qfrc, 5.0f, "qfrc_smooth matches MuJoCo C within 5.0");
        CHECK_LT(maxd_qacc, 10.0f, "qacc_smooth matches MuJoCo C within 10.0");
    }
    TEST_END();

    // Test 1: GPU vs MuJoCo C (contact-free) — 1 step from default state
    TEST_BEGIN("gpu_vs_mjc_contact_free_1step");
    {
        // MuJoCo C reference: disable contacts for apples-to-apples comparison
        mj_resetData(mj_m, mj_d);
        int saved_disableflags = mj_m->opt.disableflags;
        mj_m->opt.disableflags |= mjDSBL_CONTACT;
        mj_step(mj_m, mj_d);

        // GPU batched sim
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 1;
        cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim = mjmlx_batched_create(model, &cfg);
        if (!sim) {
            printf("    GPU sim creation failed (expected for nv>80 if Metal kernels unavailable)\n");
            CHECK(false, "GPU batched sim created");
        } else {
            mjmlx_batched_step(sim, nullptr);
            int n;
            auto* gpu_qpos = mjmlx_batched_get_qpos(sim, &n);
            auto* gpu_qvel = mjmlx_batched_get_qvel(sim, &n);
            auto* gpu_xpos = mjmlx_batched_get_xpos(sim, &n);

            float qpos_diff = max_abs_diff(gpu_qpos, mj_d->qpos, nq);
            float qvel_diff = max_abs_diff(gpu_qvel, mj_d->qvel, nv);
            float xpos_diff = max_abs_diff(gpu_xpos, mj_d->xpos, nb * 3);

            printf("    qpos max diff: %.6e\n", qpos_diff);
            printf("    qvel max diff: %.6e\n", qvel_diff);
            printf("    xpos max diff: %.6e\n", xpos_diff);

            // Print first 10 qpos values for debugging
            printf("    qpos[0:5] GPU:  ");
            for (int i = 0; i < 5 && i < nq; i++) printf("%.6f ", gpu_qpos[i]);
            printf("\n    qpos[0:5] MjC:  ");
            for (int i = 0; i < 5 && i < nq; i++) printf("%.6f ", (float)mj_d->qpos[i]);
            printf("\n");

            printf("    qvel[0:5] GPU:  ");
            for (int i = 0; i < 5 && i < nv; i++) printf("%.6f ", gpu_qvel[i]);
            printf("\n    qvel[0:5] MjC:  ");
            for (int i = 0; i < 5 && i < nv; i++) printf("%.6f ", (float)mj_d->qvel[i]);
            printf("\n");

            CHECK_LT(qpos_diff, 0.05f, "qpos: GPU matches MuJoCo C (contact-free) within 0.05");
            CHECK_LT(qvel_diff, 2.0f, "qvel: GPU matches MuJoCo C (contact-free) within 2.0");
            CHECK_LT(xpos_diff, 0.01f, "xpos: Metal FK matches MuJoCo C within 0.01");

            mjmlx_batched_free(sim);
        }

        mj_m->opt.disableflags = saved_disableflags;
    }
    TEST_END();

    // Test 2: GPU vs MuJoCo C (contact-free) — 10 steps
    TEST_BEGIN("gpu_vs_mjc_contact_free_10step");
    {
        mj_resetData(mj_m, mj_d);
        int saved = mj_m->opt.disableflags;
        mj_m->opt.disableflags |= mjDSBL_CONTACT;

        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 1; cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim = mjmlx_batched_create(model, &cfg);

        if (sim) {
            for (int s = 0; s < 10; s++) {
                mj_step(mj_m, mj_d);
                mjmlx_batched_step(sim, nullptr);
            }
            int n;
            auto* gpu_qpos = mjmlx_batched_get_qpos(sim, &n);
            auto* gpu_qvel = mjmlx_batched_get_qvel(sim, &n);

            float qpos_diff = max_abs_diff(gpu_qpos, mj_d->qpos, nq);
            float qvel_diff = max_abs_diff(gpu_qvel, mj_d->qvel, nv);

            printf("    10-step qpos max diff: %.6e\n", qpos_diff);
            printf("    10-step qvel max diff: %.6e\n", qvel_diff);

            CHECK_NO_NAN(gpu_qpos, nq, "no NaN in qpos after 10 GPU steps");
            CHECK_NO_NAN(gpu_qvel, nv, "no NaN in qvel after 10 GPU steps");
            CHECK_LT(qpos_diff, 1.0f, "10-step qpos drift < 1.0");
            CHECK_LT(qvel_diff, 10.0f, "10-step qvel drift < 10.0");

            mjmlx_batched_free(sim);
        }

        mj_m->opt.disableflags = saved;
    }
    TEST_END();

    // Test 3: GPU stability — 100 steps, check for NaN/explosion
    TEST_BEGIN("gpu_100step_stability");
    {
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 1; cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim = mjmlx_batched_create(model, &cfg);

        if (sim) {
            for (int s = 0; s < 100; s++)
                mjmlx_batched_step(sim, nullptr);

            int n;
            auto* qpos = mjmlx_batched_get_qpos(sim, &n);
            auto* qvel = mjmlx_batched_get_qvel(sim, &n);

            CHECK_NO_NAN(qpos, nq, "no NaN after 100 steps");
            CHECK_NO_NAN(qvel, nv, "no NaN in qvel after 100 steps");

            float max_vel = 0;
            for (int i = 0; i < nv; i++) {
                float v = std::abs(qvel[i]);
                if (v > max_vel) max_vel = v;
            }
            printf("    max|qvel| after 100 steps: %.2f\n", max_vel);
            CHECK_LT(max_vel, 1e4f, "max|qvel| < 1e4 after 100 steps");

            mjmlx_batched_free(sim);
        }
    }
    TEST_END();

    // Test 4: Multi-env determinism
    TEST_BEGIN("gpu_multienv_determinism");
    {
        MjmlxBatchedConfig cfg1 = {};
        cfg1.num_envs = 1; cfg1.use_gpu = 1;
        cfg1.integrator = MJMLX_INTEGRATOR_EULER;

        MjmlxBatchedConfig cfg32 = {};
        cfg32.num_envs = 32; cfg32.use_gpu = 1;
        cfg32.integrator = MJMLX_INTEGRATOR_EULER;

        auto* sim1 = mjmlx_batched_create(model, &cfg1);
        auto* sim32 = mjmlx_batched_create(model, &cfg32);

        if (sim1 && sim32) {
            for (int s = 0; s < 5; s++) {
                mjmlx_batched_step(sim1, nullptr);
                mjmlx_batched_step(sim32, nullptr);
            }
            int n;
            auto* q1 = mjmlx_batched_get_qpos(sim1, &n);
            auto* q32 = mjmlx_batched_get_qpos(sim32, &n);

            float max_diff = 0;
            for (int i = 0; i < nq; i++) {
                float d = std::abs(q1[i] - q32[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("    1-env vs env[0] of 32-env qpos diff: %.6e\n", max_diff);
            CHECK_LT(max_diff, 1e-5f, "1-env matches env[0] of 32-env batch");

            mjmlx_batched_free(sim1);
            mjmlx_batched_free(sim32);
        }
    }
    TEST_END();

    // Cleanup
    mj_deleteData(mj_d);
    mj_deleteModel(mj_m);
    mjmlx_free_model(model);

    TEST_EXIT();
}
