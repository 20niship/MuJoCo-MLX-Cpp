// SDF-SDF衝突のGPU(MKX) vs MuJoCo Cの比較。MuJoCo公式のSDFギア模型(2枚のギア、従動側に一定トルク)を1000step回し、
// 歯が噛み合って回る向きと角度が一致することを見る。SDFの距離関数はプラグイン(libsdf_plugin)が必要。
// Usage: test_sdf_gear <libsdf_plugin.so> <sdf_gear.xml> [tol_rad]
#include "test_utils.h"
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <libsdf_plugin.so> <sdf_gear.xml> [tol_rad]\n", argv[0]); return 1; }
    double tol = argc > 3 ? atof(argv[3]) : 0.3;
    mj_loadPluginLibrary(argv[1]);
    auto* model = mjmlx_load_model(argv[2]);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto* mj = (const mjModel*)mjmlx_get_mj_model(model);
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nu = info.nu;
    int drive = mj->jnt_qposadr[mj_name2id(mj, mjOBJ_JOINT, "drive")];
    int wheel = mj->jnt_qposadr[mj_name2id(mj, mjOBJ_JOINT, "freewheel")];

    const int B = 2, T = 1000;
    MjmlxBatchedConfig cg = {};
    cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedConfig cc = cg; cc.use_gpu = 0;
    auto* gpu = mjmlx_batched_create(model, &cg);
    auto* cpu = mjmlx_batched_create(model, &cc);
    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(gpu, mask.data());
    mjmlx_batched_reset(cpu, mask.data());

    std::vector<float> ctrl(B * nu, 1.0f), gq(B * nq), cq(B * nq), gv(B * nv), cv(B * nv);
    int a, b;
    for (int t = 0; t < T; t++) {
        mjmlx_batched_step(gpu, ctrl.data());
        mjmlx_batched_step(cpu, ctrl.data());
        if (getenv("SDF_TRACE") && t % 50 == 49) {
            mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
            mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
            printf("t=%4d  C: %.4f %.4f (v %.3f %.3f) GPU: %.4f %.4f (v %.3f %.3f)\n", t + 1, cq[drive], cq[wheel], cv[0], cv[1], gq[drive], gq[wheel], gv[0], gv[1]);
        }
    }
    mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
    mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
    printf("MuJoCo C: drive=%.4f freewheel=%.4f\nGPU     : drive=%.4f freewheel=%.4f\n", cq[drive], cq[wheel], gq[drive], gq[wheel]);

    TEST_BEGIN("sdf_gear_finite");
    CHECK(std::isfinite(gq[drive]) && std::isfinite(gq[wheel]), "NaN/Inf in GPU state");
    TEST_END();
    TEST_BEGIN("sdf_gear_meshing");
    CHECK_GT(std::fabs(gq[wheel]), 0.1, "従動ギアが回っていない(SDF接触なし)");
    CHECK(gq[wheel] * cq[wheel] > 0, "従動ギアの回転方向がMuJoCo Cと逆");
    TEST_END();
    TEST_BEGIN("sdf_gear_angle");
    CHECK_LT(std::fabs(gq[wheel] - cq[wheel]), tol, "従動ギア角度のCPU/GPU差");
    CHECK_LT(std::fabs(gq[drive] - cq[drive]), tol, "駆動ギア角度のCPU/GPU差");
    TEST_END();

    mjmlx_batched_free(gpu);
    mjmlx_batched_free(cpu);
    TEST_EXIT();
}
