// SDF vs mesh衝突のGPU(MKX)の検証。立方体メッシュ(ワールド原点)の上にtorus SDFを落とし、メッシュ上面で静止する高さを見る。
// 基準は既定でMuJoCo C。mesh geomに対するMuJoCoのmjc_MeshSDFはワールド原点方向のレイで距離を近似し粗い(数mmめり込む)ため、その場合はexpected_zに解析値を渡して比較する。
// SDFの距離関数はプラグイン(libsdf_plugin)が必要。Usage: test_sdf_mesh <libsdf_plugin.so> <model.xml> [tol_m] [expected_z]
#include "test_utils.h"
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <libsdf_plugin.so> <sdf_mesh.xml> [tol_m]\n", argv[0]); return 1; }
    double tol = argc > 3 ? atof(argv[3]) : 0.003;
    const bool analytic = argc > 4;
    const double expected_z = analytic ? atof(argv[4]) : 0.0;
    mj_loadPluginLibrary(argv[1]);
    auto* model = mjmlx_load_model(argv[2]);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv;

    const int B = 2, T = 600;
    MjmlxBatchedConfig cg = {};
    cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedConfig cc = cg; cc.use_gpu = 0;
    auto* gpu = mjmlx_batched_create(model, &cg);
    auto* cpu = mjmlx_batched_create(model, &cc);
    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(gpu, mask.data());
    mjmlx_batched_reset(cpu, mask.data());

    std::vector<float> gq(B * nq), cq(B * nq), gv(B * nv), cv(B * nv);
    std::vector<float> trace_c, trace_g;  // SDF_DUMP=path: env0の軌道を scripts/render_compare.py の形式で書く(CPU|GPU)
    int a, b;
    for (int t = 0; t < T; t++) {
        mjmlx_batched_step(gpu, nullptr);
        mjmlx_batched_step(cpu, nullptr);
        if (getenv("SDF_DUMP") && t % 8 == 7) {
            mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
            mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
            trace_c.insert(trace_c.end(), cq.begin(), cq.begin() + nq);
            trace_g.insert(trace_g.end(), gq.begin(), gq.begin() + nq);
        }
        if (getenv("SDF_TRACE") && t % 50 == 49) {
            mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
            mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
            printf("t=%4d  C: %.4f %.4f %.4f  GPU: %.4f %.4f %.4f\n", t + 1, cq[0], cq[1], cq[2], gq[0], gq[1], gq[2]);
        }
    }
    mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
    mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
    printf("MuJoCo C: %.4f %.4f %.4f\nGPU     : %.4f %.4f %.4f\n", cq[0], cq[1], cq[2], gq[0], gq[1], gq[2]);

    if (const char* dump = getenv("SDF_DUMP")) {
        FILE* f = fopen(dump, "wb");
        int32_t hdr[2] = {nq, (int32_t)(trace_c.size() / nq)};
        float fps = 30.0f;
        fwrite(hdr, 4, 2, f); fwrite(&fps, 4, 1, f);
        fwrite(trace_c.data(), 4, trace_c.size(), f); fwrite(trace_g.data(), 4, trace_g.size(), f);
        fclose(f);
    }
    TEST_BEGIN("sdf_mesh_finite");
    for (int i = 0; i < nq; i++) CHECK(std::isfinite(gq[i]), "NaN/Inf in GPU state");
    TEST_END();
    TEST_BEGIN("sdf_mesh_rests_on_mesh");
    CHECK_GT(gq[2], 0.32, "torusがメッシュを貫通している(SDF-mesh接触なし)");
    CHECK_LT(gq[2], 0.37, "torusがメッシュ上面で止まっていない");
    TEST_END();
    TEST_BEGIN("sdf_mesh_height");
    CHECK_LT(std::fabs(gq[2] - (analytic ? expected_z : cq[2])), tol, analytic ? "静止高さの解析値との差" : "静止高さのCPU/GPU差");
    TEST_END();

    mjmlx_batched_free(gpu);
    mjmlx_batched_free(cpu);
    TEST_EXIT();
}
