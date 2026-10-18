// GPU(MKX)バッチパイプライン vs CPU(MuJoCo C)参照の閾値付き比較。旧test_metal_synthの代替で、forward/collision/solver/eulerを通しで検証する。
// 検証: 1/10/T ステップ後のqpos・qvel誤差、NaN/Inf無し、同一初期状態envの決定性。
// Usage: test_gpu_vs_cpu <model.xml> [tol_1step] [tol_10step] [tol_T]   (T=100, 誤差は全envのmax|diff|)

#include "test_utils.h"
#include <random>
#include <vector>

static const int B = 8;
static const int T = 100;

static double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        m = std::max(m, std::fabs((double)a[i] - b[i]));
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: test_gpu_vs_cpu <model.xml> [tol_1] [tol_10] [tol_T]\n"); return 1; }
    double tol1 = argc > 2 ? atof(argv[2]) : 1e-2;
    double tol10 = argc > 3 ? atof(argv[3]) : 5e-2;
    double tolT = argc > 4 ? atof(argv[4]) : 5e-1;

    printf("=== test_gpu_vs_cpu (%s) ===\n", argv[1]);
    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nu = info.nu;
    printf("nq=%d nv=%d nu=%d\n", nq, nv, nu);

    MjmlxBatchedConfig cg = {};
    cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedConfig cc = cg; cc.use_gpu = 0;
    auto* gpu = mjmlx_batched_create(model, &cg);
    auto* cpu = mjmlx_batched_create(model, &cc);

    // env0..B/2-1 と B/2..B-1 は同一の摂動(決定性チェック用)。env間の独立性は別envペアの結果で担保される。
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(cpu, mask.data());
    mjmlx_batched_reset(gpu, mask.data());
    std::vector<float> q(B * nq), v(B * nv);
    int a, b;
    mjmlx_batched_get_state(cpu, q.data(), v.data(), &a, &b);
    for (int e = 0; e < B / 2; e++) {
        for (int j = 0; j < nq; j++) if (j < 3 || j >= 7) q[e * nq + j] += 0.01f * u(rng);
        for (int j = 0; j < nv; j++) v[e * nv + j] = 0.05f * u(rng);
        for (int j = 0; j < nq; j++) q[(e + B / 2) * nq + j] = q[e * nq + j];
        for (int j = 0; j < nv; j++) v[(e + B / 2) * nv + j] = v[e * nv + j];
    }
    mjmlx_batched_set_state(gpu, q.data(), v.data());
    mjmlx_batched_set_state(cpu, q.data(), v.data());

    std::vector<float> ctrl(B * nu), gq(B * nq), gv(B * nv), cq(B * nq), cv(B * nv);
    double eq[3] = {0, 0, 0}, ev[3] = {0, 0, 0};
    double det = 0;
    bool finite = true;
    for (int t = 1; t <= T; t++) {
        for (int e = 0; e < B / 2; e++) for (int j = 0; j < nu; j++) {
            ctrl[e * nu + j] = 0.1f * u(rng);
            ctrl[(e + B / 2) * nu + j] = ctrl[e * nu + j];
        }
        mjmlx_batched_step(gpu, nu ? ctrl.data() : nullptr);
        mjmlx_batched_step(cpu, nu ? ctrl.data() : nullptr);
        if (t != 1 && t != 10 && t != T) continue;
        mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
        mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
        int k = t == 1 ? 0 : t == 10 ? 1 : 2;
        eq[k] = max_diff(gq, cq);
        ev[k] = max_diff(gv, cv);
        printf("step %3d: qpos max|diff|=%.3e qvel max|diff|=%.3e\n", t, eq[k], ev[k]);
        for (float x : gq) finite &= std::isfinite(x);
        for (float x : gv) finite &= std::isfinite(x);
        if (t == T) {
            for (int e = 0; e < B / 2; e++) {
                for (int j = 0; j < nq; j++) det = std::max(det, (double)std::fabs(gq[e * nq + j] - gq[(e + B / 2) * nq + j]));
                for (int j = 0; j < nv; j++) det = std::max(det, (double)std::fabs(gv[e * nv + j] - gv[(e + B / 2) * nv + j]));
            }
        }
    }
    printf("determinism (同一env間の差): %.3e\n", det);

    TEST_BEGIN("gpu_finite_100step");
    CHECK(finite, "GPU state has NaN/Inf");
    TEST_END();

    TEST_BEGIN("gpu_vs_cpu_1step");
    CHECK_LT(eq[0], tol1, "qpos 1step"); CHECK_LT(ev[0], tol1 * 10, "qvel 1step");
    TEST_END();

    TEST_BEGIN("gpu_vs_cpu_10step");
    CHECK_LT(eq[1], tol10, "qpos 10step"); CHECK_LT(ev[1], tol10 * 10, "qvel 10step");
    TEST_END();

    TEST_BEGIN("gpu_vs_cpu_100step");
    CHECK_LT(eq[2], tolT, "qpos 100step"); CHECK_LT(ev[2], tolT * 10, "qvel 100step");
    TEST_END();

    TEST_BEGIN("gpu_multienv_determinism");
    CHECK_LT(det, 1e-5, "identical envs diverged");
    TEST_END();

    mjmlx_batched_free(gpu);
    mjmlx_batched_free(cpu);
    TEST_EXIT();
}
