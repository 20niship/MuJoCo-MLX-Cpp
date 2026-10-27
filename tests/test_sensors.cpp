// batched GPUのsensordata vs MuJoCo C。各step前のGPUの状態をmjDataに入れてmj_forwardした値を参照にするので、物理のCPU/GPU差とは切り分けてセンサ式だけを検証する。
// Usage: test_sensors <model.xml> <steps> <tol> [tol_acc]   tol_acc省略時はaccelerometerを比較しない(接触ありは速度差分が衝撃で乖離するため)
// accelerometerは速度の1step差分(Isaac Lab IMUと同じ)なのでMuJoCoの厳密値とは1step分遅れた近似になる。
// sensorごと(名前単位)に全env・全stepでのmax|diff|を出力し、型ごとの許容を超えたらFAIL。

#include "test_utils.h"
#include <algorithm>
#include <map>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.xml steps tol [tol_acc]\n", argv[0]); return 2; }
    const int steps = atoi(argv[2]);
    const double tol = atof(argv[3]);
    const bool check_acc = argc > 4;
    const double tol_acc = check_acc ? atof(argv[4]) : 0;
    const int B = 6;

    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { fprintf(stderr, "FAIL: load %s\n", argv[1]); return 1; }
    auto* mj = (const mjModel*)mjmlx_get_mj_model(model);
    auto info = mjmlx_model_info(model);
    const int nq = info.nq, nv = info.nv, nu = info.nu, nsd = mj->nsensordata;

    MjmlxBatchedConfig cg = {};
    cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    auto* gpu = mjmlx_batched_create(model, &cg);
    mjData* d = mj_makeData(mj);
    if (!gpu) { fprintf(stderr, "FAIL: create\n"); return 1; }
    if (mjmlx_batched_nsensordata(gpu) != nsd) { fprintf(stderr, "FAIL: nsensordata %d != %d\n", mjmlx_batched_nsensordata(gpu), nsd); return 1; }

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(gpu, mask.data());
    std::vector<float> q(B * nq), v(B * nv);
    int a, b;
    mjmlx_batched_get_state(gpu, q.data(), v.data(), &a, &b);
    for (int e = 0; e < B; e++) {
        for (int i = 0; i < nq; i++) q[e * nq + i] += 0.05f * nd(rng);
        for (int i = 0; i < nv; i++) v[e * nv + i] = 0.5f * nd(rng);
        for (int j = 0; j < mj->njnt; j++) {  // free/ballの四元数は正規化し直す
            int t = mj->jnt_type[j], adr = mj->jnt_qposadr[j] + (t == mjJNT_FREE ? 3 : 0);
            if (t == mjJNT_FREE || t == mjJNT_BALL) {
                float n = 0;
                for (int k = 0; k < 4; k++) n += q[e * nq + adr + k] * q[e * nq + adr + k];
                n = std::sqrt(n);
                for (int k = 0; k < 4; k++) q[e * nq + adr + k] /= n;
            }
        }
    }
    mjmlx_batched_set_state(gpu, q.data(), v.data());

    std::vector<float> ctrl(B * nu);
    std::map<int, std::vector<double>> acc_errs;  // accelerometerは1step差分で衝撃(関節リミット等)の瞬間に乖離するので、p95で判定する
    std::map<int, double> max_err, max_ref;       // sensor id -> max|diff|, max|参照値|
    std::vector<double> step_err_acc;
    bool finite = true;
    for (int t = 0; t < steps; t++) {
        for (int e = 0; e < B; e++) for (int j = 0; j < nu; j++) ctrl[e * nu + j] = 0.3f * std::sin(0.05f * t + j + e);
        std::vector<float> q0(B * nq), v0(B * nv);
        mjmlx_batched_get_state(gpu, q0.data(), v0.data(), &a, &b);
        mjmlx_batched_step(gpu, nu ? ctrl.data() : nullptr);
        int ng = 0;
        const float* sg = mjmlx_batched_get_sensordata(gpu, &ng);
        if (ng != B * nsd) { fprintf(stderr, "FAIL: sensordata size %d\n", ng); return 1; }
        std::vector<float> sc(B * nsd);
        for (int e = 0; e < B; e++) {
            for (int i = 0; i < nq; i++) d->qpos[i] = q0[e * nq + i];
            for (int i = 0; i < nv; i++) d->qvel[i] = v0[e * nv + i];
            for (int i = 0; i < nu; i++) d->ctrl[i] = ctrl[e * nu + i];
            mj_forward(mj, d);
            for (int i = 0; i < nsd; i++) sc[e * nsd + i] = (float)d->sensordata[i];
        }
        for (int s = 0; s < mj->nsensor; s++) {
            if (t == 0 && mj->sensor_type[s] == mjSENS_ACCELEROMETER) continue;  // 初回は差分の前stepが無い
            for (int e = 0; e < B; e++)
                for (int k = 0; k < mj->sensor_dim[s]; k++) {
                    float g = sg[e * nsd + mj->sensor_adr[s] + k], c = sc[e * nsd + mj->sensor_adr[s] + k];
                    if (!std::isfinite(g) || !std::isfinite(c)) { finite = false; continue; }
                    double d = std::fabs((double)g - c);
                    // 四元数は符号の自由度を許す
                    if (mj->sensor_type[s] == mjSENS_FRAMEQUAT || mj->sensor_type[s] == mjSENS_BALLQUAT) {
                        double dm = 0, dp = 0;
                        for (int kk = 0; kk < 4; kk++) {
                            double gg = sg[e * nsd + mj->sensor_adr[s] + kk], cc2 = sc[e * nsd + mj->sensor_adr[s] + kk];
                            dm = std::max(dm, std::fabs(gg - cc2)); dp = std::max(dp, std::fabs(gg + cc2));
                        }
                        d = std::min(dm, dp);
                    }
                    if (mj->sensor_type[s] == mjSENS_ACCELEROMETER) acc_errs[s].push_back(d);
                    max_err[s] = std::max(max_err[s], d);
                    max_ref[s] = std::max(max_ref[s], (double)std::fabs(c));
                }
        }
    }

    int fails = 0;
    printf("=== test_sensors %s (steps=%d B=%d nsensordata=%d) ===\n", argv[1], steps, B, nsd);
    for (int s = 0; s < mj->nsensor; s++) {
        bool acc = mj->sensor_type[s] == mjSENS_ACCELEROMETER;
        if (acc && !check_acc) { printf("  %-14s SKIP (accelerometer)\n", mj_id2name(mj, mjOBJ_SENSOR, s)); continue; }
        double lim = acc ? tol_acc : tol;
        if (acc && !acc_errs[s].empty()) {
            auto& v = acc_errs[s];
            std::sort(v.begin(), v.end());
            max_err[s] = v[(size_t)(0.95 * (v.size() - 1))];
        }
        bool ok = max_err[s] <= lim;
        printf("  %-14s max|diff|=%.3e (%s tol %.1e, max|ref|=%.2e) %s\n", mj_id2name(mj, mjOBJ_SENSOR, s), max_err[s], acc ? "p95," : "", lim, max_ref[s], ok ? "ok" : "FAIL");
        fails += !ok;
    }
    if (!finite) { printf("FAIL: non-finite sensordata\n"); fails++; }
    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    mj_deleteData(d); mjmlx_batched_free(gpu); mjmlx_free_model(model);
    return fails ? 1 : 0;
}
