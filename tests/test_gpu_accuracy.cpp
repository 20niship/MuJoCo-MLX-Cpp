// GPU kernel最適化の前後確認用: GPU vs CPU(参照)の数値表示と、--save/--compare による最適化前後のGPU出力一致確認
#include "mjmlx/mjmlx.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static const float kCompareTol = 1e-3f;

struct Stat { double mx = 0, mean = 0; bool bad = false; };

static Stat diff(const std::vector<float>& a, const std::vector<float>& b) {
    Stat s;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { s.bad = true; continue; }
        double d = std::fabs((double)a[i] - b[i]);
        if (d > s.mx) s.mx = d;
        s.mean += d;
    }
    s.mean /= a.size();
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_gpu_accuracy <model.xml> [num_envs=64] [num_steps=50] [--save f | --compare f] [--lift dz]\n");
        return 1;
    }
    const char* save = nullptr; const char* cmp = nullptr; float lift = 0.f;
    std::vector<const char*> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
        else if (!strcmp(argv[i], "--compare") && i + 1 < argc) cmp = argv[++i];
        else if (!strcmp(argv[i], "--lift") && i + 1 < argc) lift = (float)atof(argv[++i]);
        else pos.push_back(argv[i]);
    }
    int B = pos.size() > 1 ? atoi(pos[1]) : 64;
    int T = pos.size() > 2 ? atoi(pos[2]) : 50;

    auto* model = mjmlx_load_model(pos[0]);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nu = info.nu;

    MjmlxBatchedConfig cg = {}; cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedConfig cc = cg; cc.use_gpu = 0;
    auto* gpu = mjmlx_batched_create(model, &cg);
    auto* cpu = mjmlx_batched_create(model, &cc);

    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(cpu, mask.data());
    std::vector<float> q(B * nq), v(B * nv);
    int a, b;
    mjmlx_batched_get_state(cpu, q.data(), v.data(), &a, &b);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (size_t i = 0; i < q.size(); i++) { int k = (int)(i % nq); if (k < 3 || k >= 7) q[i] += 0.02f * u(rng); }
    for (int e = 0; e < B; e++) q[e * nq + 2] += lift;
    for (auto& x : v) x = 0.1f * u(rng);
    mjmlx_batched_set_state(gpu, q.data(), v.data());
    mjmlx_batched_set_state(cpu, q.data(), v.data());

    std::vector<float> ctrl(B * nu), gq(B * nq), gv(B * nv), cq(B * nq), cv(B * nv);
    bool fail = false;
    printf("step |  qpos max      mean     |  qvel max      mean\n");
    for (int t = 1; t <= T; t++) {
        for (auto& x : ctrl) x = 0.3f * u(rng);
        mjmlx_batched_step(gpu, nu ? ctrl.data() : nullptr);
        mjmlx_batched_step(cpu, nu ? ctrl.data() : nullptr);
        if (t == 1 || t == 10 || t == T) {
            mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
            mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
            Stat sq = diff(gq, cq), sv = diff(gv, cv);
            printf("%4d | %10.3e %10.3e | %10.3e %10.3e%s\n", t, sq.mx, sq.mean, sv.mx, sv.mean,
                   (sq.bad || sv.bad) ? "  NaN/Inf!" : "");
            fail |= sq.bad || sv.bad;
        }
    }
    // 最終stepがT<10等で未取得の場合に備え最終状態を再取得
    mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);

    std::vector<float> out(gq);
    out.insert(out.end(), gv.begin(), gv.end());
    if (save) {
        FILE* f = fopen(save, "wb");
        if (!f || fwrite(out.data(), sizeof(float), out.size(), f) != out.size()) { fprintf(stderr, "FAIL: save\n"); return 1; }
        fclose(f);
        printf("saved %zu floats to %s\n", out.size(), save);
    }
    if (cmp) {
        std::vector<float> ref(out.size());
        FILE* f = fopen(cmp, "rb");
        if (!f || fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) { fprintf(stderr, "FAIL: read %s (size mismatch?)\n", cmp); return 1; }
        fclose(f);
        Stat s = diff(out, ref);
        printf("compare: max=%.3e mean=%.3e (tol %.0e)\n", s.mx, s.mean, kCompareTol);
        if (s.bad || s.mx > kCompareTol) { printf("FAIL: compare\n"); fail = true; }
    }
    mjmlx_batched_free(gpu); mjmlx_batched_free(cpu); mjmlx_free_model(model);
    printf(fail ? "FAIL\n" : "OK\n");
    return fail ? 1 : 0;
}
