// tendon対応の性能比較: GPUバッチ(MKX)とCPU(MuJoCo C)のstep/sをバッチ数ごとに測る。tendonなしモデルでの退行確認にも使う。
// Usage: bench_tendon <model.xml> [steps=200] [B1,B2,...=1,64,1024] [--cpu-max-b N=64]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "mjmlx/mjmlx.h"

static double time_steps(MjmlxBatchedSim* sim, int B, int nq, int nv, int nu, int steps) {
    std::vector<float> ctrl((size_t)B * nu, 0.3f);
    for (int i = 0; i < 20; i++) mjmlx_batched_step(sim, ctrl.data());  // warmup(カーネルコンパイル含む)
    int a, b; std::vector<float> qq((size_t)B * nq), vv((size_t)B * nv);
    mjmlx_batched_get_state(sim, qq.data(), vv.data(), &a, &b);  // 同期
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; i++) mjmlx_batched_step(sim, ctrl.data());
    mjmlx_batched_get_state(sim, qq.data(), vv.data(), &a, &b);  // GPUの完了を待つ
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.xml> [steps] [B,B,..] [--cpu-max-b N]\n", argv[0]); return 1; }
    int steps = argc > 2 ? atoi(argv[2]) : 200;
    std::vector<int> bs = {1, 64, 1024};
    if (argc > 3 && argv[3][0] != '-') { bs.clear(); for (char* p = strtok(argv[3], ","); p; p = strtok(nullptr, ",")) bs.push_back(atoi(p)); }
    int cpu_max_b = 64;
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], "--cpu-max-b")) cpu_max_b = atoi(argv[i + 1]);

    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto info = mjmlx_model_info(model);
    printf("model=%s nv=%d nu=%d  (env-steps/s)\n%8s %14s %14s %8s\n", argv[1], info.nv, info.nu, "B", "GPU", "CPU(MuJoCo C)", "GPU/CPU");
    for (int B : bs) {
        double gps = 0, cps = 0;
        for (int gpu = 1; gpu >= 0; gpu--) {
            if (!gpu && B > cpu_max_b) continue;
            MjmlxBatchedConfig c = {};
            c.num_envs = B; c.use_gpu = gpu; c.integrator = MJMLX_INTEGRATOR_EULER;
            auto* sim = mjmlx_batched_create(model, &c);
            std::vector<int> mask(B, 1);
            mjmlx_batched_reset(sim, mask.data());
            double s = time_steps(sim, B, info.nq, info.nv, info.nu, steps);
            (gpu ? gps : cps) = (double)B * steps / s;
            mjmlx_batched_free(sim);
        }
        printf("%8d %14.0f %14s %8s\n", B, gps, cps > 0 ? std::to_string((long)cps).c_str() : "-", cps > 0 ? std::to_string(gps / cps).substr(0, 6).c_str() : "-");
    }
    return 0;
}
