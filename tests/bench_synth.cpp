// Phase 5 Benchmark: Synth GPU pipeline SPS measurement
// Measures GPU and CPU throughput at various env counts to evaluate
// the full Metal pipeline (forward + collision + solver + euler_devmem).
//
// Usage: bench_synth <synth_humanoid.xml>

#include "mjmlx/mjmlx.h"
#include <mujoco/mujoco.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    const char* label;
    int num_envs;
    int num_steps;
    double total_ms;
    double sps;
    double ms_per_step;
    float max_vel;
    bool stable;
};

static BenchResult run_bench(MjmlxModel* model, int num_envs, int num_steps,
                             bool use_gpu) {
    BenchResult r = {};
    r.num_envs = num_envs;
    r.num_steps = num_steps;
    r.label = use_gpu ? "GPU" : "CPU";

    MjmlxBatchedConfig config = {};
    config.num_envs = num_envs;
    config.use_gpu = use_gpu ? 1 : 0;
    config.foot_contacts_only = 0;
    config.integrator = MJMLX_INTEGRATOR_EULER;

    auto* sim = mjmlx_batched_create(model, &config);
    if (!sim) {
        printf("  SKIP %s %d envs: create failed\n", r.label, num_envs);
        return r;
    }

    // Warmup
    mjmlx_batched_step(sim, nullptr);
    { int tmp; mjmlx_batched_get_xpos(sim, &tmp); }

    // Timed run
    auto t0 = Clock::now();
    for (int s = 0; s < num_steps; s++) {
        mjmlx_batched_step(sim, nullptr);
    }
    { int tmp; mjmlx_batched_get_xpos(sim, &tmp); }
    auto t1 = Clock::now();

    r.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.sps = (double)(num_envs * num_steps) / (r.total_ms / 1000.0);
    r.ms_per_step = r.total_ms / num_steps;

    // Stability check
    int n;
    auto* qvel = mjmlx_batched_get_qvel(sim, &n);
    r.max_vel = 0;
    r.stable = true;
    if (qvel) {
        for (int i = 0; i < n; i++) {
            float v = std::abs(qvel[i]);
            if (v > r.max_vel) r.max_vel = v;
        }
        if (std::isnan(r.max_vel) || r.max_vel > 1e4f) r.stable = false;
    }

    mjmlx_batched_free(sim);
    return r;
}

static void print_result(const BenchResult& r) {
    printf("  %-3s %4d envs × %3d steps: %8.1f ms  %8.0f SPS  %6.1f ms/step  maxvel=%.1f %s\n",
           r.label, r.num_envs, r.num_steps,
           r.total_ms, r.sps, r.ms_per_step, r.max_vel,
           r.stable ? "" : "UNSTABLE");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: bench_synth <synth_humanoid.xml>\n");
        return 1;
    }

    auto* model = mjmlx_load_model(argv[1]);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    auto info = mjmlx_model_info(model);
    printf("=== Synth GPU Pipeline Benchmark (Phase 5) ===\n");
    printf("  Model: nq=%d nv=%d nu=%d nbody=%d ngeom=%d\n\n",
           info.nq, info.nv, info.nu, info.nbody, info.ngeom);

    // MuJoCo C single-env reference
    {
        char error[1024] = {0};
        mjModel* mj_m = mj_loadXML(argv[1], nullptr, error, sizeof(error));
        if (mj_m) {
            mjData* mj_d = mj_makeData(mj_m);
            for (int i = 0; i < 5; i++) mj_step(mj_m, mj_d);
            mj_resetData(mj_m, mj_d);
            auto t0 = Clock::now();
            int N = 200;
            for (int i = 0; i < N; i++) mj_step(mj_m, mj_d);
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("  MuJoCo C single-env: %.0f steps/sec (%.2f ms/step)\n\n",
                   N / (ms / 1000.0), ms / N);
            mj_deleteData(mj_d);
            mj_deleteModel(mj_m);
        }
    }

    std::vector<BenchResult> results;

    // CPU benchmarks
    printf("--- CPU Batched (thread pool) ---\n");
    int cpu_envs[] = {1, 4, 16, 64, 128, 252};
    for (int e : cpu_envs) {
        int steps = (e <= 16) ? 50 : 20;
        auto r = run_bench(model, e, steps, false);
        print_result(r);
        results.push_back(r);
    }

    // GPU benchmarks
    printf("\n--- GPU Metal Pipeline ---\n");
    int gpu_envs[] = {1, 4, 8, 16, 32, 64, 128, 256};
    for (int e : gpu_envs) {
        int steps = (e <= 16) ? 20 : 10;
        auto r = run_bench(model, e, steps, true);
        print_result(r);
        results.push_back(r);
    }

    // Summary
    printf("\n--- Summary ---\n");
    double best_cpu_sps = 0, best_gpu_sps = 0;
    int best_cpu_envs = 0, best_gpu_envs = 0;
    for (auto& r : results) {
        if (r.stable) {
            if (r.label[0] == 'C' && r.sps > best_cpu_sps) {
                best_cpu_sps = r.sps; best_cpu_envs = r.num_envs;
            }
            if (r.label[0] == 'G' && r.sps > best_gpu_sps) {
                best_gpu_sps = r.sps; best_gpu_envs = r.num_envs;
            }
        }
    }
    printf("  Best CPU: %.0f SPS @ %d envs\n", best_cpu_sps, best_cpu_envs);
    printf("  Best GPU: %.0f SPS @ %d envs\n", best_gpu_sps, best_gpu_envs);
    if (best_cpu_sps > 0)
        printf("  GPU/CPU ratio: %.2fx\n", best_gpu_sps / best_cpu_sps);

    mjmlx_free_model(model);
    return 0;
}
