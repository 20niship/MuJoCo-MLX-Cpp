// Benchmark framework for MuJoCo-MLX-Cpp
// Provides timing macros, statistics, CSV output, and MuJoCo C comparison.
// Builds on top of test_utils.h conventions.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include "mjmlx/mjmlx.h"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

// ── RSS memory measurement ───────────────────────────────────────────────────

static inline size_t bench_get_rss_bytes() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

// ── Git hash helper ──────────────────────────────────────────────────────────

static inline std::string bench_git_hash() {
    char buf[64] = {0};
    FILE* fp = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (fp) {
        if (fgets(buf, sizeof(buf), fp)) {
            size_t len = strlen(buf);
            if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
        }
        pclose(fp);
    }
    return strlen(buf) > 0 ? std::string(buf) : "unknown";
}

// ── Timing helper ────────────────────────────────────────────────────────────

using BenchClock = std::chrono::high_resolution_clock;
using BenchTimePoint = BenchClock::time_point;

static inline double bench_elapsed_us(BenchTimePoint start, BenchTimePoint end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// ── Statistics ───────────────────────────────────────────────────────────────

struct BenchStats {
    std::string name;
    std::string model_name;
    int num_envs = 1;
    int num_steps = 0;
    double work = 0;  // total work units (e.g., num_envs * num_steps)

    std::vector<double> samples_us;  // per-iteration timings in microseconds

    double mean_us = 0;
    double stddev_us = 0;
    double min_us = 0;
    double max_us = 0;
    double p50_us = 0;
    double p95_us = 0;
    double total_us = 0;
    double steps_per_sec = 0;

    double mj_c_steps_per_sec = 0;  // MuJoCo C reference throughput
    double mj_c_ratio = 0;          // mjmlx / mujoco_c speedup

    size_t rss_before = 0;
    size_t rss_after = 0;
    double peak_rss_mb = 0;

    void compute() {
        if (samples_us.empty()) return;
        int n = (int)samples_us.size();

        total_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
        mean_us = total_us / n;

        double sq_sum = 0;
        for (auto s : samples_us) sq_sum += (s - mean_us) * (s - mean_us);
        stddev_us = (n > 1) ? std::sqrt(sq_sum / (n - 1)) : 0;

        std::vector<double> sorted = samples_us;
        std::sort(sorted.begin(), sorted.end());
        min_us = sorted.front();
        max_us = sorted.back();
        p50_us = sorted[n / 2];
        p95_us = sorted[(int)(n * 0.95)];

        if (work > 0 && total_us > 0) {
            steps_per_sec = work / (total_us / 1e6);
        }

        if (mj_c_steps_per_sec > 0 && steps_per_sec > 0) {
            mj_c_ratio = steps_per_sec / mj_c_steps_per_sec;
        }

        peak_rss_mb = (rss_after > rss_before)
            ? (double)(rss_after - rss_before) / (1024.0 * 1024.0)
            : 0;
    }
};

// ── Benchmark tracking ───────────────────────────────────────────────────────

static std::vector<BenchStats> _bench_results;

// ── Macros ───────────────────────────────────────────────────────────────────

// BENCH_RUN: Execute a benchmark with warmup and measurement iterations.
// Usage:
//   BenchStats stats;
//   stats.name = "my_bench";
//   stats.model_name = "humanoid";
//   BENCH_RUN(stats, warmup_iters, measure_iters, {
//       // code to benchmark
//   });
//   stats.compute();
//   _bench_results.push_back(stats);

#define BENCH_RUN(stats_var, warmup, measure, block) do { \
    (stats_var).rss_before = bench_get_rss_bytes(); \
    for (int _bw = 0; _bw < (warmup); _bw++) { block; } \
    (stats_var).samples_us.clear(); \
    (stats_var).samples_us.reserve(measure); \
    for (int _bm = 0; _bm < (measure); _bm++) { \
        auto _bt0 = BenchClock::now(); \
        block; \
        auto _bt1 = BenchClock::now(); \
        (stats_var).samples_us.push_back(bench_elapsed_us(_bt0, _bt1)); \
    } \
    (stats_var).rss_after = bench_get_rss_bytes(); \
} while(0)

// ── MuJoCo C reference benchmark ─────────────────────────────────────────────

// Run N steps of MuJoCo C simulation and return steps/sec.
static inline double bench_mujoco_c_throughput(
    mjModel* m, mjData* d, int num_steps, int warmup_steps = 5)
{
    for (int i = 0; i < warmup_steps; i++) mj_step(m, d);
    mj_resetData(m, d);

    auto t0 = BenchClock::now();
    for (int i = 0; i < num_steps; i++) mj_step(m, d);
    auto t1 = BenchClock::now();

    double elapsed_s = bench_elapsed_us(t0, t1) / 1e6;
    return (elapsed_s > 0) ? num_steps / elapsed_s : 0;
}

// ── Reporting ────────────────────────────────────────────────────────────────

static inline void bench_print_stats(const BenchStats& s) {
    printf("  %-30s  mean=%10.1f us  stddev=%8.1f  p50=%10.1f  p95=%10.1f  min=%10.1f  max=%10.1f",
           s.name.c_str(), s.mean_us, s.stddev_us, s.p50_us, s.p95_us, s.min_us, s.max_us);
    if (s.steps_per_sec > 0) printf("  steps/s=%.0f", s.steps_per_sec);
    if (s.mj_c_ratio > 0) printf("  vs_mjc=%.2fx", s.mj_c_ratio);
    if (s.peak_rss_mb > 0) printf("  rss=+%.1fMB", s.peak_rss_mb);
    printf("\n");
}

static inline void bench_report_all() {
    printf("\n========== BENCHMARK RESULTS ==========\n");
    for (const auto& s : _bench_results) {
        bench_print_stats(s);
    }
    printf("========================================\n");
}

// ── CSV output ───────────────────────────────────────────────────────────────

// Append benchmark results to a CSV file.
// Creates the file with headers if it doesn't exist.
static inline void bench_write_csv(const char* csv_path) {
    bool need_header = false;
    FILE* check = fopen(csv_path, "r");
    if (!check) {
        need_header = true;
    } else {
        fclose(check);
    }

    FILE* fp = fopen(csv_path, "a");
    if (!fp) {
        fprintf(stderr, "WARNING: cannot open %s for writing\n", csv_path);
        return;
    }

    if (need_header) {
        fprintf(fp, "timestamp,git_hash,benchmark,model,num_envs,steps,"
                    "mean_us,stddev_us,p50_us,p95_us,min_us,max_us,"
                    "steps_per_sec,mj_c_steps_per_sec,mj_c_ratio,peak_rss_mb\n");
    }

    std::string git = bench_git_hash();

    // Get timestamp
    char ts[64];
    time_t now = time(nullptr);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    for (const auto& s : _bench_results) {
        fprintf(fp, "%s,%s,%s,%s,%d,%d,"
                    "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
                    "%.0f,%.0f,%.2f,%.1f\n",
                ts, git.c_str(), s.name.c_str(), s.model_name.c_str(),
                s.num_envs, s.num_steps,
                s.mean_us, s.stddev_us, s.p50_us, s.p95_us, s.min_us, s.max_us,
                s.steps_per_sec, s.mj_c_steps_per_sec, s.mj_c_ratio, s.peak_rss_mb);
    }

    fclose(fp);
    printf("Benchmark results appended to %s\n", csv_path);
}

// ── Convenience: single-env scalar benchmark ─────────────────────────────────

// Benchmark single-env stepping via batched API (num_envs=1, CPU mode).
// Automatically computes MuJoCo C comparison throughput.
static inline BenchStats bench_scalar_step(
    const char* name,
    const char* model_name,
    const char* xml_str,
    int num_steps = 100,
    int warmup_iters = 3,
    int measure_iters = 10)
{
    BenchStats stats;
    stats.name = name;
    stats.model_name = model_name;
    stats.num_envs = 1;
    stats.num_steps = num_steps;
    stats.work = num_steps;

    MjmlxModel* model = mjmlx_load_model_from_string(xml_str);
    if (!model) {
        fprintf(stderr, "  SKIP %s: failed to load mjmlx model\n", name);
        return stats;
    }

    MjmlxBatchedConfig config = {};
    config.num_envs = 1;
    config.use_gpu = 0;
    config.foot_contacts_only = 0;
    config.integrator = MJMLX_INTEGRATOR_EULER;

    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
    if (!sim) {
        fprintf(stderr, "  SKIP %s: failed to create batched sim\n", name);
        mjmlx_free_model(model);
        return stats;
    }

    BENCH_RUN(stats, warmup_iters, measure_iters, {
        std::vector<int> mask(1, 1);
        mjmlx_batched_reset(sim, mask.data());
        for (int s = 0; s < num_steps; s++) {
            mjmlx_batched_step(sim, nullptr);
        }
        int tmp;
        mjmlx_batched_get_xpos(sim, &tmp);
    });

    mjmlx_batched_free(sim);
    mjmlx_free_model(model);

    // MuJoCo C reference
    char error[1024] = {0};
    std::string path = write_temp_xml(xml_str);
    mjModel* mj_m = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
    unlink(path.c_str());
    if (mj_m) {
        mjData* mj_d = mj_makeData(mj_m);
        stats.mj_c_steps_per_sec = bench_mujoco_c_throughput(mj_m, mj_d, num_steps);
        mj_deleteData(mj_d);
        mj_deleteModel(mj_m);
    }

    stats.compute();
    return stats;
}

// ── Convenience: batched GPU benchmark ───────────────────────────────────────

// Benchmark mjmlx batched stepping: warmup (incl. Metal compile), then measure.
static inline BenchStats bench_batched_step(
    const char* name,
    const char* model_name,
    const char* xml_str,
    int num_envs = 64,
    int num_steps = 50,
    int warmup_iters = 2,
    int measure_iters = 5)
{
    BenchStats stats;
    stats.name = name;
    stats.model_name = model_name;
    stats.num_envs = num_envs;
    stats.num_steps = num_steps;
    stats.work = (double)num_envs * num_steps;

    MjmlxModel* model = mjmlx_load_model_from_string(xml_str);
    if (!model) {
        fprintf(stderr, "  SKIP %s: failed to load mjmlx model\n", name);
        return stats;
    }

    MjmlxBatchedConfig config;
    config.num_envs = num_envs;
    config.use_gpu = 1;
    config.foot_contacts_only = 0;
    config.integrator = MJMLX_INTEGRATOR_EULER;

    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
    if (!sim) {
        fprintf(stderr, "  SKIP %s: failed to create batched sim\n", name);
        mjmlx_free_model(model);
        return stats;
    }

    BENCH_RUN(stats, warmup_iters, measure_iters, {
        std::vector<int> mask(num_envs, 1);
        mjmlx_batched_reset(sim, mask.data());
        for (int s = 0; s < num_steps; s++) {
            mjmlx_batched_step(sim, nullptr);
        }
        // Force eval
        int tmp;
        mjmlx_batched_get_xpos(sim, &tmp);
    });

    mjmlx_batched_free(sim);
    mjmlx_free_model(model);

    // MuJoCo C reference (single env, for comparison)
    char error[1024] = {0};
    std::string path = write_temp_xml(xml_str);
    mjModel* mj_m = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
    unlink(path.c_str());
    if (mj_m) {
        mjData* mj_d = mj_makeData(mj_m);
        double mj_c_single = bench_mujoco_c_throughput(mj_m, mj_d, num_steps);
        stats.mj_c_steps_per_sec = mj_c_single;
        mj_deleteData(mj_d);
        mj_deleteModel(mj_m);
    }

    stats.compute();
    return stats;
}

// ── Convenience: file-path scalar/batched benchmarks (external models with mesh assets) ─────

// Like bench_scalar_step but loads from a file path so relative <mesh file="..."/> assets resolve; returns zero-throughput stats if path is empty/load fails.
static inline BenchStats bench_scalar_step_file(
    const char* name,
    const char* model_name,
    const char* path,
    int num_steps = 100,
    int warmup_iters = 3,
    int measure_iters = 10)
{
    BenchStats stats;
    stats.name = name;
    stats.model_name = model_name;
    stats.num_envs = 1;
    stats.num_steps = num_steps;
    stats.work = num_steps;

    if (!path || strlen(path) == 0) return stats;

    MjmlxModel* model = mjmlx_load_model(path);
    if (!model) {
        fprintf(stderr, "  SKIP %s: failed to load %s\n", name, path);
        return stats;
    }

    MjmlxBatchedConfig config = {};
    config.num_envs = 1;
    config.use_gpu = 0;
    config.foot_contacts_only = 0;
    config.integrator = MJMLX_INTEGRATOR_EULER;

    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
    if (!sim) {
        fprintf(stderr, "  SKIP %s: failed to create batched sim\n", name);
        mjmlx_free_model(model);
        return stats;
    }

    BENCH_RUN(stats, warmup_iters, measure_iters, {
        std::vector<int> mask(1, 1);
        mjmlx_batched_reset(sim, mask.data());
        for (int s = 0; s < num_steps; s++) {
            mjmlx_batched_step(sim, nullptr);
        }
        int tmp;
        mjmlx_batched_get_xpos(sim, &tmp);
    });

    mjmlx_batched_free(sim);
    mjmlx_free_model(model);

    char error[1024] = {0};
    mjModel* mj_m = mj_loadXML(path, nullptr, error, sizeof(error));
    if (mj_m) {
        mjData* mj_d = mj_makeData(mj_m);
        stats.mj_c_steps_per_sec = bench_mujoco_c_throughput(mj_m, mj_d, num_steps);
        mj_deleteData(mj_d);
        mj_deleteModel(mj_m);
    }

    stats.compute();
    return stats;
}

// Same as bench_batched_step but loads from a real file path (see bench_scalar_step_file).
static inline BenchStats bench_batched_step_file(
    const char* name,
    const char* model_name,
    const char* path,
    int num_envs = 32,
    int num_steps = 20,
    int warmup_iters = 1,
    int measure_iters = 3)
{
    BenchStats stats;
    stats.name = name;
    stats.model_name = model_name;
    stats.num_envs = num_envs;
    stats.num_steps = num_steps;
    stats.work = (double)num_envs * num_steps;

    if (!path || strlen(path) == 0) return stats;

    MjmlxModel* model = mjmlx_load_model(path);
    if (!model) {
        fprintf(stderr, "  SKIP %s: failed to load %s\n", name, path);
        return stats;
    }

    MjmlxBatchedConfig config = {};
    config.num_envs = num_envs;
    config.use_gpu = 1;
    config.foot_contacts_only = 0;
    config.integrator = MJMLX_INTEGRATOR_EULER;

    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
    if (!sim) {
        fprintf(stderr, "  SKIP %s: failed to create batched sim\n", name);
        mjmlx_free_model(model);
        return stats;
    }

    BENCH_RUN(stats, warmup_iters, measure_iters, {
        std::vector<int> mask(num_envs, 1);
        mjmlx_batched_reset(sim, mask.data());
        for (int s = 0; s < num_steps; s++) {
            mjmlx_batched_step(sim, nullptr);
        }
        int tmp;
        mjmlx_batched_get_xpos(sim, &tmp);
    });

    mjmlx_batched_free(sim);
    mjmlx_free_model(model);

    char error[1024] = {0};
    mjModel* mj_m = mj_loadXML(path, nullptr, error, sizeof(error));
    if (mj_m) {
        mjData* mj_d = mj_makeData(mj_m);
        stats.mj_c_steps_per_sec = bench_mujoco_c_throughput(mj_m, mj_d, num_steps);
        mj_deleteData(mj_d);
        mj_deleteModel(mj_m);
    }

    stats.compute();
    return stats;
}

// ── Temp file helper ─────────────────────────────────────────────────────────
// bench_utils.h expects test_utils.h to be included first (provides write_temp_xml).
// All benchmark files should #include "test_utils.h" before #include "bench_utils.h".
