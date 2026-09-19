// Baseline benchmark: establishes pre-conformance throughput on all models.
// Run this ONCE before starting conformance work to capture starting metrics.
// Subsequent phase benchmarks compare against these baselines.
//
// Usage: bench_baseline [humanoid.xml] [csv_path] [go2.xml] [h1.xml] -- any model path may be "".

#include "test_utils.h"
#include "bench_utils.h"
#include "test_models.h"
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    const char* humanoid_path = nullptr;
    if (argc >= 2 && strlen(argv[1]) > 0) {
        humanoid_path = argv[1];
    }
    const char* csv_path = (argc >= 3) ? argv[2] : "benchmarks/baseline.csv";
    const char* go2_path = (argc >= 4 && strlen(argv[3]) > 0) ? argv[3] : nullptr;
    const char* h1_path = (argc >= 5 && strlen(argv[4]) > 0) ? argv[4] : nullptr;

    printf("=== MuJoCo-MLX-Cpp Baseline Benchmarks ===\n");
    printf("Git: %s\n\n", bench_git_hash().c_str());

    // ── Scalar benchmarks (single env, CPU path) ─────────────────────────────

    printf("--- Scalar Benchmarks (single env) ---\n");

    // Simple pendulum (minimal model, measures pure overhead)
    {
        auto s = bench_scalar_step("scalar/pendulum", "pendulum", SIMPLE_PENDULUM_XML, 100, 3, 5);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // Cfrc_ext model (sphere on plane -- contact)
    {
        auto s = bench_scalar_step("scalar/cfrc_ext", "cfrc_ext", CFRC_EXT_XML, 50, 2, 5);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // High-DOF tree (Synth-scale, nv~69)
    {
        auto s = bench_scalar_step("scalar/high_dof_tree", "high_dof_tree", HIGH_DOF_TREE_XML, 20, 2, 3);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // Exclude model (articulated body with excludes)
    {
        auto s = bench_scalar_step("scalar/exclude", "exclude", EXCLUDE_XML, 50, 2, 5);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // T-shape push task (self-contained, no external assets)
    {
        auto s = bench_scalar_step("scalar/t_shape", "t_shape", T_SHAPE_XML, 50, 2, 5);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // External models (menagerie-style, may reference mesh assets) — skipped if path not given
    for (auto& mp : std::vector<std::pair<const char*, const char*>>{
             {"humanoid", humanoid_path}, {"go2", go2_path}, {"h1", h1_path}}) {
        std::string name = std::string("scalar/") + mp.first;
        auto s = bench_scalar_step_file(name.c_str(), mp.first, mp.second, 100, 3, 10);
        if (s.steps_per_sec > 0) {
            _bench_results.push_back(s);
            bench_print_stats(s);
        }
    }

    // ── Batched benchmarks (GPU path, B=64/256/2048 envs) ──────────────────

    const int kBatchSizes[] = {64, 256, 2048};

    printf("\n--- Batched Benchmarks (B=64/256/2048, GPU) ---\n");

    // try/catch per benchmark: one throwing (e.g. Metal resource-limit error) must not lose results already collected.
    for (int b : kBatchSizes) {
        std::string name = std::string("batched/pendulum/B") + std::to_string(b);
        try {
            auto s = bench_batched_step(name.c_str(), "pendulum", SIMPLE_PENDULUM_XML, b, 20, 1, 3);
            _bench_results.push_back(s);
            bench_print_stats(s);
        } catch (const std::exception& e) {
            printf("  CRASH %s: %s\n", name.c_str(), e.what());
        }
    }

    for (int b : kBatchSizes) {
        std::string name = std::string("batched/t_shape/B") + std::to_string(b);
        try {
            auto s = bench_batched_step(name.c_str(), "t_shape", T_SHAPE_XML, b, 20, 1, 3);
            _bench_results.push_back(s);
            bench_print_stats(s);
        } catch (const std::exception& e) {
            printf("  CRASH %s: %s\n", name.c_str(), e.what());
        }
    }

    for (int b : kBatchSizes) {
        std::string name = std::string("batched/cfrc_ext/B") + std::to_string(b);
        try {
            auto s = bench_batched_step(name.c_str(), "cfrc_ext", CFRC_EXT_XML, b, 20, 1, 3);
            _bench_results.push_back(s);
            bench_print_stats(s);
        } catch (const std::exception& e) {
            printf("  CRASH %s: %s\n", name.c_str(), e.what());
        }
    }

    for (int b : kBatchSizes) {
        std::string name = std::string("batched/exclude/B") + std::to_string(b);
        try {
            auto s = bench_batched_step(name.c_str(), "exclude", EXCLUDE_XML, b, 20, 1, 3);
            _bench_results.push_back(s);
            bench_print_stats(s);
        } catch (const std::exception& e) {
            printf("  CRASH %s: %s\n", name.c_str(), e.what());
        }
    }

    for (int b : kBatchSizes) {
        for (auto& mp : std::vector<std::pair<const char*, const char*>>{
                 {"humanoid", humanoid_path}, {"go2", go2_path}, {"h1", h1_path}}) {
            std::string name = std::string("batched/") + mp.first + "/B" + std::to_string(b);
            try {
                auto s = bench_batched_step_file(name.c_str(), mp.first, mp.second, b, 20, 1, 3);
                if (s.steps_per_sec > 0) {
                    _bench_results.push_back(s);
                    bench_print_stats(s);
                }
            } catch (const std::exception& e) {
                printf("  CRASH %s: %s\n", name.c_str(), e.what());
            }
        }
    }

    for (int b : kBatchSizes) {
        std::string name = std::string("batched/high_dof_tree/B") + std::to_string(b);
        try {
            auto s = bench_batched_step(name.c_str(), "high_dof_tree", HIGH_DOF_TREE_XML, b, 10, 1, 2);
            _bench_results.push_back(s);
            bench_print_stats(s);
        } catch (const std::exception& e) {
            printf("  CRASH %s: %s\n", name.c_str(), e.what());
        }
    }

    // ── Summary and CSV output ───────────────────────────────────────────────

    bench_report_all();
    bench_write_csv(csv_path);

    printf("\nBaseline benchmarks complete. Results in %s\n", csv_path);
    return 0;
}
