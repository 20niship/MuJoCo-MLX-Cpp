// Baseline benchmark: establishes pre-conformance throughput on all models.
// Run this ONCE before starting conformance work to capture starting metrics.
// Subsequent phase benchmarks compare against these baselines.
//
// Usage:
//   bench_baseline [humanoid.xml] [csv_path]
//
// If humanoid.xml is provided, it is included in the benchmark suite.
// CSV output defaults to benchmarks/baseline.csv.

#include "test_utils.h"
#include "bench_utils.h"
#include "test_models.h"
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    const char* humanoid_path = nullptr;
    if (argc >= 2 && strlen(argv[1]) > 0) {
        humanoid_path = argv[1];
    }
    const char* csv_path = (argc >= 3) ? argv[2] : "benchmarks/baseline.csv";

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

    // Humanoid (external, if provided)
    if (humanoid_path) {
        BenchStats hs;
        hs.name = "scalar/humanoid";
        hs.model_name = "humanoid";
        hs.num_envs = 1;
        hs.num_steps = 100;
        hs.work = 100;

        MjmlxModel* model = mjmlx_load_model(humanoid_path);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            BENCH_RUN(hs, 3, 10, {
                mjmlx_reset_data(model, data);
                for (int s = 0; s < 100; s++) mjmlx_step(model, data);
                int tmp; mjmlx_get_xpos(data, &tmp);
            });
            mjmlx_free_data(data);
            mjmlx_free_model(model);

            // MuJoCo C reference
            char error[1024] = {0};
            mjModel* mj_m = mj_loadXML(humanoid_path, nullptr, error, sizeof(error));
            if (mj_m) {
                mjData* mj_d = mj_makeData(mj_m);
                hs.mj_c_steps_per_sec = bench_mujoco_c_throughput(mj_m, mj_d, 100);
                mj_deleteData(mj_d);
                mj_deleteModel(mj_m);
            }

            hs.compute();
            _bench_results.push_back(hs);
            bench_print_stats(hs);
        } else {
            printf("  SKIP scalar/humanoid: failed to load %s\n", humanoid_path);
        }
    }

    // ── Batched benchmarks (GPU path, 64 envs) ──────────────────────────────

    printf("\n--- Batched Benchmarks (64 envs, GPU) ---\n");

    // Simple pendulum batched
    {
        auto s = bench_batched_step("batched/pendulum", "pendulum", SIMPLE_PENDULUM_XML, 64, 20, 1, 3);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // High-DOF tree batched (critical for Synth Metal pipeline)
    // Uses fewer envs and steps due to large model size
    {
        auto s = bench_batched_step("batched/high_dof_tree", "high_dof_tree", HIGH_DOF_TREE_XML, 16, 10, 1, 2);
        _bench_results.push_back(s);
        bench_print_stats(s);
    }

    // Humanoid batched (if provided)
    if (humanoid_path) {
        BenchStats hs;
        hs.name = "batched/humanoid";
        hs.model_name = "humanoid";
        hs.num_envs = 32;
        hs.num_steps = 20;
        hs.work = 32.0 * 20;

        MjmlxModel* model = mjmlx_load_model(humanoid_path);
        if (model) {
            MjmlxBatchedConfig config;
            config.num_envs = 32;
            config.use_gpu = 1;
            config.foot_contacts_only = 0;
            config.integrator = MJMLX_INTEGRATOR_EULER;

            MjmlxBatchedSim* sim = mjmlx_batched_create(model, &config);
            if (sim) {
                BENCH_RUN(hs, 1, 3, {
                    std::vector<int> mask(32, 1);
                    mjmlx_batched_reset(sim, mask.data());
                    for (int s = 0; s < 20; s++) mjmlx_batched_step(sim, nullptr);
                    int tmp; mjmlx_batched_get_xpos(sim, &tmp);
                });
                mjmlx_batched_free(sim);

                // MuJoCo C single-env reference
                char error[1024] = {0};
                mjModel* mj_m = mj_loadXML(humanoid_path, nullptr, error, sizeof(error));
                if (mj_m) {
                    mjData* mj_d = mj_makeData(mj_m);
                    hs.mj_c_steps_per_sec = bench_mujoco_c_throughput(mj_m, mj_d, 20);
                    mj_deleteData(mj_d);
                    mj_deleteModel(mj_m);
                }

                hs.compute();
                _bench_results.push_back(hs);
                bench_print_stats(hs);
            } else {
                printf("  SKIP batched/humanoid: failed to create batched sim\n");
            }
            mjmlx_free_model(model);
        }
    }

    // ── Summary and CSV output ───────────────────────────────────────────────

    bench_report_all();
    bench_write_csv(csv_path);

    printf("\nBaseline benchmarks complete. Results in %s\n", csv_path);
    return 0;
}
