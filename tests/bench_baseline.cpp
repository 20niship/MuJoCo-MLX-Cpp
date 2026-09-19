// Usage: bench_baseline [humanoid.xml] [csv_path] [go2.xml] [h1.xml]; env MJMLX_BENCH_ONLY=<name>で1件だけ実行(GPU hangが他の計測を巻き込まないように分離実行する用)。

#include "test_utils.h"
#include "bench_utils.h"
#include "test_models.h"
#include <cstdlib>
#include <cstring>
#include <functional>
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
    const char* only = std::getenv("MJMLX_BENCH_ONLY");

    printf("=== MuJoCo-MLX-Cpp Baseline Benchmarks ===\n");
    printf("Git: %s\n\n", bench_git_hash().c_str());

    std::vector<std::pair<std::string, std::function<void()>>> benches;

    auto add_scalar = [&](const char* name, const char* model, const char* xml, int steps, int warmup, int measure) {
        benches.push_back({name, [=]() {
                                auto s = bench_scalar_step(name, model, xml, steps, warmup, measure);
                                _bench_results.push_back(s);
                                bench_print_stats(s);
                            }});
    };
    add_scalar("scalar/pendulum", "pendulum", SIMPLE_PENDULUM_XML, 100, 3, 5);
    add_scalar("scalar/cfrc_ext", "cfrc_ext", CFRC_EXT_XML, 50, 2, 5);
    add_scalar("scalar/high_dof_tree", "high_dof_tree", HIGH_DOF_TREE_XML, 20, 2, 3);
    add_scalar("scalar/exclude", "exclude", EXCLUDE_XML, 50, 2, 5);
    add_scalar("scalar/t_shape", "t_shape", T_SHAPE_XML, 50, 2, 5);

    for (auto& mp : std::vector<std::pair<const char*, const char*>>{
             {"humanoid", humanoid_path}, {"go2", go2_path}, {"h1", h1_path}}) {
        std::string name = std::string("scalar/") + mp.first;
        const char* model = mp.first;
        const char* path = mp.second;
        benches.push_back({name, [=]() {
                                auto s = bench_scalar_step_file(name.c_str(), model, path, 100, 3, 10);
                                if (s.steps_per_sec > 0) {
                                    _bench_results.push_back(s);
                                    bench_print_stats(s);
                                }
                            }});
    }

    const int kBatchSizes[] = {64, 256, 2048};

    auto add_batched = [&](const char* model, const char* xml, int steps, int warmup, int measure) {
        for (int b : kBatchSizes) {
            std::string name = std::string("batched/") + model + "/B" + std::to_string(b);
            benches.push_back({name, [=]() {
                                    auto s = bench_batched_step(name.c_str(), model, xml, b, steps, warmup, measure);
                                    _bench_results.push_back(s);
                                    bench_print_stats(s);
                                }});
        }
    };
    add_batched("pendulum", SIMPLE_PENDULUM_XML, 60, 1, 3);
    add_batched("t_shape", T_SHAPE_XML, 60, 1, 3);
    add_batched("cfrc_ext", CFRC_EXT_XML, 60, 1, 3);
    add_batched("exclude", EXCLUDE_XML, 60, 1, 3);
    add_batched("high_dof_tree", HIGH_DOF_TREE_XML, 30, 1, 2);

    for (int b : kBatchSizes) {
        for (auto& mp : std::vector<std::pair<const char*, const char*>>{
                 {"humanoid", humanoid_path}, {"go2", go2_path}, {"h1", h1_path}}) {
            std::string name = std::string("batched/") + mp.first + "/B" + std::to_string(b);
            const char* model = mp.first;
            const char* path = mp.second;
            benches.push_back({name, [=]() {
                                    auto s = bench_batched_step_file(name.c_str(), model, path, b, 60, 1, 3);
                                    if (s.steps_per_sec > 0) {
                                        _bench_results.push_back(s);
                                        bench_print_stats(s);
                                    }
                                }});
        }
    }

    for (auto& [name, fn] : benches) {
        if (only && name != only) continue;
        try {
            fn();
        } catch (const std::exception& e) {
            printf("  CRASH %s: %s\n", name.c_str(), e.what());
        }
    }

    bench_report_all();
    bench_write_csv(csv_path);

    printf("\nBaseline benchmarks complete. Results in %s\n", csv_path);
    return 0;
}
