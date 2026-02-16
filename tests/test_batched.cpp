// Test batched simulation: Metal kernels + per-env step
#include "mjmlx/mjmlx.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_batched <path_to_humanoid.xml> [num_envs] [num_steps]"
                  << std::endl;
        return 1;
    }

    int num_envs = (argc >= 3) ? std::atoi(argv[2]) : 4;
    int num_steps = (argc >= 4) ? std::atoi(argv[3]) : 20;

    // Load model
    auto* model = mjmlx_load_model(argv[1]);
    if (!model) {
        std::cerr << "FAIL: load model" << std::endl;
        return 1;
    }
    auto info = mjmlx_model_info(model);
    std::cout << "Model: nq=" << info.nq << " nv=" << info.nv
              << " nu=" << info.nu << " nbody=" << info.nbody
              << " ngeom=" << info.ngeom << std::endl;

    // Create batched sim
    MjmlxBatchedConfig config;
    config.num_envs = num_envs;
    config.use_gpu = 1;
    config.foot_contacts_only = 0;
    config.integrator = MJMLX_INTEGRATOR_EULER;

    std::cout << "\n=== Creating batched sim with " << num_envs << " envs ===" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto* sim = mjmlx_batched_create(model, &config);
    auto t1 = std::chrono::high_resolution_clock::now();
    double create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!sim) {
        std::cerr << "FAIL: create batched sim" << std::endl;
        mjmlx_free_model(model);
        return 1;
    }
    std::cout << "  Created in " << create_ms << " ms" << std::endl;

    // Check initial state
    int n;
    auto* qpos = mjmlx_batched_get_qpos(sim, &n);
    std::cout << "  Initial qpos: " << n << " elements (" << num_envs << " envs x "
              << info.nq << " nq)" << std::endl;
    if (qpos && n >= info.nq * 2) {
        std::cout << "  env[0] qpos[0:3] = [" << qpos[0] << ", " << qpos[1]
                  << ", " << qpos[2] << "]" << std::endl;
        std::cout << "  env[1] qpos[0:3] = [" << qpos[info.nq] << ", "
                  << qpos[info.nq + 1] << ", " << qpos[info.nq + 2] << "]" << std::endl;
    }

    // Run batched steps
    std::cout << "\n=== Running " << num_steps << " batched steps ===" << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    bool had_nan = false;

    for (int i = 0; i < num_steps; i++) {
        mjmlx_batched_step(sim, nullptr);

        if (i < 5 || i % 10 == 0 || i == num_steps - 1) {
            auto* xpos = mjmlx_batched_get_xpos(sim, &n);
            auto* qvel_ptr = mjmlx_batched_get_qvel(sim, &n);

            // Check env 0 torso z
            float z0 = (xpos && n >= info.nbody * 3) ? xpos[3 + 2] : 0.0f;

            // Max velocity across all envs
            float max_vel = 0;
            int total_vel = num_envs * info.nv;
            if (qvel_ptr) {
                for (int j = 0; j < total_vel; j++) {
                    float v = std::abs(qvel_ptr[j]);
                    if (v > max_vel) max_vel = v;
                }
            }

            std::cout << "  step " << i << ": env0_z=" << z0
                      << "  max|qvel|=" << max_vel << std::endl;

            if (std::isnan(z0) || std::isinf(z0)) {
                std::cerr << "  NaN/Inf at step " << i << std::endl;
                had_nan = true;
                break;
            }
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double sps = (num_steps * num_envs) / (step_ms / 1000.0);

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "  " << num_steps << " steps x " << num_envs << " envs = "
              << num_steps * num_envs << " total steps" << std::endl;
    std::cout << "  Time: " << step_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << sps << " steps/sec" << std::endl;

    // Test reset
    std::cout << "\n=== Test: Reset env 0 ===" << std::endl;
    std::vector<int> mask(num_envs, 0);
    mask[0] = 1;
    mjmlx_batched_reset(sim, mask.data());

    auto* qpos_after = mjmlx_batched_get_qpos(sim, &n);
    if (qpos_after && n >= info.nq * 2) {
        std::cout << "  env[0] after reset: qpos[0:3] = ["
                  << qpos_after[0] << ", " << qpos_after[1]
                  << ", " << qpos_after[2] << "]" << std::endl;
        std::cout << "  env[1] unchanged:   qpos[0:3] = ["
                  << qpos_after[info.nq] << ", " << qpos_after[info.nq + 1]
                  << ", " << qpos_after[info.nq + 2] << "]" << std::endl;
    }

    // Cleanup
    mjmlx_batched_free(sim);
    mjmlx_free_model(model);

    if (had_nan) {
        std::cout << "\nWARN: Simulation diverged" << std::endl;
    }
    std::cout << "DONE: Batched simulation test." << std::endl;
    return 0;
}
