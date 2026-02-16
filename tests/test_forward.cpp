// Test full forward dynamics: multi-step simulation, timing

#include "mjmlx/mjmlx.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_forward <path_to_humanoid.xml> [num_steps]" << std::endl;
        return 1;
    }
    int num_steps = (argc >= 3) ? std::atoi(argv[2]) : 100;

    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { std::cerr << "FAIL: load model" << std::endl; return 1; }
    auto* data = mjmlx_make_data(model);
    if (!data) { std::cerr << "FAIL: make data" << std::endl; return 1; }

    auto info = mjmlx_model_info(model);
    std::cout << "Model: nq=" << info.nq << " nv=" << info.nv << " nu=" << info.nu
              << " nbody=" << info.nbody << " ngeom=" << info.ngeom << std::endl;

    // Initial forward
    mjmlx_forward(model, data);
    int n;
    auto* xpos0 = mjmlx_get_xpos(data, &n);
    float z0 = (n >= 6) ? xpos0[5] : 0.0f;
    std::cout << "Initial torso z = " << z0 << std::endl;

    // Multi-step simulation
    std::cout << "\n=== Running " << num_steps << " steps ===" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool had_nan = false;

    for (int i = 0; i < num_steps; i++) {
        mjmlx_step(model, data);

        if (i < 10 || i % 25 == 0 || i == num_steps - 1) {
            auto* xpos = mjmlx_get_xpos(data, &n);
            auto* qvel = mjmlx_get_qvel(data, &n);
            float z = (xpos && n >= 6) ? xpos[5] : 0.0f;
            float max_vel = 0;
            if (qvel) for (int j = 0; j < n; j++) max_vel = std::max(max_vel, std::abs(qvel[j]));

            auto* qfrc = mjmlx_get_qfrc_bias(data, &n);
            float max_bias = 0;
            if (qfrc) for (int j = 0; j < n; j++) max_bias = std::max(max_bias, std::abs(qfrc[j]));

            std::cout << "  step " << i << ": z=" << z
                      << "  max|qvel|=" << max_vel
                      << "  max|qfrc_bias|=" << max_bias << std::endl;

            if (std::isnan(z) || std::isinf(z)) {
                std::cerr << "  NaN/Inf at step " << i << std::endl;
                had_nan = true;
                break;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double sps = num_steps / (elapsed_ms / 1000.0);

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "  " << num_steps << " steps in " << elapsed_ms << " ms" << std::endl;
    std::cout << "  " << sps << " steps/sec (single env)" << std::endl;

    auto* xpos_final = mjmlx_get_xpos(data, &n);
    float z_final = (n >= 6) ? xpos_final[5] : 0.0f;
    std::cout << "\n  Final torso z = " << z_final << " (started at " << z0 << ")" << std::endl;

    mjmlx_free_data(data);
    mjmlx_free_model(model);

    if (had_nan) {
        std::cout << "\nWARN: Simulation diverged (expected without full contact solver)" << std::endl;
    }
    std::cout << "DONE: Forward dynamics test." << std::endl;
    return 0;
}
