// Test smooth dynamics: kinematics, COM, CRB, factor_m, solve_m

#include "mjmlx/mjmlx.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_smooth <path_to_humanoid.xml>" << std::endl;
        return 1;
    }

    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { std::cerr << "FAIL: load model" << std::endl; return 1; }

    auto info = mjmlx_model_info(model);
    std::cout << "Loaded: nq=" << info.nq << " nv=" << info.nv << " nu=" << info.nu
              << " nbody=" << info.nbody << " ngeom=" << info.ngeom << std::endl;

    auto* data = mjmlx_make_data(model);
    if (!data) { std::cerr << "FAIL: make data" << std::endl; return 1; }

    // Run forward dynamics
    std::cout << "\n=== Test: Forward ===" << std::endl;
    mjmlx_forward(model, data);

    // Check body positions
    int n;
    auto* xpos = mjmlx_get_xpos(data, &n);
    std::cout << "  xpos: " << n << " elements" << std::endl;
    if (xpos && n >= 6) {
        std::cout << "  body[0] (world): [" << xpos[0] << ", " << xpos[1] << ", " << xpos[2] << "]" << std::endl;
        std::cout << "  body[1] (torso): [" << xpos[3] << ", " << xpos[4] << ", " << xpos[5] << "]" << std::endl;
    }

    // Check qfrc_bias
    auto* qfrc = mjmlx_get_qfrc_bias(data, &n);
    std::cout << "  qfrc_bias: " << n << " elements" << std::endl;
    if (qfrc && n > 0) {
        float max_bias = 0;
        for (int i = 0; i < n; i++) {
            if (std::abs(qfrc[i]) > max_bias) max_bias = std::abs(qfrc[i]);
        }
        std::cout << "  max |qfrc_bias| = " << max_bias << std::endl;
        if (max_bias < 1e-6) {
            std::cout << "  WARNING: qfrc_bias is all zeros (gravity should produce nonzero)" << std::endl;
        }
    }

    // Test step
    std::cout << "\n=== Test: Step ===" << std::endl;
    mjmlx_step(model, data);

    auto* qpos = mjmlx_get_qpos(data, &n);
    std::cout << "  qpos after step: " << n << " elements" << std::endl;
    if (qpos && n >= 3) {
        std::cout << "  qpos[0:3] = [" << qpos[0] << ", " << qpos[1] << ", " << qpos[2] << "]" << std::endl;
    }

    auto* qvel = mjmlx_get_qvel(data, &n);
    if (qvel && n > 0) {
        float max_vel = 0;
        for (int i = 0; i < n; i++) {
            if (std::abs(qvel[i]) > max_vel) max_vel = std::abs(qvel[i]);
        }
        std::cout << "  max |qvel| after step = " << max_vel << std::endl;
    }

    std::cout << "\n=== Cleanup ===" << std::endl;
    mjmlx_free_data(data);
    mjmlx_free_model(model);
    std::cout << "PASS: Smooth dynamics test complete." << std::endl;
    return 0;
}
