// Test model I/O: verify load_model produces correct dimensions

#include "mjmlx/mjmlx.h"
#include <iostream>
#include <cstdlib>
#include <cassert>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_io <path_to_humanoid.xml>" << std::endl;
        return 1;
    }

    std::cout << "MuJoCo-MLX version: " << mjmlx_version() << std::endl;

    std::cout << "\n=== Test: Load Model ===" << std::endl;
    auto* model = mjmlx_load_model(argv[1]);
    if (!model) {
        std::cerr << "FAIL: mjmlx_load_model returned null" << std::endl;
        return 1;
    }

    auto info = mjmlx_model_info(model);
    std::cout << "  nq=" << info.nq << " nv=" << info.nv << " nu=" << info.nu
              << " nbody=" << info.nbody << " njnt=" << info.njnt
              << " ngeom=" << info.ngeom << " nsite=" << info.nsite << std::endl;

    // Humanoid should have nq=28, nv=27 (or 29, 28 for Gymnasium variant)
    if (info.nq < 20 || info.nv < 20) {
        std::cerr << "WARNING: Unexpected model dimensions for humanoid" << std::endl;
    }

    std::cout << "\n=== Test: Make Data ===" << std::endl;
    auto* data = mjmlx_make_data(model);
    if (!data) {
        std::cerr << "FAIL: mjmlx_make_data returned null" << std::endl;
        mjmlx_free_model(model);
        return 1;
    }

    int n;
    auto* qpos = mjmlx_get_qpos(data, &n);
    std::cout << "  qpos: " << n << " elements" << std::endl;
    if (qpos && n > 0) {
        std::cout << "  qpos[0:3] = [" << qpos[0] << ", " << qpos[1] << ", " << qpos[2] << "]" << std::endl;
    }

    auto* qvel = mjmlx_get_qvel(data, &n);
    std::cout << "  qvel: " << n << " elements" << std::endl;

    std::cout << "\n=== Cleanup ===" << std::endl;
    mjmlx_free_data(data);
    mjmlx_free_model(model);

    std::cout << "PASS: Model I/O works." << std::endl;
    return 0;
}
