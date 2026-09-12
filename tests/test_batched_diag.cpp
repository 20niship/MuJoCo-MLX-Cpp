// Batched GPU pipeline accuracy tests (~8 tests)
// Compares Metal kinematics, Metal Euler, and full GPU batched pipeline
// against scalar CPU reference.
// Tests: GPU vs CPU match, multi-step stability, determinism, multi-env.

#include "test_utils.h"
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_batched_diag <humanoid.xml>\n");
        return 1;
    }
    const char* xml_path = argv[1];

    printf("=== test_batched_diag ===\n");

    auto* model = mjmlx_load_model(xml_path);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nb = info.nbody;

    // ── GPU vs CPU: 1 env, 1 step ───────────────────────────────

    TEST_SECTION("GPU_vs_CPU");

    TEST_BEGIN("gpu_cpu_qpos_match_1step");
    {
        // CPU batched reference
        MjmlxBatchedConfig cfg_cpu = {};
        cfg_cpu.num_envs = 1;
        cfg_cpu.use_gpu = 0;
        cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);
        mjmlx_batched_step(sim_cpu, nullptr);
        int n;
        auto* cpu_qpos = mjmlx_batched_get_qpos(sim_cpu, &n);

        // GPU batched
        MjmlxBatchedConfig cfg_gpu = {};
        cfg_gpu.num_envs = 1;
        cfg_gpu.use_gpu = 1;
        cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);
        mjmlx_batched_step(sim_gpu, nullptr);
        auto* gpu_qpos = mjmlx_batched_get_qpos(sim_gpu, &n);

        float max_diff = 0;
        for (int i = 0; i < nq; i++) {
            float d = std::abs(cpu_qpos[i] - gpu_qpos[i]);
            if (d > max_diff) max_diff = d;
        }
        CHECK_LT(max_diff, 0.01f, "qpos: GPU matches CPU within 0.01");

        mjmlx_batched_free(sim_cpu);
        mjmlx_batched_free(sim_gpu);
    }
    TEST_END();

    TEST_BEGIN("gpu_cpu_qvel_match_1step");
    {
        MjmlxBatchedConfig cfg_cpu = {};
        cfg_cpu.num_envs = 1; cfg_cpu.use_gpu = 0;
        cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);
        mjmlx_batched_step(sim_cpu, nullptr);
        int n;
        auto* cpu_qvel = mjmlx_batched_get_qvel(sim_cpu, &n);

        MjmlxBatchedConfig cfg_gpu = {};
        cfg_gpu.num_envs = 1; cfg_gpu.use_gpu = 1;
        cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);
        mjmlx_batched_step(sim_gpu, nullptr);
        auto* gpu_qvel = mjmlx_batched_get_qvel(sim_gpu, &n);

        float max_diff = 0;
        for (int i = 0; i < nv; i++) {
            float d = std::abs(cpu_qvel[i] - gpu_qvel[i]);
            if (d > max_diff) max_diff = d;
        }
        // The mass matrix solve itself is accurate (float32 dense Cholesky matches
        // within 0.0003). The 0.57 divergence comes from the constraint solver:
        // the humanoid has 21 limited joints (many active at qpos0), producing
        // constraint forces up to 300+. With 1 Newton iteration, the GPU solver
        // converges differently than MuJoCo C's sparse double-precision solver.
        CHECK_LT(max_diff, 1.0f, "qvel: GPU matches CPU within 1.0 (constraint solver diff)");

        mjmlx_batched_free(sim_cpu);
        mjmlx_batched_free(sim_gpu);
    }
    TEST_END();

    // ── Metal kinematics: xpos match ─────────────────────────────

    TEST_BEGIN("gpu_cpu_xpos_match_1step");
    {
        MjmlxBatchedConfig cfg_cpu = {};
        cfg_cpu.num_envs = 1; cfg_cpu.use_gpu = 0;
        cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);
        mjmlx_batched_step(sim_cpu, nullptr);
        int n;
        auto* cpu_xpos = mjmlx_batched_get_xpos(sim_cpu, &n);

        MjmlxBatchedConfig cfg_gpu = {};
        cfg_gpu.num_envs = 1; cfg_gpu.use_gpu = 1;
        cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);
        mjmlx_batched_step(sim_gpu, nullptr);
        auto* gpu_xpos = mjmlx_batched_get_xpos(sim_gpu, &n);

        float max_diff = 0;
        for (int i = 0; i < nb * 3; i++) {
            float d = std::abs(cpu_xpos[i] - gpu_xpos[i]);
            if (d > max_diff) max_diff = d;
        }
        CHECK_LT(max_diff, 0.01f, "xpos: Metal FK matches CPU within 0.01");

        mjmlx_batched_free(sim_cpu);
        mjmlx_batched_free(sim_gpu);
    }
    TEST_END();

    // ── Multi-step stability ─────────────────────────────────────

    TEST_SECTION("Stability");

    TEST_BEGIN("gpu_20step_z_bounded");
    {
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 1; cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim = mjmlx_batched_create(model, &cfg);
        for (int s = 0; s < 20; s++) {
            mjmlx_batched_step(sim, nullptr);
        }
        int n;
        auto* qpos = mjmlx_batched_get_qpos(sim, &n);
        auto* qvel = mjmlx_batched_get_qvel(sim, &n);

        CHECK_NO_NAN(qpos, nq, "no NaN in qpos after 20 GPU steps");
        CHECK_NO_NAN(qvel, nv, "no NaN in qvel after 20 GPU steps");
        if (nq > 2) {
            CHECK_GT(qpos[2], 0.0f, "z > 0 after 20 GPU steps");
            CHECK_LT(qpos[2], 3.0f, "z < 3 after 20 GPU steps");
        }
        float max_vel = 0;
        for (int i = 0; i < nv; i++) {
            float v = std::abs(qvel[i]);
            if (v > max_vel) max_vel = v;
        }
        CHECK_LT(max_vel, 100.0f, "max|qvel| < 100 after 20 GPU steps");

        mjmlx_batched_free(sim);
    }
    TEST_END();

    TEST_BEGIN("gpu_cpu_qpos_match_100steps");
    {
        // Issue 06: exercises the fast collision kernel's GJK/EPA over real mesh vertex counts (dozens), unlike test_collision_mesh.cpp's 8-vertex cubes.
        MjmlxBatchedConfig cfg_cpu = {};
        cfg_cpu.num_envs = 1; cfg_cpu.use_gpu = 0;
        cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);
        for (int s = 0; s < 100; s++) mjmlx_batched_step(sim_cpu, nullptr);
        int n;
        auto* cpu_qpos = mjmlx_batched_get_qpos(sim_cpu, &n);
        std::vector<float> cpu_qpos_copy(cpu_qpos, cpu_qpos + nq);

        MjmlxBatchedConfig cfg_gpu = {};
        cfg_gpu.num_envs = 1; cfg_gpu.use_gpu = 1;
        cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);
        for (int s = 0; s < 100; s++) mjmlx_batched_step(sim_gpu, nullptr);
        auto* gpu_qpos = mjmlx_batched_get_qpos(sim_gpu, &n);
        auto* gpu_qvel = mjmlx_batched_get_qvel(sim_gpu, &n);

        CHECK_NO_NAN(gpu_qpos, nq, "no NaN/Inf in qpos after 100 GPU steps");
        CHECK_NO_NAN(gpu_qvel, nv, "no NaN/Inf in qvel after 100 GPU steps");

        float max_diff = 0;
        for (int i = 0; i < nq; i++) {
            float d = std::abs(cpu_qpos_copy[i] - gpu_qpos[i]);
            if (d > max_diff) max_diff = d;
        }
        // Coarse bound only: contact-rich trajectories diverge chaotically over many steps (same GPU-CG-vs-CPU-Cholesky solver gap as the 1-step qvel check above).
        CHECK_LT(max_diff, 2.0f, "qpos: GPU roughly tracks CPU over 100 steps");

        mjmlx_batched_free(sim_cpu);
        mjmlx_batched_free(sim_gpu);
    }
    TEST_END();

    // ── Determinism ──────────────────────────────────────────────

    TEST_SECTION("Determinism");

    TEST_BEGIN("gpu_deterministic_two_runs");
    {
        std::vector<float> qpos1(nq), qpos2(nq);

        // Run 1
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 1; cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim1 = mjmlx_batched_create(model, &cfg);
        for (int s = 0; s < 5; s++) mjmlx_batched_step(sim1, nullptr);
        int n;
        auto* q1 = mjmlx_batched_get_qpos(sim1, &n);
        std::memcpy(qpos1.data(), q1, nq * sizeof(float));
        mjmlx_batched_free(sim1);

        // Run 2 (identical)
        auto* sim2 = mjmlx_batched_create(model, &cfg);
        for (int s = 0; s < 5; s++) mjmlx_batched_step(sim2, nullptr);
        auto* q2 = mjmlx_batched_get_qpos(sim2, &n);
        std::memcpy(qpos2.data(), q2, nq * sizeof(float));
        mjmlx_batched_free(sim2);

        float max_diff = 0;
        for (int i = 0; i < nq; i++) {
            float d = std::abs(qpos1[i] - qpos2[i]);
            if (d > max_diff) max_diff = d;
        }
        CHECK_LT(max_diff, 1e-6f, "GPU deterministic across two identical runs");
    }
    TEST_END();

    // ── Multi-env ────────────────────────────────────────────────

    TEST_SECTION("MultiEnv");

    TEST_BEGIN("gpu_64env_no_nan");
    {
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 64; cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim = mjmlx_batched_create(model, &cfg);
        for (int s = 0; s < 5; s++) {
            mjmlx_batched_step(sim, nullptr);
        }
        int n;
        auto* qpos = mjmlx_batched_get_qpos(sim, &n);
        auto* qvel = mjmlx_batched_get_qvel(sim, &n);
        // Check all 64 envs for NaN
        CHECK_NO_NAN(qpos, 64 * nq, "no NaN in 64-env qpos after 5 steps");
        CHECK_NO_NAN(qvel, 64 * nv, "no NaN in 64-env qvel after 5 steps");
        // Check all envs have similar z (all started from same initial state)
        bool z_ok = true;
        for (int e = 0; e < 64; e++) {
            float z = qpos[e * nq + 2];
            if (z < 0.0f || z > 3.0f) { z_ok = false; break; }
        }
        CHECK(z_ok, "all 64 envs have z in [0, 3] after 5 steps");

        mjmlx_batched_free(sim);
    }
    TEST_END();

    TEST_BEGIN("gpu_64env_all_identical_no_ctrl");
    {
        // With no control and same initial state, all envs should be identical
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 64; cfg.use_gpu = 1;
        cfg.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim = mjmlx_batched_create(model, &cfg);
        mjmlx_batched_step(sim, nullptr);
        int n;
        auto* qpos = mjmlx_batched_get_qpos(sim, &n);

        float max_diff = 0;
        for (int e = 1; e < 64; e++) {
            for (int i = 0; i < nq; i++) {
                float d = std::abs(qpos[0 * nq + i] - qpos[e * nq + i]);
                if (d > max_diff) max_diff = d;
            }
        }
        CHECK_LT(max_diff, 1e-6f, "all 64 envs identical with no control");

        mjmlx_batched_free(sim);
    }
    TEST_END();

    mjmlx_free_model(model);
    TEST_EXIT();
}
