// Phase 1.5: High-DOF model verification and benchmark.
// Tests that the scalar and batched pipelines handle Synth-scale DOF (nv~69)
// correctly without NaN/Inf, and benchmarks throughput.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using Clock = std::chrono::high_resolution_clock;

int main() {
    printf("=== test_high_dof: Phase 1.5 High-DOF Metal Pipeline ===\n\n");

    // ── Test 1: Load HIGH_DOF_TREE_XML and verify dimensions ──
    {
        TEST_BEGIN("high_dof_model_dimensions");
        MjmlxModel* mh = mjmlx_load_model_from_string(HIGH_DOF_TREE_XML);
        CHECK(mh != nullptr, "model loaded");

        auto info = mjmlx_model_info(mh);
        printf("    nq=%d, nv=%d, nu=%d, nbody=%d, njnt=%d, ngeom=%d\n",
               info.nq, info.nv, info.nu, info.nbody, info.njnt, info.ngeom);

        // HIGH_DOF_TREE: free root (6 dof) + 5*3 spine/head + 2*(3*3+2*2) arms + 2*(3*3+1) legs = 67 dof
        CHECK(info.nv >= 60, "nv should be >= 60 for Synth-scale model");
        CHECK(info.nv <= 80, "nv should be <= 80");
        CHECK(info.nu > 0, "should have actuators");

        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: Scalar forward produces finite results ──
    {
        TEST_BEGIN("high_dof_scalar_forward_finite");
        MjmlxModel* mh = mjmlx_load_model_from_string(HIGH_DOF_TREE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);
        mjmlx_forward(mh, dh);

        int n = 0;
        const float* qpos = mjmlx_get_qpos(dh, &n);
        CHECK(qpos != nullptr, "qpos non-null");
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(qpos[i]) || std::isinf(qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "qpos finite after forward");

        const float* bias = mjmlx_get_qfrc_bias(dh, &n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(bias[i]) || std::isinf(bias[i])) { finite = false; break; }
        }
        CHECK(finite, "qfrc_bias finite after forward");

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: Scalar stepping 100 times stays finite ──
    {
        TEST_BEGIN("high_dof_scalar_step_stability");
        MjmlxModel* mh = mjmlx_load_model_from_string(HIGH_DOF_TREE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> ctrl_dist(-0.5f, 0.5f);

        bool finite = true;
        for (int step = 0; step < 100; step++) {
            // Apply random controls
            int nu = mh->model.nu;
            std::vector<float> ctrl(nu);
            for (int i = 0; i < nu; i++) ctrl[i] = ctrl_dist(rng);
            mjmlx_set_ctrl(dh, ctrl.data(), nu);

            mjmlx_step(mh, dh);

            if (step % 25 == 0) {
                int n = 0;
                const float* qpos = mjmlx_get_qpos(dh, &n);
                for (int i = 0; i < n; i++) {
                    if (std::isnan(qpos[i]) || std::isinf(qpos[i])) {
                        printf("    NaN/Inf at step %d, qpos[%d]\n", step, i);
                        finite = false;
                        break;
                    }
                }
                if (!finite) break;
            }
        }
        CHECK(finite, "qpos finite after 100 steps with random ctrl");

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: Compare scalar vs MuJoCo C for a few steps ──
    {
        TEST_BEGIN("high_dof_scalar_vs_mujoco_c");
        auto path = write_temp_xml(HIGH_DOF_TREE_XML);

        // MuJoCo C reference
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);

        // MLX
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Step 5 times with zero ctrl
        for (int i = 0; i < 5; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qvel = mjmlx_get_qvel(dh, &n);
        CHECK(n == nv, "nv matches");

        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qvel[i] - (float)dj->qvel[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    nv=%d, max qvel diff after 5 steps: %.6f\n", nv, max_diff);
        // Float32 vs float64 precision, plus different solvers
        CHECK(max_diff < 1.0f, "max qvel diff should be < 1.0");

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: Scalar throughput benchmark ──
    {
        TEST_BEGIN("high_dof_scalar_throughput");
        MjmlxModel* mh = mjmlx_load_model_from_string(HIGH_DOF_TREE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);

        // Warmup
        for (int i = 0; i < 5; i++) mjmlx_step(mh, dh);

        int num_steps = 50;
        auto t0 = Clock::now();
        for (int i = 0; i < num_steps; i++) {
            mjmlx_step(mh, dh);
        }
        auto t1 = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double steps_per_sec = num_steps / (elapsed_ms / 1000.0);

        printf("    Scalar: %d steps in %.1f ms = %.0f steps/sec (nv=%d)\n",
               num_steps, elapsed_ms, steps_per_sec, mh->model.nv);

        // Compare with MuJoCo C
        auto path = write_temp_xml(HIGH_DOF_TREE_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        unlink(path.c_str());

        for (int i = 0; i < 5; i++) mj_step(mj, dj);
        auto t2 = Clock::now();
        for (int i = 0; i < num_steps; i++) mj_step(mj, dj);
        auto t3 = Clock::now();
        double mj_elapsed_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double mj_steps_per_sec = num_steps / (mj_elapsed_ms / 1000.0);

        printf("    MuJoCo C: %d steps in %.1f ms = %.0f steps/sec\n",
               num_steps, mj_elapsed_ms, mj_steps_per_sec);
        printf("    Ratio (MLX/MuJoCo C): %.1fx\n", mj_steps_per_sec / steps_per_sec);

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: Batched pipeline with high-DOF model ──
    {
        TEST_BEGIN("high_dof_batched_pipeline");
        MjmlxModel* mh = mjmlx_load_model_from_string(HIGH_DOF_TREE_XML);
        CHECK(mh != nullptr, "model loaded");

        MjmlxBatchedConfig config;
        config.num_envs = 8;
        config.use_gpu = 1;
        config.solver_iterations = 1;
        config.foot_contacts_only = 0;

        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &config);
        CHECK(sim != nullptr, "batched sim created for high-DOF model");

        if (sim) {
            int nu = mh->model.nu;
            std::vector<float> ctrl(config.num_envs * nu, 0.0f);

            // Step a few times
            bool step_ok = true;
            for (int i = 0; i < 5; i++) {
                mjmlx_batched_step(sim, ctrl.data());
            }

            int n = 0;
            const float* qpos = mjmlx_batched_get_qpos(sim, &n);
            CHECK(qpos != nullptr, "batched qpos non-null");
            CHECK(n > 0, "batched qpos has data");

            bool finite = true;
            for (int i = 0; i < n && i < 1000; i++) {
                if (std::isnan(qpos[i]) || std::isinf(qpos[i])) {
                    finite = false;
                    break;
                }
            }
            CHECK(finite, "batched qpos finite after 5 steps");

            printf("    Batched: num_envs=%d, nv=%d, qpos elements=%d\n",
                   config.num_envs, mh->model.nv, n);

            // Quick batched throughput
            auto t0 = Clock::now();
            int batched_steps = 10;
            for (int i = 0; i < batched_steps; i++) {
                mjmlx_batched_step(sim, ctrl.data());
            }
            auto t1 = Clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double env_steps_per_sec = (batched_steps * config.num_envs) / (elapsed_ms / 1000.0);
            printf("    Batched: %d steps x %d envs in %.1f ms = %.0f env-steps/sec\n",
                   batched_steps, config.num_envs, elapsed_ms, env_steps_per_sec);

            mjmlx_batched_free(sim);
        }

        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
