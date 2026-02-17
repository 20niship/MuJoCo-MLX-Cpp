// Phase 2.1: Pyramidal friction (condim=3) tests.
// Validates that condim=3 contacts generate 4 pyramidal constraint rows per contact
// (instead of 1 frictionless row), and that the resulting physics matches MuJoCo C.
//
// Pyramidal approximation: each tangent direction yields 2 edge rows:
//   J_row = J_normal ± μ * J_tangent_k
// For condim=3: 4 rows total per contact (2 tangent directions × 2 edges).

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

static const char* FRICTION_SLIDE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="0.8 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.049">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1" condim="3" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* FRICTION_LATERAL_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="1.0 0.005 0.0001"/>
    <body name="box" pos="0 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1" condim="3" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="box_jntx" gear="1 0 0 0 0 0" name="push_x"/>
  </actuator>
</mujoco>
)";

static const char* FRICTION_TWO_BALL_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="0.5 0.005 0.0001"/>
    <body name="ball1" pos="-0.3 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1" condim="3" friction="0.5 0.005 0.0001"/>
    </body>
    <body name="ball2" pos="0.3 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1" condim="3" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_friction_condim3: Phase 2.1 Pyramidal Friction ===\n\n");

    // ── Test 1: Verify nefc = 4 * ncon for condim=3 ──
    {
        TEST_BEGIN("friction_condim3_row_count");
        auto path = write_temp_xml(FRICTION_SLIDE_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Forward (not step) to get constraint data from initial penetrating position
        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ncon=%d, nefc=%d\n", dj->ncon, dj->nefc);
        CHECK(dj->ncon > 0, "MuJoCo C: should have contact (ball penetrating plane)");

        // For condim=3 pyramidal: 4 rows per contact
        if (dj->ncon > 0) {
            int expected_nefc = 4 * dj->ncon;
            printf("    MuJoCo C: expected nefc=%d (4*ncon), actual=%d\n", expected_nefc, dj->nefc);
            CHECK(dj->nefc >= expected_nefc, "MuJoCo C: nefc >= 4*ncon for condim=3 pyramidal");
        }

        // Check MLX nefc
        int nefc_mlx = dh->data.nefc;
        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, nefc_mlx);

        if (ncon_mlx > 0) {
            int expected_nefc = 4 * ncon_mlx;
            printf("    MLX: expected nefc=%d (4*ncon), actual=%d\n", expected_nefc, nefc_mlx);
            CHECK(nefc_mlx >= expected_nefc, "MLX: nefc >= 4*ncon for condim=3 pyramidal");
        }

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: qacc matches MuJoCo C after 1 step ──
    {
        TEST_BEGIN("friction_condim3_qacc_vs_mujoco_c");
        auto path = write_temp_xml(FRICTION_SLIDE_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_step(mj, dj);
        mjmlx_step(mh, dh);

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qacc = mjmlx_get_qacc(dh, &n);
        CHECK(n == nv, "nv matches");

        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qacc[i] - (float)dj->qacc[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    nv=%d, max qacc diff after 1 step: %.6f\n", nv, max_diff);
        CHECK(max_diff < 0.1f, "qacc close to MuJoCo C after 1 step");

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: qfrc_constraint matches MuJoCo C after forward ──
    {
        TEST_BEGIN("friction_condim3_qfrc_constraint_vs_mujoco_c");
        auto path = write_temp_xml(FRICTION_SLIDE_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Forward only (don't step, to compare constraint forces before integration)
        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qfrc = mjmlx_get_qfrc_constraint(dh, &n);
        CHECK(n == nv, "nv matches");

        printf("    qfrc_constraint comparison (nv=%d):\n", nv);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qfrc[i] - (float)dj->qfrc_constraint[i]);
            if (diff > max_diff) max_diff = diff;
            if (std::abs((float)dj->qfrc_constraint[i]) > 0.01f || std::abs(mlx_qfrc[i]) > 0.01f) {
                printf("      [%d] MLX=%.4f, MjC=%.4f, diff=%.6f\n",
                       i, mlx_qfrc[i], (float)dj->qfrc_constraint[i], diff);
            }
        }
        printf("    max qfrc_constraint diff: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "qfrc_constraint close to MuJoCo C");

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: Ball stays on plane (friction prevents sliding) ──
    {
        TEST_BEGIN("friction_condim3_ball_stays_on_plane");
        MjmlxModel* mh = mjmlx_load_model_from_string(FRICTION_CONDIM3_XML);
        MjmlxData* dh = mjmlx_make_data(mh);

        // Step 50 times, ball should settle on plane
        for (int i = 0; i < 50; i++) mjmlx_step(mh, dh);

        int n = 0;
        const float* qpos = mjmlx_get_qpos(dh, &n);
        float z = qpos[2]; // z position of free body
        printf("    Ball z after 50 steps: %.4f (should be ~0.1 = radius)\n", z);

        // Ball should be resting near ground (z ~ radius)
        CHECK(z > 0.05f && z < 0.3f, "Ball resting on plane");

        // Check lateral velocity is near zero (friction holds)
        const float* qvel = mjmlx_get_qvel(dh, &n);
        float vx = qvel[0], vy = qvel[1];
        printf("    Lateral vel: vx=%.6f, vy=%.6f\n", vx, vy);
        CHECK(std::abs(vx) < 0.1f && std::abs(vy) < 0.1f, "Lateral velocity near zero");

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: Multiple contacts with friction ──
    {
        TEST_BEGIN("friction_condim3_two_balls");
        auto path = write_temp_xml(FRICTION_TWO_BALL_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Step 10 times
        for (int i = 0; i < 10; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qvel = mjmlx_get_qvel(dh, &n);

        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qvel[i] - (float)dj->qvel[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    nv=%d, max qvel diff after 10 steps: %.6f\n", nv, max_diff);
        CHECK(max_diff < 1.0f, "Two-ball qvel close to MuJoCo C");

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: Stability after 100 steps ──
    {
        TEST_BEGIN("friction_condim3_stability_100_steps");
        auto path = write_temp_xml(FRICTION_SLIDE_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        for (int i = 0; i < 100; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false;
                break;
            }
        }
        CHECK(finite, "qpos finite after 100 steps with friction");

        float max_qpos_diff = 0.0f;
        int nq = mj->nq;
        for (int i = 0; i < nq; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_qpos_diff) max_qpos_diff = diff;
        }
        printf("    max qpos diff after 100 steps: %.6f\n", max_qpos_diff);
        CHECK(max_qpos_diff < 5.0f, "qpos reasonably close to MuJoCo C after 100 steps");

        mj_deleteData(dj);
        mj_deleteModel(mj);
        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 7: Batched pipeline with friction ──
    {
        TEST_BEGIN("friction_condim3_batched_pipeline");
        MjmlxModel* mh = mjmlx_load_model_from_string(FRICTION_CONDIM3_XML);
        CHECK(mh != nullptr, "model loaded");

        MjmlxBatchedConfig config;
        config.num_envs = 4;
        config.use_gpu = 1;
        config.solver_iterations = 1;
        config.foot_contacts_only = 0;

        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &config);
        CHECK(sim != nullptr, "batched sim created with friction model");

        if (sim) {
            int nu = mh->model.nu;
            std::vector<float> ctrl(config.num_envs * std::max(nu, 1), 0.0f);

            for (int i = 0; i < 10; i++) {
                mjmlx_batched_step(sim, ctrl.data());
            }

            int n = 0;
            const float* qpos = mjmlx_batched_get_qpos(sim, &n);
            CHECK(qpos != nullptr, "batched qpos non-null");

            bool finite = true;
            for (int i = 0; i < n && i < 500; i++) {
                if (std::isnan(qpos[i]) || std::isinf(qpos[i])) {
                    finite = false;
                    break;
                }
            }
            CHECK(finite, "batched qpos finite after 10 steps with friction");

            mjmlx_batched_free(sim);
        }

        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
