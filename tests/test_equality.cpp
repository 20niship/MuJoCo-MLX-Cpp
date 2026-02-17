// Phase 4.1: Equality constraint tests.
// Tests CONNECT, WELD, and JOINT equality constraints against MuJoCo C.
// Verifies constraint row counts (ne), qacc agreement, and multi-step stability.

#include "test_utils.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

// ── CONNECT: two free bodies joined at a point ──────────────────────────────
static const char* CONNECT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="b2" pos="0.2 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="b1" body2="b2" anchor="0.1 0 0"/>
  </equality>
</mujoco>
)";

// ── WELD: two free bodies welded together ───────────────────────────────────
static const char* WELD_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="b2" pos="0.2 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="b1" body2="b2"/>
  </equality>
</mujoco>
)";

// ── JOINT: two hinge joints coupled by polynomial ───────────────────────────
static const char* JOINT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
  </equality>
</mujoco>
)";

// ── CONNECT to world body ───────────────────────────────────────────────────
static const char* CONNECT_WORLD_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="pendulum" pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="pendulum" anchor="0 0 0"/>
  </equality>
</mujoco>
)";

// ── Mixed: connect + contact (plane) ────────────────────────────────────────
static const char* CONNECT_CONTACT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="b1" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="b2" pos="0.2 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="b1" body2="b2" anchor="0.1 0 0"/>
  </equality>
</mujoco>
)";

int main() {
    printf("=== test_equality: Phase 4.1 Equality Constraints ===\n\n");

    // ── Test 1: CONNECT constraint row count ─────────────────────────────
    {
        TEST_BEGIN("connect_constraint_rows");
        auto path = write_temp_xml(CONNECT_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ne=%d, nefc=%d\n", dj->ne, dj->nefc);
        CHECK(dj->ne == 3, "MuJoCo C: CONNECT produces 3 equality rows");

        printf("    MLX: ne=%d, nefc=%d\n", dh->data.ne, dh->data.nefc);
        CHECK(dh->data.ne == 3, "MLX: CONNECT produces 3 equality rows");
        CHECK(dh->data.nefc == dj->nefc, "MLX nefc matches MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: CONNECT qacc comparison ──────────────────────────────────
    {
        TEST_BEGIN("connect_qacc");
        auto path = write_temp_xml(CONNECT_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_step(mj, dj);
        mjmlx_step(mh, dh);

        int nv = mj->nv, n = 0;
        const float* mlx_qacc = mjmlx_get_qacc(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qacc[i] - (float)dj->qacc[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qacc diff after 1 step: %.6f\n", max_diff);
        CHECK(max_diff < 10.0f, "CONNECT qacc close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: CONNECT 100-step stability ───────────────────────────────
    {
        TEST_BEGIN("connect_stability_100");
        auto path = write_temp_xml(CONNECT_XML);
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

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 100 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "CONNECT qpos stable after 100 steps");

        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "CONNECT qpos finite");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: CONNECT to world body ────────────────────────────────────
    {
        TEST_BEGIN("connect_world");
        auto path = write_temp_xml(CONNECT_WORLD_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ne=%d, nefc=%d\n", dj->ne, dj->nefc);
        CHECK(dj->ne == 3, "MuJoCo C: CONNECT-world produces 3 equality rows");
        CHECK(dh->data.ne == 3, "MLX: CONNECT-world produces 3 equality rows");

        // Run 50 steps — pendulum should swing without blowing up
        for (int i = 0; i < 50; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "CONNECT-world qpos stable after 50 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: JOINT constraint row count ───────────────────────────────
    {
        TEST_BEGIN("joint_constraint_rows");
        auto path = write_temp_xml(JOINT_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ne=%d, nefc=%d\n", dj->ne, dj->nefc);
        CHECK(dj->ne == 1, "MuJoCo C: JOINT produces 1 equality row");

        printf("    MLX: ne=%d, nefc=%d\n", dh->data.ne, dh->data.nefc);
        CHECK(dh->data.ne == 1, "MLX: JOINT produces 1 equality row");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: JOINT qacc comparison ────────────────────────────────────
    {
        TEST_BEGIN("joint_qacc");
        auto path = write_temp_xml(JOINT_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_step(mj, dj);
        mjmlx_step(mh, dh);

        int nv = mj->nv, n = 0;
        const float* mlx_qacc = mjmlx_get_qacc(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qacc[i] - (float)dj->qacc[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qacc diff after 1 step: %.6f\n", max_diff);
        CHECK(max_diff < 5.0f, "JOINT qacc close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 7: JOINT 100-step stability ─────────────────────────────────
    {
        TEST_BEGIN("joint_stability_100");
        auto path = write_temp_xml(JOINT_XML);
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

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 100 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "JOINT qpos stable after 100 steps");

        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "JOINT qpos finite");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 8: WELD constraint row count ────────────────────────────────
    {
        TEST_BEGIN("weld_constraint_rows");
        auto path = write_temp_xml(WELD_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ne=%d, nefc=%d\n", dj->ne, dj->nefc);
        CHECK(dj->ne == 6, "MuJoCo C: WELD produces 6 equality rows");

        printf("    MLX: ne=%d, nefc=%d\n", dh->data.ne, dh->data.nefc);
        CHECK(dh->data.ne == 6, "MLX: WELD produces 6 equality rows");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 9: WELD qacc comparison ─────────────────────────────────────
    {
        TEST_BEGIN("weld_qacc");
        auto path = write_temp_xml(WELD_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_step(mj, dj);
        mjmlx_step(mh, dh);

        int nv = mj->nv, n = 0;
        const float* mlx_qacc = mjmlx_get_qacc(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qacc[i] - (float)dj->qacc[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qacc diff after 1 step: %.6f\n", max_diff);
        CHECK(max_diff < 20.0f, "WELD qacc close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 10: WELD 100-step stability ─────────────────────────────────
    {
        TEST_BEGIN("weld_stability_100");
        auto path = write_temp_xml(WELD_XML);
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

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 100 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "WELD qpos stable after 100 steps");

        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "WELD qpos finite");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 11: CONNECT + contact mixed scene ──────────────────────────
    {
        TEST_BEGIN("connect_contact_mixed");
        auto path = write_temp_xml(CONNECT_CONTACT_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ne=%d, nefc=%d, ncon=%d\n", dj->ne, dj->nefc, dj->ncon);
        CHECK(dj->ne == 3, "MuJoCo C: equality rows correct in mixed scene");
        CHECK(dh->data.ne == 3, "MLX: equality rows correct in mixed scene");

        // Run 100 steps — drop onto plane while connected
        for (int i = 0; i < 100; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 100 steps: %.6f\n", max_diff);
        CHECK(max_diff < 2.0f, "CONNECT+contact qpos stable after 100 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 12: Batched pipeline with equality ──────────────────────────
    {
        TEST_BEGIN("equality_batched_pipeline");
        auto path = write_temp_xml(CONNECT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        unlink(path.c_str());

        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = 4;
        cfg.use_gpu = 1;
        cfg.solver_iterations = 3;
        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &cfg);
        CHECK(sim != nullptr, "batched sim created with equality constraints");

        for (int i = 0; i < 10; i++) {
            mjmlx_batched_step(sim, nullptr);
        }

        int nq = 0;
        const float* qpos = mjmlx_batched_get_qpos(sim, &nq);
        bool finite = true;
        for (int i = 0; i < nq; i++) {
            if (std::isnan(qpos[i]) || std::isinf(qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "batched equality qpos finite");

        mjmlx_batched_free(sim);
        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
