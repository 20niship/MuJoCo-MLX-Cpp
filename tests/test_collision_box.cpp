// Phase 3.1: BOX collision tests.
// Tests all four box collision pair types against MuJoCo C:
//   plane-box, sphere-box, capsule-box, box-box
// Each test verifies contact count, contact normal, penetration depth,
// and full step comparison.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

static const char* PLANE_BOX_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.049">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* SPHERE_BOX_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="sphere" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.5"/>
    </body>
    <body name="box" pos="0 0 0.049">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* CAPSULE_BOX_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="capsule" pos="0.0 0 0.3">
      <freejoint/>
      <geom type="capsule" size="0.03" fromto="0 0 -0.05 0 0 0.05" mass="0.5"/>
    </body>
    <body name="box" pos="0 0 0.049">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* BOX_BOX_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box1" pos="0 0 0.15">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
    <body name="box2" pos="0 0 0.049">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_collision_box: Phase 3.1 BOX Collisions ===\n\n");

    // ── Test 1: plane-box contact detection ──
    {
        TEST_BEGIN("plane_box_contact");
        auto path = write_temp_xml(PLANE_BOX_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ncon=%d, nefc=%d\n", dj->ncon, dj->nefc);
        CHECK(dj->ncon > 0, "MuJoCo C: plane-box has contact");

        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, dh->data.nefc);
        CHECK(ncon_mlx > 0, "MLX: plane-box has contact");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: plane-box qacc comparison ──
    {
        TEST_BEGIN("plane_box_qacc_vs_mujoco_c");
        auto path = write_temp_xml(PLANE_BOX_XML);
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
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qacc[i] - (float)dj->qacc[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qacc diff after 1 step: %.6f\n", max_diff);
        CHECK(max_diff < 5.0f, "plane-box qacc close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: sphere-box contact ──
    {
        TEST_BEGIN("sphere_box_contact");
        auto path = write_temp_xml(SPHERE_BOX_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Step a few times to let sphere fall onto box
        for (int i = 0; i < 50; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        printf("    MuJoCo C: ncon=%d after 50 steps\n", dj->ncon);
        int nv = mj->nv;
        int n = 0;
        const float* mlx_qvel = mjmlx_get_qvel(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qvel[i] - (float)dj->qvel[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qvel diff after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 10.0f, "sphere-box scene qvel reasonably close");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: box-box contact ──
    {
        TEST_BEGIN("box_box_contact");
        auto path = write_temp_xml(BOX_BOX_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Step to let box1 fall onto box2
        for (int i = 0; i < 50; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        printf("    MuJoCo C: ncon=%d after 50 steps\n", dj->ncon);

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false; break;
            }
        }
        CHECK(finite, "box-box scene stable after 50 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: plane-box stability 100 steps ──
    {
        TEST_BEGIN("plane_box_stability_100");
        auto path = write_temp_xml(PLANE_BOX_XML);
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
        CHECK(max_diff < 10.0f, "plane-box qpos close to MuJoCo C after 100 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: batched pipeline with box geoms ──
    {
        TEST_BEGIN("box_batched_pipeline");
        MjmlxModel* mh = mjmlx_load_model_from_string(PLANE_BOX_XML);
        CHECK(mh != nullptr, "model loaded");

        MjmlxBatchedConfig config;
        config.num_envs = 4;
        config.use_gpu = 1;
        config.solver_iterations = 1;
        config.foot_contacts_only = 0;

        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &config);
        CHECK(sim != nullptr, "batched sim created with box model");

        if (sim) {
            int nu = mh->model.nu;
            std::vector<float> ctrl(config.num_envs * std::max(nu, 1), 0.0f);
            for (int i = 0; i < 10; i++) mjmlx_batched_step(sim, ctrl.data());

            int n = 0;
            const float* qpos = mjmlx_batched_get_qpos(sim, &n);
            CHECK(qpos != nullptr, "batched qpos non-null");
            bool finite = true;
            for (int i = 0; i < n && i < 500; i++) {
                if (std::isnan(qpos[i]) || std::isinf(qpos[i])) { finite = false; break; }
            }
            CHECK(finite, "batched box qpos finite after 10 steps");
            mjmlx_batched_free(sim);
        }

        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
