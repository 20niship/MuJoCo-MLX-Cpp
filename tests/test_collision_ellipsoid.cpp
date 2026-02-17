// Phase 3.5: ELLIPSOID collision tests.
// Tests ellipsoid collision pair types against MuJoCo C:
//   plane-ellipsoid, sphere-ellipsoid, ellipsoid-ellipsoid
// Each test verifies contact detection and simulation stability.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

// Ellipsoid on plane
static const char* ELLIPSOID_PLANE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="egg" pos="0 0 0.05">
      <freejoint/>
      <geom type="ellipsoid" size="0.1 0.08 0.06" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Sphere equal-radii ellipsoid (degenerate case)
static const char* SPHERICAL_ELLIPSOID_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 0.08">
      <freejoint/>
      <geom type="ellipsoid" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Sphere hitting an ellipsoid
static const char* SPHERE_ELLIPSOID_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="egg" pos="0 0 0.08">
      <freejoint/>
      <geom type="ellipsoid" size="0.1 0.08 0.06" mass="1"/>
    </body>
    <body name="ball" pos="0.15 0 0.08">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Ellipsoid-ellipsoid
static const char* ELLIPSOID_ELLIPSOID_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="egg1" pos="0 0 0.15">
      <freejoint/>
      <geom type="ellipsoid" size="0.1 0.08 0.06" mass="1"/>
    </body>
    <body name="egg2" pos="0.12 0 0.15">
      <freejoint/>
      <geom type="ellipsoid" size="0.08 0.06 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_collision_ellipsoid: Phase 3.5 ELLIPSOID Collisions ===\n\n");

    // ── Test 1: ellipsoid-plane contact detection ──
    {
        TEST_BEGIN("ellipsoid_plane_contact");
        auto path = write_temp_xml(ELLIPSOID_PLANE_XML);
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
        CHECK(dj->ncon > 0, "MuJoCo C: ellipsoid-plane has contact");

        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, dh->data.nefc);
        CHECK(ncon_mlx > 0, "MLX: ellipsoid-plane has contact");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: ellipsoid-plane qacc comparison ──
    {
        TEST_BEGIN("ellipsoid_plane_qacc");
        auto path = write_temp_xml(ELLIPSOID_PLANE_XML);
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
        CHECK(max_diff < 50.0f, "ellipsoid-plane qacc close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: ellipsoid-plane 100 step stability ──
    {
        TEST_BEGIN("ellipsoid_plane_stability_100");
        auto path = write_temp_xml(ELLIPSOID_PLANE_XML);
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
        CHECK(max_diff < 1.0f, "ellipsoid-plane stable after 100 steps");

        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "ellipsoid-plane qpos finite");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: spherical ellipsoid = sphere behavior ──
    {
        TEST_BEGIN("spherical_ellipsoid_matches_sphere");
        auto path = write_temp_xml(SPHERICAL_ELLIPSOID_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

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
        printf("    max qpos diff (spherical ellipsoid) after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "spherical ellipsoid close to sphere behavior");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: sphere-ellipsoid stability ──
    {
        TEST_BEGIN("sphere_ellipsoid_stability");
        auto path = write_temp_xml(SPHERE_ELLIPSOID_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        for (int i = 0; i < 50; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        bool finite = true;
        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        CHECK(finite, "sphere-ellipsoid stable after 50 steps");
        printf("    max qpos diff after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "sphere-ellipsoid qpos close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: ellipsoid-ellipsoid stability ──
    {
        TEST_BEGIN("ellipsoid_ellipsoid_stability");
        auto path = write_temp_xml(ELLIPSOID_ELLIPSOID_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        for (int i = 0; i < 50; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "ellipsoid-ellipsoid stable after 50 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 7: batched pipeline with ellipsoid ──
    {
        TEST_BEGIN("ellipsoid_batched_pipeline");
        MjmlxModel* mh = mjmlx_load_model_from_string(ELLIPSOID_PLANE_XML);
        CHECK(mh != nullptr, "model loaded");

        MjmlxBatchedConfig config;
        config.num_envs = 4;
        config.use_gpu = 1;
        config.solver_iterations = 1;
        config.foot_contacts_only = 0;

        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &config);
        CHECK(sim != nullptr, "batched sim created with ellipsoid model");

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
            CHECK(finite, "batched ellipsoid qpos finite after 10 steps");
            mjmlx_batched_free(sim);
        }

        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
