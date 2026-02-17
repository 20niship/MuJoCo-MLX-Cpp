// Phase 3.4: HFIELD collision tests.
// Tests height field collision against MuJoCo C:
//   sphere-hfield, capsule-hfield, box-hfield
// Each test verifies contact detection and simulation stability.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

// Hfield with sphere resting on it.
// Elevation varies: center is 1.0, edges are 0.9 → MuJoCo normalizes to [0,1]
// After normalization: center data=1.0 (height=z_top*1.0=0.5), edges data=0.0
// Sphere at z=0.3, radius=0.05 → should contact center prism
static const char* FLAT_HFIELD_SPHERE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <hfield name="terrain" nrow="3" ncol="3" size="2 2 0.5 0.5"
            elevation="0.9 0.9 0.9  0.9 1.0 0.9  0.9 0.9 0.9"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Bumpy hfield with sphere
static const char* BUMPY_HFIELD_SPHERE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <hfield name="terrain" nrow="5" ncol="5" size="2 2 0.5 0.5"
            elevation="0.5 0.5 0.5 0.5 0.5
                       0.5 0.75 0.75 0.75 0.5
                       0.5 0.75 1.0 0.75 0.5
                       0.5 0.75 0.75 0.75 0.5
                       0.5 0.5 0.5 0.5 0.5"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="ball" pos="0 0 0.4">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Hfield with capsule
static const char* FLAT_HFIELD_CAPSULE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <hfield name="terrain" nrow="3" ncol="3" size="2 2 0.5 0.5"
            elevation="0.9 0.9 0.9  0.9 1.0 0.9  0.9 0.9 0.9"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="cap" pos="0 0 0.35">
      <freejoint/>
      <geom type="capsule" size="0.03" fromto="0 0 -0.04 0 0 0.04" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Hfield with box
static const char* FLAT_HFIELD_BOX_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <hfield name="terrain" nrow="3" ncol="3" size="2 2 0.5 0.5"
            elevation="0.9 0.9 0.9  0.9 1.0 0.9  0.9 0.9 0.9"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="cube" pos="0 0 0.35">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_collision_hfield: Phase 3.4 HFIELD Collisions ===\n\n");

    // ── Test 1: flat hfield-sphere contact detection ──
    {
        TEST_BEGIN("flat_hfield_sphere_contact");
        auto path = write_temp_xml(FLAT_HFIELD_SPHERE_XML);
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
        CHECK(dj->ncon > 0, "MuJoCo C: sphere-hfield has contact");

        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, dh->data.nefc);
        CHECK(ncon_mlx > 0, "MLX: sphere-hfield has contact");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: flat hfield-sphere qacc comparison ──
    {
        TEST_BEGIN("flat_hfield_sphere_qacc");
        auto path = write_temp_xml(FLAT_HFIELD_SPHERE_XML);
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
        // GJK/EPA prism contacts differ from MuJoCo C's MPR in exact position/normal
        CHECK(max_diff < 200.0f, "sphere-hfield qacc reasonable");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: flat hfield-sphere 100 step stability ──
    {
        TEST_BEGIN("flat_hfield_sphere_stability_100");
        auto path = write_temp_xml(FLAT_HFIELD_SPHERE_XML);
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
        CHECK(max_diff < 1.0f, "sphere-hfield stable after 100 steps");

        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false; break;
            }
        }
        CHECK(finite, "sphere-hfield qpos finite");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: bumpy hfield-sphere stability ──
    {
        TEST_BEGIN("bumpy_hfield_sphere_stability");
        auto path = write_temp_xml(BUMPY_HFIELD_SPHERE_XML);
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
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false; break;
            }
        }
        CHECK(finite, "bumpy hfield stable after 50 steps");
        printf("    stable=%s\n", finite ? "yes" : "no");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: flat hfield-capsule stability ──
    {
        TEST_BEGIN("flat_hfield_capsule_stability");
        auto path = write_temp_xml(FLAT_HFIELD_CAPSULE_XML);
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
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false; break;
            }
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        CHECK(finite, "capsule-hfield stable after 50 steps");
        printf("    max qpos diff after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "capsule-hfield qpos close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: flat hfield-box stability ──
    {
        TEST_BEGIN("flat_hfield_box_stability");
        auto path = write_temp_xml(FLAT_HFIELD_BOX_XML);
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
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false; break;
            }
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        CHECK(finite, "box-hfield stable after 50 steps");
        printf("    max qpos diff after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "box-hfield qpos close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 7: batched pipeline with hfield geoms ──
    {
        TEST_BEGIN("hfield_batched_pipeline");
        MjmlxModel* mh = mjmlx_load_model_from_string(FLAT_HFIELD_SPHERE_XML);
        CHECK(mh != nullptr, "model loaded");

        MjmlxBatchedConfig config;
        config.num_envs = 4;
        config.use_gpu = 1;
        config.solver_iterations = 1;
        config.foot_contacts_only = 0;

        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &config);
        CHECK(sim != nullptr, "batched sim created with hfield model");

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
            CHECK(finite, "batched hfield qpos finite after 10 steps");
            mjmlx_batched_free(sim);
        }

        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
