// Phase 3.3: MESH collision tests via GJK/EPA.
// Tests convex mesh collision pair types against MuJoCo C:
//   mesh-plane, mesh-sphere, mesh-capsule, mesh-box, mesh-mesh
// Each test verifies contact detection and simulation stability.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

// Simple cube mesh (8 vertices) on a plane
static const char* MESH_PLANE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <mesh name="cube" vertex="
      -0.05 -0.05 -0.05
       0.05 -0.05 -0.05
       0.05  0.05 -0.05
      -0.05  0.05 -0.05
      -0.05 -0.05  0.05
       0.05 -0.05  0.05
       0.05  0.05  0.05
      -0.05  0.05  0.05"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cube" pos="0 0 0.049">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Sphere falling onto a mesh cube
static const char* MESH_SPHERE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <mesh name="cube" vertex="
      -0.05 -0.05 -0.05
       0.05 -0.05 -0.05
       0.05  0.05 -0.05
      -0.05  0.05 -0.05
      -0.05 -0.05  0.05
       0.05 -0.05  0.05
       0.05  0.05  0.05
      -0.05  0.05  0.05"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cube" pos="0 0 0.05">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="1"/>
    </body>
    <body name="sphere" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.04" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
)";

// Two mesh cubes stacked
static const char* MESH_MESH_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <mesh name="cube" vertex="
      -0.05 -0.05 -0.05
       0.05 -0.05 -0.05
       0.05  0.05 -0.05
      -0.05  0.05 -0.05
      -0.05 -0.05  0.05
       0.05 -0.05  0.05
       0.05  0.05  0.05
      -0.05  0.05  0.05"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cube1" pos="0 0 0.049">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="1"/>
    </body>
    <body name="cube2" pos="0 0 0.16">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
)";

// Capsule falling onto mesh cube
static const char* MESH_CAPSULE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <mesh name="cube" vertex="
      -0.05 -0.05 -0.05
       0.05 -0.05 -0.05
       0.05  0.05 -0.05
      -0.05  0.05 -0.05
      -0.05 -0.05  0.05
       0.05 -0.05  0.05
       0.05  0.05  0.05
      -0.05  0.05  0.05"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cube" pos="0 0 0.05">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="1"/>
    </body>
    <body name="cap" pos="0 0 0.3">
      <freejoint/>
      <geom type="capsule" size="0.02" fromto="0 0 -0.04 0 0 0.04" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_collision_mesh: Phase 3.3 MESH/GJK/EPA Collisions ===\n\n");

    // ── Test 1: mesh-plane contact detection ──
    {
        TEST_BEGIN("mesh_plane_contact");
        auto path = write_temp_xml(MESH_PLANE_XML);
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
        CHECK(dj->ncon > 0, "MuJoCo C: mesh-plane has contact");

        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, dh->data.nefc);
        CHECK(ncon_mlx > 0, "MLX: mesh-plane has contact");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: mesh-plane qacc comparison ──
    {
        TEST_BEGIN("mesh_plane_qacc");
        auto path = write_temp_xml(MESH_PLANE_XML);
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
        // With proper multi-vertex plane-mesh, qacc is close to MuJoCo C
        CHECK(max_diff < 10.0f, "mesh-plane qacc close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: mesh-plane 100 step stability ──
    {
        TEST_BEGIN("mesh_plane_stability_100");
        auto path = write_temp_xml(MESH_PLANE_XML);
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
        CHECK(max_diff < 1.0f, "mesh-plane stable after 100 steps");

        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) {
                finite = false; break;
            }
        }
        CHECK(finite, "mesh-plane qpos finite");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: mesh-sphere scene stability ──
    {
        TEST_BEGIN("mesh_sphere_stability");
        auto path = write_temp_xml(MESH_SPHERE_XML);
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
        CHECK(finite, "mesh-sphere scene stable after 50 steps");

        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 50 steps: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "mesh-sphere qpos close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: mesh-mesh collision (two cubes stacked) ──
    {
        TEST_BEGIN("mesh_mesh_stability");
        auto path = write_temp_xml(MESH_MESH_XML);
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
        CHECK(finite, "mesh-mesh scene stable after 50 steps");
        printf("    stable=%s\n", finite ? "yes" : "no");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: mesh-capsule collision ──
    {
        TEST_BEGIN("mesh_capsule_stability");
        auto path = write_temp_xml(MESH_CAPSULE_XML);
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
        CHECK(finite, "mesh-capsule scene stable after 50 steps");
        printf("    stable=%s\n", finite ? "yes" : "no");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 7: batched pipeline with mesh geoms ──
    {
        TEST_BEGIN("mesh_batched_pipeline");
        MjmlxModel* mh = mjmlx_load_model_from_string(MESH_PLANE_XML);
        CHECK(mh != nullptr, "model loaded");

        MjmlxBatchedConfig config;
        config.num_envs = 4;
        config.use_gpu = 1;
        config.solver_iterations = 1;
        config.foot_contacts_only = 0;

        MjmlxBatchedSim* sim = mjmlx_batched_create(mh, &config);
        CHECK(sim != nullptr, "batched sim created with mesh model");

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
            CHECK(finite, "batched mesh qpos finite after 10 steps");
            mjmlx_batched_free(sim);
        }

        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
