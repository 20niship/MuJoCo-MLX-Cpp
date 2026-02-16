// Port of Python test_io.py (~9 tests)
// Tests model loading, data creation, shape validation, MuJoCo C comparison.

#include "test_utils.h"

static const char* SIMPLE_XML = R"(
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="box" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="hinge_body" pos="1 0 0.5">
      <joint type="hinge" axis="0 0 1" limited="true" range="-1.57 1.57"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_io_full ===\n");

    // Load MuJoCo C model for reference
    char error[1024] = {0};
    mjModel* mj_m = mj_load_xml_string(SIMPLE_XML, error, sizeof(error));
    if (!mj_m) { fprintf(stderr, "MuJoCo load failed: %s\n", error); return 1; }
    mjData* mj_d = mj_makeData(mj_m);

    // Load mjmlx model
    MjmlxModel* model = mjmlx_load_xml_string(SIMPLE_XML);
    if (!model) { fprintf(stderr, "mjmlx load failed\n"); return 1; }
    MjmlxModelInfo info = mjmlx_model_info(model);

    // ── TestPutModel ──
    TEST_SECTION("PutModel");

    TEST_BEGIN("test_counts");
    {
        CHECK(info.nq == mj_m->nq, "nq matches MuJoCo C");
        CHECK(info.nv == mj_m->nv, "nv matches MuJoCo C");
        CHECK(info.nbody == mj_m->nbody, "nbody matches MuJoCo C");
        CHECK(info.njnt == mj_m->njnt, "njnt matches MuJoCo C");
        CHECK(info.ngeom == mj_m->ngeom, "ngeom matches MuJoCo C");
    }
    TEST_END();

    TEST_BEGIN("test_nq_nv_positive");
    {
        CHECK_GT(info.nq, 0, "nq > 0");
        CHECK_GT(info.nv, 0, "nv > 0");
    }
    TEST_END();

    // ── TestMakeData ──
    TEST_SECTION("MakeData");

    MjmlxData* data = mjmlx_make_data(model);
    if (!data) { fprintf(stderr, "make_data failed\n"); return 1; }

    TEST_BEGIN("test_qpos_shape");
    {
        int n = 0;
        const float* qpos = mjmlx_get_qpos(data, &n);
        CHECK(n == info.nq, "qpos length == nq");
        CHECK(qpos != nullptr, "qpos not null");
    }
    TEST_END();

    TEST_BEGIN("test_qvel_shape");
    {
        int n = 0;
        const float* qvel = mjmlx_get_qvel(data, &n);
        CHECK(n == info.nv, "qvel length == nv");
        CHECK(qvel != nullptr, "qvel not null");
    }
    TEST_END();

    TEST_BEGIN("test_xpos_shape");
    {
        int n = 0;
        const float* xpos = mjmlx_get_xpos(data, &n);
        CHECK(n == info.nbody * 3, "xpos length == nbody*3");
        CHECK(xpos != nullptr, "xpos not null");
    }
    TEST_END();

    TEST_BEGIN("test_qpos_init");
    {
        int n = 0;
        const float* qpos = mjmlx_get_qpos(data, &n);
        // Should match MuJoCo's qpos0
        CHECK_ARRAY_CLOSE(qpos, mj_m->qpos0, info.nq, 1e-6, "qpos == qpos0");
    }
    TEST_END();

    TEST_BEGIN("test_zero_vel");
    {
        int n = 0;
        const float* qvel = mjmlx_get_qvel(data, &n);
        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(qvel[i], 0.0f, 1e-6, "qvel starts at zero");
        }
    }
    TEST_END();

    // ── TestSnapshot (step then check) ──
    TEST_SECTION("Snapshot");

    TEST_BEGIN("test_step_updates_state");
    {
        mjmlx_step(model, data);
        int n = 0;
        const float* qpos = mjmlx_get_qpos(data, &n);
        CHECK(n == info.nq, "qpos shape after step");
        // After step, time should have advanced (qpos should differ from initial)
        // For free body, z should have changed due to gravity
        bool changed = false;
        for (int i = 0; i < n; i++) {
            if (std::abs(qpos[i] - mj_m->qpos0[i]) > 1e-8) { changed = true; break; }
        }
        CHECK(changed, "qpos changed after step (gravity)");
    }
    TEST_END();

    TEST_BEGIN("test_forward_produces_xpos");
    {
        // Reset and forward
        MjmlxData* d2 = mjmlx_make_data(model);
        mjmlx_forward(model, d2);
        int n = 0;
        const float* xpos = mjmlx_get_xpos(d2, &n);
        CHECK(n == info.nbody * 3, "xpos available after forward");
        // Body 1 (box) should be at z=1
        CHECK_CLOSE(xpos[3 + 2], 1.0f, 1e-3, "box at z=1");
        mjmlx_free_data(d2);
    }
    TEST_END();

    // Cleanup
    mjmlx_free_data(data);
    mjmlx_free_model(model);
    mj_deleteData(mj_d);
    mj_deleteModel(mj_m);

    TEST_EXIT();
}
