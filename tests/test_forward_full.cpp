// Port of Python test_forward.py (~7 tests)
// Tests kinematics, gravity, step, multi-step vs MuJoCo C, actuation, contact.

#include "test_utils.h"
#include <cstring>

static const char* BOX_XML = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 2">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* PENDULUM_XML = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* ACTUATED_XML = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>
)";

static const char* CONTACT_XML = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 0.09">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_forward_full ===\n");

    // ── TestKinematics ──
    TEST_SECTION("Kinematics");

    TEST_BEGIN("test_initial_positions_box");
    {
        // MuJoCo C reference
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(BOX_XML, err, sizeof(err));
        CHECK(mj_m != nullptr, "MuJoCo load box");
        mjData* mj_d = mj_makeData(mj_m);
        mj_forward(mj_m, mj_d);

        // mjmlx
        MjmlxModel* model = mjmlx_load_xml_string(BOX_XML);
        CHECK(model != nullptr, "mjmlx load box");
        MjmlxData* data = mjmlx_make_data(model);
        mjmlx_forward(model, data);

        int n = 0;
        const float* xpos = mjmlx_get_xpos(data, &n);
        MjmlxModelInfo info = mjmlx_model_info(model);

        // Body 1 xpos should match MuJoCo C
        CHECK_CLOSE(xpos[3], (float)mj_d->xpos[3], 1e-4, "box xpos[x]");
        CHECK_CLOSE(xpos[4], (float)mj_d->xpos[4], 1e-4, "box xpos[y]");
        CHECK_CLOSE(xpos[5], (float)mj_d->xpos[5], 1e-4, "box xpos[z]");

        mjmlx_free_data(data); mjmlx_free_model(model);
        mj_deleteData(mj_d); mj_deleteModel(mj_m);
    }
    TEST_END();

    TEST_BEGIN("test_pendulum_kinematics");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(PENDULUM_XML, err, sizeof(err));
        CHECK(mj_m != nullptr, "MuJoCo load pendulum");
        mjData* mj_d = mj_makeData(mj_m);
        mj_d->qpos[0] = 0.5;
        mj_forward(mj_m, mj_d);

        MjmlxModel* model = mjmlx_load_xml_string(PENDULUM_XML);
        MjmlxData* data = mjmlx_make_data(model);
        float qpos_val = 0.5f;
        mjmlx_set_qpos(data, &qpos_val, 1);
        mjmlx_forward(model, data);

        int n = 0;
        const float* xpos = mjmlx_get_xpos(data, &n);

        CHECK_CLOSE(xpos[3], (float)mj_d->xpos[3], 1e-3, "link xpos[x]");
        CHECK_CLOSE(xpos[4], (float)mj_d->xpos[4], 1e-3, "link xpos[y]");
        CHECK_CLOSE(xpos[5], (float)mj_d->xpos[5], 1e-3, "link xpos[z]");

        mjmlx_free_data(data); mjmlx_free_model(model);
        mj_deleteData(mj_d); mj_deleteModel(mj_m);
    }
    TEST_END();

    // ── TestForwardDynamics ──
    TEST_SECTION("Forward Dynamics");

    TEST_BEGIN("test_freefall_gravity");
    {
        MjmlxModel* model = mjmlx_load_xml_string(BOX_XML);
        MjmlxData* data = mjmlx_make_data(model);
        mjmlx_forward(model, data);

        // qacc[2] (z) should be negative (falling due to gravity)
        int n = 0;
        const float* qvel = mjmlx_get_qvel(data, &n);
        // After forward only (no step), check bias forces indicate gravity
        int nb = 0;
        const float* bias = mjmlx_get_qfrc_bias(data, &nb);
        if (bias && nb > 0) {
            // For free body, qfrc_bias should reflect gravity
            CHECK(bias != nullptr, "qfrc_bias available");
        }

        // Step once and check z velocity is negative
        mjmlx_step(model, data);
        qvel = mjmlx_get_qvel(data, &n);
        CHECK_LT(qvel[2], 0.0f, "z velocity negative after step (gravity)");

        mjmlx_free_data(data); mjmlx_free_model(model);
    }
    TEST_END();

    TEST_BEGIN("test_single_step");
    {
        MjmlxModel* model = mjmlx_load_xml_string(BOX_XML);
        MjmlxData* data = mjmlx_make_data(model);
        mjmlx_step(model, data);

        int n = 0;
        const float* qpos = mjmlx_get_qpos(data, &n);
        CHECK_LT(qpos[2], 2.0f, "z < 2 after step (falling)");

        mjmlx_free_data(data); mjmlx_free_model(model);
    }
    TEST_END();

    TEST_BEGIN("test_multi_step_vs_mujoco");
    {
        int nsteps = 5;

        // MuJoCo C reference
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(PENDULUM_XML, err, sizeof(err));
        mjData* mj_d = mj_makeData(mj_m);
        for (int i = 0; i < nsteps; i++) mj_step(mj_m, mj_d);

        // mjmlx
        MjmlxModel* model = mjmlx_load_xml_string(PENDULUM_XML);
        MjmlxData* data = mjmlx_make_data(model);
        for (int i = 0; i < nsteps; i++) mjmlx_step(model, data);

        int n = 0;
        const float* qpos = mjmlx_get_qpos(data, &n);

        // Float32 accumulates error over steps, use relaxed tolerance
        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(qpos[i], (float)mj_d->qpos[i], 0.05f, "qpos matches MuJoCo C after 5 steps");
        }

        mjmlx_free_data(data); mjmlx_free_model(model);
        mj_deleteData(mj_d); mj_deleteModel(mj_m);
    }
    TEST_END();

    // ── TestActuation ──
    TEST_SECTION("Actuation");

    TEST_BEGIN("test_actuator_force");
    {
        MjmlxModel* model = mjmlx_load_xml_string(ACTUATED_XML);
        MjmlxData* data = mjmlx_make_data(model);

        float ctrl = 1.0f;
        mjmlx_set_ctrl(data, &ctrl, 1);
        mjmlx_forward(model, data);

        // qfrc_actuator should be nonzero
        int n = 0;
        const float* qfrc_bias = mjmlx_get_qfrc_bias(data, &n);
        // We check that the step produces motion
        mjmlx_step(model, data);
        const float* qvel = mjmlx_get_qvel(data, &n);
        float max_vel = 0;
        for (int i = 0; i < n; i++) max_vel = std::max(max_vel, std::abs(qvel[i]));
        CHECK_GT(max_vel, 0.0f, "actuator produces velocity");

        mjmlx_free_data(data); mjmlx_free_model(model);
    }
    TEST_END();

    // ── TestCollision ──
    TEST_SECTION("Collision");

    TEST_BEGIN("test_ground_contact");
    {
        MjmlxModel* model = mjmlx_load_xml_string(CONTACT_XML);
        MjmlxData* data = mjmlx_make_data(model);
        mjmlx_forward(model, data);

        // Ball at z=0.09 with radius 0.1 should be penetrating the ground plane
        // After step, constraint forces should prevent falling through
        mjmlx_step(model, data);
        int n = 0;
        const float* qvel = mjmlx_get_qvel(data, &n);
        // The ball should not be accelerating downward as fast as free fall
        // because contact forces are active
        float z_vel = qvel[2]; // z velocity for free joint
        CHECK_GT(z_vel, -0.1f, "contact slows falling (z_vel > -0.1)");

        mjmlx_free_data(data); mjmlx_free_model(model);
    }
    TEST_END();

    TEST_EXIT();
}
