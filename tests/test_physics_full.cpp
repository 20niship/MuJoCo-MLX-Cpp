// Port of Python test_physics.py (~28 tests)
// Multi-step accuracy, joint types, physics subsystems, parallel correctness, edge cases.

#include "test_utils.h"
#include <cstring>
#include <random>

static const char* PENDULUM_XML = R"(
<mujoco model="pendulum">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* FREE_BODY_XML = R"(
<mujoco model="free_body">
  <option timestep="0.002"/>
  <worldbody>
    <body name="ball" pos="0 0 2">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* SIMPLE_HUMANOID_XML = R"(
<mujoco model="test_humanoid">
  <option timestep="0.002" solver="Newton" iterations="50"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="torso" pos="0 0 1.2">
      <joint type="free"/>
      <geom type="capsule" fromto="0 0 -0.1 0 0 0.1" size="0.07" mass="8"/>
      <body name="thigh_right" pos="0 -0.1 -0.1">
        <joint name="hip_right" type="hinge" axis="0 1 0" range="-120 30"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="4"/>
        <body name="shin_right" pos="0 0 -0.4">
          <joint name="knee_right" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
        </body>
      </body>
      <body name="thigh_left" pos="0 0.1 -0.1">
        <joint name="hip_left" type="hinge" axis="0 1 0" range="-120 30"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="4"/>
        <body name="shin_left" pos="0 0 -0.4">
          <joint name="knee_left" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hip_right" ctrlrange="-100 100"/>
    <motor joint="knee_right" ctrlrange="-100 100"/>
    <motor joint="hip_left" ctrlrange="-100 100"/>
    <motor joint="knee_left" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
)";

static const char* BALL_CHAIN_XML = R"(
<mujoco model="ball_chain">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint type="ball"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      <body name="link2" pos="0 0 -0.5">
        <joint type="ball"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
        <body name="link3" pos="0 0 -0.5">
          <joint type="ball"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

static const char* SLIDE_XML = R"(
<mujoco model="slide_cart">
  <option timestep="0.002"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* SPRING_DAMPER_XML = R"(
<mujoco model="spring_damper">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint type="hinge" axis="0 1 0" stiffness="10" damping="0.5" springref="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
    </body>
    <body name="link2" pos="1 0 0">
      <joint type="hinge" axis="0 1 0" stiffness="5" damping="0.3" springref="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* ACTUATED_PENDULUM_XML = R"(
<mujoco model="actuated_pendulum">
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
<mujoco model="contact_test">
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

static const char* ZERO_GRAVITY_XML = R"(
<mujoco model="zero_gravity">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Helper: run n steps comparison between MuJoCo C and mjmlx
static void run_comparison(const char* xml, int nsteps,
                           float atol_qpos, float atol_qvel,
                           bool& ok, const char* label) {
    char err[1024];
    mjModel* mj_m = mj_load_xml_string(xml, err, sizeof(err));
    if (!mj_m) { printf("    MuJoCo load failed: %s\n", err); ok = false; return; }
    mjData* mj_d = mj_makeData(mj_m);

    MjmlxModel* model = mjmlx_load_xml_string(xml);
    if (!model) { printf("    mjmlx load failed\n"); ok = false; mj_deleteData(mj_d); mj_deleteModel(mj_m); return; }
    MjmlxData* data = mjmlx_make_data(model);

    for (int i = 0; i < nsteps; i++) {
        mj_step(mj_m, mj_d);
        mjmlx_step(model, data);
    }

    int nq = 0, nv = 0;
    const float* qpos = mjmlx_get_qpos(data, &nq);
    const float* qvel = mjmlx_get_qvel(data, &nv);

    for (int i = 0; i < nq; i++) {
        if (std::abs(qpos[i] - (float)mj_d->qpos[i]) > atol_qpos) {
            printf("    %s: qpos[%d] diff=%.4e (mlx=%.6f, mj=%.6f, tol=%.2e)\n",
                   label, i, std::abs(qpos[i] - (float)mj_d->qpos[i]),
                   qpos[i], (float)mj_d->qpos[i], atol_qpos);
            ok = false; break;
        }
    }
    for (int i = 0; i < nv; i++) {
        if (std::abs(qvel[i] - (float)mj_d->qvel[i]) > atol_qvel) {
            printf("    %s: qvel[%d] diff=%.4e (mlx=%.6f, mj=%.6f, tol=%.2e)\n",
                   label, i, std::abs(qvel[i] - (float)mj_d->qvel[i]),
                   qvel[i], (float)mj_d->qvel[i], atol_qvel);
            ok = false; break;
        }
    }

    mjmlx_free_data(data); mjmlx_free_model(model);
    mj_deleteData(mj_d); mj_deleteModel(mj_m);
}

int main(int argc, char** argv) {
    printf("=== test_physics_full ===\n");

    // Path to humanoid.xml (optional CLI arg)
    const char* humanoid_xml_path = nullptr;
    if (argc > 1) humanoid_xml_path = argv[1];

    // ── TestMultiStepAccuracy ──
    TEST_SECTION("MultiStepAccuracy");

    TEST_BEGIN("pendulum_10_steps");
    run_comparison(PENDULUM_XML, 10, 1e-3f, 1e-2f, _test_ok, "pendulum_10");
    TEST_END();

    TEST_BEGIN("pendulum_100_steps");
    run_comparison(PENDULUM_XML, 100, 5e-2f, 5e-1f, _test_ok, "pendulum_100");
    TEST_END();

    TEST_BEGIN("free_body_10_steps");
    run_comparison(FREE_BODY_XML, 10, 1e-3f, 1e-2f, _test_ok, "free_body_10");
    TEST_END();

    TEST_BEGIN("free_body_100_steps");
    run_comparison(FREE_BODY_XML, 100, 5e-2f, 5e-1f, _test_ok, "free_body_100");
    TEST_END();

    TEST_BEGIN("simple_humanoid_10_steps");
    run_comparison(SIMPLE_HUMANOID_XML, 10, 1e-3f, 5e-2f, _test_ok, "humanoid_10");
    TEST_END();

    TEST_BEGIN("simple_humanoid_50_steps");
    run_comparison(SIMPLE_HUMANOID_XML, 50, 5e-2f, 5e-1f, _test_ok, "humanoid_50");
    TEST_END();

    if (humanoid_xml_path) {
        TEST_BEGIN("humanoid_xml_10_steps_stability");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            CHECK(model != nullptr, "load humanoid.xml");
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                for (int i = 0; i < 10; i++) mjmlx_step(model, data);
                int nq = 0, nv = 0;
                const float* qpos = mjmlx_get_qpos(data, &nq);
                const float* qvel = mjmlx_get_qvel(data, &nv);
                CHECK_NO_NAN(qpos, nq, "no NaN in qpos");
                CHECK_NO_NAN(qvel, nv, "no NaN in qvel");
                CHECK_GT(qpos[2], 0.5f, "z > 0.5");
                CHECK_LT(qpos[2], 2.0f, "z < 2.0");
                float max_qvel = 0;
                for (int i = 0; i < nv; i++) max_qvel = std::max(max_qvel, std::abs(qvel[i]));
                CHECK_LT(max_qvel, 50.0f, "max|qvel| < 50");
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("humanoid_xml_50_steps_stability");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                for (int i = 0; i < 50; i++) mjmlx_step(model, data);
                int nq = 0, nv = 0;
                const float* qpos = mjmlx_get_qpos(data, &nq);
                const float* qvel = mjmlx_get_qvel(data, &nv);
                CHECK_NO_NAN(qpos, nq, "no NaN in qpos after 50 steps");
                CHECK_NO_NAN(qvel, nv, "no NaN in qvel after 50 steps");
                float max_qvel = 0;
                for (int i = 0; i < nv; i++) max_qvel = std::max(max_qvel, std::abs(qvel[i]));
                CHECK_LT(max_qvel, 100.0f, "max|qvel| < 100");
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();
    }

    // ── TestJointTypes ──
    TEST_SECTION("JointTypes");

    TEST_BEGIN("ball_joint_quaternion");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(BALL_CHAIN_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(BALL_CHAIN_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            MjmlxData* data = mjmlx_make_data(model);
            // Set all qvel to 0.5
            for (int i = 0; i < mj_m->nv; i++) mj_d->qvel[i] = 0.5;
            std::vector<float> qvel_f(mj_m->nv, 0.5f);
            mjmlx_set_qvel(data, qvel_f.data(), mj_m->nv);

            for (int i = 0; i < 10; i++) { mj_step(mj_m, mj_d); mjmlx_step(model, data); }

            int nq = 0;
            const float* qpos = mjmlx_get_qpos(data, &nq);
            // Check ball quaternions are unit norm (every 4 elements)
            for (int j = 0; j < nq; j += 4) {
                float norm = 0;
                for (int k = 0; k < 4; k++) norm += qpos[j+k] * qpos[j+k];
                norm = std::sqrt(norm);
                CHECK_CLOSE(norm, 1.0f, 1e-3, "ball quat unit norm");
            }
            // Check qpos matches CPU
            for (int i = 0; i < nq; i++) {
                if (std::abs(qpos[i] - (float)mj_d->qpos[i]) > 5e-2f) {
                    printf("    ball_joint: qpos[%d] diff=%.4e\n", i,
                           std::abs(qpos[i] - (float)mj_d->qpos[i]));
                    _test_ok = false; break;
                }
            }
            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        } else { _test_ok = false; }
    }
    TEST_END();

    TEST_BEGIN("slide_joint");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(SLIDE_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(SLIDE_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            MjmlxData* data = mjmlx_make_data(model);
            for (int i = 0; i < 10; i++) { mj_step(mj_m, mj_d); mjmlx_step(model, data); }

            int nq = 0;
            const float* qpos = mjmlx_get_qpos(data, &nq);
            for (int i = 0; i < nq; i++) {
                CHECK_CLOSE(qpos[i], (float)mj_d->qpos[i], 1e-2f, "slide qpos match CPU");
            }
            // z should go down under gravity
            CHECK_LT(qpos[2], 0.0f, "z slides down under gravity");

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("hinge_joint_displaced");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(PENDULUM_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(PENDULUM_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            MjmlxData* data = mjmlx_make_data(model);
            mj_d->qpos[0] = 1.0;
            float qp = 1.0f;
            mjmlx_set_qpos(data, &qp, 1);
            for (int i = 0; i < 30; i++) { mj_step(mj_m, mj_d); mjmlx_step(model, data); }

            int nq = 0;
            const float* qpos = mjmlx_get_qpos(data, &nq);
            CHECK_CLOSE(qpos[0], (float)mj_d->qpos[0], 1e-2f, "hinge qpos after 30 steps");

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    // ── TestPhysicsSubsystems ──
    TEST_SECTION("PhysicsSubsystems");

    TEST_BEGIN("com_pos_vs_cpu");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(SIMPLE_HUMANOID_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(SIMPLE_HUMANOID_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            mj_forward(mj_m, mj_d);

            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);

            int n = 0;
            const float* stcom = mjmlx_get_subtree_com(data, &n);
            if (stcom && n > 0) {
                for (int i = 0; i < mj_m->nbody * 3 && i < n; i++) {
                    if (std::abs(stcom[i] - (float)mj_d->subtree_com[i]) > 1e-3f) {
                        printf("    subtree_com[%d] diff=%.4e\n", i,
                               std::abs(stcom[i] - (float)mj_d->subtree_com[i]));
                        _test_ok = false; break;
                    }
                }
            }

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("collision_sphere_plane");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(CONTACT_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(CONTACT_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            mj_forward(mj_m, mj_d);

            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);

            CHECK_GT(mj_d->ncon, 0, "MuJoCo C detects contact");
            int mlx_ncon = mjmlx_get_ncon(data);
            int mlx_nefc = mjmlx_get_nefc(data);
            CHECK(mlx_ncon > 0 || mlx_nefc > 0, "MLX detects contact");

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("constraint_joint_limits");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(SIMPLE_HUMANOID_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(SIMPLE_HUMANOID_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            mj_d->qpos[7] = -2.5; // hip_right past limit
            mj_forward(mj_m, mj_d);

            MjmlxData* data = mjmlx_make_data(model);
            int nq = 0;
            const float* qpos_orig = mjmlx_get_qpos(data, &nq);
            std::vector<float> qpos_v(qpos_orig, qpos_orig + nq);
            qpos_v[7] = -2.5f;
            mjmlx_set_qpos(data, qpos_v.data(), nq);
            mjmlx_forward(model, data);

            CHECK_GT(mj_d->nefc, 0, "MuJoCo C has constraints");
            int mlx_nefc = mjmlx_get_nefc(data);
            CHECK_GT(mlx_nefc, 0, "MLX has constraints");

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("passive_spring_damper");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(SPRING_DAMPER_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(SPRING_DAMPER_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            mj_d->qpos[0] = 1.0; mj_d->qpos[1] = -1.0;
            mj_d->qvel[0] = 0.5; mj_d->qvel[1] = -0.3;
            mj_forward(mj_m, mj_d);

            MjmlxData* data = mjmlx_make_data(model);
            float qp[2] = {1.0f, -1.0f};
            float qv[2] = {0.5f, -0.3f};
            mjmlx_set_qpos(data, qp, 2);
            mjmlx_set_qvel(data, qv, 2);
            mjmlx_forward(model, data);

            int n = 0;
            const float* passive = mjmlx_get_qfrc_passive(data, &n);
            if (passive && n > 0) {
                float max_p = 0;
                for (int i = 0; i < n; i++) max_p = std::max(max_p, std::abs(passive[i]));
                CHECK_GT(max_p, 0.1f, "passive forces nonzero");
                // Compare with CPU
                for (int i = 0; i < n; i++) {
                    CHECK_CLOSE(passive[i], (float)mj_d->qfrc_passive[i], 1e-2f, "passive match CPU");
                }
            }

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("actuation_force");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(ACTUATED_PENDULUM_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(ACTUATED_PENDULUM_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            mj_d->ctrl[0] = 2.5;
            mj_forward(mj_m, mj_d);

            MjmlxData* data = mjmlx_make_data(model);
            float ctrl = 2.5f;
            mjmlx_set_ctrl(data, &ctrl, 1);
            mjmlx_forward(model, data);

            int n = 0;
            const float* qfrc_act = mjmlx_get_qfrc_actuator(data, &n);
            if (qfrc_act && n > 0) {
                for (int i = 0; i < n; i++) {
                    CHECK_CLOSE(qfrc_act[i], (float)mj_d->qfrc_actuator[i], 1e-4f, "actuator force match CPU");
                }
            }

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    // ── TestEdgeCases ──
    TEST_SECTION("EdgeCases");

    TEST_BEGIN("zero_velocity_start");
    {
        MjmlxModel* model = mjmlx_load_xml_string(FREE_BODY_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            int nv = 0;
            const float* qvel = mjmlx_get_qvel(data, &nv);
            float max_vel = 0;
            for (int i = 0; i < nv; i++) max_vel = std::max(max_vel, std::abs(qvel[i]));
            CHECK_CLOSE(max_vel, 0.0f, 1e-6, "initial vel is zero");

            for (int i = 0; i < 10; i++) mjmlx_step(model, data);
            int nq = 0;
            const float* qpos = mjmlx_get_qpos(data, &nq);
            qvel = mjmlx_get_qvel(data, &nv);
            CHECK_NO_NAN(qpos, nq, "no NaN qpos");
            CHECK_NO_NAN(qvel, nv, "no NaN qvel");
            CHECK_LT(qpos[2], 2.0f, "z < 2 after falling");

            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("large_velocity");
    {
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(FREE_BODY_XML, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(FREE_BODY_XML);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            double large_v[6] = {10, -5, 20, 1, 2, 3};
            for (int i = 0; i < 6; i++) mj_d->qvel[i] = large_v[i];

            MjmlxData* data = mjmlx_make_data(model);
            float large_vf[6] = {10, -5, 20, 1, 2, 3};
            mjmlx_set_qvel(data, large_vf, 6);

            for (int i = 0; i < 10; i++) { mj_step(mj_m, mj_d); mjmlx_step(model, data); }
            int nq = 0, nv = 0;
            const float* qpos = mjmlx_get_qpos(data, &nq);
            const float* qvel_r = mjmlx_get_qvel(data, &nv);
            CHECK_NO_NAN(qpos, nq, "no NaN qpos");
            CHECK_NO_NAN(qvel_r, nv, "no NaN qvel");
            for (int i = 0; i < nq; i++) {
                CHECK_CLOSE(qpos[i], (float)mj_d->qpos[i], 1e-2f, "large vel qpos match CPU");
            }

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("zero_gravity");
    {
        MjmlxModel* model = mjmlx_load_xml_string(ZERO_GRAVITY_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            float qvel_v[6] = {1, 0, 0, 0, 0, 0};
            mjmlx_set_qvel(data, qvel_v, 6);

            int nsteps = 100;
            for (int i = 0; i < nsteps; i++) mjmlx_step(model, data);

            int nq = 0, nv = 0;
            const float* qpos = mjmlx_get_qpos(data, &nq);
            const float* qvel_r = mjmlx_get_qvel(data, &nv);
            // Linear velocity should be approximately constant
            CHECK_CLOSE(qvel_r[0], 1.0f, 1e-3, "vx constant in zero gravity");
            // x should advance linearly: x ~= 100 * 0.002 * 1.0 = 0.2
            CHECK_CLOSE(qpos[0], 0.2f, 0.01f, "x advances linearly");

            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("no_contacts_free_floating");
    {
        const char* xml = R"(
<mujoco><option timestep="0.002"/>
  <worldbody><body name="ball" pos="0 0 10">
    <joint type="free"/><geom type="sphere" size="0.1" mass="1"/>
  </body></worldbody>
</mujoco>)";
        MjmlxModel* model = mjmlx_load_xml_string(xml);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);
            int nefc = mjmlx_get_nefc(data);
            CHECK(nefc == 0, "no constraints for free-floating ball");
            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("all_joint_limits_active");
    {
        const char* xml = R"(
<mujoco><option timestep="0.002" solver="Newton" iterations="10"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      <body name="link2" pos="0 0 -0.5">
        <joint type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>)";
        char err[1024];
        mjModel* mj_m = mj_load_xml_string(xml, err, sizeof(err));
        MjmlxModel* model = mjmlx_load_xml_string(xml);
        if (mj_m && model) {
            mjData* mj_d = mj_makeData(mj_m);
            mj_d->qpos[0] = 1.0; mj_d->qpos[1] = -1.0;
            mj_forward(mj_m, mj_d);

            MjmlxData* data = mjmlx_make_data(model);
            float qp[2] = {1.0f, -1.0f};
            mjmlx_set_qpos(data, qp, 2);
            mjmlx_forward(model, data);

            CHECK(mj_d->nefc >= 2, "CPU has >= 2 constraint rows");
            int mlx_nefc = mjmlx_get_nefc(data);
            CHECK(mlx_nefc >= 2, "MLX has >= 2 constraint rows");

            mjmlx_free_data(data); mjmlx_free_model(model);
            mj_deleteData(mj_d); mj_deleteModel(mj_m);
        }
    }
    TEST_END();

    TEST_BEGIN("spring_oscillation_stable");
    {
        MjmlxModel* model = mjmlx_load_xml_string(SPRING_DAMPER_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            float qp[2] = {1.0f, -1.0f};
            mjmlx_set_qpos(data, qp, 2);

            // Run 2000 steps, check stability
            float early_max_pos = 0, late_max_pos = 0;
            for (int i = 0; i < 2000; i++) {
                mjmlx_step(model, data);
                int nq = 0;
                const float* qpos = mjmlx_get_qpos(data, &nq);
                float maxp = 0;
                for (int j = 0; j < nq; j++) maxp = std::max(maxp, std::abs(qpos[j]));

                CHECK_NO_NAN(qpos, nq, "no NaN during oscillation");
                CHECK_LT(maxp, 20.0f, "|qpos| bounded");

                if (i >= 0 && i < 100) early_max_pos = std::max(early_max_pos, maxp);
                if (i >= 1900) late_max_pos = std::max(late_max_pos, maxp);
            }
            // Energy should decay with damping
            CHECK_LT(late_max_pos, early_max_pos, "energy decays over time");

            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    // ── TestParallelCorrectness (if humanoid path provided) ──
    if (humanoid_xml_path) {
        TEST_SECTION("ParallelCorrectness");

        TEST_BEGIN("batched_no_nan_64_envs");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxBatchedConfig cfg = {};
                cfg.num_envs = 64;
                cfg.use_gpu = 1;
                MjmlxBatchedSim* sim = mjmlx_batched_create(model, &cfg);
                if (sim) {
                    MjmlxModelInfo info = mjmlx_model_info(model);
                    std::vector<float> ctrl(64 * info.nu, 0.0f);
                    for (int s = 0; s < 10; s++) {
                        mjmlx_batched_step(sim, ctrl.data());
                    }
                    int n = 0;
                    const float* qpos = mjmlx_batched_get_qpos(sim, &n);
                    const float* qvel = mjmlx_batched_get_qvel(sim, &n);
                    CHECK_NO_NAN(qpos, n, "no NaN in batched qpos");
                    CHECK_NO_NAN(qvel, n, "no NaN in batched qvel");
                    mjmlx_batched_free(sim);
                }
                mjmlx_free_model(model);
            }
        }
        TEST_END();
    }

    TEST_EXIT();
}
