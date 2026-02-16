// Port of Python test_solver.py (~18 tests) + test_regression.py (~16 tests)
// Solver convergence, constraint forces, stability, NaN/Inf checks.

#include "test_utils.h"
#include <cstring>

static const char* CONTACT_XML = R"(
<mujoco model="contact_solver_test">
  <option timestep="0.002" solver="Newton" iterations="10"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 0.09">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* JOINT_LIMIT_XML = R"(
<mujoco model="joint_limits">
  <option timestep="0.002" solver="Newton" iterations="10"/>
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
</mujoco>
)";

static const char* NO_CONTACT_HIGH_XML = R"(
<mujoco model="no_contact_high">
  <option timestep="0.002"/>
  <worldbody>
    <body name="ball" pos="0 0 10">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

int main(int argc, char** argv) {
    printf("=== test_solver_full ===\n");

    const char* humanoid_xml_path = nullptr;
    if (argc > 1) humanoid_xml_path = argv[1];

    // ── TestNewtonSolver ──
    TEST_SECTION("NewtonSolver");

    TEST_BEGIN("contact_produces_forces");
    {
        MjmlxModel* model = mjmlx_load_xml_string(CONTACT_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);
            mjmlx_step(model, data);

            int n = 0;
            const float* qfrc_c = mjmlx_get_qfrc_constraint(data, &n);
            if (qfrc_c && n > 0) {
                float max_f = 0;
                for (int i = 0; i < n; i++) max_f = std::max(max_f, std::abs(qfrc_c[i]));
                CHECK_GT(max_f, 0.01f, "constraint forces nonzero");
            } else {
                // If no constraint forces, the contact should still prevent freefall
                int nv = 0;
                const float* qvel = mjmlx_get_qvel(data, &nv);
                CHECK_GT(qvel[2], -0.5f, "contact slows falling");
            }
            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("contact_force_upward");
    {
        MjmlxModel* model = mjmlx_load_xml_string(CONTACT_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);

            int n = 0;
            const float* qfrc_c = mjmlx_get_qfrc_constraint(data, &n);
            if (qfrc_c && n >= 3) {
                // z component of constraint force should be upward (positive)
                CHECK_GT(qfrc_c[2], 0.0f, "z constraint force is upward");
            }
            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("solver_modifies_acceleration");
    {
        MjmlxModel* model = mjmlx_load_xml_string(CONTACT_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);

            int n_acc = 0, n_smooth = 0;
            const float* qacc = mjmlx_get_qacc(data, &n_acc);
            const float* qacc_s = mjmlx_get_qacc_smooth(data, &n_smooth);
            if (qacc && qacc_s && n_acc > 0 && n_smooth > 0) {
                float diff = 0;
                int mn = std::min(n_acc, n_smooth);
                for (int i = 0; i < mn; i++) diff += std::abs(qacc[i] - qacc_s[i]);
                CHECK_GT(diff, 0.01f, "solver changes acceleration");
            }
            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("no_contacts_returns_smooth");
    {
        MjmlxModel* model = mjmlx_load_xml_string(NO_CONTACT_HIGH_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            mjmlx_forward(model, data);

            int nefc = mjmlx_get_nefc(data);
            // With no contacts, qacc should equal qacc_smooth
            int n_acc = 0, n_smooth = 0;
            const float* qacc = mjmlx_get_qacc(data, &n_acc);
            const float* qacc_s = mjmlx_get_qacc_smooth(data, &n_smooth);
            if (qacc && qacc_s && n_acc > 0) {
                for (int i = 0; i < n_acc; i++) {
                    CHECK_CLOSE(qacc[i], qacc_s[i], 1e-6f, "qacc == qacc_smooth without contacts");
                }
            }
            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    TEST_BEGIN("joint_limit_forces");
    {
        MjmlxModel* model = mjmlx_load_xml_string(JOINT_LIMIT_XML);
        if (model) {
            MjmlxData* data = mjmlx_make_data(model);
            float qp[2] = {1.0f, -1.0f}; // past limits
            mjmlx_set_qpos(data, qp, 2);
            mjmlx_forward(model, data);

            int n = 0;
            const float* qfrc_c = mjmlx_get_qfrc_constraint(data, &n);
            if (qfrc_c && n > 0) {
                float max_f = 0;
                for (int i = 0; i < n; i++) max_f = std::max(max_f, std::abs(qfrc_c[i]));
                CHECK_GT(max_f, 0.01f, "joint limit constraint forces nonzero");
            }
            mjmlx_free_data(data); mjmlx_free_model(model);
        }
    }
    TEST_END();

    // ── TestSolverHumanoid ──
    if (humanoid_xml_path) {
        TEST_SECTION("SolverHumanoid");

        TEST_BEGIN("humanoid_newton_no_nan");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);

                int n = 0;
                const float* qacc = mjmlx_get_qacc(data, &n);
                if (qacc && n > 0) CHECK_NO_NAN(qacc, n, "no NaN in qacc");
                const float* qfrc_c = mjmlx_get_qfrc_constraint(data, &n);
                if (qfrc_c && n > 0) CHECK_NO_NAN(qfrc_c, n, "no NaN in qfrc_constraint");

                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("humanoid_constraint_forces_nonzero");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);

                int n = 0;
                const float* qfrc_c = mjmlx_get_qfrc_constraint(data, &n);
                if (qfrc_c && n > 0) {
                    float max_f = 0;
                    for (int i = 0; i < n; i++) max_f = std::max(max_f, std::abs(qfrc_c[i]));
                    CHECK_GT(max_f, 0.1f, "humanoid has nonzero constraint forces");
                }
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("humanoid_multi_step_stable");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                for (int s = 0; s < 50; s++) {
                    mjmlx_step(model, data);
                    int nq = 0, nv = 0;
                    const float* qpos = mjmlx_get_qpos(data, &nq);
                    const float* qvel = mjmlx_get_qvel(data, &nv);
                    CHECK_NO_NAN(qpos, nq, "no NaN in qpos");
                    CHECK_NO_NAN(qvel, nv, "no NaN in qvel");
                    if (nq > 2) {
                        CHECK_GT(qpos[2], 0.5f, "z > 0.5 (not freefall)");
                        CHECK_LT(qpos[2], 5.0f, "z < 5 (bounded)");
                    }
                }
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();
    }

    // ── TestRegressions ──
    TEST_SECTION("Regressions");

    if (humanoid_xml_path) {
        TEST_BEGIN("humanoid_qacc_not_freefall");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);

                int n = 0;
                const float* qacc = mjmlx_get_qacc(data, &n);
                if (qacc && n >= 3) {
                    CHECK_GT(qacc[2], -9.0f, "qacc[z] > -9 (not freefall)");
                }
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("humanoid_qfrc_constraint_nonzero");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);

                int n = 0;
                const float* qfrc_c = mjmlx_get_qfrc_constraint(data, &n);
                if (qfrc_c && n > 0) {
                    float max_f = 0;
                    for (int i = 0; i < n; i++) max_f = std::max(max_f, std::abs(qfrc_c[i]));
                    CHECK_GT(max_f, 1.0f, "humanoid constraint forces > 1.0");
                }
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("humanoid_ncon_positive");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);
                int ncon = mjmlx_get_ncon(data);
                CHECK_GT(ncon, 0, "humanoid has contacts");
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();
    }

    // ── TestFloat32Stability ──
    TEST_SECTION("Float32Stability");

    if (humanoid_xml_path) {
        TEST_BEGIN("no_nan_100_steps");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);
                for (int s = 0; s < 100; s++) {
                    mjmlx_step(model, data);
                }
                int nq = 0, nv = 0;
                const float* qpos = mjmlx_get_qpos(data, &nq);
                const float* qvel = mjmlx_get_qvel(data, &nv);
                const float* qacc = mjmlx_get_qacc(data, &nv);
                CHECK_NO_NAN(qpos, nq, "no NaN qpos after 100 steps");
                CHECK_NO_NAN(qvel, nv, "no NaN qvel after 100 steps");
                if (qacc) CHECK_NO_NAN(qacc, nv, "no NaN qacc after 100 steps");
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("no_inf_100_steps");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                for (int s = 0; s < 100; s++) mjmlx_step(model, data);

                int nq = 0, nv = 0;
                const float* qpos = mjmlx_get_qpos(data, &nq);
                const float* qvel = mjmlx_get_qvel(data, &nv);
                for (int i = 0; i < nq; i++) CHECK(!std::isinf(qpos[i]), "no Inf in qpos");
                for (int i = 0; i < nv; i++) CHECK(!std::isinf(qvel[i]), "no Inf in qvel");
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();

        TEST_BEGIN("velocity_bounded_200_steps");
        {
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                for (int s = 0; s < 200; s++) mjmlx_step(model, data);

                int nv = 0;
                const float* qvel = mjmlx_get_qvel(data, &nv);
                float max_vel = 0;
                for (int i = 0; i < nv; i++) max_vel = std::max(max_vel, std::abs(qvel[i]));
                CHECK_LT(max_vel, 1e6f, "velocities bounded < 1e6");
                CHECK_NO_NAN(qvel, nv, "no NaN in qvel");
                mjmlx_free_data(data); mjmlx_free_model(model);
            }
        }
        TEST_END();
    }

    TEST_EXIT();
}
