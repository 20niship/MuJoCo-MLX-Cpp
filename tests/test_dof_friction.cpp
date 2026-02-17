// Phase 4.2: DOF friction loss tests.
// Tests dof_frictionloss constraint generation against MuJoCo C.
// Verifies: nf constraint rows, qacc agreement, friction decelerates spinning joint,
// mixed friction + limits + equality, batched pipeline.

#include "test_utils.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>
#include <unistd.h>

// ── Single hinge joint with friction loss ────────────────────────────────────
static const char* FRICTION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="0.5"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// ── Two hinge joints, only one has friction ──────────────────────────────────
static const char* MULTI_DOF_FRICTION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="0.3"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" frictionloss="0.0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// ── Friction + joint limits (mixed constraints) ──────────────────────────────
static const char* FRICTION_LIMITS_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="0.5"
             limited="true" range="-1.57 1.57"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// ── Free body with friction loss on slide joint ──────────────────────────────
static const char* SLIDE_FRICTION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="slider" pos="0 0 1">
      <joint name="s1" type="slide" axis="1 0 0" frictionloss="1.0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// ── Friction + equality constraint ───────────────────────────────────────────
static const char* FRICTION_EQUALITY_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="0.5"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" frictionloss="0.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
  </equality>
</mujoco>
)";

// ── High friction loss to test deceleration ──────────────────────────────────
static const char* DECEL_XML = R"(
<mujoco>
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="2.0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_dof_friction ===\n");

    // ── Test 1: friction_constraint_rows ─────────────────────────────────────
    {
        TEST_BEGIN("friction_constraint_rows");
        auto path = write_temp_xml(FRICTION_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);
        printf("    ne_c=%d nf_c=%d nl_c=%d nefc_c=%d\n", dj->ne, dj->nf, dj->nl, dj->nefc);
        printf("    ne_m=%d nf_m=%d nl_m=%d nefc_m=%d\n",
               dh->data.ne, dh->data.nf, dh->data.nl, dh->data.nefc);
        CHECK(dj->nf == 1, "MuJoCo C: 1 friction row for 1 DOF with frictionloss");
        CHECK(dh->data.nf == 1, "MLX: 1 friction row");
        CHECK(dj->ne == 0, "MuJoCo C: no equality rows");
        CHECK(dh->data.ne == 0, "MLX: no equality rows");
        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 2: multi_dof_friction_rows ──────────────────────────────────────
    {
        TEST_BEGIN("multi_dof_friction_rows");
        auto path = write_temp_xml(MULTI_DOF_FRICTION_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);
        printf("    ne_c=%d nf_c=%d nl_c=%d nefc_c=%d\n", dj->ne, dj->nf, dj->nl, dj->nefc);
        printf("    ne_m=%d nf_m=%d nl_m=%d nefc_m=%d\n",
               dh->data.ne, dh->data.nf, dh->data.nl, dh->data.nefc);
        CHECK(dj->nf == 1, "MuJoCo C: 1 friction row (only j1 has frictionloss)");
        CHECK(dh->data.nf == 1, "MLX: 1 friction row");
        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 3: friction_qacc ────────────────────────────────────────────────
    {
        TEST_BEGIN("friction_qacc");
        auto path = write_temp_xml(FRICTION_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        // Debug: compare constraint params
        int nefc_c = dj->nefc;
        printf("    Solver iterations: %d, tolerance: %g\n", mj->opt.iterations, mj->opt.tolerance);
        printf("    dof_frictionloss: %g\n", mj->dof_frictionloss[0]);
        printf("    dof_invweight0: %g\n", mj->dof_invweight0[0]);
        printf("    dof_solref: [%g, %g]\n", mj->dof_solref[0], mj->dof_solref[1]);
        printf("    dof_solimp: [%g, %g, %g, %g, %g]\n",
               mj->dof_solimp[0], mj->dof_solimp[1], mj->dof_solimp[2],
               mj->dof_solimp[3], mj->dof_solimp[4]);
        for (int i = 0; i < nefc_c; i++) {
            printf("    C efc[%d]: D=%.8g R=%.8g aref=%.8g floss=%.8g force=%.8g\n",
                   i, dj->efc_D[i], dj->efc_R[i], dj->efc_aref[i],
                   dj->efc_frictionloss[i], dj->efc_force[i]);
        }

        mx::eval(dh->data.efc_D); mx::eval(dh->data.efc_aref);
        mx::eval(dh->data.efc_frictionloss); mx::eval(dh->data.efc_force);
        auto mlx_D = dh->data.efc_D.data<float>();
        auto mlx_aref = dh->data.efc_aref.data<float>();
        auto mlx_fl = dh->data.efc_frictionloss.data<float>();
        auto mlx_force = dh->data.efc_force.data<float>();
        for (int i = 0; i < dh->data.nefc; i++) {
            printf("    MLX efc[%d]: D=%.8g aref=%.8g floss=%.8g force=%.8g\n",
                   i, mlx_D[i], mlx_aref[i], mlx_fl[i], mlx_force[i]);
        }

        mx::eval(dh->data.qacc);
        auto mlx_qacc = dh->data.qacc.data<float>();
        int nv = mj->nv;
        printf("    nv=%d, nf_c=%d, nf_mlx=%d\n", nv, dj->nf, dh->data.nf);
        printf("    qacc_smooth C: ");
        for (int i = 0; i < nv; i++) printf("%.6f ", dj->qacc_smooth[i]);
        printf("\n");
        mx::eval(dh->data.qacc_smooth);
        auto mlx_qs = dh->data.qacc_smooth.data<float>();
        printf("    qacc_smooth MLX: ");
        for (int i = 0; i < nv; i++) printf("%.6f ", mlx_qs[i]);
        printf("\n");
        for (int i = 0; i < nv; i++)
            printf("    qacc[%d]: C=%.6f MLX=%.6f diff=%.2e\n",
                   i, dj->qacc[i], mlx_qacc[i], std::abs(dj->qacc[i]-mlx_qacc[i]));
        CHECK_ARRAY_CLOSE(mlx_qacc, dj->qacc, nv, 1e-2, "qacc should match MuJoCo C");
        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 4: slide_friction_qacc ──────────────────────────────────────────
    {
        TEST_BEGIN("slide_friction_qacc");
        auto path = write_temp_xml(SLIDE_FRICTION_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);
        mx::eval(dh->data.qacc);
        auto mlx_qacc = dh->data.qacc.data<float>();
        int nv = mj->nv;
        printf("    nv=%d, nf_c=%d, nf_mlx=%d\n", nv, dj->nf, dh->data.nf);
        for (int i = 0; i < nv; i++)
            printf("    qacc[%d]: C=%.6f MLX=%.6f diff=%.2e\n",
                   i, dj->qacc[i], mlx_qacc[i], std::abs(dj->qacc[i]-mlx_qacc[i]));
        CHECK_ARRAY_CLOSE(mlx_qacc, dj->qacc, nv, 1e-2, "slide friction qacc");
        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 5: friction_limits_mixed ────────────────────────────────────────
    {
        TEST_BEGIN("friction_limits_mixed");
        auto path = write_temp_xml(FRICTION_LIMITS_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Set qpos near limit to activate limit constraint
        dj->qpos[0] = 1.5;
        mj_forward(mj, dj);

        // Reconstruct MLX data with matching qpos
        mjmlx_free_data(dh);
        dh = mjmlx_make_data(mh);
        mx::eval(dh->data.qpos);
        auto qp = dh->data.qpos.data<float>();
        std::vector<float> qpos_new(qp, qp + mj->nq);
        qpos_new[0] = 1.5f;
        dh->data.qpos = mx::array(qpos_new.data(), {(int)mj->nq}, mx::float32);
        mjmlx_forward(mh, dh);

        printf("    ne_c=%d nf_c=%d nl_c=%d nefc_c=%d\n", dj->ne, dj->nf, dj->nl, dj->nefc);
        printf("    ne_m=%d nf_m=%d nl_m=%d nefc_m=%d\n",
               dh->data.ne, dh->data.nf, dh->data.nl, dh->data.nefc);
        CHECK(dj->nf >= 1, "MuJoCo C: should have friction rows");
        CHECK(dh->data.nf >= 1, "MLX: should have friction rows");

        mx::eval(dh->data.qacc);
        auto mlx_qacc = dh->data.qacc.data<float>();
        int nv = mj->nv;
        for (int i = 0; i < nv; i++)
            printf("    qacc[%d]: C=%.6f MLX=%.6f diff=%.2e\n",
                   i, dj->qacc[i], mlx_qacc[i], std::abs(dj->qacc[i]-mlx_qacc[i]));
        CHECK_ARRAY_CLOSE(mlx_qacc, dj->qacc, nv, 1e-2, "mixed friction+limits qacc");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 6: friction_equality_mixed ──────────────────────────────────────
    {
        TEST_BEGIN("friction_equality_mixed");
        auto path = write_temp_xml(FRICTION_EQUALITY_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    ne_c=%d nf_c=%d nl_c=%d nefc_c=%d\n", dj->ne, dj->nf, dj->nl, dj->nefc);
        printf("    ne_m=%d nf_m=%d nl_m=%d nefc_m=%d\n",
               dh->data.ne, dh->data.nf, dh->data.nl, dh->data.nefc);
        CHECK(dj->ne >= 1, "MuJoCo C: should have equality rows");
        CHECK(dj->nf >= 1, "MuJoCo C: should have friction rows");
        CHECK(dh->data.ne >= 1, "MLX: should have equality rows");
        CHECK(dh->data.nf >= 1, "MLX: should have friction rows");
        CHECK(dh->data.ne == dj->ne, "ne should match");
        CHECK(dh->data.nf == dj->nf, "nf should match");

        mx::eval(dh->data.qacc);
        auto mlx_qacc = dh->data.qacc.data<float>();
        int nv = mj->nv;
        for (int i = 0; i < nv; i++)
            printf("    qacc[%d]: C=%.6f MLX=%.6f diff=%.2e\n",
                   i, dj->qacc[i], mlx_qacc[i], std::abs(dj->qacc[i]-mlx_qacc[i]));
        CHECK_ARRAY_CLOSE(mlx_qacc, dj->qacc, nv, 1e-2, "equality+friction qacc");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 7: friction_deceleration ────────────────────────────────────────
    // A spinning joint in zero gravity should decelerate due to friction loss
    {
        TEST_BEGIN("friction_deceleration");
        auto path = write_temp_xml(DECEL_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Give initial velocity
        dj->qvel[0] = 5.0;
        mjmlx_free_data(dh);
        dh = mjmlx_make_data(mh);
        std::vector<float> qvel_init = {5.0f};
        dh->data.qvel = mx::array(qvel_init.data(), {(int)mj->nv}, mx::float32);

        // Step both for 100 steps
        for (int s = 0; s < 100; s++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        mx::eval(dh->data.qvel);
        float mlx_vel = dh->data.qvel.data<float>()[0];
        float c_vel = dj->qvel[0];
        printf("    After 100 steps: C_vel=%.6f MLX_vel=%.6f\n", c_vel, mlx_vel);
        CHECK(std::abs(mlx_vel) < 5.0f, "MLX: friction should decelerate joint");
        CHECK(std::abs(c_vel) < 5.0f, "C: friction should decelerate joint");
        CHECK_CLOSE(mlx_vel, c_vel, 0.5, "velocity after 100 steps should roughly match");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 8: friction_stability_200 ───────────────────────────────────────
    {
        TEST_BEGIN("friction_stability_200");
        auto path = write_temp_xml(FRICTION_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        double max_diff = 0;
        for (int s = 0; s < 200; s++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        mx::eval(dh->data.qpos);
        auto mlx_qp = dh->data.qpos.data<float>();
        for (int i = 0; i < mj->nq; i++) {
            double diff = std::abs(mlx_qp[i] - dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 200 steps: %.2e\n", max_diff);
        CHECK(max_diff < 1e-1, "qpos should not diverge over 200 steps");
        CHECK_NO_NAN(mlx_qp, mj->nq, "qpos should not contain NaN");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_model(mh); mjmlx_free_data(dh);
        TEST_END();
    }

    // ── Test 9: batched_pipeline ─────────────────────────────────────────────
    {
        TEST_BEGIN("friction_batched_pipeline");
        auto path = write_temp_xml(FRICTION_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        MjmlxBatchedConfig cfg;
        cfg.num_envs = 4;
        cfg.use_gpu = 1;
        cfg.solver_iterations = 3;

        auto* sim = mjmlx_batched_create(mh, &cfg);
        CHECK(sim != nullptr, "batched sim created");

        mjmlx_batched_step(sim, nullptr);
        mjmlx_batched_step(sim, nullptr);

        mjmlx_batched_free(sim);
        mjmlx_free_model(mh);
        mjmlx_free_data(dh);
        TEST_END();
    }

    TEST_EXIT();
}
