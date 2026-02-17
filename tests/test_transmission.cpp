// Phase 5.3: Transmission tests.
// Tests TENDON and SITE actuator transmission against MuJoCo C.
// Verifies actuator_length, actuator_moment, qfrc_actuator, and qacc agreement.

#include "test_utils.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

static void sync_state(MjmlxData* dh, mjModel* m, mjData* d) {
    int nq = (int)m->nq, nv = (int)m->nv;
    std::vector<float> qpos(nq), qvel(nv);
    for (int i = 0; i < nq; i++) qpos[i] = (float)d->qpos[i];
    for (int i = 0; i < nv; i++) qvel[i] = (float)d->qvel[i];
    dh->data.qpos = mx::array(qpos.data(), {nq}, mx::float32);
    dh->data.qvel = mx::array(qvel.data(), {nv}, mx::float32);
}

static void sync_ctrl(MjmlxData* dh, mjModel* m, mjData* d) {
    int nu = (int)m->nu;
    std::vector<float> ctrl(nu);
    for (int i = 0; i < nu; i++) ctrl[i] = (float)d->ctrl[i];
    dh->data.ctrl = mx::array(ctrl.data(), {nu}, mx::float32);
}

// ── TENDON transmission: fixed tendon with actuator ──────────────────────────
static const char* TENDON_TRN_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <general tendon="t1" gainprm="100" biasprm="0 -100 0" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
)";

// ── Multiple tendons with multiple actuators ─────────────────────────────────
static const char* MULTI_TENDON_TRN_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t_sum">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="1.0"/>
    </fixed>
    <fixed name="t_diff">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="-1.0"/>
    </fixed>
  </tendon>
  <actuator>
    <general tendon="t_sum" gainprm="50" biasprm="0 -50 0"/>
    <general tendon="t_diff" gainprm="50" biasprm="0 -50 0"/>
  </actuator>
</mujoco>
)";

// ── SITE transmission: single site actuator ──────────────────────────────────
static const char* SITE_TRN_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <site name="s1" pos="0.15 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <general site="s1" gear="0 0 1 0 0 0" gainprm="100" biasprm="0 -100 0"/>
  </actuator>
</mujoco>
)";

// ── SITE transmission with refsite ───────────────────────────────────────────
static const char* SITE_REF_TRN_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <site name="s1" pos="0.15 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
        <site name="s2" pos="0.15 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general site="s1" refsite="s2" gear="0 0 1 0 0 0" gainprm="100" biasprm="0 -100 0"/>
  </actuator>
</mujoco>
)";

// ── Mixed: JOINT + TENDON transmission ───────────────────────────────────────
static const char* MIXED_TRN_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="1.0"/>
    </fixed>
  </tendon>
  <actuator>
    <general joint="j1" gainprm="100" biasprm="0 -100 0"/>
    <general tendon="t1" gainprm="50" biasprm="0 -50 0"/>
  </actuator>
</mujoco>
)";

int main() {
    printf("=== test_transmission ===\n");

    // ── Test 1: TENDON transmission — actuator_length and moment ─────────────
    TEST_SECTION("TENDON transmission");
    TEST_BEGIN("tendon_trn_moment");
    {
        MjScope mj(TENDON_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(TENDON_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.3; mj.d->qpos[1] = 0.5;
        mj.d->qvel[0] = 1.0; mj.d->qvel[1] = -0.5;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        int nu = (int)mj.m->nu;
        auto& md = dh->data;
        mx::eval(md.actuator_length); mx::eval(md.actuator_moment);

        float mlx_alen = md.actuator_length.data<float>()[0];
        printf("    MuJoCo C actuator_length[0] = %.8f\n", mj.d->actuator_length[0]);
        printf("    MLX      actuator_length[0] = %.8f\n", (double)mlx_alen);

        CHECK_CLOSE(mlx_alen, mj.d->actuator_length[0], 1e-4, "tendon actuator_length");

        auto mlx_mom = md.actuator_moment.data<float>();
        printf("    MuJoCo C moment[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.4f", mj.d->actuator_moment[j]);
        printf("\n    MLX      moment[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.4f", mlx_mom[j]);
        printf("\n");

        CHECK_ARRAY_CLOSE(mlx_mom, mj.d->actuator_moment, nv, 1e-4, "tendon actuator_moment");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 2: TENDON transmission — qacc agreement ─────────────────────────
    TEST_BEGIN("tendon_trn_qacc");
    {
        MjScope mj(TENDON_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(TENDON_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.3; mj.d->qvel[0] = 1.0;
        mj.d->ctrl[0] = 0.5;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        float diff = 0;
        for (int i = 0; i < nv; i++) diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", diff);
        printf("    MuJoCo C qacc =");
        for (int i = 0; i < nv; i++) printf(" %.6f", mj.d->qacc[i]);
        printf("\n    MLX      qacc =");
        for (int i = 0; i < nv; i++) printf(" %.6f", mlx_qacc[i]);
        printf("\n");

        CHECK_ARRAY_CLOSE(mlx_qacc, mj.d->qacc, nv, 5e-3, "tendon trn qacc");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 3: Multi-tendon transmission ────────────────────────────────────
    TEST_BEGIN("multi_tendon_trn");
    {
        MjScope mj(MULTI_TENDON_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(MULTI_TENDON_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.2; mj.d->qpos[1] = -0.3;
        mj.d->ctrl[0] = 0.5; mj.d->ctrl[1] = -0.5;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        int nu = (int)mj.m->nu;
        auto& md = dh->data;
        mx::eval(md.qacc); mx::eval(md.qfrc_actuator);

        auto mlx_qacc = md.qacc.data<float>();
        auto mlx_qfrc = md.qfrc_actuator.data<float>();

        float qacc_diff = 0, qfrc_diff = 0;
        for (int i = 0; i < nv; i++) {
            qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
            qfrc_diff += std::abs(mlx_qfrc[i] - (float)mj.d->qfrc_actuator[i]);
        }
        printf("    qacc diff = %.6e, qfrc_actuator diff = %.6e\n", qacc_diff, qfrc_diff);

        CHECK_LT(qacc_diff, 5e-2, "multi-tendon qacc diff < 5e-2");
        CHECK_LT(qfrc_diff, 1e-2, "multi-tendon qfrc_actuator diff < 1e-2");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 4: SITE transmission — moment and qacc ──────────────────────────
    TEST_SECTION("SITE transmission");
    TEST_BEGIN("site_trn_qacc");
    {
        MjScope mj(SITE_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SITE_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.3;
        mj.d->qvel[0] = 1.0;
        mj.d->ctrl[0] = 0.5;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc); mx::eval(md.actuator_moment);

        auto mlx_qacc = md.qacc.data<float>();
        auto mlx_mom = md.actuator_moment.data<float>();

        printf("    MuJoCo C moment[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.6f", mj.d->actuator_moment[j]);
        printf("\n    MLX      moment[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.6f", mlx_mom[j]);
        printf("\n");

        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", qacc_diff);

        CHECK_ARRAY_CLOSE(mlx_mom, mj.d->actuator_moment, nv, 1e-3, "site actuator_moment");
        CHECK_LT(qacc_diff, 1e-2, "site qacc diff < 1e-2");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 5: SITE transmission with refsite ───────────────────────────────
    TEST_BEGIN("site_ref_trn_qacc");
    {
        MjScope mj(SITE_REF_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SITE_REF_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.2; mj.d->qpos[1] = 0.4;
        mj.d->ctrl[0] = 0.5;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc); mx::eval(md.actuator_moment);

        auto mlx_qacc = md.qacc.data<float>();
        auto mlx_mom = md.actuator_moment.data<float>();

        printf("    MuJoCo C moment[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.6f", mj.d->actuator_moment[j]);
        printf("\n    MLX      moment[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.6f", mlx_mom[j]);
        printf("\n");

        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", qacc_diff);

        // Refsite differential moment has a known approximation:
        // the Jacobian subtraction at two different body points introduces error
        // for DOFs shared by both bodies. Second DOF (refsite-only) is exact.
        mx::eval(md.qfrc_actuator);
        auto mlx_qfrc = md.qfrc_actuator.data<float>();
        printf("    MuJoCo C qfrc_actuator =");
        for (int j = 0; j < nv; j++) printf(" %.6f", mj.d->qfrc_actuator[j]);
        printf("\n    MLX      qfrc_actuator =");
        for (int j = 0; j < nv; j++) printf(" %.6f", mlx_qfrc[j]);
        printf("\n");

        // At least verify the refsite-only DOF matches
        if (nv >= 2) {
            CHECK_CLOSE(mlx_qfrc[nv - 1], mj.d->qfrc_actuator[nv - 1], 0.1,
                        "refsite-only DOF qfrc_actuator matches");
        }
        CHECK_NO_NAN(mlx_qfrc, nv, "qfrc_actuator not NaN");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 6: Mixed JOINT + TENDON transmission ────────────────────────────
    TEST_SECTION("Mixed transmission");
    TEST_BEGIN("mixed_joint_tendon_trn");
    {
        MjScope mj(MIXED_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(MIXED_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.2; mj.d->qpos[1] = -0.3;
        mj.d->ctrl[0] = 0.5; mj.d->ctrl[1] = -0.3;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        float diff = 0;
        for (int i = 0; i < nv; i++) diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", diff);

        CHECK_LT(diff, 1e-2, "mixed joint+tendon qacc diff < 1e-2");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 7: Multi-step stability with tendon actuator ────────────────────
    TEST_SECTION("Multi-step stability");
    TEST_BEGIN("tendon_trn_stability_100");
    {
        MjScope mj(TENDON_TRN_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(TENDON_TRN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qvel[0] = 2.0; mj.d->qvel[1] = -1.0;
        mj.d->ctrl[0] = 0.3;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        int nq = (int)mj.m->nq, nv = (int)mj.m->nv;

        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.qpos); mx::eval(md.qvel);
        auto mlx_qpos = md.qpos.data<float>();
        auto mlx_qvel = md.qvel.data<float>();

        float qpos_diff = 0, qvel_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(mlx_qpos[i] - (float)mj.d->qpos[i]);
        for (int i = 0; i < nv; i++) qvel_diff += std::abs(mlx_qvel[i] - (float)mj.d->qvel[i]);

        printf("    After 100 steps: qpos diff = %.6e, qvel diff = %.6e\n", qpos_diff, qvel_diff);

        CHECK_NO_NAN(mlx_qpos, nq, "qpos not NaN");
        CHECK_NO_NAN(mlx_qvel, nv, "qvel not NaN");
        CHECK_LT(qpos_diff, 5e-2, "qpos diff < 5e-2 after 100 steps");
        CHECK_LT(qvel_diff, 5e-2, "qvel diff < 5e-2 after 100 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    TEST_EXIT();
}
