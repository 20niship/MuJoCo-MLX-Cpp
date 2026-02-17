// Phase 6.1: Actuator dynamics tests.
// Tests FILTER, FILTEREXACT, and INTEGRATOR activation dynamics against MuJoCo C.

#include "test_utils.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

static void sync_state(MjmlxData* dh, mjModel* m, mjData* d) {
    int nq = (int)m->nq, nv = (int)m->nv, na = (int)m->na;
    std::vector<float> qpos(nq), qvel(nv), act(na);
    for (int i = 0; i < nq; i++) qpos[i] = (float)d->qpos[i];
    for (int i = 0; i < nv; i++) qvel[i] = (float)d->qvel[i];
    for (int i = 0; i < na; i++) act[i] = (float)d->act[i];
    dh->data.qpos = mx::array(qpos.data(), {nq}, mx::float32);
    dh->data.qvel = mx::array(qvel.data(), {nv}, mx::float32);
    if (na > 0) dh->data.act = mx::array(act.data(), {na}, mx::float32);
}

static void sync_ctrl(MjmlxData* dh, mjModel* m, mjData* d) {
    int nu = (int)m->nu;
    std::vector<float> ctrl(nu);
    for (int i = 0; i < nu; i++) ctrl[i] = (float)d->ctrl[i];
    dh->data.ctrl = mx::array(ctrl.data(), {nu}, mx::float32);
}

// ── FILTER dynamics: first-order low-pass ────────────────────────────────────
static const char* FILTER_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" dyntype="filter" dynprm="0.05" gainprm="100" biasprm="0 -100 0"/>
  </actuator>
</mujoco>
)";

// ── FILTEREXACT dynamics ─────────────────────────────────────────────────────
static const char* FILTEREXACT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" dyntype="filterexact" dynprm="0.05" gainprm="100" biasprm="0 -100 0"/>
  </actuator>
</mujoco>
)";

// ── INTEGRATOR dynamics ──────────────────────────────────────────────────────
static const char* INTEGRATOR_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" dyntype="integrator" gainprm="100" biasprm="0 -100 0"/>
  </actuator>
</mujoco>
)";

// ── Multiple actuators: mixed NONE + FILTER ──────────────────────────────────
static const char* MIXED_DYN_XML = R"(
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
  <actuator>
    <general joint="j1" gainprm="100" biasprm="0 -100 0"/>
    <general joint="j2" dyntype="filter" dynprm="0.03" gainprm="100" biasprm="0 -100 0"/>
  </actuator>
</mujoco>
)";

// ── Activation-limited filter ────────────────────────────────────────────────
static const char* ACT_LIMITED_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" dyntype="filter" dynprm="0.05" gainprm="100" biasprm="0 -100 0"
             actlimited="true" actrange="-0.5 0.5"/>
  </actuator>
</mujoco>
)";

int main() {
    printf("=== test_filter_dynamics ===\n");

    // ── Test 1: FILTER act_dot computation ───────────────────────────────────
    TEST_SECTION("FILTER dynamics");
    TEST_BEGIN("filter_act_dot");
    {
        MjScope mj(FILTER_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");
        CHECK(mj.m->na == 1, "na == 1");

        auto path = write_temp_xml(FILTER_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 0.5;
        sync_state(dh, mj.m, mj.d);
        sync_ctrl(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        auto& md = dh->data;
        mx::eval(md.act_dot);

        float mlx_adot = md.act_dot.data<float>()[0];
        float mj_adot = (float)mj.d->act_dot[0];
        printf("    MuJoCo C act_dot = %.6f, MLX act_dot = %.6f\n", mj_adot, mlx_adot);
        CHECK_CLOSE(mlx_adot, mj_adot, 1e-4, "filter act_dot");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 2: FILTER activation convergence over 100 steps ─────────────────
    TEST_BEGIN("filter_convergence_100");
    {
        MjScope mj(FILTER_XML);
        auto path = write_temp_xml(FILTER_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 0.8;
        sync_ctrl(dh, mj.m, mj.d);

        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.act);
        float mlx_act = md.act.data<float>()[0];
        float mj_act = (float)mj.d->act[0];

        printf("    After 100 steps: MuJoCo C act = %.6f, MLX act = %.6f\n", mj_act, mlx_act);
        CHECK_CLOSE(mlx_act, mj_act, 5e-3, "filter act after 100 steps");

        mx::eval(md.qpos); mx::eval(md.qvel);
        float qpos_diff = std::abs(md.qpos.data<float>()[0] - (float)mj.d->qpos[0]);
        float qvel_diff = std::abs(md.qvel.data<float>()[0] - (float)mj.d->qvel[0]);
        printf("    qpos diff = %.6e, qvel diff = %.6e\n", qpos_diff, qvel_diff);
        CHECK_LT(qpos_diff, 5e-2, "filter qpos diff < 5e-2");
        CHECK_LT(qvel_diff, 1.0, "filter qvel diff < 1.0");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 3: FILTEREXACT ──────────────────────────────────────────────────
    TEST_SECTION("FILTEREXACT dynamics");
    TEST_BEGIN("filterexact_convergence");
    {
        MjScope mj(FILTEREXACT_XML);
        auto path = write_temp_xml(FILTEREXACT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 0.8;
        sync_ctrl(dh, mj.m, mj.d);

        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.act);
        float mlx_act = md.act.data<float>()[0];
        float mj_act = (float)mj.d->act[0];

        printf("    After 100 steps: MuJoCo C act = %.6f, MLX act = %.6f\n", mj_act, mlx_act);
        CHECK_CLOSE(mlx_act, mj_act, 5e-3, "filterexact act after 100 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 4: INTEGRATOR dynamics ──────────────────────────────────────────
    TEST_SECTION("INTEGRATOR dynamics");
    TEST_BEGIN("integrator_act_ramp");
    {
        MjScope mj(INTEGRATOR_XML);
        CHECK(mj.m->na == 1, "na == 1");

        auto path = write_temp_xml(INTEGRATOR_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 1.0;
        sync_ctrl(dh, mj.m, mj.d);

        for (int s = 0; s < 50; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.act);
        float mlx_act = md.act.data<float>()[0];
        float mj_act = (float)mj.d->act[0];

        printf("    After 50 steps: MuJoCo C act = %.6f, MLX act = %.6f\n", mj_act, mlx_act);
        CHECK_CLOSE(mlx_act, mj_act, 5e-3, "integrator act after 50 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 5: Mixed NONE + FILTER ──────────────────────────────────────────
    TEST_SECTION("Mixed dynamics");
    TEST_BEGIN("mixed_none_filter");
    {
        MjScope mj(MIXED_DYN_XML);
        CHECK(mj.m->na == 1, "na == 1 (only filter actuator has activation)");

        auto path = write_temp_xml(MIXED_DYN_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 0.3;  // direct (no dynamics)
        mj.d->ctrl[1] = 0.7;  // filtered
        sync_ctrl(dh, mj.m, mj.d);

        for (int s = 0; s < 50; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        int nv = (int)mj.m->nv;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    After 50 steps: qacc diff = %.6e\n", qacc_diff);
        CHECK_LT(qacc_diff, 5.0, "mixed dynamics qacc diff reasonable");

        mx::eval(md.act);
        float mlx_act = md.act.data<float>()[0];
        float mj_act = (float)mj.d->act[0];
        printf("    act: MuJoCo C = %.6f, MLX = %.6f\n", mj_act, mlx_act);
        CHECK_CLOSE(mlx_act, mj_act, 5e-2, "mixed act reasonable");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 6: Activation clamping ──────────────────────────────────────────
    TEST_SECTION("Activation limits");
    TEST_BEGIN("act_limited_clamping");
    {
        MjScope mj(ACT_LIMITED_XML);
        CHECK(mj.m->na == 1, "na == 1");

        auto path = write_temp_xml(ACT_LIMITED_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 10.0;  // Large ctrl to saturate activation
        sync_ctrl(dh, mj.m, mj.d);

        for (int s = 0; s < 200; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.act);
        float mlx_act = md.act.data<float>()[0];
        float mj_act = (float)mj.d->act[0];

        printf("    After 200 steps: MuJoCo C act = %.6f (clamped to [-0.5, 0.5])\n", mj_act);
        printf("    MLX act = %.6f\n", mlx_act);

        CHECK_CLOSE(mlx_act, mj_act, 5e-3, "activation clamped correctly");
        CHECK_LT(std::abs(mlx_act), 0.501f, "MLX act within clamp range");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 7: Long stability (FILTER) ──────────────────────────────────────
    TEST_SECTION("Stability");
    TEST_BEGIN("filter_stability_500");
    {
        MjScope mj(FILTER_XML);
        auto path = write_temp_xml(FILTER_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 0.5;
        sync_ctrl(dh, mj.m, mj.d);

        for (int s = 0; s < 500; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.qpos); mx::eval(md.qvel); mx::eval(md.act);

        CHECK_NO_NAN(md.qpos.data<float>(), (int)mj.m->nq, "qpos not NaN");
        CHECK_NO_NAN(md.qvel.data<float>(), (int)mj.m->nv, "qvel not NaN");
        CHECK_NO_NAN(md.act.data<float>(), (int)mj.m->na, "act not NaN");

        float act_diff = std::abs(md.act.data<float>()[0] - (float)mj.d->act[0]);
        printf("    After 500 steps: act diff = %.6e\n", act_diff);
        CHECK_LT(act_diff, 5e-2, "filter act diff < 5e-2 after 500 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    TEST_EXIT();
}
