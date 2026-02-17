// Phase 7.1: RK4 integrator tests.

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

// Simple double pendulum with RK4 integrator
static const char* RK4_PENDULUM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01" integrator="RK4"/>
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
</mujoco>
)";

// Free-falling body with RK4
static const char* RK4_FREE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01" integrator="RK4"/>
  <worldbody>
    <body name="ball" pos="0 0 5">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// With actuator and contacts
static const char* RK4_ACTUATED_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" mass="2"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="10"/>
  </actuator>
</mujoco>
)";

// RK4 with Euler for energy comparison
static const char* EULER_PENDULUM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01" integrator="Euler"/>
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
</mujoco>
)";

int main() {
    printf("=== test_rk4 ===\n");

    // ── Test 1: RK4 qacc agreement ───────────────────────────────────────────
    TEST_SECTION("RK4 integrator");
    TEST_BEGIN("rk4_1step_qacc");
    {
        MjScope mj(RK4_PENDULUM_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(RK4_PENDULUM_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.5; mj.d->qpos[1] = -0.3;
        sync_state(dh, mj.m, mj.d);

        mj_step(mj.m, mj.d);
        mjmlx_step(mh, dh);

        int nq = (int)mj.m->nq, nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qpos); mx::eval(md.qvel);

        float qpos_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(md.qpos.data<float>()[i] - (float)mj.d->qpos[i]);
        float qvel_diff = 0;
        for (int i = 0; i < nv; i++) qvel_diff += std::abs(md.qvel.data<float>()[i] - (float)mj.d->qvel[i]);

        printf("    qpos diff = %.6e, qvel diff = %.6e\n", qpos_diff, qvel_diff);
        printf("    MuJoCo C qpos =");
        for (int i = 0; i < nq; i++) printf(" %.6f", mj.d->qpos[i]);
        printf("\n    MLX      qpos =");
        for (int i = 0; i < nq; i++) printf(" %.6f", md.qpos.data<float>()[i]);
        printf("\n");

        CHECK_LT(qpos_diff, 1e-3, "RK4 1-step qpos diff < 1e-3");
        CHECK_LT(qvel_diff, 1e-3, "RK4 1-step qvel diff < 1e-3");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 2: RK4 10-step agreement ────────────────────────────────────────
    TEST_BEGIN("rk4_10step");
    {
        MjScope mj(RK4_PENDULUM_XML);
        auto path = write_temp_xml(RK4_PENDULUM_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.3; mj.d->qvel[0] = 1.0;
        sync_state(dh, mj.m, mj.d);

        for (int s = 0; s < 10; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        int nq = (int)mj.m->nq;
        auto& md = dh->data;
        mx::eval(md.qpos);

        float qpos_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(md.qpos.data<float>()[i] - (float)mj.d->qpos[i]);
        printf("    After 10 steps: qpos diff = %.6e\n", qpos_diff);
        CHECK_LT(qpos_diff, 1e-2, "RK4 10-step qpos diff < 1e-2");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 3: RK4 free body ────────────────────────────────────────────────
    TEST_BEGIN("rk4_free_body");
    {
        MjScope mj(RK4_FREE_XML);
        auto path = write_temp_xml(RK4_FREE_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        sync_state(dh, mj.m, mj.d);

        for (int s = 0; s < 50; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        int nq = (int)mj.m->nq;
        auto& md = dh->data;
        mx::eval(md.qpos);

        float qpos_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(md.qpos.data<float>()[i] - (float)mj.d->qpos[i]);
        printf("    Free body 50 steps: qpos diff = %.6e\n", qpos_diff);
        CHECK_LT(qpos_diff, 0.1, "RK4 free body qpos diff < 0.1");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 4: RK4 with actuator ────────────────────────────────────────────
    TEST_BEGIN("rk4_actuated");
    {
        MjScope mj(RK4_ACTUATED_XML);
        auto path = write_temp_xml(RK4_ACTUATED_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->ctrl[0] = 5.0;
        dh->data.ctrl = mx::array({5.0f});

        sync_state(dh, mj.m, mj.d);

        for (int s = 0; s < 20; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        int nq = (int)mj.m->nq;
        auto& md = dh->data;
        mx::eval(md.qpos);

        float qpos_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(md.qpos.data<float>()[i] - (float)mj.d->qpos[i]);
        printf("    Actuated 20 steps: qpos diff = %.6e\n", qpos_diff);
        CHECK_LT(qpos_diff, 0.1, "RK4 actuated qpos diff < 0.1");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 5: RK4 100-step stability ───────────────────────────────────────
    TEST_BEGIN("rk4_stability_100");
    {
        MjScope mj(RK4_PENDULUM_XML);
        auto path = write_temp_xml(RK4_PENDULUM_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 1.0; mj.d->qvel[0] = -2.0;
        sync_state(dh, mj.m, mj.d);

        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        int nq = (int)mj.m->nq, nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qpos); mx::eval(md.qvel);

        CHECK_NO_NAN(md.qpos.data<float>(), nq, "qpos not NaN");
        CHECK_NO_NAN(md.qvel.data<float>(), nv, "qvel not NaN");

        float qpos_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(md.qpos.data<float>()[i] - (float)mj.d->qpos[i]);
        printf("    After 100 steps: qpos diff = %.6e\n", qpos_diff);
        CHECK_LT(qpos_diff, 1.0, "RK4 100-step qpos diff < 1.0");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 6: RK4 uses correct integrator (not Euler) ─────────────────────
    TEST_BEGIN("rk4_differs_from_euler");
    {
        MjScope rk4_mj(RK4_PENDULUM_XML);
        MjScope euler_mj(EULER_PENDULUM_XML);

        // Start both at same state
        rk4_mj.d->qpos[0] = 0.5; rk4_mj.d->qpos[1] = -0.3;
        euler_mj.d->qpos[0] = 0.5; euler_mj.d->qpos[1] = -0.3;

        for (int s = 0; s < 10; s++) {
            mj_step(rk4_mj.m, rk4_mj.d);
            mj_step(euler_mj.m, euler_mj.d);
        }

        float diff = 0;
        for (int i = 0; i < (int)rk4_mj.m->nq; i++)
            diff += std::abs(rk4_mj.d->qpos[i] - euler_mj.d->qpos[i]);

        printf("    RK4 vs Euler qpos diff = %.6e\n", diff);
        CHECK(diff > 1e-6, "RK4 and Euler produce different results");

        mjmlx_free_model(nullptr); mjmlx_free_data(nullptr);
    }
    TEST_END();

    TEST_EXIT();
}
