// Phase 7.2: ImplicitFast integrator tests.

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
    if (m->na > 0 && d->act) {
        std::vector<float> act(m->na);
        for (int i = 0; i < m->na; i++) act[i] = (float)d->act[i];
        dh->data.act = mx::array(act.data(), {(int)m->na}, mx::float32);
    }
}

// ── Test models ─────────────────────────────────────────────────────────────

static const char* STIFF_SPRING_XML = R"(
<mujoco model="stiff_springs">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="m1" pos="0 0 2.0">
      <joint name="j1" type="slide" axis="0 0 1" stiffness="10000" damping="20"/>
      <geom type="sphere" size="0.08" mass="1.0"/>
      <body name="m2" pos="0 0 -0.3">
        <joint name="j2" type="slide" axis="0 0 1" stiffness="8000" damping="15"/>
        <geom type="sphere" size="0.08" mass="0.8"/>
        <body name="m3" pos="0 0 -0.3">
          <joint name="j3" type="slide" axis="0 0 1" stiffness="6000" damping="12"/>
          <geom type="sphere" size="0.08" mass="0.6"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

static const char* DAMPED_PENDULUM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" damping="5.0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="3.0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

static const char* ACTUATED_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="2.0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="1.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a1" joint="j1" ctrlrange="-10 10" ctrllimited="true"/>
    <motor name="a2" joint="j2" ctrlrange="-5 5" ctrllimited="true"/>
  </actuator>
</mujoco>
)";

static const char* TENDON_DAMP_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="slide" axis="0 0 1" damping="1.0"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <body name="b2" pos="0 0 -0.3">
        <joint name="j2" type="slide" axis="0 0 1" damping="1.0"/>
        <geom type="sphere" size="0.05" mass="0.8"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1" damping="5.0">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="-1.0"/>
    </fixed>
  </tendon>
</mujoco>
)";

int main() {
    printf("=== Phase 7.2: ImplicitFast Integrator Tests ===\n\n");

    // ── Test 1: 1-step qacc comparison ──
    TEST_BEGIN("implicit_1step_qacc");
    {
        MjScope mj(STIFF_SPRING_XML);
        CHECK(mj.ok(), "model loaded");

        mj.d->qpos[0] = 0.1; mj.d->qpos[1] = -0.05; mj.d->qpos[2] = 0.02;
        mj.d->qvel[0] = 0.5; mj.d->qvel[1] = -0.3;  mj.d->qvel[2] = 0.1;

        auto* mh = mjmlx_load_xml_string(STIFF_SPRING_XML);
        auto* dh = mjmlx_make_data(mh);
        sync_state(dh, mj.m, mj.d);
        dh->data.ctrl = mx::zeros({std::max(mh->model.nu, 1)});

        mj_step(mj.m, mj.d);
        mjmlx_step(mh, dh);
        mx::eval(dh->data.qacc);

        int nv = mj.m->nv;
        auto qacc_ptr = dh->data.qacc.data<float>();
        double max_err = 0.0;
        double max_rel_err = 0.0;
        for (int i = 0; i < nv; i++) {
            double mlx_val = qacc_ptr[i];
            double mj_val = (double)mj.d->qacc[i];
            double err = std::abs(mlx_val - mj_val);
            double rel = (std::abs(mj_val) > 1e-6) ? err / std::abs(mj_val) : err;
            max_err = std::max(max_err, err);
            max_rel_err = std::max(max_rel_err, rel);
            printf("    qacc[%d]: mlx=%.6f, mj=%.6f, err=%.6e\n", i, mlx_val, mj_val, err);
        }
        printf("    max abs qacc error: %.6e, max rel error: %.6e\n", max_err, max_rel_err);
        CHECK_LT(max_err, 1e-1, "qacc matches MuJoCo C");

        // Also compare qvel and qpos (trajectory is the primary correctness criterion)
        mx::eval(dh->data.qpos); mx::eval(dh->data.qvel);
        auto qpos_ptr = dh->data.qpos.data<float>();
        auto qvel_ptr = dh->data.qvel.data<float>();
        double max_pos_err = 0, max_vel_err = 0;
        for (int i = 0; i < mj.m->nq; i++)
            max_pos_err = std::max(max_pos_err, (double)std::abs(qpos_ptr[i] - (float)mj.d->qpos[i]));
        for (int i = 0; i < nv; i++)
            max_vel_err = std::max(max_vel_err, (double)std::abs(qvel_ptr[i] - (float)mj.d->qvel[i]));
        printf("    max pos err: %.6e, max vel err: %.6e\n", max_pos_err, max_vel_err);
        CHECK_LT(max_pos_err, 1e-3, "position matches MuJoCo C");
        CHECK_LT(max_vel_err, 1e-2, "velocity matches MuJoCo C");

        mjmlx_free_data(dh); mjmlx_free_model(mh);
    }
    TEST_END();

    // ── Test 2: 10-step trajectory ──
    TEST_BEGIN("implicit_10step");
    {
        MjScope mj(STIFF_SPRING_XML);
        CHECK(mj.ok(), "model loaded");

        mj.d->qpos[0] = 0.05; mj.d->qvel[0] = 1.0;

        auto* mh = mjmlx_load_xml_string(STIFF_SPRING_XML);
        auto* dh = mjmlx_make_data(mh);
        sync_state(dh, mj.m, mj.d);
        dh->data.ctrl = mx::zeros({std::max(mh->model.nu, 1)});

        for (int s = 0; s < 10; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }
        mx::eval(dh->data.qpos); mx::eval(dh->data.qvel);

        int nq = mj.m->nq, nv = mj.m->nv;
        auto qpos_ptr = dh->data.qpos.data<float>();
        auto qvel_ptr = dh->data.qvel.data<float>();
        double max_pos_err = 0, max_vel_err = 0;
        for (int i = 0; i < nq; i++)
            max_pos_err = std::max(max_pos_err, (double)std::abs(qpos_ptr[i] - (float)mj.d->qpos[i]));
        for (int i = 0; i < nv; i++)
            max_vel_err = std::max(max_vel_err, (double)std::abs(qvel_ptr[i] - (float)mj.d->qvel[i]));

        printf("    10-step max pos err: %.6e, vel err: %.6e\n", max_pos_err, max_vel_err);
        CHECK_LT(max_pos_err, 1e-1, "position after 10 steps");
        CHECK_LT(max_vel_err, 1.0, "velocity after 10 steps");

        mjmlx_free_data(dh); mjmlx_free_model(mh);
    }
    TEST_END();

    // ── Test 3: Damped pendulum ──
    TEST_BEGIN("implicit_damped_pendulum");
    {
        MjScope mj(DAMPED_PENDULUM_XML);
        CHECK(mj.ok(), "model loaded");

        mj.d->qpos[0] = 0.5; mj.d->qvel[0] = 2.0;

        auto* mh = mjmlx_load_xml_string(DAMPED_PENDULUM_XML);
        auto* dh = mjmlx_make_data(mh);
        sync_state(dh, mj.m, mj.d);
        dh->data.ctrl = mx::zeros({std::max(mh->model.nu, 1)});

        for (int s = 0; s < 20; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }
        mx::eval(dh->data.qvel);

        int nv = mj.m->nv;
        auto qvel_ptr = dh->data.qvel.data<float>();
        double max_vel_err = 0;
        for (int i = 0; i < nv; i++)
            max_vel_err = std::max(max_vel_err, (double)std::abs(qvel_ptr[i] - (float)mj.d->qvel[i]));
        printf("    20-step max vel err: %.6e\n", max_vel_err);
        CHECK_LT(max_vel_err, 2.0, "velocity after 20 steps (damped)");

        mjmlx_free_data(dh); mjmlx_free_model(mh);
    }
    TEST_END();

    // ── Test 4: 100-step stability ──
    TEST_BEGIN("implicit_stability_100");
    {
        auto* mh = mjmlx_load_xml_string(STIFF_SPRING_XML);
        auto* dh = mjmlx_make_data(mh);

        dh->data.qpos = mx::array({0.1f, 0.0f, 0.0f});
        dh->data.qvel = mx::array({1.0f, 0.0f, 0.0f});
        dh->data.ctrl = mx::zeros({std::max(mh->model.nu, 1)});

        bool stable = true;
        for (int s = 0; s < 100; s++) {
            mjmlx_step(mh, dh);
            mx::eval(dh->data.qpos);
            auto qp = dh->data.qpos.data<float>();
            for (int i = 0; i < mh->model.nq; i++) {
                if (!std::isfinite(qp[i])) {
                    printf("    DIVERGED at step %d, dof %d: %f\n", s, i, qp[i]);
                    stable = false;
                    break;
                }
            }
            if (!stable) break;
        }
        CHECK(stable, "100 steps stable");

        mjmlx_free_data(dh); mjmlx_free_model(mh);
    }
    TEST_END();

    // ── Test 5: Actuated model ──
    TEST_BEGIN("implicit_actuated");
    {
        MjScope mj(ACTUATED_XML);
        CHECK(mj.ok(), "model loaded");

        mj.d->ctrl[0] = 2.0; mj.d->ctrl[1] = -1.0;
        mj.d->qpos[0] = 0.3; mj.d->qvel[0] = 0.5;

        auto* mh = mjmlx_load_xml_string(ACTUATED_XML);
        auto* dh = mjmlx_make_data(mh);
        sync_state(dh, mj.m, mj.d);
        dh->data.ctrl = mx::array({2.0f, -1.0f});

        for (int s = 0; s < 5; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }
        mx::eval(dh->data.qpos); mx::eval(dh->data.qvel);

        int nv = mj.m->nv, nq = mj.m->nq;
        auto qpos_ptr = dh->data.qpos.data<float>();
        auto qvel_ptr = dh->data.qvel.data<float>();
        double max_pos_err = 0, max_vel_err = 0;
        for (int i = 0; i < nq; i++)
            max_pos_err = std::max(max_pos_err, (double)std::abs(qpos_ptr[i] - (float)mj.d->qpos[i]));
        for (int i = 0; i < nv; i++)
            max_vel_err = std::max(max_vel_err, (double)std::abs(qvel_ptr[i] - (float)mj.d->qvel[i]));
        printf("    5-step actuated pos err: %.6e, vel err: %.6e\n", max_pos_err, max_vel_err);
        CHECK_LT(max_pos_err, 1e-2, "actuated pos after 5 steps");
        CHECK_LT(max_vel_err, 5e-1, "actuated vel after 5 steps");

        mjmlx_free_data(dh); mjmlx_free_model(mh);
    }
    TEST_END();

    // ── Test 6: Differs from Euler ──
    TEST_BEGIN("implicit_differs_from_euler");
    {
        static const char* EULER_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="m1" pos="0 0 2.0">
      <joint name="j1" type="slide" axis="0 0 1" stiffness="10000" damping="20"/>
      <geom type="sphere" size="0.08" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
)";

        auto* mh_impl = mjmlx_load_xml_string(STIFF_SPRING_XML);
        auto* dh_impl = mjmlx_make_data(mh_impl);

        auto* mh_euler = mjmlx_load_xml_string(EULER_XML);
        auto* dh_euler = mjmlx_make_data(mh_euler);

        dh_impl->data.qpos = mx::array({0.1f, 0.0f, 0.0f});
        dh_impl->data.qvel = mx::array({1.0f, 0.0f, 0.0f});
        dh_impl->data.ctrl = mx::zeros({std::max(mh_impl->model.nu, 1)});

        dh_euler->data.qpos = mx::array({0.1f});
        dh_euler->data.qvel = mx::array({1.0f});
        dh_euler->data.ctrl = mx::zeros({std::max(mh_euler->model.nu, 1)});

        for (int s = 0; s < 5; s++) {
            mjmlx_step(mh_impl, dh_impl);
            mjmlx_step(mh_euler, dh_euler);
        }
        mx::eval(dh_impl->data.qvel); mx::eval(dh_euler->data.qvel);

        auto impl_vel = dh_impl->data.qvel.data<float>()[0];
        auto euler_vel = dh_euler->data.qvel.data<float>()[0];
        double diff = std::abs(impl_vel - euler_vel);
        printf("    vel diff (implicit vs euler): %.6e (impl=%.4f, euler=%.4f)\n",
               diff, impl_vel, euler_vel);
        CHECK_GT(diff, 1e-3, "implicit differs from euler for stiff system");

        mjmlx_free_data(dh_impl); mjmlx_free_model(mh_impl);
        mjmlx_free_data(dh_euler); mjmlx_free_model(mh_euler);
    }
    TEST_END();

    // ── Test 7: Tendon damping ──
    TEST_BEGIN("implicit_tendon_damping");
    {
        MjScope mj(TENDON_DAMP_XML);
        CHECK(mj.ok(), "model loaded");

        mj.d->qpos[0] = 0.1; mj.d->qpos[1] = -0.05;
        mj.d->qvel[0] = 0.5; mj.d->qvel[1] = -0.3;

        auto* mh = mjmlx_load_xml_string(TENDON_DAMP_XML);
        auto* dh = mjmlx_make_data(mh);
        sync_state(dh, mj.m, mj.d);
        dh->data.ctrl = mx::zeros({std::max(mh->model.nu, 1)});

        mj_step(mj.m, mj.d);
        mjmlx_step(mh, dh);
        mx::eval(dh->data.qpos); mx::eval(dh->data.qvel);

        int nv = mj.m->nv, nq = mj.m->nq;
        auto qpos_ptr = dh->data.qpos.data<float>();
        auto qvel_ptr = dh->data.qvel.data<float>();
        double max_pos_err = 0, max_vel_err = 0;
        for (int i = 0; i < nq; i++)
            max_pos_err = std::max(max_pos_err, (double)std::abs(qpos_ptr[i] - (float)mj.d->qpos[i]));
        for (int i = 0; i < nv; i++)
            max_vel_err = std::max(max_vel_err, (double)std::abs(qvel_ptr[i] - (float)mj.d->qvel[i]));
        printf("    tendon damp pos err: %.6e, vel err: %.6e\n", max_pos_err, max_vel_err);
        CHECK_LT(max_pos_err, 1e-3, "tendon damping pos");
        CHECK_LT(max_vel_err, 1e-2, "tendon damping vel");

        mjmlx_free_data(dh); mjmlx_free_model(mh);
    }
    TEST_END();

    TEST_EXIT();
}
