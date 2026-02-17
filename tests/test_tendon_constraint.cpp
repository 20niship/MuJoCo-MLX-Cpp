// Phase 5.4-5.5: Tendon limits and tendon friction loss tests.

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

// ── Tendon with length limits ────────────────────────────────────────────────
static const char* TENDON_LIMIT_XML = R"(
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
    <fixed name="t1" limited="true" range="-0.5 0.5">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Tendon with friction loss ────────────────────────────────────────────────
static const char* TENDON_FRICTION_XML = R"(
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
    <fixed name="t1" frictionloss="5.0">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="1.0"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Tendon with both limits and friction ─────────────────────────────────────
static const char* TENDON_BOTH_XML = R"(
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
    <fixed name="t1" limited="true" range="-0.3 0.3" frictionloss="2.0">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
</mujoco>
)";

int main() {
    printf("=== test_tendon_constraint ===\n");

    // ── Test 1: Tendon limit — constraint activation ─────────────────────────
    TEST_SECTION("Tendon limits");
    TEST_BEGIN("tendon_limit_qacc");
    {
        MjScope mj(TENDON_LIMIT_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(TENDON_LIMIT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Push tendon past upper limit: coef1*j1 + coef2*j2 = 1.0*0.8 + 0.5*0.0 = 0.8 > 0.5
        mj.d->qpos[0] = 0.8;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        printf("    MuJoCo C nefc=%d, MLX nefc=%d\n", mj.d->nefc, md.nefc);
        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", qacc_diff);
        printf("    MuJoCo C qacc =");
        for (int i = 0; i < nv; i++) printf(" %.4f", mj.d->qacc[i]);
        printf("\n    MLX      qacc =");
        for (int i = 0; i < nv; i++) printf(" %.4f", mlx_qacc[i]);
        printf("\n");

        // Verify constraint count matches (tendon limit should generate constraint)
        CHECK(md.nefc > 0, "tendon limit generates constraint");
        CHECK_LT(qacc_diff, 0.5, "tendon limit qacc diff < 0.5");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 2: Tendon limit — multi-step clamping ───────────────────────────
    TEST_BEGIN("tendon_limit_stability_100");
    {
        MjScope mj(TENDON_LIMIT_XML);
        auto path = write_temp_xml(TENDON_LIMIT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.8; mj.d->qvel[0] = 5.0;
        sync_state(dh, mj.m, mj.d);

        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.qpos); mx::eval(md.qvel);

        int nq = (int)mj.m->nq, nv = (int)mj.m->nv;
        CHECK_NO_NAN(md.qpos.data<float>(), nq, "qpos not NaN");
        CHECK_NO_NAN(md.qvel.data<float>(), nv, "qvel not NaN");

        float qpos_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(md.qpos.data<float>()[i] - (float)mj.d->qpos[i]);
        printf("    After 100 steps: qpos diff = %.6e\n", qpos_diff);
        CHECK_LT(qpos_diff, 1.0, "tendon limit qpos diff < 1.0 after 100 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 3: No limit violation — no constraint ───────────────────────────
    TEST_BEGIN("tendon_no_limit_violation");
    {
        MjScope mj(TENDON_LIMIT_XML);
        auto path = write_temp_xml(TENDON_LIMIT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Tendon length = 1.0*0.1 + 0.5*0.1 = 0.15 -- within [-0.5, 0.5]
        mj.d->qpos[0] = 0.1; mj.d->qpos[1] = 0.1;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff (no limit) = %.6e\n", qacc_diff);
        CHECK_LT(qacc_diff, 0.01, "no limit qacc matches");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 4: Tendon friction loss — qacc agreement ────────────────────────
    TEST_SECTION("Tendon friction loss");
    TEST_BEGIN("tendon_friction_qacc");
    {
        MjScope mj(TENDON_FRICTION_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(TENDON_FRICTION_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qvel[0] = 3.0; mj.d->qvel[1] = -2.0;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        printf("    MuJoCo C nefc=%d, MLX nefc=%d\n", mj.d->nefc, md.nefc);
        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", qacc_diff);
        printf("    MuJoCo C qacc =");
        for (int i = 0; i < nv; i++) printf(" %.4f", mj.d->qacc[i]);
        printf("\n    MLX      qacc =");
        for (int i = 0; i < nv; i++) printf(" %.4f", mlx_qacc[i]);
        printf("\n");

        CHECK(md.nefc > 0, "tendon friction generates constraint");
        CHECK_LT(qacc_diff, 0.5, "tendon friction qacc diff < 0.5");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 5: Tendon friction — damping effect over steps ──────────────────
    TEST_BEGIN("tendon_friction_damping_100");
    {
        MjScope mj(TENDON_FRICTION_XML);
        auto path = write_temp_xml(TENDON_FRICTION_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qvel[0] = 5.0;
        sync_state(dh, mj.m, mj.d);

        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        auto& md = dh->data;
        mx::eval(md.qvel);
        int nv = (int)mj.m->nv;
        auto mlx_qvel = md.qvel.data<float>();

        CHECK_NO_NAN(mlx_qvel, nv, "qvel not NaN");
        float qvel_diff = 0;
        for (int i = 0; i < nv; i++) qvel_diff += std::abs(mlx_qvel[i] - (float)mj.d->qvel[i]);
        printf("    After 100 steps: qvel diff = %.6e\n", qvel_diff);
        CHECK_LT(qvel_diff, 5.0, "tendon friction qvel diff < 5.0");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 6: Both limits + friction ───────────────────────────────────────
    TEST_SECTION("Combined");
    TEST_BEGIN("tendon_both_limits_friction");
    {
        MjScope mj(TENDON_BOTH_XML);
        auto path = write_temp_xml(TENDON_BOTH_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.5; mj.d->qvel[0] = 3.0;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = dh->data;
        mx::eval(md.qacc);
        auto mlx_qacc = md.qacc.data<float>();

        printf("    MuJoCo C nefc=%d, MLX nefc=%d\n", mj.d->nefc, md.nefc);
        float qacc_diff = 0;
        for (int i = 0; i < nv; i++) qacc_diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", qacc_diff);
        CHECK(md.nefc >= 2, "both limit + friction constraints");
        CHECK_LT(qacc_diff, 1.0, "combined qacc diff < 1.0");

        // Run 100 steps for stability
        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        mx::eval(md.qpos);
        CHECK_NO_NAN(md.qpos.data<float>(), (int)mj.m->nq, "qpos not NaN after 100 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    TEST_EXIT();
}
