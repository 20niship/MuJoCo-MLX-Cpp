// Phase 5.1: Fixed tendon tests.
// Tests fixed (joint-based) tendons against MuJoCo C.
// Verifies ten_length, ten_velocity, ten_J, and qacc agreement.

#include "test_utils.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

// Helper: copy MuJoCo C double qpos/qvel to MLX float arrays
static void sync_state(MjmlxData* dh, mjModel* m, mjData* d) {
    int nq = (int)m->nq, nv = (int)m->nv;
    std::vector<float> qpos(nq), qvel(nv);
    for (int i = 0; i < nq; i++) qpos[i] = (float)d->qpos[i];
    for (int i = 0; i < nv; i++) qvel[i] = (float)d->qvel[i];
    dh->data.qpos = mx::array(qpos.data(), {nq}, mx::float32);
    dh->data.qvel = mx::array(qvel.data(), {nv}, mx::float32);
}

// ── Single fixed tendon: one hinge joint ─────────────────────────────────────
static const char* SINGLE_TENDON_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1.5"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Multi-joint fixed tendon: two hinge joints ──────────────────────────────
static const char* MULTI_JOINT_XML = R"(
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
      <joint joint="j1" coef="2.0"/>
      <joint joint="j2" coef="-1.0"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Multiple tendons ─────────────────────────────────────────────────────────
static const char* MULTI_TENDON_XML = R"(
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
    </fixed>
    <fixed name="t2">
      <joint joint="j1" coef="0.5"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Fixed tendon with slide joint ────────────────────────────────────────────
static const char* SLIDE_TENDON_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="3.0"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Tendon with spring stiffness ─────────────────────────────────────────────
static const char* SPRING_TENDON_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1" stiffness="10" damping="0.5">
      <joint joint="j1" coef="1.0"/>
    </fixed>
  </tendon>
</mujoco>
)";

// ── Tendon + actuator (tendon transmission, if supported) ────────────────────
static const char* TENDON_MIXED_XML = R"(
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
    <fixed name="t2">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="-1.0"/>
    </fixed>
  </tendon>
</mujoco>
)";

int main() {
    printf("=== test_fixed_tendon ===\n");

    // ── Test 1: Single tendon, single joint, ten_length ──────────────────────
    TEST_SECTION("Single fixed tendon");
    TEST_BEGIN("single_tendon_length");
    {
        MjScope mj(SINGLE_TENDON_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SINGLE_TENDON_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Set a non-zero joint position
        mj.d->qpos[0] = 0.5;  // 0.5 rad
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        // MuJoCo C tendon length
        double mj_tlen = mj.d->ten_length[0];

        // MLX tendon length
        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.ten_length);
        float mlx_tlen = md.ten_length.data<float>()[0];

        printf("    MuJoCo C ten_length[0] = %.8f\n", mj_tlen);
        printf("    MLX      ten_length[0] = %.8f\n", (double)mlx_tlen);
        printf("    Expected: 1.5 * 0.5 = 0.75\n");

        CHECK_CLOSE(mlx_tlen, mj_tlen, 1e-5, "ten_length matches MuJoCo C");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 2: Single tendon, ten_velocity ──────────────────────────────────
    TEST_BEGIN("single_tendon_velocity");
    {
        MjScope mj(SINGLE_TENDON_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SINGLE_TENDON_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Set non-zero velocity
        mj.d->qvel[0] = 2.0;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        double mj_tvel = mj.d->ten_velocity[0];

        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.ten_velocity);
        float mlx_tvel = md.ten_velocity.data<float>()[0];

        printf("    MuJoCo C ten_velocity[0] = %.8f\n", mj_tvel);
        printf("    MLX      ten_velocity[0] = %.8f\n", (double)mlx_tvel);
        printf("    Expected: 1.5 * 2.0 = 3.0\n");

        CHECK_CLOSE(mlx_tvel, mj_tvel, 1e-5, "ten_velocity matches MuJoCo C");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 3: Single tendon, ten_J ─────────────────────────────────────────
    TEST_BEGIN("single_tendon_jacobian");
    {
        MjScope mj(SINGLE_TENDON_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SINGLE_TENDON_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;

        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.ten_J);
        auto mlx_J = md.ten_J.data<float>();

        printf("    MuJoCo C ten_J[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.4f", mj.d->ten_J[j]);
        printf("\n    MLX      ten_J[0,:] =");
        for (int j = 0; j < nv; j++) printf(" %.4f", mlx_J[j]);
        printf("\n");

        CHECK_ARRAY_CLOSE(mlx_J, mj.d->ten_J, nv, 1e-5, "ten_J matches MuJoCo C");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 4: Multi-joint tendon ───────────────────────────────────────────
    TEST_SECTION("Multi-joint fixed tendon");
    TEST_BEGIN("multi_joint_tendon");
    {
        MjScope mj(MULTI_JOINT_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(MULTI_JOINT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Set joint positions
        mj.d->qpos[0] = 0.3;  // j1
        mj.d->qpos[1] = 0.7;  // j2
        mj.d->qvel[0] = 1.0;
        mj.d->qvel[1] = -0.5;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.ten_length); mx::eval(md.ten_velocity); mx::eval(md.ten_J);

        float mlx_tlen = md.ten_length.data<float>()[0];
        float mlx_tvel = md.ten_velocity.data<float>()[0];

        printf("    MuJoCo C ten_length[0] = %.8f\n", mj.d->ten_length[0]);
        printf("    MLX      ten_length[0] = %.8f\n", (double)mlx_tlen);
        printf("    Expected: 2.0*0.3 + (-1.0)*0.7 = -0.1\n");
        printf("    MuJoCo C ten_velocity[0] = %.8f\n", mj.d->ten_velocity[0]);
        printf("    MLX      ten_velocity[0] = %.8f\n", (double)mlx_tvel);
        printf("    Expected: 2.0*1.0 + (-1.0)*(-0.5) = 2.5\n");

        CHECK_CLOSE(mlx_tlen, mj.d->ten_length[0], 1e-5, "multi-joint ten_length");
        CHECK_CLOSE(mlx_tvel, mj.d->ten_velocity[0], 1e-5, "multi-joint ten_velocity");

        auto mlx_J = md.ten_J.data<float>();
        CHECK_ARRAY_CLOSE(mlx_J, mj.d->ten_J, nv, 1e-5, "multi-joint ten_J");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 5: Multiple tendons ─────────────────────────────────────────────
    TEST_SECTION("Multiple tendons");
    TEST_BEGIN("multiple_tendons");
    {
        MjScope mj(MULTI_TENDON_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(MULTI_TENDON_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.4;
        mj.d->qpos[1] = 0.6;
        mj.d->qvel[0] = 1.5;
        mj.d->qvel[1] = -1.0;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nt = (int)mj.m->ntendon;
        int nv = (int)mj.m->nv;
        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.ten_length); mx::eval(md.ten_velocity); mx::eval(md.ten_J);

        CHECK(nt == 2, "2 tendons");
        CHECK(md.ten_length.shape(0) == 2, "MLX: 2 tendon lengths");

        for (int t = 0; t < nt; t++) {
            float mlx_l = md.ten_length.data<float>()[t];
            float mlx_v = md.ten_velocity.data<float>()[t];
            printf("    tendon[%d]: MuJoCo C len=%.6f vel=%.6f, MLX len=%.6f vel=%.6f\n",
                   t, mj.d->ten_length[t], mj.d->ten_velocity[t], (double)mlx_l, (double)mlx_v);
            CHECK_CLOSE(mlx_l, mj.d->ten_length[t], 1e-5, "ten_length");
            CHECK_CLOSE(mlx_v, mj.d->ten_velocity[t], 1e-5, "ten_velocity");
        }

        auto mlx_J = md.ten_J.data<float>();
        for (int t = 0; t < nt; t++) {
            CHECK_ARRAY_CLOSE(mlx_J + t * nv, mj.d->ten_J + t * nv, nv, 1e-5, "ten_J row");
        }

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 6: Slide joint tendon ───────────────────────────────────────────
    TEST_SECTION("Slide joint tendon");
    TEST_BEGIN("slide_tendon");
    {
        MjScope mj(SLIDE_TENDON_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SLIDE_TENDON_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj.d->qpos[0] = 0.2;  // 0.2 m displacement
        mj.d->qvel[0] = -1.0;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.ten_length); mx::eval(md.ten_velocity);

        float mlx_l = md.ten_length.data<float>()[0];
        float mlx_v = md.ten_velocity.data<float>()[0];

        printf("    MuJoCo C ten_length = %.8f, ten_velocity = %.8f\n",
               mj.d->ten_length[0], mj.d->ten_velocity[0]);
        printf("    MLX      ten_length = %.8f, ten_velocity = %.8f\n",
               (double)mlx_l, (double)mlx_v);

        CHECK_CLOSE(mlx_l, mj.d->ten_length[0], 1e-5, "slide ten_length");
        CHECK_CLOSE(mlx_v, mj.d->ten_velocity[0], 1e-5, "slide ten_velocity");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 7: Tendon with spring/damping — qacc agreement ──────────────────
    TEST_SECTION("Tendon spring/damping");
    TEST_BEGIN("spring_tendon_qacc");
    {
        MjScope mj(SPRING_TENDON_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(SPRING_TENDON_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Give it an initial angle and velocity
        mj.d->qpos[0] = 0.3;
        mj.d->qvel[0] = 1.0;
        sync_state(dh, mj.m, mj.d);

        mj_forward(mj.m, mj.d);
        mjmlx_forward(mh, dh);

        int nv = (int)mj.m->nv;
        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.qacc);

        auto mlx_qacc = md.qacc.data<float>();

        printf("    MuJoCo C qacc =");
        for (int i = 0; i < nv; i++) printf(" %.8f", mj.d->qacc[i]);
        printf("\n    MLX      qacc =");
        for (int i = 0; i < nv; i++) printf(" %.8f", mlx_qacc[i]);
        printf("\n");

        float diff = 0;
        for (int i = 0; i < nv; i++) diff += std::abs(mlx_qacc[i] - (float)mj.d->qacc[i]);
        printf("    qacc diff = %.6e\n", diff);

        CHECK_ARRAY_CLOSE(mlx_qacc, mj.d->qacc, nv, 1e-3, "qacc with tendon spring");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 8: Multi-step stability with tendons ────────────────────────────
    TEST_SECTION("Multi-step stability");
    TEST_BEGIN("tendon_stability_100");
    {
        MjScope mj(TENDON_MIXED_XML);
        CHECK(mj.ok(), "MuJoCo C model loaded");

        auto path = write_temp_xml(TENDON_MIXED_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        // Give initial push
        mj.d->qvel[0] = 2.0;
        mj.d->qvel[1] = -1.0;
        sync_state(dh, mj.m, mj.d);

        int nv = (int)mj.m->nv;
        int nq = (int)mj.m->nq;
        int nt = (int)mj.m->ntendon;

        // Step 100 times
        for (int s = 0; s < 100; s++) {
            mj_step(mj.m, mj.d);
            mjmlx_step(mh, dh);
        }

        // Compare final qpos, qvel, ten_length
        auto& md = ((MjmlxData*)dh)->data;
        mx::eval(md.qpos); mx::eval(md.qvel); mx::eval(md.ten_length);

        auto mlx_qpos = md.qpos.data<float>();
        auto mlx_qvel = md.qvel.data<float>();
        auto mlx_tlen = md.ten_length.data<float>();

        float qpos_diff = 0, qvel_diff = 0, tlen_diff = 0;
        for (int i = 0; i < nq; i++) qpos_diff += std::abs(mlx_qpos[i] - (float)mj.d->qpos[i]);
        for (int i = 0; i < nv; i++) qvel_diff += std::abs(mlx_qvel[i] - (float)mj.d->qvel[i]);
        for (int i = 0; i < nt; i++) tlen_diff += std::abs(mlx_tlen[i] - (float)mj.d->ten_length[i]);

        printf("    After 100 steps:\n");
        printf("    qpos diff = %.6e\n", qpos_diff);
        printf("    qvel diff = %.6e\n", qvel_diff);
        printf("    ten_length diff = %.6e\n", tlen_diff);

        CHECK_NO_NAN(mlx_qpos, nq, "qpos not NaN");
        CHECK_NO_NAN(mlx_qvel, nv, "qvel not NaN");
        CHECK_NO_NAN(mlx_tlen, nt, "ten_length not NaN");

        CHECK_LT(qpos_diff, 1e-2, "qpos diff < 1e-2 after 100 steps");
        CHECK_LT(qvel_diff, 1e-2, "qvel diff < 1e-2 after 100 steps");
        CHECK_LT(tlen_diff, 1e-3, "ten_length diff < 1e-3 after 100 steps");

        mjmlx_free_model(mh); mjmlx_free_data(dh);
    }
    TEST_END();

    // ── Test 9: Batched pipeline ─────────────────────────────────────────────
    TEST_SECTION("Batched pipeline");
    TEST_BEGIN("tendon_batched");
    {
        auto path = write_temp_xml(MULTI_JOINT_XML);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        unlink(path.c_str());

        // Create batch of 4
        int batch = 4;
        MjmlxData* batch_data[4];
        for (int b = 0; b < batch; b++) {
            batch_data[b] = mjmlx_make_data(mh);
            float qpos[] = {0.1f * (b + 1), 0.2f * (b + 1)};
            float qvel[] = {1.0f * (b + 1), -0.5f * (b + 1)};
            mjmlx_set_qpos(batch_data[b], qpos, 2);
            mjmlx_set_qvel(batch_data[b], qvel, 2);
        }

        // Step each
        bool all_ok = true;
        for (int b = 0; b < batch; b++) {
            mjmlx_step(mh, batch_data[b]);
            auto& md = ((MjmlxData*)batch_data[b])->data;
            mx::eval(md.ten_length);
            float tl = md.ten_length.data<float>()[0];
            if (std::isnan(tl) || std::isinf(tl)) {
                printf("    batch[%d] ten_length = %f (NaN/Inf!)\n", b, tl);
                all_ok = false;
            }
        }
        CHECK(all_ok, "All batched ten_length finite");

        for (int b = 0; b < batch; b++) mjmlx_free_data(batch_data[b]);
        mjmlx_free_model(mh);
    }
    TEST_END();

    TEST_EXIT();
}
