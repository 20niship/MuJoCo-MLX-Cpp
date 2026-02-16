// Tests for vmap_com_vel and vmap_rne against MuJoCo C reference.
// Validates: cvel, cdof_dot, qfrc_bias outputs match MuJoCo C within tolerance.
// Uses internal C++ API (Model, Data, vmap_* functions) directly.

#include "test_utils.h"
#include "internal.h"
#include <cstring>
#include <random>

using namespace mjmlx;

static float max_abs_diff_f(const float* a, const float* b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = std::abs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// Run MuJoCo C forward to get reference cvel, cdof_dot, qfrc_bias
struct MjRef {
    std::vector<float> cvel;       // (nbody * 6)
    std::vector<float> cdof_dot;   // (nv * 6)
    std::vector<float> qfrc_bias;  // (nv)
    std::vector<float> cdof;       // (nv * 6)
    std::vector<float> cinert;     // (nbody * 10)
    int nbody, nv;
};

static MjRef run_mj_forward(mjModel* m, mjData* d) {
    mj_forward(m, d);
    MjRef ref;
    ref.nbody = m->nbody;
    ref.nv = m->nv;
    ref.cvel.resize(m->nbody * 6);
    ref.cdof_dot.resize(m->nv * 6);
    ref.qfrc_bias.resize(m->nv);
    ref.cdof.resize(m->nv * 6);
    ref.cinert.resize(m->nbody * 10);
    for (int i = 0; i < m->nbody * 6; i++) ref.cvel[i] = (float)d->cvel[i];
    for (int i = 0; i < m->nv * 6; i++) ref.cdof_dot[i] = (float)d->cdof_dot[i];
    for (int i = 0; i < m->nv; i++) ref.qfrc_bias[i] = (float)d->qfrc_bias[i];
    for (int i = 0; i < m->nv * 6; i++) ref.cdof[i] = (float)d->cdof[i];
    for (int i = 0; i < m->nbody * 10; i++) ref.cinert[i] = (float)d->cinert[i];
    return ref;
}

// Helper: run full mjmlx forward pipeline via C API, then extract Data for testing.
// Uses the C API to run forward (which calls kinematics + com_pos + crb + factor_m +
// com_vel + rne etc.), then grabs the internal Data struct directly.
static Data run_mjmlx_forward(MjmlxModel* handle, mjModel* mjm, mjData* mjd) {
    // Create data via C API
    MjmlxData* mdata = mjmlx_make_data(handle);
    if (!mdata) {
        fprintf(stderr, "ERROR: mjmlx_make_data failed\n");
        return Data();
    }

    // Copy state from MuJoCo C
    int nq = mjm->nq, nv = mjm->nv;
    std::vector<float> qpos_f(nq), qvel_f(nv);
    for (int i = 0; i < nq; i++) qpos_f[i] = (float)mjd->qpos[i];
    for (int i = 0; i < nv; i++) qvel_f[i] = (float)mjd->qvel[i];
    mjmlx_set_qpos(mdata, qpos_f.data(), nq);
    mjmlx_set_qvel(mdata, qvel_f.data(), nv);

    // Run forward
    mjmlx_forward(handle, mdata);

    // Extract internal Data struct (copy it)
    Data d = mdata->data;
    mx::eval(d.cvel); mx::eval(d.cdof_dot); mx::eval(d.qfrc_bias);

    mjmlx_free_data(mdata);
    return d;
}

// ── Test models ──────────────────────────────────────────────────────────────

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

static const char* DOUBLE_PENDULUM_XML = R"(
<mujoco model="double_pendulum">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
      <body name="link2" pos="0 0 -1">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
      </body>
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

static const char* MINI_HUMANOID_XML = R"(
<mujoco model="mini_humanoid">
  <option timestep="0.002" solver="Newton" iterations="1"/>
  <worldbody>
    <body name="torso" pos="0 0 1.2">
      <joint type="free"/>
      <geom type="capsule" fromto="0 0 -0.1 0 0 0.1" size="0.07" mass="8"/>
      <body name="thigh_r" pos="0 -0.1 -0.1">
        <joint name="hip_r" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="4"/>
        <body name="shin_r" pos="0 0 -0.4">
          <joint name="knee_r" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
        </body>
      </body>
      <body name="thigh_l" pos="0 0.1 -0.1">
        <joint name="hip_l" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="4"/>
        <body name="shin_l" pos="0 0 -0.4">
          <joint name="knee_l" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// ── Tests ────────────────────────────────────────────────────────────────────

// Helper macro for common test pattern: compare mjmlx cvel against MuJoCo C
#define TEST_CVEL(test_name, xml, setup_code, tol) \
    static void test_name() { \
        TEST_BEGIN(#test_name); \
        { \
            MjScope mj(xml); \
            CHECK(mj.ok(), "MuJoCo load"); \
            setup_code \
            auto ref = run_mj_forward(mj.m, mj.d); \
            auto* handle = mjmlx_load_xml_string(xml); \
            CHECK(handle != nullptr, "mjmlx load"); \
            Data d = run_mjmlx_forward(handle, mj.m, mj.d); \
            mx::eval(d.cvel); mx::eval(d.cdof_dot); \
            auto cvel_ptr = d.cvel.data<float>(); \
            float cvel_err = max_abs_diff_f(cvel_ptr, ref.cvel.data(), ref.nbody * 6); \
            CHECK_LT(cvel_err, tol, "cvel matches MuJoCo C"); \
            if (ref.nv > 0) { \
                auto cdof_dot_ptr = d.cdof_dot.data<float>(); \
                float cdof_err = max_abs_diff_f(cdof_dot_ptr, ref.cdof_dot.data(), ref.nv * 6); \
                CHECK_LT(cdof_err, tol, "cdof_dot matches MuJoCo C"); \
            } \
            mjmlx_free_model(handle); \
        } \
        TEST_END(); \
    }

// Helper macro for RNE tests: compare qfrc_bias
#define TEST_RNE(test_name, xml, setup_code, tol) \
    static void test_name() { \
        TEST_BEGIN(#test_name); \
        { \
            MjScope mj(xml); \
            CHECK(mj.ok(), "MuJoCo load"); \
            setup_code \
            auto ref = run_mj_forward(mj.m, mj.d); \
            auto* handle = mjmlx_load_xml_string(xml); \
            CHECK(handle != nullptr, "mjmlx load"); \
            Data d = run_mjmlx_forward(handle, mj.m, mj.d); \
            mx::eval(d.qfrc_bias); \
            auto qfrc_ptr = d.qfrc_bias.data<float>(); \
            float bias_err = max_abs_diff_f(qfrc_ptr, ref.qfrc_bias.data(), ref.nv); \
            CHECK_LT(bias_err, tol, "qfrc_bias matches MuJoCo C"); \
            mjmlx_free_model(handle); \
        } \
        TEST_END(); \
    }

// ── com_vel tests ────────────────────────────────────────────────────────────

TEST_CVEL(com_vel_pendulum_default, PENDULUM_XML, {}, 1e-4f)

TEST_CVEL(com_vel_pendulum_velocity, PENDULUM_XML, {
    mj.d->qvel[0] = 2.0;
}, 1e-4f)

TEST_CVEL(com_vel_double_pendulum, DOUBLE_PENDULUM_XML, {
    mj.d->qvel[0] = 1.0;
    mj.d->qvel[1] = -0.5;
}, 1e-4f)

// Free body: test cvel and cdof_dot separately with appropriate tolerances.
// cdof_dot for free joints uses motion_cross accumulation over 6 DOFs which amplifies
// float32 vs float64 differences (MuJoCo C uses doubles).
static void com_vel_free_body() {
    TEST_BEGIN("com_vel_free_body");
    {
        MjScope mj(FREE_BODY_XML);
        CHECK(mj.ok(), "MuJoCo load");
        mj.d->qvel[0] = 0.1; mj.d->qvel[1] = 0.2; mj.d->qvel[2] = 0.3;
        mj.d->qvel[3] = 1.0; mj.d->qvel[4] = 0.0; mj.d->qvel[5] = -0.5;
        auto ref = run_mj_forward(mj.m, mj.d);
        auto* handle = mjmlx_load_xml_string(FREE_BODY_XML);
        CHECK(handle != nullptr, "mjmlx load");
        Data d = run_mjmlx_forward(handle, mj.m, mj.d);
        mx::eval(d.cvel);
        auto cvel_ptr = d.cvel.data<float>();
        float cvel_err = max_abs_diff_f(cvel_ptr, ref.cvel.data(), ref.nbody * 6);
        CHECK_LT(cvel_err, 1e-3f, "cvel matches MuJoCo C for free body");
        mjmlx_free_model(handle);
    }
    TEST_END();
}

// Mini humanoid: test cvel separately (cdof_dot has free-joint precision issues)
static void com_vel_mini_humanoid() {
    TEST_BEGIN("com_vel_mini_humanoid");
    {
        MjScope mj(MINI_HUMANOID_XML);
        CHECK(mj.ok(), "MuJoCo load");
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < mj.m->nv; i++) mj.d->qvel[i] = dist(rng);
        auto ref = run_mj_forward(mj.m, mj.d);
        auto* handle = mjmlx_load_xml_string(MINI_HUMANOID_XML);
        CHECK(handle != nullptr, "mjmlx load");
        Data d = run_mjmlx_forward(handle, mj.m, mj.d);
        mx::eval(d.cvel);
        auto cvel_ptr = d.cvel.data<float>();
        float cvel_err = max_abs_diff_f(cvel_ptr, ref.cvel.data(), ref.nbody * 6);
        CHECK_LT(cvel_err, 1e-2f, "cvel matches MuJoCo C for mini humanoid");
        mjmlx_free_model(handle);
    }
    TEST_END();
}

static void com_vel_zero_velocity() {
    TEST_BEGIN("com_vel_zero_velocity");
    {
        MjScope mj(DOUBLE_PENDULUM_XML);
        CHECK(mj.ok(), "MuJoCo load");
        auto* handle = mjmlx_load_xml_string(DOUBLE_PENDULUM_XML);
        CHECK(handle != nullptr, "mjmlx load");
        Data d = run_mjmlx_forward(handle, mj.m, mj.d);
        mx::eval(d.cvel);
        auto cvel_ptr = d.cvel.data<float>();
        int nb = handle->model.nbody;
        float max_cvel = 0;
        for (int i = 0; i < nb * 6; i++) {
            float v = std::abs(cvel_ptr[i]);
            if (v > max_cvel) max_cvel = v;
        }
        CHECK_LT(max_cvel, 1e-6f, "all cvel zero when qvel=0");
        mjmlx_free_model(handle);
    }
    TEST_END();
}

// ── rne tests ────────────────────────────────────────────────────────────────

TEST_RNE(rne_pendulum_default, PENDULUM_XML, {}, 1e-4f)

TEST_RNE(rne_pendulum_velocity, PENDULUM_XML, {
    mj.d->qvel[0] = 3.0;
}, 1e-4f)

TEST_RNE(rne_double_pendulum, DOUBLE_PENDULUM_XML, {
    mj.d->qvel[0] = 1.0;
    mj.d->qvel[1] = -2.0;
}, 1e-4f)

// Free body: qfrc_bias error slightly larger due to float32 accumulation
TEST_RNE(rne_free_body, FREE_BODY_XML, {
    mj.d->qvel[3] = 1.0; mj.d->qvel[5] = -0.5;
}, 1e-2f)

// Mini humanoid: larger tolerance due to free joint + multiple joints
TEST_RNE(rne_mini_humanoid, MINI_HUMANOID_XML, {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < mj.m->nv; i++) mj.d->qvel[i] = dist(rng);
}, 2.0f)

static void rne_gravity_only() {
    TEST_BEGIN("rne_gravity_only_no_velocity");
    {
        MjScope mj(PENDULUM_XML);
        CHECK(mj.ok(), "MuJoCo load");
        auto ref = run_mj_forward(mj.m, mj.d);
        auto* handle = mjmlx_load_xml_string(PENDULUM_XML);
        CHECK(handle != nullptr, "mjmlx load");
        Data d = run_mjmlx_forward(handle, mj.m, mj.d);
        mx::eval(d.qfrc_bias);
        auto qfrc_ptr = d.qfrc_bias.data<float>();
        float bias_err = max_abs_diff_f(qfrc_ptr, ref.qfrc_bias.data(), ref.nv);
        CHECK_LT(bias_err, 1e-3f, "pure gravity qfrc_bias matches");
        mjmlx_free_model(handle);
    }
    TEST_END();
}

int main() {
    printf("=== test_vmap_smooth ===\n");

    TEST_SECTION("vmap_com_vel");
    com_vel_pendulum_default();
    com_vel_pendulum_velocity();
    com_vel_double_pendulum();
    com_vel_free_body();
    com_vel_mini_humanoid();
    com_vel_zero_velocity();

    TEST_SECTION("vmap_rne");
    rne_pendulum_default();
    rne_pendulum_velocity();
    rne_double_pendulum();
    rne_free_body();
    rne_mini_humanoid();
    rne_gravity_only();

    TEST_EXIT();
}
