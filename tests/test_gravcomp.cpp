// Phase 1.1: Gravity compensation correctness tests.
// Compares qfrc_gravcomp and qfrc_passive vs MuJoCo C reference.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

// Models designed to produce non-zero gravcomp forces.
// Key: bodies must have horizontal offset from joint axis to create moment arms.

// Partial gravcomp on an L-shaped arm (horizontal offset = nonzero torque)
static const char* GRAVCOMP_LARM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" fromto="0 0 0 0.4 0 0" mass="1"/>
      <body name="mid" pos="0.4 0 0" gravcomp="0.5">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05 0.2" fromto="0 0 0 0 0 -0.4" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="50"/>
    <motor joint="j1" gear="50"/>
  </actuator>
</mujoco>
)";

// Full gravcomp on a free-floating body (simplest: direct force opposition)
static const char* GRAVCOMP_FREE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="floater" pos="0 0 2" gravcomp="1.0">
      <freejoint name="root"/>
      <geom type="sphere" size="0.1" mass="5"/>
    </body>
  </worldbody>
</mujoco>
)";

// Full gravcomp on L-arm (all bodies compensated)
static const char* GRAVCOMP_FULL_LARM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="upper" pos="0 0 1" gravcomp="1.0">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" fromto="0 0 0 0.3 0 0" mass="2"/>
      <body name="lower" pos="0.3 0 0" gravcomp="1.0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05 0.15" fromto="0 0 0 0 0 -0.3" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Helper: run MuJoCo C forward
struct MjRef {
    mjModel* m = nullptr;
    mjData* d = nullptr;
    ~MjRef() { if (d) mj_deleteData(d); if (m) mj_deleteModel(m); }
};

static MjRef mj_ref_forward(const char* xml) {
    auto path = write_temp_xml(xml);
    char err[1000] = "";
    mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    unlink(path.c_str());
    if (!m) {
        fprintf(stderr, "MuJoCo C load failed: %s\n", err);
        return {};
    }
    mjData* d = mj_makeData(m);
    mj_forward(m, d);
    return {m, d};
}

// Helper: load + forward via C API, access internal Data
struct MlxResult {
    MjmlxModel* mhandle = nullptr;
    MjmlxData* dhandle = nullptr;
    ~MlxResult() {
        if (dhandle) mjmlx_free_data(dhandle);
        if (mhandle) mjmlx_free_model(mhandle);
    }
    const mjmlx::Model& model() const { return mhandle->model; }
    mjmlx::Data& data() { return dhandle->data; }
};

static MlxResult mlx_forward(const char* xml) {
    MjmlxModel* mh = mjmlx_load_model_from_string(xml);
    MjmlxData* dh = mjmlx_make_data(mh);
    mjmlx_forward(mh, dh);
    return {mh, dh};
}

int main() {
    printf("=== test_gravcomp: Phase 1.1 Gravity Compensation ===\n\n");

    // ── Test 1: No gravcomp -- qfrc_gravcomp should be all zeros ──
    {
        static const char* NO_GRAVCOMP_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("zero_gravcomp_produces_zero_qfrc");
        auto res = mlx_forward(NO_GRAVCOMP_XML);
        mx::eval(res.data().qfrc_gravcomp);
        auto gc_ptr = res.data().qfrc_gravcomp.data<float>();
        int nv = res.model().nv;
        for (int i = 0; i < nv; i++) {
            CHECK_CLOSE(gc_ptr[i], 0.0f, 1e-6f, "qfrc_gravcomp should be zero");
        }
        TEST_END();
    }

    // ── Test 2: Free body with gravcomp=1.0 ──
    // For a free joint, the translational Jacobian is identity at the COM,
    // so qfrc_gravcomp[0:3] should be -(mass * 1.0) * gravity
    {
        TEST_BEGIN("free_body_gravcomp_vs_mujoco_c");
        auto ref = mj_ref_forward(GRAVCOMP_FREE_XML);
        CHECK(ref.m != nullptr, "MuJoCo C model loaded");

        auto res = mlx_forward(GRAVCOMP_FREE_XML);
        mx::eval(res.data().qfrc_gravcomp);
        auto gc_mlx = res.data().qfrc_gravcomp.data<float>();
        int nv = res.model().nv;

        printf("    nv=%d\n", nv);
        printf("    MuJoCo C qfrc_gravcomp:");
        for (int i = 0; i < nv; i++) printf(" %.6f", ref.d->qfrc_gravcomp[i]);
        printf("\n    MLX      qfrc_gravcomp:");
        for (int i = 0; i < nv; i++) printf(" %.6f", gc_mlx[i]);
        printf("\n");

        // Check non-zero
        bool has_nonzero = false;
        for (int i = 0; i < nv; i++) {
            if (std::abs(ref.d->qfrc_gravcomp[i]) > 1e-6) has_nonzero = true;
        }
        CHECK(has_nonzero, "MuJoCo C should produce non-zero gravcomp for free body");

        float tol = 0.05f;
        for (int i = 0; i < nv; i++) {
            float expected = (float)ref.d->qfrc_gravcomp[i];
            CHECK_CLOSE(gc_mlx[i], expected, tol, "free body qfrc_gravcomp[i]");
        }
        TEST_END();
    }

    // ── Test 3: L-arm with partial gravcomp (0.5) -- compare vs MuJoCo C ──
    {
        TEST_BEGIN("larm_partial_gravcomp_vs_mujoco_c");
        auto ref = mj_ref_forward(GRAVCOMP_LARM_XML);
        CHECK(ref.m != nullptr, "MuJoCo C model loaded");

        auto res = mlx_forward(GRAVCOMP_LARM_XML);
        mx::eval(res.data().qfrc_gravcomp);
        auto gc_mlx = res.data().qfrc_gravcomp.data<float>();
        int nv = res.model().nv;

        printf("    nv=%d, ngravcomp=%d\n", nv, res.model().ngravcomp);
        printf("    MuJoCo C qfrc_gravcomp:");
        for (int i = 0; i < nv; i++) printf(" %.6f", ref.d->qfrc_gravcomp[i]);
        printf("\n    MLX      qfrc_gravcomp:");
        for (int i = 0; i < nv; i++) printf(" %.6f", gc_mlx[i]);
        printf("\n");

        float tol = 0.05f;
        for (int i = 0; i < nv; i++) {
            float expected = (float)ref.d->qfrc_gravcomp[i];
            CHECK_CLOSE(gc_mlx[i], expected, tol, "larm qfrc_gravcomp[i]");
        }

        // Compare qfrc_passive
        mx::eval(res.data().qfrc_passive);
        auto passive_mlx = res.data().qfrc_passive.data<float>();
        printf("    MuJoCo C qfrc_passive: ");
        for (int i = 0; i < nv; i++) printf(" %.6f", ref.d->qfrc_passive[i]);
        printf("\n    MLX      qfrc_passive: ");
        for (int i = 0; i < nv; i++) printf(" %.6f", passive_mlx[i]);
        printf("\n");

        for (int i = 0; i < nv; i++) {
            float expected = (float)ref.d->qfrc_passive[i];
            CHECK_CLOSE(passive_mlx[i], expected, tol, "larm qfrc_passive[i]");
        }
        TEST_END();
    }

    // ── Test 4: Full gravcomp L-arm -- compare vs MuJoCo C ──
    {
        TEST_BEGIN("larm_full_gravcomp_vs_mujoco_c");
        auto ref = mj_ref_forward(GRAVCOMP_FULL_LARM_XML);
        CHECK(ref.m != nullptr, "MuJoCo C model loaded");

        auto res = mlx_forward(GRAVCOMP_FULL_LARM_XML);
        mx::eval(res.data().qfrc_gravcomp);
        auto gc_mlx = res.data().qfrc_gravcomp.data<float>();
        int nv = res.model().nv;

        printf("    nv=%d\n", nv);
        printf("    MuJoCo C qfrc_gravcomp:");
        for (int i = 0; i < nv; i++) printf(" %.6f", ref.d->qfrc_gravcomp[i]);
        printf("\n    MLX      qfrc_gravcomp:");
        for (int i = 0; i < nv; i++) printf(" %.6f", gc_mlx[i]);
        printf("\n");

        float tol = 0.05f;
        for (int i = 0; i < nv; i++) {
            float expected = (float)ref.d->qfrc_gravcomp[i];
            CHECK_CLOSE(gc_mlx[i], expected, tol, "full larm qfrc_gravcomp[i]");
        }
        TEST_END();
    }

    // ── Test 5: C API getter for qfrc_gravcomp ──
    {
        TEST_BEGIN("c_api_qfrc_gravcomp_getter");
        MjmlxModel* mhandle = mjmlx_load_model_from_string(GRAVCOMP_FREE_XML);
        CHECK(mhandle != nullptr, "model loaded via C API");

        MjmlxData* dhandle = mjmlx_make_data(mhandle);
        CHECK(dhandle != nullptr, "data created via C API");

        mjmlx_forward(mhandle, dhandle);

        int n = 0;
        const float* gc = mjmlx_get_qfrc_gravcomp(dhandle, &n);
        CHECK(gc != nullptr, "qfrc_gravcomp pointer non-null");
        CHECK(n > 0, "qfrc_gravcomp size > 0");

        bool has_nonzero = false;
        for (int i = 0; i < n; i++) {
            if (std::abs(gc[i]) > 1e-8f) has_nonzero = true;
        }
        CHECK(has_nonzero, "C API qfrc_gravcomp has non-zero entries");

        mjmlx_free_data(dhandle);
        mjmlx_free_model(mhandle);
        TEST_END();
    }

    // ── Test 6: Step stability with gravcomp ──
    {
        TEST_BEGIN("gravcomp_step_stability");
        MjmlxModel* mhandle = mjmlx_load_model_from_string(GRAVCOMP_FREE_XML);
        CHECK(mhandle != nullptr, "model loaded");
        MjmlxData* dhandle = mjmlx_make_data(mhandle);

        for (int i = 0; i < 100; i++) {
            mjmlx_step(mhandle, dhandle);
        }

        int n = 0;
        const float* qpos = mjmlx_get_qpos(dhandle, &n);
        CHECK(qpos != nullptr, "qpos non-null after 100 steps");
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(qpos[i]) || std::isinf(qpos[i])) {
                finite = false;
                break;
            }
        }
        CHECK(finite, "qpos finite after 100 steps with gravcomp");

        // With full gravcomp, the Z translation should barely change
        // (gravity is fully compensated, so no net force in Z)
        const float* qpos_ptr = mjmlx_get_qpos(dhandle, &n);
        float z_pos = qpos_ptr[2]; // free joint: qpos = [x, y, z, qw, qx, qy, qz]
        printf("    Z position after 100 steps: %.6f (initial: 2.0)\n", z_pos);
        CHECK_CLOSE(z_pos, 2.0f, 0.1f, "Z should stay near initial with full gravcomp");

        mjmlx_free_data(dhandle);
        mjmlx_free_model(mhandle);
        TEST_END();
    }

    // ── Test 7: Vertical chain (zero torque at initial config) ──
    // Verify the original test models produce zero at initial config (this is correct physics)
    {
        TEST_BEGIN("vertical_chain_zero_at_initial_config");
        auto ref = mj_ref_forward(GRAVCOMP_XML);
        CHECK(ref.m != nullptr, "MuJoCo C model loaded");

        // MuJoCo C should also show zero for vertical chain at initial config
        bool mj_all_zero = true;
        for (int i = 0; i < ref.m->nv; i++) {
            if (std::abs(ref.d->qfrc_gravcomp[i]) > 1e-6) mj_all_zero = false;
        }
        CHECK(mj_all_zero, "MuJoCo C should show zero gravcomp for vertical chain at init");

        auto res = mlx_forward(GRAVCOMP_XML);
        mx::eval(res.data().qfrc_gravcomp);
        auto gc_ptr = res.data().qfrc_gravcomp.data<float>();
        int nv = res.model().nv;

        bool mlx_all_zero = true;
        for (int i = 0; i < nv; i++) {
            if (std::abs(gc_ptr[i]) > 1e-6f) mlx_all_zero = false;
        }
        CHECK(mlx_all_zero, "MLX should also show zero gravcomp for vertical chain");
        TEST_END();
    }

    TEST_EXIT();
}
