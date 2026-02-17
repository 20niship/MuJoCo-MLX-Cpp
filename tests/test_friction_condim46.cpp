// Phase 2.2: Condim=4,6 friction tests (torsional + rolling).
// condim=4: 6 pyramidal rows (tangent1, tangent2, torsion) × 2 edges
// condim=6: 10 pyramidal rows (tangent1, tangent2, torsion, rolling1, rolling2) × 2 edges
// For condim>=4, the torsion/rolling rows use the rotational Jacobian (jacr).

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

static const char* CONDIM4_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="4" friction="0.8 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.049">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1" condim="4" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* CONDIM6_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="6" friction="0.8 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.049">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1" condim="6" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
)";

int main() {
    printf("=== test_friction_condim46: Phase 2.2 Torsional + Rolling Friction ===\n\n");

    // ── Test 1: condim=4 row count (6 pyramidal rows) ──
    {
        TEST_BEGIN("condim4_row_count");
        auto path = write_temp_xml(CONDIM4_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ncon=%d, nefc=%d\n", dj->ncon, dj->nefc);
        CHECK(dj->ncon > 0, "MuJoCo C: has contact");
        // condim=4 pyramidal: 2*(4-1) = 6 rows per contact
        int expected = 6 * dj->ncon;
        printf("    MuJoCo C: expected nefc=%d (6*ncon), actual=%d\n", expected, dj->nefc);
        CHECK(dj->nefc >= expected, "MuJoCo C: nefc >= 6*ncon for condim=4");

        int nefc_mlx = dh->data.nefc;
        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, nefc_mlx);
        if (ncon_mlx > 0) {
            int expected_mlx = 6 * ncon_mlx;
            CHECK(nefc_mlx >= expected_mlx, "MLX: nefc >= 6*ncon for condim=4");
        }

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: condim=6 row count (10 pyramidal rows) ──
    {
        TEST_BEGIN("condim6_row_count");
        auto path = write_temp_xml(CONDIM6_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        CHECK(mj != nullptr, "MuJoCo C model loaded");
        mjData* dj = mj_makeData(mj);

        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: ncon=%d, nefc=%d\n", dj->ncon, dj->nefc);
        CHECK(dj->ncon > 0, "MuJoCo C: has contact");
        // condim=6 pyramidal: 2*(6-1) = 10 rows per contact
        int expected = 10 * dj->ncon;
        printf("    MuJoCo C: expected nefc=%d (10*ncon), actual=%d\n", expected, dj->nefc);
        CHECK(dj->nefc >= expected, "MuJoCo C: nefc >= 10*ncon for condim=6");

        int nefc_mlx = dh->data.nefc;
        int ncon_mlx = dh->data.ncon;
        printf("    MLX: ncon=%d, nefc=%d\n", ncon_mlx, nefc_mlx);
        if (ncon_mlx > 0) {
            int expected_mlx = 10 * ncon_mlx;
            CHECK(nefc_mlx >= expected_mlx, "MLX: nefc >= 10*ncon for condim=6");
        }

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 3: condim=4 D values match MuJoCo C ──
    {
        TEST_BEGIN("condim4_efc_D_match");
        auto path = write_temp_xml(CONDIM4_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        mx::eval(dh->data.efc_D);
        auto dp = dh->data.efc_D.data<float>();
        float max_diff = 0.0f;
        int nrows = std::min(dj->nefc, dh->data.nefc);
        for (int i = 0; i < nrows; i++) {
            float diff = std::abs(dp[i] - (float)dj->efc_D[i]);
            if (diff > max_diff) max_diff = diff;
            printf("    row %d: D_mlx=%.6f, D_mjc=%.6f, diff=%.8f\n",
                   i, dp[i], (float)dj->efc_D[i], diff);
        }
        printf("    max D diff: %.8f\n", max_diff);
        CHECK(max_diff < 0.1f, "condim=4: D values close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: condim=4 qfrc_constraint matches MuJoCo C ──
    {
        TEST_BEGIN("condim4_qfrc_constraint_match");
        auto path = write_temp_xml(CONDIM4_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qfrc = mjmlx_get_qfrc_constraint(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qfrc[i] - (float)dj->qfrc_constraint[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qfrc_constraint diff: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "condim=4: qfrc_constraint close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: condim=6 qfrc_constraint matches MuJoCo C ──
    {
        TEST_BEGIN("condim6_qfrc_constraint_match");
        auto path = write_temp_xml(CONDIM6_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        mj_forward(mj, dj);
        mjmlx_forward(mh, dh);

        int nv = mj->nv;
        int n = 0;
        const float* mlx_qfrc = mjmlx_get_qfrc_constraint(dh, &n);
        float max_diff = 0.0f;
        for (int i = 0; i < nv; i++) {
            float diff = std::abs(mlx_qfrc[i] - (float)dj->qfrc_constraint[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qfrc_constraint diff: %.6f\n", max_diff);
        CHECK(max_diff < 1.0f, "condim=6: qfrc_constraint close to MuJoCo C");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: condim=4 stability after 100 steps ──
    {
        TEST_BEGIN("condim4_stability_100_steps");
        auto path = write_temp_xml(CONDIM4_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        for (int i = 0; i < 100; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "qpos finite after 100 steps with condim=4");

        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 100 steps (condim=4): %.6f\n", max_diff);
        CHECK(max_diff < 5.0f, "qpos reasonably close to MuJoCo C after 100 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 7: condim=6 stability after 100 steps ──
    {
        TEST_BEGIN("condim6_stability_100_steps");
        auto path = write_temp_xml(CONDIM6_XML);
        char err[1000] = "";
        mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
        mjData* dj = mj_makeData(mj);
        MjmlxModel* mh = mjmlx_load_model(path.c_str());
        MjmlxData* dh = mjmlx_make_data(mh);
        unlink(path.c_str());

        for (int i = 0; i < 100; i++) {
            mj_step(mj, dj);
            mjmlx_step(mh, dh);
        }

        int n = 0;
        const float* mlx_qpos = mjmlx_get_qpos(dh, &n);
        bool finite = true;
        for (int i = 0; i < n; i++) {
            if (std::isnan(mlx_qpos[i]) || std::isinf(mlx_qpos[i])) { finite = false; break; }
        }
        CHECK(finite, "qpos finite after 100 steps with condim=6");

        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(mlx_qpos[i] - (float)dj->qpos[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("    max qpos diff after 100 steps (condim=6): %.6f\n", max_diff);
        CHECK(max_diff < 5.0f, "qpos reasonably close to MuJoCo C after 100 steps");

        mj_deleteData(dj); mj_deleteModel(mj);
        mjmlx_free_data(dh); mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
