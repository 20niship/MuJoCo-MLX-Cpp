// Phase 1.3: Exclude signature (contact exclusion) tests.
// Verifies that <exclude body1="X" body2="Y"/> directives correctly
// reduce the number of collision pairs and contacts.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>

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
    if (!m) return {};
    mjData* d = mj_makeData(m);
    mj_forward(m, d);
    return {m, d};
}

// Same model as EXCLUDE_XML but without <contact> section
static const char* NO_EXCLUDE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 0.5">
      <freejoint/>
      <geom name="torso_g" type="capsule" size="0.1 0.15" mass="5"/>
      <body name="upper_arm" pos="0.2 0 0.15">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom name="ua_g" type="capsule" size="0.05 0.15" mass="1"/>
        <body name="lower_arm" pos="0 0 -0.3">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom name="la_g" type="capsule" size="0.04 0.12" mass="0.5"/>
        </body>
      </body>
      <body name="upper_leg" pos="0 0 -0.15">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom name="ul_g" type="capsule" size="0.06 0.2" mass="2"/>
        <body name="lower_leg" pos="0 0 -0.4">
          <joint name="knee" type="hinge" axis="0 1 0"/>
          <geom name="ll_g" type="capsule" size="0.05 0.15" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="50"/>
    <motor joint="elbow" gear="30"/>
    <motor joint="hip" gear="100"/>
    <motor joint="knee" gear="50"/>
  </actuator>
</mujoco>
)";

int main() {
    printf("=== test_exclude: Phase 1.3 Exclude Signature ===\n\n");

    // ── Test 1: Model without excludes -- count collision pairs ──
    {
        TEST_BEGIN("no_exclude_collision_pair_count");
        // forward() triggers init_cache() internally
        MjmlxModel* mh = mjmlx_load_model_from_string(NO_EXCLUDE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);
        mjmlx_forward(mh, dh);

        int no_exclude_pairs = mh->model.cache.max_ncon;
        printf("    collision pairs without excludes: %d\n", no_exclude_pairs);
        printf("    nexclude: %d\n", mh->model.nexclude);
        CHECK(mh->model.nexclude == 0, "nexclude should be 0");
        CHECK(no_exclude_pairs > 0, "should have collision pairs");

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 2: Excludes reduce collision pairs (non parent-child bodies) ──
    {
        // Model with 3 free bodies that would all collide
        static const char* THREE_BALLS_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="A" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
    <body name="B" pos="0.15 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
    <body name="C" pos="0.3 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
  </worldbody>
</mujoco>
)";
        static const char* THREE_BALLS_EXCLUDE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="A" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
    <body name="B" pos="0.15 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
    <body name="C" pos="0.3 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
  </worldbody>
  <contact>
    <exclude body1="A" body2="B"/>
  </contact>
</mujoco>
)";
        TEST_BEGIN("exclude_reduces_collision_pairs");
        MjmlxModel* no_ex_mh = mjmlx_load_model_from_string(THREE_BALLS_XML);
        MjmlxData* no_ex_dh = mjmlx_make_data(no_ex_mh);
        mjmlx_forward(no_ex_mh, no_ex_dh);
        int no_ex_pairs = no_ex_mh->model.cache.max_ncon;

        MjmlxModel* ex_mh = mjmlx_load_model_from_string(THREE_BALLS_EXCLUDE_XML);
        MjmlxData* ex_dh = mjmlx_make_data(ex_mh);
        mjmlx_forward(ex_mh, ex_dh);
        int ex_pairs = ex_mh->model.cache.max_ncon;

        printf("    Without excludes: %d collision pairs\n", no_ex_pairs);
        printf("    With excludes:    %d collision pairs\n", ex_pairs);
        printf("    nexclude: %d\n", ex_mh->model.nexclude);

        CHECK(ex_mh->model.nexclude == 1, "nexclude should be 1");
        CHECK(ex_pairs < no_ex_pairs, "excludes should reduce collision pair count");
        CHECK(ex_pairs == no_ex_pairs - 1, "should remove exactly 1 pair");

        mjmlx_free_data(no_ex_dh);
        mjmlx_free_model(no_ex_mh);
        mjmlx_free_data(ex_dh);
        mjmlx_free_model(ex_mh);
        TEST_END();
    }

    // ── Test 3: Compare ncon vs MuJoCo C ──
    {
        TEST_BEGIN("exclude_ncon_matches_mujoco_c");
        auto ref = mj_ref_forward(EXCLUDE_XML);
        CHECK(ref.m != nullptr, "MuJoCo C model loaded");

        MjmlxModel* mh = mjmlx_load_model_from_string(EXCLUDE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);
        mjmlx_forward(mh, dh);

        printf("    MuJoCo C: nexclude=%lld, ncon=%d\n", (long long)ref.m->nexclude, ref.d->ncon);
        printf("    MLX:      nexclude=%d, ncon=%d\n", mh->model.nexclude, dh->data.ncon);

        CHECK(ref.d->ncon == dh->data.ncon,
              "ncon should match MuJoCo C with excludes");

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 4: Excluded pairs produce 0 contacts ──
    {
        static const char* COLLIDING_EXCLUDE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="A" pos="0 0 0.5">
      <freejoint/>
      <geom name="gA" type="sphere" size="0.2" mass="1"/>
    </body>
    <body name="B" pos="0 0 0.5">
      <freejoint/>
      <geom name="gB" type="sphere" size="0.2" mass="1"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="A" body2="B"/>
  </contact>
</mujoco>
)";
        TEST_BEGIN("excluded_bodies_no_contact");
        MjmlxModel* mh = mjmlx_load_model_from_string(COLLIDING_EXCLUDE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);
        mjmlx_forward(mh, dh);

        printf("    Overlapping excluded bodies: ncon=%d\n", dh->data.ncon);
        CHECK(dh->data.ncon == 0, "excluded overlapping bodies should have 0 contacts");

        auto ref = mj_ref_forward(COLLIDING_EXCLUDE_XML);
        printf("    MuJoCo C ncon=%d\n", ref.d->ncon);

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 5: Same model without exclude -> contacts present ──
    {
        static const char* COLLIDING_NO_EXCLUDE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="A" pos="0 0 0.5">
      <freejoint/>
      <geom name="gA" type="sphere" size="0.2" mass="1"/>
    </body>
    <body name="B" pos="0 0 0.5">
      <freejoint/>
      <geom name="gB" type="sphere" size="0.2" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("non_excluded_overlapping_bodies_contact");
        MjmlxModel* mh = mjmlx_load_model_from_string(COLLIDING_NO_EXCLUDE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);
        mjmlx_forward(mh, dh);

        printf("    Overlapping non-excluded bodies: ncon=%d\n", dh->data.ncon);
        CHECK(dh->data.ncon > 0, "overlapping bodies without exclude should have contacts");

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    // ── Test 6: exclude_signature values match MuJoCo C ──
    {
        TEST_BEGIN("exclude_signature_values");
        MjmlxModel* mh = mjmlx_load_model_from_string(EXCLUDE_XML);
        MjmlxData* dh = mjmlx_make_data(mh);
        mjmlx_forward(mh, dh);

        mx::eval(mh->model.exclude_signature);
        auto sig_ptr = mh->model.exclude_signature.data<int>();
        int nexclude = mh->model.nexclude;

        printf("    nexclude=%d, signatures:", nexclude);
        for (int i = 0; i < nexclude; i++) {
            int b1 = sig_ptr[i] & 0xFFFF;
            int b2 = sig_ptr[i] >> 16;
            printf(" (%d,%d)", b1, b2);
        }
        printf("\n");

        CHECK(nexclude == 4, "nexclude should be 4");

        auto ref = mj_ref_forward(EXCLUDE_XML);
        for (int i = 0; i < nexclude; i++) {
            CHECK(sig_ptr[i] == ref.m->exclude_signature[i],
                  "exclude_signature should match MuJoCo C");
        }

        mjmlx_free_data(dh);
        mjmlx_free_model(mh);
        TEST_END();
    }

    TEST_EXIT();
}
