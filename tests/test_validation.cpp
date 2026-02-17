// Phase 1.4: Model validation warning tests.
// Verifies that loading models with unsupported features emits warnings
// to stderr rather than silently failing.

#include "test_utils.h"
#include "test_models.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <string>

// Capture stderr output during model loading
static std::string capture_stderr_load(const char* xml) {
    // Redirect stderr to a temp file
    char tmppath[] = "/tmp/test_validation_stderr_XXXXXX";
    int fd = mkstemp(tmppath);

    // Save original stderr
    int saved_stderr = dup(STDERR_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);

    // Load model (this should emit warnings to stderr)
    MjmlxModel* mh = mjmlx_load_model_from_string(xml);
    if (mh) mjmlx_free_model(mh);

    // Restore stderr
    fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    // Read captured output
    FILE* f = fopen(tmppath, "r");
    std::string result;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f)) result += buf;
        fclose(f);
    }
    unlink(tmppath);
    return result;
}

int main() {
    printf("=== test_validation: Phase 1.4 Model Validation Warnings ===\n\n");

    // ── Test 1: Clean model -- no warnings ──
    {
        static const char* CLEAN_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j" gear="50"/>
  </actuator>
</mujoco>
)";
        TEST_BEGIN("clean_model_no_warnings");
        auto warnings = capture_stderr_load(CLEAN_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.empty(), "clean model should produce no warnings");
        TEST_END();
    }

    // ── Test 2: Mesh geom warning ──
    {
        static const char* MESH_XML = R"(
<mujoco>
  <asset>
    <mesh name="teapot" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 2 3"/>
  </asset>
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom type="mesh" mesh="teapot" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("mesh_geom_no_warning");
        auto warnings = capture_stderr_load(MESH_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.find("MESH") == std::string::npos, "should NOT warn about MESH geoms (now supported)");
        TEST_END();
    }

    // ── Test 3: Box geom — no warning (box collision now supported in Phase 3.1) ──
    {
        static const char* BOX_XML = R"(
<mujoco>
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("box_geom_no_warning");
        auto warnings = capture_stderr_load(BOX_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.find("BOX") == std::string::npos, "box geoms should not produce warning");
        TEST_END();
    }

    // ── Test 4: Tendon warning ──
    {
        static const char* TENDON_XML = R"(
<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <site name="s1" pos="0.1 0 0"/>
    </body>
    <body name="b2" pos="0.5 0 1">
      <joint name="j2" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <site name="s2" pos="-0.1 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial>
      <site site="s1"/>
      <site site="s2"/>
    </spatial>
  </tendon>
</mujoco>
)";
        TEST_BEGIN("tendon_warning");
        auto warnings = capture_stderr_load(TENDON_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.find("tendon") != std::string::npos, "should warn about tendons");
        TEST_END();
    }

    // ── Test 5: Equality constraint warning ──
    {
        static const char* EQ_XML = R"(
<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="b2" pos="0.5 0 1">
      <joint name="j2" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2"/>
  </equality>
</mujoco>
)";
        TEST_BEGIN("equality_constraint_no_warning");
        auto warnings = capture_stderr_load(EQ_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.find("equality") == std::string::npos, "equality constraints now supported, no warning expected");
        TEST_END();
    }

    // ── Test 6: RK4 integrator warning ──
    {
        static const char* RK4_XML = R"(
<mujoco>
  <option integrator="RK4"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("rk4_integrator_warning");
        auto warnings = capture_stderr_load(RK4_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.find("RK4") != std::string::npos, "should warn about RK4 integrator");
        TEST_END();
    }

    // ── Test 7: Implicit integrator warning ──
    {
        static const char* IMPLICIT_XML = R"(
<mujoco>
  <option integrator="implicit"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("implicit_integrator_warning");
        auto warnings = capture_stderr_load(IMPLICIT_XML);
        printf("    warnings: '%s'\n", warnings.c_str());
        CHECK(warnings.find("implicit") != std::string::npos, "should warn about implicit integrator");
        TEST_END();
    }

    // ── Test 8: Model still loads despite warnings ──
    {
        static const char* COMPLEX_XML = R"(
<mujoco>
  <option integrator="RK4"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1" condim="3"/>
    </body>
  </worldbody>
</mujoco>
)";
        TEST_BEGIN("model_loads_despite_warnings");
        MjmlxModel* mh = mjmlx_load_model_from_string(COMPLEX_XML);
        CHECK(mh != nullptr, "model should still load with unsupported features");
        if (mh) {
            MjmlxData* dh = mjmlx_make_data(mh);
            CHECK(dh != nullptr, "data should be creatable");
            // Should be able to step without crashing
            mjmlx_step(mh, dh);
            int n = 0;
            const float* qpos = mjmlx_get_qpos(dh, &n);
            CHECK(qpos != nullptr, "qpos accessible after step");
            bool finite = true;
            for (int i = 0; i < n; i++) {
                if (std::isnan(qpos[i]) || std::isinf(qpos[i])) { finite = false; break; }
            }
            CHECK(finite, "qpos finite after step with unsupported features");
            mjmlx_free_data(dh);
            mjmlx_free_model(mh);
        }
        TEST_END();
    }

    TEST_EXIT();
}
