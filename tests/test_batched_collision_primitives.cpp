// GPU (fast Metal/GLSL kernel) vs CPU parity for non-mesh geom pairs: the fast kernel used to only detect plane-mesh/mesh-mesh, so plane-capsule etc. (used by HalfCheetah and friends) silently produced zero contacts and fell through the floor.
#include "test_utils.h"
#include <cstring>
#include <vector>
#include <cmath>
#include <sstream>

namespace {

const char* PLANE_SPHERE_XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body pos="0 0 0.5"><freejoint/><geom type="sphere" size="0.08" mass="1"/></body>
</worldbody></mujoco>)";

const char* PLANE_CAPSULE_XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body pos="0.3 0 0.5"><freejoint/><geom type="capsule" size="0.05" fromto="-0.1 0 0 0.1 0 0" mass="1"/></body>
</worldbody></mujoco>)";

const char* PLANE_BOX_XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
</worldbody></mujoco>)";

const char* PLANE_CYLINDER_XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body pos="0 0 0.5"><freejoint/><geom type="cylinder" size="0.08 0.1" mass="1"/></body>
</worldbody></mujoco>)";

const char* CAPSULE_CAPSULE_XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body pos="0 0 0.1"><geom type="capsule" size="0.05" fromto="-0.2 0 0 0.2 0 0" mass="10"/></body>
  <body pos="0 0 0.4"><freejoint/><geom type="capsule" size="0.05" fromto="-0.15 0 0 0.15 0 0" mass="1"/></body>
</worldbody></mujoco>)";

const char* SPHERE_SPHERE_XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body pos="0 0 0.1"><geom type="sphere" size="0.1" mass="10"/></body>
  <body pos="0 0 0.45"><freejoint/><geom type="sphere" size="0.08" mass="1"/></body>
</worldbody></mujoco>)";

const char* mesh_asset() {
    return R"(<mesh name="cube" vertex="
      -0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05  0.05 -0.05 -0.05  0.05 -0.05
      -0.05 -0.05  0.05  0.05 -0.05  0.05  0.05  0.05  0.05 -0.05  0.05  0.05"/>)";
}

std::string mesh_pair_xml(const char* other_body) {
    std::ostringstream ss;
    ss << "<mujoco><option gravity=\"0 0 -9.81\" timestep=\"0.005\"/><asset>" << mesh_asset() << "</asset><worldbody>"
       << "<geom name=\"floor\" type=\"plane\" size=\"5 5 0.1\"/>"
       << "<body pos=\"0 0 0.05\"><freejoint/><geom type=\"mesh\" mesh=\"cube\" mass=\"1\"/></body>"
       << other_body << "</worldbody></mujoco>";
    return ss.str();
}

struct Fixture { const char* name; std::string xml; };

std::vector<Fixture> make_fixtures() {
    return {
        {"plane_sphere", PLANE_SPHERE_XML},
        {"plane_capsule", PLANE_CAPSULE_XML},
        {"plane_box", PLANE_BOX_XML},
        {"plane_cylinder", PLANE_CYLINDER_XML},
        {"capsule_capsule", CAPSULE_CAPSULE_XML},
        {"sphere_sphere", SPHERE_SPHERE_XML},
        {"mesh_sphere", mesh_pair_xml(R"(<body pos="0 0 0.35"><freejoint/><geom type="sphere" size="0.08" mass="1"/></body>)")},
        {"mesh_capsule", mesh_pair_xml(R"(<body pos="0 0 0.35"><freejoint/><geom type="capsule" size="0.05" fromto="-0.1 0 0 0.1 0 0" mass="1"/></body>)")},
        {"mesh_box", mesh_pair_xml(R"(<body pos="0 0 0.35"><freejoint/><geom type="box" size="0.08 0.08 0.08" mass="1"/></body>)")},
        {"mesh_mesh", mesh_pair_xml(R"(<body pos="0 0 0.2"><freejoint/><geom type="mesh" mesh="cube" mass="0.5"/></body>)")},
    };
}

} // namespace

int main() {
    printf("=== test_batched_collision_primitives ===\n");
    int total = 0, failed = 0;

    for (auto& fx : make_fixtures()) {
        TEST_SECTION(fx.name);
        auto path = write_temp_xml(fx.xml.c_str());
        auto* model = mjmlx_load_model(path.c_str());
        unlink(path.c_str());
        if (!model) { fprintf(stderr, "FAIL: %s failed to load\n", fx.name); failed++; total++; continue; }
        auto info = mjmlx_model_info(model);
        int nq = info.nq;

        MjmlxBatchedConfig cfg_cpu = {};
        cfg_cpu.num_envs = 1; cfg_cpu.use_gpu = 0; cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);

        MjmlxBatchedConfig cfg_gpu = {};
        cfg_gpu.num_envs = 1; cfg_gpu.use_gpu = 1; cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
        auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);

        const int N = 200;
        for (int s = 0; s < N; s++) { mjmlx_batched_step(sim_cpu, nullptr); mjmlx_batched_step(sim_gpu, nullptr); }

        int n;
        auto* cpu_qpos = mjmlx_batched_get_qpos(sim_cpu, &n);
        std::vector<float> cpu_copy(cpu_qpos, cpu_qpos + nq);
        auto* gpu_qpos = mjmlx_batched_get_qpos(sim_gpu, &n);

        TEST_BEGIN(fx.name);
        {
            CHECK_NO_NAN(gpu_qpos, nq, "GPU qpos finite after 200 steps");

            bool caught = true;
            for (int i = 0; i < nq; i++) {
                if (!std::isfinite(gpu_qpos[i]) || std::abs(gpu_qpos[i]) > 2.0f) { caught = false; break; }
            }
            CHECK(caught, "GPU: body stayed within |qpos| < 2.0 (contact caught it, no free-fall)");

            float max_diff = 0;
            for (int i = 0; i < nq; i++) max_diff = std::max(max_diff, std::abs(cpu_copy[i] - gpu_qpos[i]));
            // Loose bound: contact-heavy trajectories diverge chaotically (GPU CG vs CPU exact solver, same gap class as test_batched_diag); this only rules out one backend not colliding at all.
            CHECK_LT(max_diff, 1.5f, "GPU qpos roughly tracks CPU after 200 steps");
        }
        TEST_END();
        total++;

        mjmlx_batched_free(sim_cpu);
        mjmlx_batched_free(sim_gpu);
        mjmlx_free_model(model);
    }

    printf("\n=== %d fixtures tested ===\n", total);
    return failed > 0 ? 1 : 0;
}
