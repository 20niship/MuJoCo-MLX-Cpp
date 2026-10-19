// Go2のtrot gait風ctrlをCPU/GPU batched simに送り、qpos軌道をCSVに記録してCPU/GPU一致を診断する。
#include <mjmlx/mjmlx.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
// IsaacLabのUNITREE_GO2_CFG.init_state.joint_pos(FL,FR,RL,RRの順)
const float kDefaultJointPos[12] = {
    0.1f, 0.8f, -1.5f,  -0.1f, 0.8f, -1.5f,  0.1f, 1.0f, -1.5f,  -0.1f, 1.0f, -1.5f
};

// trot gait風: FL/RRが同位相、FR/RLが反対位相でthigh/calfを揺らす。hipは固定。
void make_ctrl(float t, float freq_hz, std::vector<float>& ctrl) {
    float phase = 2.0f * (float)M_PI * freq_hz * t;
    float s_fl_rr = std::sin(phase);
    float s_fr_rl = std::sin(phase + (float)M_PI);
    const float thigh_amp = 0.3f, calf_amp = 0.25f;
    float legs[4] = {s_fl_rr, s_fr_rl, s_fr_rl, s_fl_rr}; // FL,FR,RL,RR
    for (int leg = 0; leg < 4; leg++) {
        ctrl[leg*3+0] = kDefaultJointPos[leg*3+0];
        ctrl[leg*3+1] = kDefaultJointPos[leg*3+1] + thigh_amp * legs[leg];
        ctrl[leg*3+2] = kDefaultJointPos[leg*3+2] + calf_amp * legs[leg];
    }
}
} // namespace

int main(int argc, char** argv) {
    const char* xml_path = (argc > 1) ? argv[1]
        : "benchmarks/models/external/mujoco_menagerie/unitree_go2/scene_mjx.xml";
    int nsteps = (argc > 2) ? atoi(argv[2]) : 500;
    const char* out_prefix = (argc > 3) ? argv[3] : "go2_traj";

    auto* model = mjmlx_load_model(xml_path);
    if (!model) { fprintf(stderr, "load failed: %s\n", xml_path); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nb = info.nbody;
    printf("model: %s  nq=%d nv=%d nbody=%d nu=%d\n", xml_path, nq, nv, nb, info.nu);

    MjmlxBatchedConfig cfg_cpu = {};
    cfg_cpu.num_envs = 1; cfg_cpu.use_gpu = 0; cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
    auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);

    MjmlxBatchedConfig cfg_gpu = {};
    cfg_gpu.num_envs = 1; cfg_gpu.use_gpu = 1; cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
    auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);

    // base高さ0.4(IsaacLab基準)+ デフォルト関節角度でスポーンさせる
    {
        std::vector<float> qpos0(nq, 0.0f), qvel0(nv, 0.0f);
        qpos0[2] = 0.40f; qpos0[3] = 1.0f; // quat identity
        for (int i = 0; i < 12 && i + 7 < nq; i++) qpos0[7+i] = kDefaultJointPos[i];
        mjmlx_batched_set_state(sim_cpu, qpos0.data(), qvel0.data());
        mjmlx_batched_set_state(sim_gpu, qpos0.data(), qvel0.data());
    }

    char cpu_path[256], gpu_path[256];
    snprintf(cpu_path, sizeof(cpu_path), "%s_cpu.csv", out_prefix);
    snprintf(gpu_path, sizeof(gpu_path), "%s_gpu.csv", out_prefix);
    FILE* fcpu = fopen(cpu_path, "w");
    FILE* fgpu = fopen(gpu_path, "w");
    fprintf(fcpu, "step");
    fprintf(fgpu, "step");
    for (int i = 0; i < nq; i++) { fprintf(fcpu, ",qpos%d", i); fprintf(fgpu, ",qpos%d", i); }
    fprintf(fcpu, "\n"); fprintf(fgpu, "\n");

    float timestep = 0.002f; // go2_mjx.xmlのMuJoCoデフォルト
    std::vector<float> ctrl(info.nu, 0.0f);
    float qpos_max_diff_overall = 0;

    for (int s = 0; s < nsteps; s++) {
        float t = s * timestep;
        make_ctrl(t, /*freq_hz=*/1.5f, ctrl);

        mjmlx_batched_step(sim_cpu, ctrl.data());
        mjmlx_batched_step(sim_gpu, ctrl.data());

        int n;
        auto* cq = mjmlx_batched_get_qpos(sim_cpu, &n);
        auto* gq = mjmlx_batched_get_qpos(sim_gpu, &n);

        fprintf(fcpu, "%d", s);
        fprintf(fgpu, "%d", s);
        for (int i = 0; i < nq; i++) { fprintf(fcpu, ",%.6f", cq[i]); fprintf(fgpu, ",%.6f", gq[i]); }
        fprintf(fcpu, "\n"); fprintf(fgpu, "\n");

        float qpos_max_diff = 0; int qpos_argmax = -1;
        for (int i = 0; i < nq; i++) {
            float d = std::abs(cq[i] - gq[i]);
            if (d > qpos_max_diff) { qpos_max_diff = d; qpos_argmax = i; }
        }
        if (qpos_max_diff > qpos_max_diff_overall) qpos_max_diff_overall = qpos_max_diff;

        if (s % 50 == 0 || s == nsteps - 1) {
            printf("step %3d: qpos_max_diff=%.5f(idx%d)  CPU base_z=%.4f GPU base_z=%.4f\n",
                   s, qpos_max_diff, qpos_argmax, cq[2], gq[2]);
        }
    }
    fclose(fcpu);
    fclose(fgpu);

    printf("\nwrote %s / %s (%d steps)\n", cpu_path, gpu_path, nsteps);
    printf("overall max qpos diff across run: %.5f\n", qpos_max_diff_overall);

    mjmlx_batched_free(sim_cpu);
    mjmlx_batched_free(sim_gpu);
    return 0;
}
