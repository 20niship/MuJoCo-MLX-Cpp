// Go2モデルでCPU batched simとGPU(Vulkan/MKX) batched simの軌道一致を確認する診断プログラム。
#include <mjmlx/mjmlx.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    const char* xml_path = (argc > 1) ? argv[1]
        : "benchmarks/models/external/mujoco_menagerie/unitree_go2/scene_mjx.xml";
    int nsteps = (argc > 2) ? atoi(argv[2]) : 100;

    auto* model = mjmlx_load_model(xml_path);
    if (!model) { fprintf(stderr, "load failed: %s\n", xml_path); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nb = info.nbody;
    printf("model: %s  nq=%d nv=%d nbody=%d\n", xml_path, nq, nv, nb);

    MjmlxBatchedConfig cfg_cpu = {};
    cfg_cpu.num_envs = 1; cfg_cpu.use_gpu = 0; cfg_cpu.integrator = MJMLX_INTEGRATOR_EULER;
    auto* sim_cpu = mjmlx_batched_create(model, &cfg_cpu);

    MjmlxBatchedConfig cfg_gpu = {};
    cfg_gpu.num_envs = 1; cfg_gpu.use_gpu = 1; cfg_gpu.integrator = MJMLX_INTEGRATOR_EULER;
    auto* sim_gpu = mjmlx_batched_create(model, &cfg_gpu);

    std::vector<float> ctrl(info.nu, 0.0f); // 無操作(重力+接触のみ)でも発散有無が分かる

    for (int s = 0; s < nsteps; s++) {
        mjmlx_batched_step(sim_cpu, ctrl.data());
        mjmlx_batched_step(sim_gpu, ctrl.data());

        int n;
        auto* cq = mjmlx_batched_get_qpos(sim_cpu, &n);
        auto* gq = mjmlx_batched_get_qpos(sim_gpu, &n);
        auto* cx = mjmlx_batched_get_xpos(sim_cpu, &n);
        auto* gx = mjmlx_batched_get_xpos(sim_gpu, &n);

        float qpos_max_diff = 0; int qpos_argmax = -1;
        for (int i = 0; i < nq; i++) {
            float d = std::abs(cq[i] - gq[i]);
            if (d > qpos_max_diff) { qpos_max_diff = d; qpos_argmax = i; }
        }
        float xpos_max_diff = 0; int xpos_argmax = -1;
        for (int i = 0; i < nb * 3; i++) {
            float d = std::abs(cx[i] - gx[i]);
            if (d > xpos_max_diff) { xpos_max_diff = d; xpos_argmax = i / 3; }
        }

        if (s % 10 == 0 || s == nsteps - 1) {
            printf("step %3d: qpos_max_diff=%.5f(idx%d) xpos_max_diff=%.5f(body%d)  "
                   "CPU base_z=%.4f GPU base_z=%.4f\n",
                   s, qpos_max_diff, qpos_argmax, xpos_max_diff, xpos_argmax, cq[2], gq[2]);
        }
    }

    printf("\n--- final joint angles (qpos[7..%d], CPU vs GPU) ---\n", nq - 1);
    {
        int n;
        auto* cq = mjmlx_batched_get_qpos(sim_cpu, &n);
        auto* gq = mjmlx_batched_get_qpos(sim_gpu, &n);
        for (int i = 7; i < nq; i++) {
            printf("  joint[%2d]: CPU=%.5f GPU=%.5f diff=%.5f\n", i - 7, cq[i], gq[i], std::abs(cq[i] - gq[i]));
        }
    }

    mjmlx_batched_free(sim_cpu);
    mjmlx_batched_free(sim_gpu);
    return 0;
}
