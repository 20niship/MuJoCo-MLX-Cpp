// GPUバッチシミュレーションをカメラセンサ(Vulkan)で描き、フレームをPPMで書き出す(動作確認動画用)。
// Usage: camera_dump <model.xml> <out_dir> [frames=60] [steps_per_frame=10] [W=320] [H=240]
// 制御は全actuatorにsin。各フレームのqposをout_dir/qpos.csvへ書く(MuJoCo標準レンダラで同じ姿勢を描く比較用)。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mjmlx/mjmlx.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.xml> <out_dir> [frames] [steps_per_frame] [W] [H]\n", argv[0]); return 1; }
    const int frames = argc > 3 ? atoi(argv[3]) : 60, per = argc > 4 ? atoi(argv[4]) : 10;
    const int W = argc > 5 ? atoi(argv[5]) : 320, H = argc > 6 ? atoi(argv[6]) : 240;
    const std::string out = argv[2];
    MjmlxModel* model = mjmlx_load_model(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto info = mjmlx_model_info(model);
    MjmlxBatchedConfig cfg{};
    cfg.num_envs = 1; cfg.integrator = MJMLX_INTEGRATOR_EULER; cfg.use_gpu = 1;
    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &cfg);
    MjmlxCameraRenderer* cam = mjmlx_camera_create(sim, W, H);
    if (!cam) { fprintf(stderr, "camera create failed\n"); return 2; }
    std::vector<uint8_t> rgb((size_t)W * H * 3);
    std::vector<float> ctrl(info.nu), q(info.nq), v(info.nv);
    std::string cmd = "mkdir -p " + out;
    if (system(cmd.c_str()) != 0) return 1;
    FILE* qf = fopen((out + "/qpos.csv").c_str(), "w");
    for (int f = 0; f < frames; f++) {
        for (int s = 0; s < per; s++) {
            double t = (f * per + s) * 0.002;
            for (int i = 0; i < info.nu; i++) ctrl[i] = (float)std::sin(2.0 * t * 3.0 + i);
            mjmlx_batched_step(sim, ctrl.data());
        }
        if (mjmlx_camera_render(cam) != 0 || mjmlx_camera_get_rgb(cam, rgb.data()) != 0) return 3;
        char name[64];
        snprintf(name, sizeof name, "/f%04d.ppm", f);
        FILE* fp = fopen((out + name).c_str(), "wb");
        fprintf(fp, "P6\n%d %d\n255\n", W, H);
        fwrite(rgb.data(), 1, rgb.size(), fp);
        fclose(fp);
        int a, b;
        mjmlx_batched_get_state(sim, q.data(), v.data(), &a, &b);
        for (int i = 0; i < info.nq; i++) fprintf(qf, i ? ",%.9g" : "%.9g", q[i]);
        fprintf(qf, "\n");
    }
    fclose(qf);
    mjmlx_camera_free(cam);
    mjmlx_batched_free(sim);
    mjmlx_free_model(model);
    return 0;
}
