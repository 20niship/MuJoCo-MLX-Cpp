// カメラセンサ(Vulkan)の描画時間: B env x 全camera を iters 回描き、1回あたりms(render + 読み戻し)を出す。
// Usage: bench_camera <model.xml> [B=64] [W=64] [H=64] [iters=20]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mjmlx/mjmlx.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.xml> [B] [W] [H] [iters]\n", argv[0]); return 1; }
    const int B = argc > 2 ? atoi(argv[2]) : 64, W = argc > 3 ? atoi(argv[3]) : 64, H = argc > 4 ? atoi(argv[4]) : 64, iters = argc > 5 ? atoi(argv[5]) : 20;
    MjmlxModel* model = mjmlx_load_model(argv[1]);
    if (!model) return 1;
    MjmlxBatchedConfig cfg{};
    cfg.num_envs = B; cfg.integrator = MJMLX_INTEGRATOR_EULER; cfg.use_gpu = 1;
    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &cfg);
    MjmlxCameraRenderer* cam = mjmlx_camera_create(sim, W, H);
    if (!cam) { fprintf(stderr, "camera create failed\n"); return 2; }
    const int nc = mjmlx_camera_count(cam);
    std::vector<uint8_t> rgb((size_t)B * nc * W * H * 3);
    mjmlx_camera_render(cam); mjmlx_camera_get_rgb(cam, rgb.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) { mjmlx_camera_render(cam); mjmlx_camera_get_rgb(cam, rgb.data()); }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    printf("B=%d cams=%d %dx%d: %.2f ms/render (render+readback)\n", B, nc, W, H, ms);
    mjmlx_camera_free(cam); mjmlx_batched_free(sim); mjmlx_free_model(model);
    return 0;
}
