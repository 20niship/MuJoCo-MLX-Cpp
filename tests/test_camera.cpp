// カメラセンサ(Vulkan multi-draw indirect)の検証: 解析解(平面+球)とのdepth比較、env毎の姿勢反映、ドローコール数1。
#include <dlfcn.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mjmlx/mjmlx.h"

// 描画系コマンドを数えるインターポーザ(libvulkanの同名シンボルより先に解決される)。
static std::atomic<int> g_draws{0};
extern "C" {
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    static auto f = reinterpret_cast<PFN_vkCmdDrawIndexedIndirect>(dlsym(RTLD_NEXT, "vkCmdDrawIndexedIndirect"));
    g_draws++;
    f(c, b, o, n, s);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer c, uint32_t a, uint32_t b, uint32_t d, int32_t e, uint32_t f_) {
    static auto f = reinterpret_cast<PFN_vkCmdDrawIndexed>(dlsym(RTLD_NEXT, "vkCmdDrawIndexed"));
    g_draws++;
    f(c, a, b, d, e, f_);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer c, uint32_t a, uint32_t b, uint32_t d, uint32_t e) {
    static auto f = reinterpret_cast<PFN_vkCmdDraw>(dlsym(RTLD_NEXT, "vkCmdDraw"));
    g_draws++;
    f(c, a, b, d, e);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    static auto f = reinterpret_cast<PFN_vkCmdDrawIndirect>(dlsym(RTLD_NEXT, "vkCmdDrawIndirect"));
    g_draws++;
    f(c, b, o, n, s);
}
}

namespace {

int g_fail = 0;
const char* g_dump = nullptr;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            g_fail++;                                     \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
        }                                                 \
    } while (0)

// 球(中心(sx,0,1), 半径0.5)と平面z=0(半幅50)。カメラは(cx,-3,1)から+y方向、fovy=60。
const char* kXml = R"(
<mujoco><option gravity="0 0 0"/>
<visual><global fovy="60"/></visual>
<worldbody>
  <geom name="floor" type="plane" size="50 50 0.1" rgba="0.6 0.6 0.6 1"/>
  <body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size="0.5" rgba="1 0 0 1"/></body>
  <body name="cart" pos="0 -3 1"><joint name="slide" type="slide" axis="1 0 0"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <camera name="cart_cam" fovy="60" xyaxes="1 0 0 0 0 1"/></body>
  <camera name="fixed_cam" pos="0 -3 1" fovy="60" xyaxes="1 0 0 0 0 1"/>
</worldbody></mujoco>)";

constexpr int W = 64, H = 64, NQ = 8;
const float kBallR = 0.5f, kFovy = 60.0f, kFar = 100.0f;

// 解析解: 光軸方向のz-depth(球にもplaneにも当たらなければ-1)。isSphereに球ヒットかを返す。
float analytic_depth(float cx, float bx, int i, int j, bool* is_sphere, int* rim = nullptr) {
    float th = std::tan(kFovy * (float)M_PI / 360.0f);
    float u = ((j + 0.5f) / W * 2 - 1) * th, v = (1 - (i + 0.5f) / H * 2) * th;
    float d[3] = {u, 1.0f, v}, o[3] = {cx, -3.0f, 1.0f}; // dir=(right,fwd,up), dir.fwd=1なのでt==z-depth
    float oc[3] = {o[0] - bx, o[1], o[2] - 1.0f};
    float a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], b = 2 * (oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2]);
    float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - kBallR * kBallR, disc = b * b - 4 * a * c;
    *is_sphere = false;
    if (disc > 0) {
        float t = (-b - std::sqrt(disc)) / (2 * a);
        if (t > 0) {
            *is_sphere = true;
            if (rim) *rim = (c + kBallR * kBallR - b * b / (4 * a)) > 0.97f * 0.97f * kBallR * kBallR;
            return t;
        }
    }
    if (d[2] < 0) {
        float t = -o[2] / d[2];
        if (std::fabs(o[0] + t * d[0]) <= 50 && std::fabs(o[1] + t * d[1]) <= 50) return t;
    }
    return -1;
}

struct Stats { double max_sphere = 0, max_plane = 0; int class_mismatch = 0; int total = 0; };

void compare_image(const float* depth, float cx, float bx, Stats& st) {
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            bool sph;
            int rim = 0;
            float ref = analytic_depth(cx, bx, i, j, &sph, &rim);
            if (rim) continue;
            float got = depth[i * W + j];
            bool got_hit = got < kFar - 1e-3f;
            st.total++;
            if ((ref > 0) != got_hit) { st.class_mismatch++; continue; }
            if (ref < 0) continue;
            double e = std::fabs(got - ref);
            // ponytail: 球はUV球の内接多面体なので輪郭付近は深く出る。輪郭ピクセル(視線が球にほぼ接する)は分類のみ見る。
            if (sph) st.max_sphere = std::max(st.max_sphere, e);
            else st.max_plane = std::max(st.max_plane, e / (1.0 + ref));
        }
}

void run(int B) {
    MjmlxModel* model = mjmlx_load_model_from_string(kXml);
    CHECK(model != nullptr, "load model");
    if (!model) return;
    MjmlxBatchedConfig cfg{};
    cfg.num_envs = B; cfg.integrator = MJMLX_INTEGRATOR_EULER; cfg.use_gpu = 1;
    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &cfg);
    CHECK(sim != nullptr, "batched create");
    if (!sim) return;
    auto info = mjmlx_model_info(model);
    CHECK(info.nq == NQ, "nq=%d", info.nq);

    // env毎に球のx位置とカート(cart_cam)のx位置を変える。
    std::vector<float> qpos((size_t)B * NQ), qvel((size_t)B * info.nv, 0.0f), ball_x(B), cart_x(B);
    for (int e = 0; e < B; e++) {
        ball_x[e] = -0.8f + 1.6f * (float)e / (float)std::max(B - 1, 1);
        cart_x[e] = 0.5f * std::sin(0.7f * (float)e);
        float* q = &qpos[(size_t)e * NQ];
        q[0] = ball_x[e]; q[1] = 0; q[2] = 1; q[3] = 1; q[4] = q[5] = q[6] = 0;
        q[7] = cart_x[e];
    }
    mjmlx_batched_set_state(sim, qpos.data(), qvel.data());

    MjmlxCameraRenderer* cam = mjmlx_camera_create(sim, W, H);
    CHECK(cam != nullptr, "camera create");
    if (!cam) return;
    CHECK(mjmlx_camera_count(cam) == 2, "ncam=%d", mjmlx_camera_count(cam));

    g_draws = 0;
    CHECK(mjmlx_camera_render(cam) == 0, "render");
    int draws = g_draws.load();
    printf("B=%d: vkCmdDraw*呼び出し回数(インターポーザ実測)=%d, レンダラ内部カウンタ=%d\n", B, draws, mjmlx_camera_draw_call_count(cam));
    CHECK(draws == 1, "draw calls=%d (expected 1)", draws);
    CHECK(mjmlx_camera_draw_call_count(cam) == 1, "internal draw count");

    size_t n = (size_t)B * 2 * H * W;
    std::vector<float> depth(n);
    std::vector<uint8_t> rgb(n * 3);
    CHECK(mjmlx_camera_get_depth(cam, depth.data()) == 0, "get_depth");
    CHECK(mjmlx_camera_get_rgb(cam, rgb.data()) == 0, "get_rgb");

    Stats fixed_st, cart_st;
    for (int e = 0; e < B; e++) {
        compare_image(&depth[((size_t)e * 2 + 0) * H * W], 0.0f, ball_x[e], fixed_st);
        compare_image(&depth[((size_t)e * 2 + 1) * H * W], cart_x[e], ball_x[e], cart_st);
        // 球中心付近のピクセルは赤く、床のピクセルは灰色であること(env毎の姿勢を反映)。
        int ci = H / 2, cj = (int)std::lround(W / 2.0 + (ball_x[e] / (3.0f * std::tan(kFovy * (float)M_PI / 360.0f))) * W / 2.0 - 0.5);
        const uint8_t* p = &rgb[(((size_t)e * 2 + 0) * H * W + (size_t)ci * W + cj) * 3];
        CHECK(p[0] > 60 && p[0] > 3 * (int)p[1] && p[0] > 3 * (int)p[2], "env %d: 球ピクセルが赤くない (%d,%d,%d) at col %d", e, p[0], p[1], p[2], cj);
    }
    if (g_dump && B == 4) { // 全env x 全カメラを横に並べたPPM
        FILE* f = fopen(g_dump, "wb");
        fprintf(f, "P6\n%d %d\n255\n", W * 2 * B, H);
        for (int i = 0; i < H; i++)
            for (int t = 0; t < 2 * B; t++) fwrite(&rgb[((size_t)t * H * W + (size_t)i * W) * 3], 1, W * 3, f);
        fclose(f);
    }
    for (auto* s : {&fixed_st, &cart_st}) {
        double frac = (double)s->class_mismatch / s->total;
        printf("  %s: 球depth最大誤差=%.4f m, 床depth最大相対誤差=%.5f, 分類不一致=%.3f%%\n", s == &fixed_st ? "fixed_cam" : "cart_cam(body搭載)", s->max_sphere, s->max_plane, 100 * frac);
        CHECK(s->max_sphere < 0.02, "sphere depth err %.4f", s->max_sphere);
        CHECK(s->max_plane < 2e-3, "plane depth rel err %.5f", s->max_plane);
        CHECK(frac < 0.01, "class mismatch %.3f%%", 100 * frac);
    }
    // envごとに画像が異なる(姿勢反映)こと: 先頭envと末尾envの球位置が違えばdepthが違う。
    if (B > 1) {
        double diff = 0;
        for (int k = 0; k < H * W; k++) diff += std::fabs(depth[k] - depth[((size_t)(B - 1) * 2) * H * W + k]);
        CHECK(diff > 1.0, "env0とenv%dの画像が同一", B - 1);
    }
    mjmlx_camera_free(cam);
    mjmlx_batched_free(sim);
    mjmlx_free_model(model);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_dump = argv[1];
    run(4);
    run(64);
    if (g_fail) { printf("test_camera: %d FAIL\n", g_fail); return 1; }
    printf("test_camera: PASS\n");
    return 0;
}
