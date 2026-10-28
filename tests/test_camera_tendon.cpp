// カメラセンサのtendon描画の検証。球wrap+sidesiteの空間tendonを描き、(1)site近傍の線分、(2)球の上を回る円弧(球の輪郭より外側にはみ出す管)が
// tendon色で描かれ、(3)env毎のsite位置が画像に反映されることを見る。MuJoCo標準レンダラとの画像比較は tests/test_camera_mujoco.py。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "mjmlx/mjmlx.h"

static int g_fail = 0;
#define CHECK(cond, ...)                              \
    do {                                              \
        if (!(cond)) {                                \
            g_fail++;                                 \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                      \
            printf("\n");                             \
        }                                             \
    } while (0)

// 球(中心(0,0,1) r=0.3、赤)の上を、a(-0.8,0,1.1+s)とb(0.8,0,1.1)を結ぶtendon(緑、半径0.06)が回る。sidesiteは球の上(0,0,1.6)。カメラ(0,-3,1.1)から+yを見る。
const char* kXml = R"(
<mujoco><option gravity="0 0 0"/>
<visual><global fovy="60"/></visual>
<worldbody>
  <geom name="ball" type="sphere" size="0.3" pos="0 0 1" rgba="1 0 0 1" contype="0" conaffinity="0"/>
  <site name="top" pos="0 0 1.6"/>
  <body name="a" pos="-0.8 0 1.1"><joint type="slide" axis="0 0 1"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/><site name="sa"/></body>
  <body name="b" pos="0.8 0 1.1"><joint type="slide" axis="0 0 1"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/><site name="sb"/></body>
  <camera name="c" pos="0 -3 1.1" fovy="60" xyaxes="1 0 0 0 0 1"/>
</worldbody>
<tendon>
  <spatial name="wrapped" width="0.06" rgba="0 1 0 1"><site site="sa"/><geom geom="ball" sidesite="top"/><site site="sb"/></spatial>
</tendon></mujoco>)";

constexpr int W = 128, H = 128;
const float kTan = 0.57735027f;  // tan(fovy/2)

// 世界点(x, y=0, z)の画素。カメラは(0,-3,1.1)。
static void project(float x, float z, int* row, int* col) {
    float depth = 3.0f;
    float u = x / depth / kTan, v = (z - 1.1f) / depth / kTan;
    *col = (int)std::lround((u + 1) * 0.5 * W - 0.5);
    *row = (int)std::lround((1 - v) * 0.5 * H - 0.5);
}

static bool green_near(const uint8_t* img, int row, int col, int rad) {
    for (int i = row - rad; i <= row + rad; i++)
        for (int j = col - rad; j <= col + rad; j++) {
            if (i < 0 || j < 0 || i >= H || j >= W) continue;
            const uint8_t* p = &img[((size_t)i * W + j) * 3];
            if (p[1] > 80 && p[1] > 3 * (int)p[0] && p[1] > 3 * (int)p[2]) return true;
        }
    return false;
}

int main(int argc, char** argv) {
    const int B = 4;
    MjmlxModel* model = mjmlx_load_model_from_string(kXml);
    CHECK(model != nullptr, "load model");
    if (!model) return 1;
    MjmlxBatchedConfig cfg{};
    cfg.num_envs = B; cfg.integrator = MJMLX_INTEGRATOR_EULER; cfg.use_gpu = 1;
    MjmlxBatchedSim* sim = mjmlx_batched_create(model, &cfg);
    CHECK(sim != nullptr, "batched create");
    auto info = mjmlx_model_info(model);
    std::vector<float> qpos((size_t)B * info.nq, 0.0f), qvel((size_t)B * info.nv, 0.0f), shift(B);
    for (int e = 0; e < B; e++) {
        shift[e] = 0.05f * e;
        qpos[(size_t)e * info.nq + 0] = shift[e];  // aのz
        qpos[(size_t)e * info.nq + 1] = 0.0f;      // bのz
    }
    mjmlx_batched_set_state(sim, qpos.data(), qvel.data());
    MjmlxCameraRenderer* cam = mjmlx_camera_create(sim, W, H);
    if (!cam) { printf("SKIP: camera unsupported on this device\n"); return 77; }
    CHECK(mjmlx_camera_render(cam) == 0, "render");
    CHECK(mjmlx_camera_draw_call_count(cam) == 1, "draw calls=%d (expected 1)", mjmlx_camera_draw_call_count(cam));
    std::vector<uint8_t> rgb((size_t)B * H * W * 3);
    CHECK(mjmlx_camera_get_rgb(cam, rgb.data()) == 0, "get_rgb");

    if (argc > 1) {  // 全envを横に並べたPPM
        FILE* f = fopen(argv[1], "wb");
        fprintf(f, "P6\n%d %d\n255\n", W * B, H);
        for (int i = 0; i < H; i++)
            for (int e = 0; e < B; e++) fwrite(&rgb[((size_t)e * H * W + (size_t)i * W) * 3], 1, W * 3, f);
        fclose(f);
    }
    for (int e = 0; e < B; e++) {
        const uint8_t* img = &rgb[(size_t)e * H * W * 3];
        int r, c;
        // 球の頂上を回る円弧(接点間の弦。MuJoCoと同じく半分の太さ)。直線(wrap無し)ならここは背景になる。
        project(0.0f, 1.0f + 0.3f + 0.04f, &r, &c);
        CHECK(green_near(img, r, c, 1), "env %d: 球の上の円弧が描かれていない (row %d col %d)", e, r, c);
        // 両端のsite近傍の線分(a側はenvごとに高さが違う)。
        project(-0.7f, 1.1f + shift[e] - 0.01f, &r, &c);
        CHECK(green_near(img, r, c, 3), "env %d: a側の線分が描かれていない (row %d col %d)", e, r, c);
        project(0.7f, 1.1f - 0.01f, &r, &c);
        CHECK(green_near(img, r, c, 3), "env %d: b側の線分が描かれていない (row %d col %d)", e, r, c);
        // 線分の外(tendonが通らない領域)は緑でない。
        project(-0.4f, 0.55f, &r, &c);
        CHECK(!green_near(img, r, c, 1), "env %d: tendonの無い場所が緑 (row %d col %d)", e, r, c);
    }
    // env毎のsite位置の差が画像に出る(env0とenv3は別画像)。
    double diff = 0;
    for (size_t k = 0; k < (size_t)H * W * 3; k++) diff += std::abs((int)rgb[k] - (int)rgb[(size_t)(B - 1) * H * W * 3 + k]);
    CHECK(diff > 1000.0, "env0とenv%dの画像が同一", B - 1);

    mjmlx_camera_free(cam);
    mjmlx_batched_free(sim);
    mjmlx_free_model(model);
    if (g_fail) { printf("test_camera_tendon: %d FAIL\n", g_fail); return 1; }
    printf("test_camera_tendon: PASS\n");
    return 0;
}
