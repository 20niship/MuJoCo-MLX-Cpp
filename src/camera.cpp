// カメラセンサ: 全env x 全カメラ x 全可視geomを1つのvkCmdDrawIndexedIndirectで描画する(物理stepとは別パス)。

#include <mujoco/mujoco.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "compat/mkx_kernels/tendon.hpp"
#include "internal.h"
#include "mjmlx/mjmlx.h"

using VB = mx::Backend;

namespace {

// ── Vulkanデバイス(mkxへ注入する)──────────────────────────────────────────────

struct CamDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    bool multi_draw_indirect = true, shader_clip_distance = true;
    bool valid = false;
    std::string reason = "camera device not prepared (mjmlx_load_model was not called before any Vulkan use)";
};
CamDevice g_dev;

// mkxへ注入したinstance/deviceの所有者。mkxのContext(関数内static、注入後の初回使用で生成)より先に構築されるので、終了時はmkxの解放が済んでから破棄される。
struct CamDeviceOwner {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    ~CamDeviceOwner() {
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};
CamDeviceOwner g_dev_owner;

// graphics+computeのqueue familyと、indirect描画に要るfeatureを持つデバイスだけをmkxに注入する。満たさなければ何もせず従来のmkx既定になる(カメラだけが使えない)。
void prepare_device() {
    if (const char* e = std::getenv("MJMLX_CAMERA_DEVICE")) { if (std::atoi(e) == 0) { g_dev.reason = "disabled by MJMLX_CAMERA_DEVICE=0"; return; } }
    VkInstance inst = VK_NULL_HANDLE;
    try {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "mjmlx"; app.apiVersion = VK_API_VERSION_1_3;
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
        std::vector<const char*> ie;
        bool portability = false;
        for (auto& e : exts) if (std::string(e.extensionName) == "VK_KHR_portability_enumeration") { ie.push_back("VK_KHR_portability_enumeration"); portability = true; }
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ii.pApplicationInfo = &app; ii.enabledExtensionCount = (uint32_t)ie.size(); ii.ppEnabledExtensionNames = ie.data();
        if (portability) ii.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        if (vkCreateInstance(&ii, nullptr, &inst) != VK_SUCCESS) { g_dev.reason = "vkCreateInstance failed"; return; }
        uint32_t nd = 0;
        vkEnumeratePhysicalDevices(inst, &nd, nullptr);
        if (nd == 0) { g_dev.reason = "no Vulkan device"; vkDestroyInstance(inst, nullptr); return; }
        std::vector<VkPhysicalDevice> devs(nd);
        vkEnumeratePhysicalDevices(inst, &nd, devs.data());
        int idx = 0;
        if (const char* e = std::getenv("MKX_VULKAN_DEVICE_INDEX")) idx = std::atoi(e);
        if (idx < 0 || idx >= (int)nd) idx = 0;
        VkPhysicalDevice phys = devs[idx];
        uint32_t nf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nf, nullptr);
        std::vector<VkQueueFamilyProperties> fam(nf);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nf, fam.data());
        int qf = -1;
        for (uint32_t i = 0; i < nf; i++)
            if ((fam[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { qf = (int)i; break; }
        VkPhysicalDeviceFeatures sup{};
        vkGetPhysicalDeviceFeatures(phys, &sup);
        if (qf < 0) { g_dev.reason = "no graphics+compute queue family"; vkDestroyInstance(inst, nullptr); return; }
        if (!sup.multiDrawIndirect || !sup.drawIndirectFirstInstance) { g_dev.reason = "device lacks multiDrawIndirect/drawIndirectFirstInstance"; vkDestroyInstance(inst, nullptr); return; }
        if (!sup.shaderClipDistance) { g_dev.reason = "device lacks shaderClipDistance"; vkDestroyInstance(inst, nullptr); return; }
        VkPhysicalDeviceFeatures en{};
        en.multiDrawIndirect = en.drawIndirectFirstInstance = en.shaderClipDistance = VK_TRUE;
        uint32_t nde = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, nullptr);
        std::vector<VkExtensionProperties> dex(nde);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, dex.data());
        std::vector<const char*> de;
        for (auto& e : dex) if (std::string(e.extensionName) == "VK_KHR_portability_subset") de.push_back("VK_KHR_portability_subset");
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = (uint32_t)qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi; di.pEnabledFeatures = &en;
        di.enabledExtensionCount = (uint32_t)de.size(); di.ppEnabledExtensionNames = de.data();
        VkDevice dev = VK_NULL_HANDLE;
        if (vkCreateDevice(phys, &di, nullptr, &dev) != VK_SUCCESS) { g_dev.reason = "vkCreateDevice failed"; vkDestroyInstance(inst, nullptr); return; }
        VkQueue q;
        vkGetDeviceQueue(dev, (uint32_t)qf, 0, &q);
        mkx::VulkanBackend::ExternalDevice ext{inst, phys, dev, q, (uint32_t)qf};
        if (!mkx::VulkanBackend::use_external_device(ext)) {
            g_dev.reason = "mkx was already initialised before mjmlx_load_model";
            vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
            return;
        }
        g_dev.instance = inst; g_dev.physical = phys; g_dev.device = dev; g_dev.queue = q; g_dev.queue_family = (uint32_t)qf;
        g_dev_owner.instance = inst; g_dev_owner.device = dev;
        g_dev.valid = true; g_dev.reason.clear();
    } catch (const std::exception& e) {
        g_dev.reason = e.what();
        if (inst && !g_dev.valid) vkDestroyInstance(inst, nullptr);
    }
}

} // namespace

namespace mjmlx {
void camera_prepare_device() {
    static std::once_flag f;
    std::call_once(f, prepare_device);
}
} // namespace mjmlx

namespace {

// ── GLSL ─────────────────────────────────────────────────────────────────────

const char* kCommon = R"(
layout(push_constant) uniform PC {
  uint B; uint ncam; uint nvis; uint tiles_x; uint tiles_y; uint nlights; uint nbody; uint H;
  float aspect; float znear; float zfar; float lodk;
} pc;
)";

// カメラ姿勢(env,cam)をbody姿勢から求める。出力は4 x vec4: {pos.xyz, 1/tan(fovy/2)}, 世界座標でのカメラx/y/z軸。
const char* kCamComp = R"(
#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 2) buffer CP { vec4 campose[]; };
layout(std430, binding = 5) readonly buffer BP { float body_xpos[]; };
layout(std430, binding = 6) readonly buffer BM { float body_xmat[]; };
struct CamC { vec4 a; vec4 b; vec4 rl0; vec4 rl1; vec4 rl2; vec4 rw0; vec4 rw1; vec4 rw2; ivec4 ids; };
layout(std430, binding = 7) readonly buffer CC { CamC camc[]; };
)" "$COMMON" R"(
vec3 bpos(uint b, uint e) { return vec3(body_xpos[(b * 3u + 0u) * pc.B + e], body_xpos[(b * 3u + 1u) * pc.B + e], body_xpos[(b * 3u + 2u) * pc.B + e]); }
mat3 bmat(uint b, uint e) {
  mat3 M;
  for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) M[j][i] = body_xmat[(b * 9u + 3u * i + j) * pc.B + e];
  return M;
}
mat3 fromRows(vec3 r0, vec3 r1, vec3 r2) { return transpose(mat3(r0, r1, r2)); }
void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.B * pc.ncam) return;
  uint e = i / pc.ncam, c = i % pc.ncam;
  CamC k = camc[c];
  int mode = int(k.a.w);
  uint bid = uint(k.ids.x);
  vec3 xb = bpos(bid, e);
  mat3 Rb = bmat(bid, e);
  vec3 p; mat3 R;
  if (mode == 1 || mode == 2) { p = xb + k.b.xyz; R = fromRows(k.rw0.xyz, k.rw1.xyz, k.rw2.xyz); }
  else {
    p = xb + Rb * k.a.xyz;
    if (mode >= 3 && k.ids.y >= 0) {
      vec3 z = normalize(p - bpos(uint(k.ids.y), e));
      vec3 x = normalize(cross(vec3(0.0, 0.0, 1.0), z));
      R = mat3(x, cross(z, x), z);
    } else R = Rb * fromRows(k.rl0.xyz, k.rl1.xyz, k.rl2.xyz);
  }
  campose[i * 4u + 0u] = vec4(p, k.b.w);
  campose[i * 4u + 1u] = vec4(R[0], 0.0);
  campose[i * 4u + 2u] = vec4(R[1], 0.0);
  campose[i * 4u + 3u] = vec4(R[2], 0.0);
}
)";

// (env,cam,geom)毎にフラスタム(球)カリングとLOD選択を行い、indirectコマンドをGPU上で書く(CPUへは戻さない)。
const char* kCull = R"(
#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer GX { float geom_xpos[]; };
layout(std430, binding = 1) readonly buffer GM { float geom_xmat[]; };
layout(std430, binding = 2) readonly buffer CP { vec4 cam[]; };
struct G { vec4 rgba; vec4 ms; uvec4 ids; vec4 tx; };
layout(std430, binding = 3) readonly buffer GT { G gt[]; };
struct Lod { vec4 sphere; uvec4 lod[4]; };
layout(std430, binding = 9) readonly buffer LD { Lod lods[]; };
layout(std430, binding = 10) writeonly buffer IC { uint cmds[]; };
)" "$COMMON" R"(
void main() {
  uint k = gl_GlobalInvocationID.x;
  if (k >= pc.B * pc.ncam * pc.nvis) return;
  uint vis = k % pc.nvis, ec = k / pc.nvis, e = ec / pc.ncam;
  uint g = gt[vis].ids.x;
  vec3 gp = vec3(geom_xpos[(g * 3u + 0u) * pc.B + e], geom_xpos[(g * 3u + 1u) * pc.B + e], geom_xpos[(g * 3u + 2u) * pc.B + e]);
  mat3 R;
  for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) R[j][i] = geom_xmat[(g * 9u + 3u * i + j) * pc.B + e];
  Lod L = lods[vis];
  float r = L.sphere.w;
  vec3 d = gp + R * L.sphere.xyz - cam[ec * 4u].xyz;
  vec3 v = vec3(dot(cam[ec * 4u + 1u].xyz, d), dot(cam[ec * 4u + 2u].xyz, d), dot(cam[ec * 4u + 3u].xyz, d));
  float w = -v.z, f = cam[ec * 4u].w, fx = f / pc.aspect;
  bool vis_ok = (w + r > pc.znear) && (w - r < pc.zfar)
      && ((abs(v.x) * fx - w) * inversesqrt(fx * fx + 1.0) <= r) && ((abs(v.y) * f - w) * inversesqrt(f * f + 1.0) <= r);
  float pr = r * f / max(w, 1e-3) * 0.5 * float(pc.H);
  float area = 3.14159 * pr * pr * pc.lodk;
  uint l = 0u;
  if (pc.lodk > 0.0) { for (l = 0u; l < 3u; l++) if (float(L.lod[l].y / 3u) <= area) break; }
  uint b = k * 5u;
  cmds[b] = L.lod[l].y; cmds[b + 1u] = vis_ok ? 1u : 0u; cmds[b + 2u] = L.lod[l].x; cmds[b + 3u] = L.lod[l].z; cmds[b + 4u] = k;
}
)";

// 1ドローで複数タイルを描くため、タイル境界はviewportでなくgl_ClipDistanceでクリップする。
const char* kVert = R"(
#version 450
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_nrm;
layout(location = 2) in vec2 in_uv;
layout(std430, binding = 0) readonly buffer GX { float geom_xpos[]; };
layout(std430, binding = 1) readonly buffer GM { float geom_xmat[]; };
layout(std430, binding = 2) readonly buffer CP { vec4 cam[]; };
struct G { vec4 rgba; vec4 ms; uvec4 ids; vec4 tx; };
layout(std430, binding = 3) readonly buffer GT { G gt[]; };
)" "$COMMON" R"(
layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_nrm;
layout(location = 2) out float v_depth;
layout(location = 3) flat out uint v_vis;
layout(location = 4) flat out uint v_cam;
layout(location = 5) out vec2 v_uv;
out float gl_ClipDistance[4];
void main() {
  uint idx = uint(gl_InstanceIndex);
  uint vis = idx % pc.nvis, ec = idx / pc.nvis;
  uint e = ec / pc.ncam;
  uint g = gt[vis].ids.x;
  vec3 gp = vec3(geom_xpos[(g * 3u + 0u) * pc.B + e], geom_xpos[(g * 3u + 1u) * pc.B + e], geom_xpos[(g * 3u + 2u) * pc.B + e]);
  mat3 R;
  for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) R[j][i] = geom_xmat[(g * 9u + 3u * i + j) * pc.B + e];
  vec3 world = gp + R * in_pos;
  v_world = world; v_nrm = R * in_nrm; v_vis = vis; v_cam = ec; v_uv = in_uv;

  vec4 c0 = cam[ec * 4u];
  vec3 d = world - c0.xyz;
  vec3 v = vec3(dot(cam[ec * 4u + 1u].xyz, d), dot(cam[ec * 4u + 2u].xyz, d), dot(cam[ec * 4u + 3u].xyz, d));
  float w = -v.z, f = c0.w;
  float cx = v.x * f / pc.aspect, cy = -v.y * f;
  float cz = pc.zfar / (pc.zfar - pc.znear) * (w - pc.znear);
  v_depth = w;

  float tx = float(ec % pc.tiles_x), ty = float(ec / pc.tiles_x);
  float nx = float(pc.tiles_x), ny = float(pc.tiles_y);
  float X = (cx + (1.0 + 2.0 * tx - nx) * w) / nx;
  float Y = (cy + (1.0 + 2.0 * ty - ny) * w) / ny;
  gl_Position = vec4(X, Y, cz, w);
  gl_ClipDistance[0] = X - (2.0 * tx - nx) / nx * w;
  gl_ClipDistance[1] = (2.0 * tx + 2.0 - nx) / nx * w - X;
  gl_ClipDistance[2] = Y - (2.0 * ty - ny) / ny * w;
  gl_ClipDistance[3] = (2.0 * ty + 2.0 - ny) / ny * w - Y;
}
)";

// Phong(ambient+diffuse+specular)。両面ライティング(裏面は法線を反転)。
const char* kFrag = R"(
#version 450
layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_nrm;
layout(location = 2) in float v_depth;
layout(location = 3) flat in uint v_vis;
layout(location = 4) flat in uint v_cam;
layout(location = 5) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_depth;
layout(std430, binding = 2) readonly buffer CP { vec4 cam[]; };
struct G { vec4 rgba; vec4 ms; uvec4 ids; vec4 tx; };
layout(std430, binding = 3) readonly buffer GT { G gt[]; };
struct L { vec4 pos; vec4 dir; vec4 diffuse; vec4 ambient; vec4 specular; };
layout(std430, binding = 4) readonly buffer LT { L lights[]; };
layout(std430, binding = 8) readonly buffer TX { uint texels[]; };
)" "$COMMON" R"(
void main() {
  G g = gt[v_vis];
  vec3 V = normalize(cam[v_cam * 4u].xyz - v_world);
  vec3 N = normalize(v_nrm);
  if (dot(N, V) < 0.0) N = -N;
  vec3 base = g.rgba.xyz;
  if (g.ids.y != 0u) {
    uint w = g.ids.w >> 16, h = g.ids.w & 0xffffu;
    vec2 f = fract(v_uv);
    uint px = texels[g.ids.z + min(uint(f.y * float(h)), h - 1u) * w + min(uint(f.x * float(w)), w - 1u)];
    base *= vec3(float(px & 255u), float((px >> 8) & 255u), float((px >> 16) & 255u)) / 255.0;
  }
  float shin = max(g.ms.y * 128.0, 1.0);
  vec3 col = vec3(0.0);
  for (uint i = 0u; i < pc.nlights; i++) {
    L l = lights[i];
    int type = int(l.pos.w);
    vec3 Ld = type == 2 ? cam[v_cam * 4u + 3u].xyz : (type == 1 ? -normalize(l.dir.xyz) : normalize(l.pos.xyz - v_world));
    float nd = max(dot(N, Ld), 0.0);
    float sp = nd > 0.0 ? pow(max(dot(N, normalize(Ld + V)), 0.0), shin) : 0.0;
    col += l.ambient.xyz * base + l.diffuse.xyz * nd * base + l.specular.xyz * sp * g.ms.x;
  }
  o_color = vec4(clamp(col, 0.0, 1.0), 1.0);
  o_depth = v_depth;
}
)";

const char* kCommonT = R"(
layout(push_constant) uniform PC {
  uint B; uint ncam; uint nvis; uint tiles_x; uint tiles_y; uint nlights; uint nbody; uint H;
  float aspect; float znear; float zfar; float lodk;
  uint ngeom; uint ntend; uint nslot;
} pc;
)";

std::string with_common(const std::string& src, bool tendon = false) {
    std::string s = src;
    size_t p;
    while ((p = s.find("$COMMON")) != std::string::npos) s.replace(p, 7, tendon ? kCommonT : kCommon);
    return s;
}

// ── tendon描画(空間tendonのみ有効)─────────────────────────────────────────────
// tendonの経路(site間・球/円柱wrapの接点)を、(env,tendon)ごとにGPUで求めて線分ごとの円柱の姿勢(仮想geom)としてtpos/tmatへ書く。
// 仮想geomはgeom表の後ろ(ids.x >= ngeom)に並べ、頂点/cullシェーダだけがそこから姿勢を読む。tendonが無いモデルは元のシェーダのまま(下の置換をしない)。

std::string tendon_variant(std::string s, bool is_cull) {
    auto rep = [&](const std::string& from, const std::string& to) {
        size_t p = s.find(from);
        if (p == std::string::npos) throw std::runtime_error("mjmlx_camera: tendon shader patch target missing");
        s.replace(p, from.size(), to);
    };
    rep("layout(std430, binding = 1) readonly buffer GM { float geom_xmat[]; };",
        "layout(std430, binding = 1) readonly buffer GM { float geom_xmat[]; };\n"
        "layout(std430, binding = 11) readonly buffer TP { float tpos[]; };\n"
        "layout(std430, binding = 12) readonly buffer TM { float tmat[]; };");
    rep("  vec3 gp = vec3(geom_xpos[(g * 3u + 0u) * pc.B + e], geom_xpos[(g * 3u + 1u) * pc.B + e], geom_xpos[(g * 3u + 2u) * pc.B + e]);\n"
        "  mat3 R;\n"
        "  for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) R[j][i] = geom_xmat[(g * 9u + 3u * i + j) * pc.B + e];\n",
        "  vec3 gp; mat3 R;\n"
        "  if (g >= pc.ngeom) {\n"
        "    uint ts = g - pc.ngeom;\n"
        "    gp = vec3(tpos[(ts * 3u + 0u) * pc.B + e], tpos[(ts * 3u + 1u) * pc.B + e], tpos[(ts * 3u + 2u) * pc.B + e]);\n"
        "    for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) R[j][i] = tmat[(ts * 9u + 3u * i + j) * pc.B + e];\n"
        "  } else {\n"
        "    gp = vec3(geom_xpos[(g * 3u + 0u) * pc.B + e], geom_xpos[(g * 3u + 1u) * pc.B + e], geom_xpos[(g * 3u + 2u) * pc.B + e]);\n"
        "    for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) R[j][i] = geom_xmat[(g * 9u + 3u * i + j) * pc.B + e];\n"
        "  }\n");
    if (is_cull)  // tendonの仮想geomはカリングせずLOD0固定(線分の長さが時変で球境界が使えない)
        rep("  uint g = gt[vis].ids.x;\n",
            "  uint g = gt[vis].ids.x;\n"
            "  if (g >= pc.ngeom) { Lod L0 = lods[vis]; uint b0 = k * 5u; cmds[b0] = L0.lod[0].y; cmds[b0 + 1u] = 1u; cmds[b0 + 2u] = L0.lod[0].x; cmds[b0 + 3u] = L0.lod[0].z; cmds[b0 + 4u] = k; return; }\n");
    return s;
}


// MuJoCoと同様、wrapはsite→接点→(接点間の弦、半分の太さ)→接点→siteの線分で描く。(env,tendon)ごとに経路をたどり、線分ごとに円柱(スロットs)の姿勢を書く。円柱メッシュは半径1・半長1なので、列に半径/半長を掛けた行列にする。
const char* kTendComp = R"(
#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer GX { float geom_xpos[]; };
layout(std430, binding = 1) readonly buffer GM { float geom_xmat[]; };
layout(std430, binding = 5) readonly buffer BP { float body_xpos[]; };
layout(std430, binding = 6) readonly buffer BM { float body_xmat[]; };
layout(std430, binding = 11) writeonly buffer TP { float tpos[]; };
layout(std430, binding = 12) writeonly buffer TM { float tmat[]; };
struct W { ivec4 a; vec4 b; ivec4 c; vec4 d; };   // a={type,objid,hasside,body}, b={site pos | radius}, c={sidebody}, d={sidepos}
layout(std430, binding = 13) readonly buffer WT { W wt[]; };
struct T { ivec4 a; vec4 col; };                    // a={adr,num,slot0,nslots}, col.x=半径
layout(std430, binding = 14) readonly buffer TT { T tt[]; };
)" "$COMMON" R"(
$WRAPFN
mat3 bmat(uint b, uint e) {
  mat3 M;
  for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) M[j][i] = body_xmat[(b * 9u + 3u * i + j) * pc.B + e];
  return M;
}
vec3 bpos(uint b, uint e) { return vec3(body_xpos[(b * 3u + 0u) * pc.B + e], body_xpos[(b * 3u + 1u) * pc.B + e], body_xpos[(b * 3u + 2u) * pc.B + e]); }
vec3 site_w(uint body, vec3 lp, uint e) { return bpos(body, e) + bmat(body, e) * lp; }
void put(uint e, uint slot, vec3 c, mat3 R) {
  for (uint k = 0u; k < 3u; k++) tpos[(slot * 3u + k) * pc.B + e] = c[k];
  for (uint i = 0u; i < 3u; i++) for (uint j = 0u; j < 3u; j++) tmat[(slot * 9u + 3u * i + j) * pc.B + e] = R[j][i];
}
void off(uint e, uint slot) { put(e, slot, vec3(0.0), mat3(0.0)); }
void seg(uint e, uint slot, vec3 a, vec3 b, float r) {
  vec3 d = b - a; float L = length(d);
  if (L < 1e-9) { off(e, slot); return; }
  vec3 z = d / L;
  vec3 t = abs(z.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 x = normalize(cross(t, z)); vec3 y = cross(z, x);
  put(e, slot, 0.5 * (a + b), mat3(x * r, y * r, z * (0.5 * L + 0.5 * r)));  // 端を半径の半分だけ延ばし、折れ曲がりの隙間を埋める(MuJoCoはカプセル)
}
void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.B * pc.ntend) return;
  uint e = i / pc.ntend, t = i % pc.ntend;
  T tn = tt[t];
  int adr = tn.a.x, num = tn.a.y; uint slot = uint(tn.a.z);
  float r = tn.col.x;
  int j = 0;
  while (j < num - 1) {
    W w0 = wt[adr + j], w1 = wt[adr + j + 1];
    if (w0.a.x == 2 || w1.a.x == 2) { j++; continue; }
    vec3 s0 = site_w(uint(w0.a.w), w0.b.xyz, e);
    if (w1.a.x == 4 || w1.a.x == 5) {
      W w2 = wt[adr + j + 2];
      vec3 s1 = site_w(uint(w2.a.w), w2.b.xyz, e);
      uint g = uint(w1.a.y);
      vec3 gx = vec3(geom_xpos[(g * 3u + 0u) * pc.B + e], geom_xpos[(g * 3u + 1u) * pc.B + e], geom_xpos[(g * 3u + 2u) * pc.B + e]);
      mat3 gR = mat3(geom_xmat[(g * 9u + 0u) * pc.B + e], geom_xmat[(g * 9u + 3u) * pc.B + e], geom_xmat[(g * 9u + 6u) * pc.B + e],
                     geom_xmat[(g * 9u + 1u) * pc.B + e], geom_xmat[(g * 9u + 4u) * pc.B + e], geom_xmat[(g * 9u + 7u) * pc.B + e],
                     geom_xmat[(g * 9u + 2u) * pc.B + e], geom_xmat[(g * 9u + 5u) * pc.B + e], geom_xmat[(g * 9u + 8u) * pc.B + e]);
      bool has = w1.a.z != 0;
      vec3 side = has ? site_w(uint(w1.c.x), w1.d.xyz, e) : vec3(0.0);
      bool sph = w1.a.x == 4;
      vec3 p0, p1;
      float wl = tw_wrap(p0, p1, s0, s1, gx, gR, w1.b.x, sph, has, side);
      if (wl < 0.0) {
        seg(e, slot, s0, s1, r);
        off(e, slot + 1u); off(e, slot + 2u);
      } else {
        seg(e, slot, s0, p0, r);
        seg(e, slot + 1u, p0, p1, 0.5 * r);
        seg(e, slot + 2u, p1, s1, r);
      }
      slot += 3u; j += 2;
    } else {
      vec3 s1 = site_w(uint(w1.a.w), w1.b.xyz, e);
      seg(e, slot, s0, s1, r);
      slot += 1u; j += 1;
    }
  }
}
)";

std::string tendon_comp_source() {
    std::string s = kTendComp;
    size_t p;
    while ((p = s.find("$WRAPFN")) != std::string::npos) s.replace(p, 7, mjmlx::mkx_kernels::detail_tendon::wrap_header());
    while ((p = s.find("$COMMON")) != std::string::npos) s.replace(p, 7, kCommonT);
    return s;
}

std::vector<uint32_t> compile_glsl(const std::string& source, const char* stage) {
    char path[] = "/tmp/mjmlx_cam_XXXXXX.glsl";
    int fd = mkstemps(path, 5);
    if (fd < 0) throw std::runtime_error("mjmlx_camera: mkstemps failed");
    close(fd);
    { std::ofstream out(path, std::ios::trunc); out << source; }
    std::string spv = std::string(path) + ".spv";
    std::string log = std::string(path) + ".log";
    std::string cmd = std::string("glslangValidator -V --target-env vulkan1.2 -S ") + stage + " -o " + spv + " " + path + " >" + log + " 2>&1";
    int rc = std::system(cmd.c_str());
    std::string msg;
    if (rc != 0) { std::ifstream l(log); msg.assign(std::istreambuf_iterator<char>(l), {}); }
    std::ifstream f(spv, std::ios::binary | std::ios::ate);
    std::vector<uint32_t> code;
    if (f) {
        size_t n = static_cast<size_t>(f.tellg());
        f.seekg(0);
        code.resize(n / 4);
        f.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(n));
    }
    std::remove(path); std::remove(spv.c_str()); std::remove(log.c_str());
    if (rc != 0 || code.empty()) throw std::runtime_error("mjmlx_camera: glslangValidator failed (" + std::string(stage) + "): " + msg);
    return code;
}

// ── ジオメトリ(geomローカル座標)────────────────────────────────────────────────

struct Vtx { float p[3]; float n[3]; float uv[2]; };
struct Mesh { std::vector<Vtx> v; std::vector<uint32_t> i; };

struct ProfilePt { float r, z, nr, nz; };

// z軸まわりの回転体(球/カプセル/円柱/楕円体)。profileは下から上へ。
Mesh revolve(const std::vector<ProfilePt>& prof, int slices) {
    Mesh m;
    int np = (int)prof.size();
    for (int s = 0; s <= slices; s++) {
        float a = 2.0f * (float)M_PI * (float)s / (float)slices, ca = std::cos(a), sa = std::sin(a);
        for (auto& q : prof) m.v.push_back({{q.r * ca, q.r * sa, q.z}, {q.nr * ca, q.nr * sa, q.nz}});
    }
    for (int s = 0; s < slices; s++)
        for (int k = 0; k + 1 < np; k++) {
            uint32_t a = s * np + k, b = (s + 1) * np + k;
            m.i.insert(m.i.end(), {a, b, a + 1, b, b + 1, a + 1});
        }
    return m;
}

std::vector<ProfilePt> arc(float R, float z0, float phi0, float phi1, int n) {
    std::vector<ProfilePt> p;
    for (int k = 0; k <= n; k++) {
        float phi = phi0 + (phi1 - phi0) * (float)k / (float)n;
        p.push_back({R * std::cos(phi), z0 + R * std::sin(phi), std::cos(phi), std::sin(phi)});
    }
    return p;
}

Mesh make_box(const float* s) {
    Mesh m;
    for (int ax = 0; ax < 3; ax++)
        for (int sg = -1; sg <= 1; sg += 2) {
            int u = (ax + 1) % 3, w = (ax + 2) % 3;
            uint32_t base = (uint32_t)m.v.size();
            for (int c = 0; c < 4; c++) {
                float su = (c == 0 || c == 3) ? -1.0f : 1.0f, sw = (c < 2) ? -1.0f : 1.0f;
                Vtx v{};
                v.p[ax] = sg * s[ax]; v.p[u] = su * s[u]; v.p[w] = sw * s[w];
                v.n[ax] = (float)sg;
                m.v.push_back(v);
            }
            m.i.insert(m.i.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    return m;
}

Mesh make_plane(const float* s) {
    float hx = s[0] > 0 ? s[0] : 100.0f, hy = s[1] > 0 ? s[1] : 100.0f; // ponytail: 無限planeは半幅100mの有限矩形で近似
    Mesh m;
    m.v = {{{-hx, -hy, 0}, {0, 0, 1}}, {{hx, -hy, 0}, {0, 0, 1}}, {{hx, hy, 0}, {0, 0, 1}}, {{-hx, hy, 0}, {0, 0, 1}}};
    m.i = {0, 1, 2, 0, 2, 3};
    return m;
}

// 背面カリングのため、各三角形の巻き方向を頂点法線(外向き)に合わせる。
void orient_outward(Mesh& m) {
    for (size_t t = 0; t + 2 < m.i.size(); t += 3) {
        const Vtx &a = m.v[m.i[t]], &b = m.v[m.i[t + 1]], &c = m.v[m.i[t + 2]];
        float e1[3], e2[3], n[3], dot = 0;
        for (int k = 0; k < 3; k++) { e1[k] = b.p[k] - a.p[k]; e2[k] = c.p[k] - a.p[k]; }
        n[0] = e1[1] * e2[2] - e1[2] * e2[1]; n[1] = e1[2] * e2[0] - e1[0] * e2[2]; n[2] = e1[0] * e2[1] - e1[1] * e2[0];
        for (int k = 0; k < 3; k++) dot += n[k] * (a.n[k] + b.n[k] + c.n[k]);
        if (dot < 0) std::swap(m.i[t + 1], m.i[t + 2]);
    }
}

Mesh make_primitive(int type, const float* s) {
    const int sl = 32, st = 16;
    switch (type) {
        case 0: return make_plane(s);
        case 2: return revolve(arc(s[0], 0, -(float)M_PI / 2, (float)M_PI / 2, 2 * st), sl);
        case 3: {
            auto p = arc(s[0], -s[1], -(float)M_PI / 2, 0, st), q = arc(s[0], s[1], 0, (float)M_PI / 2, st);
            p.insert(p.end(), q.begin(), q.end());
            return revolve(p, sl);
        }
        case 4: {
            Mesh m = revolve(arc(1.0f, 0, -(float)M_PI / 2, (float)M_PI / 2, 2 * st), sl);
            for (auto& v : m.v) {
                float n[3];
                for (int k = 0; k < 3; k++) { n[k] = v.n[k] / s[k]; v.p[k] *= s[k]; }
                float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                for (int k = 0; k < 3; k++) v.n[k] = n[k] / l;
            }
            return m;
        }
        case 5: return revolve({{0, -s[1], 0, -1}, {s[0], -s[1], 0, -1}, {s[0], -s[1], 1, 0}, {s[0], s[1], 1, 0}, {s[0], s[1], 0, 1}, {0, s[1], 0, 1}}, sl);
        case 6: return make_box(s);
    }
    return {};
}

// 平面状のUV(MuJoCoのplane/box/hfieldの2Dテクスチャ写像: 中心がu=0.5, vは+y側が小さい)。
void apply_planar_uv(Mesh& m, float hx, float hy, const float* rep, bool uniform) {
    for (auto& v : m.v) {
        v.uv[0] = (uniform ? v.p[0] * rep[0] / 2 : rep[0] * v.p[0] / (2 * hx)) + 0.5f;
        v.uv[1] = 0.5f - (uniform ? v.p[1] * rep[1] / 2 : rep[1] * v.p[1] / (2 * hy));
    }
}

void smooth_normals(Mesh& m) {
    std::vector<float> acc(m.v.size() * 3, 0.0f);
    for (size_t t = 0; t + 2 < m.i.size(); t += 3) {
        const float *a = m.v[m.i[t]].p, *b = m.v[m.i[t + 1]].p, *c = m.v[m.i[t + 2]].p;
        float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        for (int k = 0; k < 3; k++) for (int j = 0; j < 3; j++) acc[m.i[t + k] * 3 + j] += n[j];
    }
    for (size_t v = 0; v < m.v.size(); v++) {
        float* n = &acc[v * 3];
        float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (l < 1e-20f) { n[0] = 0; n[1] = 0; n[2] = 1; l = 1; }
        for (int j = 0; j < 3; j++) m.v[v].n[j] = n[j] / l;
    }
}

// 頂点クラスタリングによる簡略化(格子セルごとに頂点を平均、縮退三角形を捨てる)。法線は作り直す。
Mesh cluster_simplify(const Mesh& src, float cell, const float* lo) {
    std::map<uint64_t, uint32_t> cellid;
    std::vector<float> sum;
    std::vector<uint32_t> cnt, remap(src.v.size());
    for (size_t i = 0; i < src.v.size(); i++) {
        int64_t q[3];
        for (int k = 0; k < 3; k++) q[k] = (int64_t)std::floor((src.v[i].p[k] - lo[k]) / cell);
        uint64_t key = ((uint64_t)q[0] & 0x1fffff) | ((uint64_t)q[1] & 0x1fffff) << 21 | ((uint64_t)q[2] & 0x1fffff) << 42;
        auto it = cellid.find(key);
        if (it == cellid.end()) { it = cellid.emplace(key, (uint32_t)cnt.size()).first; cnt.push_back(0); sum.insert(sum.end(), {0, 0, 0}); }
        remap[i] = it->second;
        cnt[it->second]++;
        for (int k = 0; k < 3; k++) sum[it->second * 3 + k] += src.v[i].p[k];
    }
    Mesh out;
    out.v.resize(cnt.size());
    for (size_t c = 0; c < cnt.size(); c++) { Vtx v{}; for (int k = 0; k < 3; k++) v.p[k] = sum[c * 3 + k] / cnt[c]; out.v[c] = v; }
    for (size_t t = 0; t + 2 < src.i.size(); t += 3) {
        uint32_t a = remap[src.i[t]], b = remap[src.i[t + 1]], c = remap[src.i[t + 2]];
        if (a != b && b != c && a != c) out.i.insert(out.i.end(), {a, b, c});
    }
    smooth_normals(out);
    return out;
}

// 三角形数がratio*元の数以下になる最小のセルサイズを二分探索して簡略化する。
Mesh lod_for_ratio(const Mesh& src, double ratio) {
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (auto& v : src.v) for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], v.p[k]); hi[k] = std::max(hi[k], v.p[k]); }
    float ext = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-6f});
    size_t target = (size_t)(src.i.size() * ratio);
    float a = ext / 1024, b = ext;
    Mesh best = cluster_simplify(src, b, lo);
    for (int it = 0; it < 14; it++) {
        float mid = std::sqrt(a * b);
        Mesh m = cluster_simplify(src, mid, lo);
        if (m.i.size() <= target) { best = std::move(m); b = mid; } else a = mid;
    }
    return best;
}

// 高さ格子: x=(2c/(ncol-1)-1)*size0, y=(2r/(nrow-1)-1)*size1, z=data*size2。側面のskirtは作らない。
Mesh make_hfield(const mjModel* m, int id) {
    Mesh out;
    int nr = m->hfield_nrow[id], nc = m->hfield_ncol[id];
    const float* data = m->hfield_data + m->hfield_adr[id];
    float sx = (float)m->hfield_size[4 * id], sy = (float)m->hfield_size[4 * id + 1], sz = (float)m->hfield_size[4 * id + 2];
    for (int r = 0; r < nr; r++)
        for (int c = 0; c < nc; c++) {
            Vtx v{};
            v.p[0] = (2.0f * c / (nc - 1) - 1) * sx; v.p[1] = (2.0f * r / (nr - 1) - 1) * sy; v.p[2] = data[r * nc + c] * sz;
            out.v.push_back(v);
        }
    for (int r = 0; r + 1 < nr; r++)
        for (int c = 0; c + 1 < nc; c++) {
            uint32_t a = r * nc + c, b = a + 1, d = a + nc, e = d + 1;
            out.i.insert(out.i.end(), {a, b, e, a, e, d});
        }
    smooth_normals(out);
    return out;
}

Mesh make_mesh(const mjModel* m, int id) {
    Mesh out;
    int va = m->mesh_vertadr[id], vn = m->mesh_vertnum[id], fa = m->mesh_faceadr[id], fn = m->mesh_facenum[id];
    std::vector<float> acc(vn * 3, 0.0f);
    for (int f = 0; f < fn; f++) {
        const int* t = m->mesh_face + 3 * (fa + f);
        const float* a = m->mesh_vert + 3 * (va + t[0]);
        const float* b = m->mesh_vert + 3 * (va + t[1]);
        const float* c = m->mesh_vert + 3 * (va + t[2]);
        float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        for (int k = 0; k < 3; k++) for (int j = 0; j < 3; j++) acc[t[k] * 3 + j] += n[j];
        out.i.insert(out.i.end(), {(uint32_t)t[0], (uint32_t)t[1], (uint32_t)t[2]});
    }
    for (int v = 0; v < vn; v++) {
        float* n = &acc[v * 3];
        float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (l < 1e-20f) { n[0] = 0; n[1] = 0; n[2] = 1; l = 1; }
        const float* p = m->mesh_vert + 3 * (va + v);
        out.v.push_back({{p[0], p[1], p[2]}, {n[0] / l, n[1] / l, n[2] / l}});
    }
    return out;
}

// ── Vulkan補助 ───────────────────────────────────────────────────────────────

void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("mjmlx_camera: ") + what + " failed (VkResult=" + std::to_string((int)r) + ")");
}

struct GpuBuf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; size_t size = 0; };

// std430のvec4/uvec4配列と合わせるための構造体(CPU側)。
struct GeomRec { float rgba[4]; float ms[4]; uint32_t ids[4]; float tx[4]; };
struct LightRec { float pos[4], dir[4], diffuse[4], ambient[4], specular[4]; };
struct CamConst { float a[4], b[4], rl[3][4], rw[3][4]; int32_t ids[4]; };
struct PushC { uint32_t B, ncam, nvis, tiles_x, tiles_y, nlights, nbody, H; float aspect, znear, zfar, lodk; uint32_t ngeom, ntend, nslot; };
struct WrapRec { int32_t a[4]; float b[4]; int32_t c[4]; float d[4]; };
struct TendRec { int32_t a[4]; float col[4]; };
struct LodRec { float sphere[4]; uint32_t lod[4][4]; };

} // namespace

struct MjmlxCameraRenderer {
    mjmlx::BatchedSim* sim = nullptr;
    CamDevice h;
    int B = 0, ncam = 0, W = 0, H = 0, nbody = 0, ngeom = 0;
    uint32_t nvis = 0, nlights = 0, tiles_x = 0, tiles_y = 0, ntiles = 0;
    uint32_t img_w = 0, img_h = 0;
    float znear = 0.01f, zfar = 100.0f, lodk = 16.0f;
    int draw_calls = 0;
    uint32_t draw_count = 0;
    std::vector<float> clear_rgb = {0.2f, 0.3f, 0.4f};

    GpuBuf vbo, ibo, indirect, gtab, lights, camc, campose, texbuf, lodbuf;
    // tendon描画(空間tendonがあるモデルのみ): 線分ごとの円柱の姿勢(仮想geom)と、経路の表。
    GpuBuf tpos, tmat, wtab, ttab;
    uint32_t ntend = 0, nslot = 0;
    VkPipeline tend = VK_NULL_HANDLE;
    VkShaderModule m_tend = VK_NULL_HANDLE;
    VkImage img_color = VK_NULL_HANDLE, img_depth = VK_NULL_HANDLE, img_zbuf = VK_NULL_HANDLE;
    VkDeviceMemory mem_color = VK_NULL_HANDLE, mem_depth = VK_NULL_HANDLE, mem_zbuf = VK_NULL_HANDLE;
    VkImageView view_color = VK_NULL_HANDLE, view_depth = VK_NULL_HANDLE, view_zbuf = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline gfx = VK_NULL_HANDLE, comp = VK_NULL_HANDLE, cull = VK_NULL_HANDLE;
    VkShaderModule m_cull = VK_NULL_HANDLE, m_vert = VK_NULL_HANDLE, m_frag = VK_NULL_HANDLE, m_comp = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    GpuBuf rb_color, rb_depth;
    void *rb_color_ptr = nullptr, *rb_depth_ptr = nullptr;
    VkQueryPool qpool = VK_NULL_HANDLE;
    bool rendered = false, read_back = false;

    uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags props) const {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(h.physical, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
        throw std::runtime_error("mjmlx_camera: no suitable memory type");
    }

    GpuBuf make_buf(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
        GpuBuf b;
        b.size = std::max<size_t>(size, 4);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = b.size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(h.device, &bi, nullptr, &b.buf), "vkCreateBuffer");
        VkMemoryRequirements rq;
        vkGetBufferMemoryRequirements(h.device, b.buf, &rq);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = rq.size;
        // 読み出し用のホスト可視メモリはuncachedだと遅い(CPU読みが桁違いに遅くなる)ためcachedを優先する。
        try { ai.memoryTypeIndex = find_mem(rq.memoryTypeBits, props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT); } catch (...) { ai.memoryTypeIndex = find_mem(rq.memoryTypeBits, props); }
        vk_check(vkAllocateMemory(h.device, &ai, nullptr, &b.mem), "vkAllocateMemory(buffer)");
        vkBindBufferMemory(h.device, b.buf, b.mem, 0);
        return b;
    }

    void destroy(GpuBuf& b) {
        if (b.buf) vkDestroyBuffer(h.device, b.buf, nullptr);
        if (b.mem) vkFreeMemory(h.device, b.mem, nullptr);
        b = {};
    }

    void one_shot(const std::function<void(VkCommandBuffer)>& rec) {
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = h.command_pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(h.device, &ca, &cmd);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        rec(cmd);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vk_check(vkQueueSubmit(h.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        vkQueueWaitIdle(h.queue);
        vkFreeCommandBuffers(h.device, h.command_pool, 1, &cmd);
    }

    GpuBuf make_device_buf(const void* data, size_t size, VkBufferUsageFlags usage) {
        GpuBuf dst = make_buf(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        GpuBuf st = make_buf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* p;
        vkMapMemory(h.device, st.mem, 0, VK_WHOLE_SIZE, 0, &p);
        std::memcpy(p, data, size);
        vkUnmapMemory(h.device, st.mem);
        one_shot([&](VkCommandBuffer c) { VkBufferCopy r{0, 0, size}; vkCmdCopyBuffer(c, st.buf, dst.buf, 1, &r); });
        destroy(st);
        return dst;
    }

    void make_image(VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {img_w, img_h, 1};
        ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = usage; ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vk_check(vkCreateImage(h.device, &ii, nullptr, &img), "vkCreateImage (画像サイズかVRAMが上限を超えている可能性)");
        VkMemoryRequirements rq;
        vkGetImageMemoryRequirements(h.device, img, &rq);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = rq.size; ai.memoryTypeIndex = find_mem(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk_check(vkAllocateMemory(h.device, &ai, nullptr, &mem), "vkAllocateMemory(image)");
        vkBindImageMemory(h.device, img, mem, 0);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange = {aspect, 0, 1, 0, 1};
        vk_check(vkCreateImageView(h.device, &vi, nullptr, &view), "vkCreateImageView");
    }

    VkShaderModule make_module(const std::vector<uint32_t>& code) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = code.size() * 4; mi.pCode = code.data();
        VkShaderModule m;
        vk_check(vkCreateShaderModule(h.device, &mi, nullptr, &m), "vkCreateShaderModule");
        return m;
    }

    void build(const mjModel* m);

    ~MjmlxCameraRenderer() {
        if (!h.device) return;
        vkDeviceWaitIdle(h.device);
        if (qpool) vkDestroyQueryPool(h.device, qpool, nullptr);
        if (h.command_pool) vkDestroyCommandPool(h.device, h.command_pool, nullptr);
        for (auto* b : {&vbo, &ibo, &indirect, &gtab, &lights, &camc, &campose, &texbuf, &lodbuf, &tpos, &tmat, &wtab, &ttab, &rb_color, &rb_depth}) destroy(*b);
        if (dpool) vkDestroyDescriptorPool(h.device, dpool, nullptr);
        if (gfx) vkDestroyPipeline(h.device, gfx, nullptr);
        if (comp) vkDestroyPipeline(h.device, comp, nullptr);
        if (cull) vkDestroyPipeline(h.device, cull, nullptr);
        if (tend) vkDestroyPipeline(h.device, tend, nullptr);
        if (pipe_layout) vkDestroyPipelineLayout(h.device, pipe_layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(h.device, set_layout, nullptr);
        for (auto sm : {m_vert, m_frag, m_comp, m_cull, m_tend}) if (sm) vkDestroyShaderModule(h.device, sm, nullptr);
        if (fb) vkDestroyFramebuffer(h.device, fb, nullptr);
        if (render_pass) vkDestroyRenderPass(h.device, render_pass, nullptr);
        for (auto v : {view_color, view_depth, view_zbuf}) if (v) vkDestroyImageView(h.device, v, nullptr);
        for (auto i : {img_color, img_depth, img_zbuf}) if (i) vkDestroyImage(h.device, i, nullptr);
        for (auto mm : {mem_color, mem_depth, mem_zbuf}) if (mm) vkFreeMemory(h.device, mm, nullptr);
    }
};

void MjmlxCameraRenderer::build(const mjModel* m) {
    ngeom = m->ngeom; nbody = m->nbody;
    if (m->ncam == 0) throw std::runtime_error("mjmlx_camera: model has no <camera>");
    if (ngeom == 0) throw std::runtime_error("mjmlx_camera: model has no geoms");
    if (!h.multi_draw_indirect) throw std::runtime_error("mjmlx_camera: device lacks multiDrawIndirect/drawIndirectFirstInstance");
    if (!h.shader_clip_distance) throw std::runtime_error("mjmlx_camera: device lacks shaderClipDistance");
    ncam = m->ncam;

    // 静的シーン(qpos0)のカメラ/ライト定数。
    mjData* d = mj_makeData(m);
    mj_forward(m, d);

    // 可視geom(group 0-2)の形状を共有VB/IBへ。meshはdataid単位で重複排除。
    std::vector<Vtx> vtx;
    std::vector<uint32_t> idx;
    struct Range { uint32_t first, count; int32_t voff; };
    struct LodSet { Range lod[4]; float sphere[4]; };
    std::vector<LodSet> ranges;
    std::vector<GeomRec> grecs;
    std::map<int, LodSet> mesh_cache;
    const char* lod_env = std::getenv("MJMLX_CAM_LOD");
    const bool use_lod = !(lod_env && std::atoi(lod_env) == 0);
    std::map<int, uint32_t> tex_off;
    std::vector<uint32_t> texels;
    for (int g = 0; g < m->ngeom; g++) {
        if (m->geom_group[g] > 2) continue;
        LodSet r;
        bool is_mesh = m->geom_type[g] == mjGEOM_MESH;
        int mid = m->geom_matid[g];
        int tid = (mid >= 0 && (m->geom_type[g] == mjGEOM_PLANE || m->geom_type[g] == mjGEOM_BOX || m->geom_type[g] == mjGEOM_HFIELD)) ? m->mat_texid[mid * mjNTEXROLE + mjTEXROLE_RGB] : -1;
        if (tid >= 0 && m->tex_type[tid] != mjTEXTURE_2D) tid = -1;
        auto cached = is_mesh ? mesh_cache.find(m->geom_dataid[g]) : mesh_cache.end();
        if (cached != mesh_cache.end()) r = cached->second;
        else {
            float s[3] = {(float)m->geom_size[3 * g], (float)m->geom_size[3 * g + 1], (float)m->geom_size[3 * g + 2]};
            Mesh me = is_mesh ? make_mesh(m, m->geom_dataid[g]) : m->geom_type[g] == mjGEOM_HFIELD ? make_hfield(m, m->geom_dataid[g]) : make_primitive(m->geom_type[g], s);
            if (me.i.empty()) continue;
            if (tid >= 0) {
                bool hf = m->geom_type[g] == mjGEOM_HFIELD;
                float hx = hf ? (float)m->hfield_size[4 * m->geom_dataid[g]] : (s[0] > 0 ? s[0] : 100.0f), hy = hf ? (float)m->hfield_size[4 * m->geom_dataid[g] + 1] : (s[1] > 0 ? s[1] : 100.0f);
                apply_planar_uv(me, hx, hy, &m->mat_texrepeat[2 * mid], m->mat_texuniform[mid] != 0);
            }
            if (!is_mesh) orient_outward(me);
            auto append = [&](const Mesh& mm) {
                Range rg{(uint32_t)idx.size(), (uint32_t)mm.i.size(), (int32_t)vtx.size()};
                vtx.insert(vtx.end(), mm.v.begin(), mm.v.end());
                idx.insert(idx.end(), mm.i.begin(), mm.i.end());
                return rg;
            };
            r.lod[0] = append(me);
            for (int l = 1; l < 4; l++) {
                // 小さいメッシュと非meshは簡略化しない(LOD0と同じ範囲)。
                if (is_mesh && use_lod && me.i.size() > 768) r.lod[l] = append(lod_for_ratio(me, std::pow(0.25, l)));
                else r.lod[l] = r.lod[0];
            }
            float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
            for (auto& v : me.v) for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], v.p[k]); hi[k] = std::max(hi[k], v.p[k]); }
            float rad = 0;
            for (int k = 0; k < 3; k++) r.sphere[k] = 0.5f * (lo[k] + hi[k]);
            for (auto& v : me.v) { float dd = 0; for (int k = 0; k < 3; k++) dd += (v.p[k] - r.sphere[k]) * (v.p[k] - r.sphere[k]); rad = std::max(rad, dd); }
            r.sphere[3] = std::sqrt(rad) * 1.001f;
            if (is_mesh) mesh_cache[m->geom_dataid[g]] = r;
        }
        ranges.push_back(r);
        GeomRec gr{};
        const float* rgba = &m->geom_rgba[4 * g];
        bool def = rgba[0] == 0.5f && rgba[1] == 0.5f && rgba[2] == 0.5f && rgba[3] == 1.0f;
        const float* use = (mid >= 0 && def) ? &m->mat_rgba[4 * mid] : rgba;
        for (int k = 0; k < 4; k++) gr.rgba[k] = use[k];
        gr.ms[0] = mid >= 0 ? m->mat_specular[mid] : 0.5f;
        gr.ms[1] = mid >= 0 ? m->mat_shininess[mid] : 0.5f;
        gr.ids[0] = (uint32_t)g;
        if (tid >= 0) {
            if (!tex_off.count(tid)) {
                int tw = m->tex_width[tid], th = m->tex_height[tid], nch = m->tex_nchannel[tid];
                tex_off[tid] = (uint32_t)texels.size();
                const unsigned char* src = m->tex_data + m->tex_adr[tid];
                for (int k = 0; k < tw * th; k++) {
                    const unsigned char* q = src + k * nch;
                    texels.push_back(q[0] | (uint32_t)q[nch > 1 ? 1 : 0] << 8 | (uint32_t)q[nch > 2 ? 2 : 0] << 16 | 0xff000000u);
                }
            }
            gr.ids[1] = 1; gr.ids[2] = tex_off[tid]; gr.ids[3] = ((uint32_t)m->tex_width[tid] << 16) | (uint32_t)m->tex_height[tid];
        }
        grecs.push_back(gr);
    }
    const uint32_t nvis_geom = (uint32_t)ranges.size();
    // 空間tendon(site/geom wrapを含む)を描画対象にする。固定tendonは形が無いので描かない(MuJoCoと同じ)。
    std::vector<WrapRec> wrecs;
    std::vector<TendRec> trecs;
    std::vector<int> slot_tendon;  // スロット -> tendon
    std::vector<int> tendon_ids;   // 描画するtendonの通し番号 -> モデルのtendon id
    for (int t = 0; t < m->ntendon; t++) {
        if (m->tendon_group[t] > 2) continue;
        const int adr = m->tendon_adr[t], num = m->tendon_num[t];
        if (m->wrap_type[adr] == mjWRAP_JOINT) continue;
        TendRec tr{};
        tr.a[0] = (int)wrecs.size(); tr.a[1] = num; tr.a[2] = (int)slot_tendon.size();
        for (int k = 0; k < num; k++) {
            const int type = m->wrap_type[adr + k], obj = m->wrap_objid[adr + k];
            WrapRec w{};
            w.a[0] = type; w.a[1] = obj;
            if (type == mjWRAP_SITE) {
                w.a[3] = m->site_bodyid[obj];
                for (int q = 0; q < 3; q++) w.b[q] = (float)m->site_pos[3 * obj + q];
            } else if (type == mjWRAP_SPHERE || type == mjWRAP_CYLINDER) {
                const int side = (int)std::lround(m->wrap_prm[adr + k]);
                w.b[0] = (float)m->geom_size[3 * obj];
                w.a[2] = side >= 0;
                if (side >= 0) {
                    w.c[0] = m->site_bodyid[side];
                    for (int q = 0; q < 3; q++) w.d[q] = (float)m->site_pos[3 * side + q];
                }
            }
            wrecs.push_back(w);
        }
        int slots = 0;
        for (int j = 0; j < num - 1;) {
            const int t0 = m->wrap_type[adr + j], t1 = m->wrap_type[adr + j + 1];
            if (t0 == mjWRAP_PULLEY || t1 == mjWRAP_PULLEY) { j++; continue; }
            if (t1 == mjWRAP_SPHERE || t1 == mjWRAP_CYLINDER) { slots += 3; j += 2; }
            else { slots += 1; j += 1; }
        }
        tr.a[3] = slots;
        tr.col[0] = (float)m->tendon_width[t];
        for (int q = 0; q < slots; q++) slot_tendon.push_back((int)trecs.size());
        trecs.push_back(tr);
        tendon_ids.push_back(t);
    }
    ntend = (uint32_t)trecs.size();
    nslot = (uint32_t)slot_tendon.size();
    if (nslot > 0) {
        // 単位円柱(半径1・半長1)。姿勢行列の列に寸法を掛けて使う。
        const float one[3] = {1.0f, 1.0f, 0.0f};
        Mesh cyl = make_primitive(mjGEOM_CYLINDER, one);
        orient_outward(cyl);
        auto add_unit = [&](const Mesh& mm) {
            LodSet r{};
            Range rg{(uint32_t)idx.size(), (uint32_t)mm.i.size(), (int32_t)vtx.size()};
            vtx.insert(vtx.end(), mm.v.begin(), mm.v.end());
            idx.insert(idx.end(), mm.i.begin(), mm.i.end());
            for (int l = 0; l < 4; l++) r.lod[l] = rg;
            r.sphere[3] = 1e6f;
            return r;
        };
        const LodSet rc = add_unit(cyl);
        auto tend_rec = [&](int slot, int tendon) {
            GeomRec gr{};
            const int t = tendon_ids[tendon];
            const int mid = m->tendon_matid[t];
            const float* rgba = &m->tendon_rgba[4 * t];
            const bool def = rgba[0] == 0.5f && rgba[1] == 0.5f && rgba[2] == 0.5f && rgba[3] == 1.0f;
            const float* use = (mid >= 0 && def) ? &m->mat_rgba[4 * mid] : rgba;
            for (int k = 0; k < 4; k++) gr.rgba[k] = use[k];
            gr.ms[0] = mid >= 0 ? m->mat_specular[mid] : 0.5f;
            gr.ms[1] = mid >= 0 ? m->mat_shininess[mid] : 0.5f;
            gr.ids[0] = (uint32_t)m->ngeom + (uint32_t)slot;
            return gr;
        };
        for (uint32_t q = 0; q < nslot; q++) { ranges.push_back(rc); grecs.push_back(tend_rec((int)q, slot_tendon[q])); }
    }
    nvis = (uint32_t)ranges.size();
    (void)nvis_geom;
    if (std::getenv("MJMLX_CAM_PROF")) {
        uint64_t tri = 0;
        for (auto& rg : ranges) tri += rg.lod[0].count / 3;
        fprintf(stderr, "[cam] visible geoms=%u triangles/env=%llu vertices(shared)=%zu\n", nvis, (unsigned long long)tri, vtx.size());
    }
    if (nvis == 0) throw std::runtime_error("mjmlx_camera: no visible geoms (group 0-2)");

    // 全(env,cam)x可視geomのindirectコマンド。instance indexで(env,cam,geom)を引く。
    ntiles = (uint32_t)B * ncam;
    draw_count = ntiles * nvis;
    std::vector<LodRec> lods(nvis);
    for (uint32_t v = 0; v < nvis; v++) {
        for (int k = 0; k < 4; k++) lods[v].sphere[k] = ranges[v].sphere[k];
        for (int l = 0; l < 4; l++) {
            lods[v].lod[l][0] = ranges[v].lod[l].first; lods[v].lod[l][1] = ranges[v].lod[l].count; lods[v].lod[l][2] = (uint32_t)ranges[v].lod[l].voff;
        }
    }

    // タイル配置(ほぼ正方形)。
    tiles_x = (uint32_t)std::ceil(std::sqrt((double)ntiles));
    tiles_y = (ntiles + tiles_x - 1) / tiles_x;
    img_w = tiles_x * W; img_h = tiles_y * H;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(h.physical, &props);
    if (img_w > props.limits.maxImageDimension2D || img_h > props.limits.maxImageDimension2D || img_w > props.limits.maxFramebufferWidth || img_h > props.limits.maxFramebufferHeight)
        throw std::runtime_error("mjmlx_camera: tiled image " + std::to_string(img_w) + "x" + std::to_string(img_h) + " exceeds device limit; reduce num_envs*cameras or resolution");
    if (draw_count > props.limits.maxDrawIndirectCount) throw std::runtime_error("mjmlx_camera: draw count exceeds maxDrawIndirectCount");

    // 光源: 静的(qpos0時点)。headlightはカメラ向きの指向性光源として先頭に入れる。
    std::vector<LightRec> lrecs;
    if (m->vis.headlight.active) {
        LightRec l{};
        l.pos[3] = 2;
        for (int k = 0; k < 3; k++) { l.diffuse[k] = m->vis.headlight.diffuse[k]; l.ambient[k] = m->vis.headlight.ambient[k]; l.specular[k] = m->vis.headlight.specular[k]; }
        lrecs.push_back(l);
    }
    for (int i = 0; i < m->nlight; i++) {
        if (!m->light_active[i]) continue;
        LightRec l{};
        for (int k = 0; k < 3; k++) {
            l.pos[k] = (float)d->light_xpos[3 * i + k]; l.dir[k] = (float)d->light_xdir[3 * i + k];
            l.diffuse[k] = m->light_diffuse[3 * i + k]; l.ambient[k] = m->light_ambient[3 * i + k]; l.specular[k] = m->light_specular[3 * i + k];
        }
        l.pos[3] = m->light_type[i] == mjLIGHT_DIRECTIONAL ? 1.0f : 0.0f;
        lrecs.push_back(l);
    }
    nlights = (uint32_t)lrecs.size();
    if (lrecs.empty()) lrecs.push_back({});

    // カメラ定数: body局所の相対姿勢(fixed/target)とqpos0での世界姿勢(track)。
    std::vector<CamConst> ccs(ncam);
    for (int c = 0; c < ncam; c++) {
        int b = m->cam_bodyid[c];
        const mjtNum* Rb = d->xmat + 9 * b;
        const mjtNum* Rc = d->cam_xmat + 9 * c;
        double dp[3];
        for (int k = 0; k < 3; k++) dp[k] = d->cam_xpos[3 * c + k] - d->xpos[3 * b + k];
        CamConst& cc = ccs[c];
        std::memset(&cc, 0, sizeof(cc));
        for (int i = 0; i < 3; i++) {
            cc.a[i] = (float)(Rb[i] * dp[0] + Rb[3 + i] * dp[1] + Rb[6 + i] * dp[2]); // Rb^T dp
            cc.b[i] = (float)dp[i];
            for (int j = 0; j < 3; j++) {
                double s = 0;
                for (int k = 0; k < 3; k++) s += Rb[3 * k + i] * Rc[3 * k + j]; // (Rb^T Rc)[i][j]
                cc.rl[i][j] = (float)s;
                cc.rw[i][j] = (float)Rc[3 * i + j];
            }
        }
        cc.a[3] = (float)m->cam_mode[c];
        cc.b[3] = (float)(1.0 / std::tan(m->cam_fovy[c] * M_PI / 360.0));
        cc.ids[0] = b;
        cc.ids[1] = m->cam_targetbodyid[c];
    }
    mj_deleteData(d);

    vbo = make_device_buf(vtx.data(), vtx.size() * sizeof(Vtx), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    ibo = make_device_buf(idx.data(), idx.size() * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    indirect = make_buf((size_t)draw_count * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lodbuf = make_device_buf(lods.data(), lods.size() * sizeof(LodRec), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (texels.empty()) texels.push_back(0);
    texbuf = make_device_buf(texels.data(), texels.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gtab = make_device_buf(grecs.data(), grecs.size() * sizeof(GeomRec), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lights = make_device_buf(lrecs.data(), lrecs.size() * sizeof(LightRec), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    camc = make_device_buf(ccs.data(), ccs.size() * sizeof(CamConst), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    campose = make_buf((size_t)ntiles * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // tendon用(無いモデルでも束縛は必要なのでダミーを置く)。
    if (wrecs.empty()) wrecs.push_back({});
    if (trecs.empty()) trecs.push_back({});
    wtab = make_device_buf(wrecs.data(), wrecs.size() * sizeof(WrapRec), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ttab = make_device_buf(trecs.data(), trecs.size() * sizeof(TendRec), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    tpos = make_buf((size_t)std::max<uint32_t>(nslot, 1) * 3 * B * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tmat = make_buf((size_t)std::max<uint32_t>(nslot, 1) * 9 * B * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // 画像: RGBA8 + 線形depth(R32F、ピクセル単位の正確な値) + z-test用D32。
    make_image(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, img_color, mem_color, view_color);
    make_image(VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, img_depth, mem_depth, view_depth);
    make_image(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, img_zbuf, mem_zbuf, view_zbuf);

    VkAttachmentDescription att[3]{};
    VkFormat fmts[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT, VK_FORMAT_D32_SFLOAT};
    for (int i = 0; i < 3; i++) {
        att[i].format = fmts[i]; att[i].samples = VK_SAMPLE_COUNT_1_BIT;
        att[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[i].storeOp = i < 2 ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att[i].finalLayout = i < 2 ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    VkAttachmentReference cref[2] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
    VkAttachmentReference dref{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sub.colorAttachmentCount = 2; sub.pColorAttachments = cref; sub.pDepthStencilAttachment = &dref;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 3; rpi.pAttachments = att; rpi.subpassCount = 1; rpi.pSubpasses = &sub; rpi.dependencyCount = 2; rpi.pDependencies = deps;
    vk_check(vkCreateRenderPass(h.device, &rpi, nullptr, &render_pass), "vkCreateRenderPass");
    VkImageView views[3] = {view_color, view_depth, view_zbuf};
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = render_pass; fbi.attachmentCount = 3; fbi.pAttachments = views; fbi.width = img_w; fbi.height = img_h; fbi.layers = 1;
    vk_check(vkCreateFramebuffer(h.device, &fbi, nullptr, &fb), "vkCreateFramebuffer");

    // ディスクリプタ: 0 geom_xpos, 1 geom_xmat, 2 campose, 3 geom表, 4 光源, 5 body_xpos, 6 body_xmat, 7 カメラ定数。
    VkDescriptorSetLayoutBinding lb[15];
    for (uint32_t i = 0; i < 15; i++) lb[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 15; sli.pBindings = lb;
    vk_check(vkCreateDescriptorSetLayout(h.device, &sli, nullptr, &set_layout), "vkCreateDescriptorSetLayout");
    VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushC)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
    vk_check(vkCreatePipelineLayout(h.device, &pli, nullptr, &pipe_layout), "vkCreatePipelineLayout");
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 15};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(h.device, &dpi, nullptr, &dpool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &set_layout;
    vk_check(vkAllocateDescriptorSets(h.device, &dai, &dset), "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo bi[3] = {{campose.buf, 0, VK_WHOLE_SIZE}, {gtab.buf, 0, VK_WHOLE_SIZE}, {lights.buf, 0, VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo bc{camc.buf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo bt{texbuf.buf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo bl{lodbuf.buf, 0, VK_WHOLE_SIZE}, bic{indirect.buf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo btp{tpos.buf, 0, VK_WHOLE_SIZE}, btm{tmat.buf, 0, VK_WHOLE_SIZE}, bwt{wtab.buf, 0, VK_WHOLE_SIZE}, btt{ttab.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[11]{};
    uint32_t bind[11] = {2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14};
    const VkDescriptorBufferInfo* infos[11] = {&bi[0], &bi[1], &bi[2], &bc, &bt, &bl, &bic, &btp, &btm, &bwt, &btt};
    for (int i = 0; i < 11; i++) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = dset; w[i].dstBinding = bind[i]; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = infos[i];
    }
    vkUpdateDescriptorSets(h.device, 11, w, 0, nullptr);

    // パイプライン。
    const bool has_tend = nslot > 0;
    m_comp = make_module(compile_glsl(with_common(kCamComp), "comp"));
    if (has_tend) m_tend = make_module(compile_glsl(tendon_comp_source(), "comp"));
    m_cull = make_module(compile_glsl(with_common(has_tend ? tendon_variant(kCull, true) : std::string(kCull), has_tend), "comp"));
    VkComputePipelineCreateInfo cpi2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi2.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m_cull, "main", nullptr};
    cpi2.layout = pipe_layout;
    vk_check(vkCreateComputePipelines(h.device, VK_NULL_HANDLE, 1, &cpi2, nullptr, &cull), "vkCreateComputePipelines(cull)");
    if (has_tend) {
        VkComputePipelineCreateInfo cpt{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpt.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m_tend, "main", nullptr};
        cpt.layout = pipe_layout;
        vk_check(vkCreateComputePipelines(h.device, VK_NULL_HANDLE, 1, &cpt, nullptr, &tend), "vkCreateComputePipelines(tendon)");
    }
    m_vert = make_module(compile_glsl(with_common(has_tend ? tendon_variant(kVert, false) : std::string(kVert), has_tend), "vert"));
    m_frag = make_module(compile_glsl(with_common(kFrag), "frag"));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m_comp, "main", nullptr};
    cpi.layout = pipe_layout;
    vk_check(vkCreateComputePipelines(h.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &comp), "vkCreateComputePipelines");

    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, m_vert, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, m_frag, "main", nullptr}};
    VkVertexInputBindingDescription vb{0, sizeof(Vtx), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription va[3] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}, {2, 0, VK_FORMAT_R32G32_SFLOAT, 24}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb; vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, (float)img_w, (float)img_h, 0, 1};
    VkRect2D sc{{0, 0}, {img_w, img_h}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.rasterizerDiscardEnable = (std::getenv("MJMLX_CAM_EXP") && std::string(std::getenv("MJMLX_CAM_EXP")) == "discard") ? VK_TRUE : VK_FALSE; // 計測用: 頂点処理のみの時間
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cba[2]{};
    cba[0].colorWriteMask = cba[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 2; cb.pAttachments = cba;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = stages; gpi.pVertexInputState = &vi; gpi.pInputAssemblyState = &ia; gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &ds; gpi.pColorBlendState = &cb;
    gpi.layout = pipe_layout; gpi.renderPass = render_pass; gpi.subpass = 0;
    vk_check(vkCreateGraphicsPipelines(h.device, VK_NULL_HANDLE, 1, &gpi, nullptr, &gfx), "vkCreateGraphicsPipelines");
}

// マップしたGPU可視メモリの読み出しは単一スレッドだと約1GB/sしか出ないので、複数スレッドで分担する。
template <class F> static void parallel_chunks(size_t n, F fn) {
    unsigned nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    if (n < (1u << 16)) nt = 1;
    std::vector<std::thread> th;
    for (unsigned t = 1; t < nt; t++) th.emplace_back([=] { fn(n * t / nt, n * (t + 1) / nt); });
    fn(0, n / nt);
    for (auto& x : th) x.join();
}

// ── C API ────────────────────────────────────────────────────────────────────

extern "C" {

MJMLX_API MjmlxCameraRenderer* mjmlx_camera_create(MjmlxBatchedSim* sim, int width, int height) {
    if (!sim || sim->cpu_mode || !sim->sim.kinematics || !sim->mj_model || width <= 0 || height <= 0) {
        fprintf(stderr, "mjmlx_camera_create: GPU batched simのみ対応\n");
        return nullptr;
    }
    auto r = std::make_unique<MjmlxCameraRenderer>();
    try {
        r->sim = &sim->sim;
        r->B = sim->sim.num_envs; r->W = width; r->H = height;
        mjmlx::camera_prepare_device();
        if (!g_dev.valid) throw std::runtime_error("camera unsupported on this Vulkan setup: " + g_dev.reason);
        r->h = g_dev;
        if (const char* e = std::getenv("MJMLX_CAM_LOD_K")) r->lodk = (float)std::atof(e);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi.queueFamilyIndex = g_dev.queue_family;
        vk_check(vkCreateCommandPool(g_dev.device, &pi, nullptr, &r->h.command_pool), "vkCreateCommandPool");
        r->build(sim->mj_model);
        return r.release();
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_camera_create error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API void mjmlx_camera_free(MjmlxCameraRenderer* r) { delete r; }

MJMLX_API int mjmlx_camera_count(const MjmlxCameraRenderer* r) { return r ? r->ncam : 0; }

MJMLX_API int mjmlx_camera_draw_call_count(const MjmlxCameraRenderer* r) { return r ? r->draw_calls : 0; }

MJMLX_API int mjmlx_camera_render(MjmlxCameraRenderer* r) {
    if (!r) return -1;
    try {
        auto& s = *r->sim;
        auto t_a = std::chrono::steady_clock::now();
        auto kin = s.kinematics(s.qpos);
        std::vector<mkx::NodePtr<VB>> roots;
        for (auto& k : kin) roots.push_back(k.node());
        mkx::eval_nodes<VB>(roots);
        mkx::VulkanBackend::wait_idle();
        auto t_b = std::chrono::steady_clock::now();
        auto buf = [&](int i) { return mkx::buffer_for<VB>(kin[i].node())->buffer; };

        VkDescriptorBufferInfo bi[4] = {{buf(7), 0, VK_WHOLE_SIZE}, {buf(8), 0, VK_WHOLE_SIZE}, {buf(0), 0, VK_WHOLE_SIZE}, {buf(2), 0, VK_WHOLE_SIZE}};
        uint32_t bind[4] = {0, 1, 5, 6};
        VkWriteDescriptorSet w[4]{};
        for (int i = 0; i < 4; i++) {
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = r->dset; w[i].dstBinding = bind[i]; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(r->h.device, 4, w, 0, nullptr);

        PushC pc{(uint32_t)r->B, (uint32_t)r->ncam, r->nvis, r->tiles_x, r->tiles_y, r->nlights, (uint32_t)r->nbody, (uint32_t)r->H, (float)r->W / (float)r->H, r->znear, r->zfar, r->lodk, (uint32_t)r->ngeom, r->ntend, r->nslot};
        r->draw_calls = 0;
        const bool prof = std::getenv("MJMLX_CAM_PROF") != nullptr;
        if (prof && !r->qpool) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = 4;
            vk_check(vkCreateQueryPool(r->h.device, &qi, nullptr, &r->qpool), "vkCreateQueryPool");
        }
        r->one_shot([&](VkCommandBuffer c) {
            auto stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
            if (prof) { vkCmdResetQueryPool(c, r->qpool, 0, 4); vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r->qpool, 0); }
            vkCmdPushConstants(c, r->pipe_layout, stages, 0, sizeof(pc), &pc);
            vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe_layout, 0, 1, &r->dset, 0, nullptr);
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, r->comp);
            vkCmdDispatch(c, (r->ntiles + 63) / 64, 1, 1);
            if (r->nslot > 0) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, r->tend);
                vkCmdDispatch(c, ((uint32_t)r->B * r->ntend + 63) / 64, 1, 1);
            }
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, r->cull);
            vkCmdDispatch(c, (r->draw_count + 63) / 64, 1, 1);
            VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb2.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);
            if (prof) vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r->qpool, 1);
            VkClearValue cv[3]{};
            for (int k = 0; k < 3; k++) cv[0].color.float32[k] = r->clear_rgb[k];
            cv[0].color.float32[3] = 1.0f;
            cv[1].color.float32[0] = r->zfar;
            cv[2].depthStencil = {1.0f, 0};
            VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rb.renderPass = r->render_pass; rb.framebuffer = r->fb; rb.renderArea = {{0, 0}, {r->img_w, r->img_h}}; rb.clearValueCount = 3; rb.pClearValues = cv;
            vkCmdBeginRenderPass(c, &rb, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, r->gfx);
            vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipe_layout, 0, 1, &r->dset, 0, nullptr);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(c, 0, 1, &r->vbo.buf, &off);
            vkCmdBindIndexBuffer(c, r->ibo.buf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexedIndirect(c, r->indirect.buf, 0, r->draw_count, sizeof(VkDrawIndexedIndirectCommand));
            r->draw_calls++;
            vkCmdEndRenderPass(c);
            if (prof) vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r->qpool, 2);
        });
        if (std::getenv("MJMLX_CAM_PROF")) {
            auto t_c = std::chrono::steady_clock::now();
            uint64_t ts[3] = {0, 0, 0};
            vkGetQueryPoolResults(r->h.device, r->qpool, 0, 3, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(r->h.physical, &pp);
            double ns = pp.limits.timestampPeriod;
            fprintf(stderr, "[cam] GPU: pose+cull_compute=%.3fms render_pass(clear+draw)=%.3fms\n", (ts[1] - ts[0]) * ns * 1e-6, (ts[2] - ts[1]) * ns * 1e-6);
            fprintf(stderr, "[cam] kin=%.2fms gpu(compute+draw)=%.2fms\n", std::chrono::duration<double, std::milli>(t_b - t_a).count(), std::chrono::duration<double, std::milli>(t_c - t_b).count());
        }
        r->rendered = true; r->read_back = false;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_camera_render error: %s\n", e.what());
        return -1;
    }
}

// 画像をタイル単位で (env,cam,H,W) の連続バッファへコピーして読み出す(並べ替えはGPUのvkCmdCopyImageToBuffer領域で行う)。
static void camera_readback(MjmlxCameraRenderer* r) {
    if (r->read_back) return;
    size_t npx = (size_t)r->ntiles * r->W * r->H;
    if (!r->rb_color.buf) {
        r->rb_color = r->make_buf(npx * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        r->rb_depth = r->make_buf(npx * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        // 毎回mapするとページフォルトで遅いので常時マップしておく。
        vk_check(vkMapMemory(r->h.device, r->rb_color.mem, 0, VK_WHOLE_SIZE, 0, &r->rb_color_ptr), "vkMapMemory");
        vk_check(vkMapMemory(r->h.device, r->rb_depth.mem, 0, VK_WHOLE_SIZE, 0, &r->rb_depth_ptr), "vkMapMemory");
    }
    std::vector<VkBufferImageCopy> regs(r->ntiles);
    for (uint32_t t = 0; t < r->ntiles; t++) {
        regs[t] = {(VkDeviceSize)t * r->W * r->H * 4, (uint32_t)r->W, (uint32_t)r->H, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                   {(int32_t)((t % r->tiles_x) * r->W), (int32_t)((t / r->tiles_x) * r->H), 0}, {(uint32_t)r->W, (uint32_t)r->H, 1}};
    }
    auto tr0 = std::chrono::steady_clock::now();
    r->one_shot([&](VkCommandBuffer c) {
        vkCmdCopyImageToBuffer(c, r->img_color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r->rb_color.buf, r->ntiles, regs.data());
        vkCmdCopyImageToBuffer(c, r->img_depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r->rb_depth.buf, r->ntiles, regs.data());
    });
    auto tr1 = std::chrono::steady_clock::now();
    if (std::getenv("MJMLX_CAM_PROF")) fprintf(stderr, "[cam] readback: gpu_copy(submit+wait)=%.2fms\n", std::chrono::duration<double, std::milli>(tr1 - tr0).count());
    r->read_back = true;
}

MJMLX_API int mjmlx_camera_get_rgb(MjmlxCameraRenderer* r, uint8_t* out) {
    if (!r || !r->rendered || !out) return -1;
    try { camera_readback(r); } catch (const std::exception& e) { fprintf(stderr, "mjmlx_camera_get_rgb error: %s\n", e.what()); return -1; }
    size_t npx = (size_t)r->ntiles * r->W * r->H;
    const uint32_t* src = static_cast<const uint32_t*>(r->rb_color_ptr);
    parallel_chunks(npx, [=](size_t a, size_t b) { for (size_t i = a; i < b; i++) { uint32_t v = src[i]; out[3 * i] = (uint8_t)v; out[3 * i + 1] = (uint8_t)(v >> 8); out[3 * i + 2] = (uint8_t)(v >> 16); } });
    return 0;
}

MJMLX_API int mjmlx_camera_get_depth(MjmlxCameraRenderer* r, float* out) {
    if (!r || !r->rendered || !out) return -1;
    try { camera_readback(r); } catch (const std::exception& e) { fprintf(stderr, "mjmlx_camera_get_depth error: %s\n", e.what()); return -1; }
    const char* dsrc = static_cast<const char*>(r->rb_depth_ptr);
    parallel_chunks((size_t)r->ntiles * r->W * r->H * sizeof(float), [=](size_t a, size_t b) { std::memcpy(reinterpret_cast<char*>(out) + a, dsrc + a, b - a); });
    return 0;
}

} // extern "C"
