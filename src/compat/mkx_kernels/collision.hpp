#pragma once
// Vendored from 20niship/mlx-cross-platform tests/mujoco/collision.hpp at commit 3202339e12231f3a68f9ee67e0b6085a33c52286.

#include <cstdlib>
#include <sstream>
#include <string>

#include <mkx/ops/fast_kernel.hpp>

// MuJoCo-MLX-Cpp make_collision_source逐語GLSL移植。元コードもplane-mesh/mesh-meshの2組み合わせしか実装しておらず他のgeom型ペアはhas_contact=falseのまま、本体は形状非依存の固定テンプレートなので次元(NG/NUM_PAIRS)だけホスト側展開にした。

namespace mjmlx::mkx_kernels {

inline mkx::fast::Kernel<> make_collision_kernel(int ng, int npairs, int max_contacts_per_env = 128,
                                                int sdf_initpoints = 0, int sdf_iterations = 0, const float* pair_data_host = nullptr) {
  const int contact_stride = 8;

  std::string header = R"GLSL(
vec3 msl_cross(vec3 a, vec3 b) { return cross(a, b); }
float msl_dot(vec3 a, vec3 b) { return dot(a, b); }
float msl_len(vec3 a) { return length(a); }
vec3 msl_norm(vec3 a) {
    float l = msl_len(a);
    return l > 1e-12 ? a / l : vec3(0.0, 0.0, 1.0);
}

// 箱ローカル座標の点lcから箱への符号付き距離(外: 正、内: -最小貫入量)。球-箱/カプセル-箱の解析接触で使う。
float mjmlx_box_sdist(vec3 lc, vec3 bh) {
    vec3 q = vec3(clamp(lc.x,-bh.x,bh.x), clamp(lc.y,-bh.y,bh.y), clamp(lc.z,-bh.z,bh.z));
    float dd = msl_len(lc - q);
    if (dd > 1e-9) return dd;
    vec3 pen = vec3(bh.x-abs(lc.x), bh.y-abs(lc.y), bh.z-abs(lc.z));
    return -min(pen.x, min(pen.y, pen.z));
}

// Generic GJK/EPA support point so plane-X and X-Y contact detection work for any geom-type pair, not just mesh. MuJoCo local-frame convention: sphere size=(radius,_,_), capsule/cylinder size=(radius,halflength,_) axis along local +z, box size=(halfx,halfy,halfz), ellipsoid size=semi-axes.
vec3 mjmlx_support(int gtype, vec3 pos, float R[9], vec3 size,
                    int mesh_adr, int mesh_nverts, vec3 dir_world) {
    vec3 ld = vec3(R[0]*dir_world.x+R[3]*dir_world.y+R[6]*dir_world.z,
                   R[1]*dir_world.x+R[4]*dir_world.y+R[7]*dir_world.z,
                   R[2]*dir_world.x+R[5]*dir_world.y+R[8]*dir_world.z);
    vec3 lp;
    if (gtype == 2) { // sphere
        lp = msl_norm(ld) * size.x;
    } else if (gtype == 4) { // ellipsoid: support(d) = (a^2 dx, b^2 dy, c^2 dz) / |(a dx, b dy, c dz)|
        vec3 scaled = vec3(size.x*ld.x, size.y*ld.y, size.z*ld.z);
        float denom = msl_len(scaled);
        lp = (denom > 1e-12) ? vec3(size.x*scaled.x, size.y*scaled.y, size.z*scaled.z) / denom : vec3(0.0, 0.0, size.z);
    } else if (gtype == 3) { // capsule
        vec3 n = msl_norm(ld);
        float capz = (ld.z >= 0.0) ? size.y : -size.y;
        lp = vec3(n.x*size.x, n.y*size.x, capz + n.z*size.x);
    } else if (gtype == 5) { // cylinder: disk (xy) x interval (z)
        vec2 xy = vec2(ld.x, ld.y);
        float xyl = length(xy);
        vec2 xyn = (xyl > 1e-12) ? xy / xyl : vec2(1.0, 0.0);
        lp = vec3(xyn.x*size.x, xyn.y*size.x, (ld.z >= 0.0) ? size.y : -size.y);
    } else if (gtype == 6) { // box
        lp = vec3(ld.x >= 0.0 ? size.x : -size.x,
                  ld.y >= 0.0 ? size.y : -size.y,
                  ld.z >= 0.0 ? size.z : -size.z);
    } else if (gtype == 7) { // mesh: linear scan over vertices
        float bd = -1e30; int bi = 0;
        for (int vi = 0; vi < mesh_nverts; vi++) {
            vec3 v = vec3(mesh_verts[(mesh_adr+vi)*3], mesh_verts[(mesh_adr+vi)*3+1], mesh_verts[(mesh_adr+vi)*3+2]);
            float dd = msl_dot(v, ld);
            if (dd > bd) { bd = dd; bi = vi; }
        }
        lp = vec3(mesh_verts[(mesh_adr+bi)*3], mesh_verts[(mesh_adr+bi)*3+1], mesh_verts[(mesh_adr+bi)*3+2]);
    } else {
        lp = vec3(0.0, 0.0, 0.0);
    }
    return vec3(R[0]*lp.x+R[1]*lp.y+R[2]*lp.z+pos.x,
                R[3]*lp.x+R[4]*lp.y+R[5]*lp.z+pos.y,
                R[6]*lp.x+R[7]*lp.y+R[8]*lp.z+pos.z);
}
)GLSL" + std::string(R"GLSL(
// ── SDF-SDF (MuJoCo mjc_SDF 移植) ──
// sdf_info(geomごと20float)=[格子先頭, aabb中心(3), aabb半幅(3), 格子半幅(3), rbound, 軸ごとのセル数(3), 種別, gear属性(5)]。格子はgeomローカルの一様ボクセルで、格子外はクランプ点までの距離を足して外挿する。
float mjmlx_sdf_cell(int base, ivec3 n, ivec3 i) {
    return sdf_grid[base + (i.z * n.y + i.y) * n.x + i.x];
}
float mjmlx_sdf_dist_grid(int g, vec3 x);

// MuJoCo gearプラグイン(plugin/sdf/gear.cc)のdistance()の移植。属性=[alpha, diameter, teeth, thickness, innerdiameter]
float mjmlx_gear_mod(float x, float y) { return x - y * floor(x / y); }
float mjmlx_gear_smooth_union(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return b * (1.0 - h) + a * h - k * h * (1.0 - h);
}
float mjmlx_gear_dist(int o, vec3 p) {
    float alpha = sdf_info[o+15], D = sdf_info[o+16], N = sdf_info[o+17], thick = sdf_info[o+18], inner = sdf_info[o+19];
    const float PI = 3.14159265358979;
    float psi = 3.096e-5 * N * N - 6.557e-3 * N + 0.551;
    float R = D / 2.0;
    float rho = length(p.xy);
    float Pd = N / D;
    float P = PI / Pd;
    float a = 1.0 / Pd;
    float Ro = (D + 2.0 * a) / 2.0;
    float h = 2.2 / Pd;
    float innerR = Ro - h - 0.14 * D;
    if (inner >= 0.0) innerR = inner / 2.0;
    float d2;
    if (innerR - rho > 0.0) d2 = innerR - rho;
    else if (Ro - rho < -0.2) d2 = rho - Ro;
    else {
        float Rb = D * cos(psi) / 2.0;
        float fi = atan(p.y, p.x) + alpha;
        float alphaStride = P / R;
        float invAlpha = acos(Rb / R);
        float invPhi = tan(invAlpha) - invAlpha;
        float shift = alphaStride / 2.0 - 2.0 * invPhi;
        float fia = mjmlx_gear_mod(fi + shift / 2.0, alphaStride) - shift / 2.0;
        float fib = mjmlx_gear_mod(-fi - shift + shift / 2.0, alphaStride) - shift / 2.0;
        float dista = -1.0e6, distb = -1.0e6;
        if (Rb < rho) {
            float acr = acos(Rb / rho);
            float ta = sqrt(rho * rho - Rb * Rb);
            dista = ta - Rb * (fia + acr);
            distb = ta - Rb * (fib + acr);
        }
        float gearOuter = rho - Ro;
        float gearLowBase = rho - (Ro - h);
        float crownBase = rho - innerR;
        float cogs = max(dista, distb);
        float baseWalls = max(fia - (alphaStride - shift), fib - (alphaStride - shift));
        cogs = max(baseWalls, cogs);
        // smoothIntersection(gearOuter, cogs, k) = max(max(a,b), -smoothUnion(max(a,-b), max(b,-a), k))
        float k = 0.0035 * D;
        cogs = max(max(gearOuter, cogs), -mjmlx_gear_smooth_union(max(gearOuter, -cogs), max(cogs, -gearOuter), k));
        cogs = mjmlx_gear_smooth_union(gearLowBase, cogs, Rb - Ro + h);
        d2 = max(cogs, -crownBase);
    }
    vec2 w = vec2(d2, abs(p.z) - thick / 2.0);
    return min(max(w.x, w.y), 0.0) + length(max(w, vec2(0.0)));
}
float mjmlx_sdf_dist(int g, vec3 x) {
    int o = g * 20;
    if (sdf_info[o+14] > 0.5) return mjmlx_gear_dist(o, x);
    return mjmlx_sdf_dist_grid(g, x);
}
float mjmlx_sdf_dist_grid(int g, vec3 x) {
    int o = g * 20;
    int base = int(sdf_info[o]);
    ivec3 n = ivec3(sdf_info[o+11], sdf_info[o+12], sdf_info[o+13]);
    vec3 ctr = vec3(sdf_info[o+1], sdf_info[o+2], sdf_info[o+3]);
    vec3 gh = vec3(sdf_info[o+7], sdf_info[o+8], sdf_info[o+9]);
    vec3 xc = clamp(x, ctr - gh, ctr + gh);
    vec3 u = (xc - (ctr - gh)) / (2.0 * gh) * vec3(n - 1);
    ivec3 i0 = min(ivec3(floor(u)), (n - 2));
    vec3 f = u - vec3(i0);
    float c00 = mix(mjmlx_sdf_cell(base, n, i0), mjmlx_sdf_cell(base, n, i0 + ivec3(1,0,0)), f.x);
    float c10 = mix(mjmlx_sdf_cell(base, n, i0 + ivec3(0,1,0)), mjmlx_sdf_cell(base, n, i0 + ivec3(1,1,0)), f.x);
    float c01 = mix(mjmlx_sdf_cell(base, n, i0 + ivec3(0,0,1)), mjmlx_sdf_cell(base, n, i0 + ivec3(1,0,1)), f.x);
    float c11 = mix(mjmlx_sdf_cell(base, n, i0 + ivec3(0,1,1)), mjmlx_sdf_cell(base, n, i0 + ivec3(1,1,1)), f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z) + length(x - xc);
}
// 中心差分の勾配(刻みは1セル)
vec3 mjmlx_sdf_grad(int g, vec3 x) {
    int o = g * 20;
    bool exact = sdf_info[o+14] > 0.5;
    vec3 h = vec3(sdf_info[o+7], sdf_info[o+8], sdf_info[o+9]) * 2.0 / (vec3(ivec3(sdf_info[o+11], sdf_info[o+12], sdf_info[o+13])) - 1.0);
    if (exact) h = vec3(2e-4);
    return vec3(mjmlx_sdf_dist(g, x + vec3(h.x,0,0)) - mjmlx_sdf_dist(g, x - vec3(h.x,0,0)),
                mjmlx_sdf_dist(g, x + vec3(0,h.y,0)) - mjmlx_sdf_dist(g, x - vec3(0,h.y,0)),
                mjmlx_sdf_dist(g, x + vec3(0,0,h.z)) - mjmlx_sdf_dist(g, x - vec3(0,0,h.z))) / (2.0 * h);
}
float mjmlx_halton(int i, int b) {
    float f = 1.0, r = 0.0;
    int n = i;
    while (n > 0) { f /= float(b); r += f * float(n % b); n /= b; }
    return r;
}
// 評価点xはg2ローカル。g1ローカル座標は R21*x + p21。type 0=衝突(A+B+|max(A,B)|), 1=交差(max), 2=中間面(A-B)
float mjmlx_sdf_obj(int ga, int gb, mat3 R21, vec3 p21, vec3 x, int type) {
    float A = mjmlx_sdf_dist(ga, x);
    float B = mjmlx_sdf_dist(gb, R21 * x + p21);
    if (type == 0) return A + B + abs(max(A, B));
    if (type == 1) return max(A, B);
    return A - B;
}
vec3 mjmlx_sdf_obj_grad(int ga, int gb, mat3 R21, vec3 p21, vec3 x, int type) {
    vec3 y = R21 * x + p21;
    float A = mjmlx_sdf_dist(ga, x);
    float B = mjmlx_sdf_dist(gb, y);
    vec3 g1 = mjmlx_sdf_grad(ga, x);
    vec3 g2 = transpose(R21) * mjmlx_sdf_grad(gb, y);
    if (type == 0) return g1 + g2 + (A > B ? g1 : g2) * (max(A, B) > 0.0 ? 1.0 : -1.0);
    if (type == 1) return A > B ? g1 : g2;
    return msl_norm(msl_norm(g1) - msl_norm(g2));
}
// 勾配降下+バックトラッキング直線探索(stepGradient)。距離を返す
float mjmlx_sdf_descend(int ga, int gb, mat3 R21, vec3 p21, inout vec3 x, int type, int niter) {
    float dist = 1e10;
    for (int step = 0; step < niter; step++) {
        vec3 grad = mjmlx_sdf_obj_grad(ga, gb, R21, p21, x, type);
        if (any(isnan(grad)) || any(greaterThan(abs(grad), vec3(1e10)))) return 1e10;
        vec3 x0 = x;
        float dist0 = mjmlx_sdf_obj(ga, gb, R21, p21, x0, type);
        float alpha = 2.0;
        float wolfe = -0.1 * alpha * dot(grad, grad);
        do {
            alpha *= 0.5;
            wolfe *= 0.5;
            x = x0 - alpha * grad;
            dist = mjmlx_sdf_obj(ga, gb, R21, p21, x, type);
        } while (alpha > 1e-4 && dist - dist0 > wolfe);
        if (dist0 < dist) return dist;
    }
    return dist;
}
)GLSL");

  std::ostringstream ss;
  ss << "uint bid = gl_GlobalInvocationID.x;\n"
     << "uint NBATCH = gl_NumWorkGroups.x * gl_WorkGroupSize.x;\n"
     << "const int NG = " << ng << ";\n"
     << "const int NUM_PAIRS = " << npairs << ";\n"
     << "const int MAX_CON = " << max_contacts_per_env << ";\n"
     << "const int STRIDE = " << contact_stride << ";\n"
     << "const int SDF_INITPOINTS = " << sdf_initpoints << ";\n"
     << "const int SDF_ITERATIONS = " << sdf_iterations << ";\n\n"
     << "uint gx_off = bid * NG * 3;\n"
     << "uint gm_off = bid * NG * 9;\n"
     << "uint con_off = bid * MAX_CON * STRIDE;\n\n"
     // contact_dataの全消去はしない(1env 1024float × スレッド単位の散らばった書き込みが支配的だった)。solver/cfrcは contact_count 個しか読まない。
     << "int ncon = 0;\n\n"
     << "const int PAIR_STRIDE = 6;\n\n"
     // geomの位置は全ペアで繰り返し使うので、最初に1回だけ(独立なloadをまとめて発行して)ローカルへ読む
     << "vec3 GP[NG];\n"
     << "for (int g = 0; g < NG; g++) GP[g] = vec3(geom_xpos[uint(g*3)*NBATCH+bid], geom_xpos[uint(g*3+1)*NBATCH+bid], geom_xpos[uint(g*3+2)*NBATCH+bid]);\n\n";

  ss << R"GLSLBODY(
for (int p = 0; p < NUM_PAIRS; p++) {
    int pd = p * PAIR_STRIDE;
    PAIR_LOAD


    // (rbound判定の前に処理)
    // SDF同士: MuJoCo mjc_SDFと同じく、AABB交差内のHalton点から勾配降下で接触点を探す。接触は複数点を直接書き込む。SDFと他形状の組は未対応(接触なし)。
    if (t1 == 8 || t2 == 8) {
        if (t1 == 8 && t2 == 8 && SDF_INITPOINTS > 0) {
            // g2ローカルでの最適化。R21/p21 = g2座標->g1座標
            vec3 pos1 = vec3(geom_xpos[uint(g1*3)*NBATCH+bid], geom_xpos[uint(g1*3+1)*NBATCH+bid], geom_xpos[uint(g1*3+2)*NBATCH+bid]);
            vec3 pos2 = vec3(geom_xpos[uint(g2*3)*NBATCH+bid], geom_xpos[uint(g2*3+1)*NBATCH+bid], geom_xpos[uint(g2*3+2)*NBATCH+bid]);
            if (msl_len(pos1 - pos2) >= rbound_sum) continue;
            mat3 W1, W2;  // 列優先: W[col][row]
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
                W1[c][r] = geom_xmat[uint(g1*9+r*3+c)*NBATCH+bid];
                W2[c][r] = geom_xmat[uint(g2*9+r*3+c)*NBATCH+bid];
            }
            mat3 R21 = transpose(W1) * W2;
            vec3 p21 = transpose(W1) * (pos2 - pos1);
            mat3 R12 = transpose(R21);
            vec3 p12 = -(R12 * p21);

            // g1フレームでのAABB交差
            vec3 lo1 = vec3(1e10), hi1 = vec3(-1e10), lo2 = vec3(1e10), hi2 = vec3(-1e10);
            vec3 c1 = vec3(sdf_info[g1*20+1], sdf_info[g1*20+2], sdf_info[g1*20+3]);
            vec3 h1 = vec3(sdf_info[g1*20+4], sdf_info[g1*20+5], sdf_info[g1*20+6]);
            vec3 c2 = vec3(sdf_info[g2*20+1], sdf_info[g2*20+2], sdf_info[g2*20+3]);
            vec3 h2 = vec3(sdf_info[g2*20+4], sdf_info[g2*20+5], sdf_info[g2*20+6]);
            for (int k = 0; k < 8; k++) {
                vec3 v1 = c1 + h1 * vec3((k&1) != 0 ? 1.0 : -1.0, (k&2) != 0 ? 1.0 : -1.0, (k&4) != 0 ? 1.0 : -1.0);
                vec3 v2 = R21 * (c2 + h2 * vec3((k&1) != 0 ? 1.0 : -1.0, (k&2) != 0 ? 1.0 : -1.0, (k&4) != 0 ? 1.0 : -1.0)) + p21;
                lo1 = min(lo1, v1); hi1 = max(hi1, v1);
                lo2 = min(lo2, v2); hi2 = max(hi2, v2);
            }
            vec3 lo = max(lo1, lo2), hi = min(hi1, hi2);
            if (all(lessThanEqual(lo, hi))) {
                vec3 known[50];
                int nk = 0;
                for (int i = 0, j = 0; i < SDF_INITPOINTS; i++, j++) {
                    vec3 xg1 = lo + (hi - lo) * vec3(mjmlx_halton(j, 2), mjmlx_halton(j, 3), mjmlx_halton(j, 5));
                    vec3 x = R12 * xg1 + p12;  // g2ローカルへ
                    mjmlx_sdf_descend(g2, g1, R21, p21, x, 0, SDF_ITERATIONS);
                    // 不正確なSDFによる偽の衝突を交差距離で除く
                    float dist = mjmlx_sdf_descend(g2, g1, R21, p21, x, 1, 1);
                    if (dist > 0.0) continue;
                    bool dup = false;
                    for (int q = 0; q < nk; q++) if (distance(x, known[q]) < 1e-9) dup = true;
                    if (dup || nk >= 50 || ncon >= MAX_CON) continue;
                    known[nk++] = x;
                    // 接触点・法線: A=B(中間面)を零等値面とする。法線はg1->g2向き
                    vec3 nl = -mjmlx_sdf_obj_grad(g2, g1, R21, p21, x, 2);
                    vec3 nw = W2 * msl_norm(nl);
                    vec3 cp = W2 * x + pos2 - nw * (dist * 0.5);
                    int idx = int(con_off) + ncon * STRIDE;
                    contact_data[idx+0]=cp.x; contact_data[idx+1]=cp.y; contact_data[idx+2]=cp.z;
                    contact_data[idx+3]=nw.x; contact_data[idx+4]=nw.y; contact_data[idx+5]=nw.z;
                    contact_data[idx+6]=dist;
                    contact_data[idx+7]=float(p);
                    ncon++;
                }
            }
        }
        continue;
    }

    if (t1 != 0 && t2 != 0) {
        vec3 c1 = GP[g1];
        vec3 c2 = GP[g2];
        if (msl_len(c1 - c2) >= rbound_sum) continue;
    }

    float c_pos[3], c_norm[3], c_dist;
    bool has_contact = false;

    if (t1 == 0 || t2 == 0) {
        int plane_g = (t1 == 0) ? g1 : g2;
        int xg = (t1 == 0) ? g2 : g1;
        int xt = (t1 == 0) ? t2 : t1;
        vec3 ppos = GP[plane_g];
        float pR[9]; for (int k=0;k<9;k++) pR[k] = geom_xmat[uint(plane_g*9+k)*NBATCH+bid];
        vec3 normal = vec3(pR[2], pR[5], pR[8]);
        vec3 xpos_ = GP[xg];
        float xR[9]; for (int k=0;k<9;k++) xR[k] = geom_xmat[uint(xg*9+k)*NBATCH+bid];
        vec3 xsize = vec3(geom_size_buf[xg*3], geom_size_buf[xg*3+1], geom_size_buf[xg*3+2]);
        int mesh_id = (xt == 7) ? int(geom_dataid_buf[xg]) : 0;
        int adr = (xt == 7) ? int(mesh_vertadr_buf[mesh_id]) : 0;
        int nverts = (xt == 7) ? int(mesh_vertnum_buf[mesh_id]) : 0;

        vec3 best_w = mjmlx_support(xt, xpos_, xR, xsize, adr, nverts, -normal);
        float best_d = msl_dot(best_w - ppos, normal);
        if (best_d < pair_margin) {
            vec3 cp = best_w - normal * best_d;
            c_pos[0]=cp.x; c_pos[1]=cp.y; c_pos[2]=cp.z;
            c_norm[0]=normal.x; c_norm[1]=normal.y; c_norm[2]=normal.z;
            c_dist = best_d;
            has_contact = true;
        }
    } else {
        vec3 pos1 = GP[g1];
        vec3 pos2 = GP[g2];
        float R1[9]; for (int k=0;k<9;k++) R1[k] = geom_xmat[uint(g1*9+k)*NBATCH+bid];
        float R2[9]; for (int k=0;k<9;k++) R2[k] = geom_xmat[uint(g2*9+k)*NBATCH+bid];
        vec3 size1 = vec3(geom_size_buf[g1*3], geom_size_buf[g1*3+1], geom_size_buf[g1*3+2]);
        vec3 size2 = vec3(geom_size_buf[g2*3], geom_size_buf[g2*3+1], geom_size_buf[g2*3+2]);
        int mid1 = (t1 == 7) ? int(geom_dataid_buf[g1]) : 0;
        int mid2 = (t2 == 7) ? int(geom_dataid_buf[g2]) : 0;
        int adr1 = (t1 == 7) ? int(mesh_vertadr_buf[mid1]) : 0;
        int adr2 = (t2 == 7) ? int(mesh_vertadr_buf[mid2]) : 0;
        int nv1 = (t1 == 7) ? int(mesh_vertnum_buf[mid1]) : 0;
        int nv2 = (t2 == 7) ? int(mesh_vertnum_buf[mid2]) : 0;

        bool sphere_box = ((t1 == 2 || t1 == 3) && t2 == 6) || (t1 == 6 && (t2 == 2 || t2 == 3));
        if (sphere_box) {
            // 球/カプセル-箱は解析解で求める。GJK/EPAは貫入数mmの接触で重なり判定を外し法線が不定になり、足先が平坦な箱天面に触れた瞬間に横方向の衝撃を出していた。カプセルは軸線上で箱に最も近い点を球とみなす(接触は1点)。
            bool sph_first = (t1 != 6);
            int rt = sph_first ? t1 : t2;
            vec3 sc = sph_first ? pos1 : pos2;  vec3 rsize = sph_first ? size1 : size2;  float sr = rsize.x;
            vec3 bc = sph_first ? pos2 : pos1;  vec3 bh = sph_first ? size2 : size1;
            float BR[9]; for (int k=0;k<9;k++) BR[k] = sph_first ? R2[k] : R1[k];
            if (rt == 3) {
                float RR[9]; for (int k=0;k<9;k++) RR[k] = sph_first ? R1[k] : R2[k];
                vec3 ax = vec3(RR[2], RR[5], RR[8]) * rsize.y;
                vec3 a0 = sc - ax, a1 = sc + ax;
                float lo = 0.0, hi = 1.0;
                for (int it = 0; it < 24; it++) {
                    float m1 = lo + (hi-lo)/3.0, m2 = hi - (hi-lo)/3.0;
                    vec3 r1 = mix(a0, a1, m1) - bc, r2 = mix(a0, a1, m2) - bc;
                    vec3 l1 = vec3(BR[0]*r1.x+BR[3]*r1.y+BR[6]*r1.z, BR[1]*r1.x+BR[4]*r1.y+BR[7]*r1.z, BR[2]*r1.x+BR[5]*r1.y+BR[8]*r1.z);
                    vec3 l2 = vec3(BR[0]*r2.x+BR[3]*r2.y+BR[6]*r2.z, BR[1]*r2.x+BR[4]*r2.y+BR[7]*r2.z, BR[2]*r2.x+BR[5]*r2.y+BR[8]*r2.z);
                    if (mjmlx_box_sdist(l1, bh) < mjmlx_box_sdist(l2, bh)) hi = m2; else lo = m1;
                }
                sc = mix(a0, a1, 0.5*(lo+hi));
            }
            vec3 rel = sc - bc;
            vec3 lc = vec3(BR[0]*rel.x+BR[3]*rel.y+BR[6]*rel.z, BR[1]*rel.x+BR[4]*rel.y+BR[7]*rel.z, BR[2]*rel.x+BR[5]*rel.y+BR[8]*rel.z);
            vec3 q = vec3(clamp(lc.x,-bh.x,bh.x), clamp(lc.y,-bh.y,bh.y), clamp(lc.z,-bh.z,bh.z));
            vec3 dl = lc - q;
            float dd = msl_len(dl);
            vec3 nl;  // 箱から球へ向かう単位法線(箱ローカル)
            float sd;
            if (dd > 1e-9) { nl = dl / dd; sd = dd - sr; }
            else {
                vec3 pen = vec3(bh.x-abs(lc.x), bh.y-abs(lc.y), bh.z-abs(lc.z));
                if (pen.x <= pen.y && pen.x <= pen.z) { nl = vec3(lc.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0); sd = -pen.x - sr; }
                else if (pen.y <= pen.z) { nl = vec3(0.0, lc.y >= 0.0 ? 1.0 : -1.0, 0.0); sd = -pen.y - sr; }
                else { nl = vec3(0.0, 0.0, lc.z >= 0.0 ? 1.0 : -1.0); sd = -pen.z - sr; }
            }
            if (sd < pair_margin) {
                vec3 nw = vec3(BR[0]*nl.x+BR[1]*nl.y+BR[2]*nl.z, BR[3]*nl.x+BR[4]*nl.y+BR[5]*nl.z, BR[6]*nl.x+BR[7]*nl.y+BR[8]*nl.z);
                vec3 qw = vec3(BR[0]*q.x+BR[1]*q.y+BR[2]*q.z, BR[3]*q.x+BR[4]*q.y+BR[5]*q.z, BR[6]*q.x+BR[7]*q.y+BR[8]*q.z) + bc;
                vec3 cp = qw + nw * (sd * 0.5);  // 箱側の最近点と球表面点の中点
                vec3 n12 = sph_first ? -nw : nw;  // 接触法線はgeom1->geom2向き
                c_pos[0]=cp.x; c_pos[1]=cp.y; c_pos[2]=cp.z;
                c_norm[0]=n12.x; c_norm[1]=n12.y; c_norm[2]=n12.z;
                c_dist=sd; has_contact=true;
            }
        } else {
        #define SUPPORT1(DIR) mjmlx_support(t1, pos1, R1, size1, adr1, nv1, DIR)
        #define SUPPORT2(DIR) mjmlx_support(t2, pos2, R2, size2, adr2, nv2, DIR)

        vec3 dir = pos2 - pos1;
        if (msl_len(dir) < 1e-12f) dir = vec3(1,0,0);

        vec3 sdiff[4], sa_pts[4], sb_pts[4];
        int sn = 0;
        vec3 sup_a = SUPPORT1(dir);
        vec3 sup_b = SUPPORT2(-dir);
        sdiff[0] = sup_a - sup_b; sa_pts[0] = sup_a; sb_pts[0] = sup_b;
        sn = 1; dir = -sdiff[0];
        if (msl_len(dir) < 1e-12f) dir = vec3(1,0,0);
        bool gjk_overlap = false;

        for (int iter = 0; iter < 32; iter++) {
            sup_a = SUPPORT1(dir);
            sup_b = SUPPORT2(-dir);
            vec3 new_sd = sup_a - sup_b;
            if (msl_dot(new_sd, dir) < 0.0) break;
            sdiff[sn] = new_sd; sa_pts[sn] = sup_a; sb_pts[sn] = sup_b;
            sn++;
            if (sn == 2) {
                vec3 A=sdiff[1], B=sdiff[0], AB=B-A, AO=-A;
                if (msl_dot(AB,AO)>0.0) { dir=msl_cross(msl_cross(AB,AO),AB); if(msl_len(dir)<1e-12f)dir=AO; }
                else { sdiff[0]=A;sa_pts[0]=sa_pts[1];sb_pts[0]=sb_pts[1];sn=1;dir=AO; }
            } else if (sn == 3) {
                vec3 A=sdiff[2],B=sdiff[1],C=sdiff[0],AB=B-A,AC=C-A,AO=-A;
                vec3 ABC=msl_cross(AB,AC);
                if (msl_dot(msl_cross(ABC,AC),AO)>0.0) { sdiff[0]=C;sdiff[1]=A;sa_pts[0]=sa_pts[0];sa_pts[1]=sa_pts[2];sb_pts[0]=sb_pts[0];sb_pts[1]=sb_pts[2];sn=2;dir=msl_cross(msl_cross(AC,AO),AC);if(msl_len(dir)<1e-12f)dir=AO; }
                else if (msl_dot(msl_cross(AB,ABC),AO)>0.0) { sdiff[0]=B;sdiff[1]=A;sa_pts[0]=sa_pts[1];sa_pts[1]=sa_pts[2];sb_pts[0]=sb_pts[1];sb_pts[1]=sb_pts[2];sn=2;dir=msl_cross(msl_cross(AB,AO),AB);if(msl_len(dir)<1e-12f)dir=AO; }
                else if (msl_dot(ABC,AO)>0.0) { dir=ABC; }
                else { vec3 t=sdiff[0];sdiff[0]=sdiff[1];sdiff[1]=t;t=sa_pts[0];sa_pts[0]=sa_pts[1];sa_pts[1]=t;t=sb_pts[0];sb_pts[0]=sb_pts[1];sb_pts[1]=t;dir=-ABC; }
            } else if (sn == 4) {
                vec3 A=sdiff[3],B=sdiff[2],C=sdiff[1],D=sdiff[0];
                vec3 AB=B-A,AC=C-A,AD=D-A,AO=-A;
                vec3 ABC=msl_cross(AB,AC),ACD=msl_cross(AC,AD),ADB=msl_cross(AD,AB);
                bool abc_o=msl_dot(ABC,AO)>0.0, acd_o=msl_dot(ACD,AO)>0.0, adb_o=msl_dot(ADB,AO)>0.0;
                if (!abc_o&&!acd_o&&!adb_o) { gjk_overlap=true; break; }
                if (abc_o) { sdiff[0]=C;sdiff[1]=B;sdiff[2]=A;sa_pts[0]=sa_pts[1];sa_pts[1]=sa_pts[2];sa_pts[2]=sa_pts[3];sb_pts[0]=sb_pts[1];sb_pts[1]=sb_pts[2];sb_pts[2]=sb_pts[3];sn=3;dir=ABC; }
                else if (acd_o) { sdiff[0]=D;sdiff[1]=C;sdiff[2]=A;sa_pts[2]=sa_pts[3];sb_pts[2]=sb_pts[3];sn=3;dir=ACD; }
                else { vec3 ts;sdiff[0]=B;sdiff[1]=D;sdiff[2]=A;ts=sa_pts[0];sa_pts[0]=sa_pts[2];sa_pts[2]=sa_pts[3];sa_pts[1]=ts;ts=sb_pts[0];sb_pts[0]=sb_pts[2];sb_pts[2]=sb_pts[3];sb_pts[1]=ts;sn=3;dir=ADB; }
            }
            if (msl_len(dir) < 1e-12f) dir = vec3(1,0,0);
        }

        if (!gjk_overlap) {
            vec3 p0=sdiff[0], p1=(sn>=2)?sdiff[1]:sdiff[0];
            vec3 seg=p1-p0; float seg_sq=msl_dot(seg,seg);
            float t=(seg_sq>1e-12f)?-msl_dot(p0,seg)/seg_sq:0.0f;
            t=clamp(t,0.0f,1.0f);
            float gap_d=msl_len(p0+seg*t);
            if (gap_d < pair_margin) {
                vec3 wa=sa_pts[0]+(sa_pts[min(sn-1,1)]-sa_pts[0])*t;
                vec3 wb=sb_pts[0]+(sb_pts[min(sn-1,1)]-sb_pts[0])*t;
                vec3 n=msl_norm(wa-wb);
                vec3 cp=(wa+wb)*0.5f;
                c_pos[0]=cp.x;c_pos[1]=cp.y;c_pos[2]=cp.z;
                c_norm[0]=n.x;c_norm[1]=n.y;c_norm[2]=n.z;
                c_dist=gap_d; has_contact=true;
            }
        } else {
            float bw=1e30f; vec3 bn=vec3(0,0,1),bpa,bpb;
            const float D=0.577350269f;
            vec3 sd[14]=vec3[](vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1),
                vec3(D,D,D),vec3(-D,D,D),vec3(D,-D,D),vec3(D,D,-D),
                vec3(-D,-D,D),vec3(-D,D,-D),vec3(D,-D,-D),vec3(-D,-D,-D));
            for (int si=0;si<14;si++) {
                vec3 d=sd[si];
                vec3 pa=SUPPORT1(d), pb=SUPPORT2(-d);
                float w=msl_dot(pa-pb,d);
                if(w<bw){bw=w;bn=d;bpa=pa;bpb=pb;}
            }
            for (int fi=0;fi<4&&sn>=3;fi++) {
                vec3 e1,e2;
                if(fi==0){e1=sdiff[1]-sdiff[0];e2=sdiff[2]-sdiff[0];}
                else if(fi==1){e1=sdiff[2]-sdiff[0];e2=sdiff[min(sn-1,3)]-sdiff[0];}
                else if(fi==2){e1=sdiff[min(sn-1,3)]-sdiff[0];e2=sdiff[1]-sdiff[0];}
                else{e1=sdiff[2]-sdiff[1];e2=sdiff[min(sn-1,3)]-sdiff[1];}
                vec3 fn=msl_cross(e1,e2);
                if(msl_len(fn)<1e-8f)continue;
                fn=msl_norm(fn);
                for(int s=-1;s<=1;s+=2){
                    vec3 d=fn*float(s);
                    vec3 pa=SUPPORT1(d), pb=SUPPORT2(-d);
                    float w=msl_dot(pa-pb,d);
                    if(w<bw){bw=w;bn=d;bpa=pa;bpb=pb;}
                }
            }
            float pen=max(bw,0.0f);
            vec3 cp=(bpa+bpb)*0.5f;
            c_pos[0]=cp.x;c_pos[1]=cp.y;c_pos[2]=cp.z;
            c_norm[0]=bn.x;c_norm[1]=bn.y;c_norm[2]=bn.z;
            c_dist=-pen; has_contact=true;
        }
        #undef SUPPORT1
        #undef SUPPORT2
        }
    }

    if (has_contact && ncon < MAX_CON) {
        int idx = int(con_off) + ncon * STRIDE;
        contact_data[idx+0]=c_pos[0]; contact_data[idx+1]=c_pos[1]; contact_data[idx+2]=c_pos[2];
        contact_data[idx+3]=c_norm[0]; contact_data[idx+4]=c_norm[1]; contact_data[idx+5]=c_norm[2];
        contact_data[idx+6]=c_dist;
        contact_data[idx+7]=float(p);
        ncon++;
    }
}

contact_count[bid] = float(ncon);
)GLSLBODY";

  // ペア表(geom id・型・margin・半径和)はモデル定数なのでシェーダのconst配列に焼き込み、ペアごとのグローバル読み込みをなくす。
  std::string body = ss.str();
  std::string pair_load = "int g1 = int(pair_data[pd+0]), g2 = int(pair_data[pd+1]);\n    int t1 = int(pair_data[pd+2]), t2 = int(pair_data[pd+3]);\n    float pair_margin = pair_data[pd+4];\n    float rbound_sum = pair_data[pd+5];";
  if(pair_data_host && !getenv("MJMLX_COLL_GENERIC")) {
    std::ostringstream pc;
    pc.precision(9);
    auto arr_i = [&](const char* name, int k) { pc << "const int " << name << "[" << npairs << "] = int[]("; for(int p = 0; p < npairs; p++) pc << int(pair_data_host[p * 6 + k]) << (p + 1 < npairs ? "," : ""); pc << ");\n"; };
    auto arr_f = [&](const char* name, int k) { pc << "const float " << name << "[" << npairs << "] = float[]("; for(int p = 0; p < npairs; p++) { std::ostringstream f; f.precision(9); f << pair_data_host[p * 6 + k]; std::string t = f.str(); if(t.find('.') == std::string::npos && t.find('e') == std::string::npos) t += ".0"; pc << t << "f" << (p + 1 < npairs ? "," : ""); } pc << ");\n"; };
    arr_i("PD_G1", 0); arr_i("PD_G2", 1); arr_i("PD_T1", 2); arr_i("PD_T2", 3); arr_f("PD_M", 4); arr_f("PD_R", 5);
    header += pc.str();
    pair_load = "int g1 = PD_G1[p], g2 = PD_G2[p];\n    int t1 = PD_T1[p], t2 = PD_T2[p];\n    float pair_margin = PD_M[p];\n    float rbound_sum = PD_R[p];";
  }
  { size_t q = body.find("PAIR_LOAD"); if(q != std::string::npos) body.replace(q, 9, pair_load); }
  return mkx::fast::compute_kernel("mjmlx_collision", {"geom_xpos", "geom_xmat", "mesh_verts", "pair_data", "mesh_vertadr_buf", "mesh_vertnum_buf", "geom_dataid_buf", "geom_size_buf", "sdf_info", "sdf_grid"}, {"contact_data", "contact_count"}, body, header);
}

} // namespace mjmlx::mkx_kernels
