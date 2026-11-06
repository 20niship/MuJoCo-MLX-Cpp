#pragma once
// ponytail: メッシュは面を総当たり(AABB枝刈りのみ)でhfield/SDFはヒットなし扱い。大きなvisual meshはgeomgroupで除外し、必要になればBVHを足す。

#include <string>

#include <mkx/ops/fast_kernel.hpp>

namespace mjmlx::mkx_kernels {

constexpr int RAYCAST_GINFO = 16;

inline mkx::fast::Kernel<> make_raycast_kernel(int ngeom, int nenv) {
  std::string header = R"GLSL(
const float RQ_EPS = 1e-9;

// a t^2 + b t + c = 0 の実根(t0<=t1)。実根が無ければfalse
bool rq_roots(float a, float b, float c, out float t0, out float t1) {
    t0 = -1.0; t1 = -1.0;
    if (abs(a) < RQ_EPS) return false;
    float det = b*b - 4.0*a*c;
    if (det < 0.0) return false;
    float s = sqrt(det);
    t0 = (-b - s) / (2.0*a);
    t1 = (-b + s) / (2.0*a);
    return true;
}
// 小さい方の非負根(レイが内側にある場合は大きい方)
float rq_first(float t0, float t1) { return t0 >= 0.0 ? t0 : (t1 >= 0.0 ? t1 : -1.0); }
float rq_min(float a, float b) { return a < 0.0 ? b : (b < 0.0 ? a : min(a, b)); }

float rq_sphere(vec3 p, vec3 d, float r) {
    float t0, t1;
    if (!rq_roots(dot(d,d), 2.0*dot(p,d), dot(p,p) - r*r, t0, t1)) return -1.0;
    return rq_first(t0, t1);
}
// 円柱の側面(|z|<=h)と両端の円板
float rq_cyl_side(vec3 p, vec3 d, float r, float h) {
    float t0, t1;
    if (!rq_roots(d.x*d.x + d.y*d.y, 2.0*(p.x*d.x + p.y*d.y), p.x*p.x + p.y*p.y - r*r, t0, t1)) return -1.0;
    float best = -1.0;
    if (t0 >= 0.0 && abs(p.z + t0*d.z) <= h) best = t0;
    else if (t1 >= 0.0 && abs(p.z + t1*d.z) <= h) best = t1;
    return best;
}
float rq_cylinder(vec3 p, vec3 d, float r, float h) {
    float best = rq_cyl_side(p, d, r, h);
    if (abs(d.z) > RQ_EPS) {
        for (int k = 0; k < 2; k++) {
            float zc = k == 0 ? h : -h;
            float t = (zc - p.z) / d.z;
            if (t >= 0.0) {
                float x = p.x + t*d.x, y = p.y + t*d.y;
                if (x*x + y*y <= r*r) best = rq_min(best, t);
            }
        }
    }
    return best;
}
float rq_capsule(vec3 p, vec3 d, float r, float h) {
    float best = rq_cyl_side(p, d, r, h);
    for (int k = 0; k < 2; k++) {
        float zc = k == 0 ? h : -h;
        vec3 pc = p - vec3(0.0, 0.0, zc);
        float t0, t1;
        if (!rq_roots(dot(d,d), 2.0*dot(pc,d), dot(pc,pc) - r*r, t0, t1)) continue;
        // 端の半球側だけを採用する(円柱側と重なる部分は側面で拾う)
        if (t0 >= 0.0 && (k == 0 ? (p.z + t0*d.z >= h) : (p.z + t0*d.z <= -h))) best = rq_min(best, t0);
        else if (t1 >= 0.0 && (k == 0 ? (p.z + t1*d.z >= h) : (p.z + t1*d.z <= -h))) best = rq_min(best, t1);
    }
    return best;
}
float rq_ellipsoid(vec3 p, vec3 d, vec3 s) { return rq_sphere(p / s, d / s, 1.0); }
float rq_box(vec3 p, vec3 d, vec3 h) {
    float best = -1.0;
    for (int ax = 0; ax < 3; ax++) {
        if (abs(d[ax]) < RQ_EPS) continue;
        for (int k = 0; k < 2; k++) {
            float t = ((k == 0 ? h[ax] : -h[ax]) - p[ax]) / d[ax];
            if (t < 0.0) continue;
            vec3 q = p + t*d;
            int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
            if (abs(q[a1]) <= h[a1] + 1e-6 && abs(q[a2]) <= h[a2] + 1e-6) best = rq_min(best, t);
        }
    }
    return best;
}
// MuJoCoのray_plane: +z側から-z向きに入るレイだけ当たる。size>0なら有限の矩形
float rq_plane(vec3 p, vec3 d, vec3 size) {
    if (d.z > -RQ_EPS) return -1.0;
    float t = -p.z / d.z;
    if (t < 0.0) return -1.0;
    if (size.x > 0.0 && abs(p.x + t*d.x) > size.x) return -1.0;
    if (size.y > 0.0 && abs(p.y + t*d.y) > size.y) return -1.0;
    return t;
}
// geomローカルAABB(中心c, 半幅h)とのスラブ判定。外れていればfalse
bool rq_aabb(vec3 p, vec3 d, vec3 c, vec3 h, float tmax) {
    float t0 = 0.0, t1 = tmax;
    for (int ax = 0; ax < 3; ax++) {
        float po = p[ax] - c[ax];
        if (abs(d[ax]) < RQ_EPS) { if (abs(po) > h[ax]) return false; continue; }
        float ta = (-h[ax] - po) / d[ax], tb = (h[ax] - po) / d[ax];
        t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb));
        if (t0 > t1) return false;
    }
    return true;
}
)GLSL";

  std::string src;
  src += "uint ri = gl_GlobalInvocationID.x;\n";
  src += "const uint B = " + std::to_string(nenv) + "u;\n";
  src += "const uint NG = " + std::to_string(ngeom) + "u;\n";
  src += "const uint GI = " + std::to_string(RAYCAST_GINFO) + "u;\n";
  src += R"GLSL(
// params = [nray_per_env, max_dist(<=0で無制限), bodyexclude, flg_static, geomgroup_mask]
uint nray = uint(params[0]);
if (ri >= B * nray) return;
uint e = ri / nray;
float maxd = params[1] > 0.0 ? params[1] : 1e30;
int bodyex = int(params[2]);
bool use_static = params[3] > 0.5;
int gmask = int(params[4]);

vec3 o = vec3(origins[ri*3u], origins[ri*3u+1u], origins[ri*3u+2u]);
vec3 dw = vec3(dirs[ri*3u], dirs[ri*3u+1u], dirs[ri*3u+2u]);
float dl = length(dw);
float best = -1.0; int bestg = -1;
if (dl > 1e-12) {
    dw /= dl;
    for (uint g = 0u; g < NG; g++) {
        uint b = g * GI;
        int gtype = int(round(ginfo[b]));
        int gbody = int(round(ginfo[b+1u]));
        int ggrp = int(round(ginfo[b+3u]));
        if (gbody == bodyex) continue;
        if (!use_static && ginfo[b+4u] > 0.5) continue;
        if (ginfo[b+5u] > 0.5) continue;
        if (((gmask >> clamp(ggrp, 0, 30)) & 1) == 0) continue;
        vec3 gp = vec3(geom_xpos[(g*3u)*B+e], geom_xpos[(g*3u+1u)*B+e], geom_xpos[(g*3u+2u)*B+e]);
        float R[9];
        for (uint i = 0u; i < 9u; i++) R[i] = geom_xmat[(g*9u+i)*B+e];
        vec3 dd = o - gp;
        vec3 pl = vec3(R[0]*dd.x + R[3]*dd.y + R[6]*dd.z, R[1]*dd.x + R[4]*dd.y + R[7]*dd.z, R[2]*dd.x + R[5]*dd.y + R[8]*dd.z);
        vec3 dlc = vec3(R[0]*dw.x + R[3]*dw.y + R[6]*dw.z, R[1]*dw.x + R[4]*dw.y + R[7]*dw.z, R[2]*dw.x + R[5]*dw.y + R[8]*dw.z);
        vec3 sz = vec3(ginfo[b+13u], ginfo[b+14u], ginfo[b+15u]);
        float t = -1.0;
        if (gtype == 0) t = rq_plane(pl, dlc, sz);
        else if (gtype == 2) t = rq_sphere(pl, dlc, sz.x);
        else if (gtype == 3) t = rq_capsule(pl, dlc, sz.x, sz.y);
        else if (gtype == 4) t = rq_ellipsoid(pl, dlc, sz);
        else if (gtype == 5) t = rq_cylinder(pl, dlc, sz.x, sz.y);
        else if (gtype == 6) t = rq_box(pl, dlc, sz);
        else if (gtype == 7) {
            vec3 ac = vec3(ginfo[b+6u], ginfo[b+7u], ginfo[b+8u]);
            vec3 ah = vec3(ginfo[b+9u], ginfo[b+10u], ginfo[b+11u]);
            int mid = int(round(ginfo[b+2u]));
            if (mid >= 0 && rq_aabb(pl, dlc, ac, ah, best >= 0.0 ? best : maxd)) {
                uint va = uint(round(mesh_adr[mid*3])), fa = uint(round(mesh_adr[mid*3+1])), fn = uint(round(mesh_adr[mid*3+2]));
                for (uint f = 0u; f < fn; f++) {
                    uint i0 = va + uint(round(mesh_face[(fa+f)*3u])), i1 = va + uint(round(mesh_face[(fa+f)*3u+1u])), i2 = va + uint(round(mesh_face[(fa+f)*3u+2u]));
                    vec3 v0 = vec3(mesh_vert[i0*3u], mesh_vert[i0*3u+1u], mesh_vert[i0*3u+2u]);
                    vec3 e1 = vec3(mesh_vert[i1*3u], mesh_vert[i1*3u+1u], mesh_vert[i1*3u+2u]) - v0;
                    vec3 e2 = vec3(mesh_vert[i2*3u], mesh_vert[i2*3u+1u], mesh_vert[i2*3u+2u]) - v0;
                    vec3 pv = cross(dlc, e2);
                    float det = dot(e1, pv);
                    if (abs(det) < 1e-12) continue;
                    float inv = 1.0 / det;
                    vec3 tv = pl - v0;
                    float u = dot(tv, pv) * inv;
                    if (u < 0.0 || u > 1.0) continue;
                    vec3 qv = cross(tv, e1);
                    float v = dot(dlc, qv) * inv;
                    if (v < 0.0 || u + v > 1.0) continue;
                    t = rq_min(t, dot(e2, qv) * inv);
                }
            }
        }
        if (t >= 0.0 && t <= maxd && (best < 0.0 || t < best)) { best = t; bestg = int(g); }
    }
}
dist_out[ri] = best;
geomid_out[ri] = float(bestg);
)GLSL";

  return mkx::fast::compute_kernel("mjmlx_raycast", {"geom_xpos", "geom_xmat", "ginfo", "mesh_vert", "mesh_face", "mesh_adr", "origins", "dirs", "params"},
                                   {"dist_out", "geomid_out"}, src, header);
}

} // namespace mjmlx::mkx_kernels
