#pragma once
// Vendored from 20niship/mlx-cross-platform tests/mujoco/kinematics.hpp at commit 3202339e12231f3a68f9ee67e0b6085a33c52286.

#include <string>

#include <mkx/ops/fast_kernel.hpp>

// MuJoCo-MLX-Cpp make_kinematics_source (src/batched.cpp) の逐語GLSL移植。grid=(B,1,1)/threadgroup=(1,1,1)で1スレッド=1envのFKをそのまま踏襲、整数配列はmkx規約に従いfloatバッファをmkx_I()で丸めて読む。

namespace mjmlx::mkx_kernels {

inline mkx::fast::Kernel<> make_kinematics_kernel(int nbody, int njnt, int nq, int ngeom) {
  std::string header = R"GLSL(
int mkx_I(float x) { return int(round(x)); }

vec4 mkx_qmul(vec4 u, vec4 v) {
    return vec4(
        u.x*v.x - u.y*v.y - u.z*v.z - u.w*v.w,
        u.x*v.y + u.y*v.x + u.z*v.w - u.w*v.z,
        u.x*v.z - u.y*v.w + u.z*v.x + u.w*v.y,
        u.x*v.w + u.y*v.z - u.z*v.y + u.w*v.x
    );
}
vec3 mkx_qrot(vec4 q, vec3 v) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float t0 = 2.0 * (x*v.x + y*v.y + z*v.z);
    float t1 = w*w - (x*x + y*y + z*z);
    return vec3(
        t1*v.x + t0*x + 2.0*w*(y*v.z - z*v.y),
        t1*v.y + t0*y + 2.0*w*(z*v.x - x*v.z),
        t1*v.z + t0*z + 2.0*w*(x*v.y - y*v.x)
    );
}
vec4 mkx_qnorm(vec4 q) {
    float n = length(q);
    if (n < 1e-10) n = 1e-10;
    return q / n;
}
vec4 mkx_aa2quat(vec3 axis, float angle) {
    float ha = angle * 0.5;
    float s = sin(ha);
    return vec4(cos(ha), axis.x*s, axis.y*s, axis.z*s);
}
float[9] mkx_q2mat(vec4 q) {
    float w=q.x, x=q.y, y=q.z, z=q.w;
    float xx=x*x, yy=y*y, zz=z*z;
    float xy=x*y, xz=x*z, yz=y*z;
    float wx=w*x, wy=w*y, wz=w*z;
    float m[9];
    m[0]=1.0-2.0*(yy+zz); m[1]=2.0*(xy-wz);     m[2]=2.0*(xz+wy);
    m[3]=2.0*(xy+wz);     m[4]=1.0-2.0*(xx+zz); m[5]=2.0*(yz-wx);
    m[6]=2.0*(xz-wy);     m[7]=2.0*(yz+wx);     m[8]=1.0-2.0*(xx+yy);
    return m;
}
)GLSL";

  std::string src;
  src += "uint batch_idx = gl_GlobalInvocationID.x;\n";
  src += "const uint NB = " + std::to_string(nbody) + ";\n";
  src += "const uint NJ = " + std::to_string(njnt) + ";\n";
  src += "const uint NG = " + std::to_string(ngeom) + ";\n";
  src += "uint qoff = batch_idx * " + std::to_string(nq) + "u;\n";
  src += "uint xp_off = batch_idx * NB * 3u;\n";
  src += "uint xq_off = batch_idx * NB * 4u;\n";
  src += "uint xm_off = batch_idx * NB * 9u;\n";
  src += "uint xa_off = batch_idx * NJ * 3u;\n";
  src += "uint gxp_off = batch_idx * NG * 3u;\n";
  src += "uint gxm_off = batch_idx * NG * 9u;\n";

  src += "float bp[" + std::to_string(nbody) + " * 3];\n";
  src += "float bq[" + std::to_string(nbody) + " * 4];\n";

  src += R"GLSL(
bp[0]=body_pos[0]; bp[1]=body_pos[1]; bp[2]=body_pos[2];
bq[0]=body_quat[0]; bq[1]=body_quat[1]; bq[2]=body_quat[2]; bq[3]=body_quat[3];

for (uint b = 1u; b < NB; b++) {
    int pid = mkx_I(body_parentid[b]);
    vec3 lp = vec3(body_pos[b*3u], body_pos[b*3u+1u], body_pos[b*3u+2u]);
    vec4 pq = vec4(bq[uint(pid)*4u], bq[uint(pid)*4u+1u], bq[uint(pid)*4u+2u], bq[uint(pid)*4u+3u]);
    vec3 rp = mkx_qrot(pq, lp);
    vec3 pos = vec3(bp[uint(pid)*3u]+rp.x, bp[uint(pid)*3u+1u]+rp.y, bp[uint(pid)*3u+2u]+rp.z);
    vec4 lq = vec4(body_quat[b*4u], body_quat[b*4u+1u], body_quat[b*4u+2u], body_quat[b*4u+3u]);
    vec4 quat = mkx_qmul(pq, lq);

    int jadr = mkx_I(body_jntadr[b]);
    int jnum = mkx_I(body_jntnum[b]);
    for (int ji = jadr; ji < jadr + jnum && ji >= 0; ji++) {
        int jt = mkx_I(jnt_type[ji]);
        int qa = mkx_I(jnt_qposadr[ji]);

        if (jt == 0) {
            // FREE
            pos = vec3(qpos[qoff+uint(qa)], qpos[qoff+uint(qa)+1u], qpos[qoff+uint(qa)+2u]);
            quat = vec4(qpos[qoff+uint(qa)+3u], qpos[qoff+uint(qa)+4u], qpos[qoff+uint(qa)+5u], qpos[qoff+uint(qa)+6u]);
            quat = mkx_qnorm(quat);
            xanchor[xa_off+uint(ji)*3u]=pos.x; xanchor[xa_off+uint(ji)*3u+1u]=pos.y; xanchor[xa_off+uint(ji)*3u+2u]=pos.z;
            xaxis[xa_off+uint(ji)*3u]=0.0; xaxis[xa_off+uint(ji)*3u+1u]=0.0; xaxis[xa_off+uint(ji)*3u+2u]=1.0;
        } else if (jt == 3) {
            // HINGE
            vec3 jp = vec3(jnt_pos[ji*3+0], jnt_pos[ji*3+1], jnt_pos[ji*3+2]);
            vec3 rjp = mkx_qrot(quat, jp);
            vec3 anchor = rjp + pos;
            vec3 ja = vec3(jnt_axis[ji*3+0], jnt_axis[ji*3+1], jnt_axis[ji*3+2]);
            vec3 ax = mkx_qrot(quat, ja);
            float angle = qpos[qoff+uint(qa)] - qpos0[qa];
            vec4 qloc = mkx_aa2quat(ja, angle);
            quat = mkx_qmul(quat, qloc);
            vec3 rjp2 = mkx_qrot(quat, jp);
            pos = anchor - rjp2;
            xanchor[xa_off+uint(ji)*3u]=anchor.x; xanchor[xa_off+uint(ji)*3u+1u]=anchor.y; xanchor[xa_off+uint(ji)*3u+2u]=anchor.z;
            xaxis[xa_off+uint(ji)*3u]=ax.x; xaxis[xa_off+uint(ji)*3u+1u]=ax.y; xaxis[xa_off+uint(ji)*3u+2u]=ax.z;
        } else if (jt == 1) {
            // BALL
            vec3 jp = vec3(jnt_pos[ji*3+0], jnt_pos[ji*3+1], jnt_pos[ji*3+2]);
            vec3 rjp = mkx_qrot(quat, jp);
            vec3 anchor = rjp + pos;
            vec3 ja = vec3(jnt_axis[ji*3+0], jnt_axis[ji*3+1], jnt_axis[ji*3+2]);
            vec3 ax = mkx_qrot(quat, ja);
            vec4 qloc = vec4(qpos[qoff+uint(qa)], qpos[qoff+uint(qa)+1u], qpos[qoff+uint(qa)+2u], qpos[qoff+uint(qa)+3u]);
            qloc = mkx_qnorm(qloc);
            quat = mkx_qmul(quat, qloc);
            vec3 rjp2 = mkx_qrot(quat, jp);
            pos = anchor - rjp2;
            xanchor[xa_off+uint(ji)*3u]=anchor.x; xanchor[xa_off+uint(ji)*3u+1u]=anchor.y; xanchor[xa_off+uint(ji)*3u+2u]=anchor.z;
            xaxis[xa_off+uint(ji)*3u]=ax.x; xaxis[xa_off+uint(ji)*3u+1u]=ax.y; xaxis[xa_off+uint(ji)*3u+2u]=ax.z;
        } else if (jt == 2) {
            // SLIDE
            vec3 jp = vec3(jnt_pos[ji*3+0], jnt_pos[ji*3+1], jnt_pos[ji*3+2]);
            vec3 rjp = mkx_qrot(quat, jp);
            vec3 anchor = rjp + pos;
            vec3 ja = vec3(jnt_axis[ji*3+0], jnt_axis[ji*3+1], jnt_axis[ji*3+2]);
            vec3 ax = mkx_qrot(quat, ja);
            float dist = qpos[qoff+uint(qa)] - qpos0[qa];
            pos += ax * dist;
            xanchor[xa_off+uint(ji)*3u]=anchor.x; xanchor[xa_off+uint(ji)*3u+1u]=anchor.y; xanchor[xa_off+uint(ji)*3u+2u]=anchor.z;
            xaxis[xa_off+uint(ji)*3u]=ax.x; xaxis[xa_off+uint(ji)*3u+1u]=ax.y; xaxis[xa_off+uint(ji)*3u+2u]=ax.z;
        }
    }
    bp[b*3u]=pos.x; bp[b*3u+1u]=pos.y; bp[b*3u+2u]=pos.z;
    bq[b*4u]=quat.x; bq[b*4u+1u]=quat.y; bq[b*4u+2u]=quat.z; bq[b*4u+3u]=quat.w;
}

for (uint b = 0u; b < NB; b++) {
    for (uint i = 0u; i < 3u; i++) xpos_out[xp_off+b*3u+i] = bp[b*3u+i];
    for (uint i = 0u; i < 4u; i++) xquat_out[xq_off+b*4u+i] = bq[b*4u+i];
    vec4 q = vec4(bq[b*4u], bq[b*4u+1u], bq[b*4u+2u], bq[b*4u+3u]);
    float mat[9] = mkx_q2mat(q);
    for (uint i = 0u; i < 9u; i++) xmat_out[xm_off+b*9u+i] = mat[i];
}

for (uint b = 0u; b < NB; b++) {
    vec3 ip = vec3(body_ipos[b*3u], body_ipos[b*3u+1u], body_ipos[b*3u+2u]);
    vec4 bqt = vec4(bq[b*4u], bq[b*4u+1u], bq[b*4u+2u], bq[b*4u+3u]);
    vec3 rip = mkx_qrot(bqt, ip);
    xipos_out[xp_off+b*3u]=bp[b*3u]+rip.x; xipos_out[xp_off+b*3u+1u]=bp[b*3u+1u]+rip.y; xipos_out[xp_off+b*3u+2u]=bp[b*3u+2u]+rip.z;
    vec4 iq = vec4(body_iquat[b*4u], body_iquat[b*4u+1u], body_iquat[b*4u+2u], body_iquat[b*4u+3u]);
    vec4 cq = mkx_qmul(bqt, iq);
    float mat[9] = mkx_q2mat(cq);
    for (uint i = 0u; i < 9u; i++) ximat_out[xm_off+b*9u+i] = mat[i];
}

for (uint g = 0u; g < NG; g++) {
    int bid = mkx_I(geom_bodyid[g]);
    vec3 gp = vec3(geom_pos[g*3u], geom_pos[g*3u+1u], geom_pos[g*3u+2u]);
    vec4 bqt = vec4(bq[uint(bid)*4u], bq[uint(bid)*4u+1u], bq[uint(bid)*4u+2u], bq[uint(bid)*4u+3u]);
    vec3 rgp = mkx_qrot(bqt, gp);
    geom_xpos_out[gxp_off+g*3u]=bp[uint(bid)*3u]+rgp.x; geom_xpos_out[gxp_off+g*3u+1u]=bp[uint(bid)*3u+1u]+rgp.y; geom_xpos_out[gxp_off+g*3u+2u]=bp[uint(bid)*3u+2u]+rgp.z;
    vec4 gq = vec4(geom_quat[g*4u], geom_quat[g*4u+1u], geom_quat[g*4u+2u], geom_quat[g*4u+3u]);
    vec4 cq = mkx_qmul(bqt, gq);
    float mat[9] = mkx_q2mat(cq);
    for (uint i = 0u; i < 9u; i++) geom_xmat_out[gxm_off+g*9u+i] = mat[i];
}
)GLSL";

  return mkx::fast::compute_kernel("mjmlx_kin", {"body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_jntadr", "body_jntnum", "jnt_type", "jnt_qposadr", "jnt_pos", "jnt_axis", "qpos0", "geom_bodyid", "geom_pos", "geom_quat", "qpos"},
                                   {"xpos_out", "xquat_out", "xmat_out", "xipos_out", "ximat_out", "xanchor", "xaxis", "geom_xpos_out", "geom_xmat_out"}, src, header);
}

} // namespace mjmlx::mkx_kernels
