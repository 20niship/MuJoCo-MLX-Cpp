// Copyright 2026 Arghya Sur
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Batched simulation: Metal kernels + compile(vmap(step)) for N parallel envs.
//
// Architecture (hybrid 3-phase pipeline, same as Python mujoco-mlx):
//   Phase 1: Metal kinematics kernel (single dispatch for all N envs)
//   Phase 2: compile(vmap(forward_dynamics_without_euler))
//   Phase 3: Metal Euler kernel (single dispatch for all N envs)
//
// Metal kernels are NOT vmap-compatible, so they run outside the vmap.
// Inside vmap, Cholesky uses pure-MLX (column-vectorized, GPU-native).

#include "internal.h"
#include "mjmlx/mjmlx.h"
#include <sstream>
#include <cmath>
#include <cstring>

namespace mjmlx {

// ── MSL quaternion header (shared by kinematics and euler kernels) ────────────

static const std::string QUAT_HEADER = R"(
inline void qmul(const thread float u[4], const thread float v[4], thread float out[4]) {
    out[0] = u[0]*v[0] - u[1]*v[1] - u[2]*v[2] - u[3]*v[3];
    out[1] = u[0]*v[1] + u[1]*v[0] + u[2]*v[3] - u[3]*v[2];
    out[2] = u[0]*v[2] - u[1]*v[3] + u[2]*v[0] + u[3]*v[1];
    out[3] = u[0]*v[3] + u[1]*v[2] - u[2]*v[1] + u[3]*v[0];
}
inline void qrot(const thread float q[4], const thread float v[3], thread float out[3]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float t0 = 2.0f * (x*v[0] + y*v[1] + z*v[2]);
    float t1 = w*w - (x*x + y*y + z*z);
    out[0] = t1*v[0] + t0*x + 2.0f*w*(y*v[2] - z*v[1]);
    out[1] = t1*v[1] + t0*y + 2.0f*w*(z*v[0] - x*v[2]);
    out[2] = t1*v[2] + t0*z + 2.0f*w*(x*v[1] - y*v[0]);
}
inline void qnorm(thread float q[4]) {
    float n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n < 1e-10f) n = 1e-10f;
    float inv = 1.0f / n;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}
inline void aa2quat(const thread float axis[3], float angle, thread float out[4]) {
    float ha = angle * 0.5f;
    float s = sin(ha);
    out[0] = cos(ha); out[1] = axis[0]*s; out[2] = axis[1]*s; out[3] = axis[2]*s;
}
inline void q2mat(const thread float q[4], thread float m[9]) {
    float w=q[0], x=q[1], y=q[2], z=q[3];
    float xx=x*x, yy=y*y, zz=z*z;
    float xy=x*y, xz=x*z, yz=y*z;
    float wx=w*x, wy=w*y, wz=w*z;
    m[0]=1-2*(yy+zz); m[1]=2*(xy-wz);   m[2]=2*(xz+wy);
    m[3]=2*(xy+wz);   m[4]=1-2*(xx+zz); m[5]=2*(yz-wx);
    m[6]=2*(xz-wy);   m[7]=2*(yz+wx);   m[8]=1-2*(xx+yy);
}
)";

// ── Metal kinematics kernel source generation ────────────────────────────────

static std::string make_kinematics_source(int nbody, int njnt, int nq, int ngeom) {
    std::ostringstream ss;
    ss << "uint batch_idx = thread_position_in_grid.x;\n"
       << "const uint NB = " << nbody << ";\n"
       << "const uint NJ = " << njnt << ";\n"
       << "const uint NQ = " << nq << ";\n"
       << "const uint NG = " << ngeom << ";\n"
       << "uint qoff = batch_idx * NQ;\n"
       << "uint xp_off = batch_idx * NB * 3;\n"
       << "uint xq_off = batch_idx * NB * 4;\n"
       << "uint xm_off = batch_idx * NB * 9;\n"
       << "uint xa_off = batch_idx * NJ * 3;\n"
       << "uint gxp_off = batch_idx * NG * 3;\n"
       << "uint gxm_off = batch_idx * NG * 9;\n"

       << "float bp[" << nbody << " * 3];\n"
       << "float bq[" << nbody << " * 4];\n"

       // World body
       << "bp[0]=body_pos[0]; bp[1]=body_pos[1]; bp[2]=body_pos[2];\n"
       << "bq[0]=body_quat[0]; bq[1]=body_quat[1]; bq[2]=body_quat[2]; bq[3]=body_quat[3];\n"

       // FK tree traversal
       << "for (uint b = 1; b < NB; b++) {\n"
       << "  int pid = body_parentid[b];\n"
       << "  float lp[3] = {body_pos[b*3], body_pos[b*3+1], body_pos[b*3+2]};\n"
       << "  float pq[4] = {bq[pid*4], bq[pid*4+1], bq[pid*4+2], bq[pid*4+3]};\n"
       << "  float rp[3]; qrot(pq, lp, rp);\n"
       << "  float pos[3] = {bp[pid*3]+rp[0], bp[pid*3+1]+rp[1], bp[pid*3+2]+rp[2]};\n"
       << "  float lq[4] = {body_quat[b*4], body_quat[b*4+1], body_quat[b*4+2], body_quat[b*4+3]};\n"
       << "  float quat[4]; qmul(pq, lq, quat);\n"

       // Joints
       << "  int jadr = body_jntadr[b]; int jnum = body_jntnum[b];\n"
       << "  for (int ji = jadr; ji < jadr + jnum && ji >= 0; ji++) {\n"
       << "    int jt = jnt_type[ji]; int qa = jnt_qposadr[ji];\n"

       // FREE
       << "    if (jt == 0) {\n"
       << "      pos[0]=qpos[qoff+qa]; pos[1]=qpos[qoff+qa+1]; pos[2]=qpos[qoff+qa+2];\n"
       << "      quat[0]=qpos[qoff+qa+3]; quat[1]=qpos[qoff+qa+4]; quat[2]=qpos[qoff+qa+5]; quat[3]=qpos[qoff+qa+6];\n"
       << "      qnorm(quat);\n"
       << "      xanchor[xa_off+ji*3]=pos[0]; xanchor[xa_off+ji*3+1]=pos[1]; xanchor[xa_off+ji*3+2]=pos[2];\n"
       << "      xaxis[xa_off+ji*3]=0; xaxis[xa_off+ji*3+1]=0; xaxis[xa_off+ji*3+2]=1;\n"

       // HINGE
       << "    } else if (jt == 3) {\n"
       << "      float jp[3]={jnt_pos[ji*3],jnt_pos[ji*3+1],jnt_pos[ji*3+2]};\n"
       << "      float rjp[3]; qrot(quat,jp,rjp);\n"
       << "      float anchor[3]={rjp[0]+pos[0],rjp[1]+pos[1],rjp[2]+pos[2]};\n"
       << "      float ja[3]={jnt_axis[ji*3],jnt_axis[ji*3+1],jnt_axis[ji*3+2]};\n"
       << "      float ax[3]; qrot(quat,ja,ax);\n"
       << "      float angle=qpos[qoff+qa]-qpos0[qa];\n"
       << "      float qloc[4]; aa2quat(ja,angle,qloc);\n"
       << "      float nq4[4]; qmul(quat,qloc,nq4);\n"
       << "      quat[0]=nq4[0]; quat[1]=nq4[1]; quat[2]=nq4[2]; quat[3]=nq4[3];\n"
       << "      float rjp2[3]; qrot(quat,jp,rjp2);\n"
       << "      pos[0]=anchor[0]-rjp2[0]; pos[1]=anchor[1]-rjp2[1]; pos[2]=anchor[2]-rjp2[2];\n"
       << "      xanchor[xa_off+ji*3]=anchor[0]; xanchor[xa_off+ji*3+1]=anchor[1]; xanchor[xa_off+ji*3+2]=anchor[2];\n"
       << "      xaxis[xa_off+ji*3]=ax[0]; xaxis[xa_off+ji*3+1]=ax[1]; xaxis[xa_off+ji*3+2]=ax[2];\n"

       // BALL
       << "    } else if (jt == 1) {\n"
       << "      float jp[3]={jnt_pos[ji*3],jnt_pos[ji*3+1],jnt_pos[ji*3+2]};\n"
       << "      float rjp[3]; qrot(quat,jp,rjp);\n"
       << "      float anchor[3]={rjp[0]+pos[0],rjp[1]+pos[1],rjp[2]+pos[2]};\n"
       << "      float ja[3]={jnt_axis[ji*3],jnt_axis[ji*3+1],jnt_axis[ji*3+2]};\n"
       << "      float ax[3]; qrot(quat,ja,ax);\n"
       << "      float qloc[4]={qpos[qoff+qa],qpos[qoff+qa+1],qpos[qoff+qa+2],qpos[qoff+qa+3]};\n"
       << "      qnorm(qloc);\n"
       << "      float nq4[4]; qmul(quat,qloc,nq4);\n"
       << "      quat[0]=nq4[0]; quat[1]=nq4[1]; quat[2]=nq4[2]; quat[3]=nq4[3];\n"
       << "      float rjp2[3]; qrot(quat,jp,rjp2);\n"
       << "      pos[0]=anchor[0]-rjp2[0]; pos[1]=anchor[1]-rjp2[1]; pos[2]=anchor[2]-rjp2[2];\n"
       << "      xanchor[xa_off+ji*3]=anchor[0]; xanchor[xa_off+ji*3+1]=anchor[1]; xanchor[xa_off+ji*3+2]=anchor[2];\n"
       << "      xaxis[xa_off+ji*3]=ax[0]; xaxis[xa_off+ji*3+1]=ax[1]; xaxis[xa_off+ji*3+2]=ax[2];\n"

       // SLIDE
       << "    } else if (jt == 2) {\n"
       << "      float jp[3]={jnt_pos[ji*3],jnt_pos[ji*3+1],jnt_pos[ji*3+2]};\n"
       << "      float rjp[3]; qrot(quat,jp,rjp);\n"
       << "      float anchor[3]={rjp[0]+pos[0],rjp[1]+pos[1],rjp[2]+pos[2]};\n"
       << "      float ja[3]={jnt_axis[ji*3],jnt_axis[ji*3+1],jnt_axis[ji*3+2]};\n"
       << "      float ax[3]; qrot(quat,ja,ax);\n"
       << "      float dist=qpos[qoff+qa]-qpos0[qa];\n"
       << "      pos[0]+=ax[0]*dist; pos[1]+=ax[1]*dist; pos[2]+=ax[2]*dist;\n"
       << "      xanchor[xa_off+ji*3]=anchor[0]; xanchor[xa_off+ji*3+1]=anchor[1]; xanchor[xa_off+ji*3+2]=anchor[2];\n"
       << "      xaxis[xa_off+ji*3]=ax[0]; xaxis[xa_off+ji*3+1]=ax[1]; xaxis[xa_off+ji*3+2]=ax[2];\n"
       << "    }\n"
       << "  }\n"
       << "  bp[b*3]=pos[0]; bp[b*3+1]=pos[1]; bp[b*3+2]=pos[2];\n"
       << "  bq[b*4]=quat[0]; bq[b*4+1]=quat[1]; bq[b*4+2]=quat[2]; bq[b*4+3]=quat[3];\n"
       << "}\n"

       // Write xpos, xquat, xmat
       << "for (uint b = 0; b < NB; b++) {\n"
       << "  for (uint i = 0; i < 3; i++) xpos_out[xp_off+b*3+i] = bp[b*3+i];\n"
       << "  for (uint i = 0; i < 4; i++) xquat_out[xq_off+b*4+i] = bq[b*4+i];\n"
       << "  float q[4]={bq[b*4],bq[b*4+1],bq[b*4+2],bq[b*4+3]};\n"
       << "  float mat[9]; q2mat(q,mat);\n"
       << "  for (uint i = 0; i < 9; i++) xmat_out[xm_off+b*9+i] = mat[i];\n"
       << "}\n"

       // xipos, ximat
       << "for (uint b = 0; b < NB; b++) {\n"
       << "  float ip[3]={body_ipos[b*3],body_ipos[b*3+1],body_ipos[b*3+2]};\n"
       << "  float bqt[4]={bq[b*4],bq[b*4+1],bq[b*4+2],bq[b*4+3]};\n"
       << "  float rip[3]; qrot(bqt,ip,rip);\n"
       << "  xipos_out[xp_off+b*3]=bp[b*3]+rip[0]; xipos_out[xp_off+b*3+1]=bp[b*3+1]+rip[1]; xipos_out[xp_off+b*3+2]=bp[b*3+2]+rip[2];\n"
       << "  float iq[4]={body_iquat[b*4],body_iquat[b*4+1],body_iquat[b*4+2],body_iquat[b*4+3]};\n"
       << "  float cq[4]; qmul(bqt,iq,cq);\n"
       << "  float mat[9]; q2mat(cq,mat);\n"
       << "  for (uint i = 0; i < 9; i++) ximat_out[xm_off+b*9+i] = mat[i];\n"
       << "}\n"

       // geom transforms
       << "for (uint g = 0; g < NG; g++) {\n"
       << "  int bid=geom_bodyid[g];\n"
       << "  float gp[3]={geom_pos[g*3],geom_pos[g*3+1],geom_pos[g*3+2]};\n"
       << "  float bqt[4]={bq[bid*4],bq[bid*4+1],bq[bid*4+2],bq[bid*4+3]};\n"
       << "  float rgp[3]; qrot(bqt,gp,rgp);\n"
       << "  geom_xpos_out[gxp_off+g*3]=bp[bid*3]+rgp[0]; geom_xpos_out[gxp_off+g*3+1]=bp[bid*3+1]+rgp[1]; geom_xpos_out[gxp_off+g*3+2]=bp[bid*3+2]+rgp[2];\n"
       << "  float gq[4]={geom_quat[g*4],geom_quat[g*4+1],geom_quat[g*4+2],geom_quat[g*4+3]};\n"
       << "  float cq[4]; qmul(bqt,gq,cq);\n"
       << "  float mat[9]; q2mat(cq,mat);\n"
       << "  for (uint i = 0; i < 9; i++) geom_xmat_out[gxm_off+g*9+i] = mat[i];\n"
       << "}\n";

    return ss.str();
}

// ── Metal Euler kernel source generation ─────────────────────────────────────

static std::string make_euler_source(
    int nv, int nq, float dt,
    const std::vector<int>& simple_qa, const std::vector<int>& simple_da,
    const std::vector<std::pair<int,int>>& free_joints,
    const std::vector<std::pair<int,int>>& ball_joints,
    const std::vector<float>& dof_damping_vals)
{
    int n = nv;
    std::ostringstream ss;

    ss << "uint batch_idx = thread_position_in_grid.x;\n"
       << "const uint n = " << n << ";\n"
       << "uint moff = batch_idx * n * n;\n"
       << "uint voff = batch_idx * n;\n"
       << "uint qoff = batch_idx * " << nq << ";\n"

       // Load mass matrix
       << "float a[" << n << " * " << n << "];\n"
       << "for (uint i = 0; i < n * n; i++) a[i] = qM[moff + i];\n";

    // Damping
    for (int i = 0; i < nv; i++) {
        if (dof_damping_vals[i] != 0.0f) {
            ss << "a[" << i*n+i << "] += " << std::scientific << dt << "f * " << dof_damping_vals[i] << "f;\n";
        }
    }
    // Regularization
    for (int i = 0; i < n; i++)
        ss << "a[" << i*n+i << "] += 1e-6f;\n";

    // RHS
    ss << "float rhs[" << n << "];\n"
       << "for (uint i = 0; i < n; i++) rhs[i] = qfrc_smooth[voff+i] + qfrc_constraint[voff+i];\n"

    // Cholesky factorization
       << "float l[" << n << " * " << n << "];\n"
       << "for (uint i = 0; i < n * n; i++) l[i] = 0.0f;\n"
       << "for (uint j = 0; j < n; j++) {\n"
       << "  float s = 0.0f;\n"
       << "  for (uint k = 0; k < j; k++) s += l[j*n+k]*l[j*n+k];\n"
       << "  float diag = a[j*n+j] - s;\n"
       << "  l[j*n+j] = sqrt(max(diag, 1e-6f));\n"
       << "  for (uint i = j+1; i < n; i++) {\n"
       << "    float s2 = 0.0f;\n"
       << "    for (uint k = 0; k < j; k++) s2 += l[i*n+k]*l[j*n+k];\n"
       << "    l[i*n+j] = (a[i*n+j] - s2) / l[j*n+j];\n"
       << "  }\n"
       << "}\n"

    // Forward sub
       << "float y[" << n << "];\n"
       << "for (uint i = 0; i < n; i++) {\n"
       << "  float s = 0.0f;\n"
       << "  for (uint k = 0; k < i; k++) s += l[i*n+k]*y[k];\n"
       << "  y[i] = (rhs[i] - s) / l[i*n+i];\n"
       << "}\n"

    // Backward sub
       << "float qacc[" << n << "];\n"
       << "for (int i = n-1; i >= 0; i--) {\n"
       << "  float s = 0.0f;\n"
       << "  for (uint k = i+1; k < n; k++) s += l[k*n+i]*qacc[k];\n"
       << "  qacc[i] = (y[i] - s) / l[i*n+i];\n"
       << "}\n"

    // Velocity update
       << "float new_qvel[" << n << "];\n"
       << "for (uint i = 0; i < n; i++) {\n"
       << "  float v = qvel_in[voff+i] + qacc[i] * " << std::scientific << dt << "f;\n"
       << "  new_qvel[i] = clamp(v, -1e4f, 1e4f);\n"
       << "}\n"

    // Position integration
       << "float new_qpos[" << nq << "];\n";

    // Copy unchanged qpos
    std::set<int> touched;
    for (int qa : simple_qa) touched.insert(qa);
    for (auto [qa, da] : free_joints) for (int i = 0; i < 7; i++) touched.insert(qa + i);
    for (auto [qa, da] : ball_joints) for (int i = 0; i < 4; i++) touched.insert(qa + i);
    for (int i = 0; i < nq; i++) {
        if (touched.find(i) == touched.end())
            ss << "new_qpos[" << i << "] = qpos_in[qoff + " << i << "];\n";
    }

    // Simple joints
    for (size_t i = 0; i < simple_qa.size(); i++) {
        ss << "new_qpos[" << simple_qa[i] << "] = qpos_in[qoff + " << simple_qa[i]
           << "] + " << std::scientific << dt << "f * new_qvel[" << simple_da[i] << "];\n";
    }

    // FREE joints
    for (auto [qa, da] : free_joints) {
        ss << "new_qpos[" << qa << "] = qpos_in[qoff+" << qa << "] + " << std::scientific << dt << "f * new_qvel[" << da << "];\n"
           << "new_qpos[" << qa+1 << "] = qpos_in[qoff+" << qa+1 << "] + " << std::scientific << dt << "f * new_qvel[" << da+1 << "];\n"
           << "new_qpos[" << qa+2 << "] = qpos_in[qoff+" << qa+2 << "] + " << std::scientific << dt << "f * new_qvel[" << da+2 << "];\n"
           << "{\n"
           << "  float w[3] = {new_qvel[" << da+3 << "], new_qvel[" << da+4 << "], new_qvel[" << da+5 << "]};\n"
           << "  float wnorm = sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);\n"
           << "  float angle = " << std::scientific << dt << "f * wnorm;\n"
           << "  float ha = angle * 0.5f;\n"
           << "  float sinha = (wnorm > 1e-12f) ? sin(ha)/wnorm : 0.5f*" << std::scientific << dt << "f;\n"
           << "  float cosha = cos(ha);\n"
           << "  float dq0=cosha, dq1=sinha*w[0], dq2=sinha*w[1], dq3=sinha*w[2];\n"
           << "  float q0=qpos_in[qoff+" << qa+3 << "], q1=qpos_in[qoff+" << qa+4 << "];\n"
           << "  float q2=qpos_in[qoff+" << qa+5 << "], q3=qpos_in[qoff+" << qa+6 << "];\n"
           << "  float nq0=q0*dq0-q1*dq1-q2*dq2-q3*dq3;\n"
           << "  float nq1=q0*dq1+q1*dq0+q2*dq3-q3*dq2;\n"
           << "  float nq2=q0*dq2-q1*dq3+q2*dq0+q3*dq1;\n"
           << "  float nq3=q0*dq3+q1*dq2-q2*dq1+q3*dq0;\n"
           << "  float qn=sqrt(nq0*nq0+nq1*nq1+nq2*nq2+nq3*nq3);\n"
           << "  float inv_qn=(qn>1e-12f)?1.0f/qn:1.0f;\n"
           << "  new_qpos[" << qa+3 << "]=nq0*inv_qn;\n"
           << "  new_qpos[" << qa+4 << "]=nq1*inv_qn;\n"
           << "  new_qpos[" << qa+5 << "]=nq2*inv_qn;\n"
           << "  new_qpos[" << qa+6 << "]=nq3*inv_qn;\n"
           << "}\n";
    }

    // BALL joints
    for (auto [qa, da] : ball_joints) {
        ss << "{\n"
           << "  float w[3] = {new_qvel[" << da << "], new_qvel[" << da+1 << "], new_qvel[" << da+2 << "]};\n"
           << "  float wnorm = sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);\n"
           << "  float angle = " << std::scientific << dt << "f * wnorm;\n"
           << "  float ha = angle * 0.5f;\n"
           << "  float sinha = (wnorm > 1e-12f) ? sin(ha)/wnorm : 0.5f*" << std::scientific << dt << "f;\n"
           << "  float cosha = cos(ha);\n"
           << "  float dq0=cosha, dq1=sinha*w[0], dq2=sinha*w[1], dq3=sinha*w[2];\n"
           << "  float q0=qpos_in[qoff+" << qa << "], q1=qpos_in[qoff+" << qa+1 << "];\n"
           << "  float q2=qpos_in[qoff+" << qa+2 << "], q3=qpos_in[qoff+" << qa+3 << "];\n"
           << "  float nq0=q0*dq0-q1*dq1-q2*dq2-q3*dq3;\n"
           << "  float nq1=q0*dq1+q1*dq0+q2*dq3-q3*dq2;\n"
           << "  float nq2=q0*dq2-q1*dq3+q2*dq0+q3*dq1;\n"
           << "  float nq3=q0*dq3+q1*dq2-q2*dq1+q3*dq0;\n"
           << "  float qn=sqrt(nq0*nq0+nq1*nq1+nq2*nq2+nq3*nq3);\n"
           << "  float inv_qn=(qn>1e-12f)?1.0f/qn:1.0f;\n"
           << "  new_qpos[" << qa << "]=nq0*inv_qn;\n"
           << "  new_qpos[" << qa+1 << "]=nq1*inv_qn;\n"
           << "  new_qpos[" << qa+2 << "]=nq2*inv_qn;\n"
           << "  new_qpos[" << qa+3 << "]=nq3*inv_qn;\n"
           << "}\n";
    }

    // Write outputs
    ss << "for (uint i = 0; i < " << nq << "; i++) qpos_out[qoff+i] = new_qpos[i];\n"
       << "for (uint i = 0; i < n; i++) qvel_out[voff+i] = new_qvel[i];\n"
       << "for (uint i = 0; i < n; i++) qacc_out[voff+i] = qacc[i];\n";

    return ss.str();
}

// ── Context: Metal kernels + model constants ─────────────────────────────────

using KernelFn = mx::fast::CustomKernelFunction;

struct BatchedStepContext {
    std::optional<KernelFn> kin_kernel;
    std::optional<KernelFn> euler_kernel;

    mx::array body_parentid{mx::zeros({1}, mx::int32)};
    mx::array body_pos{mx::zeros({1})};
    mx::array body_quat{mx::zeros({1})};
    mx::array body_ipos{mx::zeros({1})};
    mx::array body_iquat{mx::zeros({1})};
    mx::array body_jntadr{mx::zeros({1}, mx::int32)};
    mx::array body_jntnum{mx::zeros({1}, mx::int32)};
    mx::array jnt_type_arr{mx::zeros({1}, mx::int32)};
    mx::array jnt_qposadr_arr{mx::zeros({1}, mx::int32)};
    mx::array jnt_pos_arr{mx::zeros({1})};
    mx::array jnt_axis_arr{mx::zeros({1})};
    mx::array qpos0{mx::zeros({1})};
    mx::array geom_bodyid_arr{mx::zeros({1}, mx::int32)};
    mx::array geom_pos_arr{mx::zeros({1})};
    mx::array geom_quat_arr{mx::zeros({1})};

    int nbody = 0, njnt = 0, nq = 0, nv = 0, nu = 0, ngeom = 0;
};

static std::shared_ptr<BatchedStepContext> build_context(const Model& m) {
    auto ctx = std::make_shared<BatchedStepContext>();
    ctx->nbody = m.nbody;
    ctx->njnt = m.njnt;
    ctx->nq = m.nq;
    ctx->nv = m.nv;
    ctx->nu = m.nu;
    ctx->ngeom = m.ngeom;

    ctx->body_parentid = mx::astype(m.body_parentid, mx::int32);
    ctx->body_pos = mx::astype(mx::flatten(m.body_pos), mx::float32);
    ctx->body_quat = mx::astype(mx::flatten(m.body_quat), mx::float32);
    ctx->body_ipos = mx::astype(mx::flatten(m.body_ipos), mx::float32);
    ctx->body_iquat = mx::astype(mx::flatten(m.body_iquat), mx::float32);
    ctx->body_jntadr = mx::astype(m.body_jntadr, mx::int32);
    ctx->body_jntnum = mx::astype(m.body_jntnum, mx::int32);
    ctx->jnt_type_arr = mx::astype(m.jnt_type, mx::int32);
    ctx->jnt_qposadr_arr = mx::astype(m.jnt_qposadr, mx::int32);
    ctx->jnt_pos_arr = (m.njnt > 0)
        ? mx::astype(mx::flatten(m.jnt_pos), mx::float32) : mx::zeros({1});
    ctx->jnt_axis_arr = (m.njnt > 0)
        ? mx::astype(mx::flatten(m.jnt_axis), mx::float32) : mx::zeros({1});
    ctx->qpos0 = mx::astype(mx::flatten(m.qpos0), mx::float32);
    ctx->geom_bodyid_arr = (m.ngeom > 0)
        ? mx::astype(m.geom_bodyid, mx::int32) : mx::zeros({1}, mx::int32);
    ctx->geom_pos_arr = (m.ngeom > 0)
        ? mx::astype(mx::flatten(m.geom_pos), mx::float32) : mx::zeros({1});
    ctx->geom_quat_arr = (m.ngeom > 0)
        ? mx::astype(mx::flatten(m.geom_quat), mx::float32) : mx::zeros({1});

    mx::eval(ctx->body_parentid); mx::eval(ctx->body_pos); mx::eval(ctx->body_quat);
    mx::eval(ctx->body_ipos); mx::eval(ctx->body_iquat);
    mx::eval(ctx->body_jntadr); mx::eval(ctx->body_jntnum);
    mx::eval(ctx->jnt_type_arr); mx::eval(ctx->jnt_qposadr_arr);
    mx::eval(ctx->jnt_pos_arr); mx::eval(ctx->jnt_axis_arr);
    mx::eval(ctx->qpos0); mx::eval(ctx->geom_bodyid_arr);
    mx::eval(ctx->geom_pos_arr); mx::eval(ctx->geom_quat_arr);

    // Build kinematics kernel
    int stack_bytes = m.nbody * 7 * 4;
    if (stack_bytes <= 24000) {
        auto source = make_kinematics_source(m.nbody, m.njnt, m.nq, m.ngeom);
        ctx->kin_kernel = mx::fast::metal_kernel(
            "mjmlx_kin_" + std::to_string(m.nbody),
            {"body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat",
             "body_jntadr", "body_jntnum", "jnt_type", "jnt_qposadr",
             "jnt_pos", "jnt_axis", "qpos0", "geom_bodyid", "geom_pos", "geom_quat",
             "qpos"},
            {"xpos_out", "xquat_out", "xmat_out", "xipos_out", "ximat_out",
             "xanchor", "xaxis", "geom_xpos_out", "geom_xmat_out"},
            source,
            QUAT_HEADER
        );
    }

    // Build Euler kernel
    int euler_stack = 2 * m.nv * m.nv + 4 * m.nv + m.nq;
    if (euler_stack * 4 <= 24000 && m.nv <= 80) {
        mx::eval(m.jnt_type); mx::eval(m.jnt_qposadr); mx::eval(m.jnt_dofadr);
        mx::eval(m.dof_damping);
        auto jt_ptr = m.jnt_type.data<int32_t>();
        auto qa_ptr = m.jnt_qposadr.data<int32_t>();
        auto da_ptr = m.jnt_dofadr.data<int32_t>();

        std::vector<int> s_qa, s_da;
        std::vector<std::pair<int,int>> fj, bj;
        for (int j = 0; j < m.njnt; j++) {
            int jt = jt_ptr[j], qa = qa_ptr[j], da = da_ptr[j];
            if (jt == (int)JointType::HINGE || jt == (int)JointType::SLIDE) {
                s_qa.push_back(qa); s_da.push_back(da);
            } else if (jt == (int)JointType::FREE) {
                fj.push_back({qa, da});
            } else if (jt == (int)JointType::BALL) {
                bj.push_back({qa, da});
            }
        }

        mx::eval(m.dof_damping);
        auto damp_ptr = m.dof_damping.data<float>();
        std::vector<float> damp_vals(damp_ptr, damp_ptr + m.nv);

        auto euler_source = make_euler_source(
            m.nv, m.nq, m.opt.timestep,
            s_qa, s_da, fj, bj, damp_vals);

        ctx->euler_kernel = mx::fast::metal_kernel(
            "mjmlx_euler_" + std::to_string(m.nv) + "_" + std::to_string(m.nq),
            {"qM", "qfrc_smooth", "qfrc_constraint", "qvel_in", "qpos_in"},
            {"qpos_out", "qvel_out", "qacc_out"},
            euler_source
        );
    }

    return ctx;
}

// ── Hybrid batched step ──────────────────────────────────────────────────────

std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu) {
    m.init_cache();
    auto ctx = build_context(m);
    int B = num_envs;

    int nq = m.nq, nv = m.nv, nu = m.nu;
    int nb = m.nbody, nj = m.njnt, ng = m.ngeom;

    bool has_kin = ctx->kin_kernel.has_value();
    bool has_euler = ctx->euler_kernel.has_value();

    const Model* mp = &m;

    if (has_kin && has_euler && use_gpu) {
        // ── Primary path: Metal kin → compile(vmap(forward)) → Metal euler ──

        // Per-env forward function: takes per-env arrays, returns per-env results
        // This runs under vmap — no eval, no data<>, pure graph building
        auto forward_fn = [mp, nq, nv, nu, nb, nj, ng](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array>
        {
            const Model& m_ref = *mp;
            Data d;

            // Unpack per-env state (vmap slices batch dim away)
            d.qpos = inputs[0];                               // (nq,)
            d.qvel = inputs[1];                               // (nv,)
            d.ctrl = (nu > 0) ? inputs[2] : mx::zeros({1});  // (nu,) or dummy

            // Kinematics results from Metal kernel
            d.xpos = mx::reshape(inputs[3], {nb, 3});
            d.xquat = mx::reshape(inputs[4], {nb, 4});
            d.xmat = mx::reshape(inputs[5], {nb, 3, 3});
            d.xipos = mx::reshape(inputs[6], {nb, 3});
            d.ximat = mx::reshape(inputs[7], {nb, 3, 3});
            if (nj > 0) {
                d.xanchor = mx::reshape(inputs[8], {nj, 3});
                d.xaxis = mx::reshape(inputs[9], {nj, 3});
            }
            if (ng > 0) {
                d.geom_xpos = mx::reshape(inputs[10], {ng, 3});
                d.geom_xmat = mx::reshape(inputs[11], {ng, 3, 3});
            }

            // Initialize non-state fields to zeros
            d.qfrc_applied = mx::zeros(mx::Shape{nv});
            d.xfrc_applied = mx::zeros(mx::Shape{nb, 6});

            // Forward dynamics (no kinematics, no euler)
            d = vmap_forward(m_ref, d);

            // Pack outputs needed by Metal euler + observations
            return {
                mx::flatten(d.qM),       // 0: mass matrix (nv*nv,)
                d.qfrc_smooth,            // 1: smooth forces (nv,)
                d.qfrc_constraint,        // 2: constraint forces (nv,)
                mx::flatten(d.xpos)       // 3: body positions (nb*3,)
            };
        };

        // vmap: batch axis 0 for all 12 inputs and 4 outputs
        std::vector<int> in_axes(12, 0);
        std::vector<int> out_axes = {0, 0, 0, 0};
        auto vmapped_fwd = mx::vmap(forward_fn, in_axes, out_axes);

        // Full hybrid pipeline: Metal kin → vmapped forward → Metal euler
        // Wrapped in compile for fused graph execution
        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> pipeline =
            [ctx, vmapped_fwd, B, nq, nv, nu, nb, nj, ng](
                const std::vector<mx::array>& state) -> std::vector<mx::array>
        {
            auto qpos_batch = state[0];  // (B, nq)
            auto qvel_batch = state[1];  // (B, nv)
            auto ctrl_batch = state[2];  // (B, max(1,nu))

            // ── Phase 1: Metal kinematics (single dispatch for all B envs) ──
            auto qpos_flat = mx::astype(mx::flatten(qpos_batch), mx::float32);

            std::vector<mx::Shape> kin_shapes = {
                mx::Shape{B * nb * 3},   // xpos
                mx::Shape{B * nb * 4},   // xquat
                mx::Shape{B * nb * 9},   // xmat
                mx::Shape{B * nb * 3},   // xipos
                mx::Shape{B * nb * 9},   // ximat
                (nj > 0) ? mx::Shape{B * nj * 3} : mx::Shape{1},  // xanchor
                (nj > 0) ? mx::Shape{B * nj * 3} : mx::Shape{1},  // xaxis
                (ng > 0) ? mx::Shape{B * ng * 3} : mx::Shape{1},  // geom_xpos
                (ng > 0) ? mx::Shape{B * ng * 9} : mx::Shape{1},  // geom_xmat
            };
            std::vector<mx::Dtype> kin_dtypes(9, mx::float32);

            auto kin = (*ctx->kin_kernel)(
                {ctx->body_parentid, ctx->body_pos, ctx->body_quat,
                 ctx->body_ipos, ctx->body_iquat,
                 ctx->body_jntadr, ctx->body_jntnum,
                 ctx->jnt_type_arr, ctx->jnt_qposadr_arr,
                 ctx->jnt_pos_arr, ctx->jnt_axis_arr,
                 ctx->qpos0,
                 ctx->geom_bodyid_arr, ctx->geom_pos_arr, ctx->geom_quat_arr,
                 qpos_flat},
                kin_shapes,
                kin_dtypes,
                std::make_tuple(B, 1, 1),
                std::make_tuple(1, 1, 1),
                {},              // template_args
                std::nullopt,    // init_value
                false,           // verbose
                {}               // default stream
            );

            // Reshape kin outputs to (B, ...)
            auto xpos   = mx::reshape(kin[0], {B, nb, 3});
            auto xquat  = mx::reshape(kin[1], {B, nb, 4});
            auto xmat   = mx::reshape(kin[2], {B, nb, 3, 3});
            auto xipos  = mx::reshape(kin[3], {B, nb, 3});
            auto ximat  = mx::reshape(kin[4], {B, nb, 3, 3});
            auto xanchor = (nj > 0) ? mx::reshape(kin[5], {B, nj, 3}) : mx::zeros({B, 1});
            auto xaxis   = (nj > 0) ? mx::reshape(kin[6], {B, nj, 3}) : mx::zeros({B, 1});
            auto gxpos   = (ng > 0) ? mx::reshape(kin[7], {B, ng, 3}) : mx::zeros({B, 1});
            auto gxmat   = (ng > 0) ? mx::reshape(kin[8], {B, ng, 3, 3}) : mx::zeros({B, 1});

            // ── Phase 2: compile(vmap(forward_dynamics)) ──
#ifdef PHASE2_SKIP
            // Dummy forward for timing Metal kernels only
            std::vector<mx::array> mid = {
                mx::zeros({B, nv*nv}),   // qM
                mx::zeros({B, nv}),       // qfrc_smooth
                mx::zeros({B, nv}),       // qfrc_constraint
                mx::flatten(xpos)         // xpos_flat → (B, nb*3) via reshape below
            };
            mid[3] = mx::reshape(mid[3], {B, nb*3});
#else
            auto mid = vmapped_fwd({
                qpos_batch, qvel_batch, ctrl_batch,
                xpos, xquat, xmat, xipos, ximat,
                xanchor, xaxis, gxpos, gxmat
            });
#endif
            // mid[0] = qM (B, nv*nv), mid[1] = qfrc_smooth (B, nv)
            // mid[2] = qfrc_constraint (B, nv), mid[3] = xpos_flat (B, nb*3)

            // ── Phase 3: Metal Euler (single dispatch for all B envs) ──
            auto euler = (*ctx->euler_kernel)(
                {mx::astype(mx::flatten(mid[0]), mx::float32),       // qM flat
                 mx::astype(mx::flatten(mid[1]), mx::float32),       // qfrc_smooth flat
                 mx::astype(mx::flatten(mid[2]), mx::float32),       // qfrc_constraint flat
                 mx::astype(mx::flatten(qvel_batch), mx::float32),   // qvel flat
                 mx::astype(mx::flatten(qpos_batch), mx::float32)},  // qpos flat
                {{B * nq}, {B * nv}, {B * nv}},                      // output shapes
                {mx::float32, mx::float32, mx::float32},             // output dtypes
                std::make_tuple(B, 1, 1),
                std::make_tuple(1, 1, 1),
                {},              // template_args
                std::nullopt,    // init_value
                false,           // verbose
                {}               // default stream
            );

            auto new_qpos = mx::reshape(euler[0], {B, nq});
            auto new_qvel = mx::reshape(euler[1], {B, nv});
            auto xpos_out = mx::reshape(mid[3], {B, nb, 3});

            return {new_qpos, new_qvel, xpos_out};
        };

        // Wrap in compile for fused Metal execution
        auto compiled = mx::compile(pipeline);
        return compiled;
    }

    // ── Fallback: per-env loop using validated scalar pipeline ──
    auto step_fn = [ctx, B, mp](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const Model& m = *mp;
        auto qpos_batch = inputs[0];
        auto qvel_batch = inputs[1];
        auto ctrl_batch = inputs[2];

        int nq = ctx->nq, nv = ctx->nv, nu = ctx->nu;
        int nb = ctx->nbody;

        std::vector<mx::array> new_qpos_list, new_qvel_list, xpos_list;
        new_qpos_list.reserve(B);
        new_qvel_list.reserve(B);
        xpos_list.reserve(B);

        for (int e = 0; e < B; e++) {
            Data d = make_data(m);
            d.qpos = mx::reshape(mx::slice(qpos_batch, {e, 0}, {e + 1, nq}), {nq});
            d.qvel = mx::reshape(mx::slice(qvel_batch, {e, 0}, {e + 1, nv}), {nv});
            if (nu > 0) {
                d.ctrl = mx::reshape(mx::slice(ctrl_batch, {e, 0}, {e + 1, nu}), {nu});
            }
            d = step(m, d);
            new_qpos_list.push_back(d.qpos);
            new_qvel_list.push_back(d.qvel);
            xpos_list.push_back(d.xpos);
        }

        return {mx::stack(new_qpos_list), mx::stack(new_qvel_list), mx::stack(xpos_list)};
    };

    return step_fn;
}

} // namespace mjmlx

// ── C API ────────────────────────────────────────────────────────────────────

extern "C" {

MJMLX_API MjmlxBatchedSim* mjmlx_batched_create(
    const MjmlxModel* model, const MjmlxBatchedConfig* config)
{
    if (!model || !config) return nullptr;
    try {
        auto* handle = new MjmlxBatchedSim();
        handle->sim.model = &model->model;
        handle->sim.num_envs = config->num_envs;
        handle->sim.config = *config;

        int B = config->num_envs;
        int nq = model->model.nq, nv = model->model.nv;
        int nu = model->model.nu;

        // Initialize batched state from qpos0
        std::vector<mx::array> qpos_list, qvel_list;
        for (int i = 0; i < B; i++) {
            qpos_list.push_back(model->model.qpos0);
            qvel_list.push_back(mx::zeros({nv}));
        }
        handle->sim.qpos = mx::stack(qpos_list);
        handle->sim.qvel = mx::stack(qvel_list);

        handle->sim.compiled_step = mjmlx::make_batched_step(
            model->model, B, config->use_gpu);

        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_batched_create error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API void mjmlx_batched_step(MjmlxBatchedSim* sim, const float* ctrl_flat) {
    if (!sim) return;
    auto& s = sim->sim;
    int B = s.num_envs;
    int nu = s.model->nu;
    int ctrl_dim = std::max(1, nu);

    mx::array ctrl = (ctrl_flat && nu > 0)
        ? mx::reshape(mx::array(ctrl_flat, {B * nu}, mx::float32), {B, nu})
        : mx::zeros({B, ctrl_dim});

    auto results = s.compiled_step({s.qpos, s.qvel, ctrl});
    s.qpos = results[0];
    s.qvel = results[1];
    if (results.size() > 2) s.xpos = results[2];
}

MJMLX_API void mjmlx_batched_get_state(
    const MjmlxBatchedSim* sim, float* qpos_out, float* qvel_out,
    int* nq_out, int* nv_out)
{
    if (!sim) return;
    auto& s = sim->sim;
    mx::eval(s.qpos); mx::eval(s.qvel);

    if (nq_out) *nq_out = s.model->nq;
    if (nv_out) *nv_out = s.model->nv;

    if (qpos_out) {
        auto p = s.qpos.data<float>();
        std::memcpy(qpos_out, p, s.num_envs * s.model->nq * sizeof(float));
    }
    if (qvel_out) {
        auto v = s.qvel.data<float>();
        std::memcpy(qvel_out, v, s.num_envs * s.model->nv * sizeof(float));
    }
}

MJMLX_API const float* mjmlx_batched_get_qpos(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.qpos);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nq;
    return sim->sim.qpos.data<float>();
}

MJMLX_API const float* mjmlx_batched_get_qvel(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.qvel);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nv;
    return sim->sim.qvel.data<float>();
}

MJMLX_API const float* mjmlx_batched_get_xpos(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.xpos);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nbody * 3;
    return sim->sim.xpos.data<float>();
}

MJMLX_API void mjmlx_batched_reset(MjmlxBatchedSim* sim, const int* reset_mask) {
    if (!sim || !reset_mask) return;
    auto& s = sim->sim;
    int B = s.num_envs;
    int nq = s.model->nq, nv = s.model->nv;

    mx::eval(s.qpos); mx::eval(s.qvel);
    std::vector<float> qp(s.qpos.data<float>(), s.qpos.data<float>() + B * nq);
    std::vector<float> qv(s.qvel.data<float>(), s.qvel.data<float>() + B * nv);

    mx::eval(s.model->qpos0);
    auto q0 = s.model->qpos0.data<float>();

    for (int i = 0; i < B; i++) {
        if (reset_mask[i]) {
            std::memcpy(&qp[i * nq], q0, nq * sizeof(float));
            std::memset(&qv[i * nv], 0, nv * sizeof(float));
        }
    }

    s.qpos = mx::reshape(mx::array(qp.data(), {B * nq}, mx::float32), {B, nq});
    s.qvel = mx::reshape(mx::array(qv.data(), {B * nv}, mx::float32), {B, nv});
}

MJMLX_API void mjmlx_batched_free(MjmlxBatchedSim* sim) {
    delete sim;
}

} // extern "C"
