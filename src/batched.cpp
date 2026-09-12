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
// ARCHITECTURE NOTE: 3-phase hybrid pipeline (same design as Python mujoco-mlx):
//   Phase 1: Metal kinematics kernel (single GPU dispatch for all N envs)
//   Phase 2: compile(vmap(forward_dynamics_without_euler))
//   Phase 3: Metal Euler kernel (single GPU dispatch for all N envs)
//
// DECISION: Metal kernels are NOT vmap-compatible (they operate on raw flat buffers
// with explicit per-env indexing via thread_position_in_grid). Therefore they must run
// OUTSIDE the vmap boundary. The vmap'd forward function (Phase 2) must use only pure
// MLX array ops -- no eval(), no data<>(), no CPU sync. Any such call would break the
// lazy computation graph that vmap traces.
//
// DECISION: The forward_fn lambda captures a Model pointer (mp). Inside vmap, this
// lambda builds a per-env computation graph. All model constants (cache arrays, option
// values) are accessed but never mutated. If solver iterations need overriding, a
// shared_ptr<Model> copy is created OUTSIDE the lambda to ensure the pointer remains
// valid for the lambda's lifetime.
//
// DECISION: Using mx::compile around the full pipeline (Metal kin → vmap(fwd) →
// Metal euler) enables MLX to fuse the computation graph. This is critical for
// performance: without compile, each MLX op dispatches a separate Metal kernel.
// With compile, MLX fuses compatible ops into fewer, larger kernels.

#include "internal.h"
#include "mjmlx/mjmlx.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <memory>

#if defined(MJMLX_BACKEND_MKX)
#include "compat/mkx_kernels/kinematics.hpp"
#include "compat/mkx_kernels/euler.hpp"
#include "compat/mkx_kernels/euler_devmem.hpp"
#include "compat/mkx_kernels/forward.hpp"
#include "compat/mkx_kernels/collision.hpp"
#include "compat/mkx_kernels/solver.hpp"
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#else
#include <thread>
#include <vector>
#endif
#include <mujoco/mujoco.h>

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

// ── Device-memory Euler kernel for large nv (>80) ────────────────────────────
//
// When nv > 80, the mass matrix (nv×nv) and Cholesky factor L can't fit in
// Metal thread-local storage (~24KB). This kernel reads/writes the matrix
// through device (global) GPU memory via L_scratch, keeping only small vectors
// in thread-local (rhs, y, qacc, new_qvel, new_qpos ≈ 4*nv+nq floats).
//
// For nv=237: thread-local ≈ 4*237+244 = 1192 floats = 4.8KB (well within 16KB)
//             device-memory L_scratch = B*237*237*4 ≈ 29MB for B=128 (reused via compile)
//
// Three-tier strategy (matching original Python mujoco-mlx):
//   nv ≤ 80:   thread-local Metal kernels (fastest, everything in registers)
//   80 < nv ≤ 2048: device-memory Metal kernel (single dispatch, L in global memory)
//   nv > 2048: CPU fallback

static const int EULER_DEVMEM_MAX_NV = 2048;

static std::string make_euler_devmem_source(
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
       << "uint qoff = batch_idx * " << nq << ";\n\n";

    // Copy mass matrix to L_scratch (device memory) and apply damping + regularization
    ss << "for (uint i = 0; i < n * n; i++) L_scratch[moff + i] = qM[moff + i];\n";

    for (int i = 0; i < nv; i++) {
        if (dof_damping_vals[i] != 0.0f) {
            ss << "L_scratch[moff + " << i*n+i << "] += "
               << std::scientific << dt << "f * " << dof_damping_vals[i] << "f;\n";
        }
    }
    for (int i = 0; i < n; i++)
        ss << "L_scratch[moff + " << i*n+i << "] += 1e-6f;\n";

    // RHS in thread-local
    ss << "\nfloat rhs[" << n << "];\n"
       << "for (uint i = 0; i < n; i++) rhs[i] = qfrc_smooth[voff+i] + qfrc_constraint[voff+i];\n\n";

    // In-place Cholesky in device memory: L_scratch goes from damped mass A → Cholesky L
    ss << "for (uint j = 0; j < n; j++) {\n"
       << "  float s = 0.0f;\n"
       << "  for (uint k = 0; k < j; k++) { float v = L_scratch[moff + j*n+k]; s += v*v; }\n"
       << "  float diag = L_scratch[moff + j*n+j] - s;\n"
       << "  diag = sqrt(max(diag, 1e-6f));\n"
       << "  L_scratch[moff + j*n+j] = diag;\n"
       << "  for (uint i = j+1; i < n; i++) {\n"
       << "    float s2 = 0.0f;\n"
       << "    for (uint k = 0; k < j; k++) s2 += L_scratch[moff + i*n+k] * L_scratch[moff + j*n+k];\n"
       << "    L_scratch[moff + i*n+j] = (L_scratch[moff + i*n+j] - s2) / diag;\n"
       << "  }\n"
       << "}\n\n";

    // Forward sub: L @ y = rhs (L from device memory, y thread-local)
    ss << "float y[" << n << "];\n"
       << "for (uint i = 0; i < n; i++) {\n"
       << "  float s = 0.0f;\n"
       << "  for (uint k = 0; k < i; k++) s += L_scratch[moff + i*n+k] * y[k];\n"
       << "  y[i] = (rhs[i] - s) / L_scratch[moff + i*n+i];\n"
       << "}\n\n";

    // Backward sub: L^T @ qacc = y (L^T from device memory, qacc thread-local)
    ss << "float qacc[" << n << "];\n"
       << "for (int i = n-1; i >= 0; i--) {\n"
       << "  float s = 0.0f;\n"
       << "  for (uint k = i+1; k < n; k++) s += L_scratch[moff + k*n+i] * qacc[k];\n"
       << "  qacc[i] = (y[i] - s) / L_scratch[moff + i*n+i];\n"
       << "}\n\n";

    // Velocity update (thread-local)
    ss << "float new_qvel[" << n << "];\n"
       << "for (uint i = 0; i < n; i++) {\n"
       << "  float v = qvel_in[voff+i] + qacc[i] * " << std::scientific << dt << "f;\n"
       << "  new_qvel[i] = clamp(v, -1e4f, 1e4f);\n"
       << "}\n\n";

    // Position integration (thread-local — identical to thread-local kernel)
    ss << "float new_qpos[" << nq << "];\n";

    std::set<int> touched;
    for (int qa : simple_qa) touched.insert(qa);
    for (auto [qa, da] : free_joints) for (int i = 0; i < 7; i++) touched.insert(qa + i);
    for (auto [qa, da] : ball_joints) for (int i = 0; i < 4; i++) touched.insert(qa + i);
    for (int i = 0; i < nq; i++) {
        if (touched.find(i) == touched.end())
            ss << "new_qpos[" << i << "] = qpos_in[qoff + " << i << "];\n";
    }

    for (size_t i = 0; i < simple_qa.size(); i++) {
        ss << "new_qpos[" << simple_qa[i] << "] = qpos_in[qoff + " << simple_qa[i]
           << "] + " << std::scientific << dt << "f * new_qvel[" << simple_da[i] << "];\n";
    }

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

    ss << "for (uint i = 0; i < " << nq << "; i++) qpos_out[qoff+i] = new_qpos[i];\n"
       << "for (uint i = 0; i < n; i++) qvel_out[voff+i] = new_qvel[i];\n"
       << "for (uint i = 0; i < n; i++) qacc_out[voff+i] = qacc[i];\n";

    return ss.str();
}

// ── Metal forward dynamics kernel (smooth dynamics for nv > 80) ──────────────
// Replaces the vmap(forward_dynamics) path for large models. Single Metal dispatch
// per batch. Computes: com_pos → cinert → cdof → crb → qM → com_vel → rne →
// passive → actuation → qfrc_smooth.
//
// Scratch layout per env (in floats):
//   crb[nb*10], cdof[nv*6], cdof_dot[nv*6], cacc[nb*6], cfrc[nb*6],
//   sub_pos[nb*3], sub_mass[nb], qfrc_bias[nv], qfrc_passive[nv]

#if defined(MJMLX_BACKEND_MKX)
// Host-side model constants for mkx_kernels::make_forward_kernel; mirrors what make_forward_source() below extracts to bake into MSL source text instead.
struct ForwardHostConsts {
    std::array<float, 3> gravity{0, 0, 0};
    std::vector<int> body_parentid, body_rootid, dof_bodyid, dof_parentid;
    std::vector<float> body_mass, body_inertia, dof_damping, dof_armature, dof_stiffness;
    std::vector<int> dof_qposadr;
    std::vector<float> qpos_spring, act_gain0, act_bias0;
    std::vector<int> dof_jtype, dof_rotaxis, dof_jid;
    std::vector<std::vector<int>> body_dofs;
    int jnt_dofadr0 = 0;
};

static ForwardHostConsts extract_forward_host_consts(const Model& m) {
    m.init_cache();
    int nb = m.nbody, nv = m.nv, nq = m.nq, nu = m.nu, njnt = m.njnt;

    mx::eval(m.body_parentid, m.body_mass, m.body_inertia, m.dof_damping);
    mx::eval(m.jnt_type, m.jnt_dofadr, m.jnt_qposadr, m.body_jntadr, m.body_jntnum);
    mx::eval(m.dof_bodyid);
    if (m.dof_armature.size() > 0) mx::eval(m.dof_armature);
    if (m.dof_parentid.size() > 0) mx::eval(m.dof_parentid);
    mx::eval(m.body_rootid);

    auto bpar_ptr = m.body_parentid.data<int>();
    auto bmass_ptr = m.body_mass.data<float>();
    auto binert_ptr = m.body_inertia.data<float>();
    auto jt_ptr = m.jnt_type.data<int>();
    auto jda_ptr = m.jnt_dofadr.data<int>();
    auto jqa_ptr = m.jnt_qposadr.data<int>();
    auto dbid_ptr = m.dof_bodyid.data<int>();
    auto ddamp_ptr = m.dof_damping.data<float>();
    auto dpar_ptr = m.dof_parentid.data<int>();
    auto broot_ptr = m.body_rootid.data<int>();

    ForwardHostConsts k;
    k.body_parentid.assign(bpar_ptr, bpar_ptr + nb);
    k.body_rootid.assign(broot_ptr, broot_ptr + nb);
    k.body_mass.assign(bmass_ptr, bmass_ptr + nb);
    k.body_inertia.assign(binert_ptr, binert_ptr + nb * 3);
    k.dof_bodyid.assign(dbid_ptr, dbid_ptr + nv);
    k.dof_parentid.assign(dpar_ptr, dpar_ptr + nv);
    k.dof_damping.assign(ddamp_ptr, ddamp_ptr + nv);

    k.dof_armature.assign(nv, 0.0f);
    if (m.dof_armature.size() > 0) {
        auto arm_ptr = m.dof_armature.data<float>();
        for (int i = 0; i < nv; i++) k.dof_armature[i] = arm_ptr[i];
    }

    k.dof_stiffness.assign(nv, 0.0f);
    if (m.jnt_stiffness.size() > 0) {
        mx::eval(m.jnt_stiffness);
        auto jstiff_ptr = m.jnt_stiffness.data<float>();
        for (int j = 0; j < njnt; j++) {
            int jt = jt_ptr[j], da = jda_ptr[j];
            if (jt == (int)JointType::HINGE || jt == (int)JointType::SLIDE) k.dof_stiffness[da] = jstiff_ptr[j];
        }
    }

    k.qpos_spring.assign(nq, 0.0f);
    if (m.qpos_spring.size() > 0) {
        mx::eval(m.qpos_spring);
        auto qsp_ptr = m.qpos_spring.data<float>();
        for (int i = 0; i < nq; i++) k.qpos_spring[i] = qsp_ptr[i];
    }

    k.act_gain0.assign(std::max(nu, 1), 0.0f);
    k.act_bias0.assign(std::max(nu, 1), 0.0f);
    if (nu > 0 && m.actuator_gainprm.size() > 0) {
        mx::eval(m.actuator_gainprm, m.actuator_biasprm);
        auto gp = m.actuator_gainprm.data<float>();
        auto bp = m.actuator_biasprm.data<float>();
        int ncol = (int)m.actuator_gainprm.shape(1);
        for (int i = 0; i < nu; i++) { k.act_gain0[i] = gp[i * ncol]; k.act_bias0[i] = bp[i * ncol]; }
    }

    std::vector<int> dof_jntid(nv, -1);
    for (int j = 0; j < njnt; j++) {
        int jt = jt_ptr[j], da = jda_ptr[j];
        int ndof = (jt == 0) ? 6 : (jt == 1) ? 3 : 1;
        for (int di = 0; di < ndof; di++) dof_jntid[da + di] = j;
    }

    k.dof_qposadr.assign(nv, 0);
    for (int j = 0; j < njnt; j++) {
        int jt = jt_ptr[j], da = jda_ptr[j], qa = jqa_ptr[j];
        int ndof = (jt == 0) ? 6 : (jt == 1) ? 3 : 1;
        for (int di = 0; di < ndof; di++) k.dof_qposadr[da + di] = qa + di;
    }

    k.body_dofs.assign(nb, {});
    for (int di = 0; di < nv; di++) k.body_dofs[dbid_ptr[di]].push_back(di);

    k.dof_jtype.assign(nv, -1);
    k.dof_rotaxis.assign(nv, -1);
    k.dof_jid.assign(nv, -1);
    for (int i = 0; i < nv; i++) {
        int ji = dof_jntid[i];
        k.dof_jid[i] = ji;
        k.dof_jtype[i] = (ji >= 0) ? jt_ptr[ji] : -1;
        int da = (ji >= 0) ? jda_ptr[ji] : 0;
        int jt = (ji >= 0) ? jt_ptr[ji] : -1;
        int sub = i - da;
        int rotax = -1;
        if (jt == 0 && sub >= 3) rotax = sub - 3;
        if (jt == 1) rotax = sub;
        k.dof_rotaxis[i] = rotax;
    }

    if (m.opt.gravity.size() > 0) {
        mx::eval(m.opt.gravity);
        auto gp = m.opt.gravity.data<float>();
        k.gravity = {gp[0], gp[1], gp[2]};
    }

    k.jnt_dofadr0 = (njnt > 0) ? jda_ptr[0] : 0;
    return k;
}
#endif // MJMLX_BACKEND_MKX

static std::string make_forward_source(const Model& m) {
    m.init_cache();
    const auto& c = m.cache;
    int nb = m.nbody, nv = m.nv, nq = m.nq, nu = m.nu, njnt = m.njnt;
    float dt = m.opt.timestep;

    // Scratch offsets per env
    int off_crb = 0;
    int off_cdof = nb * 10;
    int off_cdof_dot = off_cdof + nv * 6;
    int off_cacc = off_cdof_dot + nv * 6;
    int off_cfrc = off_cacc + nb * 6;
    int off_subpos = off_cfrc + nb * 6;
    int off_submass = off_subpos + nb * 3;
    int off_qfrc_bias = off_submass + nb;
    int off_qfrc_passive = off_qfrc_bias + nv;
    int scratch_per_env = off_qfrc_passive + nv;

    // Extract model constants to bake into source
    mx::eval(m.body_parentid, m.body_mass, m.body_inertia, m.dof_damping);
    mx::eval(m.jnt_type, m.jnt_dofadr, m.jnt_qposadr, m.body_jntadr, m.body_jntnum);
    mx::eval(m.dof_bodyid);
    if (m.dof_armature.size() > 0) mx::eval(m.dof_armature);
    if (m.dof_parentid.size() > 0) mx::eval(m.dof_parentid);

    auto bpar_ptr = m.body_parentid.data<int>();
    auto bmass_ptr = m.body_mass.data<float>();
    auto binert_ptr = m.body_inertia.data<float>();
    auto jt_ptr = m.jnt_type.data<int>();
    auto jda_ptr = m.jnt_dofadr.data<int>();
    auto jqa_ptr = m.jnt_qposadr.data<int>();
    auto bja_ptr = m.body_jntadr.data<int>();
    auto bjn_ptr = m.body_jntnum.data<int>();
    auto dbid_ptr = m.dof_bodyid.data<int>();
    auto ddamp_ptr = m.dof_damping.data<float>();
    auto dpar_ptr = m.dof_parentid.data<int>();

    // body_rootid
    mx::eval(m.body_rootid);
    auto broot_ptr = m.body_rootid.data<int>();

    // dof_armature
    std::vector<float> armature_vals(nv, 0.0f);
    if (m.dof_armature.size() > 0) {
        auto arm_ptr = m.dof_armature.data<float>();
        for (int i = 0; i < nv; i++) armature_vals[i] = arm_ptr[i];
    }

    // Passive force: extract per-dof stiffness (jnt_stiffness mapped via joint→dof)
    std::vector<float> dof_stiffness(nv, 0.0f);
    if (m.jnt_stiffness.size() > 0) {
        mx::eval(m.jnt_stiffness);
        auto jstiff_ptr = m.jnt_stiffness.data<float>();
        for (int j = 0; j < njnt; j++) {
            int jt = jt_ptr[j];
            int da = jda_ptr[j];
            if (jt == (int)JointType::HINGE || jt == (int)JointType::SLIDE) {
                dof_stiffness[da] = jstiff_ptr[j];
            } else if (jt == (int)JointType::FREE) {
                // Free joints: no spring force on translation DOFs
            }
        }
    }

    // qpos_spring: the spring reference position
    std::vector<float> qpos_spring_vals(nq, 0.0f);
    if (m.qpos_spring.size() > 0) {
        mx::eval(m.qpos_spring);
        auto qsp_ptr = m.qpos_spring.data<float>();
        for (int i = 0; i < nq; i++) qpos_spring_vals[i] = qsp_ptr[i];
    }

    // Actuator parameters
    std::vector<float> act_gain0(std::max(nu, 1), 0.0f);
    std::vector<float> act_bias0(std::max(nu, 1), 0.0f);
    std::vector<float> act_bias1(std::max(nu, 1), 0.0f);
    std::vector<float> act_bias2(std::max(nu, 1), 0.0f);
    if (nu > 0 && m.actuator_gainprm.size() > 0) {
        mx::eval(m.actuator_gainprm, m.actuator_biasprm);
        auto gp = m.actuator_gainprm.data<float>();
        auto bp = m.actuator_biasprm.data<float>();
        int ncol = (int)m.actuator_gainprm.shape(1);
        for (int i = 0; i < nu; i++) {
            act_gain0[i] = gp[i * ncol];
            act_bias0[i] = bp[i * ncol];
            act_bias1[i] = bp[i * ncol + 1];
            act_bias2[i] = bp[i * ncol + 2];
        }
    }

    // Build per-DOF joint index (which joint does each DOF belong to)
    std::vector<int> dof_jntid(nv, -1);
    for (int j = 0; j < njnt; j++) {
        int jt = jt_ptr[j];
        int da = jda_ptr[j];
        int ndof = (jt == 0) ? 6 : (jt == 1) ? 3 : 1;
        for (int k = 0; k < ndof; k++) dof_jntid[da + k] = j;
    }

    // Per-DOF qpos address (for passive spring forces)
    std::vector<int> dof_qposadr(nv, 0);
    for (int j = 0; j < njnt; j++) {
        int jt = jt_ptr[j], da = jda_ptr[j], qa = jqa_ptr[j];
        int ndof = (jt == 0) ? 6 : (jt == 1) ? 3 : 1;
        for (int k = 0; k < ndof; k++) dof_qposadr[da + k] = qa + k;
    }

    // Build body→dofs mapping
    std::vector<std::vector<int>> body_dofs(nb);
    for (int di = 0; di < nv; di++) body_dofs[dbid_ptr[di]].push_back(di);

    // Gravity
    float grav[3] = {0, 0, 0};
    if (m.opt.gravity.size() > 0) {
        mx::eval(m.opt.gravity);
        auto gp = m.opt.gravity.data<float>();
        grav[0] = gp[0]; grav[1] = gp[1]; grav[2] = gp[2];
    }

    std::ostringstream ss;
    ss << std::scientific;

    // ── Dimensions and offsets ──
    ss << "uint bid = thread_position_in_grid.x;\n"
       << "const int NB = " << nb << ";\n"
       << "const int NV = " << nv << ";\n"
       << "const int NQ = " << nq << ";\n"
       << "const int NU = " << nu << ";\n"
       << "const int NJNT = " << njnt << ";\n"
       << "const int SCRATCH_SZ = " << scratch_per_env << ";\n\n";

    // Per-env offsets for batched arrays
    ss << "uint xip_off = bid * NB * 3;\n"
       << "uint xim_off = bid * NB * 9;\n"
       << "uint xa_off  = bid * NJNT * 3;\n"
       << "uint xax_off = bid * NJNT * 3;\n"
       << "uint xm_off  = bid * NB * 9;\n"
       << "uint q_off   = bid * NQ;\n"
       << "uint v_off   = bid * NV;\n"
       << "uint u_off   = bid * NU;\n"
       << "uint qM_off  = bid * NV * NV;\n"
       << "uint qfs_off = bid * NV;\n"
       << "uint sc_off  = bid * NB * 3;\n"
       << "uint ci_off  = bid * NB * 10;\n"
       << "uint cv_off  = bid * NB * 6;\n"
       << "uint qa_off  = bid * NV;\n"
       << "uint s_off   = bid * SCRATCH_SZ;\n\n";

    // Scratch sub-offsets
    ss << "const int S_CRB = " << off_crb << ";\n"
       << "const int S_CDOF = " << off_cdof << ";\n"
       << "const int S_CDOFD = " << off_cdof_dot << ";\n"
       << "const int S_CACC = " << off_cacc << ";\n"
       << "const int S_CFRC = " << off_cfrc << ";\n"
       << "const int S_SUBP = " << off_subpos << ";\n"
       << "const int S_SUBM = " << off_submass << ";\n"
       << "const int S_BIAS = " << off_qfrc_bias << ";\n"
       << "const int S_PASS = " << off_qfrc_passive << ";\n\n";

    // ── Baked model constants ──
    ss << "// Body parent IDs\n"
       << "const int body_par[] = {";
    for (int i = 0; i < nb; i++) ss << bpar_ptr[i] << (i < nb-1 ? "," : "");
    ss << "};\n";

    ss << "const int body_root[] = {";
    for (int i = 0; i < nb; i++) ss << broot_ptr[i] << (i < nb-1 ? "," : "");
    ss << "};\n";

    ss << "const float body_mass[] = {";
    for (int i = 0; i < nb; i++) ss << bmass_ptr[i] << "f" << (i < nb-1 ? "," : "");
    ss << "};\n";

    ss << "const float body_inert[] = {";
    for (int i = 0; i < nb * 3; i++) ss << binert_ptr[i] << "f" << (i < nb*3-1 ? "," : "");
    ss << "};\n";

    ss << "const int dof_bodyid[] = {";
    for (int i = 0; i < nv; i++) ss << dbid_ptr[i] << (i < nv-1 ? "," : "");
    ss << "};\n";

    ss << "const int dof_par[] = {";
    for (int i = 0; i < nv; i++) ss << dpar_ptr[i] << (i < nv-1 ? "," : "");
    ss << "};\n";

    ss << "const float dof_damp[] = {";
    for (int i = 0; i < nv; i++) ss << ddamp_ptr[i] << "f" << (i < nv-1 ? "," : "");
    ss << "};\n";

    ss << "const float dof_arm[] = {";
    for (int i = 0; i < nv; i++) ss << armature_vals[i] << "f" << (i < nv-1 ? "," : "");
    ss << "};\n";

    ss << "const float dof_stiff[] = {";
    for (int i = 0; i < nv; i++) ss << dof_stiffness[i] << "f" << (i < nv-1 ? "," : "");
    ss << "};\n";

    ss << "const int dof_qa[] = {";
    for (int i = 0; i < nv; i++) ss << dof_qposadr[i] << (i < nv-1 ? "," : "");
    ss << "};\n";

    ss << "const float qpos_spr[] = {";
    for (int i = 0; i < nq; i++) ss << qpos_spring_vals[i] << "f" << (i < nq-1 ? "," : "");
    ss << "};\n";

    if (nu > 0) {
        ss << "const float act_g0[] = {";
        for (int i = 0; i < nu; i++) ss << act_gain0[i] << "f" << (i < nu-1 ? "," : "");
        ss << "};\n";
        ss << "const float act_b0[] = {";
        for (int i = 0; i < nu; i++) ss << act_bias0[i] << "f" << (i < nu-1 ? "," : "");
        ss << "};\n";
        ss << "const float act_b1[] = {";
        for (int i = 0; i < nu; i++) ss << act_bias1[i] << "f" << (i < nu-1 ? "," : "");
        ss << "};\n";
        ss << "const float act_b2[] = {";
        for (int i = 0; i < nu; i++) ss << act_bias2[i] << "f" << (i < nu-1 ? "," : "");
        ss << "};\n";
    }

    // Per-DOF joint type and sub-index (for cdof computation)
    ss << "const int dof_jtype[] = {";
    for (int i = 0; i < nv; i++) {
        int ji = dof_jntid[i];
        ss << (ji >= 0 ? jt_ptr[ji] : -1) << (i < nv-1 ? "," : "");
    }
    ss << "};\n";

    // For free/ball rotation DOFs: which rotation axis (0,1,2)
    ss << "const int dof_rotaxis[] = {";
    for (int i = 0; i < nv; i++) {
        int ji = dof_jntid[i];
        int da = (ji >= 0) ? jda_ptr[ji] : 0;
        int jt = (ji >= 0) ? jt_ptr[ji] : -1;
        int sub = i - da;
        int rotax = -1;
        if (jt == 0 && sub >= 3) rotax = sub - 3;
        if (jt == 1) rotax = sub;
        ss << rotax << (i < nv-1 ? "," : "");
    }
    ss << "};\n";

    // Per-DOF joint index (for anchor/axis lookup)
    ss << "const int dof_jid[] = {";
    for (int i = 0; i < nv; i++) ss << dof_jntid[i] << (i < nv-1 ? "," : "");
    ss << "};\n\n";

    // ═══ PHASE A: Subtree COM + Cinert + CDof ═══
    ss << "// ── Phase A: subtree COM, cinert, cdof ──\n";

    // Initialize subtree mass/pos
    ss << "for (int b = 0; b < NB; b++) {\n"
       << "  float m = body_mass[b];\n"
       << "  scratch[s_off + S_SUBM + b] = m;\n"
       << "  for (int k = 0; k < 3; k++)\n"
       << "    scratch[s_off + S_SUBP + b*3+k] = xipos[xip_off + b*3+k] * m;\n"
       << "}\n";

    // Backward accumulation for subtree COM
    ss << "for (int b = NB-1; b >= 1; b--) {\n"
       << "  int p = body_par[b];\n"
       << "  scratch[s_off + S_SUBM + p] += scratch[s_off + S_SUBM + b];\n"
       << "  for (int k = 0; k < 3; k++)\n"
       << "    scratch[s_off + S_SUBP + p*3+k] += scratch[s_off + S_SUBP + b*3+k];\n"
       << "}\n";

    // Compute subtree_com = sub_pos / sub_mass
    ss << "for (int b = 0; b < NB; b++) {\n"
       << "  float sm = max(scratch[s_off + S_SUBM + b], 1e-8f);\n"
       << "  for (int k = 0; k < 3; k++)\n"
       << "    subtree_com_out[sc_off + b*3+k] = scratch[s_off + S_SUBP + b*3+k] / sm;\n"
       << "}\n\n";

    // Compute cinert for each body
    ss << "for (int b = 0; b < NB; b++) {\n"
       << "  float m = body_mass[b];\n"
       << "  int rid = body_root[b];\n"
       << "  float off[3];\n"
       << "  for (int k = 0; k < 3; k++)\n"
       << "    off[k] = xipos[xip_off + b*3+k] - subtree_com_out[sc_off + rid*3+k];\n"
       // Transform inertia to global frame: R diag(I) R^T
       << "  float R[9]; for (int k=0;k<9;k++) R[k] = ximat[xim_off + b*9+k];\n"
       << "  float Ix = body_inert[b*3], Iy = body_inert[b*3+1], Iz = body_inert[b*3+2];\n"
       // RI = R * diag(Ix,Iy,Iz)
       << "  float RI[9] = {R[0]*Ix,R[1]*Iy,R[2]*Iz, R[3]*Ix,R[4]*Iy,R[5]*Iz, R[6]*Ix,R[7]*Iy,R[8]*Iz};\n"
       // I_g = RI * R^T
       << "  float Ig[9];\n"
       << "  for (int r=0;r<3;r++) for (int c=0;c<3;c++) {\n"
       << "    float s=0; for (int k=0;k<3;k++) s += RI[r*3+k]*R[c*3+k];\n"
       << "    Ig[r*3+c] = s;\n"
       << "  }\n"
       // Parallel axis theorem: I += m*(d²I - outer(off,off))
       << "  float d2 = off[0]*off[0]+off[1]*off[1]+off[2]*off[2];\n"
       << "  Ig[0] += m*(d2 - off[0]*off[0]); Ig[4] += m*(d2 - off[1]*off[1]); Ig[8] += m*(d2 - off[2]*off[2]);\n"
       << "  Ig[1] -= m*off[0]*off[1]; Ig[3] -= m*off[1]*off[0];\n"
       << "  Ig[2] -= m*off[0]*off[2]; Ig[6] -= m*off[2]*off[0];\n"
       << "  Ig[5] -= m*off[1]*off[2]; Ig[7] -= m*off[2]*off[1];\n"
       // cinert: [I00, I11, I22, I01, I02, I12, px*m, py*m, pz*m, mass]
       << "  cinert_out[ci_off + b*10+0] = Ig[0];\n"
       << "  cinert_out[ci_off + b*10+1] = Ig[4];\n"
       << "  cinert_out[ci_off + b*10+2] = Ig[8];\n"
       << "  cinert_out[ci_off + b*10+3] = Ig[1];\n"
       << "  cinert_out[ci_off + b*10+4] = Ig[2];\n"
       << "  cinert_out[ci_off + b*10+5] = Ig[5];\n"
       << "  cinert_out[ci_off + b*10+6] = off[0]*m;\n"
       << "  cinert_out[ci_off + b*10+7] = off[1]*m;\n"
       << "  cinert_out[ci_off + b*10+8] = off[2]*m;\n"
       << "  cinert_out[ci_off + b*10+9] = m;\n"
       << "}\n\n";

    // Compute cdof for each DOF
    ss << "for (int di = 0; di < NV; di++) {\n"
       << "  int bdi = dof_bodyid[di];\n"
       << "  int jt = dof_jtype[di];\n"
       << "  int ji = dof_jid[di];\n"
       << "  int ra = dof_rotaxis[di];\n"
       << "  int rid = body_root[bdi];\n"
       << "  float rc[3]; for (int k=0;k<3;k++) rc[k] = subtree_com_out[sc_off + rid*3+k];\n"
       << "  float c6[6] = {0,0,0,0,0,0};\n"
       << "  if (jt == 3) {\n"  // HINGE
       << "    float ax[3] = {xaxis[xax_off+ji*3], xaxis[xax_off+ji*3+1], xaxis[xax_off+ji*3+2]};\n"
       << "    float anc[3] = {xanchor[xa_off+ji*3], xanchor[xa_off+ji*3+1], xanchor[xa_off+ji*3+2]};\n"
       << "    float d[3]; for (int k=0;k<3;k++) d[k] = rc[k] - anc[k];\n"
       << "    c6[0]=ax[0]; c6[1]=ax[1]; c6[2]=ax[2];\n"
       << "    c6[3]=ax[1]*d[2]-ax[2]*d[1]; c6[4]=ax[2]*d[0]-ax[0]*d[2]; c6[5]=ax[0]*d[1]-ax[1]*d[0];\n"
       << "  } else if (jt == 0 && ra < 0) {\n"  // FREE translation
       << "    int sub = di - " << (njnt > 0 ? jda_ptr[0] : 0) << ";\n"  // relative dof within free joint
       << "    c6[3+sub] = 1.0f;\n"
       << "  } else if (jt == 0 && ra >= 0) {\n"  // FREE rotation
       << "    float rm[9]; for (int k=0;k<9;k++) rm[k] = xmat[xm_off + bdi*9+k];\n"
       << "    float col[3] = {rm[ra], rm[3+ra], rm[6+ra]};\n"
       << "    float anc[3] = {xanchor[xa_off+ji*3], xanchor[xa_off+ji*3+1], xanchor[xa_off+ji*3+2]};\n"
       << "    float d[3]; for (int k=0;k<3;k++) d[k] = rc[k] - anc[k];\n"
       << "    c6[0]=col[0]; c6[1]=col[1]; c6[2]=col[2];\n"
       << "    c6[3]=col[1]*d[2]-col[2]*d[1]; c6[4]=col[2]*d[0]-col[0]*d[2]; c6[5]=col[0]*d[1]-col[1]*d[0];\n"
       << "  } else if (jt == 2) {\n"  // SLIDE
       << "    float ax[3] = {xaxis[xax_off+ji*3], xaxis[xax_off+ji*3+1], xaxis[xax_off+ji*3+2]};\n"
       << "    c6[3]=ax[0]; c6[4]=ax[1]; c6[5]=ax[2];\n"
       << "  } else if (jt == 1) {\n"  // BALL
       << "    float rm[9]; for (int k=0;k<9;k++) rm[k] = xmat[xm_off + bdi*9+k];\n"
       << "    float col[3] = {rm[ra], rm[3+ra], rm[6+ra]};\n"
       << "    float anc[3] = {xanchor[xa_off+ji*3], xanchor[xa_off+ji*3+1], xanchor[xa_off+ji*3+2]};\n"
       << "    float d[3]; for (int k=0;k<3;k++) d[k] = rc[k] - anc[k];\n"
       << "    c6[0]=col[0]; c6[1]=col[1]; c6[2]=col[2];\n"
       << "    c6[3]=col[1]*d[2]-col[2]*d[1]; c6[4]=col[2]*d[0]-col[0]*d[2]; c6[5]=col[0]*d[1]-col[1]*d[0];\n"
       << "  }\n"
       << "  for (int k=0;k<6;k++) scratch[s_off + S_CDOF + di*6+k] = c6[k];\n"
       << "}\n\n";

    // ═══ PHASE B: CRB → Mass Matrix ═══
    ss << "// ── Phase B: CRB + mass matrix ──\n";

    // Copy cinert to crb scratch
    ss << "for (int b=0;b<NB;b++) for (int k=0;k<10;k++)\n"
       << "  scratch[s_off + S_CRB + b*10+k] = cinert_out[ci_off + b*10+k];\n";

    // Zero world body in crb
    ss << "for (int k=0;k<10;k++) scratch[s_off + S_CRB + k] = 0;\n";

    // Backward accumulation of CRB (leaf→root)
    ss << "for (int b=NB-1;b>=1;b--) {\n"
       << "  int p = body_par[b];\n"
       << "  for (int k=0;k<10;k++) scratch[s_off + S_CRB + p*10+k] += scratch[s_off + S_CRB + b*10+k];\n"
       << "}\n\n";

    // Compute qM using dof parent chain (efficient sparse traversal)
    // Initialize qM to zero
    ss << "for (int i=0;i<NV*NV;i++) qM_out[qM_off+i] = 0;\n";

    // For each DOF i, compute crb_cdof[i] = inert_mul(crb[body_of_i], cdof[i])
    // Then walk parent chain: qM[i,j] = dot(crb_cdof[i], cdof[j])
    ss << "for (int i=0;i<NV;i++) {\n"
       << "  int bi = dof_bodyid[i];\n"
       // inert_mul: crb[bi] * cdof[i] → cc[6]
       << "  float I00=scratch[s_off+S_CRB+bi*10], I11=scratch[s_off+S_CRB+bi*10+1], I22=scratch[s_off+S_CRB+bi*10+2];\n"
       << "  float I01=scratch[s_off+S_CRB+bi*10+3], I02=scratch[s_off+S_CRB+bi*10+4], I12=scratch[s_off+S_CRB+bi*10+5];\n"
       << "  float px=scratch[s_off+S_CRB+bi*10+6], py=scratch[s_off+S_CRB+bi*10+7], pz=scratch[s_off+S_CRB+bi*10+8];\n"
       << "  float mass=scratch[s_off+S_CRB+bi*10+9];\n"
       << "  float w0=scratch[s_off+S_CDOF+i*6], w1=scratch[s_off+S_CDOF+i*6+1], w2=scratch[s_off+S_CDOF+i*6+2];\n"
       << "  float l0=scratch[s_off+S_CDOF+i*6+3], l1=scratch[s_off+S_CDOF+i*6+4], l2=scratch[s_off+S_CDOF+i*6+5];\n"
       // ang_out = I*w + p×lin
       << "  float cc0 = I00*w0+I01*w1+I02*w2 + (py*l2-pz*l1);\n"
       << "  float cc1 = I01*w0+I11*w1+I12*w2 + (pz*l0-px*l2);\n"
       << "  float cc2 = I02*w0+I12*w1+I22*w2 + (px*l1-py*l0);\n"
       // lin_out = mass*lin - p×ang
       << "  float cc3 = mass*l0 - (py*w2-pz*w1);\n"
       << "  float cc4 = mass*l1 - (pz*w0-px*w2);\n"
       << "  float cc5 = mass*l2 - (px*w1-py*w0);\n"
       // Walk parent chain
       << "  int j = i;\n"
       << "  while (j >= 0) {\n"
       << "    float dot = 0;\n"
       << "    for (int k=0;k<6;k++) {\n"
       << "      float ck = (k==0?cc0:k==1?cc1:k==2?cc2:k==3?cc3:k==4?cc4:cc5);\n"
       << "      dot += ck * scratch[s_off+S_CDOF+j*6+k];\n"
       << "    }\n"
       << "    qM_out[qM_off + i*NV+j] = dot;\n"
       << "    qM_out[qM_off + j*NV+i] = dot;\n"
       << "    j = dof_par[j];\n"
       << "  }\n"
       << "}\n";

    // Add armature to diagonal
    ss << "for (int i=0;i<NV;i++) qM_out[qM_off + i*NV+i] += dof_arm[i];\n\n";

    // ═══ PHASE C: COM velocity ═══
    ss << "// ── Phase C: COM velocity ──\n";

    // Initialize cvel and cdof_dot to zero
    ss << "for (int b=0;b<NB;b++) for (int k=0;k<6;k++) { cvel_out[cv_off+b*6+k]=0; scratch[s_off+S_CDOFD+k]=0; }\n"
       << "for (int di=0;di<NV;di++) for (int k=0;k<6;k++) scratch[s_off+S_CDOFD+di*6+k]=0;\n";

    // Forward propagation (root→leaf): cvel[b] = cvel[parent] + sum(cdof[di]*qvel[di])
    // cdof_dot[di] = motion_cross(cvel[partial], cdof[di])
    for (int b = 1; b < nb; b++) {
        int pid = bpar_ptr[b];
        ss << "{\n"
           << "  float cv[6]; for (int k=0;k<6;k++) cv[k] = cvel_out[cv_off+" << pid << "*6+k];\n";

        for (int di : body_dofs[b]) {
            // cdof_dot[di] = motion_cross(cv, cdof[di])
            ss << "  {\n"
               << "    float ua[3]={cv[0],cv[1],cv[2]}, ul[3]={cv[3],cv[4],cv[5]};\n"
               << "    float va[3],vl[3]; for (int k=0;k<3;k++) { va[k]=scratch[s_off+S_CDOF+" << di << "*6+k]; vl[k]=scratch[s_off+S_CDOF+" << di << "*6+3+k]; }\n"
               // motion_cross: ang=ua×va, lin=ul×va+ua×vl
               << "    scratch[s_off+S_CDOFD+" << di << "*6+0]=ua[1]*va[2]-ua[2]*va[1];\n"
               << "    scratch[s_off+S_CDOFD+" << di << "*6+1]=ua[2]*va[0]-ua[0]*va[2];\n"
               << "    scratch[s_off+S_CDOFD+" << di << "*6+2]=ua[0]*va[1]-ua[1]*va[0];\n"
               << "    scratch[s_off+S_CDOFD+" << di << "*6+3]=ul[1]*va[2]-ul[2]*va[1]+ua[1]*vl[2]-ua[2]*vl[1];\n"
               << "    scratch[s_off+S_CDOFD+" << di << "*6+4]=ul[2]*va[0]-ul[0]*va[2]+ua[2]*vl[0]-ua[0]*vl[2];\n"
               << "    scratch[s_off+S_CDOFD+" << di << "*6+5]=ul[0]*va[1]-ul[1]*va[0]+ua[0]*vl[1]-ua[1]*vl[0];\n"
               // cv += cdof[di] * qvel[di]
               << "    float qv = qvel[v_off+" << di << "];\n"
               << "    for (int k=0;k<6;k++) cv[k] += scratch[s_off+S_CDOF+" << di << "*6+k] * qv;\n"
               << "  }\n";
        }

        ss << "  for (int k=0;k<6;k++) cvel_out[cv_off+" << b << "*6+k] = cv[k];\n"
           << "}\n";
    }
    ss << "\n";

    // ═══ PHASE D: RNE (Newton-Euler) ═══
    ss << "// ── Phase D: RNE ──\n";

    // Forward: cacc[b] = cacc[parent] + sum(cdof_dot[di]*qvel[di])
    ss << "for (int b=0;b<NB;b++) for (int k=0;k<6;k++) scratch[s_off+S_CACC+b*6+k]=0;\n"
       << "scratch[s_off+S_CACC+3] = " << -grav[0] << "f;\n"
       << "scratch[s_off+S_CACC+4] = " << -grav[1] << "f;\n"
       << "scratch[s_off+S_CACC+5] = " << -grav[2] << "f;\n";

    for (int b = 1; b < nb; b++) {
        int pid = bpar_ptr[b];
        ss << "{\n"
           << "  float ac[6]; for (int k=0;k<6;k++) ac[k] = scratch[s_off+S_CACC+" << pid << "*6+k];\n";
        for (int di : body_dofs[b]) {
            ss << "  { float qv=qvel[v_off+" << di << "]; for (int k=0;k<6;k++) ac[k]+=scratch[s_off+S_CDOFD+" << di << "*6+k]*qv; }\n";
        }
        ss << "  for (int k=0;k<6;k++) scratch[s_off+S_CACC+" << b << "*6+k]=ac[k];\n"
           << "}\n";
    }

    // Local forces: f = I*a + v×(I*v)
    ss << "for (int b=0;b<NB;b++) {\n"
       // inert_mul(cinert[b], cacc[b]) → Ia
       << "  float I00=cinert_out[ci_off+b*10], I11=cinert_out[ci_off+b*10+1], I22=cinert_out[ci_off+b*10+2];\n"
       << "  float I01=cinert_out[ci_off+b*10+3], I02=cinert_out[ci_off+b*10+4], I12=cinert_out[ci_off+b*10+5];\n"
       << "  float px=cinert_out[ci_off+b*10+6], py=cinert_out[ci_off+b*10+7], pz=cinert_out[ci_off+b*10+8], mass=cinert_out[ci_off+b*10+9];\n"
       << "  float aw0=scratch[s_off+S_CACC+b*6], aw1=scratch[s_off+S_CACC+b*6+1], aw2=scratch[s_off+S_CACC+b*6+2];\n"
       << "  float al0=scratch[s_off+S_CACC+b*6+3], al1=scratch[s_off+S_CACC+b*6+4], al2=scratch[s_off+S_CACC+b*6+5];\n"
       << "  float Ia0=I00*aw0+I01*aw1+I02*aw2+(py*al2-pz*al1);\n"
       << "  float Ia1=I01*aw0+I11*aw1+I12*aw2+(pz*al0-px*al2);\n"
       << "  float Ia2=I02*aw0+I12*aw1+I22*aw2+(px*al1-py*al0);\n"
       << "  float Ia3=mass*al0-(py*aw2-pz*aw1);\n"
       << "  float Ia4=mass*al1-(pz*aw0-px*aw2);\n"
       << "  float Ia5=mass*al2-(px*aw1-py*aw0);\n"
       // inert_mul(cinert[b], cvel[b]) → Iv
       << "  float vw0=cvel_out[cv_off+b*6], vw1=cvel_out[cv_off+b*6+1], vw2=cvel_out[cv_off+b*6+2];\n"
       << "  float vl0=cvel_out[cv_off+b*6+3], vl1=cvel_out[cv_off+b*6+4], vl2=cvel_out[cv_off+b*6+5];\n"
       << "  float Iv0=I00*vw0+I01*vw1+I02*vw2+(py*vl2-pz*vl1);\n"
       << "  float Iv1=I01*vw0+I11*vw1+I12*vw2+(pz*vl0-px*vl2);\n"
       << "  float Iv2=I02*vw0+I12*vw1+I22*vw2+(px*vl1-py*vl0);\n"
       << "  float Iv3=mass*vl0-(py*vw2-pz*vw1);\n"
       << "  float Iv4=mass*vl1-(pz*vw0-px*vw2);\n"
       << "  float Iv5=mass*vl2-(px*vw1-py*vw0);\n"
       // motion_cross_force(cvel, Iv): ang=va×fa+vl×fl, lin=va×fl
       << "  float mcf0=vw1*Iv2-vw2*Iv1+vl1*Iv5-vl2*Iv4;\n"
       << "  float mcf1=vw2*Iv0-vw0*Iv2+vl2*Iv3-vl0*Iv5;\n"
       << "  float mcf2=vw0*Iv1-vw1*Iv0+vl0*Iv4-vl1*Iv3;\n"
       << "  float mcf3=vw1*Iv5-vw2*Iv4;\n"
       << "  float mcf4=vw2*Iv3-vw0*Iv5;\n"
       << "  float mcf5=vw0*Iv4-vw1*Iv3;\n"
       // f = Ia + vxIv
       << "  scratch[s_off+S_CFRC+b*6+0]=Ia0+mcf0;\n"
       << "  scratch[s_off+S_CFRC+b*6+1]=Ia1+mcf1;\n"
       << "  scratch[s_off+S_CFRC+b*6+2]=Ia2+mcf2;\n"
       << "  scratch[s_off+S_CFRC+b*6+3]=Ia3+mcf3;\n"
       << "  scratch[s_off+S_CFRC+b*6+4]=Ia4+mcf4;\n"
       << "  scratch[s_off+S_CFRC+b*6+5]=Ia5+mcf5;\n"
       << "}\n";

    // Backward accumulation of cfrc
    ss << "for (int b=NB-1;b>=1;b--) {\n"
       << "  int p = body_par[b];\n"
       << "  for (int k=0;k<6;k++) scratch[s_off+S_CFRC+p*6+k] += scratch[s_off+S_CFRC+b*6+k];\n"
       << "}\n";

    // qfrc_bias[di] = dot(cdof[di], cfrc[body_of_di])
    ss << "for (int di=0;di<NV;di++) {\n"
       << "  int b=dof_bodyid[di]; float dot=0;\n"
       << "  for (int k=0;k<6;k++) dot+=scratch[s_off+S_CDOF+di*6+k]*scratch[s_off+S_CFRC+b*6+k];\n"
       << "  scratch[s_off+S_BIAS+di]=dot;\n"
       << "}\n\n";

    // ═══ PHASE E: Passive + Actuation + qfrc_smooth ═══
    ss << "// ── Phase E: passive, actuation, qfrc_smooth ──\n";

    // Passive: spring + damping
    ss << "for (int di=0;di<NV;di++) {\n"
       << "  float f = 0;\n"
       << "  if (dof_stiff[di] != 0) f -= dof_stiff[di] * (qpos[q_off+dof_qa[di]] - qpos_spr[dof_qa[di]]);\n"
       << "  if (dof_damp[di] != 0) f -= dof_damp[di] * qvel[v_off+di];\n"
       << "  scratch[s_off+S_PASS+di] = f;\n"
       << "}\n";

    // Actuation: force = gain * ctrl + bias, qfrc_actuator = moment^T @ force
    if (nu > 0) {
        ss << "for (int di=0;di<NV;di++) {\n"
           << "  float fa = 0;\n"
           << "  for (int ai=0;ai<NU;ai++) {\n"
           << "    float mom = act_moment[ai*NV+di];\n"
           << "    if (mom != 0) {\n"
           << "      float force = act_g0[ai] * ctrl[u_off+ai] + act_b0[ai];\n"
           << "      fa += mom * force;\n"
           << "    }\n"
           << "  }\n"
           << "  qfrc_actuator_out[qa_off+di] = fa;\n"
           << "}\n";
    } else {
        ss << "for (int di=0;di<NV;di++) qfrc_actuator_out[qa_off+di] = 0;\n";
    }

    // qfrc_smooth = passive - bias + actuator
    ss << "for (int di=0;di<NV;di++) {\n"
       << "  qfrc_smooth_out[qfs_off+di] = scratch[s_off+S_PASS+di] - scratch[s_off+S_BIAS+di] + qfrc_actuator_out[qa_off+di];\n"
       << "}\n";

    // Copy cdof from scratch to output
    ss << "uint cd_off = bid * NV * 6;\n"
       << "for (int di=0;di<NV;di++) for (int k=0;k<6;k++)\n"
       << "  cdof_out[cd_off + di*6+k] = scratch[s_off+S_CDOF+di*6+k];\n";

    return ss.str();
}

static int forward_scratch_per_env(const Model& m) {
    int nb = m.nbody, nv = m.nv;
    return nb*10 + nv*6 + nv*6 + nb*6 + nb*6 + nb*3 + nb + nv + nv;
}

// ── Backend-agnostic host data builders shared by MSL (metal_kernel) and GLSL (mkx_kernels) collision/solver dispatch ──
static const int CONTACT_STRIDE = 8;
static const int MAX_CONTACTS_PER_ENV = 128;
static const int MAX_EFC = 256;

static mx::array build_collision_pair_data(const Model& m) {
    m.init_cache();
    const auto& c = m.cache;
    int npairs = (int)c.collision_pairs.size();

    mx::eval(m.geom_type, m.geom_size, m.geom_dataid,
             m.mesh_vertadr, m.mesh_vertnum, m.mesh_vert);
    auto gt = m.geom_type.data<int>();
    auto gs = m.geom_size.data<float>();
    auto gdi = m.geom_dataid.data<int>();

    // Compute per-geom bounding radius
    std::vector<float> geom_rb(m.ngeom, 0.0f);
    for (int g = 0; g < m.ngeom; g++) {
        int t = gt[g];
        if (t == 0) { geom_rb[g] = 1e6f; continue; }
        if (t == 2) { geom_rb[g] = gs[g*3]; continue; }
        if (t == 3) { geom_rb[g] = std::sqrt(gs[g*3]*gs[g*3] + gs[g*3+1]*gs[g*3+1]); continue; }
        if (t == 6) { geom_rb[g] = std::sqrt(gs[g*3]*gs[g*3] + gs[g*3+1]*gs[g*3+1] + gs[g*3+2]*gs[g*3+2]); continue; }
        if (t == 7 && gdi[g] >= 0) {
            auto mva = m.mesh_vertadr.data<int>();
            auto mvn = m.mesh_vertnum.data<int>();
            auto mv = m.mesh_vert.data<float>();
            int mid = gdi[g], adr = mva[mid], nverts = mvn[mid];
            float maxr = 0;
            for (int v = 0; v < nverts; v++) {
                float x = mv[(adr+v)*3], y = mv[(adr+v)*3+1], z = mv[(adr+v)*3+2];
                float r = std::sqrt(x*x + y*y + z*z);
                if (r > maxr) maxr = r;
            }
            geom_rb[g] = maxr;
        }
    }

    // Build pair data: g1, g2, type1, type2, margin+gap, rbound_sum
    std::vector<float> data(npairs * 6);
    for (int p = 0; p < npairs; p++) {
        const auto& cp = c.collision_pairs[p];
        data[p*6+0] = (float)cp.g1;
        data[p*6+1] = (float)cp.g2;
        data[p*6+2] = (float)cp.type1;
        data[p*6+3] = (float)cp.type2;
        data[p*6+4] = cp.margin + cp.gap;
        data[p*6+5] = geom_rb[cp.g1] + geom_rb[cp.g2] + cp.margin + cp.gap;
    }
    return mx::array(data.data(), {npairs * 6}, mx::float32);
}

static int solver_scratch_per_env(const Model& m) {
    int nv = m.nv;
    return nv*nv + MAX_EFC*nv + 5*MAX_EFC + 7*nv + nv*3;
}

// Build solver pair properties buffer (18 floats per pair)
static mx::array build_solver_pair_props(const Model& m) {
    m.init_cache();
    const auto& c = m.cache;
    int npairs = (int)c.collision_pairs.size();
    std::vector<float> data(npairs * 18, 0.0f);
    for (int p = 0; p < npairs; p++) {
        const auto& cp = c.collision_pairs[p];
        data[p*18+0] = (float)cp.body1;
        data[p*18+1] = (float)cp.body2;
        data[p*18+2] = (float)cp.condim;
        data[p*18+3] = cp.margin + cp.gap;
        for (int k = 0; k < 5; k++) data[p*18+4+k] = cp.friction[k];
        data[p*18+9] = cp.solref[0];
        data[p*18+10] = cp.solref[1];
        for (int k = 0; k < 5; k++) data[p*18+11+k] = cp.solimp[k];
        data[p*18+16] = cp.invweight_t;
        data[p*18+17] = cp.invweight_r;
    }
    return mx::array(data.data(), {npairs * 18}, mx::float32);
}

// Build body_dof_masks buffer (nb × nv flat)
static mx::array build_body_dof_masks(const Model& m) {
    m.init_cache();
    int nb = m.nbody, nv = m.nv;
    std::vector<float> data(nb * nv, 0.0f);
    for (int b = 0; b < nb; b++) {
        mx::eval(m.cache.body_dof_masks[b]);
        auto ptr = m.cache.body_dof_masks[b].data<float>();
        for (int d = 0; d < nv; d++) data[b * nv + d] = ptr[d];
    }
    return mx::array(data.data(), {nb * nv}, mx::float32);
}

// Build body_rootid buffer
static mx::array build_body_rootid(const Model& m) {
    m.init_cache();
    int nb = m.nbody;
    std::vector<float> data(nb);
    for (int b = 0; b < nb; b++) data[b] = (float)m.cache.body_rootid_vec[b];
    return mx::array(data.data(), {nb}, mx::float32);
}

// ── MSL header for forward kernel (spatial algebra helpers) ──────────────────

static const std::string FORWARD_HEADER = R"(
)";

// ── Test helper: dispatch Metal forward kernel for 1 env (MLX-only, tests/test_metal_synth.cpp) ──
#if defined(MJMLX_BACKEND_MLX)

MJMLX_API MetalForwardResult test_metal_forward(
    const Model& m,
    const mx::array& xipos, const mx::array& ximat,
    const mx::array& xanchor, const mx::array& xaxis, const mx::array& xmat,
    const mx::array& qpos, const mx::array& qvel, const mx::array& ctrl)
{
    m.init_cache();
    int nb = m.nbody, nv = m.nv, nq = m.nq, nu = m.nu, njnt = m.njnt;
    int scratchSz = forward_scratch_per_env(m);

    auto fwd_source = make_forward_source(m);

    auto mm = mx::astype(mx::flatten(m.cache.make_m_mask), mx::float32);
    mx::array am = mx::zeros({std::max(nv, 1)});
    if (nu > 0 && m.cache.act_moment_const.size() > 0) {
        am = mx::astype(mx::flatten(m.cache.act_moment_const), mx::float32);
    }
    mx::eval(mm, am);

    auto kernel = mx::fast::metal_kernel(
        "mjmlx_test_forward_" + std::to_string(nb) + "_" + std::to_string(nv),
        {"xipos", "ximat", "xanchor", "xaxis", "xmat",
         "qpos", "qvel", "ctrl",
         "make_m_mask", "act_moment"},
        {"qM_out", "qfrc_smooth_out", "subtree_com_out",
         "cinert_out", "cvel_out", "qfrc_actuator_out", "scratch", "cdof_out"},
        fwd_source,
        FORWARD_HEADER
    );

    int B = 1;
    auto fwd = kernel(
        {mx::astype(mx::flatten(xipos), mx::float32),
         mx::astype(mx::flatten(ximat), mx::float32),
         mx::astype(mx::flatten(xanchor), mx::float32),
         mx::astype(mx::flatten(xaxis), mx::float32),
         mx::astype(mx::flatten(xmat), mx::float32),
         mx::astype(mx::flatten(qpos), mx::float32),
         mx::astype(mx::flatten(qvel), mx::float32),
         mx::astype(mx::flatten(ctrl), mx::float32),
         mm, am},
        {{B * nv * nv}, {B * nv}, {B * nb * 3},
         {B * nb * 10}, {B * nb * 6}, {B * nv},
         {B * scratchSz}, {B * nv * 6}},
        {mx::float32, mx::float32, mx::float32,
         mx::float32, mx::float32, mx::float32, mx::float32, mx::float32},
        std::make_tuple(B, 1, 1), std::make_tuple(1, 1, 1),
        {}, std::nullopt, false, {}
    );

    MetalForwardResult r;
    r.qM = mx::reshape(fwd[0], {nv, nv});
    r.qfrc_smooth = fwd[1];
    r.subtree_com = mx::reshape(fwd[2], {nb, 3});
    r.cinert = mx::reshape(fwd[3], {nb, 10});
    r.cvel = mx::reshape(fwd[4], {nb, 6});
    r.qfrc_actuator = fwd[5];
    return r;
}

// Forward declarations for collision helpers
static const std::string COLLISION_HEADER_FWD = R"(
inline float3 msl_cross(float3 a, float3 b) {
    return float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline float msl_dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float msl_len(float3 a) { return sqrt(msl_dot(a, a)); }
inline float3 msl_norm(float3 a) {
    float l = msl_len(a);
    return l > 1e-12f ? a / l : float3(0, 0, 1);
}

// Generic GJK/EPA support point so plane-X and X-Y contact detection work for any geom-type pair, not just mesh. MuJoCo local-frame convention: sphere size=(radius,_,_), capsule size=(radius,halflength,_) axis along local +z, box size=(halfx,halfy,halfz).
template <typename MeshVertsPtr>
inline float3 mjmlx_support(int gtype, float3 pos, thread const float* R, float3 size,
                             MeshVertsPtr mesh_verts, int mesh_adr, int mesh_nverts,
                             float3 dir_world) {
    float3 ld = float3(R[0]*dir_world.x+R[3]*dir_world.y+R[6]*dir_world.z,
                        R[1]*dir_world.x+R[4]*dir_world.y+R[7]*dir_world.z,
                        R[2]*dir_world.x+R[5]*dir_world.y+R[8]*dir_world.z);
    float3 lp;
    if (gtype == 2) { // sphere
        lp = msl_norm(ld) * size.x;
    } else if (gtype == 4) { // ellipsoid, semi-axes = size; support(d) = (a^2 dx, b^2 dy, c^2 dz) / |(a dx, b dy, c dz)|
        float3 scaled = float3(size.x*ld.x, size.y*ld.y, size.z*ld.z);
        float denom = msl_len(scaled);
        lp = (denom > 1e-12f) ? float3(size.x*scaled.x, size.y*scaled.y, size.z*scaled.z) / denom : float3(0, 0, size.z);
    } else if (gtype == 3) { // capsule
        float3 n = msl_norm(ld);
        float capz = (ld.z >= 0.0f) ? size.y : -size.y;
        lp = float3(n.x*size.x, n.y*size.x, capz + n.z*size.x);
    } else if (gtype == 5) { // cylinder: disk (xy) x interval (z), support = sum of each factor's own support point
        float2 xy = float2(ld.x, ld.y);
        float xyl = sqrt(xy.x*xy.x + xy.y*xy.y);
        float2 xyn = (xyl > 1e-12f) ? xy / xyl : float2(1, 0);
        lp = float3(xyn.x*size.x, xyn.y*size.x, (ld.z >= 0.0f) ? size.y : -size.y);
    } else if (gtype == 6) { // box
        lp = float3(ld.x >= 0.0f ? size.x : -size.x,
                    ld.y >= 0.0f ? size.y : -size.y,
                    ld.z >= 0.0f ? size.z : -size.z);
    } else if (gtype == 7) { // mesh: linear scan over vertices
        float bd = -1e30f; int bi = 0;
        for (int vi = 0; vi < mesh_nverts; vi++) {
            float3 v = float3(mesh_verts[(mesh_adr+vi)*3], mesh_verts[(mesh_adr+vi)*3+1], mesh_verts[(mesh_adr+vi)*3+2]);
            float dd = msl_dot(v, ld);
            if (dd > bd) { bd = dd; bi = vi; }
        }
        lp = float3(mesh_verts[(mesh_adr+bi)*3], mesh_verts[(mesh_adr+bi)*3+1], mesh_verts[(mesh_adr+bi)*3+2]);
    } else {
        lp = float3(0, 0, 0);
    }
    return float3(R[0]*lp.x+R[1]*lp.y+R[2]*lp.z+pos.x,
                  R[3]*lp.x+R[4]*lp.y+R[5]*lp.z+pos.y,
                  R[6]*lp.x+R[7]*lp.y+R[8]*lp.z+pos.z);
}
)";
static std::string make_collision_source(const Model& m);

// ── Test helper: dispatch Metal collision kernel for 1 env ──────────────────

MJMLX_API MetalCollisionResult test_metal_collision(
    const Model& m,
    const mx::array& geom_xpos, const mx::array& geom_xmat)
{
    m.init_cache();
    int ng = m.ngeom;

    auto coll_source = make_collision_source(m);
    auto pair_data = build_collision_pair_data(m);
    mx::eval(pair_data);

    auto mesh_verts = mx::astype(mx::flatten(m.mesh_vert), mx::float32);
    auto mesh_vertadr = mx::astype(mx::flatten(m.mesh_vertadr), mx::float32);
    auto mesh_vertnum = mx::astype(mx::flatten(m.mesh_vertnum), mx::float32);
    auto geom_dataid = mx::astype(mx::flatten(m.geom_dataid), mx::float32);
    auto geom_size = mx::astype(mx::flatten(m.geom_size), mx::float32);
    mx::eval(mesh_verts, mesh_vertadr, mesh_vertnum, geom_dataid, geom_size);

    auto kernel = mx::fast::metal_kernel(
        "mjmlx_test_collision_" + std::to_string(ng),
        {"geom_xpos", "geom_xmat",
         "mesh_verts", "pair_data",
         "mesh_vertadr_buf", "mesh_vertnum_buf", "geom_dataid_buf", "geom_size_buf"},
        {"contact_data", "contact_count"},
        coll_source,
        COLLISION_HEADER_FWD
    );

    int B = 1;
    int con_buf_sz = B * MAX_CONTACTS_PER_ENV * CONTACT_STRIDE;
    auto result = kernel(
        {mx::astype(mx::flatten(geom_xpos), mx::float32),
         mx::astype(mx::flatten(geom_xmat), mx::float32),
         mesh_verts, pair_data,
         mesh_vertadr, mesh_vertnum, geom_dataid, geom_size},
        {{con_buf_sz}, {B}},
        {mx::float32, mx::float32},
        std::make_tuple(B, 1, 1), std::make_tuple(1, 1, 1),
        {}, std::nullopt, false, {}
    );

    MetalCollisionResult r;
    r.contact_data = result[0];
    r.contact_count = result[1];
    return r;
}

// ── Metal collision kernel source ────────────────────────────────────────────

// CONTACT_STRIDE, MAX_CONTACTS_PER_ENV, COLLISION_HEADER_FWD defined above

static std::string make_collision_source(const Model& m) {
    m.init_cache();
    const auto& c = m.cache;
    int ng = m.ngeom;
    int npairs = (int)c.collision_pairs.size();

    std::ostringstream ss;
    ss << std::scientific;

    ss << "uint bid = thread_position_in_grid.x;\n"
       << "const int NG = " << ng << ";\n"
       << "const int NUM_PAIRS = " << npairs << ";\n"
       << "const int MAX_CON = " << MAX_CONTACTS_PER_ENV << ";\n"
       << "const int STRIDE = " << CONTACT_STRIDE << ";\n\n"
       << "uint gx_off = bid * NG * 3;\n"
       << "uint gm_off = bid * NG * 9;\n"
       << "uint con_off = bid * MAX_CON * STRIDE;\n\n"
       << "for (int i = 0; i < MAX_CON * STRIDE; i++) contact_data[con_off + i] = 0;\n"
       << "int ncon = 0;\n\n";

    // pair_data layout per pair: g1, g2, type1, type2, margin+gap, rbound_sum = 6 floats
    ss << "const int PAIR_STRIDE = 6;\n\n";

    // Support mesh function
    ss << R"(
for (int p = 0; p < NUM_PAIRS; p++) {
    int pd = p * PAIR_STRIDE;
    int g1 = (int)pair_data[pd+0], g2 = (int)pair_data[pd+1];
    int t1 = (int)pair_data[pd+2], t2 = (int)pair_data[pd+3];
    float pair_margin = pair_data[pd+4];
    float rbound_sum = pair_data[pd+5];

    // Bounding sphere broadphase (skip for plane pairs)
    if (t1 != 0 && t2 != 0) {
        float3 c1 = float3(geom_xpos[gx_off+g1*3], geom_xpos[gx_off+g1*3+1], geom_xpos[gx_off+g1*3+2]);
        float3 c2 = float3(geom_xpos[gx_off+g2*3], geom_xpos[gx_off+g2*3+1], geom_xpos[gx_off+g2*3+2]);
        if (msl_len(c1 - c2) >= rbound_sum) continue;
    }

    float c_pos[3], c_norm[3], c_dist;
    bool has_contact = false;

    if (t1 == 0 || t2 == 0) {
        // Plane-X: deepest point of X along -normal (X = mesh/sphere/capsule/box)
        int plane_g = (t1 == 0) ? g1 : g2;
        int xg = (t1 == 0) ? g2 : g1;
        int xt = (t1 == 0) ? t2 : t1;
        float3 ppos = float3(geom_xpos[gx_off+plane_g*3], geom_xpos[gx_off+plane_g*3+1], geom_xpos[gx_off+plane_g*3+2]);
        float pR[9]; for (int k=0;k<9;k++) pR[k] = geom_xmat[gm_off+plane_g*9+k];
        float3 normal = float3(pR[2], pR[5], pR[8]);
        float3 xpos_ = float3(geom_xpos[gx_off+xg*3], geom_xpos[gx_off+xg*3+1], geom_xpos[gx_off+xg*3+2]);
        thread float xR[9]; for (int k=0;k<9;k++) xR[k] = geom_xmat[gm_off+xg*9+k];
        float3 xsize = float3(geom_size_buf[xg*3], geom_size_buf[xg*3+1], geom_size_buf[xg*3+2]);
        int mesh_id = (xt == 7) ? (int)geom_dataid_buf[xg] : 0;
        int adr = (xt == 7) ? (int)mesh_vertadr_buf[mesh_id] : 0;
        int nverts = (xt == 7) ? (int)mesh_vertnum_buf[mesh_id] : 0;

        float3 best_w = mjmlx_support(xt, xpos_, xR, xsize, mesh_verts, adr, nverts, -normal);
        float best_d = msl_dot(best_w - ppos, normal);
        if (best_d < pair_margin) {
            float3 cp = best_w - normal * best_d;
            c_pos[0]=cp.x; c_pos[1]=cp.y; c_pos[2]=cp.z;
            c_norm[0]=normal.x; c_norm[1]=normal.y; c_norm[2]=normal.z;
            c_dist = best_d;
            has_contact = true;
        }
    } else {
        // X-Y GJK/EPA (X, Y = mesh/sphere/capsule/box, any combination)
        float3 pos1 = float3(geom_xpos[gx_off+g1*3], geom_xpos[gx_off+g1*3+1], geom_xpos[gx_off+g1*3+2]);
        float3 pos2 = float3(geom_xpos[gx_off+g2*3], geom_xpos[gx_off+g2*3+1], geom_xpos[gx_off+g2*3+2]);
        thread float R1[9]; for (int k=0;k<9;k++) R1[k] = geom_xmat[gm_off+g1*9+k];
        thread float R2[9]; for (int k=0;k<9;k++) R2[k] = geom_xmat[gm_off+g2*9+k];
        float3 size1 = float3(geom_size_buf[g1*3], geom_size_buf[g1*3+1], geom_size_buf[g1*3+2]);
        float3 size2 = float3(geom_size_buf[g2*3], geom_size_buf[g2*3+1], geom_size_buf[g2*3+2]);
        int mid1 = (t1 == 7) ? (int)geom_dataid_buf[g1] : 0;
        int mid2 = (t2 == 7) ? (int)geom_dataid_buf[g2] : 0;
        int adr1 = (t1 == 7) ? (int)mesh_vertadr_buf[mid1] : 0;
        int adr2 = (t2 == 7) ? (int)mesh_vertadr_buf[mid2] : 0;
        int nv1 = (t1 == 7) ? (int)mesh_vertnum_buf[mid1] : 0;
        int nv2 = (t2 == 7) ? (int)mesh_vertnum_buf[mid2] : 0;

        #define SUPPORT1(DIR) mjmlx_support(t1, pos1, R1, size1, mesh_verts, adr1, nv1, DIR)
        #define SUPPORT2(DIR) mjmlx_support(t2, pos2, R2, size2, mesh_verts, adr2, nv2, DIR)

        float3 dir = pos2 - pos1;
        if (msl_len(dir) < 1e-12f) dir = float3(1,0,0);

        float3 sdiff[4], sa_pts[4], sb_pts[4];
        int sn = 0;
        float3 sup_a = SUPPORT1(dir);
        float3 sup_b = SUPPORT2(-dir);
        sdiff[0] = sup_a - sup_b; sa_pts[0] = sup_a; sb_pts[0] = sup_b;
        sn = 1; dir = -sdiff[0];
        if (msl_len(dir) < 1e-12f) dir = float3(1,0,0);
        bool gjk_overlap = false;

        for (int iter = 0; iter < 32; iter++) {
            sup_a = SUPPORT1(dir);
            sup_b = SUPPORT2(-dir);
            float3 new_sd = sup_a - sup_b;
            if (msl_dot(new_sd, dir) < 0) break;
            sdiff[sn] = new_sd; sa_pts[sn] = sup_a; sb_pts[sn] = sup_b;
            sn++;
            if (sn == 2) {
                float3 A=sdiff[1], B=sdiff[0], AB=B-A, AO=-A;
                if (msl_dot(AB,AO)>0) { dir=msl_cross(msl_cross(AB,AO),AB); if(msl_len(dir)<1e-12f)dir=AO; }
                else { sdiff[0]=A;sa_pts[0]=sa_pts[1];sb_pts[0]=sb_pts[1];sn=1;dir=AO; }
            } else if (sn == 3) {
                float3 A=sdiff[2],B=sdiff[1],C=sdiff[0],AB=B-A,AC=C-A,AO=-A;
                float3 ABC=msl_cross(AB,AC);
                if (msl_dot(msl_cross(ABC,AC),AO)>0) { sdiff[0]=C;sdiff[1]=A;sa_pts[0]=sa_pts[0];sa_pts[1]=sa_pts[2];sb_pts[0]=sb_pts[0];sb_pts[1]=sb_pts[2];sn=2;dir=msl_cross(msl_cross(AC,AO),AC);if(msl_len(dir)<1e-12f)dir=AO; }
                else if (msl_dot(msl_cross(AB,ABC),AO)>0) { sdiff[0]=B;sdiff[1]=A;sa_pts[0]=sa_pts[1];sa_pts[1]=sa_pts[2];sb_pts[0]=sb_pts[1];sb_pts[1]=sb_pts[2];sn=2;dir=msl_cross(msl_cross(AB,AO),AB);if(msl_len(dir)<1e-12f)dir=AO; }
                else if (msl_dot(ABC,AO)>0) { dir=ABC; }
                else { float3 t=sdiff[0];sdiff[0]=sdiff[1];sdiff[1]=t;t=sa_pts[0];sa_pts[0]=sa_pts[1];sa_pts[1]=t;t=sb_pts[0];sb_pts[0]=sb_pts[1];sb_pts[1]=t;dir=-ABC; }
            } else if (sn == 4) {
                float3 A=sdiff[3],B=sdiff[2],C=sdiff[1],D=sdiff[0];
                float3 AB=B-A,AC=C-A,AD=D-A,AO=-A;
                float3 ABC=msl_cross(AB,AC),ACD=msl_cross(AC,AD),ADB=msl_cross(AD,AB);
                bool abc_o=msl_dot(ABC,AO)>0, acd_o=msl_dot(ACD,AO)>0, adb_o=msl_dot(ADB,AO)>0;
                if (!abc_o&&!acd_o&&!adb_o) { gjk_overlap=true; break; }
                if (abc_o) { sdiff[0]=C;sdiff[1]=B;sdiff[2]=A;sa_pts[0]=sa_pts[1];sa_pts[1]=sa_pts[2];sa_pts[2]=sa_pts[3];sb_pts[0]=sb_pts[1];sb_pts[1]=sb_pts[2];sb_pts[2]=sb_pts[3];sn=3;dir=ABC; }
                else if (acd_o) { sdiff[0]=D;sdiff[1]=C;sdiff[2]=A;sa_pts[2]=sa_pts[3];sb_pts[2]=sb_pts[3];sn=3;dir=ACD; }
                else { float3 ts;sdiff[0]=B;sdiff[1]=D;sdiff[2]=A;ts=sa_pts[0];sa_pts[0]=sa_pts[2];sa_pts[2]=sa_pts[3];sa_pts[1]=ts;ts=sb_pts[0];sb_pts[0]=sb_pts[2];sb_pts[2]=sb_pts[3];sb_pts[1]=ts;sn=3;dir=ADB; }
            }
            if (msl_len(dir) < 1e-12f) dir = float3(1,0,0);
        }

        if (!gjk_overlap) {
            float3 p0=sdiff[0], p1=(sn>=2)?sdiff[1]:sdiff[0];
            float3 seg=p1-p0; float seg_sq=msl_dot(seg,seg);
            float t=(seg_sq>1e-12f)?-msl_dot(p0,seg)/seg_sq:0.0f;
            t=clamp(t,0.0f,1.0f);
            float gap_d=msl_len(p0+seg*t);
            if (gap_d < pair_margin) {
                float3 wa=sa_pts[0]+(sa_pts[min(sn-1,1)]-sa_pts[0])*t;
                float3 wb=sb_pts[0]+(sb_pts[min(sn-1,1)]-sb_pts[0])*t;
                float3 n=msl_norm(wa-wb);
                float3 cp=(wa+wb)*0.5f;
                c_pos[0]=cp.x;c_pos[1]=cp.y;c_pos[2]=cp.z;
                c_norm[0]=n.x;c_norm[1]=n.y;c_norm[2]=n.z;
                c_dist=gap_d; has_contact=true;
            }
        } else {
            // Penetration depth via direction sampling
            float bw=1e30f; float3 bn=float3(0,0,1),bpa,bpb;
            const float D=0.577350269f;
            float3 sd[14]={float3(1,0,0),float3(-1,0,0),float3(0,1,0),float3(0,-1,0),float3(0,0,1),float3(0,0,-1),
                float3(D,D,D),float3(-D,D,D),float3(D,-D,D),float3(D,D,-D),
                float3(-D,-D,D),float3(-D,D,-D),float3(D,-D,-D),float3(-D,-D,-D)};
            for (int si=0;si<14;si++) {
                float3 d=sd[si];
                float3 pa=SUPPORT1(d), pb=SUPPORT2(-d);
                float w=msl_dot(pa-pb,d);
                if(w<bw){bw=w;bn=d;bpa=pa;bpb=pb;}
            }
            for (int fi=0;fi<4&&sn>=3;fi++) {
                float3 e1,e2;
                if(fi==0){e1=sdiff[1]-sdiff[0];e2=sdiff[2]-sdiff[0];}
                else if(fi==1){e1=sdiff[2]-sdiff[0];e2=sdiff[min(sn-1,3)]-sdiff[0];}
                else if(fi==2){e1=sdiff[min(sn-1,3)]-sdiff[0];e2=sdiff[1]-sdiff[0];}
                else{e1=sdiff[2]-sdiff[1];e2=sdiff[min(sn-1,3)]-sdiff[1];}
                float3 fn=msl_cross(e1,e2);
                if(msl_len(fn)<1e-8f)continue;
                fn=msl_norm(fn);
                for(int s=-1;s<=1;s+=2){
                    float3 d=fn*(float)s;
                    float3 pa=SUPPORT1(d), pb=SUPPORT2(-d);
                    float w=msl_dot(pa-pb,d);
                    if(w<bw){bw=w;bn=d;bpa=pa;bpb=pb;}
                }
            }
            float pen=max(bw,0.0f);
            float3 cp=(bpa+bpb)*0.5f;
            c_pos[0]=cp.x;c_pos[1]=cp.y;c_pos[2]=cp.z;
            c_norm[0]=bn.x;c_norm[1]=bn.y;c_norm[2]=bn.z;
            c_dist=-pen; has_contact=true;
        }
        #undef SUPPORT1
        #undef SUPPORT2
    }

    if (has_contact && ncon < MAX_CON) {
        int idx = con_off + ncon * STRIDE;
        contact_data[idx+0]=c_pos[0]; contact_data[idx+1]=c_pos[1]; contact_data[idx+2]=c_pos[2];
        contact_data[idx+3]=c_norm[0]; contact_data[idx+4]=c_norm[1]; contact_data[idx+5]=c_norm[2];
        contact_data[idx+6]=c_dist;
        contact_data[idx+7]=(float)p;
        ncon++;
    }
}

contact_count[bid] = (float)ncon;
)";

    return ss.str();
}

// ── Metal constraint + Newton solver kernel ──────────────────────────────────

static std::string make_solver_source(const Model& m, int solver_iters = 1, int cg_iters = 15) {
    m.init_cache();
    const auto& c = m.cache;
    int nb = m.nbody, nv = m.nv, npairs = (int)c.collision_pairs.size();
    float timestep = m.opt.timestep;
    bool use_pyramidal = (m.opt.cone == ConeType::PYRAMIDAL);
    bool refsafe = (m.opt.disableflags & DisableBit::REFSAFE) == 0;

    // Scratch layout per env (all in solver_scratch buffer)
    int S_H = 0;
    int S_J = S_H + nv * nv;
    int S_D = S_J + MAX_EFC * nv;
    int S_AREF = S_D + MAX_EFC;
    int S_FORCE = S_AREF + MAX_EFC;
    int S_GRAD = S_FORCE + MAX_EFC;
    int S_SEARCH = S_GRAD + nv;
    int S_QACC = S_SEARCH + nv;
    int S_MA = S_QACC + nv;
    int S_JAREF = S_MA + nv;
    int S_ACTIVE = S_JAREF + MAX_EFC;
    int S_MV = S_ACTIVE + MAX_EFC;
    int S_JV = S_MV + nv;
    int S_JACP = S_JV + MAX_EFC;
    int SCRATCH_PER_ENV = S_JACP + nv * 3;

    std::ostringstream ss;

    // Threadgroup-parallel solver: NV threads per env cooperate on matrix ops
    ss << "uint bid = threadgroup_position_in_grid.x;\n"
       << "uint tid = thread_position_in_threadgroup.x;\n"
       << "const int NV = " << nv << ";\n"
       << "const int NB = " << nb << ";\n"
       << "const int MAX_EFC_N = " << MAX_EFC << ";\n"
       << "const int NSOLVE = " << solver_iters << ";\n"
       << "const int CON_STRIDE = " << CONTACT_STRIDE << ";\n"
       << "const int MAX_CON = " << MAX_CONTACTS_PER_ENV << ";\n"
       << "const float TIMESTEP = " << timestep << "f;\n"
       << "const float MJMINVAL_CV = 1e-12f;\n"
       << "const float MJMINVAL_SV = 1e-14f;\n"
       << "const float MJMINIMP = 0.0001f;\n"
       << "const float MJMAXIMP = 0.9999f;\n\n";

    // Scratch offsets via macros (avoid device pointer issues in MSL)
    ss << "uint s_off = bid * " << SCRATCH_PER_ENV << ";\n"
       << "#define H(i)         solver_scratch[s_off + " << S_H << " + (i)]\n"
       << "#define J(r,c)       solver_scratch[s_off + " << S_J << " + (r)*NV+(c)]\n"
       << "#define efc_D(i)     solver_scratch[s_off + " << S_D << " + (i)]\n"
       << "#define efc_aref(i)  solver_scratch[s_off + " << S_AREF << " + (i)]\n"
       << "#define efc_force(i) solver_scratch[s_off + " << S_FORCE << " + (i)]\n"
       << "#define grad(i)      solver_scratch[s_off + " << S_GRAD << " + (i)]\n"
       << "#define search_d(i)  solver_scratch[s_off + " << S_SEARCH << " + (i)]\n"
       << "#define qacc(i)      solver_scratch[s_off + " << S_QACC << " + (i)]\n"
       << "#define Ma(i)        solver_scratch[s_off + " << S_MA << " + (i)]\n"
       << "#define Jaref(i)     solver_scratch[s_off + " << S_JAREF << " + (i)]\n"
       << "#define act(i)       solver_scratch[s_off + " << S_ACTIVE << " + (i)]\n"
       << "#define Mv_arr(i)    solver_scratch[s_off + " << S_MV << " + (i)]\n"
       << "#define Jv_arr(i)    solver_scratch[s_off + " << S_JV << " + (i)]\n"
       << "#define jacp_tmp(i)  solver_scratch[s_off + " << S_JACP << " + (i)]\n\n";

    // State offsets
    ss << "uint qm_off = bid * NV * NV;\n"
       << "uint qfs_off = bid * NV;\n"
       << "uint cd_off = bid * NV * 6;\n"
       << "uint sc_off = bid * NB * 3;\n"
       << "uint qv_off = bid * NV;\n"
       << "uint con_off = bid * MAX_CON * CON_STRIDE;\n"
       << "uint qfc_off = bid * NV;\n\n";

    // Threadgroup shared state for cross-thread communication
    ss << "threadgroup int tg_nefc;\n"
       << "threadgroup float tg_ba;\n"
       << "threadgroup float tg_rr;\n"
       << "threadgroup float tg_pAp;\n\n";

    // CG buffer aliases (reuse H region which is NV*NV — only need 3*NV for CG)
    ss << "#define cg_r(i)   H(i)\n"
       << "#define cg_p(i)   H(NV + (i))\n"
       << "#define cg_Ap(i)  H(2*NV + (i))\n";

    // CG iterations for inner linear solve
    ss << "const int CG_ITERS = " << cg_iters << ";\n\n";

    // Zero scratch (parallel: each thread zeros a stride)
    ss << "for (int ii = (int)tid; ii < " << SCRATCH_PER_ENV << "; ii += NV) solver_scratch[s_off + ii] = 0;\n"
       << "threadgroup_barrier(mem_flags::mem_device);\n\n";

    // ── Constraint construction (thread 0 only — data-dependent serial work) ──
    ss << "if (tid == 0) {\n"
       << "int ncon = (int)contact_count_in[bid];\n"
       << "int nefc = 0;\n\n";

    // ── Build constraint rows from contacts ──
    ss << R"(
for (int ci = 0; ci < ncon && nefc < MAX_EFC_N - 4; ci++) {
    int ci_off = con_off + ci * CON_STRIDE;
    float c_pos_x = contact_data_in[ci_off+0];
    float c_pos_y = contact_data_in[ci_off+1];
    float c_pos_z = contact_data_in[ci_off+2];
    float c_norm_x = contact_data_in[ci_off+3];
    float c_norm_y = contact_data_in[ci_off+4];
    float c_norm_z = contact_data_in[ci_off+5];
    float c_dist = contact_data_in[ci_off+6];
    int pair_idx = (int)contact_data_in[ci_off+7];

    int pp = pair_idx * 18;
    int body1 = (int)pair_props[pp+0];
    int body2 = (int)pair_props[pp+1];
    int condim = (int)pair_props[pp+2];
    float pair_margin_v = pair_props[pp+3];
    float fri0 = pair_props[pp+4];
    float fri1 = pair_props[pp+5];
    float solref0 = pair_props[pp+9];
    float solref1_v = pair_props[pp+10];
    float si0 = pair_props[pp+11], si1 = pair_props[pp+12];
    float si2 = pair_props[pp+13], si3 = pair_props[pp+14], si4 = pair_props[pp+15];
    float invw_t = pair_props[pp+16];

    bool contact_active = (c_dist < pair_margin_v);
    float pos = c_dist;

    float tc = solref0;
)";
    if (!refsafe) ss << "    tc = max(tc, 2.0f * TIMESTEP);\n";
    ss << R"(
    float dmin = clamp(si0, MJMINIMP, MJMAXIMP);
    float dmax_v = clamp(si1, MJMINIMP, MJMAXIMP);
    float width_v = max(si2, MJMINVAL_CV);
    float mid_v = clamp(si3, MJMINIMP, MJMAXIMP);
    float power_v = max(si4, 1.0f);

    float k_val = (tc > 0) ? 1.0f/(dmax_v*dmax_v*tc*tc*solref1_v*solref1_v) : -tc/(dmax_v*dmax_v);
    float b_val = (solref1_v > 0) ? 2.0f/(dmax_v*tc) : -solref1_v/dmax_v;

    float imp_x_val = abs(pos) / width_v;
    float imp_y_val;
    if (imp_x_val < mid_v) {
        imp_y_val = pow(imp_x_val, power_v) / pow(mid_v, power_v - 1.0f);
    } else {
        imp_y_val = 1.0f - pow(1.0f - imp_x_val, power_v) / pow(1.0f - mid_v, power_v - 1.0f);
    }
    float imp_val = clamp(dmin + imp_y_val * (dmax_v - dmin), dmin, dmax_v);
    if (imp_x_val > 1.0f) imp_val = dmax_v;

    // Compute Jacobian djacp via body_dof_masks and cdof
    for (int di = 0; di < NV*3; di++) jacp_tmp(di) = 0;

    // body2 (+)
    {
        int rid = (int)body_rootid_buf[body2];
        float ox = c_pos_x - subtree_com_in[sc_off + rid*3];
        float oy = c_pos_y - subtree_com_in[sc_off + rid*3+1];
        float oz = c_pos_z - subtree_com_in[sc_off + rid*3+2];
        for (int di = 0; di < NV; di++) {
            if (body_dof_masks_buf[body2*NV+di] < 0.5f) continue;
            float ax=cdof_in[cd_off+di*6],ay=cdof_in[cd_off+di*6+1],az=cdof_in[cd_off+di*6+2];
            float lx=cdof_in[cd_off+di*6+3],ly=cdof_in[cd_off+di*6+4],lz=cdof_in[cd_off+di*6+5];
            jacp_tmp(di*3+0) += lx + (ay*oz - az*oy);
            jacp_tmp(di*3+1) += ly + (az*ox - ax*oz);
            jacp_tmp(di*3+2) += lz + (ax*oy - ay*ox);
        }
    }
    // body1 (-)
    {
        int rid = (int)body_rootid_buf[body1];
        float ox = c_pos_x - subtree_com_in[sc_off + rid*3];
        float oy = c_pos_y - subtree_com_in[sc_off + rid*3+1];
        float oz = c_pos_z - subtree_com_in[sc_off + rid*3+2];
        for (int di = 0; di < NV; di++) {
            if (body_dof_masks_buf[body1*NV+di] < 0.5f) continue;
            float ax=cdof_in[cd_off+di*6],ay=cdof_in[cd_off+di*6+1],az=cdof_in[cd_off+di*6+2];
            float lx=cdof_in[cd_off+di*6+3],ly=cdof_in[cd_off+di*6+4],lz=cdof_in[cd_off+di*6+5];
            jacp_tmp(di*3+0) -= lx + (ay*oz - az*oy);
            jacp_tmp(di*3+1) -= ly + (az*ox - ax*oz);
            jacp_tmp(di*3+2) -= lz + (ax*oy - ay*ox);
        }
    }

    // j_normal = normal^T @ djacp^T
    float j_normal_arr[)";
    ss << nv << R"(];
    for (int di = 0; di < NV; di++)
        j_normal_arr[di] = c_norm_x*jacp_tmp(di*3) + c_norm_y*jacp_tmp(di*3+1) + c_norm_z*jacp_tmp(di*3+2);
)";

    if (use_pyramidal) {
        ss << R"(
    float3 nn = float3(c_norm_x, c_norm_y, c_norm_z);
    float3 tt1, tt2;
    if (abs(nn.z) < 0.999f) tt1 = msl_cross(nn, float3(0,0,1));
    else tt1 = msl_cross(nn, float3(0,1,0));
    tt1 = msl_norm(tt1); tt2 = msl_cross(nn, tt1);

    if (condim >= 3 && contact_active) {
        float mu0 = fri0, mu0_sq = mu0*mu0;
        float invw_py = invw_t + mu0_sq*invw_t;
        float r_first = max(invw_py*(1.0f-imp_val)/imp_val, MJMINVAL_CV);
)";
        ss << "        float r_py = max(2.0f*mu0_sq/" << std::fixed << std::setprecision(6) << m.opt.impratio << "f*r_first, MJMINVAL_CV);\n";
        ss << R"(
        float D_py = 1.0f/r_py;
        float mu_vals[2] = {fri0, fri1};
        for (int tk = 0; tk < 2; tk++) {
            float mu_k = mu_vals[tk];
            float3 tang = (tk==0) ? tt1 : tt2;
            for (int di = 0; di < NV; di++) {
                float jt = tang.x*jacp_tmp(di*3)+tang.y*jacp_tmp(di*3+1)+tang.z*jacp_tmp(di*3+2);
                J(nefc,di) = j_normal_arr[di] + mu_k*jt;
            }
            float jdot = 0;
            for (int di = 0; di < NV; di++) jdot += J(nefc,di)*qvel_in[qv_off+di];
            efc_D(nefc) = D_py;
            efc_aref(nefc) = -b_val*jdot - k_val*imp_val*pos;
            nefc++;
            for (int di = 0; di < NV; di++) {
                float jt = tang.x*jacp_tmp(di*3)+tang.y*jacp_tmp(di*3+1)+tang.z*jacp_tmp(di*3+2);
                J(nefc,di) = j_normal_arr[di] - mu_k*jt;
            }
            jdot = 0;
            for (int di = 0; di < NV; di++) jdot += J(nefc,di)*qvel_in[qv_off+di];
            efc_D(nefc) = D_py;
            efc_aref(nefc) = -b_val*jdot - k_val*imp_val*pos;
            nefc++;
        }
    } else if (contact_active) {
)";
    } else {
        ss << "    if (contact_active) {\n";
    }

    ss << R"(
        for (int di = 0; di < NV; di++) J(nefc,di) = j_normal_arr[di];
        float r = max(invw_t*(1.0f-imp_val)/imp_val, MJMINVAL_CV);
        efc_D(nefc) = 1.0f/r;
        float jdot = 0;
        for (int di = 0; di < NV; di++) jdot += j_normal_arr[di]*qvel_in[qv_off+di];
        efc_aref(nefc) = -b_val*jdot - k_val*imp_val*pos;
        nefc++;
    }
}
tg_nefc = nefc;
} // end if (tid == 0)
threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
int nefc = tg_nefc;

// ── Initial qacc = 0 ──
// Starting from zero ensures all penetrating contacts are detected as active
// (Jaref = -efc_aref < 0), so the Newton solver can converge properly.
// A diagonal solve would produce wildly inaccurate values for coupled DOFs,
// causing constraints to appear inactive.
qacc((int)tid) = 0;
threadgroup_barrier(mem_flags::mem_device);

// Early exit if no constraints (all threads)
if (nefc == 0) {
    qfrc_constraint_out[qfc_off + (int)tid] = 0;
    return;
}

// Ma = M @ qacc (parallel: each thread computes one element)
{ float s = 0; for (int j = 0; j < NV; j++) s += qM_in[qm_off + (int)tid*NV + j]*qacc(j); Ma((int)tid) = s; }
threadgroup_barrier(mem_flags::mem_device);

// Jaref, act, force (thread 0 — nefc-dependent)
if (tid == 0) {
    for (int r2 = 0; r2 < nefc; r2++) {
        float s = 0; for (int j = 0; j < NV; j++) s += J(r2,j)*qacc(j);
        Jaref(r2) = s - efc_aref(r2);
    }
    for (int r2 = 0; r2 < nefc; r2++) act(r2) = (Jaref(r2) < 0) ? 1.0f : 0.0f;
    for (int r2 = 0; r2 < nefc; r2++) efc_force(r2) = efc_D(r2)*(-Jaref(r2))*act(r2);
}
threadgroup_barrier(mem_flags::mem_device);

// grad (parallel)
{ float s = 0; for (int r2 = 0; r2 < nefc; r2++) s += J(r2,(int)tid)*efc_force(r2);
  grad((int)tid) = Ma((int)tid) - qfrc_smooth_in[qfs_off+(int)tid] - s; }
threadgroup_barrier(mem_flags::mem_device);

// ── Newton solver iterations (threadgroup-parallel) ──
#pragma clang loop unroll(disable)
for (int iter = 0; iter < NSOLVE; iter++) {
    // CG solve: search_d = -H^{-1} grad, where H = M + J^T D_act J
    // Avoids explicit H construction and numerically unstable Cholesky
    search_d((int)tid) = 0;
    cg_r((int)tid) = -grad((int)tid);
    cg_p((int)tid) = -grad((int)tid);
    threadgroup_barrier(mem_flags::mem_device);

    // Initial r·r
    if (tid == 0) { float s = 0; for (int i = 0; i < NV; i++) { float v = cg_r(i); s += v*v; } tg_rr = s; }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    #pragma clang loop unroll(disable)
    for (int cg_it = 0; cg_it < CG_ITERS; cg_it++) {
        if (tg_rr < 1e-20f) break;

        // Ap = H*p = M*p + MJMINVAL_SV*p + J^T*(D_act*(J*p))
        // M*p + regularization (parallel)
        float mp = MJMINVAL_SV * cg_p((int)tid);
        for (int j = 0; j < NV; j++) mp += qM_in[qm_off + (int)tid*NV + j] * cg_p(j);

        // J*p → D_act*(J*p) (parallel over constraint rows)
        for (int r2 = (int)tid; r2 < nefc; r2 += NV) {
            float s = 0; for (int j = 0; j < NV; j++) s += J(r2,j)*cg_p(j);
            Jv_arr(r2) = efc_D(r2)*act(r2)*s;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // J^T * (D_act * J*p) (parallel)
        float jt = 0; for (int r2 = 0; r2 < nefc; r2++) jt += J(r2,(int)tid)*Jv_arr(r2);
        cg_Ap((int)tid) = mp + jt;
        threadgroup_barrier(mem_flags::mem_device);

        // p·Ap reduction
        if (tid == 0) { float s = 0; for (int i = 0; i < NV; i++) s += cg_p(i)*cg_Ap(i); tg_pAp = s; }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        float alpha_cg = tg_rr / max(tg_pAp, 1e-30f);
        search_d((int)tid) += alpha_cg * cg_p((int)tid);
        cg_r((int)tid) -= alpha_cg * cg_Ap((int)tid);
        threadgroup_barrier(mem_flags::mem_device);

        // New r·r
        float old_rr = tg_rr;
        if (tid == 0) { float s = 0; for (int i = 0; i < NV; i++) { float v = cg_r(i); s += v*v; } tg_rr = s; }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        float beta_cg = tg_rr / max(old_rr, 1e-30f);
        cg_p((int)tid) = cg_r((int)tid) + beta_cg * cg_p((int)tid);
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Mv = M @ search (parallel)
    { float s = 0; for (int j = 0; j < NV; j++) s += qM_in[qm_off + (int)tid*NV + j]*search_d(j); Mv_arr((int)tid) = s; }
    threadgroup_barrier(mem_flags::mem_device);

    // Jv + line search + efc updates (thread 0)
    if (tid == 0) {
        for (int r2 = 0; r2 < nefc; r2++) {
            float s = 0; for (int j = 0; j < NV; j++) s += J(r2,j)*search_d(j);
            Jv_arr(r2) = s;
        }
        float qg=0,lg=0,qc=0,lc=0;
        for (int i = 0; i < NV; i++) { qg += 0.5f*search_d(i)*Mv_arr(i); lg += search_d(i)*(Ma(i)-qfrc_smooth_in[qfs_off+i]); }
        for (int r2 = 0; r2 < nefc; r2++) { qc += 0.5f*efc_D(r2)*Jv_arr(r2)*Jv_arr(r2)*act(r2); lc += efc_D(r2)*Jv_arr(r2)*Jaref(r2)*act(r2); }
        float dnom = 2.0f*(qg+qc);
        float an = clamp(-(lg+lc)/max(dnom, MJMINVAL_SV), -2.0f, 2.0f);
        float als[5] = {an, 0.5f*an, 0.1f*an, 0.01f, 0.001f};
        float bc = 1e30f, ba_local = 0;
        float cc2=0; for (int r2 = 0; r2 < nefc; r2++) cc2 += 0.5f*efc_D(r2)*Jaref(r2)*Jaref(r2)*act(r2);
        float cg=0; for (int i = 0; i < NV; i++) cg += 0.5f*(Ma(i)-qfrc_smooth_in[qfs_off+i])*qacc(i);
        bc = cc2+cg;
        for (int ai = 0; ai < 5; ai++) {
            float a = als[ai]; float tc2=0;
            for (int r2 = 0; r2 < nefc; r2++) { float x = Jaref(r2)+a*Jv_arr(r2); float ar = (x<0)?1.0f:0.0f; tc2 += 0.5f*efc_D(r2)*x*x*ar; }
            float tg2 = cg + a*lg + 0.5f*a*a*(2.0f*qg); float tt = tc2+tg2;
            if (tt < bc) { bc=tt; ba_local=a; }
        }
        tg_ba = ba_local;
        for (int r2 = 0; r2 < nefc; r2++) Jaref(r2) += ba_local*Jv_arr(r2);
        for (int r2 = 0; r2 < nefc; r2++) act(r2) = (Jaref(r2)<0)?1.0f:0.0f;
        for (int r2 = 0; r2 < nefc; r2++) efc_force(r2) = efc_D(r2)*(-Jaref(r2))*act(r2);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float ba = tg_ba;

    // Update qacc, Ma (parallel)
    qacc((int)tid) += ba * search_d((int)tid);
    Ma((int)tid) += ba * Mv_arr((int)tid);
    threadgroup_barrier(mem_flags::mem_device);

    // Update grad (parallel)
    { float s = 0; for (int r2 = 0; r2 < nefc; r2++) s += J(r2,(int)tid)*efc_force(r2);
      grad((int)tid) = Ma((int)tid) - qfrc_smooth_in[qfs_off+(int)tid] - s; }
    threadgroup_barrier(mem_flags::mem_device);
}

// Final qfrc_constraint (parallel: each thread computes one DOF)
{ float s = 0; for (int r2 = 0; r2 < nefc; r2++) s += J(r2,(int)tid)*efc_force(r2);
  qfrc_constraint_out[qfc_off + (int)tid] = s; }
)";

    return ss.str();
}

// Test helper for solver kernel
MJMLX_API MetalCollisionResult test_metal_solver(
    const Model& m,
    const mx::array& qM, const mx::array& qfrc_smooth,
    const mx::array& cdof, const mx::array& subtree_com,
    const mx::array& qvel,
    const mx::array& contact_data, const mx::array& contact_count)
{
    int nv = m.nv, nb = m.nbody;
    // GPU CG solver: cap iterations for performance. MuJoCo C defaults to 100
    // Newton iters with exact Cholesky; our approximate CG needs fewer outer
    // iterations. For RL training, 1 Newton + 10 CG gives sufficient contact
    // resolution while keeping GPU overhead manageable.
    int si = (nv > 80) ? std::min(std::max(m.opt.iterations, 1), 3) : std::max(m.opt.iterations, 1);
    int cgi = (nv > 80) ? 20 : 50;
    auto solver_source = make_solver_source(m, si, cgi);
    auto pair_props = build_solver_pair_props(m);
    auto body_dof_masks_buf = build_body_dof_masks(m);
    auto body_rootid_buf = build_body_rootid(m);
    mx::eval(pair_props, body_dof_masks_buf, body_rootid_buf);

    auto kernel = mx::fast::metal_kernel(
        "mjmlx_test_solver_" + std::to_string(nv) + "_s" + std::to_string(si) + "_c" + std::to_string(cgi),
        {"qM_in", "qfrc_smooth_in", "cdof_in", "subtree_com_in",
         "qvel_in", "contact_data_in", "contact_count_in",
         "pair_props", "body_dof_masks_buf", "body_rootid_buf"},
        {"qfrc_constraint_out", "solver_scratch"},
        solver_source,
        COLLISION_HEADER_FWD
    );

    int B = 1;
    int scratch_sz = solver_scratch_per_env(m);
    auto result = kernel(
        {mx::astype(mx::flatten(qM), mx::float32),
         mx::astype(mx::flatten(qfrc_smooth), mx::float32),
         mx::astype(mx::flatten(cdof), mx::float32),
         mx::astype(mx::flatten(subtree_com), mx::float32),
         mx::astype(mx::flatten(qvel), mx::float32),
         mx::astype(mx::flatten(contact_data), mx::float32),
         mx::astype(mx::flatten(contact_count), mx::float32),
         pair_props, body_dof_masks_buf, body_rootid_buf},
        {{B * nv}, {B * scratch_sz}},
        {mx::float32, mx::float32},
        std::make_tuple(B * nv, 1, 1), std::make_tuple(nv, 1, 1),
        {}, std::nullopt, false, {}
    );

    MetalCollisionResult r;
    r.contact_data = result[0]; // qfrc_constraint
    r.contact_count = mx::array(0.0f); // unused
    return r;
}

#endif // MJMLX_BACKEND_MLX

// ── Context: Metal kernels + model constants ─────────────────────────────────

#if defined(MJMLX_BACKEND_MLX)
using KernelFn = mx::fast::CustomKernelFunction;
#else
using KernelFn = mkx::fast::Kernel;
#endif

struct BatchedStepContext {
    std::optional<KernelFn> kin_kernel;
    std::optional<KernelFn> euler_kernel;
    std::optional<KernelFn> euler_devmem_kernel;
    std::optional<KernelFn> forward_kernel;
    bool uses_devmem_euler = false;
    bool uses_metal_forward = false;

    mx::array body_parentid = mx::zeros({1}, mx::int32);
    mx::array body_pos = mx::zeros({1});
    mx::array body_quat = mx::zeros({1});
    mx::array body_ipos = mx::zeros({1});
    mx::array body_iquat = mx::zeros({1});
    mx::array body_jntadr = mx::zeros({1}, mx::int32);
    mx::array body_jntnum = mx::zeros({1}, mx::int32);
    mx::array jnt_type_arr = mx::zeros({1}, mx::int32);
    mx::array jnt_qposadr_arr = mx::zeros({1}, mx::int32);
    mx::array jnt_pos_arr = mx::zeros({1});
    mx::array jnt_axis_arr = mx::zeros({1});
    mx::array qpos0 = mx::zeros({1});
    mx::array geom_bodyid_arr = mx::zeros({1}, mx::int32);
    mx::array geom_pos_arr = mx::zeros({1});
    mx::array geom_quat_arr = mx::zeros({1});

    // Forward kernel model constants
    mx::array make_m_mask = mx::zeros({1});
    mx::array act_moment = mx::zeros({1});
    int fwd_scratch_per_env = 0;

    // Collision kernel
    std::optional<KernelFn> collision_kernel;
    bool uses_metal_collision = false;
    mx::array coll_pair_data = mx::zeros({1});
    mx::array coll_mesh_verts = mx::zeros({1});
    mx::array coll_mesh_vertadr = mx::zeros({1});
    mx::array coll_mesh_vertnum = mx::zeros({1});
    mx::array coll_geom_dataid = mx::zeros({1});
    mx::array coll_geom_size = mx::zeros({1});
    int num_collision_pairs = 0;

    // Solver kernel
    std::optional<KernelFn> solver_kernel;
    bool uses_metal_solver = false;
    mx::array solver_pair_props = mx::zeros({1});
    mx::array solver_body_dof_masks = mx::zeros({1});
    mx::array solver_body_rootid = mx::zeros({1});
    int solver_scratch_size = 0;

    int nbody = 0, njnt = 0, nq = 0, nv = 0, nu = 0, ngeom = 0;
};

static std::shared_ptr<BatchedStepContext> build_context(const Model& m, int solver_iters_override = 0) {
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
    // DECISION: Metal kernel stack memory is limited (~24KB per thread). The kinematics
    // kernel needs ~7 floats per body (pos, quat) as working storage. If nbody is too
    // large, we skip the Metal kernel and fall back to the scalar CPU path.
    int stack_bytes = m.nbody * 7 * 4;
    if (stack_bytes <= 24000) {
#if defined(MJMLX_BACKEND_MKX)
        ctx->kin_kernel = mkx_kernels::make_kinematics_kernel(m.nbody, m.njnt, m.nq, m.ngeom);
#else
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
#endif
    }

    // Extract joint integration plan (shared by both euler kernel variants)
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

    auto damp_ptr = m.dof_damping.data<float>();
    std::vector<float> damp_vals(damp_ptr, damp_ptr + m.nv);

    // Build Euler kernel — two tiers:
    //   nv ≤ 80:   thread-local kernel (fastest, everything in registers/stack)
    //   nv ≤ 2048: device-memory kernel (L in global GPU memory, vectors thread-local)
    int euler_stack = 2 * m.nv * m.nv + 4 * m.nv + m.nq;
    if (euler_stack * 4 <= 24000 && m.nv <= 80) {
#if defined(MJMLX_BACKEND_MKX)
        ctx->euler_kernel = mkx_kernels::make_euler_kernel(m.nv, m.nq, m.opt.timestep, s_qa, s_da, fj, bj, damp_vals);
#else
        auto euler_source = make_euler_source(
            m.nv, m.nq, m.opt.timestep,
            s_qa, s_da, fj, bj, damp_vals);

        ctx->euler_kernel = mx::fast::metal_kernel(
            "mjmlx_euler_" + std::to_string(m.nv) + "_" + std::to_string(m.nq),
            {"qM", "qfrc_smooth", "qfrc_constraint", "qvel_in", "qpos_in"},
            {"qpos_out", "qvel_out", "qacc_out"},
            euler_source
        );
#endif
    } else if (m.nv <= EULER_DEVMEM_MAX_NV) {
#if defined(MJMLX_BACKEND_MKX)
        ctx->euler_devmem_kernel = mkx_kernels::make_euler_devmem_kernel(m.nv, m.nq, m.opt.timestep, s_qa, s_da, fj, bj, damp_vals);
        ctx->uses_devmem_euler = true;
#else
        auto euler_source = make_euler_devmem_source(
            m.nv, m.nq, m.opt.timestep,
            s_qa, s_da, fj, bj, damp_vals);

        ctx->euler_devmem_kernel = mx::fast::metal_kernel(
            "mjmlx_euler_devmem_" + std::to_string(m.nv) + "_" + std::to_string(m.nq),
            {"qM", "qfrc_smooth", "qfrc_constraint", "qvel_in", "qpos_in"},
            {"qpos_out", "qvel_out", "qacc_out", "L_scratch"},
            euler_source
        );
        ctx->uses_devmem_euler = true;
#endif
    }

    // Build all-Metal/GLSL forward kernel whenever there's anything to collide (see ponytail note above the collision/solver gates below for why nv alone no longer excludes this path).
    {
        ctx->fwd_scratch_per_env = forward_scratch_per_env(m);
        if (m.nu > 0 && m.cache.act_moment_const.size() > 0) {
            ctx->act_moment = mx::astype(mx::flatten(m.cache.act_moment_const), mx::float32);
        } else {
            ctx->act_moment = mx::zeros({std::max(m.nv, 1)});
        }
        mx::eval(ctx->act_moment);

#if defined(MJMLX_BACKEND_MKX)
        ForwardHostConsts fc = extract_forward_host_consts(m);
        ctx->forward_kernel = mkx_kernels::make_forward_kernel(
            m.nbody, m.nv, m.nq, m.nu, m.njnt, m.opt.timestep, fc.gravity,
            fc.body_parentid, fc.body_rootid, fc.body_mass, fc.body_inertia,
            fc.dof_bodyid, fc.dof_parentid, fc.dof_damping, fc.dof_armature,
            fc.dof_stiffness, fc.dof_qposadr, fc.qpos_spring, fc.act_gain0, fc.act_bias0,
            fc.dof_jtype, fc.dof_rotaxis, fc.dof_jid, fc.body_dofs, fc.jnt_dofadr0);
#else
        // Prepare model constant buffers
        ctx->make_m_mask = mx::astype(mx::flatten(m.cache.make_m_mask), mx::float32);
        mx::eval(ctx->make_m_mask);

        auto fwd_source = make_forward_source(m);
        ctx->forward_kernel = mx::fast::metal_kernel(
            "mjmlx_forward_" + std::to_string(m.nbody) + "_" + std::to_string(m.nv),
            {"xipos", "ximat", "xanchor", "xaxis", "xmat",
             "qpos", "qvel", "ctrl",
             "make_m_mask", "act_moment"},
            {"qM_out", "qfrc_smooth_out", "subtree_com_out",
             "cinert_out", "cvel_out", "qfrc_actuator_out", "scratch", "cdof_out"},
            fwd_source,
            FORWARD_HEADER
        );
#endif
        ctx->uses_metal_forward = true;
    }

    // Build collision kernel (replaces vmap collision) whenever there's anything to collide.
    if (m.cache.collision_pairs.size() > 0) {
        ctx->num_collision_pairs = (int)m.cache.collision_pairs.size();

        ctx->coll_pair_data = build_collision_pair_data(m);
        mx::eval(ctx->coll_pair_data);

        // Mesh vertex data
        ctx->coll_mesh_verts = mx::astype(mx::flatten(m.mesh_vert), mx::float32);
        ctx->coll_mesh_vertadr = mx::astype(mx::flatten(m.mesh_vertadr), mx::float32);
        ctx->coll_mesh_vertnum = mx::astype(mx::flatten(m.mesh_vertnum), mx::float32);
        ctx->coll_geom_dataid = mx::astype(mx::flatten(m.geom_dataid), mx::float32);
        ctx->coll_geom_size = mx::astype(mx::flatten(m.geom_size), mx::float32);
        mx::eval(ctx->coll_mesh_verts, ctx->coll_mesh_vertadr,
                 ctx->coll_mesh_vertnum, ctx->coll_geom_dataid, ctx->coll_geom_size);

#if defined(MJMLX_BACKEND_MKX)
        ctx->collision_kernel = mkx_kernels::make_collision_kernel(
            m.ngeom, (int)m.cache.collision_pairs.size(), MAX_CONTACTS_PER_ENV);
#else
        auto coll_source = make_collision_source(m);
        ctx->collision_kernel = mx::fast::metal_kernel(
            "mjmlx_collision_" + std::to_string(m.ngeom),
            {"geom_xpos", "geom_xmat",
             "mesh_verts", "pair_data",
             "mesh_vertadr_buf", "mesh_vertnum_buf", "geom_dataid_buf", "geom_size_buf"},
            {"contact_data", "contact_count"},
            coll_source,
            COLLISION_HEADER_FWD
        );
#endif
        ctx->uses_metal_collision = true;
    }

    // Build solver kernel (constraint construction + Newton solver) whenever there's anything to collide.
    if (m.cache.collision_pairs.size() > 0) {
        int raw_si = (solver_iters_override > 0) ? solver_iters_override : std::max(m.opt.iterations, 1);
        // GPU CG solver always needs this cap (100 default is for CPU exact Cholesky)
        int si = std::min(raw_si, 3);
        int cgi = 20;
        ctx->solver_scratch_size = solver_scratch_per_env(m);
        ctx->solver_pair_props = build_solver_pair_props(m);
        ctx->solver_body_dof_masks = build_body_dof_masks(m);
        ctx->solver_body_rootid = build_body_rootid(m);
        mx::eval(ctx->solver_pair_props, ctx->solver_body_dof_masks, ctx->solver_body_rootid);

#if defined(MJMLX_BACKEND_MKX)
        bool use_pyramidal = (m.opt.cone == ConeType::PYRAMIDAL);
        bool refsafe = (m.opt.disableflags & DisableBit::REFSAFE) == 0;
        ctx->solver_kernel = mkx_kernels::make_solver_kernel(
            m.nbody, m.nv, m.opt.timestep, use_pyramidal, refsafe, m.opt.impratio, si, cgi);
#else
        auto solver_source = make_solver_source(m, si, cgi);
        ctx->solver_kernel = mx::fast::metal_kernel(
            "mjmlx_solver_" + std::to_string(m.nv) + "_s" + std::to_string(si) + "_c" + std::to_string(cgi),
            {"qM_in", "qfrc_smooth_in", "cdof_in", "subtree_com_in",
             "qvel_in", "contact_data_in", "contact_count_in",
             "pair_props", "body_dof_masks_buf", "body_rootid_buf"},
            {"qfrc_constraint_out", "solver_scratch"},
            solver_source,
            COLLISION_HEADER_FWD
        );
#endif
        ctx->uses_metal_solver = true;
    }

    return ctx;
}

// ── Hybrid batched step ──────────────────────────────────────────────────────

#if defined(MJMLX_BACKEND_MKX)
// mkx::vmap only supports one input/output; this generalizes its own slice/reshape/concatenate loop (mkx/ops/transforms.hpp) to forward_fn's 12 inputs/9 outputs, all batched on axis 0.
static std::vector<mx::array> mkx_vmap_batch0(
    const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>& fn,
    const std::vector<mx::array>& batched_inputs, int batch_size)
{
    std::vector<std::vector<mx::array>> per_env(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; b++) {
        std::vector<mx::array> sliced;
        sliced.reserve(batched_inputs.size());
        for (auto& in : batched_inputs) {
            mx::Shape shp = in.shape();
            mx::Shape starts(shp.size(), 0), stops = shp;
            starts[0] = b;
            stops[0] = b + 1;
            mx::array s = mx::slice(in, starts, stops);
            mx::Shape squeezed(shp.begin() + 1, shp.end());
            sliced.push_back(mx::reshape(s, squeezed));
        }
        per_env[static_cast<size_t>(b)] = fn(sliced);
    }
    size_t num_outputs = per_env[0].size();
    std::vector<mx::array> result;
    result.reserve(num_outputs);
    for (size_t o = 0; o < num_outputs; o++) {
        std::vector<mx::array> pieces;
        pieces.reserve(static_cast<size_t>(batch_size));
        for (int b = 0; b < batch_size; b++) {
            mx::Shape shp = per_env[static_cast<size_t>(b)][o].shape();
            mx::Shape unsq = shp;
            unsq.insert(unsq.begin(), 1);
            pieces.push_back(mx::reshape(per_env[static_cast<size_t>(b)][o], unsq));
        }
        result.push_back(mx::concatenate(pieces, 0));
    }
    return result;
}
#endif

std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu, int solver_iterations_override) {
    m.init_cache();
    auto ctx = build_context(m, solver_iterations_override);
    int B = num_envs;

    int nq = m.nq, nv = m.nv, nu = m.nu;
    int nb = m.nbody, nj = m.njnt, ng = m.ngeom;

    // DECISION: Solver iteration count comes from the model XML (e.g. humanoid.xml
    // specifies iterations="1" for Newton). The caller can override via
    // solver_iterations_override > 0. We do NOT artificially inflate the iteration count
    // -- the XML value is authoritative. Previously, a forced minimum of 3 iterations
    // was used as a workaround for mass matrix bugs. That has been fixed (see io.cpp
    // make_m_mask fix) and the override removed.
    int effective_iters = (solver_iterations_override > 0) ? solver_iterations_override : m.opt.iterations;

    bool has_kin = ctx->kin_kernel.has_value();
    bool has_euler = ctx->euler_kernel.has_value();
    bool has_euler_dm = ctx->euler_devmem_kernel.has_value();

    // If we need to override iterations, create a mutable copy on heap
    // that outlives this function (captured by lambdas).
    std::shared_ptr<Model> model_override;
    const Model* mp = &m;
    if (effective_iters != m.opt.iterations) {
        model_override = std::make_shared<Model>(m);
        model_override->opt.iterations = effective_iters;
        mp = model_override.get();
    }

    if (has_kin && (has_euler || has_euler_dm) && use_gpu) {

        // For large models (nv > 80), use all-Metal pipeline:
        //   Metal kin → Metal forward → Metal euler_devmem
        // For small models (nv ≤ 80), use hybrid pipeline:
        //   Metal kin → vmap(forward) → Metal euler
        bool use_metal_fwd = ctx->uses_metal_forward;

        // vmap forward (only built for nv ≤ 80)
        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> vmapped_fwd;
        if (!use_metal_fwd) {
            auto forward_fn = [mp, nq, nv, nu, nb, nj, ng](
                const std::vector<mx::array>& inputs) -> std::vector<mx::array>
            {
                const Model& m_ref = *mp;
                Data d;
                d.qpos = inputs[0]; d.qvel = inputs[1];
                d.ctrl = (nu > 0) ? inputs[2] : mx::zeros({1});
                d.xpos = mx::reshape(inputs[3], {nb, 3});
                d.xquat = mx::reshape(inputs[4], {nb, 4});
                d.xmat = mx::reshape(inputs[5], {nb, 3, 3});
                d.xipos = mx::reshape(inputs[6], {nb, 3});
                d.ximat = mx::reshape(inputs[7], {nb, 3, 3});
                if (nj > 0) { d.xanchor = mx::reshape(inputs[8], {nj, 3}); d.xaxis = mx::reshape(inputs[9], {nj, 3}); }
                if (ng > 0) { d.geom_xpos = mx::reshape(inputs[10], {ng, 3}); d.geom_xmat = mx::reshape(inputs[11], {ng, 3, 3}); }
                d.qfrc_applied = mx::zeros(mx::Shape{nv});
                d.xfrc_applied = mx::zeros(mx::Shape{nb, 6});
                d = vmap_forward(m_ref, d, false);
                return {
                    mx::flatten(d.qM), d.qfrc_smooth, d.qfrc_constraint,
                    mx::flatten(d.xpos), mx::flatten(d.subtree_com),
                    mx::flatten(d.cinert), mx::flatten(d.cvel),
                    d.qfrc_actuator, mx::flatten(d.qfrc_bias),
                };
            };
#if defined(MJMLX_BACKEND_MKX)
            vmapped_fwd = [forward_fn, B](const std::vector<mx::array>& state) {
                return mkx_vmap_batch0(forward_fn, state, B);
            };
#else
            std::vector<int> in_axes(12, 0);
            std::vector<int> out_axes = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            vmapped_fwd = mx::vmap(forward_fn, in_axes, out_axes);
#endif
        }

        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> pipeline =
            [ctx, vmapped_fwd, model_override, use_metal_fwd, B, nq, nv, nu, nb, nj, ng](
                const std::vector<mx::array>& state) -> std::vector<mx::array>
        {
            auto qpos_batch = state[0];
            auto qvel_batch = state[1];
            auto ctrl_batch = state[2];

            // ── Phase 1: Metal kinematics ──
            auto qpos_flat = mx::astype(mx::flatten(qpos_batch), mx::float32);
            std::vector<mx::Shape> kin_shapes = {
                {B * nb * 3}, {B * nb * 4}, {B * nb * 9},
                {B * nb * 3}, {B * nb * 9},
                (nj > 0) ? mx::Shape{B * nj * 3} : mx::Shape{1},
                (nj > 0) ? mx::Shape{B * nj * 3} : mx::Shape{1},
                (ng > 0) ? mx::Shape{B * ng * 3} : mx::Shape{1},
                (ng > 0) ? mx::Shape{B * ng * 9} : mx::Shape{1},
            };
#if defined(MJMLX_BACKEND_MKX)
            auto kin_raw = (*ctx->kin_kernel)(
                mx::fast::kernel_inputs({ctx->body_parentid, ctx->body_pos, ctx->body_quat,
                 ctx->body_ipos, ctx->body_iquat,
                 ctx->body_jntadr, ctx->body_jntnum,
                 ctx->jnt_type_arr, ctx->jnt_qposadr_arr,
                 ctx->jnt_pos_arr, ctx->jnt_axis_arr,
                 ctx->qpos0,
                 ctx->geom_bodyid_arr, ctx->geom_pos_arr, ctx->geom_quat_arr,
                 qpos_flat}),
                mx::fast::kernel_shapes(kin_shapes),
                std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{1, 1, 1}
            );
            auto kin = mx::fast::kernel_outputs(kin_raw);
#else
            auto kin = (*ctx->kin_kernel)(
                {ctx->body_parentid, ctx->body_pos, ctx->body_quat,
                 ctx->body_ipos, ctx->body_iquat,
                 ctx->body_jntadr, ctx->body_jntnum,
                 ctx->jnt_type_arr, ctx->jnt_qposadr_arr,
                 ctx->jnt_pos_arr, ctx->jnt_axis_arr,
                 ctx->qpos0,
                 ctx->geom_bodyid_arr, ctx->geom_pos_arr, ctx->geom_quat_arr,
                 qpos_flat},
                kin_shapes, std::vector<mx::Dtype>(9, mx::float32),
                std::make_tuple(B, 1, 1), std::make_tuple(1, 1, 1),
                {}, std::nullopt, false, {}
            );
#endif

            auto xpos   = mx::reshape(kin[0], {B, nb, 3});
            auto xipos  = mx::reshape(kin[3], {B, nb, 3});
            auto ximat  = mx::reshape(kin[4], {B, nb, 3, 3});
            auto xanchor = (nj > 0) ? mx::reshape(kin[5], {B, nj, 3}) : mx::zeros({B, 1});
            auto xaxis   = (nj > 0) ? mx::reshape(kin[6], {B, nj, 3}) : mx::zeros({B, 1});

            mx::array qM_flat = mx::zeros({1});
            mx::array qfrc_smooth_flat = mx::zeros({1});
            mx::array qfrc_constraint_flat = mx::zeros({1});
            mx::array subtree_com_out = mx::zeros({1});
            mx::array cinert_out = mx::zeros({1});
            mx::array cvel_out = mx::zeros({1});
            mx::array qfrc_actuator_out = mx::zeros({1});

            if (use_metal_fwd) {
                // ── Phase 2a: all-Metal/GLSL forward kernel (whenever there's anything to collide) ──
                auto xmat = mx::reshape(kin[2], {B, nb, 3, 3});
                int scratchSz = ctx->fwd_scratch_per_env;

                std::vector<mx::array> fwd;
#if defined(MJMLX_BACKEND_MKX)
                {
                    auto fwd_raw = (*ctx->forward_kernel)(
                        mx::fast::kernel_inputs({mx::flatten(xipos), mx::flatten(ximat),
                         mx::flatten(xanchor), mx::flatten(xaxis), mx::flatten(xmat),
                         mx::flatten(qpos_batch), mx::flatten(qvel_batch), mx::flatten(ctrl_batch),
                         ctx->act_moment}),
                        mx::fast::kernel_shapes({{B * nv * nv}, {B * nv}, {B * nb * 3},
                         {B * nb * 10}, {B * nb * 6}, {B * nv},
                         {B * scratchSz}, {B * nv * 6}}),
                        std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{1, 1, 1}
                    );
                    fwd = mx::fast::kernel_outputs(fwd_raw);
                }
#else
                fwd = (*ctx->forward_kernel)(
                    {mx::flatten(xipos), mx::flatten(ximat),
                     mx::flatten(xanchor), mx::flatten(xaxis), mx::flatten(xmat),
                     mx::flatten(qpos_batch), mx::flatten(qvel_batch), mx::flatten(ctrl_batch),
                     ctx->make_m_mask, ctx->act_moment},
                    {{B * nv * nv}, {B * nv}, {B * nb * 3},
                     {B * nb * 10}, {B * nb * 6}, {B * nv},
                     {B * scratchSz}, {B * nv * 6}},
                    {mx::float32, mx::float32, mx::float32,
                     mx::float32, mx::float32, mx::float32, mx::float32, mx::float32},
                    std::make_tuple(B, 1, 1), std::make_tuple(1, 1, 1),
                    {}, std::nullopt, false, {}
                );
#endif

                qM_flat = fwd[0];
                qfrc_smooth_flat = fwd[1];
                subtree_com_out = mx::reshape(fwd[2], {B, nb, 3});
                cinert_out = mx::reshape(fwd[3], {B, nb, 10});
                cvel_out = mx::reshape(fwd[4], {B, nb, 6});
                qfrc_actuator_out = mx::reshape(fwd[5], {B, nv});
                auto cdof_flat = fwd[7]; // (B * nv * 6)

                // ── Collision + Solver ──
                if (ctx->uses_metal_collision) {
                    // Collision kernel: detect contacts from geom transforms
                    auto geom_xpos_flat = mx::flatten(mx::reshape(kin[7], {B, ng, 3}));
                    auto geom_xmat_flat = mx::flatten(mx::reshape(kin[8], {B, ng, 3, 3}));

                    int con_buf_sz = B * MAX_CONTACTS_PER_ENV * CONTACT_STRIDE;
                    std::vector<mx::array> coll;
#if defined(MJMLX_BACKEND_MKX)
                    {
                        auto coll_raw = (*ctx->collision_kernel)(
                            mx::fast::kernel_inputs({geom_xpos_flat, geom_xmat_flat,
                             ctx->coll_mesh_verts, ctx->coll_pair_data,
                             ctx->coll_mesh_vertadr, ctx->coll_mesh_vertnum,
                             ctx->coll_geom_dataid, ctx->coll_geom_size}),
                            mx::fast::kernel_shapes({{con_buf_sz}, {B}}),
                            std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{1, 1, 1}
                        );
                        coll = mx::fast::kernel_outputs(coll_raw);
                    }
#else
                    coll = (*ctx->collision_kernel)(
                        {geom_xpos_flat, geom_xmat_flat,
                         ctx->coll_mesh_verts, ctx->coll_pair_data,
                         ctx->coll_mesh_vertadr, ctx->coll_mesh_vertnum,
                         ctx->coll_geom_dataid, ctx->coll_geom_size},
                        {{con_buf_sz}, {B}},
                        {mx::float32, mx::float32},
                        std::make_tuple(B, 1, 1), std::make_tuple(1, 1, 1),
                        {}, std::nullopt, false, {}
                    );
#endif

                    if (ctx->uses_metal_solver) {
                        int scratch_sz = ctx->solver_scratch_size;
                        std::vector<mx::array> solver;
#if defined(MJMLX_BACKEND_MKX)
                        {
                            auto solver_raw = (*ctx->solver_kernel)(
                                mx::fast::kernel_inputs({qM_flat, qfrc_smooth_flat, cdof_flat,
                                 mx::flatten(subtree_com_out),
                                 mx::astype(mx::flatten(qvel_batch), mx::float32),
                                 coll[0], coll[1],
                                 ctx->solver_pair_props, ctx->solver_body_dof_masks,
                                 ctx->solver_body_rootid}),
                                mx::fast::kernel_shapes({{B * nv}, {B * scratch_sz}}),
                                std::array<uint32_t, 3>{static_cast<uint32_t>(B * nv), 1, 1}, std::array<uint32_t, 3>{static_cast<uint32_t>(nv), 1, 1}
                            );
                            solver = mx::fast::kernel_outputs(solver_raw);
                        }
#else
                        solver = (*ctx->solver_kernel)(
                            {qM_flat, qfrc_smooth_flat, cdof_flat,
                             mx::flatten(subtree_com_out),
                             mx::astype(mx::flatten(qvel_batch), mx::float32),
                             coll[0], coll[1],
                             ctx->solver_pair_props, ctx->solver_body_dof_masks,
                             ctx->solver_body_rootid},
                            {{B * nv}, {B * scratch_sz}},
                            {mx::float32, mx::float32},
                            std::make_tuple(B * nv, 1, 1), std::make_tuple(nv, 1, 1),
                            {}, std::nullopt, false, {}
                        );
#endif
                        qfrc_constraint_flat = solver[0];
                    } else {
                        qfrc_constraint_flat = mx::zeros({B * nv});
                    }
                } else {
                    qfrc_constraint_flat = mx::zeros({B * nv});
                }
            } else {
                // ── Phase 2b: vmap(forward) (hybrid path for nv ≤ 80) ──
                auto xquat = mx::reshape(kin[1], {B, nb, 4});
                auto xmat  = mx::reshape(kin[2], {B, nb, 3, 3});
                auto gxpos = (ng > 0) ? mx::reshape(kin[7], {B, ng, 3}) : mx::zeros({B, 1});
                auto gxmat = (ng > 0) ? mx::reshape(kin[8], {B, ng, 3, 3}) : mx::zeros({B, 1});

                auto mid = vmapped_fwd({
                    qpos_batch, qvel_batch, ctrl_batch,
                    xpos, xquat, xmat, xipos, ximat,
                    xanchor, xaxis, gxpos, gxmat
                });
                qM_flat = mx::flatten(mid[0]);
                qfrc_smooth_flat = mx::flatten(mid[1]);
                qfrc_constraint_flat = mx::flatten(mid[2]);
                subtree_com_out = mx::reshape(mid[4], {B, nb, 3});
                cinert_out = mx::reshape(mid[5], {B, nb, 10});
                cvel_out = mx::reshape(mid[6], {B, nb, 6});
                qfrc_actuator_out = mid[7];
            }

            // ── Phase 3: Metal Euler ──
            std::vector<mx::array> euler_inputs = {
                mx::astype(qM_flat, mx::float32),
                mx::astype(qfrc_smooth_flat, mx::float32),
                mx::astype(qfrc_constraint_flat, mx::float32),
                mx::astype(mx::flatten(qvel_batch), mx::float32),
                mx::astype(mx::flatten(qpos_batch), mx::float32)
            };
            std::vector<mx::array> euler;
#if defined(MJMLX_BACKEND_MKX)
            auto grid = std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1};
            auto tgroup = std::array<uint32_t, 3>{1, 1, 1};
            if (ctx->euler_kernel.has_value()) {
                auto raw = (*ctx->euler_kernel)(mx::fast::kernel_inputs(euler_inputs), mx::fast::kernel_shapes({{B * nq}, {B * nv}, {B * nv}}), grid, tgroup);
                euler = mx::fast::kernel_outputs(raw);
            } else {
                auto raw = (*ctx->euler_devmem_kernel)(mx::fast::kernel_inputs(euler_inputs), mx::fast::kernel_shapes({{B * nq}, {B * nv}, {B * nv}, {B * nv * nv}}), grid, tgroup);
                euler = mx::fast::kernel_outputs(raw);
            }
#else
            auto grid = std::make_tuple(B, 1, 1);
            auto tgroup = std::make_tuple(1, 1, 1);
            if (ctx->euler_kernel.has_value()) {
                euler = (*ctx->euler_kernel)(
                    euler_inputs,
                    {{B * nq}, {B * nv}, {B * nv}},
                    {mx::float32, mx::float32, mx::float32},
                    grid, tgroup, {}, std::nullopt, false, {}
                );
            } else {
                euler = (*ctx->euler_devmem_kernel)(
                    euler_inputs,
                    {{B * nq}, {B * nv}, {B * nv}, {B * nv * nv}},
                    {mx::float32, mx::float32, mx::float32, mx::float32},
                    grid, tgroup, {}, std::nullopt, false, {}
                );
            }
#endif

            auto new_qpos = mx::reshape(euler[0], {B, nq});
            auto new_qvel = mx::reshape(euler[1], {B, nv});
            auto xpos_out = mx::reshape(kin[0], {B, nb, 3});
            auto cfrc_ext_out = mx::zeros({B, nb, 6});

            return {new_qpos, new_qvel, xpos_out,
                    subtree_com_out, cinert_out, cvel_out,
                    qfrc_actuator_out, cfrc_ext_out};
        };

        // mx::compile fuses ops (reshape, astype, etc.) in the computation graph.
        // For nv≤80 (vmap path), this provides significant speedup by fusing many
        // small ops. For nv>80 (all-Metal path), custom kernels already do bulk work
        // per dispatch and mx::compile can deadlock with the large solver kernel.
        if (!use_metal_fwd) {
            auto compiled = mx::compile(pipeline);
            return compiled;
        }
        return pipeline;
    }

    // Metal kernels could not be built for this model.
    throw std::runtime_error(
        "GPU Metal kernels not available for this model (nv=" + std::to_string(nv) +
        ", nbody=" + std::to_string(nb) + "). "
        "Device-memory Euler requires nv <= " + std::to_string(EULER_DEVMEM_MAX_NV) +
        ". Use CPU batched mode instead.");
}

} // namespace mjmlx

// ── CPU batched helpers (used by C API and Python bindings) ──────────────────

// CPU batched: gather state from N mjData* into BatchedSim mx::array fields.
MJMLX_API void cpu_gather_state(MjmlxBatchedSim* handle) {
    auto& s = handle->sim;
    int B = s.num_envs;
    const mjModel* m = handle->cpu_model;
    int nq = m->nq, nv = m->nv, nb = m->nbody;

    std::vector<float> qp(B * nq), qv(B * nv);
    std::vector<float> xp(B * nb * 3), sc(B * nb * 3);
    std::vector<float> ci(B * nb * 10), cv(B * nb * 6);
    std::vector<float> qa(B * nv), ce(B * nb * 6);

    for (int i = 0; i < B; i++) {
        const mjData* d = handle->cpu_datas[i];
        for (int j = 0; j < nq; j++) qp[i*nq + j] = (float)d->qpos[j];
        for (int j = 0; j < nv; j++) qv[i*nv + j] = (float)d->qvel[j];
        for (int j = 0; j < nb*3; j++) xp[i*nb*3 + j] = (float)d->xpos[j];
        for (int j = 0; j < nb*3; j++) sc[i*nb*3 + j] = (float)d->subtree_com[j];
        for (int j = 0; j < nb*10; j++) ci[i*nb*10 + j] = (float)d->cinert[j];
        for (int j = 0; j < nb*6; j++) cv[i*nb*6 + j] = (float)d->cvel[j];
        for (int j = 0; j < nv; j++) qa[i*nv + j] = (float)d->qfrc_actuator[j];
        for (int j = 0; j < nb*6; j++) ce[i*nb*6 + j] = (float)d->cfrc_ext[j];
    }

    s.qpos = mx::reshape(mx::array(qp.data(), {B*nq}, mx::float32), {B, nq});
    s.qvel = mx::reshape(mx::array(qv.data(), {B*nv}, mx::float32), {B, nv});
    s.xpos = mx::reshape(mx::array(xp.data(), {B*nb*3}, mx::float32), {B, nb, 3});
    s.subtree_com = mx::reshape(mx::array(sc.data(), {B*nb*3}, mx::float32), {B, nb, 3});
    s.cinert = mx::reshape(mx::array(ci.data(), {B*nb*10}, mx::float32), {B, nb, 10});
    s.cvel = mx::reshape(mx::array(cv.data(), {B*nb*6}, mx::float32), {B, nb, 6});
    s.qfrc_actuator = mx::reshape(mx::array(qa.data(), {B*nv}, mx::float32), {B, nv});
    s.cfrc_ext = mx::reshape(mx::array(ce.data(), {B*nb*6}, mx::float32), {B, nb, 6});
}

// CPU batched: sync qpos/qvel mx::arrays back to mjData instances.
MJMLX_API void cpu_sync_state(MjmlxBatchedSim* handle) {
    auto& s = handle->sim;
    int B = s.num_envs;
    const mjModel* m = handle->cpu_model;
    int nq = m->nq, nv = m->nv;

    mx::eval(s.qpos, s.qvel);
    const float* qp = s.qpos.data<float>();
    const float* qv = s.qvel.data<float>();

    for (int i = 0; i < B; i++) {
        mjData* d = handle->cpu_datas[i];
        for (int j = 0; j < nq; j++) d->qpos[j] = (double)qp[i*nq + j];
        for (int j = 0; j < nv; j++) d->qvel[j] = (double)qv[i*nv + j];
    }
}

// CPU batched: step all environments in parallel (GCD dispatch_apply on Apple, std::thread elsewhere).
MJMLX_API void cpu_batched_step(MjmlxBatchedSim* handle, const float* ctrl_flat, int frame_skip) {
    int B = handle->sim.num_envs;
    mjModel* m = handle->cpu_model;
    int nu = m->nu;

    auto step_one = [&](size_t i) {
        mjData* d = handle->cpu_datas[i];
        if (ctrl_flat && nu > 0) {
            for (int j = 0; j < nu; j++)
                d->ctrl[j] = (double)ctrl_flat[i * nu + j];
        }
        for (int fs = 0; fs < frame_skip; fs++)
            mj_step(m, d);
    };

#if defined(__APPLE__)
    dispatch_apply((size_t)B,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t i) { step_one(i); }
    );
#else
    unsigned n_threads = std::min<unsigned>(std::thread::hardware_concurrency(), (unsigned)B);
    if (n_threads <= 1) {
        for (size_t i = 0; i < (size_t)B; i++) step_one(i);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(n_threads);
        for (unsigned t = 0; t < n_threads; t++) {
            pool.emplace_back([&, t]() {
                for (size_t i = t; i < (size_t)B; i += n_threads) step_one(i);
            });
        }
        for (auto& th : pool) th.join();
    }
#endif
}

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

        if (config->use_gpu == 0 && model->mj_model) {
            // CPU path: N independent mjData* with dispatch_apply stepping
            handle->cpu_mode = true;
            handle->cpu_model = model->mj_model;
            handle->cpu_datas.resize(B);
            for (int i = 0; i < B; i++) {
                handle->cpu_datas[i] = mj_makeData(model->mj_model);
                if (!handle->cpu_datas[i]) {
                    delete handle;
                    return nullptr;
                }
            }
            if (config->solver_iterations > 0)
                handle->cpu_model->opt.iterations = config->solver_iterations;

            // Initialize mx::array state from default qpos0/zero qvel
            std::vector<mx::array> qpos_list, qvel_list;
            for (int i = 0; i < B; i++) {
                qpos_list.push_back(model->model.qpos0);
                qvel_list.push_back(mx::zeros({nv}));
            }
            handle->sim.qpos = mx::stack(qpos_list);
            handle->sim.qvel = mx::stack(qvel_list);
        } else {
            // GPU path: compiled + vmapped MLX step function
            std::vector<mx::array> qpos_list, qvel_list;
            for (int i = 0; i < B; i++) {
                qpos_list.push_back(model->model.qpos0);
                qvel_list.push_back(mx::zeros({nv}));
            }
            handle->sim.qpos = mx::stack(qpos_list);
            handle->sim.qvel = mx::stack(qvel_list);

            handle->sim.compiled_step = mjmlx::make_batched_step(
                model->model, B, config->use_gpu, config->solver_iterations);
        }

        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "mjmlx_batched_create error: %s\n", e.what());
        return nullptr;
    }
}

MJMLX_API void mjmlx_batched_step(MjmlxBatchedSim* sim, const float* ctrl_flat) {
    if (!sim) return;

    if (sim->cpu_mode) {
        cpu_sync_state(sim);
        cpu_batched_step(sim, ctrl_flat, 1);
        cpu_gather_state(sim);
        return;
    }

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
    if (results.size() > 3) s.subtree_com = results[3];
    if (results.size() > 4) s.cinert = results[4];
    if (results.size() > 5) s.cvel = results[5];
    if (results.size() > 6) s.qfrc_actuator = results[6];
    if (results.size() > 7) s.cfrc_ext = results[7];
}

MJMLX_API void mjmlx_batched_get_state(
    const MjmlxBatchedSim* sim, float* qpos_out, float* qvel_out,
    int* nq_out, int* nv_out)
{
    if (!sim) return;
    auto& s = sim->sim;
    mx::eval(s.qpos, s.qvel);

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

MJMLX_API const float* mjmlx_batched_get_subtree_com(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.subtree_com);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nbody * 3;
    return (sim->sim.subtree_com.size() > 0) ? sim->sim.subtree_com.data<float>() : nullptr;
}

MJMLX_API const float* mjmlx_batched_get_cinert(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.cinert);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nbody * 10;
    return (sim->sim.cinert.size() > 0) ? sim->sim.cinert.data<float>() : nullptr;
}

MJMLX_API const float* mjmlx_batched_get_cvel(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.cvel);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nbody * 6;
    return (sim->sim.cvel.size() > 0) ? sim->sim.cvel.data<float>() : nullptr;
}

MJMLX_API const float* mjmlx_batched_get_qfrc_actuator(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.qfrc_actuator);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nv;
    return (sim->sim.qfrc_actuator.size() > 0) ? sim->sim.qfrc_actuator.data<float>() : nullptr;
}

MJMLX_API const float* mjmlx_batched_get_cfrc_ext(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.cfrc_ext);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nbody * 6;
    return (sim->sim.cfrc_ext.size() > 0) ? sim->sim.cfrc_ext.data<float>() : nullptr;
}

MJMLX_API void mjmlx_batched_eval_state(const MjmlxBatchedSim* sim) {
    if (!sim || sim->cpu_mode) return;
    mx::eval({sim->sim.qpos, sim->sim.qvel});
}

MJMLX_API void mjmlx_batched_reset(MjmlxBatchedSim* sim, const int* reset_mask) {
    if (!sim || !reset_mask) return;

    if (sim->cpu_mode) {
        mjModel* m = sim->cpu_model;
        for (int i = 0; i < sim->sim.num_envs; i++) {
            if (reset_mask[i])
                mj_resetData(m, sim->cpu_datas[i]);
        }
        cpu_gather_state(sim);
        return;
    }

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

MJMLX_API void mjmlx_batched_set_env_qpos(MjmlxBatchedSim* sim, int env_idx,
                                           const float* qpos, int nq) {
    if (!sim || !qpos || env_idx < 0 || env_idx >= sim->sim.num_envs) return;

    if (sim->cpu_mode) {
        int model_nq = sim->cpu_model->nq;
        if (nq != model_nq) return;
        mjData* d = sim->cpu_datas[env_idx];
        for (int j = 0; j < model_nq; j++) d->qpos[j] = (double)qpos[j];
        return;
    }

    auto& s = sim->sim;
    int B = s.num_envs;
    int model_nq = s.model->nq;
    if (nq != model_nq) return;

    mx::eval(s.qpos);
    std::vector<float> qp(s.qpos.data<float>(), s.qpos.data<float>() + B * model_nq);
    std::memcpy(&qp[env_idx * model_nq], qpos, model_nq * sizeof(float));
    s.qpos = mx::reshape(mx::array(qp.data(), {B * model_nq}, mx::float32), {B, model_nq});
}

MJMLX_API void mjmlx_batched_set_env_qvel(MjmlxBatchedSim* sim, int env_idx,
                                           const float* qvel, int nv) {
    if (!sim || !qvel || env_idx < 0 || env_idx >= sim->sim.num_envs) return;

    if (sim->cpu_mode) {
        int model_nv = sim->cpu_model->nv;
        if (nv != model_nv) return;
        mjData* d = sim->cpu_datas[env_idx];
        for (int j = 0; j < model_nv; j++) d->qvel[j] = (double)qvel[j];
        return;
    }

    auto& s = sim->sim;
    int B = s.num_envs;
    int model_nv = s.model->nv;
    if (nv != model_nv) return;

    mx::eval(s.qvel);
    std::vector<float> qv(s.qvel.data<float>(), s.qvel.data<float>() + B * model_nv);
    std::memcpy(&qv[env_idx * model_nv], qvel, model_nv * sizeof(float));
    s.qvel = mx::reshape(mx::array(qv.data(), {B * model_nv}, mx::float32), {B, model_nv});
}

MJMLX_API void mjmlx_batched_free(MjmlxBatchedSim* sim) {
    delete sim;
}

} // extern "C"
