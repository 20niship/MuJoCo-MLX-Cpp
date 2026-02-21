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
#include <cmath>
#include <cstring>
#include <memory>
#include <dispatch/dispatch.h>
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

    return ss.str();
}

static int forward_scratch_per_env(const Model& m) {
    int nb = m.nbody, nv = m.nv;
    return nb*10 + nv*6 + nv*6 + nb*6 + nb*6 + nb*3 + nb + nv + nv;
}

// ── MSL header for forward kernel (spatial algebra helpers) ──────────────────

static const std::string FORWARD_HEADER = R"(
)";

// ── Test helper: dispatch Metal forward kernel for 1 env ────────────────────

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
         "cinert_out", "cvel_out", "qfrc_actuator_out", "scratch"},
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
         {B * scratchSz}},
        {mx::float32, mx::float32, mx::float32,
         mx::float32, mx::float32, mx::float32, mx::float32},
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

// ── Context: Metal kernels + model constants ─────────────────────────────────

using KernelFn = mx::fast::CustomKernelFunction;

struct BatchedStepContext {
    std::optional<KernelFn> kin_kernel;
    std::optional<KernelFn> euler_kernel;
    std::optional<KernelFn> euler_devmem_kernel;
    std::optional<KernelFn> forward_kernel;
    bool uses_devmem_euler = false;
    bool uses_metal_forward = false;

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

    // Forward kernel model constants
    mx::array make_m_mask{mx::zeros({1})};
    mx::array act_moment{mx::zeros({1})};
    int fwd_scratch_per_env = 0;

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
    // DECISION: Metal kernel stack memory is limited (~24KB per thread). The kinematics
    // kernel needs ~7 floats per body (pos, quat) as working storage. If nbody is too
    // large, we skip the Metal kernel and fall back to the scalar CPU path.
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
        auto euler_source = make_euler_source(
            m.nv, m.nq, m.opt.timestep,
            s_qa, s_da, fj, bj, damp_vals);

        ctx->euler_kernel = mx::fast::metal_kernel(
            "mjmlx_euler_" + std::to_string(m.nv) + "_" + std::to_string(m.nq),
            {"qM", "qfrc_smooth", "qfrc_constraint", "qvel_in", "qpos_in"},
            {"qpos_out", "qvel_out", "qacc_out"},
            euler_source
        );
    } else if (m.nv <= EULER_DEVMEM_MAX_NV) {
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
    }

    // Build Metal forward kernel for large models (replaces vmap forward)
    if (m.nv > 80) {
        auto fwd_source = make_forward_source(m);
        ctx->fwd_scratch_per_env = forward_scratch_per_env(m);

        // Prepare model constant buffers
        ctx->make_m_mask = mx::astype(mx::flatten(m.cache.make_m_mask), mx::float32);
        mx::eval(ctx->make_m_mask);

        if (m.nu > 0 && m.cache.act_moment_const.size() > 0) {
            ctx->act_moment = mx::astype(mx::flatten(m.cache.act_moment_const), mx::float32);
        } else {
            ctx->act_moment = mx::zeros({std::max(m.nv, 1)});
        }
        mx::eval(ctx->act_moment);

        ctx->forward_kernel = mx::fast::metal_kernel(
            "mjmlx_forward_" + std::to_string(m.nbody) + "_" + std::to_string(m.nv),
            {"xipos", "ximat", "xanchor", "xaxis", "xmat",
             "qpos", "qvel", "ctrl",
             "make_m_mask", "act_moment"},
            {"qM_out", "qfrc_smooth_out", "subtree_com_out",
             "cinert_out", "cvel_out", "qfrc_actuator_out", "scratch"},
            fwd_source,
            FORWARD_HEADER
        );
        ctx->uses_metal_forward = true;
    }

    return ctx;
}

// ── Hybrid batched step ──────────────────────────────────────────────────────

std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu, int solver_iterations_override) {
    m.init_cache();
    auto ctx = build_context(m);
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
            std::vector<int> in_axes(12, 0);
            std::vector<int> out_axes = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            vmapped_fwd = mx::vmap(forward_fn, in_axes, out_axes);
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
                // ── Phase 2a: Metal forward kernel (all-Metal path for nv > 80) ──
                auto xmat = mx::reshape(kin[2], {B, nb, 3, 3});
                int scratchSz = ctx->fwd_scratch_per_env;

                auto fwd = (*ctx->forward_kernel)(
                    {mx::flatten(xipos), mx::flatten(ximat),
                     mx::flatten(xanchor), mx::flatten(xaxis), mx::flatten(xmat),
                     mx::flatten(qpos_batch), mx::flatten(qvel_batch), mx::flatten(ctrl_batch),
                     ctx->make_m_mask, ctx->act_moment},
                    {{B * nv * nv}, {B * nv}, {B * nb * 3},
                     {B * nb * 10}, {B * nb * 6}, {B * nv},
                     {B * scratchSz}},
                    {mx::float32, mx::float32, mx::float32,
                     mx::float32, mx::float32, mx::float32, mx::float32},
                    std::make_tuple(B, 1, 1), std::make_tuple(1, 1, 1),
                    {}, std::nullopt, false, {}
                );

                qM_flat = fwd[0];
                qfrc_smooth_flat = fwd[1];
                qfrc_constraint_flat = mx::zeros({B * nv});
                subtree_com_out = mx::reshape(fwd[2], {B, nb, 3});
                cinert_out = mx::reshape(fwd[3], {B, nb, 10});
                cvel_out = mx::reshape(fwd[4], {B, nb, 6});
                qfrc_actuator_out = mx::reshape(fwd[5], {B, nv});
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
            auto grid = std::make_tuple(B, 1, 1);
            auto tgroup = std::make_tuple(1, 1, 1);

            std::vector<mx::array> euler;
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

            auto new_qpos = mx::reshape(euler[0], {B, nq});
            auto new_qvel = mx::reshape(euler[1], {B, nv});
            auto xpos_out = mx::reshape(kin[0], {B, nb, 3});
            auto cfrc_ext_out = mx::zeros({B, nb, 6});

            return {new_qpos, new_qvel, xpos_out,
                    subtree_com_out, cinert_out, cvel_out,
                    qfrc_actuator_out, cfrc_ext_out};
        };

        auto compiled = mx::compile(pipeline);
        return compiled;
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

// CPU batched: step all environments in parallel using GCD dispatch_apply.
MJMLX_API void cpu_batched_step(MjmlxBatchedSim* handle, const float* ctrl_flat, int frame_skip) {
    int B = handle->sim.num_envs;
    mjModel* m = handle->cpu_model;
    int nu = m->nu;

    dispatch_apply((size_t)B,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t i) {
            mjData* d = handle->cpu_datas[i];
            if (ctrl_flat && nu > 0) {
                for (int j = 0; j < nu; j++)
                    d->ctrl[j] = (double)ctrl_flat[i * nu + j];
            }
            for (int fs = 0; fs < frame_skip; fs++)
                mj_step(m, d);
        }
    );
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
