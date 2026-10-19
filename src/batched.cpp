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

// Batched simulation: GPU kernels + compile(vmap(step)) for N parallel envs.
//
// ARCHITECTURE NOTE: 3-phase hybrid pipeline (same design as Python mujoco-mlx):
//   Phase 1: GPU kinematics kernel (single GPU dispatch for all N envs)
//   Phase 2: compile(vmap(forward_dynamics_without_euler))
//   Phase 3: GPU Euler kernel (single GPU dispatch for all N envs)
//
// DECISION: GPU kernels are NOT vmap-compatible (they operate on raw flat buffers
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
// DECISION: Using mx::compile around the full pipeline (GPU kin → vmap(fwd) →
// GPU euler) enables MLX to fuse the computation graph. This is critical for
// performance: without compile, each MLX op dispatches a separate GPU kernel.
// With compile, MLX fuses compatible ops into fewer, larger kernels.

#include "internal.h"
#include "mjmlx/mjmlx.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <memory>

// SIMD幅(32)を使い切るため、Bが64で割り切れるときだけ1ワークグループに64環境を載せる(割り切れなければ従来の1)。
static uint32_t env_threadgroup(int B) { return (B % 64 == 0) ? 64u : 1u; }
#include "compat/mkx_kernels/kinematics.hpp"
#include "compat/mkx_kernels/euler.hpp"
#include "compat/mkx_kernels/euler_devmem.hpp"
#include "compat/mkx_kernels/forward.hpp"
#include "compat/mkx_kernels/collision.hpp"
#include "compat/mkx_kernels/solver.hpp"
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#else
#include <thread>
#include <vector>
#endif
#include <mujoco/mujoco.h>

namespace mjmlx {

static const int EULER_DEVMEM_MAX_NV = 2048;

// ── GPU forward dynamics kernel (smooth dynamics for nv > 80) ──────────────
// Replaces the vmap(forward_dynamics) path for large models. Single GPU dispatch
// per batch. Computes: com_pos → cinert → cdof → crb → qM → com_vel → rne →
// passive → actuation → qfrc_smooth.
//
// Scratch layout per env (in floats):
//   crb[nb*10], cdof[nv*6], cdof_dot[nv*6], cacc[nb*6], cfrc[nb*6],
//   sub_pos[nb*3], sub_mass[nb], qfrc_bias[nv], qfrc_passive[nv]

// Host-side model constants for mkx_kernels::make_forward_kernel.
struct ForwardHostConsts {
    std::array<float, 3> gravity{0, 0, 0};
    std::vector<int> body_parentid, body_rootid, dof_bodyid, dof_parentid;
    std::vector<float> body_gravcomp, body_mass, body_inertia, dof_damping, dof_armature, dof_stiffness;
    std::vector<int> dof_qposadr;
    std::vector<float> qpos_spring, act_gain0, act_bias0;
    // AFFINE gain/bias(length/velocity項)算出用。JOINT/JOINTINPARENT伝達のみ対応、他はgear=0でlength=0扱い。
    std::vector<float> act_gain1, act_gain2, act_bias1, act_bias2, act_gear;
    std::vector<int> act_qposadr, act_dofadr;
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
    k.body_gravcomp.assign(nb, 0.0f);
    if (m.ngravcomp > 0 && m.body_gravcomp.size() > 0) {
        mx::eval(m.body_gravcomp);
        auto gc_ptr = m.body_gravcomp.data<float>();
        k.body_gravcomp.assign(gc_ptr, gc_ptr + nb);
    }
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
    k.act_gain1.assign(std::max(nu, 1), 0.0f);
    k.act_gain2.assign(std::max(nu, 1), 0.0f);
    k.act_bias1.assign(std::max(nu, 1), 0.0f);
    k.act_bias2.assign(std::max(nu, 1), 0.0f);
    if (nu > 0 && m.actuator_gainprm.size() > 0) {
        mx::eval(m.actuator_gainprm, m.actuator_biasprm);
        auto gp = m.actuator_gainprm.data<float>();
        auto bp = m.actuator_biasprm.data<float>();
        int ncol = (int)m.actuator_gainprm.shape(1);
        for (int i = 0; i < nu; i++) {
            k.act_gain0[i] = gp[i * ncol];
            k.act_bias0[i] = bp[i * ncol];
            if (ncol > 1) { k.act_gain1[i] = gp[i * ncol + 1]; k.act_bias1[i] = bp[i * ncol + 1]; }
            if (ncol > 2) { k.act_gain2[i] = gp[i * ncol + 2]; k.act_bias2[i] = bp[i * ncol + 2]; }
        }
    }

    k.act_gear.assign(std::max(nu, 1), 0.0f);
    k.act_qposadr.assign(std::max(nu, 1), 0);
    k.act_dofadr.assign(std::max(nu, 1), 0);
    if (nu > 0 && m.actuator_trntype.size() > 0) {
        mx::eval(m.actuator_trntype, m.actuator_trnid, m.actuator_gear);
        auto trnt_ptr = m.actuator_trntype.data<int>();
        auto trnid_ptr = m.actuator_trnid.data<int>();
        auto gear_ptr = m.actuator_gear.data<float>();
        int gear_ncol = (int)m.actuator_gear.shape(1);
        for (int ai = 0; ai < nu; ai++) {
            int trnt = trnt_ptr[ai];
            if (trnt == 0 || trnt == 1) {  // JOINT (0) or JOINTINPARENT (1)
                int ji = trnid_ptr[ai * 2];
                if (ji >= 0 && ji < njnt) {
                    k.act_gear[ai] = gear_ptr[ai * gear_ncol];
                    k.act_qposadr[ai] = jqa_ptr[ji];
                    k.act_dofadr[ai] = jda_ptr[ji];
                }
            }
        }
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
static int forward_scratch_per_env(const Model& m) {
    int nb = m.nbody, nv = m.nv;
    return nb*10 + nv*6 + nv*6 + nb*6 + nb*6 + nb*3 + nb + nv + nv;
}

// ── Backend-agnostic host data builders for GLSL (mkx_kernels) collision/solver dispatch ──
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

// solverカーネルのscratchレイアウト(GLSL版はefc長の配列7本)以上でなければならない。不足すると末尾のenvがバッファ外を読み書きしゼロ/不定値/GPUリカバリを起こす。不足すると末尾のenvがバッファ外を読み書きしゼロ/不定値/GPUリカバリを起こす。
static int solver_scratch_per_env(const Model& m) {
    int nv = m.nv;
    return nv*nv + MAX_EFC*nv + 7*MAX_EFC + 5*nv + nv*3;
}

// Build solver pair properties buffer (18 floats per pair)
// 関節リミット拘束の静的テーブル(1関節13float): dof, qposadr, lo, hi, margin, solref(2), solimp(5), invweight。hinge/slideのlimited関節のみ。
static std::vector<float> build_solver_limit_table(const Model& m, int& n_out) {
    std::vector<float> tab;
    n_out = 0;
    if ((m.opt.disableflags & DisableBit::LIMIT) || m.jnt_limited.size() == 0) return tab;
    mx::eval(m.jnt_limited, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_range, m.dof_invweight0);
    auto limited = m.jnt_limited.data<int>();
    auto jtypes = m.jnt_type.data<int>();
    auto jqpa = m.jnt_qposadr.data<int>();
    auto jda = m.jnt_dofadr.data<int>();
    auto jrange = m.jnt_range.data<float>();
    auto iw = m.dof_invweight0.data<float>();
    const float* margin = nullptr; const float* sref = nullptr; const float* simp = nullptr;
    if (m.jnt_margin.size() > 0) { mx::eval(m.jnt_margin); margin = m.jnt_margin.data<float>(); }
    if (m.jnt_solref.size() > 0) { mx::eval(m.jnt_solref); sref = m.jnt_solref.data<float>(); }
    if (m.jnt_solimp.size() > 0) { mx::eval(m.jnt_solimp); simp = m.jnt_solimp.data<float>(); }
    for (int j = 0; j < m.njnt; j++) {
        if (!limited[j]) continue;
        if (jtypes[j] != (int)JointType::SLIDE && jtypes[j] != (int)JointType::HINGE) continue;
        tab.push_back((float)jda[j]);
        tab.push_back((float)jqpa[j]);
        tab.push_back(jrange[j*2]);
        tab.push_back(jrange[j*2+1]);
        tab.push_back(margin ? margin[j] : 0.0f);
        tab.push_back(sref ? sref[j*2] : 0.02f);
        tab.push_back(sref ? sref[j*2+1] : 1.0f);
        const float dsi[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
        for (int k = 0; k < 5; k++) tab.push_back(simp ? simp[j*5+k] : dsi[k]);
        tab.push_back(iw[jda[j]]);
        n_out++;
    }
    return tab;
}

// DOF摩擦損失(frictionloss)拘束の静的テーブル(1dofあたり10float): dof, frictionloss, solref(2), solimp(5), invweight。
static std::vector<float> build_solver_friction_table(const Model& m, int& n_out) {
    std::vector<float> tab;
    n_out = 0;
    if ((m.opt.disableflags & DisableBit::FRICTIONLOSS) || m.dof_frictionloss.size() == 0) return tab;
    mx::eval(m.dof_frictionloss, m.dof_invweight0, m.dof_solref, m.dof_solimp);
    auto fl = m.dof_frictionloss.data<float>();
    auto iw = m.dof_invweight0.data<float>();
    auto sr = m.dof_solref.data<float>();
    auto si = m.dof_solimp.data<float>();
    for (int i = 0; i < m.nv; i++) {
        if (fl[i] <= 0.0f) continue;
        tab.push_back((float)i);
        tab.push_back(fl[i]);
        tab.push_back(sr[i*2]);
        tab.push_back(sr[i*2+1]);
        for (int k = 0; k < 5; k++) tab.push_back(si[i*5+k]);
        tab.push_back(iw[i]);
        n_out++;
    }
    return tab;
}

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

// ── Context: GPU kernels + model constants ─────────────────────────────────

using KernelFn = mkx::fast::Kernel<>;

struct BatchedStepContext {
    std::optional<KernelFn> kin_kernel;
    std::optional<KernelFn> euler_kernel;
    std::optional<KernelFn> euler_devmem_kernel;
    std::optional<KernelFn> forward_kernel;
    bool uses_devmem_euler = false;
    bool uses_gpu_forward = false;

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
    bool uses_gpu_collision = false;
    mx::array coll_pair_data = mx::zeros({1});
    mx::array coll_mesh_verts = mx::zeros({1});
    mx::array coll_mesh_vertadr = mx::zeros({1});
    mx::array coll_mesh_vertnum = mx::zeros({1});
    mx::array coll_geom_dataid = mx::zeros({1});
    mx::array coll_geom_size = mx::zeros({1});
    int num_collision_pairs = 0;

    // Solver kernel
    std::optional<KernelFn> solver_kernel;
    std::optional<KernelFn> cfrc_ext_kernel;
    bool uses_gpu_solver = false;
    mx::array solver_pair_props = mx::zeros({1});
    mx::array solver_body_dof_masks = mx::zeros({1});
    mx::array solver_body_rootid = mx::zeros({1});
    mx::array solver_limit_tab = mx::zeros({1});
    mx::array solver_fric_tab = mx::zeros({1});
    int solver_nlim = 0;
    int solver_nfl = 0;
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
    // DECISION: GPU kernel stack memory is limited (~24KB per thread). The kinematics
    // kernel needs ~7 floats per body (pos, quat) as working storage. If nbody is too
    // large, we skip the GPU kernel and fall back to the scalar CPU path.
    int stack_bytes = m.nbody * 7 * 4;
    if (stack_bytes <= 24000) {
        ctx->kin_kernel = mkx_kernels::make_kinematics_kernel(m.nbody, m.njnt, m.nq, m.ngeom);
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

    // mujoco本家のimplicit/implicitfast(mj_implicitSkip https://github.com/google-deepmind/mujoco/blob/71d430c71f8593a977136485e81314ec19a66e7e/src/engine/engine_forward.c#L1799 がactuatorVelDeriv https://github.com/google-deepmind/mujoco/blob/71d430c71f8593a977136485e81314ec19a66e7e/src/engine/engine_derivative.c#L2503 でd(qfrc_actuator)/d(qvel)を陰的に畳み込む)相当をdof_dampingと同じ機構に便乗させて再現(本カーネルはintegrator設定無視でexplicitのままだったためUR5eのimplicitfast前提PDゲインが発散していた)。
    if (m.nu > 0 && m.actuator_biastype.size() > 0) {
        mx::eval(m.actuator_biastype, m.actuator_biasprm, m.actuator_gaintype, m.actuator_gainprm,
                 m.actuator_trntype, m.actuator_trnid, m.actuator_gear);
        auto biastype_ptr = m.actuator_biastype.data<int32_t>();
        auto biasprm_ptr = m.actuator_biasprm.data<float>();
        auto gaintype_ptr = m.actuator_gaintype.data<int32_t>();
        auto gainprm_ptr = m.actuator_gainprm.data<float>();
        auto atrntype_ptr = m.actuator_trntype.data<int32_t>();
        auto atrnid_ptr = m.actuator_trnid.data<int32_t>();
        auto agear_ptr = m.actuator_gear.data<float>();
        int bias_ncol = (int)m.actuator_biasprm.shape(1);
        int gain_ncol = (int)m.actuator_gainprm.shape(1);
        int gear_ncol = (int)m.actuator_gear.shape(1);
        for (int ai = 0; ai < m.nu; ai++) {
            int trnt = atrntype_ptr[ai];
            if (trnt != 0 && trnt != 1) continue;  // JOINT/JOINTINPARENT以外(TENDON/SITE)は対象外
            int ji = atrnid_ptr[ai * 2];
            if (ji < 0 || ji >= m.njnt) continue;
            float bias_vel = (biastype_ptr[ai] == 1) ? biasprm_ptr[ai * bias_ncol + 2] : 0.0f;  // 1=AFFINE
            float gain_vel = (gaintype_ptr[ai] == 1) ? gainprm_ptr[ai * gain_ncol + 2] : 0.0f;   // 1=AFFINE
            if (gain_vel != 0.0f) continue;  // gain2利用時はctrl依存になり定数補正では表現不可、未対応のまま安全側にスキップ
            if (bias_vel == 0.0f) continue;
            float gear = agear_ptr[ai * gear_ncol];
            int da = da_ptr[ji];
            damp_vals[da] += -bias_vel * gear * gear;
        }
    }

    // Build Euler kernel — two tiers:
    //   nv ≤ 80:   thread-local kernel (fastest, everything in registers/stack)
    //   nv ≤ 2048: device-memory kernel (L in global GPU memory, vectors thread-local)
    int euler_stack = 2 * m.nv * m.nv + 4 * m.nv + m.nq;
    if (euler_stack * 4 <= 24000 && m.nv <= 80) {
        ctx->euler_kernel = mkx_kernels::make_euler_kernel(m.nv, m.nq, m.opt.timestep, s_qa, s_da, fj, bj, damp_vals);
    } else if (m.nv <= EULER_DEVMEM_MAX_NV) {
        ctx->euler_devmem_kernel = mkx_kernels::make_euler_devmem_kernel(m.nv, m.nq, m.opt.timestep, s_qa, s_da, fj, bj, damp_vals);
        ctx->uses_devmem_euler = true;
    }

    // Build all-GLSL forward kernel whenever there's anything to collide (see ponytail note above the collision/solver gates below for why nv alone no longer excludes this path).
    {
        ctx->fwd_scratch_per_env = forward_scratch_per_env(m);
        if (m.nu > 0 && m.cache.act_moment_const.size() > 0) {
            ctx->act_moment = mx::astype(mx::flatten(m.cache.act_moment_const), mx::float32);
        } else {
            ctx->act_moment = mx::zeros({std::max(m.nv, 1)});
        }
        mx::eval(ctx->act_moment);

        ForwardHostConsts fc = extract_forward_host_consts(m);
        ctx->forward_kernel = mkx_kernels::make_forward_kernel(
            m.nbody, m.nv, m.nq, m.nu, m.njnt, m.opt.timestep, fc.gravity,
            fc.body_parentid, fc.body_rootid, fc.body_mass, fc.body_inertia,
            fc.dof_bodyid, fc.dof_parentid, fc.dof_damping, fc.dof_armature,
            fc.dof_stiffness, fc.dof_qposadr, fc.qpos_spring, fc.act_gain0, fc.act_bias0,
            fc.dof_jtype, fc.dof_rotaxis, fc.dof_jid, fc.body_dofs, fc.jnt_dofadr0,
            fc.act_gain1, fc.act_gain2, fc.act_bias1, fc.act_bias2,
            fc.act_gear, fc.act_qposadr, fc.act_dofadr, fc.body_gravcomp);
        ctx->uses_gpu_forward = true;
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

        ctx->collision_kernel = mkx_kernels::make_collision_kernel(
            m.ngeom, (int)m.cache.collision_pairs.size(), MAX_CONTACTS_PER_ENV);
        ctx->uses_gpu_collision = true;
    }

    // Build solver kernel (constraint construction + Newton solver) whenever there's anything to collide.
    if (m.cache.collision_pairs.size() > 0) {
        {
            auto lt = build_solver_limit_table(m, ctx->solver_nlim);
            auto ft = build_solver_friction_table(m, ctx->solver_nfl);
            if (lt.empty()) lt.push_back(0.0f);
            if (ft.empty()) ft.push_back(0.0f);
            ctx->solver_limit_tab = mx::array(lt.data(), {(int)lt.size()}, mx::float32);
            ctx->solver_fric_tab = mx::array(ft.data(), {(int)ft.size()}, mx::float32);
            mx::eval(ctx->solver_limit_tab, ctx->solver_fric_tab);
        }
        int raw_si = (solver_iters_override > 0) ? solver_iters_override : std::max(m.opt.iterations, 1);
        // モデル既定のiterations(CPU厳密Cholesky向けに100等)はGPUのCGソルバでは過大なので3で頭打ちにする。呼び出し側がsolver_iterationsで明示した値は尊重する(接触が複数重なると3回では収束せず速度が発散する場合がある)。
        int si = (solver_iters_override > 0) ? solver_iters_override : std::min(raw_si, 3);
        int cgi = 10;
        // frictionloss拘束(区分二次で飽和する)は有効/無効の切り替わりを反復で辿るため、3回では静止付近で1e-2級のずれが残る(8回で1e-4)。速度優先ならconfigのsolver_iterationsで下げられる。
        if (ctx->solver_nfl > 0) si = std::max(si, std::min(raw_si, 8));
        ctx->solver_scratch_size = solver_scratch_per_env(m);
        if (ctx->solver_scratch_size != mkx_kernels::solver_scratch_floats(m.nv, MAX_EFC)) throw std::logic_error("mjmlx: solver scratch確保量がカーネルの必要量と不一致");
        ctx->solver_pair_props = build_solver_pair_props(m);
        ctx->solver_body_dof_masks = build_body_dof_masks(m);
        ctx->solver_body_rootid = build_body_rootid(m);
        mx::eval(ctx->solver_pair_props, ctx->solver_body_dof_masks, ctx->solver_body_rootid);

        bool use_pyramidal = (m.opt.cone == ConeType::PYRAMIDAL);
        bool refsafe = (m.opt.disableflags & DisableBit::REFSAFE) == 0;
        ctx->solver_kernel = mkx_kernels::make_solver_kernel(
            m.nbody, m.nv, m.nq, m.opt.timestep, use_pyramidal, refsafe, m.opt.impratio, si, cgi, ctx->solver_nlim, ctx->solver_nfl);
        ctx->cfrc_ext_kernel = mkx_kernels::make_cfrc_ext_kernel(m.nbody, CONTACT_STRIDE, MAX_CONTACTS_PER_ENV);
        ctx->uses_gpu_solver = true;
    }

    return ctx;
}

// ── Hybrid batched step ──────────────────────────────────────────────────────

// mkx::vmap(fn, batched_inputs)相当(issue #14): fnをグラフ複製なしで1回だけ呼ぶ。mx_compat_mkx.hのsum/sum_axis/matmul/reshape等はvmap-awareなmkx::側へ委譲済みなのでfn自体(forward_fn)は書き換え不要。
static std::vector<mx::array> mkx_vmap_batch0(
    const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>& fn,
    const std::vector<mx::array>& batched_inputs, int batch_size)
{
    std::function<std::vector<mx::Raw<float>>(const std::vector<mx::Raw<float>>&)> raw_fn =
        [&fn](const std::vector<mx::Raw<float>>& raw_in) -> std::vector<mx::Raw<float>> {
        std::vector<mx::array> mx_in;
        mx_in.reserve(raw_in.size());
        for (auto& r : raw_in) mx_in.push_back(mx::from_raw(r.node(), mx::Dtype::Float32));
        auto mx_out = fn(mx_in);
        std::vector<mx::Raw<float>> raw_out;
        raw_out.reserve(mx_out.size());
        for (auto& o : mx_out) raw_out.push_back(mx::to_raw(o));
        return raw_out;
    };
    std::vector<mx::Raw<float>> raw_inputs;
    raw_inputs.reserve(batched_inputs.size());
    for (auto& in : batched_inputs) raw_inputs.push_back(mx::to_raw(in));

    auto raw_results = mkx::vmap<mx::Backend>(raw_fn, raw_inputs);

    std::vector<mx::array> results;
    results.reserve(raw_results.size());
    for (auto& r : raw_results) results.push_back(mx::from_raw(r.node(), mx::Dtype::Float32));
    return results;
}

std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu, int solver_iterations_override, const void* owner) {
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

        // For large models (nv > 80), use all-GPU pipeline:
        //   GPU kin → GPU forward → GPU euler_devmem
        // For small models (nv ≤ 80), use hybrid pipeline:
        //   GPU kin → vmap(forward) → GPU euler
        bool use_gpu_fwd = ctx->uses_gpu_forward;

        // vmap forward (only built for nv ≤ 80)
        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> vmapped_fwd;
        if (!use_gpu_fwd) {
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
            vmapped_fwd = [forward_fn, B](const std::vector<mx::array>& state) {
                return mkx_vmap_batch0(forward_fn, state, B);
            };
        }

        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> pipeline =
            [ctx, vmapped_fwd, model_override, use_gpu_fwd, B, nq, nv, nu, nb, nj, ng, owner](
                const std::vector<mx::array>& state) -> std::vector<mx::array>
        {
            auto qpos_batch = state[0];
            auto qvel_batch = state[1];
            auto ctrl_batch = state[2];

            // ── Phase 1: GPU kinematics ──
            auto qpos_flat = mx::astype(mx::flatten(qpos_batch), mx::float32);
            std::vector<mx::Shape> kin_shapes = {
                {B * nb * 3}, {B * nb * 4}, {B * nb * 9},
                {B * nb * 3}, {B * nb * 9},
                (nj > 0) ? mx::Shape{B * nj * 3} : mx::Shape{1},
                (nj > 0) ? mx::Shape{B * nj * 3} : mx::Shape{1},
                (ng > 0) ? mx::Shape{B * ng * 3} : mx::Shape{1},
                (ng > 0) ? mx::Shape{B * ng * 9} : mx::Shape{1},
            };
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
                std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{env_threadgroup(B), 1, 1}
            );
            // loc_idは__LINE__由来なので1出力1行で明示(同一行だと全出力が同じキーに衝突する)。shapeは毎step固定のため全出力永続化(issue #14)。
            MX_MARK_PERSISTENT(kin_raw[0], owner);
            MX_MARK_PERSISTENT(kin_raw[1], owner);
            MX_MARK_PERSISTENT(kin_raw[2], owner);
            MX_MARK_PERSISTENT(kin_raw[3], owner);
            MX_MARK_PERSISTENT(kin_raw[4], owner);
            MX_MARK_PERSISTENT(kin_raw[5], owner);
            MX_MARK_PERSISTENT(kin_raw[6], owner);
            MX_MARK_PERSISTENT(kin_raw[7], owner);
            MX_MARK_PERSISTENT(kin_raw[8], owner);
            auto kin = mx::fast::kernel_outputs(kin_raw);

            auto xpos   = mx::reshape(kin[0], {B, nb, 3});
            auto xipos  = mx::reshape(kin[3], {B, nb, 3});
            auto ximat  = mx::reshape(kin[4], {B, nb, 3, 3});
            auto xanchor = (nj > 0) ? mx::reshape(kin[5], {B, nj, 3}) : mx::zeros({B, 1});
            auto xaxis   = (nj > 0) ? mx::reshape(kin[6], {B, nj, 3}) : mx::zeros({B, 1});

            mx::array qM_flat = mx::zeros({1});
            mx::array qfrc_smooth_flat = mx::zeros({1});
            mx::array qfrc_constraint_flat = mx::zeros({1});
            mx::array cfrc_ext_flat = mx::zeros({B * nb * 6});
            mx::array subtree_com_out = mx::zeros({1});
            mx::array cinert_out = mx::zeros({1});
            mx::array cvel_out = mx::zeros({1});
            mx::array qfrc_actuator_out = mx::zeros({1});

            if (use_gpu_fwd) {
                // ── Phase 2a: all-GLSL forward kernel (whenever there's anything to collide) ──
                auto xmat = mx::reshape(kin[2], {B, nb, 3, 3});
                int scratchSz = ctx->fwd_scratch_per_env;

                std::vector<mx::array> fwd;
                {
                    auto fwd_raw = (*ctx->forward_kernel)(
                        mx::fast::kernel_inputs({mx::flatten(xipos), mx::flatten(ximat),
                         mx::flatten(xanchor), mx::flatten(xaxis), mx::flatten(xmat),
                         mx::flatten(qpos_batch), mx::flatten(qvel_batch), mx::flatten(ctrl_batch),
                         ctx->act_moment}),
                        mx::fast::kernel_shapes({{B * nv * nv}, {B * nv}, {B * nb * 3},
                         {B * nb * 10}, {B * nb * 6}, {B * nv},
                         {B * scratchSz}, {B * nv * 6}}),
                        std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{env_threadgroup(B), 1, 1}
                    );
                    // shapeは毎step固定のため全出力永続化(issue #14: alloc/free churn削減)。
                    MX_MARK_PERSISTENT(fwd_raw[0], owner);
                    MX_MARK_PERSISTENT(fwd_raw[1], owner);
                    MX_MARK_PERSISTENT(fwd_raw[2], owner);
                    MX_MARK_PERSISTENT(fwd_raw[3], owner);
                    MX_MARK_PERSISTENT(fwd_raw[4], owner);
                    MX_MARK_PERSISTENT(fwd_raw[5], owner);
                    MX_MARK_PERSISTENT(fwd_raw[6], owner);
                    MX_MARK_PERSISTENT(fwd_raw[7], owner);
                    fwd = mx::fast::kernel_outputs(fwd_raw);
                }

                qM_flat = fwd[0];
                qfrc_smooth_flat = fwd[1];
                subtree_com_out = mx::reshape(fwd[2], {B, nb, 3});
                cinert_out = mx::reshape(fwd[3], {B, nb, 10});
                cvel_out = mx::reshape(fwd[4], {B, nb, 6});
                qfrc_actuator_out = mx::reshape(fwd[5], {B, nv});
                auto cdof_flat = fwd[7]; // (B * nv * 6)

                // ── Collision + Solver ──
                if (ctx->uses_gpu_collision) {
                    // Collision kernel: detect contacts from geom transforms
                    auto geom_xpos_flat = mx::flatten(mx::reshape(kin[7], {B, ng, 3}));
                    auto geom_xmat_flat = mx::flatten(mx::reshape(kin[8], {B, ng, 3, 3}));

                    int con_buf_sz = B * MAX_CONTACTS_PER_ENV * CONTACT_STRIDE;
                    std::vector<mx::array> coll;
                    {
                        auto coll_raw = (*ctx->collision_kernel)(
                            mx::fast::kernel_inputs({geom_xpos_flat, geom_xmat_flat,
                             ctx->coll_mesh_verts, ctx->coll_pair_data,
                             ctx->coll_mesh_vertadr, ctx->coll_mesh_vertnum,
                             ctx->coll_geom_dataid, ctx->coll_geom_size}),
                            mx::fast::kernel_shapes({{con_buf_sz}, {B}}),
                            std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{env_threadgroup(B), 1, 1}
                        );
                        // shapeはcon_buf_sz(=B*MAX_CONTACTS_PER_ENV*CONTACT_STRIDE)固定のため永続化(issue #14)。
                        MX_MARK_PERSISTENT(coll_raw[0], owner);
                        MX_MARK_PERSISTENT(coll_raw[1], owner);
                        coll = mx::fast::kernel_outputs(coll_raw);
                    }

                    if (ctx->uses_gpu_solver) {
                        int scratch_sz = ctx->solver_scratch_size;
                        std::vector<mx::array> solver;
                        {
                            auto solver_raw = (*ctx->solver_kernel)(
                                mx::fast::kernel_inputs({qM_flat, qfrc_smooth_flat, cdof_flat,
                                 mx::flatten(subtree_com_out),
                                 mx::astype(mx::flatten(qvel_batch), mx::float32),
                                 coll[0], coll[1],
                                 ctx->solver_pair_props, ctx->solver_body_dof_masks,
                                 ctx->solver_body_rootid,
                                 mx::astype(mx::flatten(qpos_batch), mx::float32),
                                 ctx->solver_limit_tab, ctx->solver_fric_tab}),
                                mx::fast::kernel_shapes({{B * nv}, {B * scratch_sz}, {B * MAX_CONTACTS_PER_ENV}}),
                                std::array<uint32_t, 3>{static_cast<uint32_t>(B * nv), 1, 1}, std::array<uint32_t, 3>{static_cast<uint32_t>(nv), 1, 1}
                            );
                            // shapeは毎step固定のため永続化(issue #14)。
                            MX_MARK_PERSISTENT(solver_raw[0], owner);
                            MX_MARK_PERSISTENT(solver_raw[1], owner);
                            MX_MARK_PERSISTENT(solver_raw[2], owner);
                            solver = mx::fast::kernel_outputs(solver_raw);
                        }
                        qfrc_constraint_flat = solver[0];
                        auto cfrc_raw = (*ctx->cfrc_ext_kernel)(
                            mx::fast::kernel_inputs({coll[0], coll[1], solver[2], ctx->solver_pair_props}),
                            mx::fast::kernel_shapes({{B * nb * 6}}),
                            std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1}, std::array<uint32_t, 3>{env_threadgroup(B), 1, 1});
                        MX_MARK_PERSISTENT(cfrc_raw[0], owner);
                        cfrc_ext_flat = mx::fast::kernel_outputs(cfrc_raw)[0];
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

            // ── Phase 3: GPU Euler ──
            std::vector<mx::array> euler_inputs = {
                mx::astype(qM_flat, mx::float32),
                mx::astype(qfrc_smooth_flat, mx::float32),
                mx::astype(qfrc_constraint_flat, mx::float32),
                mx::astype(mx::flatten(qvel_batch), mx::float32),
                mx::astype(mx::flatten(qpos_batch), mx::float32)
            };
            std::vector<mx::array> euler;
            auto grid = std::array<uint32_t, 3>{static_cast<uint32_t>(B), 1, 1};
            auto tgroup = std::array<uint32_t, 3>{env_threadgroup(B), 1, 1};
            // shapeは毎step固定のため全出力永続化(issue #14)。
            if (ctx->euler_kernel.has_value()) {
                auto raw = (*ctx->euler_kernel)(mx::fast::kernel_inputs(euler_inputs), mx::fast::kernel_shapes({{B * nq}, {B * nv}, {B * nv}}), grid, tgroup);
                MX_MARK_PERSISTENT(raw[0], owner);
                MX_MARK_PERSISTENT(raw[1], owner);
                MX_MARK_PERSISTENT(raw[2], owner);
                euler = mx::fast::kernel_outputs(raw);
            } else {
                auto raw = (*ctx->euler_devmem_kernel)(mx::fast::kernel_inputs(euler_inputs), mx::fast::kernel_shapes({{B * nq}, {B * nv}, {B * nv}, {B * nv * nv}}), grid, tgroup);
                MX_MARK_PERSISTENT(raw[0], owner);
                MX_MARK_PERSISTENT(raw[1], owner);
                MX_MARK_PERSISTENT(raw[2], owner);
                MX_MARK_PERSISTENT(raw[3], owner);
                euler = mx::fast::kernel_outputs(raw);
            }

            auto new_qpos = mx::reshape(euler[0], {B, nq});
            auto new_qvel = mx::reshape(euler[1], {B, nv});
            auto xpos_out = mx::reshape(kin[0], {B, nb, 3});
            auto xquat_out = mx::reshape(kin[1], {B, nb, 4});
            auto cfrc_ext_out = mx::reshape(cfrc_ext_flat, {B, nb, 6});

            return {new_qpos, new_qvel, xpos_out, xquat_out,
                    subtree_com_out, cinert_out, cvel_out,
                    qfrc_actuator_out, cfrc_ext_out};
        };

        // mx::compile fuses ops (reshape, astype, etc.) in the computation graph.
        // For nv≤80 (vmap path), this provides significant speedup by fusing many
        // small ops. For nv>80 (all-GPU path), custom kernels already do bulk work
        // per dispatch and mx::compile can deadlock with the large solver kernel.
        if (!use_gpu_fwd) {
            auto compiled = mx::compile(pipeline);
            return compiled;
        }
        return pipeline;
    }

    // GPU kernels could not be built for this model.
    throw std::runtime_error(
        "GPU kernels not available for this model (nv=" + std::to_string(nv) +
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
    std::vector<float> xq(B * nb * 4);
    std::vector<float> ci(B * nb * 10), cv(B * nb * 6);
    std::vector<float> qa(B * nv), ce(B * nb * 6);

    for (int i = 0; i < B; i++) {
        const mjData* d = handle->cpu_datas[i];
        for (int j = 0; j < nq; j++) qp[i*nq + j] = (float)d->qpos[j];
        for (int j = 0; j < nv; j++) qv[i*nv + j] = (float)d->qvel[j];
        for (int j = 0; j < nb*3; j++) xp[i*nb*3 + j] = (float)d->xpos[j];
        for (int j = 0; j < nb*4; j++) xq[i*nb*4 + j] = (float)d->xquat[j];
        for (int j = 0; j < nb*3; j++) sc[i*nb*3 + j] = (float)d->subtree_com[j];
        for (int j = 0; j < nb*10; j++) ci[i*nb*10 + j] = (float)d->cinert[j];
        for (int j = 0; j < nb*6; j++) cv[i*nb*6 + j] = (float)d->cvel[j];
        for (int j = 0; j < nv; j++) qa[i*nv + j] = (float)d->qfrc_actuator[j];
        for (int j = 0; j < nb*6; j++) ce[i*nb*6 + j] = (float)d->cfrc_ext[j];
    }

    s.qpos = mx::reshape(mx::array(qp.data(), {B*nq}, mx::float32), {B, nq});
    s.qvel = mx::reshape(mx::array(qv.data(), {B*nv}, mx::float32), {B, nv});
    s.xpos = mx::reshape(mx::array(xp.data(), {B*nb*3}, mx::float32), {B, nb, 3});
    s.xquat = mx::reshape(mx::array(xq.data(), {B*nb*4}, mx::float32), {B, nb, 4});
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
            {
                // fix_mjmlx_batched_init_derived: 未初期化mx::array({})読み出し対策でゼロ/恒等クォータニオン初期化。
                int nb = model->model.nbody;
                std::vector<float> xq0(static_cast<size_t>(B) * nb * 4, 0.0f);
                for (size_t i = 0; i < xq0.size(); i += 4) xq0[i] = 1.0f;  // wxyz恒等クォータニオン
                handle->sim.xpos = mx::zeros({B, nb, 3});
                handle->sim.xquat = mx::reshape(mx::array(xq0.data(), {static_cast<int>(xq0.size())}, mx::float32), {B, nb, 4});
                handle->sim.subtree_com = mx::zeros({B, nb, 3});
                handle->sim.cinert = mx::zeros({B, nb, 10});
                handle->sim.cvel = mx::zeros({B, nb, 6});
                handle->sim.qfrc_actuator = mx::zeros({B, nv});
                handle->sim.cfrc_ext = mx::zeros({B, nb, 6});
            }
        } else {
            // GPU path: compiled + vmapped MLX step function
            // 全envが同一のqpos0/zero qvelなのでstack()のO(B)回concatenateループを避け、ホスト側でB個複製した1本の配列を直接作る(初期化がO(B)dispatchからO(1)になる)。
            mx::eval(model->model.qpos0);
            const float* q0 = model->model.qpos0.data<float>();
            std::vector<float> qp0(static_cast<size_t>(B) * nq);
            for (int i = 0; i < B; i++) std::copy(q0, q0 + nq, qp0.begin() + static_cast<size_t>(i) * nq);
            handle->sim.qpos = mx::reshape(mx::array(qp0.data(), {B * nq}, mx::float32), {B, nq});
            handle->sim.qvel = mx::zeros({B, nv});
            {
                // fix_mjmlx_batched_init_derived: 未初期化mx::array({})読み出し対策でゼロ/恒等クォータニオン初期化。
                int nb = model->model.nbody;
                std::vector<float> xq0(static_cast<size_t>(B) * nb * 4, 0.0f);
                for (size_t i = 0; i < xq0.size(); i += 4) xq0[i] = 1.0f;  // wxyz恒等クォータニオン
                handle->sim.xpos = mx::zeros({B, nb, 3});
                handle->sim.xquat = mx::reshape(mx::array(xq0.data(), {static_cast<int>(xq0.size())}, mx::float32), {B, nb, 4});
                handle->sim.subtree_com = mx::zeros({B, nb, 3});
                handle->sim.cinert = mx::zeros({B, nb, 10});
                handle->sim.cvel = mx::zeros({B, nb, 6});
                handle->sim.qfrc_actuator = mx::zeros({B, nv});
                handle->sim.cfrc_ext = mx::zeros({B, nb, 6});
            }

            handle->sim.compiled_step = mjmlx::make_batched_step(
                model->model, B, config->use_gpu, config->solver_iterations, &handle->sim);
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
    if (results.size() > 3) s.xquat = results[3];
    if (results.size() > 4) s.subtree_com = results[4];
    if (results.size() > 5) s.cinert = results[5];
    if (results.size() > 6) s.cvel = results[6];
    if (results.size() > 7) s.qfrc_actuator = results[7];
    if (results.size() > 8) s.cfrc_ext = results[8];
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

MJMLX_API void mjmlx_batched_set_state(MjmlxBatchedSim* sim, const float* qpos, const float* qvel) {
    if (!sim) return;

    if (sim->cpu_mode) {
        int nq = sim->cpu_model->nq, nv = sim->cpu_model->nv;
        for (int i = 0; i < sim->sim.num_envs; i++) {
            mjData* d = sim->cpu_datas[i];
            if (qpos) for (int j = 0; j < nq; j++) d->qpos[j] = (double)qpos[i * nq + j];
            if (qvel) for (int j = 0; j < nv; j++) d->qvel[j] = (double)qvel[i * nv + j];
        }
        cpu_gather_state(sim);
        return;
    }

    auto& s = sim->sim;
    int B = s.num_envs;
    if (qpos) s.qpos = mx::reshape(mx::array(qpos, {B * s.model->nq}, mx::float32), {B, s.model->nq});
    if (qvel) s.qvel = mx::reshape(mx::array(qvel, {B * s.model->nv}, mx::float32), {B, s.model->nv});
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

MJMLX_API const float* mjmlx_batched_get_xquat(const MjmlxBatchedSim* sim, int* n_out) {
    if (!sim) return nullptr;
    mx::eval(sim->sim.xquat);
    if (n_out) *n_out = sim->sim.num_envs * sim->sim.model->nbody * 4;
    return sim->sim.xquat.data<float>();
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

    mx::eval(s.qpos, s.qvel);
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
    if (sim) mx::release_persistent_for_owner(&sim->sim);
    delete sim;
}

} // extern "C"
