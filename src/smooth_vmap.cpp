// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// Vmap-compatible smooth dynamics: COM, CRB, mass matrix factorization,
// velocity-dependent, RNE, transmission.
// NO eval(), NO data<>(), NO CPU sync. Pure MLX graph building.

#include "internal.h"

namespace mjmlx {

static constexpr float MJMINVAL_V = 1e-8f;

// Helper: row from 2D using pure array ops (no eval)
static mx::array vrow(const mx::array& arr, int i) {
    return mx::flatten(mx::slice(arr, mx::Shape{i, 0}, mx::Shape{i + 1, static_cast<int>(arr.shape(1))}));
}

// ── Vmap-compatible COM position ─────────────────────────────────────────────

Data vmap_com_pos(const Model& m, Data d) {
    const auto& c = m.cache;
    int nb = m.nbody;

    // Vectorized subtree COM: xipos * mass[:, None]
    auto mass_col = mx::reshape(m.body_mass, mx::Shape{nb, 1});  // (nb, 1)
    auto sub_pos = mx::multiply(d.xipos, mass_col);  // (nb, 3)
    auto sub_mass = mx::copy(m.body_mass);  // (nb,)

    // Level-parallel backward accumulation using precomputed scatter matrices
    for (int lvl = (int)c.tree_levels.size() - 1; lvl >= 1; lvl--) {
        const auto& level = c.tree_levels[lvl];
        if (level.empty()) continue;
        int nc = (int)level.size();
        // Gather child values
        auto child_ids = mx::array(level.data(), mx::Shape{nc}, mx::int32);
        auto child_pos = mx::take(sub_pos, child_ids, 0);  // (nc, 3)
        auto child_mass = mx::take(sub_mass, child_ids, 0);  // (nc,)
        // Build parent IDs for this level
        std::vector<int> pids(nc);
        for (int i = 0; i < nc; i++) pids[i] = c.body_parentid_vec[level[i]];
        auto parent_ids = mx::array(pids.data(), mx::Shape{nc}, mx::int32);
        // Scatter-add: for each child, add to its parent row
        // Use precomputed (nb, nc) scatter matrix: scatter_mat[pid, idx] = 1
        std::vector<float> smat(nb * nc, 0.0f);
        for (int i = 0; i < nc; i++) smat[pids[i] * nc + i] = 1.0f;
        auto scatter_mat = mx::array(smat.data(), mx::Shape{nb, nc}, mx::float32);
        sub_pos = mx::add(sub_pos, mx::matmul(scatter_mat, child_pos));
        sub_mass = mx::add(sub_mass, mx::flatten(mx::matmul(scatter_mat,
            mx::reshape(child_mass, mx::Shape{nc, 1}))));
    }

    auto safe_mass = mx::maximum(sub_mass, mx::array(MJMINVAL_V));
    d.subtree_com = mx::divide(sub_pos, mx::reshape(safe_mass, mx::Shape{nb, 1}));

    // Fully vectorized cinert computation
    auto root_ids = mx::array(c.body_rootid_vec.data(), mx::Shape{nb}, mx::int32);
    auto root_com = mx::take(d.subtree_com, root_ids, 0);  // (nb, 3)
    auto offsets = mx::subtract(d.xipos, root_com);  // (nb, 3)
    auto masses = mx::reshape(m.body_mass, mx::Shape{nb, 1, 1});  // (nb, 1, 1)

    // Transform inertia to global frame: R @ diag(I) @ R^T
    // ximat: (nb, 3, 3), body_inertia: (nb, 3) → (nb, 1, 3)
    auto RI = mx::multiply(d.ximat, mx::reshape(m.body_inertia, mx::Shape{nb, 1, 3}));
    auto ximat_T = mx::transpose(d.ximat, {0, 2, 1});
    auto I_global = mx::matmul(RI, ximat_T);  // (nb, 3, 3)

    // Parallel axis theorem: I += m * (d^2 * I3 - outer(d, d))
    auto d2 = mx::sum(mx::multiply(offsets, offsets), -1, /* keepdims = */ true);  // (nb, 1)
    auto off_col = mx::reshape(offsets, mx::Shape{nb, 3, 1});
    auto off_row = mx::reshape(offsets, mx::Shape{nb, 1, 3});
    auto outer_prod = mx::matmul(off_col, off_row);  // (nb, 3, 3)
    auto d2_3d = mx::reshape(d2, mx::Shape{nb, 1, 1});
    auto pat = mx::multiply(masses, mx::subtract(
        mx::multiply(d2_3d, mx::eye(3)), outer_prod));
    I_global = mx::add(I_global, pat);

    // Extract inertia components: (nb, 3, 3) → 6 unique components
    auto Ixx = mx::slice(I_global, mx::Shape{0,0,0}, mx::Shape{nb,1,1});  // (nb,1,1)
    auto Iyy = mx::slice(I_global, mx::Shape{0,1,1}, mx::Shape{nb,2,2});
    auto Izz = mx::slice(I_global, mx::Shape{0,2,2}, mx::Shape{nb,3,3});
    auto Ixy = mx::slice(I_global, mx::Shape{0,0,1}, mx::Shape{nb,1,2});
    auto Ixz = mx::slice(I_global, mx::Shape{0,0,2}, mx::Shape{nb,1,3});
    auto Iyz = mx::slice(I_global, mx::Shape{0,1,2}, mx::Shape{nb,2,3});
    // Reshape all to (nb, 1)
    Ixx = mx::reshape(Ixx, mx::Shape{nb, 1});
    Iyy = mx::reshape(Iyy, mx::Shape{nb, 1});
    Izz = mx::reshape(Izz, mx::Shape{nb, 1});
    Ixy = mx::reshape(Ixy, mx::Shape{nb, 1});
    Ixz = mx::reshape(Ixz, mx::Shape{nb, 1});
    Iyz = mx::reshape(Iyz, mx::Shape{nb, 1});

    // pm = offset * mass, mass_1d for last column
    auto pm = mx::multiply(offsets, mx::reshape(m.body_mass, mx::Shape{nb, 1}));  // (nb, 3)
    auto mass_1d = mx::reshape(m.body_mass, mx::Shape{nb, 1});

    // cinert: (nb, 10) = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, px, py, pz, mass]
    d.cinert = mx::concatenate({Ixx, Iyy, Izz, Ixy, Ixz, Iyz, pm, mass_1d}, 1);

    // Vectorized cdof using pre-computed plan
    if (m.nv > 0) {
        const auto& p = c.cdof_plan;

        auto anchor = mx::take(d.xanchor, p.jidxs, 0);
        auto axis = mx::take(d.xaxis, p.jidxs, 0);
        auto root_com = mx::take(d.subtree_com, p.root_bids, 0);
        auto offset = mx::subtract(root_com, anchor);

        auto xmats = mx::take(d.xmat, p.bids, 0);
        auto col0 = mx::reshape(mx::slice(xmats, mx::Shape{0,0,0}, mx::Shape{m.nv,3,1}), mx::Shape{m.nv, 3});
        auto col1 = mx::reshape(mx::slice(xmats, mx::Shape{0,0,1}, mx::Shape{m.nv,3,2}), mx::Shape{m.nv, 3});
        auto col2 = mx::reshape(mx::slice(xmats, mx::Shape{0,0,2}, mx::Shape{m.nv,3,3}), mx::Shape{m.nv, 3});
        auto rot_a = mx::add(mx::add(
            mx::multiply(col0, p.rot_col0_mask),
            mx::multiply(col1, p.rot_col1_mask)),
            mx::multiply(col2, p.rot_col2_mask));

        auto axis_x_off = mx::linalg::cross(axis, offset);
        auto rot_x_off = mx::linalg::cross(rot_a, offset);

        auto hinge_cdof = mx::concatenate({axis, axis_x_off}, 1);
        auto slide_cdof = mx::concatenate({mx::zeros(mx::Shape{m.nv, 3}), axis}, 1);
        auto ftrans_cdof = mx::concatenate({mx::zeros(mx::Shape{m.nv, 3}), p.free_trans_unit}, 1);
        auto frot_cdof = mx::concatenate({rot_a, rot_x_off}, 1);

        d.cdof = mx::add(mx::add(mx::add(mx::add(
            mx::multiply(p.is_hinge, hinge_cdof),
            mx::multiply(p.is_slide, slide_cdof)),
            mx::multiply(p.is_free_trans, ftrans_cdof)),
            mx::multiply(p.is_free_rot, frot_cdof)),
            mx::multiply(p.is_ball, frot_cdof));
    }

    return d;
}

// ── Vmap-compatible CRB ──────────────────────────────────────────────────────

Data vmap_crb(const Model& m, Data d) {
    const auto& c = m.cache;
    int nb = m.nbody;

    // Level-parallel backward accumulation using scatter matrices
    auto crb = mx::copy(d.cinert);  // (nb, 10)

    for (int lvl = (int)c.tree_levels.size() - 1; lvl >= 1; lvl--) {
        const auto& level = c.tree_levels[lvl];
        if (level.empty()) continue;
        int nc = (int)level.size();
        auto child_ids = mx::array(level.data(), mx::Shape{nc}, mx::int32);
        auto child_vals = mx::take(crb, child_ids, 0);  // (nc, 10)
        std::vector<int> pids(nc);
        for (int i = 0; i < nc; i++) pids[i] = c.body_parentid_vec[level[i]];
        std::vector<float> smat(nb * nc, 0.0f);
        for (int i = 0; i < nc; i++) smat[pids[i] * nc + i] = 1.0f;
        auto scatter_mat = mx::array(smat.data(), mx::Shape{nb, nc}, mx::float32);
        crb = mx::add(crb, mx::matmul(scatter_mat, child_vals));
    }
    // Zero out world body
    auto world_mask = mx::concatenate({mx::zeros(mx::Shape{1, 10}),
                                        mx::ones(mx::Shape{nb - 1, 10})}, 0);
    d.crb = mx::multiply(crb, world_mask);

    // Vectorized crb_cdof and mass matrix
    if (m.nv > 0) {
        auto dof_bid_arr = mx::array(c.dof_bodyid_vec.data(), mx::Shape{m.nv}, mx::int32);
        auto crb_dof = mx::take(d.crb, dof_bid_arr, 0);
        auto crb_cdof = batched_inert_mul(crb_dof, d.cdof);

        // Dense mass matrix: qM = (crb_cdof @ cdof^T) * tree_mask
        auto qm = mx::matmul(crb_cdof, mx::transpose(d.cdof));
        if (m.dof_armature.size() > 0) {
            qm = mx::add(qm, mx::diag(m.dof_armature));
        }
        qm = mx::multiply(qm, c.make_m_mask);
        auto lower = mx::tril(qm, -1);
        qm = mx::add(qm, mx::transpose(lower));
        d.qM = qm;
    }

    return d;
}

// ── Vmap-compatible mass matrix factorization (pure GPU) ─────────────────────
// Uses Neumann series to approximate M^{-1} without CPU linalg.
// For SPD mass matrices: M^{-1} ≈ D^{-1} (I + N + N^2 + N^3)
// where D = diag(M), N = I - D^{-1} M.
// All operations are GPU-native matmul + elementwise.

Data vmap_factor_m(const Model& m, Data d) {
    int n = m.nv;
    auto A = mx::add(d.qM, mx::multiply(mx::eye(n), mx::array(1e-6f)));
    d.qLD = A;  // store regularized M (used by Euler kernel via Metal)

    // GPU-native approximate inverse via Neumann series
    auto diag_A = mx::diag(A);                                     // (n,)
    auto D_inv = mx::reciprocal(mx::maximum(diag_A, mx::array(1e-10f)));  // (n,)
    auto D_inv_mat = mx::diag(D_inv);                              // (n, n)
    auto N = mx::subtract(mx::eye(n), mx::matmul(D_inv_mat, A));  // I - D^{-1}M

    // Neumann: (I + N + N^2 + N^3) @ D^{-1}
    auto N2 = mx::matmul(N, N);
    auto N3 = mx::matmul(N2, N);
    auto poly = mx::add(mx::eye(n), mx::add(N, mx::add(N2, N3)));
    d.qM_inv = mx::matmul(poly, D_inv_mat);

    return d;
}

// ── Vmap-compatible M^{-1} @ rhs using precomputed GPU inverse ──────────────

mx::array vmap_solve_m(const Model& m, const Data& d, const mx::array& rhs) {
    return mx::flatten(mx::matmul(d.qM_inv, mx::reshape(rhs, mx::Shape{m.nv, 1})));
}

// ── Vmap-compatible COM velocity ─────────────────────────────────────────────

Data vmap_com_vel(const Model& m, Data d) {
    const auto& c = m.cache;

    std::vector<mx::array> cvel(m.nbody, mx::array(0.0f));
    std::vector<mx::array> cdof_dot(m.nv, mx::array(0.0f));
    for (int i = 0; i < m.nbody; i++) cvel[i] = mx::zeros({6});
    for (int i = 0; i < m.nv; i++) cdof_dot[i] = mx::zeros({6});

    for (int lvl = 1; lvl < (int)c.tree_levels.size(); lvl++) {
        for (int bid : c.tree_levels[lvl]) {
            int pid = c.body_parentid_vec[bid];
            auto cv = cvel[pid];

            for (int di : c.body_dofs[bid]) {
                cdof_dot[di] = motion_cross(cv, vrow(d.cdof, di));
                auto qvi = mx::slice(d.qvel, mx::Shape{di}, mx::Shape{di+1});
                cv = mx::add(cv, mx::multiply(vrow(d.cdof, di), qvi));
            }
            cvel[bid] = cv;
        }
    }

    d.cvel = mx::stack(cvel);
    d.cdof_dot = (m.nv > 0) ? mx::stack(cdof_dot) : mx::zeros({0, 6});
    return d;
}

// ── Vmap-compatible RNE ──────────────────────────────────────────────────────

Data vmap_rne(const Model& m, Data d) {
    const auto& c = m.cache;

    // Forward pass: compute cacc
    std::vector<mx::array> cacc(m.nbody, mx::array(0.0f));
    cacc[0] = c.gravity_6d;

    for (int lvl = 1; lvl < (int)c.tree_levels.size(); lvl++) {
        for (int bid : c.tree_levels[lvl]) {
            int pid = c.body_parentid_vec[bid];
            auto acc = cacc[pid];

            for (int di : c.body_dofs[bid]) {
                auto qvi = mx::slice(d.qvel, mx::Shape{di}, mx::Shape{di+1});
                acc = mx::add(acc, mx::multiply(vrow(d.cdof_dot, di), qvi));
            }
            cacc[bid] = acc;
        }
    }

    // Vectorized local forces: f = I*a + v × (I*v)
    auto cacc_arr = mx::stack(cacc);
    auto Ia = batched_inert_mul(d.cinert, cacc_arr);
    auto Iv = batched_inert_mul(d.cinert, d.cvel);
    auto vxIv = batched_motion_cross_force(d.cvel, Iv);
    auto loc_cfrc = mx::add(Ia, vxIv);

    // Level-parallel backward force accumulation using scatter matrices
    auto cfrc_arr = mx::copy(loc_cfrc);  // (nb, 6)
    for (int lvl = (int)c.tree_levels.size() - 1; lvl >= 1; lvl--) {
        const auto& level = c.tree_levels[lvl];
        if (level.empty()) continue;
        int nc = (int)level.size();
        auto child_ids = mx::array(level.data(), mx::Shape{nc}, mx::int32);
        auto child_vals = mx::take(cfrc_arr, child_ids, 0);
        std::vector<int> pids(nc);
        for (int i = 0; i < nc; i++) pids[i] = c.body_parentid_vec[level[i]];
        std::vector<float> smat(m.nbody * nc, 0.0f);
        for (int i = 0; i < nc; i++) smat[pids[i] * nc + i] = 1.0f;
        auto scatter_mat = mx::array(smat.data(), mx::Shape{m.nbody, nc}, mx::float32);
        cfrc_arr = mx::add(cfrc_arr, mx::matmul(scatter_mat, child_vals));
    }
    auto dof_bid_arr = mx::array(c.dof_bodyid_vec.data(), mx::Shape{m.nv}, mx::int32);
    auto cfrc_dof = mx::take(cfrc_arr, dof_bid_arr, 0);
    d.qfrc_bias = mx::sum(mx::multiply(d.cdof, cfrc_dof), -1);

    return d;
}

// ── Vmap-compatible transmission ─────────────────────────────────────────────

Data vmap_transmission(const Model& m, Data d) {
    if (m.nu == 0) return d;
    const auto& c = m.cache;

    // Precomputed moment matrix (constant)
    d.actuator_moment = c.act_moment_const;

    // Vectorized length: qpos[qa_indices] * gear
    auto qpos_gathered = mx::take(d.qpos, c.act_qpos_idxs, 0);  // (nu,)
    d.actuator_length = mx::multiply(qpos_gathered, c.act_gear);

    return d;
}

} // namespace mjmlx
