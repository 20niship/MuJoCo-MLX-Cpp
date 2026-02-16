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

    // Level-parallel backward accumulation of subtree COM
    std::vector<mx::array> sub_pos(nb, mx::array(0.0f));
    std::vector<mx::array> sub_mass(nb, mx::array(0.0f));
    for (int i = 0; i < nb; i++) {
        sub_pos[i] = mx::multiply(vrow(d.xipos, i), mx::slice(m.body_mass, mx::Shape{i}, mx::Shape{i+1}));
        sub_mass[i] = mx::slice(m.body_mass, mx::Shape{i}, mx::Shape{i+1});
    }

    for (int lvl = (int)c.tree_levels.size() - 1; lvl >= 1; lvl--) {
        for (int bid : c.tree_levels[lvl]) {
            int pid = c.body_parentid_vec[bid];
            sub_pos[pid] = mx::add(sub_pos[pid], sub_pos[bid]);
            sub_mass[pid] = mx::add(sub_mass[pid], sub_mass[bid]);
        }
    }

    std::vector<mx::array> subtree_com_vec(nb, mx::array(0.0f));
    for (int i = 0; i < nb; i++) {
        auto safe_m = mx::maximum(sub_mass[i], mx::array(MJMINVAL_V));
        subtree_com_vec[i] = mx::divide(sub_pos[i], safe_m);
    }
    d.subtree_com = mx::stack(subtree_com_vec);

    // Vectorized cinert
    std::vector<mx::array> cinert_vec(nb, mx::array(0.0f));
    for (int i = 0; i < nb; i++) {
        int rid = c.body_rootid_vec[i];
        auto offset = mx::subtract(vrow(d.xipos, i), subtree_com_vec[rid]);
        auto inertia = vrow(m.body_inertia, i);
        auto mi = mx::slice(m.body_mass, mx::Shape{i}, mx::Shape{i+1});

        auto R = mx::reshape(mx::slice(d.ximat, mx::Shape{i,0,0}, mx::Shape{i+1,3,3}), mx::Shape{3,3});
        auto RI = mx::multiply(R, mx::reshape(inertia, mx::Shape{1, 3}));
        auto I_global = mx::matmul(RI, mx::transpose(R));

        auto d2 = mx::sum(mx::multiply(offset, offset));
        auto outer = mx::matmul(mx::reshape(offset, mx::Shape{3,1}), mx::reshape(offset, mx::Shape{1,3}));
        auto pat = mx::multiply(mi, mx::subtract(mx::multiply(d2, mx::eye(3)), outer));
        I_global = mx::add(I_global, pat);

        auto Ixx = mx::slice(mx::flatten(I_global), mx::Shape{0}, mx::Shape{1});
        auto Iyy = mx::slice(mx::flatten(I_global), mx::Shape{4}, mx::Shape{5});
        auto Izz = mx::slice(mx::flatten(I_global), mx::Shape{8}, mx::Shape{9});
        auto Ixy = mx::slice(mx::flatten(I_global), mx::Shape{1}, mx::Shape{2});
        auto Ixz = mx::slice(mx::flatten(I_global), mx::Shape{2}, mx::Shape{3});
        auto Iyz = mx::slice(mx::flatten(I_global), mx::Shape{5}, mx::Shape{6});
        auto pm = mx::multiply(offset, mi);

        cinert_vec[i] = mx::concatenate({
            Ixx, Iyy, Izz, Ixy, Ixz, Iyz,
            mx::slice(pm, mx::Shape{0}, mx::Shape{1}), mx::slice(pm, mx::Shape{1}, mx::Shape{2}), mx::slice(pm, mx::Shape{2}, mx::Shape{3}),
            mi
        }, 0);
    }
    d.cinert = mx::stack(cinert_vec);

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

        auto axis_x_off = batched_cross(axis, offset);
        auto rot_x_off = batched_cross(rot_a, offset);

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

    // Level-parallel backward accumulation
    std::vector<mx::array> crb_body(nb, mx::array(0.0f));
    for (int i = 0; i < nb; i++) {
        crb_body[i] = vrow(d.cinert, i);
    }

    for (int lvl = (int)c.tree_levels.size() - 1; lvl >= 1; lvl--) {
        for (int bid : c.tree_levels[lvl]) {
            int pid = c.body_parentid_vec[bid];
            crb_body[pid] = mx::add(crb_body[pid], crb_body[bid]);
        }
    }
    crb_body[0] = mx::zeros({10});
    d.crb = mx::stack(crb_body);

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

// ── Vmap-compatible GPU Cholesky (column-vectorized, pure MLX) ───────────────

Data vmap_factor_m(const Model& m, Data d) {
    int n = m.nv;
    auto A = mx::add(d.qM, mx::multiply(mx::eye(n), mx::array(1e-6f)));
    auto L = mx::zeros({n, n});

    for (int j = 0; j < n; j++) {
        mx::array s(0.0f);
        if (j > 0) {
            auto row_j = mx::slice(L, mx::Shape{j, 0}, mx::Shape{j + 1, j});
            s = mx::sum(mx::multiply(row_j, row_j), -1);
            s = mx::flatten(s);
        } else {
            s = mx::array(0.0f);
        }
        auto diag_val = mx::sqrt(mx::maximum(
            mx::subtract(mx::slice(mx::flatten(A), mx::Shape{j*n+j}, mx::Shape{j*n+j+1}), s),
            mx::array(1e-6f)));

        // Set L[j,j] = diag_val
        std::vector<float> mask_data(n * n, 0.0f);
        mask_data[j * n + j] = 1.0f;
        auto pos_mask = mx::array(mask_data.data(), mx::Shape{n, n}, mx::float32);
        L = mx::add(L, mx::multiply(pos_mask, diag_val));

        if (j < n - 1) {
            mx::array s2(0.0f);
            if (j > 0) {
                auto below = mx::slice(L, mx::Shape{j+1, 0}, mx::Shape{n, j});
                auto row_j2 = mx::slice(L, mx::Shape{j, 0}, mx::Shape{j + 1, j});
                s2 = mx::sum(mx::multiply(below, mx::broadcast_to(row_j2, mx::Shape{n-j-1, j})), -1);
            } else {
                s2 = mx::zeros(mx::Shape{n - j - 1});
            }

            auto a_col = mx::flatten(mx::slice(A, mx::Shape{j+1, j}, mx::Shape{n, j+1}));
            auto col = mx::divide(mx::subtract(a_col, mx::flatten(s2)), mx::flatten(diag_val));

            // Set L[j+1:n, j] = col
            std::vector<float> col_mask_data(n * n, 0.0f);
            for (int i = j + 1; i < n; i++) col_mask_data[i * n + j] = 1.0f;
            auto col_pos = mx::array(col_mask_data.data(), mx::Shape{n, n}, mx::float32);

            // Expand col to (n, n) with values in the right positions
            std::vector<float> col_expand_data(n * n, 0.0f);
            // We need a smarter approach: scatter the column values
            auto col_2d = mx::zeros({n, n});
            for (int i = j + 1; i < n; i++) {
                std::vector<float> m2(n * n, 0.0f);
                m2[i * n + j] = 1.0f;
                auto single = mx::array(m2.data(), mx::Shape{n, n}, mx::float32);
                col_2d = mx::add(col_2d, mx::multiply(single,
                    mx::slice(col, mx::Shape{i - j - 1}, mx::Shape{i - j})));
            }
            L = mx::add(L, col_2d);
        }
    }

    d.qLD = L;
    return d;
}

// ── Vmap-compatible triangular solve ─────────────────────────────────────────

mx::array vmap_solve_m(const Model& m, const Data& d, const mx::array& rhs) {
    int n = m.nv;
    auto L = d.qLD;
    auto LT = mx::transpose(L);

    // Forward substitution: L @ y = rhs
    auto y = mx::zeros({n});
    for (int i = 0; i < n; i++) {
        mx::array s(0.0f);
        if (i > 0) {
            auto L_row = mx::flatten(mx::slice(L, mx::Shape{i, 0}, mx::Shape{i + 1, i}));
            auto y_part = mx::slice(y, mx::Shape{0}, mx::Shape{i});
            s = mx::sum(mx::multiply(L_row, y_part));
        } else {
            s = mx::array(0.0f);
        }
        auto Li = mx::flatten(mx::slice(L, mx::Shape{i, i}, mx::Shape{i+1, i+1}));
        auto yi = mx::divide(mx::subtract(mx::slice(rhs, mx::Shape{i}, mx::Shape{i+1}), s), Li);

        // Set y[i] = yi
        std::vector<float> mask(n, 0.0f); mask[i] = 1.0f;
        auto mi = mx::array(mask.data(), mx::Shape{n}, mx::float32);
        y = mx::add(y, mx::multiply(mi, yi));
    }

    // Backward substitution: L^T @ x = y
    auto x = mx::zeros({n});
    for (int i = n - 1; i >= 0; i--) {
        mx::array s(0.0f);
        if (i < n - 1) {
            auto LT_row = mx::flatten(mx::slice(LT, mx::Shape{i, i+1}, mx::Shape{i+1, n}));
            auto x_part = mx::slice(x, mx::Shape{i+1}, mx::Shape{n});
            s = mx::sum(mx::multiply(LT_row, x_part));
        } else {
            s = mx::array(0.0f);
        }
        auto LTi = mx::flatten(mx::slice(LT, mx::Shape{i, i}, mx::Shape{i+1, i+1}));
        auto xi = mx::divide(mx::subtract(mx::slice(y, mx::Shape{i}, mx::Shape{i+1}), s), LTi);

        std::vector<float> mask(n, 0.0f); mask[i] = 1.0f;
        auto mi = mx::array(mask.data(), mx::Shape{n}, mx::float32);
        x = mx::add(x, mx::multiply(mi, xi));
    }

    return x;
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

    // Level-parallel backward force accumulation
    std::vector<mx::array> cfrc(m.nbody, mx::array(0.0f));
    for (int i = 0; i < m.nbody; i++) cfrc[i] = vrow(loc_cfrc, i);

    for (int lvl = (int)c.tree_levels.size() - 1; lvl >= 1; lvl--) {
        for (int bid : c.tree_levels[lvl]) {
            int pid = c.body_parentid_vec[bid];
            cfrc[pid] = mx::add(cfrc[pid], cfrc[bid]);
        }
    }

    // Vectorized joint-space projection
    auto cfrc_arr = mx::stack(cfrc);
    auto dof_bid_arr = mx::array(c.dof_bodyid_vec.data(), mx::Shape{m.nv}, mx::int32);
    auto cfrc_dof = mx::take(cfrc_arr, dof_bid_arr, 0);
    d.qfrc_bias = mx::sum(mx::multiply(d.cdof, cfrc_dof), -1);

    return d;
}

// ── Vmap-compatible transmission ─────────────────────────────────────────────

Data vmap_transmission(const Model& m, Data d) {
    if (m.nu == 0) return d;
    const auto& c = m.cache;

    auto act_length = mx::zeros({m.nu});
    auto act_moment = mx::zeros({m.nu, m.nv});

    // Use cache.dof_info and actuator properties
    // For HINGE/SLIDE actuators: length = qpos[qa] * gear[0], moment[ai, da] = gear[0]
    for (auto& ai_info : c.actuator_info) {
        int ai = ai_info.act_idx;
        int da = ai_info.dof_adr;
        float g0 = ai_info.gain;

        // Set moment[ai, da] = g0
        std::vector<float> m_mask(m.nu * m.nv, 0.0f);
        m_mask[ai * m.nv + da] = 1.0f;
        auto mask = mx::array(m_mask.data(), mx::Shape{m.nu, m.nv}, mx::float32);
        act_moment = mx::add(act_moment, mx::multiply(mask, mx::array(g0)));
    }

    // Lengths: for each simple actuator, qpos[qa] * gear
    for (auto& ai_info : c.actuator_info) {
        int ai = ai_info.act_idx;
        int ji = ai_info.jnt_idx;
        float g0 = ai_info.gain;
        int qa = c.dof_info[0].qpos_adr; // Need actual qa for this joint
        // Find the qa from cache
        for (auto& di : c.dof_info) {
            if (di.jnt_idx == ji) {
                qa = di.qpos_adr;
                break;
            }
        }

        auto qp = mx::slice(d.qpos, mx::Shape{qa}, mx::Shape{qa + 1});
        auto len_val = mx::multiply(qp, mx::array(g0));

        std::vector<float> l_mask(m.nu, 0.0f);
        l_mask[ai] = 1.0f;
        auto lm = mx::array(l_mask.data(), mx::Shape{m.nu}, mx::float32);
        act_length = mx::add(act_length, mx::multiply(lm, mx::flatten(len_val)));
    }

    d.actuator_length = act_length;
    d.actuator_moment = act_moment;
    return d;
}

} // namespace mjmlx
