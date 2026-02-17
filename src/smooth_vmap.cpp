// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// Vmap-compatible smooth dynamics: COM, CRB, mass matrix factorization,
// velocity-dependent, RNE, transmission.
// NO eval(), NO data<>(), NO CPU sync. Pure MLX graph building.

#include "internal.h"

namespace mjmlx {

// ── GPU-native Cholesky factorization (vmap-compatible) ──────────────────────
// Column-vectorized: n iterations for n×n matrix.
// Maintains single L matrix, reads submatrices directly via slice.
// All ops are pure MLX graph nodes (no eval, no data<>) → works inside mx::vmap.
// Matches Python gpu_cholesky() from gpu_linalg.py.
//
// DECISION: Uses one-hot mask + multiply to scatter each column into L, because
// MLX arrays are immutable -- there is no in-place index assignment (L[i,j] = val)
// in the computation graph. Each column is built as a full (n,1) vector and added
// via outer product with a one-hot row selector. This creates ~35 graph nodes per
// column (n columns total). For humanoid (nv=27), the graph has ~945 Cholesky nodes.
//
// DECISION: The 1e-6 floor on the diagonal prevents negative sqrt from numerical
// noise. This matches Python's gpu_cholesky implementation.
mx::array cholesky_gpu(const mx::array& A, int n) {
    auto L = mx::zeros_like(A);  // (n, n)

    for (int j = 0; j < n; j++) {
        // Diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,0:j]^2))
        auto s = (j > 0)
            ? mx::sum(mx::square(mx::slice(L, {j, 0}, {j+1, j})))
            : mx::array(0.0f);
        auto diag = mx::sqrt(mx::maximum(
            mx::subtract(mx::flatten(mx::slice(A, {j, j}, {j+1, j+1})),
                          mx::reshape(s, {1})),
            mx::array({1e-6f})));  // (1,)

        // Off-diagonal column: L[j+1:n, j] = (A[j+1:n, j] - L[j+1:n,0:j] @ L[j,0:j]^T) / diag
        mx::array col = mx::zeros({0, 1});
        if (j < n - 1) {
            auto Acol = mx::slice(A, {j+1, j}, {n, j+1});  // (n-j-1, 1)
            if (j > 0) {
                auto L_below = mx::slice(L, {j+1, 0}, {n, j});  // (n-j-1, j)
                auto L_jrow = mx::slice(L, {j, 0}, {j+1, j});   // (1, j)
                auto dot = mx::matmul(L_below, mx::transpose(L_jrow));  // (n-j-1, 1)
                Acol = mx::subtract(Acol, dot);
            }
            col = mx::divide(Acol, mx::reshape(diag, {1, 1}));  // (n-j-1, 1)
        }

        // Build full column vector and one-hot row mask, add to L
        auto diag_2d = mx::reshape(diag, {1, 1});
        mx::array full_col = (j < n - 1)
            ? mx::concatenate({mx::zeros({j, 1}), diag_2d, col}, 0)     // (n, 1)
            : mx::concatenate({mx::zeros({j, 1}), diag_2d}, 0);          // (n, 1)

        // One-hot row vector: 1 at position j
        mx::array one_hot = (j == 0)
            ? mx::concatenate({mx::ones({1, 1}), mx::zeros({1, n - 1})}, 1)
            : (j == n - 1)
                ? mx::concatenate({mx::zeros({1, n - 1}), mx::ones({1, 1})}, 1)
                : mx::concatenate({mx::zeros({1, j}), mx::ones({1, 1}), mx::zeros({1, n - j - 1})}, 1);

        L = mx::add(L, mx::multiply(full_col, one_hot));  // scatter column j into L
    }

    return L;
}

// ── GPU-native forward substitution: L y = b (vmap-compatible) ───────────────
static mx::array solve_triangular_lower(const mx::array& L, const mx::array& b, int n) {
    auto y = mx::zeros({n});
    for (int i = 0; i < n; i++) {
        auto bi = mx::slice(b, {i}, {i+1});  // (1,)
        if (i > 0) {
            auto L_row = mx::flatten(mx::slice(L, {i, 0}, {i+1, i}));  // (i,)
            auto y_prev = mx::slice(y, {0}, {i});  // (i,)
            bi = mx::subtract(bi, mx::reshape(mx::sum(mx::multiply(L_row, y_prev)), {1}));
        }
        auto Lii = mx::flatten(mx::slice(L, {i, i}, {i+1, i+1}));  // (1,)
        auto yi = mx::divide(bi, mx::maximum(Lii, mx::array({1e-10f})));
        // Update y at position i
        y = mx::concatenate({mx::slice(y, {0}, {i}), yi,
                             (i < n-1) ? mx::slice(y, {i+1}, {n}) : mx::zeros({0})}, 0);
    }
    return y;
}

// ── GPU-native backward substitution: L^T x = y (vmap-compatible) ────────────
static mx::array solve_triangular_upper(const mx::array& LT, const mx::array& y, int n) {
    auto x = mx::zeros({n});
    for (int i = n - 1; i >= 0; i--) {
        auto yi = mx::slice(y, {i}, {i+1});  // (1,)
        if (i < n - 1) {
            auto U_row = mx::flatten(mx::slice(LT, {i, i+1}, {i+1, n}));  // (n-i-1,)
            auto x_below = mx::slice(x, {i+1}, {n});  // (n-i-1,)
            yi = mx::subtract(yi, mx::reshape(mx::sum(mx::multiply(U_row, x_below)), {1}));
        }
        auto Uii = mx::flatten(mx::slice(LT, {i, i}, {i+1, i+1}));
        auto xi = mx::divide(yi, mx::maximum(Uii, mx::array({1e-10f})));
        x = mx::concatenate({(i > 0) ? mx::slice(x, {0}, {i}) : mx::zeros({0}), xi,
                             (i < n-1) ? mx::slice(x, {i+1}, {n}) : mx::zeros({0})}, 0);
    }
    return x;
}

// ── GPU-native Cholesky solve: L L^T x = b (vmap-compatible) ─────────────────
mx::array cholesky_solve_gpu(const mx::array& L, const mx::array& b, int n) {
    auto y = solve_triangular_lower(L, b, n);
    auto x = solve_triangular_upper(mx::transpose(L), y, n);
    return x;
}

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
// Uses column-vectorized Cholesky decomposition with pure MLX ops.
// All operations are GPU-native (no CPU sync) → vmap-compatible.

Data vmap_factor_m(const Model& m, Data d) {
    int n = m.nv;
    auto A = mx::add(d.qM, mx::multiply(mx::eye(n), mx::array(1e-6f)));
    d.qLD = A;  // store regularized M (used by Euler kernel via Metal)

    // Cholesky factorization: A = L L^T
    // DECISION: d.qM_inv stores the Cholesky factor L, NOT the actual inverse M^{-1}.
    // The name is a legacy from when Neumann-series approximation was used. The field
    // is consumed by vmap_solve_m() which calls cholesky_solve_gpu(L, rhs, n).
    d.qM_inv = cholesky_gpu(A, n);

    return d;
}

// ── Vmap-compatible solve M x = rhs using Cholesky factorization ────────────

mx::array vmap_solve_m(const Model& m, const Data& d, const mx::array& rhs) {
    // d.qM_inv holds the Cholesky factor L (not the actual inverse)
    return cholesky_solve_gpu(d.qM_inv, rhs, m.nv);
}

// ── Vmap-compatible COM velocity ─────────────────────────────────────────────
//
// DECISION: Per-body sequential loops are INTENTIONAL here. An attempt was made to
// vectorize this using level-parallel scatter matrices (batching all bodies in a tree
// level into a single matmul). This REGRESSED performance from 198K to 47K SPS because:
//   1. Scatter-add through the full (nb, 6) tensor creates long data dependency chains
//      that prevent the compiler from parallelizing operations.
//   2. The original per-body approach stores results in a std::vector<mx::array> where
//      each body's computation is an independent graph branch. The final mx::stack()
//      is the only convergence point, giving mx::compile maximum freedom to fuse and
//      schedule ops in parallel.
//   3. Small (6,)-element per-body ops are already very efficient on GPU after fusion.
// The backward accumulation in vmap_com_pos and vmap_crb uses scatter matrices
// successfully because scatter-add is commutative and each child's contribution to
// its parent is independent. Forward propagation (parent → child) lacks this property.

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
// DECISION: Forward pass (cacc accumulation) uses per-body loops for the same reason
// as vmap_com_vel -- scatter matrices regressed performance. The backward pass (force
// accumulation) DOES use scatter matrices because it is a commutative scatter-add.

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

// ── Vmap-compatible tendon ────────────────────────────────────────────────────

Data vmap_tendon(const Model& m, Data d) {
    if (m.ntendon == 0) {
        d.ten_length = mx::zeros({0});
        d.ten_velocity = mx::zeros({0});
        d.ten_J = mx::zeros({0, m.nv});
        return d;
    }

    // Pre-compute the tendon Jacobian as a constant matrix (for fixed tendons).
    // For each wrap object of type JOINT: ten_J[tendon_id, jnt_dofadr[wrap_objid]] = wrap_prm
    // This is the same as scalar path since the Jacobian is constant.
    mx::eval(m.tendon_adr); mx::eval(m.tendon_num);
    mx::eval(m.wrap_type); mx::eval(m.wrap_objid); mx::eval(m.wrap_prm);
    mx::eval(m.jnt_qposadr); mx::eval(m.jnt_dofadr);

    auto ten_adr = m.tendon_adr.data<int>();
    auto ten_num = m.tendon_num.data<int>();
    auto wtype = m.wrap_type.data<int>();
    auto wobjid = m.wrap_objid.data<int>();
    auto wprm = m.wrap_prm.data<float>();
    auto jqpa = m.jnt_qposadr.data<int>();
    auto jda = m.jnt_dofadr.data<int>();

    // Build constant Jacobian (same for all envs, fixed tendon coefficients)
    std::vector<float> ten_j(m.ntendon * m.nv, 0.0f);
    std::vector<int> qpos_indices;
    std::vector<float> qpos_coefs;
    std::vector<int> tendon_ids;

    for (int t = 0; t < m.ntendon; t++) {
        int adr = ten_adr[t];
        int num = ten_num[t];
        for (int w = adr; w < adr + num; w++) {
            if (wtype[w] != 1) continue;  // mjWRAP_JOINT
            int jnt = wobjid[w];
            float coef = wprm[w];
            int da = jda[jnt];
            ten_j[t * m.nv + da] += coef;
            qpos_indices.push_back(jqpa[jnt]);
            qpos_coefs.push_back(coef);
            tendon_ids.push_back(t);
        }
    }

    d.ten_J = mx::array(ten_j.data(), {m.ntendon, m.nv}, mx::float32);

    // ten_length = ten_J @ qpos (but only for scalar qpos entries that map to DOFs)
    // For fixed tendons with HINGE/SLIDE joints, we use: length = sum(coef * qpos[qa])
    // We do this via ten_J @ qpos_subset, but since qpos may have quaternions (nq != nv),
    // we compute it via gathering and dotting.
    // Actually, simplest vmap-compatible: just gather and sum.
    // Build ten_length from qpos using the gathered indices.
    if (!qpos_indices.empty()) {
        auto idx = mx::array(qpos_indices.data(), {(int)qpos_indices.size()}, mx::int32);
        auto coefs = mx::array(qpos_coefs.data(), {(int)qpos_coefs.size()}, mx::float32);
        auto tids = mx::array(tendon_ids.data(), {(int)tendon_ids.size()}, mx::int32);

        // Gather qpos values at the relevant indices
        auto qvals = mx::take(d.qpos, idx);
        auto products = mx::multiply(coefs, qvals);

        // Scatter-add into ten_length using segment_sum equivalent
        // Use mx::zeros + scatter_add (via index_put)
        // Since MLX doesn't have segment_sum directly, use a loop or scatter
        auto ten_length = mx::zeros({m.ntendon});
        // For vmap compatibility, build a one-hot and matmul
        std::vector<float> scatter_mat(qpos_indices.size() * m.ntendon, 0.0f);
        for (size_t i = 0; i < tendon_ids.size(); i++) {
            scatter_mat[i * m.ntendon + tendon_ids[i]] = 1.0f;
        }
        auto smat = mx::array(scatter_mat.data(),
            {(int)qpos_indices.size(), m.ntendon}, mx::float32);
        d.ten_length = mx::flatten(mx::matmul(mx::reshape(products, {1, (int)qpos_indices.size()}), smat));
    } else {
        d.ten_length = mx::zeros({m.ntendon});
    }

    // ten_velocity = ten_J @ qvel
    d.ten_velocity = mx::flatten(mx::matmul(d.ten_J, mx::reshape(d.qvel, {m.nv, 1})));

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
