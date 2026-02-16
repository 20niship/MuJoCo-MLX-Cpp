// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// Vmap-compatible constraint solver.
// Two modes (matches Python solver.py):
//   Newton: Form H = M + J^T*D*J, Cholesky-solve for exact search direction.
//           Converges in 1-2 iterations. Default for humanoid.
//   CG:     M^{-1} preconditioned gradient descent with Polak-Ribière updates.
//           Needs 4-10 iterations.
// Both use warmstart + 5-alpha vectorized line search + active set.
// NO eval(), NO data<>(), NO CPU sync. Pure MLX graph building.

#include "internal.h"

namespace mjmlx {

static constexpr float MJMINVAL_SV = 1e-8f;

// Dense M @ v using pure MLX ops (no eval)
static mx::array mul_m(const mx::array& qM, int nv, const mx::array& vec) {
    return mx::flatten(mx::matmul(qM, mx::reshape(vec, {nv, 1})));
}

// Active set: equality constraints always active, inequality when Jaref < 0
static mx::array get_active(const mx::array& Jaref, int ne, int nf, int nefc) {
    auto active = mx::astype(mx::less(Jaref, mx::array(0.0f)), mx::float32);
    if (ne + nf > 0) {
        auto always = mx::ones({ne + nf});
        auto rest = mx::slice(active, {ne + nf}, {nefc});
        active = mx::concatenate({always, rest}, 0);
    }
    return active;
}

Data vmap_solve(const Model& m, Data d) {
    int nefc = d.efc_J.shape(0);
    if (nefc == 0) {
        d.qacc = d.qacc_smooth;
        d.qfrc_constraint = mx::zeros({m.nv});
        return d;
    }

    int nv = m.nv;
    bool use_newton = (m.opt.solver == SolverType::NEWTON);

    // ── Warmstart ────────────────────────────────────────────────────────────
    auto qacc = d.qacc_smooth;
    if (!(m.opt.disableflags & DisableBit::WARMSTART) && d.qacc_warmstart.size() > 0) {
        auto warm_Jaref = mx::subtract(
            mx::flatten(mx::matmul(d.efc_J, mx::reshape(d.qacc_warmstart, {nv, 1}))),
            d.efc_aref);
        auto warm_active = get_active(warm_Jaref, d.ne, d.nf, nefc);
        auto warm_cost = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(warm_Jaref, warm_Jaref)), warm_active)));

        auto smooth_Jaref = mx::subtract(
            mx::flatten(mx::matmul(d.efc_J, mx::reshape(d.qacc_smooth, {nv, 1}))),
            d.efc_aref);
        auto smooth_active = get_active(smooth_Jaref, d.ne, d.nf, nefc);
        auto smooth_cost = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(smooth_Jaref, smooth_Jaref)), smooth_active)));

        // Pick whichever has lower cost
        auto use_warm = mx::less(warm_cost, smooth_cost);
        qacc = mx::where(use_warm, d.qacc_warmstart, d.qacc_smooth);
    }

    // ── Initialize solver state ──────────────────────────────────────────────
    auto Jaref = mx::subtract(
        mx::flatten(mx::matmul(d.efc_J, mx::reshape(qacc, {nv, 1}))),
        d.efc_aref);
    auto Ma = mul_m(d.qM, nv, qacc);
    auto active = get_active(Jaref, d.ne, d.nf, nefc);

    auto efc_force = mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), active);
    auto qfrc_constraint = mx::flatten(mx::matmul(
        mx::transpose(d.efc_J), mx::reshape(efc_force, {nefc, 1})));

    auto cost_c = mx::multiply(mx::array(0.5f),
        mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jaref, Jaref)), active)));
    auto gauss = mx::multiply(mx::array(0.5f),
        mx::sum(mx::multiply(mx::subtract(Ma, d.qfrc_smooth),
                              mx::subtract(qacc, d.qacc_smooth))));
    auto total_cost = mx::add(cost_c, gauss);

    auto grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));

    // For CG mode: initialize preconditioned direction
    auto Mgrad = mx::zeros({nv});
    auto search = mx::zeros({nv});
    if (!use_newton) {
        Mgrad = vmap_solve_m(m, d, grad);
        search = mx::negative(Mgrad);
    }

    // ── Solver iterations ────────────────────────────────────────────────────
    for (int iter = 0; iter < m.opt.iterations; iter++) {

        if (use_newton) {
            // ── Direct Newton: H = M + J_a^T * diag(D_a) * J_a ──────────
            auto D_active = mx::multiply(d.efc_D, active);
            auto sqrt_D = mx::sqrt(mx::maximum(D_active, mx::array(0.0f)));
            auto J_scaled = mx::multiply(d.efc_J, mx::reshape(sqrt_D, {nefc, 1}));
            auto JtDJ = mx::matmul(mx::transpose(J_scaled), J_scaled);
            auto H = mx::add(d.qM, JtDJ);
            // Regularize for numerical stability
            H = mx::add(H, mx::multiply(mx::eye(nv), mx::array(MJMINVAL_SV)));

            // Cholesky-solve for exact Newton direction
            auto L = cholesky_gpu(H, nv);
            search = mx::negative(cholesky_solve_gpu(L, grad, nv));
        }

        // ── Line search (same for Newton and CG) ────────────────────────
        auto Mv = mul_m(d.qM, nv, search);
        auto Jv = mx::flatten(mx::matmul(d.efc_J, mx::reshape(search, {nv, 1})));

        auto quad_gauss = mx::multiply(mx::array(0.5f), mx::sum(mx::multiply(search, Mv)));
        auto linear_gauss = mx::sum(mx::multiply(search, mx::subtract(Ma, d.qfrc_smooth)));
        auto quad_con = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jv, Jv)), active)));
        auto linear_con = mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jv, Jaref)), active));

        auto denom = mx::multiply(mx::array(2.0f), mx::add(quad_gauss, quad_con));
        auto alpha_n = mx::clip(
            mx::negative(mx::divide(mx::add(linear_gauss, linear_con),
                                     mx::maximum(denom, mx::array(MJMINVAL_SV)))),
            mx::array(-2.0f), mx::array(2.0f));

        // 5-alpha vectorized line search
        auto alphas = mx::stack({alpha_n,
            mx::multiply(alpha_n, mx::array(0.5f)),
            mx::multiply(alpha_n, mx::array(0.1f)),
            mx::array(0.01f), mx::array(0.001f)});

        auto x_all = mx::add(mx::reshape(Jaref, {1, nefc}),
            mx::multiply(mx::reshape(alphas, {5, 1}), mx::reshape(Jv, {1, nefc})));
        auto act_all = mx::astype(mx::less(x_all, mx::array(0.0f)), mx::float32);
        // Force equality constraints always active
        if (d.ne + d.nf > 0) {
            auto eq_mask = mx::concatenate({mx::ones({d.ne + d.nf}),
                                             mx::zeros({nefc - d.ne - d.nf})}, 0);
            eq_mask = mx::reshape(eq_mask, {1, nefc});
            act_all = mx::maximum(act_all, eq_mask);
        }

        auto c_all = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(mx::reshape(d.efc_D, {1, nefc}),
                                               mx::multiply(x_all, x_all)), act_all), 1));

        auto sMa = mx::sum(mx::multiply(search, Ma));
        auto sFs = mx::sum(mx::multiply(search, d.qfrc_smooth));
        auto sMv = mx::sum(mx::multiply(search, Mv));
        auto g_all = mx::add(gauss, mx::add(
            mx::multiply(alphas, mx::subtract(sMa, sFs)),
            mx::multiply(mx::array(0.5f), mx::multiply(mx::multiply(alphas, alphas), sMv))));
        auto total_all = mx::add(c_all, g_all);

        // Pick best alpha (including alpha=0 = no step)
        auto all_costs = mx::concatenate({mx::reshape(total_cost, {1}), total_all}, 0);
        auto all_alphas = mx::concatenate({mx::zeros({1}), alphas}, 0);
        auto best_idx = mx::argmin(all_costs);
        auto best_alpha = mx::take(all_alphas, best_idx);

        // Apply step
        qacc = mx::add(qacc, mx::multiply(search, best_alpha));
        Ma = mx::add(Ma, mx::multiply(Mv, best_alpha));
        Jaref = mx::add(Jaref, mx::multiply(Jv, best_alpha));

        // Update active set and forces
        active = get_active(Jaref, d.ne, d.nf, nefc);
        efc_force = mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), active);
        qfrc_constraint = mx::flatten(mx::matmul(
            mx::transpose(d.efc_J), mx::reshape(efc_force, {nefc, 1})));

        cost_c = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jaref, Jaref)), active)));
        gauss = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::subtract(Ma, d.qfrc_smooth),
                                  mx::subtract(qacc, d.qacc_smooth))));
        total_cost = mx::add(cost_c, gauss);

        // Update gradient and CG direction
        if (!use_newton) {
            auto prev_grad = grad;     // save BEFORE updating grad
            auto prev_Mgrad = Mgrad;
            grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));
            Mgrad = vmap_solve_m(m, d, grad);
            // Polak-Ribière CG update
            auto beta_num = mx::sum(mx::multiply(grad, mx::subtract(Mgrad, prev_Mgrad)));
            auto beta_den = mx::maximum(mx::array(MJMINVAL_SV),
                                         mx::sum(mx::multiply(prev_grad, prev_Mgrad)));
            auto beta = mx::maximum(mx::divide(beta_num, beta_den), mx::array(0.0f));
            search = mx::add(mx::negative(Mgrad), mx::multiply(beta, search));
        } else {
            grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));
        }
    }

    d.qfrc_constraint = qfrc_constraint;
    d.qacc = qacc;
    d.efc_force = efc_force;
    return d;
}

} // namespace mjmlx
