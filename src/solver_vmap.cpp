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

// Apply friction clamping in vmap-compatible way (pure MLX ops).
// Returns: (clamped_active, floss_force)
// - clamped_active: active with linear zone rows set to 0
// - floss_force: (nefc,) with +/-f for linear zone rows, 0 elsewhere
static std::pair<mx::array, mx::array> vmap_friction_clamp(
    const mx::array& Jaref, const mx::array& active, const mx::array& efc_D,
    const mx::array& efc_frictionloss, int nf, int nefc)
{
    if (nf <= 0 || efc_frictionloss.size() == 0) {
        return {active, mx::zeros({nefc})};
    }

    auto fl = efc_frictionloss;
    auto has_fl = mx::greater(fl, mx::array(0.0f));  // (nefc,)

    // r = 1/D (safe division)
    auto r_safe = mx::divide(mx::array(1.0f), mx::maximum(efc_D, mx::array(1e-15f)));
    auto rf = mx::multiply(r_safe, fl);  // threshold

    auto linear_neg = mx::logical_and(mx::less_equal(Jaref, mx::negative(rf)), has_fl);
    auto linear_pos = mx::logical_and(mx::greater_equal(Jaref, rf), has_fl);
    auto in_linear = mx::logical_or(linear_neg, linear_pos);

    // Remove linear zone rows from quadratic active set
    auto clamped_active = mx::multiply(active,
        mx::subtract(mx::array(1.0f), mx::astype(in_linear, mx::float32)));

    // Fixed forces: +f for neg zone, -f for pos zone
    auto floss_force = mx::add(
        mx::multiply(mx::astype(linear_neg, mx::float32), fl),
        mx::multiply(mx::astype(linear_pos, mx::float32), mx::negative(fl)));

    return {clamped_active, floss_force};
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

        auto use_warm = mx::less(warm_cost, smooth_cost);
        qacc = mx::where(use_warm, d.qacc_warmstart, d.qacc_smooth);
    }

    // ── Initialize solver state ──────────────────────────────────────────────
    auto Jaref = mx::subtract(
        mx::flatten(mx::matmul(d.efc_J, mx::reshape(qacc, {nv, 1}))),
        d.efc_aref);
    auto Ma = mul_m(d.qM, nv, qacc);
    auto active = get_active(Jaref, d.ne, d.nf, nefc);

    // Apply friction clamping
    auto [clamped_active, floss_force] = vmap_friction_clamp(
        Jaref, active, d.efc_D, d.efc_frictionloss, d.nf, nefc);

    auto efc_force = mx::add(
        mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), clamped_active),
        floss_force);
    auto qfrc_constraint = mx::flatten(mx::matmul(
        mx::transpose(d.efc_J), mx::reshape(efc_force, {nefc, 1})));

    auto cost_c = mx::multiply(mx::array(0.5f),
        mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jaref, Jaref)), clamped_active)));
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
            auto D_active = mx::multiply(d.efc_D, clamped_active);
            auto sqrt_D = mx::sqrt(mx::maximum(D_active, mx::array(0.0f)));
            auto J_scaled = mx::multiply(d.efc_J, mx::reshape(sqrt_D, {nefc, 1}));
            auto JtDJ = mx::matmul(mx::transpose(J_scaled), J_scaled);
            auto H = mx::add(d.qM, JtDJ);
            H = mx::add(H, mx::multiply(mx::eye(nv), mx::array(MJMINVAL_SV)));

            auto L = cholesky_gpu(H, nv);
            search = mx::negative(cholesky_solve_gpu(L, grad, nv));
        }

        // ── Line search (same for Newton and CG) ────────────────────────
        auto Mv = mul_m(d.qM, nv, search);
        auto Jv = mx::flatten(mx::matmul(d.efc_J, mx::reshape(search, {nv, 1})));

        auto quad_gauss = mx::multiply(mx::array(0.5f), mx::sum(mx::multiply(search, Mv)));
        auto linear_gauss = mx::sum(mx::multiply(search, mx::subtract(Ma, d.qfrc_smooth)));
        auto quad_con = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jv, Jv)), clamped_active)));
        auto linear_con = mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jv, Jaref)), clamped_active));

        // Friction linear zone derivative contribution:
        // neg zone: d/dalpha = -f * Jv;  pos zone: d/dalpha = f * Jv
        auto linear_floss = mx::array(0.0f);
        if (d.nf > 0 && d.efc_frictionloss.size() > 0) {
            auto fl = d.efc_frictionloss;
            auto has_fl = mx::greater(fl, mx::array(0.0f));
            auto r_safe = mx::divide(mx::array(1.0f), mx::maximum(d.efc_D, mx::array(1e-15f)));
            auto rf = mx::multiply(r_safe, fl);
            auto in_neg = mx::astype(mx::logical_and(mx::less_equal(Jaref, mx::negative(rf)), has_fl), mx::float32);
            auto in_pos = mx::astype(mx::logical_and(mx::greater_equal(Jaref, rf), has_fl), mx::float32);
            // d/dalpha for neg zone = -f*Jv, for pos zone = f*Jv
            linear_floss = mx::sum(mx::multiply(mx::subtract(in_pos, in_neg), mx::multiply(fl, Jv)));
        }

        auto denom = mx::multiply(mx::array(2.0f), mx::add(quad_gauss, quad_con));
        auto linear_total = mx::add(mx::add(linear_gauss, linear_con), linear_floss);
        auto alpha_n = mx::clip(
            mx::negative(mx::divide(linear_total,
                                     mx::maximum(denom, mx::array(MJMINVAL_SV)))),
            mx::array(-2.0f), mx::array(2.0f));

        // 5-alpha vectorized line search
        auto alphas = mx::stack({alpha_n,
            mx::multiply(alpha_n, mx::array(0.5f)),
            mx::multiply(alpha_n, mx::array(0.1f)),
            mx::array(0.01f), mx::array(0.001f)});

        auto x_all = mx::add(mx::reshape(Jaref, {1, nefc}),
            mx::multiply(mx::reshape(alphas, {5, 1}), mx::reshape(Jv, {1, nefc})));

        // Base active set for all alphas
        auto act_all = mx::astype(mx::less(x_all, mx::array(0.0f)), mx::float32);
        if (d.ne + d.nf > 0) {
            auto eq_mask = mx::concatenate({mx::ones({d.ne + d.nf}),
                                             mx::zeros({nefc - d.ne - d.nf})}, 0);
            eq_mask = mx::reshape(eq_mask, {1, nefc});
            act_all = mx::maximum(act_all, eq_mask);
        }

        // Apply friction clamping to linesearch points
        if (d.nf > 0 && d.efc_frictionloss.size() > 0) {
            auto fl = mx::reshape(d.efc_frictionloss, {1, nefc});
            auto has_fl = mx::greater(fl, mx::array(0.0f));
            auto r_safe = mx::divide(mx::array(1.0f),
                mx::maximum(mx::reshape(d.efc_D, {1, nefc}), mx::array(1e-15f)));
            auto rf = mx::multiply(r_safe, fl);
            auto linear_neg = mx::logical_and(mx::less_equal(x_all, mx::negative(rf)), has_fl);
            auto linear_pos = mx::logical_and(mx::greater_equal(x_all, rf), has_fl);
            auto in_linear = mx::logical_or(linear_neg, linear_pos);

            // Remove linear zone rows from quadratic cost
            act_all = mx::multiply(act_all,
                mx::subtract(mx::array(1.0f), mx::astype(in_linear, mx::float32)));

            // Linear zone cost: f*(-0.5*rf - x) for neg, (-f)*(0.5*rf - x) for pos
            auto floss_cost_neg = mx::multiply(mx::astype(linear_neg, mx::float32),
                mx::multiply(fl, mx::add(mx::multiply(mx::array(-0.5f), rf), mx::negative(x_all))));
            auto floss_cost_pos = mx::multiply(mx::astype(linear_pos, mx::float32),
                mx::multiply(mx::negative(fl), mx::subtract(mx::multiply(mx::array(0.5f), rf), x_all)));
            auto floss_cost_all = mx::sum(mx::add(floss_cost_neg, floss_cost_pos), 1);  // (5,)

            // Quadratic cost + friction linear cost
            auto c_all = mx::add(
                mx::multiply(mx::array(0.5f),
                    mx::sum(mx::multiply(mx::multiply(mx::reshape(d.efc_D, {1, nefc}),
                                                       mx::multiply(x_all, x_all)), act_all), 1)),
                floss_cost_all);

            auto sMa = mx::sum(mx::multiply(search, Ma));
            auto sFs = mx::sum(mx::multiply(search, d.qfrc_smooth));
            auto sMv = mx::sum(mx::multiply(search, Mv));
            auto g_all = mx::add(gauss, mx::add(
                mx::multiply(alphas, mx::subtract(sMa, sFs)),
                mx::multiply(mx::array(0.5f), mx::multiply(mx::multiply(alphas, alphas), sMv))));
            auto total_all = mx::add(c_all, g_all);

            auto all_costs = mx::concatenate({mx::reshape(total_cost, {1}), total_all}, 0);
            auto all_alphas = mx::concatenate({mx::zeros({1}), alphas}, 0);
            auto best_idx = mx::argmin(all_costs);
            auto best_alpha = mx::take(all_alphas, best_idx);

            qacc = mx::add(qacc, mx::multiply(search, best_alpha));
            Ma = mx::add(Ma, mx::multiply(Mv, best_alpha));
            Jaref = mx::add(Jaref, mx::multiply(Jv, best_alpha));
        } else {
            // No friction — original linesearch
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

            auto all_costs = mx::concatenate({mx::reshape(total_cost, {1}), total_all}, 0);
            auto all_alphas = mx::concatenate({mx::zeros({1}), alphas}, 0);
            auto best_idx = mx::argmin(all_costs);
            auto best_alpha = mx::take(all_alphas, best_idx);

            qacc = mx::add(qacc, mx::multiply(search, best_alpha));
            Ma = mx::add(Ma, mx::multiply(Mv, best_alpha));
            Jaref = mx::add(Jaref, mx::multiply(Jv, best_alpha));
        }

        // Update active set and forces with friction clamping
        active = get_active(Jaref, d.ne, d.nf, nefc);
        auto [ca, ff] = vmap_friction_clamp(
            Jaref, active, d.efc_D, d.efc_frictionloss, d.nf, nefc);
        clamped_active = ca;
        floss_force = ff;

        efc_force = mx::add(
            mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), clamped_active),
            floss_force);
        qfrc_constraint = mx::flatten(mx::matmul(
            mx::transpose(d.efc_J), mx::reshape(efc_force, {nefc, 1})));

        cost_c = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jaref, Jaref)), clamped_active)));
        gauss = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::subtract(Ma, d.qfrc_smooth),
                                  mx::subtract(qacc, d.qacc_smooth))));
        total_cost = mx::add(cost_c, gauss);

        // Update gradient and CG direction
        if (!use_newton) {
            auto prev_grad = grad;
            auto prev_Mgrad = Mgrad;
            grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));
            Mgrad = vmap_solve_m(m, d, grad);
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
