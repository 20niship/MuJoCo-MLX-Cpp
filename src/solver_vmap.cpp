// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// Vmap-compatible constraint solver.
// Fixed iteration count, batched 5-alpha linesearch, no early break.
// NO eval(), NO data<>(), NO CPU sync.

#include "internal.h"

namespace mjmlx {

static constexpr float MJMINVAL_SV = 1e-8f;

// Dense M @ v using pure MLX ops (no eval)
static mx::array vmap_mul_m(const Data& d, int nv, const mx::array& vec) {
    return mx::flatten(mx::matmul(d.qM, mx::reshape(vec, {nv, 1})));
}

Data vmap_solve(const Model& m, Data d) {
    int nefc = d.efc_J.shape(0);
    if (nefc == 0) {
        d.qacc = d.qacc_smooth;
        d.qfrc_constraint = mx::zeros({m.nv});
        return d;
    }

    // Start from smooth acceleration
    auto qacc = d.qacc_smooth;

    // J @ qacc - aref
    auto Jaref = mx::subtract(
        mx::flatten(mx::matmul(d.efc_J, mx::reshape(qacc, {m.nv, 1}))),
        d.efc_aref);
    auto Ma = vmap_mul_m(d, m.nv, qacc);

    // Active set: inequality constraints active when Jaref < 0
    auto active = mx::astype(mx::less(Jaref, mx::array(0.0f)), mx::float32);
    if (d.ne + d.nf > 0) {
        auto always = mx::ones({d.ne + d.nf});
        auto rest = mx::slice(active, mx::Shape{d.ne + d.nf}, mx::Shape{nefc});
        active = mx::concatenate({always, rest}, 0);
    }

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

    // Use M^{-1} as preconditioner for all solver types (avoids CPU cholesky in hot path)
    auto Mgrad = vmap_solve_m(m, d, grad);
    auto search = mx::negative(Mgrad);

    // Fixed-iteration solver loop
    for (int iter = 0; iter < m.opt.iterations; iter++) {
        auto Mv = vmap_mul_m(d, m.nv, search);
        auto Jv = mx::flatten(mx::matmul(d.efc_J, mx::reshape(search, {m.nv, 1})));

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

        // 5-alpha vectorized linesearch
        auto alphas = mx::stack({alpha_n,
            mx::multiply(alpha_n, mx::array(0.5f)),
            mx::multiply(alpha_n, mx::array(0.1f)),
            mx::array(0.01f), mx::array(0.001f)});

        auto x_all = mx::add(mx::reshape(Jaref, {1, nefc}),
            mx::multiply(mx::reshape(alphas, {5, 1}), mx::reshape(Jv, {1, nefc})));
        auto act_all = mx::astype(mx::less(x_all, mx::array(0.0f)), mx::float32);
        if (d.ne + d.nf > 0) {
            for (int k = 0; k < d.ne + d.nf; k++) {
                std::vector<float> ones5(5, 1.0f);
                auto mask5 = mx::reshape(mx::array(ones5.data(), mx::Shape{5}, mx::float32), mx::Shape{5, 1});
                std::vector<float> pm(nefc, 0.0f); pm[k] = 1.0f;
                auto col_mask = mx::reshape(mx::array(pm.data(), mx::Shape{nefc}, mx::float32), mx::Shape{1, nefc});
                act_all = mx::add(act_all, mx::multiply(
                    mx::multiply(mask5, col_mask),
                    mx::subtract(mx::array(1.0f), mx::multiply(
                        mx::slice(act_all, mx::Shape{0, k}, mx::Shape{5, k+1}), mx::ones(mx::Shape{5, 1})))));
            }
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

        auto all_costs = mx::concatenate({mx::reshape(total_cost, {1}), total_all}, 0);
        auto all_alphas = mx::concatenate({mx::zeros(mx::Shape{1}), alphas}, 0);
        auto best_idx = mx::argmin(all_costs);
        auto best_alpha = mx::take(all_alphas, best_idx);

        qacc = mx::add(qacc, mx::multiply(search, best_alpha));
        Ma = mx::add(Ma, mx::multiply(Mv, best_alpha));
        Jaref = mx::add(Jaref, mx::multiply(Jv, best_alpha));

        active = mx::astype(mx::less(Jaref, mx::array(0.0f)), mx::float32);
        if (d.ne + d.nf > 0) {
            auto always = mx::ones({d.ne + d.nf});
            auto rest = mx::slice(active, mx::Shape{d.ne + d.nf}, mx::Shape{nefc});
            active = mx::concatenate({always, rest}, 0);
        }

        efc_force = mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), active);
        qfrc_constraint = mx::flatten(mx::matmul(
            mx::transpose(d.efc_J), mx::reshape(efc_force, {nefc, 1})));

        cost_c = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::multiply(d.efc_D, mx::multiply(Jaref, Jaref)), active)));
        gauss = mx::multiply(mx::array(0.5f),
            mx::sum(mx::multiply(mx::subtract(Ma, d.qfrc_smooth),
                                  mx::subtract(qacc, d.qacc_smooth))));
        total_cost = mx::add(cost_c, gauss);

        grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));

        // Preconditioned CG direction using M^{-1} (all GPU, no CPU linalg)
        auto prev_grad = grad;
        auto prev_Mgrad = Mgrad;
        Mgrad = vmap_solve_m(m, d, grad);
        auto beta_num = mx::sum(mx::multiply(grad, mx::subtract(Mgrad, prev_Mgrad)));
        auto beta_den = mx::maximum(mx::array(MJMINVAL_SV),
                                     mx::sum(mx::multiply(prev_grad, prev_Mgrad)));
        auto beta = mx::maximum(mx::divide(beta_num, beta_den), mx::array(0.0f));
        search = mx::add(mx::negative(Mgrad), mx::multiply(beta, search));
    }

    d.qfrc_constraint = qfrc_constraint;
    d.qacc = qacc;
    d.efc_force = efc_force;
    return d;
}

} // namespace mjmlx
