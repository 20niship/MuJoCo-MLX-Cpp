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

// Constraint solver: CG with Polak-Ribière and linesearch.
// Port of Python mjmlx._src.solver.

#include "internal.h"
#include <cmath>

namespace mjmlx {

static constexpr float MJMINVAL_S = 1e-15f;

Data solve(const Model& m, Data d) {
    int nefc = d.efc_J.shape(0);
    if (nefc == 0) {
        d.qacc = d.qacc_smooth;
        d.qfrc_constraint = mx::zeros({m.nv});
        return d;
    }

    // Start from smooth acceleration
    auto qacc = d.qacc_smooth;

    // Compute Jaref = J @ qacc - aref
    auto Jaref = mx::subtract(mx::flatten(mx::matmul(d.efc_J, mx::reshape(qacc, {m.nv, 1}))),
                               d.efc_aref);
    auto Ma = mul_m(m, d, qacc);

    // Active constraints (Jaref < 0 for inequality)
    mx::eval(Jaref);
    auto active_mask = mx::less(Jaref, mx::array(0.0f));
    if (d.ne + d.nf > 0) {
        // Equality/friction constraints are always active
        std::vector<uint8_t> mask_data(nefc, 0);
        mx::eval(active_mask);
        auto mp = active_mask.data<bool>();
        for (int i = 0; i < nefc; i++) mask_data[i] = mp[i] ? 1 : 0;
        for (int i = 0; i < d.ne + d.nf; i++) mask_data[i] = 1;
        active_mask = mx::array(mask_data.data(), {nefc}, mx::bool_);
    }

    auto active_f = mx::astype(active_mask, mx::float32);

    // Constraint force
    auto efc_force = mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), active_f);
    auto qfrc_constraint = mx::flatten(mx::matmul(mx::transpose(d.efc_J),
                                                    mx::reshape(efc_force, {nefc, 1})));

    // Gradient
    auto grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));
    auto Mgrad = solve_m(m, d, grad);
    auto search = mx::negative(Mgrad);

    float prev_cost = 0.0f;

    // Solver iterations
    for (int iter = 0; iter < m.opt.iterations; iter++) {
        mx::eval(qacc); mx::eval(grad); mx::eval(search);

        // Cost
        float cost_c = 0.0f;
        {
            mx::eval(Jaref); mx::eval(d.efc_D); mx::eval(active_f);
            auto ja = Jaref.data<float>();
            auto dd = d.efc_D.data<float>();
            auto af = active_f.data<float>();
            for (int i = 0; i < nefc; i++) {
                cost_c += 0.5f * dd[i] * ja[i] * ja[i] * af[i];
            }
        }
        float gauss = 0.0f;
        {
            auto diff = mx::subtract(Ma, d.qfrc_smooth);
            auto diff2 = mx::subtract(qacc, d.qacc_smooth);
            auto g = mx::multiply(mx::array(0.5f), mx::sum(mx::multiply(diff, diff2)));
            mx::eval(g);
            gauss = g.item<float>();
        }
        float total_cost = cost_c + gauss;

        // Check convergence
        float scale = m.stat.meaninertia * std::max(1, m.nv);
        if (iter > 0) {
            float improvement = (prev_cost - total_cost) / scale;
            mx::eval(grad);
            auto gp = grad.data<float>();
            float gnorm = 0;
            for (int i = 0; i < m.nv; i++) gnorm += gp[i] * gp[i];
            gnorm = std::sqrt(gnorm) / scale;
            if (improvement < m.opt.tolerance || gnorm < m.opt.tolerance) break;
        }
        prev_cost = total_cost;

        // Linesearch: compute search direction products
        auto Mv = mul_m(m, d, search);
        auto Jv = mx::flatten(mx::matmul(d.efc_J, mx::reshape(search, {m.nv, 1})));

        // Newton step estimate
        mx::eval(Mv); mx::eval(Jv); mx::eval(Ma); mx::eval(d.qfrc_smooth);
        auto mv_ptr = Mv.data<float>(); auto sv = search.data<float>();
        auto jv_ptr = Jv.data<float>(); auto ma_ptr = Ma.data<float>();
        auto qs_ptr = d.qfrc_smooth.data<float>();
        auto dd = d.efc_D.data<float>(); auto ja = Jaref.data<float>();
        auto af = active_f.data<float>();

        float quad_gauss = 0, linear_gauss = 0;
        for (int i = 0; i < m.nv; i++) {
            quad_gauss += 0.5f * sv[i] * mv_ptr[i];
            linear_gauss += sv[i] * (ma_ptr[i] - qs_ptr[i]);
        }
        float quad_con = 0, linear_con = 0;
        for (int i = 0; i < nefc; i++) {
            quad_con += 0.5f * dd[i] * jv_ptr[i] * jv_ptr[i] * af[i];
            linear_con += dd[i] * jv_ptr[i] * ja[i] * af[i];
        }

        float denom = 2.0f * (quad_gauss + quad_con);
        float alpha_n = -(linear_gauss + linear_con) / std::max(denom, MJMINVAL_S);
        alpha_n = std::max(-2.0f, std::min(2.0f, alpha_n));

        // Try Newton step
        float best_alpha = 0.0f, best_cost = total_cost;
        float alphas[] = {alpha_n, alpha_n * 0.5f, alpha_n * 0.1f, 0.01f, 0.001f};
        for (float al : alphas) {
            // Evaluate cost at alpha
            float cc = 0;
            for (int i = 0; i < nefc; i++) {
                float x = ja[i] + al * jv_ptr[i];
                bool act = (x < 0) || (i < d.ne + d.nf);
                if (act) cc += 0.5f * dd[i] * x * x;
            }
            float gg = gauss;
            for (int i = 0; i < m.nv; i++) {
                gg += al * sv[i] * ma_ptr[i] - al * sv[i] * qs_ptr[i];
            }
            gg += 0.5f * al * al * quad_gauss * 2.0f;
            float tc = cc + gg;
            if (tc < best_cost) { best_alpha = al; best_cost = tc; }
        }

        // Apply step
        qacc = mx::add(qacc, mx::multiply(search, mx::array(best_alpha)));
        Ma = mx::add(Ma, mx::multiply(Mv, mx::array(best_alpha)));
        Jaref = mx::add(Jaref, mx::multiply(Jv, mx::array(best_alpha)));

        // Update active set
        active_mask = mx::less(Jaref, mx::array(0.0f));
        mx::eval(active_mask);
        {
            auto mp2 = active_mask.data<bool>();
            std::vector<uint8_t> md(nefc, 0);
            for (int i = 0; i < nefc; i++) md[i] = mp2[i] ? 1 : 0;
            for (int i = 0; i < d.ne + d.nf; i++) md[i] = 1;
            active_mask = mx::array(md.data(), {nefc}, mx::bool_);
        }
        active_f = mx::astype(active_mask, mx::float32);

        efc_force = mx::multiply(mx::multiply(d.efc_D, mx::negative(Jaref)), active_f);
        qfrc_constraint = mx::flatten(mx::matmul(mx::transpose(d.efc_J),
                                                  mx::reshape(efc_force, {nefc, 1})));

        // CG update
        auto prev_grad = grad;
        auto prev_Mgrad = Mgrad;
        grad = mx::subtract(Ma, mx::add(d.qfrc_smooth, qfrc_constraint));
        Mgrad = solve_m(m, d, grad);

        if (m.opt.solver == SolverType::NEWTON) {
            search = mx::negative(Mgrad);
        } else {
            // Polak-Ribière
            auto beta_num = mx::sum(mx::multiply(grad, mx::subtract(Mgrad, prev_Mgrad)));
            auto beta_den = mx::maximum(mx::array(MJMINVAL_S), mx::sum(mx::multiply(prev_grad, prev_Mgrad)));
            auto beta = mx::maximum(mx::divide(beta_num, beta_den), mx::array(0.0f));
            search = mx::add(mx::negative(Mgrad), mx::multiply(beta, search));
        }
    }

    d.qfrc_constraint = qfrc_constraint;
    d.qacc = qacc;
    d.efc_force = efc_force;
    return d;
}

} // namespace mjmlx
