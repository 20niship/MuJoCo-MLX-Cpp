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

// Constraint construction: Jacobians, reference accelerations, impedances.
// Port of Python mjmlx._src.constraint.

#include "internal.h"
#include <cmath>

namespace mjmlx {

static constexpr float MJMINVAL = 1e-15f;
static constexpr float MJMINIMP = 1e-3f;
static constexpr float MJMAXIMP = 0.9999f;

// Helper: extract row from 2D array
static mx::array row(const mx::array& arr, int i) {
    return mx::flatten(mx::slice(arr, {i, 0}, {i + 1, arr.shape(1)}));
}

struct KBI {
    float k, b, imp;
};

static KBI compute_kbi(const Model& m, float solref0, float solref1,
                        float solimp0, float solimp1, float solimp2,
                        float solimp3, float solimp4, float pos) {
    float timeconst = solref0;
    float dampratio = solref1;

    if (!(m.opt.disableflags & DisableBit::REFSAFE)) {
        timeconst = std::max(timeconst, 2.0f * m.opt.timestep);
    }

    float dmin = std::max(MJMINIMP, std::min(MJMAXIMP, solimp0));
    float dmax = std::max(MJMINIMP, std::min(MJMAXIMP, solimp1));
    float width = std::max(solimp2, MJMINVAL);
    float mid = std::max(MJMINIMP, std::min(MJMAXIMP, solimp3));
    float power = std::max(solimp4, 1.0f);

    float k_val = 1.0f / (dmax * dmax * timeconst * timeconst * dampratio * dampratio);
    float b_val = 2.0f / (dmax * timeconst);

    // Direct mode
    if (solref0 <= 0) k_val = -solref0 / (dmax * dmax);
    if (solref1 <= 0) b_val = -solref1 / dmax;

    // Impedance
    float imp_x = std::abs(pos) / width;
    float imp_y;
    if (imp_x < mid) {
        imp_y = (1.0f / std::pow(mid, power - 1.0f)) * std::pow(imp_x, power);
    } else {
        imp_y = 1.0f - (1.0f / std::pow(1.0f - mid, power - 1.0f)) * std::pow(1.0f - imp_x, power);
    }
    float imp = dmin + imp_y * (dmax - dmin);
    imp = std::max(dmin, std::min(dmax, imp));
    if (imp_x > 1.0f) imp = dmax;

    return {k_val, b_val, imp};
}

Data make_constraint(const Model& m, Data d) {
    if (m.opt.disableflags & DisableBit::CONSTRAINT) {
        d.efc_J = mx::zeros({0, m.nv});
        d.efc_D = mx::zeros({0});
        d.efc_aref = mx::zeros({0});
        d.efc_force = mx::zeros({0});
        d.efc_frictionloss = mx::zeros({0});
        d.nefc = 0; d.ne = 0; d.nf = 0; d.nl = 0;
        return d;
    }

    std::vector<std::vector<float>> efc_J_rows;
    std::vector<float> efc_D_vals;
    std::vector<float> efc_aref_vals;
    std::vector<float> efc_floss_vals;
    int ne = 0, nf = 0, nl = 0;

    mx::eval(d.qvel);
    auto qvel_ptr = d.qvel.data<float>();

    // ── Joint limits ──────────────────────────────────────────────────────
    if (!(m.opt.disableflags & DisableBit::LIMIT) && m.jnt_limited.size() > 0) {
        mx::eval(m.jnt_limited); mx::eval(m.jnt_type); mx::eval(m.jnt_qposadr);
        mx::eval(m.jnt_dofadr); mx::eval(d.qpos); mx::eval(m.jnt_range);

        auto limited = m.jnt_limited.data<int>();
        auto jtypes = m.jnt_type.data<int>();
        auto jqpa = m.jnt_qposadr.data<int>();
        auto jda = m.jnt_dofadr.data<int>();
        mx::eval(d.qpos);
        auto qpos_ptr = d.qpos.data<float>();

        for (int j = 0; j < m.njnt; j++) {
            if (!limited[j]) continue;
            int jt = jtypes[j];
            if (jt != static_cast<int>(JointType::SLIDE) && jt != static_cast<int>(JointType::HINGE))
                continue;

            int qa = jqpa[j], da = jda[j];
            mx::eval(m.jnt_range);
            auto jrange = m.jnt_range.data<float>();
            float qval = qpos_ptr[qa];
            float lo = jrange[j * 2], hi = jrange[j * 2 + 1];

            float margin = 0.0f;
            if (m.jnt_margin.size() > 0) {
                mx::eval(m.jnt_margin);
                margin = m.jnt_margin.data<float>()[j];
            }

            float dist_min = qval - lo;
            float dist_max = hi - qval;
            float pos = std::min(dist_min, dist_max) - margin;

            if (pos < 0) {
                float sign = (dist_min < dist_max) ? 1.0f : -1.0f;

                std::vector<float> j_row(m.nv, 0.0f);
                j_row[da] = sign;

                float solref0 = 0.02f, solref1 = 1.0f;
                if (m.jnt_solref.size() > 0) {
                    mx::eval(m.jnt_solref);
                    auto sp = m.jnt_solref.data<float>();
                    solref0 = sp[j * 2]; solref1 = sp[j * 2 + 1];
                }
                float si0 = 0.9f, si1 = 0.95f, si2 = 0.001f, si3 = 0.5f, si4 = 2.0f;
                if (m.jnt_solimp.size() > 0) {
                    mx::eval(m.jnt_solimp);
                    auto sp = m.jnt_solimp.data<float>();
                    si0 = sp[j*5]; si1 = sp[j*5+1]; si2 = sp[j*5+2]; si3 = sp[j*5+3]; si4 = sp[j*5+4];
                }

                float invw = 1.0f;
                if (m.dof_invweight0.size() > 0) {
                    mx::eval(m.dof_invweight0);
                    invw = m.dof_invweight0.data<float>()[da];
                }

                auto [k, b, imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);
                float r = std::max(invw * (1.0f - imp) / imp, MJMINVAL);

                float jdot_qvel = 0.0f;
                for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];

                float aref = -b * jdot_qvel - k * imp * pos;

                efc_J_rows.push_back(j_row);
                efc_D_vals.push_back(1.0f / r);
                efc_aref_vals.push_back(aref);
                efc_floss_vals.push_back(0.0f);
                nl++;
            }
        }
    }

    // ── Contact constraints ──────────────────────────────────────────────
    if (!(m.opt.disableflags & DisableBit::CONTACT) && d.ncon > 0) {
        mx::eval(m.geom_bodyid);
        auto gbid = m.geom_bodyid.data<int>();

        for (int ci = 0; ci < d.ncon; ci++) {
            mx::eval(d.contact.dist); mx::eval(d.contact.pos);
            mx::eval(d.contact.frame); mx::eval(d.contact.geom);
            mx::eval(d.contact.solref); mx::eval(d.contact.solimp);
            mx::eval(d.contact.includemargin); mx::eval(d.contact.dim);
            mx::eval(d.contact.friction);

            float c_dist = d.contact.dist.data<float>()[ci];
            auto c_pos = row(d.contact.pos, ci);
            auto c_frame_all = d.contact.frame;
            auto c_geom = d.contact.geom.data<int>();
            int gid1 = c_geom[ci * 2], gid2 = c_geom[ci * 2 + 1];
            int body1 = gbid[gid1], body2 = gbid[gid2];

            float incm = d.contact.includemargin.data<float>()[ci];
            float pos = c_dist - incm;

            auto sr = d.contact.solref.data<float>();
            float solref0 = sr[ci * 2], solref1 = sr[ci * 2 + 1];
            auto si = d.contact.solimp.data<float>();
            float si0 = si[ci*5], si1 = si[ci*5+1], si2 = si[ci*5+2], si3 = si[ci*5+3], si4 = si[ci*5+4];

            int condim = d.contact.dim.data<int>()[ci];
            auto fri_ptr = d.contact.friction.data<float>();

            // Get Jacobians
            auto [jacp1, jacr1] = jac(m, d, c_pos, body1);
            auto [jacp2, jacr2] = jac(m, d, c_pos, body2);
            auto djacp = mx::subtract(jacp2, jacp1);

            // Contact frame Jacobians: rotate djacp into contact frame
            auto normal = mx::flatten(mx::slice(c_frame_all, {ci, 0, 0}, {ci + 1, 1, 3}));
            auto j_normal = mx::flatten(mx::matmul(mx::reshape(normal, {1, 3}), mx::transpose(djacp)));

            float invw = 0.0f;
            if (m.body_invweight0.size() > 0) {
                mx::eval(m.body_invweight0);
                auto iw = m.body_invweight0.data<float>();
                invw = iw[body1 * 2] + iw[body2 * 2];
            }

            if (condim == 1 || m.opt.cone != ConeType::PYRAMIDAL) {
                // Frictionless: single normal constraint row
                auto [k, b, imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);
                float r = std::max(invw * (1.0f - imp) / imp, MJMINVAL);

                mx::eval(j_normal);
                auto jn_ptr = j_normal.data<float>();
                std::vector<float> j_row(jn_ptr, jn_ptr + m.nv);

                float jdot_qvel = 0.0f;
                for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];
                float aref = -b * jdot_qvel - k * imp * pos;

                efc_J_rows.push_back(j_row);
                efc_D_vals.push_back(1.0f / r);
                efc_aref_vals.push_back(aref);
                efc_floss_vals.push_back(0.0f);
            } else {
                // Pyramidal friction: 2*(condim-1) rows
                // Each tangent direction k yields two edge rows:
                //   J_edge_pos = J_normal + mu[k-1] * J_tangent_k
                //   J_edge_neg = J_normal - mu[k-1] * J_tangent_k
                int n_tangent = std::min(condim - 1, 2); // for condim=3: 2 tangents

                // Compute tangent Jacobians
                std::vector<mx::array> j_tangents;
                for (int ti = 1; ti <= n_tangent; ti++) {
                    auto tangent = mx::flatten(mx::slice(c_frame_all, {ci, ti, 0}, {ci + 1, ti + 1, 3}));
                    auto j_t = mx::flatten(mx::matmul(mx::reshape(tangent, {1, 3}), mx::transpose(djacp)));
                    j_tangents.push_back(j_t);
                }

                // Pyramidal impedance: invw_py = invw * (1 + mu^2), then R_py = 2*mu^2*R_normal
                float mu = fri_ptr[ci * 5];
                float mu_sq = mu * mu;
                float invw_py = invw + mu_sq * invw;

                auto [kbi_k, kbi_b, kbi_imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);
                float r_normal = std::max(invw_py * (1.0f - kbi_imp) / kbi_imp, MJMINVAL);
                float r_py = std::max(2.0f * mu_sq * r_normal / m.opt.impratio, MJMINVAL);

                mx::eval(j_normal);
                auto jn_ptr = j_normal.data<float>();
                std::vector<float> j_n_vec(jn_ptr, jn_ptr + m.nv);

                for (int ti = 0; ti < n_tangent; ti++) {
                    float fri_k = fri_ptr[ci * 5 + ti];
                    mx::eval(j_tangents[ti]);
                    auto jt_ptr = j_tangents[ti].data<float>();

                    // Positive edge: J_normal + mu * J_tangent
                    std::vector<float> j_pos(m.nv);
                    for (int i = 0; i < m.nv; i++) {
                        j_pos[i] = j_n_vec[i] + fri_k * jt_ptr[i];
                    }
                    float jdot_qvel_pos = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_qvel_pos += j_pos[i] * qvel_ptr[i];
                    float aref_pos = -kbi_b * jdot_qvel_pos - kbi_k * kbi_imp * pos;

                    efc_J_rows.push_back(j_pos);
                    efc_D_vals.push_back(1.0f / r_py);
                    efc_aref_vals.push_back(aref_pos);
                    efc_floss_vals.push_back(0.0f);

                    // Negative edge: J_normal - mu * J_tangent
                    std::vector<float> j_neg(m.nv);
                    for (int i = 0; i < m.nv; i++) {
                        j_neg[i] = j_n_vec[i] - fri_k * jt_ptr[i];
                    }
                    float jdot_qvel_neg = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_qvel_neg += j_neg[i] * qvel_ptr[i];
                    float aref_neg = -kbi_b * jdot_qvel_neg - kbi_k * kbi_imp * pos;

                    efc_J_rows.push_back(j_neg);
                    efc_D_vals.push_back(1.0f / r_py);
                    efc_aref_vals.push_back(aref_neg);
                    efc_floss_vals.push_back(0.0f);
                }
            }
        }
    }

    // ── Assemble ──────────────────────────────────────────────────────
    int nefc = static_cast<int>(efc_J_rows.size());
    if (nefc > 0) {
        std::vector<float> J_data(nefc * m.nv);
        for (int i = 0; i < nefc; i++) {
            for (int j = 0; j < m.nv; j++) {
                J_data[i * m.nv + j] = efc_J_rows[i][j];
            }
        }
        d.efc_J = mx::array(J_data.data(), {nefc, m.nv}, mx::float32);
        d.efc_D = mx::array(efc_D_vals.data(), {nefc}, mx::float32);
        d.efc_aref = mx::array(efc_aref_vals.data(), {nefc}, mx::float32);
        d.efc_force = mx::zeros({nefc});
        d.efc_frictionloss = mx::array(efc_floss_vals.data(), {nefc}, mx::float32);
    } else {
        d.efc_J = mx::zeros({0, m.nv});
        d.efc_D = mx::zeros({0});
        d.efc_aref = mx::zeros({0});
        d.efc_force = mx::zeros({0});
        d.efc_frictionloss = mx::zeros({0});
    }

    d.nefc = nefc;
    d.ne = ne; d.nf = nf; d.nl = nl;
    return d;
}

} // namespace mjmlx
