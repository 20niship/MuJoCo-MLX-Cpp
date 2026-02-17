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

    // ── Equality constraints (MUST be first — solver treats first ne rows as always-active) ──
    if (!(m.opt.disableflags & DisableBit::EQUALITY) && m.neq > 0) {
        mx::eval(m.eq_type); mx::eval(m.eq_obj1id); mx::eval(m.eq_obj2id);
        mx::eval(m.eq_data); mx::eval(m.eq_solref); mx::eval(m.eq_solimp);
        mx::eval(d.xpos); mx::eval(d.xmat); mx::eval(d.xquat);
        mx::eval(m.qpos0); mx::eval(d.qpos);

        auto eq_types = m.eq_type.data<int>();
        auto eq_obj1 = m.eq_obj1id.data<int>();
        auto eq_obj2 = m.eq_obj2id.data<int>();
        auto eq_dat = m.eq_data.data<float>();
        auto eq_sr = m.eq_solref.data<float>();
        auto eq_si = m.eq_solimp.data<float>();
        auto xpos_ptr = d.xpos.data<float>();
        auto xmat_ptr = d.xmat.data<float>();
        auto qpos0_ptr = m.qpos0.data<float>();
        auto qpos_ptr = d.qpos.data<float>();

        for (int e = 0; e < m.neq; e++) {
            int etype = eq_types[e];
            int id1 = eq_obj1[e];
            int id2 = eq_obj2[e];
            float solref0 = eq_sr[e * 2], solref1 = eq_sr[e * 2 + 1];
            float si0 = eq_si[e*5], si1 = eq_si[e*5+1], si2 = eq_si[e*5+2];
            float si3 = eq_si[e*5+3], si4 = eq_si[e*5+4];

            if (etype == 0) {
                // ── CONNECT: 3 translational rows ──
                int body1 = id1, body2 = id2;
                float anchor1[3] = { eq_dat[e*11+0], eq_dat[e*11+1], eq_dat[e*11+2] };
                float anchor2[3] = { eq_dat[e*11+3], eq_dat[e*11+4], eq_dat[e*11+5] };

                // pos1 = xpos[body1] + xmat[body1] @ anchor1
                const float* xp1 = xpos_ptr + body1 * 3;
                const float* xm1 = xmat_ptr + body1 * 9;
                float pos1[3], pos2[3];
                for (int i = 0; i < 3; i++) {
                    pos1[i] = xp1[i];
                    for (int j = 0; j < 3; j++)
                        pos1[i] += xm1[i * 3 + j] * anchor1[j];
                }

                const float* xp2 = xpos_ptr + body2 * 3;
                const float* xm2 = xmat_ptr + body2 * 9;
                for (int i = 0; i < 3; i++) {
                    pos2[i] = xp2[i];
                    for (int j = 0; j < 3; j++)
                        pos2[i] += xm2[i * 3 + j] * anchor2[j];
                }

                // error = pos1 - pos2
                float err[3] = { pos1[0]-pos2[0], pos1[1]-pos2[1], pos1[2]-pos2[2] };

                // Jacobians
                auto p1_arr = mx::array(pos1, {3});
                auto p2_arr = mx::array(pos2, {3});
                auto [jacp1, jacr1] = jac(m, d, p1_arr, body1);
                auto [jacp2, jacr2] = jac(m, d, p2_arr, body2);

                mx::eval(jacp1); mx::eval(jacp2);
                auto jp1 = jacp1.data<float>(); // (nv, 3)
                auto jp2 = jacp2.data<float>();

                float invw = 0.0f;
                if (m.body_invweight0.size() > 0) {
                    mx::eval(m.body_invweight0);
                    auto iw = m.body_invweight0.data<float>();
                    invw = iw[body1 * 2] + iw[body2 * 2];
                }

                float pos_norm = std::sqrt(err[0]*err[0] + err[1]*err[1] + err[2]*err[2]);

                for (int axis = 0; axis < 3; axis++) {
                    std::vector<float> j_row(m.nv, 0.0f);
                    for (int i = 0; i < m.nv; i++)
                        j_row[i] = jp1[i * 3 + axis] - jp2[i * 3 + axis];

                    float pos = err[axis];
                    auto [k, b, imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos_norm);
                    float r = std::max(invw * (1.0f - imp) / imp, MJMINVAL);

                    float jdot_qvel = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];
                    float aref = -b * jdot_qvel - k * imp * pos;

                    efc_J_rows.push_back(j_row);
                    efc_D_vals.push_back(1.0f / r);
                    efc_aref_vals.push_back(aref);
                    efc_floss_vals.push_back(0.0f);
                    ne++;
                }

            } else if (etype == 1) {
                // ── WELD: 3 translational + 3 rotational rows ──
                int body1 = id1, body2 = id2;
                // data[0:3] = anchor on body2, data[3:6] = anchor on body1 (swapped vs connect in MJX)
                float anc_body2[3] = { eq_dat[e*11+0], eq_dat[e*11+1], eq_dat[e*11+2] };
                float anc_body1[3] = { eq_dat[e*11+3], eq_dat[e*11+4], eq_dat[e*11+5] };
                float relquat[4] = { eq_dat[e*11+6], eq_dat[e*11+7], eq_dat[e*11+8], eq_dat[e*11+9] };
                float torquescale = eq_dat[e*11+10];

                // Position: pos1 = xpos[body1] + xmat[body1] @ anc_body1
                const float* xp1 = xpos_ptr + body1 * 3;
                const float* xm1 = xmat_ptr + body1 * 9;
                float pos1[3], pos2[3];
                for (int i = 0; i < 3; i++) {
                    pos1[i] = xp1[i];
                    for (int j = 0; j < 3; j++)
                        pos1[i] += xm1[i * 3 + j] * anc_body1[j];
                }

                const float* xp2 = xpos_ptr + body2 * 3;
                const float* xm2 = xmat_ptr + body2 * 9;
                for (int i = 0; i < 3; i++) {
                    pos2[i] = xp2[i];
                    for (int j = 0; j < 3; j++)
                        pos2[i] += xm2[i * 3 + j] * anc_body2[j];
                }

                float cpos[3] = { pos1[0]-pos2[0], pos1[1]-pos2[1], pos1[2]-pos2[2] };

                // Jacobians for position
                auto p1_arr = mx::array(pos1, {3});
                auto p2_arr = mx::array(pos2, {3});
                auto [jacp1, jacr1] = jac(m, d, p1_arr, body1);
                auto [jacp2, jacr2] = jac(m, d, p2_arr, body2);

                mx::eval(jacp1); mx::eval(jacp2); mx::eval(jacr1); mx::eval(jacr2);
                auto jp1 = jacp1.data<float>();
                auto jp2 = jacp2.data<float>();
                auto jr1 = jacr1.data<float>();
                auto jr2 = jacr2.data<float>();

                float invw_t = 0.0f, invw_r = 0.0f;
                if (m.body_invweight0.size() > 0) {
                    mx::eval(m.body_invweight0);
                    auto iw = m.body_invweight0.data<float>();
                    invw_t = iw[body1 * 2] + iw[body2 * 2];
                    invw_r = iw[body1 * 2 + 1] + iw[body2 * 2 + 1];
                }

                // Rotation error: conj(q2) * q1 * relquat
                mx::eval(d.xquat);
                auto xquat_ptr = d.xquat.data<float>();
                const float* q1 = xquat_ptr + body1 * 4;
                const float* q2 = xquat_ptr + body2 * 4;

                // quat = q1 * relquat
                float quat_prod[4];
                quat_prod[0] = q1[0]*relquat[0] - q1[1]*relquat[1] - q1[2]*relquat[2] - q1[3]*relquat[3];
                quat_prod[1] = q1[0]*relquat[1] + q1[1]*relquat[0] + q1[2]*relquat[3] - q1[3]*relquat[2];
                quat_prod[2] = q1[0]*relquat[2] - q1[1]*relquat[3] + q1[2]*relquat[0] + q1[3]*relquat[1];
                quat_prod[3] = q1[0]*relquat[3] + q1[1]*relquat[2] - q1[2]*relquat[1] + q1[3]*relquat[0];

                // q2_inv = conj(q2)
                float q2_inv[4] = { q2[0], -q2[1], -q2[2], -q2[3] };

                // q_err = q2_inv * quat_prod
                float q_err[4];
                q_err[0] = q2_inv[0]*quat_prod[0] - q2_inv[1]*quat_prod[1] - q2_inv[2]*quat_prod[2] - q2_inv[3]*quat_prod[3];
                q_err[1] = q2_inv[0]*quat_prod[1] + q2_inv[1]*quat_prod[0] + q2_inv[2]*quat_prod[3] - q2_inv[3]*quat_prod[2];
                q_err[2] = q2_inv[0]*quat_prod[2] - q2_inv[1]*quat_prod[3] + q2_inv[2]*quat_prod[0] + q2_inv[3]*quat_prod[1];
                q_err[3] = q2_inv[0]*quat_prod[3] + q2_inv[1]*quat_prod[2] - q2_inv[2]*quat_prod[1] + q2_inv[3]*quat_prod[0];

                // Rotation error = axis part of q_err, scaled by torquescale
                float crot[3] = { q_err[1] * torquescale, q_err[2] * torquescale, q_err[3] * torquescale };

                // Combined error for pos_imp
                float all_err[6] = { cpos[0], cpos[1], cpos[2], crot[0], crot[1], crot[2] };
                float pos_norm = 0.0f;
                for (int i = 0; i < 6; i++) pos_norm += all_err[i] * all_err[i];
                pos_norm = std::sqrt(pos_norm);

                auto [k, b, imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos_norm);

                // Rotational Jacobian correction: 0.5 * conj(q2) * (jacr1-jacr2) * q1 * relquat
                // For each DOF i, transform the rotation Jacobian column:
                //   j_corr[i] = 0.5 * quat_mul(quat_mul_axis(q2_inv, jr_diff[:,i]), quat_prod)[1:4]
                // This is the quaternion correction for the weld rotational constraint.
                // For simplicity, we compute the corrected Jacobian per DOF.
                std::vector<float> jacr_corr(m.nv * 3, 0.0f);
                for (int i = 0; i < m.nv; i++) {
                    float jrd[3] = {
                        (jr1[i*3+0] - jr2[i*3+0]) * torquescale,
                        (jr1[i*3+1] - jr2[i*3+1]) * torquescale,
                        (jr1[i*3+2] - jr2[i*3+2]) * torquescale
                    };
                    // quat_mul_axis(q2_inv, jrd): treat jrd as pure quaternion (0, jrd)
                    // q2_inv * (0, jrd) = (-q2_inv[1:]*jrd, q2_inv[0]*jrd + cross(q2_inv[1:], jrd))
                    float qm[4];
                    qm[0] = -(q2_inv[1]*jrd[0] + q2_inv[2]*jrd[1] + q2_inv[3]*jrd[2]);
                    qm[1] =  q2_inv[0]*jrd[0] + q2_inv[2]*jrd[2] - q2_inv[3]*jrd[1];
                    qm[2] =  q2_inv[0]*jrd[1] + q2_inv[3]*jrd[0] - q2_inv[1]*jrd[2];
                    qm[3] =  q2_inv[0]*jrd[2] + q2_inv[1]*jrd[1] - q2_inv[2]*jrd[0];

                    // result = qm * quat_prod (extract imaginary part, scale by 0.5)
                    float res[4];
                    res[0] = qm[0]*quat_prod[0] - qm[1]*quat_prod[1] - qm[2]*quat_prod[2] - qm[3]*quat_prod[3];
                    res[1] = qm[0]*quat_prod[1] + qm[1]*quat_prod[0] + qm[2]*quat_prod[3] - qm[3]*quat_prod[2];
                    res[2] = qm[0]*quat_prod[2] - qm[1]*quat_prod[3] + qm[2]*quat_prod[0] + qm[3]*quat_prod[1];
                    res[3] = qm[0]*quat_prod[3] + qm[1]*quat_prod[2] - qm[2]*quat_prod[1] + qm[3]*quat_prod[0];

                    jacr_corr[i*3+0] = 0.5f * res[1];
                    jacr_corr[i*3+1] = 0.5f * res[2];
                    jacr_corr[i*3+2] = 0.5f * res[3];
                }

                // 3 positional rows
                for (int axis = 0; axis < 3; axis++) {
                    std::vector<float> j_row(m.nv, 0.0f);
                    for (int i = 0; i < m.nv; i++)
                        j_row[i] = jp1[i * 3 + axis] - jp2[i * 3 + axis];

                    float pos = cpos[axis];
                    float r = std::max(invw_t * (1.0f - imp) / imp, MJMINVAL);

                    float jdot_qvel = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];
                    float aref = -b * jdot_qvel - k * imp * pos;

                    efc_J_rows.push_back(j_row);
                    efc_D_vals.push_back(1.0f / r);
                    efc_aref_vals.push_back(aref);
                    efc_floss_vals.push_back(0.0f);
                    ne++;
                }

                // 3 rotational rows
                for (int axis = 0; axis < 3; axis++) {
                    std::vector<float> j_row(m.nv, 0.0f);
                    for (int i = 0; i < m.nv; i++)
                        j_row[i] = jacr_corr[i * 3 + axis];

                    float pos = crot[axis];
                    float r = std::max(invw_r * (1.0f - imp) / imp, MJMINVAL);

                    float jdot_qvel = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];
                    float aref = -b * jdot_qvel - k * imp * pos;

                    efc_J_rows.push_back(j_row);
                    efc_D_vals.push_back(1.0f / r);
                    efc_aref_vals.push_back(aref);
                    efc_floss_vals.push_back(0.0f);
                    ne++;
                }

            } else if (etype == 2) {
                // ── JOINT: 1 row, polynomial coupling ──
                int jnt1 = id1, jnt2 = id2;

                mx::eval(m.jnt_qposadr); mx::eval(m.jnt_dofadr);
                auto jqpa = m.jnt_qposadr.data<int>();
                auto jda = m.jnt_dofadr.data<int>();

                int qa1 = jqpa[jnt1], da1 = jda[jnt1];
                float qpos1 = qpos_ptr[qa1];
                float ref1 = qpos0_ptr[qa1];

                float poly[5] = { eq_dat[e*11+0], eq_dat[e*11+1], eq_dat[e*11+2],
                                   eq_dat[e*11+3], eq_dat[e*11+4] };

                float dif = 0.0f;
                int da2 = -1;
                if (jnt2 >= 0) {
                    int qa2 = jqpa[jnt2];
                    da2 = jda[jnt2];
                    float ref2 = qpos0_ptr[qa2];
                    dif = qpos_ptr[qa2] - ref2;
                }

                // poly_val = poly[0] + poly[1]*dif + poly[2]*dif^2 + poly[3]*dif^3 + poly[4]*dif^4
                float dif_pow[5] = { 1.0f, dif, dif*dif, dif*dif*dif, dif*dif*dif*dif };
                float poly_val = 0.0f;
                for (int i = 0; i < 5; i++) poly_val += poly[i] * dif_pow[i];

                // Error: (qpos1 - ref1) - poly(dif)
                float pos = (qpos1 - ref1) - poly_val;

                // Derivative: poly[1] + 2*poly[2]*dif + 3*poly[3]*dif^2 + 4*poly[4]*dif^3
                float deriv = 0.0f;
                if (jnt2 >= 0) {
                    deriv = poly[1] + 2.0f*poly[2]*dif + 3.0f*poly[3]*dif*dif + 4.0f*poly[4]*dif*dif*dif;
                }

                // Jacobian: 1 at dof1, -deriv at dof2
                std::vector<float> j_row(m.nv, 0.0f);
                j_row[da1] = 1.0f;
                if (da2 >= 0) j_row[da2] = -deriv;

                float invw = 0.0f;
                if (m.dof_invweight0.size() > 0) {
                    mx::eval(m.dof_invweight0);
                    invw = m.dof_invweight0.data<float>()[da1];
                    if (da2 >= 0) invw += m.dof_invweight0.data<float>()[da2];
                }

                auto [k_val, b_val, imp_val] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);
                float r = std::max(invw * (1.0f - imp_val) / imp_val, MJMINVAL);

                float jdot_qvel = 0.0f;
                for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];
                float aref = -b_val * jdot_qvel - k_val * imp_val * pos;

                efc_J_rows.push_back(j_row);
                efc_D_vals.push_back(1.0f / r);
                efc_aref_vals.push_back(aref);
                efc_floss_vals.push_back(0.0f);
                ne++;
            }
            // Types 3 (TENDON) and above are not yet supported
        }
    }

    // ── DOF friction loss (MUST come after equality, before limits) ─────────
    if (!(m.opt.disableflags & DisableBit::FRICTIONLOSS) && m.dof_frictionloss.size() > 0) {
        mx::eval(m.dof_frictionloss);
        mx::eval(m.dof_invweight0);
        mx::eval(m.dof_solref);
        mx::eval(m.dof_solimp);
        auto fl_ptr = m.dof_frictionloss.data<float>();
        auto iw_ptr = m.dof_invweight0.data<float>();
        auto sr_ptr = m.dof_solref.data<float>();
        auto si_ptr = m.dof_solimp.data<float>();

        for (int i = 0; i < m.nv; i++) {
            if (fl_ptr[i] <= 0.0f) continue;

            // J = identity row for this DOF
            std::vector<float> j_row(m.nv, 0.0f);
            j_row[i] = 1.0f;

            float pos = 0.0f;  // no positional error for friction
            float invw = iw_ptr[i];
            float solref0 = sr_ptr[i * 2], solref1 = sr_ptr[i * 2 + 1];
            float si0 = si_ptr[i*5], si1 = si_ptr[i*5+1], si2 = si_ptr[i*5+2];
            float si3 = si_ptr[i*5+3], si4 = si_ptr[i*5+4];

            auto [k, b, imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);
            float r = std::max(invw * (1.0f - imp) / imp, MJMINVAL);

            float jdot_qvel = qvel_ptr[i];  // J is identity row, so J @ qvel = qvel[i]
            float aref = -b * jdot_qvel - k * imp * pos;  // pos=0, so aref = -b * qvel[i]

            efc_J_rows.push_back(j_row);
            efc_D_vals.push_back(1.0f / r);
            efc_aref_vals.push_back(aref);
            efc_floss_vals.push_back(fl_ptr[i]);
            nf++;
        }
    }

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

    // ── Tendon limits ─────────────────────────────────────────────────────
    if (!(m.opt.disableflags & DisableBit::LIMIT) && m.ntendon > 0 &&
        m.tendon_limited.size() > 0 && d.ten_length.size() > 0) {
        mx::eval(m.tendon_limited); mx::eval(m.tendon_range);
        mx::eval(d.ten_length); mx::eval(d.ten_J);

        auto tlim = m.tendon_limited.data<int>();
        auto trange = m.tendon_range.data<float>();
        auto tlen = d.ten_length.data<float>();

        float* tmargin = nullptr;
        if (m.tendon_margin.size() > 0) {
            mx::eval(m.tendon_margin);
            tmargin = const_cast<float*>(m.tendon_margin.data<float>());
        }
        float* tsolref = nullptr;
        if (m.tendon_solref_lim.size() > 0) {
            mx::eval(m.tendon_solref_lim);
            tsolref = const_cast<float*>(m.tendon_solref_lim.data<float>());
        }
        float* tsolimp = nullptr;
        if (m.tendon_solimp_lim.size() > 0) {
            mx::eval(m.tendon_solimp_lim);
            tsolimp = const_cast<float*>(m.tendon_solimp_lim.data<float>());
        }
        float* tinvw = nullptr;
        if (m.tendon_invweight0.size() > 0) {
            mx::eval(m.tendon_invweight0);
            tinvw = const_cast<float*>(m.tendon_invweight0.data<float>());
        }

        auto tenJ_ptr = d.ten_J.data<float>();

        for (int t = 0; t < m.ntendon; t++) {
            if (!tlim[t]) continue;

            float length = tlen[t];
            float lo = trange[t * 2], hi = trange[t * 2 + 1];
            float margin = tmargin ? tmargin[t] : 0.0f;

            float dist_min = length - lo;
            float dist_max = hi - length;
            float pos = std::min(dist_min, dist_max) - margin;

            if (pos < 0) {
                float sign = (dist_min < dist_max) ? 1.0f : -1.0f;

                // Jacobian = sign * ten_J[t, :]
                std::vector<float> j_row(m.nv, 0.0f);
                for (int j = 0; j < m.nv; j++)
                    j_row[j] = sign * tenJ_ptr[t * m.nv + j];

                float solref0 = tsolref ? tsolref[t * 2] : 0.02f;
                float solref1 = tsolref ? tsolref[t * 2 + 1] : 1.0f;
                float si0 = tsolimp ? tsolimp[t*5] : 0.9f;
                float si1 = tsolimp ? tsolimp[t*5+1] : 0.95f;
                float si2 = tsolimp ? tsolimp[t*5+2] : 0.001f;
                float si3 = tsolimp ? tsolimp[t*5+3] : 0.5f;
                float si4 = tsolimp ? tsolimp[t*5+4] : 2.0f;

                float invw = tinvw ? tinvw[t] : 1.0f;

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

    // ── Tendon friction loss ─────────────────────────────────────────────
    if (!(m.opt.disableflags & DisableBit::FRICTIONLOSS) && m.ntendon > 0 &&
        m.tendon_frictionloss.size() > 0 && d.ten_J.size() > 0) {
        mx::eval(m.tendon_frictionloss); mx::eval(d.ten_J);

        auto tfloss = m.tendon_frictionloss.data<float>();
        auto tenJ_ptr2 = d.ten_J.data<float>();

        float* tsolref_f = nullptr;
        if (m.tendon_solref_fri.size() > 0) {
            mx::eval(m.tendon_solref_fri);
            tsolref_f = const_cast<float*>(m.tendon_solref_fri.data<float>());
        }
        float* tsolimp_f = nullptr;
        if (m.tendon_solimp_fri.size() > 0) {
            mx::eval(m.tendon_solimp_fri);
            tsolimp_f = const_cast<float*>(m.tendon_solimp_fri.data<float>());
        }
        float* tinvw2 = nullptr;
        if (m.tendon_invweight0.size() > 0) {
            mx::eval(m.tendon_invweight0);
            tinvw2 = const_cast<float*>(m.tendon_invweight0.data<float>());
        }

        for (int t = 0; t < m.ntendon; t++) {
            if (tfloss[t] <= 0.0f) continue;

            // Jacobian = ten_J[t, :]
            std::vector<float> j_row(m.nv, 0.0f);
            for (int j = 0; j < m.nv; j++)
                j_row[j] = tenJ_ptr2[t * m.nv + j];

            float solref0 = tsolref_f ? tsolref_f[t * 2] : 0.02f;
            float solref1 = tsolref_f ? tsolref_f[t * 2 + 1] : 1.0f;
            float si0 = tsolimp_f ? tsolimp_f[t*5] : 0.9f;
            float si1 = tsolimp_f ? tsolimp_f[t*5+1] : 0.95f;
            float si2 = tsolimp_f ? tsolimp_f[t*5+2] : 0.001f;
            float si3 = tsolimp_f ? tsolimp_f[t*5+3] : 0.5f;
            float si4 = tsolimp_f ? tsolimp_f[t*5+4] : 2.0f;

            float pos = 0.0f;  // no positional error for friction
            float invw = tinvw2 ? tinvw2[t] : 1.0f;

            auto [k, b, imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);
            float r = std::max(invw * (1.0f - imp) / imp, MJMINVAL);

            float jdot_qvel = 0.0f;
            for (int i = 0; i < m.nv; i++) jdot_qvel += j_row[i] * qvel_ptr[i];

            float aref = -b * jdot_qvel - k * imp * pos;  // pos=0

            efc_J_rows.push_back(j_row);
            efc_D_vals.push_back(1.0f / r);
            efc_aref_vals.push_back(aref);
            efc_floss_vals.push_back(tfloss[t]);
            nf++;
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
                // Build contact-frame Jacobian matrix (condim rows):
                //   jac[0..2] = frame[0..2] @ djacp (translational)
                //   jac[3]   = frame[0] @ djacr  (torsion, condim>=4)
                //   jac[4..5] = frame[1..2] @ djacr (rolling, condim>=6)
                // Then for k=1..condim-1:
                //   J_pos = jac[0] + friction[k-1] * jac[k]
                //   J_neg = jac[0] - friction[k-1] * jac[k]

                // Rotational Jacobian difference (needed for condim >= 4)
                mx::array djacr = mx::zeros({0});
                if (condim > 3) {
                    djacr = mx::subtract(jacr2, jacr1);
                }

                // Rotational invweight (for torsion/rolling impedance)
                float invw_rot = 0.0f;
                if (condim > 3 && m.body_invweight0.size() > 0) {
                    mx::eval(m.body_invweight0);
                    auto iw = m.body_invweight0.data<float>();
                    invw_rot = iw[body1 * 2 + 1] + iw[body2 * 2 + 1];
                }

                // Build all direction Jacobians (jac[k] for k=1..condim-1)
                std::vector<mx::array> jac_dirs;
                // k=1,2: tangent directions (translational)
                int n_tran = std::min(condim - 1, 2);
                for (int ti = 1; ti <= n_tran; ti++) {
                    auto tangent = mx::flatten(mx::slice(c_frame_all, {ci, ti, 0}, {ci + 1, ti + 1, 3}));
                    jac_dirs.push_back(mx::flatten(mx::matmul(mx::reshape(tangent, {1, 3}), mx::transpose(djacp))));
                }
                // k=3: torsion (rotational around normal)
                if (condim >= 4) {
                    auto norm_dir = mx::flatten(mx::slice(c_frame_all, {ci, 0, 0}, {ci + 1, 1, 3}));
                    jac_dirs.push_back(mx::flatten(mx::matmul(mx::reshape(norm_dir, {1, 3}), mx::transpose(djacr))));
                }
                // k=4,5: rolling (rotational around tangent1, tangent2)
                if (condim >= 6) {
                    for (int ti = 1; ti <= 2; ti++) {
                        auto tang_dir = mx::flatten(mx::slice(c_frame_all, {ci, ti, 0}, {ci + 1, ti + 1, 3}));
                        jac_dirs.push_back(mx::flatten(mx::matmul(mx::reshape(tang_dir, {1, 3}), mx::transpose(djacr))));
                    }
                }

                // KBI (same for all rows of this contact)
                auto [kbi_k, kbi_b, kbi_imp] = compute_kbi(m, solref0, solref1, si0, si1, si2, si3, si4, pos);

                // Pyramidal impedance: ALL rows share the same D, computed from
                // the primary sliding friction (friction[0]) and translational invweight.
                // MuJoCo C: R_py = 2*μ₀²*R_first where R_first uses dA = tran + μ₀²*tran
                float mu0 = fri_ptr[ci * 5]; // primary sliding friction
                float mu0_sq = mu0 * mu0;
                float invw_py = invw + mu0_sq * invw;
                float r_first = std::max(invw_py * (1.0f - kbi_imp) / kbi_imp, MJMINVAL);
                float r_py = std::max(2.0f * mu0_sq * r_first / m.opt.impratio, MJMINVAL);
                float d_py = 1.0f / r_py;

                mx::eval(j_normal);
                auto jn_ptr = j_normal.data<float>();
                std::vector<float> j_n_vec(jn_ptr, jn_ptr + m.nv);

                // Generate 2 pyramidal rows per direction
                int n_dirs = (int)jac_dirs.size(); // condim-1 directions
                for (int ti = 0; ti < n_dirs; ti++) {
                    float fri_k = fri_ptr[ci * 5 + ti];

                    mx::eval(jac_dirs[ti]);
                    auto jt_ptr = jac_dirs[ti].data<float>();

                    // Positive edge: J_normal + mu * J_direction
                    std::vector<float> j_pos(m.nv);
                    for (int i = 0; i < m.nv; i++) j_pos[i] = j_n_vec[i] + fri_k * jt_ptr[i];
                    float jdot_pos = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_pos += j_pos[i] * qvel_ptr[i];
                    float aref_pos = -kbi_b * jdot_pos - kbi_k * kbi_imp * pos;

                    efc_J_rows.push_back(j_pos);
                    efc_D_vals.push_back(1.0f / r_py);
                    efc_aref_vals.push_back(aref_pos);
                    efc_floss_vals.push_back(0.0f);

                    // Negative edge: J_normal - mu * J_direction
                    std::vector<float> j_neg(m.nv);
                    for (int i = 0; i < m.nv; i++) j_neg[i] = j_n_vec[i] - fri_k * jt_ptr[i];
                    float jdot_neg = 0.0f;
                    for (int i = 0; i < m.nv; i++) jdot_neg += j_neg[i] * qvel_ptr[i];
                    float aref_neg = -kbi_b * jdot_neg - kbi_k * kbi_imp * pos;

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
