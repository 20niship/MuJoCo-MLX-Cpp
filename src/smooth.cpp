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

// Core smooth dynamics: kinematics, COM, CRB, mass matrix, RNE, transmission.
// Port of Python mjmlx._src.smooth.

#include "internal.h"
#include <cmath>

namespace mjmlx {

static constexpr float MJMINVAL = 1e-15f;

// Helper: extract row i from a 2D array as a 1D array
static mx::array row(const mx::array& arr, int i) {
    return mx::flatten(mx::slice(arr, {i, 0}, {i + 1, arr.shape(1)}));
}

// Helper: set row i of a 2D array from a 1D source, returns new array
static mx::array set_row(const mx::array& arr, int i, const mx::array& val) {
    int cols = arr.shape(1);
    auto before = mx::slice(arr, {0, 0}, {i, cols});
    auto after = mx::slice(arr, {i + 1, 0}, {arr.shape(0), cols});
    auto new_row = mx::reshape(val, {1, cols});
    return mx::concatenate({before, new_row, after}, 0);
}

// ── Kinematics ───────────────────────────────────────────────────────────────

Data kinematics(const Model& m, Data d) {
    mx::eval(m.body_parentid); mx::eval(m.jnt_bodyid); mx::eval(m.jnt_type);
    mx::eval(m.jnt_qposadr); mx::eval(m.jnt_dofadr); mx::eval(d.qpos);
    mx::eval(m.body_pos); mx::eval(m.body_quat); mx::eval(m.qpos0);
    mx::eval(m.jnt_pos); mx::eval(m.jnt_axis);

    auto parent_ptr = m.body_parentid.data<int>();
    auto jnt_bodyid_ptr = m.jnt_bodyid.data<int>();
    auto jnt_type_ptr = m.jnt_type.data<int>();
    auto jnt_qposadr_ptr = m.jnt_qposadr.data<int>();

    // Output storage
    std::vector<mx::array> xpos_vec;  xpos_vec.reserve(m.nbody);
    std::vector<mx::array> xquat_vec; xquat_vec.reserve(m.nbody);
    std::vector<mx::array> xmat_vec;  xmat_vec.reserve(m.nbody);
    for (int i = 0; i < m.nbody; i++) {
        xpos_vec.push_back(mx::zeros({3}));
        xquat_vec.push_back(mx::array({1.0f, 0.0f, 0.0f, 0.0f}));
        xmat_vec.push_back(mx::eye(3));
    }
    std::vector<mx::array> xanchor_vec;
    std::vector<mx::array> xaxis_vec;
    for (int i = 0; i < m.njnt; i++) {
        xanchor_vec.push_back(mx::zeros({3}));
        xaxis_vec.push_back(mx::array({0.0f, 0.0f, 1.0f}));
    }

    // World body
    xpos_vec[0] = row(m.body_pos, 0);
    xquat_vec[0] = row(m.body_quat, 0);
    xmat_vec[0] = quat_to_mat(xquat_vec[0]);

    // Walk tree root-to-leaves
    for (int body_id = 1; body_id < m.nbody; body_id++) {
        int pid = parent_ptr[body_id];
        auto pos = mx::add(xpos_vec[pid], rotate(row(m.body_pos, body_id), xquat_vec[pid]));
        auto quat = quat_mul(xquat_vec[pid], row(m.body_quat, body_id));

        // Apply joints belonging to this body
        for (int ji = 0; ji < m.njnt; ji++) {
            if (jnt_bodyid_ptr[ji] != body_id) continue;
            int jt = jnt_type_ptr[ji];
            int qa = jnt_qposadr_ptr[ji];

            if (jt == static_cast<int>(JointType::FREE)) {
                pos = mx::slice(d.qpos, {qa}, {qa + 3});
                quat = normalize(mx::slice(d.qpos, {qa + 3}, {qa + 7}));
                xanchor_vec[ji] = pos;
                xaxis_vec[ji] = mx::array({0.0f, 0.0f, 1.0f});
            } else if (jt == static_cast<int>(JointType::HINGE)) {
                auto jnt_pos_j = row(m.jnt_pos, ji);
                auto jnt_axis_j = row(m.jnt_axis, ji);
                auto anchor = mx::add(rotate(jnt_pos_j, quat), pos);
                auto axis = rotate(jnt_axis_j, quat);
                auto angle = mx::subtract(mx::slice(d.qpos, {qa}, {qa + 1}),
                                          mx::slice(m.qpos0, {qa}, {qa + 1}));
                auto qloc = axis_angle_to_quat(jnt_axis_j, mx::flatten(angle));
                quat = quat_mul(quat, qloc);
                pos = mx::subtract(anchor, rotate(jnt_pos_j, quat));
                xanchor_vec[ji] = anchor;
                xaxis_vec[ji] = axis;
            } else if (jt == static_cast<int>(JointType::SLIDE)) {
                auto jnt_pos_j = row(m.jnt_pos, ji);
                auto jnt_axis_j = row(m.jnt_axis, ji);
                auto anchor = mx::add(rotate(jnt_pos_j, quat), pos);
                auto axis = rotate(jnt_axis_j, quat);
                auto disp = mx::subtract(mx::slice(d.qpos, {qa}, {qa + 1}),
                                         mx::slice(m.qpos0, {qa}, {qa + 1}));
                pos = mx::add(pos, mx::multiply(axis, mx::flatten(disp)));
                xanchor_vec[ji] = anchor;
                xaxis_vec[ji] = axis;
            } else if (jt == static_cast<int>(JointType::BALL)) {
                auto jnt_pos_j = row(m.jnt_pos, ji);
                auto anchor = mx::add(rotate(jnt_pos_j, quat), pos);
                auto axis = rotate(row(m.jnt_axis, ji), quat);
                auto qloc = normalize(mx::slice(d.qpos, {qa}, {qa + 4}));
                quat = quat_mul(quat, qloc);
                pos = mx::subtract(anchor, rotate(jnt_pos_j, quat));
                xanchor_vec[ji] = anchor;
                xaxis_vec[ji] = axis;
            }
        }

        xpos_vec[body_id] = pos;
        xquat_vec[body_id] = quat;
        xmat_vec[body_id] = quat_to_mat(quat);
    }

    d.xpos = mx::stack(xpos_vec);
    d.xquat = mx::stack(xquat_vec);
    d.xmat = mx::stack(xmat_vec);
    d.xanchor = mx::stack(xanchor_vec);
    d.xaxis = mx::stack(xaxis_vec);

    // Compute body inertial frame
    std::vector<mx::array> xipos_vec; xipos_vec.reserve(m.nbody);
    std::vector<mx::array> ximat_vec; ximat_vec.reserve(m.nbody);
    for (int i = 0; i < m.nbody; i++) {
        xipos_vec.push_back(mx::zeros({3}));
        ximat_vec.push_back(mx::eye(3));
    }
    for (int i = 0; i < m.nbody; i++) {
        auto [p, mat] = local_to_global(xpos_vec[i], xquat_vec[i],
                                         row(m.body_ipos, i), row(m.body_iquat, i));
        xipos_vec[i] = p;
        ximat_vec[i] = mat;
    }
    d.xipos = mx::stack(xipos_vec);
    d.ximat = mx::stack(ximat_vec);

    // Geom positions
    if (m.ngeom > 0) {
        mx::eval(m.geom_bodyid);
        auto gbid = m.geom_bodyid.data<int>();
        std::vector<mx::array> gxpos; gxpos.reserve(m.ngeom);
        std::vector<mx::array> gxmat; gxmat.reserve(m.ngeom);
        for (int i2 = 0; i2 < m.ngeom; i2++) { gxpos.push_back(mx::zeros({3})); gxmat.push_back(mx::eye(3)); }
        for (int i = 0; i < m.ngeom; i++) {
            int bid = gbid[i];
            auto [p, mat] = local_to_global(xpos_vec[bid], xquat_vec[bid],
                                             row(m.geom_pos, i), row(m.geom_quat, i));
            gxpos[i] = p;
            gxmat[i] = mat;
        }
        d.geom_xpos = mx::stack(gxpos);
        d.geom_xmat = mx::stack(gxmat);
    }

    // Site positions
    if (m.nsite > 0) {
        mx::eval(m.site_bodyid);
        auto sbid = m.site_bodyid.data<int>();
        std::vector<mx::array> sxpos; sxpos.reserve(m.nsite);
        std::vector<mx::array> sxmat; sxmat.reserve(m.nsite);
        for (int i2 = 0; i2 < m.nsite; i2++) { sxpos.push_back(mx::zeros({3})); sxmat.push_back(mx::eye(3)); }
        for (int i = 0; i < m.nsite; i++) {
            int bid = sbid[i];
            auto [p, mat] = local_to_global(xpos_vec[bid], xquat_vec[bid],
                                             row(m.site_pos, i), row(m.site_quat, i));
            sxpos[i] = p;
            sxmat[i] = mat;
        }
        d.site_xpos = mx::stack(sxpos);
        d.site_xmat = mx::stack(sxmat);
    }

    return d;
}

// ── Center of Mass ───────────────────────────────────────────────────────────

Data com_pos(const Model& m, Data d) {
    mx::eval(m.body_parentid); mx::eval(m.body_mass); mx::eval(m.body_inertia);
    mx::eval(m.body_rootid); mx::eval(m.dof_bodyid); mx::eval(m.jnt_bodyid);
    mx::eval(m.jnt_type); mx::eval(m.jnt_dofadr);
    mx::eval(d.xipos); mx::eval(d.ximat); mx::eval(d.xmat);

    auto parent_ptr = m.body_parentid.data<int>();
    auto mass_ptr = m.body_mass.data<float>();
    auto rootid_ptr = m.body_rootid.data<int>();
    auto dof_bodyid_ptr = m.dof_bodyid.data<int>();

    // Subtree COM: accumulate from leaves to root
    std::vector<mx::array> subtree_pos; subtree_pos.reserve(m.nbody);
    std::vector<float> subtree_mass(m.nbody, 0.0f);

    for (int i = 0; i < m.nbody; i++) {
        subtree_pos.push_back(mx::multiply(row(d.xipos, i), mx::array(mass_ptr[i])));
        subtree_mass[i] = mass_ptr[i];
    }

    for (int i = m.nbody - 1; i > 0; i--) {
        int pid = parent_ptr[i];
        subtree_pos[pid] = mx::add(subtree_pos[pid], subtree_pos[i]);
        subtree_mass[pid] += subtree_mass[i];
    }

    std::vector<mx::array> subtree_com_vec; subtree_com_vec.reserve(m.nbody);
    for (int i = 0; i < m.nbody; i++) {
        float sm = std::max(subtree_mass[i], MJMINVAL);
        subtree_com_vec.push_back(mx::divide(subtree_pos[i], mx::array(sm)));
    }
    d.subtree_com = mx::stack(subtree_com_vec);

    // cinert: inertia in subtree-COM frame with full rotation + parallel axis theorem
    std::vector<mx::array> cinert_vec; cinert_vec.reserve(m.nbody);
    for (int i = 0; i < m.nbody; i++) {
        int rid = rootid_ptr[i];
        auto offset = mx::subtract(row(d.xipos, i), subtree_com_vec[rid]);
        auto inertia = row(m.body_inertia, i);
        float mi = mass_ptr[i];

        mx::eval(inertia); mx::eval(offset);

        // Rotate diagonal inertia to global frame: I_global = R @ diag(I) @ R^T
        // ximat is 3x3 rotation matrix
        auto R = mx::slice(d.ximat, {i, 0, 0}, {i + 1, 3, 3});
        R = mx::reshape(R, {3, 3});
        auto diag_I = mx::diag(inertia);
        auto I_global = mx::matmul(mx::matmul(R, diag_I), mx::transpose(R));

        // Parallel axis theorem: I += m * (|d|^2 * I3 - d d^T)
        mx::eval(offset);
        auto d2 = mx::sum(mx::multiply(offset, offset));
        auto outer = mx::matmul(mx::reshape(offset, {3, 1}), mx::reshape(offset, {1, 3}));
        auto pat = mx::multiply(mx::array(mi),
                                mx::subtract(mx::multiply(d2, mx::eye(3)), outer));
        I_global = mx::add(I_global, pat);

        // Extract components: (Ixx, Iyy, Izz, Ixy, Ixz, Iyz, px*m, py*m, pz*m, m)
        mx::eval(I_global);
        auto Ip = I_global.data<float>();
        float Ixx = Ip[0], Iyy = Ip[4], Izz = Ip[8];
        float Ixy = Ip[1], Ixz = Ip[2], Iyz = Ip[5];

        auto pm = mx::multiply(offset, mx::array(mi));
        mx::eval(pm);
        auto pmp = pm.data<float>();

        cinert_vec.push_back(mx::array({Ixx, Iyy, Izz, Ixy, Ixz, Iyz,
                                         pmp[0], pmp[1], pmp[2], mi}));
    }
    d.cinert = mx::stack(cinert_vec);

    // cdof: motion DOFs in global COM frame
    mx::eval(m.jnt_dofadr);
    auto jnt_bodyid_ptr = m.jnt_bodyid.data<int>();
    auto jnt_type_ptr = m.jnt_type.data<int>();
    auto jnt_dofadr_ptr = m.jnt_dofadr.data<int>();

    std::vector<mx::array> cdof_vec;
    for (int i = 0; i < m.nv; i++) cdof_vec.push_back(mx::zeros({6}));

    for (int dof_i = 0; dof_i < m.nv; dof_i++) {
        int bid = dof_bodyid_ptr[dof_i];
        int rid = rootid_ptr[bid];

        // Find which joint and local DOF offset
        int jt = -1, local_dof = 0;
        mx::array anchor = mx::zeros({3});
        mx::array axis = mx::array({0.0f, 0.0f, 1.0f});

        for (int ji = 0; ji < m.njnt; ji++) {
            if (jnt_bodyid_ptr[ji] != bid) continue;
            int jt_val = jnt_type_ptr[ji];
            int da = jnt_dofadr_ptr[ji];
            int dof_width = (jt_val == 0) ? 6 : (jt_val == 1) ? 3 : 1; // FREE=6, BALL=3, else=1
            if (da <= dof_i && dof_i < da + dof_width) {
                jt = jt_val;
                local_dof = dof_i - da;
                anchor = row(d.xanchor, ji);
                axis = row(d.xaxis, ji);
                break;
            }
        }

        if (jt < 0) continue;

        auto root_com = subtree_com_vec[rid];
        auto offset = mx::subtract(root_com, anchor);

        if (jt == static_cast<int>(JointType::FREE)) {
            if (local_dof < 3) {
                // Translation DOF
                std::vector<float> cdof_data(6, 0.0f);
                cdof_data[3 + local_dof] = 1.0f;
                cdof_vec[dof_i] = mx::array(cdof_data.data(), {6}, mx::float32);
            } else {
                // Rotation DOF
                int rot_dof = local_dof - 3;
                // Use xmat column as rotation axis
                auto xmat_body = mx::slice(d.xmat, {bid, 0, 0}, {bid + 1, 3, 3});
                xmat_body = mx::reshape(xmat_body, {3, 3});
                auto a = mx::flatten(mx::slice(xmat_body, {0, rot_dof}, {3, rot_dof + 1}));
                cdof_vec[dof_i] = mx::concatenate({a, cross(a, offset)}, 0);
            }
        } else if (jt == static_cast<int>(JointType::BALL)) {
            auto xmat_body = mx::slice(d.xmat, {bid, 0, 0}, {bid + 1, 3, 3});
            xmat_body = mx::reshape(xmat_body, {3, 3});
            auto a = mx::flatten(mx::slice(xmat_body, {0, local_dof}, {3, local_dof + 1}));
            cdof_vec[dof_i] = mx::concatenate({a, cross(a, offset)}, 0);
        } else if (jt == static_cast<int>(JointType::HINGE)) {
            cdof_vec[dof_i] = mx::concatenate({axis, cross(axis, offset)}, 0);
        } else if (jt == static_cast<int>(JointType::SLIDE)) {
            cdof_vec[dof_i] = mx::concatenate({mx::zeros({3}), axis}, 0);
        }
    }

    d.cdof = mx::stack(cdof_vec);
    return d;
}

// ── Composite Rigid Body ─────────────────────────────────────────────────────

Data crb(const Model& m, Data d) {
    mx::eval(m.body_parentid); mx::eval(m.dof_bodyid);
    auto parent_ptr = m.body_parentid.data<int>();
    auto dof_bodyid_ptr = m.dof_bodyid.data<int>();

    // Accumulate CRB from leaves to root
    std::vector<mx::array> crb_body;
    for (int i = 0; i < m.nbody; i++) {
        crb_body.push_back(row(d.cinert, i));
    }

    for (int i = m.nbody - 1; i > 0; i--) {
        int pid = parent_ptr[i];
        crb_body[pid] = mx::add(crb_body[pid], crb_body[i]);
    }
    crb_body[0] = mx::zeros({10});

    d.crb = mx::stack(crb_body);

    // Compute mass matrix: qM = crb_cdof^T @ cdof (with tree mask)
    std::vector<mx::array> crb_cdof_vec;
    for (int i = 0; i < m.nv; i++) {
        int bid = dof_bodyid_ptr[i];
        crb_cdof_vec.push_back(inert_mul(crb_body[bid], row(d.cdof, i)));
    }
    auto crb_cdof = mx::stack(crb_cdof_vec);

    d.qM = make_m(m, crb_cdof, d.cdof, m.dof_armature);
    return d;
}

// ── Mass matrix factorization ────────────────────────────────────────────────

Data factor_m(const Model& m, Data d) {
    if (!is_sparse(m)) {
        // Dense Cholesky
        auto qM_reg = mx::add(d.qM, mx::multiply(mx::eye(m.nv), mx::array(1e-8f)));
        d.qLD = mx::linalg::cholesky(qM_reg, false, mx::Device::cpu);
        return d;
    }

    // Sparse LDL factorization
    mx::eval(m.dof_parentid); mx::eval(m.dof_Madr); mx::eval(d.qM);
    auto parent_ptr = m.dof_parentid.data<int>();
    auto madr_ptr = m.dof_Madr.data<int>();

    // Copy qM to work vector
    int qm_size = d.qM.size();
    auto qm_data = std::vector<float>(d.qM.data<float>(), d.qM.data<float>() + qm_size);

    // LDL factorization
    for (int i = m.nv - 1; i >= 0; i--) {
        int madr_i = madr_ptr[i];
        float diag_i = qm_data[madr_i];
        if (std::abs(diag_i) < MJMINVAL) diag_i = MJMINVAL;

        int madr_ij = madr_i;
        int j = i;
        while (true) {
            madr_ij++;
            j = parent_ptr[j];
            if (j == -1) break;

            float val = qm_data[madr_ij];
            float scale = val / diag_i;

            // Update parent diagonal and off-diagonals
            int madr_j = madr_ptr[j];
            int madr_jk = madr_j;
            int k_ij = madr_ij;
            int kk = j;

            // Subtract outer product contribution
            int madr_ik = madr_ij;
            int ii = j;
            while (true) {
                int madr_jk2 = madr_j;
                int jj = j;
                while (jj >= ii) {
                    // Find matching pair
                    if (jj == ii) {
                        qm_data[madr_jk2] -= scale * qm_data[madr_ik];
                        break;
                    }
                    madr_jk2++;
                    jj = parent_ptr[jj];
                }
                madr_ik++;
                ii = parent_ptr[ii];
                if (ii == -1) break;
            }

            // Store L factor (normalized by diagonal)
            qm_data[madr_ij] = scale;
        }
    }

    // Extract diagonal inverse
    std::vector<float> diag_inv(m.nv);
    for (int i = 0; i < m.nv; i++) {
        float diag = qm_data[madr_ptr[i]];
        diag_inv[i] = (std::abs(diag) > MJMINVAL) ? 1.0f / diag : 0.0f;
    }

    d.qLD = mx::array(qm_data.data(), {qm_size}, mx::float32);
    d.qLDiagInv = mx::array(diag_inv.data(), {m.nv}, mx::float32);
    return d;
}

mx::array solve_m(const Model& m, const Data& d, const mx::array& rhs) {
    if (!is_sparse(m)) {
        // Dense: Cholesky solve L L^T x = rhs
        auto y = mx::linalg::solve_triangular(d.qLD, mx::reshape(rhs, {m.nv, 1}), false, mx::Device::cpu);
        auto x = mx::linalg::solve_triangular(mx::transpose(d.qLD), y, true, mx::Device::cpu);
        return mx::flatten(x);
    }

    // Sparse LDL solve
    mx::eval(m.dof_parentid); mx::eval(m.dof_Madr);
    mx::eval(d.qLD); mx::eval(d.qLDiagInv); mx::eval(rhs);

    auto parent_ptr = m.dof_parentid.data<int>();
    auto madr_ptr = m.dof_Madr.data<int>();
    auto ld_ptr = d.qLD.data<float>();
    auto dinv_ptr = d.qLDiagInv.data<float>();
    auto rhs_ptr = rhs.data<float>();

    std::vector<float> x(rhs_ptr, rhs_ptr + m.nv);

    // Forward substitution: L^T x = rhs
    for (int i = m.nv - 1; i >= 0; i--) {
        int madr_ij = madr_ptr[i];
        int j = i;
        while (true) {
            madr_ij++;
            j = parent_ptr[j];
            if (j == -1) break;
            x[j] -= ld_ptr[madr_ij] * x[i];
        }
    }

    // Diagonal: D^-1 x
    for (int i = 0; i < m.nv; i++) {
        x[i] *= dinv_ptr[i];
    }

    // Backward substitution: L x
    for (int i = 0; i < m.nv; i++) {
        int madr_ij = madr_ptr[i];
        int j = i;
        while (true) {
            madr_ij++;
            j = parent_ptr[j];
            if (j == -1) break;
            x[i] -= ld_ptr[madr_ij] * x[j];
        }
    }

    return mx::array(x.data(), {m.nv}, mx::float32);
}

// ── Velocity-dependent ───────────────────────────────────────────────────────

Data com_vel(const Model& m, Data d) {
    mx::eval(m.body_parentid); mx::eval(m.dof_bodyid);
    mx::eval(d.cdof); mx::eval(d.qvel);

    auto parent_ptr = m.body_parentid.data<int>();
    auto dof_bodyid_ptr = m.dof_bodyid.data<int>();

    std::vector<mx::array> cvel_vec;
    for (int i = 0; i < m.nbody; i++) cvel_vec.push_back(mx::zeros({6}));
    std::vector<mx::array> cdof_dot_vec;
    for (int i = 0; i < m.nv; i++) cdof_dot_vec.push_back(mx::zeros({6}));

    for (int body_id = 1; body_id < m.nbody; body_id++) {
        int pid = parent_ptr[body_id];
        auto cv = cvel_vec[pid];

        for (int di = 0; di < m.nv; di++) {
            if (dof_bodyid_ptr[di] != body_id) continue;
            cdof_dot_vec[di] = motion_cross(cv, row(d.cdof, di));
            auto qvel_i = mx::slice(d.qvel, {di}, {di + 1});
            cv = mx::add(cv, mx::multiply(row(d.cdof, di), mx::flatten(qvel_i)));
        }

        cvel_vec[body_id] = cv;
    }

    d.cvel = mx::stack(cvel_vec);
    d.cdof_dot = (m.nv > 0) ? mx::stack(cdof_dot_vec) : mx::zeros({0, 6});
    return d;
}

// ── Recursive Newton-Euler ───────────────────────────────────────────────────

Data rne(const Model& m, Data d, bool flg_acc) {
    mx::eval(m.body_parentid); mx::eval(m.dof_bodyid);
    mx::eval(d.cdof); mx::eval(d.cdof_dot); mx::eval(d.qvel);
    mx::eval(d.cinert); mx::eval(d.cvel);

    auto parent_ptr = m.body_parentid.data<int>();
    auto dof_bodyid_ptr = m.dof_bodyid.data<int>();

    // Forward pass: compute cacc (link accelerations)
    mx::array gravity_acc = mx::zeros({6});
    if (!(m.opt.disableflags & DisableBit::GRAVITY)) {
        gravity_acc = mx::concatenate({mx::zeros({3}), mx::negative(m.opt.gravity)}, 0);
    }

    std::vector<mx::array> cacc;
    cacc.push_back(gravity_acc);
    for (int i = 1; i < m.nbody; i++) cacc.push_back(mx::zeros({6}));

    for (int body_id = 1; body_id < m.nbody; body_id++) {
        int pid = parent_ptr[body_id];
        auto acc = cacc[pid];

        for (int di = 0; di < m.nv; di++) {
            if (dof_bodyid_ptr[di] != body_id) continue;
            auto qvel_i = mx::flatten(mx::slice(d.qvel, {di}, {di + 1}));
            acc = mx::add(acc, mx::multiply(row(d.cdof_dot, di), qvel_i));
            if (flg_acc) {
                auto qacc_i = mx::flatten(mx::slice(d.qacc, {di}, {di + 1}));
                acc = mx::add(acc, mx::multiply(row(d.cdof, di), qacc_i));
            }
        }
        cacc[body_id] = acc;
    }

    // Compute local forces: f = I*a + v x (I*v)
    std::vector<mx::array> cfrc;
    for (int i = 0; i < m.nbody; i++) {
        auto frc = inert_mul(row(d.cinert, i), cacc[i]);
        auto Iv = inert_mul(row(d.cinert, i), row(d.cvel, i));
        frc = mx::add(frc, motion_cross_force(row(d.cvel, i), Iv));
        cfrc.push_back(frc);
    }

    // Backward pass: accumulate forces
    for (int i = m.nbody - 1; i > 0; i--) {
        int pid = parent_ptr[i];
        cfrc[pid] = mx::add(cfrc[pid], cfrc[i]);
    }

    // Project to joint space: qfrc_bias[i] = cdof[i] . cfrc[bodyid[i]]
    std::vector<float> qfrc_data(m.nv, 0.0f);
    for (int di = 0; di < m.nv; di++) {
        int bid = dof_bodyid_ptr[di];
        auto dot = mx::sum(mx::multiply(row(d.cdof, di), cfrc[bid]));
        mx::eval(dot);
        qfrc_data[di] = dot.item<float>();
    }
    d.qfrc_bias = mx::array(qfrc_data.data(), {m.nv}, mx::float32);
    return d;
}

// ── Transmission ─────────────────────────────────────────────────────────────

Data transmission(const Model& m, Data d) {
    if (m.nu == 0) return d;

    mx::eval(m.actuator_trntype); mx::eval(m.actuator_trnid);
    mx::eval(m.jnt_type); mx::eval(m.jnt_dofadr); mx::eval(m.jnt_qposadr);
    mx::eval(m.actuator_gear); mx::eval(d.qpos);

    auto trntype_ptr = m.actuator_trntype.data<int>();
    auto jnt_type_ptr = m.jnt_type.data<int>();
    auto jnt_dofadr_ptr = m.jnt_dofadr.data<int>();
    auto jnt_qposadr_ptr = m.jnt_qposadr.data<int>();

    std::vector<mx::array> lengths;
    for (int i = 0; i < m.nu; i++) lengths.push_back(mx::array({0.0f}));
    std::vector<float> moment_data(m.nu * m.nv, 0.0f);

    mx::eval(m.actuator_trnid);
    auto trnid_ptr = m.actuator_trnid.data<int>();

    for (int i = 0; i < m.nu; i++) {
        int trntype = trntype_ptr[i];

        if (trntype == 0) {  // JOINT transmission
            int jnt_id = trnid_ptr[i * 2];  // trnid is (njnt, 2)
            int jt = jnt_type_ptr[jnt_id];
            int da = jnt_dofadr_ptr[jnt_id];

            if (jt == static_cast<int>(JointType::FREE)) {
                lengths[i] = mx::array({0.0f});
                mx::eval(m.actuator_gear);
                auto gear = m.actuator_gear.data<float>();
                for (int k = 0; k < 6; k++) {
                    moment_data[i * m.nv + da + k] = gear[i * 6 + k];
                }
            } else if (jt == static_cast<int>(JointType::BALL)) {
                lengths[i] = mx::array({0.0f});
                mx::eval(m.actuator_gear);
                auto gear = m.actuator_gear.data<float>();
                for (int k = 0; k < 3; k++) {
                    moment_data[i * m.nv + da + k] = gear[i * 6 + k];
                }
            } else {
                // HINGE or SLIDE
                int qa = jnt_qposadr_ptr[jnt_id];
                mx::eval(m.actuator_gear);
                auto gear = m.actuator_gear.data<float>();
                float g0 = gear[i * 6];
                auto qp = mx::slice(d.qpos, {qa}, {qa + 1});
                lengths[i] = mx::multiply(qp, mx::array(g0));
                moment_data[i * m.nv + da] = g0;
            }
        } else {
            lengths[i] = mx::array({0.0f});
        }
    }

    d.actuator_length = mx::concatenate(lengths, 0);
    d.actuator_moment = mx::array(moment_data.data(), {m.nu, m.nv}, mx::float32);
    return d;
}

} // namespace mjmlx
