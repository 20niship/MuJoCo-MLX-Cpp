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

// Engine support functions: coordinate transforms, mass matrix operations,
// Jacobians, and force accumulation.
// Port of Python mjmlx._src.support.

#include "internal.h"

namespace mjmlx {

// Forward declarations from math.cpp
mx::array cross(const mx::array& a, const mx::array& b);
mx::array quat_mul(const mx::array& q1, const mx::array& q2);
mx::array quat_to_mat(const mx::array& q);
mx::array rotate(const mx::array& vec, const mx::array& quat);

static constexpr float MJMINVAL = 1e-15f;

bool is_sparse(const Model& m) {
    return m.nv >= 60;
}

std::pair<mx::array, mx::array> local_to_global(
    const mx::array& world_pos, const mx::array& world_quat,
    const mx::array& local_pos, const mx::array& local_quat)
{
    auto pos = mx::add(world_pos, rotate(local_pos, world_quat));
    auto mat = quat_to_mat(quat_mul(world_quat, local_quat));
    return {pos, mat};
}

mx::array make_m(const Model& m, const mx::array& a, const mx::array& b,
                 const mx::array& d_diag)
{
    mx::eval(m.dof_parentid);
    auto parent_ptr = m.dof_parentid.data<int>();

    // Build tree connectivity (i,j) pairs
    std::vector<std::pair<int,int>> ij;
    for (int i = 0; i < m.nv; i++) {
        int j = i;
        while (j > -1) {
            ij.push_back({i, j});
            j = parent_ptr[j];
        }
    }

    if (!is_sparse(m)) {
        // Dense: qM = a @ b^T
        auto qm = mx::matmul(a, mx::transpose(b));

        if (d_diag.size() > 0) {
            qm = mx::add(qm, mx::diag(d_diag));
        }

        // Build mask on CPU, then convert to MLX
        std::vector<float> mask_data(m.nv * m.nv, 0.0f);
        for (auto& [ii, jj] : ij) {
            mask_data[ii * m.nv + jj] = 1.0f;
        }
        auto mask = mx::array(mask_data.data(), {m.nv, m.nv}, mx::float32);

        qm = mx::multiply(qm, mask);
        // Symmetrize: qM = qM + tril(qM, -1)^T
        auto lower = mx::tril(qm, -1);
        qm = mx::add(qm, mx::transpose(lower));
        return qm;
    }

    // Sparse representation
    int nnz = static_cast<int>(ij.size());
    std::vector<int> i_idx(nnz), j_idx(nnz);
    for (int k = 0; k < nnz; k++) {
        i_idx[k] = ij[k].first;
        j_idx[k] = ij[k].second;
    }

    auto i_arr = mx::array(i_idx.data(), {nnz}, mx::int32);
    auto j_arr = mx::array(j_idx.data(), {nnz}, mx::int32);

    auto a_i = mx::take(a, i_arr, 0);
    auto b_j = mx::take(b, j_arr, 0);

    auto qm = mx::sum(mx::multiply(a_i, b_j), /* axis */ -1);

    if (d_diag.size() > 0) {
        mx::eval(m.dof_Madr);
        auto madr_ptr = m.dof_Madr.data<int>();
        std::vector<int> diag_idx(m.nv);
        for (int k = 0; k < m.nv; k++) diag_idx[k] = madr_ptr[k];
        auto diag_idx_arr = mx::array(diag_idx.data(), {m.nv}, mx::int32);

        // Scatter-add d_diag into qm at diag indices
        mx::eval(qm);
        auto qm_vec = std::vector<float>(qm.data<float>(), qm.data<float>() + nnz);
        mx::eval(d_diag);
        auto diag_vec = d_diag.data<float>();
        for (int k = 0; k < m.nv; k++) {
            qm_vec[diag_idx[k]] += diag_vec[k];
        }
        qm = mx::array(qm_vec.data(), {nnz}, mx::float32);
    }

    return qm;
}

mx::array full_m(const Model& m, const Data& d) {
    if (!is_sparse(m)) {
        return d.qM;
    }

    mx::eval(m.dof_parentid);
    auto parent_ptr = m.dof_parentid.data<int>();

    std::vector<std::pair<int,int>> ij;
    for (int i = 0; i < m.nv; i++) {
        int j = i;
        while (j > -1) {
            ij.push_back({i, j});
            j = parent_ptr[j];
        }
    }

    mx::eval(d.qM);
    auto qm_ptr = d.qM.data<float>();

    std::vector<float> mat_data(m.nv * m.nv, 0.0f);
    for (int k = 0; k < static_cast<int>(ij.size()); k++) {
        auto [ii, jj] = ij[k];
        mat_data[ii * m.nv + jj] += qm_ptr[k];
    }

    auto mat = mx::array(mat_data.data(), {m.nv, m.nv}, mx::float32);
    // Symmetrize
    mat = mx::add(mat, mx::transpose(mx::tril(mat, -1)));
    return mat;
}

mx::array mul_m(const Model& m, const Data& d, const mx::array& vec) {
    if (!is_sparse(m)) {
        return mx::flatten(mx::matmul(d.qM, mx::reshape(vec, {m.nv, 1})));
    }

    mx::eval(m.dof_parentid);
    mx::eval(m.dof_Madr);
    mx::eval(d.qM);
    mx::eval(vec);

    auto parent_ptr = m.dof_parentid.data<int>();
    auto madr_ptr = m.dof_Madr.data<int>();
    auto qm_ptr = d.qM.data<float>();
    auto vec_ptr = vec.data<float>();

    // Compute M @ vec on CPU (sparse M is already evaluated)
    std::vector<float> out_data(m.nv, 0.0f);

    // Diagonal
    for (int i = 0; i < m.nv; i++) {
        out_data[i] = qm_ptr[madr_ptr[i]] * vec_ptr[i];
    }

    // Off-diagonal
    for (int i = 0; i < m.nv; i++) {
        int madr = madr_ptr[i];
        int j = i;
        while (true) {
            madr++;
            j = parent_ptr[j];
            if (j == -1) break;
            float qm_val = qm_ptr[madr];
            out_data[i] += qm_val * vec_ptr[j];
            out_data[j] += qm_val * vec_ptr[i];
        }
    }

    return mx::array(out_data.data(), {m.nv}, mx::float32);
}

mx::array xfrc_accumulate(const Model& m, const Data& d) {
    auto qfrc = mx::zeros({m.nv});

    mx::eval(d.xfrc_applied);

    for (int body_id = 0; body_id < m.nbody; body_id++) {
        // Slice row for this body: (6,)
        auto frc = mx::flatten(mx::slice(d.xfrc_applied, {body_id, 0}, {body_id + 1, 6}));

        // Quick magnitude check
        mx::eval(frc);
        auto abs_sum = mx::sum(mx::abs(frc));
        mx::eval(abs_sum);
        if (abs_sum.item<float>() < MJMINVAL) continue;

        auto force = mx::slice(frc, {0}, {3});
        auto torque = mx::slice(frc, {3}, {6});

        auto point = mx::flatten(mx::slice(d.xipos, {body_id, 0}, {body_id + 1, 3}));
        auto [jacp, jacr] = jac(m, d, point, body_id);

        auto jacp_force = mx::matmul(mx::transpose(jacp), mx::reshape(force, {3, 1}));
        auto jacr_torque = mx::matmul(mx::transpose(jacr), mx::reshape(torque, {3, 1}));
        qfrc = mx::add(qfrc, mx::add(mx::flatten(jacp_force), mx::flatten(jacr_torque)));
    }

    return qfrc;
}

std::pair<mx::array, mx::array> jac(
    const Model& m, const Data& d,
    const mx::array& point, int body_id)
{
    mx::eval(m.dof_bodyid);
    mx::eval(m.body_parentid);
    mx::eval(m.body_rootid);
    mx::eval(d.subtree_com);
    mx::eval(d.cdof);

    auto dof_bodyid_ptr = m.dof_bodyid.data<int>();
    auto body_parentid_ptr = m.body_parentid.data<int>();
    auto body_rootid_ptr = m.body_rootid.data<int>();

    // Find ancestors of body_id
    std::set<int> ancestors;
    int bid = body_id;
    while (bid > 0) {
        ancestors.insert(bid);
        bid = body_parentid_ptr[bid];
    }
    ancestors.insert(0);

    int root_id = body_rootid_ptr[body_id];
    auto subtree_com_root = mx::flatten(mx::slice(d.subtree_com, {root_id, 0}, {root_id + 1, 3}));
    auto offset = mx::subtract(point, subtree_com_root);

    // Build Jacobian on CPU: for each DOF, compute contribution
    std::vector<float> jacp_data(m.nv * 3, 0.0f);
    std::vector<float> jacr_data(m.nv * 3, 0.0f);

    mx::eval(offset);

    for (int i = 0; i < m.nv; i++) {
        int dof_body = dof_bodyid_ptr[i];
        if (ancestors.find(dof_body) == ancestors.end()) continue;

        auto cdof_i = mx::flatten(mx::slice(d.cdof, {i, 0}, {i + 1, 6}));
        auto cdof_ang = mx::slice(cdof_i, {0}, {3});
        auto cdof_lin = mx::slice(cdof_i, {3}, {6});

        auto jacp_i = mx::add(cdof_lin, cross(cdof_ang, offset));
        mx::eval(jacp_i);
        mx::eval(cdof_ang);

        auto jp = jacp_i.data<float>();
        auto jr = cdof_ang.data<float>();
        jacp_data[i * 3 + 0] = jp[0];
        jacp_data[i * 3 + 1] = jp[1];
        jacp_data[i * 3 + 2] = jp[2];
        jacr_data[i * 3 + 0] = jr[0];
        jacr_data[i * 3 + 1] = jr[1];
        jacr_data[i * 3 + 2] = jr[2];
    }

    auto jacp = mx::array(jacp_data.data(), {m.nv, 3}, mx::float32);
    auto jacr = mx::array(jacr_data.data(), {m.nv, 3}, mx::float32);
    return {jacp, jacr};
}

} // namespace mjmlx
