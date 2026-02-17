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

// Passive forces: springs and dampers.
// Port of Python mjmlx._src.passive.

#include "internal.h"

namespace mjmlx {

// DisableBit flags (matching MuJoCo C and Python types.py)
static constexpr int DISABLE_SPRING = (1 << 4);
static constexpr int DISABLE_DAMPER = (1 << 6);
static constexpr int DISABLE_GRAVITY = (1 << 7);

// Forward declarations from math.cpp
mx::array quat_mul(const mx::array& q1, const mx::array& q2);
mx::array quat_inv(const mx::array& q);

static mx::array quat_sub(const mx::array& qa, const mx::array& qb) {
    // Difference in quaternion-space -> 3D rotation vector
    // quat_sub(qa, qb) returns the xyz components of qa * conj(qb)
    auto conj_b = quat_inv(qb);
    auto diff = quat_mul(qa, conj_b);
    // Return xyz (imaginary) components as the "error"
    return mx::slice(diff, {1}, {4});
}

static mx::array spring_damper(const Model& m, const Data& d) {
    auto qfrc = mx::zeros({m.nv});

    // Extract model arrays to evaluate them
    mx::eval(m.jnt_type);
    mx::eval(m.jnt_qposadr);
    mx::eval(m.jnt_dofadr);

    // Spring forces
    if (!(m.opt.disableflags & DISABLE_SPRING) && m.jnt_stiffness.size() > 0) {
        mx::eval(m.jnt_stiffness);
        mx::eval(d.qpos);
        mx::eval(m.qpos_spring);

        for (int j = 0; j < m.njnt; j++) {
            int jt = m.jnt_type.data<int>()[j];
            int qa = m.jnt_qposadr.data<int>()[j];
            int da = m.jnt_dofadr.data<int>()[j];
            float stiff = m.jnt_stiffness.data<float>()[j];

            if (stiff == 0.0f) continue;

            if (jt == static_cast<int>(JointType::FREE)) {
                // Translation springs
                for (int k = 0; k < 3; k++) {
                    auto delta = mx::subtract(
                        mx::slice(d.qpos, {qa + k}, {qa + k + 1}),
                        mx::slice(m.qpos_spring, {qa + k}, {qa + k + 1})
                    );
                    auto force = mx::multiply(mx::array(-stiff), delta);
                    auto before = mx::slice(qfrc, {0}, {da + k});
                    auto at = mx::add(mx::slice(qfrc, {da + k}, {da + k + 1}), force);
                    auto after = mx::slice(qfrc, {da + k + 1}, {m.nv});
                    qfrc = mx::concatenate({before, at, after}, 0);
                }
                // Rotation springs
                auto q_cur = mx::slice(d.qpos, {qa + 3}, {qa + 7});
                auto q_ref = mx::slice(m.qpos_spring, {qa + 3}, {qa + 7});
                auto q_err = quat_sub(q_cur, q_ref);
                for (int k = 0; k < 3; k++) {
                    auto force = mx::multiply(mx::array(-stiff), mx::slice(q_err, {k}, {k + 1}));
                    auto before = mx::slice(qfrc, {0}, {da + 3 + k});
                    auto at = mx::add(mx::slice(qfrc, {da + 3 + k}, {da + 3 + k + 1}), force);
                    auto after = mx::slice(qfrc, {da + 3 + k + 1}, {m.nv});
                    qfrc = mx::concatenate({before, at, after}, 0);
                }
            } else if (jt == static_cast<int>(JointType::BALL)) {
                auto q_cur = mx::slice(d.qpos, {qa}, {qa + 4});
                auto q_ref = mx::slice(m.qpos_spring, {qa}, {qa + 4});
                auto q_err = quat_sub(q_cur, q_ref);
                for (int k = 0; k < 3; k++) {
                    auto force = mx::multiply(mx::array(-stiff), mx::slice(q_err, {k}, {k + 1}));
                    auto before = mx::slice(qfrc, {0}, {da + k});
                    auto at = mx::add(mx::slice(qfrc, {da + k}, {da + k + 1}), force);
                    auto after = mx::slice(qfrc, {da + k + 1}, {m.nv});
                    qfrc = mx::concatenate({before, at, after}, 0);
                }
            } else {
                // SLIDE or HINGE
                auto delta = mx::subtract(
                    mx::slice(d.qpos, {qa}, {qa + 1}),
                    mx::slice(m.qpos_spring, {qa}, {qa + 1})
                );
                auto force = mx::multiply(mx::array(-stiff), delta);
                auto before = mx::slice(qfrc, {0}, {da});
                auto at = mx::add(mx::slice(qfrc, {da}, {da + 1}), force);
                auto after = mx::slice(qfrc, {da + 1}, {m.nv});
                qfrc = mx::concatenate({before, at, after}, 0);
            }
        }
    }

    // Damping forces: qfrc -= dof_damping * qvel
    if (!(m.opt.disableflags & DISABLE_DAMPER)) {
        qfrc = mx::subtract(qfrc, mx::multiply(m.dof_damping, d.qvel));
    }

    return qfrc;
}

// Gravity compensation: for each body with gravcomp != 0,
// apply an upward force = -(mass * gravcomp * gravity) at the body COM.
// Matches MuJoCo C engine_passive.c :: mj_gravcomp.
static mx::array gravcomp(const Model& m, const Data& d) {
    auto qfrc = mx::zeros({m.nv});

    if (m.ngravcomp == 0) return qfrc;
    if (m.opt.disableflags & DISABLE_GRAVITY) return qfrc;

    mx::eval(m.body_gravcomp);
    mx::eval(m.body_mass);
    mx::eval(m.opt.gravity);
    mx::eval(d.xipos);

    auto gc_ptr = m.body_gravcomp.data<float>();
    auto mass_ptr = m.body_mass.data<float>();

    for (int i = 1; i < m.nbody; i++) {
        float gc = gc_ptr[i];
        if (gc == 0.0f) continue;

        float mass_i = mass_ptr[i];

        // force = -(mass * gravcomp) * gravity  (3D vector)
        auto force = mx::multiply(mx::array(-(mass_i * gc)), m.opt.gravity);

        // Jacobian at body COM (xipos)
        auto xipos_i = mx::flatten(mx::slice(d.xipos, {i, 0}, {i + 1, 3}));
        auto [jacp, jacr] = jac(m, d, xipos_i, i);

        // qfrc_gravcomp += J_p^T * force  =>  (nv,3) @ (3,1) -> (nv,1) -> (nv,)
        auto contrib = mx::flatten(mx::matmul(jacp, mx::reshape(force, {3, 1})));
        qfrc = mx::add(qfrc, contrib);
    }

    return qfrc;
}

Data passive(const Model& m, Data d) {
    d.qfrc_passive = spring_damper(m, d);

    // Gravity compensation
    d.qfrc_gravcomp = gravcomp(m, d);

    // Add gravcomp to passive forces (matching MuJoCo C behavior:
    // qfrc_passive += qfrc_gravcomp for joints without actgravcomp)
    if (m.ngravcomp > 0) {
        d.qfrc_passive = mx::add(d.qfrc_passive, d.qfrc_gravcomp);
    }

    return d;
}

} // namespace mjmlx
