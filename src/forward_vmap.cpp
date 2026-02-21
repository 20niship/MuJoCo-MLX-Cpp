// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// Vmap-compatible forward dynamics pipeline.
// Orchestrates smooth_vmap, collision_vmap, constraint_vmap, solver_vmap.
// skip_kinematics=true (Metal kernel handles FK outside vmap).
// NO eval(), NO data<>(), NO CPU sync.

#include "internal.h"

namespace mjmlx {

static constexpr float MJMINVAL_FV = 1e-8f;

// ── Vmap-compatible passive forces ───────────────────────────────────────────

static Data vmap_passive(const Model& m, Data d) {
    auto qfrc = mx::zeros(mx::Shape{m.nv});

    // Vectorized spring forces: -stiffness * (qpos[qa] - qpos_spring[qa])
    if (m.cache.passive_stiffness.size() > 0 && m.njnt > 0) {
        auto qpos_dofs = mx::take(d.qpos, m.cache.passive_qpos_idxs, 0);  // (nv,)
        auto qspring_dofs = mx::take(m.qpos_spring, m.cache.passive_qpos_idxs, 0);  // (nv,)
        auto diff = mx::subtract(qpos_dofs, qspring_dofs);
        qfrc = mx::subtract(qfrc, mx::multiply(m.cache.passive_stiffness, diff));
    }

    // Damper forces: -damping * qvel
    if (m.dof_damping.size() > 0) {
        qfrc = mx::subtract(qfrc, mx::multiply(m.dof_damping, d.qvel));
    }

    // Tendon passive forces: -stiffness * (len - rest) - damping * vel
    if (m.ntendon > 0 && d.ten_length.size() > 0 && d.ten_J.size() > 0) {
        auto ten_len = d.ten_length;   // (ntendon,)
        auto ten_vel = d.ten_velocity; // (ntendon,)

        // Rest length: clamp(ten_length, spring_lo, spring_hi)
        auto spring_lo = mx::flatten(mx::slice(m.tendon_lengthspring, mx::Shape{0, 0}, mx::Shape{m.ntendon, 1}));
        auto spring_hi = mx::flatten(mx::slice(m.tendon_lengthspring, mx::Shape{0, 1}, mx::Shape{m.ntendon, 2}));
        auto rest = mx::clip(ten_len, spring_lo, spring_hi);

        // force = -stiffness * (len - rest) - damping * vel
        auto spring_force = mx::multiply(mx::negative(m.tendon_stiffness),
                                          mx::subtract(ten_len, rest));
        auto damp_force = mx::multiply(mx::negative(m.tendon_damping), ten_vel);
        auto force = mx::add(spring_force, damp_force);

        // Project to joint space: qfrc += ten_J^T @ force
        auto contrib = mx::flatten(mx::matmul(
            mx::transpose(d.ten_J),
            mx::reshape(force, mx::Shape{m.ntendon, 1})));
        qfrc = mx::add(qfrc, contrib);
    }

    // Gravity compensation
    if (m.ngravcomp > 0 && d.qfrc_gravcomp.size() > 0) {
        qfrc = mx::add(qfrc, d.qfrc_gravcomp);
    }

    d.qfrc_passive = qfrc;
    return d;
}

// ── Vmap-compatible actuation ────────────────────────────────────────────────

static Data vmap_fwd_actuation(const Model& m, Data d) {
    if (m.nu == 0 || (m.opt.disableflags & DisableBit::ACTUATION)) {
        d.qfrc_actuator = mx::zeros(mx::Shape{m.nv});
        return d;
    }

    auto ctrl = d.ctrl;

    // Clamp control
    if (!(m.opt.disableflags & DisableBit::CLAMPCTRL) && m.actuator_ctrllimited.size() > 0) {
        auto limited = mx::astype(m.actuator_ctrllimited, mx::float32);
        auto lo = mx::slice(m.actuator_ctrlrange, mx::Shape{0, 0}, mx::Shape{m.nu, 1});
        auto hi = mx::slice(m.actuator_ctrlrange, mx::Shape{0, 1}, mx::Shape{m.nu, 2});
        auto clamped = mx::clip(ctrl, mx::flatten(lo), mx::flatten(hi));
        ctrl = mx::where(mx::greater(limited, mx::array(0.5f)), clamped, ctrl);
    }

    // Activation dynamics: compute act_dot, determine ctrl_act
    auto ctrl_act = ctrl;  // default: use ctrl for stateless actuators
    d.act_dot = mx::zeros(mx::Shape{std::max(m.na, 1)});

    if (m.na > 0 && d.act.size() > 0 && m.cache.act_is_stateful.size() > 0) {
        auto& c = m.cache;
        // Gather activation values for each actuator: act_vals[i] = act[actadr[i]]
        auto act_vals = mx::take(d.act, c.act_adr_safe, 0);  // (nu,)

        // ctrl_act = stateful ? act : ctrl
        ctrl_act = mx::where(mx::greater(c.act_is_stateful, mx::array(0.5f)), act_vals, ctrl);

        // act_dot for FILTER/FILTEREXACT: (ctrl - act) / tau
        auto filter_dot = mx::divide(mx::subtract(ctrl, act_vals), c.act_tau);
        // act_dot for INTEGRATOR: ctrl
        // Combine: act_dot = is_filter * filter_dot + is_integrator * ctrl
        auto per_act_dot = mx::add(
            mx::multiply(c.act_is_filter, filter_dot),
            mx::multiply(c.act_is_integrator, ctrl));

        // Scatter per-actuator act_dot to per-activation-state act_dot
        // For simple case (actnum=1, adr maps 1:1), use scatter via one-hot
        auto act_dot_arr = mx::zeros(mx::Shape{m.na});
        // Build scatter: for each actuator i with stateful dynamics, act_dot[adr[i]] += per_act_dot[i]
        auto scatter_idx = c.act_adr_safe;  // (nu,) int
        auto one_hot = mx::equal(
            mx::reshape(mx::arange(0, m.na, 1, mx::int32), mx::Shape{1, m.na}),
            mx::reshape(scatter_idx, mx::Shape{m.nu, 1}));  // (nu, na) bool
        auto scatter_mat = mx::astype(one_hot, mx::float32);  // (nu, na)
        // act_dot = scatter_mat^T @ per_act_dot  -> (na,)
        auto stateful_dot = mx::multiply(per_act_dot, c.act_is_stateful);
        act_dot_arr = mx::flatten(mx::matmul(
            mx::transpose(scatter_mat),
            mx::reshape(stateful_dot, mx::Shape{m.nu, 1})));

        d.act_dot = act_dot_arr;
    }

    // Gain and bias (vectorized for FIXED/AFFINE)
    auto gain_prm = m.actuator_gainprm;
    auto bias_prm = m.actuator_biasprm;
    auto vel = d.actuator_velocity.size() > 0 ? d.actuator_velocity : mx::zeros(mx::Shape{m.nu});
    auto length = d.actuator_length.size() > 0 ? d.actuator_length : mx::zeros(mx::Shape{m.nu});

    auto gain_fixed = mx::flatten(mx::slice(gain_prm, mx::Shape{0, 0}, mx::Shape{m.nu, 1}));
    auto gain_affine = mx::add(mx::add(
        mx::flatten(mx::slice(gain_prm, mx::Shape{0, 0}, mx::Shape{m.nu, 1})),
        mx::multiply(mx::flatten(mx::slice(gain_prm, mx::Shape{0, 1}, mx::Shape{m.nu, 2})), length)),
        mx::multiply(mx::flatten(mx::slice(gain_prm, mx::Shape{0, 2}, mx::Shape{m.nu, 3})), vel));
    auto gain = gain_fixed; // Default FIXED

    auto bias_affine = mx::add(mx::add(
        mx::flatten(mx::slice(bias_prm, mx::Shape{0, 0}, mx::Shape{m.nu, 1})),
        mx::multiply(mx::flatten(mx::slice(bias_prm, mx::Shape{0, 1}, mx::Shape{m.nu, 2})), length)),
        mx::multiply(mx::flatten(mx::slice(bias_prm, mx::Shape{0, 2}, mx::Shape{m.nu, 3})), vel));
    auto bias = bias_affine;

    auto force = mx::add(mx::multiply(gain, ctrl_act), bias);

    // Clamp force
    if (m.actuator_forcelimited.size() > 0) {
        auto f_limited = mx::astype(m.actuator_forcelimited, mx::float32);
        auto flo = mx::flatten(mx::slice(m.actuator_forcerange, mx::Shape{0, 0}, mx::Shape{m.nu, 1}));
        auto fhi = mx::flatten(mx::slice(m.actuator_forcerange, mx::Shape{0, 1}, mx::Shape{m.nu, 2}));
        auto f_clamped = mx::clip(force, flo, fhi);
        force = mx::where(mx::greater(f_limited, mx::array(0.5f)), f_clamped, force);
    }

    // Map to joint space
    if (d.actuator_moment.size() > 0) {
        d.qfrc_actuator = mx::flatten(mx::matmul(mx::transpose(d.actuator_moment),
                                                   mx::reshape(force, mx::Shape{m.nu, 1})));
    } else {
        d.qfrc_actuator = mx::zeros(mx::Shape{m.nv});
    }

    d.actuator_force = force;
    return d;
}

// ── Vmap-compatible acceleration ─────────────────────────────────────────────

static Data vmap_fwd_acceleration(const Model& m, Data d) {
    auto qfrc_smooth = mx::add(mx::subtract(d.qfrc_passive, d.qfrc_bias),
                                mx::add(d.qfrc_actuator, d.qfrc_applied));
    auto qacc_smooth = vmap_solve_m(m, d, qfrc_smooth);
    d.qfrc_smooth = qfrc_smooth;
    d.qacc_smooth = qacc_smooth;
    return d;
}

// ── Full vmap-compatible forward pipeline ────────────────────────────────────

Data vmap_forward(const Model& m, Data d, bool skip_contacts) {
    d = vmap_com_pos(m, d);
    d = vmap_crb(m, d);
    d = vmap_factor_m(m, d);

    if (skip_contacts) {
        // Contact-free path for large models (nv > 80).
        // Skips collision detection and constraint solving which create O(ngeom² + nv² × iters)
        // intermediate arrays — easily exceeding Metal's 499K buffer limit for nv=237.
        // For imitation learning, contacts are secondary to pose matching.
        d = vmap_tendon(m, d);
        d = vmap_transmission(m, d);
        if (m.nu > 0 && d.actuator_moment.size() > 0) {
            d.actuator_velocity = mx::flatten(mx::matmul(d.actuator_moment,
                                                          mx::reshape(d.qvel, mx::Shape{m.nv, 1})));
        }
        d = vmap_com_vel(m, d);
        d = vmap_passive(m, d);
        d = vmap_rne(m, d);
        d = vmap_fwd_actuation(m, d);
        d = vmap_fwd_acceleration(m, d);
        d.qacc = d.qacc_smooth;
        d.qfrc_constraint = mx::zeros(mx::Shape{m.nv});
    } else {
        d = vmap_tendon(m, d);
        d = vmap_collision(m, d);
        d = vmap_make_constraint(m, d);
        d = vmap_transmission(m, d);
        if (m.nu > 0 && d.actuator_moment.size() > 0) {
            d.actuator_velocity = mx::flatten(mx::matmul(d.actuator_moment,
                                                          mx::reshape(d.qvel, mx::Shape{m.nv, 1})));
        }
        d = vmap_com_vel(m, d);
        d = vmap_passive(m, d);
        d = vmap_rne(m, d);
        d = vmap_fwd_actuation(m, d);
        d = vmap_fwd_acceleration(m, d);
        int nefc_count = d.efc_J.shape(0);
        if (nefc_count == 0) {
            d.qacc = d.qacc_smooth;
            d.qfrc_constraint = mx::zeros(mx::Shape{m.nv});
        } else {
            d = vmap_solve(m, d);
        }
    }

    return d;
}

} // namespace mjmlx
