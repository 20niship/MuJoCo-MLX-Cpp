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

// Math utilities: quaternion ops, spatial algebra, vector norms.
// Port of Python mjmlx._src.math.
// All functions work with mx::array and are vmap/compile/grad compatible.

#include "internal.h"

namespace mjmlx {

// Float32-safe minimum value (MuJoCo C uses 1e-15 for float64)
static constexpr float MJMINVAL = 1e-8f;

// ── Vector utilities ─────────────────────────────────────────

mx::array cross(const mx::array& a, const mx::array& b) {
    auto a1 = mx::slice(a, {1}, {2});
    auto a2 = mx::slice(a, {2}, {3});
    auto a0 = mx::slice(a, {0}, {1});
    auto b1 = mx::slice(b, {1}, {2});
    auto b2 = mx::slice(b, {2}, {3});
    auto b0 = mx::slice(b, {0}, {1});

    auto c0 = mx::subtract(mx::multiply(a1, b2), mx::multiply(a2, b1));
    auto c1 = mx::subtract(mx::multiply(a2, b0), mx::multiply(a0, b2));
    auto c2 = mx::subtract(mx::multiply(a0, b1), mx::multiply(a1, b0));

    return mx::concatenate({c0, c1, c2}, 0);
}

mx::array norm(const mx::array& x) {
    auto sq = mx::multiply(x, x);
    auto s = mx::sum(sq);
    return mx::sqrt(s);
}

mx::array normalize(const mx::array& x) {
    auto n = norm(x);
    auto eps = mx::array(1e-6f);
    auto is_zero = mx::less(n, eps);
    auto safe_n = mx::where(is_zero, mx::array(1.0f), n);
    return mx::divide(x, safe_n);
}

// ── Quaternion operations ────────────────────────────────────
// Convention: [w, x, y, z] (scalar-first, matching MuJoCo)

mx::array quat_mul(const mx::array& u, const mx::array& v) {
    auto u0 = mx::slice(u, {0}, {1});
    auto u1 = mx::slice(u, {1}, {2});
    auto u2 = mx::slice(u, {2}, {3});
    auto u3 = mx::slice(u, {3}, {4});
    auto v0 = mx::slice(v, {0}, {1});
    auto v1 = mx::slice(v, {1}, {2});
    auto v2 = mx::slice(v, {2}, {3});
    auto v3 = mx::slice(v, {3}, {4});

    auto w = mx::subtract(mx::subtract(mx::subtract(
        mx::multiply(u0, v0), mx::multiply(u1, v1)),
        mx::multiply(u2, v2)), mx::multiply(u3, v3));
    auto x = mx::add(mx::subtract(mx::add(
        mx::multiply(u0, v1), mx::multiply(u1, v0)),
        mx::multiply(u3, v2)), mx::multiply(u2, v3));
    auto y = mx::add(mx::add(mx::subtract(
        mx::multiply(u0, v2), mx::multiply(u1, v3)),
        mx::multiply(u2, v0)), mx::multiply(u3, v1));
    auto z = mx::add(mx::subtract(mx::add(
        mx::multiply(u0, v3), mx::multiply(u1, v2)),
        mx::multiply(u2, v1)), mx::multiply(u3, v0));

    return mx::concatenate({w, x, y, z}, 0);
}

mx::array quat_inv(const mx::array& q) {
    auto conj = mx::array({1.0f, -1.0f, -1.0f, -1.0f});
    return mx::multiply(q, conj);
}

mx::array quat_to_mat(const mx::array& q) {
    // Direct computation from quaternion components (avoids outer product indexing)
    auto w = mx::slice(q, {0}, {1});
    auto x = mx::slice(q, {1}, {2});
    auto y = mx::slice(q, {2}, {3});
    auto z = mx::slice(q, {3}, {4});

    auto ww = mx::multiply(w, w);
    auto xx = mx::multiply(x, x);
    auto yy = mx::multiply(y, y);
    auto zz = mx::multiply(z, z);
    auto wx = mx::multiply(w, x);
    auto wy = mx::multiply(w, y);
    auto wz = mx::multiply(w, z);
    auto xy = mx::multiply(x, y);
    auto xz = mx::multiply(x, z);
    auto yz = mx::multiply(y, z);

    auto two = mx::array(2.0f);

    auto r00 = mx::subtract(mx::subtract(mx::add(ww, xx), yy), zz);
    auto r01 = mx::multiply(two, mx::subtract(xy, wz));
    auto r02 = mx::multiply(two, mx::add(xz, wy));
    auto r10 = mx::multiply(two, mx::add(xy, wz));
    auto r11 = mx::subtract(mx::subtract(mx::add(ww, yy), xx), zz);
    auto r12 = mx::multiply(two, mx::subtract(yz, wx));
    auto r20 = mx::multiply(two, mx::subtract(xz, wy));
    auto r21 = mx::multiply(two, mx::add(yz, wx));
    auto r22 = mx::subtract(mx::subtract(mx::add(ww, zz), xx), yy);

    auto row0 = mx::concatenate({r00, r01, r02}, 0);
    auto row1 = mx::concatenate({r10, r11, r12}, 0);
    auto row2 = mx::concatenate({r20, r21, r22}, 0);

    return mx::reshape(mx::concatenate({row0, row1, row2}, 0), {3, 3});
}

mx::array rotate(const mx::array& vec, const mx::array& quat) {
    auto s = mx::slice(quat, {0}, {1});
    auto u = mx::slice(quat, {1}, {4});

    auto dot_uv = mx::sum(mx::multiply(u, vec));
    auto dot_uu = mx::sum(mx::multiply(u, u));

    // r = 2*(u.v)*u + (s*s - u.u)*v + 2*s*(u x v)
    auto term1 = mx::multiply(mx::multiply(mx::array(2.0f), dot_uv), u);
    auto ss = mx::multiply(s, s);
    auto coeff = mx::subtract(mx::reshape(ss, {}), dot_uu);
    auto term2 = mx::multiply(coeff, vec);
    auto uxv = cross(u, vec);
    auto term3 = mx::multiply(mx::multiply(mx::array(2.0f), mx::reshape(s, {})), uxv);

    return mx::add(mx::add(term1, term2), term3);
}

mx::array quat_integrate(const mx::array& q, const mx::array& v, float dt) {
    auto n = norm(v);
    auto eps = mx::array(1e-8f);
    auto safe_n = mx::where(mx::less(n, eps), mx::array(1.0f), n);
    auto v_norm = mx::divide(v, safe_n);

    auto angle = mx::multiply(mx::array(dt), n);
    auto half = mx::multiply(angle, mx::array(0.5f));
    auto s = mx::cos(half);
    auto c = mx::sin(half);

    auto dq_w = mx::reshape(s, {1});
    auto dq_xyz = mx::multiply(v_norm, mx::reshape(c, {1}));
    auto dq = mx::concatenate({dq_w, dq_xyz}, 0);

    auto result = quat_mul(q, dq);
    return normalize(result);
}

// ── Spatial math (6D) ────────────────────────────────────────

mx::array inert_mul(const mx::array& inert, const mx::array& vel) {
    // Inertia layout: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, px, py, pz, mass]
    auto I00 = mx::slice(inert, {0}, {1});
    auto I11 = mx::slice(inert, {1}, {2});
    auto I22 = mx::slice(inert, {2}, {3});
    auto I01 = mx::slice(inert, {3}, {4});
    auto I02 = mx::slice(inert, {4}, {5});
    auto I12 = mx::slice(inert, {5}, {6});
    auto pos = mx::slice(inert, {6}, {9});
    auto mass = mx::slice(inert, {9}, {10});

    auto ang_in = mx::slice(vel, {0}, {3});
    auto lin_in = mx::slice(vel, {3}, {6});

    // inr @ ang_in (manual 3x3 symmetric matmul)
    auto w0 = mx::slice(ang_in, {0}, {1});
    auto w1 = mx::slice(ang_in, {1}, {2});
    auto w2 = mx::slice(ang_in, {2}, {3});

    auto inr_w0 = mx::add(mx::add(mx::multiply(I00, w0), mx::multiply(I01, w1)), mx::multiply(I02, w2));
    auto inr_w1 = mx::add(mx::add(mx::multiply(I01, w0), mx::multiply(I11, w1)), mx::multiply(I12, w2));
    auto inr_w2 = mx::add(mx::add(mx::multiply(I02, w0), mx::multiply(I12, w1)), mx::multiply(I22, w2));
    auto inr_w = mx::concatenate({inr_w0, inr_w1, inr_w2}, 0);

    auto pos_cross_lin = cross(pos, lin_in);
    auto ang_out = mx::add(inr_w, pos_cross_lin);

    auto mass_lin = mx::multiply(mass, lin_in);
    auto pos_cross_ang = cross(pos, ang_in);
    auto vel_out = mx::subtract(mass_lin, pos_cross_ang);

    return mx::concatenate({ang_out, vel_out}, 0);
}

mx::array motion_cross(const mx::array& u, const mx::array& v) {
    auto u_ang = mx::slice(u, {0}, {3});
    auto u_lin = mx::slice(u, {3}, {6});
    auto v_ang = mx::slice(v, {0}, {3});
    auto v_lin = mx::slice(v, {3}, {6});

    auto ang = cross(u_ang, v_ang);
    auto vel = mx::add(cross(u_lin, v_ang), cross(u_ang, v_lin));
    return mx::concatenate({ang, vel}, 0);
}

mx::array motion_cross_force(const mx::array& v, const mx::array& f) {
    auto v_ang = mx::slice(v, {0}, {3});
    auto v_lin = mx::slice(v, {3}, {6});
    auto f_ang = mx::slice(f, {0}, {3});
    auto f_lin = mx::slice(f, {3}, {6});

    auto ang = mx::add(cross(v_ang, f_ang), cross(v_lin, f_lin));
    auto vel = cross(v_ang, f_lin);
    return mx::concatenate({ang, vel}, 0);
}

mx::array transform_motion(const mx::array& vel, const mx::array& offset,
                            const mx::array& rotmat) {
    auto ang = mx::slice(vel, {0}, {3});
    auto lin = mx::slice(vel, {3}, {6});

    auto offset_cross_ang = cross(offset, ang);
    auto lin_shifted = mx::subtract(lin, offset_cross_ang);

    auto rotT = mx::transpose(rotmat);
    auto new_lin = mx::matmul(rotT, mx::reshape(lin_shifted, {3, 1}));
    auto new_ang = mx::matmul(rotT, mx::reshape(ang, {3, 1}));

    return mx::concatenate({mx::flatten(new_ang), mx::flatten(new_lin)}, 0);
}

} // namespace mjmlx
