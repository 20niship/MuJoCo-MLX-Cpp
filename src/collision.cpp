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

// Collision detection: primitive geometry pairs (plane, sphere, capsule).
// Port of Python mjmlx._src.collision.

#include "internal.h"

namespace mjmlx {

static constexpr float MJMINVAL = 1e-15f;

// Helper: extract row i from a 2D array as a 1D array
static mx::array row(const mx::array& arr, int i) {
    return mx::flatten(mx::slice(arr, {i, 0}, {i + 1, arr.shape(1)}));
}

// Helper: extract column j from a 3x3 matrix stored as row i of a (N,3,3) array
static mx::array mat_col(const mx::array& mats, int i, int col) {
    auto m = mx::slice(mats, {i, 0, col}, {i + 1, 3, col + 1});
    return mx::flatten(m);
}

static mx::array make_frame(const mx::array& normal) {
    auto n = normalize(normal);
    auto [b, c] = orthogonals(n);
    // Stack as (3, 3): [n; b; c]
    return mx::stack({n, b, c});
}

struct CollisionResult {
    mx::array dist;   // scalar
    mx::array pos;    // (3,)
    mx::array frame;  // (3, 3)
};

static CollisionResult plane_sphere(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& sphere_pos, const mx::array& sphere_size)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);
    auto dist = mx::subtract(mx::sum(mx::multiply(normal, mx::subtract(sphere_pos, plane_pos))),
                              mx::slice(sphere_size, {0}, {1}));
    auto pos = mx::subtract(sphere_pos,
                             mx::multiply(normal, mx::add(dist, mx::slice(sphere_size, {0}, {1}))));
    return {mx::flatten(dist), pos, make_frame(normal)};
}

static CollisionResult plane_capsule(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& cap_pos, const mx::array& cap_mat, const mx::array& cap_size)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);
    mx::eval(cap_size);
    float radius = cap_size.data<float>()[0];
    float half_len = cap_size.data<float>()[1];

    auto axis = mat_col(mx::reshape(cap_mat, {1, 3, 3}), 0, 2);
    auto p0 = mx::subtract(cap_pos, mx::multiply(axis, mx::array(half_len)));
    auto p1 = mx::add(cap_pos, mx::multiply(axis, mx::array(half_len)));

    auto d0 = mx::subtract(mx::sum(mx::multiply(normal, mx::subtract(p0, plane_pos))), mx::array(radius));
    auto d1 = mx::subtract(mx::sum(mx::multiply(normal, mx::subtract(p1, plane_pos))), mx::array(radius));

    mx::eval(d0); mx::eval(d1);
    bool use_p0 = d0.item<float>() < d1.item<float>();
    auto dist = use_p0 ? d0 : d1;
    auto center = use_p0 ? p0 : p1;
    auto pos = mx::subtract(center, mx::multiply(normal, mx::add(dist, mx::array(radius))));
    return {mx::reshape(dist, {}), pos, make_frame(normal)};
}

static CollisionResult sphere_sphere(
    const mx::array& pos1, const mx::array& size1,
    const mx::array& pos2, const mx::array& size2)
{
    auto diff = mx::subtract(pos2, pos1);
    auto d = norm(diff);
    mx::eval(d);
    float dv = d.item<float>();
    auto normal = (dv < MJMINVAL) ? mx::array({0.0f, 0.0f, 1.0f})
                                  : mx::divide(diff, mx::maximum(d, mx::array(MJMINVAL)));

    mx::eval(size1); mx::eval(size2);
    float r1 = size1.data<float>()[0], r2 = size2.data<float>()[0];
    auto dist = mx::subtract(d, mx::array(r1 + r2));
    auto pos = mx::add(pos1, mx::multiply(normal, mx::array(r1 + dv * 0.5f - r1)));
    return {mx::flatten(dist), pos, make_frame(normal)};
}

static CollisionResult sphere_capsule(
    const mx::array& sphere_pos, const mx::array& sphere_size,
    const mx::array& cap_pos, const mx::array& cap_mat, const mx::array& cap_size)
{
    mx::eval(sphere_size); mx::eval(cap_size);
    float r_s = sphere_size.data<float>()[0];
    float r_c = cap_size.data<float>()[0];
    float half_len = cap_size.data<float>()[1];

    auto axis = mat_col(mx::reshape(cap_mat, {1, 3, 3}), 0, 2);
    auto p0 = mx::subtract(cap_pos, mx::multiply(axis, mx::array(half_len)));
    auto p1 = mx::add(cap_pos, mx::multiply(axis, mx::array(half_len)));

    auto closest = closest_segment_point(p0, p1, sphere_pos);
    auto diff = mx::subtract(sphere_pos, closest);
    auto d = norm(diff);
    mx::eval(d);
    float dv = d.item<float>();
    auto normal = (dv < MJMINVAL) ? mx::array({0.0f, 0.0f, 1.0f})
                                  : mx::divide(diff, mx::maximum(d, mx::array(MJMINVAL)));

    auto dist = mx::subtract(d, mx::array(r_s + r_c));
    auto pos = mx::add(closest, mx::multiply(normal, mx::array(r_c)));
    return {mx::flatten(dist), pos, make_frame(normal)};
}

static CollisionResult capsule_capsule(
    const mx::array& pos1, const mx::array& mat1, const mx::array& size1,
    const mx::array& pos2, const mx::array& mat2, const mx::array& size2)
{
    mx::eval(size1); mx::eval(size2);
    float r1 = size1.data<float>()[0], h1 = size1.data<float>()[1];
    float r2 = size2.data<float>()[0], h2 = size2.data<float>()[1];

    auto axis1 = mat_col(mx::reshape(mat1, {1, 3, 3}), 0, 2);
    auto axis2 = mat_col(mx::reshape(mat2, {1, 3, 3}), 0, 2);

    auto a0 = mx::subtract(pos1, mx::multiply(axis1, mx::array(h1)));
    auto a1 = mx::add(pos1, mx::multiply(axis1, mx::array(h1)));
    auto b0 = mx::subtract(pos2, mx::multiply(axis2, mx::array(h2)));
    auto b1 = mx::add(pos2, mx::multiply(axis2, mx::array(h2)));

    auto [best_a, best_b] = closest_segment_to_segment_points(a0, a1, b0, b1);
    auto diff = mx::subtract(best_b, best_a);
    auto d = norm(diff);
    mx::eval(d);
    float dv = d.item<float>();
    auto normal = (dv < MJMINVAL) ? mx::array({0.0f, 0.0f, 1.0f})
                                  : mx::divide(diff, mx::maximum(d, mx::array(MJMINVAL)));

    auto dist = mx::subtract(d, mx::array(r1 + r2));
    auto pos = mx::add(best_a, mx::multiply(normal, mx::array(r1)));
    return {mx::flatten(dist), pos, make_frame(normal)};
}

Data collision(const Model& m, Data d) {
    if (m.opt.disableflags & DisableBit::CONTACT) {
        d.ncon = 0;
        d.contact = Contact();
        return d;
    }

    mx::eval(m.geom_type); mx::eval(m.geom_bodyid);
    mx::eval(m.geom_contype); mx::eval(m.geom_conaffinity);
    mx::eval(m.body_parentid); mx::eval(m.body_weldid);
    mx::eval(m.geom_margin); mx::eval(d.geom_xpos); mx::eval(d.geom_xmat);
    mx::eval(m.geom_size);

    auto gtype = m.geom_type.data<int>();
    auto gbid = m.geom_bodyid.data<int>();
    auto gcon = m.geom_contype.data<int>();
    auto gaff = m.geom_conaffinity.data<int>();
    auto bpid = m.body_parentid.data<int>();
    auto bwid = m.body_weldid.data<int>();
    auto gmargin = m.geom_margin.data<float>();

    std::vector<mx::array> c_dist, c_pos, c_frame, c_geom;
    std::vector<int> c_dim;

    for (int g1 = 0; g1 < m.ngeom; g1++) {
        for (int g2 = g1 + 1; g2 < m.ngeom; g2++) {
            int t1 = gtype[g1], t2 = gtype[g2];
            int g1_ = g1, g2_ = g2, t1_ = t1, t2_ = t2;
            if (t1 > t2) { std::swap(g1_, g2_); std::swap(t1_, t2_); }

            // Check contype/conaffinity
            int mask = (gcon[g1_] & gaff[g2_]) | (gcon[g2_] & gaff[g1_]);
            if (!mask) continue;

            // Skip same-body and parent-child
            int b1 = gbid[g1_], b2 = gbid[g2_];
            int w1 = bwid[b1], w2 = bwid[b2];
            if (w1 == w2) continue;

            if (!(m.opt.disableflags & DisableBit::FILTERPARENT)) {
                int w1p = (w1 > 0) ? bwid[bpid[w1]] : 0;
                int w2p = (w2 > 0) ? bwid[bpid[w2]] : 0;
                if (w1 != 0 && w2 != 0 && (w1 == w2p || w2 == w1p)) continue;
            }

            // Check exclude_signature: skip excluded body pairs
            if (m.nexclude > 0 && m.exclude_signature.size() > 0) {
                mx::eval(m.exclude_signature);
                auto ex_ptr = m.exclude_signature.data<int>();
                int bmin = std::min(b1, b2);
                int bmax = std::max(b1, b2);
                int sig = (bmin << 16) | bmax;
                bool excluded = false;
                for (int ei = 0; ei < m.nexclude; ei++) {
                    if (ex_ptr[ei] == sig) { excluded = true; break; }
                }
                if (excluded) continue;
            }

            // Dispatch collision
            auto gpos1 = row(d.geom_xpos, g1_);
            auto gpos2 = row(d.geom_xpos, g2_);
            auto gmat1 = mx::flatten(mx::slice(d.geom_xmat, {g1_, 0, 0}, {g1_ + 1, 3, 3}));
            auto gmat2 = mx::flatten(mx::slice(d.geom_xmat, {g2_, 0, 0}, {g2_ + 1, 3, 3}));
            gmat1 = mx::reshape(gmat1, {3, 3});
            gmat2 = mx::reshape(gmat2, {3, 3});
            auto gsize1 = row(m.geom_size, g1_);
            auto gsize2 = row(m.geom_size, g2_);

            CollisionResult result = {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};
            bool handled = false;

            if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::SPHERE)) {
                result = plane_sphere(gpos1, gmat1, gpos2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::CAPSULE)) {
                result = plane_capsule(gpos1, gmat1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::SPHERE)) {
                result = sphere_sphere(gpos1, gsize1, gpos2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::CAPSULE)) {
                result = sphere_capsule(gpos1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::CAPSULE) && t2_ == static_cast<int>(GeomType::CAPSULE)) {
                result = capsule_capsule(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            }

            if (!handled) continue;

            float margin = gmargin[g1_] + gmargin[g2_];
            mx::eval(result.dist);
            if (result.dist.item<float>() < margin) {
                int condim = 3; // default
                if (m.geom_condim.size() > 0) {
                    mx::eval(m.geom_condim);
                    auto cdp = m.geom_condim.data<int>();
                    condim = std::max(cdp[g1_], cdp[g2_]);
                }

                c_dist.push_back(result.dist);
                c_pos.push_back(result.pos);
                c_frame.push_back(result.frame);
                c_geom.push_back(mx::array({g1_, g2_}, mx::int32));
                c_dim.push_back(condim);
            }
        }
    }

    // Process explicit <pair> directives (even when contype=0, conaffinity=0)
    if (m.npair > 0 && m.pair_geom1.size() > 0) {
        mx::eval(m.pair_geom1); mx::eval(m.pair_geom2);
        auto pg1 = m.pair_geom1.data<int>();
        auto pg2 = m.pair_geom2.data<int>();

        for (int pi = 0; pi < m.npair; pi++) {
            int g1_ = pg1[pi], g2_ = pg2[pi];
            int t1_ = gtype[g1_], t2_ = gtype[g2_];
            if (t1_ > t2_) { std::swap(g1_, g2_); std::swap(t1_, t2_); }

            // Skip if already detected by auto-detect
            bool dup = false;
            for (auto& cg : c_geom) {
                mx::eval(cg);
                auto cgp = cg.data<int>();
                if ((cgp[0] == g1_ && cgp[1] == g2_) || (cgp[0] == g2_ && cgp[1] == g1_)) {
                    dup = true; break;
                }
            }
            if (dup) continue;

            auto gpos1 = row(d.geom_xpos, g1_);
            auto gpos2 = row(d.geom_xpos, g2_);
            auto gmat1 = mx::reshape(mx::flatten(mx::slice(d.geom_xmat, {g1_, 0, 0}, {g1_ + 1, 3, 3})), {3, 3});
            auto gmat2 = mx::reshape(mx::flatten(mx::slice(d.geom_xmat, {g2_, 0, 0}, {g2_ + 1, 3, 3})), {3, 3});
            auto gsize1 = row(m.geom_size, g1_);
            auto gsize2 = row(m.geom_size, g2_);

            CollisionResult result = {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};
            bool handled = false;

            if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::SPHERE)) {
                result = plane_sphere(gpos1, gmat1, gpos2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::CAPSULE)) {
                result = plane_capsule(gpos1, gmat1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::SPHERE)) {
                result = sphere_sphere(gpos1, gsize1, gpos2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::CAPSULE)) {
                result = sphere_capsule(gpos1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::CAPSULE) && t2_ == static_cast<int>(GeomType::CAPSULE)) {
                result = capsule_capsule(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            }

            if (!handled) continue;

            // For explicit pairs, use pair margin (or geom margin as fallback)
            float margin = gmargin[g1_] + gmargin[g2_];
            if (m.pair_margin.size() > 0) {
                mx::eval(m.pair_margin);
                margin = m.pair_margin.data<float>()[pi];
            }

            mx::eval(result.dist);
            if (result.dist.item<float>() < margin) {
                int condim = 3;
                if (m.pair_dim.size() > 0) {
                    mx::eval(m.pair_dim);
                    condim = m.pair_dim.data<int>()[pi];
                }

                c_dist.push_back(result.dist);
                c_pos.push_back(result.pos);
                c_frame.push_back(result.frame);
                c_geom.push_back(mx::array({g1_, g2_}, mx::int32));
                c_dim.push_back(condim);
            }
        }
    }

    if (!c_dist.empty()) {
        int ncon = static_cast<int>(c_dist.size());

        // Compute friction, solref, solimp from geom properties
        std::vector<mx::array> friction_v, solref_v, solimp_v, incmarg_v;
        std::vector<int> efc_addr_v;
        int efc_addr = 0;

        for (int i = 0; i < ncon; i++) {
            mx::eval(c_geom[i]);
            int gid1 = c_geom[i].data<int>()[0];
            int gid2 = c_geom[i].data<int>()[1];

            auto fri = mx::maximum(row(m.geom_friction, gid1), row(m.geom_friction, gid2));
            mx::eval(fri);
            auto fp = fri.data<float>();
            friction_v.push_back(mx::array({fp[0], fp[0], fp[1], fp[2], fp[2]}));

            if (m.geom_solref.size() > 0) {
                auto sr = mx::multiply(mx::add(row(m.geom_solref, gid1), row(m.geom_solref, gid2)),
                                       mx::array(0.5f));
                solref_v.push_back(sr);
            } else {
                solref_v.push_back(mx::array({0.02f, 1.0f}));
            }

            if (m.geom_solimp.size() > 0) {
                auto si = mx::multiply(mx::add(row(m.geom_solimp, gid1), row(m.geom_solimp, gid2)),
                                       mx::array(0.5f));
                solimp_v.push_back(si);
            } else {
                solimp_v.push_back(mx::array({0.9f, 0.95f, 0.001f, 0.5f, 2.0f}));
            }

            float mg = gmargin[gid1] + gmargin[gid2];
            float gap = 0.0f;
            if (m.geom_gap.size() > 0) {
                mx::eval(m.geom_gap);
                auto gp = m.geom_gap.data<float>();
                gap = gp[gid1] + gp[gid2];
            }
            incmarg_v.push_back(mx::array(mg - gap));

            efc_addr_v.push_back(efc_addr);
            efc_addr += c_dim[i];
        }

        Contact contact;
        contact.dist = mx::stack(c_dist);
        contact.pos = mx::stack(c_pos);
        contact.frame = mx::stack(c_frame);
        contact.dim = mx::array(c_dim.data(), {ncon}, mx::int32);
        contact.friction = mx::stack(friction_v);
        contact.solref = mx::stack(solref_v);
        contact.solimp = mx::stack(solimp_v);
        contact.includemargin = mx::stack(incmarg_v);
        contact.solreffriction = mx::zeros({ncon, 2});
        contact.geom = mx::stack(c_geom);
        contact.efc_address = mx::array(efc_addr_v.data(), {ncon}, mx::int32);

        d.contact = contact;
        d.ncon = ncon;
    } else {
        d.contact = Contact();
        d.ncon = 0;
    }

    return d;
}

} // namespace mjmlx
