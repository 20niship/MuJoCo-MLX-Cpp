// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0
//
// Vmap-compatible collision detection and constraint construction.
//
// DECISION: Fixed-size constraint outputs. All pre-computed collision pairs are always
// evaluated (even if not in contact), and inactive constraints get D=0. This makes the
// output shape constant across environments, which is required for vmap (which needs
// uniform shapes across the batch dimension). The solver naturally ignores D=0 rows.
//
// DECISION: The degenerate-normal fallback in vmap_make_frame uses [0,0,1] as the
// primary reference axis for computing tangent frames via cross product. When the
// normal is near-parallel to [0,0,1] (cross product length < 1e-6), it falls back to
// [0,1,0]. This is branch-free (uses mx::where) for vmap compatibility.
//
// NO eval(), NO data<>(), NO CPU sync.

#include "internal.h"

namespace mjmlx {

static constexpr float MJMINVAL_CV = 1e-8f;
static constexpr float MJMINIMP = 1e-3f;
static constexpr float MJMAXIMP = 0.9999f;

// ── Vmap-compatible KBI ──────────────────────────────────────────────────────

static void vmap_kbi(
    float timeconst_init, float dampratio, float timestep, bool refsafe,
    float si0, float si1, float si2, float si3, float si4,
    const mx::array& pos,
    mx::array& k_out, mx::array& b_out, mx::array& imp_out)
{
    float tc = timeconst_init;
    if (!refsafe) tc = std::max(tc, 2.0f * timestep);

    float dmin = std::max(MJMINIMP, std::min(MJMAXIMP, si0));
    float dmax = std::max(MJMINIMP, std::min(MJMAXIMP, si1));
    float width = std::max(si2, MJMINVAL_CV);
    float mid = std::max(MJMINIMP, std::min(MJMAXIMP, si3));
    float power = std::max(si4, 1.0f);

    float k_val = (tc > 0) ? 1.0f / (dmax*dmax*tc*tc*dampratio*dampratio)
                            : -tc / (dmax*dmax);
    float b_val = (dampratio > 0) ? 2.0f / (dmax*tc) : -dampratio / dmax;

    auto imp_x = mx::divide(mx::abs(pos), mx::array(width));
    auto imp_a = mx::multiply(
        mx::array(1.0f / std::pow(mid, power - 1.0f)),
        mx::power(imp_x, mx::array(power)));
    auto imp_b = mx::subtract(mx::array(1.0f),
        mx::multiply(mx::array(1.0f / std::pow(1.0f - mid, power - 1.0f)),
                      mx::power(mx::subtract(mx::array(1.0f), imp_x), mx::array(power))));
    auto imp_y = mx::where(mx::less(imp_x, mx::array(mid)), imp_a, imp_b);
    auto imp = mx::add(mx::array(dmin), mx::multiply(imp_y, mx::array(dmax - dmin)));
    imp = mx::clip(imp, mx::array(dmin), mx::array(dmax));
    imp = mx::where(mx::greater(imp_x, mx::array(1.0f)), mx::array(dmax), imp);

    k_out = mx::array(k_val);
    b_out = mx::array(b_val);
    imp_out = imp;
}

// ── Vmap-compatible Jacobian ─────────────────────────────────────────────────

static std::pair<mx::array, mx::array> vmap_jac(
    const Model& m, const Data& d, const mx::array& point, int body_id)
{
    const auto& c = m.cache;
    int rid = c.body_rootid_vec[body_id];
    auto root_com = mx::flatten(mx::slice(d.subtree_com, mx::Shape{rid, 0}, mx::Shape{rid+1, 3}));
    auto offset = mx::subtract(point, root_com);

    auto mask = c.body_dof_masks[body_id]; // (nv,) float

    auto cdof_ang = mx::slice(d.cdof, mx::Shape{0, 0}, mx::Shape{m.nv, 3}); // (nv, 3)
    auto cdof_lin = mx::slice(d.cdof, mx::Shape{0, 3}, mx::Shape{m.nv, 6}); // (nv, 3)

    auto off_broad = mx::broadcast_to(mx::reshape(offset, {1, 3}), {m.nv, 3});
    auto cross_result = mx::linalg::cross(cdof_ang, off_broad);

    auto jacp = mx::multiply(mx::add(cdof_lin, cross_result),
                              mx::reshape(mask, {m.nv, 1}));
    auto jacr = mx::multiply(cdof_ang, mx::reshape(mask, {m.nv, 1}));

    return {jacp, jacr}; // (nv, 3) each
}

// ── Vmap-compatible collision ────────────────────────────────────────────────

// Collision functions that never call eval/data
static mx::array vmap_norm(const mx::array& x) {
    return mx::sqrt(mx::maximum(mx::sum(mx::multiply(x, x)), mx::array(1e-16f)));
}

static mx::array vmap_normalize(const mx::array& x) {
    auto n = vmap_norm(x);
    return mx::divide(x, mx::maximum(n, mx::array(1e-8f)));
}

struct VmapCollResult {
    mx::array dist{mx::array(0.0f)};
    mx::array pos{mx::array(0.0f)};
    mx::array frame{mx::array(0.0f)};
};

static mx::array vmap_make_frame(const mx::array& normal) {
    auto n = vmap_normalize(normal);
    // Orthogonals: cross with [0,0,1], fallback to [0,1,0]
    auto t1 = mx::linalg::cross(mx::reshape(n, {1, 3}),
              mx::reshape(mx::array({0.0f, 0.0f, 1.0f}), {1, 3}));
    t1 = mx::flatten(t1);
    auto t1_len = vmap_norm(t1);
    auto t1_alt = mx::linalg::cross(mx::reshape(n, {1, 3}),
                  mx::reshape(mx::array({0.0f, 1.0f, 0.0f}), {1, 3}));
    t1_alt = mx::flatten(t1_alt);
    auto is_degen = mx::less(t1_len, mx::array(1e-6f));
    t1 = mx::where(is_degen, t1_alt, t1);
    auto b = vmap_normalize(t1);
    auto cc = vmap_normalize(cross(n, b));
    return mx::stack({n, b, cc});
}

static VmapCollResult vmap_plane_sphere(
    const mx::array& ppos, const mx::array& pmat,
    const mx::array& spos, float radius)
{
    auto normal = mx::flatten(mx::slice(mx::reshape(pmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto dist = mx::subtract(mx::sum(mx::multiply(normal, mx::subtract(spos, ppos))),
                              mx::array(radius));
    auto pos = mx::subtract(spos, mx::multiply(normal, mx::add(dist, mx::array(radius))));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

static VmapCollResult vmap_sphere_sphere(
    const mx::array& p1, float r1, const mx::array& p2, float r2)
{
    auto diff = mx::subtract(p2, p1);
    auto d = vmap_norm(diff);
    auto normal = mx::divide(diff, mx::maximum(d, mx::array(1e-8f)));
    auto fallback = mx::array({0.0f, 0.0f, 1.0f});
    normal = mx::where(mx::less(d, mx::array(1e-8f)), fallback, normal);
    auto dist = mx::subtract(d, mx::array(r1 + r2));
    auto pos = mx::add(p1, mx::multiply(normal, mx::array(r1)));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

static VmapCollResult vmap_plane_capsule(
    const mx::array& ppos, const mx::array& pmat,
    const mx::array& cpos, const mx::array& cmat, float radius, float half_len)
{
    auto normal = mx::flatten(mx::slice(mx::reshape(pmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto axis = mx::flatten(mx::slice(mx::reshape(cmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto p0 = mx::subtract(cpos, mx::multiply(axis, mx::array(half_len)));
    auto p1 = mx::add(cpos, mx::multiply(axis, mx::array(half_len)));
    auto d0 = mx::subtract(mx::sum(mx::multiply(normal, mx::subtract(p0, ppos))), mx::array(radius));
    auto d1 = mx::subtract(mx::sum(mx::multiply(normal, mx::subtract(p1, ppos))), mx::array(radius));
    auto use_p0 = mx::less(d0, d1);
    auto dist = mx::where(use_p0, d0, d1);
    auto center = mx::where(use_p0, p0, p1);
    auto pos = mx::subtract(center, mx::multiply(normal, mx::add(dist, mx::array(radius))));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

static VmapCollResult vmap_sphere_capsule(
    const mx::array& spos, float r_s,
    const mx::array& cpos, const mx::array& cmat, float r_c, float half_len)
{
    auto axis = mx::flatten(mx::slice(mx::reshape(cmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto p0 = mx::subtract(cpos, mx::multiply(axis, mx::array(half_len)));
    auto p1 = mx::add(cpos, mx::multiply(axis, mx::array(half_len)));

    // Closest point on segment
    auto seg = mx::subtract(p1, p0);
    auto t_num = mx::sum(mx::multiply(mx::subtract(spos, p0), seg));
    auto seg_sq = mx::sum(mx::multiply(seg, seg));
    auto param = mx::clip(mx::divide(t_num, mx::maximum(seg_sq, mx::array(1e-8f))),
                           mx::array(0.0f), mx::array(1.0f));
    auto closest = mx::add(p0, mx::multiply(seg, param));

    auto diff = mx::subtract(spos, closest);
    auto d = vmap_norm(diff);
    auto normal = mx::where(mx::less(d, mx::array(1e-8f)),
                             mx::array({0.0f, 0.0f, 1.0f}),
                             mx::divide(diff, mx::maximum(d, mx::array(1e-8f))));
    auto dist = mx::subtract(d, mx::array(r_s + r_c));
    auto pos = mx::add(closest, mx::multiply(normal, mx::array(r_c)));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

static VmapCollResult vmap_capsule_capsule(
    const mx::array& p1, const mx::array& m1, float r1, float h1,
    const mx::array& p2, const mx::array& m2, float r2, float h2)
{
    auto ax1 = mx::flatten(mx::slice(mx::reshape(m1, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto ax2 = mx::flatten(mx::slice(mx::reshape(m2, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto a0 = mx::subtract(p1, mx::multiply(ax1, mx::array(h1)));
    auto a1 = mx::add(p1, mx::multiply(ax1, mx::array(h1)));
    auto b0 = mx::subtract(p2, mx::multiply(ax2, mx::array(h2)));
    auto b1 = mx::add(p2, mx::multiply(ax2, mx::array(h2)));

    // Closest segment-to-segment (vmap-compatible)
    auto d1v = mx::subtract(a1, a0);
    auto d2v = mx::subtract(b1, b0);
    auto r = mx::subtract(a0, b0);
    auto a = mx::sum(mx::multiply(d1v, d1v));
    auto e = mx::sum(mx::multiply(d2v, d2v));
    auto f = mx::sum(mx::multiply(d2v, r));
    auto bv = mx::sum(mx::multiply(d1v, d2v));
    auto cv = mx::sum(mx::multiply(d1v, r));
    auto denom = mx::subtract(mx::multiply(a, e), mx::multiply(bv, bv));
    auto s_num = mx::subtract(mx::multiply(bv, f), mx::multiply(cv, e));
    auto s = mx::clip(mx::divide(s_num, mx::maximum(denom, mx::array(1e-8f))),
                       mx::array(0.0f), mx::array(1.0f));
    auto t = mx::clip(mx::divide(mx::add(mx::multiply(bv, s), f),
                                  mx::maximum(e, mx::array(1e-8f))),
                       mx::array(0.0f), mx::array(1.0f));
    s = mx::clip(mx::divide(mx::add(mx::negative(cv), mx::multiply(bv, t)),
                              mx::maximum(a, mx::array(1e-8f))),
                  mx::array(0.0f), mx::array(1.0f));

    auto best_a = mx::add(a0, mx::multiply(d1v, s));
    auto best_b = mx::add(b0, mx::multiply(d2v, t));

    auto diff = mx::subtract(best_b, best_a);
    auto d = vmap_norm(diff);
    auto normal = mx::where(mx::less(d, mx::array(1e-8f)),
                             mx::array({0.0f, 0.0f, 1.0f}),
                             mx::divide(diff, mx::maximum(d, mx::array(1e-8f))));
    auto dist = mx::subtract(d, mx::array(r1 + r2));
    auto pos = mx::add(best_a, mx::multiply(normal, mx::array(r1)));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

// ── BOX collision (vmap-compatible) ──────────────────────────────────────────

static VmapCollResult vmap_plane_box(
    const mx::array& ppos, const mx::array& pmat,
    const mx::array& bpos, const mx::array& bmat, const float* half)
{
    auto normal = mx::flatten(mx::slice(mx::reshape(pmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto R = mx::reshape(bmat, {3,3});

    // Compute R^T * normal (project normal into box frame)
    auto local_n = mx::flatten(mx::matmul(mx::transpose(R), mx::reshape(normal, {3,1})));
    // Deepest vertex: sign opposite to local_n, scaled by half-extents
    auto halves = mx::array({half[0], half[1], half[2]});
    auto signs = mx::negative(mx::sign(local_n));
    auto corner_local = mx::multiply(signs, halves);

    // World vertex = bpos + R * corner_local
    auto vertex = mx::add(bpos, mx::flatten(mx::matmul(R, mx::reshape(corner_local, {3,1}))));

    // Distance to plane
    auto dist = mx::sum(mx::multiply(normal, mx::subtract(vertex, ppos)));
    auto pos = mx::subtract(vertex, mx::multiply(normal, dist));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

static VmapCollResult vmap_sphere_box(
    const mx::array& spos, float radius,
    const mx::array& bpos, const mx::array& bmat, const float* half)
{
    auto R = mx::reshape(bmat, {3,3});
    // Transform sphere center to box-local
    auto diff = mx::subtract(spos, bpos);
    auto local = mx::flatten(mx::matmul(mx::transpose(R), mx::reshape(diff, {3,1})));

    // Clamp to box bounds
    auto halves_pos = mx::array({half[0], half[1], half[2]});
    auto halves_neg = mx::negative(halves_pos);
    auto closest_local = mx::clip(local, halves_neg, halves_pos);

    // Transform back to world
    auto closest_world = mx::add(bpos, mx::flatten(mx::matmul(R, mx::reshape(closest_local, {3,1}))));

    auto sep = mx::subtract(spos, closest_world);
    auto d = vmap_norm(sep);
    auto normal = mx::where(mx::less(d, mx::array(1e-8f)),
                             mx::array({0.0f, 0.0f, 1.0f}),
                             mx::divide(sep, mx::maximum(d, mx::array(1e-8f))));
    auto dist = mx::subtract(d, mx::array(radius));
    return {mx::reshape(dist, {}), closest_world, vmap_make_frame(normal)};
}

static VmapCollResult vmap_capsule_box(
    const mx::array& cpos, const mx::array& cmat, float r_c, float half_len,
    const mx::array& bpos, const mx::array& bmat, const float* half)
{
    auto R = mx::reshape(bmat, {3,3});
    auto RT = mx::transpose(R);
    auto halves_pos = mx::array({half[0], half[1], half[2]});
    auto halves_neg = mx::negative(halves_pos);

    // Capsule axis (z-column of capsule rotation matrix)
    auto cap_axis = mx::flatten(mx::slice(mx::reshape(cmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto e0 = mx::subtract(cpos, mx::multiply(cap_axis, mx::array(half_len)));
    auto e1 = mx::add(cpos, mx::multiply(cap_axis, mx::array(half_len)));

    // Test endpoints and midpoint against box, pick closest
    auto test_pt = [&](const mx::array& pt) -> std::pair<mx::array, mx::array> {
        auto diff = mx::subtract(pt, bpos);
        auto loc = mx::flatten(mx::matmul(RT, mx::reshape(diff, {3,1})));
        auto cl = mx::clip(loc, halves_neg, halves_pos);
        auto cw = mx::add(bpos, mx::flatten(mx::matmul(R, mx::reshape(cl, {3,1}))));
        return {pt, cw}; // segment point, box point
    };

    // For vmap: test midpoint (most common closest), then refine with segment projection
    auto diff_mid = mx::subtract(cpos, bpos);
    auto loc_mid = mx::flatten(mx::matmul(RT, mx::reshape(diff_mid, {3,1})));
    auto cl_mid = mx::clip(loc_mid, halves_neg, halves_pos);
    auto bpt_mid = mx::add(bpos, mx::flatten(mx::matmul(R, mx::reshape(cl_mid, {3,1}))));

    // Project box point onto capsule segment
    auto seg = mx::subtract(e1, e0);
    auto t_num = mx::sum(mx::multiply(mx::subtract(bpt_mid, e0), seg));
    auto seg_sq = mx::sum(mx::multiply(seg, seg));
    auto param = mx::clip(mx::divide(t_num, mx::maximum(seg_sq, mx::array(1e-8f))),
                           mx::array(0.0f), mx::array(1.0f));
    auto seg_pt = mx::add(e0, mx::multiply(seg, param));

    // Closest on box to refined segment point
    auto diff2 = mx::subtract(seg_pt, bpos);
    auto loc2 = mx::flatten(mx::matmul(RT, mx::reshape(diff2, {3,1})));
    auto cl2 = mx::clip(loc2, halves_neg, halves_pos);
    auto bpt2 = mx::add(bpos, mx::flatten(mx::matmul(R, mx::reshape(cl2, {3,1}))));

    auto sep = mx::subtract(seg_pt, bpt2);
    auto d = vmap_norm(sep);
    auto normal = mx::where(mx::less(d, mx::array(1e-8f)),
                             mx::array({0.0f, 0.0f, 1.0f}),
                             mx::divide(sep, mx::maximum(d, mx::array(1e-8f))));
    auto dist = mx::subtract(d, mx::array(r_c));
    return {mx::reshape(dist, {}), bpt2, vmap_make_frame(normal)};
}

static VmapCollResult vmap_box_box(
    const mx::array& pos1, const mx::array& mat1, const float* h1,
    const mx::array& pos2, const mx::array& mat2, const float* h2)
{
    auto R1 = mx::reshape(mat1, {3,3});
    auto R2 = mx::reshape(mat2, {3,3});
    auto T = mx::subtract(pos2, pos1);

    auto halves1 = mx::array({h1[0], h1[1], h1[2]});
    auto halves2 = mx::array({h2[0], h2[1], h2[2]});

    // SAT: test 6 face normals (3+3), skip edge cross products for vmap simplicity
    // For each axis, compute overlap = r1_proj + r2_proj - |T.axis|
    // The axis with minimum positive overlap is the contact normal

    auto best_overlap = mx::array(1e10f);
    auto best_normal = mx::array({0.0f, 0.0f, 1.0f});

    auto test_face_axis = [&](const mx::array& axis, int sign_mode) {
        // Project both boxes onto axis
        auto r1_proj = mx::array(0.0f);
        auto r2_proj = mx::array(0.0f);
        for (int j = 0; j < 3; j++) {
            auto col1 = mx::flatten(mx::slice(R1, mx::Shape{0,j}, mx::Shape{3,j+1}));
            auto d1 = mx::abs(mx::sum(mx::multiply(col1, axis)));
            r1_proj = mx::add(r1_proj, mx::multiply(mx::array(h1[j]), d1));

            auto col2 = mx::flatten(mx::slice(R2, mx::Shape{0,j}, mx::Shape{3,j+1}));
            auto d2 = mx::abs(mx::sum(mx::multiply(col2, axis)));
            r2_proj = mx::add(r2_proj, mx::multiply(mx::array(h2[j]), d2));
        }
        auto T_proj = mx::sum(mx::multiply(T, axis));
        auto d_proj = mx::abs(T_proj);
        auto overlap = mx::subtract(mx::add(r1_proj, r2_proj), d_proj);

        // Choose sign so axis points from box1 to box2
        auto signed_axis = mx::where(mx::greater_equal(T_proj, mx::array(0.0f)),
                                      axis, mx::negative(axis));
        auto is_better = mx::less(overlap, best_overlap);
        auto is_valid = mx::greater(overlap, mx::array(0.0f));
        auto use_this = mx::logical_and(is_better, is_valid);
        best_overlap = mx::where(use_this, overlap, best_overlap);
        best_normal = mx::where(use_this, signed_axis, best_normal);
    };

    // Test 3 face normals from box1
    for (int j = 0; j < 3; j++) {
        auto col = mx::flatten(mx::slice(R1, mx::Shape{0,j}, mx::Shape{3,j+1}));
        test_face_axis(col, 0);
    }
    // Test 3 face normals from box2
    for (int j = 0; j < 3; j++) {
        auto col = mx::flatten(mx::slice(R2, mx::Shape{0,j}, mx::Shape{3,j+1}));
        test_face_axis(col, 1);
    }

    // Contact point: midpoint of support points
    auto support1 = pos1;
    auto support2 = pos2;
    for (int j = 0; j < 3; j++) {
        auto col1 = mx::flatten(mx::slice(R1, mx::Shape{0,j}, mx::Shape{3,j+1}));
        auto d = mx::sum(mx::multiply(col1, best_normal));
        auto s = mx::where(mx::less(d, mx::array(0.0f)), mx::array(h1[j]), mx::array(-h1[j]));
        support1 = mx::add(support1, mx::multiply(col1, s));

        auto col2 = mx::flatten(mx::slice(R2, mx::Shape{0,j}, mx::Shape{3,j+1}));
        d = mx::sum(mx::multiply(col2, best_normal));
        s = mx::where(mx::greater(d, mx::array(0.0f)), mx::array(h2[j]), mx::array(-h2[j]));
        support2 = mx::add(support2, mx::multiply(col2, s));
    }

    auto contact_pos = mx::multiply(mx::add(support1, support2), mx::array(0.5f));
    auto dist = mx::negative(best_overlap);

    // If no overlap (separating), set dist to 1.0 (no contact)
    auto no_contact = mx::greater(best_overlap, mx::array(1e9f));
    dist = mx::where(no_contact, mx::array(1.0f), dist);

    return {mx::reshape(dist, {}), contact_pos, vmap_make_frame(best_normal)};
}

// ── CYLINDER collision (vmap-compatible) ────────────────────────────────────

static VmapCollResult vmap_plane_cylinder(
    const mx::array& ppos, const mx::array& pmat,
    const mx::array& cpos, const mx::array& cmat, float R, float H)
{
    auto normal = mx::flatten(mx::slice(mx::reshape(pmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto Rc = mx::reshape(cmat, {3,3});
    // Cylinder axis = z-column of rotation matrix
    auto axis = mx::flatten(mx::slice(Rc, mx::Shape{0,2}, mx::Shape{3,3}));

    // Rim centers: cpos ± H * axis
    auto rc0 = mx::subtract(cpos, mx::multiply(axis, mx::array(H)));
    auto rc1 = mx::add(cpos, mx::multiply(axis, mx::array(H)));

    // Perpendicular component of normal to cylinder axis
    auto ndota = mx::sum(mx::multiply(normal, axis));
    auto perp = mx::subtract(normal, mx::multiply(axis, ndota));
    auto perp_len = vmap_norm(perp);
    auto perp_dir = mx::divide(perp, mx::maximum(perp_len, mx::array(1e-8f)));
    // Offset direction on rim (towards plane = anti-perp)
    auto rim_offset = mx::multiply(mx::negative(perp_dir), mx::array(R));

    // When perp is near-zero (normal || axis), offset is zero (face center)
    auto is_parallel = mx::less(perp_len, mx::array(1e-6f));
    rim_offset = mx::where(is_parallel, mx::zeros({3}), rim_offset);

    auto pt0 = mx::add(rc0, rim_offset);
    auto pt1 = mx::add(rc1, rim_offset);

    auto d0 = mx::sum(mx::multiply(normal, mx::subtract(pt0, ppos)));
    auto d1 = mx::sum(mx::multiply(normal, mx::subtract(pt1, ppos)));

    auto use_0 = mx::less(d0, d1);
    auto dist = mx::where(use_0, d0, d1);
    auto vertex = mx::where(use_0, pt0, pt1);
    auto pos = mx::subtract(vertex, mx::multiply(normal, dist));
    return {mx::reshape(dist, {}), pos, vmap_make_frame(normal)};
}

static VmapCollResult vmap_sphere_cylinder(
    const mx::array& spos, float radius,
    const mx::array& cpos, const mx::array& cmat, float R, float H)
{
    auto Rc = mx::reshape(cmat, {3,3});
    auto RT = mx::transpose(Rc);

    // Transform sphere center to cylinder local
    auto diff = mx::subtract(spos, cpos);
    auto local = mx::flatten(mx::matmul(RT, mx::reshape(diff, {3,1})));
    auto lx = mx::slice(local, {0}, {1});
    auto ly = mx::slice(local, {1}, {2});
    auto lz = mx::slice(local, {2}, {3});

    auto rho = mx::sqrt(mx::maximum(mx::add(mx::multiply(lx,lx), mx::multiply(ly,ly)), mx::array(1e-16f)));
    auto clamped_z = mx::clip(lz, mx::array(-H), mx::array(H));

    // Closest on barrel: scale xy to R, clamp z
    auto scale = mx::divide(mx::array(R), mx::maximum(rho, mx::array(1e-8f)));
    auto barrel_x = mx::multiply(lx, mx::minimum(scale, mx::array(1.0f)));
    auto barrel_y = mx::multiply(ly, mx::minimum(scale, mx::array(1.0f)));

    // If beside barrel (rho > R, |z| <= H)
    auto on_barrel = mx::logical_and(mx::greater(rho, mx::array(R)),
                                      mx::less_equal(mx::abs(lz), mx::array(H)));
    // If above/below cap (rho <= R, |z| > H)
    auto on_cap = mx::logical_and(mx::less_equal(rho, mx::array(R)),
                                   mx::greater(mx::abs(lz), mx::array(H)));
    // Diagonal: rim
    auto on_rim = mx::logical_and(mx::greater(rho, mx::array(R)),
                                   mx::greater(mx::abs(lz), mx::array(H)));

    auto cx_barrel = mx::multiply(lx, scale);
    auto cy_barrel = mx::multiply(ly, scale);
    auto cz_barrel = clamped_z;

    auto cx_cap = lx;
    auto cy_cap = ly;
    auto cz_cap = mx::where(mx::greater(lz, mx::array(0.0f)), mx::array(H), mx::array(-H));

    auto cx_rim = mx::multiply(lx, mx::divide(mx::array(R), mx::maximum(rho, mx::array(1e-8f))));
    auto cy_rim = mx::multiply(ly, mx::divide(mx::array(R), mx::maximum(rho, mx::array(1e-8f))));
    auto cz_rim = cz_cap;

    // Default to barrel, override with cap, then rim
    auto cx = mx::where(on_barrel, cx_barrel, mx::where(on_cap, cx_cap, cx_rim));
    auto cy = mx::where(on_barrel, cy_barrel, mx::where(on_cap, cy_cap, cy_rim));
    auto cz = mx::where(on_barrel, cz_barrel, mx::where(on_cap, cz_cap, cz_rim));

    auto closest_local = mx::concatenate({cx, cy, cz}, 0);
    auto closest_world = mx::add(cpos, mx::flatten(mx::matmul(Rc, mx::reshape(closest_local, {3,1}))));

    auto sep = mx::subtract(spos, closest_world);
    auto d = vmap_norm(sep);
    auto norm = mx::where(mx::less(d, mx::array(1e-8f)),
                           mx::array({0.0f, 0.0f, 1.0f}),
                           mx::divide(sep, mx::maximum(d, mx::array(1e-8f))));
    auto dist = mx::subtract(d, mx::array(radius));
    return {mx::reshape(dist, {}), closest_world, vmap_make_frame(norm)};
}

static VmapCollResult vmap_capsule_cylinder(
    const mx::array& cap_pos, const mx::array& cap_mat, float r_c, float half_len,
    const mx::array& cpos, const mx::array& cmat, float R, float H)
{
    auto Rc = mx::reshape(cmat, {3,3});
    auto RT = mx::transpose(Rc);

    auto cap_axis = mx::flatten(mx::slice(mx::reshape(cap_mat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto e0 = mx::subtract(cap_pos, mx::multiply(cap_axis, mx::array(half_len)));
    auto e1 = mx::add(cap_pos, mx::multiply(cap_axis, mx::array(half_len)));

    // Use midpoint approach: project midpoint to cylinder, project back to segment, refine
    auto diff_mid = mx::subtract(cap_pos, cpos);
    auto loc_mid = mx::flatten(mx::matmul(RT, mx::reshape(diff_mid, {3,1})));

    auto lx = mx::slice(loc_mid, {0}, {1});
    auto ly = mx::slice(loc_mid, {1}, {2});
    auto lz = mx::slice(loc_mid, {2}, {3});
    auto rho = mx::sqrt(mx::maximum(mx::add(mx::multiply(lx,lx), mx::multiply(ly,ly)), mx::array(1e-16f)));

    // Simplified: closest on barrel/cap
    auto scale = mx::minimum(mx::divide(mx::array(R), mx::maximum(rho, mx::array(1e-8f))), mx::array(1.0f));
    auto is_outside = mx::greater(rho, mx::array(R));
    auto cx = mx::where(is_outside, mx::multiply(lx, mx::divide(mx::array(R), mx::maximum(rho, mx::array(1e-8f)))), lx);
    auto cy = mx::where(is_outside, mx::multiply(ly, mx::divide(mx::array(R), mx::maximum(rho, mx::array(1e-8f)))), ly);
    auto cz = mx::clip(lz, mx::array(-H), mx::array(H));

    auto cl_local = mx::concatenate({cx, cy, cz}, 0);
    auto bpt = mx::add(cpos, mx::flatten(mx::matmul(Rc, mx::reshape(cl_local, {3,1}))));

    // Project cylinder point onto capsule segment
    auto seg = mx::subtract(e1, e0);
    auto t_num = mx::sum(mx::multiply(mx::subtract(bpt, e0), seg));
    auto seg_sq = mx::sum(mx::multiply(seg, seg));
    auto param = mx::clip(mx::divide(t_num, mx::maximum(seg_sq, mx::array(1e-8f))),
                           mx::array(0.0f), mx::array(1.0f));
    auto seg_pt = mx::add(e0, mx::multiply(seg, param));

    // Refine: closest on cylinder to refined segment point
    auto diff2 = mx::subtract(seg_pt, cpos);
    auto loc2 = mx::flatten(mx::matmul(RT, mx::reshape(diff2, {3,1})));
    auto lx2 = mx::slice(loc2, {0}, {1});
    auto ly2 = mx::slice(loc2, {1}, {2});
    auto lz2 = mx::slice(loc2, {2}, {3});
    auto rho2 = mx::sqrt(mx::maximum(mx::add(mx::multiply(lx2,lx2), mx::multiply(ly2,ly2)), mx::array(1e-16f)));

    auto is_outside2 = mx::greater(rho2, mx::array(R));
    auto cx2 = mx::where(is_outside2, mx::multiply(lx2, mx::divide(mx::array(R), mx::maximum(rho2, mx::array(1e-8f)))), lx2);
    auto cy2 = mx::where(is_outside2, mx::multiply(ly2, mx::divide(mx::array(R), mx::maximum(rho2, mx::array(1e-8f)))), ly2);
    auto cz2 = mx::clip(lz2, mx::array(-H), mx::array(H));

    auto cl2 = mx::concatenate({cx2, cy2, cz2}, 0);
    auto bpt2 = mx::add(cpos, mx::flatten(mx::matmul(Rc, mx::reshape(cl2, {3,1}))));

    auto sep = mx::subtract(seg_pt, bpt2);
    auto d = vmap_norm(sep);
    auto norm = mx::where(mx::less(d, mx::array(1e-8f)),
                           mx::array({0.0f, 0.0f, 1.0f}),
                           mx::divide(sep, mx::maximum(d, mx::array(1e-8f))));
    auto dist = mx::subtract(d, mx::array(r_c));
    return {mx::reshape(dist, {}), bpt2, vmap_make_frame(norm)};
}

// ── MESH collision via vmap GJK (GPU-parallel) ──────────────────────────────
// Proper GJK collision for mesh geoms using pure MLX array ops.
// All branching via mx::where — no eval(), no data<>(), fully vmap-compatible.
// Runs on GPU in parallel across N environments via compile(vmap(...)).

// ── Support functions (furthest point in a direction, world space) ───────────

static mx::array vmap_support_sphere(
    const mx::array& pos, float radius, const mx::array& dir)
{
    return mx::add(pos, mx::multiply(vmap_normalize(dir), mx::array(radius)));
}

static mx::array vmap_support_capsule(
    const mx::array& pos, const mx::array& mat, float radius, float half_len,
    const mx::array& dir)
{
    auto axis = mx::flatten(mx::slice(mx::reshape(mat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto d = mx::sum(mx::multiply(axis, dir));
    auto endpoint = mx::add(pos, mx::multiply(axis, mx::where(mx::greater_equal(d, mx::array(0.0f)),
                                                                mx::array(half_len), mx::array(-half_len))));
    return mx::add(endpoint, mx::multiply(vmap_normalize(dir), mx::array(radius)));
}

static mx::array vmap_support_box(
    const mx::array& pos, const mx::array& mat, const float* size, const mx::array& dir)
{
    auto Rc = mx::reshape(mat, {3,3});
    auto xc = mx::flatten(mx::slice(Rc, mx::Shape{0,0}, mx::Shape{3,1}));
    auto yc = mx::flatten(mx::slice(Rc, mx::Shape{0,1}, mx::Shape{3,2}));
    auto zc = mx::flatten(mx::slice(Rc, mx::Shape{0,2}, mx::Shape{3,3}));
    auto sx = mx::where(mx::greater_equal(mx::sum(mx::multiply(xc, dir)), mx::array(0.0f)),
                         mx::array(size[0]), mx::array(-size[0]));
    auto sy = mx::where(mx::greater_equal(mx::sum(mx::multiply(yc, dir)), mx::array(0.0f)),
                         mx::array(size[1]), mx::array(-size[1]));
    auto sz = mx::where(mx::greater_equal(mx::sum(mx::multiply(zc, dir)), mx::array(0.0f)),
                         mx::array(size[2]), mx::array(-size[2]));
    return mx::add(pos, mx::add(mx::add(mx::multiply(xc, sx), mx::multiply(yc, sy)),
                                 mx::multiply(zc, sz)));
}

static mx::array vmap_support_cylinder(
    const mx::array& pos, const mx::array& mat, float R, float H, const mx::array& dir)
{
    auto axis = mx::flatten(mx::slice(mx::reshape(mat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto da = mx::sum(mx::multiply(axis, dir));
    auto tip = mx::add(pos, mx::multiply(axis, mx::where(mx::greater_equal(da, mx::array(0.0f)),
                                                           mx::array(H), mx::array(-H))));
    auto proj = mx::sum(mx::multiply(dir, axis));
    auto perp = mx::subtract(dir, mx::multiply(axis, proj));
    auto pl = vmap_norm(perp);
    auto offset = mx::multiply(mx::divide(perp, mx::maximum(pl, mx::array(1e-8f))), mx::array(R));
    offset = mx::where(mx::less(pl, mx::array(1e-8f)), mx::zeros({3}), offset);
    return mx::add(tip, offset);
}

static mx::array vmap_support_mesh(
    const mx::array& verts, const mx::array& pos, const mx::array& mat,
    const mx::array& dir)
{
    auto Rc = mx::reshape(mat, {3,3});
    auto RT = mx::transpose(Rc);
    auto dir_local = mx::flatten(mx::matmul(RT, mx::reshape(dir, {3,1})));
    auto dots = mx::flatten(mx::matmul(verts, mx::reshape(dir_local, {3,1})));
    auto best_idx = mx::argmax(dots);
    auto best_local = mx::flatten(mx::take(verts, mx::reshape(best_idx, {1}), 0));
    return mx::add(pos, mx::flatten(mx::matmul(Rc, mx::reshape(best_local, {3,1}))));
}

// ── Generic support function dispatcher ─────────────────────────────────────

enum VmapGeomKind { VGK_SPHERE=0, VGK_CAPSULE, VGK_BOX, VGK_CYLINDER, VGK_MESH };

struct VmapConvexShape {
    VmapGeomKind kind;
    mx::array pos{mx::zeros({3})};
    mx::array mat{mx::zeros({9})};
    float size[3] = {0,0,0};
    mx::array verts{mx::zeros({0})};
};

static mx::array vmap_support(const VmapConvexShape& g, const mx::array& dir) {
    switch (g.kind) {
    case VGK_SPHERE:   return vmap_support_sphere(g.pos, g.size[0], dir);
    case VGK_CAPSULE:  return vmap_support_capsule(g.pos, g.mat, g.size[0], g.size[1], dir);
    case VGK_BOX:      return vmap_support_box(g.pos, g.mat, g.size, dir);
    case VGK_CYLINDER: return vmap_support_cylinder(g.pos, g.mat, g.size[0], g.size[1], dir);
    case VGK_MESH:     return vmap_support_mesh(g.verts, g.pos, g.mat, dir);
    default:           return g.pos;
    }
}

static VmapGeomKind geom_type_to_kind(int t) {
    switch (t) {
    case (int)GeomType::SPHERE:   return VGK_SPHERE;
    case (int)GeomType::CAPSULE:  return VGK_CAPSULE;
    case (int)GeomType::BOX:      return VGK_BOX;
    case (int)GeomType::CYLINDER: return VGK_CYLINDER;
    case (int)GeomType::MESH:     return VGK_MESH;
    default:                      return VGK_SPHERE;
    }
}

// ── Vmap GJK: fixed 32 iterations, pure MLX ops ────────────────────────────

// GJK simplex state — stored as fixed-size (4,3) arrays
struct VmapGJKState {
    mx::array sdiff; // (4, 3) Minkowski difference points
    mx::array sa;    // (4, 3) witness points on shape A
    mx::array sb;    // (4, 3) witness points on shape B
    mx::array sn;    // scalar: simplex size (1-4)
    mx::array dir;   // (3,) search direction
    mx::array converged; // scalar bool
    mx::array overlap;   // scalar bool
};

// Helper: set row i of a (4,3) array to a (3,) vector
static mx::array set_row(const mx::array& mat, const mx::array& row_idx, const mx::array& val) {
    auto result = mat;
    for (int i = 0; i < 4; i++) {
        auto is_i = mx::equal(row_idx, mx::array(i));
        auto old_row = mx::flatten(mx::slice(mat, {i, 0}, {i+1, 3}));
        auto new_row = mx::where(is_i, val, old_row);
        // Build result by replacing row i
        auto rows_before = (i > 0) ? mx::slice(result, {0, 0}, {i, 3}) : mx::array({});
        auto rows_after = (i < 3) ? mx::slice(result, {i+1, 0}, {4, 3}) : mx::array({});
        if (i == 0) {
            result = mx::concatenate({mx::reshape(new_row, {1,3}),
                                       mx::slice(result, {1,0}, {4,3})}, 0);
        } else if (i == 3) {
            result = mx::concatenate({mx::slice(result, {0,0}, {3,3}),
                                       mx::reshape(new_row, {1,3})}, 0);
        } else {
            result = mx::concatenate({mx::slice(result, {0,0}, {i,3}),
                                       mx::reshape(new_row, {1,3}),
                                       mx::slice(result, {i+1,0}, {4,3})}, 0);
        }
    }
    return result;
}

// GJK line update: simplex has 2 points (indices sn-1=newest, sn-2=older)
static void gjk_update_line(
    const mx::array& sdiff, const mx::array& sa, const mx::array& sb,
    mx::array& out_sdiff, mx::array& out_sa, mx::array& out_sb,
    mx::array& out_sn, mx::array& out_dir, mx::array& out_overlap)
{
    auto A = mx::flatten(mx::slice(sdiff, {1, 0}, {2, 3})); // newest
    auto B = mx::flatten(mx::slice(sdiff, {0, 0}, {1, 3})); // older
    auto AB = mx::subtract(B, A);
    auto AO = mx::negative(A);
    auto ab_dot_ao = mx::sum(mx::multiply(AB, AO));
    auto toward_b = mx::greater(ab_dot_ao, mx::array(0.0f));

    // If toward B: direction = AB × AO × AB (triple cross product)
    auto abxao = mx::flatten(mx::linalg::cross(mx::reshape(AB, {1,3}), mx::reshape(AO, {1,3})));
    auto new_dir_triple = mx::flatten(mx::linalg::cross(mx::reshape(abxao, {1,3}), mx::reshape(AB, {1,3})));
    // Fallback if degenerate
    auto tdl = vmap_norm(new_dir_triple);
    new_dir_triple = mx::where(mx::less(tdl, mx::array(1e-12f)), AO, new_dir_triple);

    // If not toward B: keep only A, direction = AO
    out_dir = mx::where(toward_b, new_dir_triple, AO);
    // If not toward B, shrink simplex to just A (put A at index 0)
    out_sdiff = mx::where(toward_b, sdiff,
        mx::concatenate({mx::reshape(A, {1,3}), mx::slice(sdiff, {1,0}, {4,3})}, 0));
    out_sa = mx::where(toward_b, sa,
        mx::concatenate({mx::reshape(mx::flatten(mx::slice(sa, {1,0}, {2,3})), {1,3}),
                          mx::slice(sa, {1,0}, {4,3})}, 0));
    out_sb = mx::where(toward_b, sb,
        mx::concatenate({mx::reshape(mx::flatten(mx::slice(sb, {1,0}, {2,3})), {1,3}),
                          mx::slice(sb, {1,0}, {4,3})}, 0));
    out_sn = mx::where(toward_b, mx::array(2), mx::array(1));
    out_overlap = mx::array(false);
}

// GJK triangle update: simplex has 3 points
static void gjk_update_triangle(
    const mx::array& sdiff, const mx::array& sa, const mx::array& sb,
    mx::array& out_sdiff, mx::array& out_sa, mx::array& out_sb,
    mx::array& out_sn, mx::array& out_dir, mx::array& out_overlap)
{
    auto A = mx::flatten(mx::slice(sdiff, {2, 0}, {3, 3})); // newest
    auto B = mx::flatten(mx::slice(sdiff, {1, 0}, {2, 3}));
    auto C = mx::flatten(mx::slice(sdiff, {0, 0}, {1, 3}));
    auto AB = mx::subtract(B, A);
    auto AC = mx::subtract(C, A);
    auto AO = mx::negative(A);
    auto ABC = mx::flatten(mx::linalg::cross(mx::reshape(AB, {1,3}), mx::reshape(AC, {1,3})));

    // Test which region the origin is in
    auto abc_x_ac = mx::flatten(mx::linalg::cross(mx::reshape(ABC, {1,3}), mx::reshape(AC, {1,3})));
    auto ab_x_abc = mx::flatten(mx::linalg::cross(mx::reshape(AB, {1,3}), mx::reshape(ABC, {1,3})));

    auto ac_side = mx::greater(mx::sum(mx::multiply(abc_x_ac, AO)), mx::array(0.0f));
    auto ab_side = mx::greater(mx::sum(mx::multiply(ab_x_abc, AO)), mx::array(0.0f));
    auto above = mx::greater(mx::sum(mx::multiply(ABC, AO)), mx::array(0.0f));

    // Case 1: AC edge region — keep A, C
    auto dir_ac = mx::flatten(mx::linalg::cross(
        mx::linalg::cross(mx::reshape(AC, {1,3}), mx::reshape(AO, {1,3})),
        mx::reshape(AC, {1,3})));
    // Case 2: AB edge region — keep A, B
    auto dir_ab = mx::flatten(mx::linalg::cross(
        mx::linalg::cross(mx::reshape(AB, {1,3}), mx::reshape(AO, {1,3})),
        mx::reshape(AB, {1,3})));
    // Case 3: above triangle — direction = ABC
    // Case 4: below triangle — flip winding, direction = -ABC

    // Select direction
    auto new_dir = mx::where(ac_side, dir_ac,
                    mx::where(ab_side, dir_ab,
                    mx::where(above, ABC, mx::negative(ABC))));
    auto ndl = vmap_norm(new_dir);
    new_dir = mx::where(mx::less(ndl, mx::array(1e-12f)), AO, new_dir);

    // Select simplex: if ac_side or ab_side, shrink to 2 points
    auto shrink = mx::logical_or(ac_side, ab_side);
    auto sn_new = mx::where(shrink, mx::array(2), mx::array(3));

    // For AC: simplex = {C, A} at indices 0,1
    auto sdiff_ac = mx::concatenate({mx::reshape(C, {1,3}), mx::reshape(A, {1,3}),
                                      mx::slice(sdiff, {2,0}, {4,3})}, 0);
    auto sa_ac = mx::concatenate({mx::slice(sa, {0,0}, {1,3}), mx::slice(sa, {2,0}, {3,3}),
                                   mx::slice(sa, {2,0}, {4,3})}, 0);
    auto sb_ac = mx::concatenate({mx::slice(sb, {0,0}, {1,3}), mx::slice(sb, {2,0}, {3,3}),
                                   mx::slice(sb, {2,0}, {4,3})}, 0);
    // For AB: simplex = {B, A}
    auto sdiff_ab = mx::concatenate({mx::reshape(B, {1,3}), mx::reshape(A, {1,3}),
                                      mx::slice(sdiff, {2,0}, {4,3})}, 0);
    auto sa_ab = mx::concatenate({mx::slice(sa, {1,0}, {2,3}), mx::slice(sa, {2,0}, {3,3}),
                                   mx::slice(sa, {2,0}, {4,3})}, 0);
    auto sb_ab = mx::concatenate({mx::slice(sb, {1,0}, {2,3}), mx::slice(sb, {2,0}, {3,3}),
                                   mx::slice(sb, {2,0}, {4,3})}, 0);
    // For below: flip B and C in simplex (swap indices 0 and 1)
    auto sdiff_flip = mx::concatenate({mx::reshape(B, {1,3}), mx::reshape(C, {1,3}),
                                        mx::reshape(A, {1,3}), mx::slice(sdiff, {3,0}, {4,3})}, 0);
    auto sa_flip = mx::concatenate({mx::slice(sa, {1,0}, {2,3}), mx::slice(sa, {0,0}, {1,3}),
                                     mx::slice(sa, {2,0}, {3,3}), mx::slice(sa, {3,0}, {4,3})}, 0);
    auto sb_flip = mx::concatenate({mx::slice(sb, {1,0}, {2,3}), mx::slice(sb, {0,0}, {1,3}),
                                     mx::slice(sb, {2,0}, {3,3}), mx::slice(sb, {3,0}, {4,3})}, 0);

    out_sdiff = mx::where(ac_side, sdiff_ac, mx::where(ab_side, sdiff_ab,
                mx::where(above, sdiff, sdiff_flip)));
    out_sa = mx::where(ac_side, sa_ac, mx::where(ab_side, sa_ab,
              mx::where(above, sa, sa_flip)));
    out_sb = mx::where(ac_side, sb_ac, mx::where(ab_side, sb_ab,
              mx::where(above, sb, sb_flip)));
    out_sn = sn_new;
    out_dir = new_dir;
    out_overlap = mx::array(false);
}

// GJK tetrahedron update: simplex has 4 points
static void gjk_update_tetrahedron(
    const mx::array& sdiff, const mx::array& sa, const mx::array& sb,
    mx::array& out_sdiff, mx::array& out_sa, mx::array& out_sb,
    mx::array& out_sn, mx::array& out_dir, mx::array& out_overlap)
{
    auto A = mx::flatten(mx::slice(sdiff, {3, 0}, {4, 3})); // newest
    auto B = mx::flatten(mx::slice(sdiff, {2, 0}, {3, 3}));
    auto C = mx::flatten(mx::slice(sdiff, {1, 0}, {2, 3}));
    auto D = mx::flatten(mx::slice(sdiff, {0, 0}, {1, 3}));
    auto AO = mx::negative(A);

    auto AB = mx::subtract(B, A), AC = mx::subtract(C, A), AD = mx::subtract(D, A);
    auto ABC = mx::flatten(mx::linalg::cross(mx::reshape(AB, {1,3}), mx::reshape(AC, {1,3})));
    auto ACD = mx::flatten(mx::linalg::cross(mx::reshape(AC, {1,3}), mx::reshape(AD, {1,3})));
    auto ADB = mx::flatten(mx::linalg::cross(mx::reshape(AD, {1,3}), mx::reshape(AB, {1,3})));

    auto abc_test = mx::greater(mx::sum(mx::multiply(ABC, AO)), mx::array(0.0f));
    auto acd_test = mx::greater(mx::sum(mx::multiply(ACD, AO)), mx::array(0.0f));
    auto adb_test = mx::greater(mx::sum(mx::multiply(ADB, AO)), mx::array(0.0f));

    auto any_outside = mx::logical_or(abc_test, mx::logical_or(acd_test, adb_test));

    // If outside ABC face: keep {C, B, A} as triangle
    auto sdiff_abc = mx::concatenate({mx::reshape(C, {1,3}), mx::reshape(B, {1,3}),
                                       mx::reshape(A, {1,3}), mx::slice(sdiff, {3,0}, {4,3})}, 0);
    // If outside ACD face: keep {D, C, A}
    auto sdiff_acd = mx::concatenate({mx::reshape(D, {1,3}), mx::reshape(C, {1,3}),
                                       mx::reshape(A, {1,3}), mx::slice(sdiff, {3,0}, {4,3})}, 0);
    // If outside ADB face: keep {B, D, A}
    auto sdiff_adb = mx::concatenate({mx::reshape(B, {1,3}), mx::reshape(D, {1,3}),
                                       mx::reshape(A, {1,3}), mx::slice(sdiff, {3,0}, {4,3})}, 0);

    auto new_sdiff = mx::where(abc_test, sdiff_abc,
                      mx::where(acd_test, sdiff_acd, sdiff_adb));
    auto new_dir = mx::where(abc_test, ABC, mx::where(acd_test, ACD, ADB));

    // Witness points: same reordering
    auto sa_A = mx::slice(sa, {3,0}, {4,3}); auto sb_A = mx::slice(sb, {3,0}, {4,3});
    auto sa_B = mx::slice(sa, {2,0}, {3,3}); auto sb_B = mx::slice(sb, {2,0}, {3,3});
    auto sa_C = mx::slice(sa, {1,0}, {2,3}); auto sb_C = mx::slice(sb, {1,0}, {2,3});
    auto sa_D = mx::slice(sa, {0,0}, {1,3}); auto sb_D = mx::slice(sb, {0,0}, {1,3});

    auto new_sa = mx::where(abc_test,
        mx::concatenate({sa_C, sa_B, sa_A, sa_A}, 0),
        mx::where(acd_test,
            mx::concatenate({sa_D, sa_C, sa_A, sa_A}, 0),
            mx::concatenate({sa_B, sa_D, sa_A, sa_A}, 0)));
    auto new_sb = mx::where(abc_test,
        mx::concatenate({sb_C, sb_B, sb_A, sb_A}, 0),
        mx::where(acd_test,
            mx::concatenate({sb_D, sb_C, sb_A, sb_A}, 0),
            mx::concatenate({sb_B, sb_D, sb_A, sb_A}, 0)));

    out_sdiff = mx::where(any_outside, new_sdiff, sdiff);
    out_sa = mx::where(any_outside, new_sa, sa);
    out_sb = mx::where(any_outside, new_sb, sb);
    out_sn = mx::where(any_outside, mx::array(3), mx::array(4));
    out_dir = mx::where(any_outside, new_dir, mx::zeros({3}));
    out_overlap = mx::logical_not(any_outside); // if no face is outside, origin is inside
}

// Run vmap GJK: 32 fixed iterations
static VmapGJKState vmap_gjk(const VmapConvexShape& A, const VmapConvexShape& B) {
    auto initial_dir = mx::subtract(B.pos, A.pos);
    auto idl = vmap_norm(initial_dir);
    initial_dir = mx::where(mx::less(idl, mx::array(1e-12f)),
                             mx::array({1.0f, 0.0f, 0.0f}), initial_dir);

    // First support point
    auto sa0 = vmap_support(A, initial_dir);
    auto sb0 = vmap_support(B, mx::negative(initial_dir));
    auto sd0 = mx::subtract(sa0, sb0);

    auto sdiff = mx::concatenate({mx::reshape(sd0, {1,3}), mx::zeros({3,3})}, 0); // (4,3)
    auto s_a = mx::concatenate({mx::reshape(sa0, {1,3}), mx::zeros({3,3})}, 0);
    auto s_b = mx::concatenate({mx::reshape(sb0, {1,3}), mx::zeros({3,3})}, 0);
    auto sn = mx::array(1);
    auto dir = mx::negative(sd0);
    auto dl = vmap_norm(dir);
    dir = mx::where(mx::less(dl, mx::array(1e-12f)), mx::array({1.0f,0.0f,0.0f}), dir);
    auto converged = mx::array(false);
    auto overlap = mx::array(false);

    for (int iter = 0; iter < 32; iter++) {
        // Compute new support point
        auto new_sa = vmap_support(A, dir);
        auto new_sb = vmap_support(B, mx::negative(dir));
        auto new_sd = mx::subtract(new_sa, new_sb);

        // Check progress: if new point doesn't pass origin, no overlap
        auto progress = mx::sum(mx::multiply(new_sd, dir));
        auto no_progress = mx::less(progress, mx::array(0.0f));

        // Add point to simplex
        // Place new point at index sn (0-based)
        auto new_sdiff = sdiff, new_s_a = s_a, new_s_b = s_b;
        for (int i = 0; i < 4; i++) {
            auto is_target = mx::equal(sn, mx::array(i));
            auto old_sd_row = mx::flatten(mx::slice(sdiff, {i,0}, {i+1,3}));
            auto old_sa_row = mx::flatten(mx::slice(s_a, {i,0}, {i+1,3}));
            auto old_sb_row = mx::flatten(mx::slice(s_b, {i,0}, {i+1,3}));
            auto rep_sd = mx::where(is_target, new_sd, old_sd_row);
            auto rep_sa = mx::where(is_target, new_sa, old_sa_row);
            auto rep_sb = mx::where(is_target, new_sb, old_sb_row);
            if (i == 0) {
                new_sdiff = mx::concatenate({mx::reshape(rep_sd, {1,3}), mx::slice(new_sdiff, {1,0}, {4,3})}, 0);
                new_s_a = mx::concatenate({mx::reshape(rep_sa, {1,3}), mx::slice(new_s_a, {1,0}, {4,3})}, 0);
                new_s_b = mx::concatenate({mx::reshape(rep_sb, {1,3}), mx::slice(new_s_b, {1,0}, {4,3})}, 0);
            } else if (i == 3) {
                new_sdiff = mx::concatenate({mx::slice(new_sdiff, {0,0}, {3,3}), mx::reshape(rep_sd, {1,3})}, 0);
                new_s_a = mx::concatenate({mx::slice(new_s_a, {0,0}, {3,3}), mx::reshape(rep_sa, {1,3})}, 0);
                new_s_b = mx::concatenate({mx::slice(new_s_b, {0,0}, {3,3}), mx::reshape(rep_sb, {1,3})}, 0);
            } else {
                new_sdiff = mx::concatenate({mx::slice(new_sdiff, {0,0}, {i,3}), mx::reshape(rep_sd, {1,3}),
                                              mx::slice(new_sdiff, {i+1,0}, {4,3})}, 0);
                new_s_a = mx::concatenate({mx::slice(new_s_a, {0,0}, {i,3}), mx::reshape(rep_sa, {1,3}),
                                            mx::slice(new_s_a, {i+1,0}, {4,3})}, 0);
                new_s_b = mx::concatenate({mx::slice(new_s_b, {0,0}, {i,3}), mx::reshape(rep_sb, {1,3}),
                                            mx::slice(new_s_b, {i+1,0}, {4,3})}, 0);
            }
        }
        auto new_sn = mx::add(sn, mx::array(1));

        // Compute all three update cases
        auto z4x3 = mx::zeros({4,3}); auto z0 = mx::array(0); auto z3 = mx::zeros({3}); auto zb = mx::array(false);
        mx::array upd_sdiff2=z4x3, upd_sa2=z4x3, upd_sb2=z4x3, upd_sn2=z0, upd_dir2=z3, upd_ovl2=zb;
        gjk_update_line(new_sdiff, new_s_a, new_s_b, upd_sdiff2, upd_sa2, upd_sb2, upd_sn2, upd_dir2, upd_ovl2);

        mx::array upd_sdiff3=z4x3, upd_sa3=z4x3, upd_sb3=z4x3, upd_sn3=z0, upd_dir3=z3, upd_ovl3=zb;
        gjk_update_triangle(new_sdiff, new_s_a, new_s_b, upd_sdiff3, upd_sa3, upd_sb3, upd_sn3, upd_dir3, upd_ovl3);

        mx::array upd_sdiff4=z4x3, upd_sa4=z4x3, upd_sb4=z4x3, upd_sn4=z0, upd_dir4=z3, upd_ovl4=zb;
        gjk_update_tetrahedron(new_sdiff, new_s_a, new_s_b, upd_sdiff4, upd_sa4, upd_sb4, upd_sn4, upd_dir4, upd_ovl4);

        // Select based on new_sn (2=line, 3=triangle, 4=tetrahedron)
        auto is2 = mx::equal(new_sn, mx::array(2));
        auto is3 = mx::equal(new_sn, mx::array(3));
        // is4 = default (new_sn >= 4)

        auto sel_sdiff = mx::where(is2, upd_sdiff2, mx::where(is3, upd_sdiff3, upd_sdiff4));
        auto sel_sa = mx::where(is2, upd_sa2, mx::where(is3, upd_sa3, upd_sa4));
        auto sel_sb = mx::where(is2, upd_sb2, mx::where(is3, upd_sb3, upd_sb4));
        auto sel_sn = mx::where(is2, upd_sn2, mx::where(is3, upd_sn3, upd_sn4));
        auto sel_dir = mx::where(is2, upd_dir2, mx::where(is3, upd_dir3, upd_dir4));
        auto sel_ovl = mx::where(is2, upd_ovl2, mx::where(is3, upd_ovl3, upd_ovl4));

        // If no progress: converge with no overlap
        auto new_converged = mx::logical_or(converged, no_progress);
        auto new_overlap = mx::logical_or(overlap, sel_ovl);
        new_converged = mx::logical_or(new_converged, new_overlap);

        // Mask: if already converged, keep old state
        sdiff = mx::where(converged, sdiff, sel_sdiff);
        s_a = mx::where(converged, s_a, sel_sa);
        s_b = mx::where(converged, s_b, sel_sb);
        sn = mx::where(converged, sn, sel_sn);
        dir = mx::where(converged, dir, sel_dir);
        auto dir_l = vmap_norm(dir);
        dir = mx::where(mx::less(dir_l, mx::array(1e-12f)), mx::array({1.0f,0.0f,0.0f}), dir);
        overlap = new_overlap;
        converged = new_converged;
    }

    return {sdiff, s_a, s_b, sn, dir, converged, overlap};
}

// ── Support-based penetration depth (replaces EPA) ──────────────────────────
// Sample ~20 directions, find minimum support width for penetration depth.

static VmapCollResult vmap_gjk_depth(
    const VmapConvexShape& A, const VmapConvexShape& B,
    const VmapGJKState& state)
{
    // Sample directions: 6 axis-aligned + 8 diagonal + 4 from simplex face normals = 18
    std::vector<std::array<float,3>> sample_dirs = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {0.577f,0.577f,0.577f},{-0.577f,0.577f,0.577f},
        {0.577f,-0.577f,0.577f},{0.577f,0.577f,-0.577f},
        {-0.577f,-0.577f,0.577f},{-0.577f,0.577f,-0.577f},
        {0.577f,-0.577f,-0.577f},{-0.577f,-0.577f,-0.577f}
    };

    mx::array best_depth = mx::array(1e10f);
    mx::array best_normal = mx::array({0.0f, 0.0f, 1.0f});
    mx::array best_pa = A.pos;
    mx::array best_pb = B.pos;

    for (auto& sd : sample_dirs) {
        auto d = mx::array({sd[0], sd[1], sd[2]});
        auto pa = vmap_support(A, d);
        auto pb = vmap_support(B, mx::negative(d));
        auto width = mx::sum(mx::multiply(mx::subtract(pa, pb), d));
        auto is_better = mx::less(width, best_depth);
        best_depth = mx::where(is_better, width, best_depth);
        best_normal = mx::where(is_better, d, best_normal);
        best_pa = mx::where(is_better, pa, best_pa);
        best_pb = mx::where(is_better, pb, best_pb);
    }

    // Also try simplex face normals (more accurate for actual penetration)
    auto A0 = mx::flatten(mx::slice(state.sdiff, {0,0}, {1,3}));
    auto A1 = mx::flatten(mx::slice(state.sdiff, {1,0}, {2,3}));
    auto A2 = mx::flatten(mx::slice(state.sdiff, {2,0}, {3,3}));
    auto A3 = mx::flatten(mx::slice(state.sdiff, {3,0}, {4,3}));

    std::vector<std::pair<mx::array,mx::array>> face_pairs = {
        {mx::subtract(A1, A0), mx::subtract(A2, A0)},
        {mx::subtract(A2, A0), mx::subtract(A3, A0)},
        {mx::subtract(A3, A0), mx::subtract(A1, A0)},
        {mx::subtract(A2, A1), mx::subtract(A3, A1)}
    };

    for (auto& [e1, e2] : face_pairs) {
        auto fn = mx::flatten(mx::linalg::cross(mx::reshape(e1, {1,3}), mx::reshape(e2, {1,3})));
        auto fnl = vmap_norm(fn);
        fn = mx::where(mx::less(fnl, mx::array(1e-8f)), mx::array({0.0f,0.0f,1.0f}),
                        mx::divide(fn, mx::maximum(fnl, mx::array(1e-8f))));
        // Try both orientations
        for (int sign = -1; sign <= 1; sign += 2) {
            auto d = (sign > 0) ? fn : mx::negative(fn);
            auto pa = vmap_support(A, d);
            auto pb = vmap_support(B, mx::negative(d));
            auto width = mx::sum(mx::multiply(mx::subtract(pa, pb), d));
            auto is_better = mx::less(width, best_depth);
            best_depth = mx::where(is_better, width, best_depth);
            best_normal = mx::where(is_better, d, best_normal);
            best_pa = mx::where(is_better, pa, best_pa);
            best_pb = mx::where(is_better, pb, best_pb);
        }
    }

    auto contact_pos = mx::multiply(mx::add(best_pa, best_pb), mx::array(0.5f));
    auto dist = mx::negative(mx::maximum(best_depth, mx::array(0.0f)));
    return {mx::reshape(dist, {}), contact_pos, vmap_make_frame(best_normal)};
}

// ── Full vmap GJK collision: handles any convex pair ────────────────────────

static VmapCollResult vmap_gjk_collision(const VmapConvexShape& A, const VmapConvexShape& B) {
    auto state = vmap_gjk(A, B);

    // For overlapping case: use support-based depth estimation
    auto overlap_result = vmap_gjk_depth(A, B, state);

    // For non-overlapping case: use GJK distance
    // Closest point on simplex to origin (use last 2 points as line approximation)
    auto p0 = mx::flatten(mx::slice(state.sdiff, {0,0}, {1,3}));
    auto p1 = mx::flatten(mx::slice(state.sdiff, {1,0}, {2,3}));
    auto seg = mx::subtract(p1, p0);
    auto seg_sq = mx::sum(mx::multiply(seg, seg));
    auto t = mx::negative(mx::sum(mx::multiply(p0, seg)));
    t = mx::divide(t, mx::maximum(seg_sq, mx::array(1e-12f)));
    t = mx::clip(t, mx::array(0.0f), mx::array(1.0f));
    auto closest = mx::add(p0, mx::multiply(seg, t));
    auto gap_dist = vmap_norm(closest);

    // Witness points
    auto wa0 = mx::flatten(mx::slice(state.sa, {0,0}, {1,3}));
    auto wa1 = mx::flatten(mx::slice(state.sa, {1,0}, {2,3}));
    auto wb0 = mx::flatten(mx::slice(state.sb, {0,0}, {1,3}));
    auto wb1 = mx::flatten(mx::slice(state.sb, {1,0}, {2,3}));
    auto wit_a = mx::add(wa0, mx::multiply(mx::subtract(wa1, wa0), t));
    auto wit_b = mx::add(wb0, mx::multiply(mx::subtract(wb1, wb0), t));
    auto sep_dir = mx::subtract(wit_a, wit_b);
    auto sep_norm = vmap_normalize(sep_dir);
    auto gap_pos = mx::multiply(mx::add(wit_a, wit_b), mx::array(0.5f));
    VmapCollResult gap_result = {mx::reshape(gap_dist, {}), gap_pos, vmap_make_frame(sep_norm)};

    // Select based on overlap flag
    auto dist = mx::where(state.overlap, overlap_result.dist, gap_result.dist);
    auto pos = mx::where(state.overlap, overlap_result.pos, gap_result.pos);
    auto frame = mx::where(state.overlap, overlap_result.frame, gap_result.frame);
    return {dist, pos, frame};
}

// ── Analytic vmap plane-mesh (vertex projection, no GJK needed) ─────────────

static VmapCollResult vmap_plane_mesh_proper(
    const mx::array& ppos, const mx::array& pmat,
    const mx::array& mpos, const mx::array& mmat,
    const mx::array& mesh_verts) // (nv, 3) pre-baked constant
{
    auto normal = mx::flatten(mx::slice(mx::reshape(pmat, {3,3}), mx::Shape{0,2}, mx::Shape{3,3}));
    auto Rc = mx::reshape(mmat, {3,3});
    // Transform ALL vertices to world: verts @ R^T + pos
    auto world_verts = mx::add(
        mx::matmul(mesh_verts, mx::transpose(Rc)),
        mx::reshape(mpos, {1, 3})); // (nv, 3)
    // Distance of each vertex to plane
    auto dists = mx::matmul(
        mx::subtract(world_verts, mx::reshape(ppos, {1, 3})),
        mx::reshape(normal, {3, 1})); // (nv, 1)
    // Deepest vertex
    auto best_idx = mx::argmin(mx::flatten(dists));
    auto best_vert = mx::flatten(mx::take(world_verts, mx::reshape(best_idx, {1}), 0)); // (3,)
    auto dist = mx::reshape(mx::take(mx::flatten(dists), mx::reshape(best_idx, {1})), {}); // scalar
    auto pos = mx::subtract(best_vert, mx::multiply(normal, dist));
    return {dist, pos, vmap_make_frame(normal)};
}

// ── HFIELD collision (vmap-compatible) ───────────────────────────────────────
// Bilinear height interpolation at the geom's projected position.
// Computes a single contact from the height field surface.

static VmapCollResult vmap_hfield_collision(
    const mx::array& hf_pos, const mx::array& hf_mat,
    const mx::array& hf_data,   // (nrow, ncol) pre-baked
    const float* hf_size,        // (x_half, y_half, z_top, z_bottom)
    int hf_nrow, int hf_ncol,
    const mx::array& geom_pos, float geom_rbound)
{
    float sx = hf_size[0], sy = hf_size[1], sz_top = hf_size[2], sz_bot = hf_size[3];

    // Transform geom center to hfield local frame: local = R^T * (geom_pos - hf_pos)
    auto Rc = mx::reshape(hf_mat, {3,3});
    auto RT = mx::transpose(Rc);
    auto diff = mx::subtract(geom_pos, hf_pos);
    auto local = mx::flatten(mx::matmul(RT, mx::reshape(diff, {3,1}))); // (3,)

    // Local coordinates: x, y, z
    auto lx = mx::reshape(mx::take(local, mx::array({0})), {});
    auto ly = mx::reshape(mx::take(local, mx::array({1})), {});
    auto lz = mx::reshape(mx::take(local, mx::array({2})), {});

    // Map to grid coordinates (continuous)
    float dx = (hf_ncol > 1) ? 2.0f * sx / (hf_ncol - 1) : 2.0f * sx;
    float dy = (hf_nrow > 1) ? 2.0f * sy / (hf_nrow - 1) : 2.0f * sy;
    auto col_f = mx::divide(mx::add(lx, mx::array(sx)), mx::array(dx));
    auto row_f = mx::divide(mx::add(ly, mx::array(sy)), mx::array(dy));

    // Clamp to valid range
    col_f = mx::clip(col_f, mx::array(0.0f), mx::array((float)(hf_ncol - 2)));
    row_f = mx::clip(row_f, mx::array(0.0f), mx::array((float)(hf_nrow - 2)));

    auto c0 = mx::floor(col_f);
    auto r0 = mx::floor(row_f);
    auto cf = mx::subtract(col_f, c0);
    auto rf = mx::subtract(row_f, r0);

    // Bilinear interpolation indices: (r0,c0), (r0,c0+1), (r0+1,c0), (r0+1,c0+1)
    auto r0i = mx::astype(r0, mx::int32);
    auto c0i = mx::astype(c0, mx::int32);
    auto ncol_arr = mx::array(hf_ncol);

    auto idx00 = mx::add(mx::multiply(r0i, ncol_arr), c0i);
    auto idx01 = mx::add(idx00, mx::array(1));
    auto idx10 = mx::add(idx00, ncol_arr);
    auto idx11 = mx::add(idx10, mx::array(1));

    auto hf_flat = mx::flatten(hf_data);
    auto h00 = mx::reshape(mx::take(hf_flat, mx::reshape(idx00, {1})), {});
    auto h01 = mx::reshape(mx::take(hf_flat, mx::reshape(idx01, {1})), {});
    auto h10 = mx::reshape(mx::take(hf_flat, mx::reshape(idx10, {1})), {});
    auto h11 = mx::reshape(mx::take(hf_flat, mx::reshape(idx11, {1})), {});

    // Bilinear interpolation
    auto one_cf = mx::subtract(mx::array(1.0f), cf);
    auto one_rf = mx::subtract(mx::array(1.0f), rf);
    auto height = mx::add(
        mx::add(mx::multiply(mx::multiply(one_rf, one_cf), h00),
                mx::multiply(mx::multiply(one_rf, cf), h01)),
        mx::add(mx::multiply(mx::multiply(rf, one_cf), h10),
                mx::multiply(mx::multiply(rf, cf), h11)));

    // Scale to world height: z_surface = height * sz_top (data in [0,1])
    auto z_surface = mx::multiply(height, mx::array(sz_top));

    // Surface normal via height gradient
    auto dh_dc = mx::add(
        mx::multiply(one_rf, mx::subtract(h01, h00)),
        mx::multiply(rf, mx::subtract(h11, h10)));
    auto dh_dr = mx::add(
        mx::multiply(one_cf, mx::subtract(h10, h00)),
        mx::multiply(cf, mx::subtract(h11, h01)));
    auto dz_dx = mx::divide(mx::multiply(dh_dc, mx::array(sz_top)), mx::array(dx));
    auto dz_dy = mx::divide(mx::multiply(dh_dr, mx::array(sz_top)), mx::array(dy));

    // Normal in local frame: (-dz/dx, -dz/dy, 1), normalized
    auto n_local = mx::stack({mx::negative(dz_dx), mx::negative(dz_dy), mx::array(1.0f)});
    n_local = vmap_normalize(n_local);

    // Transform normal to world frame
    auto n_world = mx::flatten(mx::matmul(Rc, mx::reshape(n_local, {3,1})));

    // Contact point in local frame: (lx, ly, z_surface)
    auto contact_local = mx::stack({lx, ly, z_surface});
    // Transform to world
    auto contact_world = mx::add(hf_pos,
        mx::flatten(mx::matmul(Rc, mx::reshape(contact_local, {3,1}))));

    // Distance: geom center z_local - z_surface - geom_rbound
    auto dist = mx::subtract(mx::subtract(lz, z_surface), mx::array(geom_rbound));

    return {mx::reshape(dist, {}), contact_world, vmap_make_frame(n_world)};
}

// ── Vmap-compatible collision (top level) ────────────────────────────────────

Data vmap_collision(const Model& m, Data d) {
    if (m.opt.disableflags & DisableBit::CONTACT) {
        d.ncon = 0;
        d.contact = Contact();
        return d;
    }

    const auto& c = m.cache;
    int n_pairs = (int)c.collision_pairs.size();
    if (n_pairs == 0) {
        d.ncon = 0;
        d.contact = Contact();
        return d;
    }

    // Helper to get geom world pos and mat from data
    auto gpos = [&](int gi) { return mx::flatten(mx::slice(d.geom_xpos, mx::Shape{gi,0}, mx::Shape{gi+1,3})); };
    auto gmat = [&](int gi) { return mx::flatten(mx::slice(d.geom_xmat, mx::Shape{gi,0,0}, mx::Shape{gi+1,3,3})); };

    std::vector<mx::array> c_dist, c_pos, c_frame, c_geom;
    std::vector<mx::array> c_friction, c_solref, c_solimp, c_incmarg;
    std::vector<int> c_dim;

    for (int pi = 0; pi < n_pairs; pi++) {
        auto& cp = c.collision_pairs[pi];
        auto p1 = gpos(cp.g1); auto m1 = gmat(cp.g1);
        auto p2 = gpos(cp.g2); auto m2 = gmat(cp.g2);

        VmapCollResult result;
        int t1 = cp.type1, t2 = cp.type2;

        if (t1 == (int)GeomType::PLANE && t2 == (int)GeomType::SPHERE) {
            result = vmap_plane_sphere(p1, m1, p2, cp.size2[0]);
        } else if (t1 == (int)GeomType::PLANE && t2 == (int)GeomType::CAPSULE) {
            result = vmap_plane_capsule(p1, m1, p2, m2, cp.size2[0], cp.size2[1]);
        } else if (t1 == (int)GeomType::SPHERE && t2 == (int)GeomType::SPHERE) {
            result = vmap_sphere_sphere(p1, cp.size1[0], p2, cp.size2[0]);
        } else if (t1 == (int)GeomType::SPHERE && t2 == (int)GeomType::CAPSULE) {
            result = vmap_sphere_capsule(p1, cp.size1[0], p2, m2, cp.size2[0], cp.size2[1]);
        } else if (t1 == (int)GeomType::CAPSULE && t2 == (int)GeomType::CAPSULE) {
            result = vmap_capsule_capsule(p1, m1, cp.size1[0], cp.size1[1],
                                          p2, m2, cp.size2[0], cp.size2[1]);
        } else if (t1 == (int)GeomType::PLANE && t2 == (int)GeomType::BOX) {
            result = vmap_plane_box(p1, m1, p2, m2, cp.size2);
        } else if (t1 == (int)GeomType::SPHERE && t2 == (int)GeomType::BOX) {
            result = vmap_sphere_box(p1, cp.size1[0], p2, m2, cp.size2);
        } else if (t1 == (int)GeomType::CAPSULE && t2 == (int)GeomType::BOX) {
            result = vmap_capsule_box(p1, m1, cp.size1[0], cp.size1[1], p2, m2, cp.size2);
        } else if (t1 == (int)GeomType::BOX && t2 == (int)GeomType::BOX) {
            result = vmap_box_box(p1, m1, cp.size1, p2, m2, cp.size2);
        } else if (t1 == (int)GeomType::PLANE && t2 == (int)GeomType::CYLINDER) {
            result = vmap_plane_cylinder(p1, m1, p2, m2, cp.size2[0], cp.size2[1]);
        } else if (t1 == (int)GeomType::SPHERE && t2 == (int)GeomType::CYLINDER) {
            result = vmap_sphere_cylinder(p1, cp.size1[0], p2, m2, cp.size2[0], cp.size2[1]);
        } else if (t1 == (int)GeomType::CAPSULE && t2 == (int)GeomType::CYLINDER) {
            result = vmap_capsule_cylinder(p1, m1, cp.size1[0], cp.size1[1],
                                           p2, m2, cp.size2[0], cp.size2[1]);
        } else if (t2 == (int)GeomType::HFIELD || t1 == (int)GeomType::HFIELD) {
            // Hfield collision via bilinear height interpolation
            if (cp.hf_nrow > 0 && cp.hf_ncol > 0 && cp.hf_data.size() > 0) {
                auto hf_p = (t1 == (int)GeomType::HFIELD) ? p1 : p2;
                auto hf_m = (t1 == (int)GeomType::HFIELD) ? m1 : m2;
                auto geom_p = (t1 == (int)GeomType::HFIELD) ? p2 : p1;
                float geom_rb = 0.0f;
                if (t1 == (int)GeomType::HFIELD) {
                    geom_rb = std::max({cp.size2[0], cp.size2[1], cp.size2[2]});
                } else {
                    geom_rb = std::max({cp.size1[0], cp.size1[1], cp.size1[2]});
                }
                if (geom_rb < 0.001f) geom_rb = 0.05f;
                result = vmap_hfield_collision(hf_p, hf_m, cp.hf_data, cp.hf_size,
                    cp.hf_nrow, cp.hf_ncol, geom_p, geom_rb);
            } else {
                result = {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};
            }
        } else if (t1 == (int)GeomType::PLANE && t2 == (int)GeomType::MESH) {
            if (cp.mesh_verts2.size() > 0) {
                result = vmap_plane_mesh_proper(p1, m1, p2, m2, cp.mesh_verts2);
            } else {
                result = {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};
            }
        } else if (t2 == (int)GeomType::MESH || t1 == (int)GeomType::MESH) {
            VmapConvexShape shapeA;
            shapeA.kind = geom_type_to_kind(t1);
            shapeA.pos = p1; shapeA.mat = m1;
            shapeA.size[0] = cp.size1[0]; shapeA.size[1] = cp.size1[1]; shapeA.size[2] = cp.size1[2];
            shapeA.verts = cp.mesh_verts1;

            VmapConvexShape shapeB;
            shapeB.kind = geom_type_to_kind(t2);
            shapeB.pos = p2; shapeB.mat = m2;
            shapeB.size[0] = cp.size2[0]; shapeB.size[1] = cp.size2[1]; shapeB.size[2] = cp.size2[2];
            shapeB.verts = cp.mesh_verts2;

            result = vmap_gjk_collision(shapeA, shapeB);
        } else {
            result = {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};
        }

        c_dist.push_back(result.dist);
        c_pos.push_back(result.pos);
        c_frame.push_back(result.frame);
        c_geom.push_back(mx::array({cp.g1, cp.g2}, mx::int32));
        c_dim.push_back(cp.condim);
        c_friction.push_back(mx::array(cp.friction, mx::Shape{5}));
        c_solref.push_back(mx::array(cp.solref, mx::Shape{2}));
        c_solimp.push_back(mx::array(cp.solimp, mx::Shape{5}));
        c_incmarg.push_back(mx::array(cp.margin - cp.gap));
    }

    Contact contact;
    contact.dist = mx::stack(c_dist);
    contact.pos = mx::stack(c_pos);
    contact.frame = mx::stack(c_frame);
    contact.dim = mx::array(c_dim.data(), mx::Shape{n_pairs}, mx::int32);
    contact.friction = mx::stack(c_friction);
    contact.solref = mx::stack(c_solref);
    contact.solimp = mx::stack(c_solimp);
    contact.includemargin = mx::stack(c_incmarg);
    contact.geom = mx::stack(c_geom);

    d.contact = contact;
    d.ncon = n_pairs;
    return d;
}

// ── Vmap-compatible constraint construction ──────────────────────────────────

Data vmap_make_constraint(const Model& m, Data d) {
    if (m.opt.disableflags & DisableBit::CONSTRAINT) {
        d.efc_J = mx::zeros({0, m.nv});
        d.efc_D = mx::zeros({0});
        d.efc_aref = mx::zeros({0});
        d.efc_force = mx::zeros({0});
        d.efc_frictionloss = mx::zeros({0});
        d.nefc = 0;
        return d;
    }

    const auto& c = m.cache;
    bool refsafe = !(m.opt.disableflags & DisableBit::REFSAFE);

    std::vector<mx::array> J_rows, D_vals, aref_vals, floss_vals;
    int ne = 0, nf = 0, nl = 0;

    // ── Joint limits (fixed-size: always iterate all limits, mask inactive) ──
    if (!(m.opt.disableflags & DisableBit::LIMIT)) {
        for (auto& li : c.limits) {
            auto qval = mx::flatten(mx::slice(d.qpos, mx::Shape{li.dof_adr}, mx::Shape{li.dof_adr + 1}));
            auto dist_min = mx::subtract(qval, mx::array(li.range_low));
            auto dist_max = mx::subtract(mx::array(li.range_high), qval);
            auto pos = mx::subtract(mx::minimum(dist_min, dist_max), mx::array(li.margin));
            auto sign = mx::where(mx::less(dist_min, dist_max), mx::array(1.0f), mx::array(-1.0f));

            // J row: sign at da
            std::vector<float> jr(m.nv, 0.0f);
            jr[li.dof_adr] = 1.0f;
            auto J = mx::multiply(mx::array(jr.data(), mx::Shape{m.nv}, mx::float32), sign);

            mx::array k(0.0f), b(0.0f), imp(0.0f);
            vmap_kbi(li.solref[0], li.solref[1], m.opt.timestep, refsafe,
                     li.solimp[0], li.solimp[1], li.solimp[2], li.solimp[3], li.solimp[4],
                     pos, k, b, imp);

            float invw = 1.0f;
            if (m.dof_invweight0.size() > 0) {
                mx::eval(m.dof_invweight0);
                invw = m.dof_invweight0.data<float>()[li.dof_adr];
            }
            auto r = mx::maximum(mx::multiply(mx::array(invw),
                mx::divide(mx::subtract(mx::array(1.0f), imp), imp)), mx::array(MJMINVAL_CV));

            auto j_dot_qvel = mx::sum(mx::multiply(J, d.qvel));
            auto aref = mx::subtract(mx::negative(mx::multiply(b, j_dot_qvel)),
                                      mx::multiply(mx::multiply(k, imp), pos));

            auto active = mx::less(pos, mx::array(0.0f));
            auto d_val = mx::where(active, mx::divide(mx::array(1.0f), r), mx::array(0.0f));
            auto aref_val = mx::where(active, aref, mx::array(0.0f));

            J_rows.push_back(J);
            D_vals.push_back(mx::flatten(d_val));
            aref_vals.push_back(mx::flatten(aref_val));
            floss_vals.push_back(mx::array({0.0f}));
            nl++;
        }
    }

    // ── Contact constraints ──
    if (!(m.opt.disableflags & DisableBit::CONTACT) && d.ncon > 0) {
        bool use_pyramidal = (m.opt.cone == ConeType::PYRAMIDAL);

        for (int ci = 0; ci < d.ncon; ci++) {
            auto& cp = c.collision_pairs[ci];
            auto c_dist = mx::flatten(mx::slice(d.contact.dist, mx::Shape{ci}, mx::Shape{ci+1}));
            auto c_pos = mx::flatten(mx::slice(d.contact.pos, mx::Shape{ci, 0}, mx::Shape{ci+1, 3}));
            auto c_frame = mx::slice(d.contact.frame, mx::Shape{ci, 0, 0}, mx::Shape{ci+1, 3, 3});
            c_frame = mx::reshape(c_frame, {3, 3});
            auto c_incm = mx::flatten(mx::slice(d.contact.includemargin, mx::Shape{ci}, mx::Shape{ci+1}));

            auto pos = mx::subtract(c_dist, c_incm);
            auto contact_active = mx::less(c_dist, mx::array(cp.margin));

            auto [jacp1, jacr1] = vmap_jac(m, d, c_pos, cp.body1);
            auto [jacp2, jacr2] = vmap_jac(m, d, c_pos, cp.body2);

            auto djacp = mx::subtract(jacp2, jacp1);
            auto normal = mx::flatten(mx::slice(c_frame, mx::Shape{0, 0}, mx::Shape{1, 3}));
            auto j_normal = mx::flatten(mx::matmul(mx::reshape(normal, {1, 3}),
                                                     mx::transpose(djacp)));

            float invw = 0.0f;
            if (m.body_invweight0.size() > 0) {
                mx::eval(m.body_invweight0);
                auto iw = m.body_invweight0.data<float>();
                invw = iw[cp.body1 * 2] + iw[cp.body2 * 2];
            }

            if (cp.condim <= 1 || !use_pyramidal) {
                // Frictionless: single normal constraint row
                mx::array k_v(0.0f), b_v(0.0f), imp_v(0.0f);
                vmap_kbi(cp.solref[0], cp.solref[1], m.opt.timestep, refsafe,
                         cp.solimp[0], cp.solimp[1], cp.solimp[2], cp.solimp[3], cp.solimp[4],
                         pos, k_v, b_v, imp_v);

                auto r = mx::maximum(mx::multiply(mx::array(invw),
                    mx::divide(mx::subtract(mx::array(1.0f), imp_v), imp_v)), mx::array(MJMINVAL_CV));
                auto j_dot_qvel = mx::sum(mx::multiply(j_normal, d.qvel));
                auto aref = mx::subtract(mx::negative(mx::multiply(b_v, j_dot_qvel)),
                                          mx::multiply(mx::multiply(k_v, imp_v), pos));

                auto d_val = mx::where(contact_active, mx::divide(mx::array(1.0f), r), mx::array(0.0f));
                auto aref_val = mx::where(contact_active, aref, mx::array(0.0f));

                J_rows.push_back(j_normal);
                D_vals.push_back(mx::flatten(d_val));
                aref_vals.push_back(mx::flatten(aref_val));
                floss_vals.push_back(mx::array({0.0f}));
            } else {
                // Pyramidal friction: 2*(condim-1) rows per contact
                // Build direction Jacobians:
                //   dir 0,1: tangent (frame[1,2] @ djacp)
                //   dir 2:   torsion (frame[0] @ djacr) — condim>=4
                //   dir 3,4: rolling (frame[1,2] @ djacr) — condim>=6

                auto djacr = mx::subtract(jacr2, jacr1);

                float invw_rot = 0.0f;
                if (cp.condim > 3 && m.body_invweight0.size() > 0) {
                    mx::eval(m.body_invweight0);
                    auto iw = m.body_invweight0.data<float>();
                    invw_rot = iw[cp.body1 * 2 + 1] + iw[cp.body2 * 2 + 1];
                }

                std::vector<mx::array> jac_dirs;
                // Tangent directions (translational)
                int n_tran = std::min(cp.condim - 1, 2);
                for (int tk = 1; tk <= n_tran; tk++) {
                    auto tang = mx::flatten(mx::slice(c_frame, mx::Shape{tk, 0}, mx::Shape{tk + 1, 3}));
                    jac_dirs.push_back(mx::flatten(mx::matmul(mx::reshape(tang, {1, 3}),
                                                               mx::transpose(djacp))));
                }
                // Torsion (rotational around normal)
                if (cp.condim >= 4) {
                    auto norm_dir = mx::flatten(mx::slice(c_frame, mx::Shape{0, 0}, mx::Shape{1, 3}));
                    jac_dirs.push_back(mx::flatten(mx::matmul(mx::reshape(norm_dir, {1, 3}),
                                                               mx::transpose(djacr))));
                }
                // Rolling (rotational around tangent1, tangent2)
                if (cp.condim >= 6) {
                    for (int tk = 1; tk <= 2; tk++) {
                        auto tang = mx::flatten(mx::slice(c_frame, mx::Shape{tk, 0}, mx::Shape{tk + 1, 3}));
                        jac_dirs.push_back(mx::flatten(mx::matmul(mx::reshape(tang, {1, 3}),
                                                                   mx::transpose(djacr))));
                    }
                }

                mx::array k_v(0.0f), b_v(0.0f), imp_v(0.0f);
                vmap_kbi(cp.solref[0], cp.solref[1], m.opt.timestep, refsafe,
                         cp.solimp[0], cp.solimp[1], cp.solimp[2], cp.solimp[3], cp.solimp[4],
                         pos, k_v, b_v, imp_v);

                // Pyramidal impedance: ALL rows use the same D from primary friction[0]
                float mu0 = cp.friction[0];
                float mu0_sq = mu0 * mu0;
                float invw_py = invw + mu0_sq * invw;
                auto r_first = mx::maximum(mx::multiply(mx::array(invw_py),
                    mx::divide(mx::subtract(mx::array(1.0f), imp_v), imp_v)), mx::array(MJMINVAL_CV));
                auto r_py = mx::maximum(mx::multiply(mx::array(2.0f * mu0_sq / m.opt.impratio), r_first),
                                         mx::array(MJMINVAL_CV));

                int n_dirs = (int)jac_dirs.size();
                for (int tk = 0; tk < n_dirs; tk++) {
                    float fri_k = cp.friction[tk];

                    // Positive edge
                    auto j_pos = mx::add(j_normal, mx::multiply(mx::array(fri_k), jac_dirs[tk]));
                    auto jdot_pos = mx::sum(mx::multiply(j_pos, d.qvel));
                    auto aref_pos = mx::subtract(mx::negative(mx::multiply(b_v, jdot_pos)),
                                                  mx::multiply(mx::multiply(k_v, imp_v), pos));
                    auto d_pos = mx::where(contact_active, mx::divide(mx::array(1.0f), r_py), mx::array(0.0f));
                    auto aref_pos_v = mx::where(contact_active, aref_pos, mx::array(0.0f));

                    J_rows.push_back(j_pos);
                    D_vals.push_back(mx::flatten(d_pos));
                    aref_vals.push_back(mx::flatten(aref_pos_v));
                    floss_vals.push_back(mx::array({0.0f}));

                    // Negative edge
                    auto j_neg = mx::subtract(j_normal, mx::multiply(mx::array(fri_k), jac_dirs[tk]));
                    auto jdot_neg = mx::sum(mx::multiply(j_neg, d.qvel));
                    auto aref_neg = mx::subtract(mx::negative(mx::multiply(b_v, jdot_neg)),
                                                  mx::multiply(mx::multiply(k_v, imp_v), pos));
                    auto d_neg = mx::where(contact_active, mx::divide(mx::array(1.0f), r_py), mx::array(0.0f));
                    auto aref_neg_v = mx::where(contact_active, aref_neg, mx::array(0.0f));

                    J_rows.push_back(j_neg);
                    D_vals.push_back(mx::flatten(d_neg));
                    aref_vals.push_back(mx::flatten(aref_neg_v));
                    floss_vals.push_back(mx::array({0.0f}));
                }
            }
        }
    }

    int nefc = (int)J_rows.size();
    if (nefc > 0) {
        d.efc_J = mx::stack(J_rows);
        d.efc_D = mx::concatenate(D_vals, 0);
        d.efc_aref = mx::concatenate(aref_vals, 0);
        d.efc_force = mx::zeros({nefc});
        d.efc_frictionloss = mx::concatenate(floss_vals, 0);
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
