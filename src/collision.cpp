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

// ── BOX collision helpers ─────────────────────────────────────

// Transform a world-space point into box-local coordinates
static void world_to_box_local(const float* Rp, const float* bp,
                                float wx, float wy, float wz,
                                float& lx, float& ly, float& lz) {
    float dx = wx - bp[0], dy = wy - bp[1], dz = wz - bp[2];
    // local = R^T * (world - box_pos), R stored row-major: Rp[i*3+j] = R[i][j]
    lx = Rp[0]*dx + Rp[3]*dy + Rp[6]*dz;
    ly = Rp[1]*dx + Rp[4]*dy + Rp[7]*dz;
    lz = Rp[2]*dx + Rp[5]*dy + Rp[8]*dz;
}

// Transform box-local coordinates to world-space
static void box_local_to_world(const float* Rp, const float* bp,
                                float lx, float ly, float lz,
                                float& wx, float& wy, float& wz) {
    // world = box_pos + R * local
    wx = bp[0] + Rp[0]*lx + Rp[1]*ly + Rp[2]*lz;
    wy = bp[1] + Rp[3]*lx + Rp[4]*ly + Rp[5]*lz;
    wz = bp[2] + Rp[6]*lx + Rp[7]*ly + Rp[8]*lz;
}

// Closest point on an axis-aligned box (in local coords) to a local point
static void closest_on_aabb(float lx, float ly, float lz,
                             float hx, float hy, float hz,
                             float& cx, float& cy, float& cz) {
    cx = std::max(-hx, std::min(hx, lx));
    cy = std::max(-hy, std::min(hy, ly));
    cz = std::max(-hz, std::min(hz, lz));
}

// plane_box_multi: returns up to 4 contacts (vertices of the closest face)
// Matches MuJoCo C's mjc_PlaneBox algorithm.
static int plane_box_multi(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& box_pos, const mx::array& box_mat, const mx::array& box_size,
    float margin, int g1, int g2, int condim,
    std::vector<mx::array>& c_dist, std::vector<mx::array>& c_pos,
    std::vector<mx::array>& c_frame, std::vector<mx::array>& c_geom,
    std::vector<int>& c_dim)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);
    auto frame = make_frame(normal);

    mx::eval(box_size); mx::eval(box_pos); mx::eval(box_mat);
    mx::eval(normal); mx::eval(plane_pos);
    float hx = box_size.data<float>()[0];
    float hy = box_size.data<float>()[1];
    float hz = box_size.data<float>()[2];

    auto np = normal.data<float>();
    auto Rp = box_mat.data<float>();
    auto bp = box_pos.data<float>();
    auto pp = plane_pos.data<float>();

    // Project plane normal into box-local frame: local_n = R^T * normal
    float local_n[3];
    for (int j = 0; j < 3; j++)
        local_n[j] = Rp[0*3+j]*np[0] + Rp[1*3+j]*np[1] + Rp[2*3+j]*np[2];

    // Find the face axis most aligned with the plane normal
    int best_axis = 0;
    float best_dot = std::abs(local_n[0]);
    for (int i = 1; i < 3; i++) {
        float d = std::abs(local_n[i]);
        if (d > best_dot) { best_dot = d; best_axis = i; }
    }

    float halves[3] = {hx, hy, hz};
    // Sign: face center on the side opposite to the normal
    float face_sign = (local_n[best_axis] < 0) ? 1.0f : -1.0f;

    // Generate 4 vertices of the closest face
    int ax1 = (best_axis + 1) % 3;
    int ax2 = (best_axis + 2) % 3;

    int ncon_added = 0;
    for (int s1 = -1; s1 <= 1; s1 += 2) {
        for (int s2 = -1; s2 <= 1; s2 += 2) {
            float corner_local[3] = {0, 0, 0};
            corner_local[best_axis] = face_sign * halves[best_axis];
            corner_local[ax1] = s1 * halves[ax1];
            corner_local[ax2] = s2 * halves[ax2];

            float wx, wy, wz;
            box_local_to_world(Rp, bp, corner_local[0], corner_local[1], corner_local[2],
                               wx, wy, wz);

            float dist_val = np[0]*(wx-pp[0]) + np[1]*(wy-pp[1]) + np[2]*(wz-pp[2]);

            if (dist_val < margin) {
                auto vertex = mx::array({wx, wy, wz});
                auto contact_pos = mx::subtract(vertex, mx::multiply(normal, mx::array(dist_val)));
                c_dist.push_back(mx::array(dist_val));
                c_pos.push_back(contact_pos);
                c_frame.push_back(frame);
                c_geom.push_back(mx::array({g1, g2}, mx::int32));
                c_dim.push_back(condim);
                ncon_added++;
            }
        }
    }

    return ncon_added;
}

// Single-contact plane_box (for fallback / vmap dispatch compatibility)
static CollisionResult plane_box(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& box_pos, const mx::array& box_mat, const mx::array& box_size)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);

    mx::eval(box_size); mx::eval(box_pos); mx::eval(box_mat);
    mx::eval(normal); mx::eval(plane_pos);

    auto np = normal.data<float>();
    auto Rp = box_mat.data<float>();
    auto bp = box_pos.data<float>();
    auto pp = plane_pos.data<float>();
    float halves[3] = {box_size.data<float>()[0], box_size.data<float>()[1], box_size.data<float>()[2]};

    float local_n[3];
    for (int j = 0; j < 3; j++)
        local_n[j] = Rp[0*3+j]*np[0] + Rp[1*3+j]*np[1] + Rp[2*3+j]*np[2];

    float corner_local[3];
    for (int j = 0; j < 3; j++)
        corner_local[j] = (local_n[j] < 0 ? 1.0f : -1.0f) * halves[j];

    float wx, wy, wz;
    box_local_to_world(Rp, bp, corner_local[0], corner_local[1], corner_local[2],
                       wx, wy, wz);

    float dist_val = np[0]*(wx-pp[0]) + np[1]*(wy-pp[1]) + np[2]*(wz-pp[2]);
    auto dist = mx::array(dist_val);
    auto vertex_world = mx::array({wx, wy, wz});
    auto pos = mx::subtract(vertex_world, mx::multiply(normal, dist));
    return {dist, pos, make_frame(normal)};
}

static CollisionResult sphere_box(
    const mx::array& sphere_pos, const mx::array& sphere_size,
    const mx::array& box_pos, const mx::array& box_mat, const mx::array& box_size)
{
    mx::eval(sphere_size); mx::eval(sphere_pos);
    mx::eval(box_pos); mx::eval(box_mat); mx::eval(box_size);
    float r = sphere_size.data<float>()[0];
    auto sp = sphere_pos.data<float>();
    auto bp = box_pos.data<float>();
    auto Rp = box_mat.data<float>();
    float hx = box_size.data<float>()[0];
    float hy = box_size.data<float>()[1];
    float hz = box_size.data<float>()[2];

    // Transform sphere center to box-local coordinates
    float lx, ly, lz;
    world_to_box_local(Rp, bp, sp[0], sp[1], sp[2], lx, ly, lz);

    // Closest point on box (local coords)
    float cx, cy, cz;
    closest_on_aabb(lx, ly, lz, hx, hy, hz, cx, cy, cz);

    // Transform closest point back to world
    float wx, wy, wz;
    box_local_to_world(Rp, bp, cx, cy, cz, wx, wy, wz);

    // Direction from closest point to sphere center
    float fx = sp[0] - wx, fy = sp[1] - wy, fz = sp[2] - wz;
    float d = std::sqrt(fx*fx + fy*fy + fz*fz);

    float nx, ny, nz;
    if (d < MJMINVAL) {
        // Sphere center is on or inside box surface — push out along
        // the axis of least penetration
        float pen[3] = {hx - std::abs(lx), hy - std::abs(ly), hz - std::abs(lz)};
        int best = 0;
        for (int i = 1; i < 3; i++)
            if (pen[i] < pen[best]) best = i;
        float sign = (best == 0 ? lx : best == 1 ? ly : lz) >= 0 ? 1.0f : -1.0f;
        // Normal is box face normal in world coords (column of R)
        nx = Rp[0*3+best] * sign;
        ny = Rp[1*3+best] * sign;
        nz = Rp[2*3+best] * sign;
        d = pen[best] + r;
    } else {
        nx = fx/d; ny = fy/d; nz = fz/d;
    }

    float dist_val = d - r;
    auto dist = mx::array(dist_val);
    auto normal_arr = mx::array({nx, ny, nz});
    auto pos_arr = mx::array({wx, wy, wz});
    return {dist, pos_arr, make_frame(normal_arr)};
}

static CollisionResult capsule_box(
    const mx::array& cap_pos, const mx::array& cap_mat, const mx::array& cap_size,
    const mx::array& box_pos, const mx::array& box_mat, const mx::array& box_size)
{
    mx::eval(cap_size); mx::eval(cap_pos); mx::eval(cap_mat);
    mx::eval(box_pos); mx::eval(box_mat); mx::eval(box_size);
    float r_c = cap_size.data<float>()[0];
    float half_len = cap_size.data<float>()[1];
    auto cp = cap_pos.data<float>();
    auto cm = cap_mat.data<float>();
    auto bp = box_pos.data<float>();
    auto Rp = box_mat.data<float>();
    float hx = box_size.data<float>()[0];
    float hy = box_size.data<float>()[1];
    float hz = box_size.data<float>()[2];

    // Capsule axis: z-column of capsule rotation matrix
    float ax = cm[2], ay = cm[5], az = cm[8];

    // Capsule endpoints
    float e0[3] = {cp[0] - ax*half_len, cp[1] - ay*half_len, cp[2] - az*half_len};
    float e1[3] = {cp[0] + ax*half_len, cp[1] + ay*half_len, cp[2] + az*half_len};

    // Test both endpoints + midpoint, find closest to box
    float best_dist_sq = 1e20f;
    float best_seg[3], best_box[3];

    float test_pts[3][3] = {
        {e0[0], e0[1], e0[2]},
        {e1[0], e1[1], e1[2]},
        {cp[0], cp[1], cp[2]}
    };

    for (int ti = 0; ti < 3; ti++) {
        float lx, ly, lz;
        world_to_box_local(Rp, bp, test_pts[ti][0], test_pts[ti][1], test_pts[ti][2],
                           lx, ly, lz);
        float cx, cy, cz;
        closest_on_aabb(lx, ly, lz, hx, hy, hz, cx, cy, cz);
        float wx, wy, wz;
        box_local_to_world(Rp, bp, cx, cy, cz, wx, wy, wz);

        // Now find closest point on capsule segment to this box point
        // Project box point onto capsule line: t = dot(box_pt - e0, axis) / |axis|^2
        float dx = wx - e0[0], dy = wy - e0[1], dz = wz - e0[2];
        float seg_x = e1[0]-e0[0], seg_y = e1[1]-e0[1], seg_z = e1[2]-e0[2];
        float seg_sq = seg_x*seg_x + seg_y*seg_y + seg_z*seg_z;
        float t = (seg_sq > MJMINVAL) ? (dx*seg_x + dy*seg_y + dz*seg_z) / seg_sq : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        float sx = e0[0] + t*seg_x, sy = e0[1] + t*seg_y, sz = e0[2] + t*seg_z;

        // Get new closest on box to this refined segment point
        float lx2, ly2, lz2;
        world_to_box_local(Rp, bp, sx, sy, sz, lx2, ly2, lz2);
        float cx2, cy2, cz2;
        closest_on_aabb(lx2, ly2, lz2, hx, hy, hz, cx2, cy2, cz2);
        float wx2, wy2, wz2;
        box_local_to_world(Rp, bp, cx2, cy2, cz2, wx2, wy2, wz2);

        float fx = sx - wx2, fy = sy - wy2, fz = sz - wz2;
        float dsq = fx*fx + fy*fy + fz*fz;
        if (dsq < best_dist_sq) {
            best_dist_sq = dsq;
            best_seg[0] = sx; best_seg[1] = sy; best_seg[2] = sz;
            best_box[0] = wx2; best_box[1] = wy2; best_box[2] = wz2;
        }
    }

    float fx = best_seg[0] - best_box[0];
    float fy = best_seg[1] - best_box[1];
    float fz = best_seg[2] - best_box[2];
    float d = std::sqrt(fx*fx + fy*fy + fz*fz);
    float nx, ny, nz;
    if (d < MJMINVAL) {
        nx = 0; ny = 0; nz = 1;
    } else {
        nx = fx/d; ny = fy/d; nz = fz/d;
    }

    float dist_val = d - r_c;
    auto dist = mx::array(dist_val);
    auto normal_arr = mx::array({nx, ny, nz});
    auto pos_arr = mx::array({best_box[0], best_box[1], best_box[2]});
    return {dist, pos_arr, make_frame(normal_arr)};
}

static CollisionResult box_box(
    const mx::array& pos1, const mx::array& mat1, const mx::array& size1,
    const mx::array& pos2, const mx::array& mat2, const mx::array& size2)
{
    mx::eval(pos1); mx::eval(mat1); mx::eval(size1);
    mx::eval(pos2); mx::eval(mat2); mx::eval(size2);
    auto p1 = pos1.data<float>(); auto R1 = mat1.data<float>(); auto s1 = size1.data<float>();
    auto p2 = pos2.data<float>(); auto R2 = mat2.data<float>(); auto s2 = size2.data<float>();

    float h1[3] = {s1[0], s1[1], s1[2]};
    float h2[3] = {s2[0], s2[1], s2[2]};

    // Center difference
    float T[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};

    // SAT: 15 potential separating axes
    // Axes: R1 columns (3), R2 columns (3), cross products (9)
    float best_overlap = 1e10f;
    float best_axis[3] = {0, 0, 1};

    auto get_col = [](const float* R, int j, float* out) {
        out[0] = R[0*3+j]; out[1] = R[1*3+j]; out[2] = R[2*3+j];
    };

    auto dot3 = [](const float* a, const float* b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    auto test_axis = [&](float ax, float ay, float az) -> bool {
        float len = std::sqrt(ax*ax + ay*ay + az*az);
        if (len < 1e-8f) return true; // degenerate axis, skip
        ax /= len; ay /= len; az /= len;

        // Project box1 half-extents onto axis
        float r1_proj = 0;
        for (int j = 0; j < 3; j++) {
            float col[3]; get_col(R1, j, col);
            r1_proj += h1[j] * std::abs(dot3(col, (float[]){ax, ay, az}));
        }
        // Project box2 half-extents onto axis
        float r2_proj = 0;
        for (int j = 0; j < 3; j++) {
            float col[3]; get_col(R2, j, col);
            r2_proj += h2[j] * std::abs(dot3(col, (float[]){ax, ay, az}));
        }
        // Distance between centers projected onto axis
        float d_proj = std::abs(T[0]*ax + T[1]*ay + T[2]*az);
        float overlap = r1_proj + r2_proj - d_proj;
        if (overlap < 0) return false; // separating axis found
        if (overlap < best_overlap) {
            best_overlap = overlap;
            // Choose sign so axis points from box1 to box2
            float sign = (T[0]*ax + T[1]*ay + T[2]*az >= 0) ? 1.0f : -1.0f;
            best_axis[0] = ax * sign;
            best_axis[1] = ay * sign;
            best_axis[2] = az * sign;
        }
        return true;
    };

    bool colliding = true;

    // Test 3 face normals of box1
    for (int j = 0; j < 3 && colliding; j++) {
        float col[3]; get_col(R1, j, col);
        colliding = test_axis(col[0], col[1], col[2]);
    }
    // Test 3 face normals of box2
    for (int j = 0; j < 3 && colliding; j++) {
        float col[3]; get_col(R2, j, col);
        colliding = test_axis(col[0], col[1], col[2]);
    }
    // Test 9 edge-edge cross products
    for (int i = 0; i < 3 && colliding; i++) {
        float a[3]; get_col(R1, i, a);
        for (int j = 0; j < 3 && colliding; j++) {
            float b[3]; get_col(R2, j, b);
            float cx = a[1]*b[2] - a[2]*b[1];
            float cy = a[2]*b[0] - a[0]*b[2];
            float cz = a[0]*b[1] - a[1]*b[0];
            colliding = test_axis(cx, cy, cz);
        }
    }

    if (!colliding) {
        return {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};
    }

    // Contact point: midpoint between the two closest support points
    // along the separating axis
    float support1[3] = {p1[0], p1[1], p1[2]};
    float support2[3] = {p2[0], p2[1], p2[2]};
    for (int j = 0; j < 3; j++) {
        float col1[3]; get_col(R1, j, col1);
        float d = dot3(col1, best_axis);
        float sign = (d < 0) ? 1.0f : -1.0f;
        support1[0] += sign * h1[j] * col1[0];
        support1[1] += sign * h1[j] * col1[1];
        support1[2] += sign * h1[j] * col1[2];

        float col2[3]; get_col(R2, j, col2);
        d = dot3(col2, best_axis);
        sign = (d > 0) ? 1.0f : -1.0f;
        support2[0] += sign * h2[j] * col2[0];
        support2[1] += sign * h2[j] * col2[1];
        support2[2] += sign * h2[j] * col2[2];
    }

    float contact_pos[3] = {
        0.5f * (support1[0] + support2[0]),
        0.5f * (support1[1] + support2[1]),
        0.5f * (support1[2] + support2[2])
    };

    auto dist = mx::array(-best_overlap);
    auto normal_arr = mx::array({best_axis[0], best_axis[1], best_axis[2]});
    auto pos_arr = mx::array({contact_pos[0], contact_pos[1], contact_pos[2]});
    return {dist, pos_arr, make_frame(normal_arr)};
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
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::BOX)) {
                // Multi-contact: up to 4 contacts from plane-box
                float margin = gmargin[g1_] + gmargin[g2_];
                int condim = 3;
                if (m.geom_condim.size() > 0) {
                    mx::eval(m.geom_condim);
                    auto cdp = m.geom_condim.data<int>();
                    condim = std::max(cdp[g1_], cdp[g2_]);
                }
                plane_box_multi(gpos1, gmat1, gpos2, gmat2, gsize2,
                                margin, g1_, g2_, condim,
                                c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::BOX)) {
                result = sphere_box(gpos1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::CAPSULE) && t2_ == static_cast<int>(GeomType::BOX)) {
                result = capsule_box(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::BOX) && t2_ == static_cast<int>(GeomType::BOX)) {
                result = box_box(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
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
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::BOX)) {
                // Multi-contact plane-box for explicit pairs
                float margin_pb = gmargin[g1_] + gmargin[g2_];
                if (m.pair_margin.size() > 0) {
                    mx::eval(m.pair_margin);
                    margin_pb = m.pair_margin.data<float>()[pi];
                }
                int condim_pb = 3;
                if (m.pair_dim.size() > 0) {
                    mx::eval(m.pair_dim);
                    condim_pb = m.pair_dim.data<int>()[pi];
                }
                plane_box_multi(gpos1, gmat1, gpos2, gmat2, gsize2,
                                margin_pb, g1_, g2_, condim_pb,
                                c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::BOX)) {
                result = sphere_box(gpos1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::CAPSULE) && t2_ == static_cast<int>(GeomType::BOX)) {
                result = capsule_box(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::BOX) && t2_ == static_cast<int>(GeomType::BOX)) {
                result = box_box(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
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
