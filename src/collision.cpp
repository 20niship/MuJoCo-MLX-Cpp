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

// ── CYLINDER collision functions ──────────────────────────────

// Closest point on a cylinder surface to an external point, all in cylinder-local coords.
// Cylinder: radius R, half-length H, axis = local z. Surface = barrel + two caps.
static void closest_on_cylinder_local(float px, float py, float pz,
                                       float R, float H,
                                       float& cx, float& cy, float& cz) {
    float rho = std::sqrt(px*px + py*py);
    float clamped_z = std::max(-H, std::min(H, pz));

    if (rho <= R && std::abs(pz) <= H) {
        // Point is INSIDE cylinder — push out to nearest surface
        float d_barrel = R - rho;
        float d_top = H - pz;
        float d_bot = pz + H;
        float d_min = std::min({d_barrel, d_top, d_bot});
        if (d_min == d_barrel && rho > MJMINVAL) {
            cx = px * R / rho; cy = py * R / rho; cz = pz;
        } else if (d_min == d_top) {
            cx = px; cy = py; cz = H;
        } else {
            cx = px; cy = py; cz = -H;
        }
    } else if (rho <= R) {
        // Above/below cap
        cx = px; cy = py; cz = (pz > 0) ? H : -H;
    } else if (std::abs(pz) <= H) {
        // Beside barrel
        cx = px * R / rho; cy = py * R / rho; cz = pz;
    } else {
        // Diagonal: closest to rim
        cx = (rho > MJMINVAL) ? px * R / rho : R;
        cy = (rho > MJMINVAL) ? py * R / rho : 0.0f;
        cz = (pz > 0) ? H : -H;
    }
}

// plane_cylinder_multi: returns up to 3 contacts per face (2 rim points + center)
// for a total of up to 6 candidate contacts. Matches MuJoCo C's mjc_PlaneCylinder.
static int plane_cylinder_multi(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& cyl_pos, const mx::array& cyl_mat, const mx::array& cyl_size,
    float margin, int g1, int g2, int condim,
    std::vector<mx::array>& c_dist, std::vector<mx::array>& c_pos,
    std::vector<mx::array>& c_frame, std::vector<mx::array>& c_geom,
    std::vector<int>& c_dim)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);
    auto frame = make_frame(normal);

    mx::eval(cyl_size); mx::eval(cyl_pos); mx::eval(cyl_mat);
    mx::eval(normal); mx::eval(plane_pos);
    float R = cyl_size.data<float>()[0];
    float H = cyl_size.data<float>()[1];

    auto np = normal.data<float>();
    auto Rp = cyl_mat.data<float>();
    auto cp = cyl_pos.data<float>();
    auto pp = plane_pos.data<float>();

    // Cylinder axis = z-column of rotation matrix
    float ax = Rp[2], ay = Rp[5], az = Rp[8];

    // Component of normal perpendicular to cylinder axis
    float ndota = np[0]*ax + np[1]*ay + np[2]*az;
    float perp[3] = {np[0] - ndota*ax, np[1] - ndota*ay, np[2] - ndota*az};
    float perp_len = std::sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2]);

    // Normalized perpendicular direction on the face plane
    float px = 0, py = 0, pz = 0;
    // Second orthogonal direction on face plane
    float qx = 0, qy = 0, qz = 0;
    if (perp_len > 1e-6f) {
        px = perp[0] / perp_len; py = perp[1] / perp_len; pz = perp[2] / perp_len;
        // q = axis × p (orthogonal to both axis and perp, lies in the face plane)
        qx = ay*pz - az*py; qy = az*px - ax*pz; qz = ax*py - ay*px;
    } else {
        // Normal parallel to axis — pick arbitrary face directions
        // Use x-column and y-column of rotation matrix
        px = Rp[0]; py = Rp[3]; pz = Rp[6];
        qx = Rp[1]; qy = Rp[4]; qz = Rp[7];
    }

    auto add_contact = [&](float wx, float wy, float wz) -> bool {
        float d = np[0]*(wx-pp[0]) + np[1]*(wy-pp[1]) + np[2]*(wz-pp[2]);
        if (d < margin) {
            c_dist.push_back(mx::array(d));
            auto vertex = mx::array({wx, wy, wz});
            c_pos.push_back(mx::subtract(vertex, mx::multiply(normal, mx::array(d))));
            c_frame.push_back(frame);
            c_geom.push_back(mx::array({g1, g2}, mx::int32));
            c_dim.push_back(condim);
            return true;
        }
        return false;
    };

    int ncon_added = 0;
    for (int s = 0; s < 2; s++) {
        float sign = (s == 0) ? -1.0f : 1.0f;
        float fcx = cp[0] + sign * H * ax;
        float fcy = cp[1] + sign * H * ay;
        float fcz = cp[2] + sign * H * az;

        // Face center
        if (add_contact(fcx, fcy, fcz)) ncon_added++;

        // Rim point along -perp direction (closest to plane)
        if (add_contact(fcx - R*px, fcy - R*py, fcz - R*pz)) ncon_added++;

        // Rim point along +perp direction (for degenerate/upright case: second diameter point)
        if (perp_len < 1e-6f) {
            // Normal || axis: add a second rim point in orthogonal direction
            if (add_contact(fcx + R*qx, fcy + R*qy, fcz + R*qz)) ncon_added++;
        }
    }

    return ncon_added;
}

// Single-contact plane_cylinder (for vmap dispatch)
static CollisionResult plane_cylinder(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& cyl_pos, const mx::array& cyl_mat, const mx::array& cyl_size)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);

    mx::eval(cyl_size); mx::eval(cyl_pos); mx::eval(cyl_mat);
    mx::eval(normal); mx::eval(plane_pos);
    float R = cyl_size.data<float>()[0];
    float H = cyl_size.data<float>()[1];

    auto np = normal.data<float>();
    auto Rp = cyl_mat.data<float>();
    auto cp = cyl_pos.data<float>();
    auto pp = plane_pos.data<float>();

    float ax = Rp[2], ay = Rp[5], az = Rp[8];
    float ndota = np[0]*ax + np[1]*ay + np[2]*az;
    float perp[3] = {np[0] - ndota*ax, np[1] - ndota*ay, np[2] - ndota*az};
    float perp_len = std::sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2]);

    float best_dist = 1e10f;
    float best_wx = 0, best_wy = 0, best_wz = 0;

    for (int s = 0; s < 2; s++) {
        float sign = (s == 0) ? -1.0f : 1.0f;
        float fcx = cp[0] + sign * H * ax;
        float fcy = cp[1] + sign * H * ay;
        float fcz = cp[2] + sign * H * az;

        float wx, wy, wz;
        if (perp_len > MJMINVAL) {
            wx = fcx - R * perp[0] / perp_len;
            wy = fcy - R * perp[1] / perp_len;
            wz = fcz - R * perp[2] / perp_len;
        } else {
            wx = fcx; wy = fcy; wz = fcz;
        }

        float d = np[0]*(wx-pp[0]) + np[1]*(wy-pp[1]) + np[2]*(wz-pp[2]);
        if (d < best_dist) {
            best_dist = d;
            best_wx = wx; best_wy = wy; best_wz = wz;
        }
    }

    auto dist = mx::array(best_dist);
    auto vertex = mx::array({best_wx, best_wy, best_wz});
    auto pos = mx::subtract(vertex, mx::multiply(normal, dist));
    return {dist, pos, make_frame(normal)};
}

static CollisionResult sphere_cylinder(
    const mx::array& sphere_pos, const mx::array& sphere_size,
    const mx::array& cyl_pos, const mx::array& cyl_mat, const mx::array& cyl_size)
{
    mx::eval(sphere_size); mx::eval(sphere_pos);
    mx::eval(cyl_pos); mx::eval(cyl_mat); mx::eval(cyl_size);
    float r = sphere_size.data<float>()[0];
    float R = cyl_size.data<float>()[0];
    float H = cyl_size.data<float>()[1];
    auto sp = sphere_pos.data<float>();
    auto cyp = cyl_pos.data<float>();
    auto Rp = cyl_mat.data<float>();

    // Transform sphere center to cylinder-local coords
    float lx, ly, lz;
    world_to_box_local(Rp, cyp, sp[0], sp[1], sp[2], lx, ly, lz);

    // Closest point on cylinder surface
    float cx, cy, cz;
    closest_on_cylinder_local(lx, ly, lz, R, H, cx, cy, cz);

    // Transform back to world
    float wx, wy, wz;
    box_local_to_world(Rp, cyp, cx, cy, cz, wx, wy, wz);

    float fx = sp[0] - wx, fy = sp[1] - wy, fz = sp[2] - wz;
    float d = std::sqrt(fx*fx + fy*fy + fz*fz);
    float nx, ny, nz;
    if (d < MJMINVAL) {
        nx = 0; ny = 0; nz = 1;
    } else {
        nx = fx/d; ny = fy/d; nz = fz/d;
    }

    float dist_val = d - r;
    return {mx::array(dist_val), mx::array({wx, wy, wz}), make_frame(mx::array({nx, ny, nz}))};
}

static CollisionResult capsule_cylinder(
    const mx::array& cap_pos, const mx::array& cap_mat, const mx::array& cap_size,
    const mx::array& cyl_pos, const mx::array& cyl_mat, const mx::array& cyl_size)
{
    mx::eval(cap_size); mx::eval(cap_pos); mx::eval(cap_mat);
    mx::eval(cyl_pos); mx::eval(cyl_mat); mx::eval(cyl_size);
    float r_c = cap_size.data<float>()[0];
    float half_len = cap_size.data<float>()[1];
    float R = cyl_size.data<float>()[0];
    float H = cyl_size.data<float>()[1];
    auto cap_p = cap_pos.data<float>();
    auto cm = cap_mat.data<float>();
    auto cyp = cyl_pos.data<float>();
    auto Rp = cyl_mat.data<float>();

    // Capsule axis (z-column of capsule mat)
    float cax = cm[2], cay = cm[5], caz = cm[8];
    float e0[3] = {cap_p[0] - cax*half_len, cap_p[1] - cay*half_len, cap_p[2] - caz*half_len};
    float e1[3] = {cap_p[0] + cax*half_len, cap_p[1] + cay*half_len, cap_p[2] + caz*half_len};

    // Test endpoints + midpoint against cylinder
    float best_dist_sq = 1e20f;
    float best_seg[3], best_cyl[3];

    float test_pts[3][3] = {
        {e0[0], e0[1], e0[2]},
        {e1[0], e1[1], e1[2]},
        {cap_p[0], cap_p[1], cap_p[2]}
    };

    for (int ti = 0; ti < 3; ti++) {
        // Transform to cylinder local
        float lx, ly, lz;
        world_to_box_local(Rp, cyp, test_pts[ti][0], test_pts[ti][1], test_pts[ti][2],
                           lx, ly, lz);
        float cx, cy, cz;
        closest_on_cylinder_local(lx, ly, lz, R, H, cx, cy, cz);
        float wx, wy, wz;
        box_local_to_world(Rp, cyp, cx, cy, cz, wx, wy, wz);

        // Project cylinder point back onto capsule segment
        float dx = wx - e0[0], dy = wy - e0[1], dz = wz - e0[2];
        float seg_x = e1[0]-e0[0], seg_y = e1[1]-e0[1], seg_z = e1[2]-e0[2];
        float seg_sq = seg_x*seg_x + seg_y*seg_y + seg_z*seg_z;
        float t = (seg_sq > MJMINVAL) ? (dx*seg_x + dy*seg_y + dz*seg_z) / seg_sq : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        float sx = e0[0] + t*seg_x, sy = e0[1] + t*seg_y, sz = e0[2] + t*seg_z;

        // Get refined closest on cylinder
        float lx2, ly2, lz2;
        world_to_box_local(Rp, cyp, sx, sy, sz, lx2, ly2, lz2);
        float cx2, cy2, cz2;
        closest_on_cylinder_local(lx2, ly2, lz2, R, H, cx2, cy2, cz2);
        float wx2, wy2, wz2;
        box_local_to_world(Rp, cyp, cx2, cy2, cz2, wx2, wy2, wz2);

        float fx = sx - wx2, fy = sy - wy2, fz = sz - wz2;
        float dsq = fx*fx + fy*fy + fz*fz;
        if (dsq < best_dist_sq) {
            best_dist_sq = dsq;
            best_seg[0] = sx; best_seg[1] = sy; best_seg[2] = sz;
            best_cyl[0] = wx2; best_cyl[1] = wy2; best_cyl[2] = wz2;
        }
    }

    float fx = best_seg[0] - best_cyl[0];
    float fy = best_seg[1] - best_cyl[1];
    float fz = best_seg[2] - best_cyl[2];
    float d = std::sqrt(fx*fx + fy*fy + fz*fz);
    float nx, ny, nz;
    if (d < MJMINVAL) {
        nx = 0; ny = 0; nz = 1;
    } else {
        nx = fx/d; ny = fy/d; nz = fz/d;
    }

    float dist_val = d - r_c;
    return {mx::array(dist_val), mx::array({best_cyl[0], best_cyl[1], best_cyl[2]}),
            make_frame(mx::array({nx, ny, nz}))};
}

// ── GJK/EPA convex collision ──────────────────────────────────
// Used for any pair involving MESH geoms (and as fallback for other convex pairs).

struct Vec3 { float x, y, z; };
static Vec3 v3sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static Vec3 v3add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3 v3scale(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
static float v3dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3 v3cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static float v3len(Vec3 a) { return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
static Vec3 v3neg(Vec3 a) { return {-a.x, -a.y, -a.z}; }
static Vec3 v3norm(Vec3 a) {
    float l = v3len(a);
    return l > 1e-12f ? v3scale(a, 1.0f/l) : Vec3{0,0,1};
}

// Support function: furthest point on convex geom in a given direction (world space)
struct ConvexGeom {
    int type;
    float pos[3];
    float mat[9]; // 3x3 rotation, row-major
    float size[3];
    // For mesh: vertex data
    const float* verts = nullptr;
    int nverts = 0;
};

static Vec3 support(const ConvexGeom& g, Vec3 dir) {
    Vec3 center = {g.pos[0], g.pos[1], g.pos[2]};

    switch (g.type) {
    case (int)GeomType::SPHERE: {
        float r = g.size[0];
        Vec3 nd = v3norm(dir);
        return v3add(center, v3scale(nd, r));
    }
    case (int)GeomType::CAPSULE: {
        float r = g.size[0];
        float h = g.size[1];
        // Axis = z-column of rotation matrix
        Vec3 axis = {g.mat[2], g.mat[5], g.mat[8]};
        // Pick the endpoint most aligned with dir
        float d = v3dot(axis, dir);
        Vec3 endpoint = v3add(center, v3scale(axis, d >= 0 ? h : -h));
        Vec3 nd = v3norm(dir);
        return v3add(endpoint, v3scale(nd, r));
    }
    case (int)GeomType::BOX: {
        float hx = g.size[0], hy = g.size[1], hz = g.size[2];
        Vec3 xc = {g.mat[0], g.mat[3], g.mat[6]}; // x-column
        Vec3 yc = {g.mat[1], g.mat[4], g.mat[7]}; // y-column
        Vec3 zc = {g.mat[2], g.mat[5], g.mat[8]}; // z-column
        float sx = (v3dot(xc, dir) >= 0) ? hx : -hx;
        float sy = (v3dot(yc, dir) >= 0) ? hy : -hy;
        float sz = (v3dot(zc, dir) >= 0) ? hz : -hz;
        return {center.x + sx*xc.x + sy*yc.x + sz*zc.x,
                center.y + sx*xc.y + sy*yc.y + sz*zc.y,
                center.z + sx*xc.z + sy*yc.z + sz*zc.z};
    }
    case (int)GeomType::CYLINDER: {
        float R = g.size[0], H = g.size[1];
        Vec3 axis = {g.mat[2], g.mat[5], g.mat[8]}; // z-column
        // Along axis: pick the sign that aligns with dir
        float da = v3dot(axis, dir);
        Vec3 tip = v3add(center, v3scale(axis, da >= 0 ? H : -H));
        // Perpendicular to axis in dir direction
        float proj = v3dot(dir, axis);
        Vec3 perp = {dir.x - proj*axis.x, dir.y - proj*axis.y, dir.z - proj*axis.z};
        float pl = v3len(perp);
        if (pl > 1e-12f) {
            return v3add(tip, v3scale(perp, R / pl));
        }
        return tip;
    }
    case (int)GeomType::MESH: {
        // Find vertex with maximum dot product with dir
        float best_dot = -1e20f;
        int best_i = 0;
        for (int i = 0; i < g.nverts; i++) {
            float d = g.verts[i*3]*dir.x + g.verts[i*3+1]*dir.y + g.verts[i*3+2]*dir.z;
            if (d > best_dot) { best_dot = d; best_i = i; }
        }
        // Mesh vertices are in local frame; transform to world
        float lx = g.verts[best_i*3], ly = g.verts[best_i*3+1], lz = g.verts[best_i*3+2];
        float wx = center.x + g.mat[0]*lx + g.mat[1]*ly + g.mat[2]*lz;
        float wy = center.y + g.mat[3]*lx + g.mat[4]*ly + g.mat[5]*lz;
        float wz = center.z + g.mat[6]*lx + g.mat[7]*ly + g.mat[8]*lz;
        return {wx, wy, wz};
    }
    default:
        return center;
    }
}

// GJK: Minkowski difference support
struct MinkowskiPoint {
    Vec3 diff; // A - B in Minkowski space
    Vec3 a;    // support point on A
    Vec3 b;    // support point on B
};

static MinkowskiPoint mink_support(const ConvexGeom& A, const ConvexGeom& B, Vec3 dir) {
    Vec3 sa = support(A, dir);
    Vec3 sb = support(B, v3neg(dir));
    return {v3sub(sa, sb), sa, sb};
}

// GJK: update simplex toward origin. Returns true if origin is contained.
struct GJKSimplex {
    MinkowskiPoint pts[4];
    int n = 0;
    Vec3 dir;
};

static bool gjk_line(GJKSimplex& s) {
    Vec3 A = s.pts[1].diff, B = s.pts[0].diff;
    Vec3 AB = v3sub(B, A);
    Vec3 AO = v3neg(A);
    if (v3dot(AB, AO) > 0) {
        s.dir = v3cross(v3cross(AB, AO), AB);
        if (v3len(s.dir) < 1e-12f) s.dir = v3cross(AB, {1,0,0});
        if (v3len(s.dir) < 1e-12f) s.dir = v3cross(AB, {0,1,0});
    } else {
        s.pts[0] = s.pts[1];
        s.n = 1;
        s.dir = AO;
    }
    return false;
}

static bool gjk_triangle(GJKSimplex& s) {
    Vec3 A = s.pts[2].diff, B = s.pts[1].diff, C = s.pts[0].diff;
    Vec3 AB = v3sub(B, A), AC = v3sub(C, A), AO = v3neg(A);
    Vec3 ABC = v3cross(AB, AC);

    Vec3 ABperp = v3cross(AB, ABC);
    Vec3 ACperp = v3cross(ABC, AC);

    if (v3dot(ACperp, AO) > 0) {
        if (v3dot(AC, AO) > 0) {
            s.pts[0] = s.pts[0]; // C stays
            s.pts[1] = s.pts[2]; // A
            s.n = 2;
            s.dir = v3cross(v3cross(AC, AO), AC);
        } else {
            s.pts[0] = s.pts[1]; s.pts[1] = s.pts[2]; s.n = 2;
            return gjk_line(s);
        }
    } else if (v3dot(ABperp, AO) > 0) {
        s.pts[0] = s.pts[1]; s.pts[1] = s.pts[2]; s.n = 2;
        return gjk_line(s);
    } else {
        if (v3dot(ABC, AO) > 0) {
            s.dir = ABC;
        } else {
            // Flip winding
            MinkowskiPoint tmp = s.pts[0]; s.pts[0] = s.pts[1]; s.pts[1] = tmp;
            s.dir = v3neg(ABC);
        }
    }
    return false;
}

static bool gjk_tetrahedron(GJKSimplex& s) {
    Vec3 A = s.pts[3].diff, B = s.pts[2].diff, C = s.pts[1].diff, D = s.pts[0].diff;
    Vec3 AB = v3sub(B, A), AC = v3sub(C, A), AD = v3sub(D, A), AO = v3neg(A);

    Vec3 ABC = v3cross(AB, AC);
    Vec3 ACD = v3cross(AC, AD);
    Vec3 ADB = v3cross(AD, AB);

    if (v3dot(ABC, AO) > 0) {
        s.pts[0] = s.pts[1]; s.pts[1] = s.pts[2]; s.pts[2] = s.pts[3]; s.n = 3;
        return gjk_triangle(s);
    }
    if (v3dot(ACD, AO) > 0) {
        s.pts[1] = s.pts[0]; s.pts[0] = s.pts[1];
        s.pts[0] = s.pts[0]; s.pts[1] = s.pts[2]; s.pts[2] = s.pts[3]; s.n = 3;
        // Reorder: keep A, C, D
        s.pts[0] = {D.x ? s.pts[0] : s.pts[0]}; // keep D
        // Simpler: rebuild
        MinkowskiPoint pA = s.pts[3], pC = s.pts[1], pD = s.pts[0];
        s.pts[0] = pD; s.pts[1] = pC; s.pts[2] = pA; s.n = 3;
        return gjk_triangle(s);
    }
    if (v3dot(ADB, AO) > 0) {
        MinkowskiPoint pA = s.pts[3], pD = s.pts[0], pB = s.pts[2];
        s.pts[0] = pB; s.pts[1] = pD; s.pts[2] = pA; s.n = 3;
        return gjk_triangle(s);
    }
    // Origin is inside tetrahedron
    return true;
}

// Run GJK. Returns true if shapes overlap. simplex is the final GJK simplex.
static bool gjk(const ConvexGeom& A, const ConvexGeom& B, GJKSimplex& simplex) {
    Vec3 initial_dir = v3sub(Vec3{B.pos[0], B.pos[1], B.pos[2]},
                              Vec3{A.pos[0], A.pos[1], A.pos[2]});
    if (v3len(initial_dir) < 1e-12f) initial_dir = {1, 0, 0};

    simplex.pts[0] = mink_support(A, B, initial_dir);
    simplex.n = 1;

    simplex.dir = v3neg(simplex.pts[0].diff);
    if (v3len(simplex.dir) < 1e-12f) return true; // origin at support = overlap

    for (int iter = 0; iter < 64; iter++) {
        MinkowskiPoint p = mink_support(A, B, simplex.dir);
        if (v3dot(p.diff, simplex.dir) < 0) {
            return false; // no overlap
        }
        simplex.pts[simplex.n++] = p;

        bool contains_origin = false;
        switch (simplex.n) {
        case 2: contains_origin = gjk_line(simplex); break;
        case 3: contains_origin = gjk_triangle(simplex); break;
        case 4: contains_origin = gjk_tetrahedron(simplex); break;
        }
        if (contains_origin) return true;
    }
    return false;
}

// EPA: Expanding Polytope Algorithm
// Given a GJK simplex containing the origin, find the closest face and expand.
struct EPAFace {
    int a, b, c;
    Vec3 normal;
    float dist;
};

struct EPAResult {
    Vec3 normal;
    float depth;
    Vec3 point_a; // witness on shape A
    Vec3 point_b; // witness on shape B
};

static EPAResult epa(const ConvexGeom& A, const ConvexGeom& B, GJKSimplex& simplex) {
    static constexpr int MAX_VERTS = 128;
    static constexpr int MAX_FACES = 256;
    static constexpr int MAX_ITER = 64;

    MinkowskiPoint verts[MAX_VERTS];
    int nverts = 0;

    // Initialize with tetrahedron from GJK simplex
    if (simplex.n < 4) {
        // Need a full tetrahedron. Expand simplex.
        // Add support points in cardinal directions to build a tetrahedron
        Vec3 dirs[6] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        while (simplex.n < 4) {
            for (auto& d : dirs) {
                auto p = mink_support(A, B, d);
                bool dup = false;
                for (int i = 0; i < simplex.n; i++) {
                    Vec3 diff = v3sub(p.diff, simplex.pts[i].diff);
                    if (v3len(diff) < 1e-8f) { dup = true; break; }
                }
                if (!dup) {
                    simplex.pts[simplex.n++] = p;
                    if (simplex.n >= 4) break;
                }
            }
            if (simplex.n < 4) {
                // Degenerate case: shapes barely touching
                Vec3 n = v3norm(v3sub(Vec3{A.pos[0],A.pos[1],A.pos[2]},
                                       Vec3{B.pos[0],B.pos[1],B.pos[2]}));
                return {n, 0.0f, {A.pos[0],A.pos[1],A.pos[2]}, {B.pos[0],B.pos[1],B.pos[2]}};
            }
        }
    }

    for (int i = 0; i < 4; i++) verts[nverts++] = simplex.pts[i];

    // Build initial faces (tetrahedron = 4 faces)
    EPAFace faces[MAX_FACES];
    int nfaces = 0;

    auto add_face = [&](int a, int b, int c) {
        if (nfaces >= MAX_FACES) return;
        Vec3 AB = v3sub(verts[b].diff, verts[a].diff);
        Vec3 AC = v3sub(verts[c].diff, verts[a].diff);
        Vec3 n = v3cross(AB, AC);
        float nl = v3len(n);
        if (nl < 1e-12f) return;
        n = v3scale(n, 1.0f/nl);
        float d = v3dot(n, verts[a].diff);
        if (d < 0) { n = v3neg(n); d = -d; std::swap(b, c); }
        faces[nfaces++] = {a, b, c, n, d};
    };

    add_face(0,1,2); add_face(0,2,3); add_face(0,3,1); add_face(1,3,2);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        if (nfaces == 0) break;

        // Find closest face to origin
        int best_f = 0;
        float best_dist = faces[0].dist;
        for (int i = 1; i < nfaces; i++) {
            if (faces[i].dist < best_dist) { best_dist = faces[i].dist; best_f = i; }
        }

        EPAFace& closest = faces[best_f];
        Vec3 search_dir = closest.normal;

        // Get new support point
        auto p = mink_support(A, B, search_dir);
        float proj = v3dot(p.diff, search_dir);

        if (proj - best_dist < 1e-4f || nverts >= MAX_VERTS) {
            // Converged: compute witness points via barycentric coords on closest face
            Vec3 a = verts[closest.a].diff;
            Vec3 b = verts[closest.b].diff;
            Vec3 c = verts[closest.c].diff;

            // Project origin onto face plane, then barycentric
            Vec3 v0 = v3sub(b, a), v1 = v3sub(c, a), v2 = v3neg(a); // 0 - a
            float d00 = v3dot(v0, v0), d01 = v3dot(v0, v1), d11 = v3dot(v1, v1);
            float d20 = v3dot(v2, v0), d21 = v3dot(v2, v1);
            float denom = d00*d11 - d01*d01;
            float u = (denom > 1e-12f) ? (d11*d20 - d01*d21) / denom : 1.0f/3;
            float v = (denom > 1e-12f) ? (d00*d21 - d01*d20) / denom : 1.0f/3;
            float w = 1.0f - u - v;
            u = std::max(0.0f, u); v = std::max(0.0f, v); w = std::max(0.0f, w);
            float sum = u + v + w;
            if (sum > 1e-8f) { u /= sum; v /= sum; w /= sum; }

            Vec3 pa = v3add(v3add(v3scale(verts[closest.a].a, w),
                                   v3scale(verts[closest.b].a, u)),
                            v3scale(verts[closest.c].a, v));
            Vec3 pb = v3add(v3add(v3scale(verts[closest.a].b, w),
                                   v3scale(verts[closest.b].b, u)),
                            v3scale(verts[closest.c].b, v));

            return {search_dir, best_dist, pa, pb};
        }

        // Add new vertex
        verts[nverts] = p;
        int new_idx = nverts++;

        // Remove faces visible from new point, collect horizon edges
        struct Edge { int a, b; };
        Edge horizon[MAX_FACES * 3];
        int nhorizon = 0;

        bool removed[MAX_FACES] = {};
        for (int i = 0; i < nfaces; i++) {
            Vec3 to_p = v3sub(p.diff, verts[faces[i].a].diff);
            if (v3dot(faces[i].normal, to_p) > 0) removed[i] = true;
        }

        for (int i = 0; i < nfaces; i++) {
            if (!removed[i]) continue;
            int edges[3][2] = {{faces[i].a, faces[i].b},
                               {faces[i].b, faces[i].c},
                               {faces[i].c, faces[i].a}};
            for (auto& e : edges) {
                bool shared = false;
                for (int j = 0; j < nfaces; j++) {
                    if (j == i || !removed[j]) continue;
                    int fe[3][2] = {{faces[j].a, faces[j].b},
                                    {faces[j].b, faces[j].c},
                                    {faces[j].c, faces[j].a}};
                    for (auto& fe2 : fe) {
                        if ((fe2[0] == e[1] && fe2[1] == e[0]) ||
                            (fe2[0] == e[0] && fe2[1] == e[1])) {
                            shared = true; break;
                        }
                    }
                    if (shared) break;
                }
                if (!shared && nhorizon < MAX_FACES * 3) {
                    horizon[nhorizon++] = {e[0], e[1]};
                }
            }
        }

        // Compact face array
        int write = 0;
        for (int i = 0; i < nfaces; i++) {
            if (!removed[i]) faces[write++] = faces[i];
        }
        nfaces = write;

        // Add new faces from horizon edges to new vertex
        for (int i = 0; i < nhorizon; i++) {
            add_face(horizon[i].a, horizon[i].b, new_idx);
        }
    }

    // Fallback
    Vec3 n = v3norm(v3sub(Vec3{A.pos[0],A.pos[1],A.pos[2]},
                           Vec3{B.pos[0],B.pos[1],B.pos[2]}));
    return {n, 0.0f, {A.pos[0],A.pos[1],A.pos[2]}, {B.pos[0],B.pos[1],B.pos[2]}};
}

// GJK closest distance (for non-overlapping shapes)
static float gjk_distance(const GJKSimplex& simplex) {
    switch (simplex.n) {
    case 1: return v3len(simplex.pts[0].diff);
    case 2: {
        Vec3 A = simplex.pts[1].diff, B = simplex.pts[0].diff;
        Vec3 AB = v3sub(B, A);
        float t = -v3dot(A, AB) / std::max(v3dot(AB, AB), 1e-12f);
        t = std::max(0.0f, std::min(1.0f, t));
        Vec3 closest = v3add(A, v3scale(AB, t));
        return v3len(closest);
    }
    case 3: {
        // Distance from origin to triangle
        Vec3 A = simplex.pts[2].diff, B = simplex.pts[1].diff, C = simplex.pts[0].diff;
        Vec3 AB = v3sub(B, A), AC = v3sub(C, A);
        Vec3 n = v3cross(AB, AC);
        float nl = v3len(n);
        if (nl < 1e-12f) return v3len(A);
        return std::abs(v3dot(A, n)) / nl;
    }
    default: return 0.0f;
    }
}

// Witness points for non-overlapping case
static void gjk_witness(const GJKSimplex& simplex, Vec3& pa, Vec3& pb) {
    switch (simplex.n) {
    case 1:
        pa = simplex.pts[0].a;
        pb = simplex.pts[0].b;
        break;
    case 2: {
        Vec3 A = simplex.pts[1].diff, B = simplex.pts[0].diff;
        Vec3 AB = v3sub(B, A);
        float t = -v3dot(A, AB) / std::max(v3dot(AB, AB), 1e-12f);
        t = std::max(0.0f, std::min(1.0f, t));
        pa = v3add(simplex.pts[1].a, v3scale(v3sub(simplex.pts[0].a, simplex.pts[1].a), t));
        pb = v3add(simplex.pts[1].b, v3scale(v3sub(simplex.pts[0].b, simplex.pts[1].b), t));
        break;
    }
    case 3: {
        Vec3 A = simplex.pts[2].diff, B = simplex.pts[1].diff, C = simplex.pts[0].diff;
        Vec3 v0 = v3sub(B, A), v1 = v3sub(C, A), v2 = v3neg(A);
        float d00 = v3dot(v0, v0), d01 = v3dot(v0, v1), d11 = v3dot(v1, v1);
        float d20 = v3dot(v2, v0), d21 = v3dot(v2, v1);
        float denom = d00*d11 - d01*d01;
        float u = (denom > 1e-12f) ? (d11*d20 - d01*d21) / denom : 1.0f/3;
        float v = (denom > 1e-12f) ? (d00*d21 - d01*d20) / denom : 1.0f/3;
        float w = 1.0f - u - v;
        u = std::max(0.0f, u); v = std::max(0.0f, v); w = std::max(0.0f, w);
        float sum = u + v + w;
        if (sum > 1e-8f) { u /= sum; v /= sum; w /= sum; }
        pa = v3add(v3add(v3scale(simplex.pts[2].a, w), v3scale(simplex.pts[1].a, u)),
                    v3scale(simplex.pts[0].a, v));
        pb = v3add(v3add(v3scale(simplex.pts[2].b, w), v3scale(simplex.pts[1].b, u)),
                    v3scale(simplex.pts[0].b, v));
        break;
    }
    default:
        pa = {0,0,0}; pb = {0,0,0};
    }
}

// Build ConvexGeom from collision data
static ConvexGeom make_convex_geom(int type, const float* pos, const float* mat,
                                    const float* size, const Model& model, int dataid) {
    ConvexGeom g;
    g.type = type;
    for (int i = 0; i < 3; i++) g.pos[i] = pos[i];
    for (int i = 0; i < 9; i++) g.mat[i] = mat[i];
    for (int i = 0; i < 3; i++) g.size[i] = size[i];
    g.verts = nullptr;
    g.nverts = 0;

    if (type == (int)GeomType::MESH && dataid >= 0 && model.mesh_vert.size() > 0) {
        mx::eval(model.mesh_vertadr);
        mx::eval(model.mesh_vertnum);
        mx::eval(model.mesh_vert);
        auto vadr = model.mesh_vertadr.data<int>();
        auto vnum = model.mesh_vertnum.data<int>();
        g.verts = model.mesh_vert.data<float>() + vadr[dataid] * 3;
        g.nverts = vnum[dataid];
    }

    return g;
}

// Convex collision via GJK/EPA: handles any pair involving MESH geoms
static CollisionResult convex_collision(
    int type1, const mx::array& pos1, const mx::array& mat1, const mx::array& size1, int dataid1,
    int type2, const mx::array& pos2, const mx::array& mat2, const mx::array& size2, int dataid2,
    const Model& model)
{
    mx::eval(pos1); mx::eval(mat1); mx::eval(size1);
    mx::eval(pos2); mx::eval(mat2); mx::eval(size2);

    ConvexGeom A = make_convex_geom(type1, pos1.data<float>(), mat1.data<float>(),
                                     size1.data<float>(), model, dataid1);
    ConvexGeom B = make_convex_geom(type2, pos2.data<float>(), mat2.data<float>(),
                                     size2.data<float>(), model, dataid2);

    GJKSimplex simplex;
    bool overlap = gjk(A, B, simplex);

    if (overlap) {
        EPAResult r = epa(A, B, simplex);
        float dist = -r.depth;
        Vec3 contact_pos = v3scale(v3add(r.point_a, r.point_b), 0.5f);
        Vec3 normal = r.normal;
        return {mx::array(dist),
                mx::array({contact_pos.x, contact_pos.y, contact_pos.z}),
                make_frame(mx::array({normal.x, normal.y, normal.z}))};
    } else {
        float dist = gjk_distance(simplex);
        Vec3 pa, pb;
        gjk_witness(simplex, pa, pb);
        Vec3 sep = v3sub(pa, pb);
        Vec3 normal = v3norm(sep);
        Vec3 contact_pos = v3scale(v3add(pa, pb), 0.5f);
        return {mx::array(dist),
                mx::array({contact_pos.x, contact_pos.y, contact_pos.z}),
                make_frame(mx::array({normal.x, normal.y, normal.z}))};
    }
}

// plane_mesh_multi: returns contacts for all mesh vertices penetrating the plane.
// Matches MuJoCo C's convex-plane contact generation.
static int plane_mesh_multi(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& mesh_pos, const mx::array& mesh_mat, const mx::array& mesh_size,
    int dataid, const Model& model, float margin, int g1, int g2, int condim,
    std::vector<mx::array>& c_dist, std::vector<mx::array>& c_pos,
    std::vector<mx::array>& c_frame, std::vector<mx::array>& c_geom,
    std::vector<int>& c_dim)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);
    auto frame = make_frame(normal);
    mx::eval(normal); mx::eval(plane_pos);
    mx::eval(mesh_pos); mx::eval(mesh_mat);

    if (dataid < 0 || model.mesh_vert.size() == 0) return 0;

    mx::eval(model.mesh_vertadr); mx::eval(model.mesh_vertnum); mx::eval(model.mesh_vert);
    auto vadr = model.mesh_vertadr.data<int>();
    auto vnum = model.mesh_vertnum.data<int>();
    const float* verts = model.mesh_vert.data<float>() + vadr[dataid] * 3;
    int nv = vnum[dataid];

    auto np = normal.data<float>();
    auto pp = plane_pos.data<float>();
    auto mp = mesh_pos.data<float>();
    auto mm = mesh_mat.data<float>();

    int ncon_added = 0;
    for (int i = 0; i < nv; i++) {
        float lx = verts[i*3], ly = verts[i*3+1], lz = verts[i*3+2];
        float wx = mp[0] + mm[0]*lx + mm[1]*ly + mm[2]*lz;
        float wy = mp[1] + mm[3]*lx + mm[4]*ly + mm[5]*lz;
        float wz = mp[2] + mm[6]*lx + mm[7]*ly + mm[8]*lz;
        float d = np[0]*(wx-pp[0]) + np[1]*(wy-pp[1]) + np[2]*(wz-pp[2]);
        if (d < margin) {
            c_dist.push_back(mx::array(d));
            auto vertex = mx::array({wx, wy, wz});
            c_pos.push_back(mx::subtract(vertex, mx::multiply(normal, mx::array(d))));
            c_frame.push_back(frame);
            c_geom.push_back(mx::array({g1, g2}, mx::int32));
            c_dim.push_back(condim);
            ncon_added++;
        }
    }
    return ncon_added;
}

// Single-contact plane_mesh (for vmap fallback)
static CollisionResult plane_mesh_single(
    const mx::array& plane_pos, const mx::array& plane_mat,
    const mx::array& mesh_pos, const mx::array& mesh_mat, const mx::array& mesh_size,
    int dataid, const Model& model)
{
    auto normal = mat_col(mx::reshape(plane_mat, {1, 3, 3}), 0, 2);
    mx::eval(normal); mx::eval(plane_pos);
    mx::eval(mesh_pos); mx::eval(mesh_mat);

    if (dataid < 0 || model.mesh_vert.size() == 0)
        return {mx::array(1.0f), mx::zeros({3}), mx::eye(3)};

    mx::eval(model.mesh_vertadr); mx::eval(model.mesh_vertnum); mx::eval(model.mesh_vert);
    auto vadr = model.mesh_vertadr.data<int>();
    auto vnum = model.mesh_vertnum.data<int>();
    const float* verts = model.mesh_vert.data<float>() + vadr[dataid] * 3;
    int nv = vnum[dataid];

    auto np = normal.data<float>();
    auto pp = plane_pos.data<float>();
    auto mp = mesh_pos.data<float>();
    auto mm = mesh_mat.data<float>();

    float best_dist = 1e10f;
    float best_wx = 0, best_wy = 0, best_wz = 0;

    for (int i = 0; i < nv; i++) {
        float lx = verts[i*3], ly = verts[i*3+1], lz = verts[i*3+2];
        float wx = mp[0] + mm[0]*lx + mm[1]*ly + mm[2]*lz;
        float wy = mp[1] + mm[3]*lx + mm[4]*ly + mm[5]*lz;
        float wz = mp[2] + mm[6]*lx + mm[7]*ly + mm[8]*lz;
        float d = np[0]*(wx-pp[0]) + np[1]*(wy-pp[1]) + np[2]*(wz-pp[2]);
        if (d < best_dist) {
            best_dist = d;
            best_wx = wx; best_wy = wy; best_wz = wz;
        }
    }

    auto dist = mx::array(best_dist);
    auto vertex = mx::array({best_wx, best_wy, best_wz});
    auto pos = mx::subtract(vertex, mx::multiply(normal, dist));
    return {dist, pos, make_frame(normal)};
}

// ── HFIELD collision via GJK with triangular prisms ──────────────────────────
// For each grid cell overlapping the geom's bounding sphere, build 2 triangular
// prisms (6 vertices each) and test each prism against the geom using GJK/EPA.

// Get geom bounding radius (conservative)
static float geom_rbound(int type, const float* size) {
    switch (type) {
    case (int)GeomType::SPHERE:   return size[0];
    case (int)GeomType::CAPSULE:  return size[0] + size[1];
    case (int)GeomType::BOX:      return std::sqrt(size[0]*size[0] + size[1]*size[1] + size[2]*size[2]);
    case (int)GeomType::CYLINDER: return std::sqrt(size[0]*size[0] + size[1]*size[1]);
    case (int)GeomType::MESH:     return std::max({size[0], size[1], size[2]});
    default:                      return 0.1f;
    }
}

static int hfield_collision(
    const mx::array& hf_pos, const mx::array& hf_mat,
    int hf_dataid, const Model& model,
    int geom_type, const mx::array& geom_pos, const mx::array& geom_mat,
    const mx::array& geom_size, int geom_dataid,
    float margin, int g_hf, int g_other, int condim,
    std::vector<mx::array>& c_dist, std::vector<mx::array>& c_pos,
    std::vector<mx::array>& c_frame, std::vector<mx::array>& c_geom,
    std::vector<int>& c_dim)
{
    if (hf_dataid < 0 || model.hfield_data.size() == 0)
        return 0;

    mx::eval(hf_pos); mx::eval(hf_mat);
    mx::eval(geom_pos); mx::eval(geom_mat); mx::eval(geom_size);
    mx::eval(model.hfield_nrow); mx::eval(model.hfield_ncol);
    mx::eval(model.hfield_size); mx::eval(model.hfield_adr); mx::eval(model.hfield_data);

    auto hp = hf_pos.data<float>();
    auto hm = hf_mat.data<float>();
    int nrow = model.hfield_nrow.data<int>()[hf_dataid];
    int ncol = model.hfield_ncol.data<int>()[hf_dataid];
    auto hsz = model.hfield_size.data<float>() + hf_dataid * 4;
    float sx = hsz[0], sy = hsz[1], sz_top = hsz[2], sz_bot = hsz[3];
    int hadr = model.hfield_adr.data<int>()[hf_dataid];
    auto hdata = model.hfield_data.data<float>() + hadr;

    auto gp = geom_pos.data<float>();
    auto gm = geom_mat.data<float>();
    auto gs = geom_size.data<float>();

    // Transform geom center to hfield local frame
    float dx = gp[0] - hp[0], dy = gp[1] - hp[1], dz = gp[2] - hp[2];
    // hm is column-major rotation: local = R^T * world_offset
    float lx = hm[0]*dx + hm[3]*dy + hm[6]*dz;
    float ly = hm[1]*dx + hm[4]*dy + hm[7]*dz;
    float lz = hm[2]*dx + hm[5]*dy + hm[8]*dz;

    float rb = geom_rbound(geom_type, gs) + margin;

    // Grid cell spacing
    float cell_dx = (ncol > 1) ? 2.0f * sx / (ncol - 1) : 2.0f * sx;
    float cell_dy = (nrow > 1) ? 2.0f * sy / (nrow - 1) : 2.0f * sy;

    // Sub-grid: columns and rows that overlap geom bounding sphere
    int cmin = std::max(0, (int)std::floor((lx - rb + sx) / cell_dx));
    int cmax = std::min(ncol - 2, (int)std::ceil((lx + rb + sx) / cell_dx));
    int rmin = std::max(0, (int)std::floor((ly - rb + sy) / cell_dy));
    int rmax = std::min(nrow - 2, (int)std::ceil((ly + rb + sy) / cell_dy));

    if (cmin > cmax || rmin > rmax) return 0;

    // Build the "other" ConvexGeom
    ConvexGeom G = make_convex_geom(geom_type, gp, gm, gs, model, geom_dataid);

    int ncon = 0;
    const int MAX_HF_CONTACTS = 50; // MuJoCo C limit per pair

    for (int r = rmin; r <= rmax && ncon < MAX_HF_CONTACTS; r++) {
        for (int c = cmin; c <= cmax && ncon < MAX_HF_CONTACTS; c++) {
            for (int tri = 0; tri < 2 && ncon < MAX_HF_CONTACTS; tri++) {
                int r0, c0, r1, c1, r2, c2;
                if (tri == 0) {
                    r0 = r; c0 = c; r1 = r; c1 = c+1; r2 = r+1; c2 = c;
                } else {
                    r0 = r+1; c0 = c+1; r1 = r+1; c1 = c; r2 = r; c2 = c+1;
                }

                float prism_verts[18];
                for (int k = 0; k < 3; k++) {
                    int rk, ck;
                    if (k == 0) { rk = r0; ck = c0; }
                    else if (k == 1) { rk = r1; ck = c1; }
                    else { rk = r2; ck = c2; }

                    float vx = -sx + ck * cell_dx;
                    float vy = -sy + rk * cell_dy;
                    float vz_top = hdata[rk * ncol + ck] * sz_top;
                    float vz_bot = -sz_bot;

                    prism_verts[k*3 + 0] = vx;
                    prism_verts[k*3 + 1] = vy;
                    prism_verts[k*3 + 2] = vz_top;
                    prism_verts[(k+3)*3 + 0] = vx;
                    prism_verts[(k+3)*3 + 1] = vy;
                    prism_verts[(k+3)*3 + 2] = vz_bot;
                }

                ConvexGeom P;
                P.type = (int)GeomType::MESH;
                for (int i = 0; i < 3; i++) P.pos[i] = hp[i];
                for (int i = 0; i < 9; i++) P.mat[i] = hm[i];
                for (int i = 0; i < 3; i++) P.size[i] = 0;
                P.verts = prism_verts;
                P.nverts = 6;

                GJKSimplex simplex;
                bool overlap = gjk(P, G, simplex);

                if (overlap) {
                    EPAResult er = epa(P, G, simplex);
                    if (er.depth > 0) {
                        Vec3 cpos = v3scale(v3add(er.point_a, er.point_b), 0.5f);
                        c_dist.push_back(mx::array(-er.depth));
                        c_pos.push_back(mx::array({cpos.x, cpos.y, cpos.z}));
                        c_frame.push_back(make_frame(mx::array({er.normal.x, er.normal.y, er.normal.z})));
                        c_geom.push_back(mx::array({g_hf, g_other}, mx::int32));
                        c_dim.push_back(condim);
                        ncon++;
                    }
                }
            }
        }
    }
    return ncon;
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
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::CYLINDER)) {
                float margin = gmargin[g1_] + gmargin[g2_];
                int condim = 3;
                if (m.geom_condim.size() > 0) {
                    mx::eval(m.geom_condim);
                    auto cdp = m.geom_condim.data<int>();
                    condim = std::max(cdp[g1_], cdp[g2_]);
                }
                plane_cylinder_multi(gpos1, gmat1, gpos2, gmat2, gsize2,
                    margin, g1_, g2_, condim,
                    c_dist, c_pos, c_frame, c_geom, c_dim);
                continue; // already pushed contacts
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::CYLINDER)) {
                result = sphere_cylinder(gpos1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::CAPSULE) && t2_ == static_cast<int>(GeomType::CYLINDER)) {
                result = capsule_cylinder(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t2_ == static_cast<int>(GeomType::HFIELD) || t1_ == static_cast<int>(GeomType::HFIELD)) {
                // HFIELD collision via GJK with triangular prisms
                int hf_g, other_g;
                if (t1_ == static_cast<int>(GeomType::HFIELD)) { hf_g = g1_; other_g = g2_; }
                else { hf_g = g2_; other_g = g1_; }
                int hf_dataid = -1, other_dataid = -1;
                if (m.geom_dataid.size() > 0) {
                    mx::eval(m.geom_dataid);
                    auto gdid = m.geom_dataid.data<int>();
                    hf_dataid = gdid[hf_g];
                    other_dataid = gdid[other_g];
                }
                float hf_margin = gmargin[hf_g] + gmargin[other_g];
                int hf_condim = 3;
                if (m.geom_condim.size() > 0) {
                    mx::eval(m.geom_condim);
                    auto cdp = m.geom_condim.data<int>();
                    hf_condim = std::max(cdp[hf_g], cdp[other_g]);
                }
                auto hf_pos = row(d.geom_xpos, hf_g);
                auto hf_mat = mx::flatten(mx::slice(d.geom_xmat, {hf_g,0,0}, {hf_g+1,3,3}));
                auto oth_pos = row(d.geom_xpos, other_g);
                auto oth_mat = mx::flatten(mx::slice(d.geom_xmat, {other_g,0,0}, {other_g+1,3,3}));
                auto oth_size = row(m.geom_size, other_g);
                int oth_type = gtype[other_g];
                hfield_collision(hf_pos, hf_mat, hf_dataid, m,
                    oth_type, oth_pos, oth_mat, oth_size, other_dataid,
                    hf_margin, hf_g, other_g, hf_condim,
                    c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::MESH)) {
                float margin = gmargin[g1_] + gmargin[g2_];
                int condim = 3;
                if (m.geom_condim.size() > 0) {
                    mx::eval(m.geom_condim);
                    auto cdp = m.geom_condim.data<int>();
                    condim = std::max(cdp[g1_], cdp[g2_]);
                }
                int dataid2 = -1;
                if (m.geom_dataid.size() > 0) {
                    mx::eval(m.geom_dataid);
                    dataid2 = m.geom_dataid.data<int>()[g2_];
                }
                plane_mesh_multi(gpos1, gmat1, gpos2, gmat2, gsize2,
                    dataid2, m, margin, g1_, g2_, condim,
                    c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t2_ == static_cast<int>(GeomType::MESH) || t1_ == static_cast<int>(GeomType::MESH)) {
                // GJK/EPA for any pair involving MESH
                int dataid1 = -1, dataid2 = -1;
                if (m.geom_dataid.size() > 0) {
                    mx::eval(m.geom_dataid);
                    auto gdid = m.geom_dataid.data<int>();
                    dataid1 = gdid[g1_];
                    dataid2 = gdid[g2_];
                }
                result = convex_collision(t1_, gpos1, gmat1, gsize1, dataid1,
                                          t2_, gpos2, gmat2, gsize2, dataid2, m);
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
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::CYLINDER)) {
                float margin_pc = gmargin[g1_] + gmargin[g2_];
                if (m.pair_margin.size() > 0) {
                    mx::eval(m.pair_margin);
                    margin_pc = m.pair_margin.data<float>()[pi];
                }
                int condim_pc = 3;
                if (m.pair_dim.size() > 0) {
                    mx::eval(m.pair_dim);
                    condim_pc = m.pair_dim.data<int>()[pi];
                }
                plane_cylinder_multi(gpos1, gmat1, gpos2, gmat2, gsize2,
                    margin_pc, g1_, g2_, condim_pc,
                    c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t1_ == static_cast<int>(GeomType::SPHERE) && t2_ == static_cast<int>(GeomType::CYLINDER)) {
                result = sphere_cylinder(gpos1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t1_ == static_cast<int>(GeomType::CAPSULE) && t2_ == static_cast<int>(GeomType::CYLINDER)) {
                result = capsule_cylinder(gpos1, gmat1, gsize1, gpos2, gmat2, gsize2);
                handled = true;
            } else if (t2_ == static_cast<int>(GeomType::HFIELD) || t1_ == static_cast<int>(GeomType::HFIELD)) {
                int hf_g, other_g;
                if (t1_ == static_cast<int>(GeomType::HFIELD)) { hf_g = g1_; other_g = g2_; }
                else { hf_g = g2_; other_g = g1_; }
                int hf_dataid = -1, other_dataid = -1;
                if (m.geom_dataid.size() > 0) {
                    mx::eval(m.geom_dataid);
                    auto gdid = m.geom_dataid.data<int>();
                    hf_dataid = gdid[hf_g];
                    other_dataid = gdid[other_g];
                }
                float hf_margin = gmargin[hf_g] + gmargin[other_g];
                if (m.pair_margin.size() > 0) {
                    mx::eval(m.pair_margin);
                    hf_margin = m.pair_margin.data<float>()[pi];
                }
                int hf_condim = 3;
                if (m.pair_dim.size() > 0) {
                    mx::eval(m.pair_dim);
                    hf_condim = m.pair_dim.data<int>()[pi];
                }
                auto hf_pos = row(d.geom_xpos, hf_g);
                auto hf_mat_ = mx::flatten(mx::slice(d.geom_xmat, {hf_g,0,0}, {hf_g+1,3,3}));
                auto oth_pos = row(d.geom_xpos, other_g);
                auto oth_mat_ = mx::flatten(mx::slice(d.geom_xmat, {other_g,0,0}, {other_g+1,3,3}));
                auto oth_size = row(m.geom_size, other_g);
                int oth_type = gtype[other_g];
                hfield_collision(hf_pos, hf_mat_, hf_dataid, m,
                    oth_type, oth_pos, oth_mat_, oth_size, other_dataid,
                    hf_margin, hf_g, other_g, hf_condim,
                    c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t1_ == static_cast<int>(GeomType::PLANE) && t2_ == static_cast<int>(GeomType::MESH)) {
                float margin_pm = gmargin[g1_] + gmargin[g2_];
                if (m.pair_margin.size() > 0) {
                    mx::eval(m.pair_margin);
                    margin_pm = m.pair_margin.data<float>()[pi];
                }
                int condim_pm = 3;
                if (m.pair_dim.size() > 0) {
                    mx::eval(m.pair_dim);
                    condim_pm = m.pair_dim.data<int>()[pi];
                }
                int dataid2 = -1;
                if (m.geom_dataid.size() > 0) {
                    mx::eval(m.geom_dataid);
                    dataid2 = m.geom_dataid.data<int>()[g2_];
                }
                plane_mesh_multi(gpos1, gmat1, gpos2, gmat2, gsize2,
                    dataid2, m, margin_pm, g1_, g2_, condim_pm,
                    c_dist, c_pos, c_frame, c_geom, c_dim);
                continue;
            } else if (t2_ == static_cast<int>(GeomType::MESH) || t1_ == static_cast<int>(GeomType::MESH)) {
                int dataid1 = -1, dataid2 = -1;
                if (m.geom_dataid.size() > 0) {
                    mx::eval(m.geom_dataid);
                    auto gdid = m.geom_dataid.data<int>();
                    dataid1 = gdid[g1_];
                    dataid2 = gdid[g2_];
                }
                result = convex_collision(t1_, gpos1, gmat1, gsize1, dataid1,
                                          t2_, gpos2, gmat2, gsize2, dataid2, m);
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
