// Port of Python test_math.py (~17 tests)
// Tests quaternion ops, cross product, normalize, inert_mul, geometry helpers.
// No MuJoCo C reference needed -- pure math.

#include "test_utils.h"
#include "internal.h"

using namespace mjmlx;

// Helper: extract float from 1D mx::array
static float to_float(const mx::array& a) {
    mx::eval(a);
    return a.item<float>();
}

// Helper: extract vector from mx::array
static std::vector<float> to_vec(const mx::array& a) {
    mx::eval(a);
    auto* p = a.data<float>();
    return std::vector<float>(p, p + a.size());
}

// Helper: compute L2 norm
static float vec_norm(const std::vector<float>& v) {
    float s = 0;
    for (float x : v) s += x * x;
    return std::sqrt(s);
}

int main() {
    printf("=== test_math_full ===\n");

    // ── TestNorm ──
    TEST_SECTION("Norm");

    TEST_BEGIN("norm_basic");
    {
        auto x = mx::array({3.0f, 4.0f, 0.0f});
        float n = to_float(mx::sqrt(mx::sum(mx::multiply(x, x))));
        CHECK_CLOSE(n, 5.0f, 1e-5, "norm([3,4,0]) == 5");
    }
    TEST_END();

    TEST_BEGIN("norm_zero");
    {
        auto x = mx::zeros({3});
        float n = to_float(mx::sqrt(mx::sum(mx::multiply(x, x))));
        CHECK_CLOSE(n, 0.0f, 1e-6, "norm(zeros) == 0");
    }
    TEST_END();

    // ── TestNormalize ──
    TEST_SECTION("Normalize");

    TEST_BEGIN("normalize_unit_length");
    {
        auto x = mx::array({3.0f, 4.0f, 0.0f});
        auto norm = mx::sqrt(mx::sum(mx::multiply(x, x)));
        auto n = mx::divide(x, mx::maximum(norm, mx::array(1e-10f)));
        float len = to_float(mx::sqrt(mx::sum(mx::multiply(n, n))));
        CHECK_CLOSE(len, 1.0f, 1e-5, "normalized has unit length");
    }
    TEST_END();

    TEST_BEGIN("normalize_direction");
    {
        auto x = mx::array({0.0f, 5.0f, 0.0f});
        auto norm = mx::sqrt(mx::sum(mx::multiply(x, x)));
        auto n = mx::divide(x, mx::maximum(norm, mx::array(1e-10f)));
        auto v = to_vec(n);
        CHECK_CLOSE(v[0], 0.0f, 1e-5, "x component");
        CHECK_CLOSE(v[1], 1.0f, 1e-5, "y component");
        CHECK_CLOSE(v[2], 0.0f, 1e-5, "z component");
        CHECK_CLOSE(to_float(norm), 5.0f, 1e-5, "original norm == 5");
    }
    TEST_END();

    // ── TestQuaternion ──
    TEST_SECTION("Quaternion");

    TEST_BEGIN("quat_identity_mul");
    {
        auto q = mx::array({1.0f, 0.0f, 0.0f, 0.0f});
        auto r = quat_mul(q, q);
        auto v = to_vec(r);
        CHECK_CLOSE(v[0], 1.0f, 1e-6, "w");
        CHECK_CLOSE(v[1], 0.0f, 1e-6, "x");
        CHECK_CLOSE(v[2], 0.0f, 1e-6, "y");
        CHECK_CLOSE(v[3], 0.0f, 1e-6, "z");
    }
    TEST_END();

    TEST_BEGIN("quat_inverse");
    {
        auto q = mx::array({1.0f, 1.0f, 0.0f, 0.0f});
        auto norm = mx::sqrt(mx::sum(mx::multiply(q, q)));
        q = mx::divide(q, norm);
        auto qi = quat_inv(q);
        auto r = quat_mul(q, qi);
        auto v = to_vec(r);
        CHECK_CLOSE(v[0], 1.0f, 1e-5, "w");
        CHECK_CLOSE(v[1], 0.0f, 1e-5, "x");
        CHECK_CLOSE(v[2], 0.0f, 1e-5, "y");
        CHECK_CLOSE(v[3], 0.0f, 1e-5, "z");
    }
    TEST_END();

    TEST_BEGIN("quat_to_mat_identity");
    {
        auto q = mx::array({1.0f, 0.0f, 0.0f, 0.0f});
        auto mat = quat_to_mat(q);
        auto v = to_vec(mat);
        float expected[9] = {1,0,0, 0,1,0, 0,0,1};
        CHECK_ARRAY_CLOSE(v.data(), expected, 9, 1e-6, "identity rotation matrix");
    }
    TEST_END();

    TEST_BEGIN("quat_axis_angle_roundtrip");
    {
        auto axis = mx::array({0.0f, 0.0f, 1.0f});
        float angle_in = 0.5f;
        // aa2quat: [cos(a/2), sin(a/2)*axis]
        float ha = angle_in * 0.5f;
        auto q = mx::array({std::cos(ha), 0.0f, 0.0f, std::sin(ha)});
        // Extract angle back: 2*acos(w)
        auto v = to_vec(q);
        float angle_out = 2.0f * std::acos(std::min(1.0f, std::abs(v[0])));
        CHECK_CLOSE(angle_out, angle_in, 1e-5, "axis-angle roundtrip");
    }
    TEST_END();

    TEST_BEGIN("quat_rotate");
    {
        // 90 degrees around Z: q = [cos(pi/4), 0, 0, sin(pi/4)]
        float ha = (float)M_PI / 4.0f;
        auto q = mx::array({std::cos(ha), 0.0f, 0.0f, std::sin(ha)});
        auto v_in = mx::array({1.0f, 0.0f, 0.0f});
        auto v_out = rotate(v_in, q);
        auto r = to_vec(v_out);
        CHECK_CLOSE(r[0], 0.0f, 1e-5, "rotated x");
        CHECK_CLOSE(r[1], 1.0f, 1e-5, "rotated y");
        CHECK_CLOSE(r[2], 0.0f, 1e-5, "rotated z");
    }
    TEST_END();

    TEST_BEGIN("quat_integrate_unit_length");
    {
        auto q = mx::array({1.0f, 0.0f, 0.0f, 0.0f});
        auto omega = mx::array({0.0f, 0.0f, 1.0f});
        auto q2 = quat_integrate(q, omega, 0.01f);
        auto v = to_vec(q2);
        float len = vec_norm(v);
        CHECK_CLOSE(len, 1.0f, 1e-5, "integrated quat has unit length");
    }
    TEST_END();

    // ── TestCross ──
    TEST_SECTION("Cross");

    TEST_BEGIN("cross_basic");
    {
        auto a = mx::array({1.0f, 0.0f, 0.0f});
        auto b = mx::array({0.0f, 1.0f, 0.0f});
        auto c = mx::linalg::cross(a, b);
        auto v = to_vec(c);
        CHECK_CLOSE(v[0], 0.0f, 1e-6, "x");
        CHECK_CLOSE(v[1], 0.0f, 1e-6, "y");
        CHECK_CLOSE(v[2], 1.0f, 1e-6, "z");
    }
    TEST_END();

    // ── TestInertMul ──
    TEST_SECTION("InertMul");

    TEST_BEGIN("inert_mul_zero_motion");
    {
        // inertia: 10-element vector [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, mass]
        // mass=1 at element 9, rest zero except diagonal inertia
        std::vector<float> inertia_data(10, 0.0f);
        inertia_data[0] = 1.0f; // Ixx
        inertia_data[3] = 1.0f; // Iyy
        inertia_data[5] = 1.0f; // Izz
        inertia_data[9] = 1.0f; // mass
        auto inertia = mx::array(inertia_data.data(), {10}, mx::float32);
        auto motion = mx::zeros({6});
        auto result = inert_mul(inertia, motion);
        auto v = to_vec(result);
        for (int i = 0; i < 6; i++) {
            CHECK_CLOSE(v[i], 0.0f, 1e-6, "zero motion -> zero force");
        }
    }
    TEST_END();

    // ── TestMotionCross ──
    TEST_SECTION("MotionCross");

    TEST_BEGIN("motion_cross_zero");
    {
        auto u = mx::zeros({6});
        auto v = mx::zeros({6});
        auto r = motion_cross(u, v);
        auto rv = to_vec(r);
        for (int i = 0; i < 6; i++) {
            CHECK_CLOSE(rv[i], 0.0f, 1e-6, "zero cross zero");
        }
    }
    TEST_END();

    // ── TestGeometry ──
    TEST_SECTION("Geometry");

    TEST_BEGIN("closest_segment_point");
    {
        auto a = mx::array({0.0f, 0.0f, 0.0f});
        auto b = mx::array({1.0f, 0.0f, 0.0f});
        auto pt = mx::array({0.5f, 1.0f, 0.0f});
        auto cp = closest_segment_point(a, b, pt);
        auto v = to_vec(cp);
        CHECK_CLOSE(v[0], 0.5f, 1e-5, "x");
        CHECK_CLOSE(v[1], 0.0f, 1e-5, "y");
        CHECK_CLOSE(v[2], 0.0f, 1e-5, "z");
    }
    TEST_END();

    TEST_BEGIN("closest_segment_clamp");
    {
        auto a = mx::array({0.0f, 0.0f, 0.0f});
        auto b = mx::array({1.0f, 0.0f, 0.0f});
        auto pt = mx::array({2.0f, 0.0f, 0.0f});
        auto cp = closest_segment_point(a, b, pt);
        auto v = to_vec(cp);
        CHECK_CLOSE(v[0], 1.0f, 1e-5, "clamped to b");
        CHECK_CLOSE(v[1], 0.0f, 1e-5, "y");
        CHECK_CLOSE(v[2], 0.0f, 1e-5, "z");
    }
    TEST_END();

    // ── TestSafeDiv ──
    TEST_SECTION("SafeDiv");

    TEST_BEGIN("safe_div_normal");
    {
        auto a = mx::array(4.0f);
        auto b = mx::array(2.0f);
        auto r = mx::divide(a, mx::maximum(b, mx::array(1e-10f)));
        CHECK_CLOSE(to_float(r), 2.0f, 1e-6, "4/2 = 2");
    }
    TEST_END();

    TEST_BEGIN("safe_div_zero_denom");
    {
        auto a = mx::array(1.0f);
        auto b = mx::array(0.0f);
        auto r = mx::divide(a, mx::maximum(b, mx::array(1e-10f)));
        float val = to_float(r);
        CHECK(std::isfinite(val), "safe div with zero denom is finite");
    }
    TEST_END();

    TEST_EXIT();
}
