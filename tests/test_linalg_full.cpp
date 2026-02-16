// GPU-native Cholesky and solve tests (~12 tests)
// Tests cholesky_gpu and cholesky_solve_gpu from smooth_vmap.cpp.
// Validates: identity, diagonal, small SPD, humanoid mass matrix, solve roundtrips.

#include "test_utils.h"
#include "internal.h"

using namespace mjmlx;

static float max_abs_diff(const mx::array& a, const mx::array& b) {
    auto d = mx::max(mx::abs(mx::subtract(a, b)));
    mx::eval(d);
    return d.item<float>();
}

static float max_abs(const mx::array& a) {
    auto m = mx::max(mx::abs(a));
    mx::eval(m);
    return m.item<float>();
}

// Build a random SPD matrix: A = R^T R + alpha*I
static mx::array random_spd(int n, float alpha = 1.0f) {
    auto R = mx::random::normal({n, n});
    auto A = mx::add(mx::matmul(mx::transpose(R), R),
                     mx::multiply(mx::eye(n), mx::array(alpha)));
    mx::eval(A);
    return A;
}

int main(int argc, char** argv) {
    printf("=== test_linalg_full ===\n");

    const char* humanoid_xml_path = nullptr;
    if (argc > 1) humanoid_xml_path = argv[1];

    // ── Cholesky factorization tests ─────────────────────────────

    TEST_SECTION("Cholesky");

    TEST_BEGIN("cholesky_identity_2x2");
    {
        auto I = mx::eye(2);
        auto L = cholesky_gpu(I, 2);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), I);
        CHECK_LT(err, 1e-6f, "L@L^T = I for identity");
        // L should be identity for identity input
        float diag_err = max_abs_diff(L, mx::eye(2));
        CHECK_LT(diag_err, 1e-6f, "L = I for identity input");
    }
    TEST_END();

    TEST_BEGIN("cholesky_identity_5x5");
    {
        auto I = mx::eye(5);
        auto L = cholesky_gpu(I, 5);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), I);
        CHECK_LT(err, 1e-5f, "L@L^T = I for 5x5 identity");
    }
    TEST_END();

    TEST_BEGIN("cholesky_diagonal");
    {
        // diag([4, 9, 16]) -> L = diag([2, 3, 4])
        std::vector<float> dvals = {4.0f, 9.0f, 16.0f};
        auto D = mx::diag(mx::array(dvals.data(), {3}, mx::float32));
        auto L = cholesky_gpu(D, 3);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), D);
        CHECK_LT(err, 1e-5f, "L@L^T = D for diagonal");
        // Check L is lower triangular
        auto upper = mx::triu(L, 1);
        mx::eval(upper);
        CHECK_LT(max_abs(upper), 1e-6f, "L is lower triangular");
    }
    TEST_END();

    TEST_BEGIN("cholesky_2x2_known");
    {
        // A = [[4, 2], [2, 5]]  ->  L = [[2, 0], [1, 2]]
        std::vector<float> adata = {4.0f, 2.0f, 2.0f, 5.0f};
        auto A = mx::array(adata.data(), {2, 2}, mx::float32);
        auto L = cholesky_gpu(A, 2);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), A);
        CHECK_LT(err, 1e-5f, "L@L^T = A for 2x2 known SPD");
        // Verify specific L values
        auto L_flat = mx::flatten(L);
        mx::eval(L_flat);
        auto lp = L_flat.data<float>();
        CHECK_CLOSE(lp[0], 2.0f, 1e-5f, "L[0,0] = 2");
        CHECK_CLOSE(lp[1], 0.0f, 1e-5f, "L[0,1] = 0");
        CHECK_CLOSE(lp[2], 1.0f, 1e-5f, "L[1,0] = 1");
        CHECK_CLOSE(lp[3], 2.0f, 1e-5f, "L[1,1] = 2");
    }
    TEST_END();

    TEST_BEGIN("cholesky_6x6_random");
    {
        auto A = random_spd(6, 2.0f);
        auto L = cholesky_gpu(A, 6);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), A);
        CHECK_LT(err, 1e-3f, "L@L^T ≈ A for random 6x6 SPD");
        // Check lower triangular
        auto upper = mx::triu(L, 1);
        mx::eval(upper);
        CHECK_LT(max_abs(upper), 1e-6f, "L is lower triangular");
    }
    TEST_END();

    TEST_BEGIN("cholesky_15x15_random");
    {
        auto A = random_spd(15, 3.0f);
        auto L = cholesky_gpu(A, 15);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), A);
        CHECK_LT(err, 1e-2f, "L@L^T ≈ A for random 15x15 SPD");
    }
    TEST_END();

    TEST_BEGIN("cholesky_27x27_random");
    {
        // Same size as humanoid nv=27
        auto A = random_spd(27, 5.0f);
        auto L = cholesky_gpu(A, 27);
        mx::eval(L);
        float err = max_abs_diff(mx::matmul(L, mx::transpose(L)), A);
        CHECK_LT(err, 0.05f, "L@L^T ≈ A for random 27x27 SPD");
    }
    TEST_END();

    // ── Cholesky solve tests ─────────────────────────────────────

    TEST_SECTION("CholeskySolve");

    TEST_BEGIN("solve_identity");
    {
        auto I = mx::eye(3);
        auto L = cholesky_gpu(I, 3);
        std::vector<float> bdata = {1.0f, 2.0f, 3.0f};
        auto b = mx::array(bdata.data(), {3}, mx::float32);
        auto x = cholesky_solve_gpu(L, b, 3);
        mx::eval(x);
        float err = max_abs_diff(x, b);
        CHECK_LT(err, 1e-5f, "solve(I, b) = b");
    }
    TEST_END();

    TEST_BEGIN("solve_2x2_known");
    {
        // A = [[4, 2], [2, 5]], b = [10, 13]  ->  x = [1, 2] + check
        std::vector<float> adata = {4.0f, 2.0f, 2.0f, 5.0f};
        std::vector<float> bdata = {8.0f, 12.0f};
        auto A = mx::array(adata.data(), {2, 2}, mx::float32);
        auto b = mx::array(bdata.data(), {2}, mx::float32);
        auto L = cholesky_gpu(A, 2);
        auto x = cholesky_solve_gpu(L, b, 2);
        // Verify Ax = b
        auto Ax = mx::flatten(mx::matmul(A, mx::reshape(x, {2, 1})));
        mx::eval(Ax);
        float err = max_abs_diff(Ax, b);
        CHECK_LT(err, 1e-4f, "A@x ≈ b for 2x2 system");
    }
    TEST_END();

    TEST_BEGIN("solve_6x6_roundtrip");
    {
        auto A = random_spd(6, 2.0f);
        auto b = mx::random::normal({6});
        mx::eval(b);
        auto L = cholesky_gpu(A, 6);
        auto x = cholesky_solve_gpu(L, b, 6);
        auto Ax = mx::flatten(mx::matmul(A, mx::reshape(x, {6, 1})));
        mx::eval(Ax);
        float err = max_abs_diff(Ax, b);
        CHECK_LT(err, 1e-2f, "A@x ≈ b for random 6x6 solve");
    }
    TEST_END();

    TEST_BEGIN("solve_27x27_roundtrip");
    {
        auto A = random_spd(27, 5.0f);
        auto b = mx::random::normal({27});
        mx::eval(b);
        auto L = cholesky_gpu(A, 27);
        auto x = cholesky_solve_gpu(L, b, 27);
        auto Ax = mx::flatten(mx::matmul(A, mx::reshape(x, {27, 1})));
        mx::eval(Ax);
        float err = max_abs_diff(Ax, b);
        CHECK_LT(err, 0.1f, "A@x ≈ b for random 27x27 solve");
    }
    TEST_END();

    // ── Humanoid mass matrix tests ───────────────────────────────

    if (humanoid_xml_path) {
        TEST_SECTION("HumanoidMassMatrix");

        TEST_BEGIN("humanoid_qM_symmetric");
        {
            // Get mass matrix from C API forward
            MjmlxModel* model = mjmlx_load_model(humanoid_xml_path);
            if (model) {
                MjmlxData* data = mjmlx_make_data(model);
                mjmlx_forward(model, data);
                auto minfo = mjmlx_model_info(model);
                int nv = minfo.nv;

                // Get qM via MuJoCo C reference
                char error[1024] = {0};
                mjModel* mj_m = mj_loadXML(humanoid_xml_path, nullptr, error, sizeof(error));
                if (mj_m) {
                    mjData* mj_d = mj_makeData(mj_m);
                    mj_forward(mj_m, mj_d);

                    // Extract dense mass matrix from MuJoCo C (double precision)
                    std::vector<double> M_dense_d(nv * nv, 0.0);
                    mj_fullM(mj_m, M_dense_d.data(), mj_d->qM);
                    std::vector<float> M_dense(nv * nv);
                    for (int i = 0; i < nv * nv; i++) M_dense[i] = (float)M_dense_d[i];

                    auto M = mx::array(M_dense.data(), {nv, nv}, mx::float32);
                    // Verify symmetry
                    float sym_err = max_abs_diff(M, mx::transpose(M));
                    CHECK_LT(sym_err, 1e-6f, "MuJoCo C mass matrix is symmetric");

                    // Cholesky on the actual humanoid mass matrix
                    auto M_reg = mx::add(M, mx::multiply(mx::eye(nv), mx::array(1e-6f)));
                    auto L = cholesky_gpu(M_reg, nv);
                    mx::eval(L);
                    float chol_err = max_abs_diff(mx::matmul(L, mx::transpose(L)), M_reg);
                    CHECK_LT(chol_err, 0.01f, "L@L^T ≈ M_reg for humanoid mass matrix");

                    // Solve roundtrip
                    auto b = mx::random::normal({nv});
                    mx::eval(b);
                    auto x = cholesky_solve_gpu(L, b, nv);
                    auto Mx = mx::flatten(mx::matmul(M_reg, mx::reshape(x, {nv, 1})));
                    mx::eval(Mx);
                    float solve_err = max_abs_diff(Mx, b);
                    CHECK_LT(solve_err, 0.05f, "M@x ≈ b for humanoid solve");

                    mj_deleteData(mj_d);
                    mj_deleteModel(mj_m);
                }
                mjmlx_free_data(data);
                mjmlx_free_model(model);
            }
        }
        TEST_END();
    }

    TEST_EXIT();
}
