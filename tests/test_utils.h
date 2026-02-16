// Test framework for MuJoCo-MLX-Cpp
// Provides CHECK macros, array comparison, MuJoCo C reference utilities.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <functional>
#include <mujoco/mujoco.h>
#include "mjmlx/mjmlx.h"

// ── Test tracking ────────────────────────────────────────────────────────────

static int _tests_run = 0;
static int _tests_passed = 0;
static int _tests_failed = 0;

#define TEST_BEGIN(name) do { \
    _tests_run++; \
    const char* _test_name = name; \
    bool _test_ok = true; \
    (void)_test_name;

#define TEST_END() \
    if (_test_ok) { _tests_passed++; printf("  PASS: %s\n", _test_name); } \
    else { _tests_failed++; printf("  FAIL: %s\n", _test_name); } \
} while(0)

#define TEST_SUMMARY() do { \
    printf("\n=== %d/%d tests passed", _tests_passed, _tests_run); \
    if (_tests_failed > 0) printf(" (%d FAILED)", _tests_failed); \
    printf(" ===\n"); \
} while(0)

#define TEST_EXIT() do { \
    TEST_SUMMARY(); \
    return _tests_failed > 0 ? 1 : 0; \
} while(0)

// ── Assertions ───────────────────────────────────────────────────────────────

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("    ASSERT FAILED: %s\n      at %s:%d\n", msg, __FILE__, __LINE__); \
        _test_ok = false; \
    } \
} while(0)

#define CHECK_CLOSE(a, b, tol, msg) do { \
    double _a = (double)(a), _b = (double)(b), _t = (double)(tol); \
    if (std::abs(_a - _b) > _t) { \
        printf("    ASSERT FAILED: %s\n      expected %.8g, got %.8g (diff=%.2e, tol=%.2e)\n      at %s:%d\n", \
               msg, _b, _a, std::abs(_a - _b), _t, __FILE__, __LINE__); \
        _test_ok = false; \
    } \
} while(0)

#define CHECK_ARRAY_CLOSE(a, b, n, tol, msg) do { \
    for (int _i = 0; _i < (n); _i++) { \
        double _ai = (double)(a)[_i], _bi = (double)(b)[_i]; \
        if (std::abs(_ai - _bi) > (double)(tol)) { \
            printf("    ASSERT FAILED: %s [%d]\n      expected %.8g, got %.8g (diff=%.2e, tol=%.2e)\n      at %s:%d\n", \
                   msg, _i, _bi, _ai, std::abs(_ai - _bi), (double)(tol), __FILE__, __LINE__); \
            _test_ok = false; \
            break; \
        } \
    } \
} while(0)

#define CHECK_NO_NAN(arr, n, msg) do { \
    for (int _i = 0; _i < (n); _i++) { \
        if (std::isnan((arr)[_i]) || std::isinf((arr)[_i])) { \
            printf("    ASSERT FAILED: %s [%d] = %g (NaN/Inf)\n      at %s:%d\n", \
                   msg, _i, (double)(arr)[_i], __FILE__, __LINE__); \
            _test_ok = false; \
            break; \
        } \
    } \
} while(0)

#define CHECK_GT(a, b, msg) do { \
    double _a = (double)(a), _b = (double)(b); \
    if (!(_a > _b)) { \
        printf("    ASSERT FAILED: %s\n      expected > %.8g, got %.8g\n      at %s:%d\n", \
               msg, _b, _a, __FILE__, __LINE__); \
        _test_ok = false; \
    } \
} while(0)

#define CHECK_LT(a, b, msg) do { \
    double _a = (double)(a), _b = (double)(b); \
    if (!(_a < _b)) { \
        printf("    ASSERT FAILED: %s\n      expected < %.8g, got %.8g\n      at %s:%d\n", \
               msg, _b, _a, __FILE__, __LINE__); \
        _test_ok = false; \
    } \
} while(0)

// ── Temp file helper ─────────────────────────────────────────────────────────

#include <unistd.h>

static std::string write_temp_xml(const char* xml_str) {
    char tmppath[] = "/tmp/mjmlx_test_XXXXXX.xml";
    int fd = mkstemps(tmppath, 4);
    if (fd < 0) return "";
    write(fd, xml_str, strlen(xml_str));
    close(fd);
    return std::string(tmppath);
}

// ── MuJoCo helpers ───────────────────────────────────────────────────────────

// Load MuJoCo C model from XML string (via temp file)
static mjModel* mj_load_xml_string(const char* xml_str, char* error, int error_sz) {
    auto path = write_temp_xml(xml_str);
    if (path.empty()) return nullptr;
    mjModel* m = mj_loadXML(path.c_str(), nullptr, error, error_sz);
    unlink(path.c_str());
    return m;
}

// RAII scope for MuJoCo C model+data
struct MjScope {
    mjModel* m = nullptr;
    mjData* d = nullptr;
    MjScope(const char* xml_str) {
        char error[1024] = {0};
        m = mj_load_xml_string(xml_str, error, sizeof(error));
        if (!m) {
            fprintf(stderr, "MuJoCo load failed: %s\n", error);
            return;
        }
        d = mj_makeData(m);
    }
    ~MjScope() {
        if (d) mj_deleteData(d);
        if (m) mj_deleteModel(m);
    }
    bool ok() const { return m && d; }
};

// Load mjmlx model from XML string (uses C API directly)
static MjmlxModel* mjmlx_load_xml_string(const char* xml_str) {
    return mjmlx_load_model_from_string(xml_str);
}

// ── Section headers ──────────────────────────────────────────────────────────

#define TEST_SECTION(name) printf("\n--- %s ---\n", name)
