#!/bin/bash
# Run all MuJoCo-MLX-Cpp tests
# Usage: ./run_tests.sh [path/to/humanoid.xml]
#
# If no path given, falls back to HUMANOID env var.
# Model-dependent tests are skipped if no model path is available.

set -e

MODEL="${1:-${HUMANOID:-}}"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: build/ directory not found. Run cmake --build build first."
    exit 1
fi

cd "$BUILD_DIR"

PASS=0
FAIL=0
SKIP=0
TOTAL=0
FAILED_TESTS=""

run_test() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    printf "%-30s " "$name"
    if ./"$name" "$@" > /tmp/mjmlx_test_output_$$.txt 2>&1; then
        # Check if any FAIL in output
        if grep -q "FAIL" /tmp/mjmlx_test_output_$$.txt; then
            FAIL=$((FAIL + 1))
            FAILED_TESTS="$FAILED_TESTS $name"
            echo "FAIL"
            # Show summary line
            grep -E "(FAIL|PASS|===)" /tmp/mjmlx_test_output_$$.txt | tail -3
        else
            PASS=$((PASS + 1))
            echo "PASS"
        fi
    else
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name"
        echo "FAIL (exit code $?)"
        tail -5 /tmp/mjmlx_test_output_$$.txt
    fi
    rm -f /tmp/mjmlx_test_output_$$.txt
}

echo "========================================"
echo "  MuJoCo-MLX-Cpp Test Suite"
echo "========================================"
if [ -n "$MODEL" ]; then
    echo "  Model: $MODEL"
else
    echo "  Model: (none -- model-dependent tests will be skipped)"
fi
echo ""

# Self-contained tests (no model needed)
echo "--- Self-contained tests ---"
run_test test_math_full
run_test test_io_full
run_test test_forward_full
run_test test_physics_full
run_test test_collision_full
run_test test_linalg_full
run_test test_vmap_smooth

# Model-dependent tests
if [ -n "$MODEL" ]; then
    echo ""
    echo "--- Model-dependent tests ---"
    run_test test_solver_full "$MODEL"
    run_test test_linalg_full "$MODEL"
    run_test test_batched_diag "$MODEL"
    run_test test_batched "$MODEL" 64 5
fi

echo ""
echo "========================================"
echo "  Results: $PASS/$TOTAL passed"
if [ $FAIL -gt 0 ]; then
    echo "  FAILED ($FAIL):$FAILED_TESTS"
fi
if [ $SKIP -gt 0 ]; then
    echo "  Skipped: $SKIP"
fi
echo "========================================"

exit $FAIL
