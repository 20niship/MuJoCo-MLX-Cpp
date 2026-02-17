#!/bin/bash
# bench_check.sh -- Run all benchmarks and check for regressions.
#
# Usage:
#   scripts/bench_check.sh [humanoid.xml] [threshold_pct]
#
# Arguments:
#   humanoid.xml   - Optional path to humanoid model for full benchmark suite
#   threshold_pct  - Regression threshold percentage (default: 10)
#
# This script:
#   1. Builds the benchmark executables
#   2. Runs bench_baseline (and any phase benchmarks that exist)
#   3. Appends results to benchmarks/history.csv
#   4. Compares latest run against the previous run
#   5. Flags regressions exceeding the threshold

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
HISTORY_CSV="${PROJECT_DIR}/benchmarks/history.csv"
THRESHOLD_PCT="${2:-10}"
HUMANOID="${1:-}"

echo "=== MuJoCo-MLX-Cpp Benchmark Regression Check ==="
echo "Project:   ${PROJECT_DIR}"
echo "Threshold: ${THRESHOLD_PCT}%"
echo "Git:       $(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

# Ensure build directory exists and benchmarks are built
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found at ${BUILD_DIR}"
    echo "Run 'cmake -B build && cmake --build build' first."
    exit 1
fi

# Ensure benchmarks directory exists
mkdir -p "${PROJECT_DIR}/benchmarks"

# Run bench_baseline
BENCH_BASELINE="${BUILD_DIR}/bench_baseline"
if [ ! -x "$BENCH_BASELINE" ]; then
    echo "ERROR: bench_baseline not found. Build with -DMJMLX_BUILD_TESTS=ON."
    exit 1
fi

echo "--- Running bench_baseline ---"
if [ -n "$HUMANOID" ]; then
    "$BENCH_BASELINE" "$HUMANOID" "$HISTORY_CSV"
else
    "$BENCH_BASELINE" "" "$HISTORY_CSV"
fi

# Run any phase-specific benchmarks that exist
for bench_exe in "${BUILD_DIR}"/bench_*; do
    if [ -x "$bench_exe" ] && [ "$(basename "$bench_exe")" != "bench_baseline" ]; then
        echo ""
        echo "--- Running $(basename "$bench_exe") ---"
        if [ -n "$HUMANOID" ]; then
            "$bench_exe" "$HUMANOID" "$HISTORY_CSV" || true
        else
            "$bench_exe" "" "$HISTORY_CSV" || true
        fi
    fi
done

echo ""

# Check for regressions by comparing last two runs
if [ ! -f "$HISTORY_CSV" ]; then
    echo "No history CSV found. First run -- no regression check possible."
    exit 0
fi

# Count unique timestamps (each run has one timestamp)
TIMESTAMPS=$(tail -n +2 "$HISTORY_CSV" | cut -d',' -f1 | sort -u)
NUM_RUNS=$(echo "$TIMESTAMPS" | wc -l | tr -d ' ')

if [ "$NUM_RUNS" -lt 2 ]; then
    echo "Only ${NUM_RUNS} run(s) in history. Need at least 2 for regression check."
    echo "Results saved to ${HISTORY_CSV}"
    exit 0
fi

# Get the two most recent timestamps
LATEST=$(echo "$TIMESTAMPS" | tail -1)
PREVIOUS=$(echo "$TIMESTAMPS" | tail -2 | head -1)

echo "=== Regression Check ==="
echo "Comparing: ${LATEST} vs ${PREVIOUS}"
echo "Threshold: ${THRESHOLD_PCT}%"
echo ""

# Compare steps_per_sec for each benchmark name
REGRESSIONS=0
printf "%-35s %12s %12s %8s %s\n" "Benchmark" "Previous" "Latest" "Change" "Status"
printf "%-35s %12s %12s %8s %s\n" "---------" "--------" "------" "------" "------"

# Get unique benchmark names from the latest run
BENCH_NAMES=$(grep "^${LATEST}," "$HISTORY_CSV" | cut -d',' -f3 | sort -u)

while IFS= read -r bench; do
    [ -z "$bench" ] && continue

    # Get steps_per_sec (column 13) for this benchmark from each run
    PREV_SPS=$(grep "^${PREVIOUS},.*,${bench}," "$HISTORY_CSV" | head -1 | cut -d',' -f13)
    CURR_SPS=$(grep "^${LATEST},.*,${bench}," "$HISTORY_CSV" | head -1 | cut -d',' -f13)

    if [ -z "$PREV_SPS" ] || [ -z "$CURR_SPS" ] || [ "$PREV_SPS" = "0" ]; then
        printf "%-35s %12s %12s %8s %s\n" "$bench" "${PREV_SPS:-N/A}" "${CURR_SPS:-N/A}" "N/A" "SKIP"
        continue
    fi

    # Calculate percentage change
    CHANGE=$(echo "scale=1; (($CURR_SPS - $PREV_SPS) / $PREV_SPS) * 100" | bc 2>/dev/null || echo "0")

    # Check if regression exceeds threshold
    IS_REGRESSION=$(echo "$CHANGE < -${THRESHOLD_PCT}" | bc 2>/dev/null || echo "0")

    if [ "$IS_REGRESSION" = "1" ]; then
        STATUS="REGRESSION"
        REGRESSIONS=$((REGRESSIONS + 1))
    else
        STATUS="OK"
    fi

    printf "%-35s %12.0f %12.0f %7.1f%% %s\n" "$bench" "$PREV_SPS" "$CURR_SPS" "$CHANGE" "$STATUS"
done <<< "$BENCH_NAMES"

echo ""
if [ "$REGRESSIONS" -gt 0 ]; then
    echo "WARNING: ${REGRESSIONS} benchmark(s) regressed by more than ${THRESHOLD_PCT}%!"
    exit 1
else
    echo "All benchmarks within ${THRESHOLD_PCT}% threshold."
    exit 0
fi
