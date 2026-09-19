#!/bin/bash
# 単発だと稀に遅い測定が出るため同条件でN回走らせmin/max/avgを取る(引数: humanoid.xml [num_runs=5])。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
HISTORY_CSV="${HISTORY_CSV:-${PROJECT_DIR}/benchmarks/history.csv}"
HUMANOID="${1:-}"
NUM_RUNS="${2:-5}"

BENCH_BASELINE="${BUILD_DIR}/bench_baseline"
if [ ! -x "$BENCH_BASELINE" ]; then
    echo "ERROR: bench_baseline not found. Build with -DMJMLX_BUILD_TESTS=ON."
    exit 1
fi

mkdir -p "${PROJECT_DIR}/benchmarks"
"${SCRIPT_DIR}/fetch_models.sh"
MENAGERIE_DIR="${PROJECT_DIR}/benchmarks/models/external/mujoco_menagerie"
GO2_MODEL="${MENAGERIE_DIR}/unitree_go2/scene.xml"
H1_MODEL="${MENAGERIE_DIR}/unitree_h1/scene.xml"

echo "=== MuJoCo-MLX-Cpp Benchmark Repeat (${NUM_RUNS} runs) ==="
echo "Git: $(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

FIRST_TS=""
for i in $(seq 1 "$NUM_RUNS"); do
    echo "--- run ${i}/${NUM_RUNS} ---"
    "$BENCH_BASELINE" "$HUMANOID" "$HISTORY_CSV" "$GO2_MODEL" "$H1_MODEL" | grep -E "steps/s=" || true
    TS=$(tail -n +2 "$HISTORY_CSV" | cut -d',' -f1 | sort -u | tail -1)
    [ -z "$FIRST_TS" ] && FIRST_TS="$TS"
    LAST_TS="$TS"
done

echo ""
echo "=== min/max/avg across ${NUM_RUNS} runs (timestamps ${FIRST_TS}..${LAST_TS}) ==="
python3 "${SCRIPT_DIR}/bench_repeat_aggregate.py" "$HISTORY_CSV" "$FIRST_TS" "$LAST_TS"
