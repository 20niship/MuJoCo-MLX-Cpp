#!/bin/bash
# fetch_models.sh -- idempotent download of Unitree Go2/H1 (mujoco_menagerie) into benchmarks/models/external/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MENAGERIE_DIR="${PROJECT_DIR}/benchmarks/models/external/mujoco_menagerie"

if [ -d "${MENAGERIE_DIR}/unitree_go2" ] && [ -d "${MENAGERIE_DIR}/unitree_h1" ]; then
    echo "fetch_models.sh: models already present, skipping."
    exit 0
fi

mkdir -p "$(dirname "$MENAGERIE_DIR")"
echo "fetch_models.sh: fetching mujoco_menagerie (unitree_go2, unitree_h1)..."
/bin/rm -rf "$MENAGERIE_DIR"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
git -C "$MENAGERIE_DIR" sparse-checkout set unitree_go2 unitree_h1
echo "fetch_models.sh: done."
