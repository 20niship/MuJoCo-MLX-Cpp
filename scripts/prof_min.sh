#!/usr/bin/env bash
# bench_go2_step.py をMJMLX_PROF=1で複数回回し、段ごとの.eval(ms/substep)の最小値を出す。他プロセスのGPU/CPU負荷で値がぶれるので最小値で比べる。
# 使い方: scripts/prof_min.sh [build-dir] [回数]   (環境変数は呼び出し側で設定)
B=${1:-build-new}; N=${2:-5}
for i in $(seq "$N"); do
  MJMLX_PROF=1 .venv/bin/python scripts/bench_go2_step.py --build-dir "$B" --steps 40 --warmup 3 2>&1 | grep prof | tail -1
done | tr ' ' '\n' | grep "\.eval=" | awk -F= '{ if (!($1 in m) || $2 < m[$1]) m[$1] = $2 } END { for (k in m) printf "%s=%.1f ", k, m[k]; printf "\n" }'
