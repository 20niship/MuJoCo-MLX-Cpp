#!/usr/bin/env python3
# Usage: scripts/bench_check.py [humanoid.xml] [threshold_pct]; env BUILD_DIR/HISTORY_CSV/REPEATS override defaults.
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def bold(s: str) -> str:
    return c("1", s)


def green(s: str) -> str:
    return c("92", s)


def red(s: str) -> str:
    return c("91", s)


def yellow(s: str) -> str:
    return c("93", s)


def git_hash(project_dir: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_dir, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def read_rows(history_csv: Path) -> list[dict[str, str]]:
    if not history_csv.exists():
        return []
    with open(history_csv, newline="") as f:
        return list(csv.DictReader(f))


SCALAR_MODELS = ["pendulum", "cfrc_ext", "high_dof_tree", "exclude", "t_shape", "humanoid", "go2", "h1"]
BATCHED_MODELS = ["pendulum", "t_shape", "cfrc_ext", "exclude", "high_dof_tree", "humanoid", "go2", "h1"]
BATCH_SIZES = [64, 256, 2048]
ALL_BENCH_NAMES = [f"scalar/{m}" for m in SCALAR_MODELS] + [f"batched/{m}/B{b}" for m in BATCHED_MODELS for b in BATCH_SIZES]


NAME_RE = re.compile(r"(?:batched|scalar)/([^/]+)(?:/B(\d+))?$")


def maybe_plot(per_bench: dict[str, list[float]], all_rows: list[dict[str, str]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(yellow("matplotlib not installed, skipping plot"))
        return

    history_best: dict[str, float] = {}
    for row in all_rows:
        sps = float(row["steps_per_sec"])
        if sps <= 0:
            continue
        name = row["benchmark"]
        if sps > history_best.get(name, 0.0):
            history_best[name] = sps

    models: dict[str, list[tuple[int, str]]] = {}
    for name in per_bench:
        m = NAME_RE.match(name)
        if not m:
            continue
        model, b = m.groups()
        models.setdefault(model, []).append((int(b) if b else 1, name))

    if not models:
        return

    names_sorted = sorted(models)
    cols = min(4, len(names_sorted))
    rows_n = (len(names_sorted) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)

    for idx, model in enumerate(names_sorted):
        ax = axes[idx // cols][idx % cols]
        entries = sorted(models[model])
        xs = [e[0] for e in entries]
        bench_names = [e[1] for e in entries]
        avgs = [sum(per_bench[n]) / len(per_bench[n]) for n in bench_names]
        mins = [min(per_bench[n]) for n in bench_names]
        maxs = [max(per_bench[n]) for n in bench_names]
        bests = [history_best.get(n, 0.0) for n in bench_names]
        x_pos = list(range(len(xs)))
        yerr = [[a - lo for a, lo in zip(avgs, mins)], [hi - a for a, hi in zip(avgs, maxs)]]
        ax.errorbar(x_pos, avgs, yerr=yerr, fmt="o-", label="current", color="tab:blue", capsize=3)
        ax.plot(x_pos, bests, "s--", label="best", color="tab:orange")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(model)
        ax.set_ylabel("steps/sec")
        ax.set_yscale("log")
        ax.legend(fontsize=6)

    for idx in range(len(names_sorted), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    build_dir = Path(os.environ.get("BUILD_DIR", project_dir / "build"))
    history_csv = Path(os.environ.get("HISTORY_CSV", project_dir / "benchmarks" / "history.csv"))
    repeats = int(os.environ.get("REPEATS", "5"))
    timeout_sec = float(os.environ.get("BENCH_TIMEOUT_SEC", "120"))
    humanoid = sys.argv[1] if len(sys.argv) > 1 else ""
    threshold_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    print(bold("=== MuJoCo-MLX-Cpp Benchmark Regression Check ==="))
    print(f"Project:   {project_dir}")
    print(f"Threshold: {threshold_pct}%")
    print(f"Git:       {git_hash(project_dir)}")
    print()

    if not build_dir.is_dir():
        print(red(f"ERROR: Build directory not found at {build_dir}"))
        print("Run 'cmake -B build && cmake --build build' first.")
        return 1

    (project_dir / "benchmarks").mkdir(exist_ok=True)
    subprocess.run([str(script_dir / "fetch_models.sh")], check=True)
    menagerie_dir = project_dir / "benchmarks" / "models" / "external" / "mujoco_menagerie"
    go2_model = menagerie_dir / "unitree_go2" / "scene.xml"
    h1_model = menagerie_dir / "unitree_h1" / "scene.xml"

    bench_baseline = build_dir / "bench_baseline"
    if not os.access(bench_baseline, os.X_OK):
        print(red("ERROR: bench_baseline not found. Build with -DMJMLX_BUILD_TESTS=ON."))
        return 1

    # 1プロセス=1ベンチマークで起動する: GPU hangでプロセスが落ちても他の計測を巻き込まないようにするため。
    print(bold(f"--- Running bench_baseline ({repeats}x per benchmark, 1 process each, for min/max/avg) ---"))
    run_timestamps: set[str] = set()
    crashes: list[tuple[int, str]] = []
    bench_names = [n for n in ALL_BENCH_NAMES if humanoid or "humanoid" not in n]
    total = repeats * len(bench_names)
    done = 0
    env = dict(os.environ)
    for i in range(1, repeats + 1):
        for name in bench_names:
            done += 1
            before = len(read_rows(history_csv))
            env["MJMLX_BENCH_ONLY"] = name
            try:
                result = subprocess.run(
                    [str(bench_baseline), humanoid, str(history_csv), str(go2_model), str(h1_model)],
                    env=env, capture_output=True, text=True, timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                crashes.append((i, name))
                print(red(f"  [{done}/{total}] TIMEOUT {name} (> {timeout_sec:.0f}s, likely GPU hang)"))
                continue
            after_rows = read_rows(history_csv)
            if result.returncode != 0:
                crashes.append((i, name))
                print(red(f"  [{done}/{total}] CRASH {name} (exit {result.returncode})"))
                tail = "\n".join((result.stdout + result.stderr).splitlines()[-5:])
                if tail:
                    print(f"    {tail}")
                continue
            if len(after_rows) <= before:
                print(yellow(f"  [{done}/{total}] SKIP {name} (model path not given)"))
                continue
            run_timestamps.add(after_rows[-1]["timestamp"])
            line = next((l for l in result.stdout.splitlines() if l.strip().startswith(name)), "")
            print(f"  [{done}/{total}] {line.strip() or name}")

    if crashes:
        print()
        print(yellow(f"{len(crashes)} of {total} runs crashed (excluded from aggregation):"))
        for i, name in crashes:
            print(f"  run {i}: {name}")

    rows = read_rows(history_csv)
    this_run_ts = run_timestamps
    per_bench: dict[str, list[float]] = {}
    for row in rows:
        if row["timestamp"] not in this_run_ts:
            continue
        sps = float(row["steps_per_sec"])
        if sps <= 0:
            continue
        per_bench.setdefault(row["benchmark"], []).append(sps)

    print()
    print(bold(f"=== min/max/avg steps/sec across {repeats} runs ==="))
    print(f"{'benchmark':<32} {'n':>3} {'min':>12} {'avg':>12} {'max':>12}")
    for name in sorted(per_bench):
        vals = per_bench[name]
        print(f"{name:<32} {len(vals):>3} {min(vals):>12.0f} {sum(vals) / len(vals):>12.0f} {max(vals):>12.0f}")

    plot_suffix = history_csv.stem.removeprefix("history")  # history.csv->"", history-mkx.csv->"-mkx"
    maybe_plot(per_bench, rows, project_dir / "benchmarks" / f"bench_plot{plot_suffix}.png")

    for bench_exe in sorted(build_dir.glob("bench_*")):
        if bench_exe.name == "bench_baseline" or not os.access(bench_exe, os.X_OK):
            continue
        print()
        print(bold(f"--- Running {bench_exe.name} ---"))
        subprocess.run([str(bench_exe), humanoid, str(history_csv)])

    print()

    rows = read_rows(history_csv)
    if not rows:
        print(yellow("No history CSV found. First run -- no regression check possible."))
        return 0

    timestamps = sorted({row["timestamp"] for row in rows})
    if len(timestamps) < 2:
        print(yellow(f"Only {len(timestamps)} run(s) in history. Need at least 2 for regression check."))
        print(f"Results saved to {history_csv}")
        return 0

    latest, previous = timestamps[-1], timestamps[-2]
    print(bold("=== Regression Check ==="))
    print(f"Comparing: {latest} vs {previous}")
    print(f"Threshold: {threshold_pct}%")
    print()

    latest_by_name = {row["benchmark"]: float(row["steps_per_sec"]) for row in rows if row["timestamp"] == latest}
    previous_by_name = {row["benchmark"]: float(row["steps_per_sec"]) for row in rows if row["timestamp"] == previous}

    print(f"{'Benchmark':<35} {'Previous':>12} {'Latest':>12} {'Change':>8} Status")
    print(f"{'---------':<35} {'--------':>12} {'------':>12} {'------':>8} ------")

    regressions = 0
    for name in sorted(latest_by_name):
        curr = latest_by_name[name]
        prev = previous_by_name.get(name)
        if prev is None or prev == 0:
            print(f"{name:<35} {'N/A':>12} {curr:>12.0f} {'N/A':>8} {yellow('SKIP')}")
            continue
        change = (curr - prev) / prev * 100
        if change < -threshold_pct:
            status = red("REGRESSION")
            regressions += 1
        else:
            status = green("OK")
        print(f"{name:<35} {prev:>12.0f} {curr:>12.0f} {change:>7.1f}% {status}")

    print()
    if regressions > 0:
        print(red(f"WARNING: {regressions} benchmark(s) regressed by more than {threshold_pct}%!"))
        return 1
    print(green(f"All benchmarks within {threshold_pct}% threshold."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
