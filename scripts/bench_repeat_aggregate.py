#!/usr/bin/env python3
# CSV列: [0]timestamp [2]name ... [12]steps_per_sec (bench_utils.hのbench_write_csv列順に合わせる)
import csv
import sys


def main() -> None:
    csv_path, first_ts, last_ts = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    ts_idx, name_idx, sps_idx = 0, 2, 12

    by_name: dict[str, list[float]] = {}
    for row in rows:
        if not (first_ts <= row[ts_idx] <= last_ts):
            continue
        sps = float(row[sps_idx])
        if sps <= 0:
            continue
        by_name.setdefault(row[name_idx], []).append(sps)

    print(f"{'benchmark':<28} {'n':>3} {'min':>12} {'avg':>12} {'max':>12}")
    for name in sorted(by_name):
        vals = by_name[name]
        print(f"{name:<28} {len(vals):>3} {min(vals):>12.0f} {sum(vals) / len(vals):>12.0f} {max(vals):>12.0f}")


if __name__ == "__main__":
    main()
