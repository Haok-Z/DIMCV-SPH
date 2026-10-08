"""Dump which PLY frames each stage actually samples, for one manifest case.

Usage:
    python tools/dump_stage_frames.py [--write-csv]

Reads flow_statistics.csv of every method, reproduces the target-time grid of
analyze_spatial_spectrum_3d.target_times(), and resolves each target to the
nearest available frame. Also reports the alignment error against maxTimeError.
"""
import argparse
import csv
import json
from pathlib import Path

MANIFEST = Path("spectrum_experiments_3d.json")
CASE = "mixer_coarse"
RESULT_DIRS = {
    "dimcv": "results/DIM_von_karman_vortex_3d_dimcv_rotating_mixer_unified_coarse/dimcv",
    "dfsph": "results/DIM_von_karman_vortex_3d_dfsph_rotating_mixer_unified_coarse/dfsph",
    "dfsph_segment": "results/DIM_von_karman_vortex_3d_dfsph_segment_rotating_mixer_unified_coarse/dfsph_segment",
}


def load_frames(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [(int(r["frame"]), float(r["time"])) for r in csv.DictReader(stream)]


def target_times(intervals, sample_interval):
    values = []
    for start, stop in intervals:
        count = int((stop - start) / sample_interval + 1e-9)
        values.extend(start + sample_interval * i for i in range(count + 1))
    return sorted({round(v, 6) for v in values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=CASE)
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case = manifest["cases"][args.case]
    interval = float(manifest.get("sampleInterval", 0.25))
    max_error = float(case.get("maxTimeError", 0.020))
    frames = {m: load_frames(Path(d) / "flow_statistics.csv") for m, d in RESULT_DIRS.items()}

    rows, overlaps = [], []
    seen = {}
    for stage, intervals in case["stages"].items():
        for t in target_times(intervals, interval):
            pick, errs = {}, []
            for m, data in frames.items():
                fr, tt = min(data, key=lambda x: abs(x[1] - t))
                pick[m] = (fr, tt)
                errs.append(abs(tt - t))
            if t in seen:
                overlaps.append((t, seen[t], stage))
            seen[t] = stage
            rows.append({
                "stage": stage, "target_time": round(t, 3),
                **{f"{m}_frame": pick[m][0] for m in frames},
                "actual_time": pick["dimcv"][1],
                "max_abs_error": round(max(errs), 5),
                "within_tolerance": bool(max(errs) <= max_error),
                "kinetic_energy": None,
            })

    series = load_ke()
    for r in rows:
        _, value = min(series, key=lambda x: abs(x[0] - r["target_time"]))
        r["kinetic_energy"] = round(value, 3)

    out = Path("results/spectrum3d_mixer") / "stage_frames.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"frames per method: {[len(v) for v in frames.values()]}")
    print(f"rows written: {len(rows)} -> {out}")
    if overlaps:
        print("boundary frames shared by two stages:")
        for t, a, b in overlaps:
            print(f"  t={t:.2f}: {a} and {b}")


def load_ke():
    p = Path(RESULT_DIRS["dimcv"]) / "flow_statistics.csv"
    with p.open(newline="", encoding="utf-8") as stream:
        data = [(float(r["time"]), float(r["kinetic_energy_total"])) for r in csv.DictReader(stream)]
    return data


if __name__ == "__main__":
    main()
