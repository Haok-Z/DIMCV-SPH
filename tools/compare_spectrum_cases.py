"""Compare spectrum summaries of two batch runs side by side.

The two runs share the same dimcv/dfsph scenes, so any change in those two
columns is a pure common-mask artifact: swapping the dfsph_segment scene
changes the three-way intersection mask and therefore re-clips every field.

Usage:
    python tools/compare_spectrum_cases.py
"""
import csv
from pathlib import Path

OLD = Path("results/spectrum3d_mixer/summary.csv")
NEW = Path("results/spectrum3d_mixer_residual_circulation/summary.csv")
OUT = Path("results/spectrum3d_mixer_residual_circulation/comparison_vs_baseline.csv")

KEY = ("region", "stage", "spacing_target", "quantity", "field")
METRICS = ("total_energy", "spectral_centroid", "slope", "log_distance_to_dfsph",
           "high_k_fraction", "common_mask_fraction")


def load(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            if not r["spacing_target"]:
                continue
            rows.append(r)
    return {tuple(r[k] for k in KEY): r for r in rows}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    old, new = load(OLD), load(NEW)
    keys = [k for k in new if k in old]

    out_rows = []
    for k in sorted(keys):
        row = dict(zip(KEY, k))
        for m in METRICS:
            a, b = num(old[k][m]), num(new[k][m])
            row[f"{m}_old"] = a
            row[f"{m}_new"] = b
            row[f"{m}_rel_change"] = (b - a) / a if a not in (None, 0) and b is not None else None
        out_rows.append(row)

    with OUT.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    lines = [f"compared {len(out_rows)} rows -> {OUT}", ""]

    lines.append("== 1. Segment contribution (hybrid decomposition) ==")
    lines.append(f"{'region':<11}{'stage':<8}{'dx':<8}{'field':<18}{'old':>14}{'new':>14}{'x':>10}")
    for k in sorted(keys):
        if k[3] != "hybrid" or k[4] not in ("segment_bs_vel", "segment_bs_curl"):
            continue
        a, b = num(old[k]["total_energy"]), num(new[k]["total_energy"])
        lines.append(f"{k[0]:<11}{k[1]:<8}{k[2]:<8}{k[4]:<18}{a:>14.4e}{b:>14.4e}{(b / a if a else 0):>9.1f}x")

    lines.append("")
    lines.append("== 2. final_vel vs pre_feedback_vel gain (%) ==")
    lines.append(f"{'region':<11}{'stage':<8}{'dx':<8}{'old %':>10}{'new %':>10}")
    for k in sorted(keys):
        if k[3] != "hybrid" or k[4] != "final_vel":
            continue
        pk = k[:4] + ("pre_feedback_vel",)
        a = num(old[k]["total_energy"]) / num(old[pk]["total_energy"]) - 1
        b = num(new[k]["total_energy"]) / num(new[pk]["total_energy"]) - 1
        lines.append(f"{k[0]:<11}{k[1]:<8}{k[2]:<8}{a * 100:>10.3f}{b * 100:>10.3f}")

    lines.append("")
    lines.append("== 3. log_distance_to_dfsph (deviation from DFSPH) ==")
    lines.append(f"{'region':<11}{'stage':<8}{'dx':<8}{'quantity':<14}{'field':<15}{'old':>9}{'new':>9}{'ratio':>8}")
    for k in sorted(keys):
        if k[3] == "hybrid" or k[4] == "dfsph":
            continue
        a, b = num(old[k]["log_distance_to_dfsph"]), num(new[k]["log_distance_to_dfsph"])
        lines.append(f"{k[0]:<11}{k[1]:<8}{k[2]:<8}{k[3]:<14}{k[4]:<15}"
                     f"{a:>9.4f}{b:>9.4f}{(b / a if a else 0):>8.2f}")

    lines.append("")
    lines.append("== 4. dimcv / dfsph drift (common-mask artifact check) ==")
    lines.append(f"{'region':<11}{'stage':<8}{'dx':<8}{'quantity':<14}{'field':<7}"
                 f"{'old E':>11}{'new E':>11}{'rel%':>8}")
    for k in sorted(keys):
        if k[3] == "hybrid" or k[4] not in ("dimcv", "dfsph") or k[3] != "velocity":
            continue
        a, b = num(old[k]["total_energy"]), num(new[k]["total_energy"])
        lines.append(f"{k[0]:<11}{k[1]:<8}{k[2]:<8}{k[3]:<14}{k[4]:<7}"
                     f"{a:>11.6f}{b:>11.6f}{(b / a - 1) * 100:>8.2f}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
