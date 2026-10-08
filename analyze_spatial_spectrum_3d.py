"""Batch 3D spectra for manifest-defined SPH method comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config_builder import resolve_output_paths
from spatial_spectrum_3d import (
    analyze_reconstructed_fields,
    grid_for_spacing,
    load_particle_frame,
    load_particle_radius,
    reconstruct_sph_vector_grid,
    spectrum_metrics,
    vector_curl_spectral,
    radial_shell_spectrum,
)

METHODS = ("dimcv", "dfsph", "dfsph_segment")
DISPLAY_NAMES = {
    "dimcv": "DIMCV", "dfsph": "DFSPH", "dfsph_segment": "DFSPH+Segment", "mpsph": "MPSPH",
}
COLORS = {
    "dimcv": "#d18b00", "dfsph": "#1f77b4", "dfsph_segment": "#228b5b", "mpsph": "#b23a48",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def output_paths(scene_path: Path, method: str) -> tuple[Path, Path, dict]:
    cfg = load_json(scene_path)["Configuration"]
    _, ply_dir = resolve_output_paths(scene_path, cfg, method)
    return ply_dir, ply_dir.parent / "flow_statistics.csv", cfg


def statistics_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing flow statistics: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty flow statistics: {path}")
    return rows


def nearest_frame(rows: list[dict], target_time: float) -> tuple[int, float]:
    candidate = min(rows, key=lambda row: abs(float(row["time"]) - target_time))
    return int(candidate["frame"]), float(candidate["time"])


def target_times(intervals: list[list[float]], sample_interval: float) -> np.ndarray:
    values = []
    for start, stop in intervals:
        count = int(np.floor((stop - start) / sample_interval + 1e-9))
        values.extend(start + sample_interval * np.arange(count + 1))
    return np.unique(np.asarray(values, dtype=np.float64))


def save_comparison_plot(path: Path, spectra: dict[str, dict], column: int, ylabel: str,
                         reliable_k: float, ratio: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    baseline = spectra.get("dfsph")
    for method, values in spectra.items():
        radial, deviation = values["mean"], values["std"][:, column]
        k, y = radial[:, 0], radial[:, column]
        if ratio:
            if baseline is None:
                continue
            base = baseline["mean"][:, column]
            y = np.divide(y, base, out=np.full_like(y, np.nan), where=base > 0)
            deviation = np.divide(deviation, base, out=np.zeros_like(deviation), where=base > 0)
        positive = np.isfinite(y) & (y > 0)
        ax.loglog(k[positive], y[positive], color=COLORS.get(method), label=DISPLAY_NAMES.get(method, method))
        if not ratio:
            lower, upper = np.maximum(y - deviation, 0), y + deviation
            shaded = positive & (upper > 0)
            ax.fill_between(k[shaded], lower[shaded], upper[shaded], color=COLORS.get(method), alpha=0.16)
    ax.axvline(reliable_k, color="#555555", linestyle="--", linewidth=1, label="reliable limit")
    ax.set_xlabel("Wavenumber |k| (rad/m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_stage_comparison(path: Path, spectra: dict[str, dict], quantity: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = tuple(spectra)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["k"]
        for method in methods:
            header += [f"{method}_mean", f"{method}_std", f"{method}_normalized_mean"]
        writer.writerow(header)
        for index, k in enumerate(spectra[methods[0]]["mean"][:, 0]):
            row = [k]
            for method in methods:
                mean, std = spectra[method]["mean"][index], spectra[method]["std"][index]
                row += [mean[1], std[1], mean[2]]
            writer.writerow(row)
    (path.with_suffix(".json")).write_text(json.dumps({"quantity": quantity, **metadata}, indent=2), encoding="utf-8")


def aggregate(samples: dict[str, list[np.ndarray]]) -> dict[str, dict]:
    result = {}
    for method, curves in samples.items():
        stack = np.stack(curves)
        result[method] = {"mean": stack.mean(axis=0), "std": stack.std(axis=0, ddof=0)}
    return result


def convergence_rows(stage_aggregates: dict[tuple[str, float], tuple[dict[str, dict], dict]],
                     case_name: str, region: str, stage: str) -> list[dict]:
    """Compare coarse/fine reconstruction grids on their shared reliable band."""
    rows = []
    for quantity in ("velocity", "vorticity", "velocity_curl"):
        candidates = sorted(spacing for key_quantity, spacing in stage_aggregates if key_quantity == quantity)
        if len(candidates) < 2:
            continue
        fine_spacing, coarse_spacing = candidates[0], candidates[-1]
        fine, _ = stage_aggregates[(quantity, fine_spacing)]
        coarse, coarse_metadata = stage_aggregates[(quantity, coarse_spacing)]
        reliable_k = 0.8 * np.pi / max(coarse_metadata["spacing"])
        for method in METHODS:
            coarse_curve = coarse[method]["mean"]
            fine_curve = fine[method]["mean"]
            fine_at_coarse = np.interp(coarse_curve[:, 0], fine_curve[:, 0], fine_curve[:, 1], left=np.nan, right=np.nan)
            valid = (coarse_curve[:, 0] <= reliable_k) & (coarse_curve[:, 1] > 0) & (fine_at_coarse > 0)
            relative = np.abs(coarse_curve[valid, 1] - fine_at_coarse[valid]) / fine_at_coarse[valid]
            median_difference = float(np.median(relative)) if relative.size else float("nan")
            rows.append({
                "case": case_name, "region": region, "stage": stage, "quantity": quantity, "method": method,
                "coarse_spacing": coarse_spacing, "fine_spacing": fine_spacing,
                "reliable_k_max": reliable_k, "median_relative_difference": median_difference,
                "converged_5pct": bool(median_difference <= 0.05) if np.isfinite(median_difference) else False,
            })
    return rows


def analyze_sample(scene_paths: dict[str, Path], region_lo: np.ndarray, region_hi: np.ndarray,
                   spacing_target: float, frames: dict[str, int]) -> tuple[dict[str, dict], dict]:
    """Reconstruct one aligned sample; all final-method fields share one mask."""
    particle_radius = load_particle_radius(scene_paths["dimcv"])
    for method in METHODS[1:]:
        if not np.isclose(load_particle_radius(scene_paths[method]), particle_radius):
            raise ValueError("All methods in one spectrum comparison must use the same particleRadius")
    shape, spacing = grid_for_spacing(region_lo, region_hi, spacing_target)
    frames_data = {}
    fields_by_method: dict[str, dict[str, np.ndarray]] = {}
    weights: dict[str, np.ndarray] = {}
    for method in METHODS:
        ply_dir, _, cfg = output_paths(scene_paths[method], method)
        ply_path = ply_dir / f"frame_{frames[method]:04d}.ply"
        names = ("velocity", "vorticity")
        if method == "dfsph_segment":
            names += ("pre_feedback_vel", "segment_bs_vel")
        frame = load_particle_frame(
            ply_path, region_lo, region_hi, names,
            ply_axis_convention=str(cfg.get("exportPLYAxisConvention", "y_up")),
        )
        frames_data[method] = frame
        fields = {}
        for name, values in frame.vectors.items():
            field, weight = reconstruct_sph_vector_grid(
                frame.positions, values, region_lo, region_hi, shape, particle_radius * 4.0
            )
            fields[name] = field
            if name == "velocity":
                weights[method] = weight
        fields_by_method[method] = fields

    velocity_fields = {method: fields_by_method[method]["velocity"] for method in METHODS}
    velocity_results, common_mask = analyze_reconstructed_fields(velocity_fields, weights, spacing)
    vorticity_fields = {method: fields_by_method[method]["vorticity"] for method in METHODS}
    vorticity_results, _ = analyze_reconstructed_fields(vorticity_fields, weights, spacing)

    # This uses the exact same windowed velocity field as the energy spectrum,
    # making it a diagnostic consistency check against raw SPH vorticity.
    velocity_curl_results = {}
    for method in METHODS:
        curl = vector_curl_spectral(velocity_results[method]["prepared"], spacing)
        radial, parseval = radial_shell_spectrum(curl, spacing)
        velocity_curl_results[method] = {
            "radial": radial, "parseval": parseval,
            "metrics": {"total_energy": parseval["total_energy"]},
        }
    result = {"velocity": velocity_results, "vorticity": vorticity_results,
              "velocity_curl": velocity_curl_results}
    # Hybrid diagnostics use the same particle coverage as the final Hybrid field.
    hybrid_fields = {
        "pre_feedback_vel": fields_by_method["dfsph_segment"]["pre_feedback_vel"],
        "segment_bs_vel": fields_by_method["dfsph_segment"]["segment_bs_vel"],
        "final_vel": fields_by_method["dfsph_segment"]["velocity"],
    }
    hybrid_weight_map = {name: weights["dfsph_segment"] for name in hybrid_fields}
    hybrid_results, _ = analyze_reconstructed_fields(hybrid_fields, hybrid_weight_map, spacing)
    curl = vector_curl_spectral(hybrid_results["segment_bs_vel"]["prepared"], spacing)
    radial, parseval = radial_shell_spectrum(curl, spacing)
    hybrid_results["segment_bs_curl"] = {
        "radial": radial, "parseval": parseval,
        "metrics": {"total_energy": parseval["total_energy"]},
    }
    result["hybrid"] = hybrid_results
    coverage = {method: float((weights[method] > 1e-12).mean()) for method in METHODS}
    metadata = {
        "shape": shape.tolist(), "spacing": spacing.tolist(), "support_radius": particle_radius * 4.0,
        "common_mask_fraction": float(common_mask.mean()), "method_coverage": coverage,
        "particles": {method: frames_data[method].stats for method in METHODS},
    }
    return result, metadata


def run_case(case_name: str, case: dict, output_root: Path, sample_interval: float,
             only_region: str | None = None) -> list[dict]:
    global METHODS
    configured_methods = tuple(case.get("methods", case["scenes"].keys()))
    if not configured_methods:
        raise ValueError(f"{case_name}: methods must not be empty")
    missing = set(configured_methods) - set(case["scenes"])
    if missing:
        raise ValueError(f"{case_name}: missing scene paths for methods {sorted(missing)}")
    METHODS = configured_methods
    scene_paths = {method: Path(path) for method, path in case["scenes"].items()}
    rows = {method: statistics_rows(output_paths(scene_paths[method], method)[1]) for method in METHODS}
    max_error = float(case.get("maxTimeError", 0.020))
    regions = {"domain": None, **case["regions"]}
    if only_region:
        regions = {only_region: regions[only_region]}
    summaries: list[dict] = []
    for region_name, region in regions.items():
        if region is None:
            _, _, cfg = output_paths(scene_paths["dimcv"], "dimcv")
            lo, hi = np.asarray(cfg["domainStart"], dtype=float), np.asarray(cfg["domainEnd"], dtype=float)
        else:
            lo, hi = np.asarray(region["start"], dtype=float), np.asarray(region["end"], dtype=float)
        for stage_name, intervals in case["stages"].items():
            # Keys include stage/region/grid spacing, so do not pre-seed this
            # mapping with plain quantity names.
            samples_by_quantity: dict[tuple[str, str, float], dict[str, list[np.ndarray]]] = {}
            sample_metadata = []
            for target in target_times(intervals, sample_interval):
                selected = {method: nearest_frame(rows[method], float(target)) for method in METHODS}
                errors = {method: abs(time - target) for method, (_, time) in selected.items()}
                if max(errors.values()) > max_error:
                    raise ValueError(f"{case_name}/{stage_name}: target t={target:.3f} has alignment error {errors}")
                frames = {method: frame for method, (frame, _) in selected.items()}
                for spacing_target in case["gridSpacings"]:
                    key = (stage_name, region_name, float(spacing_target))
                    result, metadata = analyze_sample(scene_paths, lo, hi, float(spacing_target), frames)
                    metadata.update({"target_time": float(target), "frames": frames,
                                     "actual_times": {method: time for method, (_, time) in selected.items()}})
                    sample_metadata.append((key, metadata))
                    for quantity in ("velocity", "vorticity", "velocity_curl"):
                        for method in METHODS:
                            samples_by_quantity.setdefault((key, quantity), defaultdict(list))[method].append(
                                result[quantity][method]["radial"]
                            )
                    for name, values in result["hybrid"].items():
                        samples_by_quantity.setdefault((key, "hybrid"), defaultdict(list))[name].append(values["radial"])

            metadata_by_key = defaultdict(list)
            for key, metadata in sample_metadata:
                metadata_by_key[key].append(metadata)
            stage_aggregates: dict[tuple[str, float], tuple[dict[str, dict], dict]] = {}
            for (key, quantity), values in samples_by_quantity.items():
                stage, region_label, spacing_target = key
                spectra = aggregate(values)
                common_fraction = float(np.mean([item["common_mask_fraction"] for item in metadata_by_key[key]]))
                reliable_k = float(0.8 * np.pi / max(metadata_by_key[key][0]["spacing"]))
                out_dir = output_root / case_name / region_label / f"dx_{spacing_target:g}" / stage / quantity
                write_stage_comparison(out_dir / "comparison.csv", spectra, quantity, {
                    "sample_count": len(metadata_by_key[key]), "common_mask_fraction_mean": common_fraction,
                    "samples": metadata_by_key[key], "reliable_k_max": reliable_k,
                })
                if quantity != "hybrid":
                    label = {
                        "velocity": "Velocity energy spectrum",
                        "vorticity": "Raw SPH vorticity spectrum",
                        "velocity_curl": "Vorticity spectrum from reconstructed velocity",
                    }[quantity]
                    save_comparison_plot(out_dir / "absolute.png", spectra, 1, label, reliable_k)
                    save_comparison_plot(out_dir / "normalized.png", spectra, 2, "Normalized shell energy", reliable_k)
                    save_comparison_plot(out_dir / "relative_to_dfsph.png", spectra, 1, "Spectrum / DFSPH", reliable_k, ratio=True)
                else:
                    save_comparison_plot(out_dir / "absolute.png", spectra, 1, "Hybrid diagnostic spectrum", reliable_k)
                for method, values_ in spectra.items():
                    mean = values_["mean"]
                    total = float(mean[:, 1].sum())
                    metrics = spectrum_metrics(mean, reliable_k)
                    baseline = spectra.get("dfsph", {}).get("mean")
                    log_distance = 0.0
                    if quantity != "hybrid" and method != "dfsph" and baseline is not None:
                        valid = (mean[:, 0] <= reliable_k) & (mean[:, 1] > 0) & (baseline[:, 1] > 0)
                        log_distance = float(np.sqrt(np.mean((np.log(mean[valid, 1]) - np.log(baseline[valid, 1])) ** 2))) if valid.any() else float("nan")
                    summaries.append({"case": case_name, "region": region_label, "stage": stage,
                                      "spacing_target": spacing_target, "quantity": quantity, "field": method,
                                      "total_energy": total, "high_k_fraction": float(mean[mean[:, 0] >= .6 * mean[:, 0].max(), 1].sum() / total) if total else 0.0,
                                      "spectral_centroid": float(np.sum(mean[:, 0] * mean[:, 1]) / total) if total else 0.0,
                                      "slope": metrics["slope"], "log_distance_to_dfsph": log_distance,
                                      "common_mask_fraction": common_fraction, "reliable_k_max": reliable_k})
                stage_aggregates[(quantity, spacing_target)] = (spectra, metadata_by_key[key][0])
            convergence = convergence_rows(stage_aggregates, case_name, region_name, stage_name)
            if convergence:
                convergence_path = output_root / case_name / region_name / stage_name / "grid_convergence.csv"
                write_summary(convergence_path, convergence)
                summaries.extend(convergence)
    return summaries


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 3D SPH vector spectra with common reconstruction masks")
    parser.add_argument("--manifest", type=Path, default=Path("spectrum_experiments_3d.json"))
    parser.add_argument("--case", action="append", help="Case key; repeat to run multiple cases")
    parser.add_argument("--region", help="Analyze only domain or one manifest region")
    parser.add_argument("--output-dir", type=Path, default=Path("results/spectrum3d"))
    parser.add_argument("--sample-interval", type=float, default=None, help="Override manifest sample interval in seconds")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    cases = args.case or list(manifest["cases"])
    unknown = set(cases) - set(manifest["cases"])
    if unknown:
        parser.error(f"Unknown cases: {sorted(unknown)}")
    all_rows = []
    interval = float(args.sample_interval or manifest.get("sampleInterval", 0.25))
    for name in cases:
        print(f"[spectrum] case={name}")
        all_rows.extend(run_case(name, manifest["cases"][name], args.output_dir, interval, args.region))
    write_summary(args.output_dir / "summary.csv", all_rows)
    print(f"[spectrum] summary={args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
