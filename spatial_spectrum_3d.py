"""Robust offline 3D velocity and vorticity spectra from SPH PLY snapshots.

The module intentionally does not fill empty cells. A caller comparing several
methods builds one common validity mask and FFTs only values on that mask.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

FLUID_PARTICLE_TYPES = (0, 1)
POSITION_COLUMNS = ("x", "y", "z")
VORTICITY_COLUMNS = ("vort_x", "vort_y", "vort_z")
VELOCITY_COLUMNS = ("vel_x", "vel_y", "vel_z")


@dataclass(frozen=True)
class ParticleFrame:
    positions: np.ndarray
    vectors: dict[str, np.ndarray]
    stats: dict


def read_ascii_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Read an ASCII vertex PLY keyed by property name."""
    path = Path(path)
    with path.open("rb") as stream:
        header: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
    if not header or header[0] != "ply" or "format ascii 1.0" not in header:
        raise ValueError(f"Only ASCII PLY is supported: {path}")
    vertex_line = next((line for line in header if line.startswith("element vertex ")), None)
    if vertex_line is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    count = int(vertex_line.split()[-1])
    properties = [line.split()[-1] for line in header if line.startswith("property ")]
    missing = [name for name in POSITION_COLUMNS if name not in properties]
    if missing:
        raise ValueError(f"PLY missing position properties {missing}: {path}")
    data = np.loadtxt(path, skiprows=len(header), max_rows=count, ndmin=2)
    if data.shape != (count, len(properties)):
        raise ValueError(f"PLY vertex count/property mismatch: {path}")
    return {name: data[:, index] for index, name in enumerate(properties)}


def vector_columns(name: str) -> tuple[str, str, str]:
    if name == "velocity":
        return VELOCITY_COLUMNS
    if name == "vorticity":
        return VORTICITY_COLUMNS
    return (f"{name}_x", f"{name}_y", f"{name}_z")


def load_domain(scene_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(scene_path).open("r", encoding="utf-8") as stream:
        cfg = json.load(stream).get("Configuration", {})
    lo = np.asarray(cfg.get("domainStart"), dtype=np.float64).reshape(-1)
    hi = np.asarray(cfg.get("domainEnd"), dtype=np.float64).reshape(-1)
    if lo.size != 3 or hi.size != 3 or np.any(hi <= lo):
        raise ValueError("Configuration.domainStart/domainEnd must define a 3D domain")
    return lo, hi


def load_particle_radius(scene_path: str | Path) -> float:
    with Path(scene_path).open("r", encoding="utf-8") as stream:
        radius = json.load(stream).get("Configuration", {}).get("particleRadius")
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Configuration.particleRadius must be positive")
    return radius


def load_particle_frame(
    ply_path: str | Path,
    lo: np.ndarray,
    hi: np.ndarray,
    vector_names: Iterable[str],
    particle_types: Iterable[int] = FLUID_PARTICLE_TYPES,
    ply_axis_convention: str = "y_up",
) -> ParticleFrame:
    """Load fluid particles plus requested vector-valued PLY properties."""
    data = read_ascii_ply(ply_path)
    pos = np.column_stack([data[name] for name in POSITION_COLUMNS]).astype(np.float64)
    requested = tuple(vector_names)
    missing = [column for name in requested for column in vector_columns(name) if column not in data]
    if missing:
        raise ValueError(f"{ply_path} lacks {missing}. Re-run after enabling the new PLY export.")
    vectors = {
        name: np.column_stack([data[column] for column in vector_columns(name)]).astype(np.float64)
        for name in requested
    }
    # PLY uses the DCC-facing convention configured by the scene. Restore the
    # simulation axes before applying domain/ROI coordinates from that scene.
    convention = (ply_axis_convention or "y_up").lower().strip()
    if convention in ("z_up", "houdini"):
        pos = pos[:, [0, 2, 1]]
        vectors = {name: values[:, [0, 2, 1]] for name, values in vectors.items()}
    elif convention not in ("y_up", "same", "sim", "", "xy", "2d", "plane_xy", "plane"):
        raise ValueError(f"Unsupported exportPLYAxisConvention {ply_axis_convention!r}")
    mask = np.all(np.isfinite(pos), axis=1) & np.all((pos >= lo) & (pos <= hi), axis=1)
    for values in vectors.values():
        mask &= np.all(np.isfinite(values), axis=1)
    if "particle_type" in data:
        mask &= np.isin(data["particle_type"].astype(np.int64), tuple(particle_types))
    if not mask.any():
        raise ValueError(f"No valid fluid particles remain in {ply_path}")
    return ParticleFrame(
        positions=pos[mask],
        vectors={name: values[mask] for name, values in vectors.items()},
        stats={"input_particles": int(len(pos)), "selected_particles": int(mask.sum()),
               "excluded_particles": int((~mask).sum()), "input": str(ply_path)},
    )


def grid_for_spacing(lo: np.ndarray, hi: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    if spacing <= 0:
        raise ValueError("grid spacing must be positive")
    shape = np.maximum(4, np.ceil((hi - lo) / spacing).astype(np.int64))
    return shape, (hi - lo) / shape


def cubic_kernel_3d(distance: np.ndarray, support_radius: float) -> np.ndarray:
    """Cubic kernel matching ParticleSystem's support-radius convention."""
    q = distance / support_radius
    result = np.zeros_like(q, dtype=np.float64)
    scale = 2.5465 / support_radius**3
    near = q <= 0.5
    far = (q > 0.5) & (q <= 1.0)
    result[near] = scale * (6.0 * q[near] ** 3 - 6.0 * q[near] ** 2 + 1.0)
    result[far] = scale * 2.0 * (1.0 - q[far]) ** 3
    return result


def reconstruct_sph_vector_grid(
    positions: np.ndarray, vectors: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    shape: tuple[int, int, int] | np.ndarray, support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-normalized particle-to-grid reconstruction without gap filling.

    Particles are first deposited at their nearest cell center. The numerator
    and denominator are then convolved with the discrete cubic SPH kernel
    using zero-padded FFTs. This is the same normalized kernel estimator as
    direct particle-to-cell summation, while avoiding O(N_particles * N_kernel)
    work for the fine comparison grid.
    """
    shape = np.asarray(shape, dtype=np.int64)
    spacing = (hi - lo) / shape
    index = np.floor((positions - lo) / spacing).astype(np.int64)
    inside = np.all((index >= 0) & (index < shape), axis=1)
    index, vectors = index[inside], vectors[inside]
    impulses = np.zeros(tuple(shape) + (3,), dtype=np.float64)
    count = np.zeros(tuple(shape), dtype=np.float64)
    address = tuple(index.T)
    np.add.at(count, address, 1.0)
    for component in range(3):
        np.add.at(impulses[..., component], address, vectors[:, component])

    radius = np.ceil(support_radius / spacing).astype(np.int64)
    kernel_index = np.stack(np.meshgrid(
        np.arange(-radius[0], radius[0] + 1),
        np.arange(-radius[1], radius[1] + 1),
        np.arange(-radius[2], radius[2] + 1), indexing="ij"), axis=-1)
    kernel = cubic_kernel_3d(np.linalg.norm(kernel_index * spacing, axis=-1), support_radius)
    kernel *= float(np.prod(spacing))
    fft_shape = tuple((shape + 2 * radius).tolist())
    # The centered local kernel must be moved to FFT origin. Zero padding
    # avoids circular wrap at the physical reconstruction-domain boundary.
    kernel_padded = np.zeros(fft_shape, dtype=np.float64)
    kernel_padded[:kernel.shape[0], :kernel.shape[1], :kernel.shape[2]] = np.fft.ifftshift(kernel)
    kernel_fft = np.fft.rfftn(kernel_padded, axes=(0, 1, 2))

    def convolve_same(values: np.ndarray) -> np.ndarray:
        convolved = np.fft.irfftn(
            np.fft.rfftn(values, fft_shape, axes=(0, 1, 2)) * kernel_fft,
            fft_shape,
            axes=(0, 1, 2),
        )
        return convolved[:shape[0], :shape[1], :shape[2]]

    weight = convolve_same(count)
    field = np.empty_like(impulses)
    for component in range(3):
        field[..., component] = convolve_same(impulses[..., component])
    np.divide(field, weight[..., None], out=field, where=weight[..., None] > 1e-12)
    field[weight <= 1e-12] = 0.0
    return field, weight


def common_valid_mask(weights: Iterable[np.ndarray], minimum_weight: float = 1e-12) -> np.ndarray:
    weights = tuple(weights)
    if not weights:
        raise ValueError("At least one weight grid is required")
    mask = np.ones_like(weights[0], dtype=bool)
    for weight in weights:
        if weight.shape != mask.shape:
            raise ValueError("All reconstructed grids must have the same shape")
        mask &= weight > minimum_weight
    if not mask.any():
        raise ValueError("The common reconstruction mask is empty")
    return mask


def tukey_window_1d(size: int, alpha: float = 0.25) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("Tukey alpha must be in [0, 1]")
    if size < 2 or alpha == 0:
        return np.ones(size, dtype=np.float64)
    coordinate = np.linspace(0.0, 1.0, size)
    if alpha == 1:
        return np.hanning(size)
    window = np.ones(size, dtype=np.float64)
    left, right = coordinate < alpha / 2.0, coordinate > 1.0 - alpha / 2.0
    window[left] = 0.5 * (1 + np.cos(np.pi * (2 * coordinate[left] / alpha - 1)))
    window[right] = 0.5 * (1 + np.cos(np.pi * (2 * coordinate[right] / alpha - 2 / alpha + 1)))
    return window


def prepare_masked_field(field: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Remove valid-cell mean and apply an energy-normalized 3D Tukey window."""
    if field.shape[:3] != mask.shape or field.shape[-1] != 3:
        raise ValueError("field must be (nx, ny, nz, 3) and match mask")
    centered = np.zeros_like(field, dtype=np.float64)
    centered[mask] = field[mask] - field[mask].mean(axis=0)
    window = (tukey_window_1d(field.shape[0], alpha)[:, None, None]
              * tukey_window_1d(field.shape[1], alpha)[None, :, None]
              * tukey_window_1d(field.shape[2], alpha)[None, None, :]) * mask
    norm = float(np.sum(window * window))
    if norm <= 0:
        raise ValueError("No valid window samples")
    window *= np.sqrt(float(mask.sum()) / norm)
    return centered * window[..., None], window


def wave_axes(shape: tuple[int, int, int], spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(2.0 * np.pi * np.fft.fftfreq(size, d=float(dx)) for size, dx in zip(shape, spacing))


def vector_curl_spectral(field: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    kx, ky, kz = np.meshgrid(*wave_axes(field.shape[:3], spacing), indexing="ij")
    transform = np.fft.fftn(field, axes=(0, 1, 2))
    curl_hat = np.empty_like(transform)
    curl_hat[..., 0] = 1j * (ky * transform[..., 2] - kz * transform[..., 1])
    curl_hat[..., 1] = 1j * (kz * transform[..., 0] - kx * transform[..., 2])
    curl_hat[..., 2] = 1j * (kx * transform[..., 1] - ky * transform[..., 0])
    return np.fft.ifftn(curl_hat, axes=(0, 1, 2)).real


def radial_shell_spectrum(field: np.ndarray, spacing: np.ndarray, bins: int | None = None) -> tuple[np.ndarray, dict]:
    """Return a physically normalized, spherical-shell integrated vector spectrum."""
    shape = field.shape[:3]
    transform = np.fft.fftn(field, axes=(0, 1, 2))
    kx, ky, kz = np.meshgrid(*wave_axes(shape, spacing), indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky + kz * kz).ravel()
    power = np.sum(np.abs(transform) ** 2, axis=-1).ravel()
    bins = bins or min(shape) // 2
    nyquist = np.pi / spacing
    edges = np.linspace(0.0, float(np.linalg.norm(nyquist)), bins + 1)
    index = np.clip(np.digitize(radius, edges, right=False) - 1, 0, bins - 1)
    shell_power = np.bincount(index, weights=power, minlength=bins)
    shell_count = np.bincount(index, minlength=bins)
    cell_volume = float(np.prod(spacing))
    shell_energy = 0.5 * cell_volume / np.prod(shape) * shell_power
    total = float(shell_energy.sum())
    normalized = shell_energy / total if total > 0 else np.zeros_like(shell_energy)
    radial = np.column_stack((0.5 * (edges[:-1] + edges[1:]), shell_energy, normalized,
                              shell_count, np.cumsum(shell_energy)))
    return radial, {"total_energy": total,
                    "physical_space_energy": float(0.5 * cell_volume * np.sum(field * field)),
                    "nyquist_min": float(nyquist.min()), "reliable_k_max": float(0.8 * nyquist.min())}


def spectrum_metrics(radial: np.ndarray, reliable_k_max: float) -> dict:
    k, energy = radial[:, 0], radial[:, 1]
    total = float(energy.sum())
    if total <= 0:
        return {"total_energy": 0.0, "high_k_fraction": 0.0, "spectral_centroid": 0.0, "slope": float("nan")}
    fit = (k >= 0.25 * reliable_k_max) & (k <= reliable_k_max) & (energy > 0)
    return {"total_energy": total, "high_k_fraction": float(energy[k >= 0.6 * k.max()].sum() / total),
            "spectral_centroid": float(np.sum(k * energy) / total),
            "slope": float(np.polyfit(np.log(k[fit]), np.log(energy[fit]), 1)[0]) if fit.sum() >= 2 else float("nan")}


def analyze_reconstructed_fields(fields: dict[str, np.ndarray], weights: dict[str, np.ndarray], spacing: np.ndarray,
                                alpha: float = 0.25, bins: int | None = None) -> tuple[dict[str, dict], np.ndarray]:
    """Analyze named vector fields with one common occupied-cell mask."""
    mask = common_valid_mask(weights.values())
    result: dict[str, dict] = {}
    for name, field in fields.items():
        prepared, window = prepare_masked_field(field, mask, alpha)
        radial, parseval = radial_shell_spectrum(prepared, spacing, bins)
        result[name] = {"radial": radial, "prepared": prepared, "window": window, "parseval": parseval,
                        "metrics": spectrum_metrics(radial, parseval["reliable_k_max"])}
    return result, mask


def write_spectrum(path: str | Path, radial: np.ndarray, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, radial, delimiter=",",
               header="k,shell_energy,normalized_energy,shell_count,cumulative_energy", comments="")
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
