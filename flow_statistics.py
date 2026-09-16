"""统一记录 SPH 流场的时间序列统计量。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class FlowStatisticsRecorder:
    """在导出帧上记录可跨方法比较的涡量和动能统计。"""

    fieldnames = [
        "frame",
        "time",
        "fluid_count",
        "vorticity_field",
        "vorticity_threshold",
        "vortex_particle_count",
        "vortex_particle_fraction",
        "vorticity_mean_signed_z",
        "vorticity_mean_abs_z",
        "vorticity_mean_magnitude",
        "vorticity_rms_magnitude",
        "kinetic_energy_total",
        "kinetic_energy_mean_per_particle",
        "vortex_kinetic_energy_total",
        "vortex_kinetic_energy_mean_per_particle",
    ]

    def __init__(self, output_dir: Path, config, method_name: str):
        enabled = config.get_cfg("statisticsEnabled")
        self.enabled = bool(enabled) if enabled is not None else True
        threshold = config.get_cfg("statisticsVorticityThreshold")
        self.threshold = float(threshold) if threshold is not None else 0.0
        self.field_name = str(config.get_cfg("statisticsVorticityField") or "raw").lower()
        filename = str(config.get_cfg("statisticsOutputName") or "flow_statistics.csv")
        self.path = output_dir / filename
        self.method_name = method_name
        self._file = None
        self._writer = None

        if self.enabled:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._file.flush()
            print(f"[{method_name}] statistics output: {self.path}")

    def record(self, solver, frame: int, simulation_time: float):
        if not self.enabled:
            return

        # 统一在导出时从当前速度场重算，避免 DIMCV、DFSPH、Hybrid 使用不同时间点的涡量。
        solver.compute_vorticity()
        solver.compute_vorticity_vis()

        n = int(solver.ps.particle_num[None])
        material = solver.ps.material.to_numpy()[:n]
        active = solver.ps.is_active.to_numpy()[:n].astype(bool)
        fluid = (material == int(solver.ps.material_fluid)) & active
        fluid_count = int(np.count_nonzero(fluid))

        velocity = solver.ps.v.to_numpy()[:n, : int(solver.ps.dim)].astype(np.float64, copy=False)
        mass = solver.ps.m.to_numpy()[:n].astype(np.float64, copy=False)
        omega_field = solver.ps.vorticity_vis if self.field_name in ("vis", "smoothed", "vorticity_vis") else solver.ps.vorticity
        omega = omega_field.to_numpy()[:n].astype(np.float64, copy=False)

        v = velocity[fluid]
        m = mass[fluid]
        w = omega[fluid]
        omega_magnitude = np.linalg.norm(w, axis=1)
        selected = omega_magnitude >= self.threshold
        ws = w[selected]
        magnitudes = omega_magnitude[selected]
        particle_ke = 0.5 * m * np.einsum("ij,ij->i", v, v)
        vortex_particle_ke = particle_ke[selected]

        row = {
            "frame": int(frame),
            "time": float(simulation_time),
            "fluid_count": fluid_count,
            "vorticity_field": "vorticity_vis" if omega_field is solver.ps.vorticity_vis else "vorticity",
            "vorticity_threshold": self.threshold,
            "vortex_particle_count": int(np.count_nonzero(selected)),
            "vortex_particle_fraction": float(np.mean(selected)) if fluid_count else 0.0,
            "vorticity_mean_signed_z": float(np.mean(ws[:, 2])) if ws.size else 0.0,
            "vorticity_mean_abs_z": float(np.mean(np.abs(ws[:, 2]))) if ws.size else 0.0,
            "vorticity_mean_magnitude": float(np.mean(magnitudes)) if magnitudes.size else 0.0,
            "vorticity_rms_magnitude": float(np.sqrt(np.mean(magnitudes ** 2))) if magnitudes.size else 0.0,
            "kinetic_energy_total": float(np.sum(particle_ke)),
            "kinetic_energy_mean_per_particle": float(np.mean(particle_ke)) if particle_ke.size else 0.0,
            "vortex_kinetic_energy_total": float(np.sum(vortex_particle_ke)),
            "vortex_kinetic_energy_mean_per_particle": float(np.mean(vortex_particle_ke)) if vortex_particle_ke.size else 0.0,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
