"""
DFSPH Kármán solver coupled with Lagrangian vortex segments (research prototype).

Pipeline each ``substep`` (same ``dt`` as SPH):
  1) Standard DFSPH through ``pressure_solve``.
  2) ``compute_vorticity`` — SPH estimate ω from velocity (same formulation as DIMCV).
  3) Segment subsystem: optional inlet emit → optional SPH ω deposit → BS → RK4 → topology.
  4) Biot–Savart induced velocity at fluid particle positions → add ``β u_BS`` to ``ps.v``.

  Experiment ``sphToSegmentDepositEnabled: false`` + ``emitterEnabled: true`` (2D
  ``emitterOrientation: streamwise_x``): segments inject at the fluid inlet like particles,
  evolve with BS/background only, and drive SPH via β coupling (no ω→γ deposit).

Coupling parameters live under ``SegmentConfiguration`` in the scene JSON:

  - ``sphSegmentBsCoupling`` (default 0.12): β, BS → SPH velocity correction strength.
  - ``sphToSegmentGammaBlend`` (default 0.06): convex blend toward γ target each step.
  - ``sphToSegmentGammaScale``: scales target γ after projection (tune with BS strength).
  - ``sphToSegmentGammaProjection``: ``tangential``, ``norm``, ``norm_signed`` (see note above).
  - ``sphToSegmentDepositSource``: ``vorticity_vis`` (default) uses the same smoothed field as
    rendering; ``raw`` uses ``ps.vorticity`` (unnormalized SPH curl sum, often much smaller in
    magnitude than ``vorticity_vis``). Weights use ``m_V * cubic_kernel`` like ``compute_vorticity_vis``.

Notes:
  • Correction after ``pressure_solve`` is not divergence-free; treat as vorticity-style boost.
  • Segment advection: BS + ``segmentAdvectionBackgroundVelocity`` (RK4 端点速度);
    ``backgroundVelocity`` 用于边界 RHS 等，默认 **不** 叠加到 SPH 的 BS 耦合。
  • For ``initType: random_uniform`` with ``randomGammaInitial: 0``, set ``deleteGammaThreshold: 0``
    so weak segments are not culled before SPH deposits circulation.
  • Requires ``SimConfig.scene_file_path`` and a ``SegmentConfiguration`` block in that JSON.
  • Use ``simulationMethod``: 2 with e.g. ``DIM_von_karman_vortex_dfsph_segment.json``.
  • PNG: ``<image_path>/sph/vorticity_XXXX.png`` (fluid + cylinder, no segments) and
    ``<image_path>/segments/vorticity_XXXX.png`` (interior segments by ``vorticity_vis.z``;
    boundary virtual segments in ``imageBoundarySegmentColor`` if ``imageShowBoundarySegments``;
    optional cylinder via ``imageSegmentPanelIncludeSolid``).
  • When ``exportPLY`` runs for particles, ``exportSegmentPLY`` (SegmentConfiguration, default true)
    also writes ``segments_XXXX.ply`` beside ``frame_XXXX.ply``.
    Particle positions/vorticity use ``Configuration.exportPLYAxisConvention``; segment vertices use
    ``SegmentConfiguration.exportPLYAxisConvention`` (same string meanings as in ``segment_export``).
"""

import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from matplotlib.colors import Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dfsph_karman_solver import DFSPHKarmanVortexSolver
from segment_config import SegmentConfig
from segment_system import SegmentSystem
from segment_solver import SegmentSolver
from segment_export import SegmentExporter


class DFSPHSegmentHybridKarmanSolver(DFSPHKarmanVortexSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        path = getattr(self.ps.cfg, "scene_file_path", None)
        if not path:
            raise ValueError(
                "DFSPHSegmentHybridKarmanSolver requires SimConfig.scene_file_path "
                "(scene JSON path used to load SegmentConfiguration)."
            )
        self.seg_cfg = SegmentConfig(scene_file_path=path)
        self.ss_seg = SegmentSystem(self.seg_cfg)
        self.seg_solver = SegmentSolver(self.ss_seg)
        self._initial_sph_advect_steps = max(
            1,
            int(self.seg_cfg.get_cfg("segmentInitialSphVelocitySteps", 1)),
        )
        self._initial_sph_advect_decay = float(
            self.seg_cfg.get_cfg("segmentInitialSphVelocityDecay", 1.0)
        )
        if self.seg_solver.has_boundary:
            self.seg_solver.boundary.set_particle_system(particle_system)

        self.segment_bs_coupling_beta = float(
            self.seg_cfg.get_cfg("sphSegmentBsCoupling", 0.12)
        )
        self.sph_to_segment_blend = float(
            self.seg_cfg.get_cfg("sphToSegmentGammaBlend", 0.06)
        )
        _dep_en = self.seg_cfg.get_cfg("sphToSegmentDepositEnabled", None)
        if _dep_en is None:
            self._sph_to_segment_deposit_enabled = self.sph_to_segment_blend > 0.0
        else:
            self._sph_to_segment_deposit_enabled = bool(_dep_en)
        self.sph_to_segment_gamma_scale = float(
            self.seg_cfg.get_cfg("sphToSegmentGammaScale", 5e-4)
        )
        self._deposit_skip_seg_type = int(
            self.seg_cfg.get_cfg("boundarySegmentTypeId", 2)
        ) if bool(self.seg_cfg.get_cfg("sphToSegmentDepositSkipBoundary", True)) else -1
        _proj = str(
            self.seg_cfg.get_cfg("sphToSegmentGammaProjection", "tangential")
            or "tangential"
        ).lower().strip()
        if _proj in ("norm", "magnitude", "omega_norm"):
            self._deposit_gamma_proj = 1
        elif _proj in ("norm_signed", "signed_norm"):
            self._deposit_gamma_proj = 2
        else:
            self._deposit_gamma_proj = 0

        _depsrc = str(
            self.seg_cfg.get_cfg("sphToSegmentDepositSource", "vorticity_vis")
            or "vorticity_vis"
        ).lower().strip()
        self._deposit_src_vis = 0 if _depsrc in ("raw", "vorticity") else 1

        self._sph_velocity_to_segment_enabled = bool(
            self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionEnabled", False)
        )
        self._segment_advection_mode = str(
            self.seg_cfg.get_cfg("segmentAdvectionMode", "") or ""
        ).lower().strip()
        self._segment_initial_sph_then_bs = self._segment_advection_mode in (
            "initial_sph_velocity_then_bs",
            "sph_initial_then_bs",
            "initial_sph_then_bs",
            "bs_with_initial_sph_velocity",
        )
        self._sph_velocity_to_segment_scale = float(
            self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionScale", 1.0)
        )
        self._sph_velocity_to_segment_blend = float(
            self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionBlend", 1.0)
        )
        self._sph_velocity_to_segment_skip_type = int(
            self.seg_cfg.get_cfg("boundarySegmentTypeId", 2)
        ) if bool(self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionSkipBoundary", True)) else -1
        self._bs_feedback_skip_boundary = bool(
            self.seg_cfg.get_cfg("sphSegmentBsFeedbackSkipBoundary", False)
        )
        self._bs_feedback_skip_type = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))

        self.u_seg_bs = ti.Vector.field(
            3, dtype=float, shape=self.ps.particle_max_num
        )
        self.u_vortex_ghost_sph = ti.Vector.field(
            self.ps.dim, dtype=float, shape=self.ps.particle_max_num
        )
        self._feedback_mode = str(
            self.seg_cfg.get_cfg(
                "sphSegmentFeedbackMode",
                "vortex_ghost_velocity" if self.seg_solver._bs_2d_point else "bs_direct",
            )
            or "bs_direct"
        ).lower().strip()
        self._vortex_ghost_velocity_feedback = self._feedback_mode in (
            "vortex_ghost_velocity",
            "ghost_velocity",
            "velocity_ghost",
            "ghost",
        )
        self._vortex_ghost_displacement_feedback = self._feedback_mode in (
            "vortex_ghost_displacement",
            "ghost_displacement",
            "displacement_ghost",
            "ghost_position",
        )
        self._vortex_ghost_feedback = (
            self._vortex_ghost_velocity_feedback
            or self._vortex_ghost_displacement_feedback
        )
        self._vortex_ghost_beta = float(
            self.seg_cfg.get_cfg("vortexGhostVelocityCouplingBeta", self.segment_bs_coupling_beta)
        )
        self._vortex_ghost_support_radius_scale = float(
            self.seg_cfg.get_cfg("vortexGhostVelocityCouplingSupportRadiusScale", 1.0)
        )
        self._vortex_ghost_grid_capacity = int(
            self.seg_cfg.get_cfg("vortexGhostGridCellCapacity", 64)
        )
        self._vortex_ghost_skip_boundary = bool(
            self.seg_cfg.get_cfg("vortexGhostVelocityCouplingSkipBoundary", self._bs_feedback_skip_boundary)
        )
        self._vortex_ghost_skip_type = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
        _life_mode = str(
            self.seg_cfg.get_cfg(
                "pointVortexLifetimeMode",
                "one_step_ghost_impulse" if self._vortex_ghost_feedback and self.seg_solver._bs_2d_point else "persistent",
            )
            or "persistent"
        ).lower().strip()
        self._one_step_vortex_impulse = self._vortex_ghost_feedback and _life_mode in (
            "one_step",
            "one_step_ghost",
            "one_step_ghost_impulse",
            "temporary",
            "transient",
            "impulse",
        )
        self.vortex_ghost_num = ti.field(dtype=ti.i32, shape=())
        self.vortex_ghost_x = ti.Vector.field(
            self.ps.dim, dtype=float, shape=self.ss_seg.segment_max_num
        )
        self.vortex_ghost_v = ti.Vector.field(
            self.ps.dim, dtype=float, shape=self.ss_seg.segment_max_num
        )
        self.vortex_ghost_mV = ti.field(dtype=float, shape=self.ss_seg.segment_max_num)
        self.vortex_ghost_gamma = ti.field(dtype=float, shape=self.ss_seg.segment_max_num)
        self.vortex_ghost_grid_count = ti.field(dtype=ti.i32, shape=self.ps.flattened_grid_num)
        self.vortex_ghost_grid_indices = ti.field(
            dtype=ti.i32,
            shape=(self.ps.flattened_grid_num, self._vortex_ghost_grid_capacity),
        )

        self._export_seg_line_width = float(
            self.seg_cfg.get_cfg("imageSegmentLineWidth", 0.85)
        )

        self._segment_exporter = SegmentExporter(self.ss_seg)
        self._export_segment_ply = bool(self.seg_cfg.get_cfg("exportSegmentPLY", True))

        # [DEBUG 沉积] SegmentConfiguration.sphToSegmentDepositDebug=true 时，在沉积 kernel 内统计全局最大值
        self._deposit_debug = bool(
            self.seg_cfg.get_cfg("sphToSegmentDepositDebug", False)
        )
        self._deposit_debug_interval = max(
            1, int(self.seg_cfg.get_cfg("sphToSegmentDepositDebugInterval", 50))
        )
        self._deposit_debug_step = 0
        self._dep_dbg_max_w = ti.field(dtype=ti.f32, shape=())
        self._dep_dbg_max_wsum = ti.field(dtype=ti.f32, shape=())
        self._dep_dbg_max_omega_norm = ti.field(dtype=ti.f32, shape=())
        self._dep_dbg_max_om_eff = ti.field(dtype=ti.f32, shape=())
        self._dep_dbg_max_g_tgt = ti.field(dtype=ti.f32, shape=())
        self._dep_dbg_max_gamma = ti.field(dtype=ti.f32, shape=())

        self._seg_vis_scalar = ti.field(
            dtype=float, shape=int(self.ss_seg.segment_max_num)
        )
        self._export_sph_subdir = str(
            self.seg_cfg.get_cfg("imageExportSubfolderSph", "sph") or "sph"
        ).strip() or "sph"
        self._export_seg_subdir = str(
            self.seg_cfg.get_cfg("imageExportSubfolderSegments", "segments")
            or "segments"
        ).strip() or "segments"
        self._export_ghost_subdir = str(
            self.seg_cfg.get_cfg("imageExportSubfolderGhost", "ghost") or "ghost"
        ).strip() or "ghost"
        self._export_ghost_panel = bool(
            self.seg_cfg.get_cfg("imageExportVortexGhostPanel", True)
        )
        self._img_vort_vmin = float(self.seg_cfg.get_cfg("imageVorticityVmin", -40.0))
        self._img_vort_vmax = float(self.seg_cfg.get_cfg("imageVorticityVmax", 40.0))
        self._export_segment_panel = bool(
            self.seg_cfg.get_cfg("imageExportSegmentPanel", True)
        )
        self._segment_panel_include_solid = bool(
            self.seg_cfg.get_cfg("imageSegmentPanelIncludeSolid", True)
        )
        self._segment_cmap = str(
            self.seg_cfg.get_cfg("imageSegmentColormap", "coolwarm") or "coolwarm"
        )
        self._sph_vort_debug = bool(self.seg_cfg.get_cfg("sphVorticityDebugPrint", False))
        self._sph_vort_debug_interval = max(
            1, int(self.seg_cfg.get_cfg("sphVorticityDebugPrintInterval", 10))
        )
        self._debug_last_residual_particle = -1
        self._substep_timing_enabled = bool(
            self.seg_cfg.get_cfg("debugHybridSubstepTiming", False)
        )
        self._substep_timing_interval = max(
            1, int(self.seg_cfg.get_cfg("debugHybridSubstepTimingInterval", 10))
        )
        self._substep_timing_sync = bool(
            self.seg_cfg.get_cfg("debugHybridSubstepTimingSync", True)
        )
        self._segment_timing_enabled = bool(
            self.seg_cfg.get_cfg("debugSegmentAdvanceTiming", False)
        )
        self._segment_timing_interval = max(
            1, int(self.seg_cfg.get_cfg("debugSegmentAdvanceTimingInterval", self._substep_timing_interval))
        )

    def initialize(self):
        super().initialize()
        self.seg_solver.initialize()
        if bool(self.seg_cfg.get_cfg("emitterSyncFluidEmitInterval", False)):
            self.seg_solver._emitter_interval_override = max(
                1, int(self.emit_interval)
            )
        # 若 initialize_only 在 seg_solver.initialize 内未成功提交，下一步再试
        ssol = self.seg_solver
        if (
            ssol.has_boundary
            and int(self.ss_seg.segment_num[None]) == 0
            and not ssol._boundary_one_shot_done
        ):
            ssol._run_boundary_injection_pipeline(strip_committed_first=False)
            ssol.boundary.mark_one_shot_complete_if_applicable(ssol)
            self.ss_seg.update_segment_geometry()

    def _cubic_kernel_np(self, r_norm, support_radius: float):
        h = float(support_radius)
        if h <= 1e-12:
            return np.zeros_like(np.asarray(r_norm, dtype=np.float64))
        q = np.asarray(r_norm, dtype=np.float64) / h
        if int(self.ps.dim) == 2:
            k = 1.8189 / (h * h)
        elif int(self.ps.dim) == 3:
            k = 2.5465 / (h * h * h)
        else:
            k = 1.3333 / h
        w = np.zeros_like(q, dtype=np.float64)
        m0 = q <= 0.5
        m1 = (q > 0.5) & (q <= 1.0)
        w[m0] = k * (6.0 * q[m0] ** 3 - 6.0 * q[m0] ** 2 + 1.0)
        w[m1] = k * 2.0 * (1.0 - q[m1]) ** 3
        return w

    def _cubic_kernel_grad_np(self, r_vec, support_radius: float):
        h = float(support_radius)
        rv = np.asarray(r_vec, dtype=np.float64)
        if h <= 1e-12 or rv.size == 0:
            return np.zeros_like(rv, dtype=np.float64)
        rn = np.linalg.norm(rv, axis=1)
        q = rn / h
        if int(self.ps.dim) == 2:
            k = 1.8189 / (h * h)
        elif int(self.ps.dim) == 3:
            k = 2.5465 / (h * h * h)
        else:
            k = 1.3333 / h
        dwdq = np.zeros_like(q, dtype=np.float64)
        m0 = q <= 0.5
        m1 = (q > 0.5) & (q <= 1.0)
        dwdq[m0] = k * (18.0 * q[m0] ** 2 - 12.0 * q[m0])
        dwdq[m1] = k * (-6.0 * (1.0 - q[m1]) ** 2)
        dwdr = dwdq / h
        grad = np.zeros_like(rv, dtype=np.float64)
        nz = rn > 1e-12
        grad[nz] = rv[nz] * (dwdr[nz] / rn[nz])[:, None]
        return grad

    def _active_point_vortex_arrays(self):
        nseg = int(self.ss_seg.segment_num[None])
        dim = int(self.ps.dim)
        if nseg <= 0:
            empty_x = np.zeros((0, dim), dtype=np.float32)
            empty_1 = np.zeros((0,), dtype=np.float32)
            return empty_x, empty_1, empty_1, empty_x
        active = self.ss_seg.active.to_numpy()[:nseg] == 1
        seg_type = self.ss_seg.seg_type.to_numpy()[:nseg]
        skip_type = int(self._vortex_ghost_skip_type if self._vortex_ghost_skip_boundary else -999999)
        if self._vortex_ghost_skip_boundary:
            active = active & (seg_type != skip_type)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            empty_x = np.zeros((0, dim), dtype=np.float32)
            empty_1 = np.zeros((0,), dtype=np.float32)
            return empty_x, empty_1, empty_1, empty_x
        xm = self.ss_seg.x_minus.to_numpy()[:nseg, :dim]
        xp = self.ss_seg.x_plus.to_numpy()[:nseg, :dim]
        centers = (0.5 * (xm[idx] + xp[idx])).astype(np.float32)
        gamma = self.ss_seg.gamma.to_numpy()[:nseg][idx].astype(np.float32)
        volume = self.seg_solver.point_vortex_volume.to_numpy()[:nseg][idx].astype(np.float32)
        vm = self.seg_solver.v_minus.to_numpy()[:nseg, :dim]
        vp = self.seg_solver.v_plus.to_numpy()[:nseg, :dim]
        vel = (0.5 * (vm[idx] + vp[idx])).astype(np.float32)
        return centers, gamma, volume, vel

    def _estimate_vortex_omega_from_gamma_at_points(self, points, support_radius: float):
        pts = np.asarray(points, dtype=np.float32)
        out = np.zeros((pts.shape[0], 3), dtype=np.float32)
        centers, gamma, volume, _ = self._active_point_vortex_arrays()
        if pts.size == 0 or centers.shape[0] == 0:
            return out
        valid = np.isfinite(gamma) & np.isfinite(volume) & (volume > 1e-12)
        if not np.any(valid):
            return out
        centers = centers[valid]
        gamma = gamma[valid].astype(np.float64)
        volume = volume[valid].astype(np.float64)
        for i, p in enumerate(pts[:, : centers.shape[1]]):
            r = p[None, :] - centers
            rn = np.linalg.norm(r, axis=1)
            near = rn < float(support_radius)
            if not np.any(near):
                continue
            w = self._cubic_kernel_np(rn[near], support_radius)
            wsum = float(np.sum(volume[near] * w))
            if wsum > 1e-12:
                out[i, 2] = float(np.sum(w * gamma[near]) / wsum)
        return out

    def _estimate_vortex_omega_from_velocity_curl_at_points(self, points, support_radius: float):
        pts = np.asarray(points, dtype=np.float32)
        out = np.zeros((pts.shape[0], 3), dtype=np.float32)
        if int(self.ps.dim) != 2:
            return out
        centers, _, volume, vel = self._active_point_vortex_arrays()
        if pts.size == 0 or centers.shape[0] == 0:
            return out
        valid = np.isfinite(volume) & (volume > 1e-12)
        if not np.any(valid):
            return out
        centers = centers[valid].astype(np.float64)
        volume = volume[valid].astype(np.float64)
        vel = vel[valid].astype(np.float64)
        for i, p in enumerate(pts[:, :2].astype(np.float64)):
            r = p[None, :] - centers
            rn = np.linalg.norm(r, axis=1)
            near = rn < float(support_radius)
            if not np.any(near):
                continue
            w = self._cubic_kernel_np(rn[near], support_radius)
            wsum = float(np.sum(volume[near] * w))
            if wsum <= 1e-12:
                continue
            ui = np.sum((volume[near] * w)[:, None] * vel[near], axis=0) / wsum
            grad = self._cubic_kernel_grad_np(r[near], support_radius)
            dv = vel[near] - ui[None, :]
            curl_z = np.sum(volume[near] * (dv[:, 1] * grad[:, 0] - dv[:, 0] * grad[:, 1]))
            out[i, 2] = float(curl_z)
        return out

    def _inject_segments_from_sph_boundary_vorticity(self):
        if not bool(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionEnabled", False)):
            return
        step = int(self.seg_solver._sim_step_index)
        interval = max(1, int(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionIntervalSteps", 10)))
        start = int(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionStartStep", 0))
        if step < start or (step - start) % interval != 0:
            return
        if self.ps.dim not in (2, 3):
            return
        if int(self.ss_seg.dim) != int(self.ps.dim):
            return

        n = int(self.ps.particle_num[None])
        if n <= 0:
            return
        x = self.ps.x.to_numpy()[:n].astype(np.float32, copy=False)
        mat = self.ps.material.to_numpy()[:n]
        fluid_mask = mat == int(self.ps.material_fluid)
        if not np.any(fluid_mask):
            return

        vort_src = str(
            self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSource", "vorticity")
            or "vorticity"
        ).lower().strip()
        if vort_src in ("vorticity_loss", "loss", "d_vort", "vorticity_loss_pred", "loss_pred", "vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed"):
            self.compute_v_star()
            self.compute_vorticity()
            self.compute_grad_v()
            if vort_src in ("vorticity_loss_pred", "loss_pred"):
                self.compute_vorticity_pred()
                self.compute_d_vorticity_pred()
            else:
                self.compute_d_vorticity()
            if vort_src in ("vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed"):
                self.compute_all_d_vort_smoothed()
        if vort_src in ("vis", "vorticity_vis", "smoothed", "smooth"):
            vort = self.ps.vorticity_vis.to_numpy()[:n].astype(np.float32, copy=False)
        elif vort_src in ("vorticity_loss", "loss", "d_vort", "vorticity_loss_pred", "loss_pred"):
            vort = self.d_vort.to_numpy()[:n].astype(np.float32, copy=False)
        elif vort_src in ("vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed"):
            vort = self.d_vort_smoothed.to_numpy()[:n].astype(np.float32, copy=False)
        else:
            vort = self.ps.vorticity.to_numpy()[:n].astype(np.float32, copy=False)

        threshold_src = str(
            self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionThresholdSource", vort_src)
            or vort_src
        ).lower().strip()
        if threshold_src in ("vorticity_loss", "loss", "d_vort", "vorticity_loss_pred", "loss_pred", "vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed") and not vort_src in ("vorticity_loss", "loss", "d_vort", "vorticity_loss_pred", "loss_pred", "vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed"):
            self.compute_v_star()
            self.compute_vorticity()
            self.compute_grad_v()
            if threshold_src in ("vorticity_loss_pred", "loss_pred"):
                self.compute_vorticity_pred()
                self.compute_d_vorticity_pred()
            else:
                self.compute_d_vorticity()
            if threshold_src in ("vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed"):
                self.compute_all_d_vort_smoothed()
        if threshold_src in ("vis", "vorticity_vis", "smoothed", "smooth"):
            threshold_vort = self.ps.vorticity_vis.to_numpy()[:n].astype(np.float32, copy=False)
        elif threshold_src in ("vorticity_loss", "loss", "d_vort", "vorticity_loss_pred", "loss_pred"):
            threshold_vort = self.d_vort.to_numpy()[:n].astype(np.float32, copy=False)
        elif threshold_src in ("vorticity_loss_smoothed", "loss_smoothed", "d_vort_smoothed"):
            threshold_vort = self.d_vort_smoothed.to_numpy()[:n].astype(np.float32, copy=False)
        else:
            threshold_vort = self.ps.vorticity.to_numpy()[:n].astype(np.float32, copy=False)
        vel = self.ps.v.to_numpy()[:n].astype(np.float32, copy=False)
        mv = self.ps.m_V.to_numpy()[:n].astype(np.float32, copy=False)
        center_cfg = self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionCircleCenter", None)
        if center_cfg is None:
            center_cfg = self.seg_cfg.get_cfg("boundaryCircleCenter", [0.65, 0.5])
        c = np.asarray(center_cfg, dtype=np.float32).reshape(-1)
        cx = float(c[0]) if c.size > 0 else 0.65
        cy = float(c[1]) if c.size > 1 else 0.5
        radius = float(
            self.seg_cfg.get_cfg(
                "sphBoundaryVorticityInjectionCircleRadius",
                self.seg_cfg.get_cfg("boundaryCircleRadius", 0.1),
            )
        )
        band = float(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionDistance", 0.02))
        threshold = float(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionThreshold", 10.0))
        max_per = int(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionMaxPerStep", 32))
        if max_per <= 0:
            return

        dx = x[:, 0] - np.float32(cx)
        dy = x[:, 1] - np.float32(cy)
        rr = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        signed_dist = rr - np.float32(radius)
        omega_z = vort[:, 2]
        omega_mag = np.linalg.norm(vort, axis=1).astype(np.float32)
        threshold_mag = np.linalg.norm(threshold_vort, axis=1).astype(np.float32)
        region = str(
            self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionRegion", "circle_band")
            or "circle_band"
        ).lower().strip()
        if region in ("all", "global", "whole_domain", "domain", "anywhere"):
            mask = fluid_mask & (threshold_mag >= np.float32(threshold))
        else:
            mask = fluid_mask & (signed_dist >= 0.0) & (signed_dist <= np.float32(band)) & (threshold_mag >= np.float32(threshold))
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return
        candidate_threshold_z = threshold_vort[idx, 2].astype(np.float32, copy=False)
        candidate_pos = int(np.sum(candidate_threshold_z > 0.0))
        candidate_neg = int(np.sum(candidate_threshold_z < 0.0))
        candidate_zero = int(idx.size - candidate_pos - candidate_neg)
        if idx.size > max_per:
            selection = str(
                self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSelection", "top_strength")
                or "top_strength"
            ).lower().strip()
            strength = threshold_mag[idx]
            if selection in ("proportional_sign", "sign_proportional", "weighted_sign", "ratio_sign"):
                threshold_omega_z = threshold_vort[:, 2]
                sign_src = threshold_omega_z[idx]
                pos = idx[sign_src > 0.0]
                neg = idx[sign_src < 0.0]
                pos = pos[np.argsort(-threshold_mag[pos])]
                neg = neg[np.argsort(-threshold_mag[neg])]
                n_pos = int(pos.size)
                n_neg = int(neg.size)
                n_signed = n_pos + n_neg
                if n_signed > 0:
                    take_pos = int(round(float(max_per) * float(n_pos) / float(n_signed)))
                    take_pos = max(0, min(max_per, take_pos))
                    take_neg = max_per - take_pos
                else:
                    take_pos = 0
                    take_neg = 0
                take_pos = min(take_pos, n_pos)
                take_neg = min(take_neg, n_neg)
                chosen_parts = []
                if take_pos > 0:
                    chosen_parts.append(pos[:take_pos])
                if take_neg > 0:
                    chosen_parts.append(neg[:take_neg])
                chosen = np.concatenate(chosen_parts) if chosen_parts else np.zeros((0,), dtype=idx.dtype)
                if chosen.size < max_per:
                    chosen_set = set(int(i) for i in chosen.tolist())
                    rest = np.array([int(i) for i in idx if int(i) not in chosen_set], dtype=idx.dtype)
                    if rest.size > 0:
                        rest = rest[np.argsort(-threshold_mag[rest])]
                        chosen = np.concatenate([chosen, rest[: max_per - chosen.size]])
                idx = chosen.astype(idx.dtype, copy=False)
            elif selection in ("balanced_sign", "sign_balanced", "pos_neg", "positive_negative"):
                threshold_omega_z = threshold_vort[:, 2]
                sign_src = threshold_omega_z[idx]
                pos = idx[sign_src > 0.0]
                neg = idx[sign_src < 0.0]
                pos_strength = threshold_mag[pos]
                neg_strength = threshold_mag[neg]
                pos = pos[np.argsort(-pos_strength)]
                neg = neg[np.argsort(-neg_strength)]
                half = max_per // 2
                chosen_parts = []
                if half > 0:
                    chosen_parts.append(pos[:half])
                    chosen_parts.append(neg[:half])
                chosen = np.concatenate(chosen_parts) if chosen_parts else np.zeros((0,), dtype=idx.dtype)
                if chosen.size < max_per:
                    chosen_set = set(int(i) for i in chosen.tolist())
                    rest = np.array([int(i) for i in idx if int(i) not in chosen_set], dtype=idx.dtype)
                    if rest.size > 0:
                        rest_strength = threshold_mag[rest]
                        rest = rest[np.argsort(-rest_strength)]
                        chosen = np.concatenate([chosen, rest[: max_per - chosen.size]])
                idx = chosen.astype(idx.dtype, copy=False)
            else:
                pick = np.argpartition(-strength, max_per - 1)[:max_per]
                idx = idx[pick]

        selected_threshold_z = threshold_vort[idx, 2].astype(np.float32, copy=False)
        selected_pos = int(np.sum(selected_threshold_z > 0.0))
        selected_neg = int(np.sum(selected_threshold_z < 0.0))
        selected_zero = int(idx.size - selected_pos - selected_neg)

        seg_len = float(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSegmentLength", self.seg_cfg.get_cfg("boundarySegmentLength", 0.018)))
        gamma_scale = float(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionGammaScale", 1.0))
        seg_type = int(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSegmentTypeId", self.seg_cfg.get_cfg("initSegmentTypeId", 0)))
        orientation = str(
            self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionOrientation", "omega")
            or "omega"
        ).lower().strip()
        dim = int(self.ps.dim)
        tangent = np.zeros((idx.size, dim), dtype=np.float32)
        if dim == 3 and orientation in ("auto", "omega", "vorticity", "vorticity_direction"):
            tangent = vort[idx, :3].astype(np.float32, copy=True)
            tn = np.linalg.norm(tangent, axis=1, keepdims=True)
            weak = tn[:, 0] < 1e-8
            tangent = tangent / (tn + 1e-8)
            tangent[weak, 0] = 1.0
            tangent[weak, 1] = 0.0
            tangent[weak, 2] = 0.0
        else:
            use_circle_tangent = orientation in ("circle", "circle_tangent", "boundary_tangent") or (
                orientation == "auto" and region not in ("all", "global", "whole_domain", "domain", "anywhere")
            )
            if use_circle_tangent:
                outward = np.stack([dx[idx], dy[idx]], axis=1).astype(np.float32)
                outward_norm = np.linalg.norm(outward, axis=1, keepdims=True) + 1e-8
                outward = outward / outward_norm
                tangent[:, :2] = np.stack([-outward[:, 1], outward[:, 0]], axis=1).astype(np.float32)
            elif orientation in ("x", "streamwise", "streamwise_x"):
                tangent[:, 0] = 1.0
            elif orientation in ("omega", "vorticity", "vorticity_direction") and dim == 2:
                tangent[:, 0] = 1.0
            else:
                tangent[:, :2] = vel[idx, :2].astype(np.float32, copy=True)
                tn = np.linalg.norm(tangent[:, :2], axis=1, keepdims=True)
                weak = tn[:, 0] < 1e-8
                tangent[:, :2] = tangent[:, :2] / (tn + 1e-8)
                tangent[weak, 0] = 1.0
                tangent[weak, 1] = 0.0
        sign = np.sign(omega_z[idx]).astype(np.float32)
        sign[sign == 0.0] = 1.0
        if dim == 2 and bool(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionOrientBySign", True)):
            tangent *= sign[:, None]

        centers = x[idx, :dim].astype(np.float32, copy=False)
        xm_d = centers - np.float32(0.5 * seg_len) * tangent
        xp_d = centers + np.float32(0.5 * seg_len) * tangent
        xm = np.zeros((idx.size, 3), dtype=np.float32)
        xp = np.zeros((idx.size, 3), dtype=np.float32)
        xm[:, :dim] = xm_d
        xp[:, :dim] = xp_d
        gamma_source_vort = vort
        residual_enabled = bool(
            self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionUseResidualVorticity", False)
        ) or vort_src in (
            "vorticity_residual",
            "residual_vorticity",
            "vorticity_minus_vortex",
            "sph_vorticity_residual",
        )
        omega_vortex_gamma_sel = None
        omega_vortex_curl_sel = None
        if residual_enabled:
            support = float(
                self.ps.support_radius
                * float(self.seg_cfg.get_cfg("sphBoundaryVortexResidualSupportRadiusScale", 1.0))
            )
            omega_vortex_gamma_sel = self._estimate_vortex_omega_from_gamma_at_points(
                centers, support
            )
            gamma_source_vort = vort.copy()
            gamma_source_vort[idx, :3] = vort[idx, :3] - omega_vortex_gamma_sel[:, :3]
            if bool(self.seg_cfg.get_cfg("sphBoundaryVorticityResidualDebug", False)):
                omega_vortex_curl_sel = self._estimate_vortex_omega_from_velocity_curl_at_points(
                    centers, support
                )
        gamma_mode = str(
            self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionGammaMode", "omega_dot_t")
            or "omega_dot_t"
        ).lower().strip()
        gamma_omega_z = gamma_source_vort[:, 2]
        gamma_omega_mag = np.linalg.norm(gamma_source_vort, axis=1).astype(np.float32)
        if gamma_mode in ("omega_norm", "source_norm", "norm", "magnitude"):
            gamma = (gamma_scale * gamma_omega_mag[idx] * mv[idx]).astype(np.float32)
        elif gamma_mode in ("omega_dot_t", "source_dot_t", "dot_t", "projection"):
            gamma = (gamma_scale * np.sum(gamma_source_vort[idx, :dim] * tangent, axis=1) * mv[idx]).astype(np.float32)
        elif gamma_mode in ("source_component", "omega_component", "z", "z_component"):
            gamma = (gamma_scale * gamma_omega_z[idx] * mv[idx]).astype(np.float32)
        else:
            gamma = (gamma_scale * gamma_omega_z[idx] * mv[idx]).astype(np.float32)

        offset = int(self.ss_seg.segment_num[None])
        cap = int(self.ss_seg.segment_max_num)
        if offset >= cap:
            return
        n_new = min(int(idx.size), cap - offset)
        if n_new <= 0:
            return
        gamma_pos = int(np.sum(gamma[:n_new] > 0.0))
        gamma_neg = int(np.sum(gamma[:n_new] < 0.0))
        gamma_zero = int(n_new - gamma_pos - gamma_neg)
        self.seg_solver._seed_segments_kernel(offset, n_new, xm[:n_new], xp[:n_new], gamma[:n_new], seg_type)
        self.seg_solver._set_point_vortex_volume_kernel(offset, n_new, mv[idx[:n_new]].astype(np.float32, copy=False))
        if self._segment_initial_sph_then_bs and not self._one_step_vortex_impulse:
            init_vel = vel[idx[:n_new], :].astype(np.float32, copy=False)
            self._set_segment_initial_sph_advect_velocity_kernel(
                offset, n_new, init_vel, int(self._initial_sph_advect_steps)
            )
        self.ss_seg.segment_num[None] = offset + n_new
        if residual_enabled and bool(self.seg_cfg.get_cfg("sphBoundaryVorticityResidualDebug", False)):
            dbg_local = int(
                np.argmax(np.abs(gamma[:n_new])) if n_new > 0 else 0
            )
            dbg_p = int(idx[dbg_local])
            self._debug_last_residual_particle = dbg_p
            ov_g = omega_vortex_gamma_sel[dbg_local] if omega_vortex_gamma_sel is not None else np.zeros((3,), dtype=np.float32)
            ov_c = omega_vortex_curl_sel[dbg_local] if omega_vortex_curl_sel is not None else np.zeros((3,), dtype=np.float32)
            res_v = gamma_source_vort[dbg_p]
            print(
                f"[sph-vort-residual-debug] step={step} particle={dbg_p} "
                f"pos=({float(x[dbg_p, 0]):.4f},{float(x[dbg_p, 1]):.4f}) "
                f"omega_sph_z={float(vort[dbg_p, 2]):.6e} "
                f"omega_vortex_gamma_z={float(ov_g[2]):.6e} "
                f"omega_vortex_curl_z={float(ov_c[2]):.6e} "
                f"omega_res_z={float(res_v[2]):.6e} "
                f"gamma_new={float(gamma[dbg_local]):.6e}"
            )
        if bool(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSignDebug", False)):
            print(
                f"[sph-vort-inject-sign] step={step} "
                f"candidate(+/-/0)={candidate_pos}/{candidate_neg}/{candidate_zero} "
                f"selected(+/-/0)={selected_pos}/{selected_neg}/{selected_zero} "
                f"gamma(+/-/0)={gamma_pos}/{gamma_neg}/{gamma_zero}"
            )
        if bool(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionLog", False)):
            print(
                f"[sph-vort-inject] step={step} region={region} added={n_new} "
                f"|gamma|max={float(np.max(np.abs(gamma[:n_new]))):.4e}"
            )

    @ti.kernel
    def _set_segment_initial_sph_advect_velocity_kernel(
        self,
        offset: int,
        n_new: int,
        init_vel: ti.types.ndarray(),
        active_steps: ti.i32,
    ):
        for k in range(n_new):
            i = offset + k
            if ti.static(self.ps.dim == 2):
                v = ti.Vector([init_vel[k, 0], init_vel[k, 1]])
                self.seg_solver.sph_advect_velocity_minus[i] = v
                self.seg_solver.sph_advect_velocity_plus[i] = v
            else:
                v = ti.Vector([init_vel[k, 0], init_vel[k, 1], init_vel[k, 2]])
                self.seg_solver.sph_advect_velocity_minus[i] = v
                self.seg_solver.sph_advect_velocity_plus[i] = v
            self.seg_solver.initial_sph_advect_remaining[i] = active_steps

    @ti.kernel
    def _advance_segment_initial_sph_advect_velocity_kernel(self, decay: float):
        for i in range(self.ss_seg.segment_max_num):
            if self.ss_seg.active[i] != 1:
                self.seg_solver.sph_advect_velocity_minus[i] *= 0.0
                self.seg_solver.sph_advect_velocity_plus[i] *= 0.0
                self.seg_solver.initial_sph_advect_remaining[i] = 0
                continue
            if self.seg_solver.initial_sph_advect_remaining[i] > 1:
                self.seg_solver.initial_sph_advect_remaining[i] -= 1
                self.seg_solver.sph_advect_velocity_minus[i] *= decay
                self.seg_solver.sph_advect_velocity_plus[i] *= decay
            elif self.seg_solver.initial_sph_advect_remaining[i] == 1:
                self.seg_solver.initial_sph_advect_remaining[i] = 0
                self.seg_solver.sph_advect_velocity_minus[i] *= 0.0
                self.seg_solver.sph_advect_velocity_plus[i] *= 0.0
            else:
                self.seg_solver.sph_advect_velocity_minus[i] *= 0.0
                self.seg_solver.sph_advect_velocity_plus[i] *= 0.0

    def _delete_interior_segments_inside_obstacles(self):
        if not bool(self.seg_cfg.get_cfg("deleteInteriorSegmentsInsideObstacles", False)):
            return
        n = int(self.ss_seg.segment_num[None])
        if n <= 0:
            return
        dim = int(self.ss_seg.dim)
        x_minus = self.ss_seg.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss_seg.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss_seg.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss_seg.active.to_numpy()[:n].astype(np.int32)
        age = self.ss_seg.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss_seg.seg_type.to_numpy()[:n].astype(np.int32)
        sph_v_minus = self.seg_solver.sph_advect_velocity_minus.to_numpy()[:n].astype(np.float32)
        sph_v_plus = self.seg_solver.sph_advect_velocity_plus.to_numpy()[:n].astype(np.float32)
        initial_remaining = self.seg_solver.initial_sph_advect_remaining.to_numpy()[:n].astype(np.int32)
        point_vortex_volume = self.seg_solver.point_vortex_volume.to_numpy()[:n].astype(np.float32)
        centers = 0.5 * (x_minus[:, :dim] + x_plus[:, :dim])

        target = self.seg_cfg.get_cfg("deleteInsideObstacleSegmentTypeIds", None)
        if target is None:
            target = [int(self.seg_cfg.get_cfg("deleteInsideObstacleSegmentTypeId", self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSegmentTypeId", 0)))]
        target_types = set(int(t) for t in target)
        margin_cfg = self.seg_cfg.get_cfg("deleteInsideObstacleMargin", None)
        if margin_cfg is None:
            margin = 0.5 * float(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSegmentLength", self.seg_cfg.get_cfg("boundarySegmentLength", 0.018)))
        else:
            margin = float(margin_cfg)

        remove = np.zeros((n,), dtype=bool)
        candidate = (active == 1) & np.isin(seg_type, list(target_types))
        if not np.any(candidate):
            return

        if bool(self.seg_cfg.get_cfg("deleteInsideObstacleIncludeRigidBlocks", True)):
            default_ex = self.seg_cfg.get_cfg("boundaryRigidBlockExcludeObjectIds", []) or []
            exclude_ids = set(int(x) for x in (self.seg_cfg.get_cfg("deleteInsideObstacleRigidBlockExcludeObjectIds", default_ex) or []))
            include_ids_cfg = self.seg_cfg.get_cfg("deleteInsideObstacleRigidBlockObjectIds", None)
            include_ids = None if include_ids_cfg is None else set(int(x) for x in include_ids_cfg)
            for blk in self.seg_cfg.get_obstacles():
                oid = int(blk.get("objectId", -999999))
                if oid in exclude_ids:
                    continue
                if include_ids is not None and oid not in include_ids:
                    continue
                start = np.asarray(blk.get("start", [0.0] * dim), dtype=np.float32).reshape(-1)
                end = np.asarray(blk.get("end", [0.0] * dim), dtype=np.float32).reshape(-1)
                trans = np.asarray(blk.get("translation", [0.0] * dim), dtype=np.float32).reshape(-1)
                scale = np.asarray(blk.get("scale", [1.0] * dim), dtype=np.float32).reshape(-1)
                if start.size < dim or end.size < dim:
                    continue
                if trans.size < dim:
                    trans = np.pad(trans, (0, dim - trans.size), constant_values=0.0)
                if scale.size < dim:
                    scale = np.pad(scale, (0, dim - scale.size), constant_values=1.0)
                lo = start[:dim] + trans[:dim]
                hi = lo + (end[:dim] - start[:dim]) * scale[:dim]
                lo2 = np.minimum(lo, hi) - margin
                hi2 = np.maximum(lo, hi) + margin
                inside = np.all((centers >= lo2[None, :]) & (centers <= hi2[None, :]), axis=1)
                remove |= candidate & inside

        if bool(self.seg_cfg.get_cfg("deleteInsideObstacleIncludeCylinders", True)) and dim >= 2:
            cylinders = []
            cylinders.extend(self.seg_cfg.config.get("RigidCylinders", []) or [])
            cylinders.extend(self.seg_cfg.config.get("Cylinders", []) or [])
            if bool(self.seg_cfg.get_cfg("deleteInsideObstacleIncludeBoundaryCircle", True)):
                bc = self.seg_cfg.get_cfg("boundaryCircleCenter", None)
                br = self.seg_cfg.get_cfg("boundaryCircleRadius", None)
                if bc is not None and br is not None:
                    cylinders.append({"center": bc, "radius": br})
            for cyl in cylinders:
                c = np.asarray(cyl.get("center", [0.0, 0.0]), dtype=np.float32).reshape(-1)
                if c.size < 2:
                    continue
                r = float(cyl.get("radius", 0.0)) + margin
                dxy = centers[:, :2] - c[None, :2]
                inside = np.sum(dxy * dxy, axis=1) <= r * r
                if dim >= 3 and c.size >= 3 and cyl.get("height", None) is not None:
                    axis = np.asarray(cyl.get("axis", [0.0, 0.0, 1.0]), dtype=np.float32).reshape(-1)
                    if axis.size < 3:
                        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    axis = axis[:3]
                    axis = axis / (np.linalg.norm(axis) + 1e-8)
                    rel = centers[:, :3] - c[None, :3]
                    axial = np.abs(np.sum(rel * axis[None, :], axis=1))
                    inside &= axial <= 0.5 * float(cyl.get("height", 0.0)) + margin
                remove |= candidate & inside

        if not np.any(remove):
            return
        keep = (active == 1) & (~remove)
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            self.ss_seg.segment_num[None] = 0
            return
        self.seg_solver._overwrite_segments_kernel(
            int(idx.size),
            x_minus[idx],
            x_plus[idx],
            gamma[idx],
            age[idx],
            seg_type[idx].astype(np.int32),
        )
        self.seg_solver._overwrite_initial_sph_advect_state_kernel(
            int(idx.size),
            sph_v_minus[idx],
            sph_v_plus[idx],
            initial_remaining[idx],
        )
        self.seg_solver._overwrite_point_vortex_volume_kernel(
            int(idx.size), point_vortex_volume[idx]
        )
        self.ss_seg.segment_num[None] = int(idx.size)
        self.ss_seg.update_segment_geometry()
        if bool(self.seg_cfg.get_cfg("deleteInsideObstacleLog", False)):
            print(f"[segment-obstacle-cull] removed={int(np.sum(remove))} kept={int(idx.size)}")

    def _delete_slow_internal_point_vortices(self):
        if not bool(self.seg_cfg.get_cfg("deleteSlowPointVorticesEnabled", False)):
            return
        n = int(self.ss_seg.segment_num[None])
        if n <= 0:
            return
        threshold = float(self.seg_cfg.get_cfg("deleteSlowPointVortexSpeedThreshold", 1e-6))
        if threshold <= 0.0:
            return
        boundary_type = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
        target = self.seg_cfg.get_cfg("deleteSlowPointVortexSegmentTypeIds", None)
        if target is None:
            target_types = {int(self.seg_cfg.get_cfg("sphBoundaryVorticityInjectionSegmentTypeId", 0))}
        else:
            target_types = set(int(t) for t in target)
        dim = int(self.ss_seg.dim)
        x_minus = self.ss_seg.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss_seg.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss_seg.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss_seg.active.to_numpy()[:n].astype(np.int32)
        age = self.ss_seg.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss_seg.seg_type.to_numpy()[:n].astype(np.int32)
        sph_v_minus = self.seg_solver.sph_advect_velocity_minus.to_numpy()[:n].astype(np.float32)
        sph_v_plus = self.seg_solver.sph_advect_velocity_plus.to_numpy()[:n].astype(np.float32)
        initial_remaining = self.seg_solver.initial_sph_advect_remaining.to_numpy()[:n].astype(np.int32)
        point_vortex_volume = self.seg_solver.point_vortex_volume.to_numpy()[:n].astype(np.float32)
        v_minus = self.seg_solver.v_minus.to_numpy()[:n, :dim].astype(np.float32)
        v_plus = self.seg_solver.v_plus.to_numpy()[:n, :dim].astype(np.float32)
        speed = np.linalg.norm(0.5 * (v_minus + v_plus), axis=1)
        candidate = (
            (active == 1)
            & (seg_type != boundary_type)
            & np.isin(seg_type, list(target_types))
        )
        remove = candidate & (speed < np.float32(threshold))
        if not np.any(remove):
            return
        keep = (active == 1) & (~remove)
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            self.ss_seg.segment_num[None] = 0
            return
        self.seg_solver._overwrite_segments_kernel(
            int(idx.size),
            x_minus[idx],
            x_plus[idx],
            gamma[idx],
            age[idx],
            seg_type[idx].astype(np.int32),
        )
        self.seg_solver._overwrite_initial_sph_advect_state_kernel(
            int(idx.size),
            sph_v_minus[idx],
            sph_v_plus[idx],
            initial_remaining[idx],
        )
        self.seg_solver._overwrite_point_vortex_volume_kernel(
            int(idx.size), point_vortex_volume[idx]
        )
        self.ss_seg.segment_num[None] = int(idx.size)
        self.ss_seg.update_segment_geometry()
        if bool(self.seg_cfg.get_cfg("deleteSlowPointVortexLog", False)):
            print(
                f"[slow-point-vortex-cull] removed={int(np.sum(remove))} "
                f"kept={int(idx.size)} threshold={threshold:.6e} "
                f"min_speed={float(np.min(speed[candidate])) if np.any(candidate) else 0.0:.6e}"
            )

    def _clear_one_step_internal_point_vortices(self):
        n = int(self.ss_seg.segment_num[None])
        if n <= 0:
            return
        boundary_type = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
        x_minus = self.ss_seg.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss_seg.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss_seg.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss_seg.active.to_numpy()[:n].astype(np.int32)
        age = self.ss_seg.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss_seg.seg_type.to_numpy()[:n].astype(np.int32)
        sph_v_minus = self.seg_solver.sph_advect_velocity_minus.to_numpy()[:n].astype(np.float32)
        sph_v_plus = self.seg_solver.sph_advect_velocity_plus.to_numpy()[:n].astype(np.float32)
        initial_remaining = self.seg_solver.initial_sph_advect_remaining.to_numpy()[:n].astype(np.int32)
        point_vortex_volume = self.seg_solver.point_vortex_volume.to_numpy()[:n].astype(np.float32)
        keep = (active == 1) & (seg_type == boundary_type)
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            self.seg_solver._clear_active_kernel(n)
            self.ss_seg.segment_num[None] = 0
            return
        self.seg_solver._overwrite_segments_kernel(
            int(idx.size),
            x_minus[idx],
            x_plus[idx],
            gamma[idx],
            age[idx],
            seg_type[idx].astype(np.int32),
        )
        self.seg_solver._overwrite_initial_sph_advect_state_kernel(
            int(idx.size),
            sph_v_minus[idx],
            sph_v_plus[idx],
            initial_remaining[idx],
        )
        self.seg_solver._overwrite_point_vortex_volume_kernel(
            int(idx.size), point_vortex_volume[idx]
        )
        self.ss_seg.segment_num[None] = int(idx.size)
        self.ss_seg.update_segment_geometry()

    def _print_internal_segment_gamma_debug(self):
        if not bool(self.seg_cfg.get_cfg("debugInternalSegmentGamma", False)):
            return
        interval = max(1, int(self.seg_cfg.get_cfg("debugInternalSegmentGammaInterval", 10)))
        step = int(self.seg_solver._sim_step_index)
        if step % interval != 0:
            return
        n = int(self.ss_seg.segment_num[None])
        if n <= 0:
            print(f"[segment-gamma-debug] step={step} internal_count=0")
            return
        gamma = self.ss_seg.gamma.to_numpy()[:n].astype(np.float32, copy=False)
        active = self.ss_seg.active.to_numpy()[:n].astype(np.int32, copy=False)
        seg_type = self.ss_seg.seg_type.to_numpy()[:n].astype(np.int32, copy=False)
        boundary_type = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
        internal_mask = (active == 1) & (seg_type != boundary_type)
        ids = np.nonzero(internal_mask)[0]
        if ids.size == 0:
            print(f"[segment-gamma-debug] step={step} internal_count=0")
            return
        g = gamma[ids]
        abs_g = np.abs(g)
        print(
            f"[segment-gamma-debug] step={step} internal_count={int(ids.size)} "
            f"abs_gamma_min={float(np.min(abs_g)):.6e} "
            f"abs_gamma_max={float(np.max(abs_g)):.6e} "
            f"abs_gamma_mean={float(np.mean(abs_g)):.6e}"
        )

    def _advance_segments_coupled(self):
        ssol = self.seg_solver
        timing = self._segment_timing_enabled and (
            int(ssol._sim_step_index) % self._segment_timing_interval == 0
        )
        records = []

        def sync_if_needed():
            if timing and self._substep_timing_sync:
                ti.sync()

        def run_stage(name, fn):
            if not timing:
                fn()
                return
            sync_if_needed()
            t0 = time.perf_counter()
            fn()
            sync_if_needed()
            records.append((name, (time.perf_counter() - t0) * 1000.0))

        def boundary_pipeline():
            if ssol.has_boundary:
                if ssol._boundary_schedule == "each_step":
                    strip = bool(
                        ssol.ss.cfg.get_cfg("boundaryReplaceCommittedEachStep", False)
                    )
                    ssol._run_boundary_injection_pipeline(strip_committed_first=strip)
                elif not ssol._boundary_one_shot_done:
                    strip = bool(
                        ssol.ss.cfg.get_cfg("boundaryReplaceCommittedEachStep", False)
                    )
                    ssol._run_boundary_injection_pipeline(strip_committed_first=strip)
                    ssol.boundary.mark_one_shot_complete_if_applicable(ssol)

        def periodic_emit():
            ssol._emit_periodic_parallel_x_layers()

        def residual_vorticity_injection():
            self._inject_segments_from_sph_boundary_vorticity()

        def update_segment_geometry_pre():
            ssol.ss.update_segment_geometry()

        def vorticity_deposit_to_segments():
            dbg_deposit = 0
            if self._sph_to_segment_deposit_enabled and self.sph_to_segment_blend > 0.0:
                if self._deposit_debug:
                    self._deposit_debug_step += 1
                    if self._deposit_debug_step % self._deposit_debug_interval == 0:
                        dbg_deposit = 1
                        self._deposit_debug_reset()
                self._deposit_vorticity_to_segments_kernel(
                    float(self.sph_to_segment_blend),
                    float(self.sph_to_segment_gamma_scale),
                    int(self.ps.material_fluid),
                    int(self._deposit_gamma_proj),
                    int(self._deposit_src_vis),
                    dbg_deposit,
                    int(self._deposit_skip_seg_type),
                )
                if dbg_deposit:
                    self._print_deposit_debug_maxima()

        def sph_velocity_deposit_to_segments():
            if self._sph_velocity_to_segment_enabled and not self._segment_initial_sph_then_bs:
                self._deposit_velocity_to_segments_advect_kernel(
                    float(self._sph_velocity_to_segment_blend),
                    float(self._sph_velocity_to_segment_scale),
                    int(self.ps.material_fluid),
                    int(self._sph_velocity_to_segment_skip_type),
                )

        run_stage("boundary_pipeline", boundary_pipeline)
        run_stage("periodic_emit", periodic_emit)
        run_stage("residual_vorticity_injection", residual_vorticity_injection)
        run_stage("update_segment_geometry_pre", update_segment_geometry_pre)
        run_stage("vorticity_deposit_to_segments", vorticity_deposit_to_segments)
        run_stage("sph_velocity_deposit_to_segments", sph_velocity_deposit_to_segments)

        if self._one_step_vortex_impulse:
            run_stage("delete_inside_obstacles", self._delete_interior_segments_inside_obstacles)
            old_bg = int(ssol._segment_advect_use_background[None])
            old_sph = int(ssol._segment_advect_use_sph[None])
            old_bs = int(ssol._segment_advect_use_bs[None])
            ssol._segment_advect_use_background[None] = 0
            ssol._segment_advect_use_sph[None] = 0
            ssol._segment_advect_use_bs[None] = 1
            run_stage("compute_endpoint_velocity", ssol.compute_endpoint_velocity)
            ssol._segment_advect_use_background[None] = old_bg
            ssol._segment_advect_use_sph[None] = old_sph
            ssol._segment_advect_use_bs[None] = old_bs
            run_stage("update_segment_geometry_post", ssol.ss.update_segment_geometry)
        else:
            sph_advect_bs_ghost = bool(
                self.seg_cfg.get_cfg("segmentSphAdvectButGhostUseBsVelocity", False)
            )
            if sph_advect_bs_ghost:
                old_bg = int(ssol._segment_advect_use_background[None])
                old_sph = int(ssol._segment_advect_use_sph[None])
                old_bs = int(ssol._segment_advect_use_bs[None])
                ssol._segment_advect_use_background[None] = 0
                ssol._segment_advect_use_sph[None] = 1
                ssol._segment_advect_use_bs[None] = 0
                run_stage("compute_endpoint_velocity", ssol.compute_endpoint_velocity)
                run_stage("advect_segments_rk4", ssol.advect_segments_rk4)
                ssol._segment_advect_use_background[None] = old_bg
                ssol._segment_advect_use_sph[None] = old_sph
                ssol._segment_advect_use_bs[None] = old_bs
            else:
                run_stage("compute_endpoint_velocity", ssol.compute_endpoint_velocity)
                run_stage("advect_segments_rk4", ssol.advect_segments_rk4)
            if self._segment_initial_sph_then_bs:
                run_stage(
                    "initial_sph_advect_decay",
                    lambda: self._advance_segment_initial_sph_advect_velocity_kernel(
                        float(self._initial_sph_advect_decay)
                    ),
                )
            run_stage("update_segment_geometry_post", ssol.ss.update_segment_geometry)
            run_stage("delete_inside_obstacles", self._delete_interior_segments_inside_obstacles)
            run_stage("delete_slow_point_vortices", self._delete_slow_internal_point_vortices)
            if not getattr(ssol, "_bs_2d_point", False):
                run_stage("split_segments", ssol.split_segments)
                run_stage("merge_segments", ssol.merge_segments)
                run_stage("restore_frozen_segment_geometry", ssol.restore_frozen_segment_geometry)
            run_stage("delete_weak_segments", ssol.delete_weak_segments)
            if sph_advect_bs_ghost:
                old_bg = int(ssol._segment_advect_use_background[None])
                old_sph = int(ssol._segment_advect_use_sph[None])
                old_bs = int(ssol._segment_advect_use_bs[None])
                ssol._segment_advect_use_background[None] = 0
                ssol._segment_advect_use_sph[None] = 0
                ssol._segment_advect_use_bs[None] = 1
                run_stage("recompute_bs_velocity_for_ghost", ssol.compute_endpoint_velocity)
                ssol._segment_advect_use_background[None] = old_bg
                ssol._segment_advect_use_sph[None] = old_sph
                ssol._segment_advect_use_bs[None] = old_bs
        run_stage("print_internal_segment_gamma_debug", self._print_internal_segment_gamma_debug)
        if ssol._use_leapfrog_initial_impulse[None] == 1:
            run_stage("decay_impulse", ssol._decay_impulse_kernel)

        if timing:
            total = sum(v for _, v in records)
            parts = " ".join(f"{name}={ms:.3f}ms" for name, ms in records)
            print(
                f"[segment-advance-timing] step={int(ssol._sim_step_index)} "
                f"total={total:.3f}ms {parts}"
            )
        ssol._sim_step_index += 1

    @ti.kernel
    def _deposit_debug_reset(self):
        # [DEBUG 沉积] 每轮打印前清零，供 atomic_max 汇总
        self._dep_dbg_max_w[None] = 0.0
        self._dep_dbg_max_wsum[None] = 0.0
        self._dep_dbg_max_omega_norm[None] = 0.0
        self._dep_dbg_max_om_eff[None] = 0.0
        self._dep_dbg_max_g_tgt[None] = 0.0
        self._dep_dbg_max_gamma[None] = 0.0

    def _print_deposit_debug_maxima(self):
        # [DEBUG 沉积] 仅打印沉积 kernel 内统计到的全局最大值（见 _deposit_vorticity_to_segments_kernel）
        print(
            "[DEBUG deposit] "
            f"max_w(m_V*W)={float(self._dep_dbg_max_w[None]):.6e} "
            f"max_wsum={float(self._dep_dbg_max_wsum[None]):.6e} "
            f"max_|omega_avg|={float(self._dep_dbg_max_omega_norm[None]):.6e} "
            f"max_|om_eff|={float(self._dep_dbg_max_om_eff[None]):.6e} "
            f"max_|g_tgt|={float(self._dep_dbg_max_g_tgt[None]):.6e} "
            f"max_|gamma|={float(self._dep_dbg_max_gamma[None]):.6e}"
        )

    @ti.func
    def _sample_sph_velocity_at_position_from_grid(self, pos, mf: ti.i32):
        h = self.ps.support_radius
        center_cell = self.ps.pos_to_index(pos)
        vacc = ti.Vector([0.0 for _ in ti.static(range(self.ps.dim))])
        wsum = 0.0
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
            cell = center_cell + offset
            valid = 1
            for d in ti.static(range(self.ps.dim)):
                if cell[d] < 0 or cell[d] >= self.ps.grid_num[d]:
                    valid = 0
            if valid == 0:
                continue
            flat = -1
            if ti.static(self.ps.dim == 2):
                flat = self.ps.flatten_grid_index_2d(cell)
            else:
                flat = self.ps.flatten_grid_index_3d(cell)
            start = 0
            if flat > 0:
                start = self.ps.grid_particles_num[flat - 1]
            end = self.ps.grid_particles_num[flat]
            for p in range(start, end):
                if self.ps.material[p] != mf:
                    continue
                r = pos - self.ps.x[p]
                rn = r.norm()
                if rn < h:
                    mv_w = self.ps.m_V[p] * self.cubic_kernel(rn)
                    vacc += mv_w * self.ps.v[p]
                    wsum += mv_w
        return vacc, wsum

    @ti.kernel
    def _deposit_velocity_to_segments_advect_kernel(
        self,
        blend: float,
        scale: float,
        mf: ti.i32,
        skip_seg_type: ti.i32,
    ):
        nseg = self.ss_seg.segment_num[None]
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
                continue
            if skip_seg_type >= 0 and self.ss_seg.seg_type[i] == skip_seg_type:
                self.seg_solver.sph_advect_velocity_minus[i] *= 0.0
                self.seg_solver.sph_advect_velocity_plus[i] *= 0.0
                continue
            xm = self.ss_seg.x_minus[i]
            xp = self.ss_seg.x_plus[i]
            vacc_m, wsum_m = self._sample_sph_velocity_at_position_from_grid(xm, mf)
            vacc_p, wsum_p = self._sample_sph_velocity_at_position_from_grid(xp, mf)
            if wsum_m > 1e-12:
                v_target_m = scale * (vacc_m / wsum_m)
                self.seg_solver.sph_advect_velocity_minus[i] = (
                    (1.0 - blend) * self.seg_solver.sph_advect_velocity_minus[i]
                    + blend * v_target_m
                )
            else:
                self.seg_solver.sph_advect_velocity_minus[i] *= (1.0 - blend)
            if wsum_p > 1e-12:
                v_target_p = scale * (vacc_p / wsum_p)
                self.seg_solver.sph_advect_velocity_plus[i] = (
                    (1.0 - blend) * self.seg_solver.sph_advect_velocity_plus[i]
                    + blend * v_target_p
                )
            else:
                self.seg_solver.sph_advect_velocity_plus[i] *= (1.0 - blend)
        for i in range(nseg, self.ss_seg.segment_max_num):
            self.seg_solver.sph_advect_velocity_minus[i] *= 0.0
            self.seg_solver.sph_advect_velocity_plus[i] *= 0.0

    @ti.func
    def _vorticity_deposit_sample(self, p: int, src_vis: ti.i32) -> float:
        """2D 平面流：取涡量标量 ω_z（存于向量第 3 分量）。"""
        omega_z = self.ps.vorticity[p][2]
        if src_vis != 0:
            omega_z = self.ps.vorticity_vis[p][2]
        return omega_z

    @ti.kernel
    def _deposit_vorticity_to_segments_2d_kernel(
        self,
        blend: float,
        gamma_scale: float,
        mf: ti.i32,
        proj_mode: ti.i32,
        src_vis: ti.i32,
        dbg: ti.i32,
        skip_seg_type: ti.i32,
    ):
        h = self.ps.support_radius
        nseg = self.ss_seg.segment_num[None]
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
                continue
            if skip_seg_type >= 0 and self.ss_seg.seg_type[i] == skip_seg_type:
                continue
            xc = self.ss_seg.center[i]
            omega_acc_z = 0.0
            wsum = 0.0
            for p in range(self.ps.particle_num[None]):
                if self.ps.material[p] != mf:
                    continue
                r = xc - self.ps.x[p]
                rn = r.norm()
                if rn >= h:
                    continue
                mv_w = self.ps.m_V[p] * self.cubic_kernel(rn)
                if dbg != 0:
                    ti.atomic_max(self._dep_dbg_max_w[None], mv_w)
                omega_acc_z += mv_w * self._vorticity_deposit_sample(p, src_vis)
                wsum += mv_w
            if wsum > 1e-12:
                omega_z = omega_acc_z / wsum
                om_eff = omega_z
                if proj_mode == 1:
                    om_eff = ti.abs(omega_z)
                elif proj_mode == 2:
                    om_eff = omega_z
                g_tgt = gamma_scale * om_eff
                gamma_new = (1.0 - blend) * self.ss_seg.gamma[i] + blend * g_tgt
                self.ss_seg.gamma[i] = gamma_new
                if dbg != 0:
                    ti.atomic_max(self._dep_dbg_max_wsum[None], wsum)
                    ti.atomic_max(self._dep_dbg_max_omega_norm[None], ti.abs(omega_z))
                    ti.atomic_max(self._dep_dbg_max_om_eff[None], ti.abs(om_eff))
                    ti.atomic_max(self._dep_dbg_max_g_tgt[None], ti.abs(g_tgt))
                    ti.atomic_max(self._dep_dbg_max_gamma[None], ti.abs(gamma_new))

    @ti.kernel
    def _deposit_vorticity_to_segments_3d_kernel(
        self,
        blend: float,
        gamma_scale: float,
        mf: ti.i32,
        proj_mode: ti.i32,
        src_vis: ti.i32,
        dbg: ti.i32,
        skip_seg_type: ti.i32,
    ):
        h = self.ps.support_radius
        nseg = self.ss_seg.segment_num[None]
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
                continue
            if skip_seg_type >= 0 and self.ss_seg.seg_type[i] == skip_seg_type:
                continue
            xc = self.ss_seg.center[i]
            tdir = self.ss_seg.tangent[i]
            omega_acc = ti.Vector([0.0, 0.0, 0.0])
            wsum = 0.0
            for p in range(self.ps.particle_num[None]):
                if self.ps.material[p] != mf:
                    continue
                r = xc - self.ps.x[p]
                rn = r.norm()
                if rn >= h:
                    continue
                mv_w = self.ps.m_V[p] * self.cubic_kernel(rn)
                if dbg != 0:
                    ti.atomic_max(self._dep_dbg_max_w[None], mv_w)
                if src_vis != 0:
                    omega_acc += mv_w * self.ps.vorticity_vis[p]
                else:
                    omega_acc += mv_w * self.ps.vorticity[p]
                wsum += mv_w
            if wsum > 1e-12:
                omega_avg = omega_acc / wsum
                dot_ot = omega_avg.dot(tdir)
                om_eff = dot_ot
                if proj_mode == 1:
                    om_eff = omega_avg.norm()
                elif proj_mode == 2:
                    nrm = omega_avg.norm()
                    om_eff = nrm
                    if dot_ot < 0.0:
                        om_eff = -nrm
                g_tgt = gamma_scale * om_eff
                gamma_new = (1.0 - blend) * self.ss_seg.gamma[i] + blend * g_tgt
                self.ss_seg.gamma[i] = gamma_new
                if dbg != 0:
                    ti.atomic_max(self._dep_dbg_max_wsum[None], wsum)
                    ti.atomic_max(self._dep_dbg_max_omega_norm[None], omega_avg.norm())
                    ti.atomic_max(self._dep_dbg_max_om_eff[None], ti.abs(om_eff))
                    ti.atomic_max(self._dep_dbg_max_g_tgt[None], ti.abs(g_tgt))
                    ti.atomic_max(self._dep_dbg_max_gamma[None], ti.abs(gamma_new))

    def _deposit_vorticity_to_segments_kernel(
        self,
        blend: float,
        gamma_scale: float,
        mf: ti.i32,
        proj_mode: ti.i32,
        src_vis: ti.i32,
        dbg: ti.i32,
        skip_seg_type: int,
    ):
        if self.ps.dim == 2:
            self._deposit_vorticity_to_segments_2d_kernel(
                blend, gamma_scale, mf, proj_mode, src_vis, dbg, int(skip_seg_type)
            )
        else:
            self._deposit_vorticity_to_segments_3d_kernel(
                blend, gamma_scale, mf, proj_mode, src_vis, dbg, int(skip_seg_type)
            )

    @ti.func
    def _vortex_ghost_cubic_kernel(self, r_norm: float, support_radius: float):
        res = ti.cast(0.0, ti.f32)
        inv_h = 1.0 / support_radius
        k = 1.0
        if ti.static(self.ps.dim == 1):
            k = 1.3333
        elif ti.static(self.ps.dim == 2):
            k = 1.8189
        elif ti.static(self.ps.dim == 3):
            k = 2.5465
        k *= inv_h ** self.ps.dim
        q = r_norm * inv_h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                res = k * 2.0 * ti.pow(1.0 - q, 3.0)
        return res

    @ti.func
    def _vortex_ghost_flatten_cell(self, cell):
        flat = -1
        if ti.static(self.ps.dim == 2):
            flat = self.ps.flatten_grid_index_2d(cell)
        elif ti.static(self.ps.dim == 3):
            flat = self.ps.flatten_grid_index_3d(cell)
        return flat

    @ti.kernel
    def _sync_vortex_ghost_particles_from_segments(self, skip_seg_type: ti.i32, copy_velocity: ti.i32):
        self.vortex_ghost_num[None] = 0
        for c in range(self.ps.flattened_grid_num):
            self.vortex_ghost_grid_count[c] = 0
        nseg = self.ss_seg.segment_num[None]
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
                continue
            if skip_seg_type >= 0 and self.ss_seg.seg_type[i] == skip_seg_type:
                continue
            g = ti.atomic_add(self.vortex_ghost_num[None], 1)
            if g >= self.ss_seg.segment_max_num:
                continue
            for d in ti.static(range(self.ps.dim)):
                self.vortex_ghost_x[g][d] = 0.5 * (self.ss_seg.x_minus[i][d] + self.ss_seg.x_plus[i][d])
                self.vortex_ghost_v[g][d] = 0.0
                if copy_velocity != 0:
                    self.vortex_ghost_v[g][d] = 0.5 * (
                        self.seg_solver.v_minus[i][d] + self.seg_solver.v_plus[i][d]
                    )
            self.vortex_ghost_mV[g] = self.seg_solver.point_vortex_volume[i]
            self.vortex_ghost_gamma[g] = self.ss_seg.gamma[i]
        if self.vortex_ghost_num[None] > self.ss_seg.segment_max_num:
            self.vortex_ghost_num[None] = self.ss_seg.segment_max_num

    @ti.kernel
    def _build_vortex_ghost_grid(self):
        for g in range(self.vortex_ghost_num[None]):
            cell = self.ps.pos_to_index(self.vortex_ghost_x[g])
            valid = 1
            for d in ti.static(range(self.ps.dim)):
                if cell[d] < 0 or cell[d] >= self.ps.grid_num[d]:
                    valid = 0
            if valid == 0:
                continue
            flat = self._vortex_ghost_flatten_cell(cell)
            if flat < 0 or flat >= self.ps.flattened_grid_num:
                continue
            slot = ti.atomic_add(self.vortex_ghost_grid_count[flat], 1)
            if slot < ti.static(self._vortex_ghost_grid_capacity):
                self.vortex_ghost_grid_indices[flat, slot] = g

    @ti.kernel
    def _compute_vortex_ghost_bs_velocity_for_fluid(self, mf: ti.i32, support_radius: float, reg_radius: float):
        for p in range(self.ps.particle_max_num):
            self.u_vortex_ghost_sph[p] *= 0.0
        R2 = reg_radius * reg_radius
        inv2pi = 1.0 / (2.0 * ti.math.pi)
        normal_sign = self.seg_solver._bs_2d_normal_sign[None]
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] != mf:
                continue
            center_cell = self.ps.pos_to_index(self.ps.x[p])
            u = ti.Vector([0.0 for _ in ti.static(range(self.ps.dim))])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                cell = center_cell + offset
                valid = 1
                for d in ti.static(range(self.ps.dim)):
                    if cell[d] < 0 or cell[d] >= self.ps.grid_num[d]:
                        valid = 0
                if valid == 0:
                    continue
                flat = self._vortex_ghost_flatten_cell(cell)
                if flat < 0 or flat >= self.ps.flattened_grid_num:
                    continue
                count = self.vortex_ghost_grid_count[flat]
                if count > ti.static(self._vortex_ghost_grid_capacity):
                    count = ti.static(self._vortex_ghost_grid_capacity)
                for slot in range(count):
                    g = self.vortex_ghost_grid_indices[flat, slot]
                    r = self.ps.x[p] - self.vortex_ghost_x[g]
                    rn = r.norm()
                    if rn < support_radius:
                        Gamma = self.vortex_ghost_gamma[g]
                        if ti.static(self.ps.dim == 2):
                            rx = r[0]
                            ry = r[1]
                            denom = rx * rx + ry * ry + R2 + 1e-12
                            u[0] += normal_sign * inv2pi * Gamma * (-ry) / denom
                            u[1] += normal_sign * inv2pi * Gamma * (rx) / denom
                        else:
                            for d in ti.static(range(self.ps.dim)):
                                u[d] += 0.0
            self.u_vortex_ghost_sph[p] = u

    @ti.kernel
    def _compute_vortex_ghost_velocity_for_fluid(self, mf: ti.i32, support_radius: float):
        for p in range(self.ps.particle_max_num):
            self.u_vortex_ghost_sph[p] *= 0.0
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] != mf:
                continue
            center_cell = self.ps.pos_to_index(self.ps.x[p])
            v_acc = ti.Vector([0.0 for _ in ti.static(range(self.ps.dim))])
            wsum = 0.0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                cell = center_cell + offset
                valid = 1
                for d in ti.static(range(self.ps.dim)):
                    if cell[d] < 0 or cell[d] >= self.ps.grid_num[d]:
                        valid = 0
                if valid == 0:
                    continue
                flat = self._vortex_ghost_flatten_cell(cell)
                if flat < 0 or flat >= self.ps.flattened_grid_num:
                    continue
                count = self.vortex_ghost_grid_count[flat]
                if count > ti.static(self._vortex_ghost_grid_capacity):
                    count = ti.static(self._vortex_ghost_grid_capacity)
                for slot in range(count):
                    g = self.vortex_ghost_grid_indices[flat, slot]
                    r = self.ps.x[p] - self.vortex_ghost_x[g]
                    rn = r.norm()
                    if rn < support_radius:
                        w = self.vortex_ghost_mV[g] * self._vortex_ghost_cubic_kernel(rn, support_radius)
                        v_acc += w * self.vortex_ghost_v[g]
                        wsum += w
            if wsum > 1e-12:
                self.u_vortex_ghost_sph[p] = v_acc / wsum

    @ti.kernel
    def _add_cached_vortex_ghost_velocity_to_fluid(self, beta: float, mf: ti.i32):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == mf:
                self.ps.v[p] += beta * self.u_vortex_ghost_sph[p]

    @ti.kernel
    def _apply_cached_vortex_ghost_displacement_to_fluid(self, beta: float, mf: ti.i32):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == mf:
                self.ps.x[p] += self.dt[None] * beta * self.u_vortex_ghost_sph[p]

    @ti.kernel
    def _add_vortex_ghost_velocity_to_fluid(self, beta: float, mf: ti.i32, support_radius: float):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] != mf:
                continue
            center_cell = self.ps.pos_to_index(self.ps.x[p])
            v_acc = ti.Vector([0.0 for _ in ti.static(range(self.ps.dim))])
            wsum = 0.0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                cell = center_cell + offset
                valid = 1
                for d in ti.static(range(self.ps.dim)):
                    if cell[d] < 0 or cell[d] >= self.ps.grid_num[d]:
                        valid = 0
                if valid == 0:
                    continue
                flat = self._vortex_ghost_flatten_cell(cell)
                if flat < 0 or flat >= self.ps.flattened_grid_num:
                    continue
                count = self.vortex_ghost_grid_count[flat]
                if count > ti.static(self._vortex_ghost_grid_capacity):
                    count = ti.static(self._vortex_ghost_grid_capacity)
                for slot in range(count):
                    g = self.vortex_ghost_grid_indices[flat, slot]
                    r = self.ps.x[p] - self.vortex_ghost_x[g]
                    rn = r.norm()
                    if rn < support_radius:
                        w = self.vortex_ghost_mV[g] * self._vortex_ghost_cubic_kernel(rn, support_radius)
                        v_acc += w * self.vortex_ghost_v[g]
                        wsum += w
            if wsum > 1e-12:
                self.ps.v[p] += beta * (v_acc / wsum)

    @ti.kernel
    def _add_segment_bs_to_fluid_velocity(self, beta: float, mf: ti.i32):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == mf:
                for d in ti.static(range(self.ps.dim)):
                    self.ps.v[p][d] += beta * self.u_seg_bs[p][d]

    def _print_sph_vorticity_debug(self):
        if not self._sph_vort_debug:
            return
        step = int(self.seg_solver._sim_step_index)
        if step % self._sph_vort_debug_interval != 0:
            return
        n = int(self.ps.particle_num[None])
        if n <= 0:
            return
        mat = self.ps.material.to_numpy()[:n]
        fluid = mat == int(self.ps.material_fluid)
        if not np.any(fluid):
            return
        vort = self.ps.vorticity.to_numpy()[:n].astype(np.float32, copy=False)[fluid]
        vort_vis = self.ps.vorticity_vis.to_numpy()[:n].astype(np.float32, copy=False)[fluid]
        raw_norm = np.linalg.norm(vort, axis=1)
        vis_norm = np.linalg.norm(vort_vis, axis=1)
        raw_z_abs = np.abs(vort[:, 2])
        vis_z_abs = np.abs(vort_vis[:, 2])
        i_vis = int(np.argmax(vis_z_abs))
        max_vis_z = float(vis_z_abs[i_vis])
        signed_vis_z = float(vort_vis[i_vis, 2])
        max_raw_z = float(np.max(raw_z_abs))
        max_vis_norm = float(np.max(vis_norm))
        max_raw_norm = float(np.max(raw_norm))
        print(
            f"[SPH vorticity] step={step} "
            f"max|vorticity_vis.z|={max_vis_z:.6e} "
            f"signed={signed_vis_z:.6e} "
            f"max|vorticity.z|={max_raw_z:.6e} "
            f"max|vorticity_vis|={max_vis_norm:.6e} "
            f"max|vorticity|={max_raw_norm:.6e}"
        )

    def _debug_print_sph_ghost_velocity(self, stage: str, particle_id=None, v_before=None):
        if not bool(self.seg_cfg.get_cfg("vortexGhostVelocityDebug", False)):
            return
        step = int(self.seg_solver._sim_step_index)
        interval = max(1, int(self.seg_cfg.get_cfg("vortexGhostVelocityDebugInterval", 1)))
        if step % interval != 0:
            return
        n = int(self.ps.particle_num[None])
        if n <= 0:
            return
        p = particle_id
        if p is None or int(p) < 0 or int(p) >= n:
            cfg_p = self.seg_cfg.get_cfg("vortexGhostVelocityDebugParticle", None)
            if cfg_p is not None:
                p = int(cfg_p)
            else:
                p = int(self._debug_last_residual_particle)
        if p is None or int(p) < 0 or int(p) >= n:
            return
        p = int(p)
        x_p = self.ps.x.to_numpy()[p].astype(np.float64)
        v_now = self.ps.v.to_numpy()[p].astype(np.float64)
        support = float(self.ps.support_radius * self._vortex_ghost_support_radius_scale)
        ng = int(self.vortex_ghost_num[None])
        ghost_v = np.zeros((int(self.ps.dim),), dtype=np.float64)
        wsum = 0.0
        near_count = 0
        if ng > 0:
            xg = self.vortex_ghost_x.to_numpy()[:ng].astype(np.float64)
            vg = self.vortex_ghost_v.to_numpy()[:ng].astype(np.float64)
            mv = self.vortex_ghost_mV.to_numpy()[:ng].astype(np.float64)
            r = x_p[: int(self.ps.dim)][None, :] - xg[:, : int(self.ps.dim)]
            rn = np.linalg.norm(r, axis=1)
            near = (rn < support) & np.isfinite(mv) & (mv > 0.0)
            near_count = int(np.sum(near))
            if near_count > 0:
                w = mv[near] * self._cubic_kernel_np(rn[near], support)
                wsum = float(np.sum(w))
                if wsum > 1e-12:
                    ghost_v = np.sum(w[:, None] * vg[near, : int(self.ps.dim)], axis=0) / wsum
        beta = float(self._vortex_ghost_beta)
        cached_v = self.u_vortex_ghost_sph.to_numpy()[p].astype(np.float64)[: int(self.ps.dim)]
        velocity_source = str(
            self.seg_cfg.get_cfg("vortexGhostFeedbackVelocitySource", "direct_bs")
            or "direct_bs"
        ).lower().strip()
        if velocity_source in ("direct_bs", "bs", "gamma_bs", "point_vortex_bs") and self.seg_solver._bs_2d_point:
            ghost_v = cached_v
        beta_ghost = beta * ghost_v
        before_s = ""
        if v_before is not None:
            vb = np.asarray(v_before, dtype=np.float64)[: int(self.ps.dim)]
            dv = v_now[: int(self.ps.dim)] - vb
            before_s = (
                f" v_before=({float(vb[0]):.6e},{float(vb[1]):.6e})"
                f" dv_actual=({float(dv[0]):.6e},{float(dv[1]):.6e})"
            )
        print(
            f"[ghost-velocity-debug] step={step} stage={stage} particle={p} "
            f"pos=({float(x_p[0]):.4f},{float(x_p[1]):.4f}) "
            f"near_ghost={near_count} wsum={wsum:.6e} beta={beta:.6e} "
            f"ghost_v=({float(ghost_v[0]):.6e},{float(ghost_v[1]):.6e}) "
            f"beta_ghost=({float(beta_ghost[0]):.6e},{float(beta_ghost[1]):.6e}) "
            f"sph_v=({float(v_now[0]):.6e},{float(v_now[1]):.6e})"
            f"{before_s}"
        )

    def _apply_segment_feedback_to_sph(self):
        if self._vortex_ghost_feedback:
            skip_type = int(self._vortex_ghost_skip_type if self._vortex_ghost_skip_boundary else -1)
            velocity_source = str(
                self.seg_cfg.get_cfg("vortexGhostFeedbackVelocitySource", "direct_bs")
                or "direct_bs"
            ).lower().strip()
            direct_bs_source = (
                velocity_source in ("direct_bs", "bs", "gamma_bs", "point_vortex_bs")
                and self.seg_solver._bs_2d_point
            )
            copy_ghost_velocity = 0 if direct_bs_source else 1
            self._sync_vortex_ghost_particles_from_segments(skip_type, copy_ghost_velocity)
            self._build_vortex_ghost_grid()
            v_before = None
            dbg_p = int(self._debug_last_residual_particle)
            support = float(self.ps.support_radius * self._vortex_ghost_support_radius_scale)
            if direct_bs_source:
                self._compute_vortex_ghost_bs_velocity_for_fluid(
                    int(self.ps.material_fluid), support, float(self.seg_solver.reg_radius)
                )
            else:
                self._compute_vortex_ghost_velocity_for_fluid(
                    int(self.ps.material_fluid), support
                )
            if bool(self.seg_cfg.get_cfg("vortexGhostVelocityDebug", False)) and dbg_p >= 0:
                n_dbg = int(self.ps.particle_num[None])
                if dbg_p < n_dbg:
                    v_before = self.ps.v.to_numpy()[dbg_p].copy()
                    self._debug_print_sph_ghost_velocity("before_feedback", dbg_p)
            if self._vortex_ghost_velocity_feedback:
                self._add_cached_vortex_ghost_velocity_to_fluid(
                    float(self._vortex_ghost_beta), int(self.ps.material_fluid)
                )
                if v_before is not None:
                    self._debug_print_sph_ghost_velocity("after_feedback", dbg_p, v_before=v_before)
            else:
                self._debug_print_sph_ghost_velocity("cached_displacement_feedback", dbg_p)
            if self._one_step_vortex_impulse:
                self._clear_one_step_internal_point_vortices()
        elif self._feedback_mode in ("bs_direct", "direct_bs", "biot_savart", "bs"):
            self.seg_solver.accumulate_bs_velocity_at_fluid_particles(
                self.ps,
                self.u_seg_bs,
                float(self.seg_solver.reg_radius),
                int(self._bs_feedback_skip_type if self._bs_feedback_skip_boundary else -999999),
            )
            self._add_segment_bs_to_fluid_velocity(
                float(self.segment_bs_coupling_beta),
                int(self.ps.material_fluid),
            )

    def _apply_vortex_ghost_displacement_feedback(self):
        if not self._vortex_ghost_displacement_feedback:
            return
        self._apply_cached_vortex_ghost_displacement_to_fluid(
            float(self._vortex_ghost_beta), int(self.ps.material_fluid)
        )
        self._debug_print_sph_ghost_velocity(
            "after_displacement", int(self._debug_last_residual_particle)
        )

    def substep(self):
        timing = self._substep_timing_enabled and (
            int(self.seg_solver._sim_step_index) % self._substep_timing_interval == 0
        )
        records = []

        def sync_if_needed():
            if timing and self._substep_timing_sync:
                ti.sync()

        def run_stage(name, fn):
            if not timing:
                fn()
                return
            sync_if_needed()
            t0 = time.perf_counter()
            fn()
            sync_if_needed()
            records.append((name, (time.perf_counter() - t0) * 1000.0))

        run_stage("compute_densities", self.compute_densities)
        run_stage("compute_DFSPH_factor", self.compute_DFSPH_factor)
        run_stage("divergence_solve", self.divergence_solve)
        run_stage("compute_non_pressure_forces", self.compute_non_pressure_forces)
        run_stage("predict_velocity", self.predict_velocity)
        run_stage("compute_vorticity_pre", self.compute_vorticity)
        run_stage("compute_vorticity_vis_pre", self.compute_vorticity_vis)
        run_stage("print_sph_vorticity_debug", self._print_sph_vorticity_debug)
        run_stage("advance_segments_coupled", self._advance_segments_coupled)
        run_stage("apply_segment_feedback_to_sph", self._apply_segment_feedback_to_sph)
        run_stage("pressure_solve", self.pressure_solve)
        run_stage(
            "ghost_velocity_debug_after_pressure",
            lambda: self._debug_print_sph_ghost_velocity(
                "after_pressure", int(self._debug_last_residual_particle)
            ),
        )
        run_stage("compute_vorticity_post", self.compute_vorticity)
        run_stage("compute_vorticity_vis_post", self.compute_vorticity_vis)
        run_stage("copy_x_temp", self.copy_x_temp)
        run_stage("advect", self.advect)
        run_stage("apply_vortex_ghost_displacement", self._apply_vortex_ghost_displacement_feedback)

        if timing:
            total = sum(v for _, v in records)
            parts = " ".join(f"{name}={ms:.3f}ms" for name, ms in records)
            print(
                f"[hybrid-substep-timing] step={int(self.seg_solver._sim_step_index)} "
                f"total={total:.3f}ms {parts}"
            )

    @ti.kernel
    def _compute_segment_center_vort_vis_z(self, mf: ti.i32):
        """段心处 m_V*W 加权平均的 vorticity_vis.z，与粒子 PNG 上色标量一致。"""
        h = self.ps.support_radius
        nseg = self.ss_seg.segment_num[None]
        for i in range(self.ss_seg.segment_max_num):
            self._seg_vis_scalar[i] = 0.0
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
                continue
            xc = self.ss_seg.center[i]
            acc = 0.0
            wsum = 0.0
            for p in range(self.ps.particle_num[None]):
                if self.ps.material[p] != mf:
                    continue
                r = xc - self.ps.x[p]
                rn = r.norm()
                if rn >= h:
                    continue
                w = self.ps.m_V[p] * self.cubic_kernel(rn)
                acc += w * self.ps.vorticity_vis[p][2]
                wsum += w
            if wsum > 1e-12:
                self._seg_vis_scalar[i] = acc / wsum

    def _boundary_segment_draw_style(self):
        """边界虚拟段 PNG 样式（与 segment_export 配置项一致）。"""
        btype = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
        show = bool(self.seg_cfg.get_cfg("imageShowBoundarySegments", True))
        raw = self.seg_cfg.get_cfg("imageBoundarySegmentColor", [255, 165, 0])
        if raw is None:
            raw = [255, 165, 0]
        arr = np.asarray(raw, dtype=np.float64).ravel()
        if arr.size >= 3 and float(np.max(arr)) > 1.0:
            color = (arr[:3] / 255.0).astype(np.float64)
        else:
            color = arr[:3] if arr.size >= 3 else np.array([1.0, 0.65, 0.0])
        lw_cfg = self.seg_cfg.get_cfg("imageBoundarySegmentLineWidth", None)
        lw = float(lw_cfg) if lw_cfg is not None else self._export_seg_line_width * 1.35
        alpha = float(self.seg_cfg.get_cfg("imageBoundarySegmentAlpha", 0.95))
        return btype, show, tuple(color.tolist()), lw, alpha

    def _active_segment_index_groups(self, n_seg: int):
        """将活跃段分为内部段与边界虚拟段索引。"""
        if n_seg <= 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
        act = self.ss_seg.active.to_numpy()[:n_seg]
        idx = np.nonzero(act == 1)[0]
        if idx.size == 0:
            return idx, idx
        btype, show_bnd, _, _, _ = self._boundary_segment_draw_style()
        if not show_bnd:
            return idx, np.zeros((0,), dtype=np.int64)
        seg_type = self.ss_seg.seg_type.to_numpy()[:n_seg]
        is_bnd = seg_type[idx] == btype
        return idx[~is_bnd], idx[is_bnd]

    def _segment_line_width_for_export(self) -> float:
        return float(
            self.seg_cfg.get_cfg("imageSegmentLineWidth", self._export_seg_line_width)
        )

    def _segment_gamma_color_norm(self, gamma_values=None):
        auto = bool(self.seg_cfg.get_cfg("imageSegmentGammaAutoScale", False))
        if auto and gamma_values is not None and len(gamma_values) > 0:
            g = np.asarray(gamma_values, dtype=np.float64)
            vmax_abs = float(np.percentile(np.abs(g), 98))
            if vmax_abs <= 1e-12:
                vmax_abs = 1.0
            return Normalize(vmin=-vmax_abs, vmax=vmax_abs)
        vmin = float(self.seg_cfg.get_cfg("imageSegmentGammaVmin", -1.0e-3))
        vmax = float(self.seg_cfg.get_cfg("imageSegmentGammaVmax", 1.0e-3))
        if vmax <= vmin + 1e-12:
            vmax = vmin + 1.0
        return Normalize(vmin=vmin, vmax=vmax)

    def _parse_rgb_cfg(self, key: str, default):
        raw = self.seg_cfg.get_cfg(key, default)
        if raw is None:
            raw = default
        arr = np.asarray(raw, dtype=np.float64).ravel()
        if arr.size >= 3 and float(np.max(arr[:3])) > 1.0:
            arr = arr[:3] / 255.0
        else:
            arr = arr[:3] if arr.size >= 3 else np.asarray(default, dtype=np.float64)[:3]
        return tuple(arr.tolist())

    def _add_vortex_ghost_overlay_2d(self, ax, force: bool = False):
        show = self.seg_cfg.get_cfg("imageShowVortexGhostParticles", None)
        if show is None:
            show = self._vortex_ghost_feedback
        if not force and not bool(show):
            return
        n = int(self.vortex_ghost_num[None])
        if n <= 0:
            return
        xg = self.vortex_ghost_x.to_numpy()[:n]
        vg = self.vortex_ghost_v.to_numpy()[:n]
        mv = self.vortex_ghost_mV.to_numpy()[:n]
        valid = np.isfinite(xg[:, 0]) & np.isfinite(xg[:, 1]) & (mv > 0.0)
        if not np.any(valid):
            return
        xg = xg[valid]
        vg = vg[valid]
        mv = mv[valid]
        dot_color = self._parse_rgb_cfg("imageVortexGhostParticleColor", [255, 0, 255])
        arrow_color = self._parse_rgb_cfg("imageVortexGhostArrowColor", [0, 0, 0])
        alpha = float(self.seg_cfg.get_cfg("imageVortexGhostParticleAlpha", 0.9))
        radius_scale = float(self.seg_cfg.get_cfg("imageVortexGhostParticleRadiusScale", 1.0))
        min_radius = float(self.seg_cfg.get_cfg("imageVortexGhostParticleMinRadius", 0.0))
        edge_lw = float(self.seg_cfg.get_cfg("imageVortexGhostParticleEdgeLineWidth", 0.35))
        for p, vol in zip(xg, mv):
            r = radius_scale * float(np.sqrt(max(float(vol), 0.0) / np.pi))
            r = max(r, min_radius)
            if r <= 0.0:
                continue
            ax.add_patch(
                Circle(
                    (float(p[0]), float(p[1])),
                    r,
                    facecolor=to_rgba(dot_color, alpha),
                    edgecolor=to_rgba((1.0, 1.0, 1.0), min(1.0, alpha)),
                    linewidth=edge_lw,
                    zorder=8,
                )
            )
        if not bool(self.seg_cfg.get_cfg("imageShowVortexGhostVelocityArrows", True)):
            return
        speed = np.linalg.norm(vg[:, :2], axis=1)
        moving = speed > 1e-12
        if not np.any(moving):
            return
        arrow_len = float(self.seg_cfg.get_cfg("imageVortexGhostArrowLength", 0.035))
        arrow_scale_by_speed = bool(self.seg_cfg.get_cfg("imageVortexGhostArrowScaleBySpeed", False))
        vv = vg[moving, :2].astype(np.float64, copy=True)
        pp = xg[moving, :2].astype(np.float64, copy=False)
        ss = speed[moving].astype(np.float64, copy=False)
        dirs = vv / (ss[:, None] + 1e-12)
        lengths = np.full_like(ss, arrow_len)
        if arrow_scale_by_speed:
            ref = float(self.seg_cfg.get_cfg("imageVortexGhostArrowSpeedRef", np.percentile(ss, 90)))
            if ref <= 1e-12:
                ref = 1.0
            lengths = arrow_len * np.minimum(ss / ref, 2.0)
        vec = dirs * lengths[:, None]
        ax.quiver(
            pp[:, 0],
            pp[:, 1],
            vec[:, 0],
            vec[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=arrow_color,
            alpha=float(self.seg_cfg.get_cfg("imageVortexGhostArrowAlpha", 0.95)),
            width=float(self.seg_cfg.get_cfg("imageVortexGhostArrowWidth", 0.003)),
            zorder=9,
        )

    def _add_sph_ghost_bs_velocity_overlay_2d(self, ax):
        if not bool(self.seg_cfg.get_cfg("imageShowSphGhostBsVelocityArrows", False)):
            return
        n = int(self.ps.particle_num[None])
        if n <= 0:
            return
        material = self.ps.material.to_numpy()[:n]
        fluid = material == int(self.ps.material_fluid)
        if not np.any(fluid):
            return
        x = self.x_temp.to_numpy()[:n]
        u = self.u_vortex_ghost_sph.to_numpy()[:n]
        x = x[fluid, :2].astype(np.float64, copy=False)
        u = u[fluid, :2].astype(np.float64, copy=False)
        speed = np.linalg.norm(u, axis=1)
        min_speed = float(self.seg_cfg.get_cfg("imageSphGhostBsArrowMinSpeed", 1e-8))
        valid = np.isfinite(x[:, 0]) & np.isfinite(x[:, 1]) & np.isfinite(speed) & (speed > min_speed)
        if not np.any(valid):
            return
        x = x[valid]
        u = u[valid]
        speed = speed[valid]
        stride = max(1, int(self.seg_cfg.get_cfg("imageSphGhostBsArrowStride", 16)))
        if stride > 1:
            x = x[::stride]
            u = u[::stride]
            speed = speed[::stride]
        max_count = int(self.seg_cfg.get_cfg("imageSphGhostBsArrowMaxCount", 2500))
        if max_count > 0 and x.shape[0] > max_count:
            ids = np.linspace(0, x.shape[0] - 1, max_count).astype(np.int64)
            x = x[ids]
            u = u[ids]
            speed = speed[ids]
        if x.shape[0] == 0:
            return
        arrow_len = float(self.seg_cfg.get_cfg("imageSphGhostBsArrowLength", 0.025))
        scale_by_speed = bool(self.seg_cfg.get_cfg("imageSphGhostBsArrowScaleBySpeed", False))
        dirs = u / (speed[:, None] + 1e-12)
        lengths = np.full_like(speed, arrow_len)
        if scale_by_speed:
            ref = float(self.seg_cfg.get_cfg("imageSphGhostBsArrowSpeedRef", np.percentile(speed, 90)))
            if ref <= 1e-12:
                ref = 1.0
            lengths = arrow_len * np.minimum(speed / ref, 2.0)
        vec = dirs * lengths[:, None]
        ax.quiver(
            x[:, 0],
            x[:, 1],
            vec[:, 0],
            vec[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=self._parse_rgb_cfg("imageSphGhostBsArrowColor", [0, 255, 0]),
            alpha=float(self.seg_cfg.get_cfg("imageSphGhostBsArrowAlpha", 0.85)),
            width=float(self.seg_cfg.get_cfg("imageSphGhostBsArrowWidth", 0.0015)),
            zorder=10,
        )

    def _export_vortex_ghost_png_2d(self, cnt: int, dir_ghost: Path, ds, de):
        if not self._export_ghost_panel:
            return
        dir_ghost.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=200)
        bg = self.seg_cfg.get_cfg("imageVortexGhostPanelBackgroundColor", [20, 20, 30])
        bg_arr = np.asarray(bg, dtype=np.float64).ravel()
        if bg_arr.size >= 3 and float(np.max(bg_arr[:3])) > 1.0:
            bg_arr = bg_arr[:3] / 255.0
        else:
            bg_arr = bg_arr[:3] if bg_arr.size >= 3 else np.array([0.08, 0.08, 0.12])
        fig.patch.set_facecolor(bg_arr)
        ax.set_facecolor(bg_arr)
        if bool(self.seg_cfg.get_cfg("imageVortexGhostPanelIncludeSph", True)):
            N = int(self.ps.particle_num[None])
            if N > 0:
                material = self.ps.material.to_numpy()[:N]
                obj_id = self.ps.object_id.to_numpy()[:N]
                fluid_mask = material == self.ps.material_fluid
                solid_mask = (obj_id == 1) | (obj_id == 2)
                x = self.x_temp.to_numpy()[:N]
                vort_np = self.ps.vorticity_vis.to_numpy()[:N]
                vort = vort_np[:, 2] if vort_np.shape[1] > 2 else vort_np[:, 0]
                fluid_x = x[fluid_mask]
                fluid_vort = vort[fluid_mask]
                solid_x = x[solid_mask]
                _auto = self.ps.cfg.get_cfg("imageVorticityAutoScale")
                if _auto and fluid_vort.size > 0:
                    vmin = float(np.percentile(fluid_vort, 2))
                    vmax = float(np.percentile(fluid_vort, 98))
                    if vmax <= vmin + 1e-12:
                        vmax = vmin + 1.0
                else:
                    _vmin = self.ps.cfg.get_cfg("imageVorticityVmin")
                    _vmax = self.ps.cfg.get_cfg("imageVorticityVmax")
                    vmin = float(_vmin if _vmin is not None else self._img_vort_vmin)
                    vmax = float(_vmax if _vmax is not None else self._img_vort_vmax)
                vnorm = Normalize(vmin=vmin, vmax=vmax)
                pt_size = float(
                    self.seg_cfg.get_cfg(
                        "imageVortexGhostPanelSphPointSize",
                        self.ps.cfg.get_cfg("imageFluidPointSize") or 1.0,
                    )
                )
                pt_alpha = float(
                    self.seg_cfg.get_cfg(
                        "imageVortexGhostPanelSphAlpha",
                        self.ps.cfg.get_cfg("imageFluidAlpha") or 1.0,
                    )
                )
                if fluid_x.shape[0] > 0:
                    ax.scatter(
                        fluid_x[:, 0],
                        fluid_x[:, 1],
                        c=fluid_vort,
                        cmap=self._segment_cmap,
                        s=pt_size,
                        norm=vnorm,
                        edgecolors="none",
                        alpha=pt_alpha,
                        zorder=1,
                    )
                if bool(self.seg_cfg.get_cfg("imageVortexGhostPanelIncludeSolid", True)) and solid_x.shape[0] > 0:
                    ax.scatter(
                        solid_x[:, 0],
                        solid_x[:, 1],
                        color="#00C853",
                        s=float(self.seg_cfg.get_cfg("imageVortexGhostPanelSolidPointSize", 1.0)),
                        edgecolors="none",
                        zorder=2,
                    )
        self._add_vortex_ghost_overlay_2d(ax, force=True)
        self._add_sph_ghost_bs_velocity_overlay_2d(ax)
        ax.set_xlim(float(ds[0]), float(de[0]))
        ax.set_ylim(float(ds[1]), float(de[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        transparent = bool(self.seg_cfg.get_cfg("imageVortexGhostPanelTransparent", False))
        plt.savefig(
            dir_ghost / f"vorticity_{cnt:04}.png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=transparent,
            dpi=400,
            facecolor=fig.get_facecolor() if not transparent else "none",
        )
        plt.close("all")

    def _add_point_vortices_2d(self, ax, centers, idx, scalars, vnorm):
        if idx.size == 0:
            return
        volumes = self.seg_solver.point_vortex_volume.to_numpy()
        seg_alpha = float(self.seg_cfg.get_cfg("imageSegmentLineAlpha", 0.92))
        min_radius = float(self.seg_cfg.get_cfg("imagePointVortexMinRadius", 0.0))
        radius_scale = float(self.seg_cfg.get_cfg("imagePointVortexRadiusScale", 1.0))
        draw_underlay = bool(self.seg_cfg.get_cfg("imageSegmentDrawWhiteUnderlay", False))
        under_scale = float(self.seg_cfg.get_cfg("imagePointVortexWhiteUnderlayRadiusScale", 1.18))
        under_a = float(self.seg_cfg.get_cfg("imageSegmentWhiteUnderlayAlpha", 0.55))
        cmap = plt.get_cmap(self._segment_cmap)
        for local, i in enumerate(idx):
            cx, cy = float(centers[i, 0]), float(centers[i, 1])
            if not np.isfinite([cx, cy]).all():
                continue
            vol = max(float(volumes[i]), 0.0) if i < volumes.shape[0] else 0.0
            radius = radius_scale * float(np.sqrt(vol / np.pi)) if vol > 0.0 else min_radius
            radius = max(radius, min_radius)
            if radius <= 0.0:
                continue
            if draw_underlay:
                ax.add_patch(
                    Circle(
                        (cx, cy),
                        radius * under_scale,
                        facecolor=(1.0, 1.0, 1.0, under_a),
                        edgecolor="none",
                        zorder=3,
                    )
                )
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius,
                    facecolor=to_rgba(cmap(vnorm(float(scalars[local]))), seg_alpha),
                    edgecolor="none",
                    zorder=4,
                )
            )

    def _add_interior_segments_2d(self, ax, xm, xp, idx, scalars, vnorm):
        if idx.size == 0:
            return
        segments = self._segment_polylines_2d(xm, xp, idx)
        lw = self._segment_line_width_for_export()
        seg_alpha = float(self.seg_cfg.get_cfg("imageSegmentLineAlpha", 0.92))
        if bool(self.seg_cfg.get_cfg("imageSegmentDrawWhiteUnderlay", False)):
            under_lw = float(
                self.seg_cfg.get_cfg(
                    "imageSegmentWhiteUnderlayLineWidth", lw * 1.35
                )
            )
            under_a = float(self.seg_cfg.get_cfg("imageSegmentWhiteUnderlayAlpha", 0.55))
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=(1.0, 1.0, 1.0, under_a),
                    linewidths=under_lw,
                    zorder=3,
                    capstyle="round",
                )
            )
        lc = LineCollection(
            segments,
            array=scalars.astype(np.float64),
            cmap=self._segment_cmap,
            norm=vnorm,
            linewidths=lw,
            alpha=seg_alpha,
            zorder=4,
            capstyle="round",
        )
        ax.add_collection(lc)

    def _segment_polylines_2d(self, xm, xp, idx):
        """每条段为 (2,2) 折线；兼容 Taichi 2D 端点 (n,2)。"""
        segs = np.stack([xm[idx], xp[idx]], axis=1)
        if segs.shape[-1] > 2:
            segs = segs[:, :, :2]
        return [np.asarray(s, dtype=np.float64) for s in segs]

    def _add_boundary_segments_2d(self, ax, xm, xp, idx):
        if idx.size == 0:
            return
        _, _, color, lw, alpha = self._boundary_segment_draw_style()
        # 用 ax.plot 逐条绘制，避免 LineCollection 单色/裁剪导致边界段不可见
        for i in idx:
            x0, y0 = float(xm[i, 0]), float(xm[i, 1])
            x1, y1 = float(xp[i, 0]), float(xp[i, 1])
            if not (np.isfinite([x0, y0, x1, y1]).all()):
                continue
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=color,
                linewidth=lw,
                alpha=alpha,
                zorder=6,
                solid_capstyle="round",
            )

    def _log_segment_export_stats(
        self, cnt: int, n_seg: int, idx_int, idx_bnd, xm, xp
    ):
        if not bool(
            self.seg_cfg.get_cfg("boundaryInjectionLog", False)
            or self.seg_cfg.get_cfg("imageExportSegmentDebug", False)
        ):
            return
        if n_seg <= 0:
            print(f"[export segments] frame {cnt:04d}: segment_num=0")
            return
        st = self.ss_seg.seg_type.to_numpy()[:n_seg]
        act = self.ss_seg.active.to_numpy()[:n_seg]
        btype = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
        msg = (
            f"[export segments] frame {cnt:04d}: n_seg={n_seg} active={int(np.sum(act == 1))} "
            f"draw_interior={idx_int.size} draw_boundary={idx_bnd.size} "
            f"type_counts={{0:{int(np.sum(st == 0))}, {btype}:{int(np.sum(st == btype))}}}"
        )
        if idx_bnd.size > 0:
            ii = idx_bnd
            c = 0.5 * (xm[ii] + xp[ii])
            Ls = np.linalg.norm(xp[ii] - xm[ii], axis=1)
            msg += (
                f" | bbox x=[{float(c[:, 0].min()):.3f},{float(c[:, 0].max()):.3f}]"
                f" y=[{float(c[:, 1].min()):.3f},{float(c[:, 1].max()):.3f}]"
                f" |L|_med={float(np.median(Ls)):.4f}"
            )
        print(msg)

    def _add_interior_segments_3d(self, ax, xm, xp, idx, scalars, vnorm):
        if idx.size == 0:
            return
        seg_xyz = np.stack([xm[idx], xp[idx]], axis=1)
        lc = Line3DCollection(
            list(seg_xyz),
            array=scalars.astype(np.float64),
            cmap=self._segment_cmap,
            norm=vnorm,
            linewidths=self._export_seg_line_width,
            alpha=0.92,
        )
        ax.add_collection3d(lc)

    def _add_boundary_segments_3d(self, ax, xm, xp, idx):
        if idx.size == 0:
            return
        _, _, color, lw, alpha = self._boundary_segment_draw_style()
        seg_xyz = [np.asarray(s, dtype=np.float64) for s in np.stack([xm[idx], xp[idx]], axis=1)]
        rgba = to_rgba(color, alpha)
        lc = Line3DCollection(
            seg_xyz,
            colors=[rgba] * len(seg_xyz),
            linewidths=lw,
        )
        ax.add_collection3d(lc)

    def export_png(self, cnt, image_path):
        """SPH 与涡段分目录导出；涡段颜色 = 段心处与粒子相同的 m_V*W 加权 vorticity_vis.z。"""
        if self.ps.dim == 2:
            self._export_png_2d(cnt, image_path)
            return

        path_base = Path(image_path)
        dir_sph = path_base / self._export_sph_subdir
        dir_seg = path_base / self._export_seg_subdir
        dir_sph.mkdir(parents=True, exist_ok=True)
        dir_seg.mkdir(parents=True, exist_ok=True)

        vnorm = Normalize(
            vmin=self._img_vort_vmin, vmax=self._img_vort_vmax
        )

        N = self.ps.particle_num[None]
        material = self.ps.material.to_numpy()[:N]
        obj_id = self.ps.object_id.to_numpy()[:N]
        fluid_mask = material == self.ps.material_fluid
        solid_mask = (obj_id == 1) | (obj_id == 2)

        x = self.x_temp.to_numpy()[:N]
        vort = self.ps.vorticity_vis.to_numpy()[:N][:, 2]

        fluid_x = x[fluid_mask]
        fluid_vort = vort[fluid_mask]
        solid_x = x[solid_mask]

        # --- [EXPORT] SPH 粒子图（无涡段）---
        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=-60)
        ax.scatter(
            fluid_x[:, 0],
            fluid_x[:, 1],
            fluid_x[:, 2],
            c=fluid_vort,
            cmap=self._segment_cmap,
            s=0.2,
            norm=vnorm,
            edgecolors="none",
            alpha=0.6,
        )
        ax.scatter(
            solid_x[:, 0],
            solid_x[:, 1],
            solid_x[:, 2],
            color="#00C853",
            s=0.5,
            edgecolors="none",
        )
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_box_aspect((4, 1, 1))
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            dir_sph / f"vorticity_{cnt:04}.png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=400,
        )
        plt.cla()
        plt.close("all")

        # --- [EXPORT] 涡段图（按段心涡量 z 分量着色，色标与 SPH 一致）---
        if self._export_segment_panel:
            n_seg = int(self.ss_seg.segment_num[None])
            fig2 = plt.figure(figsize=(10, 4), dpi=200)
            ax2 = fig2.add_subplot(111, projection="3d")
            ax2.view_init(elev=30, azim=-60)
            if self._segment_panel_include_solid and solid_x.shape[0] > 0:
                ax2.scatter(
                    solid_x[:, 0],
                    solid_x[:, 1],
                    solid_x[:, 2],
                    color="#00C853",
                    s=0.5,
                    edgecolors="none",
                )
            if n_seg > 0:
                xm = self.ss_seg.x_minus.to_numpy()[:n_seg]
                xp = self.ss_seg.x_plus.to_numpy()[:n_seg]
                idx_int, idx_bnd = self._active_segment_index_groups(n_seg)
                if idx_int.size > 0:
                    gamma_values = self.ss_seg.gamma.to_numpy()[:n_seg]
                    scalars = gamma_values[idx_int].astype(np.float64)
                    seg_norm = self._segment_gamma_color_norm(scalars)
                    self._add_interior_segments_3d(ax2, xm, xp, idx_int, scalars, seg_norm)
                self._add_boundary_segments_3d(ax2, xm, xp, idx_bnd)
            ax2.set_xlim(0, 4)
            ax2.set_ylim(0, 1)
            ax2.set_zlim(0, 1)
            ax2.set_box_aspect((4, 1, 1))
            ax2.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(
                dir_seg / f"vorticity_{cnt:04}.png",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=400,
            )
            plt.cla()
            plt.close("all")

    def _export_png_2d(self, cnt, image_path):
        """2D 卡门涡街：平面散点 + 涡段折线。"""
        path_base = Path(image_path)
        dir_sph = path_base / self._export_sph_subdir
        dir_seg = path_base / self._export_seg_subdir
        dir_ghost = path_base / self._export_ghost_subdir
        dir_sph.mkdir(parents=True, exist_ok=True)
        dir_seg.mkdir(parents=True, exist_ok=True)

        ds = self.ps.domain_start
        de = self.ps.domain_end

        N = self.ps.particle_num[None]
        material = self.ps.material.to_numpy()[:N]
        obj_id = self.ps.object_id.to_numpy()[:N]
        fluid_mask = material == self.ps.material_fluid
        solid_mask = (obj_id == 1) | (obj_id == 2)

        x = self.x_temp.to_numpy()[:N]
        vort_np = self.ps.vorticity_vis.to_numpy()[:N]
        vort = vort_np[:, 2] if vort_np.shape[1] > 2 else vort_np[:, 0]

        fluid_x = x[fluid_mask]
        fluid_vort = vort[fluid_mask]
        solid_x = x[solid_mask]

        _auto = self.ps.cfg.get_cfg("imageVorticityAutoScale")
        if _auto and fluid_vort.size > 0:
            vmin = float(np.percentile(fluid_vort, 2))
            vmax = float(np.percentile(fluid_vort, 98))
            if vmax <= vmin + 1e-12:
                vmax = vmin + 1.0
        else:
            _vmin = self.ps.cfg.get_cfg("imageVorticityVmin")
            _vmax = self.ps.cfg.get_cfg("imageVorticityVmax")
            vmin = float(_vmin if _vmin is not None else self._img_vort_vmin)
            vmax = float(_vmax if _vmax is not None else self._img_vort_vmax)
        vnorm = Normalize(vmin=vmin, vmax=vmax)

        _pt = self.ps.cfg.get_cfg("imageFluidPointSize")
        _al = self.ps.cfg.get_cfg("imageFluidAlpha")
        pt_size = float(_pt if _pt is not None else 2.5)
        pt_alpha = float(_al if _al is not None else 1.0)

        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=200)
        ax.scatter(
            fluid_x[:, 0], fluid_x[:, 1], c=fluid_vort, cmap=self._segment_cmap,
            s=pt_size, norm=vnorm, edgecolors="none", alpha=pt_alpha,
        )
        if solid_x.shape[0] > 0:
            ax.scatter(
                solid_x[:, 0], solid_x[:, 1], color="#00C853",
                s=1.0, edgecolors="none",
            )
        if bool(self.seg_cfg.get_cfg("imageOverlayVortexGhostOnSph", False)):
            self._add_vortex_ghost_overlay_2d(ax)
        ax.set_xlim(float(ds[0]), float(de[0]))
        ax.set_ylim(float(ds[1]), float(de[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            dir_sph / f"vorticity_{cnt:04}.png",
            bbox_inches="tight", pad_inches=0, transparent=True, dpi=400,
        )
        plt.close("all")
        self._export_vortex_ghost_png_2d(cnt, dir_ghost, ds, de)

        if self._export_segment_panel:
            n_seg = int(self.ss_seg.segment_num[None])
            fig2, ax2 = plt.subplots(figsize=(10, 2.5), dpi=200)
            pad = float(self.seg_cfg.get_cfg("imageSegmentAxisPad", 0.02))
            hide_solid = bool(self.seg_cfg.get_cfg("imageSegmentPanelHideSolid", False))
            draw_solid = (
                self._segment_panel_include_solid
                and solid_x.shape[0] > 0
                and not hide_solid
            )
            seg_only_bg = bool(self.seg_cfg.get_cfg("imageSegmentPanelOpaqueBackground", False))
            if seg_only_bg and hide_solid:
                bg = self.seg_cfg.get_cfg("imageSegmentPanelBackgroundColor", [255, 255, 255])
                if bg is None:
                    bg = [255, 255, 255]
                bg_arr = np.asarray(bg, dtype=np.float64).ravel()
                if bg_arr.size >= 3 and float(np.max(bg_arr)) > 1.0:
                    bg_arr = bg_arr[:3] / 255.0
                else:
                    bg_arr = bg_arr[:3] if bg_arr.size >= 3 else np.array([1.0, 1.0, 1.0])
                fig2.patch.set_facecolor(bg_arr)
                ax2.set_facecolor(bg_arr)
            if draw_solid:
                ax2.scatter(
                    solid_x[:, 0], solid_x[:, 1], color="#00C853",
                    s=0.6, edgecolors="none", zorder=1, alpha=0.55,
                )
            if n_seg > 0:
                xm = self.ss_seg.x_minus.to_numpy()[:n_seg]
                xp = self.ss_seg.x_plus.to_numpy()[:n_seg]
                idx_int, idx_bnd = self._active_segment_index_groups(n_seg)
                if idx_int.size > 0:
                    gamma_values = self.ss_seg.gamma.to_numpy()[:n_seg]
                    scalars = gamma_values[idx_int].astype(np.float64)
                    seg_norm = self._segment_gamma_color_norm(scalars)
                    if getattr(self.seg_solver, "_bs_2d_point", False):
                        centers = 0.5 * (xm + xp)
                        self._add_point_vortices_2d(ax2, centers, idx_int, scalars, seg_norm)
                    else:
                        self._add_interior_segments_2d(ax2, xm, xp, idx_int, scalars, seg_norm)
                self._add_boundary_segments_2d(ax2, xm, xp, idx_bnd)
                self._log_segment_export_stats(cnt, n_seg, idx_int, idx_bnd, xm, xp)
            fit_seg = bool(
                self.seg_cfg.get_cfg("imageSegmentPanelFitToSegments", True)
            )
            use_sim_domain = bool(
                self.seg_cfg.get_cfg("imageSegmentPanelUseSimulationDomain", False)
            )
            if (
                n_seg > 0
                and fit_seg
                and not use_sim_domain
            ):
                draw_idx = idx_bnd if idx_bnd.size > 0 else np.arange(n_seg)
                if draw_idx.size > 0:
                    xs = np.concatenate([xm[draw_idx, 0], xp[draw_idx, 0]])
                    ys = np.concatenate([xm[draw_idx, 1], xp[draw_idx, 1]])
                    if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
                        ax2.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
                        ax2.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
                    else:
                        ax2.set_xlim(float(ds[0]), float(de[0]))
                        ax2.set_ylim(float(ds[1]), float(de[1]))
                else:
                    ax2.set_xlim(float(ds[0]), float(de[0]))
                    ax2.set_ylim(float(ds[1]), float(de[1]))
            else:
                ax2.set_xlim(float(ds[0]), float(de[0]))
                ax2.set_ylim(float(ds[1]), float(de[1]))
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            seg_transparent = not (seg_only_bg and hide_solid)
            plt.savefig(
                dir_seg / f"vorticity_{cnt:04}.png",
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=seg_transparent,
                dpi=400,
                facecolor=fig2.get_facecolor() if not seg_transparent else "none",
            )
            plt.close("all")

    def export_ply(self, cnt, ply_path):
        """导出 SPH 粒子 PLY（父类），并在同一目录写入涡段 ``segments_{cnt:04}.ply``。"""
        super().export_ply(cnt, ply_path)
        if self._export_segment_ply:
            self._segment_exporter.export_segments_ply(cnt, ply_path)
