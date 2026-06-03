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
from pathlib import Path
from matplotlib.colors import Normalize, to_rgba
from matplotlib.collections import LineCollection
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
        self._sph_velocity_to_segment_scale = float(
            self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionScale", 1.0)
        )
        self._sph_velocity_to_segment_blend = float(
            self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionBlend", 1.0)
        )
        self._sph_velocity_to_segment_skip_type = int(
            self.seg_cfg.get_cfg("boundarySegmentTypeId", 2)
        ) if bool(self.seg_cfg.get_cfg("sphVelocityToSegmentAdvectionSkipBoundary", True)) else -1

        self.u_seg_bs = ti.Vector.field(
            3, dtype=float, shape=self.ps.particle_max_num
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

    def _advance_segments_coupled(self):
        ssol = self.seg_solver
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

        ssol._emit_inlet_segments()
        ssol._emit_periodic_parallel_x_layers()
        ssol.ss.update_segment_geometry()
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
        if self._sph_velocity_to_segment_enabled:
            self._deposit_velocity_to_segments_advect_kernel(
                float(self._sph_velocity_to_segment_blend),
                float(self._sph_velocity_to_segment_scale),
                int(self.ps.material_fluid),
                int(self._sph_velocity_to_segment_skip_type),
            )
        ssol.compute_endpoint_velocity()
        ssol.advect_segments_rk4()
        ssol.ss.update_segment_geometry()
        ssol.split_segments()
        ssol.merge_segments()
        ssol.restore_frozen_segment_geometry()
        ssol.delete_weak_segments()
        if ssol._use_leapfrog_initial_impulse[None] == 1:
            ssol._decay_impulse_kernel()
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

    @ti.kernel
    def _deposit_velocity_to_segments_advect_kernel(
        self,
        blend: float,
        scale: float,
        mf: ti.i32,
        skip_seg_type: ti.i32,
    ):
        h = self.ps.support_radius
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
            wsum_m = 0.0
            wsum_p = 0.0
            if ti.static(self.ps.dim == 2):
                vacc_m2 = ti.Vector([0.0, 0.0])
                vacc_p2 = ti.Vector([0.0, 0.0])
                for p in range(self.ps.particle_num[None]):
                    if self.ps.material[p] != mf:
                        continue
                    rm = xm - self.ps.x[p]
                    rnm = rm.norm()
                    if rnm < h:
                        mv_wm = self.ps.m_V[p] * self.cubic_kernel(rnm)
                        vacc_m2 += mv_wm * self.ps.v[p]
                        wsum_m += mv_wm
                    rp = xp - self.ps.x[p]
                    rnp = rp.norm()
                    if rnp < h:
                        mv_wp = self.ps.m_V[p] * self.cubic_kernel(rnp)
                        vacc_p2 += mv_wp * self.ps.v[p]
                        wsum_p += mv_wp
                if wsum_m > 1e-12:
                    v_target_m = scale * (vacc_m2 / wsum_m)
                    self.seg_solver.sph_advect_velocity_minus[i] = (
                        (1.0 - blend) * self.seg_solver.sph_advect_velocity_minus[i]
                        + blend * v_target_m
                    )
                else:
                    self.seg_solver.sph_advect_velocity_minus[i] *= (1.0 - blend)
                if wsum_p > 1e-12:
                    v_target_p = scale * (vacc_p2 / wsum_p)
                    self.seg_solver.sph_advect_velocity_plus[i] = (
                        (1.0 - blend) * self.seg_solver.sph_advect_velocity_plus[i]
                        + blend * v_target_p
                    )
                else:
                    self.seg_solver.sph_advect_velocity_plus[i] *= (1.0 - blend)
            else:
                vacc_m3 = ti.Vector([0.0, 0.0, 0.0])
                vacc_p3 = ti.Vector([0.0, 0.0, 0.0])
                for p in range(self.ps.particle_num[None]):
                    if self.ps.material[p] != mf:
                        continue
                    rm = xm - self.ps.x[p]
                    rnm = rm.norm()
                    if rnm < h:
                        mv_wm = self.ps.m_V[p] * self.cubic_kernel(rnm)
                        vacc_m3 += mv_wm * self.ps.v[p]
                        wsum_m += mv_wm
                    rp = xp - self.ps.x[p]
                    rnp = rp.norm()
                    if rnp < h:
                        mv_wp = self.ps.m_V[p] * self.cubic_kernel(rnp)
                        vacc_p3 += mv_wp * self.ps.v[p]
                        wsum_p += mv_wp
                if wsum_m > 1e-12:
                    v_target_m = scale * (vacc_m3 / wsum_m)
                    self.seg_solver.sph_advect_velocity_minus[i] = (
                        (1.0 - blend) * self.seg_solver.sph_advect_velocity_minus[i]
                        + blend * v_target_m
                    )
                else:
                    self.seg_solver.sph_advect_velocity_minus[i] *= (1.0 - blend)
                if wsum_p > 1e-12:
                    v_target_p = scale * (vacc_p3 / wsum_p)
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

    @ti.kernel
    def _add_segment_bs_to_fluid_velocity(self, beta: float, mf: ti.i32):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == mf:
                for d in ti.static(range(self.ps.dim)):
                    self.ps.v[p][d] += beta * self.u_seg_bs[p][d]

    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        self.divergence_solve()
        self.compute_non_pressure_forces()
        self.predict_velocity()
        self.pressure_solve()

        self.compute_vorticity()
        self.compute_vorticity_vis()
        self._advance_segments_coupled()

        self.seg_solver.accumulate_bs_velocity_at_fluid_particles(
            self.ps,
            self.u_seg_bs,
            float(self.seg_solver.reg_radius),
        )
        self._add_segment_bs_to_fluid_velocity(
            float(self.segment_bs_coupling_beta),
            int(self.ps.material_fluid),
        )

        self.copy_x_temp()
        self.advect()

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
            self._compute_segment_center_vort_vis_z(int(self.ps.material_fluid))
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
                    scalars = self._seg_vis_scalar.to_numpy()[:n_seg][idx_int].astype(
                        np.float64
                    )
                    self._add_interior_segments_3d(ax2, xm, xp, idx_int, scalars, vnorm)
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

        if self._export_segment_panel:
            self._compute_segment_center_vort_vis_z(int(self.ps.material_fluid))
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
                btype = int(self.seg_cfg.get_cfg("boundarySegmentTypeId", 2))
                st = self.ss_seg.seg_type.to_numpy()[:n_seg]
                act = self.ss_seg.active.to_numpy()[:n_seg]
                # 若类型标记异常，仍把全部活跃段按边界色绘制
                if idx_bnd.size == 0 and int(np.sum(act == 1)) > 0:
                    idx_all = np.nonzero(act == 1)[0]
                    if int(np.sum(st[idx_all] == btype)) == 0:
                        idx_bnd = idx_all
                    elif int(np.sum(st[idx_all] == btype)) > 0:
                        idx_bnd = idx_all[st[idx_all] == btype]
                if idx_int.size > 0:
                    scalars = self._seg_vis_scalar.to_numpy()[:n_seg][idx_int].astype(
                        np.float64
                    )
                    self._add_interior_segments_2d(ax2, xm, xp, idx_int, scalars, vnorm)
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
