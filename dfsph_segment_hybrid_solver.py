"""
DFSPH Kármán solver coupled with Lagrangian vortex segments (research prototype).

Pipeline each ``substep`` (same ``dt`` as SPH):
  1) Standard DFSPH through ``pressure_solve``.
  2) ``compute_vorticity`` — SPH estimate ω from velocity (same formulation as DIMCV).
  3) Segment subsystem: emit → geometry → relax γ toward kernel-weighted SPH ω at segment
     centers (projection configurable: ω·t̂ vs ‖ω‖ for random segments) → BS → RK4 → topology.
  4) Biot–Savart induced velocity at fluid particle positions → add ``β u_BS`` to ``ps.v``.

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
  • Segment advection uses BS + ``backgroundVelocity`` only (no SPH interpolation at endpoints yet).
  • For ``initType: random_uniform`` with ``randomGammaInitial: 0``, set ``deleteGammaThreshold: 0``
    so weak segments are not culled before SPH deposits circulation.
  • Requires ``SimConfig.scene_file_path`` and a ``SegmentConfiguration`` block in that JSON.
  • Use ``simulationMethod``: 2 with e.g. ``DIM_von_karman_vortex_dfsph_segment.json``.
  • PNG: ``<image_path>/sph/vorticity_XXXX.png`` (fluid + cylinder, no segments) and
    ``<image_path>/segments/vorticity_XXXX.png`` (segments colored by kernel-weighted
    ``vorticity_vis.z`` at segment center, same colormap / ``imageVorticityVmin`` / ``Max`` as SPH;
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
from matplotlib.colors import Normalize
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

        self.segment_bs_coupling_beta = float(
            self.seg_cfg.get_cfg("sphSegmentBsCoupling", 0.12)
        )
        self.sph_to_segment_blend = float(
            self.seg_cfg.get_cfg("sphToSegmentGammaBlend", 0.06)
        )
        self.sph_to_segment_gamma_scale = float(
            self.seg_cfg.get_cfg("sphToSegmentGammaScale", 5e-4)
        )
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
                ssol._boundary_one_shot_done = True

        ssol._emit_inlet_segments()
        ssol.ss.update_segment_geometry()
        dbg_deposit = 0
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
        )
        if dbg_deposit:
            self._print_deposit_debug_maxima()
        ssol.compute_endpoint_velocity()
        ssol.advect_segments_rk4()
        ssol.ss.update_segment_geometry()
        ssol.split_segments()
        ssol.merge_segments()
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
    ):
        h = self.ps.support_radius
        nseg = self.ss_seg.segment_num[None]
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
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
    ):
        h = self.ps.support_radius
        nseg = self.ss_seg.segment_num[None]
        for i in range(nseg):
            if self.ss_seg.active[i] != 1:
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
    ):
        if self.ps.dim == 2:
            self._deposit_vorticity_to_segments_2d_kernel(
                blend, gamma_scale, mf, proj_mode, src_vis, dbg
            )
        else:
            self._deposit_vorticity_to_segments_3d_kernel(
                blend, gamma_scale, mf, proj_mode, src_vis, dbg
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
                act = self.ss_seg.active.to_numpy()[:n_seg]
                idx = np.nonzero(act == 1)[0]
                if idx.shape[0] > 0:
                    seg_xyz = np.stack([xm[idx], xp[idx]], axis=1)
                    scalars = self._seg_vis_scalar.to_numpy()[:n_seg][idx].astype(
                        np.float64
                    )
                    lc = Line3DCollection(
                        list(seg_xyz),
                        array=scalars,
                        cmap=self._segment_cmap,
                        norm=vnorm,
                        linewidths=self._export_seg_line_width,
                        alpha=0.92,
                    )
                    ax2.add_collection3d(lc)
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
            if self._segment_panel_include_solid and solid_x.shape[0] > 0:
                ax2.scatter(
                    solid_x[:, 0], solid_x[:, 1], color="#00C853",
                    s=1.0, edgecolors="none",
                )
            if n_seg > 0:
                xm = self.ss_seg.x_minus.to_numpy()[:n_seg]
                xp = self.ss_seg.x_plus.to_numpy()[:n_seg]
                act = self.ss_seg.active.to_numpy()[:n_seg]
                idx = np.nonzero(act == 1)[0]
                if idx.shape[0] > 0:
                    scalars = self._seg_vis_scalar.to_numpy()[:n_seg][idx].astype(
                        np.float64
                    )
                    segs = np.stack([xm[idx], xp[idx]], axis=1)
                    lc = LineCollection(
                        list(segs[:, :, :2]),
                        array=scalars,
                        cmap=self._segment_cmap,
                        norm=vnorm,
                        linewidths=self._export_seg_line_width,
                        alpha=0.92,
                    )
                    ax2.add_collection(lc)
            ax2.set_xlim(float(ds[0]), float(de[0]))
            ax2.set_ylim(float(ds[1]), float(de[1]))
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(
                dir_seg / f"vorticity_{cnt:04}.png",
                bbox_inches="tight", pad_inches=0, transparent=True, dpi=400,
            )
            plt.close("all")

    def export_ply(self, cnt, ply_path):
        """导出 SPH 粒子 PLY（父类），并在同一目录写入涡段 ``segments_{cnt:04}.ply``。"""
        super().export_ply(cnt, ply_path)
        if self._export_segment_ply:
            self._segment_exporter.export_segments_ply(cnt, ply_path)
