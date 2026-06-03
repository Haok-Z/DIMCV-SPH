import taichi as ti
import numpy as np


def _cfg_biot_savart_is_finite(cfg) -> bool:
    """与 segment_solver 中约定一致：biotSavartModel 为 blob_center 时用中点 blob。"""
    m = cfg.get_cfg("biotSavartModel", "finite_segment")
    if m is None:
        return True
    m = str(m).lower().strip()
    if m in ("blob", "blob_center", "center_blob", "lumped"):
        return False
    return True


def _cfg_use_2d_point_vortex_bs(cfg, spatial_dim: int) -> bool:
    """与 segment_solver 一致：二维场景可用 TOG2021 式 (7)。"""
    m = cfg.get_cfg("biotSavartModel", None)
    if m is not None:
        m = str(m).lower().strip()
        if m in ("point_vortex_2d", "2d_point", "eq7", "formula_7", "2d"):
            return True
        if m in ("finite_segment", "finite", "3d", "blob", "blob_center"):
            return False
    return spatial_dim == 2


@ti.data_oriented
class SegmentBoundaryHandler:
    def __init__(self, segment_system):
        self.ss = segment_system
        self.enable_boundary_injection = bool(
            self.ss.cfg.get_cfg("enableBoundaryInjection", False)
        )

        # 边界采样状态（目前在 CPU 侧维护；最小二乘求解也大概率在 CPU 侧进行）。
        self.nb = int(self.ss.cfg.get_cfg("numBoundarySamples", 0) or 0)
        self._boundary_initialized = False
        self._b_points = None  # (Nb, 3) float32，边界采样点 b_i
        self._b_vel = None  # (Nb, 3) float32，采样点处规定的边界速度 u_b(b_i)
        self._b_owner = None  # (Nb,) int32，障碍全局索引：0..N_aabb-1 为 RigidBlocks，其后为圆柱
        self._b_normals = None  # (Nb, 3) float32；boundarySampleSource=sph_solid 时由流体侧方向估计

        src = str(self.ss.cfg.get_cfg("boundarySampleSource", "rigid_geometry") or "rigid_geometry")
        self._boundary_sample_source = src.lower().strip()
        self._ps = None  # 由混合求解器注入，用于从 SPH 固体粒子采样边界

        # 缓存 rigid blocks 以及（可选）动态平移信息
        self._rigid_blocks = self.ss.cfg.get_obstacles() if hasattr(self.ss.cfg, "get_obstacles") else []
        exclude_rigid_ids = set(int(x) for x in (self.ss.cfg.get_cfg("boundaryRigidBlockExcludeObjectIds", []) or []))
        if len(exclude_rigid_ids) > 0:
            self._rigid_blocks = [
                blk for blk in self._rigid_blocks
                if int(blk.get("objectId", -999999)) not in exclude_rigid_ids
            ]
        self._num_aabb_obstacles = int(len(self._rigid_blocks))
        self._block_dynamic = []
        self._block_vel = []
        self._block_translation = []
        for blk in self._rigid_blocks:
            is_dyn = bool(blk.get("isDynamic", False))
            vel = np.array(blk.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            tr = np.array(blk.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            self._block_dynamic.append(is_dyn)
            self._block_vel.append(vel)
            self._block_translation.append(tr)

        self._cylinders = self.ss.cfg.get_cylinders() if hasattr(self.ss.cfg, "get_cylinders") else []
        self._cylinder_dynamic = []
        self._cylinder_vel = []
        self._cylinder_translation = []
        for cyl in self._cylinders:
            is_dyn = bool(cyl.get("isDynamic", False))
            vel = np.array(cyl.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            tr = np.array(cyl.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            self._cylinder_dynamic.append(is_dyn)
            self._cylinder_vel.append(vel)
            self._cylinder_translation.append(tr)

        # 边界虚拟段（候选段）状态：本模块先在 CPU 侧生成，后续 commit 时写入段池
        self.ng = int(self.ss.cfg.get_cfg("numGeneratedBoundarySegments", 0) or 0)
        self._g_initialized = False
        self._g_x_minus = None  # (Ng,3) float32
        self._g_x_plus = None  # (Ng,3) float32
        self._g_gamma = None  # (Ng,) float32，待求解
        self._g_active = None  # (Ng,) bool
        self._g_owner_b = None  # (Ng,) int32，对应的边界采样点索引 i

        # 生成虚拟段时用到的几何参数（默认值给一个保守实现，后续可调参/替换为论文更严谨版本）
        self._g_length = float(self.ss.cfg.get_cfg("boundarySegmentLength", 0.05))
        self._g_inset = float(self.ss.cfg.get_cfg("boundarySegmentInset", 0.01))  # 往流体侧内缩距离
        self._g_tol = float(self.ss.cfg.get_cfg("boundaryNormalTolerance", 1e-5))
        self._g_seed = int(self.ss.cfg.get_cfg("boundarySegmentSeed", 1))

        # K 矩阵缓存：K 的形状为 (3*Nb, Ng)，列对应每条候选虚拟段
        self._K = None  # float32
        self._K_nb = 0
        self._K_ng = 0

        # RHS 缓存：U 的形状为 (3*Nb,)
        self._U = None  # float32
        self._U_nb = 0
        self._u_d = None  # (Nb,3) float32，用于调试：内部段诱导速度
        self._last_boundary_commit_count = 0

        self._gpu_boundary_cap_nb = max(1, int(self.nb))
        self._gpu_boundary_cap_ng = max(1, int(self.ng if self.ng > 0 else self.nb))
        self._gpu_boundary_b = ti.Vector.field(3, dtype=ti.f32, shape=self._gpu_boundary_cap_nb)
        self._gpu_boundary_ub = ti.Vector.field(3, dtype=ti.f32, shape=self._gpu_boundary_cap_nb)
        self._gpu_boundary_ud = ti.Vector.field(3, dtype=ti.f32, shape=self._gpu_boundary_cap_nb)
        self._gpu_boundary_U = ti.field(dtype=ti.f32, shape=3 * self._gpu_boundary_cap_nb)
        self._gpu_boundary_P = ti.field(dtype=ti.f32, shape=(self._gpu_boundary_cap_ng, 3 * self._gpu_boundary_cap_nb))
        self._gpu_boundary_gamma = ti.field(dtype=ti.f32, shape=self._gpu_boundary_cap_ng)
        self._gpu_boundary_projection_ready = False
        self._gpu_boundary_samples_synced = False

        self._use_2d_point_bs = _cfg_use_2d_point_vortex_bs(self.ss.cfg, int(self.ss.dim))

    def set_particle_system(self, particle_system):
        """混合 DFSPH+Segment 求解器在构造后调用，以支持 boundarySampleSource=sph_solid。"""
        self._ps = particle_system

    def mark_one_shot_complete_if_applicable(self, solver) -> None:
        """
        initialize_only 调度：SPH 边界需成功 commit 后才标记完成；
        此前若在流体不足时误标 done，会导致永远不注入（混合求解器曾有的 bug）。
        """
        if solver._boundary_one_shot_done:
            return
        if self._sample_source_is_sph() and self._ps is not None:
            if self._last_boundary_commit_count > 0:
                solver._boundary_one_shot_done = True
            return
        if self._boundary_initialized:
            solver._boundary_one_shot_done = True

    def _sample_source_is_sph(self) -> bool:
        return self._boundary_sample_source in (
            "sph_solid",
            "sph",
            "sph_boundary",
            "sph_particles",
        )

    def _boundary_virtual_mode(self) -> str:
        m = self.ss.cfg.get_cfg("boundaryVirtualSegmentMode", "least_squares")
        if m is None:
            return "least_squares"
        return str(m).lower().strip()

    def _uses_random_boundary_virtual(self) -> bool:
        return self._boundary_virtual_mode() in (
            "random",
            "random_near_solid",
            "sph_random",
            "random_sph",
        )

    def _orient_normals_into_domain(
        self, normals: np.ndarray, pts: np.ndarray
    ) -> np.ndarray:
        """
        将法向翻转为指向 Segment 计算域内部（通道内、流体侧）。
        固体格点外壳法向在边墙上常指向域外，会导致 p+inset*n 落在画图范围之外。
        """
        out = normals.astype(np.float32, copy=True)
        dim = int(self.ss.dim)
        lo = self.ss.domain_start.astype(np.float64)[:dim]
        hi = self.ss.domain_end.astype(np.float64)[:dim]
        center = 0.5 * (lo + hi)
        for i in range(int(pts.shape[0])):
            n = out[i, :dim].astype(np.float64)
            to_in = center - pts[i, :dim].astype(np.float64)
            if float(np.dot(n, to_in)) < 0.0:
                n = -n
            ln = float(np.linalg.norm(n))
            if ln < 1e-8:
                n = to_in
                ln = float(np.linalg.norm(n)) + 1e-8
            out[i, :dim] = (n / ln).astype(np.float32)
        return out

    @staticmethod
    def _estimate_outward_normals_solid_shell(
        pts: np.ndarray,
        x_solid: np.ndarray,
        dim: int,
        spacing: float,
    ) -> np.ndarray:
        """无流体时：由固体格点外壳的空邻居方向估计外法向（指向流体侧）。"""
        nb = int(pts.shape[0])
        normals = np.zeros((nb, 3), dtype=np.float32)
        if nb == 0:
            return normals
        scale = 1.0 / max(float(spacing), 1e-8)
        keys = [
            tuple(np.round(x_solid[i, :dim].astype(np.float64) * scale).astype(np.int64))
            for i in range(int(x_solid.shape[0]))
        ]
        keyset = set(keys)
        if dim == 2:
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            offsets = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            ]
        for i in range(nb):
            key = tuple(
                np.round(pts[i, :dim].astype(np.float64) * scale).astype(np.int64)
            )
            acc = np.zeros((dim,), dtype=np.float64)
            for off in offsets:
                nk = tuple(key[j] + off[j] for j in range(dim))
                if nk not in keyset:
                    for j in range(dim):
                        acc[j] += float(off[j]) / scale
            ln = float(np.linalg.norm(acc))
            if ln < 1e-8:
                acc = np.ones((dim,), dtype=np.float64)
                ln = float(np.linalg.norm(acc))
            normals[i, :dim] = (acc / ln).astype(np.float32)
        return normals

    @staticmethod
    def _rebalance_boundary_sample_counts(
        counts: list,
        nb: int,
        min_per_obstacle: int,
    ) -> list:
        """
        在按表面积比例分配后，抬高小面积障碍（如圆柱）的采样下限，并从大障碍收回采样，
        使总数仍为 nb。若 min_per_obstacle * n > nb，则允许个别障碍最终低于 min（从大障碍扣）。
        """
        counts = [int(max(0, c)) for c in counts]
        n_obs = len(counts)
        if n_obs == 0 or nb <= 0:
            return counts
        mp = int(max(0, min_per_obstacle))
        if mp <= 0:
            s = sum(counts)
            if s == nb:
                return counts
            if s < nb:
                k = 0
                while sum(counts) < nb:
                    counts[k % n_obs] += 1
                    k += 1
            else:
                while sum(counts) > nb:
                    j = int(np.argmax(counts))
                    if counts[j] <= 0:
                        break
                    counts[j] -= 1
            return counts

        for i in range(n_obs):
            if counts[i] < mp:
                counts[i] = mp
        s = sum(counts)
        while s > nb:
            j = int(np.argmax(counts))
            if counts[j] <= 1:
                break
            counts[j] -= 1
            s -= 1
        k = 0
        while sum(counts) < nb:
            counts[k % n_obs] += 1
            k += 1
        return counts

    @ti.kernel
    def _commit_segments_kernel(
        self,
        offset: int,
        n_new: int,
        x_minus: ti.types.ndarray(),
        x_plus: ti.types.ndarray(),
        gamma: ti.types.ndarray(),
        seg_type: int,
    ):
        for k in range(n_new):
            i = offset + k
            self.ss.set_segment_ends_from_ndarray_row(i, k, x_minus, x_plus)
            self.ss.gamma[i] = gamma[k]
            self.ss.active[i] = 1
            self.ss.age[i] = 0.0
            self.ss.seg_type[i] = seg_type

    def _get_background_velocity(self) -> np.ndarray:
        v = self.ss.cfg.get_cfg("backgroundVelocity", [0.0, 0.0, 0.0])
        return np.array(v, dtype=np.float32)

    @staticmethod
    def _bs_velocity_finite_line(
        query: np.ndarray,
        x_minus: np.ndarray,
        x_plus: np.ndarray,
        gamma_seg: float,
        reg_radius: float,
    ) -> np.ndarray:
        """
        论文 TOG2021 式 (6)（与 segment_solver 有限段 BS 一致）。
        Γ = gamma_seg * L；分母正则为 ||(x^−−x)×(x^+−x)||^2 + R^2。
        """
        am = x_minus.astype(np.float32, copy=False)
        ap = x_plus.astype(np.float32, copy=False)
        dvec = ap - am
        L = float(np.linalg.norm(dvec)) + 1e-8
        Gamma = float(gamma_seg) * L
        R2 = np.float32(reg_radius * reg_radius)
        apx = ap[None, :] - query
        ambx = am[None, :] - query
        n_ap = np.linalg.norm(apx, axis=1).astype(np.float32) + np.float32(reg_radius)
        n_am = np.linalg.norm(ambx, axis=1).astype(np.float32) + np.float32(reg_radius)
        u1 = apx / n_ap[:, None]
        u2 = ambx / n_am[:, None]
        d_exp = dvec.astype(np.float32, copy=False)[None, :]
        sc = Gamma * np.sum((u1 - u2) * d_exp, axis=1)
        cross = np.cross(ambx, apx).astype(np.float32)
        cross_sq = np.sum(cross * cross, axis=1).astype(np.float32) + R2
        coef = np.float32(1.0 / (4.0 * np.pi)) * sc / np.maximum(cross_sq, np.float32(1e-20))
        return (coef[:, None] * cross).astype(np.float32)

    @staticmethod
    def _bs_velocity_2d_point_vortex(
        query_xy: np.ndarray,
        vortex_xy: np.ndarray,
        gamma_strength: float,
        reg_radius: float,
    ) -> np.ndarray:
        """TOG2021 式 (7)，与 segment_solver._bs_velocity_2d_point_vortex 一致。"""
        inv2pi = np.float32(1.0 / (2.0 * np.pi))
        R2 = np.float32(reg_radius * reg_radius)
        rx = query_xy[:, 0] - np.float32(vortex_xy[0])
        ry = query_xy[:, 1] - np.float32(vortex_xy[1])
        r2 = rx * rx + ry * ry + np.float32(1e-12)
        denom = r2 + R2
        ux = inv2pi * np.float32(gamma_strength) * (-ry) / denom
        uy = inv2pi * np.float32(gamma_strength) * (rx) / denom
        out = np.zeros((query_xy.shape[0], 3), dtype=np.float32)
        out[:, 0] = ux.astype(np.float32)
        out[:, 1] = uy.astype(np.float32)
        return out

    def _bs_velocity_segment_at_queries(
        self,
        query: np.ndarray,
        x_minus: np.ndarray,
        x_plus: np.ndarray,
        gamma_seg: float,
        reg_radius: float,
    ) -> np.ndarray:
        """在 query 点上求单条虚拟段（强度 gamma_seg）的诱导速度。"""
        if self._use_2d_point_bs:
            c = 0.5 * (x_minus + x_plus)
            L = float(np.linalg.norm(x_plus - x_minus)) + 1e-8
            if query.shape[1] >= 2:
                qxy = query[:, :2].astype(np.float32, copy=False)
            else:
                qxy = query.astype(np.float32, copy=False)
            Gamma = float(gamma_seg) * L
            return self._bs_velocity_2d_point_vortex(
                qxy, c[:2].astype(np.float32, copy=False), Gamma, reg_radius
            )
        return self._bs_velocity_finite_line(query, x_minus, x_plus, gamma_seg, reg_radius)

    @staticmethod
    def _pad_vec3(arr: np.ndarray, dim: int) -> np.ndarray:
        out = np.zeros((arr.shape[0], 3), dtype=np.float32)
        d = min(int(dim), 3)
        out[:, :d] = arr[:, :d].astype(np.float32, copy=False)
        return out

    @staticmethod
    def _sph_surface_indices(x_solid: np.ndarray, dim: int, spacing: float) -> np.ndarray:
        """固体粒子中仅保留“外表面”格点（邻格无固体则视为边界）。"""
        ns = int(x_solid.shape[0])
        if ns == 0:
            return np.zeros((0,), dtype=np.int64)
        scale = 1.0 / max(float(spacing), 1e-8)
        keys = [
            tuple(np.round(x_solid[i, :dim].astype(np.float64) * scale).astype(np.int64))
            for i in range(ns)
        ]
        keyset = set(keys)
        if dim == 2:
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            offsets = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            ]
        surface = []
        for i, key in enumerate(keys):
            for off in offsets:
                nk = tuple(key[j] + off[j] for j in range(dim))
                if nk not in keyset:
                    surface.append(i)
                    break
        return np.asarray(surface, dtype=np.int64)

    @staticmethod
    def _estimate_outward_normals_from_fluid(
        pts: np.ndarray,
        fluid_xyz: np.ndarray,
        dim: int,
    ) -> np.ndarray:
        """由最近流体粒子方向估计外法向（指向流体）。"""
        nb = int(pts.shape[0])
        normals = np.zeros((nb, 3), dtype=np.float32)
        if fluid_xyz.shape[0] == 0:
            normals[:, 0] = 1.0
            return normals
        f = fluid_xyz[:, :dim].astype(np.float32, copy=False)
        p = pts[:, :dim].astype(np.float32, copy=False)
        for i in range(nb):
            d = f - p[i]
            j = int(np.argmin(np.sum(d * d, axis=1)))
            n = d[j]
            ln = float(np.linalg.norm(n))
            if ln < 1e-8:
                n = np.array([1.0, 0.0], dtype=np.float32)[:dim]
                ln = 1.0
            normals[i, :dim] = (n / ln).astype(np.float32)
        return normals

    def sph_boundary_sampling_ready(self) -> bool:
        """
        SPH 固体边界是否可采样。
        - random_near_solid：有固体粒子即可（不等待流体）
        - least_squares：默认需足够流体粒子以估计外法向
        """
        if not self._sample_source_is_sph() or self._ps is None:
            return True
        if self._boundary_initialized:
            return True
        ps = self._ps
        n = int(ps.particle_num[None])
        if n <= 0:
            return False
        mat = ps.material.to_numpy()[:n]
        ns = int(np.sum(mat == int(ps.material_solid)))
        if self._uses_random_boundary_virtual():
            return ns > 0
        min_fl = int(self.ss.cfg.get_cfg("boundarySphMinFluidParticles", 800) or 0)
        if min_fl <= 0:
            return True
        nf = int(np.sum(mat == int(ps.material_fluid)))
        return nf >= min_fl

    @staticmethod
    def _pad_vec(v, dim: int = 3, dtype=np.float32) -> np.ndarray:
        a = np.array(v, dtype=dtype).ravel()
        out = np.zeros((dim,), dtype=dtype)
        m = min(dim, int(a.size))
        if m > 0:
            out[:m] = a[:m]
        return out

    @staticmethod
    def _apply_block_transform_vec3(blk, translation_override=None):
        start = SegmentBoundaryHandler._pad_vec(blk.get("start", [0.0, 0.0, 0.0]), 3)
        end = SegmentBoundaryHandler._pad_vec(blk.get("end", [0.0, 0.0, 0.0]), 3)
        scale_raw = blk.get("scale", [1.0, 1.0, 1.0])
        scale = SegmentBoundaryHandler._pad_vec(scale_raw, 3)
        if len(scale_raw) < 3:
            scale[2] = 1.0
        tr = SegmentBoundaryHandler._pad_vec(blk.get("translation", [0.0, 0.0, 0.0]), 3)
        if translation_override is not None:
            tr = SegmentBoundaryHandler._pad_vec(translation_override, 3)
        lo = start + tr
        hi = start + tr + (end - start) * scale
        return np.minimum(lo, hi).astype(np.float32), np.maximum(lo, hi).astype(np.float32)

    @staticmethod
    def _aabb_perimeter_2d(blk) -> float:
        lo, hi = SegmentBoundaryHandler._apply_block_transform_vec3(blk)
        ex = max(float(hi[0] - lo[0]), 0.0)
        ey = max(float(hi[1] - lo[1]), 0.0)
        return 2.0 * (ex + ey)

    @staticmethod
    def _sample_points_on_aabb_perimeter_2d(blk, n: int):
        lo, hi = SegmentBoundaryHandler._apply_block_transform_vec3(blk)
        x0, y0 = float(lo[0]), float(lo[1])
        x1, y1 = float(hi[0]), float(hi[1])
        ex = max(x1 - x0, 1e-8)
        ey = max(y1 - y0, 1e-8)
        lengths = np.array([ex, ey, ex, ey], dtype=np.float64)
        cum = np.cumsum(lengths)
        per = float(cum[-1])
        svals = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1) * per
        pts = np.zeros((n, 3), dtype=np.float32)
        normals = np.zeros((n, 3), dtype=np.float32)
        for i, sv in enumerate(svals):
            if sv < cum[0]:
                a = sv / ex
                pts[i, 0] = x0 + a * ex
                pts[i, 1] = y0
                normals[i, 1] = -1.0
            elif sv < cum[1]:
                a = (sv - cum[0]) / ey
                pts[i, 0] = x1
                pts[i, 1] = y0 + a * ey
                normals[i, 0] = 1.0
            elif sv < cum[2]:
                a = (sv - cum[1]) / ex
                pts[i, 0] = x1 - a * ex
                pts[i, 1] = y1
                normals[i, 1] = 1.0
            else:
                a = (sv - cum[2]) / ey
                pts[i, 0] = x0
                pts[i, 1] = y1 - a * ey
                normals[i, 0] = -1.0
        return pts, normals

    def _init_boundary_samples_circle_2d(self):
        """Directly sample a 2D circular obstacle for least-squares virtual boundary segments."""
        nb = int(self.nb)
        if nb <= 0:
            self._b_points = np.zeros((0, 3), dtype=np.float32)
            self._b_vel = np.zeros((0, 3), dtype=np.float32)
            self._b_owner = np.zeros((0,), dtype=np.int32)
            self._b_normals = np.zeros((0, 3), dtype=np.float32)
            self._boundary_initialized = True
            return

        center_cfg = self.ss.cfg.get_cfg("boundaryCircleCenter", [0.65, 0.5])
        center = np.array(center_cfg, dtype=np.float32).ravel()
        cx = float(center[0]) if center.size > 0 else 0.65
        cy = float(center[1]) if center.size > 1 else 0.5
        radius = float(self.ss.cfg.get_cfg("boundaryCircleRadius", 0.1))
        radius = max(radius, 1e-8)

        endpoint = bool(self.ss.cfg.get_cfg("boundaryCircleIncludeEndpoint", False))
        theta = np.linspace(0.0, 2.0 * np.pi, nb, endpoint=endpoint, dtype=np.float32)
        pts = np.zeros((nb, 3), dtype=np.float32)
        normals = np.zeros((nb, 3), dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        pts[:, 0] = np.float32(cx) + np.float32(radius) * c
        pts[:, 1] = np.float32(cy) + np.float32(radius) * s
        normals[:, 0] = c
        normals[:, 1] = s

        vel_cfg = self.ss.cfg.get_cfg("boundaryCircleVelocity", [0.0, 0.0, 0.0])
        vel = np.array(vel_cfg, dtype=np.float32).ravel()
        bvel = np.zeros((nb, 3), dtype=np.float32)
        bvel[:, : min(3, vel.size)] = vel[: min(3, vel.size)][None, :]

        self._b_points = pts
        self._b_vel = bvel
        self._b_owner = np.full((nb,), -1, dtype=np.int32)
        self._b_normals = normals
        self._boundary_initialized = True

    def _init_boundary_samples_rigid_geometry_2d(self):
        """Sample 2D circle/cylinder and RigidBlocks perimeters for virtual boundary segments."""
        nb = int(self.nb)
        if nb <= 0:
            self._b_points = np.zeros((0, 3), dtype=np.float32)
            self._b_vel = np.zeros((0, 3), dtype=np.float32)
            self._b_owner = np.zeros((0,), dtype=np.int32)
            self._b_normals = np.zeros((0, 3), dtype=np.float32)
            self._boundary_initialized = True
            return

        shapes = []
        if bool(self.ss.cfg.get_cfg("boundaryIncludeCircle", True)):
            center_cfg = self.ss.cfg.get_cfg("boundaryCircleCenter", [0.65, 0.5])
            center = np.array(center_cfg, dtype=np.float32).ravel()
            cx = float(center[0]) if center.size > 0 else 0.65
            cy = float(center[1]) if center.size > 1 else 0.5
            r = max(float(self.ss.cfg.get_cfg("boundaryCircleRadius", 0.1)), 1e-8)
            shapes.append(("circle", (cx, cy, r), 2.0 * np.pi * r))
        for blk in self._rigid_blocks:
            shapes.append(("block", blk, self._aabb_perimeter_2d(blk)))

        if len(shapes) == 0:
            self._b_points = np.zeros((0, 3), dtype=np.float32)
            self._b_vel = np.zeros((0, 3), dtype=np.float32)
            self._b_owner = np.zeros((0,), dtype=np.int32)
            self._b_normals = np.zeros((0, 3), dtype=np.float32)
            self._boundary_initialized = True
            return

        min_per_shape = int(self.ss.cfg.get_cfg("boundaryMinSamplesPerObstacle", 24) or 0)
        weights = np.array([max(float(s[2]), 0.0) for s in shapes], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0:
            counts = [nb // len(shapes) for _ in shapes]
        else:
            raw = weights / float(np.sum(weights)) * float(nb)
            counts = np.floor(raw).astype(int).tolist()
        while sum(counts) < nb:
            counts[int(np.argmax(weights))] += 1
        while sum(counts) > nb:
            j = int(np.argmax(counts))
            counts[j] -= 1

        pts_all = []
        normals_all = []
        for (kind, data, _), cnt in zip(shapes, counts):
            cnt = int(cnt)
            if cnt <= 0:
                continue
            if kind == "circle":
                cx, cy, r = data
                theta = np.linspace(0.0, 2.0 * np.pi, cnt, endpoint=False, dtype=np.float32)
                pts = np.zeros((cnt, 3), dtype=np.float32)
                normals = np.zeros((cnt, 3), dtype=np.float32)
                c, s = np.cos(theta), np.sin(theta)
                pts[:, 0] = np.float32(cx) + np.float32(r) * c
                pts[:, 1] = np.float32(cy) + np.float32(r) * s
                normals[:, 0] = c
                normals[:, 1] = s
            else:
                pts, normals = self._sample_points_on_aabb_perimeter_2d(data, cnt)
            pts_all.append(pts)
            normals_all.append(normals)

        pts3 = np.concatenate(pts_all, axis=0).astype(np.float32)
        normals3 = np.concatenate(normals_all, axis=0).astype(np.float32)
        inset = float(self.ss.cfg.get_cfg("boundarySegmentInset", 0.0) or 0.0)
        if inset != 0.0:
            test_pts = pts3 + inset * normals3
            lo = self.ss.domain_start.astype(np.float32)
            hi = self.ss.domain_end.astype(np.float32)
            inside = (
                (test_pts[:, 0] >= lo[0]) & (test_pts[:, 0] <= hi[0]) &
                (test_pts[:, 1] >= lo[1]) & (test_pts[:, 1] <= hi[1])
            )
            normals3[~inside] *= -1.0

        vel_cfg = self.ss.cfg.get_cfg("boundaryCircleVelocity", [0.0, 0.0, 0.0])
        vel = np.array(vel_cfg, dtype=np.float32).ravel()
        bvel = np.zeros((pts3.shape[0], 3), dtype=np.float32)
        bvel[:, : min(3, vel.size)] = vel[: min(3, vel.size)][None, :]
        self._b_points = pts3
        self._b_vel = bvel
        self._b_owner = np.full((pts3.shape[0],), -1, dtype=np.int32)
        self._b_normals = normals3
        self._boundary_initialized = True

    def _init_boundary_samples_from_sph(self):
        """从 SPH 固体粒子（外表面）采样边界点，供后续虚拟段生成与约束。"""
        ps = self._ps
        if ps is None:
            raise ValueError(
                "boundarySampleSource=sph_solid 需要混合求解器调用 "
                "SegmentBoundaryHandler.set_particle_system(ps)"
            )
        n = int(ps.particle_num[None])
        if n <= 0 or self.nb <= 0:
            self._b_points = np.zeros((0, 3), dtype=np.float32)
            self._b_vel = np.zeros((0, 3), dtype=np.float32)
            self._b_owner = np.zeros((0,), dtype=np.int32)
            self._b_normals = np.zeros((0, 3), dtype=np.float32)
            self._boundary_initialized = True
            return

        dim = int(ps.dim)
        x_all = ps.x.to_numpy()[:n].astype(np.float32, copy=False)
        mat = ps.material.to_numpy()[:n]
        oid = ps.object_id.to_numpy()[:n]
        ms = int(ps.material_solid)
        solid_mask = mat == ms

        obj_filter = self.ss.cfg.get_cfg("boundarySphSolidObjectIds", None)
        if obj_filter is not None and len(obj_filter) > 0:
            want = {int(o) for o in obj_filter}
            solid_mask &= np.isin(oid, list(want))

        solid_idx = np.nonzero(solid_mask)[0]
        if solid_idx.size == 0:
            self._b_points = np.zeros((0, 3), dtype=np.float32)
            self._b_vel = np.zeros((0, 3), dtype=np.float32)
            self._b_owner = np.zeros((0,), dtype=np.int32)
            self._b_normals = np.zeros((0, 3), dtype=np.float32)
            self._boundary_initialized = True
            return

        x_solid = x_all[solid_idx]
        spacing = float(ps.particle_radius) * 2.0
        if bool(self.ss.cfg.get_cfg("boundarySphSurfaceOnly", True)):
            surf_local = self._sph_surface_indices(x_solid, dim, spacing)
            x_surf = x_solid[surf_local]
        else:
            x_surf = x_solid

        nb_target = int(self.nb)
        seed = int(self.ss.cfg.get_cfg("boundarySampleSeed", 0))
        rng = np.random.default_rng(seed)
        if x_surf.shape[0] > nb_target:
            pick = rng.choice(x_surf.shape[0], size=nb_target, replace=False)
            pts = x_surf[pick]
        else:
            pts = x_surf
            if pts.shape[0] < nb_target and pts.shape[0] > 0:
                extra = rng.choice(
                    pts.shape[0], size=nb_target - pts.shape[0], replace=True
                )
                pts = np.concatenate([pts, pts[extra]], axis=0)

        fluid_mask = mat == int(ps.material_fluid)
        fluid_xyz = x_all[fluid_mask]
        cap = int(self.ss.cfg.get_cfg("boundarySphFluidSampleCap", 25000) or 0)
        if cap > 0 and fluid_xyz.shape[0] > cap:
            pick_f = rng.choice(fluid_xyz.shape[0], size=cap, replace=False)
            fluid_xyz = fluid_xyz[pick_f]

        pts3 = self._pad_vec3(pts, dim)
        self._b_points = pts3
        self._b_vel = np.zeros((pts3.shape[0], 3), dtype=np.float32)
        self._b_owner = np.zeros((pts3.shape[0],), dtype=np.int32)
        prefer_shell = bool(
            self.ss.cfg.get_cfg("boundaryNormalsFromSolidShell", False)
        )
        if fluid_xyz.shape[0] > 0 and not prefer_shell:
            self._b_normals = self._estimate_outward_normals_from_fluid(
                pts3, fluid_xyz, dim
            )
        else:
            self._b_normals = self._estimate_outward_normals_solid_shell(
                pts3, x_solid, dim, spacing
            )
        self._b_normals = self._orient_normals_into_domain(self._b_normals, pts3)
        self._boundary_initialized = True

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < eps:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def _pick_perpendicular(n: np.ndarray) -> np.ndarray:
        """
        给定法向 n，选取一个稳定的切向方向（不要求物理最优，仅用于生成可用候选段）。
        """
        # 选一个与 n 不共线的轴
        ax = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ay = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        base = ax if abs(float(np.dot(n, ax))) < 0.9 else ay
        t = np.cross(n, base)
        return SegmentBoundaryHandler._normalize(t)

    def _boundary_tangent_direction(
        self, n: np.ndarray, u_dir: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        边界虚拟段的切向 t（必须与段同处物理平面）。
        2D 时不能用 cross(n, e_x) 当 t 落在 z 上，否则写入 2D 端点后长度恒为 0。
        """
        n3 = np.asarray(n, dtype=np.float32).ravel()
        if n3.size < 3:
            n3 = np.array(
                [float(n3[0]) if n3.size > 0 else 0.0,
                 float(n3[1]) if n3.size > 1 else 0.0,
                 0.0],
                dtype=np.float32,
            )
        if dim == 2:
            nx, ny = float(n3[0]), float(n3[1])
            t2 = np.array([-ny, nx], dtype=np.float64)
            ln = float(np.linalg.norm(t2))
            if ln < 1e-8:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            t2 /= ln
            return np.array([float(t2[0]), float(t2[1]), 0.0], dtype=np.float32)
        t = np.cross(n3, u_dir.astype(np.float32))
        t = self._normalize(t)
        if float(np.linalg.norm(t)) < 1e-6:
            t = self._pick_perpendicular(n3)
        return t.astype(np.float32, copy=False)

    def _estimate_aabb_normal(self, p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        对于 AABB 表面点 p，通过与 lo/hi 的距离估计外法向（仅用于 RigidBlocks）。
        """
        tol = self._g_tol
        # 判断贴近哪个面
        if abs(float(p[0] - lo[0])) <= tol:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(p[0] - hi[0])) <= tol:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(p[1] - lo[1])) <= tol:
            return np.array([0.0, -1.0, 0.0], dtype=np.float32)
        if abs(float(p[1] - hi[1])) <= tol:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(p[2] - lo[2])) <= tol:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if abs(float(p[2] - hi[2])) <= tol:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 兜底：取最近面的法向
        d = np.array(
            [
                abs(float(p[0] - lo[0])),
                abs(float(p[0] - hi[0])),
                abs(float(p[1] - lo[1])),
                abs(float(p[1] - hi[1])),
                abs(float(p[2] - lo[2])),
                abs(float(p[2] - hi[2])),
            ],
            dtype=np.float32,
        )
        fid = int(np.argmin(d))
        normals = [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        ]
        return np.array(normals[fid], dtype=np.float32)

    def _get_dt(self) -> float:
        return float(self.ss.cfg.get_cfg("timeStepSize", 0.002))

    @staticmethod
    def _apply_block_transform(blk, translation_override=None):
        """
        将一个 RigidBlock 条目转换为世界坐标系下的 AABB： [lo, hi]。

        这里复用/对齐了 SPH 中初始化立方体粒子时的几何逻辑：
        - lower_corner = start + translation
        - cube_size = (end - start) * scale
        """
        start = np.array(blk.get("start", [0.0, 0.0, 0.0]), dtype=np.float32)
        end = np.array(blk.get("end", [0.0, 0.0, 0.0]), dtype=np.float32)
        scale = np.array(blk.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        tr = np.array(blk.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        if translation_override is not None:
            tr = np.array(translation_override, dtype=np.float32)

        lo = start + tr
        hi = start + tr + (end - start) * scale
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        return lo2, hi2

    @staticmethod
    def _cylinder_world_geometry(cyl, translation_override=None):
        tr = np.array(cyl.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        if translation_override is not None:
            tr = np.array(translation_override, dtype=np.float32)
        center = np.array(cyl.get("center", [0.0, 0.0, 0.0]), dtype=np.float32) + tr
        r = float(cyl.get("radius", 0.1))
        hh = float(cyl.get("halfHeight", 0.5))
        axis = str(cyl.get("axis", "z")).lower().strip()
        return center, axis, r, hh

    @staticmethod
    def _cylinder_lateral_area(cyl, translation_override=None):
        _, _, r, hh = SegmentBoundaryHandler._cylinder_world_geometry(cyl, translation_override)
        h = 2.0 * max(hh, 1e-8)
        return float(2.0 * np.pi * max(r, 1e-8) * h)

    @staticmethod
    def _sample_points_on_cylinder_lateral(cyl, n, rng: np.random.Generator, translation_override=None):
        center, axis, r, hh = SegmentBoundaryHandler._cylinder_world_geometry(cyl, translation_override)
        u = rng.random((n, 2))
        theta = (2.0 * np.pi * u[:, 0]).astype(np.float32)
        s = ((2.0 * u[:, 1] - 1.0) * hh).astype(np.float32)
        rf = np.float32(r)
        pts = np.zeros((n, 3), dtype=np.float32)
        c0, c1, c2 = float(center[0]), float(center[1]), float(center[2])
        if axis == "z":
            pts[:, 0] = np.float32(c0) + np.cos(theta) * rf
            pts[:, 1] = np.float32(c1) + np.sin(theta) * rf
            pts[:, 2] = np.float32(c2) + s
        elif axis == "y":
            pts[:, 0] = np.float32(c0) + np.cos(theta) * rf
            pts[:, 1] = np.float32(c1) + s
            pts[:, 2] = np.float32(c2) + np.sin(theta) * rf
        else:
            pts[:, 0] = np.float32(c0) + s
            pts[:, 1] = np.float32(c1) + np.cos(theta) * rf
            pts[:, 2] = np.float32(c2) + np.sin(theta) * rf
        return pts

    def _estimate_cylinder_normal(self, p: np.ndarray, cyl, translation_override=None):
        center, axis, _, _ = self._cylinder_world_geometry(cyl, translation_override)
        d = p.astype(np.float64) - center.astype(np.float64)
        ax = str(axis).lower()
        n = np.zeros(3, dtype=np.float64)
        if ax == "z":
            n[0], n[1] = d[0], d[1]
        elif ax == "y":
            n[0], n[2] = d[0], d[2]
        else:
            n[1], n[2] = d[1], d[2]
        nn = float(np.linalg.norm(n))
        if nn < 1e-10:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (n / nn).astype(np.float32)

    @staticmethod
    def _sample_points_on_aabb_surface(lo, hi, n, rng: np.random.Generator):
        """
        在轴对齐盒子的表面均匀采样 n 个点。
        返回形状为 (n, 3) 的 float32 数组。
        """
        lo = lo.astype(np.float32)
        hi = hi.astype(np.float32)
        extent = np.maximum(hi - lo, 1e-8)
        ex, ey, ez = float(extent[0]), float(extent[1]), float(extent[2])

        # 六个面的面积（用于按面积比例选择面）
        a_xy = ex * ey
        a_xz = ex * ez
        a_yz = ey * ez
        areas = np.array([a_yz, a_yz, a_xz, a_xz, a_xy, a_xy], dtype=np.float64)
        area_sum = float(np.sum(areas))
        if area_sum <= 0.0:
            # 退化盒子：直接返回重复的中心点
            c = 0.5 * (lo + hi)
            return np.repeat(c[None, :], n, axis=0).astype(np.float32)

        probs = areas / area_sum
        face_ids = rng.choice(6, size=n, p=probs)
        u = rng.random((n, 2), dtype=np.float32)

        pts = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            fid = int(face_ids[i])
            s, t = float(u[i, 0]), float(u[i, 1])
            if fid == 0:  # x = lo.x（YZ 面）
                pts[i, 0] = lo[0]
                pts[i, 1] = lo[1] + s * ey
                pts[i, 2] = lo[2] + t * ez
            elif fid == 1:  # x = hi.x（YZ 面）
                pts[i, 0] = hi[0]
                pts[i, 1] = lo[1] + s * ey
                pts[i, 2] = lo[2] + t * ez
            elif fid == 2:  # y = lo.y（XZ 面）
                pts[i, 0] = lo[0] + s * ex
                pts[i, 1] = lo[1]
                pts[i, 2] = lo[2] + t * ez
            elif fid == 3:  # y = hi.y（XZ 面）
                pts[i, 0] = lo[0] + s * ex
                pts[i, 1] = hi[1]
                pts[i, 2] = lo[2] + t * ez
            elif fid == 4:  # z = lo.z（XY 面）
                pts[i, 0] = lo[0] + s * ex
                pts[i, 1] = lo[1] + t * ey
                pts[i, 2] = lo[2]
            else:  # fid == 5，z = hi.z（XY 面）
                pts[i, 0] = lo[0] + s * ex
                pts[i, 1] = lo[1] + t * ey
                pts[i, 2] = hi[2]
        return pts

    def update_boundary_pose(self):
        """
        TODO：
        根据边界运动更新边界采样点。
        - 静态边界：无需更新
        - 运动刚体边界：按当前位姿更新采样点
        """
        if not self.enable_boundary_injection:
            return

        if self.nb <= 0:
            return

        # 第一次调用时初始化边界采样点
        if (not self._boundary_initialized) or (self._b_points is None):
            if self._sample_source_is_sph():
                if not self.sph_boundary_sampling_ready():
                    return
                self._init_boundary_samples_from_sph()
                return

            # 可通过配置覆盖随机种子，保证可复现
            if self._boundary_sample_source in ("rigid_geometry_2d", "geometry_2d", "mixed_2d"):
                self._init_boundary_samples_rigid_geometry_2d()
                return

            if self._boundary_sample_source in ("circle_2d", "circle", "circular_obstacle_2d"):
                self._init_boundary_samples_circle_2d()
                return

            seed = self.ss.cfg.get_cfg("boundarySampleSeed", 0)
            rng = np.random.default_rng(int(seed))

            if len(self._rigid_blocks) == 0 and len(self._cylinders) == 0:
                self._b_points = np.zeros((0, 3), dtype=np.float32)
                self._b_vel = np.zeros((0, 3), dtype=np.float32)
                self._b_owner = np.zeros((0,), dtype=np.int32)
                self._boundary_initialized = True
                return

            # 按每个 AABB 与圆柱侧面积的表面积比例分配采样数（圆柱仅采侧表面）
            obstacle_areas = []
            for bi, blk in enumerate(self._rigid_blocks):
                lo, hi = self._apply_block_transform(blk, translation_override=self._block_translation[bi])
                extent = np.maximum(hi - lo, 0.0)
                ex, ey, ez = float(extent[0]), float(extent[1]), float(extent[2])
                area = 2.0 * (ex * ey + ex * ez + ey * ez)
                obstacle_areas.append(max(area, 0.0))
            for ci, cyl in enumerate(self._cylinders):
                area = self._cylinder_lateral_area(cyl, translation_override=self._cylinder_translation[ci])
                obstacle_areas.append(max(area, 0.0))

            # 有效面积权重：在总采样数 nb 固定时，提高圆柱权重可让圆柱上采样更密、平板相对更稀。
            aabb_w = float(self.ss.cfg.get_cfg("boundaryAabbSampleWeight", 1.0) or 1.0)
            cyl_w = float(self.ss.cfg.get_cfg("boundaryCylinderSampleWeight", 1.0) or 1.0)
            aabb_w = max(0.0, aabb_w)
            cyl_w = max(0.0, cyl_w)
            for idx in range(0, min(self._num_aabb_obstacles, len(obstacle_areas))):
                obstacle_areas[idx] *= aabb_w
            for idx in range(self._num_aabb_obstacles, len(obstacle_areas)):
                obstacle_areas[idx] *= cyl_w

            n_obs = len(obstacle_areas)
            area_sum = float(np.sum(obstacle_areas))

            if area_sum <= 0.0 or n_obs == 0:
                per = max(1, self.nb // max(1, n_obs))
                counts = [per for _ in range(n_obs)]
                counts[0] += self.nb - sum(counts)
                min_bo = int(self.ss.cfg.get_cfg("boundaryMinSamplesPerObstacle", 0) or 0)
                counts = self._rebalance_boundary_sample_counts(counts, self.nb, min_bo)
            else:
                raw = np.array(obstacle_areas, dtype=np.float64) / area_sum * float(self.nb)
                counts = np.floor(raw).astype(int).tolist()
                remaining = self.nb - int(np.sum(counts))
                if remaining > 0:
                    frac = raw - np.floor(raw)
                    order = np.argsort(-frac)
                    for k in range(remaining):
                        counts[int(order[k % len(counts)])] += 1

            min_bo = int(self.ss.cfg.get_cfg("boundaryMinSamplesPerObstacle", 0) or 0)
            counts = self._rebalance_boundary_sample_counts(counts, self.nb, min_bo)

            pts_all = []
            vel_all = []
            owner_all = []
            for bi, blk in enumerate(self._rigid_blocks):
                n_i = int(counts[bi])
                if n_i <= 0:
                    continue
                lo, hi = self._apply_block_transform(blk, translation_override=self._block_translation[bi])
                pts = self._sample_points_on_aabb_surface(lo, hi, n_i, rng)
                pts_all.append(pts)
                v = self._block_vel[bi] if self._block_dynamic[bi] else np.zeros(3, dtype=np.float32)
                vel_all.append(np.repeat(v[None, :], n_i, axis=0))
                owner_all.append(np.full((n_i,), bi, dtype=np.int32))

            for ci, cyl in enumerate(self._cylinders):
                n_i = int(counts[self._num_aabb_obstacles + ci])
                if n_i <= 0:
                    continue
                obs_id = self._num_aabb_obstacles + ci
                pts = self._sample_points_on_cylinder_lateral(
                    cyl, n_i, rng, translation_override=self._cylinder_translation[ci]
                )
                pts_all.append(pts)
                v = self._cylinder_vel[ci] if self._cylinder_dynamic[ci] else np.zeros(3, dtype=np.float32)
                vel_all.append(np.repeat(v[None, :], n_i, axis=0))
                owner_all.append(np.full((n_i,), obs_id, dtype=np.int32))

            if len(pts_all) == 0:
                self._b_points = np.zeros((0, 3), dtype=np.float32)
                self._b_vel = np.zeros((0, 3), dtype=np.float32)
                self._b_owner = np.zeros((0,), dtype=np.int32)
            else:
                self._b_points = np.concatenate(pts_all, axis=0).astype(np.float32)
                self._b_vel = np.concatenate(vel_all, axis=0).astype(np.float32)
                self._b_owner = np.concatenate(owner_all, axis=0).astype(np.int32)

            self._boundary_initialized = True
            return

        # 若存在动态边界（目前仅支持平移），则更新缓存采样点
        dt = self._get_dt()
        any_dynamic = any(self._block_dynamic) or any(self._cylinder_dynamic)
        if not any_dynamic:
            return

        for bi, is_dyn in enumerate(self._block_dynamic):
            if is_dyn:
                self._block_translation[bi] = self._block_translation[bi] + self._block_vel[bi] * dt

        for ci, is_dyn in enumerate(self._cylinder_dynamic):
            if is_dyn:
                self._cylinder_translation[ci] = self._cylinder_translation[ci] + self._cylinder_vel[ci] * dt

        for bi, is_dyn in enumerate(self._block_dynamic):
            if not is_dyn:
                continue
            mask = self._b_owner == bi
            if np.any(mask):
                self._b_points[mask] += self._block_vel[bi][None, :] * dt

        for ci, is_dyn in enumerate(self._cylinder_dynamic):
            if not is_dyn:
                continue
            oid = self._num_aabb_obstacles + ci
            mask = self._b_owner == oid
            if np.any(mask):
                self._b_points[mask] += self._cylinder_vel[ci][None, :] * dt

    def _boundary_segment_normal_at(self, bi: int) -> np.ndarray:
        p = self._b_points[bi]
        if self._b_normals is not None and self._b_normals.shape[0] == self._b_points.shape[0]:
            return self._b_normals[bi].astype(np.float32, copy=False)
        owner = int(self._b_owner[bi])
        if owner < self._num_aabb_obstacles:
            blk = self._rigid_blocks[owner]
            lo, hi = self._apply_block_transform(
                blk, translation_override=self._block_translation[owner]
            )
            return self._estimate_aabb_normal(p, lo, hi)
        ci = owner - self._num_aabb_obstacles
        cyl = self._cylinders[ci]
        return self._estimate_cylinder_normal(
            p, cyl, translation_override=self._cylinder_translation[ci]
        )

    def _build_boundary_candidate_segments(
        self,
        randomize_strength: bool,
        rng: np.random.Generator,
    ):
        if self._b_points is None or self._b_vel is None or self._b_owner is None:
            return

        nb = int(self._b_points.shape[0])
        if nb == 0:
            return

        ng = int(self.ng if self.ng > 0 else nb)
        ng = min(ng, nb)

        if ng < nb:
            chosen = rng.choice(nb, size=ng, replace=False)
        else:
            chosen = np.arange(nb, dtype=np.int32)

        g_xm = np.zeros((ng, 3), dtype=np.float32)
        g_xp = np.zeros((ng, 3), dtype=np.float32)
        g_gamma = np.zeros((ng,), dtype=np.float32)
        g_active = np.ones((ng,), dtype=bool)
        g_owner_b = chosen.astype(np.int32, copy=True)

        u_inf = self._get_background_velocity()
        u_dir = self._normalize(u_inf)
        dim = int(self.ss.dim)
        if dim == 2 and float(np.linalg.norm(u_dir[:2])) < 1e-6:
            ref = self.ss.cfg.get_cfg("boundaryTangentReferenceVelocity", [1.0, 0.0, 0.0])
            if ref is None:
                ref = [1.0, 0.0, 0.0]
            u_dir = self._normalize(np.array(ref, dtype=np.float32))
        inset = float(self._g_inset)

        if randomize_strength:
            L0 = float(self.ss.cfg.get_cfg("boundaryRandomLengthMin", 0.0) or 0.0)
            L1 = float(self.ss.cfg.get_cfg("boundaryRandomLengthMax", 0.0) or 0.0)
            if L1 <= L0 + 1e-12:
                L0 = 0.75 * float(self._g_length)
                L1 = 1.25 * float(self._g_length)
            g_min = float(self.ss.cfg.get_cfg("boundaryRandomGammaMin", -0.02))
            g_max = float(self.ss.cfg.get_cfg("boundaryRandomGammaMax", 0.02))
            if g_max < g_min:
                g_min, g_max = g_max, g_min
            tangent_jitter = bool(
                self.ss.cfg.get_cfg("boundaryRandomTangentJitter", True)
            )
        else:
            L0 = L1 = float(self._g_length)
            g_min = g_max = 0.0
            tangent_jitter = False

        for k in range(ng):
            bi = int(chosen[k])
            n = self._boundary_segment_normal_at(bi)
            place_on_solid = bool(
                self.ss.cfg.get_cfg("boundaryPlaceOnSolidParticles", True)
            )
            if place_on_solid:
                # 段心落在固体采样点（与 SPH 绿点一致），仅切向随机
                p_center = self._b_points[bi].astype(np.float32, copy=False)
            else:
                p_center = self._b_points[bi] + inset * n

            t = self._boundary_tangent_direction(n, u_dir, dim)

            if randomize_strength and tangent_jitter:
                if dim == 2:
                    phi = float(rng.uniform(0.0, 2.0 * np.pi))
                    tx, ty = float(t[0]), float(t[1])
                    c, s = np.cos(phi), np.sin(phi)
                    t = np.array([c * tx - s * ty, s * tx + c * ty, 0.0], dtype=np.float32)
                    tn = float(np.linalg.norm(t[:2]))
                    if tn > 1e-8:
                        t = (t / tn).astype(np.float32)
                else:
                    jitter = rng.standard_normal(3).astype(np.float32)
                    t = self._normalize(t + 0.35 * jitter)

            L = float(rng.uniform(L0, L1)) if randomize_strength else L0
            g_xm[k] = p_center - 0.5 * L * t
            g_xp[k] = p_center + 0.5 * L * t
            if randomize_strength:
                g_gamma[k] = float(rng.uniform(g_min, g_max))

        self._g_x_minus = g_xm
        self._g_x_plus = g_xp
        self._g_gamma = g_gamma
        self._g_active = g_active
        self._g_owner_b = g_owner_b
        self._g_initialized = True

    def generate_boundary_segments(self):
        """在边界采样点附近生成候选虚拟段（γ=0，供最小二乘求解）。"""
        if not self.enable_boundary_injection:
            return
        rng = np.random.default_rng(self._g_seed)
        self._build_boundary_candidate_segments(randomize_strength=False, rng=rng)

    def generate_boundary_segments_random(self):
        """
        在 SPH 固体表面采样点附近放置随机短虚拟段（随机切向、长度、γ），
        跳过 K 矩阵最小二乘。
        """
        if not self.enable_boundary_injection:
            return
        rng = np.random.default_rng(int(self._g_seed) + 99173)
        self._build_boundary_candidate_segments(randomize_strength=True, rng=rng)

    def compute_k_matrix(self):
        """
        TODO：
        构建 K 矩阵，其中 K[i, a] 表示单位强度虚拟段 a
        在边界采样点 b_i 处诱导的速度。
        """
        if not self.enable_boundary_injection:
            return

        if not self._g_initialized or self._g_x_minus is None or self._g_x_plus is None:
            # 还没有候选虚拟段
            return
        if self._b_points is None:
            return

        b = self._b_points.astype(np.float32, copy=False)
        nb = int(b.shape[0])
        ng = int(self._g_x_minus.shape[0])
        if nb == 0 or ng == 0:
            return

        R = float(self.ss.cfg.get_cfg("regularizationRadiusR", 0.01))
        R2 = R * R
        use_finite = _cfg_biot_savart_is_finite(self.ss.cfg)

        xm = self._g_x_minus.astype(np.float32, copy=False)
        xp = self._g_x_plus.astype(np.float32, copy=False)
        K = np.zeros((3 * nb, ng), dtype=np.float32)
        inv4pi = np.float32(1.0 / (4.0 * np.pi))

        if self._use_2d_point_bs or use_finite:
            # K 列 a：单位强度（gamma=1）虚拟段在边界点上的诱导速度。
            for a in range(ng):
                u = self._bs_velocity_segment_at_queries(b, xm[a], xp[a], 1.0, R)
                K[0::3, a] = u[:, 0]
                K[1::3, a] = u[:, 1]
                K[2::3, a] = u[:, 2]
        else:
            # 中点 blob：omega = t * L（单位 gamma）
            c = 0.5 * (xm + xp)
            d = xp - xm
            L = np.linalg.norm(d, axis=1).astype(np.float32) + 1e-8
            t = (d.T / L).T
            omega = (t.T * L).T
            for a in range(ng):
                ca_tmp = c[a]
                wa = omega[a]
                r = b - ca_tmp[None, :]
                r2 = np.sum(r * r, axis=1) + np.float32(R2)
                denom = np.power(r2, 1.5).astype(np.float32)
                cross = np.cross(np.repeat(wa[None, :], nb, axis=0), r).astype(np.float32)
                u = inv4pi * (cross.T / denom).T
                K[0::3, a] = u[:, 0]
                K[1::3, a] = u[:, 1]
                K[2::3, a] = u[:, 2]

        self._K = K
        self._K_nb = nb
        self._K_ng = ng

    def _gpu_boundary_enabled(self) -> bool:
        return bool(self.ss.cfg.get_cfg("enableGpuBoundarySolve", False)) and self._use_2d_point_bs and int(self.ss.dim) == 2

    def _prepare_gpu_boundary_projection(self) -> bool:
        if not self._gpu_boundary_enabled():
            return False
        if self._K is None or self._b_points is None or self._b_vel is None:
            return False
        nb = int(self._K_nb)
        ng = int(self._K_ng)
        if nb <= 0 or ng <= 0:
            return False
        if nb > self._gpu_boundary_cap_nb or ng > self._gpu_boundary_cap_ng:
            return False
        if self._gpu_boundary_projection_ready and self._gpu_boundary_samples_synced:
            return True

        K = self._K.astype(np.float64, copy=False)
        eps = max(float(self.ss.cfg.get_cfg("boundaryLeastSquaresEps", 1e-4)), 0.0)
        A = K.T @ K
        if eps > 0.0:
            A = A + eps * np.eye(ng, dtype=np.float64)
        try:
            P = np.linalg.solve(A, K.T).astype(np.float32)
        except np.linalg.LinAlgError:
            P = np.linalg.lstsq(A, K.T, rcond=None)[0].astype(np.float32)

        p_host = np.zeros((self._gpu_boundary_cap_ng, 3 * self._gpu_boundary_cap_nb), dtype=np.float32)
        p_host[:ng, : 3 * nb] = P
        self._gpu_boundary_P.from_numpy(p_host)

        b_host = np.zeros((self._gpu_boundary_cap_nb, 3), dtype=np.float32)
        ub_host = np.zeros((self._gpu_boundary_cap_nb, 3), dtype=np.float32)
        b_host[:nb] = self._b_points[:nb].astype(np.float32, copy=False)
        ub_host[:nb] = self._b_vel[:nb].astype(np.float32, copy=False)
        self._gpu_boundary_b.from_numpy(b_host)
        self._gpu_boundary_ub.from_numpy(ub_host)

        self._gpu_boundary_projection_ready = True
        self._gpu_boundary_samples_synced = True
        return True

    @ti.kernel
    def _compute_rhs_2d_point_gpu_kernel(
        self,
        nb: int,
        ns: int,
        boundary_type: int,
        reg_radius: float,
        ux: float,
        uy: float,
        uz: float,
    ):
        R2 = reg_radius * reg_radius
        for i in range(nb):
            b = self._gpu_boundary_b[i]
            ud = ti.Vector([0.0, 0.0, 0.0])
            for j in range(ns):
                if self.ss.active[j] == 1 and self.ss.seg_type[j] != boundary_type:
                    am = self.ss.x_minus[j]
                    ap = self.ss.x_plus[j]
                    c = 0.5 * (am + ap)
                    L = (ap - am).norm() + 1e-8
                    Gamma = self.ss.gamma[j] * L
                    r = ti.Vector([b[0] - c[0], b[1] - c[1], 0.0])
                    r2 = r[0] * r[0] + r[1] * r[1] + R2
                    denom = r2 + 1e-12
                    coeff = Gamma / (2.0 * ti.math.pi * denom)
                    ud[0] += -r[1] * coeff
                    ud[1] += r[0] * coeff
            self._gpu_boundary_ud[i] = ud
            uinf = ti.Vector([ux, uy, uz])
            U = self._gpu_boundary_ub[i] - ud - uinf
            self._gpu_boundary_U[3 * i + 0] = U[0]
            self._gpu_boundary_U[3 * i + 1] = U[1]
            self._gpu_boundary_U[3 * i + 2] = U[2]

    @ti.kernel
    def _solve_gamma_projection_gpu_kernel(self, nb: int, ng: int):
        for a in range(ng):
            acc = 0.0
            for r in range(3 * nb):
                acc += self._gpu_boundary_P[a, r] * self._gpu_boundary_U[r]
            self._gpu_boundary_gamma[a] = acc

    def _compute_rhs_and_solve_gpu(self) -> bool:
        if not self._prepare_gpu_boundary_projection():
            return False
        nb = int(self._K_nb)
        ng = int(self._K_ng)
        ns = int(self.ss.segment_num[None])
        boundary_type = int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2))
        reg_radius = float(self.ss.cfg.get_cfg("regularizationRadiusR", 0.01))
        u_inf = self._get_background_velocity().astype(np.float32, copy=False)
        self._compute_rhs_2d_point_gpu_kernel(nb, ns, boundary_type, reg_radius, float(u_inf[0]), float(u_inf[1]), float(u_inf[2]))
        self._solve_gamma_projection_gpu_kernel(nb, ng)
        gamma = self._gpu_boundary_gamma.to_numpy()[:ng].astype(np.float32, copy=False)
        if self._g_gamma is None or self._g_gamma.shape[0] != ng:
            self._g_gamma = np.zeros((ng,), dtype=np.float32)
        self._g_gamma[:ng] = gamma
        return True

    def compute_rhs(self):
        """
        TODO：
        构建边界采样点上的右端项 U = u_b - u_d - u_inf。
        """
        if not self.enable_boundary_injection:
            return

        if self._b_points is None or self._b_vel is None:
            return
        b = self._b_points.astype(np.float32, copy=False)
        ub = self._b_vel.astype(np.float32, copy=False)
        nb = int(b.shape[0])
        if nb == 0:
            return

        u_inf = self._get_background_velocity().astype(np.float32, copy=False)

        # 是否将内部段（segment_system 中的段）诱导速度计入 u_d（更接近论文公式）
        include_internal = bool(self.ss.cfg.get_cfg("rhsIncludeInternalSegments", False))

        u_d = np.zeros((nb, 3), dtype=np.float32)
        if include_internal:
            Ns = int(self.ss.segment_num[None])
            if Ns > 0:
                xm = self.ss.x_minus.to_numpy()[:Ns].astype(np.float32, copy=False)
                xp = self.ss.x_plus.to_numpy()[:Ns].astype(np.float32, copy=False)
                active = self.ss.active.to_numpy()[:Ns].astype(np.int32, copy=False)
                gamma = self.ss.gamma.to_numpy()[:Ns].astype(np.float32, copy=False)

                R = float(self.ss.cfg.get_cfg("regularizationRadiusR", 0.01))
                use_finite = _cfg_biot_savart_is_finite(self.ss.cfg)

                btype = int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2))
                seg_types = self.ss.seg_type.to_numpy()[:Ns].astype(np.int32, copy=False)
                if self._use_2d_point_bs or use_finite:
                    for j in range(Ns):
                        if active[j] != 1 or seg_types[j] == btype:
                            continue
                        u_d += self._bs_velocity_segment_at_queries(
                            b, xm[j], xp[j], float(gamma[j]), R
                        )
                else:
                    R2 = R * R
                    inv4pi = np.float32(1.0 / (4.0 * np.pi))
                    c = 0.5 * (xm + xp)
                    d = xp - xm
                    L = np.linalg.norm(d, axis=1).astype(np.float32) + 1e-8
                    t = (d.T / L).T
                    omega = (t.T * (gamma * L)).T
                    for j in range(Ns):
                        if active[j] != 1:
                            continue
                        cj = c[j]
                        wj = omega[j]
                        r = b - cj[None, :]
                        r2 = np.sum(r * r, axis=1) + np.float32(R2)
                        denom = np.power(r2, 1.5).astype(np.float32)
                        cross = np.cross(np.repeat(wj[None, :], nb, axis=0), r).astype(np.float32)
                        u = inv4pi * (cross.T / denom).T
                        u_d += u

        U = (ub - u_d - u_inf[None, :]).astype(np.float32)
        U_flat = np.zeros((3 * nb,), dtype=np.float32)
        U_flat[0::3] = U[:, 0]
        U_flat[1::3] = U[:, 1]
        U_flat[2::3] = U[:, 2]

        self._u_d = u_d
        self._U = U_flat
        self._U_nb = nb

    def solve_linear_system(self):
        """
        TODO：
        求解正则化最小二乘：
            gamma = (K^T K + eps I)^(-1) K^T U
        以恢复虚拟段强度。
        """
        if not self.enable_boundary_injection:
            return

        if self._K is None or self._U is None:
            return

        K = self._K
        U = self._U
        nb = int(self._K_nb)
        ng = int(self._K_ng)
        if nb <= 0 or ng <= 0:
            return
        if U.shape[0] != 3 * nb:
            return

        # 仅对活跃的候选虚拟段求解（如果未来支持部分失活）
        if self._g_active is None:
            active_mask = np.ones((ng,), dtype=bool)
        else:
            active_mask = self._g_active.astype(bool, copy=False)
            if active_mask.shape[0] != ng:
                active_mask = np.ones((ng,), dtype=bool)

        active_ids = np.nonzero(active_mask)[0]
        if active_ids.size == 0:
            return

        K_a = K[:, active_ids]  # (3*Nb, Ng_active)

        eps = float(self.ss.cfg.get_cfg("boundaryLeastSquaresEps", 1e-4))
        eps = max(eps, 0.0)

        # 正则化最小二乘：解 (K^T K + eps I) gamma = K^T U
        # 这里的矩阵应为对称正定（eps>0 时），可用 solve/Cholesky。
        A = (K_a.T @ K_a).astype(np.float64, copy=False)
        if eps > 0.0:
            A = A + (eps * np.eye(A.shape[0], dtype=np.float64))
        b = (K_a.T @ U).astype(np.float64, copy=False)

        try:
            gamma_a = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            # 兜底：用最小二乘（可能更慢，但更鲁棒）
            gamma_a = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

        # 写回到候选虚拟段强度数组（不活跃的保持 0）
        if self._g_gamma is None or self._g_gamma.shape[0] != ng:
            self._g_gamma = np.zeros((ng,), dtype=np.float32)
        else:
            self._g_gamma.fill(0.0)
        self._g_gamma[active_ids] = gamma_a

    def release_vorticity_to_internal_segments(self):
        """Release strong virtual boundary segments into movable internal segments."""
        if not bool(self.ss.cfg.get_cfg("enableBoundaryVorticityRelease", False)):
            return
        if self._g_x_minus is None or self._g_x_plus is None or self._g_gamma is None:
            return
        if self.ss.dim != 2:
            return

        threshold = float(self.ss.cfg.get_cfg("boundaryReleaseGammaThreshold", 0.05))
        scale = float(self.ss.cfg.get_cfg("boundaryReleaseScale", 0.2))
        max_per = int(self.ss.cfg.get_cfg("boundaryReleaseMaxPerStep", 8))
        if max_per <= 0 or scale == 0.0:
            return

        center_cfg = self.ss.cfg.get_cfg("boundaryCircleCenter", [0.65, 0.5])
        circle_center = np.array(center_cfg, dtype=np.float32).ravel()
        cx = float(circle_center[0]) if circle_center.size > 0 else 0.65
        cy = float(circle_center[1]) if circle_center.size > 1 else 0.5
        radius = float(self.ss.cfg.get_cfg("boundaryCircleRadius", 0.1))
        rear_x = cx + float(self.ss.cfg.get_cfg("boundaryReleaseRearXOffset", 0.0))
        offset = float(self.ss.cfg.get_cfg("boundaryReleaseOffset", 0.018))
        st = int(self.ss.cfg.get_cfg("boundaryReleaseSegmentTypeId", self.ss.cfg.get_cfg("initSegmentTypeId", 0)))

        xm = self._g_x_minus.astype(np.float32, copy=False)
        xp = self._g_x_plus.astype(np.float32, copy=False)
        gg = self._g_gamma.astype(np.float32, copy=False)
        centers = 0.5 * (xm + xp)
        strong = np.abs(gg) >= threshold
        release_region = str(self.ss.cfg.get_cfg("boundaryReleaseRegion", "rear_half") or "rear_half").lower().strip()
        if release_region in ("whole", "all", "whole_circle", "circle"):
            rear = np.ones_like(strong, dtype=bool)
        else:
            rear = centers[:, 0] >= rear_x
        near_circle = np.linalg.norm(centers[:, :2] - np.array([cx, cy], dtype=np.float32)[None, :], axis=1) <= radius + 3.0 * max(offset, 1e-6)
        ids = np.nonzero(strong & rear & near_circle)[0]
        if ids.size == 0:
            return
        selection_mode = str(self.ss.cfg.get_cfg("boundaryReleaseSelection", "balanced_sign") or "balanced_sign").lower().strip()
        if selection_mode in ("balanced_sign", "sign_balanced", "positive_negative", "pos_neg"):
            pos = ids[gg[ids] > 0.0]
            neg = ids[gg[ids] < 0.0]
            pos = pos[np.argsort(-np.abs(gg[pos]))]
            neg = neg[np.argsort(-np.abs(gg[neg]))]
            half = max_per // 2
            chosen_parts = []
            if half > 0:
                chosen_parts.append(pos[:half])
                chosen_parts.append(neg[:half])
            chosen = np.concatenate(chosen_parts) if len(chosen_parts) > 0 else np.zeros((0,), dtype=np.int64)
            if chosen.size < max_per:
                chosen_set = set(int(i) for i in chosen.tolist())
                rest = np.array([int(i) for i in ids if int(i) not in chosen_set], dtype=np.int64)
                if rest.size > 0:
                    rest = rest[np.argsort(-np.abs(gg[rest]))]
                    chosen = np.concatenate([chosen, rest[: max_per - chosen.size]])
            ids = chosen.astype(np.int64, copy=False)
        else:
            order = ids[np.argsort(-np.abs(gg[ids]))]
            ids = order[:max_per]

        new_xm = np.zeros((ids.size, 3), dtype=np.float32)
        new_xp = np.zeros((ids.size, 3), dtype=np.float32)
        new_g = np.zeros((ids.size,), dtype=np.float32)
        cxy = np.array([cx, cy], dtype=np.float32)
        for out_i, src_i in enumerate(ids):
            c = centers[src_i].copy()
            n2 = c[:2] - cxy
            ln = float(np.linalg.norm(n2))
            if ln < 1e-8:
                n2 = np.array([1.0, 0.0], dtype=np.float32)
                ln = 1.0
            n2 = n2 / ln
            shift = np.array([n2[0] * offset, n2[1] * offset, 0.0], dtype=np.float32)
            new_xm[out_i] = xm[src_i] + shift
            new_xp[out_i] = xp[src_i] + shift
            new_g[out_i] = float(scale * gg[src_i])

        offset_idx = int(self.ss.segment_num[None])
        cap = int(self.ss.segment_max_num)
        if offset_idx >= cap:
            return
        n_new = min(int(ids.size), cap - offset_idx)
        if n_new <= 0:
            return
        self._commit_segments_kernel(offset_idx, n_new, new_xm[:n_new], new_xp[:n_new], new_g[:n_new], st)
        self.ss.segment_num[None] = offset_idx + n_new

        if bool(self.ss.cfg.get_cfg("boundaryReleaseLog", False)):
            print(
                f"[boundary-release] released {n_new} internal segments "
                f"(|gamma| max={float(np.max(np.abs(new_g[:n_new]))):.4e})"
            )

    def commit_boundary_segments(self):
        """
        TODO：
        将求解后的边界虚拟段合并到全局段池，
        或写入用于速度评估的边界段池。
        """
        if not self.enable_boundary_injection:
            self._last_boundary_commit_count = 0
            return

        if not self._g_initialized or self._g_x_minus is None or self._g_x_plus is None or self._g_gamma is None:
            self._last_boundary_commit_count = 0
            return

        ng = int(self._g_x_minus.shape[0])
        if ng == 0:
            self._last_boundary_commit_count = 0
            return

        active_mask = self._g_active.astype(bool, copy=False) if self._g_active is not None else np.ones((ng,), dtype=bool)
        active_ids = np.nonzero(active_mask)[0]
        if active_ids.size == 0:
            return

        # 仅提交强度非零/足够大的段（避免把求解噪声写入段池）
        gamma_eps = float(self.ss.cfg.get_cfg("boundaryCommitGammaThreshold", 0.0))
        if gamma_eps > 0.0:
            keep = np.abs(self._g_gamma[active_ids]) >= gamma_eps
            active_ids = active_ids[keep]
            if active_ids.size == 0:
                self._last_boundary_commit_count = 0
                return

        xm = self._g_x_minus[active_ids].astype(np.float32, copy=False)
        xp = self._g_x_plus[active_ids].astype(np.float32, copy=False)
        gg = self._g_gamma[active_ids].astype(np.float32, copy=False)

        n_new = int(active_ids.size)
        offset = int(self.ss.segment_num[None])
        capacity = int(self.ss.segment_max_num)
        if offset >= capacity:
            self._last_boundary_commit_count = 0
            return

        # 截断以防止越界
        n_new = min(n_new, capacity - offset)
        xm = xm[:n_new]
        xp = xp[:n_new]
        gg = gg[:n_new]

        seg_type = int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2))
        self._commit_segments_kernel(offset, n_new, xm, xp, gg, seg_type)
        self.ss.segment_num[None] = offset + n_new
        self._last_boundary_commit_count = int(n_new)

        if bool(self.ss.cfg.get_cfg("boundaryInjectionLog", False)):
            gmax = float(np.max(np.abs(gg))) if n_new > 0 else 0.0
            c = 0.5 * (xm + xp)
            Lseg = np.linalg.norm(xp - xm, axis=1)
            dim = int(self.ss.dim)
            lo = self.ss.domain_start.astype(np.float64)[:dim]
            hi = self.ss.domain_end.astype(np.float64)[:dim]
            inside = np.all(
                (c[:, :dim] >= lo[None, :]) & (c[:, :dim] <= hi[None, :]), axis=1
            )
            print(
                f"[boundary] committed {n_new} virtual segments "
                f"(type={seg_type}, |gamma|_max={gmax:.4e}, "
                f"segment_num={int(self.ss.segment_num[None])})"
            )
            print(
                f"[boundary] center x=[{float(c[:, 0].min()):.4f},{float(c[:, 0].max()):.4f}] "
                f"y=[{float(c[:, 1].min()):.4f},{float(c[:, 1].max()):.4f}] "
                f"|L| med={float(np.median(Lseg)):.4f} "
                f"in_segment_domain={float(np.mean(inside)):.3f}"
            )

        # 可选：提交后失活候选段，避免被重复提交
        if bool(self.ss.cfg.get_cfg("boundaryClearCandidatesAfterCommit", True)):
            self._g_active[:] = False
