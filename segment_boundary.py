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

        # 缓存 rigid blocks 以及（可选）动态平移信息
        self._rigid_blocks = self.ss.cfg.get_obstacles() if hasattr(self.ss.cfg, "get_obstacles") else []
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
            # 可通过配置覆盖随机种子，保证可复现
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

    def generate_boundary_segments(self):
        """
        TODO：
        在边界采样点附近生成候选虚拟段。
        这些段暂不写入全局段池。
        """
        if not self.enable_boundary_injection:
            return

        if self._b_points is None or self._b_vel is None or self._b_owner is None:
            # 还未初始化边界采样点
            return

        nb = int(self._b_points.shape[0])
        if nb == 0:
            return

        ng = int(self.ng if self.ng > 0 else nb)
        ng = min(ng, nb)

        rng = np.random.default_rng(self._g_seed)
        # 若 ng < nb，则随机选一部分边界采样点生成候选段
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
        L = float(self._g_length)
        inset = float(self._g_inset)

        for k in range(ng):
            bi = int(chosen[k])
            p = self._b_points[bi]
            owner = int(self._b_owner[bi])

            if owner < self._num_aabb_obstacles:
                blk = self._rigid_blocks[owner]
                lo, hi = self._apply_block_transform(blk, translation_override=self._block_translation[owner])
                n = self._estimate_aabb_normal(p, lo, hi)
            else:
                ci = owner - self._num_aabb_obstacles
                cyl = self._cylinders[ci]
                n = self._estimate_cylinder_normal(p, cyl, translation_override=self._cylinder_translation[ci])

            # 将段放到“流体侧”一点点（沿 -n 方向内缩）
            p_in = p - inset * n

            # 构造段的方向 t：优先取 n × u_inf（使其与边界/来流都有关系）
            t = np.cross(n, u_dir)
            t = self._normalize(t)
            if float(np.linalg.norm(t)) < 1e-6:
                # 若 n 与 u_dir 平行导致叉积接近 0，则选一个稳定的切向方向兜底
                t = self._pick_perpendicular(n)

            g_xm[k] = p_in - 0.5 * L * t
            g_xp[k] = p_in + 0.5 * L * t

        self._g_x_minus = g_xm
        self._g_x_plus = g_xp
        self._g_gamma = g_gamma
        self._g_active = g_active
        self._g_owner_b = g_owner_b
        self._g_initialized = True

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

        if use_finite:
            # K 列 a：单位强度（gamma=1）虚拟段在边界点上的诱导速度；Γ=L_a。
            for a in range(ng):
                u = self._bs_velocity_finite_line(b, xm[a], xp[a], 1.0, R)
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

                if use_finite:
                    for j in range(Ns):
                        if active[j] != 1:
                            continue
                        u_d += self._bs_velocity_finite_line(b, xm[j], xp[j], float(gamma[j]), R)
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

    def commit_boundary_segments(self):
        """
        TODO：
        将求解后的边界虚拟段合并到全局段池，
        或写入用于速度评估的边界段池。
        """
        if not self.enable_boundary_injection:
            return

        if not self._g_initialized or self._g_x_minus is None or self._g_x_plus is None or self._g_gamma is None:
            return

        ng = int(self._g_x_minus.shape[0])
        if ng == 0:
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
                return

        xm = self._g_x_minus[active_ids].astype(np.float32, copy=False)
        xp = self._g_x_plus[active_ids].astype(np.float32, copy=False)
        gg = self._g_gamma[active_ids].astype(np.float32, copy=False)

        n_new = int(active_ids.size)
        offset = int(self.ss.segment_num[None])
        capacity = int(self.ss.segment_max_num)
        if offset >= capacity:
            return

        # 截断以防止越界
        n_new = min(n_new, capacity - offset)
        xm = xm[:n_new]
        xp = xp[:n_new]
        gg = gg[:n_new]

        seg_type = int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2))
        self._commit_segments_kernel(offset, n_new, xm, xp, gg, seg_type)
        self.ss.segment_num[None] = offset + n_new

        # 可选：提交后失活候选段，避免被重复提交
        if bool(self.ss.cfg.get_cfg("boundaryClearCandidatesAfterCommit", True)):
            self._g_active[:] = False
