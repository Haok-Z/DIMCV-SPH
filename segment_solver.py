import taichi as ti
import numpy as np

from segment_boundary import SegmentBoundaryHandler


@ti.data_oriented
class SegmentSolver:
    def __init__(self, segment_system):
        self.ss = segment_system
        self.dt = float(self.ss.cfg.get_cfg("timeStepSize", 0.002))
        self.reg_radius = float(self.ss.cfg.get_cfg("regularizationRadiusR", 0.01))
        self.gamma_decay = float(self.ss.cfg.get_cfg("gammaDecay", 1.0))
        self.split_len_threshold = float(self.ss.cfg.get_cfg("splitLengthThreshold", 0.08))
        self.delete_gamma_threshold = float(self.ss.cfg.get_cfg("deleteGammaThreshold", 1e-4))
        self.background_velocity = np.array(
            self.ss.cfg.get_cfg("backgroundVelocity", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.u_inf = ti.Vector.field(3, dtype=float, shape=())
        self.u_inf.from_numpy(self.background_velocity)

        self.boundary = SegmentBoundaryHandler(self.ss)
        self.has_boundary = self.boundary.enable_boundary_injection

        self.v_minus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.v_plus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)

        # RK4 需要的中间导数（端点速度）
        self.k1_minus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.k2_minus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.k3_minus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.k4_minus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)

        self.k1_plus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.k2_plus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.k3_plus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)
        self.k4_plus = ti.Vector.field(3, dtype=float, shape=self.ss.segment_max_num)

    def initialize(self):
        self.ss.clear()
        self.seed_initial_segments()
        self.ss.update_segment_geometry()

    def seed_initial_segments(self):
        """
        TODO：
        按用户指定方式初始化段云，例如：
        - 解析形式的涡环 / 涡管
        - 基于入口发射器区域的初始播种
        - 从预计算段文件加载
        """
        init_type = self.ss.cfg.get_cfg("initType", "ring")
        if init_type is None:
            init_type = "ring"
        init_type = str(init_type).lower()

        if init_type == "none":
            return

        if init_type == "triple_parallel_filaments_x":
            self._seed_triple_parallel_filaments_x()
            return

        if init_type == "v_bundle_pair":
            self._seed_v_bundle_pair()
            return

        if init_type != "ring":
            raise NotImplementedError(
                f"initType={init_type} 尚未实现（当前支持 ring / triple_parallel_filaments_x / v_bundle_pair / none）"
            )

        # 读取初始化参数
        n_seg = int(self.ss.cfg.get_cfg("initSegmentNum", 64))
        n_seg = max(3, n_seg)
        ring_radius = float(self.ss.cfg.get_cfg("initRingRadius", 0.15))
        gamma0 = float(self.ss.cfg.get_cfg("initGamma", 0.25))
        seg_type = int(self.ss.cfg.get_cfg("initSegmentTypeId", 0))

        # 环中心：默认域中心，也允许配置覆盖
        center_cfg = self.ss.cfg.get_cfg("initCenter", None)
        if center_cfg is None:
            c = 0.5 * (self.ss.domain_start + self.ss.domain_end)
        else:
            c = np.array(center_cfg, dtype=np.float32)

        # 环法向：默认取背景速度方向（若为 0 则回退到 x 轴）
        axis_cfg = self.ss.cfg.get_cfg("initRingAxis", None)
        if axis_cfg is None:
            axis = self.background_velocity.copy()
            if float(np.linalg.norm(axis)) < 1e-6:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis = np.array(axis_cfg, dtype=np.float32)

        axis = axis / (np.linalg.norm(axis) + 1e-8)

        # 构造环平面上的正交基 (e1, e2)
        # 选择一个不与 axis 共线的向量作为参考
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(axis, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e1 = np.cross(axis, ref)
        e1 = e1 / (np.linalg.norm(e1) + 1e-8)
        e2 = np.cross(axis, e1)
        e2 = e2 / (np.linalg.norm(e2) + 1e-8)

        # 生成环上点并拼成 segments（相邻点连线）
        angles = np.linspace(0.0, 2.0 * np.pi, n_seg + 1, dtype=np.float32)[:-1]
        pts = (c[None, :] + ring_radius * (np.cos(angles)[:, None] * e1[None, :] +
                                           np.sin(angles)[:, None] * e2[None, :])).astype(np.float32)

        x_minus = pts
        x_plus = np.roll(pts, shift=-1, axis=0)
        gamma = (gamma0 * np.ones((n_seg,), dtype=np.float32))

        offset = int(self.ss.segment_num[None])
        if offset >= int(self.ss.segment_max_num):
            return

        n_new = min(n_seg, int(self.ss.segment_max_num) - offset)
        if n_new <= 0:
            return

        self._seed_segments_kernel(offset, n_new, x_minus[:n_new], x_plus[:n_new], gamma[:n_new], seg_type)
        self.ss.segment_num[None] = offset + n_new

    def _seed_triple_parallel_filaments_x(self):
        """
        初始化沿 x 轴方向的平行“很长涡丝”（每条一个长 segment）。

        设计意图：多条同向平行涡丝在各自的诱导作用下，易产生绕 x 轴的整体旋转趋势。

        使用 SegmentConfiguration 的可选参数：
        - initFilamentCount（默认 3，支持任意 >=2）
        - initFilamentGamma（默认 initGamma）
        - initFilamentRadiusYZ（默认 0.18）：涡丝在 y-z 平面上的环半径
        - initFilamentYZCenter（默认 initCenter 的 y/z）：y-z 平面中心
        - initFilamentMargin（默认 0.05）：x 方向端点离域边界的安全边距
        - initSegmentTypeId（段类型标记）
        """
        n_fil = int(self.ss.cfg.get_cfg("initFilamentCount", 3))
        if n_fil < 2:
            raise ValueError("initFilamentCount 必须 >= 2")

        gamma0 = float(self.ss.cfg.get_cfg("initFilamentGamma", self.ss.cfg.get_cfg("initGamma", 0.25)))
        seg_type = int(self.ss.cfg.get_cfg("initSegmentTypeId", 0))

        x_margin = float(self.ss.cfg.get_cfg("initFilamentMargin", 0.05))
        x0 = float(self.ss.domain_start[0] + x_margin)
        x1 = float(self.ss.domain_end[0] - x_margin)

        yz_center = self.ss.cfg.get_cfg("initFilamentYZCenter", None)
        if yz_center is None:
            init_center = self.ss.cfg.get_cfg("initCenter", None)
            if init_center is None:
                init_center = 0.5 * (self.ss.domain_start + self.ss.domain_end)
            yz_center = [float(init_center[1]), float(init_center[2])]
        cy = float(yz_center[0])
        cz = float(yz_center[1])

        r_yz = float(self.ss.cfg.get_cfg("initFilamentRadiusYZ", 0.18))
        # 在 y-z 平面按环半径均匀分配角度位置
        angles = np.linspace(0.0, 2.0 * np.pi, n_fil, endpoint=False, dtype=np.float32)
        y_off = (r_yz * np.cos(angles)).astype(np.float32)
        z_off = (r_yz * np.sin(angles)).astype(np.float32)

        x_minus = np.zeros((n_fil, 3), dtype=np.float32)
        x_plus = np.zeros((n_fil, 3), dtype=np.float32)
        gamma = (gamma0 * np.ones((n_fil,), dtype=np.float32))

        for k in range(n_fil):
            y = cy + float(y_off[k])
            z = cz + float(z_off[k])
            x_minus[k] = np.array([x0, y, z], dtype=np.float32)
            x_plus[k] = np.array([x1, y, z], dtype=np.float32)

        offset = int(self.ss.segment_num[None])
        if offset >= int(self.ss.segment_max_num):
            return

        n_new = min(n_fil, int(self.ss.segment_max_num) - offset)
        if n_new <= 0:
            return

        self._seed_segments_kernel(
            offset,
            n_new,
            x_minus[:n_new],
            x_plus[:n_new],
            gamma[:n_new],
            seg_type,
        )
        self.ss.segment_num[None] = offset + n_new

    def _seed_v_bundle_pair(self):
        """
        初始化一对形如 >< 的折线涡丝簇。
        每个簇用两段折线骨架表示，每段上均匀离散多个中心点，
        再在每个中心点周围的正交平面上生成若干条“小涡丝”段，形成一个簇。

        SegmentConfiguration 可用参数：
        - vBundleCenter: [x, y, z]，整体中心，默认 domain 中心
        - vBundleSize: [sx, sy, sz]，控制折线几何尺度，默认 [1.2, 0.6, 0.4]
        - vBundleSegmentsPerLeg: 每条骨架腿离散段数（默认 16）
        - vBundleFilamentsPerCenter: 每个中心点周围生成的涡丝条数（默认 6）
        - vBundleRadius: 每个簇在正交平面的截面半径（默认 0.05）
        - vBundleGamma: 涡丝强度（默认 initGamma）
        - initSegmentTypeId: 段类型
        """
        cfg = self.ss.cfg

        # 几何中心与尺度
        center_cfg = cfg.get_cfg("vBundleCenter", None)
        if center_cfg is None:
            c = 0.5 * (self.ss.domain_start + self.ss.domain_end)
        else:
            c = np.array(center_cfg, dtype=np.float32)

        size_cfg = cfg.get_cfg("vBundleSize", [2.0, 0.4, 0.4])
        s = np.array(size_cfg, dtype=np.float32)

        # 折线骨架点：在 x-z 平面做张开角度接近 180° 的 >< 结构，
        #            两个长条整体主要沿 x 轴近似平行，y 固定为 c[1]
        y0 = float(c[1])
        hx = 0.5 * float(s[0])   # 沿 x 的一半长度（控制整体条带长度）
        dz = 0.5 * float(s[2])   # 在 z 方向的轻微折弯量（决定开口角度）

        # 左簇骨架：P0 -> P1 -> P2（整体略向上倾斜）
        P0 = np.array([c[0] - hx, y0, c[2] - dz], dtype=np.float32)
        P1 = np.array([c[0],      y0, c[2]],      dtype=np.float32)
        P2 = np.array([c[0] + hx, y0, c[2] + dz], dtype=np.float32)
        # 右簇骨架：Q0 -> Q1 -> Q2（与左簇在 z 上镜像，对出一个 ><）
        Q0 = np.array([c[0] - hx, y0, c[2] + dz], dtype=np.float32)
        Q1 = np.array([c[0],      y0, c[2]],      dtype=np.float32)
        Q2 = np.array([c[0] + hx, y0, c[2] - dz], dtype=np.float32)

        segments_per_leg = int(cfg.get_cfg("vBundleSegmentsPerLeg", 16))
        segments_per_leg = max(1, segments_per_leg)
        filaments_per_center = int(cfg.get_cfg("vBundleFilamentsPerCenter", 6))
        filaments_per_center = max(1, filaments_per_center)
        bundle_radius = float(cfg.get_cfg("vBundleRadius", 0.05))
        gamma0 = float(cfg.get_cfg("vBundleGamma", cfg.get_cfg("initGamma", 0.2)))
        seg_type = int(cfg.get_cfg("initSegmentTypeId", 0))

        def build_leg_points(A, B):
            """离散单条骨架腿，返回中心点与方向。"""
            A = A.astype(np.float32)
            B = B.astype(np.float32)
            dir_vec = B - A
            L = np.linalg.norm(dir_vec) + 1e-8
            t = dir_vec / L
            # 中心点：均匀分布在 (0,1) 区间
            alphas = (np.arange(segments_per_leg, dtype=np.float32) + 0.5) / float(segments_per_leg)
            centers = (A[None, :] + alphas[:, None] * dir_vec[None, :]).astype(np.float32)

            # 为每条腿构造两个正交向量 e1, e2（垂直于 t），用于在截面上打圈
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(t, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            e1 = np.cross(t, ref)
            e1 = e1 / (np.linalg.norm(e1) + 1e-8)
            e2 = np.cross(t, e1)
            e2 = e2 / (np.linalg.norm(e2) + 1e-8)
            return centers, t, e1, e2, L

        bundle_centers = []
        bundle_dirs = []
        bundle_e1e2 = []
        bundle_seg_len = []

        # 左簇：两条腿 P0-P1, P1-P2
        for A, B in [(P0, P1), (P1, P2)]:
            centers, t, e1, e2, L_leg = build_leg_points(A, B)
            bundle_centers.append(centers)
            bundle_dirs.append(t)
            bundle_e1e2.append((e1, e2))
            # 每个小段的长度取为腿总长的 1/segments_per_leg
            bundle_seg_len.append(L_leg / float(segments_per_leg))

        # 右簇：两条腿 Q0-Q1, Q1-Q2
        for A, B in [(Q0, Q1), (Q1, Q2)]:
            centers, t, e1, e2, L_leg = build_leg_points(A, B)
            bundle_centers.append(centers)
            bundle_dirs.append(t)
            bundle_e1e2.append((e1, e2))
            bundle_seg_len.append(L_leg / float(segments_per_leg))

        # 统计将要生成的 segment 数量
        total_centers = sum(c.shape[0] for c in bundle_centers)
        n_new = total_centers * filaments_per_center

        offset = int(self.ss.segment_num[None])
        capacity = int(self.ss.segment_max_num)
        if offset >= capacity or n_new <= 0:
            return
        n_new = min(n_new, capacity - offset)

        x_minus = np.zeros((n_new, 3), dtype=np.float32)
        x_plus = np.zeros((n_new, 3), dtype=np.float32)
        gamma = (gamma0 * np.ones((n_new,), dtype=np.float32))

        idx = 0
        for leg_idx, centers in enumerate(bundle_centers):
            t = bundle_dirs[leg_idx]
            e1, e2 = bundle_e1e2[leg_idx]
            seg_len = bundle_seg_len[leg_idx]

            # 每个中心点周围生成 filaments_per_center 条小段，截面上按等角分布
            thetas = np.linspace(0.0, 2.0 * np.pi, filaments_per_center, endpoint=False, dtype=np.float32)
            for c0 in centers:
                for th in thetas:
                    if idx >= n_new:
                        break
                    offset_vec = bundle_radius * (np.cos(th) * e1 + np.sin(th) * e2)
                    center_seg = c0 + offset_vec
                    xm = center_seg - 0.5 * seg_len * t
                    xp = center_seg + 0.5 * seg_len * t
                    x_minus[idx] = xm.astype(np.float32)
                    x_plus[idx] = xp.astype(np.float32)
                    idx += 1
                if idx >= n_new:
                    break
            if idx >= n_new:
                break

        # 实际生成数量可能小于 n_new（若提前打满 capacity），按 idx 截断
        n_final = idx
        if n_final <= 0:
            return

        self._seed_segments_kernel(
            offset,
            n_final,
            x_minus[:n_final],
            x_plus[:n_final],
            gamma[:n_final],
            seg_type,
        )
        self.ss.segment_num[None] = offset + n_final

    @ti.kernel
    def _seed_segments_kernel(
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
            self.ss.x_minus[i] = ti.Vector([x_minus[k, 0], x_minus[k, 1], x_minus[k, 2]])
            self.ss.x_plus[i] = ti.Vector([x_plus[k, 0], x_plus[k, 1], x_plus[k, 2]])
            self.ss.gamma[i] = gamma[k]
            self.ss.active[i] = 1
            self.ss.age[i] = 0.0
            self.ss.seg_type[i] = seg_type

    def compute_endpoint_velocity(self):
        """
        TODO：
        为每条活跃段计算两端点速度。
        预期行为：
        1) 从背景速度 u_inf 开始
        2) 叠加所有活跃内部段的诱导速度
        3) 叠加活跃边界虚拟段的诱导速度
        4) 结果写入 self.v_minus / self.v_plus

        说明：
        - 第一版可用 O(N^2) 直接求和
        - 后续可用网格或多极近似进行加速
        """
        self._compute_endpoint_velocity_bs(float(self.reg_radius))

    @ti.kernel
    def _compute_endpoint_velocity_bs(self, reg_radius: float):
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = reg_radius * reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i]
            xpi = self.ss.x_plus[i]
            ui_m = self.u_inf[None]
            ui_p = self.u_inf[None]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                xmj = self.ss.x_minus[j]
                xpj = self.ss.x_plus[j]
                cj = 0.5 * (xmj + xpj)
                dj = xpj - xmj
                Lj = dj.norm() + 1e-8
                tj = dj / Lj
                omega = self.ss.gamma[j] * Lj * tj  # (3,)

                # endpoint x_minus
                r = xmi - cj
                r2 = r.dot(r) + R2
                denom = r2 * ti.sqrt(r2)
                ui_m += inv4pi * omega.cross(r) / (denom + 1e-12)

                # endpoint x_plus
                r2p = (xpi - cj)
                s2 = r2p.dot(r2p) + R2
                denom_p = s2 * ti.sqrt(s2)
                ui_p += inv4pi * omega.cross(r2p) / (denom_p + 1e-12)

            self.v_minus[i] = ui_m
            self.v_plus[i] = ui_p

    def advect_segments_rk4(self):
        """
        TODO：
        使用 RK4 对每条段的两个端点进行对流推进：
            dx_minus/dt = u(x_minus)
            dx_plus/dt  = u(x_plus)

        建议分阶段实现：
        - 先用临时 Euler 版本打通流程
        - u_at_point 就绪后替换为完整 RK4
        """
        # 1) k1：在当前端点位置计算速度
        self._copy_v_to_k1()

        # 2) k2：在 x + 0.5*dt*k1 位置计算速度
        self._compute_velocity_at_factor(0.5, self.k1_minus, self.k1_plus,
                                          self.k2_minus, self.k2_plus)

        # 3) k3：在 x + 0.5*dt*k2 位置计算速度
        self._compute_velocity_at_factor(0.5, self.k2_minus, self.k2_plus,
                                          self.k3_minus, self.k3_plus)

        # 4) k4：在 x + dt*k3 位置计算速度
        self._compute_velocity_at_factor(1.0, self.k3_minus, self.k3_plus,
                                          self.k4_minus, self.k4_plus)

        # 5) 更新端点位置并推进 gamma / age
        self._update_endpoints_rk4()

    @ti.kernel
    def _advect_segments_euler_placeholder(self):
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] == 1:
                self.ss.x_minus[i] += self.dt * self.v_minus[i]
                self.ss.x_plus[i] += self.dt * self.v_plus[i]
                self.ss.gamma[i] *= self.gamma_decay
                self.ss.age[i] += self.dt

    @ti.kernel
    def _copy_v_to_k1(self):
        """
        将当前端点速度（v_minus/v_plus）复制为 RK4 的 k1。
        """
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] == 1:
                self.k1_minus[i] = self.v_minus[i]
                self.k1_plus[i] = self.v_plus[i]

    @ti.kernel
    def _compute_velocity_at_factor(
        self,
        factor: float,
        k_minus_in: ti.template(),
        k_plus_in: ti.template(),
        out_minus: ti.template(),
        out_plus: ti.template(),
    ):
        """
        计算速度：在查询点
            x_minus_query = x_minus + factor * dt * k_minus_in
            x_plus_query  = x_plus  + factor * dt * k_plus_in
        处求 u(x)。

        该 kernel 复用当前段池中段的几何（端点/强度不在 RK4 子步变化）。
        """
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = self.reg_radius * self.reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i] + factor * self.dt * k_minus_in[i]
            xpi = self.ss.x_plus[i] + factor * self.dt * k_plus_in[i]

            ui_m = self.u_inf[None]
            ui_p = self.u_inf[None]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                xmj = self.ss.x_minus[j]
                xpj = self.ss.x_plus[j]
                dj = xpj - xmj  # d = x_plus - x_minus
                omega = self.ss.gamma[j] * dj  # 等效涡量向量（单位强度下的正则化形式）

                # endpoint x_minus_query 的诱导速度
                r_m = xmi - (0.5 * (xmj + xpj))
                r2_m = r_m.dot(r_m) + R2
                denom_m = r2_m * ti.sqrt(r2_m) + 1e-12
                ui_m += inv4pi * omega.cross(r_m) / denom_m

                # endpoint x_plus_query 的诱导速度
                r_p = xpi - (0.5 * (xmj + xpj))
                r2_p = r_p.dot(r_p) + R2
                denom_p = r2_p * ti.sqrt(r2_p) + 1e-12
                ui_p += inv4pi * omega.cross(r_p) / denom_p

            out_minus[i] = ui_m
            out_plus[i] = ui_p

    @ti.kernel
    def _update_endpoints_rk4(self):
        """
        使用 RK4 更新端点位置：
            x_new = x + dt/6 * (k1 + 2k2 + 2k3 + k4)
        """
        n = self.ss.segment_num[None]
        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            self.ss.x_minus[i] += (self.dt / 6.0) * (
                self.k1_minus[i] + 2.0 * self.k2_minus[i] + 2.0 * self.k3_minus[i] + self.k4_minus[i]
            )
            self.ss.x_plus[i] += (self.dt / 6.0) * (
                self.k1_plus[i] + 2.0 * self.k2_plus[i] + 2.0 * self.k3_plus[i] + self.k4_plus[i]
            )

            self.ss.gamma[i] *= self.gamma_decay
            self.ss.age[i] += self.dt

    def delete_weak_segments(self):
        """
        删除弱段并压缩段池：
        - 必选条件：|gamma| >= delete_gamma_threshold
        - 可选条件：年龄阈值、是否删除域外段
        """
        n = int(self.ss.segment_num[None])
        if n <= 0:
            return

        x_minus = self.ss.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss.active.to_numpy()[:n].astype(np.int32)
        age = self.ss.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss.seg_type.to_numpy()[:n].astype(np.int32)

        keep = (active == 1) & (np.abs(gamma) >= float(self.delete_gamma_threshold))

        # 可选：按年龄删除
        max_age = self.ss.cfg.get_cfg("deleteMaxAge", None)
        if max_age is not None:
            keep &= (age <= float(max_age))

        # 可选：删除域外段（要求两端点都在域内）
        delete_outside = bool(self.ss.cfg.get_cfg("deleteOutsideDomain", False))
        if delete_outside:
            lo = self.ss.domain_start.astype(np.float32)
            hi = self.ss.domain_end.astype(np.float32)
            in_m = np.all((x_minus >= lo[None, :]) & (x_minus <= hi[None, :]), axis=1)
            in_p = np.all((x_plus >= lo[None, :]) & (x_plus <= hi[None, :]), axis=1)
            keep &= (in_m & in_p)

        idx = np.nonzero(keep)[0]
        new_n = int(idx.size)

        if new_n <= 0:
            self._clear_active_kernel(n)
            self.ss.segment_num[None] = 0
            return

        out_xm = x_minus[idx]
        out_xp = x_plus[idx]
        out_g = gamma[idx]
        out_a = age[idx]
        out_t = seg_type[idx]

        self._overwrite_segments_kernel(new_n, out_xm, out_xp, out_g, out_a, out_t)
        self.ss.segment_num[None] = new_n

    @ti.kernel
    def _clear_active_kernel(self, n_old: int):
        for i in range(n_old):
            self.ss.active[i] = 0

    def split_segments(self):
        """
        TODO（仅 3D）：
        对长度超过 split_len_threshold 的段进行分裂。
        典型操作：
        - m = 0.5 * (x_minus + x_plus)
        - 当前段更新为 [x_minus, m]
        - 追加新段 [m, x_plus]
        - 按守恒策略复制或重分配 gamma
        """
        old_n = int(self.ss.segment_num[None])
        if old_n <= 0:
            return
        self._split_segments_kernel(old_n, float(self.split_len_threshold))

    @ti.kernel
    def _split_segments_kernel(self, old_n: int, split_len_threshold: float):
        """
        分裂策略（论文式(12)的最小实现）：
        - 仅处理 old_n 范围内原有段，避免本次新增段再次被处理
        - 对每条满足 length > threshold 的活跃段：
          1) 计算中点 m
          2) 原段改为 [x_minus, m]
          3) 追加新段 [m, x_plus_old]
        - 新段继承 gamma / seg_type，age 置 0
        - 若容量不足则跳过追加（原段仍会被截断到一半）
        """
        max_n = self.ss.segment_max_num
        for i in range(old_n):
            if self.ss.active[i] != 1:
                continue
            if self.ss.length[i] <= split_len_threshold:
                continue

            xm = self.ss.x_minus[i]
            xp_old = self.ss.x_plus[i]
            mid = 0.5 * (xm + xp_old)

            # 更新原段为 [xm, mid]
            self.ss.x_plus[i] = mid

            # 追加新段 [mid, xp_old]
            new_idx = ti.atomic_add(self.ss.segment_num[None], 1)
            if new_idx < max_n:
                self.ss.x_minus[new_idx] = mid
                self.ss.x_plus[new_idx] = xp_old
                self.ss.gamma[new_idx] = self.ss.gamma[i]
                self.ss.active[new_idx] = 1
                self.ss.age[new_idx] = 0.0
                self.ss.seg_type[new_idx] = self.ss.seg_type[i]
            else:
                # 容量不足：回滚计数（尽量保持一致）
                ti.atomic_add(self.ss.segment_num[None], -1)

            # 立即更新几何缓存（原段）
            d0 = self.ss.x_plus[i] - self.ss.x_minus[i]
            l0 = d0.norm() + 1e-8
            self.ss.center[i] = 0.5 * (self.ss.x_plus[i] + self.ss.x_minus[i])
            self.ss.tangent[i] = d0 / l0
            self.ss.length[i] = l0

            # 立即更新几何缓存（新段）——仅当成功追加
            if new_idx < max_n:
                d1 = self.ss.x_plus[new_idx] - self.ss.x_minus[new_idx]
                l1 = d1.norm() + 1e-8
                self.ss.center[new_idx] = 0.5 * (self.ss.x_plus[new_idx] + self.ss.x_minus[new_idx])
                self.ss.tangent[new_idx] = d1 / l1
                self.ss.length[new_idx] = l1

    def merge_segments(self):
        """
        TODO（仅 3D）：
        合并满足以下条件的段对：
        - 中心距离 < mergeDistanceLambda
        - 方向判据满足（近反向或你选定的判据）
        - 可选：gamma 兼容性判据
        """
        n = int(self.ss.segment_num[None])
        if n <= 1:
            return

        merge_dist = float(self.ss.cfg.get_cfg("mergeDistanceLambda", 0.03))
        merge_angle = float(self.ss.cfg.get_cfg("mergeAngleThreshold", 5.0 * np.pi / 6.0))
        # 近反向：dot(t_i, t_j) <= cos(theta)，theta 接近 pi 时更严格
        dot_th = float(np.cos(merge_angle))

        x_minus = self.ss.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss.active.to_numpy()[:n].astype(np.int32)
        age = self.ss.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss.seg_type.to_numpy()[:n].astype(np.int32)
        center = self.ss.center.to_numpy()[:n].astype(np.float32)
        tangent = self.ss.tangent.to_numpy()[:n].astype(np.float32)
        length = self.ss.length.to_numpy()[:n].astype(np.float32)

        used = np.zeros((n,), dtype=bool)
        out_xm = []
        out_xp = []
        out_g = []
        out_a = []
        out_t = []

        for i in range(n):
            if active[i] != 1 or used[i]:
                continue

            # 找到可合并的最佳候选 j（最近）
            best_j = -1
            best_d2 = 1e30
            ci = center[i]
            ti = tangent[i]
            for j in range(i + 1, n):
                if active[j] != 1 or used[j]:
                    continue
                d2 = float(np.sum((ci - center[j]) ** 2))
                if d2 > merge_dist * merge_dist:
                    continue
                # 方向判据：近反向
                dotv = float(np.dot(ti, tangent[j]))
                if dotv > dot_th:
                    continue
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j

            if best_j < 0:
                # 不合并，直接保留
                used[i] = True
                out_xm.append(x_minus[i])
                out_xp.append(x_plus[i])
                out_g.append(gamma[i])
                out_a.append(age[i])
                out_t.append(seg_type[i])
                continue

            j = best_j
            used[i] = True
            used[j] = True

            # 合并规则：用“涡量向量”守恒构造新段
            di = x_plus[i] - x_minus[i]
            dj = x_plus[j] - x_minus[j]
            Li = float(np.linalg.norm(di) + 1e-8)
            Lj = float(np.linalg.norm(dj) + 1e-8)
            wi = gamma[i] * di
            wj = gamma[j] * dj
            w = wi + wj

            c_new = 0.5 * (center[i] + center[j])
            L_new = 0.5 * (Li + Lj)
            w_norm = float(np.linalg.norm(w))

            if w_norm < 1e-8:
                # 两段近乎相互抵消：等效删除（不输出）
                continue

            t_new = (w / (w_norm + 1e-8)).astype(np.float32)
            xm_new = c_new - 0.5 * L_new * t_new
            xp_new = c_new + 0.5 * L_new * t_new
            g_new = w_norm / (L_new + 1e-8)

            out_xm.append(xm_new.astype(np.float32))
            out_xp.append(xp_new.astype(np.float32))
            out_g.append(np.float32(g_new))
            out_a.append(np.float32(min(age[i], age[j])))
            out_t.append(seg_type[i])

        new_n = len(out_xm)
        if new_n <= 0:
            self.ss.segment_num[None] = 0
            return

        out_xm = np.asarray(out_xm, dtype=np.float32)
        out_xp = np.asarray(out_xp, dtype=np.float32)
        out_g = np.asarray(out_g, dtype=np.float32)
        out_a = np.asarray(out_a, dtype=np.float32)
        out_t = np.asarray(out_t, dtype=np.int32)

        self._overwrite_segments_kernel(new_n, out_xm, out_xp, out_g, out_a, out_t)
        self.ss.segment_num[None] = new_n

    @ti.kernel
    def _overwrite_segments_kernel(
        self,
        new_n: int,
        x_minus: ti.types.ndarray(),
        x_plus: ti.types.ndarray(),
        gamma: ti.types.ndarray(),
        age: ti.types.ndarray(),
        seg_type: ti.types.ndarray(),
    ):
        # 先清理旧 active，避免尾部脏数据影响后续逻辑
        for i in range(self.ss.segment_num[None]):
            self.ss.active[i] = 0

        for i in range(new_n):
            self.ss.x_minus[i] = ti.Vector([x_minus[i, 0], x_minus[i, 1], x_minus[i, 2]])
            self.ss.x_plus[i] = ti.Vector([x_plus[i, 0], x_plus[i, 1], x_plus[i, 2]])
            self.ss.gamma[i] = gamma[i]
            self.ss.age[i] = age[i]
            self.ss.seg_type[i] = seg_type[i]
            self.ss.active[i] = 1

    def cull_segments(self):
        """
        TODO：
        可选后处理清理，例如：
        - 删除域外段
        - 删除寿命过长段
        - 删除远场低影响段
        """
        n = int(self.ss.segment_num[None])
        if n <= 0:
            return

        # 总开关：关闭时直接返回
        if not bool(self.ss.cfg.get_cfg("enableCullSegments", True)):
            return

        x_minus = self.ss.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss.active.to_numpy()[:n].astype(np.int32)
        age = self.ss.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss.seg_type.to_numpy()[:n].astype(np.int32)
        length = self.ss.length.to_numpy()[:n].astype(np.float32)
        center = self.ss.center.to_numpy()[:n].astype(np.float32)

        keep = (active == 1)

        # 1) 域约束：中心点在 [domain_start-margin, domain_end+margin] 内
        cull_outside = bool(self.ss.cfg.get_cfg("cullOutsideDomain", True))
        if cull_outside:
            margin = float(self.ss.cfg.get_cfg("cullDomainMargin", 0.0))
            lo = (self.ss.domain_start - margin).astype(np.float32)
            hi = (self.ss.domain_end + margin).astype(np.float32)
            in_center = np.all((center >= lo[None, :]) & (center <= hi[None, :]), axis=1)
            keep &= in_center

        # 2) 长度约束（可选）
        min_len = self.ss.cfg.get_cfg("cullMinLength", None)
        if min_len is not None:
            keep &= (length >= float(min_len))

        max_len = self.ss.cfg.get_cfg("cullMaxLength", None)
        if max_len is not None:
            keep &= (length <= float(max_len))

        # 3) 年龄约束（可选）
        cull_max_age = self.ss.cfg.get_cfg("cullMaxAge", None)
        if cull_max_age is not None:
            keep &= (age <= float(cull_max_age))

        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            self._clear_active_kernel(n)
            self.ss.segment_num[None] = 0
            return

        # 4) 数量上限约束（可选）：按 |gamma| 重要性保留前 K 个
        max_keep = self.ss.cfg.get_cfg("cullMaxSegments", None)
        if max_keep is not None:
            k = int(max_keep)
            if k > 0 and idx.size > k:
                imp = np.abs(gamma[idx]).astype(np.float32)
                # 保留重要性最大的 k 个（argpartition O(N)）
                pick_local = np.argpartition(-imp, kth=k - 1)[:k]
                idx = idx[pick_local]

        # 统一按原索引排序，减少时序跳变
        idx = np.sort(idx)
        new_n = int(idx.size)

        out_xm = x_minus[idx]
        out_xp = x_plus[idx]
        out_g = gamma[idx]
        out_a = age[idx]
        out_t = seg_type[idx]

        self._overwrite_segments_kernel(new_n, out_xm, out_xp, out_g, out_a, out_t)
        self.ss.segment_num[None] = new_n

    def step(self):
        """
        独立 Segment 仿真的主步骤模板：
        1) 更新边界几何（若存在边界）
        2) 生成并求解边界虚拟段
        3) 计算端点速度场贡献
        4) 对端点进行对流推进（RK4）
        5) 拓扑操作：split / merge / delete
        6) 可选清理
        """
        if self.has_boundary:
            self.boundary.update_boundary_pose()
            self.boundary.generate_boundary_segments()
            self.boundary.compute_k_matrix()
            self.boundary.compute_rhs()
            self.boundary.solve_linear_system()
            self.boundary.commit_boundary_segments()

        self.compute_endpoint_velocity()
        self.advect_segments_rk4()
        self.ss.update_segment_geometry()
        self.split_segments()
        self.merge_segments()
        self.delete_weak_segments()
        #self.cull_segments()
