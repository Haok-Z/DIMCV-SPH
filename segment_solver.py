import taichi as ti
import numpy as np
import time
from typing import List, Set, Tuple

from segment_boundary import SegmentBoundaryHandler


def _segment_cfg_biot_savart_is_finite(cfg) -> bool:
    """
    SegmentConfiguration.biotSavartModel:
    - finite_segment（默认）：有限长直线段 Biot–Savart 闭式（与 Vortex Segment 类论文常用离散一致）
    - blob_center：段中点 + 平滑 blob（旧实现，数值更钝、易调）
    - point_vortex_2d：二维点涡式 (7)，见 _segment_cfg_use_2d_point_vortex_bs
    """
    m = cfg.get_cfg("biotSavartModel", "finite_segment")
    if m is None:
        return True
    m = str(m).lower().strip()
    if m in ("blob", "blob_center", "center_blob", "lumped"):
        return False
    if m in ("point_vortex_2d", "2d_point", "eq7", "formula_7", "2d"):
        return False
    return True


def _segment_cfg_use_2d_point_vortex_bs(cfg, spatial_dim: int) -> bool:
    """TOG2021 式 (7)：u = Γ/(2π) e_z × (x−x_j) / (|x−x_j|²+R²)。"""
    m = cfg.get_cfg("biotSavartModel", None)
    if m is not None:
        m = str(m).lower().strip()
        if m in ("point_vortex_2d", "2d_point", "eq7", "formula_7", "2d"):
            return True
        if m in ("finite_segment", "finite", "3d", "blob", "blob_center"):
            return False
    return spatial_dim == 2


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
        self.advection_background_velocity = self._build_advection_background_velocity()
        self._advect_fix_segment_center = bool(
            self.ss.cfg.get_cfg("advectFixSegmentCenter", False)
        )
        self._advect_fix_segment_length = bool(
            self.ss.cfg.get_cfg("advectFixSegmentLength", False)
        )
        # 仅用于涡段端点 RK4 平流；对 SPH 的 BS 耦合见 accumulate_bs_*（不含此项）
        self.u_inf = ti.Vector.field(3, dtype=float, shape=())
        self.u_inf.from_numpy(self.advection_background_velocity)
        self._grav_add = ti.Vector.field(3, dtype=float, shape=())
        gv = self._build_segment_gravity_velocity_numpy()
        # Taichi Vector(dtype=float) 为 f32；from_numpy 用 f64 会触发 Assign may lose precision 警告
        self._grav_add.from_numpy(np.asarray(gv, dtype=np.float32))
        # Leapfrog 两个环可选的独立背景速度（默认关闭）
        self._use_leapfrog_ring_bg = ti.field(dtype=ti.i32, shape=())
        self._ring1_seg_type = ti.field(dtype=ti.i32, shape=())
        self._ring2_seg_type = ti.field(dtype=ti.i32, shape=())
        self._ring1_u_inf = ti.Vector.field(3, dtype=float, shape=())
        self._ring2_u_inf = ti.Vector.field(3, dtype=float, shape=())
        self._use_leapfrog_ring_bg[None] = 0
        self._ring1_seg_type[None] = int(self.ss.cfg.get_cfg("leapfrogRing1SegmentTypeId", 101))
        self._ring2_seg_type[None] = int(self.ss.cfg.get_cfg("leapfrogRing2SegmentTypeId", 102))
        self._ring1_u_inf.from_numpy(
            np.array(self.ss.cfg.get_cfg("leapfrogRing1BackgroundVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        self._ring2_u_inf.from_numpy(
            np.array(self.ss.cfg.get_cfg("leapfrogRing2BackgroundVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        # Leapfrog 两个环可选“一次性初始脉冲速度”（仅首步生效）
        self._use_leapfrog_initial_impulse = ti.field(dtype=ti.i32, shape=())
        self._ring1_impulse = ti.Vector.field(3, dtype=float, shape=())
        self._ring2_impulse = ti.Vector.field(3, dtype=float, shape=())
        self._use_leapfrog_initial_impulse[None] = 0
        self._ring1_impulse.from_numpy(
            np.array(self.ss.cfg.get_cfg("leapfrogRing1InitialImpulseVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        self._ring2_impulse.from_numpy(
            np.array(self.ss.cfg.get_cfg("leapfrogRing2InitialImpulseVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        # 脉冲速度衰减参数：dv/dt = -v/tau  => 每步乘 exp(-dt/tau)
        self._impulse_decay_factor = ti.field(dtype=float, shape=())
        self._impulse_decay_stop_eps = ti.field(dtype=float, shape=())
        tau = float(self.ss.cfg.get_cfg("leapfrogImpulseDecayTau", 0.15))
        if tau > 0.0:
            decay = float(np.exp(-self.dt / tau))
        else:
            # tau<=0 退化为“单步脉冲”
            decay = 0.0
        self._impulse_decay_factor[None] = decay
        self._impulse_decay_stop_eps[None] = float(self.ss.cfg.get_cfg("leapfrogImpulseStopEps", 1e-4))

        self.boundary = SegmentBoundaryHandler(self.ss)
        self.has_boundary = self.boundary.enable_boundary_injection
        self._bs_finite = _segment_cfg_biot_savart_is_finite(self.ss.cfg)
        self._bs_2d_point = _segment_cfg_use_2d_point_vortex_bs(
            self.ss.cfg, self.ss.dim
        )
        sch = str(self.ss.cfg.get_cfg("boundaryInjectionSchedule", "each_step") or "each_step").lower().strip()
        if sch in ("init", "initialize", "initialize_only", "once", "static", "static_once"):
            self._boundary_schedule = "initialize_only"
        else:
            self._boundary_schedule = "each_step"
        self._boundary_one_shot_done = False
        self._sim_step_index = 0
        self._step_timing_enabled = bool(self.ss.cfg.get_cfg("debugStepTiming", False))
        self._step_timing_interval = max(1, int(self.ss.cfg.get_cfg("debugStepTimingInterval", 1)))
        self._step_timing_sync = bool(self.ss.cfg.get_cfg("debugStepTimingSync", True))
        self._emitter_interval_override = None
        self._emitter_slot_cursor = 0
        self._emitter_rng = np.random.default_rng(
            int(self.ss.cfg.get_cfg("emitterSeed", 0) or 0)
        )

        _vd = self.ss.dim
        self.v_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.v_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)

        # RK4 需要的中间导数（端点速度）
        self.k1_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.k2_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.k3_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.k4_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)

        self.k1_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.k2_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.k3_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.k4_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)

        self._compact_x_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self._compact_x_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self._compact_gamma = ti.field(dtype=float, shape=self.ss.segment_max_num)
        self._compact_age = ti.field(dtype=float, shape=self.ss.segment_max_num)
        self._compact_seg_type = ti.field(dtype=int, shape=self.ss.segment_max_num)
        self._compact_counter = ti.field(dtype=ti.i32, shape=())

        self.sph_advect_velocity = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.sph_advect_velocity_minus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)
        self.sph_advect_velocity_plus = ti.Vector.field(_vd, dtype=float, shape=self.ss.segment_max_num)

        self._fz_a = ti.field(dtype=ti.i32, shape=())
        self._fz_b = ti.field(dtype=ti.i32, shape=())
        self._fz_c = ti.field(dtype=ti.i32, shape=())
        self._fz_d = ti.field(dtype=ti.i32, shape=())
        _fz_list: List[int] = []
        if bool(self.ss.cfg.get_cfg("advectFreezeBoundarySegments", False)):
            _fz_list.append(int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2)))
        _extras = self.ss.cfg.get_cfg("advectFreezeSegmentTypeIds", None)
        if _extras is not None:
            for _x in _extras:
                _xi = int(_x)
                if _xi not in _fz_list and len(_fz_list) < 4:
                    _fz_list.append(_xi)
        while len(_fz_list) < 4:
            _fz_list.append(-1)
        self._fz_a[None] = _fz_list[0]
        self._fz_b[None] = _fz_list[1]
        self._fz_c[None] = _fz_list[2]
        self._fz_d[None] = _fz_list[3]

    def _build_advection_background_velocity(self) -> np.ndarray:
        """
        涡段自身平流用的背景速度（写入端点速度/RK4）。

        - ``segmentAdvectionBackgroundVelocity``：仅段平流（推荐与来流一致）
        - ``backgroundVelocity``：边界最小二乘等；默认不再用于段平流，避免与 SPH 来流重复
        - 未设 ``segmentAdvectionBackgroundVelocity`` 时，若
          ``backgroundVelocityAffectsSegmentAdvection`` 为 true（默认 false），才回退到
          ``backgroundVelocity``
        """
        adv = self.ss.cfg.get_cfg("segmentAdvectionBackgroundVelocity", None)
        if adv is not None:
            return np.asarray(adv, dtype=np.float32).reshape(-1)[:3]
        affects = self.ss.cfg.get_cfg("backgroundVelocityAffectsSegmentAdvection", None)
        if affects is None:
            affects = True
        if bool(affects):
            return self.background_velocity.astype(np.float32, copy=True)
        return np.zeros(3, dtype=np.float32)

    def _boundary_type_id(self) -> int:
        return int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2))

    def _topology_skip_type_ids(self) -> Set[int]:
        """
        不参与 split / merge / delete 等拓扑操作的 seg_type。
        boundarySkipTopology=true（默认）时自动包含 boundarySegmentTypeId。
        """
        skip: Set[int] = set()
        if bool(self.ss.cfg.get_cfg("boundarySkipTopology", True)):
            if self.has_boundary or bool(
                self.ss.cfg.get_cfg("advectFreezeBoundarySegments", False)
            ):
                skip.add(self._boundary_type_id())
        for key in (
            "splitSkipSegmentTypeIds",
            "mergeSkipSegmentTypeIds",
            "deleteSkipSegmentTypeIds",
        ):
            extra = self.ss.cfg.get_cfg(key, None)
            if extra is not None:
                for x in extra:
                    skip.add(int(x))
        return skip

    @staticmethod
    def _skip_ids_for_kernel(skip: Set[int], slots: int = 4) -> Tuple[int, ...]:
        ids = sorted(skip)[:slots]
        while len(ids) < slots:
            ids.append(-1)
        return tuple(ids)

    @ti.func
    def _advect_frozen(self, st: ti.i32) -> ti.i32:
        """若为 1，则该 seg_type 不参与 RK4/Euler 端点推进（仍参与 BS 诱导速度）。"""
        # Taichi：@ti.func 内不能在非 static 的 if 里 return，只能末尾单一 return
        out = 0
        if self._fz_a[None] != -1 and st == self._fz_a[None]:
            out = 1
        if self._fz_b[None] != -1 and st == self._fz_b[None]:
            out = 1
        if self._fz_c[None] != -1 and st == self._fz_c[None]:
            out = 1
        if self._fz_d[None] != -1 and st == self._fz_d[None]:
            out = 1
        return out

    @ti.func
    def _segment_background(self, i: ti.i32):
        if ti.static(self.ss.dim == 2):
            u = ti.Vector([self.u_inf[None][0], self.u_inf[None][1]])
            if self._use_leapfrog_ring_bg[None] == 1:
                st = self.ss.seg_type[i]
                if st == self._ring1_seg_type[None]:
                    u = u + ti.Vector([
                        self._ring1_u_inf[None][0], self._ring1_u_inf[None][1]
                    ])
                elif st == self._ring2_seg_type[None]:
                    u = u + ti.Vector([
                        self._ring2_u_inf[None][0], self._ring2_u_inf[None][1]
                    ])
            if self._use_leapfrog_initial_impulse[None] == 1:
                st = self.ss.seg_type[i]
                if st == self._ring1_seg_type[None]:
                    u = u + ti.Vector([
                        self._ring1_impulse[None][0], self._ring1_impulse[None][1]
                    ])
                elif st == self._ring2_seg_type[None]:
                    u = u + ti.Vector([
                        self._ring2_impulse[None][0], self._ring2_impulse[None][1]
                    ])
            return u + ti.Vector([self._grav_add[None][0], self._grav_add[None][1]])
        u = self.u_inf[None]
        if self._use_leapfrog_ring_bg[None] == 1:
            st = self.ss.seg_type[i]
            if st == self._ring1_seg_type[None]:
                u = u + self._ring1_u_inf[None]
            elif st == self._ring2_seg_type[None]:
                u = u + self._ring2_u_inf[None]
        if self._use_leapfrog_initial_impulse[None] == 1:
            st = self.ss.seg_type[i]
            if st == self._ring1_seg_type[None]:
                u = u + self._ring1_impulse[None]
            elif st == self._ring2_seg_type[None]:
                u = u + self._ring2_impulse[None]
        return u + self._grav_add[None]

    def _build_segment_gravity_velocity_numpy(self) -> np.ndarray:
        """
        将重力映射为端点对流中的常速度偏置（m/s），与 Biot–Savart 诱导速度叠加。
        优先 SegmentConfiguration.segmentGravitationVelocity；
        否则用 Configuration.gravitation * segmentGravitationScale。
        """
        raw = self.ss.cfg.get_cfg("segmentGravitationVelocity", None)
        if raw is not None:
            return np.array(raw, dtype=np.float32).reshape(3)
        scale = float(self.ss.cfg.get_cfg("segmentGravitationScale", 0.0))
        if scale == 0.0:
            return np.zeros(3, dtype=np.float32)
        conf = self.ss.cfg.get_configuration_dict() if hasattr(self.ss.cfg, "get_configuration_dict") else {}
        g_acc = np.array(conf.get("gravitation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        return (g_acc * np.float32(scale)).astype(np.float32)

    @ti.kernel
    def _decay_impulse_kernel(self):
        if self._use_leapfrog_initial_impulse[None] == 1:
            f = self._impulse_decay_factor[None]
            self._ring1_impulse[None] = f * self._ring1_impulse[None]
            self._ring2_impulse[None] = f * self._ring2_impulse[None]
            e = self._impulse_decay_stop_eps[None]
            if self._ring1_impulse[None].norm() < e and self._ring2_impulse[None].norm() < e:
                self._use_leapfrog_initial_impulse[None] = 0

    def initialize(self):
        self.ss.clear()
        self._boundary_one_shot_done = False
        self.seed_initial_segments()
        self.ss.update_segment_geometry()
        if self.has_boundary and self._boundary_schedule == "initialize_only":
            defer_sph = (
                self.boundary._sample_source_is_sph()
                and self.boundary._ps is not None
                and not self.boundary.sph_boundary_sampling_ready()
            )
            if not defer_sph:
                self._run_boundary_injection_pipeline(strip_committed_first=False)
                self.boundary.mark_one_shot_complete_if_applicable(self)
        if self._advect_fix_segment_length:
            self.ss.update_segment_geometry()
            self._snapshot_segment_ref_geometry()

    @ti.kernel
    def _snapshot_segment_ref_geometry(self):
        """记录当前活跃段的参考段心与长度（用于固定几何投影）。"""
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] != 1:
                continue
            xm = self.ss.x_minus[i]
            xp = self.ss.x_plus[i]
            d = xp - xm
            l = d.norm() + 1e-8
            self.ss.center_ref[i] = 0.5 * (xm + xp)
            self.ss.tangent_ref[i] = d / l
            self.ss.length_ref[i] = l

    @ti.kernel
    def _project_endpoints_fixed_ref_geometry(self):
        """将端点投影到参考段心 + 参考长度，仅保留切向旋转。"""
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] != 1:
                continue
            if self._advect_frozen(self.ss.seg_type[i]) == 1:
                continue
            C = self.ss.center_ref[i]
            L = self.ss.length_ref[i]
            d = self.ss.tangent_ref[i]
            dn = d.norm()
            if dn < 1e-8:
                continue
            t = d / dn
            half = 0.5 * L
            self.ss.x_minus[i] = C - half * t
            self.ss.x_plus[i] = C + half * t

    @ti.kernel
    def _snapshot_frozen_segment_geometry(self):
        """记录冻结段（如边界虚拟段）的端点几何，供 merge 后复位。"""
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] != 1:
                continue
            if self._advect_frozen(self.ss.seg_type[i]) != 1:
                continue
            xm = self.ss.x_minus[i]
            xp = self.ss.x_plus[i]
            d = xp - xm
            l = d.norm() + 1e-8
            self.ss.center_ref[i] = 0.5 * (xm + xp)
            self.ss.tangent_ref[i] = d / l
            self.ss.length_ref[i] = l

    @ti.kernel
    def _restore_frozen_segment_geometry(self):
        """将冻结段端点恢复为提交时的参考几何（位置不变）。"""
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] != 1:
                continue
            if self._advect_frozen(self.ss.seg_type[i]) != 1:
                continue
            C = self.ss.center_ref[i]
            L = self.ss.length_ref[i]
            d = self.ss.tangent_ref[i]
            dn = d.norm()
            if dn < 1e-8:
                if ti.static(self.ss.dim == 2):
                    d = ti.Vector([1.0, 0.0])
                else:
                    d = ti.Vector([1.0, 0.0, 0.0])
                dn = d.norm()
            t = d / dn
            half = 0.5 * L
            self.ss.x_minus[i] = C - half * t
            self.ss.x_plus[i] = C + half * t

    def snapshot_frozen_segment_geometry(self):
        if not bool(self.ss.cfg.get_cfg("boundarySnapshotFrozenGeometry", True)):
            return
        self.ss.update_segment_geometry()
        self._snapshot_frozen_segment_geometry()

    def restore_frozen_segment_geometry(self):
        if not bool(self.ss.cfg.get_cfg("boundarySnapshotFrozenGeometry", True)):
            return
        self._restore_frozen_segment_geometry()
        self.ss.update_segment_geometry()

    def _run_boundary_injection_pipeline(self, strip_committed_first: bool):
        """
        边界虚拟段：可选先剥旧类型，再更新位姿、生成候选、K、RHS、求解、提交。
        """
        if not self.has_boundary:
            return
        if (
            self.boundary._sample_source_is_sph()
            and self.boundary._ps is not None
            and not self.boundary.sph_boundary_sampling_ready()
        ):
            return
        if strip_committed_first:
            self._strip_segments_of_type(int(self.ss.cfg.get_cfg("boundarySegmentTypeId", 2)))
        self.boundary.update_boundary_pose()
        if self.boundary._uses_random_boundary_virtual():
            self.boundary.generate_boundary_segments_random()
            self.boundary.commit_boundary_segments()
            self.boundary.release_vorticity_to_internal_segments()
        else:
            self.boundary.generate_boundary_segments()
            if (not bool(self.ss.cfg.get_cfg("boundaryProjectionCache", True))) or self.boundary._K is None:
                self.boundary.compute_k_matrix()
            if not self.boundary._compute_rhs_and_solve_gpu():
                self.boundary.compute_rhs()
                self.boundary.solve_linear_system()
            self.boundary.commit_boundary_segments()
            self.boundary.release_vorticity_to_internal_segments()
        self.ss.update_segment_geometry()
        if self.boundary._last_boundary_commit_count > 0:
            self.snapshot_frozen_segment_geometry()

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
        # 默认关闭按环独立背景速度，仅 leapfrog_rings 初始化会开启
        self._use_leapfrog_ring_bg[None] = 0
        self._use_leapfrog_initial_impulse[None] = 0

        if init_type == "none":
            return

        if init_type == "triple_parallel_filaments_x":
            self._seed_triple_parallel_filaments_x()
            return

        if init_type == "v_bundle_pair":
            self._seed_v_bundle_pair()
            return

        if init_type == "leapfrog_rings":
            self._seed_leapfrog_rings()
            return

        if init_type in ("random_uniform", "random_cloud"):
            self._seed_random_uniform_segments()
            return

        if init_type in ("parallel_x_layers", "parallel_x_filaments_grid", "x_parallel_layers"):
            self._seed_parallel_x_layers_filaments()
            return

        if init_type != "ring":
            raise NotImplementedError(
                f"initType={init_type} 尚未实现（当前支持 ring / triple_parallel_filaments_x / "
                f"v_bundle_pair / leapfrog_rings / random_uniform / random_cloud / "
                f"parallel_x_layers / none）"
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

        # 可选：对环中心与环上点加随机扰动（打破对称，便于观察涡环演化）
        # SegmentConfiguration:
        # - initPerturbSeed: int | null，固定种子可复现；不设则每次运行不同
        # - initCenterJitter: 标量或 [ax,ay,az]，在 [-j,j] 上均匀扰动中心各分量
        # - initRingRadialJitter: 每段顶点在环平面内径向扰动（米），半径 = initRingRadius + 噪声
        # - initRingPhaseJitter: 每段顶点角度扰动（弧度），在 [-p,p] 均匀
        seed = self.ss.cfg.get_cfg("initPerturbSeed", None)
        rng = np.random.default_rng(seed)
        jc = self.ss.cfg.get_cfg("initCenterJitter", 0.0)
        if jc is not None:
            if isinstance(jc, (list, tuple)) and len(jc) >= 3:
                w = np.array(jc[:3], dtype=np.float64)
                c = (c.astype(np.float64) + rng.uniform(-w, w)).astype(np.float32)
            else:
                jf = float(jc)
                if jf > 0.0:
                    c = (c.astype(np.float64) + rng.uniform(-jf, jf, size=3)).astype(np.float32)

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
        angles = np.linspace(0.0, 2.0 * np.pi, n_seg + 1, dtype=np.float64)[:-1]
        jr = float(self.ss.cfg.get_cfg("initRingRadialJitter", 0.0))
        jp = float(self.ss.cfg.get_cfg("initRingPhaseJitter", 0.0))
        if jp != 0.0:
            angles = angles + rng.uniform(-jp, jp, size=n_seg)
        radii = np.full((n_seg,), float(ring_radius), dtype=np.float64)
        if jr != 0.0:
            radii = radii + rng.uniform(-jr, jr, size=n_seg)
        radii = np.maximum(radii, float(self.ss.cfg.get_cfg("initRingRadiusMin", 1e-4)))
        pts = (
            c[None, :].astype(np.float64)
            + radii[:, None] * (np.cos(angles)[:, None] * e1[None, :].astype(np.float64)
                                + np.sin(angles)[:, None] * e2[None, :].astype(np.float64))
        ).astype(np.float32)

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

    def _seed_random_uniform_segments(self):
        """
        在域内随机放置短线段：随机中点、随机朝向、长度 ∈ [Lmin, Lmax]，γ 默认全 0。
        用于 SPH–涡段耦合：强度完全由后续沉积与演化给出。

        SegmentConfiguration:
        - randomSegmentNum（默认 4096）
        - randomSegmentLengthMin / randomSegmentLengthMax（默认 0.015 / 0.04）
        - randomPositionMargin（默认 0.05）：相对 domain 的内缩，避免端点出域
        - randomGammaInitial（默认 0.0）
        - randomSegmentSeed：种子；若缺省则用 initPerturbSeed，再无则 0
        - initSegmentTypeId
        """
        n_seg = int(self.ss.cfg.get_cfg("randomSegmentNum", 4096))
        n_seg = max(1, n_seg)
        L0 = float(self.ss.cfg.get_cfg("randomSegmentLengthMin", 0.015))
        L1 = float(self.ss.cfg.get_cfg("randomSegmentLengthMax", 0.04))
        if L1 < L0:
            L0, L1 = L1, L0
        margin = float(self.ss.cfg.get_cfg("randomPositionMargin", 0.05))
        g0 = float(self.ss.cfg.get_cfg("randomGammaInitial", 0.0))
        seg_type = int(self.ss.cfg.get_cfg("initSegmentTypeId", 0))

        seed = self.ss.cfg.get_cfg("randomSegmentSeed", None)
        if seed is None:
            seed = self.ss.cfg.get_cfg("initPerturbSeed", None)
        if seed is None:
            seed = 0
        rng = np.random.default_rng(int(seed))

        lo = self.ss.domain_start.astype(np.float64) + margin + 0.5 * L1
        hi = self.ss.domain_end.astype(np.float64) - margin - 0.5 * L1
        if np.any(lo >= hi):
            raise ValueError(
                "random_uniform: domain too small for randomPositionMargin and max segment length; "
                "reduce margin or randomSegmentLengthMax."
            )

        x_minus = np.zeros((n_seg, 3), dtype=np.float32)
        x_plus = np.zeros((n_seg, 3), dtype=np.float32)
        gamma = np.full((n_seg,), g0, dtype=np.float32)

        for k in range(n_seg):
            center = rng.uniform(lo, hi).astype(np.float32)
            if self.ss.dim == 2:
                q = rng.standard_normal(2).astype(np.float64)
                nq = float(np.linalg.norm(q))
                if nq < 1e-8:
                    t2 = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    t2 = (q / nq).astype(np.float32)
                L = float(rng.uniform(L0, L1))
                half2 = 0.5 * L * t2
                x_minus[k] = np.array([center[0] - half2[0], center[1] - half2[1], 0.0],
                                      dtype=np.float32)
                x_plus[k] = np.array([center[0] + half2[0], center[1] + half2[1], 0.0],
                                     dtype=np.float32)
                continue
            q = rng.standard_normal(3).astype(np.float64)
            nq = float(np.linalg.norm(q))
            if nq < 1e-8:
                t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                t = (q / nq).astype(np.float32)
            L = float(rng.uniform(L0, L1))
            half = 0.5 * L * t
            x_minus[k] = center - half
            x_plus[k] = center + half

        offset = int(self.ss.segment_num[None])
        if offset >= int(self.ss.segment_max_num):
            return

        n_new = min(n_seg, int(self.ss.segment_max_num) - offset)
        if n_new <= 0:
            return

        self._seed_segments_kernel(
            offset, n_new, x_minus[:n_new], x_plus[:n_new], gamma[:n_new], seg_type
        )
        self.ss.segment_num[None] = offset + n_new

    def _parallel_x_layout_bounds(self, margin: float):
        """
        parallel_x_layers 铺段用的轴对齐盒子 [lo, hi]（已加 margin 内缩）。

        默认用 SegmentConfiguration.domainStart/End；
        若设置 parallelXDomainStart / parallelXDomainEnd，则仅在该子域内生成
        （再与全局 domain 求交，避免越界）。
        """
        d = self.ss.dim
        dom_lo = self.ss.domain_start.astype(np.float64)
        dom_hi = self.ss.domain_end.astype(np.float64)

        box_lo = self.ss.cfg.get_cfg("parallelXDomainStart", None)
        box_hi = self.ss.cfg.get_cfg("parallelXDomainEnd", None)
        if box_lo is not None and box_hi is not None:
            lo = np.asarray(box_lo, dtype=np.float64).reshape(-1)[:d]
            hi = np.asarray(box_hi, dtype=np.float64).reshape(-1)[:d]
            lo = np.maximum(lo, dom_lo[:d])
            hi = np.minimum(hi, dom_hi[:d])
        else:
            lo = dom_lo[:d].copy()
            hi = dom_hi[:d].copy()

        lo = lo + margin
        hi = hi - margin
        return lo, hi

    def _seed_parallel_x_layers_filaments(self):
        """
        在 SegmentConfiguration.domainStart/End 盒子内，沿 +X 方向铺许多短涡段，
        组成 n 条平行线；共 L 层（层与层沿可选轴分开）。

        SegmentConfiguration:
        - parallelXLayers（默认 3）：层数 L
        - parallelXFilamentsPerLayer（默认 8）：每层平行线数 n（沿层法向的垂直方向均匀排布）
        - parallelXSegmentsPerLine（默认 24）：每条线沿 x 切成多少段
        - parallelXSegmentGap（默认 0.0）：相邻两段在 x 上的间隙（上一段 x_plus 到下一段 x_minus 的距离）；
          段长 L = (可用 x 跨度 − (M−1)×gap) / M；gap=0 时与原先首尾相接一致
        - parallelXLayerAxis（默认 "z"）：层间分离轴，"z" 表示不同 z 平面为不同层，平行线在 y 向排开；
          "y" 则不同 y 为层，线在 z 向排开
        - parallelXMargin（默认 0.06）：相对铺段盒子三轴内缩，避免段端点贴边；
          使用 parallelXDomainStart/End 时若需精确坐标，可设为 0
        - parallelXDomainStart / parallelXDomainEnd（可选）：仅在此子域内铺平行涡段
          （与 SegmentConfiguration.domain 求交）；未设则用全局 domain
        - parallelXFilamentGamma（默认 initGamma 或 0.0）
        - parallelXGammaAlternateFilament（默认 true）：相邻平行线 γ 取反（可选打破对称）
        - parallelXGammaAlternateLayer（默认 false）：相邻层 γ 再取反
        - initSegmentTypeId
        """
        L = int(self.ss.cfg.get_cfg("parallelXLayers", 3))
        n = int(self.ss.cfg.get_cfg("parallelXFilamentsPerLayer", 8))
        M = int(self.ss.cfg.get_cfg("parallelXSegmentsPerLine", 24))
        L = max(1, L)
        n = max(1, n)
        M = max(1, M)

        margin = float(self.ss.cfg.get_cfg("parallelXMargin", 0.06))
        g0 = float(self.ss.cfg.get_cfg("parallelXFilamentGamma", self.ss.cfg.get_cfg("initGamma", 0.0)))
        seg_type = int(self.ss.cfg.get_cfg("initSegmentTypeId", 0))
        layer_axis = str(self.ss.cfg.get_cfg("parallelXLayerAxis", "z") or "z").lower().strip()
        alt_f = bool(self.ss.cfg.get_cfg("parallelXGammaAlternateFilament", True))
        alt_L = bool(self.ss.cfg.get_cfg("parallelXGammaAlternateLayer", False))

        lo, hi = self._parallel_x_layout_bounds(margin)
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            raise ValueError(
                "parallel_x_layers: domain too small after parallelXMargin; reduce margin or enlarge domain."
            )
        if self.ss.dim == 3 and lo[2] >= hi[2]:
            raise ValueError(
                "parallel_x_layers: domain too small after parallelXMargin (z); reduce margin or enlarge domain."
            )

        span_x = float(hi[0] - lo[0])
        gap_x = float(self.ss.cfg.get_cfg("parallelXSegmentGap", 0.0))
        gap_x = max(0.0, gap_x)
        if M > 1:
            usable_x = span_x - float(M - 1) * gap_x
        else:
            usable_x = span_x
        if usable_x <= 0.0:
            raise ValueError(
                "parallel_x_layers: parallelXSegmentGap too large for parallelXSegmentsPerLine and x-span; "
                "reduce gap, reduce M, or widen domain (after margin)."
            )
        seg_len_x = usable_x / float(M)

        total = L * n * M
        x_minus = np.zeros((total, 3), dtype=np.float32)
        x_plus = np.zeros((total, 3), dtype=np.float32)
        gamma = np.zeros((total,), dtype=np.float32)

        idx = 0
        if self.ss.dim == 2:
            # 2D：L 层沿 y，每层 n 条平行线（不同 y），段沿 +x，z = 0
            ys_layer = np.linspace(float(lo[1]), float(hi[1]), L, dtype=np.float64)
            ys_line = np.linspace(float(lo[1]), float(hi[1]), n, dtype=np.float64)
            if L == 1:
                ys_layer = ys_line
            for li in range(L):
                y_layer = float(ys_layer[li]) if L > 1 else float(lo[1])
                for fi in range(n):
                    yc = float(ys_line[fi]) if L == 1 else y_layer
                    sgn = 1.0
                    if alt_f and (fi % 2) == 1:
                        sgn *= -1.0
                    if alt_L and (li % 2) == 1:
                        sgn *= -1.0
                    g_line = float(sgn * g0)
                    for k in range(M):
                        x0 = float(lo[0] + float(k) * (seg_len_x + gap_x))
                        x1 = float(x0 + seg_len_x)
                        x_minus[idx] = np.array([x0, yc, 0.0], dtype=np.float32)
                        x_plus[idx] = np.array([x1, yc, 0.0], dtype=np.float32)
                        gamma[idx] = g_line
                        idx += 1
        elif layer_axis in ("y", "layer_y"):
            # L 层：不同 y；每层 n 条线：不同 z
            ys = np.linspace(float(lo[1]), float(hi[1]), L, dtype=np.float64)
            zs = np.linspace(float(lo[2]), float(hi[2]), n, dtype=np.float64)
            for li in range(L):
                yc = float(ys[li])
                for fi in range(n):
                    zc = float(zs[fi])
                    sgn = 1.0
                    if alt_f and (fi % 2) == 1:
                        sgn *= -1.0
                    if alt_L and (li % 2) == 1:
                        sgn *= -1.0
                    g_line = float(sgn * g0)
                    for k in range(M):
                        x0 = float(lo[0] + float(k) * (seg_len_x + gap_x))
                        x1 = float(x0 + seg_len_x)
                        x_minus[idx] = np.array([x0, yc, zc], dtype=np.float32)
                        x_plus[idx] = np.array([x1, yc, zc], dtype=np.float32)
                        gamma[idx] = g_line
                        idx += 1
        else:
            # 默认 "z"：L 层不同 z；每层 n 条线不同 y
            zs = np.linspace(float(lo[2]), float(hi[2]), L, dtype=np.float64)
            ys = np.linspace(float(lo[1]), float(hi[1]), n, dtype=np.float64)
            for li in range(L):
                zc = float(zs[li])
                for fi in range(n):
                    yc = float(ys[fi])
                    sgn = 1.0
                    if alt_f and (fi % 2) == 1:
                        sgn *= -1.0
                    if alt_L and (li % 2) == 1:
                        sgn *= -1.0
                    g_line = float(sgn * g0)
                    for k in range(M):
                        x0 = float(lo[0] + float(k) * (seg_len_x + gap_x))
                        x1 = float(x0 + seg_len_x)
                        x_minus[idx] = np.array([x0, yc, zc], dtype=np.float32)
                        x_plus[idx] = np.array([x1, yc, zc], dtype=np.float32)
                        gamma[idx] = g_line
                        idx += 1

        assert idx == total

        offset = int(self.ss.segment_num[None])
        if offset >= int(self.ss.segment_max_num):
            return

        n_new = min(total, int(self.ss.segment_max_num) - offset)
        if n_new <= 0:
            return

        self._seed_segments_kernel(
            offset, n_new, x_minus[:n_new], x_plus[:n_new], gamma[:n_new], seg_type
        )
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

    def _seed_leapfrog_rings(self):
        """
        蛙跳涡（Leapfrogging vortex rings）初始化：
        - 一个小半径/强度更大（更快）的涡环从后方追赶
        - 一个大半径/强度更小（更慢）的涡环在前方
        两个涡环都具有“厚度”：用截面圆上多股涡丝（多条环状 filament）拼成涡管。

        SegmentConfiguration 参数（默认值给一个可跑的起点）：
        - leapfrogAxis: [ax,ay,az] 轴向（默认 [1,0,0]，沿 x 方向蛙跳）
        - leapfrogCenterYZ: [y,z] 两环的 y/z 中心（默认取 domain 中心）
        - leapfrogRing1CenterX / leapfrogRing2CenterX: 两环的 x 位置（默认 0.8 / 1.3）
        - leapfrogRing1Radius / leapfrogRing2Radius: 两环主半径（默认 0.16 / 0.24）
        - leapfrogRing1Gamma / leapfrogRing2Gamma: 两环总强度参数（默认 0.9 / 0.5）
        - leapfrogTubeRadius: 厚度（截面半径，默认 0.04）
        - leapfrogSegmentsPerRing: 环向离散段数（默认 256）
        - leapfrogFilamentsPerSection: 截面上 filament 条数（默认 8）
        - leapfrogGammaSplit: 是否将 ringGamma 均分到 filament（默认 true）
        - leapfrogRing1SegmentTypeId / leapfrogRing2SegmentTypeId: 两环段类型 id（用于独立背景速度）
        - leapfrogRing1BackgroundVelocity / leapfrogRing2BackgroundVelocity: 两环背景速度增量
        """
        cfg = self.ss.cfg
        seg_type1 = int(cfg.get_cfg("leapfrogRing1SegmentTypeId", self._ring1_seg_type[None]))
        seg_type2 = int(cfg.get_cfg("leapfrogRing2SegmentTypeId", self._ring2_seg_type[None]))
        # 打开“按环叠加独立背景速度”开关；速度值可为 0（不生效）
        self._use_leapfrog_ring_bg[None] = 1
        self._ring1_seg_type[None] = seg_type1
        self._ring2_seg_type[None] = seg_type2
        # 按需打开“一次性初始脉冲速度”
        imp1 = np.array(cfg.get_cfg("leapfrogRing1InitialImpulseVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        imp2 = np.array(cfg.get_cfg("leapfrogRing2InitialImpulseVelocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._ring1_impulse.from_numpy(imp1)
        self._ring2_impulse.from_numpy(imp2)
        if float(np.linalg.norm(imp1)) > 0.0 or float(np.linalg.norm(imp2)) > 0.0:
            self._use_leapfrog_initial_impulse[None] = 1
        else:
            self._use_leapfrog_initial_impulse[None] = 0

        axis_cfg = cfg.get_cfg("leapfrogAxis", [1.0, 0.0, 0.0])
        axis = np.array(axis_cfg, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        yz_center = cfg.get_cfg("leapfrogCenterYZ", None)
        if yz_center is None:
            c_dom = 0.5 * (self.ss.domain_start + self.ss.domain_end)
            cy, cz = float(c_dom[1]), float(c_dom[2])
        else:
            cy, cz = float(yz_center[0]), float(yz_center[1])

        x1 = float(cfg.get_cfg("leapfrogRing1CenterX", 0.8))
        x2 = float(cfg.get_cfg("leapfrogRing2CenterX", 1.3))
        R1 = float(cfg.get_cfg("leapfrogRing1Radius", 0.16))
        R2 = float(cfg.get_cfg("leapfrogRing2Radius", 0.24))
        G1 = float(cfg.get_cfg("leapfrogRing1Gamma", 0.9))
        G2 = float(cfg.get_cfg("leapfrogRing2Gamma", 0.5))
        a = float(cfg.get_cfg("leapfrogTubeRadius", 0.04))

        n_theta = int(cfg.get_cfg("leapfrogSegmentsPerRing", 256))
        n_theta = max(16, n_theta)
        n_phi = int(cfg.get_cfg("leapfrogFilamentsPerSection", 8))
        n_phi = max(1, n_phi)
        gamma_split = bool(cfg.get_cfg("leapfrogGammaSplit", True))

        # 构造环平面基 (e1, e2)：与 ring axis 垂直
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e1 = np.cross(axis, ref)
        e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(axis, e1)
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)

        def build_thick_ring(center: np.ndarray, R: float, ring_gamma: float):
            """
            返回 x_minus/x_plus/gamma 三个数组，表示一条“厚涡环”由 n_phi 条 filament 组成；
            每条 filament 由 n_theta 个小段闭合成环。
            """
            angles = np.linspace(0.0, 2.0 * np.pi, n_theta + 1, endpoint=True, dtype=np.float64)[:-1]
            cos_t = np.cos(angles)
            sin_t = np.sin(angles)

            # 环中心线上每个角度点的中心位置（不带厚度）
            cline = center[None, :] + R * (cos_t[:, None] * e1[None, :] + sin_t[:, None] * e2[None, :])

            # 环向切向与径向（用于构造截面平面）
            t_hat = (-sin_t[:, None] * e1[None, :] + cos_t[:, None] * e2[None, :])
            t_hat = t_hat / (np.linalg.norm(t_hat, axis=1, keepdims=True) + 1e-12)
            n_hat = (cos_t[:, None] * e1[None, :] + sin_t[:, None] * e2[None, :])
            n_hat = n_hat / (np.linalg.norm(n_hat, axis=1, keepdims=True) + 1e-12)
            b_hat = np.cross(t_hat, n_hat)
            b_hat = b_hat / (np.linalg.norm(b_hat, axis=1, keepdims=True) + 1e-12)

            phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float64)
            cos_p = np.cos(phis)
            sin_p = np.sin(phis)

            pts = np.zeros((n_phi, n_theta, 3), dtype=np.float64)
            for p in range(n_phi):
                offset = a * (cos_p[p] * n_hat + sin_p[p] * b_hat)  # (n_theta,3)
                pts[p] = cline + offset

            # 连接每条 filament 的相邻点形成小段
            xm = pts.reshape(-1, 3).astype(np.float32)  # (n_phi*n_theta,3)
            # x_plus：沿 theta 方向 roll（每条 filament 独立闭合）
            pts_roll = np.roll(pts, shift=-1, axis=1)
            xp = pts_roll.reshape(-1, 3).astype(np.float32)

            if gamma_split and n_phi > 0:
                g_each = float(ring_gamma) / float(n_phi)
            else:
                g_each = float(ring_gamma)
            g = (g_each * np.ones((n_phi * n_theta,), dtype=np.float32))
            return xm, xp, g

        c1 = np.array([x1, cy, cz], dtype=np.float64)
        c2 = np.array([x2, cy, cz], dtype=np.float64)

        xm1, xp1, g1 = build_thick_ring(c1, R1, G1)
        xm2, xp2, g2 = build_thick_ring(c2, R2, G2)

        offset = int(self.ss.segment_num[None])
        capacity = int(self.ss.segment_max_num)
        if offset >= capacity:
            return

        n1 = int(min(xm1.shape[0], capacity - offset))
        if n1 > 0:
            self._seed_segments_kernel(offset, n1, xm1[:n1], xp1[:n1], g1[:n1], seg_type1)
            offset += n1
        if offset >= capacity:
            return
        n2 = int(min(xm2.shape[0], capacity - offset))
        if n2 <= 0:
            self.ss.segment_num[None] = offset
            return

        self._seed_segments_kernel(offset, n2, xm2[:n2], xp2[:n2], g2[:n2], seg_type2)
        self.ss.segment_num[None] = offset + n2

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
            self.ss.set_segment_ends_from_ndarray_row(i, k, x_minus, x_plus)
            self.ss.gamma[i] = gamma[k]
            self.ss.active[i] = 1
            self.ss.age[i] = 0.0
            self.ss.seg_type[i] = seg_type

    @ti.func
    def _bs_velocity_2d_point_vortex(
        self, x_query: ti.template(), x_vortex: ti.template(), Gamma: float, R2: float
    ):
        """
        TOG2021 式 (7)：u^BS = Γ/(2π) e_z × (x−x_j) / (|x−x_j|²+R²)，xy 平面内速度。
        """
        inv2pi = 1.0 / (2.0 * ti.math.pi)
        rx = x_query[0] - x_vortex[0]
        ry = x_query[1] - x_vortex[1]
        r2 = rx * rx + ry * ry + 1e-12
        ux = inv2pi * Gamma * (-ry) / (r2 + R2)
        uy = inv2pi * Gamma * (rx) / (r2 + R2)
        return ti.Vector([ux, uy])

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
        if self._bs_2d_point:
            self._compute_endpoint_velocity_bs_2d_point(float(self.reg_radius))
        elif self._bs_finite:
            self._compute_endpoint_velocity_bs_finite(float(self.reg_radius))
            if bool(self.ss.cfg.get_cfg("debugPrintEndpointVelocity", False)):
                n = int(self.ss.segment_num[None])
                v_m = self.v_minus.to_numpy()[:n]
                v_p = self.v_plus.to_numpy()[:n]
                print("v_minus[0:5] =", v_m[:5])
                print("v_plus[0:5]  =", v_p[:5])
                print("speed_minus mean/max =", (v_m**2).sum(axis=1).mean()**0.5, ((v_m**2).sum(axis=1).max())**0.5)
                print("speed_plus  mean/max =", (v_p**2).sum(axis=1).mean()**0.5, ((v_p**2).sum(axis=1).max())**0.5)
        else:
            self._compute_endpoint_velocity_bs_blob(float(self.reg_radius))

    @ti.kernel
    def _compute_endpoint_velocity_bs_blob(self, reg_radius: float):
        """段中点涡 blob（旧版）：omega = gamma * L * t。"""
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = reg_radius * reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i]
            xpi = self.ss.x_plus[i]
            ui_m = self._segment_background(i) + self.sph_advect_velocity_minus[i]
            ui_p = self._segment_background(i) + self.sph_advect_velocity_plus[i]

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

    @ti.kernel
    def _compute_endpoint_velocity_bs_2d_point(self, reg_radius: float):
        """
        TOG2021 式 (7)：二维点涡 u^BS = Γ/(2π) e_z×(x−x_j)/(|x−x_j|²+R²)。
        每段以中点 x_j 与 Γ_j = γ_j L_j 参与求和（式 (8) 的离散项）。
        """
        R2 = reg_radius * reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i]
            xpi = self.ss.x_plus[i]
            ui_m = self._segment_background(i) + self.sph_advect_velocity_minus[i]
            ui_p = self._segment_background(i) + self.sph_advect_velocity_plus[i]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                am = self.ss.x_minus[j]
                ap = self.ss.x_plus[j]
                cj = 0.5 * (am + ap)
                Lj = (ap - am).norm() + 1e-8
                Gamma = self.ss.gamma[j] * Lj

                bs_m = self._bs_velocity_2d_point_vortex(xmi, cj, Gamma, R2)
                bs_p = self._bs_velocity_2d_point_vortex(xpi, cj, Gamma, R2)
                ui_m[0] += bs_m[0]
                ui_m[1] += bs_m[1]
                ui_p[0] += bs_p[0]
                ui_p[1] += bs_p[1]

            self.v_minus[i] = ui_m
            self.v_plus[i] = ui_p

    @ti.kernel
    def _compute_endpoint_velocity_bs_finite(self, reg_radius: float):
        """
        论文 TOG2021 式 (6)：三维涡段云上的 Biot–Savart（Weißmann & Pinkall 形式）。

        u_j^BS(x) = Γ_j/(4π) * [ ( (x_j^+−x)/(|x_j^+−x|+R) − (x_j^−−x)/(|x_j^−−x|+R) ) · (x_j^+−x_j^−) ]
                              * [ (x_j^−−x)×(x_j^+−x) / ( |(x_j^−−x)×(x_j^+−x)|^2 + R^2 ) ]

        三维流用式 (6)；二维请用 biotSavartModel: point_vortex_2d（式 (7)）。

        数据约定：段上涡向量取 gamma*(x^+−x^−)，则 Γ_j = gamma_j * L_j。
        """
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = reg_radius * reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i]
            xpi = self.ss.x_plus[i]
            ui_m = self._segment_background(i) + self.sph_advect_velocity_minus[i]
            ui_p = self._segment_background(i) + self.sph_advect_velocity_plus[i]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                am = self.ss.x_minus[j]
                ap = self.ss.x_plus[j]
                d = ap - am
                Lj = d.norm() + 1e-8
                gamma_j = self.ss.gamma[j]
                Gamma = gamma_j * Lj

                # --- 查询 x = xmi ---
                apx = ap - xmi
                ambx = am - xmi
                n_ap = apx.norm() + reg_radius
                n_am = ambx.norm() + reg_radius
                u1 = apx / (n_ap + 1e-12)
                u2 = ambx / (n_am + 1e-12)
                sc = (u1 - u2).dot(d)
                cr = ambx.cross(apx)
                cross_sq = cr.dot(cr) + R2
                ui_m += inv4pi * Gamma * sc / (cross_sq + 1e-20) * cr

                # --- 查询 x = xpi ---
                apxp = ap - xpi
                ambxp = am - xpi
                n_app = apxp.norm() + reg_radius
                n_amp = ambxp.norm() + reg_radius
                u1p = apxp / (n_app + 1e-12)
                u2p = ambxp / (n_amp + 1e-12)
                scp = (u1p - u2p).dot(d)
                crp = ambxp.cross(apxp)
                cross_sqp = crp.dot(crp) + R2
                ui_p += inv4pi * Gamma * scp / (cross_sqp + 1e-20) * crp

            self.v_minus[i] = ui_m
            self.v_plus[i] = ui_p

    def accumulate_bs_velocity_at_fluid_particles(self, ps, out_u, reg_radius: float):
        """
        For each fluid particle at ``ps.x[p]``, sum **pure Biot–Savart** from active segments.

        Does **not** add ``backgroundVelocity`` / ``segmentAdvectionBackgroundVelocity``
        (SPH 流体已有入口 ``velocity``，避免来流重复叠加).

        Non-fluid indices are zeroed.
        """
        if self._bs_2d_point:
            self._accumulate_bs_at_particles_2d_point(
                ps.particle_num,
                ps.x,
                ps.material,
                int(ps.material_fluid),
                out_u,
                float(reg_radius),
            )
            return
        if not self._bs_finite:
            raise NotImplementedError(
                "accumulate_bs_velocity_at_fluid_particles requires finite_segment or point_vortex_2d"
            )
        self._accumulate_bs_at_particles_finite(
            ps.particle_num,
            ps.x,
            ps.material,
            int(ps.material_fluid),
            out_u,
            float(reg_radius),
        )

    @ti.kernel
    def _accumulate_bs_at_particles_2d_point(
        self,
        particle_num: ti.template(),
        x: ti.template(),
        material: ti.template(),
        mf: ti.i32,
        out_u: ti.template(),
        reg_radius: float,
    ):
        R2 = reg_radius * reg_radius
        nseg = self.ss.segment_num[None]
        for p in range(particle_num[None]):
            out_u[p] = ti.Vector([0.0, 0.0, 0.0])
            if material[p] != mf:
                continue
            xp_query = x[p]
            ux = 0.0
            uy = 0.0
            for j in range(nseg):
                if self.ss.active[j] != 1:
                    continue
                am = self.ss.x_minus[j]
                ap = self.ss.x_plus[j]
                cj = 0.5 * (am + ap)
                Lj = (ap - am).norm() + 1e-8
                Gamma = self.ss.gamma[j] * Lj
                bsu = self._bs_velocity_2d_point_vortex(xp_query, cj, Gamma, R2)
                ux += bsu[0]
                uy += bsu[1]
            out_u[p] = ti.Vector([ux, uy, 0.0])

    @ti.kernel
    def _accumulate_bs_at_particles_finite(
        self,
        particle_num: ti.template(),
        x: ti.template(),
        material: ti.template(),
        mf: ti.i32,
        out_u: ti.template(),
        reg_radius: float,
    ):
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = reg_radius * reg_radius
        nseg = self.ss.segment_num[None]
        for p in range(particle_num[None]):
            out_u[p] = ti.Vector([0.0, 0.0, 0.0])
            if material[p] != mf:
                continue
            xp_query = x[p]
            u = ti.Vector([0.0, 0.0, 0.0])
            for j in range(nseg):
                if self.ss.active[j] != 1:
                    continue
                am = self.ss.x_minus[j]
                ap = self.ss.x_plus[j]
                d = ap - am
                Lj = d.norm() + 1e-8
                Gamma = self.ss.gamma[j] * Lj

                apx = ap - xp_query
                ambx = am - xp_query
                u1 = apx / (apx.norm() + reg_radius + 1e-12)
                u2 = ambx / (ambx.norm() + reg_radius + 1e-12)
                sc = (u1 - u2).dot(d)
                cr = ambx.cross(apx)
                cross_sq = cr.dot(cr) + R2
                u += inv4pi * Gamma * sc / (cross_sq + 1e-20) * cr
            out_u[p] = u

    @ti.kernel
    def _center_fix_velocity_pair(
        self,
        u_minus: ti.template(),
        u_plus: ti.template(),
    ):
        """
        去掉端点速度的共模分量 u_c = (u^- + u^+)/2，使 d/dt(center)=0。
        保留相对运动，涡段可在原位旋转/伸缩，整体不平移。
        """
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] != 1:
                continue
            u_c = 0.5 * (u_minus[i] + u_plus[i])
            u_minus[i] -= u_c
            u_plus[i] -= u_c

    def advect_segments_rk4(self):
        """
        TODO：
        使用 RK4 对每条段的两个端点进行对流推进：
            dx_minus/dt = u(x_minus)
            dx_plus/dt  = u(x_plus)

        SegmentConfiguration.advectFixSegmentCenter=true 时：
            dx^±/dt = u(x^±) - (u(x^-) + u(x^+))/2  （段心固定，仅相对运动）
        """
        # 1) k1：在当前端点位置计算速度
        self._copy_v_to_k1()
        if self._advect_fix_segment_center:
            self._center_fix_velocity_pair(self.k1_minus, self.k1_plus)

        # 2) k2：在 x + 0.5*dt*k1 位置计算速度
        self._velocity_at_factor_dispatch(0.5, self.k1_minus, self.k1_plus,
                                          self.k2_minus, self.k2_plus)
        if self._advect_fix_segment_center:
            self._center_fix_velocity_pair(self.k2_minus, self.k2_plus)

        # 3) k3：在 x + 0.5*dt*k2 位置计算速度
        self._velocity_at_factor_dispatch(0.5, self.k2_minus, self.k2_plus,
                                          self.k3_minus, self.k3_plus)
        if self._advect_fix_segment_center:
            self._center_fix_velocity_pair(self.k3_minus, self.k3_plus)

        # 4) k4：在 x + dt*k3 位置计算速度
        self._velocity_at_factor_dispatch(1.0, self.k3_minus, self.k3_plus,
                                          self.k4_minus, self.k4_plus)
        if self._advect_fix_segment_center:
            self._center_fix_velocity_pair(self.k4_minus, self.k4_plus)

        # 5) 更新端点位置并推进 gamma / age
        self._update_endpoints_rk4()
        if self._advect_fix_segment_length:
            self._project_endpoints_fixed_ref_geometry()

    @ti.kernel
    def _advect_segments_euler_placeholder(self):
        for i in range(self.ss.segment_num[None]):
            if self.ss.active[i] == 1:
                if self._advect_frozen(self.ss.seg_type[i]) == 1:
                    continue
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

    def _velocity_at_factor_dispatch(
        self,
        factor: float,
        k_minus_in,
        k_plus_in,
        out_minus,
        out_plus,
    ):
        if self._bs_2d_point:
            self._compute_velocity_at_factor_2d_point(
                factor, k_minus_in, k_plus_in, out_minus, out_plus, float(self.reg_radius)
            )
        elif self._bs_finite:
            self._compute_velocity_at_factor_finite(
                factor, k_minus_in, k_plus_in, out_minus, out_plus, float(self.reg_radius)
            )
        else:
            self._compute_velocity_at_factor_blob(
                factor, k_minus_in, k_plus_in, out_minus, out_plus
            )

    @ti.kernel
    def _compute_velocity_at_factor_2d_point(
        self,
        factor: float,
        k_minus_in: ti.template(),
        k_plus_in: ti.template(),
        out_minus: ti.template(),
        out_plus: ti.template(),
        reg_radius: float,
    ):
        """RK4 子步：在 x + factor*dt*k 处用论文式 (7) 求诱导速度。"""
        R2 = reg_radius * reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i] + factor * self.dt * k_minus_in[i]
            xpi = self.ss.x_plus[i] + factor * self.dt * k_plus_in[i]

            ui_m = self._segment_background(i) + self.sph_advect_velocity_minus[i]
            ui_p = self._segment_background(i) + self.sph_advect_velocity_plus[i]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                am = self.ss.x_minus[j]
                ap = self.ss.x_plus[j]
                cj = 0.5 * (am + ap)
                Lj = (ap - am).norm() + 1e-8
                Gamma = self.ss.gamma[j] * Lj

                bs_m = self._bs_velocity_2d_point_vortex(xmi, cj, Gamma, R2)
                bs_p = self._bs_velocity_2d_point_vortex(xpi, cj, Gamma, R2)
                ui_m[0] += bs_m[0]
                ui_m[1] += bs_m[1]
                ui_p[0] += bs_p[0]
                ui_p[1] += bs_p[1]

            out_minus[i] = ui_m
            out_plus[i] = ui_p

    @ti.kernel
    def _compute_velocity_at_factor_blob(
        self,
        factor: float,
        k_minus_in: ti.template(),
        k_plus_in: ti.template(),
        out_minus: ti.template(),
        out_plus: ti.template(),
    ):
        """
        在试探点 x + factor*dt*k 处求 u(x)（中点 blob）。
        """
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = self.reg_radius * self.reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i] + factor * self.dt * k_minus_in[i]
            xpi = self.ss.x_plus[i] + factor * self.dt * k_plus_in[i]

            ui_m = self._segment_background(i) + self.sph_advect_velocity_minus[i]
            ui_p = self._segment_background(i) + self.sph_advect_velocity_plus[i]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                xmj = self.ss.x_minus[j]
                xpj = self.ss.x_plus[j]
                dj = xpj - xmj
                omega = self.ss.gamma[j] * dj

                r_m = xmi - (0.5 * (xmj + xpj))
                r2_m = r_m.dot(r_m) + R2
                denom_m = r2_m * ti.sqrt(r2_m) + 1e-12
                ui_m += inv4pi * omega.cross(r_m) / denom_m

                r_p = xpi - (0.5 * (xmj + xpj))
                r2_p = r_p.dot(r_p) + R2
                denom_p = r2_p * ti.sqrt(r2_p) + 1e-12
                ui_p += inv4pi * omega.cross(r_p) / denom_p

            out_minus[i] = ui_m
            out_plus[i] = ui_p

    @ti.kernel
    def _compute_velocity_at_factor_finite(
        self,
        factor: float,
        k_minus_in: ti.template(),
        k_plus_in: ti.template(),
        out_minus: ti.template(),
        out_plus: ti.template(),
        reg_radius: float,
    ):
        """RK4 子步：论文式 (6) 与 _compute_endpoint_velocity_bs_finite 一致。"""
        inv4pi = 1.0 / (4.0 * ti.math.pi)
        R2 = reg_radius * reg_radius
        n = self.ss.segment_num[None]

        for i in range(n):
            if self.ss.active[i] != 1:
                continue

            xmi = self.ss.x_minus[i] + factor * self.dt * k_minus_in[i]
            xpi = self.ss.x_plus[i] + factor * self.dt * k_plus_in[i]

            ui_m = self._segment_background(i) + self.sph_advect_velocity_minus[i]
            ui_p = self._segment_background(i) + self.sph_advect_velocity_plus[i]

            for j in range(n):
                if j == i or self.ss.active[j] != 1:
                    continue

                am = self.ss.x_minus[j]
                ap = self.ss.x_plus[j]
                d = ap - am
                Lj = d.norm() + 1e-8
                Gamma = self.ss.gamma[j] * Lj

                apx = ap - xmi
                ambx = am - xmi
                u1 = apx / (apx.norm() + reg_radius + 1e-12)
                u2 = ambx / (ambx.norm() + reg_radius + 1e-12)
                sc = (u1 - u2).dot(d)
                cr = ambx.cross(apx)
                cross_sq = cr.dot(cr) + R2
                ui_m += inv4pi * Gamma * sc / (cross_sq + 1e-20) * cr

                apxp = ap - xpi
                ambxp = am - xpi
                u1p = apxp / (apxp.norm() + reg_radius + 1e-12)
                u2p = ambxp / (ambxp.norm() + reg_radius + 1e-12)
                scp = (u1p - u2p).dot(d)
                crp = ambxp.cross(apxp)
                cross_sqp = crp.dot(crp) + R2
                ui_p += inv4pi * Gamma * scp / (cross_sqp + 1e-20) * crp

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
            if self._advect_frozen(self.ss.seg_type[i]) == 1:
                continue

            self.ss.x_minus[i] += (self.dt / 6.0) * (
                self.k1_minus[i] + 2.0 * self.k2_minus[i] + 2.0 * self.k3_minus[i] + self.k4_minus[i]
            )
            self.ss.x_plus[i] += (self.dt / 6.0) * (
                self.k1_plus[i] + 2.0 * self.k2_plus[i] + 2.0 * self.k3_plus[i] + self.k4_plus[i]
            )

            self.ss.gamma[i] *= self.gamma_decay
            self.ss.age[i] += self.dt

    def _delete_weak_segments_gpu(self, n: int, skip_topo: Set[int]) -> bool:
        if not bool(self.ss.cfg.get_cfg("enableGpuDeleteCompact", False)):
            return False
        if n <= 0:
            return True
        sa, sb, sc, sd = self._skip_ids_for_kernel(skip_topo)
        enable_weak = 1 if bool(self.ss.cfg.get_cfg("enableDeleteWeakSegments", True)) else 0
        ofx_cfg = self.ss.cfg.get_cfg("outflowDeleteCenterBeyondX", None)
        has_outflow = 1 if ofx_cfg is not None else 0
        outflow_x = float(ofx_cfg) if ofx_cfg is not None else 0.0
        max_age_cfg = self.ss.cfg.get_cfg("deleteMaxAge", None)
        has_max_age = 1 if max_age_cfg is not None else 0
        max_age = float(max_age_cfg) if max_age_cfg is not None else 0.0
        delete_outside = 1 if bool(self.ss.cfg.get_cfg("deleteOutsideDomain", False)) else 0
        lo = self.ss.domain_start.astype(np.float32)
        hi = self.ss.domain_end.astype(np.float32)
        lo2 = float(lo[1]) if self.ss.dim >= 2 else 0.0
        hi2 = float(hi[1]) if self.ss.dim >= 2 else 0.0
        lo3 = float(lo[2]) if self.ss.dim >= 3 else 0.0
        hi3 = float(hi[2]) if self.ss.dim >= 3 else 0.0
        self._compact_counter[None] = 0
        self._compact_segments_kernel(
            n,
            enable_weak,
            float(self.delete_gamma_threshold),
            has_outflow,
            outflow_x,
            has_max_age,
            max_age,
            delete_outside,
            float(lo[0]),
            lo2,
            lo3,
            float(hi[0]),
            hi2,
            hi3,
            sa,
            sb,
            sc,
            sd,
        )
        self._copy_compacted_segments_kernel(n)
        return True

    @ti.kernel
    def _compact_segments_kernel(
        self,
        n: int,
        enable_weak: int,
        gamma_threshold: float,
        has_outflow: int,
        outflow_x: float,
        has_max_age: int,
        max_age: float,
        delete_outside: int,
        lo0: float,
        lo1: float,
        lo2: float,
        hi0: float,
        hi1: float,
        hi2: float,
        skip_a: int,
        skip_b: int,
        skip_c: int,
        skip_d: int,
    ):
        for i in range(n):
            st = self.ss.seg_type[i]
            skip = (st == skip_a) or (st == skip_b) or (st == skip_c) or (st == skip_d)
            keep = self.ss.active[i] == 1
            if keep and enable_weak == 1 and not skip:
                keep = ti.abs(self.ss.gamma[i]) >= gamma_threshold
            if keep and has_outflow == 1 and not skip:
                mid_x = 0.5 * (self.ss.x_minus[i][0] + self.ss.x_plus[i][0])
                keep = mid_x <= outflow_x
            if keep and has_max_age == 1:
                keep = self.ss.age[i] <= max_age
            if keep and delete_outside == 1 and not skip:
                if ti.static(self.ss.dim == 2):
                    xm = self.ss.x_minus[i]
                    xp = self.ss.x_plus[i]
                    keep = (
                        xm[0] >= lo0 and xm[0] <= hi0 and xm[1] >= lo1 and xm[1] <= hi1 and
                        xp[0] >= lo0 and xp[0] <= hi0 and xp[1] >= lo1 and xp[1] <= hi1
                    )
                else:
                    xm = self.ss.x_minus[i]
                    xp = self.ss.x_plus[i]
                    keep = (
                        xm[0] >= lo0 and xm[0] <= hi0 and xm[1] >= lo1 and xm[1] <= hi1 and xm[2] >= lo2 and xm[2] <= hi2 and
                        xp[0] >= lo0 and xp[0] <= hi0 and xp[1] >= lo1 and xp[1] <= hi1 and xp[2] >= lo2 and xp[2] <= hi2
                    )
            if keep:
                j = ti.atomic_add(self._compact_counter[None], 1)
                self._compact_x_minus[j] = self.ss.x_minus[i]
                self._compact_x_plus[j] = self.ss.x_plus[i]
                self._compact_gamma[j] = self.ss.gamma[i]
                self._compact_age[j] = self.ss.age[i]
                self._compact_seg_type[j] = self.ss.seg_type[i]

    @ti.kernel
    def _copy_compacted_segments_kernel(self, n_old: int):
        n_new = self._compact_counter[None]
        for i in range(n_old):
            self.ss.active[i] = 0
        for i in range(n_new):
            self.ss.x_minus[i] = self._compact_x_minus[i]
            self.ss.x_plus[i] = self._compact_x_plus[i]
            self.ss.gamma[i] = self._compact_gamma[i]
            self.ss.age[i] = self._compact_age[i]
            self.ss.seg_type[i] = self._compact_seg_type[i]
            self.ss.active[i] = 1
            d = self.ss.tangent_ref[i]
            l = d.norm() + 1e-8
            self.ss.center[i] = 0.5 * (self.ss.x_minus[i] + self.ss.x_plus[i])
            self.ss.tangent[i] = d / l
            self.ss.length[i] = l
        self.ss.segment_num[None] = n_new

    def _delete_weak_segments_cpu(self):
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

        skip_topo = self._topology_skip_type_ids()
        if self._delete_weak_segments_gpu(n, skip_topo):
            return
        keep = active == 1
        if bool(self.ss.cfg.get_cfg("enableDeleteWeakSegments", True)):
            keep &= np.abs(gamma) >= float(self.delete_gamma_threshold)
        if len(skip_topo) > 0:
            keep |= np.isin(seg_type, list(skip_topo))

        ofx = self.ss.cfg.get_cfg("outflowDeleteCenterBeyondX", None)
        if ofx is not None:
            mid_x = 0.5 * (x_minus[:, 0] + x_plus[:, 0])
            outflow = mid_x > float(ofx)
            if len(skip_topo) > 0:
                keep &= (~outflow) | np.isin(seg_type, list(skip_topo))
            else:
                keep &= ~outflow

        # 可选：按年龄删除
        max_age = self.ss.cfg.get_cfg("deleteMaxAge", None)
        if max_age is not None:
            keep &= (age <= float(max_age))

        # 可选：删除域外段（要求两端点都在域内）
        delete_outside = bool(self.ss.cfg.get_cfg("deleteOutsideDomain", False))
        if delete_outside:
            d = self.ss.dim
            lo = self.ss.domain_start.astype(np.float32)[:d]
            hi = self.ss.domain_end.astype(np.float32)[:d]
            in_m = np.all(
                (x_minus[:, :d] >= lo[None, :]) & (x_minus[:, :d] <= hi[None, :]), axis=1
            )
            in_p = np.all(
                (x_plus[:, :d] >= lo[None, :]) & (x_plus[:, :d] <= hi[None, :]), axis=1
            )
            inside_keep = in_m & in_p
            if len(skip_topo) > 0:
                inside_keep |= np.isin(seg_type, list(skip_topo))
            keep &= inside_keep

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

    def delete_weak_segments(self):
        n = int(self.ss.segment_num[None])
        if n <= 0:
            return
        skip_topo = self._topology_skip_type_ids()
        if self._delete_weak_segments_gpu(n, skip_topo):
            return
        self._delete_weak_segments_cpu()

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
        if not bool(self.ss.cfg.get_cfg("enableSplitSegments", True)):
            return
        old_n = int(self.ss.segment_num[None])
        if old_n <= 0:
            return
        sa, sb, sc, sd = self._skip_ids_for_kernel(self._topology_skip_type_ids())
        self._split_segments_kernel(old_n, float(self.split_len_threshold), sa, sb, sc, sd)

    @ti.kernel
    def _split_segments_kernel(
        self,
        old_n: int,
        split_len_threshold: float,
        skip_type_a: int,
        skip_type_b: int,
        skip_type_c: int,
        skip_type_d: int,
    ):
        """
        分裂策略（论文式(12)的最小实现）：
        - 仅处理 old_n 范围内原有段，避免本次新增段再次被处理
        - 对每条满足 length > threshold 的活跃段：
          1) 计算中点 m
          2) 原段改为 [x_minus, m]
          3) 追加新段 [m, x_plus_old]
        - 新段继承 gamma / seg_type，age 置 0
        - 若容量不足则跳过追加（原段仍会被截断到一半）
        - skip_type_*：若为 != -1 且 seg_type 等于其中之一，则不对该段分裂（用于边界虚拟段等）
        """
        max_n = self.ss.segment_max_num
        for i in range(old_n):
            if self.ss.active[i] != 1:
                continue
            if self.ss.length[i] <= split_len_threshold:
                continue

            st = self.ss.seg_type[i]
            if skip_type_a != -1 and st == skip_type_a:
                continue
            if skip_type_b != -1 and st == skip_type_b:
                continue
            if skip_type_c != -1 and st == skip_type_c:
                continue
            if skip_type_d != -1 and st == skip_type_d:
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
        if not bool(self.ss.cfg.get_cfg("enableMergeSegments", True)):
            return
        n = int(self.ss.segment_num[None])
        if n <= 1:
            return

        merge_dist = float(self.ss.cfg.get_cfg("mergeDistanceLambda", 0.03))
        merge_angle = float(self.ss.cfg.get_cfg("mergeAngleThreshold", 5.0 * np.pi / 6.0))
        # 近反向：dot(t_i, t_j) <= cos(theta)，theta 接近 pi 时更严格
        dot_th = float(np.cos(merge_angle))
        skip_merge_types = self._topology_skip_type_ids()

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
            if int(seg_type[i]) in skip_merge_types:
                used[i] = True
                out_xm.append(x_minus[i])
                out_xp.append(x_plus[i])
                out_g.append(gamma[i])
                out_a.append(age[i])
                out_t.append(seg_type[i])
                continue

            # 找到可合并的最佳候选 j（最近）
            best_j = -1
            best_d2 = 1e30
            ci = center[i]
            ti = tangent[i]
            for j in range(i + 1, n):
                if active[j] != 1 or used[j]:
                    continue
                if int(seg_type[j]) in skip_merge_types:
                    continue
                if bool(self.ss.cfg.get_cfg("mergeRequireSameSegmentType", False)):
                    if int(seg_type[i]) != int(seg_type[j]):
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

            out_xm.append(xm_new)
            out_xp.append(xp_new)
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
            self.ss.set_segment_ends_from_ndarray_row(i, i, x_minus, x_plus)
            self.ss.gamma[i] = gamma[i]
            self.ss.age[i] = age[i]
            self.ss.seg_type[i] = seg_type[i]
            self.ss.active[i] = 1

    def _merge_candidate_indices_from_grid(self, grid, cell, radius: int, dim: int):
        if dim == 2:
            cx, cy = cell
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    yield from grid.get((cx + dx, cy + dy), ())
            return
        cx, cy, cz = cell
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    yield from grid.get((cx + dx, cy + dy, cz + dz), ())

    def merge_segments(self):
        if not bool(self.ss.cfg.get_cfg("enableMergeSegments", True)):
            return
        interval = max(1, int(self.ss.cfg.get_cfg("mergeIntervalSteps", 1)))
        if interval > 1 and int(self._sim_step_index) % interval != 0:
            return
        n = int(self.ss.segment_num[None])
        if n <= 1:
            return

        merge_dist = float(self.ss.cfg.get_cfg("mergeDistanceLambda", 0.03))
        if merge_dist <= 0.0:
            return
        merge_angle = float(self.ss.cfg.get_cfg("mergeAngleThreshold", 5.0 * np.pi / 6.0))
        dot_th = float(np.cos(merge_angle))
        skip_merge_types = self._topology_skip_type_ids()
        require_same_type = bool(self.ss.cfg.get_cfg("mergeRequireSameSegmentType", False))

        x_minus = self.ss.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss.active.to_numpy()[:n].astype(np.int32)
        age = self.ss.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss.seg_type.to_numpy()[:n].astype(np.int32)
        center = self.ss.center.to_numpy()[:n].astype(np.float32)
        tangent = self.ss.tangent.to_numpy()[:n].astype(np.float32)

        d = int(self.ss.dim)
        use_hash = bool(self.ss.cfg.get_cfg("mergeSpatialHashEnabled", True))
        cell_size = max(float(self.ss.cfg.get_cfg("mergeSpatialHashCellSize", merge_dist)), 1e-8)
        inv_cell = 1.0 / cell_size
        neighbor_radius = max(1, int(np.ceil(merge_dist / cell_size)))

        grid = {}
        if use_hash:
            for i in range(n):
                if active[i] != 1:
                    continue
                if d == 2:
                    cell = (
                        int(np.floor(float(center[i, 0]) * inv_cell)),
                        int(np.floor(float(center[i, 1]) * inv_cell)),
                    )
                else:
                    cell = (
                        int(np.floor(float(center[i, 0]) * inv_cell)),
                        int(np.floor(float(center[i, 1]) * inv_cell)),
                        int(np.floor(float(center[i, 2]) * inv_cell)),
                    )
                grid.setdefault(cell, []).append(i)

        used = np.zeros((n,), dtype=bool)
        out_xm = []
        out_xp = []
        out_g = []
        out_a = []
        out_t = []
        merge_dist2 = merge_dist * merge_dist

        for i in range(n):
            if active[i] != 1 or used[i]:
                continue
            if int(seg_type[i]) in skip_merge_types:
                used[i] = True
                out_xm.append(x_minus[i])
                out_xp.append(x_plus[i])
                out_g.append(gamma[i])
                out_a.append(age[i])
                out_t.append(seg_type[i])
                continue

            best_j = -1
            best_d2 = 1e30
            ci = center[i]
            ti = tangent[i]
            if use_hash:
                if d == 2:
                    cell_i = (
                        int(np.floor(float(ci[0]) * inv_cell)),
                        int(np.floor(float(ci[1]) * inv_cell)),
                    )
                else:
                    cell_i = (
                        int(np.floor(float(ci[0]) * inv_cell)),
                        int(np.floor(float(ci[1]) * inv_cell)),
                        int(np.floor(float(ci[2]) * inv_cell)),
                    )
                candidates = self._merge_candidate_indices_from_grid(grid, cell_i, neighbor_radius, d)
            else:
                candidates = range(i + 1, n)

            for j in candidates:
                if j <= i:
                    continue
                if active[j] != 1 or used[j]:
                    continue
                if int(seg_type[j]) in skip_merge_types:
                    continue
                if require_same_type and int(seg_type[i]) != int(seg_type[j]):
                    continue
                dc = ci[:d] - center[j, :d]
                d2 = float(np.dot(dc, dc))
                if d2 > merge_dist2:
                    continue
                dotv = float(np.dot(ti[:d], tangent[j, :d]))
                if dotv > dot_th:
                    continue
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = int(j)

            if best_j < 0:
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
                continue

            t_new = (w / (w_norm + 1e-8)).astype(np.float32)
            xm_new = c_new - 0.5 * L_new * t_new
            xp_new = c_new + 0.5 * L_new * t_new
            g_new = w_norm / (L_new + 1e-8)

            out_xm.append(xm_new)
            out_xp.append(xp_new)
            out_g.append(np.float32(g_new))
            out_a.append(np.float32(min(age[i], age[j])))
            out_t.append(seg_type[i])

        new_n = len(out_xm)
        if new_n <= 0:
            self.ss.segment_num[None] = 0
            return

        self._overwrite_segments_kernel(
            new_n,
            np.asarray(out_xm, dtype=np.float32),
            np.asarray(out_xp, dtype=np.float32),
            np.asarray(out_g, dtype=np.float32),
            np.asarray(out_a, dtype=np.float32),
            np.asarray(out_t, dtype=np.int32),
        )
        self.ss.segment_num[None] = new_n

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

    def _strip_segments_of_type(self, type_id: int):
        n = int(self.ss.segment_num[None])
        if n <= 0:
            return
        x_minus = self.ss.x_minus.to_numpy()[:n].astype(np.float32)
        x_plus = self.ss.x_plus.to_numpy()[:n].astype(np.float32)
        gamma = self.ss.gamma.to_numpy()[:n].astype(np.float32)
        active = self.ss.active.to_numpy()[:n].astype(np.int32)
        age = self.ss.age.to_numpy()[:n].astype(np.float32)
        seg_type = self.ss.seg_type.to_numpy()[:n].astype(np.int32)
        if not np.any((active == 1) & (seg_type == int(type_id))):
            return
        keep = (active == 1) & (seg_type != int(type_id))
        idx = np.nonzero(keep)[0]
        new_n = int(idx.size)
        if new_n <= 0:
            self._clear_active_kernel(n)
            self.ss.segment_num[None] = 0
            return
        self._overwrite_segments_kernel(
            new_n,
            x_minus[idx],
            x_plus[idx],
            gamma[idx],
            age[idx],
            seg_type[idx],
        )
        self.ss.segment_num[None] = new_n

    def _resolve_emitter_inlet_bounds(self):
        """与 FluidEmitters[0] 或 emitterX/Y0/Y1 对齐的入口铺段区域。"""
        use_fe = bool(self.ss.cfg.get_cfg("emitterUseFluidEmitterLayout", True))
        emitters = self.ss.cfg.get_inflow() if hasattr(self.ss.cfg, "get_inflow") else []
        if use_fe and len(emitters) > 0:
            e0 = emitters[0]
            c = np.array(e0["squareCenter"], dtype=np.float64).reshape(-1)
            s = np.array(e0["squareSize"], dtype=np.float64).reshape(-1)
            lo = c - 0.5 * s
            hi = c + 0.5 * s
            if float(s[0]) < 1e-8:
                x_emit = float(self.ss.cfg.get_cfg("emitterPlaneInsetX", 0.02))
            else:
                x_emit = float(0.5 * (lo[0] + hi[0]))
            y0, y1 = float(lo[1]), float(hi[1])
            if self.ss.dim >= 3 and s.size >= 3:
                z0, z1 = float(lo[2]), float(hi[2])
            else:
                z0, z1 = 0.0, 0.0
        else:
            x_emit = float(self.ss.cfg.get_cfg("emitterX", 0.05))
            y0 = float(self.ss.cfg.get_cfg("emitterY0", 0.1))
            y1 = float(self.ss.cfg.get_cfg("emitterY1", 0.9))
            z0 = float(self.ss.cfg.get_cfg("emitterZ0", 0.1))
            z1 = float(self.ss.cfg.get_cfg("emitterZ1", 0.9))
        return x_emit, y0, y1, z0, z1

    def _emitter_gamma_sign_for_y_slot(self, iy: int, ny: int) -> float:
        mode = str(self.ss.cfg.get_cfg("emitterGammaYMode", "alternate") or "alternate").lower().strip()
        if mode in ("split_half", "half", "y_half", "upper_lower", "front_back_half"):
            split = int(self.ss.cfg.get_cfg("emitterGammaYSplitIndex", max(1, ny // 2)))
            split = max(1, min(int(ny), split))
            first_sign = float(self.ss.cfg.get_cfg("emitterGammaFirstHalfSign", 1.0))
            return first_sign if int(iy) < split else -first_sign
        alternate_y = bool(self.ss.cfg.get_cfg("emitterGammaAlternateY", True))
        return 1.0 if (not alternate_y or (int(iy) % 2) == 0) else -1.0

    def _emit_inlet_segments_streamwise_x(
        self, x_emit: float, y0: float, y1: float
    ):
        """2D：在入口 x 平面铺沿 +X 的短涡段（与来流平行），y 向均匀排布。"""
        ny = max(
            1,
            int(
                self.ss.cfg.get_cfg(
                    "emitterNy", self.ss.cfg.get_cfg("emitterMeshNy", 12)
                )
            ),
        )
        nx_sub = max(
            1,
            int(
                self.ss.cfg.get_cfg(
                    "emitterSubdivisionsX",
                    self.ss.cfg.get_cfg("emitterMeshNx", 1),
                )
            ),
        )
        nx_cap = int(self.ss.cfg.get_cfg("emitterSegmentsPerYPerEmit", 0))
        if nx_cap > 0:
            nx_sub = min(nx_sub, nx_cap)
        seg_len = float(self.ss.cfg.get_cfg("emitterSegmentLength", 0.018))
        g0 = float(self.ss.cfg.get_cfg("emitterGamma", 0.02))
        st = int(self.ss.cfg.get_cfg("emitterSegTypeId", 0))
        jy_cfg = self.ss.cfg.get_cfg("emitterYJitter", None)
        if jy_cfg is not None:
            jyz = float(jy_cfg)
        else:
            jyz = float(self.ss.cfg.get_cfg("emitterJitterYZ", 0.0))
        alternate_y = bool(self.ss.cfg.get_cfg("emitterGammaAlternateY", True))

        ys = np.linspace(y0, y1, ny, dtype=np.float32)
        if jyz > 0.0:
            ys = ys + self._emitter_rng.uniform(-jyz, jyz, size=ny).astype(np.float32)
            ys = np.clip(
                ys,
                float(self.ss.domain_start[1]) + 1e-4,
                float(self.ss.domain_end[1]) - 1e-4,
            )

        xm_list = []
        xp_list = []
        g_list = []
        for iy in range(ny):
            yj = float(ys[iy])
            sgn_y = self._emitter_gamma_sign_for_y_slot(iy, ny)
            for _ in range(nx_sub):
                xm_list.append([x_emit - 0.5 * seg_len, yj, 0.0])
                xp_list.append([x_emit + 0.5 * seg_len, yj, 0.0])
                g_list.append(float(sgn_y * g0))

        xm_list, xp_list, g_list = self._select_emit_batch(xm_list, xp_list, g_list)
        self._commit_emitted_segments(xm_list, xp_list, g_list, st)

    def _select_emit_batch(self, xm_list, xp_list, g_list):
        """
        可选限制单次发射总段数（默认 0 = 不限制）。

        2D 入口典型用法：y 向 ``emitterNy`` 个发射口同时各放 1 段（``emitterSubdivisionsX: 1``、
        ``emitterSegmentsPerYPerEmit: 1``），用 ``emitterIntervalSteps`` / ``emitterIntervalStride``
        控制时间频率；不要用 round_robin 关掉多 y 口。
        """
        n = len(xm_list)
        if n <= 0:
            return xm_list, xp_list, g_list
        max_per = int(self.ss.cfg.get_cfg("emitterMaxSegmentsPerEmit", 0))
        if max_per <= 0:
            return xm_list, xp_list, g_list
        mode = str(self.ss.cfg.get_cfg("emitterBatchMode", "all") or "all").lower().strip()
        max_per = min(max_per, n)
        if mode in ("round_robin", "sequential", "one", "single"):
            idx = int(self._emitter_slot_cursor) % n
            self._emitter_slot_cursor = (idx + 1) % n
            return [xm_list[idx]], [xp_list[idx]], [g_list[idx]]
        return xm_list[:max_per], xp_list[:max_per], g_list[:max_per]

    def _emit_inlet_segments_spanwise_z(
        self, x_emit: float, y0: float, y1: float, z0: float, z1: float
    ):
        """3D：沿 z 的涡线（固定 x、y），在 y 方向堆叠。"""
        ny = max(
            1,
            int(
                self.ss.cfg.get_cfg(
                    "emitterNy", self.ss.cfg.get_cfg("emitterMeshNy", 12)
                )
            ),
        )
        nz_sub = max(
            1,
            int(
                self.ss.cfg.get_cfg(
                    "emitterSubdivisionsZ",
                    self.ss.cfg.get_cfg("emitterMeshNz", 1),
                )
            ),
        )
        g0 = float(self.ss.cfg.get_cfg("emitterGamma", 0.02))
        st = int(self.ss.cfg.get_cfg("emitterSegTypeId", 0))
        jy_cfg = self.ss.cfg.get_cfg("emitterYJitter", None)
        if jy_cfg is not None:
            jyz = float(jy_cfg)
        else:
            jyz = float(self.ss.cfg.get_cfg("emitterJitterYZ", 0.0))
        alternate_y = bool(self.ss.cfg.get_cfg("emitterGammaAlternateY", True))

        ys = np.linspace(y0, y1, ny, dtype=np.float32)
        zs = np.linspace(z0, z1, nz_sub + 1, dtype=np.float32)
        if jyz > 0.0:
            ys = ys + self._emitter_rng.uniform(-jyz, jyz, size=ny).astype(np.float32)
            ys = np.clip(
                ys,
                float(self.ss.domain_start[1]) + 1e-4,
                float(self.ss.domain_end[1]) - 1e-4,
            )

        xm_list = []
        xp_list = []
        g_list = []
        for iy in range(ny):
            yj = float(ys[iy])
            sgn_y = self._emitter_gamma_sign_for_y_slot(iy, ny)
            for iz in range(nz_sub):
                sgn = sgn_y if alternate_y else 1.0
                z0s = float(zs[iz])
                z1s = float(zs[iz + 1])
                xm_list.append([x_emit, yj, z0s])
                xp_list.append([x_emit, yj, z1s])
                g_list.append(float(sgn * g0))

        xm_list, xp_list, g_list = self._select_emit_batch(xm_list, xp_list, g_list)
        self._commit_emitted_segments(xm_list, xp_list, g_list, st)

    def _commit_emitted_segments(self, xm_list, xp_list, g_list, seg_type: int):
        n_seg = len(xm_list)
        if n_seg <= 0:
            return
        xm = np.asarray(xm_list, dtype=np.float32)
        xp = np.asarray(xp_list, dtype=np.float32)
        g = np.asarray(g_list, dtype=np.float32)
        offset = int(self.ss.segment_num[None])
        cap = int(self.ss.segment_max_num)
        if offset >= cap:
            return
        n_new = min(n_seg, cap - offset)
        if n_new <= 0:
            return
        self._seed_segments_kernel(
            offset, n_new, xm[:n_new], xp[:n_new], g[:n_new], seg_type
        )
        self.ss.segment_num[None] = offset + n_new

    def _emitter_burst_active(self) -> bool:
        if not bool(self.ss.cfg.get_cfg("emitterBurstEnabled", False)):
            return True
        on_steps = int(self.ss.cfg.get_cfg("emitterBurstOnSteps", 1))
        off_steps = int(self.ss.cfg.get_cfg("emitterBurstOffSteps", 0))
        start_step = int(self.ss.cfg.get_cfg("emitterBurstStartStep", 0))
        on_steps = max(0, on_steps)
        off_steps = max(0, off_steps)
        local_step = int(self._sim_step_index) - start_step
        if local_step < 0:
            return False
        cycle = on_steps + off_steps
        if cycle <= 0:
            return False
        phase = local_step % cycle
        return phase < on_steps

    def _emit_periodic_parallel_x_layers(self):
        if not bool(self.ss.cfg.get_cfg("parallelXRepeatEnabled", False)):
            return
        interval = max(1, int(self.ss.cfg.get_cfg("parallelXRepeatIntervalSteps", 100)))
        start_step = int(self.ss.cfg.get_cfg("parallelXRepeatStartStep", interval))
        step = int(self._sim_step_index)
        if step < start_step:
            return
        if (step - start_step) % interval != 0:
            return
        max_batches = int(self.ss.cfg.get_cfg("parallelXRepeatMaxBatches", 0) or 0)
        if max_batches > 0:
            batch_idx = (step - start_step) // interval
            if batch_idx >= max_batches:
                return
        self._seed_parallel_x_layers_filaments()

    def _emit_inlet_segments(self):
        if not bool(self.ss.cfg.get_cfg("emitterEnabled", False)):
            return
        if not self._emitter_burst_active():
            return
        stride = max(1, int(self.ss.cfg.get_cfg("emitterIntervalStride", 1)))
        if self._emitter_interval_override is not None:
            interval = max(1, int(self._emitter_interval_override)) * stride
        else:
            interval = max(
                1, int(self.ss.cfg.get_cfg("emitterIntervalSteps", 1))
            ) * stride
        if int(self._sim_step_index) % interval != 0:
            return

        x_emit, y0, y1, z0, z1 = self._resolve_emitter_inlet_bounds()
        orient = str(
            self.ss.cfg.get_cfg("emitterOrientation", "") or ""
        ).lower().strip()
        if not orient:
            orient = "streamwise_x" if self.ss.dim == 2 else "spanwise_z"

        if orient in ("streamwise_x", "x", "along_x", "2d_inlet"):
            self._emit_inlet_segments_streamwise_x(x_emit, y0, y1)
        else:
            self._emit_inlet_segments_spanwise_z(x_emit, y0, y1, z0, z1)

    def _step_timing_should_print(self) -> bool:
        return self._step_timing_enabled and (int(self._sim_step_index) % self._step_timing_interval == 0)

    def _step_timing_mark(self, marks, name: str):
        if not self._step_timing_enabled:
            return
        if self._step_timing_sync:
            ti.sync()
        marks.append((name, time.perf_counter()))

    def _step_timing_report(self, marks, active_before: int, active_after: int):
        if not self._step_timing_should_print() or len(marks) < 2:
            return
        total = (marks[-1][1] - marks[0][1]) * 1000.0
        parts = []
        for i in range(1, len(marks)):
            dt_ms = (marks[i][1] - marks[i - 1][1]) * 1000.0
            parts.append(f"{marks[i][0]}={dt_ms:.3f}ms")
        print(
            f"[StepTiming] step={int(self._sim_step_index)} "
            f"segments={active_before}->{active_after} total={total:.3f}ms | "
            + ", ".join(parts)
        )

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
            if self._boundary_schedule == "each_step":
                strip = bool(self.ss.cfg.get_cfg("boundaryReplaceCommittedEachStep", False))
                self._run_boundary_injection_pipeline(strip_committed_first=strip)
            elif not self._boundary_one_shot_done:
                strip = bool(self.ss.cfg.get_cfg("boundaryReplaceCommittedEachStep", False))
                self._run_boundary_injection_pipeline(strip_committed_first=strip)
                self.boundary.mark_one_shot_complete_if_applicable(self)

        self._emit_inlet_segments()
        self._emit_periodic_parallel_x_layers()
        self.ss.update_segment_geometry()
        self.compute_endpoint_velocity()
        self.advect_segments_rk4()
        self.ss.update_segment_geometry()
        self.split_segments()
        self.merge_segments()
        self.restore_frozen_segment_geometry()
        self.delete_weak_segments()
        # 初始脉冲只生效一个时间步
        if self._use_leapfrog_initial_impulse[None] == 1:
            self._decay_impulse_kernel()
        self._sim_step_index += 1
        # self.cull_segments()

    def step(self):
        """
        独立 Segment 仿真的主步骤模板，带可选分段计时。
        """
        timing_marks = []
        active_before = int(self.ss.segment_num[None])
        self._step_timing_mark(timing_marks, "start")

        if self.has_boundary:
            if self._boundary_schedule == "each_step":
                strip = bool(self.ss.cfg.get_cfg("boundaryReplaceCommittedEachStep", False))
                self._run_boundary_injection_pipeline(strip_committed_first=strip)
            elif not self._boundary_one_shot_done:
                strip = bool(self.ss.cfg.get_cfg("boundaryReplaceCommittedEachStep", False))
                self._run_boundary_injection_pipeline(strip_committed_first=strip)
                self.boundary.mark_one_shot_complete_if_applicable(self)
        self._step_timing_mark(timing_marks, "boundary")

        self._emit_inlet_segments()
        self._emit_periodic_parallel_x_layers()
        self._step_timing_mark(timing_marks, "emitter")
        self.ss.update_segment_geometry()
        self._step_timing_mark(timing_marks, "geom_pre")
        self.compute_endpoint_velocity()
        self._step_timing_mark(timing_marks, "velocity")
        self.advect_segments_rk4()
        self._step_timing_mark(timing_marks, "advect")
        self.ss.update_segment_geometry()
        self._step_timing_mark(timing_marks, "geom_post")
        self.split_segments()
        self._step_timing_mark(timing_marks, "split")
        self.merge_segments()
        self._step_timing_mark(timing_marks, "merge")
        self.restore_frozen_segment_geometry()
        self._step_timing_mark(timing_marks, "restore_frozen")
        self.delete_weak_segments()
        self._step_timing_mark(timing_marks, "delete")
        if self._use_leapfrog_initial_impulse[None] == 1:
            self._decay_impulse_kernel()
        self._step_timing_mark(timing_marks, "impulse")
        active_after = int(self.ss.segment_num[None])
        self._step_timing_mark(timing_marks, "end")
        self._step_timing_report(timing_marks, active_before, active_after)
        self._sim_step_index += 1
