import taichi as ti
import numpy as np
from segment_config import SegmentConfig


@ti.data_oriented
class SegmentSystem:
    def __init__(self, config: SegmentConfig):
        self.cfg = config

        self.domain_start = np.array(self.cfg.get_domain_start(), dtype=np.float32)
        self.domain_end = np.array(self.cfg.get_domain_end(), dtype=np.float32)
        self.dim = int(len(self.domain_end))
        if self.dim not in (2, 3):
            raise ValueError(
                f"Segment domain must be 2D or 3D (domainStart/End length={self.dim})"
            )
        self.segment_max_num = int(self.cfg.get_cfg("segmentMaxNum", 200000))

        self.segment_num = ti.field(dtype=int, shape=())
        self.segment_num[None] = 0

        self.x_minus = ti.Vector.field(self.dim, dtype=float, shape=self.segment_max_num)
        self.x_plus = ti.Vector.field(self.dim, dtype=float, shape=self.segment_max_num)
        self.gamma = ti.field(dtype=float, shape=self.segment_max_num)
        self.active = ti.field(dtype=int, shape=self.segment_max_num)
        self.age = ti.field(dtype=float, shape=self.segment_max_num)
        self.seg_type = ti.field(dtype=int, shape=self.segment_max_num)

        self.center = ti.Vector.field(self.dim, dtype=float, shape=self.segment_max_num)
        self.tangent = ti.Vector.field(self.dim, dtype=float, shape=self.segment_max_num)
        self.length = ti.field(dtype=float, shape=self.segment_max_num)

    @ti.kernel
    def clear(self):
        self.segment_num[None] = 0
        self.active.fill(0)
        self.age.fill(0.0)
        self.gamma.fill(0.0)
        self.length.fill(0.0)
        self.seg_type.fill(0)

    @ti.func
    def set_segment_ends_from_ndarray_row(
        self, i: int, row: int, x_minus_arr: ti.template(), x_plus_arr: ti.template()
    ):
        """从 ndarray 行写入端点；2D 场只取前两列（第三列可忽略）。"""
        if ti.static(self.dim == 2):
            self.x_minus[i] = ti.Vector([x_minus_arr[row, 0], x_minus_arr[row, 1]])
            self.x_plus[i] = ti.Vector([x_plus_arr[row, 0], x_plus_arr[row, 1]])
        else:
            self.x_minus[i] = ti.Vector([
                x_minus_arr[row, 0], x_minus_arr[row, 1], x_minus_arr[row, 2]
            ])
            self.x_plus[i] = ti.Vector([
                x_plus_arr[row, 0], x_plus_arr[row, 1], x_plus_arr[row, 2]
            ])

    @ti.kernel
    def update_segment_geometry(self):
        for i in range(self.segment_num[None]):
            if self.active[i] == 1:
                d = self.x_plus[i] - self.x_minus[i]
                l = d.norm() + 1e-8
                self.center[i] = 0.5 * (self.x_plus[i] + self.x_minus[i])
                self.tangent[i] = d / l
                self.length[i] = l

