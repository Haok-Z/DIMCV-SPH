import taichi as ti
import numpy as np
from segment_config import SegmentConfig


@ti.data_oriented
class SegmentSystem:
    def __init__(self, config: SegmentConfig):
        self.cfg = config
        self.dim = 3

        self.domain_start = np.array(self.cfg.get_domain_start(), dtype=np.float32)
        self.domain_end = np.array(self.cfg.get_domain_end(), dtype=np.float32)

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

    @ti.kernel
    def update_segment_geometry(self):
        for i in range(self.segment_num[None]):
            if self.active[i] == 1:
                d = self.x_plus[i] - self.x_minus[i]
                l = d.norm() + 1e-8
                self.center[i] = 0.5 * (self.x_plus[i] + self.x_minus[i])
                self.tangent[i] = d / l
                self.length[i] = l

