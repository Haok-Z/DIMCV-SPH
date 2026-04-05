from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class SegmentExporter:
    def __init__(self, segment_system):
        self.ss = segment_system

    def export_segments_ply(self, frame_id: int, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"segments_{frame_id:04}.ply"

        n = self.ss.segment_num[None]
        if n == 0:
            return

        x_minus = self.ss.x_minus.to_numpy()[:n]
        x_plus = self.ss.x_plus.to_numpy()[:n]
        gamma = self.ss.gamma.to_numpy()[:n]
        active = self.ss.active.to_numpy()[:n].astype(bool)
        seg_type = self.ss.seg_type.to_numpy()[:n]

        # Export segment endpoints as point cloud for quick inspection.
        pts = []
        for i in range(n):
            if not active[i]:
                continue
            pts.append((x_minus[i], gamma[i], seg_type[i], 0))
            pts.append((x_plus[i], gamma[i], seg_type[i], 1))

        if len(pts) == 0:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float gamma\n")
            f.write("property int segment_type\n")
            f.write("property uchar endpoint\n")
            f.write("end_header\n")
            for p, g, t, e in pts:
                f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {g:.7f} {int(t)} {int(e)}\n")

    def export_segments_png(self, frame_id: int, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"segments_{frame_id:04}.png"

        n = self.ss.segment_num[None]
        if n == 0:
            return

        x_minus = self.ss.x_minus.to_numpy()[:n]
        x_plus = self.ss.x_plus.to_numpy()[:n]
        gamma = self.ss.gamma.to_numpy()[:n]
        active = self.ss.active.to_numpy()[:n].astype(bool)

        if not np.any(active):
            return

        x_minus = x_minus[active]
        x_plus = x_plus[active]
        gamma = gamma[active]

        # 使用段中心做着色可视化，颜色编码 gamma
        center = 0.5 * (x_minus + x_plus)

        # 仿照 karman_vortex.py 的风格：白底 + 3D 视角 + tight 输出
        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=-60)

        # 颜色映射范围：支持显式配置，默认按数据自适应
        vmin = self.ss.cfg.get_cfg("imageGammaVmin", None)
        vmax = self.ss.cfg.get_cfg("imageGammaVmax", None)
        if vmin is None:
            vmin = float(np.min(gamma))
        else:
            vmin = float(vmin)
        if vmax is None:
            vmax = float(np.max(gamma))
        else:
            vmax = float(vmax)
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-8
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "coolwarm"

        # 先画涡丝线段（按 gamma 着色）
        line_width = float(self.ss.cfg.get_cfg("imageLineWidth", 0.8))
        line_alpha = float(self.ss.cfg.get_cfg("imageLineAlpha", 0.9))
        cm = plt.get_cmap(cmap)
        for i in range(x_minus.shape[0]):
            col = cm(norm(float(gamma[i])))
            ax.plot(
                [float(x_minus[i, 0]), float(x_plus[i, 0])],
                [float(x_minus[i, 1]), float(x_plus[i, 1])],
                [float(x_minus[i, 2]), float(x_plus[i, 2])],
                color=col,
                linewidth=line_width,
                alpha=line_alpha,
            )

        # 再画散点（中心点）
        sc = ax.scatter(
            center[:, 0],
            center[:, 1],
            center[:, 2],
            c=gamma,
            cmap=cmap,
            s=float(self.ss.cfg.get_cfg("imagePointSize", 1.0)),
            norm=norm,
            edgecolors="none",
            alpha=float(self.ss.cfg.get_cfg("imagePointAlpha", 0.6)),
        )
        if bool(self.ss.cfg.get_cfg("imageShowColorbar", False)):
            fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)

        domain_start = self.ss.domain_start.astype(np.float32)
        domain_end = self.ss.domain_end.astype(np.float32)
        span = np.maximum(domain_end - domain_start, 1e-6)

        # 视野范围支持三种方式（优先级从高到低）：
        # 1) imageFollowSegments = true：每帧跟随当前涡丝范围（推荐做动画时使用）
        # 2) 显式配置 imageViewStart / imageViewEnd：固定视野
        # 3) imageAutoFitRange = true：按当前段中心包围盒自适应
        # 4) 默认：基于 domain 范围外扩
        follow = bool(self.ss.cfg.get_cfg("imageFollowSegments", False))
        if follow:
            cmin = np.min(center, axis=0).astype(np.float32)
            cmax = np.max(center, axis=0).astype(np.float32)
            c0 = 0.5 * (cmin + cmax)

            follow_span_cfg = self.ss.cfg.get_cfg("imageFollowSpan", None)
            if follow_span_cfg is not None:
                follow_span = np.array(follow_span_cfg, dtype=np.float32)
            else:
                # 视野大小：至少覆盖当前段范围，再乘一个系数
                base_span = np.maximum(cmax - cmin, 1e-6)
                extra = float(self.ss.cfg.get_cfg("imageFollowExtraScale", 2.0))
                follow_span = base_span * extra

                # 允许设置一个最小视野尺寸，避免段很小时画面过近
                min_span_cfg = self.ss.cfg.get_cfg("imageFollowMinSpan", None)
                if min_span_cfg is not None:
                    min_span = np.array(min_span_cfg, dtype=np.float32)
                    follow_span = np.maximum(follow_span, min_span)

            # 额外 margin（比例），让视野更大一点
            margin_ratio = float(self.ss.cfg.get_cfg("imageViewMarginRatio", 0.3))
            follow_span = follow_span * (1.0 + margin_ratio)

            view_start = c0 - 0.5 * follow_span
            view_end = c0 + 0.5 * follow_span
        else:
            view_start_cfg = self.ss.cfg.get_cfg("imageViewStart", None)
            view_end_cfg = self.ss.cfg.get_cfg("imageViewEnd", None)
            if view_start_cfg is not None and view_end_cfg is not None:
                view_start = np.array(view_start_cfg, dtype=np.float32)
                view_end = np.array(view_end_cfg, dtype=np.float32)
            else:
                margin_ratio = float(self.ss.cfg.get_cfg("imageViewMarginRatio", 0.5))
                margin = span * margin_ratio
                auto_fit = bool(self.ss.cfg.get_cfg("imageAutoFitRange", False))
                if auto_fit:
                    cmin = np.min(center, axis=0).astype(np.float32)
                    cmax = np.max(center, axis=0).astype(np.float32)
                    view_start = cmin - margin
                    view_end = cmax + margin
                else:
                    view_start = domain_start - margin
                    view_end = domain_end + margin

        ax.set_xlim(float(view_start[0]), float(view_end[0]))
        ax.set_ylim(float(view_start[1]), float(view_end[1]))
        ax.set_zlim(float(view_start[2]), float(view_end[2]))
        ax.set_box_aspect(
            (
                float(max(view_end[0] - view_start[0], 1e-6)),
                float(max(view_end[1] - view_start[1], 1e-6)),
                float(max(view_end[2] - view_start[2], 1e-6)),
            )
        )
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path, bbox_inches="tight", pad_inches=0, transparent=True, dpi=400)
        plt.close(fig)
