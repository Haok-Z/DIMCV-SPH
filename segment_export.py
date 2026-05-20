from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class SegmentExporter:
    def __init__(self, segment_system):
        self.ss = segment_system

    @staticmethod
    def _ply_vec3(p) -> np.ndarray:
        """仿真坐标可能是 2D (x,y) 或 3D；PLY 顶点统一为 (x,y,z)。"""
        arr = np.asarray(p, dtype=np.float64).ravel()
        if arr.size >= 3:
            return np.array([arr[0], arr[1], arr[2]], dtype=np.float64)
        if arr.size >= 2:
            return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
        if arr.size >= 1:
            return np.array([arr[0], 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _apply_ply_axis_convention(p, convention: str):
        """
        将仿真坐标写到 PLY 顶点。

        SegmentConfiguration.exportPLYAxisConvention:
        - y_up / same / sim：不变换（默认）。若仿真里竖直已是 +Z、与 DCC 一致，请用此项。
        - xy / 2d / plane_xy：2D 平面流 (x,y)，补 z=0（与 y_up 对 2D 输入等价，显式用于 2D 场景）。
        - z_up / houdini：(x,y,z)_sim -> (x, z, y)。用于「仿真竖直为 +Y、希望导出到 Z 朝上的 DCC」
          的旧约定；若仿真竖直为 +Z，勿用此项，否则会把高度与宽度轴对调。
        """
        c = (convention or "y_up").lower().strip()
        v = SegmentExporter._ply_vec3(p)
        if c in ("y_up", "same", "sim", "", "xy", "2d", "plane_xy", "plane"):
            return v
        if c in ("z_up", "houdini"):
            x, y, z = float(v[0]), float(v[1]), float(v[2])
            return np.array([x, z, y], dtype=np.float64)
        raise ValueError(
            f"Unknown exportPLYAxisConvention={convention!r}; "
            "use y_up, xy (2D), or z_up (houdini)."
        )

    @staticmethod
    def _rigid_block_lo_hi(blk):
        """与 segment_boundary._apply_block_transform 一致的 AABB。"""
        start = np.array(blk.get("start", [0.0, 0.0, 0.0]), dtype=np.float64)
        end = np.array(blk.get("end", [0.0, 0.0, 0.0]), dtype=np.float64)
        scale = np.array(blk.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        tr = np.array(blk.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        lo = start + tr
        hi = start + tr + (end - start) * scale
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        return lo2.astype(np.float64), hi2.astype(np.float64)

    @staticmethod
    def _aabb_wire_edges(lo, hi):
        """12 条边的 (p0,p1)。"""
        x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
        x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
        c = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
        c = [np.array(p, dtype=np.float64) for p in c]
        e = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        return [(c[a], c[b]) for a, b in e]

    @staticmethod
    def _cylinder_world(cyl):
        tr = np.array(cyl.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        center = np.array(cyl.get("center", [0.0, 0.0, 0.0]), dtype=np.float64) + tr
        r = float(cyl.get("radius", 0.1))
        hh = float(cyl.get("halfHeight", 0.5))
        axis = str(cyl.get("axis", "z")).lower().strip()
        return center.astype(np.float64), axis, r, hh

    @staticmethod
    def _cylinder_wire_edges(cyl, n_theta=40, n_meridian=8):
        """圆柱侧棱线：上下圆 + 若干母线。"""
        center, axis, r, hh = SegmentExporter._cylinder_world(cyl)
        th = np.linspace(0.0, 2.0 * np.pi, int(n_theta) + 1)
        out = []
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        ax = axis
        if ax == "z":
            z0, z1 = cz - hh, cz + hh
            for i in range(len(th) - 1):
                c0, s0 = np.cos(th[i]) * r, np.sin(th[i]) * r
                c1, s1 = np.cos(th[i + 1]) * r, np.sin(th[i + 1]) * r
                p0 = np.array([cx + c0, cy + s0, z0], dtype=np.float64)
                p1 = np.array([cx + c1, cy + s1, z0], dtype=np.float64)
                out.append((p0, p1))
                q0 = np.array([cx + c0, cy + s0, z1], dtype=np.float64)
                q1 = np.array([cx + c1, cy + s1, z1], dtype=np.float64)
                out.append((q0, q1))
            step = max(1, len(th) // max(2, int(n_meridian)))
            for k in range(0, len(th), step):
                c0, s0 = np.cos(th[k]) * r, np.sin(th[k]) * r
                p0 = np.array([cx + c0, cy + s0, z0], dtype=np.float64)
                p1 = np.array([cx + c0, cy + s0, z1], dtype=np.float64)
                out.append((p0, p1))
        elif ax == "y":
            y0, y1 = cy - hh, cy + hh
            for i in range(len(th) - 1):
                c0, s0 = np.cos(th[i]) * r, np.sin(th[i]) * r
                c1, s1 = np.cos(th[i + 1]) * r, np.sin(th[i + 1]) * r
                p0 = np.array([cx + c0, cy + y0, cz + s0], dtype=np.float64)
                p1 = np.array([cx + c1, cy + y0, cz + s1], dtype=np.float64)
                out.append((p0, p1))
                q0 = np.array([cx + c0, cy + y1, cz + s0], dtype=np.float64)
                q1 = np.array([cx + c1, cy + y1, cz + s1], dtype=np.float64)
                out.append((q0, q1))
            step = max(1, len(th) // max(2, int(n_meridian)))
            for k in range(0, len(th), step):
                c0, s0 = np.cos(th[k]) * r, np.sin(th[k]) * r
                p0 = np.array([cx + c0, cy + y0, cz + s0], dtype=np.float64)
                p1 = np.array([cx + c0, cy + y1, cz + s0], dtype=np.float64)
                out.append((p0, p1))
        else:
            x0, x1 = cx - hh, cx + hh
            for i in range(len(th) - 1):
                c0, s0 = np.cos(th[i]) * r, np.sin(th[i]) * r
                c1, s1 = np.cos(th[i + 1]) * r, np.sin(th[i + 1]) * r
                p0 = np.array([x0, cy + c0, cz + s0], dtype=np.float64)
                p1 = np.array([x0, cy + c1, cz + s1], dtype=np.float64)
                out.append((p0, p1))
                q0 = np.array([x1, cy + c0, cz + s0], dtype=np.float64)
                q1 = np.array([x1, cy + c1, cz + s1], dtype=np.float64)
                out.append((q0, q1))
            step = max(1, len(th) // max(2, int(n_meridian)))
            for k in range(0, len(th), step):
                c0, s0 = np.cos(th[k]) * r, np.sin(th[k]) * r
                p0 = np.array([x0, cy + c0, cz + s0], dtype=np.float64)
                p1 = np.array([x1, cy + c0, cz + s0], dtype=np.float64)
                out.append((p0, p1))
        return out

    @staticmethod
    def _solid_scene_bounds(blocks, cyls):
        """用于无段时估计视野。"""
        pts = []
        for blk in blocks:
            lo, hi = SegmentExporter._rigid_block_lo_hi(blk)
            pts.append(lo)
            pts.append(hi)
        for cyl in cyls:
            c, axis, r, hh = SegmentExporter._cylinder_world(cyl)
            if axis == "z":
                lo = c + np.array([-r, -r, -hh], dtype=np.float64)
                hi = c + np.array([r, r, hh], dtype=np.float64)
            elif axis == "y":
                lo = c + np.array([-r, -hh, -r], dtype=np.float64)
                hi = c + np.array([r, hh, r], dtype=np.float64)
            else:
                lo = c + np.array([-hh, -r, -r], dtype=np.float64)
                hi = c + np.array([hh, r, r], dtype=np.float64)
            pts.append(lo)
            pts.append(hi)
        if not pts:
            return None, None
        arr = np.stack(pts, axis=0)
        return np.min(arr, axis=0), np.max(arr, axis=0)

    @staticmethod
    def _solid_ply_samples(blocks, cyls, n_theta_cyl=24):
        """固体线框采样点（去重边端点），用于 PLY。"""
        seen = set()
        pts = []

        def add_pt(p):
            key = (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5))
            if key in seen:
                return
            seen.add(key)
            pts.append(np.array(p, dtype=np.float64))

        for blk in blocks:
            lo, hi = SegmentExporter._rigid_block_lo_hi(blk)
            for p0, p1 in SegmentExporter._aabb_wire_edges(lo, hi):
                add_pt(p0)
                add_pt(p1)
        th = np.linspace(0.0, 2.0 * np.pi, int(n_theta_cyl) + 1)
        for cyl in cyls:
            center, axis, r, hh = SegmentExporter._cylinder_world(cyl)
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            if axis == "z":
                z0, z1 = cz - hh, cz + hh
                for i in range(len(th)):
                    c0, s0 = np.cos(th[i]) * r, np.sin(th[i]) * r
                    add_pt([cx + c0, cy + s0, z0])
                    add_pt([cx + c0, cy + s0, z1])
            elif axis == "y":
                y0, y1 = cy - hh, cy + hh
                for i in range(len(th)):
                    c0, s0 = np.cos(th[i]) * r, np.sin(th[i]) * r
                    add_pt([cx + c0, y0, cz + s0])
                    add_pt([cx + c0, y1, cz + s0])
            else:
                x0, x1 = cx - hh, cx + hh
                for i in range(len(th)):
                    c0, s0 = np.cos(th[i]) * r, np.sin(th[i]) * r
                    add_pt([x0, cy + c0, cz + s0])
                    add_pt([x1, cy + c0, cz + s0])
        return pts

    @staticmethod
    def _to_rgb01(v, default_rgb):
        arr = np.array(v if v is not None else default_rgb, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            arr = np.array(default_rgb, dtype=np.float32)
        rgb = arr[:3]
        # 兼容 0-255 与 0-1 两种输入
        if float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)

    def export_segments_ply(self, frame_id: int, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"segments_{frame_id:04}.ply"

        include_solids = bool(self.ss.cfg.get_cfg("exportPLYIncludeSolids", False))
        blocks = self.ss.cfg.get_obstacles() if hasattr(self.ss.cfg, "get_obstacles") else []
        cyls = self.ss.cfg.get_cylinders() if hasattr(self.ss.cfg, "get_cylinders") else []

        n = int(self.ss.segment_num[None])
        conv = str(self.ss.cfg.get_cfg("exportPLYAxisConvention", "y_up"))

        pts = []
        if n > 0:
            x_minus = self.ss.x_minus.to_numpy()[:n]
            x_plus = self.ss.x_plus.to_numpy()[:n]
            gamma = self.ss.gamma.to_numpy()[:n]
            active = self.ss.active.to_numpy()[:n].astype(bool)
            seg_type = self.ss.seg_type.to_numpy()[:n]
            for i in range(n):
                if not active[i]:
                    continue
                pm = self._apply_ply_axis_convention(x_minus[i], conv)
                pp = self._apply_ply_axis_convention(x_plus[i], conv)
                if include_solids:
                    pts.append((pm, gamma[i], seg_type[i], 0, 0))
                    pts.append((pp, gamma[i], seg_type[i], 1, 0))
                else:
                    pts.append((pm, gamma[i], seg_type[i], 0))
                    pts.append((pp, gamma[i], seg_type[i], 1))

        if include_solids and (len(blocks) > 0 or len(cyls) > 0):
            n_cyl_th = int(self.ss.cfg.get_cfg("exportPLYSolidCylinderTheta", 24))
            solid_pts = self._solid_ply_samples(blocks, cyls, n_theta_cyl=max(8, n_cyl_th))
            for sp in solid_pts:
                p = self._apply_ply_axis_convention(sp, conv)
                pts.append((p, 0.0, -1, 0, 1))

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
            if include_solids:
                f.write("property uchar vertex_kind\n")
            f.write("end_header\n")
            for row in pts:
                if include_solids:
                    p, g, t, e, vk = row
                    f.write(
                        f"{float(p[0]):.7f} {float(p[1]):.7f} {float(p[2]):.7f} "
                        f"{g:.7f} {int(t)} {int(e)} {int(vk)}\n"
                    )
                else:
                    p, g, t, e = row
                    f.write(
                        f"{float(p[0]):.7f} {float(p[1]):.7f} {float(p[2]):.7f} "
                        f"{g:.7f} {int(t)} {int(e)}\n"
                    )

    def export_segments_png(self, frame_id: int, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"segments_{frame_id:04}.png"

        show_solids = bool(self.ss.cfg.get_cfg("imageShowSolids", True))
        blocks = self.ss.cfg.get_obstacles() if hasattr(self.ss.cfg, "get_obstacles") else []
        cyls = self.ss.cfg.get_cylinders() if hasattr(self.ss.cfg, "get_cylinders") else []
        has_solids = show_solids and (len(blocks) > 0 or len(cyls) > 0)

        n = int(self.ss.segment_num[None])
        x_minus = np.zeros((0, 3), dtype=np.float64)
        x_plus = np.zeros((0, 3), dtype=np.float64)
        gamma = np.zeros((0,), dtype=np.float64)
        seg_type = np.zeros((0,), dtype=np.int32)
        has_segments = False
        if n > 0:
            x_minus = self.ss.x_minus.to_numpy()[:n].astype(np.float64)
            x_plus = self.ss.x_plus.to_numpy()[:n].astype(np.float64)
            gamma = self.ss.gamma.to_numpy()[:n].astype(np.float64)
            active = self.ss.active.to_numpy()[:n].astype(bool)
            seg_type = self.ss.seg_type.to_numpy()[:n].astype(np.int32)
            has_segments = bool(np.any(active))
            if has_segments:
                x_minus = x_minus[active]
                x_plus = x_plus[active]
                gamma = gamma[active]
                seg_type = seg_type[active]

        if not has_segments and not has_solids:
            return

        center = 0.5 * (x_minus + x_plus) if x_minus.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
        slo, shi = self._solid_scene_bounds(blocks, cyls) if has_solids else (None, None)

        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=-60)

        if has_solids:
            col = self._to_rgb01(
                self.ss.cfg.get_cfg("imageSolidEdgeColor", [85, 78, 72]),
                [0.35, 0.31, 0.28],
            )
            lw = float(self.ss.cfg.get_cfg("imageSolidLineWidth", 1.05))
            sa = float(self.ss.cfg.get_cfg("imageSolidLineAlpha", 0.92))
            n_th = int(self.ss.cfg.get_cfg("imageSolidCylinderTheta", 44))
            n_mer = int(self.ss.cfg.get_cfg("imageSolidCylinderMeridians", 8))
            for blk in blocks:
                lo, hi = self._rigid_block_lo_hi(blk)
                for p0, p1 in self._aabb_wire_edges(lo, hi):
                    ax.plot(
                        [p0[0], p1[0]],
                        [p0[1], p1[1]],
                        [p0[2], p1[2]],
                        color=col,
                        linewidth=lw,
                        alpha=sa,
                    )
            for cyl in cyls:
                for p0, p1 in self._cylinder_wire_edges(cyl, n_theta=n_th, n_meridian=n_mer):
                    ax.plot(
                        [p0[0], p1[0]],
                        [p0[1], p1[1]],
                        [p0[2], p1[2]],
                        color=col,
                        linewidth=lw,
                        alpha=sa,
                    )

        vmin = self.ss.cfg.get_cfg("imageGammaVmin", None)
        vmax = self.ss.cfg.get_cfg("imageGammaVmax", None)
        if x_minus.shape[0] > 0:
            if vmin is None:
                vmin = float(np.min(gamma))
            else:
                vmin = float(vmin)
            if vmax is None:
                vmax = float(np.max(gamma))
            else:
                vmax = float(vmax)
        else:
            if vmin is None:
                vmin = -1.0
            else:
                vmin = float(vmin)
            if vmax is None:
                vmax = 1.0
            else:
                vmax = float(vmax)
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-8
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "coolwarm"

        line_width = float(self.ss.cfg.get_cfg("imageLineWidth", 0.8))
        line_alpha = float(self.ss.cfg.get_cfg("imageLineAlpha", 0.9))
        cm = plt.get_cmap(cmap)
        use_type_colors = bool(self.ss.cfg.get_cfg("imageUseSegmentTypeColors", False))
        ring1_type = int(self.ss.cfg.get_cfg("leapfrogRing1SegmentTypeId", 101))
        ring2_type = int(self.ss.cfg.get_cfg("leapfrogRing2SegmentTypeId", 102))
        ring1_color = self._to_rgb01(
            self.ss.cfg.get_cfg("imageRing1Color", [255, 80, 80]),
            [1.0, 0.31, 0.31],
        )
        ring2_color = self._to_rgb01(
            self.ss.cfg.get_cfg("imageRing2Color", [80, 170, 255]),
            [0.31, 0.67, 1.0],
        )
        other_color = self._to_rgb01(
            self.ss.cfg.get_cfg("imageOtherSegmentColor", [210, 210, 210]),
            [0.82, 0.82, 0.82],
        )
        for i in range(x_minus.shape[0]):
            if use_type_colors:
                st = int(seg_type[i])
                if st == ring1_type:
                    col = ring1_color
                elif st == ring2_type:
                    col = ring2_color
                else:
                    col = other_color
            else:
                col = cm(norm(float(gamma[i])))
            ax.plot(
                [float(x_minus[i, 0]), float(x_plus[i, 0])],
                [float(x_minus[i, 1]), float(x_plus[i, 1])],
                [float(x_minus[i, 2]), float(x_plus[i, 2])],
                color=col,
                linewidth=line_width,
                alpha=line_alpha,
            )

        if center.shape[0] > 0:
            if use_type_colors:
                point_colors = np.zeros((center.shape[0], 3), dtype=np.float32)
                for i in range(center.shape[0]):
                    st = int(seg_type[i])
                    if st == ring1_type:
                        point_colors[i] = ring1_color
                    elif st == ring2_type:
                        point_colors[i] = ring2_color
                    else:
                        point_colors[i] = other_color
                ax.scatter(
                    center[:, 0],
                    center[:, 1],
                    center[:, 2],
                    c=point_colors,
                    s=float(self.ss.cfg.get_cfg("imagePointSize", 1.0)),
                    edgecolors="none",
                    alpha=float(self.ss.cfg.get_cfg("imagePointAlpha", 0.6)),
                )
            else:
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

        follow = bool(self.ss.cfg.get_cfg("imageFollowSegments", False))
        if follow:
            if center.shape[0] > 0:
                cmin = np.min(center, axis=0).astype(np.float32)
                cmax = np.max(center, axis=0).astype(np.float32)
            else:
                cmin = domain_start.copy()
                cmax = domain_end.copy()
            if slo is not None and shi is not None:
                cmin = np.minimum(cmin, slo.astype(np.float32))
                cmax = np.maximum(cmax, shi.astype(np.float32))
            c0 = 0.5 * (cmin + cmax)

            follow_span_cfg = self.ss.cfg.get_cfg("imageFollowSpan", None)
            if follow_span_cfg is not None:
                follow_span = np.array(follow_span_cfg, dtype=np.float32)
            else:
                base_span = np.maximum(cmax - cmin, 1e-6)
                extra = float(self.ss.cfg.get_cfg("imageFollowExtraScale", 2.0))
                follow_span = base_span * extra
                min_span_cfg = self.ss.cfg.get_cfg("imageFollowMinSpan", None)
                if min_span_cfg is not None:
                    min_span = np.array(min_span_cfg, dtype=np.float32)
                    follow_span = np.maximum(follow_span, min_span)

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
                if auto_fit and center.shape[0] > 0:
                    cmin = np.min(center, axis=0).astype(np.float32)
                    cmax = np.max(center, axis=0).astype(np.float32)
                    if slo is not None and shi is not None:
                        cmin = np.minimum(cmin, slo.astype(np.float32))
                        cmax = np.maximum(cmax, shi.astype(np.float32))
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
