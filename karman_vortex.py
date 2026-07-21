import taichi as ti
from dimcv_sph import DIMCVSPHSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from segment_export import SegmentExporter


class KarmanVortexSolver(DIMCVSPHSolver):

    def __init__(self, particle_system):
        super().__init__(particle_system)
        cfg_dict = getattr(self.ps.cfg, "config", {}).get("Configuration", {})

        def cfg_get(name, default=None):
            return cfg_dict.get(name, default)

        # Mesh obstacles can share the legacy motion schedule without being
        # clipped to the circular proxy used by the Karman cylinder scenes.
        cc = cfg_get("movingRigidStartCenter", cfg_get("cylinderCenter", None))
        if cc is not None:
            self.circle_pos = np.array(cc, dtype=np.float64)
        elif self.ps.dim == 2:
            self.circle_pos = np.array([0.65, 0.5], dtype=np.float64)
        else:
            self.circle_pos = np.array([0.65, 0.5, 0.5], dtype=np.float64)
        _cr = cfg_get("cylinderRadius", None)
        self.circle_radius = float(_cr if _cr is not None else 0.1)
        self.circle_vis = ti.Vector.field(self.ps.dim, dtype=float, shape=1)
        if self.ps.dim == 2:
            self.circle_vis[0] = ti.Vector(
                [self.circle_pos[0] * 0.25, self.circle_pos[1]])
        else:
            self.circle_vis[0] = ti.Vector([
                self.circle_pos[0] * 0.25, self.circle_pos[1], self.circle_pos[2]
            ])
        self._cylinder_obstacle_enabled = bool(
            cfg_get("cylinderObstacleEnabled", True)
        )
        self._moving_rigid_object_full_body = bool(
            cfg_get("movingRigidObjectFullBody", False)
        )
        if self._cylinder_obstacle_enabled:
            self.init_cylinder()
        self._emit_stop_step = cfg_get("emitStopStep", None)
        self._emit_stop_time = cfg_get("emitStopTime", None)
        self._emit_stop_fluid_particle_num = cfg_get("emitStopFluidParticleNum", None)
        self._cylinder_motion_mode = str(
            cfg_get("cylinderMotionMode", "oscillation") or "oscillation"
        ).lower().strip()
        self._cylinder_oscillation_enabled = bool(
            cfg_get("cylinderOscillationEnabled", False)
        )
        self._cylinder_oscillation_object_id = int(
            cfg_get("cylinderOscillationObjectId", 2)
        )
        self._cylinder_oscillation_axis = np.asarray(
            cfg_get("cylinderOscillationAxis", [1.0, 0.0]), dtype=np.float64
        ).reshape(-1)
        if self._cylinder_oscillation_axis.size < self.ps.dim:
            self._cylinder_oscillation_axis = np.pad(
                self._cylinder_oscillation_axis,
                (0, self.ps.dim - self._cylinder_oscillation_axis.size),
                constant_values=0.0,
            )
        self._cylinder_oscillation_axis = self._cylinder_oscillation_axis[: self.ps.dim]
        axis_norm = np.linalg.norm(self._cylinder_oscillation_axis)
        if axis_norm > 1e-12:
            self._cylinder_oscillation_axis = self._cylinder_oscillation_axis / axis_norm
        self._cylinder_oscillation_amplitude = float(
            cfg_get("cylinderOscillationAmplitude", 0.0)
        )
        self._cylinder_oscillation_period = max(
            1e-12, float(cfg_get("cylinderOscillationPeriod", 1.0))
        )
        self._cylinder_oscillation_start_time = float(
            cfg_get("cylinderOscillationStartTime", 0.0)
        )
        self._cylinder_linear_velocity = np.asarray(
            cfg_get("cylinderLinearVelocity", [0.0, 0.0]), dtype=np.float64
        ).reshape(-1)
        if self._cylinder_linear_velocity.size < self.ps.dim:
            self._cylinder_linear_velocity = np.pad(
                self._cylinder_linear_velocity,
                (0, self.ps.dim - self._cylinder_linear_velocity.size),
                constant_values=0.0,
            )
        self._cylinder_linear_velocity = self._cylinder_linear_velocity[: self.ps.dim]

        # When true (linear mode only), clamp the cylinder translation so its
        # center stops at the position symmetric to circle_pos about the domain
        # center:  stop = domain_start + domain_end - circle_pos.
        self._cylinder_linear_stop_at_symmetric = bool(
            cfg_get("cylinderLinearStopAtSymmetric", False)
        )
        # Current applied offset (after clamping); other subsystems (e.g. the
        # segment hybrid solver) read this to sync obstacle geometry.
        self._cylinder_current_offset = np.zeros(self.ps.dim, dtype=np.float64)

    @ti.kernel
    def init_circle(self):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == 2:
                if (self.ps.x[p] - self.circle_pos).norm() > self.circle_radius:
                    for d in ti.static(range(self.ps.dim)):
                        self.ps.x[p][d] = 0.0
                    self.ps.is_active[p] = 0

    @ti.kernel
    def init_cylinder(self):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == 2:
                dx = self.ps.x[p][0] - self.circle_pos[0]
                dy = self.ps.x[p][1] - self.circle_pos[1]
                # Cylinder axis along z, so radius in x-y plane
                if ti.sqrt(dx * dx + dy * dy) > self.circle_radius:
                    for d in ti.static(range(self.ps.dim)):
                        self.ps.x[p][d] = 0.0
                    self.ps.is_active[p] = 0

    @ti.kernel
    def _translate_object_from_rest_kernel(
        self, object_id: ti.i32, off0: float, off1: float, off2: float, vel0: float, vel1: float, vel2: float
    ):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == object_id:
                dx = self.ps.x_0[p][0] - self.circle_pos[0]
                dy = self.ps.x_0[p][1] - self.circle_pos[1]
                inside_circle = ti.sqrt(dx * dx + dy * dy) <= self.circle_radius
                if inside_circle:
                    self.ps.is_active[p] = 1
                    self.ps.x[p] = self.ps.x_0[p] + ti.Vector([off0, off1])
                    self.ps.v[p] = ti.Vector([vel0, vel1])
                else:
                    self.ps.x[p] = self.ps.x_0[p] + ti.Vector([off0, off1, off2])
                    self.ps.v[p] = ti.Vector([vel0, vel1, vel2])

    @ti.kernel
    def _translate_object_from_rest_kernel(
        self, object_id: ti.i32, off0: float, off1: float, off2: float, vel0: float, vel1: float, vel2: float
    ):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == object_id:
                dx = self.ps.x_0[p][0] - self.circle_pos[0]
                dy = self.ps.x_0[p][1] - self.circle_pos[1]
                inside_circle = ti.sqrt(dx * dx + dy * dy) <= self.circle_radius
                if inside_circle:
                    self.ps.is_active[p] = 1
                    if ti.static(self.ps.dim == 2):
                        self.ps.x[p] = self.ps.x_0[p] + ti.Vector([off0, off1])
                        self.ps.v[p] = ti.Vector([vel0, vel1])
                    else:
                        self.ps.x[p] = self.ps.x_0[p] + ti.Vector([off0, off1, off2])
                        self.ps.v[p] = ti.Vector([vel0, vel1, vel2])
                else:
                    self.ps.is_active[p] = 0
                    for d in ti.static(range(self.ps.dim)):
                        self.ps.x[p][d] = 0.0
                        self.ps.v[p][d] = 0.0

    @ti.kernel
    def _translate_whole_object_from_rest_kernel(
        self, object_id: ti.i32, off0: float, off1: float, off2: float,
        vel0: float, vel1: float, vel2: float,
    ):
        """Move every particle of a mesh/rigid object without proxy clipping."""
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == object_id:
                self.ps.is_active[p] = 1
                if ti.static(self.ps.dim == 2):
                    self.ps.x[p] = self.ps.x_0[p] + ti.Vector([off0, off1])
                    self.ps.v[p] = ti.Vector([vel0, vel1])
                else:
                    self.ps.x[p] = self.ps.x_0[p] + ti.Vector([off0, off1, off2])
                    self.ps.v[p] = ti.Vector([vel0, vel1, vel2])

    def _should_emit_this_step(self) -> bool:
        if self._emit_stop_step is not None and int(self.cnt) >= int(self._emit_stop_step):
            return False
        if self._emit_stop_fluid_particle_num is not None:
            if int(self.ps.fluid_particle_num[None]) >= int(self._emit_stop_fluid_particle_num):
                return False
        if self._emit_stop_time is not None:
            t = float(self.cnt) * float(self.dt[None])
            if t >= float(self._emit_stop_time):
                return False
        return True

    def _update_oscillating_cylinder(self):
        if not self._cylinder_oscillation_enabled:
            self._cylinder_current_offset[: self.ps.dim] = 0.0
            return
        t = float(self.cnt) * float(self.dt[None])
        if t < self._cylinder_oscillation_start_time:
            self._cylinder_current_offset[: self.ps.dim] = 0.0
            return
        tau = t - self._cylinder_oscillation_start_time
        if self._cylinder_motion_mode in ("linear", "translate", "translation"):
            offset = tau * self._cylinder_linear_velocity
            vel = self._cylinder_linear_velocity.copy()
            if self._cylinder_linear_stop_at_symmetric:
                # Symmetric stop position about the domain center:
                #   stop = domain_start + domain_end - circle_pos
                #   max_offset = stop - circle_pos
                #             = (domain_start + domain_end) - 2 * circle_pos
                # Clamp the offset so the cylinder center does not travel past
                # the symmetric position; zero the velocity once it arrives.
                max_offset = (
                    self.ps.domain_start + self.ps.domain_end
                ) - 2.0 * self.circle_pos
                for d in range(self.ps.dim):
                    if abs(max_offset[d]) > 1e-12:
                        if max_offset[d] > 0.0 and offset[d] >= max_offset[d]:
                            offset[d] = max_offset[d]
                            vel[d] = 0.0
                        elif max_offset[d] < 0.0 and offset[d] <= max_offset[d]:
                            offset[d] = max_offset[d]
                            vel[d] = 0.0
        else:
            omega = 2.0 * np.pi / self._cylinder_oscillation_period
            amp = self._cylinder_oscillation_amplitude
            offset = amp * np.sin(omega * tau) * self._cylinder_oscillation_axis
            vel = amp * omega * np.cos(omega * tau) * self._cylinder_oscillation_axis
        self._cylinder_current_offset[: self.ps.dim] = offset[: self.ps.dim]
        off3 = np.zeros((3,), dtype=np.float64)
        vel3 = np.zeros((3,), dtype=np.float64)
        off3[: self.ps.dim] = offset[: self.ps.dim]
        vel3[: self.ps.dim] = vel[: self.ps.dim]
        translate = (
            self._translate_whole_object_from_rest_kernel
            if self._moving_rigid_object_full_body
            else self._translate_object_from_rest_kernel
        )
        translate(
            self._cylinder_oscillation_object_id,
            float(off3[0]), float(off3[1]), float(off3[2]),
            float(vel3[0]), float(vel3[1]), float(vel3[2]),
        )

    def export_png(self, cnt, image_path):
        N = self.ps.particle_num[None]
        material = self.ps.material.to_numpy()[:N]
        obj_id = self.ps.object_id.to_numpy()[:N]
        fluid_mask = (material == self.ps.material_fluid)
        # Render the channel walls/cylinder unless the scene hides an object id.
        invisible_objects = self.ps.cfg.get_cfg("invisibleObjects") or []
        invisible_ids = np.asarray(invisible_objects, dtype=obj_id.dtype)
        solid_mask = (obj_id == 1) | (obj_id == 2)
        if invisible_ids.size > 0:
            solid_mask &= ~np.isin(obj_id, invisible_ids)

        x = self.x_temp.to_numpy()[:N]
        vort_np = self.ps.vorticity_vis.to_numpy()[:N]
        vort = vort_np[:, 2] if vort_np.shape[1] > 2 else vort_np[:, 0]

        fluid_x = x[fluid_mask]
        fluid_vort = vort[fluid_mask]
        solid_x = x[solid_mask]

        ds = self.ps.domain_start
        de = self.ps.domain_end
        _vmin = self.ps.cfg.get_cfg("imageVorticityVmin")
        _vmax = self.ps.cfg.get_cfg("imageVorticityVmax")
        _auto_scale = self.ps.cfg.get_cfg("imageVorticityAutoScale")
        if _auto_scale and fluid_vort.size > 0:
            vmin = float(np.percentile(fluid_vort, 2))
            vmax = float(np.percentile(fluid_vort, 98))
            if vmax <= vmin + 1e-12:
                vmax = vmin + 1.0
        else:
            vmin = float(_vmin if _vmin is not None else -40)
            vmax = float(_vmax if _vmax is not None else 40)
        vnorm = Normalize(vmin=vmin, vmax=vmax)

        _pt = self.ps.cfg.get_cfg("imageFluidPointSize")
        _al = self.ps.cfg.get_cfg("imageFluidAlpha")
        if self.ps.dim == 2:
            pt_size = float(_pt if _pt is not None else 2.5)
            pt_alpha = float(_al if _al is not None else 1.0)
        else:
            pt_size = float(_pt if _pt is not None else 0.2)
            pt_alpha = float(_al if _al is not None else 0.6)

        if self.ps.dim == 2:
            fig, ax = plt.subplots(figsize=(10, 2.5), dpi=200)
            ax.scatter(
                fluid_x[:, 0], fluid_x[:, 1], c=fluid_vort, cmap="coolwarm",
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
                image_path / f"vorticity_{cnt:04}.png",
                bbox_inches="tight", pad_inches=0, transparent=True, dpi=400,
            )
            plt.close("all")
            return

        domain_extent = de - ds
        fig_width = 12.0
        fig_height = max(
            2.0,
            min(4.0, fig_width * max(float(domain_extent[1]), float(domain_extent[2])) /
                max(float(domain_extent[0]), 1e-12)),
        )
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=-60)
        sc = ax.scatter(fluid_x[:, 0],
                        fluid_x[:, 1],
                        fluid_x[:, 2],
                        c=fluid_vort,
                        cmap='coolwarm',
                        s=pt_size,
                        norm=vnorm,
                        edgecolors='none',
                        alpha=pt_alpha)
        ax.scatter(solid_x[:, 0],
                   solid_x[:, 1],
                   solid_x[:, 2],
                   color="#00C853",
                   s=0.5,
                   edgecolors='none')

        ax.set_xlim(float(ds[0]), float(de[0]))
        ax.set_ylim(float(ds[1]), float(de[1]))
        ax.set_zlim(float(ds[2]), float(de[2]))
        ax.set_box_aspect(tuple(float(v) for v in domain_extent))
        ax.set_axis_off()  # 如果想看坐标轴可以注释掉这一行
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        '''fluid_x_x = fluid_x[:, 0]
        fluid_x_y = fluid_x[:, 1]
        solid_x_x = solid_x[:, 0]
        solid_x_y = solid_x[:, 1]

        filtered = (fluid_x_x <= 4.0)
        fluid_x_x = fluid_x_x[filtered]
        fluid_x_y = fluid_x_y[filtered]
        fluid_vort = fluid_vort[filtered]

        norm = Normalize(vmin=-40, vmax=40)
        cmap = 'coolwarm'
        solid_color = "#00C853"

        plt.figure(figsize=(5.12, 1.28), dpi=400)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.scatter(fluid_x_x,
                    fluid_x_y,
                    edgecolors='none',
                    c=fluid_vort,
                    cmap=cmap,
                    s=0.5,
                    norm=norm)
        plt.scatter(solid_x_x,
                    solid_x_y,
                    edgecolors='none',
                    s=1.0,
                    color=solid_color)

        plt.xlim(0, 4)
        plt.ylim(0, 1)
        '''

        plt.savefig(image_path / f"vorticity_{cnt:04}.png",
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True,
                    dpi=400)
        plt.cla()
        plt.close('all')

    def export_ply(self, cnt, ply_path):
        """
        Export current particles to an ASCII PLY file with:
        - position (x, y, z)
        - vorticity (vort_x, vort_y, vort_z)
        - color (r, g, b)
        - object_id
        - particle_type:
            0 = fluid block
            1 = fluid emitter
            2 = rigid block
            3 = rigid body
            4 = other / unknown

        ``Configuration.exportPLYAxisConvention`` matches ``segment_export``:
        ``y_up`` / ``same`` / ``sim`` (no change), or ``z_up`` / ``houdini``
        maps (x,y,z) and vorticity as (x,z,y).
        """
        N = self.ps.particle_num[None]
        if N == 0:
            return

        x = self.x_temp.to_numpy()[:N]
        vorticity = self.ps.vorticity.to_numpy()[:N]
        color = self.ps.color.to_numpy()[:N]
        obj_id = self.ps.object_id.to_numpy()[:N]
        material = self.ps.material.to_numpy()[:N]
        is_active = self.ps.is_active.to_numpy()[:N].astype(bool)

        # Only keep active particles
        x = x[is_active]
        vorticity = vorticity[is_active]
        color = color[is_active]
        obj_id = obj_id[is_active]
        material = material[is_active]

        if x.shape[0] == 0:
            return

        num_vertices = x.shape[0]

        # Build particle_type code per particle
        particle_type = np.full(num_vertices, 4, dtype=np.uint8)  # default: other

        # Helper to safely create masks when there may be no ids
        def mask_from_ids(ids_set):
            ids = list(ids_set)
            if len(ids) == 0:
                return np.zeros_like(obj_id, dtype=bool)
            ids_arr = np.array(ids, dtype=obj_id.dtype)
            return np.isin(obj_id, ids_arr)

        emitter_mask = mask_from_ids(self.ps.obj_id_emitters)
        fluid_block_mask = mask_from_ids(self.ps.obj_id_fluid_blocks)
        rigid_block_mask = mask_from_ids(self.ps.obj_id_rigid_blocks)
        rigid_body_mask = mask_from_ids(self.ps.object_id_rigid_body)

        # Assign types; keep material information implicit via these codes
        particle_type[fluid_block_mask] = 0
        particle_type[emitter_mask] = 1
        particle_type[rigid_block_mask] = 2
        particle_type[rigid_body_mask] = 3

        ply_axis_conv = str(
            self.ps.cfg.get_cfg("exportPLYAxisConvention", "y_up") or "y_up"
        )

        # Ensure output directory exists
        ply_path.mkdir(parents=True, exist_ok=True)
        file_path = ply_path / f"frame_{cnt:04}.ply"

        with open(file_path, "w") as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float vort_x\n")
            f.write("property float vort_y\n")
            f.write("property float vort_z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property int object_id\n")
            f.write("property uchar particle_type\n")
            f.write("end_header\n")

            # Data rows (axis convention for DCC: same rules as segment PLY)
            for i in range(num_vertices):
                p = SegmentExporter._apply_ply_axis_convention(x[i], ply_axis_conv)
                w = SegmentExporter._apply_ply_axis_convention(
                    np.asarray(vorticity[i], dtype=np.float64), ply_axis_conv
                )
                px, py, pz = float(p[0]), float(p[1]), float(p[2])
                vx, vy, vz = float(w[0]), float(w[1]), float(w[2])
                r, g, b = color[i]
                oid = int(obj_id[i])
                ptype = int(particle_type[i])
                f.write(
                    f"{px:.7f} {py:.7f} {pz:.7f} "
                    f"{vx:.7f} {vy:.7f} {vz:.7f} "
                    f"{int(r)} {int(g)} {int(b)} {oid} {ptype}\n"
                )

    def step(self):
        # Cull out-of-domain fluid and compact first so emit sees freed slots.
        self.ps.initialize_particle_system()
        if self._cylinder_oscillation_enabled:
            reset_translation = (
                self._translate_whole_object_from_rest_kernel
                if self._moving_rigid_object_full_body
                else self._translate_object_from_rest_kernel
            )
            reset_translation(
                self._cylinder_oscillation_object_id,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            )
        if self.cnt % self.emit_interval == 0 and self._should_emit_this_step():
            self.dump_num_particles_each_emitters_ti2np()
            self.emit_particle()
            self.dump_num_particles_each_emitters_np2ti()
            self.ps.rebuild_neighbor_grid()
        self._update_oscillating_cylinder()
        self.cnt += 1
        self.compute_moving_boundary_volume()
        if int(self.ps.fluid_particle_num[None]) <= 0:
            return
        self.substep()

        if self.ps.cfg.get_cfg("bounceBackBoundary"):
            if self.ps.dim == 2:
                self.enforce_boundary_2D(self.ps.material_fluid)
            elif self.ps.dim == 3:
                self.enforce_boundary_3D(self.ps.material_fluid)

    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        self.divergence_solve()
        self.compute_non_pressure_forces()
        self.dimcv()
        self.predict_velocity()
        self.pressure_solve()
        self.copy_x_temp()
        self.compute_vorticity_vis()
        self.advect()
