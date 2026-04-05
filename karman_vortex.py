import taichi as ti
from dimcv_sph import DIMCVSPHSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class KarmanVortexSolver(DIMCVSPHSolver):

    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.circle_pos = ti.Vector([0.65, 0.5, 0.5])
        self.circle_vis = ti.Vector.field(self.ps.dim, dtype=float, shape=1)
        self.circle_vis[0] = ti.Vector(
            [self.circle_pos[0] * 0.25, self.circle_pos[1], self.circle_pos[2]])
        self.circle_radius = 0.1
        self.init_cylinder()

    @ti.kernel
    def init_circle(self):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == 2:
                if (self.ps.x[p] - self.circle_pos).norm() > self.circle_radius:
                    self.ps.x[p] = ti.Vector([0, 0, 0])
                    self.ps.is_active[p] = 0

    @ti.kernel
    def init_cylinder(self):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == 2:
                dx = self.ps.x[p][0] - self.circle_pos[0]
                dy = self.ps.x[p][1] - self.circle_pos[1]
                # Cylinder axis along z, so radius in x-y plane
                if ti.sqrt(dx * dx + dy * dy) > self.circle_radius:
                    self.ps.x[p] = ti.Vector([0, 0, 0])
                    self.ps.is_active[p] = 0

    def export_png(self, cnt, image_path):
        N = self.ps.particle_num[None]
        material = self.ps.material.to_numpy()[:N]
        obj_id = self.ps.object_id.to_numpy()[:N]
        fluid_mask = (material == self.ps.material_fluid)
        #solid_mask = (material == self.ps.material_solid)
        solid_mask = (obj_id == 2)

        x = self.x_temp.to_numpy()[:N]
        vort = self.ps.vorticity_vis.to_numpy()[:N][:, 2]

        fluid_x = x[fluid_mask]
        fluid_vort = vort[fluid_mask]
        solid_x = x[solid_mask]

        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=-60)
        sc = ax.scatter(fluid_x[:, 0],
                        fluid_x[:, 1],
                        fluid_x[:, 2],
                        c=fluid_vort,
                        cmap='coolwarm',
                        s=0.2,  # 3D 模式下粒子尺寸建议调小
                        norm=Normalize(vmin=-40, vmax=40),
                        edgecolors='none',
                        alpha=0.6)
        ax.scatter(solid_x[:, 0],
                   solid_x[:, 1],
                   solid_x[:, 2],
                   color="#00C853",
                   s=0.5,
                   edgecolors='none')

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_box_aspect((4, 1, 1))
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

            # Data rows
            for i in range(num_vertices):
                px, py, pz = x[i]
                vx, vy, vz = vorticity[i]
                r, g, b = color[i]
                oid = int(obj_id[i])
                ptype = int(particle_type[i])
                f.write(
                    f"{px:.7f} {py:.7f} {pz:.7f} "
                    f"{vx:.7f} {vy:.7f} {vz:.7f} "
                    f"{int(r)} {int(g)} {int(b)} {oid} {ptype}\n"
                )

    def step(self):
        self.dump_num_particles_each_emitters_ti2np()
        if self.cnt % self.emit_interval == 0:
            self.emit_particle()
            self.dump_num_particles_each_emitters_np2ti()
        self.cnt += 1
        self.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
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
