import taichi as ti
from dimc_sph import DIMCSPHSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class KarmanVortexSolver(DIMCSPHSolver):

    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.circle_pos = ti.Vector([0.4, 0.5])
        self.circle_vis = ti.Vector.field(self.ps.dim, dtype=float, shape=1)
        self.circle_vis[0] = ti.Vector(
            [self.circle_pos[0] * 0.25, self.circle_pos[1]])
        self.circle_radius = 0.1
        self.init_circle()

    @ti.kernel
    def init_circle(self):
        for p in range(self.ps.particle_num[None]):
            if self.ps.object_id[p] == 2:
                if (self.ps.x[p] -
                        self.circle_pos).norm() > self.circle_radius:
                    self.ps.x[p] = ti.Vector([0, 0])
                    self.ps.is_active[p] = 0

    def export_png(self, cnt, image_path):
        N = self.ps.particle_num[None]
        material = self.ps.material.to_numpy()[:N]
        obj_id = self.ps.object_id.to_numpy()[:N]
        fluid_mask = (material == self.ps.material_fluid)
        solid_mask = (obj_id == 2)

        x = self.x_temp.to_numpy()[:N]
        vort = self.ps.vorticity_vis.to_numpy()[:N][:, 2]

        fluid_x = x[fluid_mask]
        fluid_vort = vort[fluid_mask]
        solid_x = x[solid_mask]

        fluid_x_x = fluid_x[:, 0]
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

        plt.savefig(image_path / f"vorticity_{cnt:04}.png",
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=400)
        plt.cla()
        plt.close('all')

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
        self.dimc()
        self.predict_velocity()
        self.pressure_solve()
        self.copy_x_temp()
        self.compute_vorticity_vis()
        self.advect()
