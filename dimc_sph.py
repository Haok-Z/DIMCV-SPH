import taichi as ti
from sph_base import SPHBase
import numpy as np


@ti.data_oriented
class DIMCSPHSolver(SPHBase):

    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.surface_tension = 0
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.eps = 1e-6
        self.inv_eps = 1.0 / self.eps
        self.volume_scale = self.ps.cfg.get_cfg("volumeScale")
        if self.volume_scale != None:
            self.vp_volume = self.volume_scale * self.ps.m_V0

        #======= DFSPH Setting =============#
        self.m_max_iterations_v = 100
        self.m_max_iterations = 100
        self.m_eps = 1e-5
        self.max_error_V = 0.1
        self.max_error = 0.05
        self.inv_dt = 1.0 / self.dt[None]
        self.inv_dt2 = 1.0 / (self.dt[None] * self.dt[None])

        self.d_vort = ti.Vector.field(3,
                                      dtype=float,
                                      shape=self.ps.particle_max_num)
        self.d_vort_smoothed = ti.Vector.field(3,
                                               dtype=float,
                                               shape=self.ps.particle_max_num)
        self.v_star = ti.Vector.field(self.ps.dim,
                                      dtype=float,
                                      shape=self.ps.particle_max_num)
        self.grad_v = ti.Matrix.field(self.ps.dim,
                                      self.ps.dim,
                                      dtype=float,
                                      shape=self.ps.particle_max_num)
        self.x_temp = ti.Vector.field(self.ps.dim,
                                      dtype=float,
                                      shape=self.ps.particle_max_num)
        self.num_samples = ti.field(dtype=int, shape=())
        self.sample_idx = ti.field(dtype=int, shape=self.ps.particle_max_num)
        self.kinematic_vort_num = ti.field(dtype=float,
                                           shape=self.ps.particle_max_num)
        self.kernel_density = ti.field(dtype=float,
                                       shape=self.ps.particle_max_num)
        self.kden_sum = ti.field(dtype=float, shape=())
        self.kernel_sum = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.vortex_enforcing_domain_start = self.ps.cfg.get_cfg(
            "vortexEnforcingDomainStart")
        self.vortex_enforcing_domain_end = self.ps.cfg.get_cfg(
            "vortexEnforcingDomainEnd")
        self.a = self.ps.cfg.get_cfg("GenProbPara")
        self.b = self.ps.cfg.get_cfg("DelProbPara")
        num_samples = self.ps.cfg.get_cfg("numSamples")
        if num_samples:
            self.num_samples[None] = num_samples
        self.divergence_transit_area = self.ps.cfg.get_cfg(
            "divergenceTransitArea")

        #============ emitter ==============#
        self.cnt = 0
        self.emit_interval = 1
        for emitter in self.ps.fluid_emitters:
            init_v = np.array(emitter["velocity"])
            self.emit_interval = np.ceil(
                self.ps.particle_diameter /
                (np.linalg.norm(init_v) * self.dt[None])).astype(int)
            break

    @ti.func
    def is_in_vortex_domain(self, pos) -> bool:
        flag = True
        if ti.static(self.vortex_enforcing_domain_start != None):
            for i in ti.static(range(self.ps.dim)):
                if flag and (pos[i] < self.vortex_enforcing_domain_start[i]
                             or pos[i] > self.vortex_enforcing_domain_end[i]):
                    flag = False
        return flag

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.ps.dim)])
            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            self.ps.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)
            sum_grad_p_k = ret[self.ps.dim]
            for i in ti.static(range(self.ps.dim)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()
            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-6:
                factor = 1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.ps.dfsph_factor[p_i] = factor

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            grad_p_j = self.ps.m[p_j] * self.cubic_kernel_derivative(
                self.ps.x[p_i] - self.ps.x[p_j])
            ret[self.ps.dim] += grad_p_j.norm_sqr()  # sum_grad_p_k
            for i in ti.static(range(self.ps.dim)):  # grad_p_i
                ret[i] += grad_p_j[i]
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = self.ps.m_V[
                p_j] * self.density_0 * self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
            for i in ti.static(range(self.ps.dim)):  # grad_p_i
                ret[i] += grad_p_j[i]

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        ############## Surface Tension ###############
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[
                    p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[
                    p_j] * r * self.cubic_kernel(
                        ti.Vector([self.ps.particle_diameter, 0.0, 0.0
                                   ]).norm())

        ############### Viscosity Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)

        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * self.ps.m_V[p_j] * v_xy / (
                r.norm()**2 + 0.01 *
                self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (
                self.density_0 * self.ps.m_V[p_j] /
                (self.ps.density[p_i])) * v_xy / (
                    r.norm()**2 + 0.01 * self.ps.support_radius**2
                ) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[
                    p_j] += -f_v * self.ps.density[p_i] / self.ps.density[p_j]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(
                    p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Update position
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i]:
                if self.ps.is_dynamic_rigid_body(p_i):
                    self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                # self.ps.v[p_i] += self.v_vort[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    @ti.kernel
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i] and self.ps.material[
                    p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]

    @ti.func
    def compute_vorticity_task_diff(self, p_i, p_j, vort: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            if ti.static(self.ps.dim == 3):
                vort += self.ps.m_V[p_j] * (
                    self.ps.v[p_j] - self.ps.v[p_i]).cross(
                        self.cubic_kernel_derivative(self.ps.x[p_i] -
                                                     self.ps.x[p_j]))
            elif ti.static(self.ps.dim == 2):
                v_i = ti.Vector([0.0 for _ in ti.static(range(3))])
                v_j = ti.Vector([0.0 for _ in ti.static(range(3))])
                kernel_derivative_3d = ti.Vector(
                    [0.0 for _ in ti.static(range(3))])
                kernel_derivative_2d = self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
                for i in ti.static(range(self.ps.dim)):
                    v_i[i] = self.ps.v[p_i][i]
                    v_j[i] = self.ps.v[p_j][i]
                    kernel_derivative_3d[i] = kernel_derivative_2d[i]
                vort += self.ps.m_V[p_j] * (v_j -
                                            v_i).cross(kernel_derivative_3d)

    @ti.func
    def compute_grad_v_task(self, p_i, p_j, grad_v: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            vji_vec = self.ps.v[p_j] - self.ps.v[p_i]
            kernel_deriv_vec = self.cubic_kernel_derivative(self.ps.x[p_i] -
                                                            self.ps.x[p_j])
            vji_mat = ti.Matrix([[0.0] for _ in ti.static(range(self.ps.dim))])
            kernel_deriv_trans_mat = ti.Matrix(
                [[0.0 for _ in ti.static(range(self.ps.dim))]])
            for i in ti.static(range(self.ps.dim)):
                vji_mat[i, 0] = vji_vec[i]
                kernel_deriv_trans_mat[0, i] = kernel_deriv_vec[i]
            grad_v += self.ps.m_V[p_j] * vji_mat @ kernel_deriv_trans_mat

    @ti.kernel
    def compute_v_star(self):
        self.v_star.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.v_star[p_i] = self.ps.v[
                p_i] + self.ps.acceleration[p_i] * self.dt[None]

    @ti.kernel
    def compute_grad_v(self):
        self.grad_v.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            grad_v = ti.Matrix([[0.0] * self.ps.dim
                                for _ in ti.static(range(self.ps.dim))])
            self.ps.for_all_neighbors(p_i, self.compute_grad_v_task, grad_v)
            self.grad_v[p_i] = grad_v

    @ti.kernel
    def compute_vorticity(self):
        self.ps.vorticity.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            vort = ti.Vector([0.0 for _ in ti.static(range(3))])
            self.ps.for_all_neighbors(p_i, self.compute_vorticity_task_diff,
                                      vort)
            self.ps.vorticity[p_i] = vort

    @ti.func
    def get_produce_prob(self, kvn, t):
        t0 = ti.math.clamp(t, 0.0, 1.0)
        r = 0.018 * ti.exp(-self.a * t0) + 0.004
        return 1.0 - 1.0 / ((kvn + 1.0)**r)

    @ti.func
    def get_del_prob(self, kvn):
        return ti.exp(-self.b * kvn)

    @ti.kernel
    def compute_kinematic_vorticity_num_sample(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[
                    p_i] != self.ps.material_fluid or not self.is_in_vortex_domain(
                        self.ps.x[p_i]):
                continue
            grad_v = self.grad_v[p_i]
            grad_v_trans = grad_v.transpose()
            vorticity_tensor = grad_v - grad_v_trans
            strain_tensor = grad_v + grad_v_trans
            KVN = vorticity_tensor.norm() / (strain_tensor.norm() + 1e-8)
            produce_prob = self.get_produce_prob(KVN, self.ps.life_time[p_i])
            del_prob = self.get_del_prob(KVN)
            self.kinematic_vort_num[p_i] = KVN
            if self.ps.is_sample[p_i] == 0 and ti.random(float) < produce_prob:
                self.ps.is_sample[p_i] = True
            elif self.ps.is_sample[p_i] == 1 and KVN < 2.0 and ti.random(
                    float) < del_prob:
                self.ps.is_sample[p_i] = 0

    @ti.kernel
    def record_vort_part_sample_idx(self):
        idx = 0
        self.num_samples[None] = 0
        self.sample_idx.fill(0)
        for p in range(self.ps.particle_num[None]):
            if self.ps.is_sample[p] == 1:
                idx_tmp = ti.atomic_add(idx, 1)
                self.sample_idx[idx_tmp] = p
        self.num_samples[None] = idx

    @ti.kernel
    def compute_density_change(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            self.ps.for_all_neighbors(p_i, self.compute_density_change_task,
                                      ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if self.ps.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0

            self.ps.density_adv[p_i] = density_adv

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        # Compute the number of neighbors
        ret.num_neighbors += 1

    def divergence_solve(self):
        # TODO: warm start
        # Compute velocity of density change
        self.compute_density_change()
        #inv_dt = 1.0 / self.dt[None]
        self.multiply_time_step(self.ps.dfsph_factor, self.inv_dt)
        m_iterations_v = 0
        # Start solver
        avg_density_err = 0.0
        while m_iterations_v < 1 or m_iterations_v < self.m_max_iterations_v:
            avg_density_err = self.divergence_solver_iteration()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = self.inv_dt * self.max_error_V * 0.01 * self.density_0
            m_iterations_v += 1
            if avg_density_err <= eta:
                break
        self.multiply_time_step(self.ps.dfsph_factor, self.dt[None])

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in range(self.ps.particle_num[None]):
            if self.ps.material[I] == self.ps.material_fluid:
                field[I] *= time_step

    @ti.func
    def compute_divergence_transit_weight(self, x_pos):
        return ti.math.clamp((x_pos - self.divergence_transit_area[0]) /
                             (self.divergence_transit_area[1] -
                              self.divergence_transit_area[0]), 0.0, 1.0)

    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for I in range(self.ps.particle_num[None]):
            if self.ps.material[I] == self.ps.material_fluid:
                if ti.static(self.divergence_transit_area == None):
                    # if self.divergence_transit_area == None:
                    density_error += self.ps.density_adv[I] - offset
                else:
                    weight = self.compute_divergence_transit_weight(
                        self.ps.x[I][0])
                    density_error += weight * (self.ps.density_adv[I] - offset)
        return density_error

    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # evaluate rhs
            b_i = self.ps.density_adv[p_i]
            k_i = b_i * self.ps.dfsph_factor[p_i] * self.ps.density[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.ps.dim)]),
                            k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            self.ps.for_all_neighbors(p_i,
                                      self.divergence_solver_iteration_task,
                                      ret)
            if ti.static(self.divergence_transit_area == None):
                # if self.divergence_transit_area == None:
                self.ps.v[p_i] += ret.dv
            else:
                weight = self.compute_divergence_transit_weight(
                    self.ps.x[p_i][0])
                self.ps.v[p_i] += weight * ret.dv

    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j]
            k_j = b_j * self.ps.dfsph_factor[p_j] * self.ps.density[p_j]
            k_sum = ret.k_i + k_j
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = self.ps.m[p_j] * self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = self.ps.m_V[
                    p_j] * self.density_0 * self.cubic_kernel_derivative(
                        self.ps.x[p_i] - self.ps.x[p_j])
                vel_change = -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[
                        p_j] += -vel_change * self.inv_dt * self.ps.density[
                            p_i] / self.ps.density[p_j]

    @ti.kernel
    def compute_density_adv(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            delta = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_density_adv_task,
                                      delta)
            density_adv = self.ps.density[p_i] + self.dt[None] * delta
            self.ps.density_adv[p_i] = ti.max(density_adv, self.density_0)

    @ti.func
    def compute_density_adv_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret += self.ps.m[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret += self.ps.m_V[p_j] * self.density_0 * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))

    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # Evaluate rhs
            b_i = self.ps.density_adv[p_i] - self.density_0
            k_i = b_i * self.ps.dfsph_factor[p_i]

            # TODO: if warmstart
            # get kappa V
            self.ps.for_all_neighbors(p_i, self.pressure_solve_iteration_task,
                                      k_i)

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j] - self.density_0
            k_j = b_j * self.ps.dfsph_factor[p_j]
            k_sum = k_i + k_j
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = self.ps.m[p_j] * self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
                # Directly update velocities instead of storing pressure accelerations
                if ti.static(self.divergence_transit_area == None):
                    # if self.divergence_transit_area == None:
                    self.ps.v[p_i] -= self.dt[
                        None] * k_sum * grad_p_j  # ki, kj already contain inverse density
                else:
                    weight = self.compute_divergence_transit_weight(
                        self.ps.x[p_i][0])
                    self.ps.v[p_i] -= weight * self.dt[None] * k_sum * grad_p_j
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(k_i) > self.m_eps:
                grad_p_j = self.ps.m_V[
                    p_j] * self.density_0 * self.cubic_kernel_derivative(
                        self.ps.x[p_i] - self.ps.x[p_j])

                # Directly update velocities instead of storing pressure accelerations
                vel_change = -self.dt[
                    None] * 1.0 * k_i * grad_p_j  # kj already contains inverse density
                if ti.static(self.divergence_transit_area == None):
                    self.ps.v[p_i] += vel_change
                else:
                    weight = self.compute_divergence_transit_weight(
                        self.ps.x[p_i][0])
                    self.ps.v[p_i] += weight * vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[
                        p_j] += -vel_change * self.inv_dt * self.ps.density[
                            p_i] / self.ps.density[p_j]

    @ti.func
    def compute_curl_v_star_task(self, p_i, p_j, curl_v_star: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            if ti.static(self.ps.dim == 3):
                curl_v_star += self.ps.m_V[p_j] * (
                    self.v_star[p_j] - self.v_star[p_i]).cross(
                        self.cubic_kernel_derivative(self.ps.x[p_i] -
                                                     self.ps.x[p_j]))
            elif ti.static(self.ps.dim == 2):
                v_star_i = ti.Vector([0.0 for _ in ti.static(range(3))])
                v_star_j = ti.Vector([0.0 for _ in ti.static(range(3))])
                kernel_derivative_3d = ti.Vector(
                    [0.0 for _ in ti.static(range(3))])
                kernel_derivative_2d = self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
                for i in ti.static(range(self.ps.dim)):
                    v_star_i[i] = self.v_star[p_i][i]
                    v_star_j[i] = self.v_star[p_j][i]
                    kernel_derivative_3d[i] = kernel_derivative_2d[i]
                curl_v_star += self.ps.m_V[p_j] * (
                    v_star_j - v_star_i).cross(kernel_derivative_3d)

    @ti.kernel
    def compute_d_vorticity(self):
        self.d_vort.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            curl_v_star = ti.Vector([0.0 for _ in ti.static(range(3))])
            self.ps.for_all_neighbors(p_i, self.compute_curl_v_star_task,
                                      curl_v_star)
            d_vorticity = self.ps.vorticity[p_i] - curl_v_star
            self.d_vort[p_i] = d_vorticity

    def pressure_solve_iteration(self):
        self.pressure_solve_iteration_kernel()
        self.compute_density_adv()
        density_err = self.compute_density_error(self.density_0)
        return density_err / self.ps.fluid_particle_num[None]

    def divergence_solver_iteration(self):
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_error(0.0)
        return density_err / self.ps.fluid_particle_num[None]

    def pressure_solve(self):
        inv_dt2 = self.inv_dt2

        # Compute rho_adv
        self.compute_density_adv()
        self.multiply_time_step(self.ps.dfsph_factor, inv_dt2)
        m_iterations = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations < 1 or m_iterations < self.m_max_iterations:
            avg_density_err = self.pressure_solve_iteration()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01 * self.density_0
            m_iterations += 1
            if avg_density_err <= eta:
                break

    def emit_particle(self):
        for emitter in self.ps.fluid_emitters:
            obj_id = emitter["objectId"]
            squareCenter = np.array(emitter["squareCenter"])
            squareSize = np.array(emitter["squareSize"])
            init_v = np.array(emitter["velocity"])
            density = emitter["density"]
            color = emitter["color"]
            lower_corner = squareCenter - 0.5 * squareSize
            cube_size = squareSize + self.ps.particle_radius * np.abs(
                init_v) / np.linalg.norm(init_v)
            self.ps.object_collection[obj_id][
                "particleNum"] += self.ps.compute_cube_particle_num(
                    lower_corner, lower_corner + cube_size)
            self.ps.add_cube(object_id=obj_id,
                             lower_corner=lower_corner,
                             cube_size=cube_size,
                             velocity=init_v,
                             density=density,
                             is_dynamic=1,
                             color=color,
                             material=1)

    def dump_num_particles_each_emitters_ti2np(self):
        for obj_id in self.ps.obj_id_emitters:
            self.ps.object_collection[obj_id][
                "particleNum"] = self.ps.num_particles_each_emitter[obj_id]

    def dump_num_particles_each_emitters_np2ti(self):
        for obj_id in self.ps.obj_id_emitters:
            self.ps.num_particles_each_emitter[
                obj_id] = self.ps.object_collection[obj_id]["particleNum"]

    @ti.func
    def vort2vel_neighbors_2d_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            r_2d = self.ps.x[p_i] - self.ps.x[p_j]
            r = ti.Vector([r_2d[0], r_2d[1], 0.0])
            r_norm = r.norm()
            ret.v_vort += (1.0 - ti.exp(-r_norm * self.inv_eps)) / (
                r_norm * (r_norm + self.eps * ti.exp(-r_norm * self.inv_eps))
            ) * ti.math.cross(r, self.d_vort_smoothed[p_j])
            ret.kden_sum_neighbors += self.kernel_density[p_j]

    @ti.func
    def smooth_d_vorticity_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            ret += self.ps.m_V[p_j] * self.d_vort[p_j] * self.cubic_kernel(
                (self.ps.x[p_i] - self.ps.x[p_j]).norm())

    @ti.kernel
    def compute_all_d_vort_smoothed(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[
                    p_i] != self.ps.material_fluid or not self.is_in_vortex_domain(
                        self.ps.x[p_i]):
                continue
            d_vort_smoothed = self.ps.m_V[p_i] * self.d_vort[
                p_i] * self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.smooth_d_vorticity_task,
                                      d_vort_smoothed)
            self.d_vort_smoothed[p_i] = d_vort_smoothed

    @ti.kernel
    def vort2vel(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[
                    p_i] != self.ps.material_fluid or not self.is_in_vortex_domain(
                        self.ps.x[p_i]):
                continue
            ret_neighbors = ti.Struct(
                v_vort=ti.Vector([0.0 for _ in ti.static(range(3))]),
                kden_sum_neighbors=self.kernel_density[p_i])
            self.ps.for_all_neighbors(p_i, self.vort2vel_neighbors_2d_task,
                                      ret_neighbors)
            ret_neighbors.v_vort *= 0.1592 * self.vp_volume

            # far samples
            v_vort_samples = ti.Vector([0.0 for _ in ti.static(range(3))])
            added_sample_num = 0
            for j in range(self.num_samples[None]):
                p_j = self.sample_idx[j]
                r_2d = self.ps.x[p_i] - self.ps.x[p_j]
                r_norm = r_2d.norm()
                if r_norm > self.ps.support_radius:
                    r = ti.Vector([r_2d[0], r_2d[1], 0.0])
                    v_vort_samples += (
                        1.0 - ti.exp(-r_norm * self.inv_eps)
                    ) / (r_norm *
                         (r_norm + self.eps * ti.exp(-r_norm * self.inv_eps))
                         ) * ti.math.cross(r, self.d_vort_smoothed[p_j]) * (
                             self.kden_sum[None] -
                             ret_neighbors.kden_sum_neighbors) / (
                                 self.kernel_density[p_j] + 1e-8)
                    added_sample_num += 1
            if added_sample_num == 0:
                v_vort_samples.fill(0.0)
            else:
                v_vort_samples *= 0.1592 * self.vp_volume / added_sample_num
            v_vort = v_vort_samples + ret_neighbors.v_vort

            if v_vort.norm() < self.ps.v[p_i].norm():
                for i in ti.static(range(self.ps.dim)):
                    self.ps.v[p_i][i] += v_vort[i]

    @ti.func
    def epanechnikov_kernel(self, r):
        ret = 1.0
        for i in ti.static(range(self.ps.dim)):
            q = ti.abs(r[i]) * self.ps.inv_support_radius
            if q < 1.0:
                ret *= 0.75 * (1 - q * q)
            else:
                ret *= 0.0
        return ret

    @ti.func
    def compute_kernel_density_task(self, p_i, p_j, kden: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            kden += self.kinematic_vort_num[p_j] * self.epanechnikov_kernel(
                (self.ps.x[p_i] - self.ps.x[p_j]))

    @ti.kernel
    def compute_kernel_density(self):
        self.kden_sum[None] = 0.0
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[
                    p_i] != self.ps.material_fluid or not self.is_in_vortex_domain(
                        self.ps.x[p_i]):
                continue
            r0 = ti.Vector([0.0 for _ in ti.static(range(self.ps.dim))])
            kden = self.kinematic_vort_num[p_i] * self.epanechnikov_kernel(r0)
            self.ps.for_all_neighbors(p_i, self.compute_kernel_density_task,
                                      kden)
            self.kernel_density[p_i] = kden
            self.kden_sum[None] += kden

    @ti.kernel
    def update_particle_life_time(self):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[
                    p] != self.ps.material_fluid or not self.is_in_vortex_domain(
                        self.ps.x[p]):
                continue
            self.ps.life_time[p] += self.dt[None]

    @ti.func
    def compute_kernel_sum_task(self, p_i, p_j, ret: ti.template()):
        ret += self.ps.m_V[p_j] * self.cubic_kernel(
            (self.ps.x[p_i] - self.ps.x[p_j]).norm())

    @ti.kernel
    def compute_kernel_sum(self):
        self.kernel_sum.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            kernel_sum = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_kernel_sum_task,
                                      kernel_sum)
            self.kernel_sum[p_i] = kernel_sum

    @ti.func
    def compute_vorticity_vis_task(self, p_i, p_j, ret: ti.template()):
        w = self.ps.m_V[p_j] * self.cubic_kernel(
            (self.ps.x[p_i] - self.ps.x[p_j]).norm())
        ret.vort += w * self.ps.vorticity[p_j]
        ret.w += w

    @ti.kernel
    def compute_vorticity_vis(self):
        self.ps.vorticity_vis.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            w = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            ret = ti.Struct(vort=w * self.ps.vorticity[p_i], w=w)
            self.ps.for_all_neighbors(p_i, self.compute_vorticity_vis_task,
                                      ret)
            self.ps.vorticity_vis[p_i] = ret.vort / (ret.w + 1e-6)

    @ti.kernel
    def copy_x_temp(self):
        self.x_temp.fill(0.0)
        for p in range(self.ps.particle_num[None]):
            self.x_temp[p] = self.ps.x[p]

    def init(self):
        self.compute_v_star()
        self.compute_vorticity()
        self.compute_grad_v()

    def update_sample(self):
        self.compute_kinematic_vorticity_num_sample()
        self.record_vort_part_sample_idx()

    def dimc(self):
        self.init()
        self.compute_d_vorticity()
        self.compute_all_d_vort_smoothed()
        self.update_sample()
        self.compute_kernel_density()
        self.vort2vel()
        self.update_particle_life_time()
