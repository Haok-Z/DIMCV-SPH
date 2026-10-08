"""Micropolar SPH solver coupled to the existing DFSPH projection.

This implements the inviscid MPSPH model from Bender et al. (2018): a
particle carries a micro-rotation field and exchanges angular/linear motion
through the transfer coefficient ``nu_t``.  Pressure and boundary handling
remain the DFSPH implementation used by the other Karman solvers.
"""

import time

import taichi as ti
import numpy as np

from dfsph_karman_solver import DFSPHKarmanVortexSolver


@ti.data_oriented
class MPSPHKarmanSolver(DFSPHKarmanVortexSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        cfg = particle_system.cfg
        def get(name, default):
            value = cfg.get_cfg(name)
            return default if value is None else value
        self.nu_t = float(get("mpsphTransferCoefficient", 0.05) or 0.0)
        self.theta = max(float(get("mpsphMicroInertia", 2.0) or 2.0), 1e-8)
        self.gamma = float(get("mpsphLinearXSPH", 0.002) or 0.0)
        self.eta = float(get("mpsphAngularXSPH", 0.125) or 0.0)
        self.enable_xsph = bool(get("mpsphEnableXSPH", True))
        self.boundary_mode = str(get("mpsphBoundaryMode", "sph") or "sph").lower()
        self.clamp_micro_omega = bool(get("mpsphClampMicroOmega", False))
        self.micro_omega_clamp = abs(float(get("mpsphMicroOmegaClamp", 0.0) or 0.0))

        n = self.ps.particle_max_num
        self.curl_v = ti.Vector.field(3, dtype=float, shape=n)
        self.curl_micro_omega = ti.Vector.field(3, dtype=float, shape=n)
        self.micro_omega_tmp = ti.Vector.field(3, dtype=float, shape=n)
        self.mpsph_linear_coupling = ti.Vector.field(3, dtype=float, shape=n)
        self.mpsph_angular_coupling = ti.Vector.field(3, dtype=float, shape=n)
        self._last_coupling_ms = 0.0

    @ti.func
    def _vel3(self, p):
        out = ti.Vector([0.0, 0.0, 0.0])
        for d in ti.static(range(self.ps.dim)):
            out[d] = self.ps.v[p][d]
        return out

    @ti.func
    def _curl_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            aj = self._vel3(p_j)
            ai = self._vel3(p_i)
            ret.v += self.ps.m_V[p_j] * (
                aj - ai
            ).cross(self._grad3(p_i, p_j))
            ret.w += self.ps.m_V[p_j] * (
                self.ps.micro_omega[p_j] - self.ps.micro_omega[p_i]
            ).cross(self._grad3(p_i, p_j))
        elif self.boundary_mode != "none":
            # Boundary particles carry their current rigid/body velocity. Their
            # micro-rotation is zero unless a future boundary model supplies it.
            aj = self._vel3(p_j)
            ai = self._vel3(p_i)
            ret.v += self.ps.m_V[p_j] * (
                aj - ai
            ).cross(self._grad3(p_i, p_j))

    @ti.func
    def _grad3(self, p_i, p_j):
        g = self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
        out = ti.Vector([0.0, 0.0, 0.0])
        for d in ti.static(range(self.ps.dim)):
            out[d] = g[d]
        return out

    @ti.kernel
    def compute_mpsph_curls(self):
        self.curl_v.fill(0.0)
        self.curl_micro_omega.fill(0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(v=ti.Vector([0.0, 0.0, 0.0]), w=ti.Vector([0.0, 0.0, 0.0]))
            self.ps.for_all_neighbors(p_i, self._curl_task, ret)
            self.curl_v[p_i] = ret.v
            self.curl_micro_omega[p_i] = ret.w

    @ti.kernel
    def compute_mpsph_coupling(self, nu_t: float, theta: float):
        for p_i in range(self.ps.particle_num[None]):
            self.mpsph_linear_coupling[p_i] = ti.Vector([0.0, 0.0, 0.0])
            self.mpsph_angular_coupling[p_i] = ti.Vector([0.0, 0.0, 0.0])
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            linear = nu_t * self.curl_micro_omega[p_i]
            angular = (nu_t / theta) * (self.curl_v[p_i] - 2.0 * self.ps.micro_omega[p_i])
            self.mpsph_linear_coupling[p_i] = linear
            self.mpsph_angular_coupling[p_i] = angular
            self.ps.acceleration[p_i] += linear

    @ti.kernel
    def predict_micro_omega(self, dt: float, clamp_enabled: ti.i32, clamp_value: float):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            w = self.ps.micro_omega[p_i] + dt * self.mpsph_angular_coupling[p_i]
            if clamp_enabled != 0 and clamp_value > 0.0:
                n = w.norm()
                if n > clamp_value:
                    w = w * (clamp_value / n)
            self.micro_omega_tmp[p_i] = w

    @ti.func
    def _xsph_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            w = self.ps.m_V[p_j] * self.cubic_kernel(
                (self.ps.x[p_i] - self.ps.x[p_j]).norm()
            )
            ret.dv += w * (self.ps.v[p_j] - self.ps.v[p_i])
            ret.dw += w * (self.micro_omega_tmp[p_j] - self.micro_omega_tmp[p_i])

    @ti.kernel
    def apply_xsph(self, gamma: float, eta: float):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(
                dv=ti.Vector([0.0 for _ in ti.static(range(self.ps.dim))]),
                dw=ti.Vector([0.0, 0.0, 0.0]),
            )
            self.ps.for_all_neighbors(p_i, self._xsph_task, ret)
            self.ps.v[p_i] += gamma * ret.dv
            self.ps.micro_omega[p_i] = self.micro_omega_tmp[p_i] + eta * ret.dw

    @ti.kernel
    def commit_micro_omega(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.micro_omega[p_i] = self.micro_omega_tmp[p_i]

    def ply_extra_vector_fields(self, n):
        values = self.ps.micro_omega.to_numpy()[:n].astype(np.float32, copy=False)
        return {
            "micro_omega": values,
        }

    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        self.divergence_solve()
        self.compute_non_pressure_forces()

        start = time.perf_counter()
        self.compute_mpsph_curls()
        self.compute_mpsph_coupling(self.nu_t, self.theta)
        self.predict_micro_omega(
            float(self.dt[None]),
            1 if self.clamp_micro_omega else 0,
            self.micro_omega_clamp,
        )
        self.predict_velocity()
        self.pressure_solve()
        if self.enable_xsph and (abs(self.gamma) > 0.0 or abs(self.eta) > 0.0):
            self.apply_xsph(self.gamma, self.eta)
        else:
            self.commit_micro_omega()
        self._last_coupling_ms = (time.perf_counter() - start) * 1000.0

        self.copy_x_temp()
        self.compute_vorticity()
        self.compute_vorticity_vis()
        self.advect()
