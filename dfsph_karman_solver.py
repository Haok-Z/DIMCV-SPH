"""
Pure DFSPH Kármán vortex solver (no DIMCV / KVN / vort2vel Monte Carlo).

Substep matches DIMCVSPHSolver except ``dimcv()`` is omitted: density, DFSPH
factor, divergence solve, non-pressure forces, predict_velocity, pressure
solve, ``compute_vorticity`` (raw SPH curl), ``compute_vorticity_vis`` (kernel
smoothed field for PNG / consistency), copy_x_temp, advection.

PLY ``vort_x/y/z`` come from ``ps.vorticity``; PNG coloring uses ``vorticity_vis``
(ω_z component in ``KarmanVortexSolver.export_png``).

Use ``simulationMethod: 1`` in Configuration with the same scene geometry as
``DIM_von_karman_vortex.json`` (see ``DIM_von_karman_vortex_dfsph.json``).
"""

from karman_vortex import KarmanVortexSolver


class DFSPHKarmanVortexSolver(KarmanVortexSolver):
    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        self.divergence_solve()
        self.compute_non_pressure_forces()
        self.predict_velocity()
        self.pressure_solve()
        self.copy_x_temp()
        self.compute_vorticity()
        self.compute_vorticity_vis()
        self.advect()
