"""Run the 2018 micropolar SPH method (simulationMethod=3)."""

import argparse
import os
import shutil
import time

import taichi as ti

from config_builder import SimConfig, resolve_output_paths
from flow_statistics import FlowStatisticsRecorder
from particle_system import ParticleSystem


ti.init(arch=ti.cuda, device_memory_fraction=1, debug=False,
        random_seed=int(time.time()), kernel_profiler=False)


def main():
    parser = argparse.ArgumentParser(description="MPSPH + DFSPH Karman solver")
    parser.add_argument("--scene_file", required=True,
                        help="Scene JSON with Configuration.simulationMethod=3")
    parser.add_argument("--max_time", type=float, default=None,
                        help="Optional wall-clock simulation-time cap for smoke tests")
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    image_path, ply_path = resolve_output_paths(
        scene_path, config.config["Configuration"], "mpsph"
    )
    if image_path.exists():
        shutil.rmtree(image_path)
    image_path.mkdir(parents=True, exist_ok=True)
    export_ply = bool(config.get_cfg("exportPLY") or False)
    if export_ply:
        if ply_path.exists():
            shutil.rmtree(ply_path)
        ply_path.mkdir(parents=True, exist_ok=True)
    else:
        ply_path = None

    method = config.get_cfg("simulationMethod")
    if method is not None and int(method) != 3:
        print(f"[MPSPH] Warning: simulationMethod={method} (expected 3)")
    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()
    statistics = FlowStatisticsRecorder(image_path.parent, config, "MPSPH")
    substeps = int(config.get_cfg("numberOfStepsPerRenderUpdate") or 1)
    output_interval = max(1, int(0.016 / float(config.get_cfg("timeStepSize"))))
    simulation_time = float(config.get_cfg("simulationTime") or 0.0)
    if args.max_time is not None:
        simulation_time = min(simulation_time, max(0.0, args.max_time))
    t = 0.0
    frame = 0
    output_frame = 0
    while True:
        for _ in range(substeps):
            solver.step()
        t += float(solver.dt[None]) * substeps
        frame += 1
        if frame % output_interval == 0:
            statistics.record(solver, output_frame, t)
            solver.export_png(output_frame, image_path)
            if ply_path is not None:
                solver.export_ply(output_frame, ply_path)
            output_frame += 1
        if frame % 50 == 0:
            print(f"[MPSPH] Simulation Time = {t:.3f}s, "
                  f"Fluid_particle_num = {solver.ps.fluid_particle_num[None]}")
        if t > simulation_time or os.path.exists("stop"):
            break
    statistics.close()


if __name__ == "__main__":
    main()
