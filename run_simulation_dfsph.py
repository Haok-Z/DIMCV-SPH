"""
Baseline DFSPH Kármán street (no DIMCV / KVN).

Requires Configuration.simulationMethod == 1 (see data/scenes/DIM_von_karman_vortex_dfsph.json).

Example (3D):
    python run_simulation_dfsph.py --scene_file ./data/scenes/DIM_von_karman_vortex_dfsph.json

Example (2D):
    python run_simulation_dfsph.py --scene_file ./data/scenes/DIM_von_karman_vortex_2d_dfsph.json
"""

import os
import argparse
import taichi as ti
from config_builder import SimConfig, resolve_output_paths
from particle_system import ParticleSystem
from flow_statistics import FlowStatisticsRecorder
from pathlib import Path
import shutil
import time

ti.init(
    arch=ti.cuda,
    device_memory_fraction=1,
    debug=False,
    random_seed=int(time.time()),
    kernel_profiler=False,
)


def main():
    parser = argparse.ArgumentParser(
        description="DFSPH-only Kármán vortex (no DIMCV / KVN vort2vel correction)"
    )
    parser.add_argument(
        "--scene_file",
        default="./data/scenes/DIM_von_karman_vortex_dfsph.json",
        help="Scene JSON; must use simulationMethod 1 for DFSPH-only solver",
    )
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    image_path, ply_path = resolve_output_paths(
        scene_path, config.config["Configuration"], "dfsph"
    )
    if image_path.exists():
        shutil.rmtree(image_path)
    image_path.mkdir(parents=True, exist_ok=True)

    sm = config.get_cfg("simulationMethod")
    if sm is not None and int(sm) != 1:
        print(
            f"[run_simulation_dfsph] Warning: simulationMethod={sm} "
            f"(expected 1 for DFSPH-only). ParticleSystem will still follow JSON."
        )

    export_ply = config.get_cfg("exportPLY")
    export_ply = bool(export_ply) if export_ply is not None else False
    if export_ply:
        if ply_path.exists():
            shutil.rmtree(ply_path)
        ply_path.mkdir(parents=True, exist_ok=True)
    else:
        ply_path = None
    print(f"[DFSPH] image output: {image_path}")
    if ply_path is not None:
        print(f"[DFSPH] PLY output: {ply_path}")

    simulation_time = config.get_cfg("simulationTime")
    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))

    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()
    statistics = FlowStatisticsRecorder(image_path.parent, config, "DFSPH")

    cnt = 0
    cnt_output = 0
    t = 0.0
    while True:
        for _ in range(substeps):
            solver.step()
        cnt += 1
        t += solver.dt[None] * substeps
        if cnt % output_interval == 0:
            statistics.record(solver, cnt_output, t)
            solver.export_png(cnt_output, image_path)
            if export_ply and hasattr(solver, "export_ply"):
                solver.export_ply(cnt_output, ply_path)
            cnt_output += 1
        if cnt % 50 == 0:
            print(f"[DFSPH] Simulation Time = {t:.2f}s")
            print(f"[DFSPH] Fluid_particle_num = {solver.ps.fluid_particle_num[None]}")
        if t > simulation_time or os.path.exists("stop"):
            break
    statistics.close()


if __name__ == "__main__":
    main()
