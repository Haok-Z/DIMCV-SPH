import os
import argparse
import taichi as ti
from config_builder import SimConfig, resolve_output_paths
from particle_system import ParticleSystem
from flow_statistics import FlowStatisticsRecorder
from pathlib import Path
import shutil
import time

ti.init(arch=ti.cuda,
        device_memory_fraction=1,
        debug=False,
        random_seed=int(time.time()),
        kernel_profiler=False)

def main():
    parser = argparse.ArgumentParser(description='Dynamic Importance Monte Carlo Vortical SPH')
    parser.add_argument('--scene_file', default='', help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    image_path, ply_path = resolve_output_paths(
        scene_path, config.config["Configuration"], "dimcv"
    )
    # Clean previous PNGs for this scene and method only.
    if image_path.exists():
        shutil.rmtree(image_path)
    image_path.mkdir(parents=True, exist_ok=True)

    export_ply = config.get_cfg("exportPLY")
    export_ply = bool(export_ply) if export_ply is not None else False
    if export_ply:
        if ply_path.exists():
            shutil.rmtree(ply_path)
        ply_path.mkdir(parents=True, exist_ok=True)
    else:
        ply_path = None
    print(f"[DIMCV] image output: {image_path}")
    if ply_path is not None:
        print(f"[DIMCV] PLY output: {ply_path}")
    simulation_time = config.get_cfg("simulationTime")
    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))

    ps = ParticleSystem(config, GGUI=True) 
    solver = ps.build_solver()
    solver.initialize()
    statistics = FlowStatisticsRecorder(image_path.parent, config, "DIMCV")

    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

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
            # Export corresponding PLY point cloud with particle attributes
            if export_ply and hasattr(solver, "export_ply"):
                solver.export_ply(cnt_output, ply_path)
            cnt_output += 1
        if cnt % 50 == 0:
            print(f"Simulation Time = {t:.2f}s")
            print("Fluid_particle_num = {}".format(solver.ps.fluid_particle_num[None]))
            print("Sample_num = ", solver.num_samples[None])
        if t > simulation_time or os.path.exists("stop"):
            break
    statistics.close()

if __name__ == "__main__":
    main()