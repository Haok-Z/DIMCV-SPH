import os
import argparse
import taichi as ti
from config_builder import SimConfig
from particle_system import ParticleSystem
from pathlib import Path
import time

ti.init(arch=ti.cuda,
        device_memory_fraction=0.5,
        debug=False,
        random_seed=int(time.time()),
        kernel_profiler=False)

def main():
    parser = argparse.ArgumentParser(description='Dynamic Importance Monte Carlo Vortical SPH')
    parser.add_argument('--scene_file', default='', help='scene file')
    image_path = Path("result_images")
    image_path.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    simulation_time = config.get_cfg("simulationTime")
    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))

    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()

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
        t += solver.dt[None]
        if cnt % output_interval == 0:
            solver.export_png(cnt_output, image_path)
            cnt_output += 1
        if cnt % 50 == 0:
            print(f"Simulation Time = {t:.2f}s")
            print("Fluid_particle_num = {}".format(solver.ps.fluid_particle_num[None]))
            print("Sample_num = ", solver.num_samples[None])
        if t > simulation_time or os.path.exists("stop"):
            break

if __name__ == "__main__":
    main()