"""
DFSPH + vortex segment coupling for Kármán-type scenes (simulationMethod 2).

Exports two PNGs per frame under ``result_images_dfsph_seg``:
``sph/vorticity_*.png`` (particles only) and ``segments/vorticity_*.png`` (segments colored by
kernel-weighted ``vorticity_vis.z`` at segment center, same colormap as SPH). Subfolder names
are configurable in ``SegmentConfiguration`` (``imageExportSubfolderSph`` / ``imageExportSubfolderSegments``).
"""

import os
import argparse
import taichi as ti
from pathlib import Path
import shutil
import time

from config_builder import SimConfig
from particle_system import ParticleSystem


ti.init(
    arch=ti.cuda,
    device_memory_fraction=0.5,
    debug=False,
    random_seed=int(time.time()),
    kernel_profiler=False,
)


def main():
    parser = argparse.ArgumentParser(
        description="DFSPH Kármán + coupled vortex segments (Biot–Savart feedback)"
    )
    parser.add_argument(
        "--scene_file",
        default="./data/scenes/DIM_von_karman_vortex_2d_dfsph_segment.json",
        help="JSON with Configuration + SegmentConfiguration. "
        "Emit-only experiment: DIM_von_karman_vortex_2d_dfsph_segment_emit_bs.json",
    )
    image_path = Path("result_images_dfsph_seg_3D_1")
    if image_path.exists():
        for p in image_path.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    image_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)

    sm = config.get_cfg("simulationMethod")
    if sm is not None and int(sm) != 2:
        print(
            f"[run_simulation_dfsph_segment] Warning: simulationMethod={sm} "
            f"(expected 2). Solver follows JSON."
        )

    export_ply = config.get_cfg("exportPLY")
    export_ply = bool(export_ply) if export_ply is not None else False
    ply_path = None
    if export_ply:
        ply_path = Path("result_ply_dfsph_seg_3D_1")
        if ply_path.exists():
            for p in ply_path.iterdir():
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
        ply_path.mkdir(parents=True, exist_ok=True)

    simulation_time = config.get_cfg("simulationTime")
    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))

    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()

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
            if export_ply and hasattr(solver, "export_ply"):
                solver.export_ply(cnt_output, ply_path)
            cnt_output += 1
        if cnt % 50 == 0:
            print(f"[DFSPH+Seg] Simulation Time = {t:.2f}s")
            print(f"[DFSPH+Seg] Fluid_particle_num = {solver.ps.fluid_particle_num[None]}")
            if hasattr(solver, "ss_seg"):
                print(f"[DFSPH+Seg] segment_num = {solver.ss_seg.segment_num[None]}")
        if t > simulation_time or os.path.exists("stop"):
            break


if __name__ == "__main__":
    main()
