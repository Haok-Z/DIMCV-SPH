import argparse
import shutil
import time
from pathlib import Path

import taichi as ti

from segment_config import SegmentConfig
from segment_system import SegmentSystem
from segment_solver import SegmentSolver
from segment_export import SegmentExporter


ti.init(
    arch=ti.cuda,
    device_memory_fraction=0.5,
    debug=False,
    random_seed=int(time.time()),
    kernel_profiler=False,
)


def clear_dir(path: Path):
    if path.exists():
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Standalone Vortex Segment Simulation")
    parser.add_argument("--scene_file", default="", help="scene file path")
    args = parser.parse_args()

    cfg = SegmentConfig(scene_file_path=args.scene_file)
    simulation_time = float(cfg.get_cfg("simulationTime", 5.0))
    dt = float(cfg.get_cfg("timeStepSize", 0.002))
    export_interval = int(cfg.get_cfg("exportInterval", 8))
    export_ply = bool(cfg.get_cfg("exportPLY", True))
    export_images = bool(cfg.get_cfg("exportImages", True))

    out_ply = Path("result_segment_ply_TripleLine")
    out_img = Path("result_segment_images_TripleLine")
    if export_ply:
        clear_dir(out_ply)
    if export_images:
        clear_dir(out_img)

    ss = SegmentSystem(cfg)
    solver = SegmentSolver(ss)
    exporter = SegmentExporter(ss)

    solver.initialize()

    t = 0.0
    frame = 0
    step_id = 0
    while t < simulation_time:
        solver.step()
        step_id += 1
        t += dt

        if step_id % export_interval == 0:
            if export_ply:
                exporter.export_segments_ply(frame, out_ply)
            if export_images:
                exporter.export_segments_png(frame, out_img)
            frame += 1

        if step_id % 50 == 0:
            print(f"[Segment] t={t:.3f}s, active_segments~{ss.segment_num[None]}")


if __name__ == "__main__":
    main()
