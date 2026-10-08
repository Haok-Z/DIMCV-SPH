"""绘制一个或多个实验的流场统计时间曲线。"""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_right
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
if CHINESE_FONT_PATH.exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


METRICS = {
    "vorticity_mean_magnitude": "平均涡量模长",
    "vorticity_mean_abs_z": "平均 |omega_z|",
    "vorticity_rms_magnitude": "涡量 RMS",
    "vortex_particle_fraction": "阈值涡粒子占比",
    "kinetic_energy_total": "总动能",
    "kinetic_energy_mean_per_particle": "平均单粒子动能",
    "vortex_kinetic_energy_total": "阈值涡粒子总动能",
    "vortex_kinetic_energy_mean_per_particle": "阈值涡粒子平均动能",
}

PARTICLE_FIELDS = {
    "frame",
    "time",
    "particle_id",
    "vorticity_field",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
    "vorticity_magnitude",
    "kinetic_energy",
}

METHOD_COLORS = {
    "DFSPH": "#2F6BFF",
    "DIMCV": "#F2C94C",
    "DFSPH+Segment": "#27AE60",
}
METHOD_ALIASES = {
    "dfsph": "DFSPH",
    "dimcv": "DIMCV",
    "dfsph_segment": "DFSPH+Segment",
    "dfsph+segment": "DFSPH+Segment",
    "df sph segment": "DFSPH+Segment",
}


def read_statistics(path: Path, vorticity_threshold: float | None):
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if PARTICLE_FIELDS.issubset(fieldnames):
            return aggregate_particle_statistics(reader, path, vorticity_threshold)
        rows = list(reader)
    if not rows:
        raise ValueError(f"统计文件为空: {path}")
    if vorticity_threshold is not None:
        raise ValueError(
            f"{path} 是逐帧汇总 CSV，不能在绘图时重新筛选阈值；"
            "请传入 flow_vorticity_particles.csv。"
        )
    return rows


def aggregate_particle_statistics(reader, path: Path, vorticity_threshold: float | None):
    """由全域逐粒子 CSV 在绘图时精确重算每帧统计量。"""
    threshold = 0.0 if vorticity_threshold is None else vorticity_threshold
    frames = {}
    for row in reader:
        try:
            frame = int(row["frame"])
            time = float(row["time"])
            magnitude = float(row["vorticity_magnitude"])
            omega_z = float(row["vorticity_z"])
            kinetic_energy = float(row["kinetic_energy"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"逐粒子统计 CSV 包含无效数值: {path}") from error

        stats = frames.setdefault(
            frame,
            {
                "frame": frame,
                "time": time,
                "vorticity_field": row["vorticity_field"],
                "fluid_count": 0,
                "vortex_particle_count": 0,
                "signed_z_sum": 0.0,
                "abs_z_sum": 0.0,
                "magnitude_sum": 0.0,
                "magnitude_squared_sum": 0.0,
                "kinetic_energy_total": 0.0,
                "vortex_kinetic_energy_total": 0.0,
            },
        )
        stats["fluid_count"] += 1
        stats["kinetic_energy_total"] += kinetic_energy
        if magnitude >= threshold:
            stats["vortex_particle_count"] += 1
            stats["signed_z_sum"] += omega_z
            stats["abs_z_sum"] += abs(omega_z)
            stats["magnitude_sum"] += magnitude
            stats["magnitude_squared_sum"] += magnitude * magnitude
            stats["vortex_kinetic_energy_total"] += kinetic_energy

    if not frames:
        raise ValueError(f"统计文件为空: {path}")

    rows = []
    for stats in sorted(frames.values(), key=lambda item: (item["time"], item["frame"])):
        fluid_count = stats["fluid_count"]
        vortex_count = stats["vortex_particle_count"]
        rows.append(
            {
                "frame": stats["frame"],
                "time": stats["time"],
                "fluid_count": fluid_count,
                "vorticity_field": stats["vorticity_field"],
                "vorticity_threshold": threshold,
                "vortex_particle_count": vortex_count,
                "vortex_particle_fraction": vortex_count / fluid_count if fluid_count else 0.0,
                "vorticity_mean_signed_z": stats["signed_z_sum"] / vortex_count if vortex_count else 0.0,
                "vorticity_mean_abs_z": stats["abs_z_sum"] / vortex_count if vortex_count else 0.0,
                "vorticity_mean_magnitude": stats["magnitude_sum"] / vortex_count if vortex_count else 0.0,
                "vorticity_rms_magnitude": math.sqrt(
                    stats["magnitude_squared_sum"] / vortex_count
                ) if vortex_count else 0.0,
                "kinetic_energy_total": stats["kinetic_energy_total"],
                "kinetic_energy_mean_per_particle": stats["kinetic_energy_total"] / fluid_count if fluid_count else 0.0,
                "vortex_kinetic_energy_total": stats["vortex_kinetic_energy_total"],
                "vortex_kinetic_energy_mean_per_particle": (
                    stats["vortex_kinetic_energy_total"] / vortex_count if vortex_count else 0.0
                ),
            }
        )
    return rows


def normalize_method(path: Path) -> str:
    key = path.parent.name.strip().lower().replace(" ", " ")
    method = METHOD_ALIASES.get(key)
    if method is None:
        raise ValueError(
            f"无法从 CSV 所在目录识别方法: {path.parent.name}；"
            f"支持目录名: {', '.join(METHOD_ALIASES)}"
        )
    return method


def interpolate(times, values, target_times):
    """按时间线性插值，目标时间超出范围或数据无效时返回 NaN。"""
    pairs = sorted(
        (float(t), float(v))
        for t, v in zip(times, values)
        if math.isfinite(float(t)) and math.isfinite(float(v))
    )
    if not pairs:
        return [math.nan] * len(target_times)
    unique = {}
    for time, value in pairs:
        unique[time] = value
    source_times = sorted(unique)
    source_values = [unique[time] for time in source_times]
    result = []
    for target in target_times:
        if target < source_times[0] or target > source_times[-1]:
            result.append(math.nan)
            continue
        index = bisect_right(source_times, target)
        if index == 0:
            result.append(source_values[0])
        elif index == len(source_times):
            result.append(source_values[-1])
        else:
            left_time, right_time = source_times[index - 1], source_times[index]
            left_value, right_value = source_values[index - 1], source_values[index]
            if right_time == left_time:
                result.append(right_value)
            else:
                ratio = (target - left_time) / (right_time - left_time)
                result.append(left_value + ratio * (right_value - left_value))
    return result


def main():
    parser = argparse.ArgumentParser(description="绘制流场统计对比曲线")
    parser.add_argument("csv_files", nargs="+", help="一个或多个 flow_statistics.csv 或 flow_vorticity_particles.csv 文件")
    parser.add_argument("--metric", default="vorticity_mean_magnitude", choices=METRICS)
    parser.add_argument("--all-metrics", action="store_true", help="一次导出全部预定义统计指标")
    parser.add_argument("--labels", nargs="*", help="每条曲线的显示名称，数量须与 CSV 文件相同")
    parser.add_argument("--output", default="flow_statistics_comparison", help="输出 PNG 路径或输出目录")
    parser.add_argument("--linewidth", type=float, default=1.2, help="曲线线宽，默认 1.2")
    parser.add_argument("--normalize", action="store_true", help="以 DFSPH 为基线输出相对比值曲线")
    parser.add_argument(
        "--vorticity-threshold",
        type=float,
        help="逐粒子 CSV 的涡粒子判定阈值；未指定时使用 0.0。汇总 CSV 不支持此选项。",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.csv_files):
        parser.error("--labels 的数量必须与 CSV 文件数量一致")
    if args.linewidth <= 0:
        parser.error("--linewidth 必须大于 0")
    if args.vorticity_threshold is not None and args.vorticity_threshold < 0:
        parser.error("--vorticity-threshold 不能小于 0")

    datasets = []
    methods = {}
    for index, filename in enumerate(args.csv_files):
        path = Path(filename)
        try:
            rows = read_statistics(path, args.vorticity_threshold)
        except ValueError as error:
            parser.error(str(error))
        missing = [metric for metric in METRICS if metric not in rows[0]]
        if missing:
            parser.error(f"统计文件缺少字段 {missing}: {path}")
        try:
            method = normalize_method(path)
        except ValueError as error:
            parser.error(str(error))
        if method in methods:
            parser.error(f"输入中重复出现方法 {method}: {methods[method]} 和 {path}")
        methods[method] = path
        label = args.labels[index] if args.labels else method
        times = [float(row["time"]) for row in rows]
        datasets.append({"label": label, "method": method, "time": times, "rows": rows})

    if args.normalize:
        required = set(METHOD_COLORS)
        missing_methods = required - set(methods)
        if missing_methods:
            parser.error(f"归一化模式缺少方法: {sorted(missing_methods)}")

    metrics = METRICS if args.all_metrics else {args.metric: METRICS[args.metric]}
    output = Path(args.output)
    output_dir = output if args.all_metrics else output.parent
    if args.normalize:
        output_dir = output_dir / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, metric_name in metrics.items():
        plot_datasets = datasets
        if args.normalize:
            baseline = next(item for item in datasets if item["method"] == "DFSPH")
            target_times = baseline["time"]
            base_values = [float(row[metric]) for row in baseline["rows"]]
            plot_datasets = []
            for item in datasets:
                if item["method"] == "DFSPH":
                    values = [1.0] * len(target_times)
                else:
                    raw_values = [float(row[metric]) for row in item["rows"]]
                    aligned = interpolate(item["time"], raw_values, target_times)
                    values = [
                        value / base if math.isfinite(value) and math.isfinite(base) and base != 0 else math.nan
                        for value, base in zip(aligned, base_values)
                    ]
                plot_datasets.append({**item, "time": target_times, "values": values})

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        for item in plot_datasets:
            values = item.get("values")
            if values is None:
                values = [float(row[metric]) for row in item["rows"]]
            ax.plot(
                item["time"], values,
                color=METHOD_COLORS[item["method"]],
                linewidth=args.linewidth,
                label=item["label"],
            )
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel(f"{metric_name} / DFSPH" if args.normalize else metric_name)
        ax.set_title(f"{metric_name} 相对 DFSPH 比值" if args.normalize else f"{metric_name} 随时间变化")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = output_dir / f"{metric}.png" if args.all_metrics else output_dir / output.name
        fig.savefig(target)
        plt.close(fig)
        print(f"曲线已写入: {target}")


if __name__ == "__main__":
    main()
