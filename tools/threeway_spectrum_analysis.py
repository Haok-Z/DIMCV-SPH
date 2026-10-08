"""三方横向谱分析：residual_circulation vs DFSPH vs DIMCV。

读取 spectrum3d_mixer_residual_circulation 内的逐 k comparison.csv，
输出：
  1) threeway_perk.csv      —— 可靠区内逐 k 的三方谱值与两两比值
  2) threeway_bands.csv     —— 分波段（大/中/小尺度）积分能量与两两偏差
  3) threeway_distance.csv  —— 两两对数距离矩阵（含分数差）
  4) stdout 摘要
"""

import csv
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(
    ROOT, "results", "spectrum3d_mixer_residual_circulation", "mixer_residual_circulation"
)
OUT_DIR = os.path.join(ROOT, "results", "spectrum3d_mixer_residual_circulation")

FIELDS = ["dimcv", "dfsph", "dfsph_segment"]
LABEL = {"dimcv": "DIMCV", "dfsph": "DFSPH", "dfsph_segment": "SegRC"}

# 波段边界（rad/m）：大尺度 / 中尺度 / 小尺度
BANDS = [(0.0, 30.0, "large"), (30.0, 80.0, "mid"), (80.0, 1e9, "small")]


def read_summary():
    """返回 (region, stage, dx, quantity, field) -> dict(指标)"""
    idx = {}
    path = os.path.join(OUT_DIR, "summary.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("spacing_target"):
                continue  # grid_convergence 行无 spacing
            key = (
                r["region"], r["stage"], float(r["spacing_target"]),
                r["quantity"], r["field"],
            )
            idx[key] = r
    return idx


def read_comparison(region, stage, dx, quantity):
    path = os.path.join(
        CASE_DIR, region, "dx_%g" % dx, stage, quantity, "comparison.csv"
    )
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("k"):
                continue
            try:
                k = float(r["k"])
            except ValueError:
                continue
            rows.append((k, r))
    rows.sort(key=lambda t: t[0])
    return rows


def log_distance(a, b):
    """两个谱之间的对数距离：mean |ln(E_a/E_b)|"""
    vals = []
    for ea, eb in zip(a, b):
        if ea > 0 and eb > 0:
            vals.append(abs(math.log(ea / eb)))
    return statistics.fmean(vals) if vals else float("nan")


def integrate(sel, step_fallback=24.4):
    """对一个波段的 k 序列做梯形积分，返回 {field: energy} 与 {field: centroid}。

    用 center-based  bin 边界：每段 bin 的左右边界取相邻样本中点，
    两端再各向外延半个 bin，使积分区间覆盖整个波段。
    """
    if not sel:
        return None, None
    if len(sel) == 1:
        step = step_fallback
    else:
        step = sel[1][0] - sel[0][0]

    # 构造每个样本的左右边界
    edges = []
    for i, (k, _r) in enumerate(sel):
        if i == 0:
            lo = k - 0.5 * step
        else:
            lo = 0.5 * (sel[i - 1][0] + k)
        if i == len(sel) - 1:
            hi = k + 0.5 * step
        else:
            hi = 0.5 * (k + sel[i + 1][0])
        edges.append((lo, hi, k))

    energy = {f: 0.0 for f in FIELDS}
    moment = {f: 0.0 for f in FIELDS}
    for lo, hi, k in edges:
        if hi <= lo:
            continue
        width = hi - lo
        idx = edges.index((lo, hi, k))
        row = sel[idx][1]
        for f in FIELDS:
            v = float(row[f + "_mean"])
            energy[f] += v * width
            moment[f] += v * width * k
    centroid = {f: (moment[f] / energy[f] if energy[f] > 0 else float("nan")) for f in FIELDS}
    return energy, centroid


def band_energy(rows, kmax):
    """按波段做 band-limited 积分，返回 {band: (energy_dict, centroid_dict)}"""
    out = {}
    pts = [(k, r) for k, r in rows if k <= kmax]
    for lo, hi, name in BANDS:
        sel = [(k, r) for k, r in pts if lo <= k < hi]
        if not sel:
            continue
        e, c = integrate(sel)
        if e is None:
            continue
        out[name] = (e, c)
    return out


def main():
    summary = read_summary()
    perk_rows = []
    band_rows = []
    dist_rows = []

    combos = []
    for region in ("domain", "mixing_roi"):
        for stage in ("driven", "coast"):
            for dx in (0.025, 0.0125):
                for q in ("velocity", "velocity_curl", "vorticity"):
                    rows = read_comparison(region, stage, dx, q)
                    if rows:
                        combos.append((region, stage, dx, q, rows))

    for region, stage, dx, q, rows in combos:
        ref = summary.get((region, stage, dx, q, "dfsph"), {})
        kmax = float(ref.get("reliable_k_max") or 1e9)

        # ---- 1. 逐 k ----
        for k, r in rows:
            if k > kmax:
                continue
            rec = {
                "region": region, "stage": stage, "dx": dx, "quantity": q, "k": k,
            }
            for fld in FIELDS:
                rec[fld] = float(r[fld + "_mean"])
                rec[fld + "_cv"] = (
                    float(r[fld + "_std"]) / float(r[fld + "_mean"])
                    if float(r[fld + "_mean"]) > 0 else float("nan")
                )
            rec["seg_vs_dfsph"] = rec["dfsph_segment"] / rec["dfsph"]
            rec["dimcv_vs_dfsph"] = rec["dimcv"] / rec["dfsph"]
            rec["seg_vs_dimcv"] = rec["dfsph_segment"] / rec["dimcv"]
            perk_rows.append(rec)

        # ---- 2. 波段积分 ----
        bands = band_energy(rows, kmax)
        for lo, hi, name in BANDS:
            if name not in bands:
                continue
            e, cent = bands[name]
            if not isinstance(e, dict) or not e.get("dfsph"):
                continue
            rec = {
                "region": region, "stage": stage, "dx": dx, "quantity": q,
                "band": name, "k_lo": lo, "k_hi": min(hi, kmax),
            }
            for fld in FIELDS:
                rec[fld + "_energy"] = e[fld]
                rec[fld + "_centroid"] = cent.get(fld, float("nan"))
            tot = sum(e[f] for f in FIELDS)
            for fld in FIELDS:
                rec[fld + "_share"] = e[fld] / tot if tot > 0 else float("nan")
            rec["seg_vs_dfsph"] = e["dfsph_segment"] / e["dfsph"]
            rec["dimcv_vs_dfsph"] = e["dimcv"] / e["dfsph"]
            rec["seg_vs_dimcv"] = e["dfsph_segment"] / e["dimcv"]
            band_rows.append(rec)

        # ---- 3. 两两距离 ----
        series = {f: [float(r[f + "_mean"]) for k, r in rows if k <= kmax] for f in FIELDS}
        for a, b in (("dfsph_segment", "dfsph"), ("dimcv", "dfsph"), ("dfsph_segment", "dimcv")):
            d = log_distance(series[a], series[b])
            # 帧间变异导致的距离本底：用相对标准差估算
            dist_rows.append({
                "region": region, "stage": stage, "dx": dx, "quantity": q,
                "pair": "%s vs %s" % (LABEL[a], LABEL[b]),
                "log_distance": d,
            })

    # ---- 写出 ----
    def dump(name, rows, cols=None):
        if not rows:
            return
        path = os.path.join(OUT_DIR, name)
        cols = cols or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", path, len(rows), "rows")

    dump("threeway_perk.csv", perk_rows)
    dump("threeway_bands.csv", band_rows)
    dump("threeway_distance.csv", dist_rows)

    # ---- stdout 摘要 ----
    print("\n" + "=" * 78)
    print("两两对数距离（可靠区内，mean|ln(Ea/Eb)|）")
    print("=" * 78)
    hdr = "%-10s %-7s %-7s %-14s %14s %14s %14s" % (
        "region", "stage", "dx", "quantity", "SegRC-DFSPH", "DIMCV-DFSPH", "SegRC-DIMCV")
    print(hdr)
    for region in ("domain", "mixing_roi"):
        for stage in ("driven", "coast"):
            for dx in (0.025, 0.0125):
                for q in ("velocity", "velocity_curl", "vorticity"):
                    got = {}
                    for r in dist_rows:
                        if (r["region"], r["stage"], r["dx"], r["quantity"]) == (region, stage, dx, q):
                            got[r["pair"]] = r["log_distance"]
                    if not got:
                        continue
                    print("%-10s %-7s %-7g %-14s %14.5f %14.5f %14.5f" % (
                        region, stage, dx, q,
                        got.get("SegRC vs DFSPH", float("nan")),
                        got.get("DIMCV vs DFSPH", float("nan")),
                        got.get("SegRC vs DIMCV", float("nan"))))

    print("\n" + "=" * 78)
    print("稳健性交叉检验：中尺度波段(mid)比值，四种独立估计")
    print("  两种物理量(velocity / velocity_curl) × 两种分析网格(dx)")
    print("=" * 78)
    print("%-10s %-7s %-14s %-7s %11s %11s   %s" % (
        "region", "stage", "quantity", "dx", "SegRC/DF", "DIMCV/DF", "方向"))
    agree = {}
    for region in ("domain", "mixing_roi"):
        for stage in ("driven", "coast"):
            for q in ("velocity", "velocity_curl"):
                line = []
                for dx in (0.025, 0.0125):
                    r = next((x for x in band_rows
                              if x["region"] == region and x["stage"] == stage
                              and x["quantity"] == q and x["dx"] == dx
                              and x["band"] == "mid"), None)
                    if r is None:
                        continue
                    print("%-10s %-7s %-14s %-7g %11.4f %11.4f   %s" % (
                        region, stage, q, dx, r["seg_vs_dfsph"], r["dimcv_vs_dfsph"],
                        "SegRC%sDIMCV" % ("<" if r["seg_vs_dfsph"] < r["dimcv_vs_dfsph"] else ">")))
                    line.append((r["seg_vs_dfsph"], r["dimcv_vs_dfsph"]))
                if line:
                    agree[(region, stage, q)] = line

    # ---- hybrid 上界验算 ----
    print("\n" + "=" * 78)
    print("Hybrid 分解检验：final - pre 的能量差能否由 bs 场解释？")
    print("  若 final = pre + b，则 ΔE = 2<pre·b> + <b·b> ≤ 2√(E_pre·E_bs) + E_bs")
    print("  比值 > 1 表示数学上不可能由叠加 bs 产生")
    print("=" * 78)
    print("%-10s %-7s %-7s %12s %12s %12s %10s %8s" % (
        "region", "stage", "dx", "E_pre", "E_final", "ΔE", "ΔE/上界", "E_bs占比"))
    hyb_rows = []
    for region in ("domain", "mixing_roi"):
        for stage in ("driven", "coast"):
            for dx in (0.025, 0.0125):
                g = lambda f: summary.get((region, stage, dx, "hybrid", f))
                pre, fin, bs = g("pre_feedback_vel"), g("final_vel"), g("segment_bs_vel")
                if not (pre and fin and bs):
                    continue
                ep = float(pre["total_energy"])
                ef = float(fin["total_energy"])
                eb = float(bs["total_energy"])
                de = ef - ep
                bound = 2.0 * math.sqrt(ep * eb) + eb
                ratio = de / bound if bound > 0 else float("nan")
                print("%-10s %-7s %-7g %12.6f %12.6f %12.3e %9.2fx %7.1e" % (
                    region, stage, dx, ep, ef, de, ratio, eb / ep))
                hyb_rows.append({
                    "region": region, "stage": stage, "dx": dx,
                    "E_pre": ep, "E_final": ef, "E_bs": eb,
                    "delta_E": de, "upper_bound": bound, "ratio": ratio,
                    "bs_fraction": eb / ep, "delta_fraction": de / ep,
                })
    with open(os.path.join(OUT_DIR, "threeway_hybrid_bound.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(hyb_rows[0].keys()))
        w.writeheader()
        w.writerows(hyb_rows)
    print("wrote threeway_hybrid_bound.csv")

    print("\n" + "=" * 78)
    print("混合 ROI 逐相位（dx=0.025，velocity + velocity_curl 一致性）")
    print("=" * 78)
    for region in ("mixing_roi",):
        for stage in ("driven", "coast"):
            print("\n--- %s / %s ---" % (region, stage))
            print("%8s  %-28s %-28s" % ("k", "velocity", "velocity_curl"))
            print("%8s  %9s %9s     %9s %9s" % ("", "Seg/DF", "DIM/DF", "Seg/DF", "DIM/DF"))
            va = [r for r in perk_rows if r["region"] == region and r["stage"] == stage
                  and r["dx"] == 0.025 and r["quantity"] == "velocity"]
            vb = [r for r in perk_rows if r["region"] == region and r["stage"] == stage
                  and r["dx"] == 0.025 and r["quantity"] == "velocity_curl"]
            for a in va:
                b = next((x for x in vb if abs(x["k"] - a["k"]) < 1e-6), None)
                if b is None:
                    continue
                print("%8.2f  %9.3f %9.3f     %9.3f %9.3f" % (
                    a["k"], a["seg_vs_dfsph"], a["dimcv_vs_dfsph"],
                    b["seg_vs_dfsph"], b["dimcv_vs_dfsph"]))

    print("\n" + "=" * 78)
    print("逐 k 比值（reliable 内，dx=0.025）")
    print("=" * 78)
    for region in ("domain", "mixing_roi"):
        for stage in ("driven", "coast"):
            for q in ("velocity", "velocity_curl"):
                sel = [r for r in perk_rows
                       if r["region"] == region and r["stage"] == stage
                       and r["dx"] == 0.025 and r["quantity"] == q]
                if not sel:
                    continue
                print("\n--- %s / %s / %s ---" % (region, stage, q))
                print("%10s  %12s %12s %12s   %8s %8s" % (
                    "k", "DIMCV", "DFSPH", "SegRC", "Seg/DF", "DIM/DF"))
                for r in sel:
                    print("%10.2f  %12.3e %12.3e %12.3e   %7.3f %7.3f" % (
                        r["k"], r["dimcv"], r["dfsph"], r["dfsph_segment"],
                        r["seg_vs_dfsph"], r["dimcv_vs_dfsph"]))

    print("\n" + "=" * 78)
    print("帧间变异系数 CV = std/mean（越大越不稳）")
    print("=" * 78)
    print("%-10s %-7s %-7s %-14s %10s %10s %10s" % (
        "region", "stage", "dx", "quantity", "DIMCV", "DFSPH", "SegRC"))
    for region in ("domain", "mixing_roi"):
        for stage in ("driven", "coast"):
            for dx in (0.025, 0.0125):
                for q in ("velocity", "velocity_curl", "vorticity"):
                    sel = [r for r in perk_rows
                           if r["region"] == region and r["stage"] == stage
                           and r["dx"] == dx and r["quantity"] == q]
                    if not sel:
                        continue
                    print("%-10s %-7s %-7g %-14s %10.3f %10.3f %10.3f" % (
                        region, stage, dx, q,
                        statistics.fmean([r["dimcv_cv"] for r in sel]),
                        statistics.fmean([r["dfsph_cv"] for r in sel]),
                        statistics.fmean([r["dfsph_segment_cv"] for r in sel])))


if __name__ == "__main__":
    main()
