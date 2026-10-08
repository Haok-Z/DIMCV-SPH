import csv
from pathlib import Path

p = Path("results/DIM_von_karman_vortex_3d_dimcv_rotating_mixer_unified_coarse/dimcv/flow_statistics.csv")
with p.open(newline="", encoding="utf-8") as f:
    rows = [(float(r["time"]), float(r["kinetic_energy_total"])) for r in csv.DictReader(f)]

lines = []
for i, (t, ke) in enumerate(rows):
    if i % 10 == 0 or i == len(rows) - 1:
        lines.append(f"{t:.4f},{ke:.6f}")
print("SERIES")
print(";".join(lines))

sample_t = []
for a, b in [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [4.0, 6.0], [8.0, 10.0]]:
    n = int((b - a) / 0.25 + 1e-9)
    sample_t.extend(round(a + 0.25 * i, 4) for i in range(n + 1))
sample_t = sorted(set(sample_t))
print("SAMPLES")
out = []
for t in sample_t:
    _, ke = min(rows, key=lambda x: abs(x[1 - 1] - t))
    out.append(f"{t:.2f},{ke:.6f}")
print(";".join(out))
