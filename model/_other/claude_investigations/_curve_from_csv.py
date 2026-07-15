"""Pooled V-vs-in-degree curve per topology from a centrality_perf CSV.
Usage: py _curve_from_csv.py [csv-filename-in-output-dir]"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

fname = sys.argv[1] if len(sys.argv) > 1 else "centrality_perf_agents.csv"
rows = list(csv.DictReader(open(Path(__file__).parent / "output" / fname)))
by = defaultdict(list)
for r in rows:
    by[(r['topo'], int(r['indeg']))].append(float(r['meanV']))

TOPOS = ['Scale Free', 'Random', 'Small World', 'Hierarchical']
print('pooled mean V by in-degree (n agents); 10 seeds x 80 agents per topology')
hdr = 'indeg'.rjust(6) + ''.join(t.rjust(22) for t in TOPOS)
print(hdr)
for k in range(0, 27):
    if not any(by.get((t, k)) for t in TOPOS):
        continue
    cells = []
    for t in TOPOS:
        v = by.get((t, k))
        cells.append(f'{np.mean(v):6.1f} ({len(v):3d})' if v else '-'.rjust(12))
    print(str(k).rjust(6) + ''.join(c.rjust(22) for c in cells))

print()
for t in TOPOS:
    ks = [int(r['indeg']) for r in rows if r['topo'] == t]
    hi = [float(r['meanV']) for r in rows if r['topo'] == t and int(r['indeg']) >= 10]
    lo = [float(r['meanV']) for r in rows if r['topo'] == t and int(r['indeg']) <= 1]
    print(f"{t:<13} indeg range {min(ks)}-{max(ks)};  "
          f"V(indeg>=10) = {np.mean(hi):5.2f} (n={len(hi)});  "
          f"V(indeg<=1)  = {np.mean(lo):5.2f} (n={len(lo)})")

# JSON-ish means for plotting
print("\n--- means for plotting ---")
for t in TOPOS:
    pts = [(k, round(float(np.mean(by[(t, k)])), 2), len(by[(t, k)]))
           for k in range(0, 27) if (t, k) in by]
    print(t, '=', pts)
