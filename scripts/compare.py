"""Print a comparison table of every results/*.summary.json.

    python scripts/compare.py [results_dir]
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
    s = json.load(open(p))
    rows.append(s)
if not rows:
    sys.exit(f"no summaries in {d}")

cols = ["name", "n", "pass_rate", "mean_reward", "mean_calls", "finished_cleanly_rate", "mean_judge", "errors", "wall_seconds"]
w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
print(" | ".join(c.ljust(w[c]) for c in cols))
print("-+-".join("-" * w[c] for c in cols))
for r in rows:
    print(" | ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))

cats = sorted({c for r in rows for c in r.get("by_category", {})})
if cats:
    print("\nper-category pass rate")
    names = [r["name"] for r in rows]
    cw = max(len(c) for c in cats)
    print("category".ljust(cw), " | ", " | ".join(n.ljust(8) for n in names))
    for c in cats:
        print(c.ljust(cw), " | ", " | ".join(str(r.get("by_category", {}).get(c, {}).get("pass_rate", "-")).ljust(8) for r in rows))
