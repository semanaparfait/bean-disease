"""Turn locust/results/*_stats.csv into a markdown table + chart (locust/results/summary.md, latency.png)."""
import re
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "locust" / "results"
rows = []
for f in sorted(RES.glob("*_stats.csv")):
    df = pd.read_csv(f)
    r = df[df["Name"] == "/predict"].iloc[0]
    n = re.search(r"(\d+)", f.stem.replace("_stats", "")).group(1)
    rows.append({"Setup": f.stem.replace("_stats", ""), "Replicas": int(n), "Requests": int(r["Request Count"]),
                 "Failures": int(r["Failure Count"]), "Avg (ms)": round(r["Average Response Time"]),
                 "Median (ms)": round(r["Median Response Time"]), "p95 (ms)": round(r["95%"]),
                 "p99 (ms)": round(r["99%"]), "Max (ms)": round(r["Max Response Time"]),
                 "Throughput (req/s)": round(r["Requests/s"], 2)})
if not rows:
    sys.exit("no results found in " + str(RES))
table = pd.DataFrame(rows).sort_values(["Setup"])
md = table.to_markdown(index=False)
(RES / "summary.md").write_text(md + "\n")
print(md)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.6))
x = range(len(table))
ax.bar([i - 0.2 for i in x], table["Median (ms)"], width=0.38, color="#2a78d6", label="median")
ax.bar([i + 0.2 for i in x], table["p95 (ms)"], width=0.38, color="#eb6834", label="p95")
ax.set_xticks(list(x), [f"{n} replica" + ("s" if n > 1 else "") for n in table["Replicas"]])
ax.set_ylabel("response time (ms)"); ax.set_title("/predict latency vs number of replicas", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(RES / "latency.png", dpi=130)
