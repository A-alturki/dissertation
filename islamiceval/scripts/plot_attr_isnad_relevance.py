#!/usr/bin/env python3
"""
plot_attr_isnad_relevance.py
============================
Aggregate figure over the COMPLETE Fanar set (train+dev+test) in
data/__FINAL_ANALYSIS_SET__/:

  - source-attribution accuracy : claimed_source segments, correct/(correct+incorrect)
                                  (N/A = no source claimed, excluded)
  - isnad accuracy              : isnad segments, correct/(correct+incorrect)
  - relevance accuracy          : share of spans labeled relevant (1) in
                                  train_task_4.tsv + dev_task_4.tsv

Writes outputs/figures/attr_isnad_relevance.{pdf,png} and prints the numbers.
"""
import os, sys, json, csv

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "..", "data", "__FINAL_ANALYSIS_SET__")
OUTDIR = os.path.join(HERE, "..", "outputs", "figures")

def seg_counts(jsonl_path):
    ok = {"claimed_source": 0, "isnad": 0}
    bad = {"claimed_source": 0, "isnad": 0}
    for line in open(jsonl_path, encoding="utf-8"):
        if not line.strip():
            continue
        for a in json.loads(line).get("annotations", []):
            for s in a.get("segments", []):
                t = s.get("type")
                if t in ok:
                    if s.get("label") == "correct":
                        ok[t] += 1
                    elif s.get("label") == "incorrect":
                        bad[t] += 1
    return ok, bad

def relevance_counts(paths):
    rel = tot = 0
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            lab = r["relevance_label"].strip()
            if lab in ("0", "1"):
                tot += 1
                rel += lab == "1"
    return rel, tot

ok, bad = seg_counts(os.path.join(FINAL, "fanar_all_corrected.jsonl"))
rel, rtot = relevance_counts([os.path.join(FINAL, "train_task_4.tsv"),
                              os.path.join(FINAL, "dev_task_4.tsv")])

metrics = [
    ("Source attribution", ok["claimed_source"], ok["claimed_source"] + bad["claimed_source"]),
    ("Isnad",              ok["isnad"],          ok["isnad"] + bad["isnad"]),
    ("Relevance",          rel,                  rtot),
]
for name, k, n in metrics:
    print(f"{name:20} {k:>5}/{n:<5} = {100*k/n:.2f}%")

# ── figure ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

names = [m[0] for m in metrics]
vals = [100 * k / n for _, k, n in metrics]
x = range(len(metrics))
bars = ax.bar(x, vals, width=0.52, color=BLUE, zorder=3)

for xi, v, (_, k, n) in zip(x, vals, metrics):
    ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", va="bottom",
            fontsize=10.5, color=INK, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{name}\n(n = {n:,})" for name, _, n in metrics],
                   fontsize=9.5, color=INK)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("Accuracy (%)", fontsize=9.5, color=INK)
ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASE)

fig.tight_layout()
os.makedirs(OUTDIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"attr_isnad_relevance.{ext}"),
                facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {os.path.join(os.path.relpath(OUTDIR, HERE), 'attr_isnad_relevance.pdf/.png')}")
