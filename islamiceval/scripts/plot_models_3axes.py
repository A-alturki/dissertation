#!/usr/bin/env python3
"""
plot_models_3axes.py
====================
Per-model figure over data/__FINAL_ANALYSIS_SET__/: every model in the two
vanilla corrected sets (8 + 12) plus the Fanar release set, on three axes:

  1. Source-attribution accuracy  claimed_source segments, correct/(correct+incorrect)
  2. Isnad accuracy               isnad segments, correct/(correct+incorrect)
  3. Relevance                    share of spans labeled relevant (1) — labels exist
                                  only for the Fanar release set (train/dev_task_4)

Bars are annotated with n (labeled segments); models with no labeled segments
on an axis show a dash. Writes outputs/figures/models_3axes.{pdf,png}.
"""
import os, sys, json, csv, math
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "..", "data", "__FINAL_ANALYSIS_SET__")
OUTDIR = os.path.join(HERE, "..", "outputs", "figures")

FANAR = "Fanar (release)"

stats = defaultdict(lambda: {"att_ok": 0, "att_bad": 0, "isn_ok": 0, "isn_bad": 0})

def scan(path, fixed=None):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        m = fixed or r["model"]
        for a in r.get("annotations", []):
            for s in a.get("segments", []):
                t, l = s["type"], s.get("label")
                if t == "claimed_source" and l in ("correct", "incorrect"):
                    stats[m]["att_ok" if l == "correct" else "att_bad"] += 1
                elif t == "isnad" and l in ("correct", "incorrect"):
                    stats[m]["isn_ok" if l == "correct" else "isn_bad"] += 1

scan(os.path.join(FINAL, "vanilla8_corrected.jsonl"))
scan(os.path.join(FINAL, "vanilla12_corrected.jsonl"))
scan(os.path.join(FINAL, "fanar_all_corrected.jsonl"), fixed=FANAR)

rel = rtot = 0
for f in ("train_task_4.tsv", "dev_task_4.tsv"):
    for r in csv.DictReader(open(os.path.join(FINAL, f), encoding="utf-8"), delimiter="\t"):
        lab = r["relevance_label"].strip()
        if lab in ("0", "1"):
            rtot += 1
            rel += lab == "1"

def acc(ok, bad):
    n = ok + bad
    return (100 * ok / n if n else math.nan), n

rows = []   # (model, (att%, att_n), (isn%, isn_n), (rel%, rel_n))
for m, s in stats.items():
    att = acc(s["att_ok"], s["att_bad"])
    isn = acc(s["isn_ok"], s["isn_bad"])
    rv = (100 * rel / rtot, rtot) if m == FANAR else (math.nan, 0)
    rows.append((m, att, isn, rv))
rows.sort(key=lambda r: r[1][0], reverse=True)

print(f"{'model':18}{'att%':>7}{'n':>6}{'isn%':>7}{'n':>5}{'rel%':>7}{'n':>6}")
for m, (a, an), (i, inn), (rv, rn) in rows:
    fmt = lambda v: f"{v:.1f}" if not math.isnan(v) else "  --"
    print(f"{m:18}{fmt(a):>7}{an:>6}{fmt(i):>7}{inn:>5}{fmt(rv):>7}{rn:>6}")

# ── figure ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, SEC, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                        "#898781", "#e1e0d9", "#c3c2b7")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3
PANELS = ["Source attribution", "Isnad", "Relevance"]

models = [r[0] for r in rows]
n_m = len(models)
y = list(range(n_m))[::-1]                          # best at top

fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.34 * n_m + 1.6),
                         dpi=200, sharey=True)
fig.patch.set_facecolor(SURFACE)

for pi, (ax, color, title) in enumerate(zip(axes, COLORS, PANELS)):
    ax.set_facecolor(SURFACE)
    vals = [rows[mi][pi + 1] for mi in range(n_m)]  # (pct, n)
    for yi, (pct, n) in zip(y, vals):
        if math.isnan(pct):
            ax.text(2, yi, "n/a", ha="left", va="center", fontsize=7.5, color=MUTED)
            continue
        ax.barh(yi, pct, height=0.62, color=color, zorder=3)
        label = f"{pct:.1f}"
        if pct > 55:   # label inside the bar, white
            ax.text(pct - 2, yi, label, ha="right", va="center", fontsize=7.8,
                    color="#ffffff", fontweight="bold", zorder=4)
            ax.text(3, yi, f"n={n:,}", ha="left", va="center", fontsize=6.8,
                    color="#ffffff", alpha=0.85, zorder=4)
        else:          # label after the bar, ink
            ax.text(pct + 2, yi, label, ha="left", va="center", fontsize=7.8,
                    color=INK, fontweight="bold", zorder=4)
            ax.text(pct + 6 + 2.3 * len(label), yi, f"n={n:,}", ha="left",
                    va="center", fontsize=6.8, color=MUTED, zorder=4)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.set_xlabel("% correct" if pi < 2 else "% relevant", fontsize=8.5, color=SEC)

axes[0].set_yticks(y)
axes[0].set_yticklabels(models, fontsize=9, color=INK)
axes[0].tick_params(axis="y", length=0)
for lbl in axes[0].get_yticklabels():
    if lbl.get_text() == FANAR:
        lbl.set_fontweight("bold")

fig.text(0.01, 0.005,
         "Accuracies over labeled segments only (N/A excluded). "
         "Relevance labels exist only for the Fanar release set (train+dev).",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))

os.makedirs(OUTDIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"models_3axes.{ext}"),
                facecolor=SURFACE, bbox_inches="tight")
print(f"\nwrote {os.path.join(os.path.relpath(OUTDIR, HERE), 'models_3axes.pdf/.png')}")
