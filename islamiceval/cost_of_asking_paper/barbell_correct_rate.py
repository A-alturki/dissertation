"""
barbell_correct_rate.py

Companion to the cost-of-asking decomposition: a barbell (dumbbell) chart of
every model's citation accuracy under the two prompts.

The decomposition figure shows the pooled MECHANISM (selection / quality /
volume). It cannot show per-model movement, because pooling is the whole point
of it. This chart shows the other half: where each model starts under the
implicit prompt, where it lands under the explicit one, and how far it moved.

Reads the CSV the decomposition writes, so the two figures can never disagree:
    outputs/cost_of_asking_3class_permodel.csv   (--axis fidelity)

Outputs
-------
new_paper/outputs/fig2_barbell_correct.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NEW_PAPER_DIR = Path(__file__).resolve().parent
OUT_DIR = NEW_PAPER_DIR / "outputs"
CSV_PATH = OUT_DIR / "cost_of_asking_3class_permodel.csv"

FRONTIER = {"gpt-5.4", "gemini-3.1-pro"}
DISPLAY = {
    "llama-3.1-8b": "Llama-3.1-8B", "phi-3.5": "Phi-3.5-mini",
    "gemma-3-12b": "Gemma-3-12B", "qwen3-8b": "Qwen3-8B",
    "acegpt-8b": "AceGPT-v2-8B", "command-r-7b": "Command-R-7B",
    "mistral-7b": "Mistral-7B", "allam-7b": "ALLaM-7B",
    "fanar-1-9b": "Fanar-1-9B", "jais-13b": "Jais-13B",
    "gpt-5.4": "GPT-5.4$^\\dagger$", "gemini-3.1-pro": "Gemini3.1-Pro$^\\dagger$",
}

# Same two hues as the fidelity figure and taxonomy_tree.pdf, so "green = good
# direction, red = bad direction" means the same thing across every figure.
# Direction is never carried by colour alone: the dots' left-right order and
# the arrowhead both state it.
UP, DOWN = "#77be80", "#a33a36"
IMP_FACE, EXP_EDGE = "#ffffff", "#2b2b2b"
INK, MUTED, GRID = "#1a1a19", "#55544f", "#e4e3df"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv, encoding="utf-8"))
            if not r["model"].startswith("__")]
    data = [(r["model"], float(r["rate_imp__correct"]), float(r["rate_exp__correct"]))
            for r in rows]
    data.sort(key=lambda t: t[2])  # by explicit rate, worst at the bottom

    fig, ax = plt.subplots(figsize=(3.35, 3.9))
    ys = range(len(data))

    for y, (m, imp, exp) in zip(ys, data):
        colour = UP if exp >= imp else DOWN
        # A plain segment, not an arrow. FancyArrowPatch's shrink overshoots on
        # near-zero-length arrows and drew ALLaM's head outside its own range;
        # the open-vs-filled dots already say which end is which.
        ax.hlines(y, min(imp, exp), max(imp, exp), color=colour, linewidth=2.0, zorder=2)
        ax.plot(imp, y, "o", ms=5.0, mfc=IMP_FACE, mec=EXP_EDGE, mew=1.1, zorder=3)
        ax.plot(exp, y, "o", ms=5.0, mfc=EXP_EDGE, mec=EXP_EDGE, zorder=4)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([DISPLAY[m] for m, _, _ in data], fontsize=7.4)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlim(-4, 104)
    ax.set_xlabel("Citations that are textually correct (%)", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=7.4, color="#b9b8b3")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b3")

    handles = [
        Line2D([], [], marker="o", linestyle="none", ms=5.2, mfc=IMP_FACE,
               mec=EXP_EDGE, mew=1.1, label="Implicit"),
        Line2D([], [], marker="o", linestyle="none", ms=5.2, mfc=EXP_EDGE,
               mec=EXP_EDGE, label="Explicit"),
        Line2D([], [], color=DOWN, linewidth=1.8, label="Worse when asked"),
        Line2D([], [], color=UP, linewidth=1.8, label="Better when asked"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, fontsize=7.0, handletextpad=0.5,
              columnspacing=1.2, labelspacing=0.35)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "fig2_barbell_correct.png"
    pdf = args.out_dir / "fig2_barbell_correct.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {png}\nWrote {pdf}\n")
    for m, imp, exp in reversed(data):
        print(f"  {DISPLAY[m].replace('$^\\dagger$', '*'):18s} {imp:5.1f} -> {exp:5.1f}  ({exp - imp:+5.1f})")


if __name__ == "__main__":
    main()
