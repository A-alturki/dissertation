"""
Stage 6: Empirical Analysis — Tueki's Dissertation
Benchmarks all models on IslamicEval 2026 across Subtasks A, B, and C.

Dimensions:
  - Per-subtask performance
  - Per-error-class breakdown
  - Model size scaling (within-family)
  - Model family comparison (Arabic-centric vs multilingual)
  - Quran vs Hadith performance gap
  - Dense vs MoE architecture comparison

Usage:
    python analyze.py --gold-dir ../outputs/annotations/ --pred-dir ../outputs/answers/
"""

import os, json, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

SUBTASK_B_LABELS = [
    "correct",
    "misattributed_source",
    "fabricated",
    "paraphrased_as_verbatim",
    "variant_attested_matn",
    "fabricated_isnad",
]

MODEL_FAMILIES = {
    "Arabic-centric": ["allam-7b", "jais-13b", "acegpt-8b", "silma-9b"],
    "Qwen":           ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"],
    "Llama":          ["llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b"],
    "Gemma":          ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b"],
    "Mistral":        ["mistral-7b", "mistral-small-24b", "mixtral-8x7b"],
    "Other":          ["phi-4-14b", "deepseek-v3", "command-r-7b"],
}

MODEL_PARAMS = {
    "allam-7b": 7, "jais-13b": 13, "acegpt-8b": 8, "silma-9b": 9,
    "qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4, "qwen3-8b": 8,
    "qwen3-14b": 14, "qwen3-32b": 32,
    "llama-3.2-3b": 3, "llama-3.1-8b": 8, "llama-3.3-70b": 70,
    "gemma-3-4b": 4, "gemma-3-12b": 12, "gemma-3-27b": 27,
    "mistral-7b": 7, "mistral-small-24b": 24, "mixtral-8x7b": 56,
    "phi-4-14b": 14, "deepseek-v3": 685, "command-r-7b": 7,
}


def get_family(model_name: str) -> str:
    for family, members in MODEL_FAMILIES.items():
        if model_name in members:
            return family
    return "Other"


def flatten_spans(items: list, source_filter: str = None) -> tuple[list, list]:
    gold_labels, pred_labels = [], []
    for item in items:
        gold = item.get("annotations", [])
        pred = item.get("predicted_annotations", [])
        for g, p in zip(gold, pred):
            if source_filter and g.get("source_type") != source_filter:
                continue
            gold_labels.append(g["label"])
            pred_labels.append(p["label"])
    return gold_labels, pred_labels


def compute_metrics(gold_labels: list, pred_labels: list) -> dict:
    if not gold_labels:
        return {}
    return {
        "macro_f1": f1_score(gold_labels, pred_labels, average="macro",
                             labels=SUBTASK_B_LABELS, zero_division=0),
        "per_label": {
            label: f1_score(
                [1 if l == label else 0 for l in gold_labels],
                [1 if l == label else 0 for l in pred_labels],
                zero_division=0,
            )
            for label in SUBTASK_B_LABELS
        },
    }


def plot_heatmap(df: pd.DataFrame, title: str, output_path: str):
    plt.figure(figsize=(12, max(6, len(df) * 0.4)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Empirical analysis (Stage 6)")
    parser.add_argument("--gold-dir",   default="../outputs/annotations/")
    parser.add_argument("--pred-dir",   default="../outputs/answers/")
    parser.add_argument("--output-dir", default="../outputs/analysis/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for fname in sorted(os.listdir(args.gold_dir)):
        if not fname.endswith("_agreed.json"):
            continue
        model_name = fname.replace("_agreed.json", "")
        gold_path  = os.path.join(args.gold_dir, fname)

        with open(gold_path, encoding="utf-8") as f:
            data = json.load(f)

        gold_all,  pred_all  = flatten_spans(data)
        gold_qur,  pred_qur  = flatten_spans(data, source_filter="quran")
        gold_had,  pred_had  = flatten_spans(data, source_filter="hadith")

        m_all = compute_metrics(gold_all,  pred_all)
        m_qur = compute_metrics(gold_qur,  pred_qur)
        m_had = compute_metrics(gold_had,  pred_had)

        row = {
            "model":          model_name,
            "family":         get_family(model_name),
            "params_b":       MODEL_PARAMS.get(model_name, None),
            "macro_f1_all":   m_all.get("macro_f1", None),
            "macro_f1_quran": m_qur.get("macro_f1", None),
            "macro_f1_hadith":m_had.get("macro_f1", None),
        }
        for label in SUBTASK_B_LABELS:
            row[f"f1_{label}"] = m_all.get("per_label", {}).get(label, None)

        rows.append(row)
        print(f"{model_name:20s}  macro_f1={row['macro_f1_all']:.3f}  "
              f"quran={row['macro_f1_quran']:.3f}  hadith={row['macro_f1_hadith']:.3f}")

    if not rows:
        print("No annotated results found. Run annotate.py first.")
        return

    df = pd.DataFrame(rows).sort_values("macro_f1_all", ascending=False)
    df.to_csv(os.path.join(args.output_dir, "subtask_b_results.csv"), index=False)

    # Per-label heatmap
    label_cols = [f"f1_{l}" for l in SUBTASK_B_LABELS]
    heatmap_df = df.set_index("model")[label_cols].rename(
        columns=lambda c: c.replace("f1_", "")
    )
    plot_heatmap(heatmap_df, "Subtask B: F1 per error class",
                 os.path.join(args.output_dir, "per_label_heatmap.png"))

    # Family comparison
    family_df = df.groupby("family")["macro_f1_all"].mean().sort_values(ascending=False)
    family_df.to_csv(os.path.join(args.output_dir, "family_comparison.csv"))

    print(f"\nResults saved -> {args.output_dir}")
    print(df[["model", "family", "params_b", "macro_f1_all"]].to_string(index=False))


if __name__ == "__main__":
    main()
