"""
build_8class_table.py

Per-model distribution over the EIGHT leaf classes of the citation taxonomy
(figures/taxonomy_tree.pdf), as percentage and count, for Section 5.1
"Fine-Grained Benchmark Results".

The eight leaves
----------------
The tree crosses the textual-fidelity axis (correct / corrupted / fabricated)
with the attributional-accuracy axis, giving eight terminal classes:

  textual = correct     attribution correct    -> Correct Citation
                        attribution incorrect  -> Misattribution
                        none offered           -> Unsourced Quotation
  textual = corrupted   attribution correct    -> Textual Corruption
                        attribution incorrect  -> Corrupted Misattribution
                        none offered           -> Unsourced Corruption
  textual = fabricated  attribution offered    -> Attributed Fabrication
                        none offered           -> Bare Fabrication

The tree prints the two "unsourced" leaves with the same words ("Unsourced
Quotation"); they are distinct leaves on different branches, so the corrupted
one is named "Unsourced Corruption" here to keep the table columns unique.

How attribution state is read off the data
------------------------------------------
Attribution lives in the isnad and claimed_source segments, which carry
correct / incorrect / none / N/A. Two separate things both mean "the model
offered no source", and BOTH must map to the unsourced branch:

  * the citation has no attribution segment at all (the released files only
    emit a segment where the citation supplies one) -- the common case, e.g.
    127 correct-text citations in fanar_main; and
  * an attribution segment explicitly labelled "none" -- the rare case, 27.

"N/A" is the deterministic fabrication gate: the segment exists (the model DID
name a source) but there is nothing to attribute, so it is not a judgement of
the attribution and must not be read as "no source offered". A fabricated
citation with an N/A segment is therefore an Attributed Fabrication, and one
with no attribution segment at all is a Bare Fabrication.

A Hadith citation can carry two attribution components (isnad AND
claimed_source) which are judged independently. The citation counts as
correctly attributed only if EVERY judged component is correct; one wrong
component makes it a misattribution.

Conditions
----------
The twelve original models were run under both prompts; the eight newer
open-weight models exist ONLY under the implicit prompt (a2a_8models is
byte-identical to vanilla8_corrected). So:

  --condition implicit : all 20 models, rule between the 12 and the 8. The
                         only internally consistent "12 + 8" table.
  --condition explicit : the 12 original models only. Putting the 8 newer
                         models in an explicit-condition table is impossible,
                         and mixing conditions across the rule would compare
                         models that were asked different questions.

Outputs
-------
new_paper/outputs/table_8class_<condition>.tex
new_paper/outputs/table_8class_<condition>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NEW_PAPER_DIR = Path(__file__).resolve().parent
OUT_DIR = NEW_PAPER_DIR / "outputs"

TEXT_TYPES = ("Ayah", "matn")
ATTR_TYPES = ("isnad", "claimed_source")

# (key, column header, textual-fidelity group)
LEAVES = [
    # Headers name the taxonomy LEAVES, matching taxonomy_tree.pdf, so a
    # reader moving between the figure and the table sees the same words.
    # The group header supplies the fidelity half of each leaf name, which is
    # why "Misattr." and "Unsourced" each appear twice -- under "Correct text"
    # that is the Misattribution leaf, under "Corrupted text" the Corrupted
    # Misattribution leaf. Full names ("Attributed Fabrication") do not fit;
    # the tree itself already abbreviates "Corrupted Misattr.".
    ("correct_citation",       "Correct",     "correct"),
    ("misattribution",         "Misattr.",    "correct"),
    ("unsourced_quotation",    "Unsourced",   "correct"),
    ("textual_corruption",     "Corruption",  "corrupted"),
    ("corrupted_misattr",      "Misattr.",    "corrupted"),
    ("unsourced_corruption",   "Unsourced",   "corrupted"),
    ("attributed_fabrication", "Attributed",  "fabricated"),
    ("bare_fabrication",       "Bare",        "fabricated"),
]
LEAF_KEYS = [k for k, _, _ in LEAVES]

LEAF_GROUP = {k: g for k, _, g in LEAVES}

# Each fidelity group is printed as an "All" column -- the model's overall rate
# for that textual-fidelity class, i.e. the sum of the group's leaves -- followed
# by the leaves that decompose it. The three All columns sum to 100%.
GROUPS = [
    ("correct", "Correct text"),
    ("corrupted", "Corrupted text"),
    ("fabricated", "Fabricated"),
]


def column_specs():
    """-> [(kind, key, header, group)] in printed order; kind is 'all'|'leaf'."""
    cols = []
    for g, _ in GROUPS:
        cols.append(("all", g, "All", g))
        for k, header, kg in LEAVES:
            if kg == g:
                cols.append(("leaf", k, header, g))
    return cols


COLUMNS = column_specs()
GROUP_SPAN = [(title, sum(1 for c in COLUMNS if c[3] == g)) for g, title in GROUPS]

ORIGINAL_OPEN = [
    "acegpt-8b", "allam-7b", "command-r-7b", "fanar-1-9b", "gemma-3-12b",
    "jais-13b", "llama-3.1-8b", "mistral-7b", "phi-3.5", "qwen3-8b",
]
ORIGINAL_FRONTIER = ["gpt-5.4", "gemini-3.1-pro"]
NEWER_OPEN = [
    "acegpt-32b", "fanar-2-27b", "gemma-4-12b", "gemma-4-31b",
    "karnak-40b", "ministral-3-14b", "qwen3.5-9b", "qwen3.5-27b",
]

# Spellings follow Table~\ref{tab:model_selection_effects} in the paper
# ("Phi-3.5-mini", "Gemini3.1-Pro") so the two per-model tables agree.
DISPLAY_NAMES = {
    "llama-3.1-8b": "Llama-3.1-8B", "phi-3.5": "Phi-3.5-mini",
    "gemma-3-12b": "Gemma-3-12B", "qwen3-8b": "Qwen3-8B",
    "acegpt-8b": "AceGPT-v2-8B", "command-r-7b": "Command-R-7B",
    "mistral-7b": "Mistral-7B", "allam-7b": "ALLaM-7B",
    "fanar-1-9b": "Fanar-1-9B", "jais-13b": "Jais-13B",
    "gpt-5.4": r"GPT-5.4\fr", "gemini-3.1-pro": r"Gemini3.1-Pro\fr",
    "acegpt-32b": "AceGPT-v2-32B", "fanar-2-27b": "Fanar-2-27B",
    "gemma-4-12b": "Gemma-4-12B", "gemma-4-31b": "Gemma-4-31B",
    "karnak-40b": "Karnak-40B", "ministral-3-14b": "Ministral-3-14B",
    "qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-27b": "Qwen3.5-27B",
}

CONDITION_FILES = {
    "explicit": ["fanar_main.jsonl", "fanar_frontier.jsonl"],
    "implicit": ["a2a_12models.jsonl", "a2a_8models.jsonl"],
}


def normalize_model(name):
    return "gemini-3.1-pro" if name == "gemini-pro-3.1" else name


def classify(annotation):
    """-> leaf key, or None if the annotation carries no text segment.

    Mirrors the tree's decision procedure exactly; see the module docstring
    for why 'none' and an absent segment both mean "no attribution offered"
    while 'N/A' does not.
    """
    segs = annotation.get("segments", [])
    text_segs = [s for s in segs if s.get("type") in TEXT_TYPES]
    attr_segs = [s for s in segs if s.get("type") in ATTR_TYPES]
    if not text_segs:
        return None

    labels = {s["label"] for s in text_segs}
    if len(labels) != 1:
        raise ValueError(f"citation with conflicting text labels: {labels}")
    fidelity = labels.pop()
    if fidelity not in ("correct", "corrupted", "fabricated"):
        raise ValueError(f"unexpected text label {fidelity!r}")

    offered = [s for s in attr_segs if s["label"] != "none"]
    judged = [s for s in attr_segs if s["label"] in ("correct", "incorrect")]

    if fidelity == "fabricated":
        return "attributed_fabrication" if offered else "bare_fabrication"

    if not judged:
        return "unsourced_quotation" if fidelity == "correct" else "unsourced_corruption"
    all_correct = all(s["label"] == "correct" for s in judged)
    if fidelity == "correct":
        return "correct_citation" if all_correct else "misattribution"
    return "textual_corruption" if all_correct else "corrupted_misattr"


RELEASE_DIR = NEW_PAPER_DIR.parent / "data" / "islamiceval2026_RELEASE"


def release_splits():
    """(question_id, generated_answer) -> split, over the public release."""
    out = {}
    for sp in ("train", "dev", "test"):
        with open(RELEASE_DIR / f"{sp}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[(r["question_id"], r["generated_answer"])] = sp
    return out


def public_pairs(keep=("train", "dev")):
    """(model, question_id) pairs whose EXPLICIT response is in `keep`.

    `split` is a property of a response, not a question -- all 500 questions
    appear in train -- so the restriction can only be applied response-wise.
    """
    sp = release_splits()
    pairs = set()
    for fn in CONDITION_FILES["explicit"]:
        with open(NEW_PAPER_DIR / fn, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if sp.get((r["question_id"], r["generated_answer"])) in keep:
                    pairs.add((normalize_model(r["source_model"]), r["question_id"]))
    return pairs


def tally(paths, keep_pairs=None, exempt=(), drop_pairs=None):
    """-> ({model: {leaf: count}}, n_no_text_segment)

    An annotation with no Ayah/matn segment carries no textual-fidelity label,
    so it cannot be placed on the tree at all and is excluded. These are
    counted and reported rather than dropped silently.

    keep_pairs restricts to the public train+dev universe. `exempt` lists models
    that bypass it: the extended panel has no explicit run, so no response of
    theirs can inherit a split label. Every one of the 500 questions appears in
    train, so for those models train+dev is the whole set in any case.
    """
    counts = {}
    skipped = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                model = normalize_model(row["source_model"])
                pair = (model, row["question_id"])
                if (keep_pairs is not None and model not in exempt
                        and pair not in keep_pairs):
                    continue
                if drop_pairs is not None and pair in drop_pairs:
                    continue
                bucket = counts.setdefault(model, dict.fromkeys(LEAF_KEYS, 0))
                for ann in row["annotations"]:
                    leaf = classify(ann)
                    if leaf is None:
                        skipped += 1
                    else:
                        bucket[leaf] += 1
    return counts, skipped


def pooled(counts, models):
    out = dict.fromkeys(LEAF_KEYS, 0)
    for m in models:
        for k, v in counts.get(m, {}).items():
            out[k] += v
    return out


def group_count(counts, group):
    return sum(v for k, v in counts.items() if LEAF_GROUP[k] == group)


def fmt_pct(n, total, bold=False):
    r"""Percentage with an escaped percent sign.

    Counts are deliberately NOT inlined here. With the three "All" columns the
    table is 13 wide; "52.3\% (1234)" in every cell runs roughly 170pt past
    \textwidth even at \scriptsize. Every count stays recoverable from the
    N column, and the exact integers are in the companion CSV.
    """
    if total == 0:
        return "--"
    s = rf"{100.0 * n / total:.1f}\%"
    return rf"\textbf{{{s}}}" if bold else s


def build_rows(counts, blocks):
    """blocks: [(label_or_None, [model keys]), ...] -> list of row dicts."""
    rows = []
    for sep_label, models in blocks:
        rows.append({"kind": "sep", "label": sep_label})
        for m in models:
            c = counts.get(m)
            if c is None:
                rows.append({"kind": "missing", "model": m})
                continue
            total = sum(c.values())
            rows.append({"kind": "model", "model": m, "total": total, "counts": c})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=("explicit", "implicit", "both"), default="both")
    ap.add_argument("--splits", default="train,dev",
                    help="public release splits to keep (comma-separated); "
                         "use train,dev,test for the unrestricted set")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conditions = ("explicit", "implicit") if args.condition == "both" else (args.condition,)

    for condition in conditions:
        paths = [NEW_PAPER_DIR / f for f in CONDITION_FILES[condition]]
        keep = tuple(x.strip() for x in args.splits.split(",") if x.strip())
        held_out = {"train", "dev", "test"} - set(keep)
        if held_out and condition == "explicit":
            # The explicit arm IS the released data, with a genuine held-out
            # test split, so it is filtered response-wise.
            counts, skipped = tally(paths, keep_pairs=public_pairs(keep))
        else:
            # The implicit arm is NOT filtered, at any --splits setting. It was
            # never released, so no implicit response is held out: the test
            # split holds out annotated EXPLICIT responses. The implicit answer
            # to the same question is a different response, from a different
            # prompt, with its own annotations, and the 500 questions are public
            # in any case. Coupling it to its explicit twin's split would
            # discard clean data to protect nothing.
            counts, skipped = tally(paths)

        present = set(counts)
        expected = set(ORIGINAL_OPEN + ORIGINAL_FRONTIER)
        if condition == "implicit":
            expected |= set(NEWER_OPEN)
        missing, extra = expected - present, present - expected
        if missing:
            raise SystemExit(f"[{condition}] models missing from the data: {sorted(missing)}")
        if extra:
            raise SystemExit(f"[{condition}] unexpected models in the data: {sorted(extra)}")

        blocks = [(None, ORIGINAL_OPEN + ORIGINAL_FRONTIER)]
        # Open-weight and frontier are pooled separately. A single "Pooled (12)"
        # row averaged two tiers whose distributions barely overlap -- frontier
        # fabricates ~0% and the open models 44% -- so the combined figure
        # described no model in the table.
        pooled_rows = [("Open-weight (10)", ORIGINAL_OPEN),
                       ("Frontier (2)", ORIGINAL_FRONTIER)]
        if condition == "implicit":
            blocks.append(("newer", NEWER_OPEN))
            pooled_rows.append(("Extended panel (8)", NEWER_OPEN))

        rows = build_rows(counts, blocks)

        grand = sum(sum(c.values()) for c in counts.values())
        print(f"=== {condition}: {len(counts)} models, {grand:,} citations classified"
              f" ({skipped} excluded: no Ayah/matn segment, so no fidelity label)")
        for label, models in pooled_rows:
            p = pooled(counts, models)
            t = sum(p.values())
            print(f"  {label:22s} n={t:6,}  " +
                  "  ".join(f"{k.split('_')[0][:9]}={100*v/t:5.1f}%" for k, v in p.items()))

        # ---- CSV ----
        csv_path = args.out_dir / f"table_8class_{condition}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            fid = [g for g, _ in GROUPS]
            w.writerow(["model", "group", "n_citations"] +
                       [f"n_{k}" for k in LEAF_KEYS] + [f"pct_{k}" for k in LEAF_KEYS] +
                       [f"n_all_{g}" for g in fid] + [f"pct_all_{g}" for g in fid])

            def emit(name, grp, c):
                t = sum(c.values())
                pc = (lambda n: f"{100.0 * n / t:.2f}" if t else "")
                w.writerow([name, grp, t] + [c[k] for k in LEAF_KEYS] +
                           [pc(c[k]) for k in LEAF_KEYS] +
                           [group_count(c, g) for g in fid] +
                           [pc(group_count(c, g)) for g in fid])

            for m in ORIGINAL_OPEN + ORIGINAL_FRONTIER + (NEWER_OPEN if condition == "implicit" else []):
                emit(m, "frontier" if m in ORIGINAL_FRONTIER
                     else "newer-open" if m in NEWER_OPEN else "original-open", counts[m])
            for label, models in pooled_rows:
                emit(label, "pooled", pooled(counts, models))
        print(f"  wrote {csv_path}")

        # ---- LaTeX ----
        ncol = len(COLUMNS)
        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        # \scriptsize, not \footnotesize: the headers name the taxonomy leaves
        # ("Corruption", "Attributed", "Unsourced") rather than bare attribution
        # states, and at 9pt those 13 columns run ~50pt past \textwidth.
        lines.append(r"\scriptsize")
        # 1.5pt, not the usual 6pt: 13 columns need every point of the width.
        lines.append(r"\setlength{\tabcolsep}{1.5pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.15}")
        # Every numeric column is right-aligned so the decimal points line up.
        # An explicit 6pt gap opens before each fidelity group: at tabcolsep
        # 1.5pt the boundary between one group's last column and the next
        # group's "All" is the same 3pt as the gaps inside a group, so the
        # blocks visually run together. @{} replaces the normal glue at that
        # boundary rather than adding to it, so this costs only ~3pt each.
        gap = r"@{\hspace{6pt}}"
        colspec = " ".join(f"{gap} " + "r" * w for _, w in GROUP_SPAN)
        lines.append(rf"\begin{{tabular}}{{l r {colspec}}}")
        lines.append(r"\toprule")

        # Group header, then the leaf header. Deliberately no \multirow: the
        # paper's preamble does not load the multirow package, so the Model /
        # N labels go in the second header row instead of spanning both.
        spans = [rf"\multicolumn{{{w}}}{{c}}{{\textbf{{{name}}}}}" for name, w in GROUP_SPAN]
        lines.append(" & ".join(["", ""] + spans) + r" \\")
        cm, start = [], 3
        for _, width in GROUP_SPAN:
            cm.append(rf"\cmidrule(lr){{{start}-{start + width - 1}}}")
            start += width
        lines.append("".join(cm))
        lines.append(" & ".join([r"\textbf{Model}", r"\textbf{N}"] +
                                [rf"\textbf{{{h}}}" for _, _, h, _ in COLUMNS]) + r" \\")
        lines.append(r"\midrule")

        for row in rows:
            if row["kind"] == "sep":
                if row["label"] == "newer":
                    lines.append(r"\midrule")
                    lines.append(rf"\multicolumn{{{2 + ncol}}}{{l}}{{\emph{{Newer open-weight models "
                                 r"(implicit prompt only)}} \\")
                continue
            if row["kind"] == "missing":
                continue
            c, t = row["counts"], row["total"]
            cells = [fmt_pct(group_count(c, key) if kind == "all" else c[key], t,
                             bold=(kind == "all"))
                     for kind, key, _, _ in COLUMNS]
            lines.append(f"{DISPLAY_NAMES[row['model']]} & {t:,} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        for label, models in pooled_rows:
            p = pooled(counts, models)
            t = sum(p.values())
            cells = [fmt_pct(group_count(p, key) if kind == "all" else p[key], t,
                             bold=(kind == "all"))
                     for kind, key, _, _ in COLUMNS]
            lines.append(rf"\textbf{{{label}}} & {t:,} & " + " & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        scope = ("the publicly released train and dev splits"
                 if set(keep) == {"train", "dev"} else "+".join(sorted(keep)))
        cap = (r"Per-model distribution over the eight terminal classes of the citation "
               r"taxonomy (Figure~\ref{fig:decision-tree}), as percentage of that model's "
               r"$N$ citations. The bold \textbf{All} column in each group is the model's "
               r"overall rate for that textual-fidelity class and equals the sum of the "
               r"leaf columns beside it; the three \textbf{All} columns sum to 100\%, as "
               r"do the eight leaf columns, in both cases to within rounding. Counts are "
               rf"recoverable as the percentage of $N$. Citations were generated under the "
               rf"{condition} prompt" + (rf", restricted to {scope}. " if condition == "explicit" else ". "))
        if condition == "implicit":
            cap += (r"The implicit condition was never released and has no held-out split, "
                    r"so all of it is used; the eight models below the rule were run under "
                    rf"this prompt only. $N$ counts classifiable citations: {skipped} "
                    r"annotations carry no Ayah or Matn segment and so have no "
                    r"textual-fidelity label. ")
        cap += (r"Column groups give the textual-fidelity axis; the columns within each "
                r"group give the attributional one, so the eight leaves of "
                r"Figure~\ref{fig:decision-tree} read, left to right: Correct Citation, "
                r"Misattribution, Unsourced Quotation, Textual Corruption, Corrupted "
                r"Misattribution, Unsourced Corruption, Attributed Fabrication, Bare "
                r"Fabrication. \emph{Unsrc.} denotes a citation that offered no source at "
                r"all; attribution is judged only where a source was offered, so a "
                r"component the model never supplied is not counted against it, and a "
                r"Hadith citation counts as correctly attributed only if every component "
                r"it did supply is correct. Under \emph{Fabricated} there is no "
                r"attribution to judge, so the split is merely whether a source was "
                r"offered. \fr marks frontier models.")
        lines.append(rf"\caption{{{cap}}}")
        lines.append(rf"\label{{tab:eight_class_{condition}}}")
        lines.append(r"\end{table*}")

        tex_path = args.out_dir / f"table_8class_{condition}.tex"
        tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {tex_path}\n")


if __name__ == "__main__":
    main()
