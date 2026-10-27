"""
cost_of_asking_3class.py

Three-class version of the "cost of asking explicitly" decomposition, for the
second paper: instead of a single accuracy (correct vs. incorrect), each
citation carries the fine-grained label

    correct / corrupted / fabricated

and the Selection / Quality / Volume decomposition is run separately on the
RATE of each of the three outcomes.

Relation to scripts/cost_of_asking.py
-------------------------------------
Same estimator, same bootstrap, same join unit. Two differences:

1. Outcome is three-way, not binary. The three rates sum to 1 within every
   sub-population, so for each of Delta / Selection / Quality / Volume the
   three per-outcome values sum to 0 (asserted). The `correct` column
   reproduces the binary analysis exactly on the same subset of responses.
2. The response universe is restricted to the PUBLIC release splits
   (train + dev by default), because this paper may only use released data.

Data
----
EXPLICIT  new_paper/fanar_main.jsonl       (10 open models)
          new_paper/fanar_frontier.jsonl   (2 frontier models)
IMPLICIT  new_paper/a2a_12models.jsonl     (same 500 questions, vanilla prompt)

CAVEAT, read before citing this figure
--------------------------------------
The implicit ("a2a" / vanilla) condition is NOT part of the public release:
matching every a2a response against train/dev/test.jsonl by
(question_id, generated_answer) finds 2 of 5,845 rows in the release. The
public-split restriction below is therefore applied to the EXPLICIT arm only
-- the implicit arm is unreleased data in its entirety. Running this script
with --splits train,dev gives "public explicit vs. unreleased implicit", not
a fully public figure. This was a deliberate call; it is recorded here so the
limitation is not lost between the script and the paper.

Public-split restriction
------------------------
`split` is a property of a RESPONSE, not of a question: all 500 questions
appear in train, so holding out test can only be done response-wise. We keep
explicit responses whose (question_id, generated_answer) lands in the chosen
splits, and then (default --implicit-universe paired) keep the implicit
response for exactly those (model, question_id) pairs, so the join unit stays
coherent. --implicit-universe all keeps every implicit response instead,
which lets I(all) see pairs with no public explicit counterpart.

Definitions (per outcome o, unchanged in form from the binary script)
---------------------------------------------------------------------
A CITATION is an annotation segment with type in {"Ayah", "matn"}; isnad and
claimed_source are ignored. Citations within an answer are ordered by
span_start. Join unit = (model, question_id).

    E_o(all)  = share of ALL explicit citations whose label == o
    I_o(all)  = share of ALL implicit citations whose label == o
    shared    = (model, question_id) pairs with >=1 citation under BOTH
                prompt conditions
    k_q       = number of implicit citations for a shared pair
    first-k   = the first k_q explicit citations (by span_start) for that pair

    Delta_o     = E_o(all) - I_o(all)
    Quality_o   = E_o(shared, first-k) - I_o(shared, all)
    Volume_o    = E_o(shared, all)     - E_o(shared, first-k)
    Selection_o = [E_o(all) - E_o(shared)] + [I_o(shared) - I_o(all)]

Selection_o + Quality_o + Volume_o == Delta_o, asserted to 1e-9 per outcome
per model. 95% CIs come from a question-level cluster bootstrap; one resample
of question_ids drives all three outcomes and all four terms jointly, so the
sum-to-zero structure holds inside every resample.

Outputs (everything for this paper stays under new_paper/)
----------------------------------------------------------
new_paper/outputs/cost_of_asking_3class_permodel.csv
new_paper/outputs/fig1_cost_of_asking_3class.png
new_paper/outputs/fig1_cost_of_asking_3class.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

NEW_PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = NEW_PAPER_DIR.parent

EXPLICIT_PATHS = [NEW_PAPER_DIR / "fanar_main.jsonl", NEW_PAPER_DIR / "fanar_frontier.jsonl"]
IMPLICIT_PATH = NEW_PAPER_DIR / "a2a_12models.jsonl"

# The only read outside new_paper/: the release splits, used to decide which
# responses are public.
RELEASE_DIR = REPO_ROOT / "data" / "islamiceval2026_RELEASE"
RELEASE_SPLIT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}

OUT_DIR = NEW_PAPER_DIR / "outputs"

CITATION_TYPES = {"Ayah", "matn"}
ATTR_TYPES = {"isnad", "claimed_source"}

# --axis selects which of the taxonomy's two axes the decomposition runs on.
# Both are measured over the SAME denominator -- every citation -- so the
# outcome rates sum to 1 and the estimator telescopes identically.
#
# The attribution axis carries a fourth class, "No text" (the deterministic
# N/A gate: the citation's text is fabricated, so there is nothing to
# attribute). Keeping it as an explicit class is what makes this analysis
# honest. The tempting alternative -- restricting to citations whose
# attribution is judgeable and comparing accuracy there -- conditions on a
# POST-TREATMENT variable: the explicit prompt changes how much a model
# fabricates, so it changes which citations are judgeable at all, and the
# resulting "attribution accuracy" difference mixes that selection in. The
# conditional view is still printed, clearly marked, for reference.
AXES = {
    "fidelity": {
        "outcomes": ("correct", "corrupted", "fabricated"),
        "labels": {"correct": "Correct", "corrupted": "Corrupted",
                   "fabricated": "Fabricated"},
        # Lifted from taxonomy_tree.pdf; see OUTCOME_COLORS note below.
        "colors": {"correct": "#77be80", "corrupted": "#d9992f",
                   "fabricated": "#a33a36"},
        "caption": ("Each citation is labelled correct / corrupted / fabricated; bars are the explicit-minus-implicit\n"
                    "change in the share of citations carrying that label, so the three bars in a group sum to zero."),
        "title": "The cost of asking, by failure mode: selection, quality, and volume",
        "ylabel": "Explicit $-$ implicit rate (points)",
        "stem": "3class",
    },
    "attribution": {
        "outcomes": ("attr_correct", "attr_incorrect", "unsourced", "no_text"),
        "labels": {"attr_correct": "Attributed correctly",
                   "attr_incorrect": "Misattributed",
                   "unsourced": "No source offered",
                   "no_text": "No text to attribute"},
        # Green/red/amber as on the fidelity axis, plus grey for the null
        # class. The palette validator FAILS grey on its chroma floor -- that
        # check exists so a real series never reads as grey. "No text to
        # attribute" is not a series, it is a not-applicable slot, and grey is
        # the conventional encoding for it. Every other check passes
        # (CVD dE 8.1, normal-vision 15.0), which is why this grey and not the
        # lighter #8a9099 from the tree (14.9 against the green, below floor).
        "colors": {"attr_correct": "#77be80", "attr_incorrect": "#a33a36",
                   "unsourced": "#d9992f", "no_text": "#6b7280"},
        "caption": ("Bars are the explicit-minus-implicit change in the share of citations in each attribution class, so the\n"
                    "four bars in a group sum to zero. Attribution is undefined where the quoted text is fabricated, so that case\n"
                    "is kept as its own class rather than dropped: dropping it would condition on an outcome the prompt changes."),
        "title": "The cost of asking, by attribution outcome: selection, quality, and volume",
        "ylabel": "Explicit $-$ implicit rate (points)",
        "stem": "attribution",
    },
}

# Rebound in main() from the chosen --axis; every helper reads them at call
# time, so the estimator itself is axis-agnostic.
AXIS = "fidelity"
OUTCOMES = AXES["fidelity"]["outcomes"]

FRONTIER_MODELS = ["gpt-5.4", "gemini-3.1-pro"]
OPEN_WEIGHT_MODELS = [
    "acegpt-8b", "allam-7b", "command-r-7b", "fanar-1-9b", "gemma-3-12b",
    "jais-13b", "llama-3.1-8b", "mistral-7b", "phi-3.5", "qwen3-8b",
]
ALL_MODELS = OPEN_WEIGHT_MODELS + FRONTIER_MODELS

DISPLAY_NAMES = {
    # Spellings follow the paper's other per-model tables.
    "llama-3.1-8b": "Llama-3.1-8B",
    "phi-3.5": "Phi-3.5-mini",
    "gemma-3-12b": "Gemma-3-12B",
    "qwen3-8b": "Qwen3-8B",
    "acegpt-8b": "AceGPT-v2-8B",
    "command-r-7b": "Command-R-7B",
    "mistral-7b": "Mistral-7B",
    "allam-7b": "ALLaM-7B",
    "fanar-1-9b": "Fanar-1-9B",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro": "Gemini3.1-Pro",
    "jais-13b": "Jais-13B",
    "__OPEN__": "Open-weight pooled",
    "__FRONTIER__": "Frontier pooled",
}

# Lifted from taxonomy_tree.pdf so the two figures read as one system: the
# tree colours its leaves green / amber / red by textual outcome, and these are
# three of its own fills.
#
# The tree's SATURATED green (#2e7d32) cannot be used here. Against its red
# (#a33a36) it measures CVD deltaE 1.8 under deuteranopia -- the two are
# effectively the same colour for a red-green colourblind reader. The tree gets
# away with it because every leaf is labelled and boxed; a bar chart leans much
# harder on fill alone. Taking the tree's LIGHTER green leaf tint instead adds
# the missing lightness separation and lifts the worst pair to 8.1, clearing
# the validator on all checks (band / chroma / CVD / normal-vision), with hue
# order still reading green -> amber -> red as severity rises.
#
# The amber and green sit below 3:1 on white, so the relief rule applies: every
# bar carries a visible value label and the legend names all three outcomes --
# identity is never colour-alone. Sign is carried by bar direction, never hue.
OUTCOME_COLORS = {"correct": "#77be80", "corrupted": "#d9992f", "fabricated": "#a33a36"}
TERMS = ("delta", "selection", "quality", "volume")
TERM_LABELS = {
    "delta": "Total\n(unconditioned)",
    "selection": "Selection",
    "quality": "Quality",
    "volume": "Volume",
}


def normalize_model(name: str) -> str:
    if name == "gemini-pro-3.1":
        return "gemini-3.1-pro"
    return name


# ---------------------------------------------------------------- loading ---

def load_release_splits(release_dir: Path) -> dict:
    """(question_id, generated_answer) -> split name, over the public release."""
    splits = {}
    for split, fname in RELEASE_SPLIT_FILES.items():
        path = release_dir / fname
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                splits[(row["question_id"], row["generated_answer"])] = split
    return splits


def attribution_outcome(fidelity, attr_segs):
    """Attribution class for one citation.

    'none' and an absent segment both mean no source was offered; 'N/A' does
    not -- it is the fabrication gate, and the segment's presence means the
    model DID name a source. A Hadith citation counts as correctly attributed
    only if every judged component (isnad and claimed_source) is correct.
    """
    if fidelity == "fabricated":
        return "no_text"
    judged = [s for s in attr_segs if s["label"] in ("correct", "incorrect")]
    if not judged:
        return "unsourced"
    return "attr_correct" if all(s["label"] == "correct" for s in judged) else "attr_incorrect"


def extract_citations(annotations: list) -> list:
    """[(span_start, outcome)] for every citation, sorted by span_start.

    One citation == one annotation carrying exactly one Ayah/matn segment.
    Annotations with no text segment have no fidelity label and so cannot be
    placed on either axis; they are skipped (5 across the whole corpus).
    """
    cites = []
    for ann in annotations:
        segs = ann.get("segments", [])
        text = [s for s in segs if s.get("type") in CITATION_TYPES]
        if not text:
            continue
        labels = {s["label"] for s in text}
        if len(labels) != 1:
            raise ValueError(f"citation with conflicting text labels: {labels}")
        fidelity = labels.pop()
        if fidelity not in AXES["fidelity"]["outcomes"]:
            raise ValueError(f"unexpected text-segment label {fidelity!r}")

        if AXIS == "fidelity":
            outcome = fidelity
        else:
            outcome = attribution_outcome(
                fidelity, [s for s in segs if s.get("type") in ATTR_TYPES])
        cites.append((min(s["span_start"] for s in text), outcome))
    cites.sort(key=lambda x: x[0])
    return cites


def load_explicit(paths, release_splits, keep_splits):
    """-> (records, split_counts). records: (model, qid) -> [(span_start, outcome)]"""
    records = {}
    counts = {"train": 0, "dev": 0, "test": 0, "NOT_IN_RELEASE": 0}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                split = release_splits.get((row["question_id"], row["generated_answer"]),
                                           "NOT_IN_RELEASE")
                counts[split] += 1
                if split not in keep_splits:
                    continue
                key = (normalize_model(row["source_model"]), row["question_id"])
                if key in records:
                    raise ValueError(f"duplicate explicit (model, qid) pair: {key}")
                records[key] = extract_citations(row["annotations"])
    return records, counts


def load_implicit(path, allowed_pairs=None):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (normalize_model(row["source_model"]), row["question_id"])
            if allowed_pairs is not None and key not in allowed_pairs:
                continue
            if key in records:
                raise ValueError(f"duplicate implicit (model, qid) pair: {key}")
            records[key] = extract_citations(row["annotations"])
    return records


# ------------------------------------------------------------ count arrays ---

def per_model_question_arrays(model, question_ids, q_index, exp_records, imp_records):
    """Per-question count arrays for one model.

    Denominator arrays (n_*) are outcome-independent; numerators are one array
    per outcome, held under the flat key "c_<pop>__<outcome>" so that pooling
    across models and bootstrap weighting stay a plain elementwise sum.
    """
    n = len(question_ids)
    arrs = {
        "n_imp": np.zeros(n, dtype=np.int64),
        "n_exp": np.zeros(n, dtype=np.int64),
        "n_exp_firstk": np.zeros(n, dtype=np.int64),
        "shared": np.zeros(n, dtype=np.int64),
        "n_resp_imp": np.zeros(n, dtype=np.int64),
        "n_resp_exp": np.zeros(n, dtype=np.int64),
        "n_resp_imp_citing": np.zeros(n, dtype=np.int64),
        "n_resp_exp_citing": np.zeros(n, dtype=np.int64),
    }
    for pop in ("c_imp", "c_exp", "c_exp_firstk"):
        for o in OUTCOMES:
            arrs[f"{pop}__{o}"] = np.zeros(n, dtype=np.int64)

    for qid in question_ids:
        idx = q_index[qid]
        imp_cites = imp_records.get((model, qid))
        exp_cites = exp_records.get((model, qid))

        # n_resp_* counts EVERY answer in the arm, including answers that cite
        # nothing -- that is the denominator for "citations / answer" in the
        # figure. n_resp_*_citing counts only answers with >=1 citation; it is
        # carried to the CSV for reference but must NOT be used as the
        # denominator. Under the vanilla prompt most answers cite nothing, and
        # because every explicit answer here cites at least once, an implicit
        # answer that cites is automatically a shared pair -- so dividing by
        # the citing count silently reports the rate on shared pairs only.
        if imp_cites is not None:
            arrs["n_resp_imp"][idx] = 1
            arrs["n_resp_imp_citing"][idx] = 1 if imp_cites else 0
            arrs["n_imp"][idx] = len(imp_cites)
            for _, o in imp_cites:
                arrs[f"c_imp__{o}"][idx] += 1
        if exp_cites is not None:
            arrs["n_resp_exp"][idx] = 1
            arrs["n_resp_exp_citing"][idx] = 1 if exp_cites else 0
            arrs["n_exp"][idx] = len(exp_cites)
            for _, o in exp_cites:
                arrs[f"c_exp__{o}"][idx] += 1

        if imp_cites and exp_cites:
            arrs["shared"][idx] = 1
            first_k = exp_cites[:len(imp_cites)]
            arrs["n_exp_firstk"][idx] = len(first_k)
            for _, o in first_k:
                arrs[f"c_exp_firstk__{o}"][idx] += 1

    return arrs


def make_derived_arrs(arrs):
    """Add the shared-masked populations.

    Masking by `shared` happens here, per model, BEFORE any cross-model
    pooling. Pooling raw counts first and masking afterwards would leak
    non-shared citations from other models into the shared totals.
    """
    shared = arrs["shared"]
    d = dict(arrs)
    d["n_exp_shared"] = arrs["n_exp"] * shared
    d["n_imp_shared"] = arrs["n_imp"] * shared
    for o in OUTCOMES:
        d[f"c_exp_shared__{o}"] = arrs[f"c_exp__{o}"] * shared
        d[f"c_imp_shared__{o}"] = arrs[f"c_imp__{o}"] * shared
    return d


def sum_group_arrs(list_of_arrs):
    keys = list_of_arrs[0].keys()
    return {k: sum(a[k] for a in list_of_arrs) for k in keys}


def safe_div(num, den):
    return num / den if den > 0 else float("nan")


def decompose(d, w=1):
    """Decomposition for every outcome at once, under question weights `w`.

    Returns {"denoms": {...}, "<outcome>": {delta, selection, quality, volume,
    rate_imp, rate_exp, rate_exp_firstk, rate_exp_beyondk}}.
    """
    N_imp_all = (d["n_imp"] * w).sum()
    N_exp_all = (d["n_exp"] * w).sum()
    N_exp_fk = (d["n_exp_firstk"] * w).sum()
    N_exp_sh = (d["n_exp_shared"] * w).sum()
    N_imp_sh = (d["n_imp_shared"] * w).sum()

    out = {"denoms": {
        "n_cites_imp": int(N_imp_all), "n_cites_exp": int(N_exp_all),
        "n_cites_exp_firstk": int(N_exp_fk), "n_cites_exp_shared": int(N_exp_sh),
        "n_cites_imp_shared": int(N_imp_sh), "n_beyondk": int(N_exp_sh - N_exp_fk),
        "n_resp_imp": int((d["n_resp_imp"] * w).sum()),
        "n_resp_exp": int((d["n_resp_exp"] * w).sum()),
        "n_resp_imp_citing": int((d["n_resp_imp_citing"] * w).sum()),
        "n_resp_exp_citing": int((d["n_resp_exp_citing"] * w).sum()),
        "n_shared": int((d["shared"] * w).sum()),
    }}

    for o in OUTCOMES:
        C_imp_all = (d[f"c_imp__{o}"] * w).sum()
        C_exp_all = (d[f"c_exp__{o}"] * w).sum()
        C_exp_fk = (d[f"c_exp_firstk__{o}"] * w).sum()
        C_exp_sh = (d[f"c_exp_shared__{o}"] * w).sum()
        C_imp_sh = (d[f"c_imp_shared__{o}"] * w).sum()

        E_all = safe_div(C_exp_all, N_exp_all)
        I_all = safe_div(C_imp_all, N_imp_all)
        E_sh = safe_div(C_exp_sh, N_exp_sh)
        I_sh = safe_div(C_imp_sh, N_imp_sh)
        E_fk = safe_div(C_exp_fk, N_exp_fk)

        out[o] = {
            "delta": E_all - I_all,
            "quality": E_fk - I_sh,
            "volume": E_sh - E_fk,
            "selection": (E_all - E_sh) + (I_sh - I_all),
            "rate_imp": I_all,
            "rate_exp": E_all,
            "rate_exp_firstk": E_fk,
            "rate_exp_beyondk": safe_div(C_exp_sh - C_exp_fk, N_exp_sh - N_exp_fk),
        }
    return out


def bootstrap_group(derived, n_questions, B, seed):
    """Question-level cluster bootstrap.

    One resample of question_ids per replicate drives all three outcomes and
    all four terms, so Selection+Quality+Volume==Delta and the sum-to-zero
    across outcomes both hold inside every replicate, exactly as they do in
    the point estimate.
    """
    rng = np.random.default_rng(seed)
    boot = {o: {t: np.full(B, np.nan) for t in TERMS} for o in OUTCOMES}
    n_nan = 0

    for b in range(B):
        idx = rng.integers(0, n_questions, size=n_questions)
        w = np.bincount(idx, minlength=n_questions)

        # Denominators are outcome-independent, so a degenerate resample is
        # degenerate for all three outcomes at once -> skip it wholesale.
        if ((derived["n_exp"] * w).sum() == 0 or (derived["n_imp"] * w).sum() == 0
                or (derived["n_exp_shared"] * w).sum() == 0
                or (derived["n_imp_shared"] * w).sum() == 0
                or (derived["n_exp_firstk"] * w).sum() == 0):
            n_nan += 1
            continue

        res = decompose(derived, w)
        for o in OUTCOMES:
            for t in TERMS:
                boot[o][t][b] = res[o][t]

    return boot, n_nan


def pct(x):
    return float("nan") if x != x else 100.0 * x


def ci_from_boot(vals):
    return pct(np.nanpercentile(vals, 2.5)), pct(np.nanpercentile(vals, 97.5))


def fmt(x):
    return "NaN" if x != x else f"{x:.1f}"


# ----------------------------------------------------------------- figure ---

FOCUS_LABEL = {"fidelity": "textually correct", "attribution": "correctly attributed"}


def make_latex_table(rows, out_dir, n_boot):
    r"""Per-model decomposition for the axis's focus outcome, in the paper's
    own \est / \sigest style, with the implicit and explicit citation counts
    so a reader can see which rows rest on thin data.
    """
    focus = OUTCOMES[0]
    sig = lambda lo, hi: not (lo <= 0 <= hi)

    def cell(r, term):
        v, lo, hi = r[f"{term}__{focus}"], r[f"{term}__{focus}__lo"], r[f"{term}__{focus}__hi"]
        if v != v:
            return "--"
        mac = r"\sigest" if sig(lo, hi) else r"\est"
        return rf"{mac}{{{v:+.1f}}}{{{lo:.1f}}}{{{hi:.1f}}}"

    model_rows = [r for r in rows if r["model"] in ALL_MODELS]
    model_rows.sort(key=lambda r: r[f"delta__{focus}"])
    pooled_rows = [r for r in rows if r["model"].startswith("__")]

    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.25}",
         r"\begin{tabular}{l rr llll}", r"\toprule",
         r"\textbf{Model} & \textbf{$N_{\text{imp}}$} & \textbf{$N_{\text{exp}}$} "
         r"& \textbf{$\Delta$ Total (pp)} & \textbf{$\Delta$ Selection (pp)} "
         r"& \textbf{$\Delta$ Quality (pp)} & \textbf{$\Delta$ Volume (pp)} \\", r"\midrule"]
    for group in (model_rows, pooled_rows):
        if group is pooled_rows:
            L.append(r"\midrule")
        for r in group:
            name = DISPLAY_NAMES[r["model"]]
            if r["model"] in FRONTIER_MODELS:
                name += r"\fr"
            if r["model"].startswith("__"):
                name = rf"\textbf{{{name}}}"
            L.append(f"{name} & {r['n_cites_imp']:,} & {r['n_cites_exp']:,} & "
                     + " & ".join(cell(r, t) for t in TERMS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L.append(
        rf"\caption{{Per-model decomposition of the change in the share of citations that are "
        rf"\emph{{{FOCUS_LABEL[AXIS]}}}, from the implicit to the explicit prompt. "
        r"Selection represents citations under shared questions. Quality represents first shared "
        r"citations. Volume represents explicit-only citations beyond the implicit count. "
        r"$N_{\text{imp}}$ and $N_{\text{exp}}$ are the citation counts the estimate rests on. "
        rf"Values are estimated changes with 95\% confidence intervals from a question-level "
        rf"cluster bootstrap ({n_boot:,} resamples); bold estimates have intervals excluding zero. "
        r"\fr marks frontier models.}")
    L.append(rf"\label{{tab:cost_of_asking_{AXES[AXIS]['stem']}}}")
    L.append(r"\end{table*}")

    path = out_dir / f"cost_of_asking_{AXES[AXIS]['stem']}_permodel.tex"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def make_figure(open_row, frontier_row, out_dir, n_boot, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Panel labels are identified in the LaTeX subcaptions, so the figure
    # itself carries only the citations/answer line.
    panels = [("", open_row), ("", frontier_row)]

    # The footnote runs 4 lines on the fidelity axis and 6 on attribution, so
    # the figure grows and the reserved bottom strip grows with it; a fixed
    # margin lets the longer caption collide with the x tick labels.
    # No footnote strip: the methods text lives in the LaTeX \caption, so the
    # figure keeps only the plot. AXES[...]["caption"] is retained as the
    # source of that wording -- print_captions() emits it for the paper.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle(AXES[AXIS]["title"], fontsize=15.5, y=0.985)

    group_x = np.arange(len(TERMS))
    bar_w = 0.26

    for ax, (panel_title, row) in zip(axes, panels):
        cites_imp = row["n_cites_imp"] / row["n_resp_imp"]
        cites_exp = row["n_cites_exp"] / row["n_resp_exp"]
        head = f"{panel_title}\n" if panel_title else ""
        ax.set_title(f"{head}citations / answer: {cites_imp:.2f} "
                     f"$\\rightarrow$ {cites_exp:.2f}", fontsize=13, pad=12)

        labels = []  # (x, value, whisker_lo, whisker_hi) -- annotated after autoscale
        for j, o in enumerate(OUTCOMES):
            offs = (j - 1) * bar_w
            vals = np.array([row[f"{t}__{o}"] for t in TERMS])
            los = np.array([row[f"{t}__{o}__lo"] for t in TERMS])
            his = np.array([row[f"{t}__{o}__hi"] for t in TERMS])
            # errorbar wants distances from the point estimate, clipped at 0 so
            # a bootstrap percentile landing on the wrong side of the point
            # cannot render as a negative-length whisker.
            yerr = np.vstack([np.clip(vals - los, 0, None), np.clip(his - vals, 0, None)])

            ax.bar(group_x + offs, vals, bar_w * 0.88, color=AXES[AXIS]["colors"][o],
                   edgecolor="white", linewidth=0.8, zorder=3)
            ax.errorbar(group_x + offs, vals, yerr=yerr, fmt="none",
                        ecolor="#2b2b2b", elinewidth=1.3, capsize=3.5,
                        capthick=1.3, zorder=4)
            labels.extend(zip(group_x + offs, vals, vals - yerr[0], vals + yerr[1]))

        ax.axhline(0, color="#1a1a19", linewidth=1.1, zorder=2)
        ax.set_xticks(group_x)
        ax.set_xticklabels([TERM_LABELS[t] for t in TERMS], fontsize=12)
        ax.set_xlim(-0.55, len(TERMS) - 0.45)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b3")
        ax.tick_params(axis="y", labelsize=11, color="#b9b8b3")
        ax.tick_params(axis="x", length=0)
        ax.margins(y=0.24)

        # Direct-label every bar -- the size of each term IS the finding, and
        # with twelve bars per panel a reader cannot recover them off the axis.
        # Done after margins() so the offset is a fixed share of the FINAL
        # y-range; computing it mid-loop reads a y-limit that autoscaling has
        # not finished with yet.
        y0, y1 = ax.get_ylim()
        pad = 0.035 * (y1 - y0)
        for x, v, lo, hi in labels:
            above = v >= 0
            ax.annotate(f"{v:+.1f}", (x, (hi + pad) if above else (lo - pad)),
                        ha="center", va="bottom" if above else "top",
                        fontsize=9.5, fontweight="bold", color="#1a1a19", zorder=5)

    axes[0].set_ylabel(AXES[AXIS]["ylabel"], fontsize=13)

    handles = [Patch(facecolor=AXES[AXIS]["colors"][o], edgecolor="white",
                     label=AXES[AXIS]["labels"][o]) for o in OUTCOMES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(OUTCOMES), frameon=False, fontsize=11.5)

    fig.tight_layout(rect=(0, 0.01, 1, 0.87))

    stem = f'fig1_cost_of_asking_{AXES[AXIS]["stem"]}'
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description="Three-class cost-of-asking decomposition")
    ap.add_argument("--axis", choices=("fidelity", "attribution", "both"), default="fidelity",
                    help="which taxonomy axis to decompose; see AXES")
    ap.add_argument("--bootstrap", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="train,dev",
                    help="public release splits to keep on the explicit arm "
                         "(comma-separated; use train,dev,test to disable the restriction)")
    ap.add_argument("--implicit-universe", choices=("paired", "all"), default="all",
                    help="'all' (default): keep every implicit response. The implicit "
                         "condition was never released and has no held-out split, so "
                         "nothing in it needs withholding. 'paired': additionally require "
                         "an explicit counterpart in the kept splits -- this changes only "
                         "Delta and Selection (Quality and Volume use shared pairs, which "
                         "are identical either way) and discards clean implicit data.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    axes = ("fidelity", "attribution") if args.axis == "both" else (args.axis,)
    for ax in axes:
        run_axis(args, ax)


def run_axis(args, axis):
    """Run the whole decomposition for one taxonomy axis.

    AXIS / OUTCOMES are module-level because every helper reads them at call
    time; rebinding them here is what keeps the estimator axis-agnostic
    without threading the axis through a dozen signatures.
    """
    global AXIS, OUTCOMES
    AXIS = axis
    OUTCOMES = AXES[axis]["outcomes"]
    print("\n" + "#" * 70)
    print("# AXIS: " + axis)
    print("#" * 70)

    t0 = time.time()
    keep_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    bad = keep_splits - set(RELEASE_SPLIT_FILES)
    if bad:
        ap.error(f"unknown split(s): {sorted(bad)}")

    print(f"Loading release splits from {RELEASE_DIR} ...")
    release_splits = load_release_splits(RELEASE_DIR)
    print(f"  {len(release_splits)} released responses")

    print(f"Loading explicit condition from {[p.name for p in EXPLICIT_PATHS]} ...")
    exp_records, split_counts = load_explicit(EXPLICIT_PATHS, release_splits, keep_splits)
    print(f"  split breakdown of the explicit files: {split_counts}")
    print(f"  keeping splits {sorted(keep_splits)}: {len(exp_records)} (model, question) pairs")

    allowed = set(exp_records) if args.implicit_universe == "paired" else None
    print(f"Loading implicit condition from {IMPLICIT_PATH.name} "
          f"(universe={args.implicit_universe}) ...")
    imp_records = load_implicit(IMPLICIT_PATH, allowed)
    print(f"  {len(imp_records)} (model, question) pairs")

    question_ids = sorted({q for (_, q) in exp_records} | {q for (_, q) in imp_records})
    q_index = {q: i for i, q in enumerate(question_ids)}
    print(f"  {len(question_ids)} unique question_ids")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_model_arrs = {m: per_model_question_arrays(m, question_ids, q_index,
                                                   exp_records, imp_records)
                      for m in ALL_MODELS}

    rows = []
    nan_counts = {}

    def process_group(name, models, group_label):
        derived = sum_group_arrs([make_derived_arrs(per_model_arrs[m]) for m in models])
        point = decompose(derived, w=1)

        for o in OUTCOMES:
            p = point[o]
            assert abs((p["selection"] + p["quality"] + p["volume"]) - p["delta"]) < 1e-9, \
                f"telescoping failed for {name}/{o}: {p}"
        for t in TERMS:
            total = sum(point[o][t] for o in OUTCOMES)
            assert abs(total) < 1e-9, f"{t} does not sum to zero across outcomes for {name}: {total}"

        boot, n_nan = bootstrap_group(derived, len(question_ids), args.bootstrap, args.seed)
        nan_counts[name] = n_nan

        row = {"model": name, "group": group_label, **point["denoms"]}
        for o in OUTCOMES:
            # raw counts alongside the rates: with four attribution classes
            # some cells get thin, and a rate without its numerator hides that
            row[f"n_imp__{o}"] = int(round(point[o]["rate_imp"] * point["denoms"]["n_cites_imp"]))
            row[f"n_exp__{o}"] = int(round(point[o]["rate_exp"] * point["denoms"]["n_cites_exp"]))
            row[f"rate_imp__{o}"] = pct(point[o]["rate_imp"])
            row[f"rate_exp__{o}"] = pct(point[o]["rate_exp"])
            row[f"rate_exp_firstk__{o}"] = pct(point[o]["rate_exp_firstk"])
            row[f"rate_exp_beyondk__{o}"] = pct(point[o]["rate_exp_beyondk"])
            for t in TERMS:
                lo, hi = ci_from_boot(boot[o][t])
                row[f"{t}__{o}"] = pct(point[o][t])
                row[f"{t}__{o}__lo"] = lo
                row[f"{t}__{o}__hi"] = hi
        return row

    for model in ALL_MODELS:
        rows.append(process_group(
            model, [model], "frontier" if model in FRONTIER_MODELS else "open-weight"))
    rows.append(process_group("__OPEN__", OPEN_WEIGHT_MODELS, "open-weight-pooled"))
    rows.append(process_group("__FRONTIER__", FRONTIER_MODELS, "frontier-pooled"))

    # ---- CSV ----
    csv_path = args.out_dir / f'cost_of_asking_{AXES[AXIS]["stem"]}_permodel.csv'
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ---- stdout table ----
    print("\nDelta / Selection / Quality / Volume, in points, per outcome")
    header = f"{'model':20s} {'outcome':11s} {'ncit_i':>7s} {'ncit_e':>7s} " \
             f"{'rate_i':>7s} {'rate_e':>7s} {'delta':>18s} {'sel':>18s} {'qual':>18s} {'vol':>18s}"
    print(header)
    print("-" * len(header))
    ordered = [r for r in rows if r["model"] in ALL_MODELS]
    # sort by the delta of the axis's "good" outcome, whichever that is
    ordered.sort(key=lambda r: r[f"delta__{OUTCOMES[0]}"])
    for r in ordered + [r for r in rows if r["model"].startswith("__")]:
        if r["model"].startswith("__"):
            print("-" * len(header))
        for o in OUTCOMES:
            cells = " ".join(
                f"{fmt(r[f'{t}__{o}']):>6s}[{fmt(r[f'{t}__{o}__lo'])},{fmt(r[f'{t}__{o}__hi'])}]"
                for t in TERMS)
            print(f"{DISPLAY_NAMES[r['model']]:20s} {o:11s} {r['n_cites_imp']:7d} "
                  f"{r['n_cites_exp']:7d} {fmt(r[f'rate_imp__{o}']):>7s} "
                  f"{fmt(r[f'rate_exp__{o}']):>7s} {cells}")

    print("\nDegenerate bootstrap resamples (no shared pairs), per group:")
    for name, cnt in nan_counts.items():
        print(f"  {DISPLAY_NAMES[name]:22s} {cnt}")

    open_row = next(r for r in rows if r["model"] == "__OPEN__")
    frontier_row = next(r for r in rows if r["model"] == "__FRONTIER__")

    if AXIS == "attribution":
        # Is the effect carried by enough data to be worth reporting? Print the
        # raw cell counts behind every pooled rate, plus the conditional view
        # the reader will reach for, with its own denominators.
        print("\n=== Attribution: cell counts behind the pooled rates ===")
        hdr = f"{'group':20s} {'condition':10s} " + " ".join(f"{o:>16s}" for o in OUTCOMES) + f" {'total':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for name, r in (("Open-weight pooled", open_row), ("Frontier pooled", frontier_row)):
            for cond, nk in (("implicit", "n_cites_imp"), ("explicit", "n_cites_exp")):
                cells = " ".join(f"{r[f'n_{cond[:3]}__{o}']:16,d}" for o in OUTCOMES)
                print(f"{name:20s} {cond:10s} {cells} {r[nk]:8,d}")
        print("\nConditional view (accuracy AMONG citations whose attribution is judgeable,")
        print("i.e. excluding fabricated text and citations offering no source). This")
        print("conditions on a post-treatment variable -- the prompt changes which")
        print("citations are judgeable -- so it is reported for reference, NOT as the")
        print("headline effect:")
        for name, r in (("Open-weight pooled", open_row), ("Frontier pooled", frontier_row)):
            for cond in ("imp", "exp"):
                ok = r[f"n_{cond}__attr_correct"]
                bad = r[f"n_{cond}__attr_incorrect"]
                den = ok + bad
                acc = 100.0 * ok / den if den else float("nan")
                print(f"  {name:20s} {'implicit' if cond == 'imp' else 'explicit':9s} "
                      f"{acc:5.1f}% correct of {den:,} judgeable ({ok:,} correct, {bad:,} misattributed)")

    if AXIS == "fidelity":
        print("\n=== Continuity check vs. the binary figure ===")
        print("The 'correct' outcome is the binary accuracy decomposition. On the full")
        print("data (all splits, both arms) the published values were:")
        print("  open:     delta -14.4  selection -10.2  quality +0.8  volume -5.0")
        print("  frontier: delta  +3.3  selection  +1.3  quality +3.0  volume -1.1")
        print(f"Here (splits={sorted(keep_splits)}, implicit-universe={args.implicit_universe}):")
        for label, r in (("open", open_row), ("frontier", frontier_row)):
            print(f"  {label:9s} delta {r['delta__correct']:+6.1f}  "
                  f"selection {r['selection__correct']:+6.1f}  "
                  f"quality {r['quality__correct']:+6.1f}  volume {r['volume__correct']:+6.1f}")
        print("Differences from the published values are expected: the response universe")
        print("is restricted here, so these are not the same populations.")

    print(f"\nWrote {make_latex_table(rows, args.out_dir, args.bootstrap)}")

    if not args.no_figure:
        png, pdf = make_figure(open_row, frontier_row, args.out_dir, args.bootstrap, args.dpi)
        print(f"\nWrote {png}")
        print(f"Wrote {pdf}")

    print(f"\nWall-clock time: {time.time() - t0:.1f}s (B={args.bootstrap})")


if __name__ == "__main__":
    main()
