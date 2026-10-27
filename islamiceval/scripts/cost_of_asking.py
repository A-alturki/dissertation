"""
cost_of_asking.py

Decomposes the "cost of asking explicitly" for citation-grounding accuracy
into Selection / Quality / Volume components, comparing an EXPLICIT-prompt
condition (models told to cite Ayah/Hadith) against an IMPLICIT-prompt
("vanilla") condition, over the same 500 questions x 12 models.

Definitions
-----------
A CITATION is an annotation segment with type in {"Ayah", "matn"}
(claimed_source / isnad segments are ignored). A citation is CORRECT iff its
segment label == "correct". Citations within an answer are ordered by
span_start.

Join unit = (model, question_id).

Per model (and pooled "Open-weight (10)" / "Frontier (2)" groups):
    E(all)   = accuracy over ALL explicit citations
    I(all)   = accuracy over ALL implicit citations
    shared   = (model, question_id) pairs with >=1 citation under BOTH
               prompt conditions
    k_q      = number of implicit citations for a shared pair
    explicit "first-k" = the first k_q explicit citations (by span_start)
               for that pair (or all of them if explicit has < k_q)

    Delta     = E(all) - I(all)
    Quality   = E(shared, first-k) - I(shared, all)
    Volume    = E(shared, all)     - E(shared, first-k)
    Selection = [E(all) - E(shared)] + [I(shared) - I(all)]

These telescope exactly: Selection + Quality + Volume == Delta (asserted
to 1e-9 per model).

A question-level bootstrap (resample the 500 question_ids with replacement)
gives 95% CIs for Delta / Selection / Quality / Volume.

Outputs
-------
outputs/analysis/cost_of_asking/cost_of_asking_permodel.csv
outputs/analysis/cost_of_asking/cost_of_asking_permodel.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPLICIT_PATH = REPO_ROOT / "data" / "__FINAL_ANALYSIS_SET__" / "fanar_all_corrected.jsonl"
IMPLICIT_PATH = REPO_ROOT / "data" / "__FINAL_ANALYSIS_SET__" / "vanilla12_corrected.jsonl"
MAPPING_PATH = REPO_ROOT / "data" / "islamiceval2026_RELEASE" / "private_mapping.csv"

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "cost_of_asking"

CITATION_TYPES = {"Ayah", "matn"}

FRONTIER_MODELS = ["gpt-5.4", "gemini-3.1-pro"]
OPEN_WEIGHT_MODELS = [
    "acegpt-8b", "allam-7b", "command-r-7b", "fanar-1-9b", "gemma-3-12b",
    "jais-13b", "llama-3.1-8b", "mistral-7b", "phi-3.5", "qwen3-8b",
]
ALL_MODELS = OPEN_WEIGHT_MODELS + FRONTIER_MODELS

DISPLAY_NAMES = {
    "llama-3.1-8b": "Llama-3.1-8B",
    "phi-3.5": "Phi-3.5-mini-4B",
    "gemma-3-12b": "Gemma-3-12B",
    "qwen3-8b": "Qwen3-8B",
    "acegpt-8b": "AceGPT-v2-8B",
    "command-r-7b": "Command-R-7B",
    "mistral-7b": "Mistral-7B",
    "allam-7b": "ALLaM-7B",
    "fanar-1-9b": "Fanar-1-9B",
    "gpt-5.4": r"GPT-5.4\fr",
    "gemini-3.1-pro": r"Gemini-3.1-Pro\fr",
    "jais-13b": "Jais-13B",
    "__OPEN__": "Open-weight pooled",
    "__FRONTIER__": "Frontier pooled",
}


def normalize_model(name: str) -> str:
    if name == "gemini-pro-3.1":
        return "gemini-3.1-pro"
    return name


def extract_citations(annotations: list) -> list:
    """Return list of (span_start, is_correct) for all Ayah/matn segments in
    an answer, sorted by span_start."""
    cites = []
    for ann in annotations:
        for seg in ann.get("segments", []):
            if seg.get("type") in CITATION_TYPES:
                label = seg.get("label")
                cites.append((seg["span_start"], label == "correct"))
    cites.sort(key=lambda x: x[0])
    return cites


def load_explicit(path: Path, mapping_path: Path):
    with open(mapping_path, encoding="utf-8") as f:
        mapping = {row["new_id"]: normalize_model(row["source_model"])
                   for row in csv.DictReader(f)}

    records = {}  # (model, qid) -> list[(span_start, correct)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            model = mapping.get(row["id"])
            if model is None:
                continue
            qid = row["question_id"]
            cites = extract_citations(row["annotations"])
            key = (model, qid)
            if key in records:
                raise ValueError(f"duplicate explicit (model, qid) pair: {key}")
            records[key] = cites
    return records


def load_implicit(path: Path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            model = row["model"]
            qid = row["question_id"]
            cites = extract_citations(row["annotations"])
            key = (model, qid)
            if key in records:
                raise ValueError(f"duplicate implicit (model, qid) pair: {key}")
            records[key] = cites
    return records


def build_question_index(question_ids):
    return {qid: i for i, qid in enumerate(question_ids)}


def per_model_question_arrays(model, question_ids, q_index, exp_records, imp_records):
    """Build per-question count arrays for one model.

    Returns a dict of numpy arrays, each length len(question_ids):
      n_imp, c_imp, n_exp, c_exp, n_exp_firstk, c_exp_firstk, shared (0/1)
    """
    n = len(question_ids)
    n_imp = np.zeros(n, dtype=np.int64)
    c_imp = np.zeros(n, dtype=np.int64)
    n_exp = np.zeros(n, dtype=np.int64)
    c_exp = np.zeros(n, dtype=np.int64)
    n_exp_firstk = np.zeros(n, dtype=np.int64)
    c_exp_firstk = np.zeros(n, dtype=np.int64)
    shared = np.zeros(n, dtype=np.int64)

    for qid in question_ids:
        idx = q_index[qid]
        imp_cites = imp_records.get((model, qid))
        exp_cites = exp_records.get((model, qid))

        if imp_cites is not None:
            n_imp[idx] = len(imp_cites)
            c_imp[idx] = sum(1 for _, ok in imp_cites if ok)
        if exp_cites is not None:
            n_exp[idx] = len(exp_cites)
            c_exp[idx] = sum(1 for _, ok in exp_cites if ok)

        if imp_cites and exp_cites:
            shared[idx] = 1
            k = len(imp_cites)
            first_k = exp_cites[:k]
            n_exp_firstk[idx] = len(first_k)
            c_exp_firstk[idx] = sum(1 for _, ok in first_k if ok)

    return {
        "n_imp": n_imp, "c_imp": c_imp,
        "n_exp": n_exp, "c_exp": c_exp,
        "n_exp_firstk": n_exp_firstk, "c_exp_firstk": c_exp_firstk,
        "shared": shared,
    }


def safe_div(num, den):
    return num / den if den > 0 else float("nan")


def make_derived_arrs(arrs):
    """From ONE model's per-question arrays, build the "shared-masked"
    derived arrays. Masking by `shared` is done here, per model, BEFORE any
    cross-model pooling — this is essential: pooling raw per-question counts
    across models first and only then masking by an aggregated shared-count
    would leak non-shared citations from other models into shared totals.

    n_exp_firstk / c_exp_firstk are already zero for non-shared questions
    (per_model_question_arrays only fills them for shared pairs), so they
    need no extra masking.
    """
    shared = arrs["shared"]
    return {
        "n_imp": arrs["n_imp"], "c_imp": arrs["c_imp"],
        "n_exp": arrs["n_exp"], "c_exp": arrs["c_exp"],
        "n_exp_firstk": arrs["n_exp_firstk"], "c_exp_firstk": arrs["c_exp_firstk"],
        "n_exp_shared": arrs["n_exp"] * shared, "c_exp_shared": arrs["c_exp"] * shared,
        "n_imp_shared": arrs["n_imp"] * shared, "c_imp_shared": arrs["c_imp"] * shared,
    }


def compute_stats_from_derived(d, weights=None):
    """weights: None (unit weights, i.e. observed data) or a length-n
    integer array of question multiplicities (for bootstrap resamples).
    `d` must already have shared-masking baked in per make_derived_arrs
    (summed across models is fine, since masking happened before summing).
    """
    w = 1 if weights is None else weights

    N_imp_all = (d["n_imp"] * w).sum()
    C_imp_all = (d["c_imp"] * w).sum()
    N_exp_all = (d["n_exp"] * w).sum()
    C_exp_all = (d["c_exp"] * w).sum()

    N_exp_fk = (d["n_exp_firstk"] * w).sum()
    C_exp_fk = (d["c_exp_firstk"] * w).sum()
    N_exp_sh = (d["n_exp_shared"] * w).sum()
    C_exp_sh = (d["c_exp_shared"] * w).sum()
    N_imp_sh = (d["n_imp_shared"] * w).sum()
    C_imp_sh = (d["c_imp_shared"] * w).sum()

    E_all = safe_div(C_exp_all, N_exp_all)
    I_all = safe_div(C_imp_all, N_imp_all)
    E_shared_all = safe_div(C_exp_sh, N_exp_sh)
    I_shared_all = safe_div(C_imp_sh, N_imp_sh)
    E_shared_fk = safe_div(C_exp_fk, N_exp_fk)

    delta = E_all - I_all
    quality = E_shared_fk - I_shared_all
    volume = E_shared_all - E_shared_fk
    selection = (E_all - E_shared_all) + (I_shared_all - I_all)

    beyondk_n = N_exp_sh - N_exp_fk
    beyondk_c = C_exp_sh - C_exp_fk
    acc_beyondk = safe_div(beyondk_c, beyondk_n)

    return {
        "n_cites_imp": int(N_imp_all),
        "n_cites_exp": int(N_exp_all),
        "acc_imp": I_all,
        "acc_exp": E_all,
        "delta": delta,
        "selection": selection,
        "quality": quality,
        "volume": volume,
        "acc_exp_firstk": E_shared_fk,
        "acc_exp_beyondk": acc_beyondk,
        "n_beyondk": int(beyondk_n),
    }


def bootstrap_group(derived, question_ids, B, seed):
    """derived: dict of length-n arrays, already shared-masked per model and
    summed across the models in the group (see make_derived_arrs / the sum
    of several such dicts). Returns dict of arrays: delta[B], selection[B],
    quality[B], volume[B], each length B (NaN where a resample has no
    shared/implicit/explicit citations at all for this group).
    """
    rng = np.random.default_rng(seed)
    n = len(question_ids)

    n_imp = derived["n_imp"]; c_imp = derived["c_imp"]
    n_exp = derived["n_exp"]; c_exp = derived["c_exp"]
    n_exp_fk = derived["n_exp_firstk"]; c_exp_fk = derived["c_exp_firstk"]
    n_exp_shared = derived["n_exp_shared"]; c_exp_shared = derived["c_exp_shared"]
    n_imp_shared = derived["n_imp_shared"]; c_imp_shared = derived["c_imp_shared"]

    deltas = np.full(B, np.nan)
    selections = np.full(B, np.nan)
    qualities = np.full(B, np.nan)
    volumes = np.full(B, np.nan)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        w = np.bincount(idx, minlength=n)

        N_imp_all = (n_imp * w).sum()
        C_imp_all = (c_imp * w).sum()
        N_exp_all = (n_exp * w).sum()
        C_exp_all = (c_exp * w).sum()

        N_exp_fk_sh = (n_exp_fk * w).sum()
        C_exp_fk_sh = (c_exp_fk * w).sum()
        N_exp_sh = (n_exp_shared * w).sum()
        C_exp_sh = (c_exp_shared * w).sum()
        N_imp_sh = (n_imp_shared * w).sum()
        C_imp_sh = (c_imp_shared * w).sum()

        if N_exp_all == 0 or N_imp_all == 0 or N_exp_sh == 0 or N_imp_sh == 0 or N_exp_fk_sh == 0:
            continue

        E_all = C_exp_all / N_exp_all
        I_all = C_imp_all / N_imp_all
        E_sh = C_exp_sh / N_exp_sh
        I_sh = C_imp_sh / N_imp_sh
        E_fk = C_exp_fk_sh / N_exp_fk_sh

        delta = E_all - I_all
        quality = E_fk - I_sh
        volume = E_sh - E_fk
        selection = (E_all - E_sh) + (I_sh - I_all)

        deltas[b] = delta
        selections[b] = selection
        qualities[b] = quality
        volumes[b] = volume

    return {"delta": deltas, "selection": selections, "quality": qualities, "volume": volumes}


def sum_group_arrs(list_of_arrs):
    keys = list_of_arrs[0].keys()
    return {k: sum(a[k] for a in list_of_arrs) for k in keys}


def pct(x):
    return float("nan") if x != x else 100.0 * x


def ci_from_boot(boot_vals):
    lo = np.nanpercentile(boot_vals, 2.5)
    hi = np.nanpercentile(boot_vals, 97.5)
    n_nan = int(np.isnan(boot_vals).sum())
    return pct(lo), pct(hi), n_nan


def fmt_num(x):
    if x != x:
        return "NaN"
    return f"{x:.1f}"


def latex_cell(point, lo, hi, bold):
    p = f"{point:.1f}"
    lo_s = f"{lo:.1f}"
    hi_s = f"{hi:.1f}"
    p_math = rf"\mathbf{{{p}}}" if bold else p
    return rf"${p_math}$ {{\scriptsize[${lo_s}$, ${hi_s}$]}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=8000, help="number of bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    t0 = time.time()

    print(f"Loading explicit condition from {EXPLICIT_PATH} ...")
    exp_records = load_explicit(EXPLICIT_PATH, MAPPING_PATH)
    print(f"  {len(exp_records)} (model, question_id) pairs")

    print(f"Loading implicit condition from {IMPLICIT_PATH} ...")
    imp_records = load_implicit(IMPLICIT_PATH)
    print(f"  {len(imp_records)} (model, question_id) pairs")

    question_ids = sorted({qid for (_, qid) in exp_records} | {qid for (_, qid) in imp_records})
    q_index = build_question_index(question_ids)
    print(f"  {len(question_ids)} unique question_ids")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model per-question arrays
    per_model_arrs = {}
    for model in ALL_MODELS:
        per_model_arrs[model] = per_model_question_arrays(
            model, question_ids, q_index, exp_records, imp_records
        )

    rows = []
    nan_counts = {}

    def process_group(name, models, group_label):
        arrs_list = [per_model_arrs[m] for m in models]

        # Mask by `shared` at the per-model level BEFORE pooling across
        # models (see make_derived_arrs docstring) — this is required for
        # correctness of both the point estimate and the bootstrap.
        derived_list = [make_derived_arrs(a) for a in arrs_list]
        derived_summed = sum_group_arrs(derived_list)

        point = compute_stats_from_derived(derived_summed, weights=None)

        # n_q_imp / n_q_exp / n_shared for a pooled group: count of
        # (model,question) pairs, not distinct questions -> compute directly.
        point["n_q_imp"] = int(sum((a["n_imp"] > 0).sum() for a in arrs_list))
        point["n_q_exp"] = int(sum((a["n_exp"] > 0).sum() for a in arrs_list))
        point["n_shared"] = int(sum((a["shared"] > 0).sum() for a in arrs_list))

        # sanity: telescoping
        assert abs((point["selection"] + point["quality"] + point["volume"]) - point["delta"]) < 1e-9, \
            f"telescoping failed for {name}: {point}"

        boot = bootstrap_group(derived_summed, question_ids, args.bootstrap, args.seed)
        d_lo, d_hi, d_nan = ci_from_boot(boot["delta"])
        s_lo, s_hi, s_nan = ci_from_boot(boot["selection"])
        q_lo, q_hi, q_nan = ci_from_boot(boot["quality"])
        v_lo, v_hi, v_nan = ci_from_boot(boot["volume"])

        nan_counts[name] = d_nan  # they all share the same "no shared pairs" condition

        row = {
            "model": name,
            "group": group_label,
            "n_q_imp": point["n_q_imp"],
            "n_q_exp": point["n_q_exp"],
            "n_shared": point["n_shared"],
            "n_cites_imp": point["n_cites_imp"],
            "n_cites_exp": point["n_cites_exp"],
            "acc_imp": pct(point["acc_imp"]),
            "acc_exp": pct(point["acc_exp"]),
            "delta": pct(point["delta"]), "delta_lo": d_lo, "delta_hi": d_hi,
            "selection": pct(point["selection"]), "sel_lo": s_lo, "sel_hi": s_hi,
            "quality": pct(point["quality"]), "qual_lo": q_lo, "qual_hi": q_hi,
            "volume": pct(point["volume"]), "vol_lo": v_lo, "vol_hi": v_hi,
            "acc_exp_firstk": pct(point["acc_exp_firstk"]),
            "acc_exp_beyondk": pct(point["acc_exp_beyondk"]),
            "n_beyondk": point["n_beyondk"],
        }
        return row

    for model in ALL_MODELS:
        group_label = "frontier" if model in FRONTIER_MODELS else "open-weight"
        rows.append(process_group(model, [model], group_label))

    rows.append(process_group("__OPEN__", OPEN_WEIGHT_MODELS, "open-weight-pooled"))
    rows.append(process_group("__FRONTIER__", FRONTIER_MODELS, "frontier-pooled"))

    # ---- write CSV ----
    csv_path = args.out_dir / "cost_of_asking_permodel.csv"
    fieldnames = [
        "model", "group", "n_q_imp", "n_q_exp", "n_shared", "n_cites_imp", "n_cites_exp",
        "acc_imp", "acc_exp", "delta", "delta_lo", "delta_hi",
        "selection", "sel_lo", "sel_hi", "quality", "qual_lo", "qual_hi",
        "volume", "vol_lo", "vol_hi", "acc_exp_firstk", "acc_exp_beyondk", "n_beyondk",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # ---- LaTeX table ----
    model_rows = [r for r in rows if r["model"] in ALL_MODELS]
    model_rows.sort(key=lambda r: r["delta"])  # ascending, most negative first
    pooled_rows = [r for r in rows if r["model"] in ("__OPEN__", "__FRONTIER__")]

    def sig(lo, hi):
        return not (lo <= 0 <= hi)

    def tex_row(r):
        name = DISPLAY_NAMES[r["model"]]
        d = latex_cell(r["delta"], r["delta_lo"], r["delta_hi"], sig(r["delta_lo"], r["delta_hi"]))
        s = latex_cell(r["selection"], r["sel_lo"], r["sel_hi"], sig(r["sel_lo"], r["sel_hi"]))
        q = latex_cell(r["quality"], r["qual_lo"], r["qual_hi"], sig(r["qual_lo"], r["qual_hi"]))
        v = latex_cell(r["volume"], r["vol_lo"], r["vol_hi"], sig(r["vol_lo"], r["vol_hi"]))
        return f"{name} & {d} & {s} & {q} & {v} \\\\"

    tex_lines = []
    tex_lines.append(r"\begin{tabular}{lrrrr}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Model & $\Delta$ (pp) & Selection & Quality & Volume \\")
    tex_lines.append(r"\midrule")
    for r in model_rows:
        tex_lines.append(tex_row(r))
    tex_lines.append(r"\midrule")
    for r in pooled_rows:
        tex_lines.append(tex_row(r))
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")

    tex_path = args.out_dir / "cost_of_asking_permodel.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Wrote {tex_path}")

    # ---- plain text table to stdout ----
    header = (
        f"{'model':16s} {'n_q_i':>6s} {'n_q_e':>6s} {'n_sh':>6s} "
        f"{'ncit_i':>7s} {'ncit_e':>7s} {'acc_i':>7s} {'acc_e':>7s} "
        f"{'delta':>9s} {'sel':>9s} {'qual':>9s} {'vol':>9s} "
        f"{'exp_fk':>7s} {'exp_bk':>7s} {'n_bk':>6s}"
    )
    print("\n" + header)
    print("-" * len(header))
    all_print_rows = model_rows + pooled_rows
    for r in all_print_rows:
        name = DISPLAY_NAMES.get(r["model"], r["model"])
        print(
            f"{name:16s} {r['n_q_imp']:6d} {r['n_q_exp']:6d} {r['n_shared']:6d} "
            f"{r['n_cites_imp']:7d} {r['n_cites_exp']:7d} "
            f"{fmt_num(r['acc_imp']):>7s} {fmt_num(r['acc_exp']):>7s} "
            f"{fmt_num(r['delta']):>5s}[{fmt_num(r['delta_lo'])},{fmt_num(r['delta_hi'])}] "
            f"{fmt_num(r['selection']):>5s}[{fmt_num(r['sel_lo'])},{fmt_num(r['sel_hi'])}] "
            f"{fmt_num(r['quality']):>5s}[{fmt_num(r['qual_lo'])},{fmt_num(r['qual_hi'])}] "
            f"{fmt_num(r['volume']):>5s}[{fmt_num(r['vol_lo'])},{fmt_num(r['vol_hi'])}] "
            f"{fmt_num(r['acc_exp_firstk']):>7s} {fmt_num(r['acc_exp_beyondk']):>7s} {r['n_beyondk']:6d}"
        )

    print("\nNaN bootstrap resamples per model/group (no shared pairs in resample):")
    for name, cnt in nan_counts.items():
        disp = DISPLAY_NAMES.get(name, name)
        print(f"  {disp:20s} {cnt}")

    elapsed = time.time() - t0
    print(f"\nWall-clock time: {elapsed:.1f}s (B={args.bootstrap})")

    # ---- sanity checks against known targets (open-weight pooled) ----
    open_row = next(r for r in rows if r["model"] == "__OPEN__")
    frontier_row = next(r for r in rows if r["model"] == "__FRONTIER__")
    print("\n=== Sanity check (pooled open-weight 10) ===")
    print(f"  delta={open_row['delta']:.1f} (target -14.4)")
    print(f"  selection={open_row['selection']:.1f} (target -10.2)")
    print(f"  quality={open_row['quality']:.1f} (target +0.8)")
    print(f"  volume={open_row['volume']:.1f} (target -5.0)")
    print(f"  acc_exp_firstk={open_row['acc_exp_firstk']:.1f}% (target 41.1%)")
    print(f"  acc_exp_beyondk={open_row['acc_exp_beyondk']:.1f}% n={open_row['n_beyondk']} (target 31.4%, n=1603)")
    print(f"  n_cites_imp={open_row['n_cites_imp']} (target 1745)")
    print(f"  n_cites_exp={open_row['n_cites_exp']} (target 12782)")
    print(f"  frontier delta={frontier_row['delta']:.1f} (target +3.3)")


if __name__ == "__main__":
    main()
