#!/usr/bin/env python3
"""
validate_clean.py
=================
Stratified BOUNDARY-sampling validation for clean_answers.py — for MANUAL adjudication
(no LLM judge). The earlier `_audit_clean.py` only eyeballed drops; this samples both
error directions, specifically where each threshold is riskiest:

  * DROP-near bands  (filter dropped it, but the metric is just past the threshold)
        -> tests DROP PRECISION  (false positives: was this wrongly dropped?)
  * KEEP-inside bands (filter kept it, but the metric is just shy of the threshold)
        -> tests KEEP RECALL     (false negatives: should this have been dropped?)

For each category we recompute the SAME metrics clean_answers uses and replicate its
priority-ordered decision, so `filter_decision` is exactly what clean_answers does. Then
we draw ~--n samples per band (seeded) and write a review sheet with the question, the
cleaned answer, the triggering metric value vs its threshold, and BLANK `verdict`/`note`
columns for you to fill in by hand. Disagreement (your verdict != filter) = a real error.

Bands (thresholds from clean_answers defaults; --eng-max etc. mirror them):
  foreign     drop 0.30<=r<0.40        keep 0.20<=r<0.30
  comp        drop 0.15<=c<0.21        keep 0.21<=c<0.27      (zlib compression ratio)
  word_rep    drop 0.40<=w<0.50        keep 0.30<=w<0.40      (only when comp didn't trigger)
  sent_repeat drop cnt>=4, 0.10<=cov<0.15   keep cnt>=4, 0.07<=cov<0.10
  ngram_loop  drop 0.12<=s<0.16        keep 0.09<=s<0.12
  truncated   drop (all at budget; sampled)  keep budget-40<=tok<budget-8   (needs tokenizer)
  empty       drop (sampled to confirm junk; no keep counterpart)

Usage (run from scripts/):
  python validate_clean.py --input "../outputs/answers/explicit_large_token_size/*_alt2025-explicit.json" \
         --truncation --n 20 --out ../outputs/answers/clean_validation_boundary.xlsx
  python validate_clean.py --input "...*.json" --no-trunc        # skip tokenizer loading
"""
import json, argparse, glob, os, sys, random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_answers as C

# (category, side) -> human label for the sheet
SIDE = {"drop": "FP? (dropped, near threshold)", "keep": "FN? (kept, near threshold)"}


def metrics_and_reason(raw, A, trunc):
    """Recompute every metric once and replay clean_answers' priority-ordered decision.
    Returns (cleaned, reason|None, m) where m holds the metric values + trigger info."""
    cleaned = C.strip_cot(C.strip_leak(raw or "")).strip()
    m = {"foreign": None, "comp": None, "rep": None, "sent_cnt": 0, "sent_cov": 0.0,
         "ngram": None, "tok": None, "budget": None, "deg_trigger": None,
         "ends_punct": C.ends_punct(raw)}
    if not cleaned:
        return cleaned, "empty", m
    m["foreign"] = C.foreign_ratio(cleaned)
    if m["foreign"] >= A["eng_max"]:
        return cleaned, "foreign", m
    if A["min_len"] and len(cleaned) < A["min_len"]:
        return cleaned, "short", m
    m["comp"] = C.comp_ratio(cleaned)
    m["rep"]  = C.word_rep(cleaned)
    if m["comp"] < A["comp_min"]:
        m["deg_trigger"] = "comp"
        return cleaned, "degenerate", m
    if m["rep"] >= A["rep_max"]:
        m["deg_trigger"] = "rep"
        return cleaned, "degenerate", m
    m["sent_cnt"], m["sent_cov"] = C.rep_sentence(cleaned)
    if m["sent_cnt"] >= A["sent_rep"] and m["sent_cov"] >= A["sent_cover"]:
        return cleaned, "sent_repeat", m
    m["ngram"] = C.top_ngram_share(cleaned)
    if m["ngram"] >= A["ngram_max"]:
        return cleaned, "ngram_loop", m
    if trunc is not None:
        tok, budget, margin, broken = trunc
        n = len(tok.encode(raw or "", add_special_tokens=False))
        m["tok"] = n / 2 if broken else n      # store the TRUE count (ALLaM /2)
        m["budget"] = budget
        if broken:
            if 1400 <= m["tok"] <= 1600 and not m["ends_punct"]:
                return cleaned, "truncated", m
        elif n >= budget - margin:
            return cleaned, "truncated", m
    return cleaned, None, m


def band_of(reason, m, A):
    """(category, side, metric_label, value, threshold_str) if the record falls in a
    boundary band, else None. drop-side keys on the actual reason; keep-side on reason=None."""
    e, cmn, rmx = A["eng_max"], A["comp_min"], A["rep_max"]
    sc, sr = A["sent_cover"], A["sent_rep"]
    nmx = A["ngram_max"]
    # ---- DROP-near (false-positive risk) ----
    if reason == "foreign" and e <= m["foreign"] < e + 0.10:
        return "foreign", "drop", "foreign_ratio", m["foreign"], f">= {e}"
    if reason == "degenerate" and m["deg_trigger"] == "comp" and cmn - 0.06 <= m["comp"] < cmn:
        return "comp", "drop", "comp_ratio", m["comp"], f"< {cmn}"
    if reason == "degenerate" and m["deg_trigger"] == "rep" and rmx <= m["rep"] < rmx + 0.10:
        return "word_rep", "drop", "word_rep", m["rep"], f">= {rmx}"
    if reason == "sent_repeat" and sc <= m["sent_cov"] < sc + 0.05:
        return "sent_repeat", "drop", "sent_cov", m["sent_cov"], f"cnt>={sr} & cov>={sc}"
    if reason == "ngram_loop" and nmx <= m["ngram"] < nmx + 0.04:
        return "ngram_loop", "drop", "ngram_share", m["ngram"], f">= {nmx}"
    if reason == "truncated":
        return "truncated", "drop", "tokens", m["tok"], f">= budget({m['budget']})-{A['trunc_margin']}"
    if reason == "empty":
        return "empty", "drop", "-", 0, "nothing left after cleaning"
    # ---- KEEP-inside (false-negative risk); reason is None => passed everything ----
    if reason is None:
        if e - 0.10 <= m["foreign"] < e:
            return "foreign", "keep", "foreign_ratio", m["foreign"], f">= {e}"
        if cmn <= m["comp"] < cmn + 0.06:
            return "comp", "keep", "comp_ratio", m["comp"], f"< {cmn}"
        if rmx - 0.10 <= m["rep"] < rmx:
            return "word_rep", "keep", "word_rep", m["rep"], f">= {rmx}"
        if m["sent_cnt"] >= sr and sc - 0.03 <= m["sent_cov"] < sc:
            return "sent_repeat", "keep", "sent_cov", m["sent_cov"], f"cnt>={sr} & cov>={sc}"
        if m["ngram"] is not None and nmx - 0.03 <= m["ngram"] < nmx:
            return "ngram_loop", "keep", "ngram_share", m["ngram"], f">= {nmx}"
        if m["tok"] is not None and m["budget"] - 40 <= m["tok"] < m["budget"] - A["trunc_margin"]:
            return "truncated", "keep", "tokens", m["tok"], f">= budget({m['budget']})-{A['trunc_margin']}"
    return None


def load_trunc(path, data, args):
    if not args.truncation:
        return None
    model = C.resolve_model(path, data, args)
    if model is None:
        print(f"  WARNING: cannot resolve model for {os.path.basename(path)}; truncation bands skipped")
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(C.TOK_REPO[model])
        return (tok, C.budget_for(model), args.trunc_margin, model in C.BROKEN_RT)
    except Exception as e:
        print(f"  WARNING: tokenizer load failed for {model} ({str(e)[:60]}); truncation bands skipped")
        return None


def score(path):
    """Read a hand-filled sheet and report per-(category,side) agreement.
    verdict is keep/drop. On DROP-near bands filter said 'drop' -> verdict 'keep' = a
    false positive (wrongly dropped) => drop-precision = agree/reviewed. On KEEP-inside
    bands filter said 'keep' -> verdict 'drop' = a false negative (wrongly kept)."""
    df = pd.read_excel(path, sheet_name="boundary")
    df = df[df["verdict"].astype(str).str.strip().str.lower().isin(["keep", "drop"])].copy()
    if df.empty:
        sys.exit("no rows have a keep/drop verdict yet — fill the `verdict` column first.")
    df["v"] = df["verdict"].astype(str).str.strip().str.lower()
    df["filter_keep"] = df["filter_decision"].eq("kept")
    df["verdict_keep"] = df["v"].eq("keep")
    df["agree"] = df["filter_keep"] == df["verdict_keep"]
    rows = []
    for (cat, side), g in df.groupby(["category", "side"]):
        n, ag = len(g), int(g["agree"].sum())
        kind = "drop-precision" if side.startswith("FP") else "keep-correct"
        rows.append({"category": cat, "side": side, "reviewed": n, "agree": ag,
                     "error": n - ag, kind: round(ag / n, 3)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"\noverall agreement {df['agree'].mean():.3f} over {len(df)} reviewed rows")


def main():
    ap = argparse.ArgumentParser(description="Boundary-sampling validation set for clean_answers (manual review).")
    ap.add_argument("--score", metavar="XLSX", help="read a hand-filled sheet and report per-category agreement")
    ap.add_argument("--input", nargs="+")
    ap.add_argument("--out", default="../outputs/answers/clean_validation_boundary.xlsx")
    ap.add_argument("--n", type=int, default=20, help="samples per band (per side, pooled across models)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truncation", action="store_true", default=True)
    ap.add_argument("--no-trunc", dest="truncation", action="store_false")
    ap.add_argument("--ans-chars", type=int, default=0, help="cap answer text in the sheet to N chars (0 = full, never truncate)")
    # mirror clean_answers thresholds
    ap.add_argument("--eng-max", type=float, default=0.30)
    ap.add_argument("--comp-min", type=float, default=0.21)
    ap.add_argument("--rep-max", type=float, default=0.40)
    ap.add_argument("--sent-rep", type=int, default=4)
    ap.add_argument("--sent-cover", type=float, default=0.10)
    ap.add_argument("--ngram-max", type=float, default=0.12)
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--trunc-margin", type=int, default=8)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    if args.score:
        return score(args.score)
    if not args.input:
        ap.error("--input is required (unless using --score)")
    A = dict(eng_max=args.eng_max, comp_min=args.comp_min, rep_max=args.rep_max,
             sent_rep=args.sent_rep, sent_cover=args.sent_cover, ngram_max=args.ngram_max,
             min_len=args.min_len, trunc_margin=args.trunc_margin)

    paths = []
    for pat in args.input:
        paths.extend(sorted(glob.glob(pat)) or [pat])

    pools = {}  # (category, side) -> list of row dicts (the full population)
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        trunc = load_trunc(path, data, args)
        model = (C.resolve_model(path, data, args) or
                 os.path.basename(path).split("_alt2025")[0])
        for r in data:
            raw = r.get("answer", "")
            cleaned, reason, m = metrics_and_reason(raw, A, trunc)
            b = band_of(reason, m, A)
            if b is None:
                continue
            cat, side, label, val, thr = b
            pools.setdefault((cat, side), []).append({
                "model": model, "id": r.get("id"),
                "category": cat, "side": SIDE[side],
                "filter_decision": reason or "kept",
                "metric": label, "value": round(val, 4) if isinstance(val, float) else val,
                "threshold": thr,
                "question": (r.get("prompt") or ""),
                "answer": (cleaned[:args.ans_chars] if args.ans_chars else cleaned),
                # context from the RAW answer: tail shows where truncation cut off;
                # for 'empty' the cleaned text is blank, so show the raw head (what got stripped).
                "raw_tail": (raw[-120:] if cat == "truncated"
                             else (raw[:200] if cat == "empty" else "")),
                "verdict": "", "note": "",
            })
        print(f"  scanned {os.path.basename(path)}")

    rng = random.Random(args.seed)
    rows, summary = [], []
    for (cat, side) in sorted(pools):
        pop = pools[(cat, side)]
        k = min(args.n, len(pop))
        rows.extend(rng.sample(pop, k))
        summary.append({"category": cat, "side": SIDE[side], "population": len(pop), "sampled": k})

    df = pd.DataFrame(rows, columns=["model", "id", "category", "side", "filter_decision",
                                     "metric", "value", "threshold", "question", "answer",
                                     "raw_tail", "verdict", "note"])
    sdf = pd.DataFrame(summary)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="boundary", index=False)
        sdf.to_excel(xw, sheet_name="population", index=False)
        ws = xw.sheets["boundary"]
        widths = {"I": 60, "J": 90, "K": 30, "L": 12, "M": 30}  # question/answer/raw_tail/verdict/note
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        from openpyxl.styles import Alignment
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")

    print("\n=== boundary population (full counts) ===")
    print(sdf.to_string(index=False))
    print(f"\nwrote {len(df)} review rows -> {args.out}")
    print("Fill the `verdict` column (keep/drop) by hand; verdict != filter_decision = an error.")


if __name__ == "__main__":
    main()
