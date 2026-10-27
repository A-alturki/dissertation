#!/usr/bin/env python3
"""
score_precision_proposed.py — precision restricted to spans where the model actually
PROPOSED a correction (units nonempty). Declines are excluded entirely (not counted as
either correct or incorrect, and not in the denominator) — this answers "when the model
does propose a correction, is the text any good", with no credit for correctly declining.

Matched against the Tier-2 gold (GOLD_CORR: every annotator-sanctioned candidate, not just
majority-backed ones) via the same umatch/dedup logic as score_manual_gold.py.
Per-span precision = (# proposed candidates that match some gold candidate) / (# proposed);
reported value = macro-average of that fraction over all spans where the model proposed
>=1 candidate (same averaging style as the existing precision column).

    python score_precision_proposed.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unified_eval_local as U
import norm

GOLD_FILE = os.path.join(HERE, "..", "..", "classification", "gold_100_final.json")

gd = json.load(open(GOLD_FILE, encoding="utf-8"))
def gunit(c):
    return {"text": norm.normalize_text(c["text"]),
            "ref": norm.extract_ref((c.get("sources") or [""])[0] or c["text"])}
ref_of, GOLD_CORR = {}, {}
for s in gd["spans"]:
    ns = norm.normalize_text(s["span"]); ref = s["ref"]; ref_of[ns] = ref
    corr = [gunit(c) for c in s["gold"] if not c["is_khata"]]
    GOLD_CORR[ns] = U.dedup(corr, ref)
ALL = sorted(ref_of)

def compute(spec):
    units, declined, _vm, _preds = U.model_accessor(spec)
    out = {"name": spec["name"]}
    for scope in [None, "quran", "hadith"]:
        spans = [ns for ns in ALL if scope is None or ref_of[ns] == scope]
        prec_sum = n_proposed = n_total_cands = n_correct_cands = 0
        n_declined = 0
        for ns in spans:
            ref = ref_of[ns]
            mu = units(ns)
            if not mu:
                n_declined += 1
                continue
            g = GOLD_CORR[ns]
            correct = sum(1 for mc in mu if any(U.umatch(mc, gc, ref) for gc in g))
            prec_sum += correct / len(mu)
            n_proposed += 1
            n_total_cands += len(mu)
            n_correct_cands += correct
        n = len(spans)
        out[scope or "all"] = {
            "n_spans": n,
            "n_proposed": n_proposed,
            "n_declined": n_declined,
            "precision_macro": round(prec_sum / n_proposed, 4) if n_proposed else None,
            "precision_micro": round(n_correct_cands / n_total_cands, 4) if n_total_cands else None,
            "n_candidates": n_total_cands,
            "n_correct_candidates": n_correct_cands,
        }
    return out

def main():
    results = [compute(m) for m in U.MODELS]
    print(f"{'Model':16} {'precision(all)':>15} {'n_proposed':>11} {'quran':>10} {'hadith':>10}")
    for r in results:
        a, q, h = r["all"], r["quran"], r["hadith"]
        print(f"{r['name']:16} {a['precision_macro']!s:>15} {a['n_proposed']:>11} "
              f"{q['precision_macro']!s:>10} {h['precision_macro']!s:>10}")
    json.dump(results, open(os.path.join(HERE, "precision_proposed_results.json"), "w", encoding="utf-8"),
               ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
