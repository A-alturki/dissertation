#!/usr/bin/env python3
"""
score_false_khata.py — mirror of score_manual_gold.py's false-corr metric, but for the
OPPOSITE error: spans where the majority gold has an actual correction (MAJ_CORR nonempty,
no khata-majority), and the model wrongly declined (said خطأ) instead of proposing one.

false-corr    = model proposed X | majority gold says خطأ   (over-correction)
false-khata   = model said خطأ  | majority gold says correctable  (under-correction / missed fix)

    python score_false_khata.py
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
ref_of, MAJ_CORR, KHATA_MAJ = {}, {}, {}
for s in gd["spans"]:
    ns = norm.normalize_text(s["span"]); ref = s["ref"]; ref_of[ns] = ref
    majc = [gunit(c) for c in s["gold"] if not c["is_khata"] and c["count"] >= 2]
    MAJ_CORR[ns] = U.dedup(majc, ref)
    KHATA_MAJ[ns] = any(c["is_khata"] and c["count"] >= 2 for c in s["gold"])
ALL = sorted(ref_of)

def compute(spec):
    units, declined, _vm, _preds = U.model_accessor(spec)
    out = {"name": spec["name"]}
    for scope in [None, "quran", "hadith"]:
        spans = [ns for ns in ALL if scope is None or ref_of[ns] == scope]
        fk = fk_n = 0
        for ns in spans:
            if MAJ_CORR[ns] and not KHATA_MAJ[ns]:   # majority says: there IS a correction
                fk_n += 1
                dec = declined(ns)
                if dec:                               # model declined (said خطأ) -> missed it
                    fk += 1
        out[scope or "all"] = {"false_khata": fk, "correctable_maj_spans": fk_n}
    return out

def main():
    results = [compute(m) for m in U.MODELS]
    print(f"{'Model':16} {'false-khata (all)':>20} {'quran':>12} {'hadith':>12}")
    for r in results:
        a, q, h = r["all"], r["quran"], r["hadith"]
        print(f"{r['name']:16} {str(a['false_khata'])+'/'+str(a['correctable_maj_spans']):>20} "
              f"{str(q['false_khata'])+'/'+str(q['correctable_maj_spans']):>12} "
              f"{str(h['false_khata'])+'/'+str(h['correctable_maj_spans']):>12}")
    json.dump(results, open(os.path.join(HERE, "false_khata_results.json"), "w", encoding="utf-8"),
               ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
