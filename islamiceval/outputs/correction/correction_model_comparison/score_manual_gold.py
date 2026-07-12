#!/usr/bin/env python3
"""
score_manual_gold.py — re-score every model against the CURATOR gold (gold_100_final.json),
replacing the buggy annotator-derived gold. Reuses the harness's matching (umatch/shares/
dedup) and model loaders (unified_eval_local.model_accessor) so only the gold changes.

Gold model (per span): a set of accepted corrections, each with a backer `count`; خطأ is a
candidate too. Majority = count>=2. Tier-2 = any accepted candidate.

  python score_manual_gold.py   ->  prints tables + writes manual_gold_results.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unified_eval_local as U          # matching + model accessors (its own gold is ignored)
import norm

GOLD_FILE = os.path.join(HERE, "..", "..", "classification", "gold_100_final.json")

# ---- build gold structures from the curator file, keyed by normalized span ----
gd = json.load(open(GOLD_FILE, encoding="utf-8"))
def gunit(c):
    return {"text": norm.normalize_text(c["text"]),
            "ref": norm.extract_ref((c.get("sources") or [""])[0] or c["text"])}
ref_of, GOLD_CORR, MAJ_CORR, KHATA_ANY, KHATA_MAJ = {}, {}, {}, {}, {}
for s in gd["spans"]:
    ns = norm.normalize_text(s["span"]); ref = s["ref"]; ref_of[ns] = ref
    corr = [gunit(c) for c in s["gold"] if not c["is_khata"]]
    majc = [gunit(c) for c in s["gold"] if not c["is_khata"] and c["count"] >= 2]
    GOLD_CORR[ns] = U.dedup(corr, ref)
    MAJ_CORR[ns]  = U.dedup(majc, ref)
    KHATA_ANY[ns] = any(c["is_khata"] for c in s["gold"])
    KHATA_MAJ[ns] = any(c["is_khata"] and c["count"] >= 2 for c in s["gold"])
ALL = sorted(ref_of)


def compute(spec):
    units, declined, _vm, _preds = U.model_accessor(spec)
    out = {"name": spec["name"]}
    for scope in [None, "quran", "hadith"]:
        spans = [ns for ns in ALL if scope is None or ref_of[ns] == scope]
        mg_ok = mg_n = dec_ok = dec_n = fc = fc_n = 0
        rec = prec = np_ = ge1 = cov_all = cov_part = cov_none = 0
        for ns in spans:
            ref = ref_of[ns]; mu = units(ns); dec = declined(ns)
            has_maj = KHATA_MAJ[ns] or bool(MAJ_CORR[ns])
            # --- Tier-1 majority-gold accuracy + decision + false-correction ---
            if has_maj:
                mg_n += 1
                mg_ok += (dec and KHATA_MAJ[ns]) or U.shares(mu, MAJ_CORR[ns], ref)
                dec_n += 1
                dec_ok += (dec and KHATA_MAJ[ns]) or ((not dec) and bool(MAJ_CORR[ns]))
            if KHATA_MAJ[ns] and not MAJ_CORR[ns]:          # majority says خطأ (no majority fix)
                fc_n += 1
                if not dec:
                    fc += 1
            # --- Tier-2 coverage / recall / precision (every span has gold) ---
            g = GOLD_CORR[ns]
            n_gold = len(g) + (1 if KHATA_ANY[ns] else 0)
            m = sum(1 for gc in g if any(U.umatch(mc, gc, ref) for mc in mu))
            if KHATA_ANY[ns] and dec:
                m += 1
            if n_gold:
                rec += m / n_gold
            if m >= 1: ge1 += 1
            if m == n_gold: cov_all += 1
            elif m >= 1: cov_part += 1
            else: cov_none += 1
            if mu:
                prec += sum(1 for mc in mu if any(U.umatch(mc, gc, ref) for gc in g)) / len(mu); np_ += 1
            elif dec and KHATA_ANY[ns]:
                prec += 1; np_ += 1
        n = len(spans)
        out[scope or "all"] = {
            "n": n,
            "majority_gold": round(mg_ok / mg_n, 4) if mg_n else None,
            "decision_acc":  round(dec_ok / dec_n, 4) if dec_n else None,
            "ge1_candidate": round(ge1 / n, 4) if n else None,
            "recall":        round(rec / n, 4) if n else None,
            "precision":     round(prec / np_, 4) if np_ else None,
            "false_corr": fc, "khata_maj_spans": fc_n,
            "cov_all": cov_all, "cov_partial": cov_part, "cov_none": cov_none,
        }
    return out


def fmt(v): return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main():
    results = [compute(m) for m in U.MODELS]
    json.dump(results, open(os.path.join(HERE, "manual_gold_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"CURATOR-GOLD scoring on {len(ALL)} spans "
          f"(Quran {sum(1 for n in ALL if ref_of[n]=='quran')} / Hadith {sum(1 for n in ALL if ref_of[n]=='hadith')})")
    print("majority-gold = matches a >=2-backed answer (a majority correction, or خطأ if خطأ is majority) | "
          "decision = correctable-vs-خطأ | false-corr = proposed where majority says خطأ\n")
    for scope in ["all", "quran", "hadith"]:
        print(f"## {scope.upper()}")
        print(f"{'Model':16} {'maj-gold':>9} {'decision':>9} {'≥1-cand':>8} {'recall':>7} {'precision':>10} {'false-corr':>11}")
        for r in results:
            x = r[scope]
            print(f"{r['name']:16} {fmt(x['majority_gold']):>9} {fmt(x['decision_acc']):>9} "
                  f"{fmt(x['ge1_candidate']):>8} {fmt(x['recall']):>7} {fmt(x['precision']):>10} "
                  f"{str(x['false_corr'])+'/'+str(x['khata_maj_spans']):>11}")
        print()
    print("Coverage tiers (ALL): covered-all / partial / none")
    for r in results:
        x = r["all"]; print(f"  {r['name']:16} {x['cov_all']} / {x['cov_partial']} / {x['cov_none']}")


if __name__ == "__main__":
    main()
