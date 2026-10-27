#!/usr/bin/env python3
"""
score_raw_vs_snapped.py — reconstruct the "effect of snapping + expansion" table
(raw -> snapped+expanded) for all 5 correctors on the 100 curated spans, scored
against the CURATOR gold (gold_100_final.json) with the exact harness matching
used by score_manual_gold.py (Table `tab:corrector` is reproduced by that script).

Definitions (locked to the definitive Table-2 pipeline):
  * RAW side
      - GPT (orig) / Gemini (orig): candidates straight from pipeline_oneshot.json
        (per_sample[].oneshot[model].prediction.candidates), text-only, parsed with
        parse2.parse_units — the same treatment the harness gives prediction files.
      - Pro / Flash / GPT-v3 (revised): EXACTLY unified_eval_local.model_accessor on
        the prediction jsonl (these are the rows of Table 2, which is raw for them).
  * SNAPPED+EXPANDED side
      - GPT (orig) / Gemini (orig): EXACTLY the annotated slots of
        annotators_corrected_enriched.json (snapped+expanded strings the annotators
        judged; these are the rows of Table 2 for the orig models).
      - Pro / Flash / GPT-v3 (revised): each raw candidate resolved with
        ground_corrections.resolve_correction under the LOCKED release config
        (jaccard snap 0.40 + char fallback 0.85, jaccard expand 0.35, skip>=15,
        ALEF_DEL, PARTIAL, HADITH_EXPAND) — the same path as
        build_release_deliverables.ground_span — then formatted like the enriched
        strings ("text  [source]" joined by " ||| ") and parsed with parse_units.
        A span whose candidates ALL eliminate counts as declined (خطأ), matching
        release semantics.

Metrics per (model, side): majority-gold acc, recall, precision, >=1-cand,
and the total number of candidate units the scorer sees (the "Candidates" column).

  python score_raw_vs_snapped.py            -> prints table + writes raw_vs_snapped_results.json
  python score_raw_vs_snapped.py --no-ground  -> skip grounding (raw sides + orig snapped only)
"""
import json, os, re, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "scripts")))
import score_manual_gold as SM      # curator gold structures + harness matching
import unified_eval_local as U
import norm
from parse2 import parse_units

ONESHOT = os.path.abspath(os.path.join(HERE, "..", "pipeline_oneshot.json"))
PREDS = {
    "Gemini-3.1-Pro (revised)": U.PRED_PRO,
    "Gemini-Flash-3.5 (revised)": U.PRED_FLASH,
    "GPT-5.4 (revised)": U.PRED_GPT_V3,
}

# ---------------- raw candidates (with sources) ----------------
def load_oneshot_raw(model_key):
    """pipeline_oneshot.json -> {ns: {'cands':[{text,source}], 'declined':bool, 'blocked':bool}}"""
    d = json.load(open(ONESHOT, encoding="utf-8"))
    out = {}
    for s in d["per_sample"]:
        ns = norm.normalize_text(s["span"])
        rec = (s.get("oneshot") or {}).get(model_key) or {}
        p = rec.get("prediction")
        if not p:
            out[ns] = {"cands": [], "blocked": True}
            continue
        cs = p.get("candidates") or []
        out[ns] = {"cands": [{"text": c.get("text", ""), "source": c.get("source", "")}
                             for c in cs],
                   "blocked": False}
    return out

def load_prediction_raw(path):
    """prediction jsonl -> same shape as load_oneshot_raw (keeps candidate sources)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        t = d["request"]["contents"][0]["parts"][0]["text"]
        m = re.search(r"<<<\s*(.*?)\s*>>>", t, re.S)
        if not m:
            continue
        ns = norm.normalize_text(m.group(1).strip())
        cand = d["response"]["candidates"][0]
        if "content" not in cand or "parts" not in cand.get("content", {}):
            out[ns] = {"cands": [], "blocked": True}
            continue
        try:
            p = json.loads(cand["content"]["parts"][0]["text"])
        except Exception:
            out[ns] = {"cands": [], "blocked": True}
            continue
        cs = p.get("candidates") or []
        out[ns] = {"cands": [{"text": c.get("text", ""), "source": c.get("source", "")}
                             for c in cs],
                   "blocked": False}
    return out

def raw_accessors(raw):
    def units(ns):
        p = raw.get(ns)
        return parse_units([c["text"] for c in p["cands"]]) if p and p["cands"] else []
    def declined(ns):
        p = raw.get(ns)
        return bool(p) and (not p["blocked"]) and len(units(ns)) == 0
    return units, declined

# ---------------- grounding (locked release config) ----------------
def make_grounder():
    import ground_corrections as G
    G.SNAP_METRIC = "jaccard"; G.SNAP_JAC_TH = 0.40; G.SNAP_CHAR_FALLBACK_TH = 0.85
    G.EXPAND_METRIC = "jaccard"; G.EXPAND_JAC_TH = 0.35
    G.EXPAND_MAX = 0; G.EXPAND_SKIP_GE = 15
    G.HADITH_EXPAND = True; G.ALEF_DEL = True; G.PARTIAL = True
    print("loading corpora for grounding (locked release config)...", flush=True)
    quran, mushaf, hadith = G.load_corpus()
    snapc = G.load_snap_corpus()
    WS = re.compile(r"\s+")
    def ground(cands, ref):
        """[{text,source}] -> deduped [(text, [sources])] surviving grounding."""
        rows = []
        for c in cands:
            r = G.resolve_correction(c["text"], ref, c.get("source", ""),
                                     quran, mushaf, hadith, snapc)
            if r["status"] == "eliminated":
                continue
            for vt, vs in (r.get("variants") or [(r["gold_text"], r["sources"])]):
                vt = WS.sub(" ", vt or "").strip()
                if vt:
                    rows.append((vt, vs if isinstance(vs, list) else [vs]))
        seen, out = {}, []
        for t, s in rows:
            k = G.norm_dedup(t)
            if not k:
                continue
            if k in seen:
                for src in s:
                    if src and src not in seen[k][1]:
                        seen[k][1].append(src)
                continue
            d = (t, [x for x in s if x])
            seen[k] = d
            out.append(d)
        return out
    return ground

def snapped_accessors(raw, ground, ref_of):
    """Ground every raw candidate once; build enriched-style strings -> parse_units."""
    cache = {}
    for ns, p in raw.items():
        if ns not in ref_of:
            continue
        if not p["cands"]:
            cache[ns] = {"units": [], "proposed": False, "blocked": p["blocked"]}
            continue
        rows = ground(p["cands"], ref_of[ns])
        parts = []
        for t, srcs in rows:
            parts.append(f"{t}  [{srcs[0]}]" if srcs else t)
        us = parse_units([" ||| ".join(parts)]) if parts else []
        cache[ns] = {"units": us, "proposed": True, "blocked": False}
    def units(ns):
        c = cache.get(ns)
        return c["units"] if c else []
    def declined(ns):
        c = cache.get(ns)
        # raw decline OR everything eliminated -> خطأ (release semantics)
        return bool(c) and (not c["blocked"]) and len(c["units"]) == 0
    return units, declined, cache

# ---------------- scoring (identical math to score_manual_gold.compute) ----------------
def score(units, declined):
    """precision = Table-2 definition (score_precision_proposed.py precision_macro):
    macro-average, over spans where the model PROPOSED >=1 candidate, of the fraction
    of its proposed corrections accepted (matching a Tier-2 gold candidate). Declines
    are excluded entirely — no abstention credit."""
    mg_ok = mg_n = 0
    rec = ge1 = 0
    prec_sum = n_proposed = 0
    n_cand = 0
    for ns in SM.ALL:
        ref = SM.ref_of[ns]; mu = units(ns); dec = declined(ns)
        n_cand += len(mu)
        has_maj = SM.KHATA_MAJ[ns] or bool(SM.MAJ_CORR[ns])
        if has_maj:
            mg_n += 1
            mg_ok += (dec and SM.KHATA_MAJ[ns]) or U.shares(mu, SM.MAJ_CORR[ns], ref)
        g = SM.GOLD_CORR[ns]
        n_gold = len(g) + (1 if SM.KHATA_ANY[ns] else 0)
        m = sum(1 for gc in g if any(U.umatch(mc, gc, ref) for mc in mu))
        if SM.KHATA_ANY[ns] and dec:
            m += 1
        if n_gold:
            rec += m / n_gold
        if m >= 1:
            ge1 += 1
        if mu:
            prec_sum += sum(1 for mc in mu if any(U.umatch(mc, gc, ref) for gc in g)) / len(mu)
            n_proposed += 1
    n = len(SM.ALL)
    return {"maj_gold": round(mg_ok / mg_n, 4), "recall": round(rec / n, 4),
            "precision": round(prec_sum / n_proposed, 4) if n_proposed else None,
            "n_proposed": n_proposed,
            "ge1": round(ge1 / n, 4), "candidates": n_cand}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ground", action="store_true")
    args = ap.parse_args()

    rows = []  # (display name, raw metrics, snapped metrics)

    # ---- orig models: raw from pipeline_oneshot, snapped from the annotated slots ----
    for disp, key, slot in [("GPT-5.4 (orig.)", "gpt-5.4", "first"),
                            ("Gemini-3.1-Pro (orig.)", "gemini-3.1-pro-preview", "second")]:
        raw = load_oneshot_raw(key)
        missing = [ns for ns in SM.ALL if ns not in raw]
        if missing:
            print(f"WARN {disp}: {len(missing)} gold spans not in pipeline_oneshot", flush=True)
        ru, rd = raw_accessors(raw)
        su, sd, _vm, _ = U.model_accessor({"name": disp, "kind": "annotated", "slot": slot})
        rows.append((disp, score(ru, rd), score(su, sd)))

    # ---- revised models: raw = harness accessors, snapped = locked grounding ----
    ground = None if args.no_ground else make_grounder()
    for disp, path in PREDS.items():
        ru, rd, _vm, _ = U.model_accessor({"name": disp, "kind": "prediction", "path": path})
        if ground is None:
            rows.append((disp, score(ru, rd), None))
            continue
        rawp = load_prediction_raw(path)
        su, sd, cache = snapped_accessors(rawp, ground, SM.ref_of)
        elim = sum(1 for ns in SM.ALL
                   if (c := cache.get(ns)) and c["proposed"] and not c["units"])
        print(f"{disp}: spans where ALL candidates eliminated -> خطأ: {elim}", flush=True)
        rows.append((disp, score(ru, rd), score(su, sd)))

    # ---- report ----
    def f3(v): return "—" if v is None else f"{v:.3f}"
    def f2(v): return "—" if v is None else f"{v:.2f}"
    print(f"\nRAW -> SNAPPED+EXPANDED on the {len(SM.ALL)} curated spans (curator gold)\n")
    hdr = f"{'Model':28} {'maj-gold':>16} {'recall':>16} {'precision':>16} {'>=1-cand':>13} {'cands':>11}"
    print(hdr); print("-" * len(hdr))
    for disp, r, s in rows:
        if s is None:
            print(f"{disp:28} {f3(r['maj_gold']):>16} {f3(r['recall']):>16} "
                  f"{f3(r['precision']):>16} {f2(r['ge1']):>13} {r['candidates']:>11}")
        else:
            print(f"{disp:28} {f3(r['maj_gold'])+' -> '+f3(s['maj_gold']):>16} "
                  f"{f3(r['recall'])+' -> '+f3(s['recall']):>16} "
                  f"{f3(r['precision'])+' -> '+f3(s['precision']):>16} "
                  f"{f2(r['ge1'])+' -> '+f2(s['ge1']):>13} "
                  f"{str(r['candidates'])+' -> '+str(s['candidates']):>11}")

    json.dump([{"name": d, "raw": r, "snapped": s} for d, r, s in rows],
              open(os.path.join(HERE, "raw_vs_snapped_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote raw_vs_snapped_results.json")

    # LaTeX rows
    print("\nLaTeX rows:")
    for disp, r, s in rows:
        if s is None:
            continue
        print(f"{disp} & {r['maj_gold']:.3f}\\,$\\to$\\,{s['maj_gold']:.3f} & "
              f"{r['recall']:.3f}\\,$\\to$\\,{s['recall']:.3f} & "
              f"{r['precision']:.3f}\\,$\\to$\\,{s['precision']:.3f} & "
              f"{r['ge1']:.2f}\\,$\\to$\\,{s['ge1']:.2f} & "
              f"{r['candidates']}\\,$\\to$\\,{s['candidates']} \\\\")

if __name__ == "__main__":
    main()
