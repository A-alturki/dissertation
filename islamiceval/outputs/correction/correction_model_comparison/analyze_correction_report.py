#!/usr/bin/env python3
"""
analyze_correction_report.py — recompute ALL correction-subtask statistics fresh, against the
CURRENT spans (keep100), the CLEAN curator gold (gold_100_final.json), and the LOCKED grounding
config (ground_corrections module defaults). Fixes the earlier annotator-verdict-overwrite bug by
reading per-corrector verdicts directly from the annotator export.

Outputs `report_stats.json` (+ prints). Parts:
  A. Annotator agreement (correctable / full-solution / >=1-shared): all-3-agree, mean-pairwise, Fleiss.
  B. Unanimous-annotator subset (n) + each model's accuracy on it.
  C. Cohen's kappa (annotator-vs-gold verdict, annotator-annotator correctable, model-vs-gold correctable).
  D. Model scores vs the manual gold, BEFORE vs AFTER snapping+expansion, for ALL 5 models, uniformly.
"""
import json, os, sys, itertools
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import norm
from parse2 import parse_units
import unified_eval_local as U       # reuse umatch/cj/shares/dedup (pure) — NOT its buggy gold
import ground_corrections as G       # module defaults = the LOCKED grounding config
alef = G.load_corpus(); snapc = G.load_snap_corpus()
SENT = norm.SENTINEL
umatch, shares, dedup = U.umatch, U.shares, U.dedup

# ───────────────────────── load spans + gold ─────────────────────────
keep = set(tuple(x) for x in json.load(open(os.path.join(HERE, "keep100.json"), encoding="utf-8")))
gold = json.load(open(os.path.join(ROOT, "outputs", "classification", "gold_100_final.json"), encoding="utf-8"))["spans"]
def gunit(c):
    return {"text": norm.normalize_text(c["text"]),
            "ref": norm.extract_ref((c.get("sources") or [""])[0] or c["text"])}
ref_of, GOLD_CORR, MAJ_CORR, KHATA_ANY, KHATA_MAJ = {}, {}, {}, {}, {}
for s in gold:
    ns = norm.normalize_text(s["span"]); ref = s["ref"]; ref_of[ns] = ref
    GOLD_CORR[ns] = dedup([gunit(c) for c in s["gold"] if not c["is_khata"]], ref)
    MAJ_CORR[ns]  = dedup([gunit(c) for c in s["gold"] if not c["is_khata"] and c["count"] >= 2], ref)
    KHATA_ANY[ns] = any(c["is_khata"] for c in s["gold"])
    KHATA_MAJ[ns] = any(c["is_khata"] and c["count"] >= 2 for c in s["gold"])
ALL = sorted(ref_of); QU = [n for n in ALL if ref_of[n]=="quran"]; HA = [n for n in ALL if ref_of[n]=="hadith"]
SCOPES = {"all": ALL, "quran": QU, "hadith": HA}

# ───────────────────────── annotator data (per-corrector, CORRECT) ─────────────────────────
A = json.load(open(os.path.join(HERE, "..", "annotators_corrected_enriched.json"), encoding="utf-8"))["annotators"]
ANN = [a["annotator"] for a in A]
# per (ns, ann): {correctable: bool, khata: bool, units: [candidate units]}
adata = defaultdict(dict)
for a in A:
    nm = a["annotator"]
    rows = defaultdict(dict)
    for r in a["annotations"]:
        if (r["id"], r["model"], r["span"]) not in keep or r["judgment"] is None: continue
        rows[norm.normalize_text(r["span"])][r["corrector"]] = r
    for ns, rr in rows.items():
        fr, sr = rr.get("first"), rr.get("second")
        endorsed_prop = ((fr and fr["judgment"]=="correct" and SENT not in fr["gpt_correction"]) or
                         (sr and sr["judgment"]=="correct" and SENT not in sr["gemini_correction"]))
        endorsed_decl = ((fr and fr["judgment"]=="correct" and SENT in fr["gpt_correction"]) or
                         (sr and sr["judgment"]=="correct" and SENT in sr["gemini_correction"]))
        manual = []
        for x in (fr, sr):
            if x: manual += (x.get("manual_corrections") or [])
        munits = parse_units(manual)
        # candidate set for this annotator = endorsed model texts + own manuals
        cands = []
        if fr and fr["judgment"]=="correct" and SENT not in fr["gpt_correction"]: cands += parse_units([fr["gpt_correction"]])
        if sr and sr["judgment"]=="correct" and SENT not in sr["gemini_correction"]: cands += parse_units([sr["gemini_correction"]])
        cands += munits
        # correctable = has a correction (endorsed proposal or manual); not-correctable = only endorsed decline
        correctable = bool(endorsed_prop or munits) or not endorsed_decl
        adata[ns][nm] = {"correctable": correctable, "khata": endorsed_decl and not (endorsed_prop or munits),
                         "units": dedup(cands, ref_of[ns])}

def fleiss_binary(flag_fn, spans):
    """Fleiss kappa for a binary category over 3 raters."""
    N = 0; Pbar = 0.0; p1 = 0
    for ns in spans:
        vals = [flag_fn(ns, nm) for nm in ANN if nm in adata[ns]]
        if len(vals) < 2: continue
        n = len(vals); k1 = sum(1 for v in vals if v); k0 = n - k1
        Pi = (k1*(k1-1) + k0*(k0-1)) / (n*(n-1))
        Pbar += Pi; p1 += k1 / n; N += 1
    if not N: return None
    Pbar /= N; p1 /= N
    Pe = p1**2 + (1-p1)**2
    return (Pbar - Pe)/(1 - Pe) if (1-Pe) else None

def all3_agree(flag_fn, spans):
    ok = tot = 0
    for ns in spans:
        vals = [flag_fn(ns, nm) for nm in ANN if nm in adata[ns]]
        if len(vals) < 3: continue
        tot += 1; ok += len(set(vals)) == 1
    return ok, tot

def mean_pairwise(flag_fn, spans):
    agr = []
    for a, b in itertools.combinations(ANN, 2):
        m = n = 0
        for ns in spans:
            if a in adata[ns] and b in adata[ns]:
                n += 1; m += flag_fn(ns, a) == flag_fn(ns, b)
        if n: agr.append(m/n)
    return sum(agr)/len(agr) if agr else None

def sets_equal(ns, a, b):
    Ua, Ub = adata[ns][a]["units"], adata[ns][b]["units"]
    ref = ref_of[ns]
    if not Ua and not Ub: return True
    if not Ua or not Ub: return False
    return (all(any(umatch(u,w,ref) for w in Ub) for u in Ua) and
            all(any(umatch(w,u,ref) for u in Ua) for w in Ub))
def sets_share(ns, a, b):
    Ua, Ub = adata[ns][a]["units"], adata[ns][b]["units"]
    if not Ua and not Ub: return True     # both declined = they agree there is no correction
    return shares(Ua, Ub, ref_of[ns])

corr = lambda ns, nm: adata[ns][nm]["correctable"]
out = {"n": {k: len(v) for k,v in SCOPES.items()}}

# ── PART A: agreement ──
partA = {}
for sc, spans in SCOPES.items():
    fs_ok, fs_tot = all3_agree(lambda ns,nm: None, spans)  # placeholder; full-solution uses pairwise below
    # full-solution all-3-agree = all pairs equal
    fs3 = 0; fs3t = 0
    for ns in spans:
        av = [nm for nm in ANN if nm in adata[ns]]
        if len(av) < 3: continue
        fs3t += 1
        fs3 += all(sets_equal(ns,a,b) for a,b in itertools.combinations(av,2))
    sh3 = 0; sh3t = 0
    for ns in spans:
        av = [nm for nm in ANN if nm in adata[ns]]
        if len(av) < 3: continue
        sh3t += 1
        sh3 += all(sets_share(ns,a,b) for a,b in itertools.combinations(av,2))
    c_ok, c_tot = all3_agree(corr, spans)
    partA[sc] = {
        "correctable_all3": round(c_ok/c_tot,3), "correctable_all3_n": f"{c_ok}/{c_tot}",
        "correctable_pairwise": round(mean_pairwise(corr, spans),3),
        "correctable_fleiss": round(fleiss_binary(corr, spans),3),
        "fullsolution_all3": round(fs3/fs3t,3), "fullsolution_all3_n": f"{fs3}/{fs3t}",
        "fullsolution_pairwise": round(sum(sets_equal(ns,a,b) for ns in spans for a,b in itertools.combinations([nm for nm in ANN if nm in adata[ns]],2)) /
                                       max(1,sum(1 for ns in spans for _ in itertools.combinations([nm for nm in ANN if nm in adata[ns]],2))),3),
        "share1_all3": round(sh3/sh3t,3), "share1_all3_n": f"{sh3}/{sh3t}",
        "share1_pairwise": round(sum(sets_share(ns,a,b) for ns in spans for a,b in itertools.combinations([nm for nm in ANN if nm in adata[ns]],2)) /
                                 max(1,sum(1 for ns in spans for _ in itertools.combinations([nm for nm in ANN if nm in adata[ns]],2))),3),
    }
out["A_agreement"] = partA

# ── majority gold correctable (for kappa vs gold) ──
def gold_correctable(ns):  # majority of annotators say correctable
    v = [adata[ns][nm]["correctable"] for nm in ANN if nm in adata[ns]]
    return sum(v) > len(v)/2 if v else None

# ── PART C: Cohen kappa ──
def cohen(pairs):
    # pairs: list of (a,b) binary
    n = len(pairs)
    if not n: return None
    po = sum(1 for a,b in pairs if a==b)/n
    pa1 = sum(1 for a,_ in pairs if a)/n; pb1 = sum(1 for _,b in pairs if b)/n
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    return round((po-pe)/(1-pe),3) if (1-pe) else None
partC = {"annotator_vs_gold_correctable": {}, "annotator_annotator_correctable": {}}
for nm in ANN:
    pr = [(adata[ns][nm]["correctable"], gold_correctable(ns)) for ns in ALL if nm in adata[ns] and gold_correctable(ns) is not None]
    partC["annotator_vs_gold_correctable"][nm] = cohen(pr)
for a,b in itertools.combinations(ANN,2):
    pr = [(adata[ns][a]["correctable"], adata[ns][b]["correctable"]) for ns in ALL if a in adata[ns] and b in adata[ns]]
    partC["annotator_annotator_correctable"][f"{a[:8]}|{b[:8]}"] = cohen(pr)
out["C_kappa"] = partC

# ───────────────────────── model raw candidates ─────────────────────────
import re
def load_pred_jsonl(path):
    res = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        d = json.loads(line)
        t = d["request"]["contents"][0]["parts"][0]["text"]
        m = re.search(r"<<<\s*(.*?)\s*>>>", t, re.S)
        if not m: continue
        ns = norm.normalize_text(m.group(1).strip())
        c = d["response"]["candidates"][0]
        if "content" not in c or "parts" not in c.get("content", {}): res[ns] = {"verdict":None,"cands":[]}; continue
        try: p = json.loads(c["content"]["parts"][0]["text"])
        except: res[ns] = {"verdict":None,"cands":[]}; continue
        res[ns] = {"verdict": p.get("verdict"), "cands": [x.get("text","") for x in (p.get("candidates") or [])]}
    return res
def load_pipeline(path, model):
    d = json.load(open(path, encoding="utf-8")); res = {}
    for s in d["per_sample"]:
        ns = norm.normalize_text(s["span"])
        pr = (s.get("oneshot",{}).get(model,{}) or {}).get("prediction") or {}
        res[ns] = {"verdict": pr.get("verdict"), "cands": [x.get("text","") for x in (pr.get("candidates") or [])]}
    return res
base = os.path.join(ROOT, "outputs", "correction")
MODELS = {
    "GPT (orig)":   load_pipeline(os.path.join(base,"pipeline_oneshot.json"), "gpt-5.4"),
    "Gemini (orig)":load_pipeline(os.path.join(base,"pipeline_oneshot.json"), "gemini-3.1-pro-preview"),
    "Pro (new)":    load_pred_jsonl(os.path.join(base,"batch_v3","download","gemini-3.1-pro-preview","predictions_pro.jsonl")),
    "Flash (new)":  load_pred_jsonl(os.path.join(base,"batch_v3","download","gemini-3.5-flash","predictions_flash.jsonl")),
    "GPT-v3 (new)": load_pred_jsonl(os.path.join(HERE,"predictions_gpt_v3.jsonl")),
}

def raw_units(pred, ns):
    return dedup(parse_units(pred.get("cands") or []), ref_of[ns])
def snapped_units(pred, ns):
    texts = []
    for t in (pred.get("cands") or []):
        r = G.resolve_correction(t, ref_of[ns], "", *alef, snapc)
        if r["status"] == "eliminated": continue
        for vt, vs in (r.get("variants") or [(r["gold_text"], r["sources"])]):
            if vt: texts.append(vt)
    return dedup([{"text": norm.normalize_text(t), "ref": norm.extract_ref(t)} for t in texts], ref_of[ns])
def declined(pred):
    return (pred.get("verdict") == "خطأ") or (pred.get("verdict")=="ok" and not (pred.get("cands") or []))

def score(units_fn, model_preds):
    res = {}
    for sc, spans in SCOPES.items():
        mg_ok=mg_n=dec_ok=dec_n=fc=fc_n=0; rec=prec=np_=ge1=cov_all=cov_none=0; ncand=0
        for ns in spans:
            p = model_preds.get(ns, {"verdict":None,"cands":[]})
            mu = units_fn(p, ns); dec = declined(p); ncand += len(mu)
            has_maj = KHATA_MAJ[ns] or bool(MAJ_CORR[ns])
            if has_maj:
                mg_n += 1; mg_ok += (dec and KHATA_MAJ[ns]) or shares(mu, MAJ_CORR[ns], ref_of[ns])
                dec_n += 1; dec_ok += (dec and KHATA_MAJ[ns]) or ((not dec) and bool(MAJ_CORR[ns]))
            if KHATA_MAJ[ns] and not MAJ_CORR[ns]:
                fc_n += 1; fc += (not dec)
            g = GOLD_CORR[ns]; n_gold = len(g) + (1 if KHATA_ANY[ns] else 0)
            m = sum(1 for gc in g if any(umatch(mc,gc,ref_of[ns]) for mc in mu))
            if KHATA_ANY[ns] and dec: m += 1
            if n_gold: rec += m/n_gold
            ge1 += m>=1; cov_all += (m==n_gold); cov_none += (m==0)
            if mu: prec += sum(1 for mc in mu if any(umatch(mc,gc,ref_of[ns]) for gc in g))/len(mu); np_ += 1
            elif dec and KHATA_ANY[ns]: prec += 1; np_ += 1
        n = len(spans)
        res[sc] = {"majority_gold": round(mg_ok/mg_n,3) if mg_n else None,
                   "decision": round(dec_ok/dec_n,3) if dec_n else None,
                   "ge1_candidate": round(ge1/n,3), "recall": round(rec/n,3),
                   "precision": round(prec/np_,3) if np_ else None,
                   "false_corr": f"{fc}/{fc_n}", "cov_all": cov_all, "cov_none": cov_none, "n_candidates": ncand}
    return res

partD = {}
for name, preds in MODELS.items():
    partD[name] = {"raw": score(raw_units, preds), "snapped_expanded": score(snapped_units, preds)}
out["D_models_before_after"] = partD

# ── PART B: unanimous subset + model accuracy on it ──
def all3_correctable_val(ns):
    v = [adata[ns][nm]["correctable"] for nm in ANN if nm in adata[ns]]
    return v[0] if (len(v)==3 and len(set(v))==1) else None
def all3_solution(ns):
    av = [nm for nm in ANN if nm in adata[ns]]
    if len(av)<3: return None
    if not all(sets_equal(ns,a,b) for a,b in itertools.combinations(av,2)): return None
    return dedup(adata[ns][av[0]]["units"], ref_of[ns])
partB = {"n_unanimous_correctable": {}, "n_unanimous_solution": {}, "model_on_unanimous": {}}
for sc, spans in SCOPES.items():
    uc = [ns for ns in spans if all3_correctable_val(ns) is not None]
    us = [ns for ns in spans if all3_solution(ns) is not None]
    partB["n_unanimous_correctable"][sc] = len(uc); partB["n_unanimous_solution"][sc] = len(us)
for name, preds in MODELS.items():
    row = {}
    for sc, spans in SCOPES.items():
        uc = [ns for ns in spans if all3_correctable_val(ns) is not None]
        us = [ns for ns in spans if all3_solution(ns) is not None]
        # decision matches unanimous correctable (use snapped units for decision? decision is verdict-based)
        d_ok = sum(1 for ns in uc if (not declined(preds.get(ns,{"verdict":None,"cands":[]}))) == all3_correctable_val(ns))
        s_ok = 0
        for ns in us:
            g = all3_solution(ns); mu = snapped_units(preds.get(ns,{"verdict":None,"cands":[]}), ns)
            if not g: s_ok += declined(preds.get(ns,{"verdict":None,"cands":[]}))
            else: s_ok += shares(mu, g, ref_of[ns])
        row[sc] = {"decision_matches": f"{d_ok}/{len(uc)}", "decision_acc": round(d_ok/len(uc),3) if uc else None,
                   "solution_matches": f"{s_ok}/{len(us)}", "solution_acc": round(s_ok/len(us),3) if us else None}
    partB["model_on_unanimous"][name] = row
out["B_unanimous"] = partB

json.dump(out, open(os.path.join(HERE,"report_stats.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote report_stats.json")
print("\n=== A. Annotator agreement (all-3 / pairwise / Fleiss) ===")
for sc in ["all","quran","hadith"]:
    a=partA[sc]; print(f"  {sc:6} correctable: all3 {a['correctable_all3']} ({a['correctable_all3_n']}) pairwise {a['correctable_pairwise']} Fleiss {a['correctable_fleiss']} | "
                        f"fullsol all3 {a['fullsolution_all3']} pairwise {a['fullsolution_pairwise']} | share1 pairwise {a['share1_pairwise']}")
print("\n=== C. Cohen kappa ===")
print("  annotator vs gold (correctable):", partC["annotator_vs_gold_correctable"])
print("  annotator-annotator (correctable):", partC["annotator_annotator_correctable"])
print("\n=== B. Unanimous subset n ===", partB["n_unanimous_correctable"], partB["n_unanimous_solution"])
for nm in MODELS: print(f"  {nm:14}", {sc: partB['model_on_unanimous'][nm][sc]['decision_acc'] for sc in ['all']}, {sc: partB['model_on_unanimous'][nm][sc]['solution_acc'] for sc in ['all']})
print("\n=== D. Models BEFORE vs AFTER snap+expand (ALL scope) ===")
for nm, v in partD.items():
    r,s = v["raw"]["all"], v["snapped_expanded"]["all"]
    print(f"  {nm:14} maj-gold {r['majority_gold']}->{s['majority_gold']} | recall {r['recall']}->{s['recall']} | prec {r['precision']}->{s['precision']} | ge1 {r['ge1_candidate']}->{s['ge1_candidate']} | cand {r['n_candidates']}->{s['n_candidates']}")
