#!/usr/bin/env python3
"""
Unified, reproducible correction-subtask evaluation.

Adds any number of models to the SAME tables:
  - Annotated originals (GPT, Gemini) that live inside the annotator JSON, with human verdicts.
  - Prediction files (.jsonl) from any new prompt/model, joined by span, without verdicts.

To add a model: append an entry to MODELS below (or pass a JSON config via --config).
Then: python3 unified_eval.py   ->  writes unified_results.json + unified_report.md

Metric availability:
  * Universal (all models): decision_acc, approx_correct, ge1_candidate, coverage recall/precision,
    false_correction_rate, unanimous decision/solution match.
  * Annotated-only (verdicts exist): verdict_maj (true human majority-correct).
"""
import json, re, sys, os, argparse
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
import norm
from parse2 import parse_units
SENT=norm.SENTINEL

# local ports of the web-sandbox paths
DATA=os.path.join(HERE,'..','annotators_corrected_enriched.json')   # == _judged gold (snapped+expanded)
KEEP=os.path.join(HERE,'keep100.json')
PRED_PRO=os.path.join(HERE,'..','batch_v3','download','gemini-3.1-pro-preview','predictions_pro.jsonl')
PRED_FLASH=os.path.join(HERE,'..','batch_v3','download','gemini-3.5-flash','predictions_flash.jsonl')
PRED_GPT_V3=os.path.join(HERE,'predictions_gpt_v3.jsonl')

# ------------- model registry -------------
# kind: 'annotated' (inside the annotator JSON; slot=first/second) or 'prediction' (.jsonl file)
MODELS=[
  {'name':'GPT (orig)','kind':'annotated','slot':'first'},
  {'name':'Gemini (orig)','kind':'annotated','slot':'second'},
  {'name':'Pro (new)','kind':'prediction','path':PRED_PRO},
  {'name':'Flash (new)','kind':'prediction','path':PRED_FLASH},
]
if os.path.exists(PRED_GPT_V3):
  MODELS.append({'name':'GPT-v3 (new)','kind':'prediction','path':PRED_GPT_V3})

# ------------- matching (locked rules) -------------
BOIL=set('صلي الله عليه وسلم رسول عن قال ان النبي رضي عنه انه يا ايها الذين امنوا الراوي المحدث المصدر التخريج'.split())
def cj(a,b):
    sa=set(t for t in a.split() if t not in BOIL); sb=set(t for t in b.split() if t not in BOIL)
    return len(sa&sb)/len(sa|sb) if (sa|sb) else 0.0
def umatch(u,v,ref):
    if u['text'] and u['text']==v['text']: return True
    if cj(u['text'],v['text'])>=0.6: return True
    ru,rv=u['ref'],v['ref']
    if ref=='quran' and ru and rv and ru[0]=='quran'==rv[0] and ru[1]==rv[1]:
        try:
            if ru[2] and rv[2] and abs(int(ru[2])-int(rv[2]))<=1: return True
        except: pass
    return False
def dedup(units,ref):
    out=[]
    for u in units:
        if not any(umatch(u,o,ref) for o in out): out.append(u)
    return out
def shares(A,B,ref): return any(umatch(a,b,ref) for a in A for b in B)

# ------------- load annotator data, build gold + originals + unanimous sets -------------
d=json.load(open(DATA,encoding='utf-8')); A=d['annotators']; ANN=[a['annotator'] for a in A]
keep=set(tuple(x) for x in json.load(open(KEEP,encoding='utf-8')))

ref_of={}; gold_raw=defaultdict(list); GOLD_KHATA=set()
orig_units=defaultdict(dict)            # ns -> slot(first/second) -> units
V=defaultdict(lambda: defaultdict(dict))# ns -> slot -> ann -> verdict  (for annotated verdict_maj)
K=defaultdict(dict)                      # ns -> slot -> kind
MANU=defaultdict(lambda: defaultdict(list))  # ns -> ann -> units (manual corrections)
for a in A:
    nm=a['annotator']
    for r in a['annotations']:
        it=(r['id'],r['model'],r['span'])
        if it not in keep or r['judgment'] is None: continue
        ns=norm.normalize_text(r['span']); ref_of[ns]=r['ref']
        MANU[ns][nm]=parse_units(r['manual_corrections'])
        for slot,txt in [('first',r['gpt_correction']),('second',r['gemini_correction'])]:
            V[ns][slot][nm]=r['judgment']
            K[ns][slot]='sentinel' if SENT in txt else 'proposed'
            if slot not in orig_units[ns]:
                orig_units[ns][slot]=[] if SENT in txt else parse_units([txt])
            # gold: endorsed model correction
            if r['judgment']=='correct' and SENT not in txt:
                gold_raw[ns].extend(parse_units([txt]))
            # gold: endorsed خطأ (model declined and annotator judged that correct)
            if r['judgment']=='correct' and SENT in txt:
                GOLD_KHATA.add(ns)
        # gold: manual corrections when wrong
        if r['judgment']=='wrong' and r['manual_corrections']:
            gold_raw[ns].extend(parse_units(r['manual_corrections']))
# TIER 2 — gold candidate set: any annotator-sanctioned answer (any correction span) + خطأ flag
GOLD={ns:dedup(us,ref_of[ns]) for ns,us in gold_raw.items()}
ALL=sorted(ref_of.keys())

def _ann_correctable(ns,ann):
    fl=[]
    for slot in ['first','second']:
        vd=V[ns].get(slot,{})
        if ann not in vd: continue
        fl.append(False if (K[ns][slot]=='sentinel' and vd[ann]=='correct') else True)
    return None if not fl else (sum(fl)>len(fl)/2)

# per span: how many annotators declined (said no correction), and each annotator's correction units
def _n_declined(ns):
    return sum(1 for a in ANN if _ann_correctable(ns,a) is False)

# TIER 2 also carries a خطأ option: it is an acceptable answer if ANY annotator declined.
def gold_allows_khata(ns): return _n_declined(ns)>=1

# TIER 1 — majority gold. خطأ if >=2 declined; else the set of correction spans >=2 annotators share.
def majority_gold(ns):
    if _n_declined(ns)>=2:
        return {'khata':True,'spans':[]}
    # cluster annotator correction units; keep those backed by >=2 annotators
    cl=[]
    for a in ANN:
        for u in MANU[ns].get(a,[]):
            hit=False
            for c in cl:
                if umatch(u,c['rep'],ref_of[ns]): c['anns'].add(a); hit=True; break
            if not hit: cl.append({'rep':u,'anns':{a}})
    spans=[c['rep'] for c in cl if len(c['anns'])>=2]
    return {'khata':False,'spans':spans}

def gold_has_khata(ns): return ns in GOLD_KHATA
def has_gold(ns): return (ns in GOLD and len(GOLD[ns])>0) or gold_has_khata(ns)

def model_matches_candidate_set(ns, units_fn, declined_fn):
    """Tier-2: does the model's answer match ANY gold candidate (a correction span, or خطأ)?"""
    g=GOLD.get(ns,[])
    if g and shares(units_fn(ns),g,ref_of[ns]): return True
    if gold_has_khata(ns) and declined_fn(ns): return True
    return False


# annotator-majority correctable (fix exists) per span
def ann_correctable(ns,ann):
    flags=[]
    for slot in ['first','second']:
        vd=V[ns].get(slot,{})
        if ann not in vd: continue
        flags.append(False if (K[ns][slot]=='sentinel' and vd[ann]=='correct') else True)
    if not flags: return None
    return sum(flags)>len(flags)/2
def majority_correctable(ns):
    fl=[ann_correctable(ns,a) for a in ANN]; fl=[x for x in fl if x is not None]
    return None if not fl else sum(fl)>len(fl)/2

# unanimous subsets (from original annotated data)
def all3_correctable(ns):
    fl=[ann_correctable(ns,a) for a in ANN]
    if any(x is None for x in fl): return None
    return len(set(fl))==1
def maj_solution(ns):
    cl=[]
    for a in ANN:
        for u in MANU[ns].get(a,[]):
            h=False
            for c in cl:
                if umatch(u,c['rep'],ref_of[ns]): c['anns'].add(a); h=True; break
            if not h: cl.append({'rep':u,'anns':{a}})
    return [c['rep'] for c in cl if len(c['anns'])>=2]
def all3_full(ns):
    pres=[MANU[ns].get(a,[]) for a in ANN if any(a in V[ns].get(s,{}) for s in ['first','second'])]
    if len(pres)<3: return None
    def sf(U,W):
        if not U and not W: return True
        if not U or not W: return False
        return all(any(umatch(u,w,ref_of[ns]) for w in W) for u in U) and all(any(umatch(w,u,ref_of[ns]) for u in U) for w in W)
    return sf(pres[0],pres[1]) and sf(pres[1],pres[2]) and sf(pres[0],pres[2])

# ------------- prediction file loader -------------
def load_prediction(path):
    out={}
    for line in open(path,encoding='utf-8'):
        if not line.strip(): continue
        d=json.loads(line)
        t=d['request']['contents'][0]['parts'][0]['text']
        m=re.search(r'<<<\s*(.*?)\s*>>>',t,re.S)
        if not m: continue
        ns=norm.normalize_text(m.group(1).strip())
        cand=d['response']['candidates'][0]
        if 'content' not in cand or 'parts' not in cand.get('content',{}):
            out[ns]={'units':[],'blocked':True}; continue
        try: p=json.loads(cand['content']['parts'][0]['text'])
        except: out[ns]={'units':[],'blocked':True}; continue
        cs=p.get('candidates',[]) or []
        out[ns]={'units':parse_units([c['text'] for c in cs]) if cs else [],'blocked':False}
    return out

# ------------- resolve each model to a units-by-span function + optional verdicts -------------
def model_accessor(spec):
    if spec['kind']=='annotated':
        slot=spec['slot']
        def units(ns): return orig_units[ns].get(slot,[])
        def declined(ns): return len(orig_units[ns].get(slot,[]))==0
        def verdict_maj(ns):
            vd=V[ns].get(slot,{})
            if not vd: return None
            return sum(1 for v in vd.values() if v=='correct')>len(vd)/2
        return units,declined,verdict_maj,None
    else:
        preds=load_prediction(spec['path'])
        def units(ns):
            p=preds.get(ns); return p['units'] if p else []
        def declined(ns):
            p=preds.get(ns); return bool(p) and (not p['blocked']) and len(p['units'])==0
        return units,declined,(lambda ns:None),preds

# ------------- metric computation -------------
def compute(spec):
    units,declined,verdict_maj,preds=model_accessor(spec)
    out={'name':spec['name'],'annotated':spec['kind']=='annotated'}
    # precompute majority gold per span
    MG={ns:majority_gold(ns) for ns in ALL}
    for scope in [None,'quran','hadith']:
        spans=[ns for ns in ALL if scope is None or ref_of[ns]==scope]
        # TIER 1 partition by MAJORITY gold
        khata_spans=[ns for ns in spans if MG[ns]['khata']]
        corr_spans =[ns for ns in spans if not MG[ns]['khata'] and MG[ns]['spans']]
        # spans where majority is neither khata nor a >=2-agreed correction (no majority answer)
        nomaj_spans=[ns for ns in spans if not MG[ns]['khata'] and not MG[ns]['spans']]

        # --- majority-gold decision accuracy ---
        # correct if: khata-span & model declined; OR corr-span & model matches a majority span.
        mg_ok=mg_n=0
        for ns in khata_spans:
            mg_n+=1; mg_ok+=declined(ns)
        for ns in corr_spans:
            mg_n+=1; mg_ok+=shares(units(ns),MG[ns]['spans'],ref_of[ns])
        # (nomaj spans excluded from majority-gold accuracy: no majority answer to score against)

        # --- majority-gold DECISION only (correctable vs khata), ignores which span ---
        dec_ok=dec_n=0
        for ns in spans:
            if MG[ns]['khata']:
                dec_n+=1; dec_ok+=declined(ns)
            elif MG[ns]['spans']:
                dec_n+=1; dec_ok+=(not declined(ns))
        # false correction = proposing on a khata-majority span
        fc=sum(1 for ns in khata_spans if not declined(ns))

        # --- TIER 2 coverage: recall/precision over spans that have >=1 gold candidate (incl خطأ) ---
        t2_spans=[ns for ns in spans if has_gold(ns)]
        rec=prec=0; np_=0; cov_all=cov_part=cov_none=0; ge1=0
        for ns in t2_spans:
            g=list(GOLD.get(ns,[])); mu=units(ns); ref=ref_of[ns]
            khata_ok=gold_has_khata(ns)
            n_gold=len(g)+(1 if khata_ok else 0)          # خطأ counts as one gold candidate
            m=sum(1 for gc in g if any(umatch(mc,gc,ref) for mc in mu))  # correction matches
            if khata_ok and declined(ns): m+=1            # declining matches the خطأ candidate
            rec+=m/n_gold if n_gold else 0
            if m>=1: ge1+=1
            if m==n_gold: cov_all+=1
            elif m>=1: cov_part+=1
            else: cov_none+=1
            # precision: model's own outputs that hit a gold candidate.
            if mu:  # model proposed something
                mp=sum(1 for mc in mu if any(umatch(mc,gc,ref) for gc in g)); prec+=mp/len(mu); np_+=1
            elif declined(ns) and khata_ok:  # model declined and خطأ was correct -> precision 1
                prec+=1; np_+=1

        # --- approx-correct (verdict-maj proxy), TIER 1 based ---
        # correct if model's answer matches the majority gold (khata->declined ; corr->matches a majority span)
        appr = mg_ok  # same numerator
        n=len(spans)
        out[scope or 'all']={
            'n':n,'n_khata_maj':len(khata_spans),'n_corr_maj':len(corr_spans),'n_nomajority':len(nomaj_spans),
            'majority_gold_acc':round(mg_ok/mg_n,4) if mg_n else None,   # matches majority answer
            'decision_acc':round(dec_ok/dec_n,4) if dec_n else None,      # correctable-vs-khata decision
            'approx_correct':round(appr/mg_n,4) if mg_n else None,
            'verdict_maj':(round(sum(1 for ns in spans if verdict_maj(ns))/n,4) if spec['kind']=='annotated' else None),
            'ge1_candidate':round(ge1/len(t2_spans),4) if t2_spans else None,
            'recall':round(rec/len(t2_spans),4) if t2_spans else None,
            'precision':round(prec/np_,4) if np_ else None,
            'cov_all':cov_all,'cov_partial':cov_part,'cov_none':cov_none,
            'false_corrections':fc,'false_correction_rate':round(fc/len(khata_spans),4) if khata_spans else None,
        }
    # unanimous subsets
    dec_n=dec_ok=sol_n=sol_ok=0
    for ns in ALL:
        if all3_correctable(ns)==True:
            mc=majority_correctable(ns)
            dec_n+=1; dec_ok+=((not declined(ns))==mc)
        if all3_full(ns)==True:
            # agreed solution = full gold pool for this span (endorsed corrections + manuals),
            # matched uniformly for every model. Endorsement credit lives in GOLD already.
            g=GOLD.get(ns,[])
            sol_n+=1
            if not g: sol_ok+=declined(ns)          # unanimous 'no fix' -> correct to decline
            else: sol_ok+=shares(units(ns),g,ref_of[ns])
    out['unanimous']={'decision_n':dec_n,'decision_acc':round(dec_ok/dec_n,4) if dec_n else None,
                      'solution_n':sol_n,'solution_acc':round(sol_ok/sol_n,4) if sol_n else None}
    if preds is not None:
        out['blocked']=sum(1 for v in preds.values() if v.get('blocked'))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',help='JSON file with a "models" list to override the default registry')
    args=ap.parse_args()
    models=MODELS
    if args.config:
        models=json.load(open(args.config))['models']
    results=[compute(m) for m in models]
    json.dump(results,open(os.path.join(HERE,'unified_results_local.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    write_report(results)
    print('wrote unified_results_local.json and unified_report_local.md')

def _fmt(v): return '—' if v is None else (f'{v:.3f}' if isinstance(v,float) else str(v))
def write_report(results):
    L=[]
    L.append('# Unified Correction-Subtask Evaluation\n')
    L.append('All models scored on the same 100 spans (50 Quran + 50 Hadith). '
             'Two-tier gold: **majority gold** is the answer 2+ annotators agree on — either **خطأ** '
             '(2+ said no correction) or the set of correction spans 2+ annotators share; the '
             '**gold candidate set** (Tier 2) is every annotator-sanctioned correction span, used for '
             'coverage/recall. `verdict_maj` is the true human majority-correct rate (annotated models '
             'only); `majority-gold` = matches the majority answer (correct decline on a خطأ span, or '
             'a majority correction span); `decision-acc` = correctable-vs-خطأ decision only; '
             '`approx-correct` mirrors majority-gold and is comparable across ALL models.\n')
    for scope in ['all','quran','hadith']:
        L.append(f'\n## {scope.capitalize()} \n')
        L.append('| Model | verdict-maj | majority-gold | approx-correct | decision-acc | ≥1-cand | recall | precision | false-corr |')
        L.append('|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|')
        for r in results:
            x=r[scope]
            L.append(f"| {r['name']} | {_fmt(x['verdict_maj'])} | {_fmt(x['majority_gold_acc'])} | "
                     f"{_fmt(x['approx_correct'])} | {_fmt(x['decision_acc'])} | {_fmt(x['ge1_candidate'])} | "
                     f"{_fmt(x['recall'])} | {_fmt(x['precision'])} | {x['false_corrections']}/{x['n_khata_maj']} |")
    L.append('\n## Coverage tiers (gold-bearing spans, All scope)\n')
    L.append('| Model | covered all | partial | none |')
    L.append('|-------|:-:|:-:|:-:|')
    for r in results:
        x=r['all']; L.append(f"| {r['name']} | {x['cov_all']} | {x['cov_partial']} | {x['cov_none']} |")
    L.append('\n## Model accuracy on unanimous annotator decisions\n')
    L.append('| Model | all-3 correctable → decision matches (n) | all-3 full solution → matches (n) |')
    L.append('|-------|:-:|:-:|')
    for r in results:
        u=r['unanimous']
        L.append(f"| {r['name']} | {_fmt(u['decision_acc'])} ({u['decision_n']}) | {_fmt(u['solution_acc'])} ({u['solution_n']}) |")
    L.append('\n*verdict-maj is blank for prediction-file models (no human verdicts on their outputs); '
             'approx-correct is the comparable substitute. Reproduce: `python3 unified_eval.py` '
             '(optionally `--config models.json`).*\n')
    open(os.path.join(HERE,'unified_report_local.md'),'w',encoding='utf-8').write('\n'.join(L))

if __name__=='__main__': main()
