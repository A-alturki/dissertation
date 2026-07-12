#!/usr/bin/env python3
"""Analyze human annotations of model corrections (IslamicEval correction subtask).

Reads the blind 200-span item file (`correction_annotation.json`) and every annotator
export (JSON or CSV) in outputs/correction/, then reports:

  * coverage per annotator (judged / 400 slots),
  * per-corrector human-judged accuracy  (first = GPT-5.4, second = Gemini-3.1-pro),
    split by whether the model proposed a correction ("ok") or declined ("خطأ"),
  * breakdown by ref (quran / hadith) and by error_type,
  * inter-annotator raw agreement + Cohen's kappa on the slots both judged,
  * inappropriate-question flags (union across annotators).

A "slot" = (id, model, span, corrector). Judgments are binary {correct, wrong}; blank
/ None = unjudged.

Usage: python analyze_correction_annotations.py [--items PATH] [--xlsx OUT]
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CORR = os.path.join(HERE, "..", "outputs", "correction")

CORRECTOR_NAME = {"first": "GPT-5.4", "second": "Gemini-3.1-pro"}


def slot_key(id_, model, span, corrector):
    return (id_, model, (span or "").strip(), corrector)


def load_items(path):
    """-> {(id,model,span): {ref, error_type, verdict_first, verdict_second}}"""
    d = json.load(open(path, encoding="utf-8"))
    items = d["items"] if isinstance(d, dict) else d
    out = {}
    for it in items:
        key = (it["id"], it["model"], (it["span"] or "").strip())
        cor = it.get("correctors", {})
        out[key] = {
            "ref": it.get("ref"),
            "error_type": it.get("error_type") or "",
            "verdict_first": (cor.get("first") or {}).get("verdict"),
            "verdict_second": (cor.get("second") or {}).get("verdict"),
        }
    return out


def norm_judgment(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in ("correct", "صحيح", "1", "true", "ok"):
        return "correct"
    if v in ("wrong", "خطأ", "incorrect", "0", "false"):
        # NB: hisham CSV uses '' for unjudged, '0' is NOT used for judgment here
        return "wrong"
    return None


def norm_judgment_csv(v):
    # CSV: judgment column holds 'correct'/'wrong'/'' (inappropriate is a separate col)
    if v is None:
        return None
    v = str(v).strip().lower()
    if v == "correct":
        return "correct"
    if v == "wrong":
        return "wrong"
    return None


def load_annotator(path):
    """-> (name, {slot_key: {'judgment','inappropriate','manual'}})"""
    recs = {}
    name = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        name = d.get("annotator", name)
        anns = d["annotations"] if isinstance(d, dict) else d
        for a in anns:
            k = slot_key(a["id"], a.get("model"), a.get("span"), a.get("corrector"))
            recs[k] = {
                "judgment": norm_judgment_csv(a.get("judgment")),
                "inappropriate": bool(a.get("inappropriate_question")),
                "manual": list(a.get("manual_corrections") or []),
            }
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("annotator"):
                    name = r["annotator"]
                k = slot_key(r["id"], r.get("model"), r.get("span"), r.get("corrector"))
                inappr = str(r.get("inappropriate_question", "")).strip() in ("1", "true", "True")
                man = r.get("manual_corrections") or ""
                recs[k] = {
                    "judgment": norm_judgment_csv(r.get("judgment")),
                    "inappropriate": inappr,
                    "manual": [man] if man.strip() else [],
                }
    return name, recs


def find_exports():
    files = []
    for p in glob.glob(os.path.join(CORR, "correction_annotations*")):
        low = p.lower()
        if low.endswith(".json") or low.endswith(".csv"):
            files.append(p)
    # exclude builder/intermediate artifacts + generated outputs that are not annotator exports
    skip = ("_first_topup", "_topup", "_original", "_combined", "_batch",
            "_consolidated", "_results")
    return [p for p in files if not any(s in os.path.basename(p) for s in skip)]


def pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "  n/a"


def kappa(pairs):
    """Cohen's kappa on a list of (a, b) binary labels."""
    n = len(pairs)
    if n == 0:
        return None, 0
    labels = ["correct", "wrong"]
    obs = sum(1 for a, b in pairs if a == b) / n
    pa = {l: sum(1 for a, _ in pairs if a == l) / n for l in labels}
    pb = {l: sum(1 for _, b in pairs if b == l) / n for l in labels}
    pe = sum(pa[l] * pb[l] for l in labels)
    k = (obs - pe) / (1 - pe) if (1 - pe) else 1.0
    return k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(CORR, "correction_annotation.json"))
    ap.add_argument("--xlsx", default="")
    args = ap.parse_args()

    items = load_items(args.items)
    print(f"Items: {len(items)} spans, {len(set(k[0] for k in items))} distinct questions")
    total_slots = 2 * len(items)

    exports = find_exports()
    annotators = []
    for p in exports:
        name, recs = load_annotator(p)
        annotators.append((name, recs, p))
    print(f"Annotators ({len(annotators)}): " + ", ".join(n for n, _, _ in annotators))
    print()

    # ---- per-annotator report ----
    for name, recs, path in annotators:
        judged = {k: v for k, v in recs.items() if v["judgment"] is not None}
        print(f"=== {name} ===  ({os.path.basename(path)})")
        print(f"  judged {len(judged)}/{total_slots} slots ({pct(len(judged), total_slots)})")
        jc = Counter(v["judgment"] for v in judged.values())
        print(f"  judgments: correct={jc['correct']}  wrong={jc['wrong']}")

        # per corrector, split by model's own verdict (ok vs خطأ)
        for cor in ("first", "second"):
            rows = [(k, v) for k, v in judged.items() if k[3] == cor]
            if not rows:
                continue
            corr_ok = sum(1 for _, v in rows if v["judgment"] == "correct")
            # split by model verdict
            att, att_ok, dec, dec_ok = 0, 0, 0, 0
            for k, v in rows:
                meta = items.get((k[0], k[1], k[2]))
                mv = meta["verdict_%s" % cor] if meta else None
                attempted = (mv == "ok")
                if attempted:
                    att += 1
                    att_ok += v["judgment"] == "correct"
                else:
                    dec += 1
                    dec_ok += v["judgment"] == "correct"
            print(f"  {CORRECTOR_NAME[cor]:14s}: human-correct {corr_ok}/{len(rows)} ({pct(corr_ok,len(rows))})"
                  f"  | corrected→ok {att_ok}/{att} ({pct(att_ok,att)})"
                  f"  | declined→ok {dec_ok}/{dec} ({pct(dec_ok,dec)})")

        # by ref
        for ref in ("quran", "hadith"):
            rows = [(k, v) for k, v in judged.items()
                    if (items.get((k[0], k[1], k[2])) or {}).get("ref") == ref]
            ok = sum(1 for _, v in rows if v["judgment"] == "correct")
            print(f"  ref={ref:7s}: human-correct {ok}/{len(rows)} ({pct(ok,len(rows))})")
        inappr = sorted(set((k[0], k[1]) for k, v in recs.items() if v["inappropriate"]))
        if inappr:
            print(f"  flagged inappropriate: {len(inappr)} -> {inappr}")
        print()

    # ---- inter-annotator agreement (all pairs) ----
    if len(annotators) >= 2:
        print("=== Inter-annotator agreement (slots both judged) ===")
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                na, ra, _ = annotators[i]
                nb, rb, _ = annotators[j]
                shared = []
                for k in set(ra) & set(rb):
                    a, b = ra[k]["judgment"], rb[k]["judgment"]
                    if a is not None and b is not None:
                        shared.append((a, b))
                agree = sum(1 for a, b in shared if a == b)
                k_, n_ = kappa(shared)
                kstr = f"{k_:.3f}" if k_ is not None else "n/a"
                print(f"  {na}  vs  {nb}: shared judged={n_}, raw agreement {pct(agree,n_)}, Cohen's κ={kstr}")
                # disagreements
                dis = [k for k in set(ra) & set(rb)
                       if ra[k]["judgment"] and rb[k]["judgment"]
                       and ra[k]["judgment"] != rb[k]["judgment"]]
                print(f"    disagreements: {len(dis)}  (adjudication queue)")
                dis.sort()
                for k in dis:
                    meta = items.get((k[0], k[1], k[2])) or {}
                    mv = meta.get("verdict_%s" % k[3])
                    print(f"      {k[0]:9s} {k[1]:22s} {CORRECTOR_NAME[k[3]]:14s} "
                          f"ref={meta.get('ref','?'):6s} modelverdict={mv or '?':4s} "
                          f"| {na}={ra[k]['judgment']:7s} {nb}={rb[k]['judgment']}")

    if args.xlsx:
        write_xlsx(args.xlsx, items, annotators, total_slots)


def write_xlsx(out, items, annotators, total_slots):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("openpyxl not installed; skipping xlsx", file=sys.stderr)
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "slots"
    hdr = ["id", "model", "ref", "error_type", "corrector", "corrector_model",
           "model_verdict", "span"]
    for name, _, _ in annotators:
        hdr.append(name)
    ws.append(hdr)
    for (id_, model, span), meta in items.items():
        for cor in ("first", "second"):
            row = [id_, model, meta["ref"], meta["error_type"], cor,
                   CORRECTOR_NAME[cor], meta["verdict_%s" % cor], span]
            k = (id_, model, span, cor)
            for _, recs, _ in annotators:
                row.append((recs.get(k) or {}).get("judgment") or "")
            ws.append(row)
    wb.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
