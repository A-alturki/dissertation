#!/usr/bin/env python3
"""
precompute_gold_pool.py
=======================
Assemble the per-span CANDIDATE POOL for manual gold construction on the frozen 100.

For each span it collects every correction any annotator SANCTIONED — their endorsed
GPT/Gemini corrections + their own manual corrections + a خطأ (decline) flag — then
CLUSTERS equivalent texts across annotators (so "X" typed two ways counts once, backed
by 2), canonicalizing each to corpus gold where possible. خطأ is a first-class candidate
with its own backer count.

Output: gold_pool_100.json — one record per span, candidates each carrying
{text, sources, grounded, provenance, backers[], count}. The gold-builder viewer loads
this; the curator confirms/edits clusters, drops junk, sets خطأ, and exports final gold.

Nothing here decides the gold — it only assembles the pool + counts. Majority (count>=2)
is derivable, not baked in.
"""
import json, os, sys, hashlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # islamiceval/
sys.path.insert(0, HERE)                                              # norm, parse2
sys.path.insert(0, os.path.join(ROOT, "scripts"))                    # ground_corrections
import norm
from parse2 import parse_units
import ground_corrections as G
from hadith_similarity import jaccard as _hjac

SENT = norm.SENTINEL
DATA = os.path.join(HERE, "..", "annotators_corrected_enriched.json")
KEEP = os.path.join(HERE, "keep100.json")
OUT  = os.path.join(HERE, "..", "gold_pool_100.json")

# grounding for canonicalization: snap each candidate to its corpus text. NO expansion —
# we want the single canonical identity per candidate, not the multi-book variant set.
G.ALEF_DEL = True; G.PARTIAL = True; G.HADITH_EXPAND = False
_alef = G.load_corpus()
_snap = G.load_snap_corpus()


def canonicalize(text, ref):
    """Snap a raw candidate to corpus gold. Returns (canonical_text, sources, grounded)."""
    r = G.resolve_correction(text, ref, "", *_alef, _snap)
    if r["status"] == "eliminated" or not r.get("gold_text"):
        return text.strip(), [], False
    return r["gold_text"], r.get("sources", []), True


def same_candidate(a, b, ref):
    """CONSERVATIVE cluster test: merge only when it's unambiguously the SAME text —
    identical normalized form, or both grounded to the same corpus ref. Distinct matn
    variants (same ruling, different wording, e.g. يُجمِع vs يُبيّت) stay SEPARATE; the
    curator merges them by hand if desired. (No fuzzy Jaccard merge — that conflated
    genuine variants.)"""
    if a["grounded"] and b["grounded"] and set(a["sources"]) & set(b["sources"]):
        return True
    return G.norm_dedup(a["text"]) == G.norm_dedup(b["text"])


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    A = d["annotators"]; ANN = [a["annotator"] for a in A]
    keep = set(tuple(x) for x in json.load(open(KEEP, encoding="utf-8")))

    # gather both corrector rows per (annotator, span)
    rows = defaultdict(dict)
    for a in A:
        nm = a["annotator"]
        for r in a["annotations"]:
            if (r["id"], r["model"], r["span"]) not in keep or r["judgment"] is None:
                continue
            ns = norm.normalize_text(r["span"])
            rows[(nm, ns)][r["corrector"]] = r
            rows[(nm, ns)]["meta"] = {"ref": r["ref"], "span": r["span"], "id": r["id"],
                                       "model": r["model"], "question": r.get("question", ""),
                                       "answer": r.get("answer", "")}

    # per span: each annotator's sanctioned raw candidates + khata flag
    spans = defaultdict(lambda: {"raw": defaultdict(list), "khata": {}, "meta": None})
    for (nm, ns), v in rows.items():
        spans[ns]["meta"] = v["meta"]; ref = v["meta"]["ref"]
        fr, sr = v.get("first"), v.get("second")
        cand = []
        # model corrections can list several variants joined by ||| — split them so each
        # becomes its own candidate (and grounds/clusters independently)
        if fr and fr["judgment"] == "correct" and SENT not in fr["gpt_correction"]:
            for part in fr["gpt_correction"].split("|||"):
                if part.strip():
                    cand.append(("GPT", part.strip()))
        if sr and sr["judgment"] == "correct" and SENT not in sr["gemini_correction"]:
            for part in sr["gemini_correction"].split("|||"):
                if part.strip():
                    cand.append(("Gemini", part.strip()))
        manual = []
        for rr in (fr, sr):
            if rr:
                manual += (rr.get("manual_corrections") or [])
        for u in parse_units(manual):
            cand.append(("manual", u["text"]))
        spans[ns]["raw"][nm] = cand
        # خطأ heuristic: endorsed a model's decline
        spans[ns]["khata"][nm] = bool(
            (fr and fr["judgment"] == "correct" and SENT in fr["gpt_correction"]) or
            (sr and sr["judgment"] == "correct" and SENT in sr["gemini_correction"]))

    # DISPLAY block — the raw material shown in the two slots (like the results viewer):
    # each corrector's model correction + verdict, and every annotator's judgment + manual.
    disp = defaultdict(lambda: {"first": {"model_text": None, "model_verdict": None, "annotators": {}},
                                 "second": {"model_text": None, "model_verdict": None, "annotators": {}},
                                 "s": None, "e": None})
    for (nm, ns), v in rows.items():
        D = disp[ns]; answer = v["meta"]["answer"]; span = v["meta"]["span"]
        if D["s"] is None:
            i = answer.find(span)
            if i >= 0:
                D["s"], D["e"] = i, i + len(span)
        for slot, field in [("first", "gpt_correction"), ("second", "gemini_correction")]:
            r = v.get(slot)
            if not r:
                continue
            txt = r[field]
            if D[slot]["model_verdict"] is None:
                D[slot]["model_text"] = None if SENT in txt else txt
                D[slot]["model_verdict"] = "خطأ" if SENT in txt else "ok"
            D[slot]["annotators"][nm] = {"judgment": r["judgment"],
                                          "manual": r.get("manual_corrections") or []}

    out = []
    for ns, S in spans.items():
        ref = S["meta"]["ref"]
        # canonicalize every raw candidate, tagging provenance + backer
        items = []
        for nm, cands in S["raw"].items():
            for prov, text in cands:
                ct, srcs, gr = canonicalize(text, ref)
                items.append({"text": ct, "raw": text, "sources": srcs, "grounded": gr,
                               "prov": prov, "ann": nm})
        # greedy cluster
        clusters = []
        for it in items:
            for c in clusters:
                if same_candidate(c["rep"], it, ref):
                    c["members"].append(it); c["backers"].add(it["ann"])
                    c["prov"].add(it["prov"])
                    if it["grounded"] and not c["rep"]["grounded"]:
                        c["rep"] = it            # prefer a grounded representative
                    break
            else:
                clusters.append({"rep": it, "members": [it], "backers": {it["ann"]},
                                  "prov": {it["prov"]}})
        candidates = [{
            "text": c["rep"]["text"],
            "sources": sorted(set(s for m in c["members"] for s in m["sources"])),
            "grounded": any(m["grounded"] for m in c["members"]),
            "provenance": sorted(c["prov"]),
            "backers": sorted(c["backers"]),
            "count": len(c["backers"]),
            "raw_forms": sorted(set(m["raw"] for m in c["members"])),
        } for c in clusters]
        candidates.sort(key=lambda x: -x["count"])
        khata_backers = sorted(a for a in ANN if S["khata"].get(a))
        out.append({
            "uid": S["meta"]["id"] + "_" + hashlib.md5(ns.encode("utf-8")).hexdigest()[:6],
            **S["meta"],
            "n_annotators": len([a for a in ANN if a in S["raw"] or S["khata"].get(a)]),
            "display": disp[ns],
            "candidates": candidates,
            "khata": {"backers": khata_backers, "count": len(khata_backers)},
        })

    out.sort(key=lambda x: (x["ref"], x["id"]))
    json.dump({"n": len(out), "annotators": ANN, "spans": out},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # quick summary
    import statistics
    nc = [len(s["candidates"]) for s in out]
    maj = [sum(1 for c in s["candidates"] if c["count"] >= 2) + (1 if s["khata"]["count"] >= 2 else 0) for s in out]
    print(f"wrote {os.path.relpath(OUT, HERE)}: {len(out)} spans")
    print(f"  candidates/span: avg={statistics.mean(nc):.1f} med={statistics.median(nc):.0f} max={max(nc)}")
    print(f"  spans with >=1 majority candidate: {sum(1 for m in maj if m>=1)}")
    print(f"  spans with a majority خطأ: {sum(1 for s in out if s['khata']['count']>=2)}")
    print(f"  ungrounded candidates (need curator eyes): "
          f"{sum(1 for s in out for c in s['candidates'] if not c['grounded'])}")


if __name__ == "__main__":
    main()
