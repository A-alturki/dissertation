#!/usr/bin/env python3
"""
verify_typology.py
==================
Corpus-grounded evaluation of a typology annotation file (output of
annotate_typology.py). Two checks, no human gold needed:

1. SPAN PRECISION / RECALL  (shingle overlap vs the corpora)
   - Treat the Quran + six-books corpora as the source of truth for what
     AUTHENTIC scripture is present in each answer.
   - gold Quranic 4-grams in an answer = answer shingles that exist in the Quran.
   - Precision(Quran) = of the 4-grams inside the model's Quran spans, how many
     are actually Quranic.   Recall(Quran) = of the answer's Quranic 4-grams, how
     many fall inside the model's Quran spans.  Same for Hadith.
   - CAVEAT: corpus = authentic text only, so recall here is recall on REAL
     citations; fabricated citations are (correctly) not in the corpus.

2. VERDICT ACCURACY  (Correct/Incorrect vs corpus)
   - For each Quran span: corpus-true label = Correct iff the (normalized) span
     text is an exact substring of the Quran, else Incorrect.
   - For each Hadith_Matn span: Correct iff it is a substring of some canonical
     hadith text. (Hadith matching is fuzzier — flagged as approximate.)
   - Compare to the model's label and report a confusion matrix + disagreements.

Usage:
    python verify_typology.py ../outputs/annotations/allam-7b__gpt-5.4-mini_typology.json
"""
import os, re, sys, json, argparse

CORP = os.path.join(os.path.dirname(__file__), "..", "data", "corpora")
K = 4  # shingle size (words)

# --- Arabic normalization (same scheme as citation_lookup.py) ---
_DEL = {c: None for c in (list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660))
                          + [0x0670, 0x0640] + list(range(0x06D6, 0x06EE)))}
_REPL = {0x0623: 0x0627, 0x0625: 0x0627, 0x0622: 0x0627, 0x0671: 0x0627,
         0x0649: 0x064A, 0x0629: 0x0647, 0x0624: 0x0648, 0x0626: 0x064A}
_TABLE = {**_DEL, **_REPL}
_keep = re.compile("[^" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A) + r"\s]")

def norm(t: str) -> str:
    t = str(t).translate(_TABLE)
    t = _keep.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def shingles(text: str):
    w = norm(text).split()
    return {" ".join(w[i:i+K]) for i in range(len(w) - K + 1)}

MIN_REGION = 6  # a contiguous corpus-matched run must span >= this many words to be a "gold citation"

def region_gold(ans, QSET, HSET):
    """Gold Quran/Hadith shingles = those inside CONTIGUOUS corpus-matched runs of
    >= MIN_REGION words. This excludes incidental short matches (common phrases that
    happen to appear in scripture but aren't citations)."""
    w = norm(ans).split()
    grams = [" ".join(w[i:i+K]) for i in range(len(w) - K + 1)]
    tags = ["q" if g in QSET else "h" if g in HSET else "." for g in grams]  # Quran wins ties
    gold = {"q": set(), "h": set()}
    i, L = 0, len(tags)
    while i < L:
        if tags[i] == ".":
            i += 1; continue
        t = tags[i]; j = i
        while j < L and tags[j] == t:
            j += 1
        covered = (j - 1 + K) - i  # words spanned by this run
        if covered >= MIN_REGION:
            gold[t].update(grams[i:j])
        i = j
    return gold["q"], gold["h"]


def load_corpora():
    print("Loading corpora (one-time, ~30s for hadith)...", file=sys.stderr)
    quran = json.load(open(os.path.join(CORP, "quranic_verses.json"), encoding="utf-8"))
    hadith = json.load(open(os.path.join(CORP, "six_hadith_books.json"), encoding="utf-8"))
    # Quran shingle set + ordered concatenation for exact substring tests
    QSET = set()
    for v in quran:
        QSET |= shingles(v["ayah_text"])
    ordered = sorted(quran, key=lambda v: (v.get("surah_id", 0), v.get("ayah_id", 0)))
    QCONCAT = " " + " ".join(norm(v["ayah_text"]) for v in ordered) + " "
    # Hadith shingle set + newline-joined concatenation for exact substring tests.
    # "\n" separators stop a span from matching ACROSS two different hadith (normalized
    # text is Arabic letters + spaces only, so it can never span a newline).
    HSET = set()
    hnorms = []
    for h in hadith:
        n = norm(h.get("hadithTxt", ""))
        hnorms.append(n)
        w = n.split()
        HSET.update(" ".join(w[i:i+K]) for i in range(len(w) - K + 1))
    HCONCAT = "\n" + "\n".join(hnorms) + "\n"
    print(f"  Quran shingles {len(QSET):,} | Hadith shingles {len(HSET):,}", file=sys.stderr)
    return QSET, QCONCAT, HSET, HCONCAT


def longest_corpus_run(nslice, CSET):
    """Longest contiguous run (in 4-grams) of the span that are all in the corpus.
    Boundary-robust: a real verse/hadith quote keeps a long matched run even if the
    span sloppily includes a leading cue ('قال الله تعالى') or braces."""
    w = nslice.split()
    best = cur = 0
    for i in range(len(w) - K + 1):
        g = " ".join(w[i:i+K])
        if g in CSET:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best  # >=3 grams ≈ 6+ contiguous corpus-matched words

def quran_exact(nslice, QSET, QCONCAT):
    # real if the whole (normalized) span is verbatim Quran (handles SHORT verses),
    # OR it contains a long contiguous Quranic run (handles spans that sloppily
    # include a leading cue / braces around a long verse).
    if len(nslice.split()) >= 2 and nslice in QCONCAT:
        return True
    return longest_corpus_run(nslice, QSET) >= 3

def hadith_exact(nslice, HSET, HCONCAT):
    # real if the whole matn is a verbatim substring of some canonical hadith (handles
    # SHORT matns like «لا ضرر ولا ضرار»), OR it contains a long contiguous hadith run
    # (handles boundary-noisy long spans). APPROXIMATE — matn variants may miss.
    if len(nslice.split()) >= 2 and nslice in HCONCAT:
        return True
    return longest_corpus_run(nslice, HSET) >= 3


def micro_pr(num, den):
    return (num / den) if den else None


def main():
    ap = argparse.ArgumentParser(description="Corpus-grounded check of a typology file")
    ap.add_argument("typology_json")
    ap.add_argument("--dump", default=None, help="write per-disagreement detail to this JSON")
    args = ap.parse_args()

    data = json.load(open(args.typology_json, encoding="utf-8"))
    ps = data.get("per_sample", data)
    QSET, QCONCAT, HSET, HCONCAT = load_corpora()

    # ---- aggregate counters ----
    pr = {"Quran": dict(p_num=0, p_den=0, r_num=0, r_den=0),
          "Hadith_Matn": dict(p_num=0, p_den=0, r_num=0, r_den=0)}
    # verdict confusion: key (type) -> {(model_label, corpus_label): count}
    conf = {"Quran": {}, "Hadith_Matn": {}}
    disagreements = []
    n_spans = {"Quran": 0, "Hadith_Matn": 0, "Hadith_Isnad": 0, "Reference": 0}

    for r in ps:
        ans = r.get("answer", "") or ""
        spans = r.get("spans") or []
        # gold shingles = contiguous corpus-matched regions (>= MIN_REGION words)
        gold_q, gold_h = region_gold(ans, QSET, HSET)
        # model span shingles by type
        msh = {"Quran": set(), "Hadith_Matn": set()}
        for sp in spans:
            t = sp.get("type")
            n_spans[t] = n_spans.get(t, 0) + 1
            s, e = sp.get("span_start"), sp.get("span_end")
            slice_ = ans[s:e] if isinstance(s, int) and isinstance(e, int) else ""
            nslice = norm(slice_)
            if t in msh:
                msh[t] |= shingles(slice_)
            # verdict check for Quran / Hadith_Matn
            if t == "Quran":
                truth = "Correct" if quran_exact(nslice, QSET, QCONCAT) else "Incorrect"
            elif t == "Hadith_Matn":
                truth = "Correct" if hadith_exact(nslice, HSET, HCONCAT) else "Incorrect"
            else:
                truth = None
            if truth is not None:
                ml = sp.get("label")
                conf[t][(ml, truth)] = conf[t].get((ml, truth), 0) + 1
                if ml != truth:
                    disagreements.append({"id": r.get("sample_id"), "type": t,
                                          "model_label": ml, "corpus_label": truth,
                                          "span_text": sp.get("span_text"),
                                          "slice": slice_[:120]})
        # precision/recall accumulation
        for t in ("Quran", "Hadith_Matn"):
            gold = gold_q if t == "Quran" else gold_h
            m = msh[t]
            corpus_set = QSET if t == "Quran" else HSET
            pr[t]["p_num"] += len(m & corpus_set); pr[t]["p_den"] += len(m)
            pr[t]["r_num"] += len(m & gold);        pr[t]["r_den"] += len(gold)

    # ---- report ----
    print(f"\n=== {os.path.basename(args.typology_json)} ===")
    print(f"answers={len(ps)}  spans: " + ", ".join(f"{k}={v}" for k, v in n_spans.items()))

    print("\n-- SPAN PRECISION / RECALL (shingle overlap vs corpora) --")
    print(f"{'type':14}{'precision':>12}{'recall':>12}   (precision=spans are real; recall=real cites covered)")
    for t in ("Quran", "Hadith_Matn"):
        p = micro_pr(pr[t]["p_num"], pr[t]["p_den"])
        rc = micro_pr(pr[t]["r_num"], pr[t]["r_den"])
        ps_ = f"{p:.1%}" if p is not None else "n/a"
        rs_ = f"{rc:.1%}" if rc is not None else "n/a"
        print(f"{t:14}{ps_:>12}{rs_:>12}   (P {pr[t]['p_num']}/{pr[t]['p_den']}, R {pr[t]['r_num']}/{pr[t]['r_den']} shingles)")

    print("\n-- VERDICT ACCURACY (model label vs corpus) --")
    for t in ("Quran", "Hadith_Matn"):
        c = conf[t]
        total = sum(c.values())
        agree = sum(n for (ml, tr), n in c.items() if ml == tr)
        acc = f"{agree/total:.1%}" if total else "n/a"
        note = "" if t == "Quran" else "  (hadith match is approximate — verify disagreements manually)"
        print(f"  {t}: {agree}/{total} agree ({acc}){note}")
        for (ml, tr), n in sorted(c.items()):
            mark = "OK " if ml == tr else "XX "
            print(f"      {mark}model={ml:9} corpus={tr:9} : {n}")

    print(f"\ndisagreements: {len(disagreements)}")
    if args.dump:
        json.dump({"pr": pr, "confusion": {t: {f"{k[0]}|{k[1]}": v for k, v in c.items()}
                                            for t, c in conf.items()},
                   "disagreements": disagreements},
                  open(args.dump, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"detail -> {args.dump}")


if __name__ == "__main__":
    main()
