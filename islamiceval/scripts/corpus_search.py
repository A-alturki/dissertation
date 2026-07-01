#!/usr/bin/env python3
"""
corpus_search.py
================
Deterministic candidate retrieval for the correction task (Method 2 = "judge").

Given an INCORRECT Quran / Hadith span, find the top-K canonical texts that share
the longest CONTIGUOUS word-run with the span, searching diacritic-insensitively
(reuses citation_lookup.norm). These candidates are then handed to an LLM judge
(see correction_pipeline.py), which picks the genuine distortion or rules
"خطأ". If this search returns nothing, the caller emits "خطأ" WITHOUT an LLM call.

Search policy (per the agreed "contiguity knob"):
  • Global gate (both types): >= MIN_OVERLAP distinct shared words AND a contiguous
    run of >= the per-type minimum.
  • Quran  is STRICT  — contiguous run must be >= QURAN_MIN_RUN (3) words, and the
    search also covers MULTI-AYAH quotes that cross an ayah boundary (matched over
    the concatenated mushaf, surfaced as a single synthetic candidate).
  • Hadith is LENIENT — contiguous run >= HADITH_MIN_RUN (2) words is enough (matns
    are long, so even a short verbatim spine is a useful anchor for the judge).

Ranking: (contiguous run length desc, then total word-overlap desc); deduped by
source; top K returned.

Public API:
    search(span: str, ref: str, k: int = 5) -> list[dict]
      ref ∈ {"quran","hadith"}. Each candidate dict:
        {"text", "source", "matched_run", "run_len", "overlap"}
      Empty list  =>  nothing met the gate.
"""
import os
import json
import difflib
from collections import Counter
from typing import Dict, List

from citation_lookup import norm

CORP = os.path.join(os.path.dirname(__file__), "..", "data", "corpora")

# ── gate / ranking thresholds ────────────────────────────────────────────────
MIN_OVERLAP    = 3    # >= this many DISTINCT shared (normalized) words, both types
QURAN_MIN_RUN  = 3    # Quran: contiguous run must be >= 3 words   (strict)
HADITH_MIN_RUN = 2    # Hadith: contiguous run >= 2 words is enough (lenient)
MAX_CAND_DOCS  = 80   # run the (cheap) contiguous-run check only on the N most-
                      # overlapping docs (keeps it fast over 35k hadith)
_SEP = "\x00"         # mushaf surah-separator sentinel (never matches answer text)


# ── per-doc normalized word lists + inverted index ───────────────────────────
def _load_quran():
    q = json.load(open(os.path.join(CORP, "quranic_verses.json"), encoding="utf-8"))
    q.sort(key=lambda v: (v["surah_id"], v["ayah_id"]))
    docs = []
    for v in q:
        docs.append({
            "words":  norm(v["ayah_text"]).split(),
            "text":   v["ayah_text"],
            "source": f"سورة {v['surah_name']} {v['ayah_id']}",
        })
    return q, docs


def _load_hadith():
    h = json.load(open(os.path.join(CORP, "six_hadith_books.json"), encoding="utf-8"))
    docs = []
    for x in h:
        matn = x.get("Matn")
        if not matn or matn == "None":            # no isolated matn -> skip (Matn-only, like
            continue                              # load_corpus; do NOT fall back to isnad-laden hadithTxt)
        coll = (x.get("title", "").split(" - ")[0]).strip() or "؟"
        docs.append({
            "words":  norm(matn).split(),
            "text":   matn,
            "source": f"{coll} {x.get('hadithID', '')}".strip(),
        })
    return docs


def _build_index(docs):
    idx: Dict[str, List[int]] = {}
    for i, d in enumerate(docs):
        for w in set(d["words"]):
            idx.setdefault(w, []).append(i)
    return idx


def _build_mushaf(qraw):
    """Concatenated mushaf word stream for multi-ayah matches. Returns
    (flat_words, ayah_of_word, ayah_meta) — ayah_of_word[i] is -1 at separators."""
    flat, aof, ameta, prev = [], [], [], None
    for v in qraw:
        if prev is not None and v["surah_id"] != prev:
            flat.append(_SEP); aof.append(-1)
        prev = v["surah_id"]
        ai = len(ameta)
        ameta.append(v)
        for w in norm(v["ayah_text"]).split():
            flat.append(w); aof.append(ai)
    return flat, aof, ameta


print("[corpus_search] loading corpora...", flush=True)
_QRAW, _Q_DOCS = _load_quran()
_H_DOCS        = _load_hadith()
_Q_IDX         = _build_index(_Q_DOCS)
_H_IDX         = _build_index(_H_DOCS)
_FLAT, _AOF, _AMETA = _build_mushaf(_QRAW)
print(f"[corpus_search] quran ayahs={len(_Q_DOCS):,}  hadith={len(_H_DOCS):,}  "
      f"mushaf words={len(_FLAT):,}", flush=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def _longest_run(a: List[str], b: List[str]):
    """Longest CONTIGUOUS common word-run between two word lists → (len, words)."""
    if not a or not b:
        return 0, []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    return m.size, a[m.a:m.a + m.size]


def _multi_ayah_candidate(span_words: List[str]):
    """If the span's longest verbatim run over the concatenated mushaf crosses an
    ayah boundary (>= 2 ayahs) and is >= QURAN_MIN_RUN words, return a synthetic
    candidate covering those ayahs; else None."""
    best_len, best_pos = 0, None
    n = len(span_words)
    i = 0
    while i < n:
        # extend a contiguous match starting at span position i, anywhere in the mushaf
        cand_positions = _seed_positions(span_words, i)
        for c in cand_positions:
            L = 0
            while (i + L < n and c + L < len(_FLAT)
                   and _AOF[c + L] != -1 and _FLAT[c + L] == span_words[i + L]):
                L += 1
            if L > best_len:
                best_len, best_pos = L, c
        i += 1
    if best_pos is None or best_len < QURAN_MIN_RUN:
        return None
    ayahs = sorted(set(a for a in _AOF[best_pos:best_pos + best_len] if a != -1))
    if len(ayahs) < 2:
        return None                                   # single-ayah → inverted index has it
    first, last = _AMETA[ayahs[0]], _AMETA[ayahs[-1]]
    text   = " ".join(_AMETA[a]["ayah_text"] for a in ayahs)
    if first["surah_id"] == last["surah_id"]:
        source = f"سورة {first['surah_name']} {first['ayah_id']}-{last['ayah_id']}"
    else:
        source = f"سورة {first['surah_name']} {first['ayah_id']}+"
    run_words = _FLAT[best_pos:best_pos + best_len]
    return {"text": text, "source": source, "matched_run": " ".join(run_words),
            "run_len": best_len, "overlap": best_len}


# seed index for the mushaf: 3-gram → start positions (built once, lazily)
_MUSHAF_SEED: Dict[str, List[int]] = {}
_SEED_K = 3
for _p in range(len(_FLAT) - _SEED_K + 1):
    _win = _FLAT[_p:_p + _SEED_K]
    if _SEP in _win:
        continue
    _MUSHAF_SEED.setdefault(" ".join(_win), []).append(_p)


def _seed_positions(span_words: List[str], i: int) -> List[int]:
    if i + _SEED_K > len(span_words):
        return []
    return _MUSHAF_SEED.get(" ".join(span_words[i:i + _SEED_K]), [])


# ── public API ───────────────────────────────────────────────────────────────
def search(span: str, ref: str, k: int = 5) -> List[Dict]:
    sw = norm(span).split()
    if not sw:
        return []
    if ref == "quran":
        docs, idx, min_run = _Q_DOCS, _Q_IDX, QURAN_MIN_RUN
    elif ref == "hadith":
        docs, idx, min_run = _H_DOCS, _H_IDX, HADITH_MIN_RUN
    else:
        raise ValueError(f"ref must be 'quran' or 'hadith', got {ref!r}")

    # candidate docs by distinct-word overlap (inverted index), capped
    cnt: Counter = Counter()
    for w in set(sw):
        for i in idx.get(w, ()):
            cnt[i] += 1
    cand_ids = [i for i, c in cnt.most_common() if c >= MIN_OVERLAP][:MAX_CAND_DOCS]

    results: List[Dict] = []
    for i in cand_ids:
        d = docs[i]
        run_len, run_words = _longest_run(sw, d["words"])
        if run_len >= min_run and cnt[i] >= MIN_OVERLAP:
            results.append({"text": d["text"], "source": d["source"],
                            "matched_run": " ".join(run_words),
                            "run_len": run_len, "overlap": cnt[i]})

    if ref == "quran":
        ma = _multi_ayah_candidate(sw)
        if ma:
            results.append(ma)

    # rank by (run length, overlap); dedupe by source; top-k
    seen, uniq = set(), []
    for r in sorted(results, key=lambda r: (-r["run_len"], -r["overlap"])):
        if r["source"] in seen:
            continue
        seen.add(r["source"])
        uniq.append(r)
    return uniq[:k]


if __name__ == "__main__":
    import sys
    ref = sys.argv[1] if len(sys.argv) > 1 else "quran"
    span = sys.argv[2] if len(sys.argv) > 2 else "رَبِّنَا هَبْ لَنَا مِنَ الصَّالِحِينَ"
    for j, c in enumerate(search(span, ref)):
        print(f"[{j}] run={c['run_len']} ov={c['overlap']} {c['source']}")
        print(f"     run: {c['matched_run']}")
        print(f"     txt: {c['text'][:90]}")
