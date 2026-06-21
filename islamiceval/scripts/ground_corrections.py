#!/usr/bin/env python3
"""
ground_corrections.py
=====================
Validate that the CORRECTION TEXTS the models output are actually grounded in the
canonical corpora we ship to participants (Quran + the six Hadith books).

The match is SPACE-INSENSITIVE and un-diacriticized: we strip tashkeel/tatweel,
fold letter variants (أإآٱ→ا, ى→ي, ة→ه, ؤ→و, ئ→ي), drop every non-Arabic-letter,
AND remove ALL whitespace, then test whether the candidate is a contiguous
substring of a corpus entry. This makes "ها هنا" == "هاهنا" and ignores the
formatting drift between the model's wording and the corpus wording.

A correction text is GROUNDED iff its space-stripped normal form is a contiguous
substring of some corpus entry of the right ref type (Quran multi-ayah quotes are
matched against the concatenated mushaf as a fallback).

For each candidate it prints whether it matched and the corpus source(s); at the
end it reports a per-model grounding/correctness percentage.

Input: a correction pipeline JSON (default `pipeline_oneshot.json`); reads the
`oneshot` block (or `correction` block via --block). Candidate = each entry in a
model's `candidates` list when verdict == "ok".

Usage:
    python ground_corrections.py
    python ground_corrections.py --input ../outputs/correction/pipeline_results.json --block correction
    python ground_corrections.py --show all      # print every candidate, not just summary+misses
"""
import os, re, sys, json, argparse, difflib
from collections import defaultdict, Counter

# print Arabic without crashing on a cp1252 console (Windows default)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.join(HERE, "..", "data", "corpora")
DEF_INPUT = os.path.join(HERE, "..", "outputs", "correction", "pipeline_oneshot.json")

# ── un-diacriticized + letter-folded, SPACE-STRIPPED normalization ──────────────
_DEL = {c: None for c in (list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660))
                          + [0x0670, 0x0640, 0x0621] + list(range(0x06D6, 0x06EE)))}  # tashkeel + tatweel + standalone hamza ء
_REPL = {0x0623: 0x0627, 0x0625: 0x0627, 0x0622: 0x0627, 0x0671: 0x0627,        # أإآٱ -> ا
         0x0649: 0x064A, 0x0629: 0x0647, 0x0624: 0x0648, 0x0626: 0x064A}        # ى->ي ة->ه ؤ->و ئ->ي
_TABLE = {**_DEL, **_REPL}
_nonletter = re.compile("[^" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A) + "]")

# When --alef_deletion is set, drop every alif-class char too (full alif ا, plus the
# dagger/wasla/madda/hamza-carrier alifs which _TABLE already folds to ا). This bridges
# the Uthmani-rasm vs corpus-orthography gap where the long ā is written as a superscript
# alif on one side and a full alif letter on the other (e.g. model إخوَٰنا vs corpus إخوانا).
ALEF_DEL = False           # module toggle, set once from CLI before load_corpus()
def norm_ns(t: str) -> str:
    """Normalize and remove ALL whitespace -> contiguous Arabic-letter string.
    If ALEF_DEL, also drop every alif (ا) so matching is alif-insensitive."""
    if not t:
        return ""
    s = _nonletter.sub("", t.translate(_TABLE))
    if ALEF_DEL:
        s = s.replace("ا", "")
    return s


def norm_dedup(t: str) -> str:
    """Dedup key for grounded corrections: un-diacritized + letter-folded + space-
    stripped, but ALIF KEPT (independent of the ALEF_DEL toggle). Two matns that are
    byte-identical un-diacritized collapse to ONE line (same hadith across books);
    two matns differing only by an alif stay DISTINCT (separate variant lines)."""
    if not t:
        return ""
    return _nonletter.sub("", t.translate(_TABLE))


# ── surah-name normalization + Quran reference parsing (for the ref fallback) ────
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
def _norm_surah(name):
    """Fixed surah-name key: fold letters, drop diacritics/spaces and a leading 'ال'."""
    s = _nonletter.sub("", (name or "").translate(_TABLE))     # letters only (no spaces)
    return s[2:] if s.startswith("ال") else s

def parse_quran_ref(src):
    """'سورة البقرة 185' / 'البقرة ١٨٥' / 'البقرة 185-186' -> (surah_key, ayah:int) or None."""
    s = (src or "").translate(_AR_DIGITS)
    m = re.search(r"\d+", s)
    if not m:
        return None
    name = s[:m.start()].replace("سورة", "").replace("الآية", "").replace("اية", "")
    return _norm_surah(name), int(m.group(0))


# ── corpus (space-stripped blobs) ──────────────────────────────────────────────
class Mushaf:
    """Concatenated normalized mushaf for multi-ayah quotes. Behaves like the old
    string for `cn in mushaf` / `len(mushaf)`; `locate(cn)` gives the surah+ayah range
    label, `locate_full(cn)` also returns the full raw text of the covered ayah(s),
    `raw_of(source)` returns a single ayah's raw text, and `ref_ayah(surah,ayah)`
    maps a parsed reference to its source string."""
    def __init__(self, q):
        parts, bounds, pos = [], [], 0
        self.raw, self.ordered, self.by_ref = {}, [], {}
        for v in q:
            src = f"سورة {v['surah_name']} {v['ayah_id']}"
            blob = norm_ns(v["ayah_text"])
            parts.append(blob)
            bounds.append((pos, pos + len(blob), src))     # [start, end) -> source
            pos += len(blob)
            self.raw[src] = v["ayah_text"]
            self.ordered.append((src, v["ayah_text"]))
            self.by_ref[(_norm_surah(v["surah_name"]), int(v["ayah_id"]))] = src
        self.text = "".join(parts)
        self.bounds = bounds

    def __contains__(self, s):
        return s in self.text

    def __len__(self):
        return len(self.text)

    def _cover(self, cn):
        i = self.text.find(cn)
        if i < 0:
            return []
        j = i + len(cn)
        return [k for k, (a, b, _) in enumerate(self.bounds) if a < j and b > i]

    def locate(self, cn):
        """Source label for the ayah(s) the substring cn spans, or None if absent."""
        cov = self._cover(cn)
        if not cov:
            return None
        srcs = [self.bounds[k][2] for k in cov]
        if len(srcs) == 1:
            return srcs[0]
        (pf, fa), (pl, la) = srcs[0].rsplit(" ", 1), srcs[-1].rsplit(" ", 1)
        return f"{pf} {fa}-{la}" if pf == pl else f"{srcs[0]} - {srcs[-1]}"

    def locate_full(self, cn):
        """(label, full raw text of covered ayahs) or (None, None)."""
        cov = self._cover(cn)
        if not cov:
            return None, None
        return self.locate(cn), " ".join(self.ordered[k][1] for k in cov)

    def raw_of(self, src):
        return self.raw.get(src)

    def ref_ayah(self, surah_key, ayah):
        return self.by_ref.get((surah_key, ayah))


def load_corpus():
    q = json.load(open(os.path.join(CORP, "quranic_verses.json"), encoding="utf-8"))
    q.sort(key=lambda v: (v["surah_id"], v["ayah_id"]))
    quran = [(f"سورة {v['surah_name']} {v['ayah_id']}", norm_ns(v["ayah_text"])) for v in q]
    mushaf = Mushaf(q)                                     # concatenated for multi-ayah quotes

    h = json.load(open(os.path.join(CORP, "six_hadith_books.json"), encoding="utf-8"))
    hadith = []
    for x in h:
        matn = x.get("Matn", "")
        # if not matn or matn == "None":
        #     matn = x.get("hadithTxt", "")
        coll = (x.get("title", "").split(" - ")[0]).strip() or "؟"
        hadith.append((f"{coll} {x.get('hadithID', '')}".strip(), norm_ns(matn)))
    return quran, mushaf, hadith


# --partial: also count a match when a full corpus entry is an exact substring of
# the candidate (the model wrapped/extended a real text, or quoted a shorter book's
# matn). Guarded by a min length so trivially short corpus entries don't spuriously
# match inside a long candidate.
PARTIAL = False
PARTIAL_MIN_LEN = 15        # min normalized (space-stripped) corpus-entry length for reverse hits

def _search_one(cn, ref, quran, mushaf, hadith, max_src):
    """Return (sources, kind). kind: 'in' = candidate ⊂ corpus entry (a fragment of
    a real text); 'contains' = corpus entry ⊂ candidate (model extended a real text)."""
    docs = quran if ref == "quran" else hadith
    srcs = [src for src, blob in docs if blob and cn in blob]
    if srcs:
        return srcs[:max_src], "in"
    if ref == "quran":                                    # multi-ayah / cross-ayah quote
        loc = mushaf.locate(cn)
        if loc:
            return [loc], "in"
    if PARTIAL:                                           # reverse: corpus entry ⊂ candidate
        rev = [src for src, blob in docs
               if blob and len(blob) >= PARTIAL_MIN_LEN and blob in cn]
        if rev:
            print("partial returned:", rev)
            return rev[:max_src], "contains"
    return [], None

def lookup(cand_text, ref, quran, mushaf, hadith, max_src=5):
    """Grounded iff the space-stripped normal form shares a full contiguous overlap
    with a corpus entry (candidate ⊂ entry, or — with --partial — entry ⊂ candidate).
    Searches the span's ref type first, then the OTHER corpus as a fallback.

    Returns (grounded, sources, matched_ref, kind). matched_ref != ref flags a
    cross-type (ref-mismatch) grounding; kind ∈ {'in','contains',None}."""
    cn = norm_ns(cand_text)
    if not cn:
        return False, [], None, None
    srcs, kind = _search_one(cn, ref, quran, mushaf, hadith, max_src)
    if srcs:
        return True, srcs, ref, kind
    other = "hadith" if ref == "quran" else "quran"
    srcs, kind = _search_one(cn, other, quran, mushaf, hadith, max_src)
    if srcs:
        return True, srcs, other, kind
    return False, [], None, None


# ─────────────────────────────────────────────────────────────────────────────
# SNAPPER — for an ungrounded near-miss, snap to the single closest corpus entry
# when difflib char-ratio >= threshold, and adopt that entry's VERBATIM text as gold.
# Ratio is computed on a fixed normalization (un-diacritized + space-stripped, alif
# KEPT — independent of --alef_deletion) so it is a true letter-match percentage.
# ─────────────────────────────────────────────────────────────────────────────
SNAP_THRESHOLD = 0.90
_nonletter_sp = re.compile("[^" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A) + r"\s]")

def _norm_words(t):
    """Un-diacritized + letter-folded, SPACE-PRESERVING (keeps alif). For tokenizing."""
    if not t:
        return ""
    return re.sub(r"\s+", " ", _nonletter_sp.sub(" ", t.translate(_TABLE))).strip()

def load_snap_corpus():
    """Per-corpus list of (source, raw_text, sim_str, words) + a word->idx index."""
    q = json.load(open(os.path.join(CORP, "quranic_verses.json"), encoding="utf-8"))
    q.sort(key=lambda v: (v["surah_id"], v["ayah_id"]))
    Q = []
    for v in q:
        w = _norm_words(v["ayah_text"]).split()
        Q.append((f"سورة {v['surah_name']} {v['ayah_id']}", v["ayah_text"], "".join(w), w))
    H = []
    for x in json.load(open(os.path.join(CORP, "six_hadith_books.json"), encoding="utf-8")):
        matn = x.get("Matn", "")
        if not matn or matn == "None":
            continue                       # snap gold must be a real matn — never the
                                           # isnad-laden hadithTxt fallback. If nothing with a
                                           # matn clears the threshold, we simply don't snap.
        coll = (x.get("title", "").split(" - ")[0]).strip() or "؟"
        src = f"{coll} {x.get('hadithID', '')}".strip()
        w = _norm_words(matn).split()
        H.append((src, matn, "".join(w), w))
    def index(docs):
        idx = {}
        for i, d in enumerate(docs):
            for tok in set(d[3]):
                idx.setdefault(tok, []).append(i)
        return idx
    return {"quran": (Q, index(Q)), "hadith": (H, index(H)),
            "quran_raw":  {s: r for s, r, _, _ in Q},
            "hadith_raw": {s: r for s, r, _, _ in H}}

def _best_in(cand_sim, cand_words, docs, idx, cap=80):
    """Highest difflib ratio among the `cap` corpus docs sharing the most words."""
    cnt = Counter()
    for tok in set(cand_words):
        for i in idx.get(tok, ()):
            cnt[i] += 1
    best = (0.0, None)
    for i, _ in cnt.most_common(cap):
        r = difflib.SequenceMatcher(None, cand_sim, docs[i][2], autojunk=False).ratio()
        if r > best[0]:
            best = (r, i)
    return best

def snap(cand_text, ref, snapc, threshold=SNAP_THRESHOLD, cross=True):
    """Snap a near-miss to its single closest corpus entry. Searches the span's ref
    type, then (if cross) the other corpus; returns the global best if >= threshold,
    else None. Result carries the corpus VERBATIM text to use as gold."""
    cw = _norm_words(cand_text).split()
    cs = "".join(cw)
    if not cs:
        return None
    other = "hadith" if ref == "quran" else "quran"
    best = None
    for rtype in ((ref, other) if cross else (ref,)):
        docs, idx = snapc[rtype]
        score, i = _best_in(cs, cw, docs, idx)
        if i is not None and (best is None or score > best["score"]):
            src, raw, _, _ = docs[i]
            best = {"score": round(score, 4), "source": src, "gold_text": raw,
                    "matched_ref": rtype}
    if best and best["score"] >= threshold:
        return best
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE — the layered correction→corpus policy (used by build_corrections_excel
# and for rollout). Quran: always resolve to the FULL ayah (text match → snap →
# reference fallback), else eliminate. Hadith: snap (exact/≥90%), then OPTIONAL
# expansion (--hadith_expansion) to every ≥95%-or-substring corpus matn, else
# eliminate. Any correction not grounded to the corpus is eliminated.
# ─────────────────────────────────────────────────────────────────────────────
HADITH_EXPAND = False          # module toggle (set from --hadith_expansion)
EXPAND_TH     = 0.90           # hadith expansion similarity threshold
REF_COVER_TH  = 0.90           # Quran reference-fallback: min cand-coverage within the cited ayah

def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _coverage(cand_norm, target_norm):
    """Fraction of cand chars matched within target (length-insensitive to target):
    tolerates a small orthographic slip, but a long corrupted candidate scores low."""
    if not cand_norm:
        return 0.0
    sm = difflib.SequenceMatcher(None, cand_norm, target_norm, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(cand_norm)

def expand_hadith(matn, snapc, th=EXPAND_TH):
    """Every corpus hadith (source, matn_text) whose matn is >= th ratio to, OR a
    (normalized) substring/superstring of, `matn`. Single-hop from `matn` (no drift).
    Returns RAW (source, text) pairs in best-overlap order — collapsing identical
    matns across books is the caller's job (_group_variants), so alif-distinct
    variants are NOT lost here."""
    docs, idx = snapc["hadith"]
    cw = _norm_words(matn).split()
    cs = "".join(cw)
    if not cs:
        return []
    cnt = Counter()
    for tok in set(cw):
        for i in idx.get(tok, ()):
            cnt[i] += 1
    out = []
    for i, _ in cnt.most_common(150):
        ds = docs[i][2]
        if ds and (cs in ds or ds in cs
                   or difflib.SequenceMatcher(None, cs, ds, autojunk=False).ratio() >= th):
            out.append((docs[i][0], docs[i][1]))
    return out


def _group_variants(pairs):
    """[(source, matn_text), ...] -> [(matn_text, [sources]), ...]. Collapse EXACT
    (norm_dedup, alif-kept) duplicate matns across books into a single entry with
    their sources unioned, preserving first-seen order. Distinct variants stay apart."""
    by, order = {}, []
    for src, text in pairs:
        k = norm_dedup(text)
        if not k:
            continue
        if k not in by:
            by[k] = [text, []]; order.append(k)
        if src and src not in by[k][1]:
            by[k][1].append(src)
    return [(by[k][0], by[k][1]) for k in order]

def _resolve_quran(cand_text, cn, claimed_source, quran, mushaf, hadith, snapc):
    # tier 1: text match (substring / multi-ayah) -> snap to the FULL ayah(s)
    srcs, _ = _search_one(cn, "quran", quran, mushaf, hadith, 5)
    if srcs:
        full = mushaf.raw_of(srcs[0]) or mushaf.locate_full(cn)[1]
        return {"status": "grounded", "gold_text": full, "sources": [srcs[0]], "matched_ref": "quran"}
    # tier 2: >=90% snap to a full ayah (quran-only here; cross handled by caller)
    sn = snap(cand_text, "quran", snapc, SNAP_THRESHOLD, cross=False)
    if sn:
        return {"status": "snapped", "gold_text": sn["gold_text"], "sources": [sn["source"]], "matched_ref": "quran"}
    # tier 3: reference fallback — the cited ayah, lenient-substring / small-mistake
    pr = parse_quran_ref(claimed_source)
    if pr:
        src = mushaf.ref_ayah(*pr)
        if src:
            ayah_norm = norm_ns(mushaf.raw_of(src))
            if cn and (cn in ayah_norm or _coverage(cn, ayah_norm) >= REF_COVER_TH):
                return {"status": "ref", "gold_text": mushaf.raw_of(src), "sources": [src], "matched_ref": "quran"}
    return {"status": "eliminated", "gold_text": None, "sources": [], "matched_ref": None}

def _resolve_hadith(cand_text, cn, quran, mushaf, hadith, snapc):
    # tier 1: exact substring -> the matched corpus matn is the anchor for expansion
    srcs, _ = _search_one(cn, "hadith", quran, mushaf, hadith, 5)
    if srcs:
        if HADITH_EXPAND:
            anchor = snapc["hadith_raw"].get(srcs[0], cand_text)   # full canonical matn
            variants = _group_variants(expand_hadith(anchor, snapc)) or [(anchor, list(srcs))]
            return {"status": "grounded", "gold_text": variants[0][0], "sources": variants[0][1],
                    "variants": variants, "matched_ref": "hadith"}
        return {"status": "grounded", "gold_text": cand_text, "sources": _dedup(srcs),
                "variants": [(cand_text, _dedup(srcs))], "matched_ref": "hadith"}
    # tier 2: >=90% snap -> the corpus matn is the anchor for expansion
    sn = snap(cand_text, "hadith", snapc, SNAP_THRESHOLD, cross=False)
    if sn:
        if HADITH_EXPAND:
            variants = _group_variants(expand_hadith(sn["gold_text"], snapc)) \
                       or [(sn["gold_text"], [sn["source"]])]
            return {"status": "snapped", "gold_text": variants[0][0], "sources": variants[0][1],
                    "variants": variants, "matched_ref": "hadith"}
        return {"status": "snapped", "gold_text": sn["gold_text"], "sources": [sn["source"]],
                "variants": [(sn["gold_text"], [sn["source"]])], "matched_ref": "hadith"}
    return {"status": "eliminated", "gold_text": None, "sources": [], "variants": [], "matched_ref": None}

def resolve_correction(cand_text, ref, claimed_source, quran, mushaf, hadith, snapc):
    """Resolve one model correction to corpus gold. Returns
    {status: grounded|snapped|ref|eliminated, gold_text, sources, matched_ref}.
    Tries the span's ref type first; if eliminated, tries the OTHER corpus once
    (cross-ref) before final elimination."""
    cn = norm_ns(cand_text)
    if not cn:
        return {"status": "eliminated", "gold_text": None, "sources": [], "matched_ref": None}
    fn = _resolve_quran if ref == "quran" else _resolve_hadith
    r = fn(cand_text, cn, *( (claimed_source, quran, mushaf, hadith, snapc) if ref == "quran"
                             else (quran, mushaf, hadith, snapc) ))
    if r["status"] != "eliminated":
        return r
    # cross-ref: a real ayah cited in a hadith-position span, or vice versa
    if ref == "quran":
        return _resolve_hadith(cand_text, cn, quran, mushaf, hadith, snapc)
    return _resolve_quran(cand_text, cn, claimed_source, quran, mushaf, hadith, snapc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEF_INPUT)
    ap.add_argument("--block", default="oneshot", choices=["oneshot", "correction"],
                    help="which pipeline block holds the candidates")
    ap.add_argument("--show", default="miss", choices=["miss", "all"],
                    help="'miss' prints only ungrounded candidates; 'all' prints every one")
    ap.add_argument("--alef_deletion", action="store_true",
                    help="alif-insensitive matching: drop every alif (ا) on both sides "
                         "(bridges Uthmani-rasm dagger-alif vs corpus full-alif, e.g. إخوٰنا/إخوانا)")
    ap.add_argument("--partial", action="store_true",
                    help="also count a match when a full corpus entry is an exact substring "
                         f"of the candidate (model wrapped/extended a real text; min "
                         f"{PARTIAL_MIN_LEN} normalized chars)")
    ap.add_argument("--snap", action="store_true",
                    help="for each ungrounded miss, snap to the single closest corpus entry "
                         f"when difflib char-ratio >= threshold (default {SNAP_THRESHOLD}); "
                         "adopt that entry's verbatim text as gold")
    ap.add_argument("--snap_threshold", type=float, default=SNAP_THRESHOLD)
    ap.add_argument("--snap_out", default=os.path.join(HERE, "..", "outputs", "correction",
                                                       "snapped_corrections.json"))
    args = ap.parse_args()

    global ALEF_DEL, PARTIAL
    ALEF_DEL = args.alef_deletion
    PARTIAL  = args.partial

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: input not found: {args.input}\n"
                 f"  pass --input <path to a correction pipeline json>")
    print(f"input: {args.input}", flush=True)
    print(f"matching: un-diacriticized + space-insensitive"
          f"{' + ALIF-INSENSITIVE (--alef_deletion)' if ALEF_DEL else ''}"
          f"{' + PARTIAL/reverse-substring (--partial)' if PARTIAL else ''}", flush=True)
    print("loading corpora...", flush=True)
    quran, mushaf, hadith = load_corpus()
    print(f"quran ayahs={len(quran):,}  hadith={len(hadith):,}  mushaf chars={len(mushaf):,}\n")
    snapc = load_snap_corpus() if args.snap else None

    data = json.load(open(args.input, encoding="utf-8"))
    per_model = defaultdict(lambda: {"n": 0, "grounded": 0, "xref": 0, "contains": 0, "snapped": 0})
    miss_by_ref = defaultdict(int)
    gold = []                                    # per-candidate gold table (written when --snap)

    for s in data["per_sample"]:
        block = s.get(args.block, {}) or {}
        for model, rec in block.items():
            p = (rec or {}).get("prediction") or {}
            if p.get("verdict") != "ok":
                continue
            for c in (p.get("candidates") or []):
                text, ref = c.get("text", ""), s["ref"]
                claimed = c.get("source", "")
                ok, srcs, mref, kind = lookup(text, ref, quran, mushaf, hadith)
                d = per_model[model]
                d["n"] += 1; d["grounded"] += ok
                xref = ok and mref != ref
                d["xref"] += xref
                d["contains"] += (kind == "contains")
                snapped = None
                if not ok and args.snap:
                    snapped = snap(text, ref, snapc, args.snap_threshold)
                if not ok and not snapped:
                    miss_by_ref[ref] += 1
                if snapped:
                    d["snapped"] += 1
                if args.snap:
                    status = "grounded" if ok else ("snapped" if snapped else "miss")
                    gold.append({
                        "id": s["id"], "model": model, "ref": ref,
                        "original_text": text, "claimed_source": claimed, "status": status,
                        "source": (snapped["source"] if snapped else (srcs[0] if srcs else None)),
                        "matched_ref": (snapped["matched_ref"] if snapped else (mref if ok else None)),
                        "gold_text": (snapped["gold_text"] if snapped else (text if ok else None)),
                        "snap_score": (snapped["score"] if snapped else None),
                    })
                if args.show == "all" or (not ok):
                    flag = "MISS" if not ok else ("OK*x" if xref else "OK  ")
                    if snapped:
                        flag = "SNAP"
                    print(f"[{flag}] {model:24} {s['id']} {ref:6} claim='{claimed}'")
                    print(f"        text: {text[:90]}")
                    if ok:
                        tags = []
                        if xref: tags.append(f"cross-ref -> {mref}")
                        if kind == "contains": tags.append("corpus ⊂ candidate")
                        tag = f" ({'; '.join(tags)})" if tags else ""
                        print(f"        corpus{tag}: {', '.join(srcs)}")
                    elif snapped:
                        xr = f"; cross-ref -> {snapped['matched_ref']}" if snapped["matched_ref"] != ref else ""
                        print(f"        SNAP {snapped['score']:.3f} -> {snapped['source']}{xr}")
                        print(f"        gold: {snapped['gold_text'][:90]}")

    print("\n================ GROUNDING / CORRECTNESS ================")
    print("  (grounded = correction text is a verbatim substring of the corpus,")
    print("   un-diacriticized + space-insensitive; *x = grounded in the other corpus)\n")
    tot_n = tot_g = tot_x = tot_c = tot_s = 0
    for m, d in sorted(per_model.items()):
        n, g, x, ct, sp = d["n"], d["grounded"], d["xref"], d["contains"], d["snapped"]
        tot_n += n; tot_g += g; tot_x += x; tot_c += ct; tot_s += sp
        extra = ", ".join(filter(None, [f"{x} cross-ref" if x else "",
                                        f"{ct} corpus⊂cand" if ct else ""]))
        xs = f"  (incl. {extra})" if extra else ""
        snp = f"  + {sp} snapped = {100*(g+sp)/n:5.1f}% covered" if (args.snap and n) else ""
        print(f"  {m:26} {g:3}/{n:<3} grounded = {100*g/n:5.1f}%{xs}{snp}" if n
              else f"  {m:26} no candidates")
    if tot_n:
        extra = ", ".join(filter(None, [f"{tot_x} cross-ref" if tot_x else "",
                                        f"{tot_c} corpus⊂cand" if tot_c else ""]))
        snp = f"  + {tot_s} snapped = {100*(tot_g+tot_s)/tot_n:5.1f}% covered" if args.snap else ""
        print(f"  {'ALL':26} {tot_g:3}/{tot_n:<3} grounded = {100*tot_g/tot_n:5.1f}%  "
              f"(incl. {extra}){snp}")
        print(f"\n  still ungrounded {tot_n - tot_g - tot_s}: by ref {dict(miss_by_ref)}")
        print("  NB remaining misses are mostly Quran Uthmani-rasm (الصلوة/الصلاة) and")
        print("     genuine hadith matn variants (أجسام/أجساد) — not diacritic/space issues.")

    if args.snap:
        os.makedirs(os.path.dirname(args.snap_out), exist_ok=True)
        json.dump({"threshold": args.snap_threshold, "n": len(gold),
                   "n_snapped": tot_s, "rows": gold},
                  open(args.snap_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n  -> wrote gold table ({tot_s} snapped) to {args.snap_out}")


if __name__ == "__main__":
    main()
