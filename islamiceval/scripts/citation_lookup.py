"""
Quick & rough citation estimator (whole-text scan).

Scans the ENTIRE answer (not just bracketed quotes): every K-word window is tagged
as Quran / Hadith / neither by membership in the corpora, then contiguous matched
windows are merged into "passages". A passage counts as a citation if it covers at
least the per-type minimum number of words (QURAN_MIN / HADITH_MIN). Diacritic-
insensitive. Rough — see caveats at the bottom.

Defaults: K=3 shingles, Quran counts at >= 3 contiguous matched words (sensitive,
to catch short verse quotes), Hadith at >= 6 (matns are long; 3-word hits are noise).

Importable: `from citation_lookup import ref_counts, scan, QSET, HSET`.

CLI:
    python citation_lookup.py ../outputs/answers/explicit/*.json
    python citation_lookup.py --quran-min 3 --hadith-min 6 -k 3 file.json
"""
import sys, os, re, json, glob, argparse

CORP = os.path.join(os.path.dirname(__file__), "..", "data", "corpora")

# Defaults (the Quran threshold is now 3 contiguous words).
K = 3            # window size (words) used to build corpus shingles
QURAN_MIN = 3    # a Quran run must cover >= this many words to count as a citation
HADITH_MIN = 6   # hadith matns are long; require more contiguous words to avoid noise

# Arabic normalization built from code points (ASCII source — no pasted glyphs).
_DEL = {c: None for c in (list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660))
                          + [0x0670, 0x0640] + list(range(0x06D6, 0x06EE)))}   # tashkeel + tatweel
_REPL = {0x0623: 0x0627, 0x0625: 0x0627, 0x0622: 0x0627, 0x0671: 0x0627,        # أإآٱ -> ا
         0x0649: 0x064A, 0x0629: 0x0647, 0x0624: 0x0648, 0x0626: 0x064A}        # ى->ي ة->ه ؤ->و ئ->ي
_TABLE = {**_DEL, **_REPL}
# keep only Arabic letters (0621-063A, 0641-064A) and whitespace
_keep = re.compile("[^" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A) + r"\s]")
def norm(t: str) -> str:
    t = t.translate(_TABLE)
    t = _keep.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def grams(words, k=K):
    return [hash(" ".join(words[i:i+k])) for i in range(len(words) - k + 1)]

def build_set(texts, k=K):
    s = set()
    for t in texts:
        s.update(grams(norm(t).split(), k))
    return s

print("Loading corpora...", file=sys.stderr)
quran = json.load(open(os.path.join(CORP, "quranic_verses.json"), encoding="utf-8"))
hadith = json.load(open(os.path.join(CORP, "six_hadith_books.json"), encoding="utf-8"))
QSET = build_set((v["ayah_text"] for v in quran), K)
HSET = build_set((h["hadithTxt"] for h in hadith), K)
print(f"  Quran shingles: {len(QSET):,} | Hadith shingles: {len(HSET):,}  (K={K})", file=sys.stderr)

# --- attribution ("claimed source") counting ---
# digit-keeping normalizer (norm() strips digits, which we need for surah+number)
_keepd = re.compile("[^" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A)
                    + "0-9" + "".join(chr(c) for c in range(0x0660, 0x066A)) + r"\s]")
def dnorm(t: str) -> str:
    t = t.translate(_TABLE)
    t = _keepd.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

_DIGIT = re.compile("[0-9" + "".join(chr(c) for c in range(0x0660, 0x066A)) + "]")
# surah names from the corpus (>=3 chars to avoid 1-2 letter names matching noise)
_SURAHS = sorted({norm(v["surah_name"]) for v in quran if len(norm(v["surah_name"])) >= 3},
                 key=len, reverse=True)
_surah_re = re.compile("|".join(re.escape(s) for s in _SURAHS))
# hadith attribution keywords (normalized so hamza/diacritic variants match)
_HKW = sorted({dnorm(k) for k in ["رواه", "أخرجه", "متفق عليه", "البخاري", "مسلم",
              "الترمذي", "النسائي", "ابن ماجه", "أبو داود", "أحمد", "إلخ"] if dnorm(k)},
              key=len, reverse=True)
_hkw_re = re.compile("|".join(re.escape(k) for k in _HKW))

def attributions(ans):
    """Return (quran_attributions, hadith_attributions) — named sources, not content.
    Quran: a surah name followed by a number within the next 5 words (tolerates an
    intervening word like 'الآية')."""
    t = dnorm(ans)
    qa = sum(1 for m in _surah_re.finditer(t)
             if any(_DIGIT.search(w) for w in t[m.end():].split()[:5]))
    ha = len(_hkw_re.findall(t))
    return qa, ha

def scan(ans, k=K, quran_min=QURAN_MIN, hadith_min=HADITH_MIN):
    """Return (n_words, quran_passages, hadith_passages, scripture_words)."""
    w = norm(ans).split()
    tags = ["q" if s in QSET else "h" if s in HSET else "." for s in grams(w, k)]  # Quran wins ties
    qp = hp = sw = 0
    i, L = 0, len(tags)
    while i < L:
        if tags[i] == ".":
            i += 1; continue
        t = tags[i]; j = i
        while j < L and tags[j] == t:
            j += 1
        covered = (j - 1 + k) - i                # words spanned by this run
        minreg = quran_min if t == "q" else hadith_min
        if covered >= minreg:
            sw += covered
            qp += (t == "q"); hp += (t == "h")
        i = j
    return len(w), qp, hp, sw

def ref_counts(ans):
    """Convenience: (quran_passages, hadith_passages) at module defaults."""
    _, qp, hp, _ = scan(ans)
    return qp, hp


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-k", type=int, default=K, help="shingle size (words)")
    ap.add_argument("--quran-min", type=int, default=QURAN_MIN)
    ap.add_argument("--hadith-min", type=int, default=HADITH_MIN)
    args = ap.parse_args()
    files = []
    for a in args.files:
        files += sorted(glob.glob(a))
    if not files:
        sys.exit("give one or more answers .json files")

    hdr = (f"{'file':40}{'n':>6}{'avg_w':>7}{'quran':>8}{'hadith':>8}{'cites':>7}"
           f"{'q_attr':>8}{'h_attr':>8}{'%script':>8}")
    print(hdr); print("-" * len(hdr))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        nw = q = h = sw = qa = ha = 0
        for x in d:
            ans = str(x.get("answer", ""))
            a, qp, hp, s = scan(ans, args.k, args.quran_min, args.hadith_min)
            nw += a; q += qp; h += hp; sw += s
            dqa, dha = attributions(ans)
            qa += dqa; ha += dha
        n = len(d)
        print(f"{os.path.basename(f):40}{n:>6}{nw//n:>7}{q:>8}{h:>8}{q+h:>7}"
              f"{qa:>8}{ha:>8}{100*sw/max(nw,1):>7.1f}%")


if __name__ == "__main__":
    _main()
