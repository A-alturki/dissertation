#!/usr/bin/env python3
"""
hadith_similarity.py
====================
Single source of truth for the two hadith-matn similarity metrics under evaluation
for expand_hadith:

  char_ratio(a,b)  — the CURRENT metric: difflib char-ratio on the space-stripped,
                     un-diacritized, letter-folded string (frame/boilerplate included).
  jaccard(a,b)     — the CANDIDATE metric: strip isnad/honorific frame + stopwords +
                     leading و/ف, then Jaccard over the remaining content-token SET.

Both consume the same normalization as ground_corrections (_norm_words). Keeping the
definitions here means the blind labeling study, the threshold analysis, and any future
engine change all use identical scoring.
"""
import difflib
import ground_corrections as G

# ── isnad / honorific / attribution frame (normalized: ة→ه ى→ي أإآ→ا, no diacritics) ──
FRAME = set("""
صلي الله عليه وسلم سلم رضي عنه عنها عنهم عنهما عز وجل سبحانه تعالي تبارك جل جلاله السلام عليهم
قال قالت قالوا يقول تقول قل رسول النبي نبي محمد يا
حدثنا حدثني اخبرنا اخبرني انبانا انباني سمعت سمعنا عن انه انها ان
""".split())
# generic Arabic function-word stopwords (Jaccard noise)
STOP = set("""
من في الي علي ما لا هو هي به له بها لها كل او ثم قد حتي الذي التي هذا هذه وهو وهي
اذا اذ عند بعد قبل كان يكون هم هن انا نحن انت وعن
""".split())
# narration openers — only removed when extended=True
OPEN = set("جاء جاءه اتي اتاه سالت سال سالنا رجل جاءت اتت".split())


def _strip_wf(tok):
    while tok and tok[0] in "وف":
        tok = tok[1:]
    return tok


def content_tokens(text, extended=False, use_stop=True):
    """Normalized content-token SET: drop frame (+openers if extended) + stopwords,
    strip leading و/ف, drop 1-char tokens."""
    drop = FRAME | (OPEN if extended else set()) | (STOP if use_stop else set())
    out = set()
    for t in G._norm_words(text).split():
        t = _strip_wf(t)
        if len(t) > 1 and t not in drop:
            out.add(t)
    return out


def content_seq(text, extended=False, use_stop=True):
    """ORDERED content tokens (frame/stopwords removed), for sequence metrics like
    char-ratio. Same filter as content_tokens but order-preserving (no set)."""
    drop = FRAME | (OPEN if extended else set()) | (STOP if use_stop else set())
    out = []
    for t in G._norm_words(text).split():
        t = _strip_wf(t)
        if len(t) > 1 and t not in drop:
            out.append(t)
    return out


def raw_token_set(text):
    """Normalized token SET with the frame KEPT (no removal) — for the ablation."""
    return set(t for t in G._norm_words(text).split() if len(t) > 1)


def jaccard_raw(a, b):
    """Set-overlap Jaccard on ALL tokens (frame kept). The 'SET + raw' 2x2 cell."""
    A, B = raw_token_set(a), raw_token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def jaccard(a, b, extended=False, use_stop=True):
    A = content_tokens(a, extended, use_stop)
    B = content_tokens(b, extended, use_stop)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def char_ratio(a, b):
    """difflib char-ratio on the FULL space-stripped string (frame INCLUDED) — legacy."""
    ca = "".join(G._norm_words(a).split())
    cb = "".join(G._norm_words(b).split())
    if not ca or not cb:
        return 0.0
    return difflib.SequenceMatcher(None, ca, cb, autojunk=False).ratio()


def char_content(a, b, extended=False, use_stop=True):
    """difflib char-ratio on the FRAME-STRIPPED content string — the ablation that gives
    char-ratio the same common-word removal as Jaccard, isolating metric vs stripping."""
    ca = "".join(content_seq(a, extended, use_stop))
    cb = "".join(content_seq(b, extended, use_stop))
    if not ca or not cb:
        return 0.0
    return difflib.SequenceMatcher(None, ca, cb, autojunk=False).ratio()
