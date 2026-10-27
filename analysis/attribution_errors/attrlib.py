"""
attrlib.py -- shared loading, normalisation and lookup for the attribution-error
analysis. Reads only; writes nothing.

Scope note: no train/dev/test restriction is applied. The implicit condition was
never released so has no split, and the per-model validation targets supplied
with the task (Fanar-1-9B 14.9/6.0, ALLaM-7B 9.2/3.2) reproduce only over the
full implicit set. The explicit side is therefore also taken in full, so the two
conditions are counted the same way.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict

ROOT = r"C:\Users\Sh\OneDrive\Desktop\dissertation\islamiceval"
NP = os.path.join(ROOT, "new_paper")
QURAN = os.path.join(ROOT, "data", "corpora", "quranic_verses.json")

TXT_TYPES = {"Ayah", "matn"}
ATT_TYPES = {"isnad", "claimed_source"}

FRONTIER = ["gpt-5.4", "gemini-3.1-pro"]
MATCHED = ["acegpt-8b", "allam-7b", "command-r-7b", "fanar-1-9b", "gemma-3-12b",
           "jais-13b", "llama-3.1-8b", "mistral-7b", "phi-3.5", "qwen3-8b"]
EXTENDED = ["acegpt-32b", "fanar-2-27b", "gemma-4-12b", "gemma-4-31b",
            "karnak-40b", "ministral-3-14b", "qwen3.5-9b", "qwen3.5-27b"]
TIER = ({m: "frontier" for m in FRONTIER} | {m: "matched" for m in MATCHED}
        | {m: "extended" for m in EXTENDED})

DISPLAY = {
    "gpt-5.4": "GPT-5.4", "gemini-3.1-pro": "Gemini-3.1-Pro",
    "acegpt-8b": "AceGPT-v2-8B", "allam-7b": "ALLaM-7B", "command-r-7b": "Command-R-7B",
    "fanar-1-9b": "Fanar-1-9B", "gemma-3-12b": "Gemma-3-12B", "jais-13b": "Jais-13B",
    "llama-3.1-8b": "Llama-3.1-8B", "mistral-7b": "Mistral-7B",
    "phi-3.5": "Phi-3.5-mini", "qwen3-8b": "Qwen3-8B",
    "acegpt-32b": "AceGPT-v2-32B", "fanar-2-27b": "Fanar-2-27B",
    "gemma-4-12b": "Gemma-4-12B", "gemma-4-31b": "Gemma-4-31B",
    "karnak-40b": "Karnak-40B", "ministral-3-14b": "Ministral-3-14B",
    "qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-27b": "Qwen3.5-27B",
}

CONDITION_FILES = {
    "explicit": ["fanar_main.jsonl", "fanar_frontier.jsonl"],
    "implicit": ["a2a_12models.jsonl", "a2a_8models.jsonl"],
}

# ------------------------------------------------------------------ Arabic ---

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_NONLETTER = re.compile(r"[^\u0621-\u064A\s]")


def ar_norm(s: str) -> str:
    """Strip diacritics/tatweel, unify alef/ya/ta-marbuta, collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _DIACRITICS.sub("", s)
    s = (s.replace("\u0671", "\u0627").replace("\u0622", "\u0627")
           .replace("\u0623", "\u0627").replace("\u0625", "\u0627")
           .replace("\u0649", "\u064A").replace("\u0629", "\u0647")
           .replace("\u06A9", "\u0643"))
    s = _NONLETTER.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------- Quran ---

def load_quran():
    """-> (verses, name2id, surah_len). Canonical names come from the corpus."""
    data = json.load(open(QURAN, encoding="utf-8"))
    verses, surah_len, name2id = [], defaultdict(int), {}
    for v in data:
        sid, aid = int(v["surah_id"]), int(v["ayah_id"])
        verses.append((sid, aid, ar_norm(v["ayah_text"])))
        surah_len[sid] = max(surah_len[sid], aid)
        name2id.setdefault(ar_norm(v["surah_name"]), sid)
    # common surface variants beyond the corpus spelling
    extra = {
        "ال عمران": 3, "آل عمران": 3, "الاسراء": 17, "بني اسراييل": 17, "بنو اسراييل": 17,
        "الكهف": 18, "فصلت": 41, "حم السجده": 41, "غافر": 40, "المومن": 40,
        "الانسان": 76, "الدهر": 76, "المسد": 111, "اللهب": 111, "الاخلاص": 112,
        "التوحيد": 112, "الاسرا": 17, "محمد": 47, "القتال": 47, "النمل": 27,
        "فاطر": 35, "الملايكه": 35, "التوبه": 9, "براءه": 9, "برايه": 9,
    }
    for k, v in extra.items():
        name2id.setdefault(ar_norm(k), v)
    # bare names without the definite article
    for nm, sid in list(name2id.items()):
        if nm.startswith("ال") and len(nm) > 3:
            name2id.setdefault(nm[2:], sid)
    return verses, dict(name2id), dict(surah_len)


VERSES, NAME2ID, SURAH_LEN = load_quran()
_VERSE_INDEX = [(s, a, t) for s, a, t in VERSES if t]


def locate_ayah(cited: str):
    """True (surah_id, ayah_id) for a cited Arabic string, by exact containment
    in the canonical corpus. Returns None when nothing matches (typical for
    corrupted text) or when the match is ambiguous across surahs."""
    q = ar_norm(cited)
    if len(q) < 12:
        return None
    hits = {(s, a) for s, a, t in _VERSE_INDEX if q in t}
    if len(hits) == 1:
        return hits.pop()
    if len(hits) > 1:
        surahs = {s for s, _ in hits}
        return None if len(surahs) > 1 else sorted(hits)[0]
    return None


_NUM = r"(\d{1,3})"
# Order matters. Pattern 1 handles a number that PRECEDES the surah
# ("<ayah> 151 <min> <sura> al-Baqara"); without it that number is dropped.
# Minimum name length is 1, not 2, so single-letter surahs resolve; the
# claim "Qaf: 22" previously failed to parse at all.
_CLAIM_PATTERNS = [
    re.compile(r"سور[ةه]\s+([^\d،,:()\n]{1,30}?)[\s،,:]*(?:الآي[ةه]|الاي[ةه]|اي[ةه])?\s*" + _NUM),
    re.compile(r"(?:الآي[ةه]|الاي[ةه]|اي[ةه])\s*" + _NUM + r"\s*(?:من)?\s*سور[ةه]?\s*([^\d،,:()\n]{1,30})"),
    re.compile(r"([\u0621-\u064A\s]{2,30}?)\s*[،,]\s*(?:الآي[ةه]|الاي[ةه]|اي[ةه])?\s*[:：]?\s*" + _NUM),
    re.compile(r"سور[ةه]\s+([^\d،,:()\n]{1,30})"),
    re.compile(r"([\u0621-\u064A\s]{1,30}?)\s*[:：]\s*" + _NUM),
]


def parse_quran_claim(text: str):
    """-> (surah_id|None, surah_surface, ayah_no|None) from a claimed_source span."""
    if not text:
        return None, "", None
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    for i, pat in enumerate(_CLAIM_PATTERNS):
        m = pat.search(t)
        if not m:
            continue
        if i == 1:                     # 'ayah N of surah X' -> groups reversed
            ayah, surface = int(m.group(1)), m.group(2)
        else:
            surface = m.group(1)
            ayah = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else None
        surface = surface.strip(" ،,:-")
        sid = NAME2ID.get(ar_norm(surface))
        if sid is not None:
            return sid, surface, ayah
        if i in (0, 3):                # 'surah <unresolvable name>' is still a claim
            return None, surface, ayah
    m = re.search(_NUM, t)
    return None, t.strip(), (int(m.group(1)) if m else None)


# ------------------------------------------------------------------ loading ---

def citations(condition):
    """Yield one dict per citation (annotation with a text segment)."""
    for fn in CONDITION_FILES[condition]:
        for line in open(os.path.join(NP, fn), encoding="utf-8"):
            r = json.loads(line)
            model = r["source_model"]
            model = "gemini-3.1-pro" if model == "gemini-pro-3.1" else model
            ga = r["generated_answer"]
            for a in r["annotations"]:
                txt = [s for s in a["segments"] if s["type"] in TXT_TYPES]
                if not txt:
                    continue
                att = [s for s in a["segments"] if s["type"] in ATT_TYPES]
                cs = next((s for s in att if s["type"] == "claimed_source"), None)
                isn = next((s for s in att if s["type"] == "isnad"), None)
                sup = [s for s in att if s["label"] in ("correct", "incorrect")]
                yield {
                    "condition": condition,
                    "model": model,
                    "tier": TIER.get(model, "?"),
                    "qid": r["question_id"],
                    "kind": a["label"],                     # Ayah | Hadith
                    "fidelity": txt[0]["label"],            # correct|corrupted|fabricated
                    "text": ga[txt[0]["span_start"]:txt[0]["span_end"]],
                    "claimed_source": ga[cs["span_start"]:cs["span_end"]] if cs else None,
                    "cs_label": cs["label"] if cs else None,
                    "isnad": ga[isn["span_start"]:isn["span_end"]] if isn else None,
                    "isnad_label": isn["label"] if isn else None,
                    "supplied": bool(sup),
                    "misattributed": any(s["label"] == "incorrect" for s in sup),
                    "reason": a.get("reason") or "",
                }
