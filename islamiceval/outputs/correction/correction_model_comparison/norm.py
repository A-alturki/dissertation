"""
Normalization + solution-parsing utilities for the IslamicEval 2026 correction-subtask
IAA / accuracy analysis. Deterministic, documented for a methods section.

Decisions locked with the user:
- Slot -> model: corrector 'first' == GPT, 'second' == GEMINI.
- Gold per model-row = annotator MAJORITY verdict (accuracy = model agreement with humans).
- Accuracy grain = model-row (each model's correction judged separately),
  agreement over the 3 annotators' verdicts on that model-row.
- correctable flag = MAJORITY of annotators imply a fix is needed (item-level).
- Solution exact-normalized normalizer: NFC + strip tashkeel + normalize alef/yaa
  + collapse whitespace + strip refs/punct. Also a ref-only metric (parsed source tag).
- Solution matching reported: exact-normalized AND ref-only, plus >=1-span overlap.
- .txt typology used as qualitative color only (no hard join).
"""
import re, unicodedata

SENTINEL = "لا تصحيح مقترح"            # substring test for "no correction proposed"
SPAN_SEP = "|||"                        # multi-span separator inside a single solution string

# --- Arabic normalization -------------------------------------------------
# Tashkeel / harakat (fatha, damma, kasra, tanwin, shadda, sukun, superscript alef, etc.)
_TASHKEEL = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]')
# Tatweel (kashida)
_TATWEEL = '\u0640'

def strip_tashkeel(s: str) -> str:
    s = s.replace(_TATWEEL, '')
    return _TASHKEEL.sub('', s)

def norm_letters(s: str) -> str:
    # alef variants -> bare alef
    s = re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', s)
    # alef maqsura -> yaa ; and final yaa handling kept simple (yaa stays yaa)
    s = s.replace('\u0649', '\u064A')
    # taa marbuta -> haa (common normalization; conservative)
    s = s.replace('\u0629', '\u0647')
    # remove Arabic-Indic / Quranic decorative marks left over
    return s

# Bracketed ref e.g. [سورة المائدة 38]  /  [صحيح البخاري 3174]
_REF_BRACKET = re.compile(r'\[[^\]]*\]')
# A source keyword that marks something as a reference line.
_REF_KEYWORD = re.compile(r'(سور[ةه]|صحيح|سنن|مسند|موطأ|جامع|رواه|أخرجه)')

def strip_refs(s: str):
    """Separate the correction BODY (ayah/hadith text) from its reference.
    A reference is either a [bracketed] tag or a trailing line/segment that
    begins with a source keyword (سورة / صحيح / سنن ...). Only a genuine
    trailing reference is stripped; the body is preserved for exact matching.
    Returns (body_text, ref_string_or_None)."""
    ref = None
    m = _REF_BRACKET.search(s)
    if m:
        ref = m.group(0)
        s = _REF_BRACKET.sub(' ', s)
    # Look for a trailing reference on its own line first (most common form).
    lines = [ln for ln in s.split('\n')]
    if len(lines) > 1 and _REF_KEYWORD.search(lines[-1]) and len(lines[-1].strip()) < 40:
        if ref is None:
            ref = lines[-1].strip()
        s = '\n'.join(lines[:-1])
    return s, ref

_PUNCT = re.compile(r'[^\u0600-\u06FF\s]')  # keep only Arabic letters + space

def normalize_text(s: str) -> str:
    """Full exact-match normalizer: NFC, strip refs+punct, tashkeel, letters, whitespace."""
    if s is None:
        return ''
    s = unicodedata.normalize('NFC', s)
    s, _ = strip_refs(s)
    s = strip_tashkeel(s)
    s = norm_letters(s)
    s = _PUNCT.sub(' ', s)          # strip punctuation, ref digits, latin, etc.
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- Reference (source) extraction for ref-only matching -------------------
# surah name may be one or two tokens (e.g. "آل عمران"); grab up to the number
# allow ':' / punctuation between surah name and ayah number
_SURAH = re.compile(r'سور[ةه]\s+([\u0600-\u06FF\s]+?)\s*[:：.\-]?\s*(\d+)')
_SURAH_NONUM = re.compile(r'سور[ةه]\s+([\u0600-\u06FF]+)')
# collection name may be up to two tokens (e.g. "أبي داود", "ابن ماجه")
_COLL  = re.compile(r'(صحيح|سنن|مسند|موطأ|جامع)\s+([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?)\s*[:：.\-]?\s*(\d+)?')

def extract_ref(s: str):
    """Return a canonical (source_type, name, number|None) tuple or None.
    Reads the number adjacent to the surah/collection name in the ref segment."""
    if s is None:
        return None
    s = unicodedata.normalize('NFC', s)
    _, reftxt = strip_refs(s)
    hay = strip_tashkeel(reftxt if reftxt else s)
    m = _COLL.search(hay)
    if m:
        return ('hadith', norm_letters(m.group(2)), m.group(3))
    m = _SURAH.search(hay)
    if m:
        return ('quran', norm_letters(m.group(1).strip()), m.group(2))
    m = _SURAH_NONUM.search(hay)
    if m:
        return ('quran', norm_letters(m.group(1).strip()), None)
    return None

# --- Solution (manual_corrections list) parsing ----------------------------
def parse_solution_spans(manual_corrections):
    """A single annotator's manual_corrections is a list; each entry may itself
    contain multiple spans joined by '|||' or by newlines. Return a set of
    normalized span texts and a set of refs (for >=1-span overlap + ref-only)."""
    spans, refs = [], []
    for entry in (manual_corrections or []):
        # split on explicit separator, then on newlines (each ayah/hadith on its own line)
        parts = re.split(r'\|\|\||\n', entry)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            spans.append(normalize_text(p))
            r = extract_ref(p)
            if r:
                refs.append(r)
    return set(x for x in spans if x), set(refs)

if __name__ == '__main__':
    import json
    d = json.load(open('/mnt/user-data/uploads/annotators_corrected_enriched_judged.json'))
    A = d['annotators']
    # sample a few manual corrections and show normalizer + ref extraction
    shown = 0
    for a in A:
        for r in a['annotations']:
            if r['manual_corrections'] and shown < 4:
                print('RAW  :', r['manual_corrections'][0][:90].replace('\n',' / '))
                sp, rf = parse_solution_spans(r['manual_corrections'])
                print('NORM :', list(sp)[0][:90] if sp else '(none)')
                print('REFS :', rf)
                print('---')
                shown += 1
    # sentinel detection sanity
    print('sentinel norm ->', repr(normalize_text('خطأ — لا تصحيح مقترح')))
