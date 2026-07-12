import re
import norm

COLL_RE=re.compile(r'^(صحيح|سنن|مسند|موطأ|جامع|السنن|مسلم|البخاري)\b')
REF_LINE=re.compile(r'(صحيح|سنن|مسند|موطأ|جامع|رواه|أخرجه|سورة|سوره|الراوي|المحدث|المصدر|الصفحة|الصفحه|خلاصة|خلاصه|التخريج)')
JUNK_EXACT={'الصفحة او الرقم','الصفحه او الرقم','خلاصة حكم المحدث','خلاصه حكم المحدث',
            'الحديث ضعيف','المصدر','الصفحة أو الرقم'}
BARE_COLL={'ابو داود','النسائي','الترمذي','البخاري','مسلم','ابن ماجه','ابن حبان','احمد',
           'رواه مسلم','اخرجه الترمذي','البخاري ، مسلم','البخاري ، مسلم'}

def _norm_plain(s):
    return norm.normalize_text(re.sub(r'\[[^\]]*\]',' ',s))

_META_PAT=re.compile(r'(الراوي|المحدث|المصدر|التخريج|خلاصة حكم|خلاصه حكم|الصفحة أو الرقم|الصفحه او الرقم)')
def _is_ref_or_junk_line(line):
    """A line that is only a citation/metadata (no scripture text of its own)."""
    # Strip a bracketed ref first: "قل أعوذ برب الفلق [سورة الفلق 1]" is real scripture, not a
    # citation line — judge only what remains after removing [..].
    line_nb=re.sub(r'\[[^\]]*\]',' ',line)
    t=_norm_plain(line_nb).strip()
    if not t: return True
    if t in JUNK_EXACT or t in BARE_COLL: return True
    raw_nt=norm.strip_tashkeel(line_nb)
    # isnad/provenance metadata lines (الراوي ... المصدر ...) carry no matn -> junk
    if _META_PAT.search(raw_nt):
        # if the line is dominated by metadata markers and has no long scripture clause, drop it
        # heuristic: metadata line rarely exceeds a dozen tokens and always contains a marker
        if len(t.split())<=12: return True
    # short + contains a ref/metadata keyword + NO real scripture content => treat as ref line
    toks=t.split()
    cite=set('صحيح سنن مسند موطا جامع مسلم البخاري النسائي الترمذي داود ابي ابن ماجه احمد '
             'رواه اخرجه سوره صفحه الراوي المحدث المصدر الصفحه الرقم او خلاصه حكم و'.split())
    content=[w for w in toks if w not in cite and not w.isdigit()]
    if len(toks)<=4 and REF_LINE.search(raw_nt) and len(content)<2: return True
    return False

def parse_units(manual_corrections):
    """Return list of candidate UNITS. Each unit = {'text': normalized scripture text,
    'ref': (type,name,num) or None, 'raw': raw}. Text+citation kept together.
    - Split on ||| (explicit separator).
    - Within a chunk, join a scripture line with a following ref-only line.
    - A newline that introduces a NEW scripture line starts a new unit.
    - Drop empty / pure-metadata units."""
    units=[]
    for entry in (manual_corrections or []):
        for chunk in entry.split('|||'):
            lines=[l for l in chunk.split('\n')]
            cur_text=[]; cur_ref=None
            def flush():
                nonlocal cur_text,cur_ref
                txt=' '.join(cur_text).strip()
                ntxt=_norm_plain(txt).strip()
                if ntxt:  # has real scripture content
                    units.append({'text':ntxt,'ref':cur_ref,'raw':txt})
                cur_text=[]; cur_ref=None
            for line in lines:
                if not line.strip(): continue
                if _is_ref_or_junk_line(line):
                    # citation/metadata for the current text; capture ref if any, don't start new unit
                    r=norm.extract_ref(line)
                    if r and cur_ref is None: cur_ref=r
                    # if it's junk with no ref and we have no text yet, skip entirely
                    continue
                else:
                    # a scripture line. If we already have text accumulated, this is a NEW unit.
                    if cur_text: flush()
                    cur_text.append(line)
                    r=norm.extract_ref(line)
                    if r: cur_ref=r
            flush()
    # dedupe by (text,ref)
    seen=set(); out=[]
    for u in units:
        k=(u['text'],u['ref'])
        if k in seen: continue
        seen.add(k); out.append(u)
    # FINAL FILTER: drop citation-only units (text is just source names/numbers, no scripture).
    _CITE_TOK=set('صحيح سنن مسند موطا جامع مسلم البخاري النسائي الترمذي داود ابي ابن ماجه احمد '
                  'رواه اخرجه سوره صفحه الراوي المحدث المصدر الصفحه الرقم او خلاصه حكم و'.split())
    kept=[]
    for u in out:
        content=[w for w in u['text'].split() if w not in _CITE_TOK and not w.isdigit()]
        if len(content)>=3:      # needs real scripture content to be a correction
            kept.append(u)
    return kept

if __name__=='__main__':
    import json
    d=json.load(open('/mnt/user-data/uploads/annotators_corrected_enriched_judged.json'))
    A=d['annotators']
    # test on the problematic items
    tests={}
    for a in A:
        for r in a['annotations']:
            if r['id'] in ('Q00004','Q00439','Q00016','Q01298') and r['manual_corrections'] and r['corrector']=='first':
                tests.setdefault((r['id'],a['annotator'][:6]),r['manual_corrections'])
    for (qid,ann),mc in list(tests.items())[:8]:
        print(f'=== {qid} {ann} ===')
        for u in parse_units(mc):
            print(f"   text[{len(u['text'].split())}tok]={u['text'][:55]!r} ref={u['ref']}")
