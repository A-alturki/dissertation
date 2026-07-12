import json
import os
import re
import pandas as pd
from collections import defaultdict

'''
Task 3 - Correction scoring.

1- Place a TSV of the expected solutions in {reference_dir}
2- Place your prediction TSV in {prediction_dir}

Both TSVs use these columns:
    Response_ID, Annotation_ID, Segment_Type, Correction
  - Segment_Type is the citation type: "Ayah" (Quran) or "matn" (Hadith).
  - Correction is the corrected canonical text (the full Uthmani ayah for Quran,
    the matn for Hadith), or the string "خطأ" when no faithful correction is possible.
  - In the REFERENCE only, the Correction cell may list several acceptable
    corrections separated by " ||| " (e.g. the same hadith attested across books,
    or a verse quoted at more than one length). A prediction is counted CORRECT if
    it matches ANY one of them.

Quran is diacritic sensitive, and hadith is diacritic insensitive

A prediction matches if it equals ANY reference correction.
Reported scores:
  - accuracy       : overall accuracy over all rows
  - accuracy_Ayah  : accuracy on Quran rows only.
  - accuracy_matn  : accuracy on Hadith rows only.
'''

ROOT_DIR = os.environ.get('SCORING_ROOT', '/app/')
reference_dir = os.path.join(ROOT_DIR, 'input', 'ref')
prediction_dir = os.path.join(ROOT_DIR, 'input', 'res')
score_dir = os.path.join(ROOT_DIR, 'output')

SEP = ' ||| '
_WS = re.compile(r'\s+')

# letter-variant folding shared by both types (alef, yaa, taa-marbuta, hamza carriers)
_FOLD = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ى': 'ي', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي'}

# --- Quran: fold the letter variants, then standardize only the redundant "default"
#     same rule as last year's scorer
def standardize_quran(text):
    text = str(text)
    for a, b in _FOLD.items():
        text = text.replace(a, b)
    text = text.replace('َا', 'ا').replace('ِي', 'ي').replace('ُو', 'و') 
    text = text.replace('الْ', 'ال')                                       
    text = text.replace('ْ', '')                                          
    text = text.replace('اَ', 'ا').replace('اِ', 'ا').replace('لِا', 'لا')
    text = text.replace('اً', 'ًا')
    return _WS.sub(' ', text).strip()

# lenient -  diacritic-insensitive matching
_NOT_LETTER = re.compile(r'[^ء-ي\s]')
def strip_hadith(text):
    text = standardize_quran(text).replace('ـ', '')  # remove tatweel too
    text = _NOT_LETTER.sub('', text)                 # drop remaining diacritics + punctuation
    return _WS.sub(' ', text).strip()

def normalize(text, seg_type):
    return standardize_quran(text) if seg_type == 'Ayah' else strip_hadith(text)

# --- read prediction -----------------------------------------------------------
print('Reading prediction')
try:
    pred_file = [f for f in os.listdir(prediction_dir) if f.endswith('.tsv')][0]
    print(f'Found prediction file: {pred_file}')
except IndexError:
    raise FileNotFoundError('No prediction file found in the prediction directory. Ensure that there is exactly one file in TSV format.')

print('Contents of prediction directory (Should include your tsv submission):')
print(os.listdir(prediction_dir))
pred_data = pd.read_csv(os.path.join(prediction_dir, pred_file), sep='\t', dtype=str, keep_default_na=False)

for col in ('Response_ID', 'Annotation_ID', 'Correction'):
    if col not in pred_data.columns:
        raise ValueError(f'Prediction file must contain a "{col}" column.')

pred_lookup = {(r['Response_ID'], r['Annotation_ID']): r['Correction']
               for _, r in pred_data.iterrows()}

# --- read reference ------------------------------------------------------------
try:
    ref_file = os.path.join(reference_dir,
        [f for f in os.listdir(reference_dir) if f.lower().endswith('.tsv')][0])
except IndexError:
    raise FileNotFoundError('No reference file found in the reference directory. Contact the organizers if you see this error.')
ref_data = pd.read_csv(ref_file, sep='\t', dtype=str, keep_default_na=False)

# --- score ---------------------------------------------------------------------
print('Checking Accuracy')
per_type = defaultdict(lambda: [0, 0])   # Segment_Type -> [correct, total]
missing = 0
for _, r in ref_data.iterrows():
    seg_type = r['Segment_Type']
    key = (r['Response_ID'], r['Annotation_ID'])
    per_type[seg_type][1] += 1

    gold = {normalize(g, seg_type) for g in r['Correction'].split(SEP) if g.strip()}
    if key not in pred_lookup:
        missing += 1
        continue
    pred = {normalize(p, seg_type) for p in pred_lookup[key].split(SEP) if p.strip()}
    if gold & pred:                       # any acceptable correction matched
        per_type[seg_type][0] += 1

per_type_acc = {t: c / n for t, (c, n) in sorted(per_type.items()) if n > 0}
tot_correct = sum(c for c, _ in per_type.values())
tot_rows = sum(n for _, n in per_type.values())
overall = tot_correct / tot_rows if tot_rows else 0.0   

print('Scores:')
scores = {'accuracy': overall}
scores.update({f'accuracy_{t}': a for t, a in per_type_acc.items()})
print(scores)
if missing:
    print(f'(note: {missing} reference rows had no matching prediction and were scored as wrong)')
os.makedirs(score_dir, exist_ok=True)
with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
