import json
import os
import pandas as pd

'''
1- Place a TSV of the expected solutions in {reference_dir}
2- Place your prediction TSV in {prediction_dir}
'''

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')
try:
    pred_file = [f for f in os.listdir(prediction_dir) if f.endswith('.tsv')][0]
    print(f'Found prediction file: {pred_file}')
except IndexError:
    raise FileNotFoundError('No prediction file found in the prediction directory. Ensure that there is exactly one file in TSV format.')

print('Contents of prediction directory (Should include your tsv submission):')
print(os.listdir(prediction_dir))
pred_data = pd.read_csv(os.path.join(prediction_dir, pred_file), sep='\t', dtype=str, keep_default_na=False)

try:
    ref_file = os.path.join(reference_dir,
        [f for f in os.listdir(reference_dir) if f.endswith('HIDDEN.tsv')][0])
except IndexError:
    raise FileNotFoundError('No reference file found in the reference directory. Contact the organizers if you see this error.')
ref_data = pd.read_csv(ref_file, sep='\t', dtype=str, keep_default_na=False)

for col in ('Response_ID', 'Annotation_ID', 'Segment_Type', 'Label'):
    if col not in pred_data.columns:
        raise ValueError(f'Prediction file must contain a "{col}" column.')

if any(v not in ['correct', 'incorrect'] for v in pred_data['Label'].values):
    raise ValueError('Prediction file "Label" column must contain only "correct" or "incorrect".')

key = ['Response_ID', 'Annotation_ID', 'Segment_Type']
pred_lookup = {tuple(r[k] for k in key): r['Label'] for _, r in pred_data.iterrows()}

# exclude gold N/A
ref_scored = ref_data[ref_data['Label'] != 'N/A']

print('Checking Accuracy')
from collections import defaultdict
per_type = defaultdict(lambda: [0, 0])
missing = 0
for _, r in ref_scored.iterrows():
    k = tuple(r[c] for c in key)
    per_type[r['Segment_Type']][1] += 1
    if k in pred_lookup:
        if pred_lookup[k] == r['Label']:
            per_type[r['Segment_Type']][0] += 1
    else:
        missing += 1

per_type_acc = {t: c / n for t, (c, n) in sorted(per_type.items()) if n > 0}
macro = sum(per_type_acc.values()) / len(per_type_acc) if per_type_acc else 0.0

print('Scores:')
scores = {'accuracy': macro}
scores.update({f'accuracy_{t}': a for t, a in per_type_acc.items()})
print(scores)
os.makedirs(score_dir, exist_ok=True)
with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
