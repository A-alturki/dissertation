import json
import os
import pandas as pd
from sklearn.metrics import f1_score

'''
1- Place the reference JSONL (responses) and a tsv of the expected solution in {reference_dir}
2- Place your prediction TSV in {prediction_dir}

Spans are character offsets into generated_answer, end-exclusive.
'''

ROOT_DIR = '/app/'
reference_dir = os.path.join(ROOT_DIR, 'input/', 'ref')
prediction_dir = os.path.join(ROOT_DIR, 'input/', 'res')
score_dir = os.path.join(ROOT_DIR, 'output/')

CLASSES = {'neither': 0, 'Ayah': 1, 'matn': 2, 'isnad': 3, 'claimed_source': 4}

print('Reading reference')

try:
    resp_file = os.path.join(reference_dir,
        [f for f in os.listdir(reference_dir) if f.endswith('.jsonl')][0])
except IndexError:
    raise FileNotFoundError('No reference JSONL (responses) found in the reference directory. Contact the organizers if you see this error.')

qid_response_mapping = {}
with open(resp_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            qid_response_mapping[rec['id']] = rec['generated_answer']

try:
    ref_file = os.path.join(reference_dir,
        [f for f in os.listdir(reference_dir) if f.endswith('HIDDEN.tsv')][0])
except IndexError:
    raise FileNotFoundError('No reference file found in the reference directory. Contact the organizers if you see this error.')
ref_data = pd.read_csv(ref_file, sep='\t', dtype=str, keep_default_na=False)

print('Reading prediction')
try:
    pred_file = [f for f in os.listdir(prediction_dir) if f.endswith('.tsv')][0]
    print(f'Found prediction file: {pred_file}')
except IndexError:
    raise FileNotFoundError('No prediction file found in the prediction directory. Ensure that there is exactly one file in TSV format.')

print('Contents of prediction directory (Should include your tsv submission):')
print(os.listdir(prediction_dir))
pred_data = pd.read_csv(os.path.join(prediction_dir, pred_file), sep='\t', dtype=str, keep_default_na=False)

allowed = set(CLASSES.keys()) | {'NoAnnotation'}
if any(t not in allowed for t in pred_data['Segment_Type'].values):
    raise ValueError('Prediction file "Segment_Type" column must contain only '
                     'Ayah, matn, isnad, claimed_source, or NoAnnotation.')

def paint(rows, L):
    arr = [0] * L
    for _, r in rows.iterrows():
        typ = r['Segment_Type']
        if typ == 'NoAnnotation' or str(r['Span_Start']) in ('-', '', 'nan'):
            continue
        s, e = int(r['Span_Start']), int(r['Span_End'])
        cls = CLASSES.get(typ, 0)
        for i in range(s, min(e, L)):
            arr[i] = cls
    return arr

print('Calculating F1 Score')
g_all, p_all = [], []
for rid, text in qid_response_mapping.items():
    L = len(text)
    g_all.extend(paint(ref_data[ref_data['Response_ID'] == rid], L))
    p_all.extend(paint(pred_data[pred_data['Response_ID'] == rid], L))

labels = list(CLASSES.values())
f1_score_value = f1_score(g_all, p_all, average='macro', labels=labels, zero_division=0)
print(f'F1 Score: {f1_score_value}')

scores = {'F1 Score': f1_score_value}
print(scores)
os.makedirs(score_dir, exist_ok=True)
with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
