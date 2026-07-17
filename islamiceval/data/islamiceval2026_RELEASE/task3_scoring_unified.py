import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd

'''
IslamicEval 2026 - Task 3 (Correction) scorer.

Input files:
1- Place a TSV of the expected solutions in {ref} directory (or pass it by
   command line using --ref)
2- Place your prediction TSV in {res} directory (or pass it via --pred)
3- Place {ref} and {res} directories in a directory named {input}

Both TSVs use these columns:
    Response_ID, Annotation_ID, Segment_Type, Correction
  - Segment_Type is the citation type: "Ayah" (Quran) or "matn" (Hadith).
  - Correction is the corrected canonical text (the full Uthmani ayah for Quran,
    the matn for Hadith), or the string "خطأ" when no faithful correction is possible.
  - In the REFERENCE only, the Correction cell may list several acceptable
    corrections separated by " ||| " (e.g. the same hadith attested across books,
    or a verse quoted at more than one length). A prediction is counted CORRECT if
    it matches ANY one of them.

Quran is diacritic sensitive, and hadith is diacritic insensitive.

A prediction matches if it equals ANY reference correction.
Reported scores:
  - accuracy       : overall accuracy over all rows
  - accuracy_Ayah  : accuracy on Quran rows only.
  - accuracy_matn  : accuracy on Hadith rows only.

Usage:
    python task3_scoring.py
        (uses the default competition directories below)
    OR
    python task3_scoring.py --pred my_preds.tsv --ref gold.tsv --output ./out
        (uses explicit files/dirs instead)
'''

ROOT_DIR = '/app/'
DEFAULT_REFERENCE_DIR = os.path.join(ROOT_DIR, 'input/', 'ref')
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, 'input/', 'res')
DEFAULT_SCORE_DIR = os.path.join(ROOT_DIR, 'output/')

REQUIRED_COLUMNS = ('Response_ID', 'Annotation_ID', 'Segment_Type', 'Correction')
KEY_COLUMNS = ['Response_ID', 'Annotation_ID']
VALID_SEGMENT_TYPES = {'Ayah', 'matn'}

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


# lenient - diacritic-insensitive matching
_NOT_LETTER = re.compile(r'[^ء-ي\s]')


def strip_hadith(text):
    text = standardize_quran(text).replace('ـ', '')  # remove tatweel too
    text = _NOT_LETTER.sub('', text)                  # drop remaining diacritics + punctuation
    return _WS.sub(' ', text).strip()


def normalize(text, seg_type):
    return standardize_quran(text) if seg_type == 'Ayah' else strip_hadith(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description='IslamicEval 2026 - Task 3 (Correction) scorer.'
    )
    parser.add_argument(
        '--pred', '-p', default=None,
        help=f'Prediction TSV file, or a directory containing exactly one '
             f'.tsv file (default: {DEFAULT_PREDICTION_DIR})'
    )
    parser.add_argument(
        '--ref', '-r', default=None,
        help=f'Reference TSV file, or a directory containing exactly one '
             f'.tsv file (default: {DEFAULT_REFERENCE_DIR})'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help=f'Directory to write scores.json into (default: {DEFAULT_SCORE_DIR})'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help='Print the computed scores to stdout. Use this flag if you want to print output on stdout'
    )
    return parser.parse_args()


def resolve_tsv_file(path, label):
    """Accept either a direct file path or a directory containing a single
    (or several, in which case the alphabetically-first is used) .tsv file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} path does not exist: {path}')

    if os.path.isfile(path):
        return path

    candidates = sorted(f for f in os.listdir(path) if f.lower().endswith('.tsv'))
    if not candidates:
        raise FileNotFoundError(
            f'No {label.lower()} file found in "{path}". '
            f'Ensure there is exactly one file in TSV format.'
        )
    if len(candidates) > 1:
        print(
            f'WARNING: multiple .tsv files found in "{path}": {candidates}. '
            f'Using "{candidates[0]}". Pass an explicit file path to avoid ambiguity.'
        )
    chosen = candidates[0]
    print(f'Found {label.lower()} file: {chosen}')
    return os.path.join(path, chosen)


def load_tsv(path):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def validate_columns(df, name):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{name} file must contain column(s): {", ".join(missing)}.')


def validate_segment_types(df, name):
    bad_rows = df.index[~df['Segment_Type'].isin(VALID_SEGMENT_TYPES)].tolist()
    if bad_rows:
        bad_values = df.loc[bad_rows, 'Segment_Type'].unique().tolist()
        raise ValueError(
            f'{name} file "Segment_Type" column must contain only '
            f'{", ".join(sorted(VALID_SEGMENT_TYPES))}. '
            f'Found invalid value(s) {bad_values} at row(s) {bad_rows}.'
        )


def build_pred_lookup(pred_data):
    lookup = {}
    duplicates = []
    for _, r in pred_data.iterrows():
        k = tuple(r[c] for c in KEY_COLUMNS)
        if k in lookup:
            duplicates.append(k)
        lookup[k] = r['Correction']
    if duplicates:
        raise ValueError(
            f'Prediction file contains duplicate rows for the same '
            f'(Response_ID, Annotation_ID) key(s): {duplicates[:10]}'
            f'{" ..." if len(duplicates) > 10 else ""}. Each combination must '
            f'appear exactly once.'
        )
    return lookup


def score(pred_data, ref_data, verbose=False):
    validate_columns(pred_data, 'Prediction')
    validate_columns(ref_data, 'Reference')
    validate_segment_types(ref_data, 'Reference')

    pred_lookup = build_pred_lookup(pred_data)

    per_type = defaultdict(lambda: [0, 0])   # Segment_Type -> [correct, total]
    missing_total = 0
    missing_per_type = defaultdict(int)

    for _, r in ref_data.iterrows():
        seg_type = r['Segment_Type']
        key = tuple(r[c] for c in KEY_COLUMNS)
        per_type[seg_type][1] += 1

        gold = {normalize(g, seg_type) for g in r['Correction'].split(SEP) if g.strip()}
        if key not in pred_lookup:
            missing_total += 1
            missing_per_type[seg_type] += 1
            continue
        pred = {normalize(p, seg_type) for p in pred_lookup[key].split(SEP) if p.strip()}
        if gold & pred:  # any acceptable correction matched
            per_type[seg_type][0] += 1

    per_type_acc = {t: c / n for t, (c, n) in sorted(per_type.items()) if n > 0}
    tot_correct = sum(c for c, _ in per_type.values())
    tot_rows = sum(n for _, n in per_type.values())
    overall = tot_correct / tot_rows if tot_rows else 0.0

    scores = {'accuracy': overall}
    scores.update({f'accuracy_{t}': a for t, a in per_type_acc.items()})
    scores['missing_predictions'] = missing_total
    scores.update({
        f'missing_predictions_{t}': n for t, n in sorted(missing_per_type.items())
    })

    if missing_total and verbose:
        print(
            f'WARNING: {missing_total} reference row(s) had no matching prediction '
            f'and were scored as wrong: {dict(sorted(missing_per_type.items()))}'
        )

    return scores


def main():
    args = parse_args()

    pred_path = args.pred if args.pred is not None else DEFAULT_PREDICTION_DIR
    ref_path = args.ref if args.ref is not None else DEFAULT_REFERENCE_DIR
    score_dir = args.output if args.output is not None else DEFAULT_SCORE_DIR

    print('Reading prediction')
    pred_file = resolve_tsv_file(pred_path, 'Prediction')
    pred_data = load_tsv(pred_file)

    print('Reading reference')
    ref_file = resolve_tsv_file(ref_path, 'Reference')
    ref_data = load_tsv(ref_file)

    print('Checking Accuracy')
    scores = score(pred_data, ref_data, verbose=args.verbose)

    os.makedirs(score_dir, exist_ok=True)
    out_path = os.path.join(score_dir, 'scores.json')
    with open(out_path, 'w') as score_file:
        score_file.write(json.dumps(scores, indent=2))

    if args.verbose:
        print('Scores:')
        print(scores)
    else:
        print('Submission processed and scored successfully.')


if __name__ == '__main__':
    main()
