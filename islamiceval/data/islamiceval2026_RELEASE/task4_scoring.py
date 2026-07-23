#!/usr/bin/env python3
"""Evaluate submitted predications to Task 4 - IslamicEval 2026 """

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

'''
IslamicEval 2026 - Task 4 (Labelling) scorer.

1- Place a TSV of the expected solutions in {ref} directory (or pass --ref)
2- Place your prediction TSV in {res} directory (or pass --pred)

Prediction TSV columns (tab-separated, WITH header):
    question_id   response_id   Annotation_ID   span_type   span_text   relevance_label

Given a user question and an LLM response with its correctly extracted Qur'anic and Hadith citation spans,
the task is to determine whether each citation span is relevant to answering the question.
The question_id, Response_ID, and Annotation_ID fields link to the main JSONL file, which contains the
full text of the user's question and the LLM's response.
span_type identifies the citation as either an Ayah or a matn.
span_text provides the correct text of the citation.
The task is to predict the relevance_label for each span using a binary classification:
 Relevant (1) or Non-relevant (0).

Refer to the detailed relevance label definitions published at Subtask 4 webpage https://sites.google.com/view/islamiceval2026/subtask-4

Output: a single scores.json with three top-level keys:
    "Macro-averaged F1"        : float, the macro-averaged F1 across questions
    "per_question_score"       : dict keyed by question_id, each value holding
                                  that question's Response_ID(s), span_count,
                                  tp/fp/fn/tn, precision, recall, and F1 Score

Usage:
    python task4_scoring.py
        (uses the default competition directories below)
    OR
    python task4_scoring.py --ref gold.tsv --pred my_preds.tsv --output ./out
        (uses explicit files/dirs instead)
'''

ROOT_DIR = Path("/app/")
DEFAULT_REFERENCE_DIR = ROOT_DIR / "input" / "ref"
DEFAULT_PREDICTION_DIR = ROOT_DIR / "input" / "res"
DEFAULT_SCORE_DIR = ROOT_DIR / "output"

EXPECTED_COLUMNS = [
    "question_id",
    "Response_ID",
    "Annotation_ID",
    "span_type",
    "span_text",
    "relevance_label",
]
KEY_COLUMNS = ("question_id", "Response_ID", "Annotation_ID")


def binary_label(value):
    value = value.strip()
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0", "2", "2.0"}:
        return 1
    raise ValueError(
        f"unsupported relevance_label {value!r}; expected 0, 1, or 2"
    )


def validate_tsv(path, *, allow_empty=False):
    if path.suffix.lower() != ".tsv":
        raise ValueError(f"{path}: input must have a .tsv extension")
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: input is not valid UTF-8") from error

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: columns must be exactly, in order: "
                + ", ".join(EXPECTED_COLUMNS)
            )

        records, seen = [], set()
        for row_number, row in enumerate(reader, 2):
            if None in row or any(row[column] is None for column in EXPECTED_COLUMNS):
                raise ValueError(
                    f"{path}:{row_number}: incorrect number of TSV columns"
                )
            key = tuple(row[column].strip() for column in KEY_COLUMNS)
            if any(not value for value in key):
                raise ValueError(f"{path}:{row_number}: blank identifier value")
            if key in seen:
                raise ValueError(
                    f"{path}:{row_number}: duplicate key {'/'.join(key)}"
                )
            seen.add(key)
            record = {
                column: row[column].strip() for column in EXPECTED_COLUMNS
            }
            try:
                record["binary_label"] = binary_label(row["relevance_label"])
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: {error}") from error
            records.append(record)

    if not records and not allow_empty:
        raise ValueError(f"{path}: no data rows")
    return records


def row_key(record):
    return tuple(str(record[column]) for column in KEY_COLUMNS)


def align_rows(gold, predictions):
    """Align rows while retaining wholly missing prediction questions.

    Every row of a question may be omitted from the prediction file; that
    question is passed to the scorer with ``None`` predictions and receives
    F1 = 0. A partially omitted question remains an input error because its
    row-level alignment is ambiguous/incomplete. Extra prediction rows also
    remain errors.
    """
    gold_by_key = {row_key(row): row for row in gold}
    prediction_by_key = {row_key(row): row for row in predictions}
    gold_keys = set(gold_by_key)
    prediction_keys = set(prediction_by_key)
    missing = sorted(gold_keys - prediction_keys)
    extra = sorted(prediction_keys - gold_keys)
    prediction_question_ids = {key[0] for key in prediction_keys}
    partial_missing = [
        key for key in missing if key[0] in prediction_question_ids
    ]

    if partial_missing or extra:
        details = []
        if partial_missing:
            details.append(
                f"missing {len(partial_missing)} prediction row(s) from a "
                f"partially submitted question, e.g. {'/'.join(partial_missing[0])}"
            )
        if extra:
            details.append(
                f"extra {len(extra)} prediction row(s), e.g. {'/'.join(extra[0])}"
            )
        raise ValueError("; ".join(details))

    return [
        (gold_by_key[key], prediction_by_key.get(key))
        for key in sorted(gold_by_key)
    ]


def question_scores(pairs):
    grouped = defaultdict(list)
    for gold, prediction in pairs:
        grouped[str(gold["question_id"])].append((gold, prediction))

    results, no_relevant_count = [], 0
    for question_id in sorted(grouped):
        question_pairs = grouped[question_id]
        prediction_missing = all(
            prediction is None for _, prediction in question_pairs
        )
        counts = Counter()

        for gold, prediction in question_pairs:
            actual = int(gold["binary_label"])
            # Missing questions are counted as having no positive predictions,
            # then explicitly assigned F1 = 0 below in all gold-label cases.
            predicted = 0 if prediction is None else int(prediction["binary_label"])
            if actual == 1 and predicted == 1:
                counts["tp"] += 1
            elif actual == 0 and predicted == 1:
                counts["fp"] += 1
            elif actual == 1 and predicted == 0:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

        tp, fp, fn, tn = (counts[name] for name in ("tp", "fp", "fn", "tn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        if tp + fn == 0:
            no_relevant_count += 1

        if prediction_missing:
            f1 = 0.0
        elif tp + fn == 0:
            f1 = 1.0 if fp == 0 else 0.0
        else:
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )

        results.append(
            {
                "question_id": question_id,
                "Response_ID": ", ".join(
                    sorted(
                        {
                            str(gold["Response_ID"])
                            for gold, _ in question_pairs
                        }
                    )
                ),
                "span_count": len(question_pairs),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "F1 Score": f1,
            }
        )
    return results, no_relevant_count


def evaluate(gold_path, prediction_path):
    """Read and score a submission without writing output files."""
    gold = validate_tsv(gold_path)
    predictions = validate_tsv(prediction_path, allow_empty=True)
    rows, no_relevant_count = question_scores(align_rows(gold, predictions))
    macro_f1 = sum(float(row["F1 Score"]) for row in rows) / len(rows)
    return macro_f1, no_relevant_count, rows


def save_scores(macro_f1, no_relevant_count, rows, out_dir):
    """Write the consolidated scores.json file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_question_score = {
        row["question_id"]: {
            key: value for key, value in row.items() if key != "question_id"
        }
        for row in rows
    }
    scores = {
        "Macro-averaged F1": round(macro_f1, 6),
        "per_question_score": per_question_score,
    }
    scores_path = out_dir / "scores.json"
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2)
    return scores_path


def resolve_tsv_path(path, label):
    """Resolve a direct TSV path or a directory containing TSV files."""
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if path.is_file():
        return path

    candidates = sorted(
        candidate for candidate in path.iterdir()
        if candidate.suffix.lower() == ".tsv"
    )
    if not candidates:
        raise FileNotFoundError(
            f'No {label.lower()} file found in "{path}". '
            "Ensure there is exactly one .tsv file."
        )
    if len(candidates) > 1:
        print(
            f'WARNING: multiple .tsv files found in "{path}": '
            f"{[candidate.name for candidate in candidates]}. "
            f'Using "{candidates[0].name}". '
            "Pass an explicit file path to avoid ambiguity."
        )
    print(f"Found {label.lower()} file: {candidates[0].name}")
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref",
        "-r",
        type=Path,
        default=None,
        help="Gold TSV file, or a directory containing a .tsv file "
        f"(default: {DEFAULT_REFERENCE_DIR})",
    )
    parser.add_argument(
        "--pred",
        "-p",
        type=Path,
        default=None,
        help="Prediction TSV file, or a directory containing a .tsv file "
        f"(default: {DEFAULT_PREDICTION_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Directory to write scores.json into "
        f"(default: {DEFAULT_SCORE_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print the computed macro F1 and output path.",
    )
    args = parser.parse_args(argv)

    gold_path = args.ref if args.ref is not None else DEFAULT_REFERENCE_DIR
    pred_path = args.pred if args.pred is not None else DEFAULT_PREDICTION_DIR
    out_dir = args.output if args.output is not None else DEFAULT_SCORE_DIR

    try:
        gold_file = resolve_tsv_path(gold_path, "Gold")
        pred_file = resolve_tsv_path(pred_path, "Prediction")
        macro_f1, no_relevant_count, rows = evaluate(gold_file, pred_file)
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error

    scores_path = save_scores(macro_f1, no_relevant_count, rows, out_dir)
    if args.verbose:
        print(f"Macro-averaged F1: {macro_f1:.6f}")
        print(f"Saved scores to: {scores_path}")
    else:
        print("Submission processed and scored successfully.")
    return macro_f1

if __name__ == "__main__":
    main()
