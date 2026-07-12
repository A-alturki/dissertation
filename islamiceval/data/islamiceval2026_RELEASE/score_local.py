#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_local.py — score a Task-3 prediction locally, exactly the way CodaBench does.

It builds the CodaBench directory layout in a temp folder
    <tmp>/input/ref/<gold>_HIDDEN.tsv      (the gold answers)
    <tmp>/input/res/<your prediction>.tsv  (what you submit)
    <tmp>/output/scores.json               (the result)
runs the real task3_scoring.py against it, prints scores.json, then cleans up.

Usage:
    python score_local.py PREDICTION.tsv
    python score_local.py PREDICTION.tsv --gold path/to/gold.tsv

Default gold = ../islamiceval2026_RELEASE_correction/dev_corrected_correction.tsv
Prediction/gold TSV columns: Response_ID, Annotation_ID, Segment_Type, Correction
(gold may list several acceptable corrections in one cell joined by ' ||| ').
"""
import os, sys, json, shutil, subprocess, tempfile, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "task3_scoring.py")
DEFAULT_GOLD = os.path.join(HERE, "..", "islamiceval2026_RELEASE_correction",
                            "dev_corrected_correction.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prediction", help="your prediction TSV")
    ap.add_argument("--gold", default=DEFAULT_GOLD, help="gold TSV (default: dev_corrected_correction.tsv)")
    args = ap.parse_args()

    for label, path in (("prediction", args.prediction), ("gold", args.gold)):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {label} file not found: {path}")

    root = tempfile.mkdtemp(prefix="task3_local_")
    try:
        ref = os.path.join(root, "input", "ref")
        res = os.path.join(root, "input", "res")
        out = os.path.join(root, "output")
        for d in (ref, res, out):
            os.makedirs(d)
        # CodaBench requires the gold to end in *_HIDDEN.tsv, and exactly one tsv in res/
        shutil.copyfile(args.gold, os.path.join(ref, "gold_HIDDEN.tsv"))
        shutil.copyfile(args.prediction, os.path.join(res, os.path.basename(args.prediction)))

        env = dict(os.environ, SCORING_ROOT=root + os.sep)
        r = subprocess.run([sys.executable, SCORER], env=env)
        if r.returncode != 0:
            sys.exit(f"scorer exited with code {r.returncode}")

        scores = json.load(open(os.path.join(out, "scores.json"), encoding="utf-8"))
        print("\n==== scores.json ====")
        print(json.dumps(scores, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
