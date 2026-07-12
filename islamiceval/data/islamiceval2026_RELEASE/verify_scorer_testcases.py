# -*- coding: utf-8 -*-
"""
Verify task3_scoring.py against the exhaustive bench built by make_scorer_testcases.py.

Two independent checks:
  1) PER-CASE: replay the scorer's OWN normalize()/matching per (Response_ID,Annotation_ID)
     and assert the correct/wrong decision equals the hand-authored `expected`.
  2) END-TO-END: run the real task3_scoring.py via SCORING_ROOT and assert its
     scores.json equals the hand-computed aggregate.
Prints a per-case table; exits non-zero if anything disagrees.
"""
import os, sys, io, json, subprocess, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "scorer_selftest")

# import normalize() etc. from the scorer without executing its I/O body
src = open(os.path.join(HERE, "task3_scoring.py"), encoding="utf-8").read()
ns = {}
exec(src.split("# --- read prediction")[0], ns)
normalize, SEP = ns["normalize"], ns["SEP"]

import pandas as pd
ref = pd.read_csv(os.path.join(ROOT, "input", "ref", "tests_HIDDEN.tsv"), sep="\t", dtype=str, keep_default_na=False)
pred = pd.read_csv(os.path.join(ROOT, "input", "res", "pred.tsv"), sep="\t", dtype=str, keep_default_na=False)
# mirror the scorer's lookup (dict keeps LAST on dup key)
plook = {(r["Response_ID"], r["Annotation_ID"]): r["Correction"] for _, r in pred.iterrows()}
exp = json.load(open(os.path.join(ROOT, "expected.json"), encoding="utf-8"))

def decide(seg, gold_cell, key):
    gold = {normalize(g, seg) for g in gold_cell.split(SEP) if g.strip()}
    if key not in plook:
        return False
    pr = {normalize(p, seg) for p in plook[key].split(SEP) if p.strip()}
    return bool(gold & pr)

# ---- 1) per-case ----
refmap = {(r["Response_ID"], r["Annotation_ID"]): (r["Segment_Type"], r["Correction"]) for _, r in ref.iterrows()}
print("%-24s %-5s %-8s %-8s %s" % ("CASE", "SEG", "EXPECT", "ACTUAL", "RESULT"))
print("-" * 78)
fails = 0
for cid, meta in exp["cases"].items():
    key = tuple(meta["key"]); seg, gold_cell = refmap[key]
    actual = decide(seg, gold_cell, key)
    ok = (actual == meta["expected"])
    fails += (not ok)
    print("%-24s %-5s %-8s %-8s %s" % (
        cid, seg, "CORRECT" if meta["expected"] else "WRONG",
        "CORRECT" if actual else "WRONG", "PASS" if ok else "*** FAIL ***"))

# ---- 2) end-to-end real scorer ----
env = dict(os.environ, SCORING_ROOT=ROOT, PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(HERE, "task3_scoring.py")], env=env, check=True,
               stdout=subprocess.DEVNULL)
got = json.load(open(os.path.join(ROOT, "output", "scores.json"), encoding="utf-8"))
want = exp["aggregate"]
print("\nEND-TO-END scores.json vs expected aggregate:")
agg_ok = True
for k in ("accuracy", "accuracy_Ayah", "accuracy_matn"):
    match = round(got.get(k, -1), 6) == round(want[k], 6)
    agg_ok &= match
    print("  %-16s got=%.6f  want=%.6f  %s" % (k, got.get(k, -1), want[k], "PASS" if match else "*** FAIL ***"))

print("\nPER-CASE: %d/%d passed | AGGREGATE: %s" %
      (len(exp["cases"]) - fails, len(exp["cases"]), "PASS" if agg_ok else "FAIL"))
sys.exit(1 if (fails or not agg_ok) else 0)
