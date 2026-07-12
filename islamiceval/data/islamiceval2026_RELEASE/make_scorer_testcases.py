# -*- coding: utf-8 -*-
"""
Build an EXHAUSTIVE test bench for task3_scoring.py and self-verify it.

Produces (under <this_dir>/scorer_selftest/):
    input/ref/tests_HIDDEN.tsv   the reference (gold) TSV
    input/res/pred.tsv           the participant prediction TSV
    tests_annotated.tsv          human-readable: case_id | seg | gold | pred | expected | reason
    expected.json                machine expectation (per-case + aggregate)

Then run the REAL scorer:
    SCORING_ROOT=<...>/scorer_selftest python task3_scoring.py
and compare its scores.json to expected.json (see verify_scorer_testcases.py).

Every "correct" case uses REAL diacritized Quran/Hadith text pulled from the gold
(_scorer_test_bases.json); wrong/variant cases are derived programmatically so the
Arabic is never hand-mangled.
"""
import os, re, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))
BASES = json.load(open(os.path.join(HERE, "_scorer_test_bases.json"), encoding="utf-8"))
SEP = " ||| "

SA = BASES["single_ayah"]                       # real diacritized ayah, 1 variant
MA = BASES["multi_ayah"].split(SEP)             # 3 real ayah variants
SM = BASES["single_matn"]                       # real diacritized matn, 1 variant
MM = BASES["multi_matn"].split(SEP)             # 4 real matn variants
KH = "خطأ"

# ---- text transforms (to derive variants) ----------------------------------
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")  # all Arabic marks incl. superscript-alef
def strip_diac(s):   return _DIAC.sub("", s)
def plain_alef(s):
    for a in "أإآٱ": s = s.replace(a, "ا")
    return s
def add_tatweel(s):
    # insert a tatweel after the first Arabic letter of each word (harmless elongation)
    return " ".join((w[:1] + "ـ" + w[1:]) if len(w) > 1 else w for w in s.split(" "))
def add_punct(s):    return "«" + s.replace(" ", "، ", 1) + " .»"
def strip_sukun(s):  return s.replace("ْ", "")
def messy_ws(s):     return "   " + re.sub(r" ", "  ", s) + "  "

# ---- case table: (case_id, seg, gold_cell, pred_cell|None, expected_bool, reason) ----
C = []
def case(cid, seg, gold, pred, exp, reason): C.append((cid, seg, gold, pred, exp, reason))

# ===== QURAN (Ayah) — diacritic-SENSITIVE (except standardized defaults) =====
case("Q01_exact",        "Ayah", SA, SA,                        True,  "identical diacritized ayah")
case("Q02_nodiac",       "Ayah", SA, strip_diac(SA),            False, "Quran is diacritic-sensitive: fully undiacritized must NOT match")
case("Q03_alef_fold",    "Ayah", SA, plain_alef(SA),            True,  "hamza-alef variants folded (أإآٱ->ا) on both sides")
case("Q04_ya_swap",      "Ayah", SA, SA.replace("ي","ى"),       True,  "alef-maqsura/ya folded (ى->ي)")
case("Q05_wrong_verse",  "Ayah", SA, MA[0],                     False, "different verse -> wrong")
case("Q06_pred_khata",   "Ayah", SA, KH,                        False, "declined a real, groundable verse -> wrong")
case("Q07_khata_khata",  "Ayah", KH, KH,                        True,  "gold خطأ, predicted خطأ -> correct")
case("Q08_khata_overreach","Ayah", KH, SA,                      False, "gold خطأ but predicted a verse (over-reach) -> wrong")
case("Q09_multi_first",  "Ayah", SEP.join(MA), MA[0],           True,  "multi-variant gold: match FIRST variant")
case("Q10_multi_second", "Ayah", SEP.join(MA), MA[1],           True,  "multi-variant gold: match SECOND variant")
case("Q11_multi_last",   "Ayah", SEP.join(MA), MA[-1],          True,  "multi-variant gold: match LAST variant")
case("Q12_multi_none",   "Ayah", SEP.join(MA), SA,              False, "multi-variant gold: match NONE -> wrong")
case("Q13_whitespace",   "Ayah", SA, messy_ws(SA),             True,  "leading/trailing/inner whitespace collapsed")
case("Q14_missing",      "Ayah", SA, None,                      False, "no prediction row for this key -> counted wrong (missing)")
case("Q15_pred_multi",   "Ayah", SA, SEP.join([MA[0], SA]),     True,  "prediction is multi-variant; its 2nd part matches gold")
case("Q16_pred_segtype", "Ayah", SA, SA,                        True,  "pred row mislabels Segment_Type=matn; scorer uses REF type -> still correct")
case("Q17_sukun_default","Ayah", SA, strip_sukun(SA),           True,  "removing redundant sukun tolerated by standardization")

# ===== HADITH (matn) — diacritic-INSENSITIVE (letters only) =================
case("H01_exact",        "matn", SM, SM,                        True,  "identical diacritized matn")
case("H02_nodiac_pred",  "matn", SM, strip_diac(SM),            True,  "hadith lenient: undiacritized prediction matches diacritized gold")
case("H03_nodiac_gold",  "matn", strip_diac(SM), SM,            True,  "hadith lenient: diacritized prediction matches undiacritized gold")
case("H04_punct",        "matn", SM, add_punct(SM),             True,  "punctuation (« » ، .) stripped for hadith")
case("H05_tatweel",      "matn", SM, add_tatweel(SM),           True,  "tatweel (ـ) removed for hadith")
case("H06_wrong_matn",   "matn", SM, MM[0],                     False, "different matn -> wrong")
case("H07_khata_khata",  "matn", KH, KH,                        True,  "gold خطأ, predicted خطأ -> correct")
case("H08_khata_overreach","matn", KH, SM,                      False, "gold خطأ but predicted a matn -> wrong")
case("H09_multi_first",  "matn", SEP.join(MM), MM[0],           True,  "multi-variant gold: match FIRST variant")
case("H10_multi_second", "matn", SEP.join(MM), MM[1],           True,  "multi-variant gold: match SECOND variant")
case("H11_multi_last",   "matn", SEP.join(MM), MM[-1],          True,  "multi-variant gold: match LAST variant")
case("H12_multi_none",   "matn", SEP.join(MM), SM,              False, "multi-variant gold: match NONE -> wrong")
case("H13_pred_multi_last","matn", SM, SEP.join([MM[0], SM]),   True,  "prediction multi-variant; its LAST part matches gold")
case("H14_alef_fold",    "matn", SM, plain_alef(SM),            True,  "hamza-alef folded for hadith too")
case("H15_missing",      "matn", SM, None,                      False, "no prediction row -> counted wrong (missing)")
case("H16_all_lenient",  "matn", SM, add_punct(add_tatweel(strip_diac(SM))), True, "combined: undiacritized+tatweel+punct still matches")

# ===== EDGE: duplicate prediction key (documents silent last-wins collapse) ==
# ref has ONE row; pred file will carry TWO rows with the same key (correct then wrong).
# Scorer's dict keeps the LAST -> a correct answer is clobbered -> WRONG.
case("E01_dupkey_lastwins","Ayah", SA, "<DUP>",                 False, "duplicate prediction key: scorer keeps LAST row (wrong) -> correct answer lost")

# ---- assign unique keys & write files --------------------------------------
os.makedirs(os.path.join(HERE, "scorer_selftest", "input", "ref"), exist_ok=True)
os.makedirs(os.path.join(HERE, "scorer_selftest", "input", "res"), exist_ok=True)
os.makedirs(os.path.join(HERE, "scorer_selftest", "output"), exist_ok=True)

ref_rows, pred_rows, ann_rows, expected = [], [], [], {}
for i, (cid, seg, gold, pred, exp, reason) in enumerate(C, 1):
    rid = "R9%05d" % i
    aid = "1"
    ref_rows.append([rid, aid, seg, gold])
    # prediction handling
    if cid == "E01_dupkey_lastwins":
        pred_rows.append([rid, aid, seg, SA])   # first (correct)
        pred_rows.append([rid, aid, seg, MA[0]])# last (wrong) -> wins
    elif cid == "Q16_pred_segtype":
        pred_rows.append([rid, aid, "matn", pred])   # deliberately wrong Segment_Type in pred
    elif pred is not None:
        pred_rows.append([rid, aid, seg, pred])
    # (Q14/H15 missing: no pred row emitted)
    expected[cid] = {"key": [rid, aid], "seg": seg, "expected": exp, "reason": reason}
    ann_rows.append([cid, seg, gold, ("<missing>" if pred is None else pred), "CORRECT" if exp else "WRONG", reason])

# add one extra prediction row with NO matching ref key (must be ignored)
pred_rows.append(["R999999", "1", "Ayah", SA])

def wtsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header); w.writerows(rows)

root = os.path.join(HERE, "scorer_selftest")
wtsv(os.path.join(root, "input", "ref", "tests_HIDDEN.tsv"),
     ["Response_ID", "Annotation_ID", "Segment_Type", "Correction"], ref_rows)
wtsv(os.path.join(root, "input", "res", "pred.tsv"),
     ["Response_ID", "Annotation_ID", "Segment_Type", "Correction"], pred_rows)
wtsv(os.path.join(root, "tests_annotated.tsv"),
     ["Case", "Segment_Type", "Gold", "Prediction", "Expected", "Reason"], ann_rows)

# aggregate expectation (mirror the scorer: overall=micro, per-type)
per = {"Ayah": [0, 0], "matn": [0, 0]}
for cid, meta in expected.items():
    per[meta["seg"]][1] += 1
    if meta["expected"]:
        per[meta["seg"]][0] += 1
agg = {
    "accuracy": round(sum(c for c, _ in per.values()) / sum(n for _, n in per.values()), 6),
    "accuracy_Ayah": round(per["Ayah"][0] / per["Ayah"][1], 6),
    "accuracy_matn": round(per["matn"][0] / per["matn"][1], 6),
    "n_Ayah": per["Ayah"][1], "n_matn": per["matn"][1],
}
json.dump({"cases": expected, "aggregate": agg}, open(os.path.join(root, "expected.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("wrote %d ref rows, %d pred rows, %d cases" % (len(ref_rows), len(pred_rows), len(C)))
print("expected aggregate:", json.dumps(agg, ensure_ascii=False))
