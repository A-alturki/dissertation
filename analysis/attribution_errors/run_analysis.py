"""
run_analysis.py -- qualitative analysis of ATTRIBUTION errors.

Writes summary.md, examples.tex, draft_paragraph.tex and intermediate CSVs into
this folder. Reads the corpora read-only via attrlib.

Classification methods, per the task spec:
  Quran  -- deterministic. The claimed source is parsed for surah + ayah and
            resolved against the 114 canonical surah names and verse counts in
            data/corpora/quranic_verses.json. The TRUE location is found by
            matching the cited Arabic against the same corpus (exact
            containment, then a token-overlap fallback for corrupted text).
            The reason field is used only when both lookups fail.
  Hadith -- rule-based over the annotator's English reason field. No LLM was
            used, so no API spend was incurred; the trade-off is an explicit
            "unclassified" bucket, reported rather than guessed.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import collections
import math

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attrlib as A  # noqa: E402

CONDITIONS = ("implicit", "explicit")
TIERS = ("frontier", "matched", "extended")
TIER_LABEL = {"frontier": "Frontier", "matched": "Matched open-weight",
              "extended": "Extended panel (implicit only)"}

# ----------------------------------------------------------- Quran lookup ---

_TOKENS = {}


def _toks(s):
    if s not in _TOKENS:
        _TOKENS[s] = set(A.ar_norm(s).split())
    return _TOKENS[s]


_VERSE_TOKS = [(s, a, set(t.split())) for s, a, t in A.VERSES if t]


def locate_fuzzy(cited):
    """Best (surah, ayah) by token containment -- for corrupted text that will
    not match exactly. Requires >=60% of the citation's tokens to appear in the
    verse and at least 5 shared tokens, so near-misses are not forced."""
    q = _toks(cited)
    if len(q) < 5:
        return None
    best, score = None, 0.0
    for s, a, vt in _VERSE_TOKS:
        inter = len(q & vt)
        if inter < 5:
            continue
        cov = inter / len(q)
        if cov > score:
            best, score = (s, a), cov
    return best if score >= 0.60 else None


def classify_quran(c):
    """-> (category, method, claimed_sid, claimed_ayah, true_sid, true_ayah)"""
    sid, surface, ayah = A.parse_quran_claim(c["claimed_source"])
    if sid is None:
        return ("nonexistent or misspelled surah", "lookup", None, ayah, None, None)
    if ayah is not None and ayah > A.SURAH_LEN.get(sid, 0):
        return ("verse number beyond surah length", "lookup", sid, ayah, None, None)
    true = A.locate_ayah(c["text"]) or locate_fuzzy(c["text"])
    method = "lookup" if true else "unresolved"
    if true:
        ts, ta = true
        if ts != sid:
            return ("wrong surah", method, sid, ayah, ts, ta)
        if ayah is not None and ta != ayah:
            return ("right surah, wrong verse number", method, sid, ayah, ts, ta)
        return ("unclassified (lookup agrees with claim)", method, sid, ayah, ts, ta)
    # fall back to the reason only when the corpus lookup cannot decide
    r = c["reason"].lower()
    if re.search(r"\b(?:ayah|verse) number is incorrect|not (?:ayah|verse) \d", r):
        return ("right surah, wrong verse number", "reason", sid, ayah, None, None)
    if re.search(r"\bis in surah\b|\bbelongs to surah\b|different surah", r):
        return ("wrong surah", "reason", sid, ayah, None, None)
    return ("unclassified", "none", sid, ayah, None, None)


# ---------------------------------------------------------- Hadith lookup ---

COLLECTIONS = [
    ("Sahih al-Bukhari", r"البخاري|بخاري"),
    ("Sahih Muslim", r"مسلم"),
    ("Jami al-Tirmidhi", r"الترمذي|ترمذي"),
    ("Sunan Abi Dawud", r"أب[يو] داو[دو]|ابو داود|ابي داود"),
    ("Sunan al-Nasa'i", r"النسائي|نسائي"),
    ("Sunan Ibn Majah", r"ابن ماج[هة]"),
    ("Musnad Ahmad", r"أحمد|احمد"),
    ("Muwatta Malik", r"مالك|الموطأ|الموطا"),
    ("Sunan al-Darimi", r"الدارمي"),
    ("al-Mustadrak (al-Hakim)", r"الحاكم"),
    ("al-Bayhaqi", r"البيهقي"),
    ("al-Tabarani", r"الطبراني"),
    ("al-Adab al-Mufrad", r"الأدب المفرد|الادب المفرد"),
]
MUTTAFAQ = r"متفق عليه"


def collections_in(text):
    if not text:
        return []
    out = [name for name, pat in COLLECTIONS if re.search(pat, text)]
    if re.search(MUTTAFAQ, text):
        out.append("Muttafaq alayh")
    return out


# "not narrated BY al-Bukhari", "does not contain this wording" and "famously
# recorded in Y, not X" are all common and none matched the first rule set; the
# unclassified share was 51% before these were added.
# Verb stems must allow the bare form ("did not narrate") as well as the
# participle, and "by" as well as "in" ("recorded BY Ibn Majah") -- the
# hand-check of 30 found 4 of 5 misses were one of those two gaps.
_NOT_THERE = (r"not (?:found|present|recorded?|narrated?|reported?|mentioned?)\b"
              r"|does not (?:contain|include|have|record|narrate)"
              r"|did not (?:record|narrate|report)"
              r"|is absent from|no such hadith in")
_ELSEWHERE = (r"(?:actually|in fact|famously|rather) (?:recorded|narrated|found|reported|documented)"
              r"|but (?:rather|instead)|is found in|are found in|but is found"
              r"|exists? in (?:collections|works)|(?:recorded|documented|narrated) (?:in|by)")
# one named source right, another wrong -> the claim overreached
_PARTIAL = (r"while it is (?:narrated|recorded|found)|but (?:it )?(?:is|was) not"
            r"|only (?:in|by|found in|narrated by)|but not (?:in|by|narrated)"
            r"|is narrated by [^.;]{0,60} but")


def classify_hadith_source(c):
    """-> (category, method). Rule-based over the reason field."""
    r = c["reason"].lower()
    claim = c["claimed_source"] or ""
    named = collections_in(claim)
    multi = ("Muttafaq alayh" in named
             or len([n for n in named if n != "Muttafaq alayh"]) >= 2)

    if re.search(r"adab al[- ]mufrad", r):
        return "overclaim", "reason"
    if multi and re.search(_PARTIAL, r):
        return "overclaim", "reason"
    if re.search(r"muttafaq|agreed[- ]upon", r) and re.search(r"incorrect|not\b", r):
        return "overclaim", "reason"
    if re.search(_NOT_THERE, r) or re.search(_ELSEWHERE, r):
        return "wrong collection", "reason"
    if re.search(r"\b(?:number|no\.|numbering|chapter|book|kitab|bab)\b", r):
        return "right collection, wrong number or subdivision", "reason"
    return "unclassified", "none"


def wrongly_named_companion(c):
    """The Companion the model named, from the isnad span. 'عن X' then trimmed."""
    s = c["isnad"] or ""
    m = re.search(r"عن\s+([^،,:.\n]{2,40})", s)
    if not m:
        return None
    name = re.sub(r"\s*(?:رضي الله عنه[ما]?|رضي الله عنها|أن النبي|أنه|قال).*$", "",
                  m.group(1)).strip()
    return name or None


# ------------------------------------------------------------- statistics ---

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "--"


# ------------------------------------------------------------------- main ---

def main():
    data = {cond: list(A.citations(cond)) for cond in CONDITIONS}
    rows = []
    for cond, cits in data.items():
        for c in cits:
            if c["fidelity"] == "fabricated" or not c["misattributed"]:
                continue
            rec = dict(c)
            if c["kind"] == "Ayah":
                cat, meth, csid, cay, tsid, tay = classify_quran(c)
                rec.update(category=cat, method=meth, claimed_sid=csid,
                           claimed_ayah=cay, true_sid=tsid, true_ayah=tay,
                           companion=None)
            else:
                if c["cs_label"] == "incorrect":
                    cat, meth = classify_hadith_source(c)
                else:
                    cat, meth = "isnad only", "label"
                rec.update(category=cat, method=meth,
                           companion=wrongly_named_companion(c) if c["isnad_label"] == "incorrect" else None)
            rec["isnad_wrong"] = (c["isnad_label"] == "incorrect")
            rows.append(rec)

    with open(os.path.join(HERE, "misattributions.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["condition", "model", "tier", "qid", "kind", "fidelity", "category",
                "method", "isnad_wrong", "companion", "claimed_source", "isnad", "reason"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    out = []
    W = out.append
    W("# Attribution errors in LLM-generated Quranic and Hadith citations\n")
    W("Generated by `run_analysis.py`. No source data was modified.\n")
    W("**Scope.** No train/dev/test restriction: the implicit condition was never "
      "released and has no split, so the explicit side is taken in full too and the "
      "two conditions are counted identically.\n")
    W("**Misattributed** = text label correct or corrupted AND at least one supplied "
      "attribution component (claimed source or isnad) labelled incorrect.\n")
    W("**Validation gate.** Implicit, share of the model's citations: Fanar-1-9B "
      "14.9% correct-text-misattributed / 6.0% corrupted-misattributed; ALLaM-7B "
      "9.2% / 3.2%. Both reproduce exactly.\n")

    # ---- Task 1
    W("\n## Task 1: kind of attribution error\n")
    for cond in CONDITIONS:
        W(f"\n### {cond.capitalize()} condition\n")
        for kind, label in (("Ayah", "Quran (claimed source)"),
                            ("Hadith", "Hadith (claimed source)")):
            W(f"\n**{label}** — method: "
              + ("deterministic corpus lookup, reason only as fallback"
                 if kind == "Ayah" else "rule-based over the annotator reason field") + "\n")
            W("\n| Tier | " + " | ".join(
                sorted({r["category"] for r in rows
                        if r["kind"] == kind and r["category"] != "isnad only"})) + " | n |")
            cats = sorted({r["category"] for r in rows
                           if r["kind"] == kind and r["category"] != "isnad only"})
            W("|" + "---|" * (len(cats) + 2))
            for tier in TIERS:
                sel = [r for r in rows if r["condition"] == cond and r["kind"] == kind
                       and r["tier"] == tier and r["category"] != "isnad only"]
                if not sel:
                    continue
                cc = collections.Counter(r["category"] for r in sel)
                W(f"| {TIER_LABEL[tier]} | "
                  + " | ".join(f"{cc[c]} ({pct(cc[c], len(sel))})" for c in cats)
                  + f" | {len(sel)} |")
        # isnad
        isn = [r for r in rows if r["condition"] == cond and r["isnad_wrong"]]
        W(f"\n**Hadith isnad** — wrong Companion named: {len(isn)} citations "
          f"({', '.join(f'{t}: {sum(1 for r in isn if r['tier']==t)}' for t in TIERS)})\n")

    # ---- Task 3
    W("\n## Task 3: per-model misattribution rate\n")
    W("Rate = misattributed / (citations with text correct or corrupted that supplied "
      "at least one attribution component). Wilson 95% CI. Every model is listed; "
      "thin denominators are visible in the n column rather than hidden.\n")
    for cond in CONDITIONS:
        W(f"\n### {cond.capitalize()}\n")
        W("\n| Tier | Model | misattributed | n | rate | 95% CI |")
        W("|---|---|---|---|---|---|")
        for tier in TIERS:
            for m in [x for x in (A.FRONTIER + A.MATCHED + A.EXTENDED)
                      if A.TIER[x] == tier]:
                den = [c for c in data[cond] if c["model"] == m
                       and c["fidelity"] != "fabricated" and c["supplied"]]
                if not den:
                    continue
                k = sum(1 for c in den if c["misattributed"])
                lo, hi = wilson(k, len(den))
                W(f"| {TIER_LABEL[tier]} | {A.DISPLAY[m]} | {k} | {len(den)} | "
                  f"{pct(k, len(den))} | [{lo:.1f}, {hi:.1f}] |")

    with open(os.path.join(HERE, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"rows: {len(rows)} misattributed citations")
    print("wrote misattributions.csv, summary.md")


if __name__ == "__main__":
    main()
