"""
run_analysis2.py -- Task 2 (who gets falsely credited), the Task 3 concentration
check, Task 4 (examples) and the draft paragraph. Appends to summary.md and
writes examples.tex / draft_paragraph.tex. Reads only.
"""
from __future__ import annotations

import csv, os, re, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attrlib as A                      # noqa: E402
from run_analysis import (collections_in, wrongly_named_companion,  # noqa: E402
                          classify_quran, wilson, pct)

CONDS = ("implicit", "explicit")
DATA = {c: list(A.citations(c)) for c in CONDS}
MIS = list(csv.DictReader(open(os.path.join(HERE, "misattributions.csv"), encoding="utf-8")))

out = ["\n\n## Task 2: who gets falsely credited\n"]
W = out.append


def top(counter, n=5):
    tot = sum(counter.values())
    return [(k, v, f"{100*v/tot:.1f}%") for k, v in counter.most_common(n)], tot


# ---- 2a: collections wrongly named, Companions wrongly named ----
for cond in CONDS:
    W(f"\n### {cond.capitalize()} — collections named on *misattributed* Hadith citations\n")
    cc = collections.Counter()
    for r in MIS:
        if r["condition"] == cond and r["kind"] == "Hadith":
            for name in collections_in(r["claimed_source"] or ""):
                cc[name] += 1
    rowsx, tot = top(cc, 5)
    W("\n| Collection named | citations | share of named mentions |")
    W("|---|---:|---:|")
    for k, v, p in rowsx:
        W(f"| {k} | {v} | {p} |")
    W(f"\nDenominator: {tot} collection mentions across "
      f"{sum(1 for r in MIS if r['condition']==cond and r['kind']=='Hadith')} misattributed "
      f"Hadith citations (a citation naming two collections contributes twice).\n")

    comp = collections.Counter(r["companion"] for r in MIS
                               if r["condition"] == cond and r["companion"])
    rowsx, tot = top(comp, 5)
    W(f"\n**Companions wrongly named in isnads ({cond})** — denominator {tot}\n")
    W("\n| Companion named | citations |")
    W("|---|---:|")
    for k, v, _ in rowsx:
        W(f"| {k} | {v} |")

# ---- 2b: fabricated vs correct-text Hadith, collection attached ----
W("\n### Collections attached to FABRICATED vs CORRECT-text Hadith citations\n")
W("\n| Condition | Collection | fabricated | share | correct-text | share |")
W("|---|---|---:|---:|---:|---:|")
for cond in CONDS:
    fab = collections.Counter()
    cor = collections.Counter()
    nf = nc = 0
    for c in DATA[cond]:
        if c["kind"] != "Hadith" or not c["claimed_source"]:
            continue
        names = collections_in(c["claimed_source"])
        if c["fidelity"] == "fabricated":
            nf += 1
            for n in names:
                fab[n] += 1
        elif c["fidelity"] == "correct":
            nc += 1
            for n in names:
                cor[n] += 1
    for name, _ in fab.most_common(5):
        W(f"| {cond} | {name} | {fab[name]} | {pct(fab[name], nf)} | "
          f"{cor[name]} | {pct(cor[name], nc)} |")
    W(f"| {cond} | *(denominators)* | {nf} | | {nc} | |")

# ---- 2b (cont): Companions on fabricated vs correct ----
W("\n### Companions attached to FABRICATED vs CORRECT-text Hadith citations\n")
W("\n| Condition | Companion | fabricated | correct-text |")
W("|---|---|---:|---:|")
for cond in CONDS:
    fab, cor = collections.Counter(), collections.Counter()
    for c in DATA[cond]:
        if c["kind"] != "Hadith" or not c["isnad"]:
            continue
        nm = wrongly_named_companion(c)
        if not nm:
            continue
        (fab if c["fidelity"] == "fabricated" else cor if c["fidelity"] == "correct"
         else collections.Counter())[nm] += 1
    for nm, v in fab.most_common(5):
        W(f"| {cond} | {nm} | {v} | {cor.get(nm, 0)} |")

# ---- 2c: surahs on fabricated vs correct Ayahs ----
W("\n### Surahs named on FABRICATED vs CORRECT Ayah citations\n")
W("\n| Condition | Surah claimed | fabricated | share | correct | share |")
W("|---|---|---:|---:|---:|---:|")
for cond in CONDS:
    fab, cor = collections.Counter(), collections.Counter()
    nf = nc = 0
    for c in DATA[cond]:
        if c["kind"] != "Ayah" or not c["claimed_source"]:
            continue
        sid, surface, _ = A.parse_quran_claim(c["claimed_source"])
        key = surface if sid is None else f"{sid}"
        if c["fidelity"] == "fabricated":
            nf += 1
            fab[key] += 1
        elif c["fidelity"] == "correct":
            nc += 1
            cor[key] += 1
    name_of = {str(s): n for n, s in A.NAME2ID.items()}
    for k, v in fab.most_common(5):
        W(f"| {cond} | {name_of.get(k, k)[:28]} | {v} | {pct(v, nf)} | "
          f"{cor.get(k, 0)} | {pct(cor.get(k, 0), nc)} |")
    W(f"| {cond} | *(denominators)* | {nf} | | {nc} | |")

# ---- Task 3 concentration check ----
W("\n\n## Task 3 (cont.): concentration on frequently cited texts\n")
W("For each model, the share of its CORRECTLY ATTRIBUTED citations whose cited text "
  "is among the 50 most frequently cited texts in that condition. Text key: the "
  "located (surah, ayah) for Ayahs, else the first 10 normalised tokens.\n")
for cond in CONDS:
    def key(c):
        if c["kind"] == "Ayah":
            loc = A.locate_ayah(c["text"])
            if loc:
                return f"Q{loc[0]}:{loc[1]}"
        return " ".join(A.ar_norm(c["text"]).split()[:10])

    freq = collections.Counter(key(c) for c in DATA[cond])
    top50 = {k for k, _ in freq.most_common(50)}
    W(f"\n### {cond.capitalize()}\n")
    W("\n| Model | correctly attributed | in top-50 | share |")
    W("|---|---:|---:|---:|")
    for m in A.FRONTIER + A.MATCHED + A.EXTENDED:
        ok = [c for c in DATA[cond] if c["model"] == m and c["supplied"]
              and c["fidelity"] != "fabricated" and not c["misattributed"]]
        if not ok:
            continue
        hit = sum(1 for c in ok if key(c) in top50)
        W(f"| {A.DISPLAY[m]} | {len(ok)} | {hit} | {pct(hit, len(ok))} |")

with open(os.path.join(HERE, "summary.md"), "a", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("appended Task 2 + concentration check to summary.md")
