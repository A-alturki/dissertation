#!/usr/bin/env python3
"""
compare_expansion_metrics.py
============================
Diff two correction audits produced with different hadith-expansion metrics
(char-ratio vs Jaccard) on the SAME predictions, to see what changed.

Compares per-hadith-span correction sets (matched by normalized text) and reports:
  * list-size distribution + max, each metric
  * spans where the set SHRANK (jaccard removed frame-sharers) or GREW (jaccard added
    variants char missed), with counts and examples
  * the specific large-list ids

Usage:
  python compare_expansion_metrics.py \
     --char ../outputs/correction/release_corrections_grounded.json \
     --jac  ../outputs/correction/release_corrections_grounded_jaccard.json
"""
import os, sys, json, argparse
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ground_corrections as G


def texts(entry):
    return {G.norm_dedup(c["text"]): c["text"] for c in (entry.get("corrections") or [])}


def sizes(aud):
    return Counter(len(a.get("corrections") or []) for a in aud.values()
                   if a.get("ref") == "hadith" and a.get("corrections"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default=os.path.join(HERE, "..", "outputs", "correction", "release_corrections_grounded.json"))
    ap.add_argument("--jac",  default=os.path.join(HERE, "..", "outputs", "correction", "release_corrections_grounded_jaccard.json"))
    args = ap.parse_args()
    C = json.load(open(args.char, encoding="utf-8"))
    J = json.load(open(args.jac, encoding="utf-8"))

    sc, sj = sizes(C), sizes(J)
    def summ(s):
        n=sum(s.values()); tot=sum(k*v for k,v in s.items())
        return f"spans={n} totalvariants={tot} avg={tot/n:.2f} max={max(s) if s else 0}"
    print("HADITH correction-list sizes (grounded spans only):")
    print(f"  char-ratio: {summ(sc)}")
    print(f"  jaccard   : {summ(sj)}")
    print(f"  char sizes {dict(sorted(sc.items()))}")
    print(f"  jac  sizes {dict(sorted(sj.items()))}\n")

    shrank, grew, same, changed_ex = [], [], 0, []
    for sid, cj in J.items():
        cc = C.get(sid)
        if not cc or cj.get("ref") != "hadith": continue
        tc, tj = texts(cc), texts(cj)
        if not tc and not tj: continue
        removed = set(tc) - set(tj)     # char had, jaccard dropped (frame-sharers)
        added   = set(tj) - set(tc)     # jaccard added (variants char missed)
        if len(tj) < len(tc): shrank.append((sid, len(tc), len(tj), removed, tc))
        elif len(tj) > len(tc): grew.append((sid, len(tc), len(tj), added, tj))
        else: same += 1

    print(f"per-span change (hadith): shrank {len(shrank)} | grew {len(grew)} | same {same}\n")

    print("=== SHRANK — jaccard removed suspected frame-sharers (precision gain) ===")
    for sid, nc, nj, removed, tc in sorted(shrank, key=lambda x:x[1]-x[2], reverse=True)[:6]:
        print(f"  {sid}: {nc} -> {nj}  (removed {len(removed)})")
        for k in list(removed)[:3]:
            print(f"     - {tc[k][:70]}")
    print("\n=== GREW — jaccard added variants char-ratio missed (recall gain) ===")
    for sid, nc, nj, added, tj in sorted(grew, key=lambda x:x[2]-x[1], reverse=True)[:6]:
        print(f"  {sid}: {nc} -> {nj}  (added {len(added)})")
        for k in list(added)[:3]:
            print(f"     + {tj[k][:70]}")

    # verdict-level: does خطأ vs grounded change? (a span could lose all variants)
    flip_to_khata = flip_to_ok = 0
    for sid, cj in J.items():
        cc = C.get(sid)
        if not cc or cj.get("ref")!="hadith": continue
        ck = cc.get("result")=="خطأ"; jk = cj.get("result")=="خطأ"
        if not ck and jk: flip_to_khata+=1
        if ck and not jk: flip_to_ok+=1
    print(f"\nspan verdict flips: grounded->خطأ {flip_to_khata} | خطأ->grounded {flip_to_ok}")


if __name__ == "__main__":
    main()
