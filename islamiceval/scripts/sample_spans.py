#!/usr/bin/env python3
"""
sample_spans.py
===============
Draw a REPRESENTATIVE sample of incorrect citation spans from citation_spans.csv
for the correction-task method comparison / prompt tuning.

Design (agreed):
  • Stratify by ref: a forced 50/50 Quran / Hadith split (--per-ref each).
  • Within each ref stratum, allocate slots to models PROPORTIONAL to each model's
    share of incorrect spans (largest-remainder method → exact integer quotas, not
    left to the luck of a uniform draw), then random-sample within each model.
  • Prompt variants are COLLAPSED to their base model for the weighting only
    (gemma-3-12b-explicit → gemma-3-12b, fanar-1-9b-explicit → fanar-1-9b); the
    emitted rows keep the ORIGINAL model name (so the answer/prompt-variant is known).
  • Reproducible via --seed.

Why not round-robin (the old sample50): round-robin over-weighted low-volume models
and force-included frontier models, whose incorrect spans are mostly recoverable
near-misses — biasing the sample toward "ok" and away from "خطأ".

Output: same columns as the input (id, model, span, ref, correctness), UTF-8-BOM.

Usage:
    python sample_spans.py                       # 50+50 -> sample100_spans.csv
    python sample_spans.py --per-ref 75 --seed 7 --out ../outputs/correction/sample150_spans.csv
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict


# Frontier reference models — present in the answer set for OTHER purposes (they are
# the gold/reference answers, not the dissertation's weak-model subjects). Excluded
# from the correction sample by default.
FRONTIER = {"gpt-5", "gpt-5.4", "gpt-5.5", "gemini-3-flash-preview", "gemini-3.1-pro-preview"}


def base_model(m: str) -> str:
    """Collapse a prompt-variant to its base model (for proportional weighting)."""
    return m[:-len("-explicit")] if m.endswith("-explicit") else m


def largest_remainder(counts: dict, total_slots: int) -> dict:
    """Allocate total_slots across keys proportional to counts (Hamilton method)."""
    pool = sum(counts.values())
    if pool == 0:
        return {k: 0 for k in counts}
    raw = {k: total_slots * c / pool for k, c in counts.items()}
    quota = {k: int(v) for k, v in raw.items()}
    remainder = total_slots - sum(quota.values())
    # hand out the leftover slots to the largest fractional parts
    frac = sorted(counts, key=lambda k: (raw[k] - quota[k], counts[k]), reverse=True)
    for k in frac[:remainder]:
        quota[k] += 1
    return quota


def sample_stratum(rows, total_slots, rng):
    """rows = incorrect spans of one ref type. Return the sampled rows."""
    by_base = defaultdict(list)
    for r in rows:
        by_base[base_model(r["model"])].append(r)
    counts = {m: len(v) for m, v in by_base.items()}
    quota = largest_remainder(counts, total_slots)

    picked = []
    deficit = 0
    for m, q in quota.items():
        avail = by_base[m]
        take = min(q, len(avail))
        deficit += q - take
        picked.extend(rng.sample(avail, take))
    # redistribute any deficit (a model had fewer spans than its quota) to the rest
    if deficit:
        remaining = [r for r in rows if r not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[:deficit])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../outputs/correction/citation_spans.csv")
    ap.add_argument("--out", default="../outputs/correction/sample100_spans.csv")
    ap.add_argument("--per-ref", type=int, default=50, help="spans per ref type (quran, hadith)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-frontier", action="store_true",
                    help="keep frontier reference models (gpt-5/5.4, gemini); default excludes them")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = list(csv.DictReader(open(args.input, encoding="utf-8-sig")))
    inc = [r for r in rows if r["correctness"] == "incorrect" and r["ref"] in ("quran", "hadith")]
    if not args.include_frontier:
        n_before = len(inc)
        inc = [r for r in inc if base_model(r["model"]) not in FRONTIER]
        print(f"Excluded {n_before - len(inc)} frontier-model spans "
              f"({', '.join(sorted(FRONTIER))}); {len(inc)} incorrect spans remain.")

    sampled = []
    for ref in ("quran", "hadith"):
        stratum = [r for r in inc if r["ref"] == ref]
        picked = sample_stratum(stratum, args.per_ref, rng)
        sampled.extend(picked)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "model", "span", "ref", "correctness"])
        w.writeheader()
        w.writerows(sampled)

    # ── summary ──
    print(f"Wrote {len(sampled)} spans -> {args.out}  (seed={args.seed})")
    print(f"  by ref: {dict(Counter(r['ref'] for r in sampled))}")
    print(f"  unique questions: {len(set(r['id'] for r in sampled))} | "
          f"distinct base models: {len(set(base_model(r['model']) for r in sampled))}")
    print("  per base-model (sampled / available-incorrect):")
    avail = Counter(base_model(r["model"]) for r in inc)
    got = Counter(base_model(r["model"]) for r in sampled)
    for m, n in got.most_common():
        print(f"     {n:3d} / {avail[m]:4d}   {m}")


if __name__ == "__main__":
    main()
