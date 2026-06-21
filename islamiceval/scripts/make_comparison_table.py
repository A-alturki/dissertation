#!/usr/bin/env python3
"""Build the explicit(512) vs explicit_1500 comparison table (markdown)."""
import json, statistics as st
from citation_lookup import ref_counts

OLD = "../outputs/answers/explicit"
NEW = "../outputs/answers/explicit_1500"
MODELS = ["acegpt-8b", "allam-7b", "command-r-7b", "deepseek-r1-llama-8b",
          "jais-13b", "llama-3.1-8b", "mistral-7b", "phi-3.5", "qwen3-8b"]

# Exact truncation % (old@512, new@1500/3000) from token-count runs; None = gated tokenizer.
TRUNC = {
    "phi-3.5": (86.6, 3.1), "qwen3-8b": (94.9, 16.8), "acegpt-8b": (5.5, 1.5),
    "mistral-7b": (74.3, 30.0), "llama-3.1-8b": (41.9, 15.7),
    "deepseek-r1-llama-8b": (55.4, 11.3),
    "allam-7b": (None, None), "command-r-7b": (None, None), "jais-13b": (None, None),
}

def load(p): return json.load(open(p, encoding="utf-8"))
def words(a): return len(str(a).split())

def rep_score(a):
    w = str(a).split()
    if len(w) < 6: return 0.0
    tri = [" ".join(w[i:i+3]) for i in range(len(w)-2)]
    return 1 - len(set(tri))/len(tri)

def stats(records):
    """avg words, refs/answer, %answers with >=1 ref, %high-repeat."""
    n = len(records)
    w = ref = has = degen = 0
    for r in records:
        a = str(r.get("answer", ""))
        w += words(a)
        q, h = ref_counts(a)
        c = q + h
        ref += c
        has += (c >= 1)
        degen += (rep_score(a) > 0.5)
    return n, w/n, ref/n, 100*has/n, 100*degen/n

def fmt(x, suf=""): return "—" if x is None else f"{x}{suf}"

rows = []
for m in MODELS:
    old = load(f"{OLD}/{m}_alt2025-explicit.json")
    new = load(f"{NEW}/{m}_alt2025-explicit.json")
    no, wo, ro, ho, _   = stats(old)
    nn, wn, rn, hn, dn  = stats(new)
    to, tn = TRUNC[m]
    dpct = 100*(wn-wo)/wo if wo else 0
    rows.append((m, wo, wn, dpct, to, tn, ro, rn, ho, hn, dn))
    print(f"done {m}", flush=True)

# markdown
print("\n| Model | prev words | new words | %Δ | prev trunc% | new trunc% | "
      "prev refs/ans | new refs/ans | prev %≥1ref | new %≥1ref | degen%(new) |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for (m, wo, wn, d, to, tn, ro, rn, ho, hn, dn) in rows:
    print(f"| {m} | {wo:.0f} | {wn:.0f} | +{d:.0f}% | {fmt(to,'%')} | {fmt(tn,'%')} | "
          f"{ro:.2f} | {rn:.2f} | {ho:.0f}% | {hn:.0f}% | {dn:.1f}% |")
