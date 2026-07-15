# -*- coding: utf-8 -*-
"""
Build the Ansari-based shared-task TEST question set.

1. Parse ansari_arabic_questions_all.md  (blocks: **Qn.** <text> ... ---)
2. Assign stable QIDs QA%05d (from the md's Qn number) -> ansari_questions_all.json
3. Light QUALITY GATE (fragments / pasted essays / non-Arabic / exact dups)
4. Seeded random 500 from the gated pool          -> ansari_test_500.json
5. Print validation: per-rule drop counts + samples of dropped & kept, and a
   topic/length comparison of the 500 vs the full pool (confirm random ~unbiased).

Files use the inference schema {"id","prompt"} so ansari_test_500.json feeds
straight into inference_vllm.py / inference_hugging_face.py.
"""
import re, sys, io, json, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC   = "../data/islamiceval2026_RELEASE/ansari_arabic_questions_all.md"
ALL   = "../data/islamiceval2026_RELEASE/ansari_questions_all.json"
TEST  = "../data/islamiceval2026_RELEASE/ansari_test_500.json"
N, SEED = 500, 42

# gate thresholds
MIN_CHARS, MIN_WORDS, MAX_CHARS, MIN_AR_RATIO = 15, 3, 800, 0.5

AR = re.compile(r'[؀-ۿ]')
def ar_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(1 for c in letters if AR.match(c)) / len(letters) if letters else 0.0
def norm(t):
    t = re.sub(r'[ًٌٍَُِّْـ]', '', t); t = re.sub(r'[^ء-ي ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

# ---- 1+2. parse -> QID ----
txt = open(SRC, encoding="utf-8").read()
parts = re.split(r'\*\*Q(\d+)\.\*\*', txt)
allq = []
for i in range(1, len(parts), 2):
    num = int(parts[i])
    body = re.sub(r'\n\s*---\s*$', '', parts[i+1].strip()).strip().strip('- \n')
    if body:
        allq.append({"id": f"QA{num:05d}", "prompt": body})
json.dump(allq, open(ALL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"parsed {len(allq)} questions -> {ALL}")

# ---- 3. quality gate ----
drops = {"too_short": [], "too_long": [], "non_arabic": [], "dup": []}
kept, seen = [], set()
for q in allq:
    t = q["prompt"]; nwords = len(t.split())
    if len(t) < MIN_CHARS or nwords < MIN_WORDS:
        drops["too_short"].append(q)
    elif len(t) > MAX_CHARS:
        drops["too_long"].append(q)
    elif ar_ratio(t) < MIN_AR_RATIO:
        drops["non_arabic"].append(q)
    else:
        key = norm(t)
        if key in seen:
            drops["dup"].append(q)
        else:
            seen.add(key); kept.append(q)

print(f"\n=== QUALITY GATE (min {MIN_CHARS} chars & {MIN_WORDS} words; max {MAX_CHARS}; "
      f"ar>={MIN_AR_RATIO}) ===")
for k, v in drops.items():
    print(f"  dropped {k:11s}: {len(v)}")
print(f"  --> gated pool: {len(kept)} / {len(allq)}")

# ---- 4. seeded random 500 ----
random.seed(SEED)
sample = random.sample(kept, min(N, len(kept)))
sample.sort(key=lambda q: q["id"])          # stable output order
json.dump(sample, open(TEST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nsampled {len(sample)} (seed {SEED}) -> {TEST}")

# ---- 5. validation output ----
def show(title, items, n=8):
    print(f"\n--- {title} (showing {min(n,len(items))}/{len(items)}) ---")
    for q in items[:n]:
        print(f"  [{q['id']}] {q['prompt'][:100].replace(chr(10),' / ')}")

for k in ("too_short", "too_long", "non_arabic", "dup"):
    if drops[k]: show(f"DROPPED · {k}", drops[k])
show("SAMPLED 500 (first few by id)", sample, 12)

# topic/length comparison: sampled 500 vs full pool (random should be ~unbiased)
def profile(items):
    def has(t, ws): return any(w in t for w in ws)
    H = ["حديث","سند","رواه","صحيح البخاري","صحيح مسلم","السنة"]
    Qr = ["آية","سورة","القرآن","تفسير","قوله تعالى"]
    F = ["حكم","يجوز","حرام","حلال","واجب","الصلاة","الصيام","الزكاة","الطلاق","الحج","الوضوء"]
    n = len(items)
    ls = sorted(len(q["prompt"]) for q in items)
    return {
        "n": n,
        "cite_elicit%": round(100*sum(1 for q in items if has(q["prompt"],H) or has(q["prompt"],Qr))/n),
        "fiqh%":        round(100*sum(1 for q in items if has(q["prompt"],F))/n),
        "median_chars": ls[n//2],
    }
print("\n--- profile: SAMPLED 500 vs FULL GATED POOL (random should match) ---")
print("  sampled:", profile(sample))
print("  pool   :", profile(kept))
