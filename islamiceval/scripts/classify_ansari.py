# -*- coding: utf-8 -*-
"""
Stage-1 "answerable" classification of the Ansari test-set candidate pool.

Pipeline (mirrors the memory plan):
  1. Load ansari_questions_all.json (3562 parsed Ansari questions).
  2. Apply the SAME light quality gate as build_ansari_test_set.py -> gated pool (~3523).
  3. Classify each gated prompt as 1 (about Islam AND answerable by citing Quran/Hadith
     as primary evidence) or 0, using gpt-5.4 with REASONING OFF (reasoning_effort="none").
     - Parallelised with a thread pool.
     - Resumable: reuses labels already in the output CSV.
  4. Save labels CSV (id, prompt, label) + an answerable-only JSON.
  5. Optionally (--resample) draw a seeded random 500 from the answerable subset and
     (over)write ansari_test_500.json, backing up the previous file first.

The classification rubric (SYSTEM_PROMPT) is imported verbatim from classify_prompts.py
so the Ansari pool is filtered by the exact same definition the train/dev sets were.

Usage:
    python classify_ansari.py                      # classify (resumable)
    python classify_ansari.py --resample           # classify, then draw the 500
    python classify_ansari.py --resample --workers 12
"""
import os, re, io, sys, csv, json, time, random, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from openai import OpenAI

# Reuse the exact Stage-1 rubric that filtered train/dev.
from classify_prompts import SYSTEM_PROMPT

HERE = os.path.dirname(os.path.abspath(__file__))
def rp(*p):  # repo-relative path
    return os.path.normpath(os.path.join(HERE, "..", *p))

ALL_JSON  = rp("data", "islamiceval2026_RELEASE", "ansari_questions_all.json")
LABELS    = rp("outputs", "classification", "ansari_answerable_labels.csv")
KEEP_JSON = rp("outputs", "classification", "ansari_answerable.json")
TEST_500  = rp("data", "islamiceval2026_RELEASE", "ansari_test_500.json")

# ---- quality gate (identical thresholds to build_ansari_test_set.py) ----
MIN_CHARS, MIN_WORDS, MAX_CHARS, MIN_AR_RATIO = 15, 3, 800, 0.5
AR = re.compile(r'[؀-ۿ]')
def ar_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(1 for c in letters if AR.match(c)) / len(letters) if letters else 0.0
def norm(t):
    t = re.sub(r'[ًٌٍَُِّْـ]', '', t); t = re.sub(r'[^ء-ي ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def gate(allq):
    """Return (kept, drop_counts). Same logic as build_ansari_test_set.py."""
    drops = {"too_short": 0, "too_long": 0, "non_arabic": 0, "dup": 0}
    kept, seen = [], set()
    for q in allq:
        t = q["prompt"]; nwords = len(t.split())
        if len(t) < MIN_CHARS or nwords < MIN_WORDS:
            drops["too_short"] += 1
        elif len(t) > MAX_CHARS:
            drops["too_long"] += 1
        elif ar_ratio(t) < MIN_AR_RATIO:
            drops["non_arabic"] += 1
        else:
            key = norm(t)
            if key in seen:
                drops["dup"] += 1
            else:
                seen.add(key); kept.append(q)
    return kept, drops

# ---- classifier ----
_client = None
def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def parse_label(content):
    if not content:
        return -1
    t = content.strip()
    if t in ("0", "1"):
        return int(t)
    # fall back to the last 0/1 character emitted
    for ch in reversed(t):
        if ch in "01":
            return int(ch)
    return -1

def classify(prompt, model="gpt-5.4", max_retries=4):
    for attempt in range(max_retries):
        try:
            r = client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": str(prompt)},
                ],
                max_completion_tokens=2000,   # headroom; reasoning off -> ~1 output tok
                reasoning_effort="none",
            )
            return parse_label(r.choices[0].message.content)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [WARN] attempt {attempt+1}/{max_retries} failed: {e}; retry in {wait}s")
            time.sleep(wait)
    return -1

# ---- resumable CSV I/O ----
def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                lab = row.get("label", "")
                if lab not in ("", None):
                    try:
                        v = int(lab)
                        if v in (0, 1):        # only treat 0/1 as done; -1 => retry
                            done[row["id"]] = {"id": row["id"], "prompt": row["prompt"], "label": v}
                    except ValueError:
                        pass
    return done

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "prompt", "label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)

def run_classify(model, workers, save_every):
    allq = json.load(open(ALL_JSON, encoding="utf-8"))
    kept, drops = gate(allq)
    print(f"parsed {len(allq)} -> gate drops {drops} -> gated pool {len(kept)}")

    done = load_done(LABELS)
    todo = [q for q in kept if q["id"] not in done]
    print(f"resume: {len(done)} already labeled, {len(todo)} to classify (model={model}, workers={workers})")

    results = dict(done)  # id -> row
    lock_order = [q["id"] for q in kept]

    def flush():
        rows = [results[i] for i in lock_order if i in results]
        write_csv(LABELS, rows)

    if todo:
        n_done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(classify, q["prompt"], model): q for q in todo}
            for fut in as_completed(futs):
                q = futs[fut]
                lab = fut.result()
                results[q["id"]] = {"id": q["id"], "prompt": q["prompt"], "label": lab}
                n_done += 1
                if n_done % save_every == 0:
                    flush()
                    print(f"  ...{n_done}/{len(todo)} classified")
        flush()

    rows = [results[i] for i in lock_order if i in results]
    n1 = sum(1 for r in rows if r["label"] == 1)
    n0 = sum(1 for r in rows if r["label"] == 0)
    nerr = sum(1 for r in rows if r["label"] == -1)
    print(f"\n=== CLASSIFY DONE (n={len(rows)}) ===")
    print(f"  answerable (1): {n1} ({100*n1/len(rows):.1f}%)")
    print(f"  dropped    (0): {n0} ({100*n0/len(rows):.1f}%)")
    print(f"  errors    (-1): {nerr}")
    print(f"  labels -> {LABELS}")

    keep = [{"id": r["id"], "prompt": r["prompt"]} for r in rows if r["label"] == 1]
    json.dump(keep, open(KEEP_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  answerable-only -> {KEEP_JSON} ({len(keep)})")

    # sanity: show a few kept and a few dropped
    def show(title, items, n=10):
        print(f"\n--- {title} (showing {min(n,len(items))}/{len(items)}) ---")
        for r in items[:n]:
            print(f"  [{r['id']}] {r['prompt'][:90].replace(chr(10),' / ')}")
    show("KEPT · answerable==1", [r for r in rows if r["label"] == 1])
    show("DROPPED · label==0",   [r for r in rows if r["label"] == 0])
    if nerr:
        show("ERRORS · label==-1", [r for r in rows if r["label"] == -1])
    return rows, keep

# ---- sampling ----
def profile(items):
    def has(t, ws): return any(w in t for w in ws)
    H  = ["حديث","سند","رواه","صحيح البخاري","صحيح مسلم","السنة"]
    Qr = ["آية","سورة","القرآن","تفسير","قوله تعالى"]
    F  = ["حكم","يجوز","حرام","حلال","واجب","الصلاة","الصيام","الزكاة","الطلاق","الحج","الوضوء"]
    n = len(items); ls = sorted(len(q["prompt"]) for q in items)
    return {"n": n,
            "cite_elicit%": round(100*sum(1 for q in items if has(q["prompt"],H) or has(q["prompt"],Qr))/n),
            "fiqh%": round(100*sum(1 for q in items if has(q["prompt"],F))/n),
            "median_chars": ls[n//2]}

def resample(keep, n=500, seed=42):
    if len(keep) < n:
        print(f"\n[WARN] answerable pool ({len(keep)}) < {n}; sampling all {len(keep)}.")
        n = len(keep)
    # back up existing 500
    if os.path.exists(TEST_500):
        bak = TEST_500.replace(".json", "_prelabel.json")
        if not os.path.exists(bak):
            os.replace(TEST_500, bak)
            print(f"\nbacked up previous test set -> {bak}")
        else:
            print(f"\nbackup already exists ({bak}); leaving it untouched")
    random.seed(seed)
    sample = random.sample(keep, n)
    sample.sort(key=lambda q: q["id"])
    json.dump(sample, open(TEST_500, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"sampled {len(sample)} (seed {seed}) -> {TEST_500}")
    print("\n--- profile: SAMPLED vs ANSWERABLE POOL (random should match) ---")
    print("  sampled:", profile(sample))
    print("  pool   :", profile(keep))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resample", action="store_true", help="after classify, draw the seeded 500")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, keep = run_classify(args.model, args.workers, args.save_every)
    if args.resample:
        resample(keep, n=args.n, seed=args.seed)

if __name__ == "__main__":
    main()
