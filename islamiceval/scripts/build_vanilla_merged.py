# -*- coding: utf-8 -*-
"""
build_vanilla_merged.py
=======================
Merge every per-model vanilla-prompt generation in `outputs/answers/vanilla_500/`
into ONE flat json of `{questionid, question, answer, model}` rows — the same shape
as `ansari_test_500/ansari_frontier_answers.json`.

Sources: all `<model>_vanilla.json` files in the folder (local generations
{id, prompt, answer, model[, thinking]} + the frontier `gpt-5.4_vanilla.json` /
`gemini-3.1-pro_vanilla.json`). Error/empty answers are dropped (the local files are
already clean; only stray frontier refusals/empties get filtered).

Usage
  python build_vanilla_merged.py
  python build_vanilla_merged.py --out ../outputs/answers/vanilla_500/vanilla_all_answers.json
"""
import os, re, sys, json, glob, argparse
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VDIR = os.path.join(ROOT, "outputs", "answers", "vanilla_500")

_AR = re.compile(r'[؀-ۿ]')


def bad(a):
    """Drop-reason for an error/degenerate answer, else None (keep). Same rule as
    frontier_batch_answers._bad_answer."""
    a = (a or "").strip()
    if not a or a.startswith("ERROR"):          return "error/empty"
    if len(a) < 5:                              return "too_short"
    letters = [c for c in a if c.isalpha()]
    if letters and sum(1 for c in letters if _AR.match(c)) / len(letters) < 0.3:
        return "foreign"
    toks = a.split()
    if len(toks) >= 8 and len(set(toks)) <= 2:  return "loop"
    return None


def main():
    ap = argparse.ArgumentParser(description="Merge all vanilla_500 per-model answers into one json")
    ap.add_argument("--dir", default=VDIR, help="vanilla_500 folder")
    ap.add_argument("--out", default=os.path.join(VDIR, "vanilla_all_answers.json"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_vanilla.json")))
    if not files:
        sys.exit(f"no *_vanilla.json files in {args.dir}")

    kept, dropped = [], Counter()
    per_model = Counter()
    for fp in files:
        rows = json.load(open(fp, encoding="utf-8"))
        rows = rows if isinstance(rows, list) else list(rows.values())
        for r in rows:
            model = r.get("model") or os.path.basename(fp).replace("_vanilla.json", "")
            ans = r.get("answer", "")
            reason = bad(ans)
            if reason:
                dropped[f"{model}:{reason}"] += 1
                continue
            kept.append({"questionid": r.get("id") or r.get("questionid"),
                         "question": r.get("prompt") or r.get("question"),
                         "answer": ans,
                         "model": model})
            per_model[model] += 1

    json.dump(kept, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"merged {len(files)} model files -> {len(kept)} rows -> {os.path.relpath(args.out, HERE)}")
    print(f"models ({len(per_model)}):")
    for m, n in sorted(per_model.items()):
        print(f"   {m:<22} {n}")
    if dropped:
        print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()
