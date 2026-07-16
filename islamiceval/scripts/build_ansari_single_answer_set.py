# -*- coding: utf-8 -*-
"""
Build a 500-row single-answer annotation set from the Ansari test-500 generations.

Requirement:
  - 500 unique questions (the ansari_test_500 set), each answered by exactly ONE model.
  - Each model used exactly 50 times (10 models x 50 = 500).
  - Random assignment, BUT only a (question, model) pair whose answer SURVIVED cleaning
    (i.e. exists in that model's *_clean.json) may be used -> accommodates removals.

This is a capacitated bipartite matching: questions <-> models, each model capacity 50,
edge (q, m) exists iff model m has a clean answer for q. Solved with randomized
augmenting paths (most-constrained question first), which yields a random *feasible*
balanced assignment. Reproducible via --seed.

Output (one file, raw answer schema with the model kept in each record):
  ansari_annotation_set_500.json  -> [{id, prompt, answer, model}]  (seed-shuffled order)
"""
import os, json, glob, random, argparse

ANS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "answers", "ansari_test_500")
TEST_500 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "islamiceval2026_RELEASE", "ansari_test_500.json")
CAP = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    qs = json.load(open(TEST_500, encoding="utf-8"))
    qorder = [q["id"] for q in qs]
    qtext = {q["id"]: q["prompt"] for q in qs}

    clean = {}  # model -> {qid: answer}
    for f in sorted(glob.glob(os.path.join(ANS_DIR, "*_alt2025-explicit_clean.json"))):
        model = os.path.basename(f).replace("_alt2025-explicit_clean.json", "")
        clean[model] = {r["id"]: r["answer"] for r in json.load(open(f, encoding="utf-8"))}
    models = sorted(clean)
    assert len(models) * args.cap == len(qorder), \
        f"{len(models)} models x {args.cap} != {len(qorder)} questions"

    avail = {q: [m for m in models if q in clean[m]] for q in qorder}
    uncoverable = [q for q in qorder if not avail[q]]
    assert not uncoverable, f"{len(uncoverable)} questions have no clean answer: {uncoverable[:5]}"

    # ---- randomized augmenting-path assignment ----
    assign, load, by_model = {}, {m: 0 for m in models}, {m: [] for m in models}

    def augment(q, visited):
        ms = avail[q][:]; rng.shuffle(ms)
        for m in ms:
            if m in visited:
                continue
            visited.add(m)
            if load[m] < args.cap:
                assign[q] = m; load[m] += 1; by_model[m].append(q); return True
            cand = by_model[m][:]; rng.shuffle(cand)
            for q2 in cand:                                   # try to relocate one of m's questions
                by_model[m].remove(q2); load[m] -= 1; del assign[q2]
                if augment(q2, visited):
                    assign[q] = m; load[m] += 1; by_model[m].append(q); return True
                assign[q2] = m; load[m] += 1; by_model[m].append(q2)   # restore
        return False

    order = sorted(qorder, key=lambda q: (len(avail[q]), rng.random()))   # most-constrained first
    unassigned = [q for q in order if not augment(q, set())]
    assert not unassigned, f"could not assign: {unassigned}"

    # ---- verify invariants ----
    assert len(assign) == len(qorder)
    assert all(load[m] == args.cap for m in models), load
    assert all(assign[q] in avail[q] for q in qorder)         # every pair is a clean answer
    assert len({q for q in assign}) == len(qorder)            # each question exactly once

    # ---- build one file, raw schema {id, prompt, answer, model} ----
    rows = [{"id": q, "prompt": qtext[q], "answer": clean[assign[q]][q], "model": assign[q]}
            for q in qorder]
    rng.shuffle(rows)                                         # spread models through the file

    op = os.path.join(ANS_DIR, "ansari_annotation_set_500.json")
    json.dump(rows, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # drop the old split key file if it exists (model now lives in the main file)
    kp = os.path.join(ANS_DIR, "ansari_annotation_key_500.json")
    if os.path.exists(kp):
        os.remove(kp)

    from collections import Counter
    print(f"assigned {len(rows)} unique question/answer pairs (cap {args.cap}/model, seed {args.seed})")
    print("per-model counts:", dict(sorted(Counter(r['model'] for r in rows).items())))
    print("fields:", list(rows[0].keys()))
    print(f"-> {op}")


if __name__ == "__main__":
    main()
