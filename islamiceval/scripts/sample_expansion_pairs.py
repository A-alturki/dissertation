#!/usr/bin/env python3
"""
sample_expansion_pairs.py
=========================
Build a BLIND, disagreement-weighted sample of hadith matn pairs for manual labeling,
to defensibly choose between char-ratio and Jaccard for expand_hadith and locate each
metric's threshold.

Pool: real hadith anchors (the grounded/snapped matns from the dev correction run) ×
their top word-overlap corpus candidates (exactly the pool expand_hadith scores).
Each pair scored by BOTH metrics (hadith_similarity.char_ratio / .jaccard).

Selection (disagreement-weighted — every label maximally informative):
  A frame-sharer disagreement : char>=0.55 & jac<=0.30      (char keeps / jac drops)
  B reverse disagreement      : 0.45<=char<0.88 & jac>=0.40 (char drops / jac keeps → Mecca-type)
  C jaccard boundary          : 0.20<=jac<=0.50             (locate the jac threshold)
  D clear same                : jac>=0.60 & char>=0.60       (calibration)
  E clear different           : jac<=0.10 & char<=0.35       (calibration)

Outputs (outputs/correction/expansion_study/):
  pairs_blind.json  [{pid, A, B}]  — shuffled, NO scores (feed the labeler)
  pairs_key.json    {pid: {char, jac, jac_ext, bucket, anchor_id, cand_source}}

Usage:  python sample_expansion_pairs.py [--n 40] [--seed 7]
"""
import os, sys, json, random, argparse
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ground_corrections as G
import hadith_similarity as H

CORR = os.path.join(HERE, "..", "outputs", "correction")
OUTDIR = os.path.join(CORR, "expansion_study")


def load_anchors():
    """Distinct hadith correction gold texts from the dev audit = real anchors."""
    aud = json.load(open(os.path.join(CORR, "release_corrections_grounded.json"), encoding="utf-8"))
    seen, anchors = set(), []
    for sid, a in aud.items():
        if a.get("ref") != "hadith":
            continue
        for c in (a.get("corrections") or []):
            t = c.get("text", "")
            k = G.norm_dedup(t)
            if k and k not in seen and len(H.content_tokens(t)) >= 3:
                seen.add(k); anchors.append(t)
    return anchors


def scores(a, b):
    """All five metrics for the 2x2 (metric x frame-stripping) + extended."""
    return {"char":         round(H.char_ratio(a, b), 4),          # SEQUENCE + raw
            "char_content":  round(H.char_content(a, b), 4),       # SEQUENCE + content
            "jac_raw":       round(H.jaccard_raw(a, b), 4),        # SET + raw
            "jac":           round(H.jaccard(a, b), 4),            # SET + content
            "jac_ext":       round(H.jaccard(a, b, extended=True), 4)}


def build_pool(anchors, snapc, per_anchor=40):
    """For each anchor, its top word-overlap corpus matns; score all metrics."""
    docs, idx = snapc["hadith"]
    pool, seen_pair = [], set()
    for aid, anchor in enumerate(anchors):
        aw = H.content_tokens(anchor)                     # content words to index on
        cnt = Counter()
        for tok in aw:
            for i in idx.get(tok, ()):
                cnt[i] += 1
        ak = G.norm_dedup(anchor)
        for i, _ in cnt.most_common(per_anchor):
            cand = docs[i][1]
            ck = G.norm_dedup(cand)
            if not ck or ck == ak:                        # skip exact-identical text
                continue
            pk = tuple(sorted((ak, ck)))
            if pk in seen_pair:
                continue
            seen_pair.add(pk)
            pool.append({"anchor_id": aid, "A": anchor, "B": cand,
                         "cand_source": docs[i][0], **scores(anchor, cand)})
    return pool


def bucket_of(p):
    c, j = p["char"], p["jac"]
    if c >= 0.55 and j <= 0.30:            return "A_frame_disagree"
    if 0.45 <= c < 0.88 and j >= 0.40:     return "B_reverse_disagree"
    if 0.20 <= j <= 0.50:                  return "C_jac_boundary"
    if j >= 0.60 and c >= 0.60:            return "D_clear_same"
    if j <= 0.10 and c <= 0.35:            return "E_clear_diff"
    return None


# bucket proportions (scaled to --n)
PROP = {"A_frame_disagree": 0.35, "B_reverse_disagree": 0.15, "C_jac_boundary": 0.25,
        "D_clear_same": 0.125, "E_clear_diff": 0.125}


def make_target(n):
    t = {k: max(1, round(p * n)) for k, p in PROP.items()}
    # nudge the largest bucket so the total hits n exactly
    diff = n - sum(t.values())
    t["A_frame_disagree"] += diff
    return t


TARGET = make_target(40)     # replaced in main() once --n is known


def select(pool, rng):
    by = {k: [] for k in TARGET}
    for p in pool:
        b = bucket_of(p)
        if b:
            p["bucket"] = b; by[b].append(p)
    picked = []
    for b, need in TARGET.items():
        rng.shuffle(by[b])
        # de-dup anchors within a bucket where possible (variety)
        seen_anchor, chosen = set(), []
        for p in by[b]:
            if p["anchor_id"] in seen_anchor and len(by[b]) > need:
                continue
            seen_anchor.add(p["anchor_id"]); chosen.append(p)
            if len(chosen) >= need:
                break
        if len(chosen) < need:                              # top up ignoring anchor variety
            for p in by[b]:
                if p not in chosen:
                    chosen.append(p)
                    if len(chosen) >= need:
                        break
        print(f"  {b:20} available {len(by[b]):4}  picked {len(chosen)}")
        picked += chosen
    return picked


GEMINI_INSTRUCTIONS = (
    "أنت خبير في علم الحديث ومتخصص في تخريج الأحاديث وتمييز رواياتها.\n"
    "لديك أزواج من متون الأحاديث (بدون الإسناد). لكل زوج (المتن الأول A والمتن الثاني B) "
    "احكم: هل هما «نفس الحديث» أم «حديثان مختلفان»؟\n\n"
    "«نفس الحديث» (same): روايتان لنفس الحديث الواحد — نفس الواقعة أو نفس القول النبوي — "
    "ولو اختلفت الألفاظ أو الرواية أو التقديم والتأخير أو زيادة/نقص جملة أو المرادفات. "
    "العبرة أن يكون أصل الخبر/الحكم/الواقعة واحدًا بعينه.\n\n"
    "«حديثان مختلفان» (different): خبران أو قولان مختلفان، حتى لو:\n"
    "  • تشابها في الموضوع أو الفكرة العامة (كلاهما عن الصدقة، أو عن الصلاة، أو عن فضل عمل) — "
    "التشابه في الموضوع لا يكفي إطلاقًا.\n"
    "  • تشاركا فقط في العبارة الافتتاحية أو الإطار (مثل «قال رسول الله صلى الله عليه وسلم» أو "
    "«نهى رسول الله عن» أو «جاء رجل إلى النبي فقال») ثم اختلف المضمون بعدها.\n"
    "  • ذكرَا حكمين أو نهيين مختلفين وإن تشابهت الصيغة (نهيٌ عن شيءٍ ≠ نهيٌ عن شيءٍ آخر).\n\n"
    "القاعدة الحاسمة: اسأل نفسك — هل يرجع المتنان إلى حديثٍ واحدٍ بعينه في كتب السنة (نفس الخبر "
    "النبوي)، أم هما حديثان مستقلّان لا يجمعهما إلا الموضوع أو الصيغة؟ الأول → same، الثاني → "
    "different، وعند الشك الحقيقي → unsure. لا تحكم بـ same لمجرد اشتراكهما في الموضوع أو الإطار.\n\n"
    "احكم بالمتن (المضمون) فقط، وتجاهل الإسناد والألقاب والعبارات الافتتاحية.\n"
    "أخرج النتيجة كمصفوفة JSON فقط، بهذا الشكل بالضبط ولا شيء غيرها، وأرجِع جميع الأزواج (100) دون حذف:\n"
    "[{\"pid\":\"P000\",\"label\":\"same\"}, {\"pid\":\"P001\",\"label\":\"different\"}, ...]\n"
    "استخدم نفس قيمة pid المرفقة مع كل زوج. لا تضف أي شرح.\n"
)


def write_gemini_prompt(path, blind):
    with open(path, "w", encoding="utf-8") as f:
        f.write(GEMINI_INSTRUCTIONS)
        f.write("\n=== الأزواج ===\n")
        f.write(json.dumps(blind, ensure_ascii=False, indent=1))


def ckey(A, B):
    return tuple(sorted((G.norm_dedup(A), G.norm_dedup(B))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-anchor", type=int, default=40)
    ap.add_argument("--include-blind", default=None,
                    help="pin these already-labeled pairs (a pairs_blind.json) as the FIRST pairs, "
                         "keeping their pids; the rest are sampled fresh (excluding them)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # fixed prefix = an existing labeled blind set (keeps pid + A/B so old labels align)
    fixed, fixed_keys = [], set()
    if args.include_blind:
        fixed = json.load(open(args.include_blind, encoding="utf-8"))
        fixed_keys = {ckey(p["A"], p["B"]) for p in fixed}
        print(f"pinning {len(fixed)} labeled pairs as the prefix (from {os.path.basename(args.include_blind)})")

    n_new = args.n - len(fixed)
    global TARGET
    TARGET = make_target(n_new)
    print(f"sampling {n_new} NEW pairs; target buckets: {TARGET}")

    print("loading snap corpus + anchors...", flush=True)
    snapc = G.load_snap_corpus()
    anchors = load_anchors()
    pool = [p for p in build_pool(anchors, snapc, args.per_anchor)
            if ckey(p["A"], p["B"]) not in fixed_keys]     # exclude the pinned pairs
    print(f"  {len(anchors)} anchors | {len(pool)} candidate pairs (after excluding pinned)\n")

    new = select(pool, rng)
    rng.shuffle(new)
    new = new[:n_new]

    os.makedirs(OUTDIR, exist_ok=True)
    SCORE_KEYS = ("char", "char_content", "jac_raw", "jac", "jac_ext")
    blind, key = [], {}
    # 1) pinned prefix — keep pid + orientation exactly
    for p in fixed:
        pid, A, B = p["pid"], p["A"], p["B"]
        blind.append({"pid": pid, "A": A, "B": B})
        sc = scores(A, B)
        key[pid] = {**sc, "bucket": bucket_of({"char": sc["char"], "jac": sc["jac"]}) or "pinned",
                    "anchor_id": -1, "cand_source": "pinned"}
    # 2) new pairs — continue pids after the prefix, randomize orientation
    for j, p in enumerate(new):
        pid = f"P{len(fixed) + j:03d}"
        A, B = (p["A"], p["B"]) if rng.random() < 0.5 else (p["B"], p["A"])
        blind.append({"pid": pid, "A": A, "B": B})
        key[pid] = {**{k: p[k] for k in SCORE_KEYS},
                    "bucket": p["bucket"], "anchor_id": p["anchor_id"], "cand_source": p["cand_source"]}

    tag = f"_{args.n}"
    fb = os.path.join(OUTDIR, f"pairs_blind{tag}.json")
    fk = os.path.join(OUTDIR, f"pairs_key{tag}.json")
    fg = os.path.join(OUTDIR, f"gemini_labeling{tag}.md")
    json.dump(blind, open(fb, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(key, open(fk, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    write_gemini_prompt(fg, blind)
    print(f"\n  wrote {len(blind)} blind pairs -> {os.path.relpath(fb, HERE)}"
          f"  (first {len(fixed)} = your labeled set)")
    print(f"  wrote key (5 metric scores) -> {os.path.basename(fk)}")
    print(f"  wrote tuned Gemini prompt -> {os.path.basename(fg)}")
    print(f"  bucket mix: {dict(Counter(v['bucket'] for v in key.values()))}")


if __name__ == "__main__":
    main()
