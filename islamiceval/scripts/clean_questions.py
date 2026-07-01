#!/usr/bin/env python3
"""
clean_questions.py — word-based question cleaning stage (after the Gemini severity pass).

Pipeline position:
  [Gemini severity classifier  qflags_full_flash3.json]   <- already done
        -> THIS STAGE: keep level-1 only, then remove vulgar(tier-1) + sectarian prompts
        -> output cleaned JSON  -> [Gemini VULGARITY_RUBRIC pass]  (run separately)
        -> final cleaned set

What it does, in order, tracking every removal with reason + matched term:
  0. SEVERITY  — keep only Gemini level-1 questions ("1|N"); everything >=2 (and the 1
     malformed record) is removed here, tagged with its level.
  1. VULGAR    — remove prompts hitting the TIER-1 vulgar/sexual/slur lexicon.
  2. SECTARIAN — remove prompts naming a sect/creed/Sufi-order/heterodox group/pejorative
     (creedal/shia/sufi/heterodox/pejorative). NOT madhhabs, NOT Sunni terms, NOT the
     مذاهب/طوائف "meta" words (those are kept — they're benign comparative-fiqh).

Matching: diacritics stripped + alef/yaa/hamza normalized, with Arabic token boundaries and
optional proclitics (و ف ب ك ل ال يا) — so no matching inside unrelated words.

Outputs (../outputs/classification/):
  questions_level1_wordcleaned.json   survivors [{qid,prompt}] -> feed to the vulgarity flash pass
  questions_removed_log.json          every removed record: {qid, stage, reason, matched, level, prompt}
  question_cleaning_stats.json        counts per reason
and prints the stats table.
"""
import sys, os, re, json, argparse
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── normalization (explicit codepoints so letters 0621-064A are never touched) ──────
_DIAC = re.compile("[ؐ-ؚـً-ٰٟۖ-ۭ]")  # signs/harakat/tatweel
def norm(s):
    s = _DIAC.sub("", str(s))
    return s.translate(str.maketrans("أإآىئؤ", "اااايء"))

AL = "ء-ي"                                   # Arabic letter block (for boundaries)
PREF = r"(?:[وفبكل]|ال|وال|بال|فال|كال|لل|يا)?"        # optional proclitics
def matcher(stems):
    alt = "|".join(re.escape(norm(s)) for s in stems if norm(s))
    return re.compile(rf"(?<![{AL}]){PREF}(?:{alt})(?![{AL}])")

def first_hit(rx, normed):
    m = rx.search(normed)
    return m.group(0) if m else None

# ── lexicons ────────────────────────────────────────────────────────────────────────
# TIER-1 vulgar / sexual / slur — strong signal, rarely legitimate
VULGAR = ["كس", "كسم", "كسمك", "طيز", "زب", "زبر", "قحبة", "شرموطة", "شرموط", "شرموطه",
          "عاهرة", "منيوك", "منيوكة", "متناك", "متناكة", "نياكة", "عرص", "اعرص",
          "خول", "ديوث", "لوطي", "يلعن دينك", "كلب ابن"]

# Sectarian-identity stems (NO madhhabs, NO Sunni terms, NO مذاهب/طوائف meta words)
creedal   = ["اشعري", "اشاعرة", "ماتريدي", "ماتريدية", "معتزلي", "معتزلة", "سلفي", "سلفية",
             "مرجئة", "مرجئي", "جهمية", "جهمي", "قدرية", "خوارج", "خارجي", "خوارجي"]
shia      = ["شيعة", "شيعي", "شيعية", "رافضة", "روافض", "رافضي", "اثنا عشرية", "اثني عشرية",
             "اثنا عشري", "جعفري", "جعفرية", "امامية", "امامي", "زيدية", "زيدي",
             "اسماعيلية", "اسماعيلي", "نصيرية", "نصيري", "علوي", "علوية", "علويين",
             "باطنية", "باطني"]
sufi      = ["صوفي", "صوفية", "تصوف", "نقشبندية", "نقشبندي", "قادرية", "قادري",
             "تيجانية", "تيجاني", "شاذلية", "شاذلي", "مولوية", "مولوي"]
heterodox = ["اباضية", "اباضي", "دروز", "درزي", "درزية", "بهائية", "بهائي", "بهائيون",
             "احمدية", "احمدي", "قاديانية", "قادياني"]
pejorative = ["نواصب", "ناصبي", "مجوس", "مجوسي", "وهابي", "وهابية"]
SECTS = {"creedal": creedal, "shia": shia, "sufi": sufi,
         "heterodox": heterodox, "pejorative": pejorative}


def main():
    ap = argparse.ArgumentParser(description="Word-based question cleaning (level-1 -> vulgar -> sectarian).")
    ap.add_argument("--input", default="../outputs/classification/qflags_full_flash3.json")
    ap.add_argument("--out", default="../outputs/classification/questions_level1_wordcleaned.json")
    ap.add_argument("--removed", default="../outputs/classification/questions_removed_log.json")
    ap.add_argument("--stats", default="../outputs/classification/question_cleaning_stats.json")
    args = ap.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    total = len(data)

    rx_vulgar = matcher(VULGAR)
    rx_sects = {name: matcher(stems) for name, stems in SECTS.items()}

    def level(r):
        return (r.get("output") or "").split("|")[0]

    kept, removed = [], []
    for r in data:
        qid, prompt = r.get("qid"), r.get("prompt", "")
        lvl = level(r)
        # stage 0 — severity: keep only valid level-1
        if not (r.get("valid") and lvl == "1"):
            removed.append({"qid": qid, "stage": "severity",
                            "reason": f"gemini_level_{lvl or 'invalid'}",
                            "matched": r.get("output"), "level": lvl, "prompt": prompt})
            continue
        n = norm(prompt)
        # stage 1 — vulgar (tier-1)
        hit = first_hit(rx_vulgar, n)
        if hit:
            removed.append({"qid": qid, "stage": "vulgar", "reason": "tier1_vulgar",
                            "matched": hit, "level": lvl, "prompt": prompt})
            continue
        # stage 2 — sectarian (first matching category)
        sect_hit = None
        for name, rx in rx_sects.items():
            h = first_hit(rx, n)
            if h:
                sect_hit = (name, h); break
        if sect_hit:
            removed.append({"qid": qid, "stage": "sectarian", "reason": f"sect_{sect_hit[0]}",
                            "matched": sect_hit[1], "level": lvl, "prompt": prompt})
            continue
        kept.append({"qid": qid, "prompt": prompt})

    # ── write ──
    for path, obj in [(args.out, kept), (args.removed, removed)]:
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    by_reason = Counter(r["reason"] for r in removed)
    by_stage = Counter(r["stage"] for r in removed)
    stats = {"input_total": total, "kept": len(kept), "removed": len(removed),
             "by_stage": dict(by_stage), "by_reason": dict(by_reason.most_common())}
    json.dump(stats, open(args.stats, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ── report ──
    print(f"=== clean_questions ===  input {total}")
    print(f"  STAGE 0 severity  : removed {by_stage.get('severity',0):5}  (kept only level-1)")
    print(f"  STAGE 1 vulgar    : removed {by_stage.get('vulgar',0):5}")
    print(f"  STAGE 2 sectarian : removed {by_stage.get('sectarian',0):5}")
    print(f"  ---------------------------------------------")
    print(f"  KEPT (-> vulgarity flash pass): {len(kept)}  ({100*len(kept)/total:.1f}%)")
    print("\n  removed-by-reason:")
    for reason, c in by_reason.most_common():
        print(f"    {reason:22} {c:5}")
    print("\n  vulgar removals (verify FPs e.g. الخول=الدخول typo):")
    for r in [x for x in removed if x["stage"] == "vulgar"]:
        print(f"    {r['qid']} [{r['matched']}] {r['prompt'][:60]}")
    print("\n  sectarian sample (5/category):")
    for name in SECTS:
        ex = [x for x in removed if x["reason"] == f"sect_{name}"][:5]
        if ex:
            print(f"    [{name}] " + " / ".join(f"{x['matched']}:{x['prompt'][:22]}" for x in ex))
    print(f"\n-> {args.out}\n-> {args.removed}\n-> {args.stats}")


if __name__ == "__main__":
    main()
