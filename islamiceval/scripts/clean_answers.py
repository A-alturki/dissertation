#!/usr/bin/env python3
"""
clean_answers.py
================
Filter Stage-2 generations before Stage-3 span detection. For each answer it (in order):

  1. STRIP leaked chat/template tokens  ([INST], <s>, </s>, <br>, <|...|>, mistral byte
     tokens ...)  — these are model template leakage, not content. Fanar's citation
     markers <aya_start>/<aya_end>/<hadith_start>/<hadith_end> are WHITELISTED (kept,
     and never counted as foreign — they delimit the gold citations).
  2. STRIP leading CoT — deepseek-style reasoning is a leading English block; we slice
     off leading lines while they are English-dominant and keep from the first
     Arabic-dominant line. No-op for answers that already start in Arabic. Preserves
     any English that appears *inside* the Arabic answer.

Then DROPS the entry (does not keep it) if any of:
  - empty after cleaning
  - FOREIGN ratio >= --eng-max  (Latin + CJK lumped together / all letters; default 0.30).
    A couple of stray CJK chars in an Arabic answer are fine — only heavy English/Chinese
    code-switching or OOM dumps cross the threshold.
  - degenerate loop:  zlib compression ratio < --comp-min  OR  repeated word-trigram
    ratio >= --rep-max
  - same sentence/citation repeated >= --sent-rep× AND filling >= --sent-cover of the text
  - one word-4gram >= --ngram-max share (min 30 words) — token loop
  - (optional) shorter than --min-len chars
  - TRUNCATED (only with --truncation): the generation hit its token budget (1500, or 3000
    for the forced-CoT models). Counted on the RAW answer with the model's own tokenizer:
    drop if tokens >= budget - margin. ALLaM's tokenizer does not round-trip (decode->encode
    ~doubles), so for it we use tokens/2 and require 1400-1600 AND a non-punctuation ending.

Outputs the kept entries (same schema, cleaned answer) to --output, and prints a per-file
drop breakdown. --calibrate prints comp/rep distributions instead of writing; --show N prints
sample drops.

Usage:
    python clean_answers.py --input ".../explicit_large_token_size/*.json" --calibrate
    python clean_answers.py --input FILE --eng-max 0.30 --comp-min 0.21 --rep-max 0.40 --output OUT.json
    python clean_answers.py --input FILE --truncation        # also drop budget-truncated answers
"""
import json, argparse, re, zlib, os, sys, glob, statistics
from collections import Counter

# drop reasons, in the order clean_one checks them
REASONS = ["empty", "foreign", "short", "degenerate", "sent_repeat", "ngram_loop", "truncated"]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

AR  = re.compile(r"[؀-ۿ]")                       # Arabic block
LAT = re.compile(r"[A-Za-z]")
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")   # CJK ideographs + kana

# fanar's legit citation markers — keep in output, ignore when measuring "foreign"
GOOD_MARKERS = re.compile(r"</?(?:aya|hadith|ref)(?:_start|_end)?>")

# leaked chat/template/byte tokens to remove (NOT the good markers above)
LEAK = re.compile(
    r"\[/?INST\]|<</?SYS>>|</?s>|<br\s*/?>|<\|[^>]{0,40}\|>"
    r"|<start_of_turn>|<end_of_turn>|<INST>|<ST[XT]*>|<STT>", re.I)

# ── truncation: each model's generation budget + tokenizer for re-counting ───────
THINKING_3000 = {"deepseek-r1-llama-8b", "deepseek-r1-qwen-32b",
                 "deepseek-r1-llama-70b", "lfm2.5-8b-a1b"}     # forced-CoT -> 3000-token budget
BROKEN_RT = {"allam-7b"}     # tokenizer decode->encode ~doubles; use tokens/2 + band + ending
TOK_REPO = {
    "allam-7b":             "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "jais-13b":             "inceptionai/jais-13b-chat",
    "acegpt-8b":            "FreedomIntelligence/AceGPT-v2-8B-Chat",
    "fanar-1-9b":           "QCRI/Fanar-1-9B-Instruct",
    "qwen3-8b":             "Qwen/Qwen3-8B",
    "qwen3.5-9b":           "Qwen/Qwen3.5-9B",
    "llama-3.1-8b":         "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-3-12b":          "google/gemma-3-12b-it",
    "mistral-7b":           "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-r1-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "phi-3.5":              "microsoft/Phi-3.5-mini-instruct",
    "command-r-7b":         "CohereForAI/c4ai-command-r7b-12-2024",
}
ENDPUNC = set(".!?؟۔،؛:\"'’”»«›‹)]}…")


def letters(s):
    """(arabic, latin, cjk) char counts after removing whitelisted citation markers."""
    t = GOOD_MARKERS.sub(" ", s)
    return len(AR.findall(t)), len(LAT.findall(t)), len(CJK.findall(t))

def foreign_ratio(s):
    """(latin + CJK) / all letters — English and Chinese lumped as 'foreign'."""
    a, l, c = letters(s)
    tot = a + l + c
    return (l + c) / tot if tot else 0.0

def strip_leak(s):
    return LEAK.sub("", s)

def strip_cot(s):
    """Drop leading foreign-dominant lines; keep from the first Arabic-dominant line
    (>=3 Arabic chars and Arabic-majority among letters). '' if no Arabic line."""
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        a, l, c = letters(ln)
        if a >= 3 and a > l + c:
            return "\n".join(lines[i:]).strip()
    return ""

def comp_ratio(s):
    b = s.encode("utf-8")
    return len(zlib.compress(b)) / len(b) if b else 1.0

def word_rep(s):
    """Repeated word-trigram ratio (1 - distinct/total). High => phrase loop."""
    w = s.split()
    if len(w) < 4:
        return 0.0
    tg = list(zip(w, w[1:], w[2:]))
    return 1 - len(set(tg)) / len(tg)

def rep_sentence(s):
    """(count, coverage) for the single most-repeated 'real' sentence (>15 chars):
    how many times it occurs, and the fraction of the answer those repeats fill
    (count*len(sentence)/len(answer)). A genuine loop repeats many times AND fills much
    of the text; a legitimately-repeated short attribution like «(رواه البخاري ومسلم)»
    after several hadiths repeats but covers little — coverage screens it out."""
    parts = [p.strip() for p in re.split(r"[.!؟?\n]", s) if len(p.strip()) > 15]
    if not parts:
        return 0, 0.0
    sent, cnt = Counter(parts).most_common(1)[0]
    return cnt, cnt * len(sent) / max(1, len(s))

def top_ngram_share(s, n=4, min_words=30):
    """Share of the single most-frequent word n-gram among all n-grams (0.0 if < min_words
    tokens). A high share means ONE short phrase dominates the text (token loop)."""
    w = s.split()
    if len(w) < min_words:
        return 0.0
    g = list(zip(*[w[i:] for i in range(n)]))
    if not g:
        return 0.0
    return Counter(g).most_common(1)[0][1] / len(g)


# ── truncation helpers ──────────────────────────────────────────────────────────
def budget_for(model):
    return 3000 if model in THINKING_3000 else 1500

def ends_punct(s):
    s = (s or "").rstrip()
    return bool(s) and s[-1] in ENDPUNC

def is_truncated(raw, trunc):
    """trunc = (tokenizer, budget, margin, broken). Counted on the RAW answer (the actual
    generation, incl. any stripped CoT/leak), so we see whether it hit the budget."""
    tok, budget, margin, broken = trunc
    n = len(tok.encode(raw or "", add_special_tokens=False))
    if broken:                       # ALLaM: re-encode ~doubles -> tokens/2 ~ true count
        half = n / 2
        return 1400 <= half <= 1600 and not ends_punct(raw)
    return n >= budget - margin


def clean_one(raw, eng_max, comp_min, rep_max, min_len, sent_rep, sent_cover, ngram_max, trunc=None):
    """Return (cleaned_answer, drop_reason|None). Reasons checked in priority order."""
    a = strip_cot(strip_leak(raw or "")).strip()
    if not a:
        return a, "empty"
    if foreign_ratio(a) >= eng_max:
        return a, "foreign"            # Latin + CJK together cross the threshold
    if min_len and len(a) < min_len:
        return a, "short"
    if comp_ratio(a) < comp_min or word_rep(a) >= rep_max:
        return a, "degenerate"
    rc, rcov = rep_sentence(a)
    if rc >= sent_rep and rcov >= sent_cover:
        return a, "sent_repeat"
    if top_ngram_share(a) >= ngram_max:
        return a, "ngram_loop"
    if trunc is not None and is_truncated(raw, trunc):
        return a, "truncated"          # generation hit its token budget (on the RAW answer)
    return a, None


def resolve_model(path, data, args):
    """Model key for tokenizer/budget: --model, else the filename stem, else the record field."""
    if args.model:
        return args.model
    cand = os.path.basename(path).split("_alt2025")[0].split("_clean")[0]
    if cand in TOK_REPO:
        return cand
    m = (data[0].get("model") if data else "") or ""
    m = m[:-len("-explicit")] if m.endswith("-explicit") else m
    return m if m in TOK_REPO else None


def process(path, args):
    data = json.load(open(path, encoding="utf-8"))
    trunc = None
    if args.truncation:
        model = resolve_model(path, data, args)
        if model is None:
            sys.exit(f"--truncation: cannot resolve model for {os.path.basename(path)}; pass --model")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(TOK_REPO[model])
            trunc = (tok, budget_for(model), args.trunc_margin, model in BROKEN_RT)
            print(f"  truncation: {model} budget={budget_for(model)}"
                  f"{' (ALLaM /2 band 1400-1600 + non-punct end)' if model in BROKEN_RT else ''}")
        except Exception as e:
            print(f"  WARNING: tokenizer load failed for {model} ({str(e)[:60]}); skipping truncation")

    kept, drops = [], {k: 0 for k in REASONS}
    samples = {k: [] for k in drops}
    survivors = []          # (comp, rep) for answers passing empty/foreign (calibration)
    for r in data:
        cleaned, reason = clean_one(r.get("answer", ""), args.eng_max, args.comp_min, args.rep_max,
                                    args.min_len, args.sent_rep, args.sent_cover, args.ngram_max, trunc)
        if cleaned and reason not in ("empty", "foreign"):
            survivors.append((comp_ratio(cleaned), word_rep(cleaned)))
        if reason is None:
            kept.append({**r, "answer": cleaned})
        else:
            drops[reason] += 1
            if len(samples[reason]) < args.show:
                samples[reason].append((r.get("id"), cleaned[:120]))

    name = os.path.basename(path)
    total = len(data)
    print(f"\n=== {name}  ({total} answers) ===")
    print(f"  kept {len(kept)}  ({100*len(kept)/total:.1f}%)  | dropped " +
          (", ".join(f"{k}={v}" for k, v in drops.items() if v) or "none"))
    if args.show:
        for k, ex in samples.items():
            for (i, t) in ex:
                print(f"    [{k}] {i}: {t!r}")
    if args.calibrate and survivors:
        comps = sorted(c for c, _ in survivors)
        reps = sorted(rp for _, rp in survivors)
        def pct(xs, p): return xs[min(len(xs) - 1, int(len(xs) * p))]
        print("  calibrate (over answers passing empty/foreign):")
        print("    compression  p01 {:.3f}  p05 {:.3f}  p10 {:.3f}  p25 {:.3f}  median {:.3f}"
              .format(pct(comps, .01), pct(comps, .05), pct(comps, .10), pct(comps, .25), statistics.median(comps)))
        print("    word-rep     p99 {:.3f}  p95 {:.3f}  p90 {:.3f}  p75 {:.3f}  median {:.3f}"
              .format(pct(reps, .99), pct(reps, .95), pct(reps, .90), pct(reps, .75), statistics.median(reps)))

    if not args.calibrate:
        out = args.output or path.replace(".json", "_clean.json")
        json.dump(kept, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  -> wrote {len(kept)} -> {out}")
    return total, len(kept), drops


def main():
    ap = argparse.ArgumentParser(description="Clean Stage-2 answers for span detection.")
    ap.add_argument("--input", nargs="+", required=True, help="answer JSON file(s) or glob(s)")
    ap.add_argument("--output", default=None, help="output path (single input only; default <in>_clean.json)")
    ap.add_argument("--eng-max", type=float, default=0.30, help="drop if foreign (Latin+CJK) letter ratio >= this")
    ap.add_argument("--comp-min", type=float, default=0.21, help="drop if zlib compression ratio < this (loop)")
    ap.add_argument("--rep-max", type=float, default=0.40, help="drop if repeated word-trigram ratio >= this")
    ap.add_argument("--sent-rep", type=int, default=4, help="drop if a >15-char sentence repeats >= this many times (with --sent-cover)")
    ap.add_argument("--sent-cover", type=float, default=0.10, help="...and those repeats fill >= this fraction of the answer")
    ap.add_argument("--ngram-max", type=float, default=0.12, help="drop if the top word-4gram's share >= this (min 30 words)")
    ap.add_argument("--min-len", type=int, default=0, help="drop if cleaned answer shorter than this (0=off)")
    ap.add_argument("--truncation", action="store_true", help="also drop budget-truncated answers (loads the model tokenizer)")
    ap.add_argument("--trunc-margin", type=int, default=8, help="truncated if RAW tokens >= budget - this margin")
    ap.add_argument("--model", default=None, help="override model key for tokenizer/budget (else inferred from filename)")
    ap.add_argument("--calibrate", action="store_true", help="don't write; print comp/rep distributions")
    ap.add_argument("--show", type=int, default=0, help="print N sample drops per reason")
    args = ap.parse_args()

    paths = []
    for pat in args.input:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    if args.output and len(paths) > 1:
        sys.exit("--output only valid with a single input file")

    grand = {k: 0 for k in REASONS}
    tot = kep = 0
    for p in paths:
        t, k, d = process(p, args)
        tot += t; kep += k
        for r in grand: grand[r] += d[r]
    if len(paths) > 1:
        print(f"\n=== TOTAL  {tot} answers -> kept {kep} ({100*kep/tot:.1f}%) ===")
        print("  dropped:", ", ".join(f"{k}={v}" for k, v in grand.items() if v))


if __name__ == "__main__":
    main()
