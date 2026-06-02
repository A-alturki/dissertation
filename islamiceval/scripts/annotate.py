"""
Stage 3 (Subtask A): Frontier-LLM span detection with the Gemini API.

Gemini reads each answer and returns every Qur'an / Hadith citation it finds as
{text, type, start, end}. We store its output VERBATIM — including the start/end
indices it produces. The point of the annotation tool is to *measure* how accurate
Gemini is, both at the source attribution (q/h, and whether the citation is real)
and at the character offsets, so we deliberately do NOT auto-correct anything.

Indices are sent to Gemini and stored against the answer text EXACTLY as it is in
the answers file (no normalization), so the annotation tool — which loads the same
file — shows the answer in the same frame Gemini saw.

Optional: pass --auto-index to (re)compute start/end in Python via exact / diacritic
-stripped search. This is a fallback to use only if Gemini's own indices turn out
unreliable; it is OFF by default.

Output (Format B, what the annotation tool consumes):
    {"spans": [{"id", "type", "text", "start", "end"}, ...]}

Usage:
    python annotate.py --input ../outputs/answers/allam-7b.json --limit 3   # smoke test
    python annotate.py --input ../outputs/answers/allam-7b.json
    python annotate.py --input ../outputs/answers/allam-7b.json --auto-index # opt-in matching

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in .env.
"""

import os, re, json, time, argparse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm.auto import tqdm

DEFAULT_MODEL = "gemini-3.1-pro-preview"

# ── Optional Python matching (only used with --auto-index) ──────────────────────
_DIAC = re.compile(r'[ً-ٰۖ-ۜ۟-۪ۤۧۨ-ۭ]')
def is_diac(c):    return bool(_DIAC.match(c))
def strip_diac(t): return _DIAC.sub('', t)

def find_in_answer(ans_text, query):
    """Return (start, end, method) for `query` in `ans_text`, in the same string
    frame as `ans_text`. exact -> diacritic-stripped fallback."""
    idx = ans_text.find(query)
    if idx != -1:
        end = idx + len(query)
        while end < len(ans_text) and is_diac(ans_text[end]):
            end += 1
        return idx, end, 'exact'

    a_s, q_s = strip_diac(ans_text), strip_diac(query)
    idx = a_s.find(q_s)
    if idx == -1:
        return None, None, None
    count, orig_start = 0, -1
    for i, c in enumerate(ans_text):
        if count == idx: orig_start = i; break
        if not is_diac(c): count += 1
    count, orig_end = 0, len(ans_text)
    for i in range(orig_start, len(ans_text)):
        if not is_diac(ans_text[i]):
            count += 1
            if count == len(q_s):
                j = i + 1
                while j < len(ans_text) and is_diac(ans_text[j]): j += 1
                orig_end = j; break
    return orig_start, orig_end, 'stripped'

# ── Gemini ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in the Qur'an and Hadith. You are given an Arabic answer produced by an LLM. Find every span in the answer that represents a quotation of the Quran or hadith, whether correct or hallucinated.

For each citation span return four fields:
- "text": the quoted Qur'an / Hadith text copied EXACTLY and VERBATIM from the answer, character-for-character. Do NOT add, drop, reorder, or normalize any diacritics (tashkeel). Copy it precisely as it appears.
  * Do NOT include surrounding quotation marks, curly braces { }, brackets, or parentheses.
  * Do NOT include the source reference that follows the quote, e.g. "(النحل: 90)" or "(رواه البخاري)".
  * Include ONLY the quoted scripture / narration text itself.
- "type": "q" for a Qur'an quotation, "h" for a Hadith quotation.
- "start": the 0-based index of the FIRST character of "text" within the answer string.
- "end": the 0-based index ONE PAST the last character of "text", so that answer[start:end] == text exactly.

Index every Unicode character, including spaces, punctuation, newlines, and diacritics. The answer string is given to you exactly as stored.

Only mark text the answer presents as an actual quotation. Do NOT mark paraphrase, commentary, or the model's own prose.

Respond with ONLY valid JSON in exactly this shape:
{"spans": [{"text": "...", "type": "q", "start": 0, "end": 0}]}
If there are no quotations, respond with {"spans": []}.
Do NOT output any other field, reasoning, or commentary."""

def _parse_json(raw):
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.split('```', 2)[1]
        if raw.startswith('json'): raw = raw[4:]
        raw = raw.strip().rstrip('`').strip()
    return json.loads(raw)

def detect_spans(client, model, question, answer, retries=3):
    """One Gemini call for one answer; returns its raw list of span dicts."""
    user = f"Question:\n{question}\n\nAnswer:\n{answer}"
    last = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return _parse_json(resp.text).get('spans', [])
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Subtask A span detection via Gemini")
    ap.add_argument("--input",      required=True, help="answers JSON [{id,prompt,answer,model}]")
    ap.add_argument("--output-dir", default="../outputs/annotations/")
    ap.add_argument("--model",      default=DEFAULT_MODEL)
    ap.add_argument("--limit",      type=int, default=None, help="annotate only first N answers")
    ap.add_argument("--auto-index", action="store_true",
                    help="OPT-IN: overwrite Gemini's start/end with a Python search "
                         "(exact / diacritic-stripped). Off by default.")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
    client = genai.Client(api_key=api_key)

    with open(args.input, encoding="utf-8") as f:
        answers = json.load(f)
    if not isinstance(answers, list):
        answers = [answers]
    if args.limit:
        answers = answers[:args.limit]

    model_name = os.path.basename(args.input).replace(".json", "")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{model_name}_gemini_spans.json")

    # Resume: keep ids already present in the output file
    all_spans, done = [], set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            all_spans = json.load(f).get("spans", [])
        done = {s["id"] for s in all_spans}

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"spans": all_spans}, f, ensure_ascii=False, indent=2)

    for item in tqdm(answers, desc=f"annotating {model_name}"):
        pid = item["id"]
        if pid in done:
            continue
        answer = item.get("answer", "")      # exactly as stored — no normalization
        try:
            raw_spans = detect_spans(client, args.model, item.get("prompt", ""), answer)
        except Exception as e:
            print(f"\nERROR id={pid}: {e}")
            all_spans.append({"id": pid, "type": "q", "text": "",
                              "start": None, "end": None, "error": str(e)})
            save()
            continue

        for sp in raw_spans:
            text  = sp.get("text", "")
            typ   = sp.get("type", "q")
            start = sp.get("start")
            end   = sp.get("end")
            if args.auto_index:                                  # opt-in fallback
                s, e, _ = find_in_answer(answer, text.strip())
                if s is not None:
                    start, end = s, e
            all_spans.append({"id": pid, "type": typ, "text": text,
                              "start": start, "end": end})
        save()

    n = len([s for s in all_spans if not s.get("error")])
    print(f"\nWrote {n} spans across {len(set(s['id'] for s in all_spans))} answers -> {out_path}")
    print(f"  Verify:  python ../tools/annotation_server.py {args.input} {out_path}")
    print(f"  (or load both files in tools/annotation_tool.html)")


if __name__ == "__main__":
    main()
