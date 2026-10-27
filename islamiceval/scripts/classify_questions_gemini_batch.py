#!/usr/bin/env python3
"""
classify_questions_gemini_batch.py
==================================
Question taxonomy re-run with the DISSERTATION prompt (12 categories +
difficulty 1-5 + reasoning/multi_step booleans, NO divergence), batched through
Vertex AI batch prediction on Gemini (same GCS path as correction_oneshot_batch).

Input : outputs/classification/questions_final_cleaned.json  (7,700 {qid,prompt})
Output: outputs/classification/labels_<model>_12cat.json
        [{"id","prompt","difficulty","reasoning","multi_step","category_type"}, ...]

Subcommands
-----------
  submit   build the request JSONL (+ keys sidecar), upload to GCS, create the
           batch job. `--dry-run` writes the JSONL locally and stops.
  poll     report the job state.
  collect  download prediction shards, join by line index, write the labels
           JSON and print the per-category summary table.

Usage
-----
  python classify_questions_gemini_batch.py submit --limit 5 --dry-run
  python classify_questions_gemini_batch.py submit
  python classify_questions_gemini_batch.py poll
  python classify_questions_gemini_batch.py collect
"""
import os, sys, json, argparse, re
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL   = "gemini-3.5-flash"
DEFAULT_PROJECT = "islamiceval-500212"
DEFAULT_LOCATION = "global"          # gemini-3.x models are only on the global endpoint
DEFAULT_BUCKET  = "tueki-islamiceval-batch"
DEFAULT_PREFIX  = "islamiceval/classify_12cat"

DEFAULT_INPUT   = os.path.join(HERE, "..", "outputs", "classification", "questions_final_cleaned.json")
DEFAULT_WORKDIR = os.path.join(HERE, "..", "outputs", "classification", "batch_12cat")
DEFAULT_RESULTS = os.path.join(HERE, "..", "outputs", "classification", f"labels_{DEFAULT_MODEL}_12cat.json")

CATEGORIES = [
    "Islamic Creed", "Jurisprudence", "Inheritance Law", "Hadith Studies",
    "Qur'anic Studies", "Prophetic Biography", "Islamic History",
    "Islamic Ethics and Morality", "Islamic Finance and Economics",
    "Islamic Family Law", "Comparative Religion", "Contemporary Issues",
]

# The dissertation prompt, verbatim.
SYSTEM_PROMPT = """You are an expert evaluator of Islamic
knowledge questions. Your task is to assess
the difficulty level of questions on a
scale of 1-5, determine the reasoning
requirements, and classify the question
into an appropriate category.
Difficulty Scale:
1 = Very Easy: Basic factual recall, simple
definitions, or straightforward yes/no
questions
2 = Easy: Requires basic understanding of
concepts with minimal reasoning
3 = Moderate: Requires understanding multiple
concepts and some analytical reasoning
4 = Hard: Requires deep understanding,
synthesis of multiple sources, and nuanced
reasoning
5 = Very Hard: Requires expert-level analysis,
balancing competing interests, and
consideration of complex ethical frameworks
Reasoning Assessment:
- reasoning: Does answering this question
require reasoning beyond simple recall?
(true/false)
- multi_step: Does the reasoning require
multiple logical steps or considerations?
(true/false)
Examples of multi-step: comparing multiple
sources, weighing competing principles,
applying rules to specific contexts,
building logical chains
Category Classification:
Classify the question into ONE of these
categories:
1. "Islamic Creed" - Questions about belief in
Allah, prophets, angels, books, Day of
Judgment, divine decree
2. "Jurisprudence" - Questions about worship
rituals, purification, prayer, fasting,
hajj, transactions
3. "Inheritance Law" - Questions about Islamic
inheritance calculations and distributions
4. "Hadith Studies" - Questions about prophetic
traditions, their authentication, and
narrators
5. "Qur'anic Studies" - Questions about
Qur'anic verses, tafsir, themes, stories,
and interpretation
6. "Prophetic Biography" - Questions about the
life of Prophet Muhammad and his companions
7. "Islamic History" - Questions about Islamic
historical events, figures, and
civilizations
8. "Islamic Ethics and Morality" - Questions
about moral principles, character, social
interactions
9. "Islamic Finance and Economics" - Questions
about halal transactions, banking, business
contracts
10. "Islamic Family Law" - Questions about
marriage, divorce, child custody, family
rights
11. "Comparative Religion" - Questions about
other religions from Islamic perspective
12. "Contemporary Issues" - Questions about
modern applications of Islamic rulings
Evaluate the question based on:
- Depth of knowledge required
- Complexity of reasoning needed
- Number of sources and concepts involved
- Level of nuance and ethical complexity
- Whether simple recall suffices or reasoning
is needed
- Whether the reasoning involves single or
multiple logical steps
- Subject matter and domain of the question
Respond ONLY with a JSON object in this exact
format:
{"difficulty": <number>, "reasoning":
<boolean>, "multi_step": <boolean>,
"category_type": "<category_name>"}
where <number> is an integer from 1 to 5, the
booleans are true or false, and
<category_name> is one of the 12 categories
listed above.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "difficulty":    {"type": "integer"},
        "reasoning":     {"type": "boolean"},
        "multi_step":    {"type": "boolean"},
        "category_type": {"type": "string", "enum": CATEGORIES},
    },
    "required": ["difficulty", "reasoning", "multi_step", "category_type"],
}

MAX_OUTPUT_TOKENS = 4096   # headroom for thinking tokens (counted in the budget)


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [{"id": q["qid"], "prompt": q["prompt"]} for q in json.load(f)]


def build_request(prompt_text, thinking_budget=None):
    gen = {
        "responseMimeType": "application/json",
        "responseSchema": SCHEMA,
        "maxOutputTokens": MAX_OUTPUT_TOKENS,
    }
    if thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": gen,
        }
    }


def write_jsonl(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ── submit ───────────────────────────────────────────────────────────────────
def cmd_submit(args):
    qs = load_questions(args.input)
    if args.limit:
        qs = qs[:args.limit]
    tb = args.thinking_budget
    print(f"questions: {len(qs)} | model: {args.model} | thinking_budget={tb}")

    reqs = [build_request(q["prompt"], tb) for q in qs]
    keys = [{"line": i, "id": q["id"], "prompt": q["prompt"]} for i, q in enumerate(qs)]
    jpath = os.path.join(args.workdir, "input", "requests.jsonl")
    kpath = os.path.join(args.workdir, "keys", "keys.jsonl")
    write_jsonl(jpath, reqs)
    write_jsonl(kpath, keys)
    print(f"wrote {os.path.relpath(jpath, HERE)} (+ keys sidecar)")

    if args.dry_run:
        print("\n─── DRY RUN: first request line ───")
        print(json.dumps(reqs[0], ensure_ascii=False, indent=2))
        print("[dry-run] no GCS upload, no job created.")
        return

    from google.cloud import storage
    from google import genai
    from google.genai.types import HttpOptions, CreateBatchJobConfig

    client = genai.Client(vertexai=True, project=args.project, location=args.location,
                          http_options=HttpOptions(api_version="v1"))
    sc = storage.Client(project=args.project)

    in_uri  = f"gs://{args.bucket}/{args.prefix}/input/requests.jsonl"
    out_pfx = f"gs://{args.bucket}/{args.prefix}/output/"
    sc.bucket(args.bucket).blob(f"{args.prefix}/input/requests.jsonl").upload_from_filename(jpath)
    print(f"uploaded -> {in_uri}")
    job = client.batches.create(model=f"publishers/google/models/{args.model}",
                                src=in_uri, config=CreateBatchJobConfig(dest=out_pfx))
    print(f"created  -> {job.name}  state={job.state}")

    manifest = {"created": datetime.now(timezone.utc).isoformat(), "project": args.project,
                "location": args.location, "bucket": args.bucket, "prefix": args.prefix,
                "model": args.model, "job": job.name, "src": in_uri, "dest": out_pfx,
                "keys_path": os.path.abspath(kpath), "n": len(reqs),
                "input": os.path.abspath(args.input),
                "results": os.path.abspath(args.results), "state": str(job.state)}
    mpath = os.path.join(args.workdir, "batch_manifest.json")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {os.path.relpath(mpath, HERE)}")


# ── poll ─────────────────────────────────────────────────────────────────────
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
            "JOB_STATE_PAUSED", "JOB_STATE_EXPIRED"}


def load_manifest(workdir):
    mpath = os.path.join(workdir, "batch_manifest.json")
    if not os.path.exists(mpath):
        sys.exit(f"No manifest at {mpath} — run `submit` first.")
    return json.load(open(mpath, encoding="utf-8")), mpath


def cmd_poll(args):
    manifest, mpath = load_manifest(args.workdir)
    from google import genai
    from google.genai.types import HttpOptions
    client = genai.Client(vertexai=True, project=manifest["project"],
                          location=manifest["location"],
                          http_options=HttpOptions(api_version="v1"))
    job = client.batches.get(name=manifest["job"])
    state = str(job.state)
    manifest["state"] = state
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{manifest['model']}: {state}")
    if not any(t in state for t in TERMINAL):
        sys.exit(1)


# ── collect ──────────────────────────────────────────────────────────────────
def _extract_text(resp):
    try:
        cands = resp.get("candidates") or []
        if not cands:
            pf = (resp.get("promptFeedback") or {}).get("blockReason")
            return None, f"blocked:{pf}" if pf else "no_candidates"
        c0 = cands[0]
        parts = ((c0.get("content") or {}).get("parts")) or []
        txt = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not txt:
            return None, f"empty:{c0.get('finishReason')}"
        return txt, None
    except Exception as e:
        return None, f"parse_err:{e}"


def _parse_labels(txt):
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
    obj = json.loads(txt)
    d = int(obj["difficulty"])
    if not 1 <= d <= 5:
        raise ValueError(f"difficulty out of range: {d}")
    cat = obj["category_type"]
    if cat not in CATEGORIES:
        raise ValueError(f"bad category: {cat}")
    return {"difficulty": d, "reasoning": bool(obj["reasoning"]),
            "multi_step": bool(obj["multi_step"]), "category_type": cat}


def cmd_collect(args):
    manifest, mpath = load_manifest(args.workdir)
    from google.cloud import storage
    from google import genai
    from google.genai.types import HttpOptions
    gclient = genai.Client(vertexai=True, project=manifest["project"],
                           location=manifest["location"],
                           http_options=HttpOptions(api_version="v1"))
    manifest["state"] = str(gclient.batches.get(name=manifest["job"]).state)
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if "SUCCEEDED" not in manifest["state"]:
        sys.exit(f"job state = {manifest['state']} — not collecting.")

    sc = storage.Client(project=manifest["project"])
    dest = manifest["dest"]
    bkt, _, pfx = dest[5:].partition("/")
    local_dir = os.path.join(args.workdir, "download")
    os.makedirs(local_dir, exist_ok=True)
    shards = []
    for blob in sc.list_blobs(bkt, prefix=pfx):
        if blob.name.endswith(".jsonl"):
            lp = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(lp)
            shards.append(lp)
    shards.sort()

    keys = [json.loads(l) for l in open(manifest["keys_path"], encoding="utf-8") if l.strip()]
    by_prompt = {k["prompt"]: k for k in keys}
    out_lines = []
    for sp in shards:
        out_lines += [json.loads(l) for l in open(sp, encoding="utf-8") if l.strip()]
    print(f"{len(out_lines)} predictions across {len(shards)} shard(s)")

    results, errors = {}, 0
    for i, line in enumerate(out_lines):
        # join primarily by the question text echoed in the request (shards may
        # not preserve global input order); fall back to line index.
        k = None
        try:
            echoed = line["request"]["contents"][0]["parts"][0]["text"]
            k = by_prompt.get(echoed)
        except Exception:
            pass
        if k is None and i < len(keys):
            k = keys[i]
        if k is None:
            print(f"  ! line {i}: no join"); errors += 1; continue
        rec = {"id": k["id"], "prompt": k["prompt"]}
        txt, err = _extract_text(line.get("response") or {})
        if err:
            rec["error"] = err; errors += 1
        else:
            try:
                rec.update(_parse_labels(txt))
            except Exception as e:
                rec["error"] = f"bad_json:{str(e)[:120]}"; rec["raw"] = txt[:300]; errors += 1
        results[rec["id"]] = rec

    # keep input order
    ordered = [results[k["id"]] for k in keys if k["id"] in results]
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    print(f"wrote {os.path.relpath(args.results, HERE)}  ({len(ordered)} records, {errors} errors)")
    summarize(ordered)


def summarize(recs):
    ok = [r for r in recs if "category_type" in r]
    if not ok:
        return
    from collections import defaultdict
    by = defaultdict(list)
    for r in ok:
        by[r["category_type"]].append(r)
    n = len(ok)
    print(f"\n{'Category':32}{'n':>6}{'%':>7}{'meanD':>7}{'%D>=4':>7}{'%reas':>7}{'%multi':>7}")
    rows = sorted(by.items(), key=lambda kv: -len(kv[1]))
    for cat, rs in rows:
        m = sum(r["difficulty"] for r in rs) / len(rs)
        hd = 100 * sum(r["difficulty"] >= 4 for r in rs) / len(rs)
        re_ = 100 * sum(r["reasoning"] for r in rs) / len(rs)
        ms = 100 * sum(r["multi_step"] for r in rs) / len(rs)
        print(f"{cat:32}{len(rs):>6}{100*len(rs)/n:>7.1f}{m:>7.2f}{hd:>7.0f}{re_:>7.0f}{ms:>7.0f}")
    m = sum(r["difficulty"] for r in ok) / n
    hd = 100 * sum(r["difficulty"] >= 4 for r in ok) / n
    re_ = 100 * sum(r["reasoning"] for r in ok) / n
    ms = 100 * sum(r["multi_step"] for r in ok) / n
    print(f"{'TOTAL':32}{n:>6}{100.0:>7.1f}{m:>7.2f}{hd:>7.0f}{re_:>7.0f}{ms:>7.0f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Gemini Vertex-batch question taxonomy (12 categories, no divergence)")
    sub = ap.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default=DEFAULT_WORKDIR)
    common.add_argument("--input", default=DEFAULT_INPUT)
    common.add_argument("--results", default=DEFAULT_RESULTS)

    ps = sub.add_parser("submit", parents=[common])
    ps.add_argument("--model", default=DEFAULT_MODEL)
    ps.add_argument("--limit", type=int, default=None)
    ps.add_argument("--thinking-budget", type=int, default=None,
                    help="thinkingConfig.thinkingBudget (default: omit = model default)")
    ps.add_argument("--dry-run", action="store_true")
    ps.add_argument("--project", default=DEFAULT_PROJECT)
    ps.add_argument("--location", default=DEFAULT_LOCATION)
    ps.add_argument("--bucket", default=DEFAULT_BUCKET)
    ps.add_argument("--prefix", default=DEFAULT_PREFIX)
    ps.set_defaults(func=cmd_submit)

    pp = sub.add_parser("poll", parents=[common])
    pp.set_defaults(func=cmd_poll)

    pc = sub.add_parser("collect", parents=[common])
    pc.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
