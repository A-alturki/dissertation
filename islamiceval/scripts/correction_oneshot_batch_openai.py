#!/usr/bin/env python3
"""
correction_oneshot_batch_openai.py
==================================
GPT one-shot correction (gate+correct in one call) via the **OpenAI Batch API**
(async, 24h window, ~50% cheaper). The GPT analogue of correction_oneshot_batch.py
(which is Gemini-only on Vertex batch). GPT was the stronger corrector, so this
batches it. Does NOT touch correction_pipeline.

Task-defining pieces are imported so results join back + reuse cp.save/cp.analyze:
oneshot_user, ONE_SHOT_CORRECTION_PROMPT, SCHEMA_CORRECTION, CORRECT_REASONING,
CORRECT_MAX_TOKENS from correction_pipeline; load_spans_multi + DEFAULT_INPUT from
correction_oneshot_batch; get_client/MODEL_REGISTRY/parse_json_response from annotate_spans.

Join is trivial here: OpenAI Batch echoes a `custom_id`, so custom_id = uid (no
line-index sidecar like Vertex needed). Body == the live chat.completions call:
strict json_schema (SCHEMA_CORRECTION keeps additionalProperties — OpenAI strict
REQUIRES it, unlike Vertex which rejects it), reasoning_effort=medium, 16k tokens.

Subcommands
-----------
  submit   build the batch JSONL (one line/span, custom_id=uid), upload via Files
           API, create the batch. `--dry-run` writes JSONL locally + prints the first
           request, no upload/no job.
  poll     retrieve the batch, print status + request counts.
  collect  download the output file, join by custom_id, write pipeline_results-shaped
           JSON, run analyze(stage="oneshot").

Usage
-----
  python correction_oneshot_batch_openai.py submit --limit 100 --dry-run
  python correction_oneshot_batch_openai.py submit --limit 100
  python correction_oneshot_batch_openai.py poll
  python correction_oneshot_batch_openai.py collect
"""
import os, sys, json, argparse
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import correction_pipeline as cp
from correction_pipeline import (
    oneshot_user, ONE_SHOT_CORRECTION_PROMPT, SCHEMA_CORRECTION,
    CORRECT_REASONING, CORRECT_MAX_TOKENS,
)
from correction_oneshot_batch import load_spans_multi, DEFAULT_INPUT, PROMPTS
from annotate_spans import get_client, MODEL_REGISTRY, parse_json_response

# PROMPTS (from the Vertex batch module) = {name: (raw_system_prompt, schema)}; alt/v3
# add the "paraphrase" confidence tier + v3 drops matched_run. The variant schemas keep
# additionalProperties:false, so they stay OpenAI-strict compatible.
ENDPOINT = "/v1/chat/completions"
DEFAULT_WORKDIR = os.path.join(HERE, "..", "outputs", "correction", "batch_gpt")
DEFAULT_RESULTS = os.path.join(HERE, "..", "outputs", "correction", "pipeline_oneshot_gpt.json")
TERMINAL = {"completed", "failed", "expired", "cancelled"}


# ─────────────────────────────────────────────────────────────────────────────
def build_line(sp, model_id, sys_prompt, schema):
    """One OpenAI Batch request line for a span (custom_id = uid → exact join)."""
    return {
        "custom_id": sp["uid"],
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": model_id,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": oneshot_user(sp["span"], sp["ref"])},
            ],
            "max_completion_tokens": CORRECT_MAX_TOKENS,
            "reasoning_effort": CORRECT_REASONING,   # "medium"
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "correction", "strict": True, "schema": schema},
            },
        },
    }


def write_jsonl(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in lines:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def manifest_path(workdir):
    return os.path.join(workdir, "batch_manifest.json")


# ─────────────────────────────────────────────────────────────────────────────
# submit
# ─────────────────────────────────────────────────────────────────────────────
def cmd_submit(args):
    if args.model not in MODEL_REGISTRY or MODEL_REGISTRY[args.model][0] != "openai":
        sys.exit(f"{args.model} is not an OpenAI model in MODEL_REGISTRY.")
    model_id = MODEL_REGISTRY[args.model][1]

    raw_prompt, schema = PROMPTS[args.prompt]
    sys_prompt = raw_prompt.format()                       # collapse doubled braces
    spans = load_spans_multi(args.input)
    if args.limit:
        spans = spans[:args.limit]
    lines = [build_line(sp, model_id, sys_prompt, schema) for sp in spans]
    print(f"spans: {len(spans)} | model: {args.model} ({model_id}) | prompt={args.prompt} | "
          f"reasoning={CORRECT_REASONING} max_tokens={CORRECT_MAX_TOKENS}")

    jpath = os.path.join(args.workdir, "input", f"{args.model}.jsonl")
    write_jsonl(jpath, lines)
    print(f"  wrote {os.path.relpath(jpath, HERE)}  ({len(lines)} requests)")

    if args.dry_run:
        print("\n─── DRY RUN: first request line (pretty) ───────────────────────")
        print(json.dumps(lines[0], ensure_ascii=False, indent=2))
        print("────────────────────────────────────────────────────────────────")
        print(f"\n[dry-run] no upload, no batch created. Re-run without --dry-run to launch.")
        return

    client = get_client("openai")
    if client is None:
        sys.exit("OPENAI_API_KEY not set.")
    up = client.files.create(file=open(jpath, "rb"), purpose="batch")
    print(f"  uploaded input file -> {up.id}")
    batch = client.batches.create(input_file_id=up.id, endpoint=ENDPOINT,
                                  completion_window=args.window)
    print(f"  created batch -> {batch.id}  status={batch.status}")

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "provider": "openai", "model": args.model, "model_id": model_id,
        "prompt": args.prompt,
        "batch_id": batch.id, "input_file_id": up.id, "status": batch.status,
        "window": args.window,
        "input": [os.path.abspath(p) for p in args.input],
        "results": os.path.abspath(args.results),
        "submitted_uids": [sp["uid"] for sp in spans],
    }
    os.makedirs(args.workdir, exist_ok=True)
    json.dump(manifest, open(manifest_path(args.workdir), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nwrote {os.path.relpath(manifest_path(args.workdir), HERE)} — resume anchor for poll/collect.")


# ─────────────────────────────────────────────────────────────────────────────
# poll
# ─────────────────────────────────────────────────────────────────────────────
def load_manifest(workdir):
    p = manifest_path(workdir)
    if not os.path.exists(p):
        sys.exit(f"No manifest at {p} — run `submit` (without --dry-run) first.")
    return json.load(open(p, encoding="utf-8")), p


def cmd_poll(args):
    manifest, p = load_manifest(args.workdir)
    client = get_client("openai")
    batch = client.batches.retrieve(manifest["batch_id"])
    manifest["status"] = batch.status
    json.dump(manifest, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    rc = getattr(batch, "request_counts", None)
    counts = f"  total={rc.total} completed={rc.completed} failed={rc.failed}" if rc else ""
    print(f"  {manifest['model']:12} batch {batch.id}: {batch.status}{counts}")
    if batch.status not in TERMINAL:
        print("…not terminal yet.")
        sys.exit(1)
    print("batch terminal.")


# ─────────────────────────────────────────────────────────────────────────────
# collect
# ─────────────────────────────────────────────────────────────────────────────
def _from_completion(body):
    """chat.completion body -> (prediction, raw_text, usage) or error."""
    try:
        txt = body["choices"][0]["message"]["content"]
    except Exception as e:
        return {"prediction": None, "raw_response": None, "usage": None, "error": f"no_content:{e}"}
    u = body.get("usage") or {}
    usage = {"input_tokens": u.get("prompt_tokens", 0) or 0,
             "output_tokens": u.get("completion_tokens", 0) or 0}
    return {"prediction": parse_json_response(txt), "raw_response": txt, "usage": usage}


def cmd_collect(args):
    manifest, _ = load_manifest(args.workdir)
    model = manifest["model"]
    client = get_client("openai")
    batch = client.batches.retrieve(manifest["batch_id"])
    if batch.status != "completed":
        print(f"  batch status={batch.status} (need 'completed'); nothing collected.")
        if batch.status not in TERMINAL:
            sys.exit(1)

    # restrict recs to submitted uids, carrying id/ref/error_type from the input
    by_uid = {sp["uid"]: sp for sp in load_spans_multi(manifest.get("input", args.input))}
    submitted = manifest.get("submitted_uids") or list(by_uid)
    recs = {uid: {**by_uid[uid], "gate": {}, "correction": {}, "oneshot": {}}
            for uid in submitted if uid in by_uid}

    def read_file(fid, tag):
        if not fid:
            return []
        raw = client.files.content(fid).text
        out_dir = os.path.join(args.workdir, "download")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, f"{model}.{tag}.jsonl"), "w", encoding="utf-8").write(raw)
        return [json.loads(l) for l in raw.splitlines() if l.strip()]

    out_lines = read_file(batch.output_file_id, "output")
    err_lines = read_file(getattr(batch, "error_file_id", None), "error")
    print(f"  {model}: {len(out_lines)} output + {len(err_lines)} error line(s)")

    matched = 0
    for line in out_lines:
        uid = line.get("custom_id")
        if uid not in recs:
            print(f"    ! custom_id {uid} not in submitted set; skipped")
            continue
        resp = line.get("response") or {}
        if resp.get("status_code") == 200 and resp.get("body"):
            recs[uid]["oneshot"][model] = _from_completion(resp["body"])
        else:
            recs[uid]["oneshot"][model] = {"prediction": None, "raw_response": None,
                                           "usage": None, "error": f"http:{resp.get('status_code')}"}
        matched += 1
    for line in err_lines:
        uid = line.get("custom_id")
        if uid in recs:
            recs[uid]["oneshot"][model] = {"prediction": None, "raw_response": None,
                                           "usage": None, "error": "request_failed"}
    print(f"    joined {matched}/{len(out_lines)}")

    cp.save(args.results, recs)
    cp.analyze(recs, stage="oneshot")
    print(f"\n  -> wrote {os.path.relpath(args.results, HERE)}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="GPT one-shot correction via OpenAI Batch API")
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default=DEFAULT_WORKDIR)
    common.add_argument("--input", nargs="+", default=DEFAULT_INPUT,
                        help="span CSV(s) (default = the 200-span set)")
    common.add_argument("--results", default=DEFAULT_RESULTS)

    ps = sub.add_parser("submit", parents=[common])
    ps.add_argument("--model", default="gpt-5.4")
    ps.add_argument("--prompt", choices=list(PROMPTS), default="oneshot",
                    help="correction system prompt (v3 = recovery-oriented, paraphrase tier)")
    ps.add_argument("--limit", type=int, default=None)
    ps.add_argument("--window", default="24h")
    ps.add_argument("--dry-run", action="store_true")
    ps.set_defaults(func=cmd_submit)

    pp = sub.add_parser("poll", parents=[common])
    pp.set_defaults(func=cmd_poll)

    pc = sub.add_parser("collect", parents=[common])
    pc.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
