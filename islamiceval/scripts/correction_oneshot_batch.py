#!/usr/bin/env python3
"""
correction_oneshot_batch.py
===========================
GEMINI-ONLY offline path for the ONE-SHOT correction stage (gate+correct in a
single call), run through Vertex AI **batch prediction** (async GCS job) instead
of live per-span API calls. Parallel to `correction_pipeline.py --stage oneshot`
(which stays the live path); this file never touches it.

GPT models are out of scope: any non-Gemini entry in CORRECT_MODELS is skipped.

Everything task-defining (spans, prompt, schema, reasoning/token budgets, uid
scheme) is IMPORTED from correction_pipeline so results join back to the CSV and
the existing analyze() runs unchanged.

Subcommands
-----------
  submit   build one JSONL request file per Gemini model (+ a keys sidecar),
           upload to GCS, create a Vertex batch job. `--dry-run` writes the
           JSONL + keys locally and stops (no GCS, no job) — use this to eyeball
           the schema/prompt before spending a real job.
  poll     read batch_manifest.json, report each job's state.
  collect  download prediction shards for SUCCEEDED jobs, parse + join back to
           uids, write pipeline_results-shaped JSON, run analyze(stage="oneshot").

Usage
-----
  python correction_oneshot_batch.py submit --limit 2 --dry-run
  python correction_oneshot_batch.py submit --project P --location us-central1 --bucket B
  python correction_oneshot_batch.py poll
  python correction_oneshot_batch.py collect
"""
import os, sys, csv, json, copy, argparse
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import correction_pipeline as cp
from correction_pipeline import (
    oneshot_user, ONE_SHOT_CORRECTION_PROMPT, GEMINI_CORRECTION_PROMPT_ALTERNATIVE,
    GEMINI_CORRECTION_PROMPT_V3, SCHEMA_CORRECTION, uid_for, REF_LABEL,
    CORRECT_REASONING, CORRECT_MAX_TOKENS, CORRECT_MODELS,
)
from annotate_spans import parse_json_response, _strip_additional_properties

# ── correction system prompts (+ the schema each expects) ────────────────────
# Newer prompts add a "paraphrase" confidence tier for semantic recoveries, so
# their schema must permit it (SCHEMA_CORRECTION only allows low/medium/high). V3
# also drops matched_run from the candidate output, so its schema drops it too.
def _schema_variant(extra_conf=(), drop=()):
    s = copy.deepcopy(SCHEMA_CORRECTION)
    it = s["properties"]["candidates"]["items"]
    for c in extra_conf:
        if c not in it["properties"]["confidence"]["enum"]:
            it["properties"]["confidence"]["enum"].append(c)
    for field in drop:
        it["properties"].pop(field, None)
        it["required"] = [r for r in it["required"] if r != field]
    return s

PROMPTS = {   # name -> (system prompt, schema)
    "oneshot": (ONE_SHOT_CORRECTION_PROMPT, SCHEMA_CORRECTION),
    "alt":     (GEMINI_CORRECTION_PROMPT_ALTERNATIVE, _schema_variant(extra_conf=["paraphrase"])),
    "v3":      (GEMINI_CORRECTION_PROMPT_V3, _schema_variant(extra_conf=["paraphrase"], drop=["matched_run"])),
}

# ─────────────────────────────────────────────────────────────────────────────
# constants derived from the live pipeline (kept identical for parity)
# ─────────────────────────────────────────────────────────────────────────────
# Gemini reasoning has no "effort" knob — map the pipeline's effort to a token budget.
REASONING_TO_THINKING = {"minimal": 0, "low": 1024, "medium": 8192, "high": 24576}
THINK_BUDGET  = REASONING_TO_THINKING.get(CORRECT_REASONING, 8192)

DEFAULT_WORKDIR = os.path.join(HERE, "..", "outputs", "correction", "batch")
_CORR           = os.path.join(HERE, "..", "outputs", "correction")
# the frozen 200-span human-eval set = original 100 + topup 100
DEFAULT_INPUT   = [os.path.join(_CORR, "sample_correction_final_classified.csv"),
                   os.path.join(_CORR, "sample_correction_topup_final.csv")]
DEFAULT_RESULTS = os.path.join(_CORR, "pipeline_oneshot_batch.json")


def load_spans_multi(paths):
    """Concatenate rows from one or more span CSVs, then index continuously so
    uids stay globally unique across the join (mirrors correction_pipeline.load_spans
    but over several files). uid_for uses the running line index i."""
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8-sig")))
    out = []
    for i, r in enumerate(rows):
        if r["ref"] not in REF_LABEL:
            continue
        out.append({"uid": uid_for(i, r), "id": r["id"], "answer_model": r["model"],
                    "ref": r["ref"], "span": r["span"], "error_type": r.get("error_type", "")})
    return out


def gemini_models():
    """Corrector models that route through Vertex batch (Gemini only)."""
    return [m for m in CORRECT_MODELS if "gemini" in m.lower()]


def sanitize(name):
    return name.replace("/", "_").replace(":", "_")


# ─────────────────────────────────────────────────────────────────────────────
# request building
# ─────────────────────────────────────────────────────────────────────────────
def build_request(span, sys_prompt, schema, use_thinking=True):
    """One Vertex batch `generateContent` request line for a single span."""
    gen = {
        "responseMimeType": "application/json",
        "responseSchema": _strip_additional_properties(schema),   # Vertex rejects additionalProperties
        "maxOutputTokens": CORRECT_MAX_TOKENS,
    }
    if use_thinking:
        gen["thinkingConfig"] = {"thinkingBudget": THINK_BUDGET}
    return {
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": oneshot_user(span["span"], span["ref"])}]}
            ],
            "systemInstruction": {"parts": [{"text": sys_prompt.format()}]},  # collapse doubled braces
            "generationConfig": gen,
        }
    }


def write_jsonl(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# submit
# ─────────────────────────────────────────────────────────────────────────────
def cmd_submit(args):
    spans = load_spans_multi(args.input)
    if args.limit:
        spans = spans[:args.limit]
    models = args.models or gemini_models()
    non_gemini = [m for m in models if "gemini" not in m.lower()]
    if non_gemini:
        sys.exit(f"This path is Gemini-only; drop {non_gemini}.")
    if not models:
        sys.exit("No Gemini model selected — nothing to do.")
    sys_prompt, schema = PROMPTS[args.prompt]
    print(f"spans: {len(spans)} | prompt: {args.prompt} | gemini models: {models} | "
          f"thinking_budget={THINK_BUDGET if not args.no_thinking else 'off'} "
          f"max_tokens={CORRECT_MAX_TOKENS}")

    in_dir   = os.path.join(args.workdir, "input")
    keys_dir = os.path.join(args.workdir, "keys")

    per_model = {}   # model -> {local_jsonl, keys_path, requests}
    for model in models:
        reqs, keys = [], []
        for i, sp in enumerate(spans):
            reqs.append(build_request(sp, sys_prompt, schema, use_thinking=not args.no_thinking))
            # sidecar: line index is the join key back to the uid (Vertex preserves
            # input order and does NOT echo a custom id inside `request`). span is
            # carried too so collect can re-validate the join by content.
            keys.append({"line": i, "uid": sp["uid"], "model": model, "span": sp["span"]})
        jpath = os.path.join(in_dir,   f"{sanitize(model)}.jsonl")
        kpath = os.path.join(keys_dir, f"{sanitize(model)}.keys.jsonl")
        write_jsonl(jpath, reqs)
        write_jsonl(kpath, keys)
        per_model[model] = {"local_jsonl": jpath, "keys_path": kpath, "n": len(reqs)}
        print(f"  wrote {os.path.relpath(jpath, HERE)}  ({len(reqs)} reqs) + keys sidecar")

    if args.dry_run:
        first_line = build_request(spans[0], sys_prompt, schema, use_thinking=not args.no_thinking)
        print("\n─── DRY RUN: first request line (pretty) ───────────────────────")
        print(json.dumps(first_line, ensure_ascii=False, indent=2))
        print("────────────────────────────────────────────────────────────────")
        print(f"\n[dry-run] wrote {len(spans)} reqs/model to {os.path.relpath(args.workdir, HERE)}; "
              f"no GCS upload, no job created. Re-run without --dry-run (and with "
              f"--project/--location/--bucket) to launch.")
        return

    # ── live: upload to GCS + create batch job ─────────────────────────────────
    project  = args.project  or os.environ.get("GCP_PROJECT")  or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = args.location or os.environ.get("GCP_LOCATION") or "us-central1"
    bucket   = args.bucket   or os.environ.get("GCS_BUCKET")
    if not (project and bucket):
        sys.exit("Live submit needs --project and --bucket (or GCP_PROJECT / GCS_BUCKET).")
    prefix = args.prefix

    from google.cloud import storage
    from google import genai
    from google.genai.types import HttpOptions, CreateBatchJobConfig

    client = genai.Client(vertexai=True, project=project, location=location,
                          http_options=HttpOptions(api_version="v1"))
    sc = storage.Client(project=project)
    gbucket = sc.bucket(bucket)

    manifest = {"created": datetime.now(timezone.utc).isoformat(), "project": project,
                "location": location, "bucket": bucket, "prefix": prefix,
                "prompt": args.prompt,
                "input": [os.path.abspath(p) for p in args.input],
                "results": os.path.abspath(args.results),
                "models": {}}
    for model, info in per_model.items():
        in_uri  = f"gs://{bucket}/{prefix}/input/{sanitize(model)}.jsonl"
        out_pfx = f"gs://{bucket}/{prefix}/output/{sanitize(model)}/"
        gbucket.blob(f"{prefix}/input/{sanitize(model)}.jsonl").upload_from_filename(info["local_jsonl"])
        print(f"  uploaded -> {in_uri}")
        job = client.batches.create(model=f"publishers/google/models/{model}",
                                    src=in_uri, config=CreateBatchJobConfig(dest=out_pfx))
        print(f"  created  -> {job.name}  state={job.state}")
        manifest["models"][model] = {"job": job.name, "src": in_uri, "dest": out_pfx,
                                     "keys_path": info["keys_path"], "n": info["n"],
                                     "state": str(job.state)}

    mpath = os.path.join(args.workdir, "batch_manifest.json")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {os.path.relpath(mpath, HERE)} — resume anchor for poll/collect.")


# ─────────────────────────────────────────────────────────────────────────────
# poll
# ─────────────────────────────────────────────────────────────────────────────
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
            "JOB_STATE_PAUSED", "JOB_STATE_EXPIRED"}


def load_manifest(workdir):
    mpath = os.path.join(workdir, "batch_manifest.json")
    if not os.path.exists(mpath):
        sys.exit(f"No manifest at {mpath} — run `submit` (without --dry-run) first.")
    return json.load(open(mpath, encoding="utf-8")), mpath


def cmd_poll(args):
    manifest, mpath = load_manifest(args.workdir)
    from google import genai
    from google.genai.types import HttpOptions
    client = genai.Client(vertexai=True, project=manifest["project"],
                          location=manifest["location"],
                          http_options=HttpOptions(api_version="v1"))
    all_done = True
    for model, info in manifest["models"].items():
        job = client.batches.get(name=info["job"])
        state = str(job.state)
        info["state"] = state
        done = state.split(".")[-1] in TERMINAL or any(t in state for t in TERMINAL)
        all_done = all_done and done
        print(f"  {model:26} {state}")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not all_done:
        print("…not all terminal yet.")
        sys.exit(1)
    print("all jobs terminal.")


# ─────────────────────────────────────────────────────────────────────────────
# collect
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text(resp):
    """text from a generateContent response dict, or (None, reason)."""
    try:
        cands = resp.get("candidates") or []
        if not cands:
            pf = (resp.get("promptFeedback") or {}).get("blockReason")
            return None, f"blocked:{pf}" if pf else "no_candidates"
        c0 = cands[0]
        parts = ((c0.get("content") or {}).get("parts")) or []
        txt = "".join(p.get("text", "") for p in parts)
        if not txt:
            return None, f"empty:{c0.get('finishReason')}"
        return txt, None
    except Exception as e:
        return None, f"parse_err:{e}"


def _usage(resp):
    um = resp.get("usageMetadata") or {}
    return {"input_tokens": um.get("promptTokenCount", 0) or 0,
            "output_tokens": um.get("candidatesTokenCount", 0) or 0}


def _download_shards(sc, dest_uri, local_dir):
    """dest_uri = gs://bucket/prefix/output/model/ -> list of local jsonl paths."""
    assert dest_uri.startswith("gs://")
    bkt, _, pfx = dest_uri[5:].partition("/")
    os.makedirs(local_dir, exist_ok=True)
    paths = []
    for blob in sc.list_blobs(bkt, prefix=pfx):
        if not blob.name.endswith(".jsonl"):
            continue
        lp = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(lp)
        paths.append(lp)
    return sorted(paths)


def cmd_collect(args):
    manifest, mpath = load_manifest(args.workdir)
    from google.cloud import storage
    from google import genai
    from google.genai.types import HttpOptions
    sc = storage.Client(project=manifest["project"])
    gclient = genai.Client(vertexai=True, project=manifest["project"],
                           location=manifest["location"],
                           http_options=HttpOptions(api_version="v1"))

    # refresh each job's live state (so collect works without a prior `poll`)
    for model, info in manifest["models"].items():
        try:
            info["state"] = str(gclient.batches.get(name=info["job"]).state)
        except Exception as e:
            print(f"  warn: could not refresh {model} state: {str(e)[:60]}")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # rebuild the span index so recs carry id/ref/error_type like the live path,
    # but restrict recs to the uids ACTUALLY submitted (per the keys sidecars) so
    # n reflects the batch size even when submit used --limit.
    by_uid = {sp["uid"]: sp for sp in load_spans_multi(manifest.get("input", args.input))}
    submitted = set()
    for info in manifest["models"].values():
        for l in open(info["keys_path"], encoding="utf-8"):
            if l.strip():
                submitted.add(json.loads(l)["uid"])
    recs = {uid: {**by_uid[uid], "gate": {}, "correction": {}, "oneshot": {}}
            for uid in submitted if uid in by_uid}

    for model, info in manifest["models"].items():
        if "SUCCEEDED" not in info.get("state", ""):
            print(f"  skip {model}: state={info.get('state')}")
            continue
        keys = [json.loads(l) for l in open(info["keys_path"], encoding="utf-8") if l.strip()]
        by_span = {k["span"]: k["uid"] for k in keys}          # fallback join
        local_dir = os.path.join(args.workdir, "download", sanitize(model))
        shards = _download_shards(sc, info["dest"], local_dir)
        out_lines = []
        for sp_path in shards:
            out_lines += [json.loads(l) for l in open(sp_path, encoding="utf-8") if l.strip()]
        print(f"  {model}: {len(out_lines)} predictions across {len(shards)} shard(s)")

        matched = 0
        for i, line in enumerate(out_lines):
            # primary join: line index within the (single-model) file
            k = keys[i] if i < len(keys) else None
            uid = k["uid"] if k else None
            # validate / fall back by span echoed in the request
            echoed = _echoed_span(line)
            if echoed is not None and echoed in by_span:
                if uid is None or by_span[echoed] != uid:
                    uid = by_span[echoed]
            if uid is None or uid not in recs:
                print(f"    ! line {i}: could not join to a uid; skipped")
                continue
            resp = line.get("response") or {}
            txt, err = _extract_text(resp)
            if err:
                recs[uid]["oneshot"][model] = {"prediction": None, "raw_response": None,
                                               "usage": _usage(resp), "error": err}
            else:
                recs[uid]["oneshot"][model] = {"prediction": parse_json_response(txt),
                                               "raw_response": txt, "usage": _usage(resp)}
            matched += 1
        print(f"    joined {matched}/{len(out_lines)}")

    cp.save(args.results, recs)
    cp.analyze(recs, stage="oneshot")
    print(f"\n  -> wrote {os.path.relpath(args.results, HERE)}")


def _echoed_span(line):
    """recover the span text from an output line's echoed request (<<< span >>>)."""
    try:
        txt = line["request"]["contents"][0]["parts"][0]["text"]
    except Exception:
        return None
    if "<<<" in txt and ">>>" in txt:
        return txt.split("<<<", 1)[1].rsplit(">>>", 1)[0].strip("\n")
    return None


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Gemini-only batched one-shot correction (Vertex batch prediction)")
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default=DEFAULT_WORKDIR)
    common.add_argument("--input", nargs="+", default=DEFAULT_INPUT,
                        help="one or more span CSVs (default = the 200-span set)")
    common.add_argument("--results", default=DEFAULT_RESULTS)

    ps = sub.add_parser("submit", parents=[common])
    ps.add_argument("--limit", type=int, default=None)
    ps.add_argument("--prompt", choices=list(PROMPTS), default="oneshot",
                    help="which correction system prompt (alt = paraphrase-tier prompt)")
    ps.add_argument("--models", nargs="+", default=None,
                    help="gemini models to run (default = CORRECT_MODELS gemini subset)")
    ps.add_argument("--dry-run", action="store_true", help="write JSONL + keys locally, no GCS, no job")
    ps.add_argument("--no-thinking", action="store_true", help="omit thinkingConfig (if model rejects it)")
    ps.add_argument("--project", default=None)
    ps.add_argument("--location", default=None)
    ps.add_argument("--bucket", default=None)
    ps.add_argument("--prefix", default="islamiceval/correction_oneshot")
    ps.set_defaults(func=cmd_submit)

    pp = sub.add_parser("poll", parents=[common])
    pp.set_defaults(func=cmd_poll)

    pc = sub.add_parser("collect", parents=[common])
    pc.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
