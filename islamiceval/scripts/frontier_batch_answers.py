# -*- coding: utf-8 -*-
"""
frontier_batch_answers.py
=========================
Generate frontier-model answers to the 500 Ansari test questions, via the async
BATCH APIs, then merge into one json ready for the annotation pipeline.

Models
  gpt-5.4         -> OpenAI Batch API           (/v1/chat/completions, 24h window)
  gemini-3.1-pro  -> Gemini Developer API batch (AI-Studio inline batch; no GCS)

Protocol (user-confirmed 2026-07-20): REASONING ON (reference-quality).
  - system prompt = the SAME alt2025-explicit prompt the local 500 used (extracted
    at runtime from inference_hugging_face.py so it stays byte-identical).
  - gpt-5.4        : reasoning_effort=medium, max_completion_tokens=8192
  - gemini-3.1-pro : default (dynamic) thinking, max_output_tokens=8192
  - RAW: no cleaning.

Join keys: OpenAI echoes custom_id=QID; Gemini inline preserves order + we also tag
each request metadata={"qid": QID}. Manifests under the workdir make poll/collect resumable.

Subcommands
  submit  --provider {openai,gemini} [--limit N] [--dry-run]
  poll    --provider {openai,gemini}
  collect --provider {openai,gemini}          # writes per-provider raw json
  merge                                        # -> ansari_frontier_answers.json

Usage
  python frontier_batch_answers.py submit --provider openai
  python frontier_batch_answers.py submit --provider gemini
  python frontier_batch_answers.py poll   --provider openai
  python frontier_batch_answers.py collect --provider gemini
  python frontier_batch_answers.py merge
"""
import os, re, sys, json, argparse
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

MODELS = {   # label -> provider model id
    "openai": ("gpt-5.4",        "gpt-5.4"),
    "gemini": ("gemini-3.1-pro", "gemini-3.1-pro-preview"),
}
MAX_TOKENS = 8192
GPT_REASONING = "medium"

# ── prompt/job presets ───────────────────────────────────────────────────────
# Each preset = the system-prompt variable to extract from inference_hugging_face.py
# (kept byte-identical to the local runs) + the input questions + output folder.
# `explicit` reproduces the original Ansari-500 job unchanged; `vanilla` runs the
# SAME two frontier models on the 500 annotated questions under the bare prompt,
# writing per-model files alongside the local vanilla_500 generations.
PRESETS = {
    "explicit": {
        "promptvar": "ALT_PROMPT_2025_explicit",
        "tag":       "alt2025-explicit",
        "input":     os.path.join(ROOT, "data", "islamiceval2026_RELEASE", "ansari_test_500.json"),
        "outdir":    os.path.join(ROOT, "outputs", "answers", "ansari_test_500"),
        "merged":    "ansari_frontier_answers.json",
    },
    "vanilla": {
        "promptvar": "VANILLA_PROMPT",
        "tag":       "vanilla",
        "input":     os.path.join(ROOT, "data", "islamiceval2026_RELEASE", "batch1_500_questions.json"),
        "outdir":    os.path.join(ROOT, "outputs", "answers", "vanilla_500"),
        "merged":    "frontier_vanilla_answers.json",
    },
}

# populated by configure()
INPUT = OUTDIR = WORKDIR = MERGED = SYSTEM = TAG = None


def _extract_prompt(varname):
    """Extract a system-prompt string literal verbatim from the HF inference script so
    the frontier answers use byte-identical instructions to the local runs (no import ->
    avoids pulling torch/transformers on the PC)."""
    src = open(os.path.join(HERE, "inference_hugging_face.py"), encoding="utf-8").read()
    m = re.search(varname + r'\s*=\s*(".*?"|\'.*?\'|""".*?"""|\'\'\'.*?\'\'\')', src, re.DOTALL)
    if not m:
        sys.exit(f"could not extract {varname} from inference_hugging_face.py")
    return eval(m.group(1))            # safe: matched a single string literal


def configure(prompt, input_override=None, outdir_override=None):
    global INPUT, OUTDIR, WORKDIR, MERGED, SYSTEM, TAG
    p = PRESETS[prompt]
    OUTDIR  = outdir_override or p["outdir"]
    INPUT   = input_override or p["input"]
    WORKDIR = os.path.join(OUTDIR, "frontier_batch")
    MERGED  = os.path.join(OUTDIR, p["merged"])
    SYSTEM  = _extract_prompt(p["promptvar"])
    TAG     = p["tag"]


def load_questions(limit=None):
    qs = json.load(open(INPUT, encoding="utf-8"))
    if limit:
        qs = qs[:limit]
    return qs   # [{id, prompt}, ...]


def manifest_path(provider):
    return os.path.join(WORKDIR, f"{provider}_manifest.json")


def save_manifest(provider, m):
    os.makedirs(WORKDIR, exist_ok=True)
    json.dump(m, open(manifest_path(provider), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def load_manifest(provider):
    p = manifest_path(provider)
    if not os.path.exists(p):
        sys.exit(f"no manifest at {p} — run `submit --provider {provider}` first.")
    return json.load(open(p, encoding="utf-8"))


def _write_per_model(rows, label, tag):
    """Also write a per-model file in the LOCAL generation schema
    ({id, prompt, answer, model}) into OUTDIR, so the frontier answers sit alongside
    the local `<model>_<tag>.json` files (e.g. vanilla_500/gpt-5.4_vanilla.json)."""
    local = [{"id": r["questionid"], "prompt": r["question"],
              "answer": r["answer"], "model": r["model"]} for r in rows]
    outp = os.path.join(OUTDIR, f"{label}_{tag}.json")
    json.dump(local, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[{rows[0]['model'] if rows else label}] per-model -> {os.path.relpath(outp, HERE)}")


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Batch API
# ─────────────────────────────────────────────────────────────────────────────
def _openai_client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set.")
    return OpenAI(api_key=key)


def submit_openai(qs, dry_run):
    label, model_id = MODELS["openai"]
    lines = [{
        "custom_id": q["id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_id,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": q["prompt"]}],
            "max_completion_tokens": MAX_TOKENS,
            "reasoning_effort": GPT_REASONING,
        },
    } for q in qs]

    jpath = os.path.join(WORKDIR, "input", "openai.jsonl")
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    with open(jpath, "w", encoding="utf-8") as f:
        for o in lines:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[openai] {len(lines)} requests -> {os.path.relpath(jpath, HERE)} "
          f"(model={model_id}, reasoning={GPT_REASONING}, max_tokens={MAX_TOKENS})")

    if dry_run:
        print("\n[dry-run] first request:\n" + json.dumps(lines[0], ensure_ascii=False, indent=2)[:900])
        return

    client = _openai_client()
    up = client.files.create(file=open(jpath, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"[openai] uploaded {up.id} -> batch {batch.id}  status={batch.status}")
    save_manifest("openai", {"created": datetime.now(timezone.utc).isoformat(),
                             "provider": "openai", "model": label, "model_id": model_id,
                             "tag": TAG, "batch_id": batch.id, "input_file_id": up.id,
                             "qids": [q["id"] for q in qs]})


def poll_openai():
    m = load_manifest("openai")
    client = _openai_client()
    b = client.batches.retrieve(m["batch_id"])
    rc = getattr(b, "request_counts", None)
    counts = f" total={rc.total} completed={rc.completed} failed={rc.failed}" if rc else ""
    print(f"[openai] batch {b.id}: {b.status}{counts}")
    return b.status


def collect_openai():
    m = load_manifest("openai")
    client = _openai_client()
    b = client.batches.retrieve(m["batch_id"])
    if b.status != "completed":
        print(f"[openai] status={b.status}; not collecting yet.")
        return None
    qmap = {q["id"]: q["prompt"] for q in load_questions()}
    raw = client.files.content(b.output_file_id).text
    dl = os.path.join(WORKDIR, "download"); os.makedirs(dl, exist_ok=True)
    open(os.path.join(dl, "openai.output.jsonl"), "w", encoding="utf-8").write(raw)
    rows, trunc = [], 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        qid = o.get("custom_id")
        resp = o.get("response") or {}
        body = resp.get("body") or {}
        ans, err = "", None
        if resp.get("status_code") == 200 and body.get("choices"):
            ch = body["choices"][0]
            ans = ch["message"]["content"] or ""
            if ch.get("finish_reason") == "length":
                trunc += 1
        else:
            err = f"http:{resp.get('status_code')}"
        rows.append({"questionid": qid, "question": qmap.get(qid, ""),
                     "answer": ans if ans else f"ERROR:{err or 'empty'}",
                     "model": m["model"]})
    outp = os.path.join(WORKDIR, "openai_answers.json")
    json.dump(rows, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    _write_per_model(rows, m["model"], m.get("tag") or TAG)
    ok = sum(1 for r in rows if not r["answer"].startswith("ERROR"))
    print(f"[openai] collected {len(rows)} ({ok} ok, {trunc} length-truncated) -> {os.path.relpath(outp, HERE)}")
    return outp


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Developer API batch (AI-Studio, inline)
# ─────────────────────────────────────────────────────────────────────────────
def _gemini_client():
    import google.genai as genai
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set.")
    return genai.Client(api_key=key)


def submit_gemini(qs, dry_run):
    from google.genai import types
    label, model_id = MODELS["gemini"]
    # REASONING ON: default (dynamic) thinking -> do NOT set thinking_config.
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=MAX_TOKENS)
    reqs = [types.InlinedRequest(contents=q["prompt"], config=cfg, metadata={"qid": q["id"]})
            for q in qs]
    print(f"[gemini] {len(reqs)} inline requests (model={model_id}, default thinking, "
          f"max_output_tokens={MAX_TOKENS})")

    if dry_run:
        print("\n[dry-run] first request contents:", repr(qs[0]["prompt"][:120]),
              "\n           metadata:", {"qid": qs[0]["id"]})
        return

    client = _gemini_client()
    job = client.batches.create(model=model_id, src=reqs,
                                config={"display_name": f"{TAG}_frontier_gemini"})
    print(f"[gemini] created batch {job.name}  state={job.state}")
    save_manifest("gemini", {"created": datetime.now(timezone.utc).isoformat(),
                             "provider": "gemini", "model": label, "model_id": model_id,
                             "tag": TAG, "job": job.name, "state": str(job.state),
                             "qids": [q["id"] for q in qs]})


def poll_gemini():
    m = load_manifest("gemini")
    client = _gemini_client()
    job = client.batches.get(name=m["job"])
    m["state"] = str(job.state); save_manifest("gemini", m)
    print(f"[gemini] batch {job.name}: {job.state}")
    return str(job.state)


def collect_gemini():
    m = load_manifest("gemini")
    client = _gemini_client()
    job = client.batches.get(name=m["job"])
    state = str(job.state)
    if "SUCCEEDED" not in state:
        print(f"[gemini] state={state}; not collecting yet.")
        return None
    qids = m["qids"]
    qmap = {q["id"]: q["prompt"] for q in load_questions()}
    dest = job.dest
    responses = getattr(dest, "inlined_responses", None) if dest else None
    if responses is None:
        sys.exit("[gemini] no inlined_responses on the completed job — inspect job.dest manually.")
    rows, trunc = [], 0
    for i, ir in enumerate(responses):
        qid = None
        if getattr(ir, "metadata", None):
            qid = ir.metadata.get("qid")
        if not qid:                                   # fall back to submission order
            qid = qids[i] if i < len(qids) else f"IDX{i}"
        ans, err = "", None
        if getattr(ir, "error", None):
            err = f"job_error:{getattr(ir.error,'message',ir.error)}"
        else:
            resp = ir.response
            try:
                ans = resp.text or ""
            except Exception as e:
                ans, err = "", f"no_text:{e}"
            fr = None
            try:
                fr = str(resp.candidates[0].finish_reason)
            except Exception:
                pass
            if fr and "MAX_TOKENS" in fr:
                trunc += 1
        rows.append({"questionid": qid, "question": qmap.get(qid, ""),
                     "answer": ans if ans else f"ERROR:{err or 'empty'}",
                     "model": m["model"]})
    outp = os.path.join(WORKDIR, "gemini_answers.json")
    json.dump(rows, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    _write_per_model(rows, m["model"], m.get("tag") or TAG)
    ok = sum(1 for r in rows if not r["answer"].startswith("ERROR"))
    print(f"[gemini] collected {len(rows)} ({ok} ok, {trunc} max-token-truncated) -> {os.path.relpath(outp, HERE)}")
    return outp


# ─────────────────────────────────────────────────────────────────────────────
_AR = re.compile(r'[؀-ۿ]')
def _bad_answer(a):
    """Return a drop-reason for an error/degenerate answer, else None (keep)."""
    a = (a or "").strip()
    if not a or a.startswith("ERROR"):        return "error/empty"
    if len(a) < 5:                            return "too_short"
    letters = [c for c in a if c.isalpha()]
    if letters and sum(1 for c in letters if _AR.match(c)) / len(letters) < 0.3:
        return "foreign"
    toks = a.split()
    if len(toks) >= 8 and len(set(toks)) <= 2: return "loop"
    return None


def cmd_merge(args):
    from collections import Counter
    kept, dropped = [], Counter()
    for prov in ("openai", "gemini"):
        p = os.path.join(WORKDIR, f"{prov}_answers.json")
        if not os.path.exists(p):
            print(f"  [warn] {prov}_answers.json missing — run collect --provider {prov} first.")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        for r in rows:
            reason = _bad_answer(r["answer"])
            if reason:
                dropped[f"{r['model']}:{reason}"] += 1
            else:
                kept.append({"questionid": r["questionid"], "question": r["question"],
                             "answer": r["answer"], "model": r["model"]})
    json.dump(kept, open(MERGED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"CLEANED merge: kept {len(kept)} rows -> {os.path.relpath(MERGED, HERE)}")
    print("per-model kept:", dict(Counter(r["model"] for r in kept)))
    if dropped:
        print("dropped:", dict(dropped))
    print("fields:", list(kept[0].keys()) if kept else "(none)")


def main():
    ap = argparse.ArgumentParser(description="Frontier answers (gpt-5.4 + gemini-3.1-pro) to a 500-question set via batch APIs")
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_common(sp):
        sp.add_argument("--prompt", choices=list(PRESETS), default="explicit",
                        help="preset: explicit=Ansari-500 (default) | vanilla=annotated-500 bare prompt")
        sp.add_argument("--input", default=None, help="override the questions json ([{id,prompt}])")
        sp.add_argument("--outdir", default=None, help="override the output folder")

    for name in ("submit", "poll", "collect"):
        sp = sub.add_parser(name)
        sp.add_argument("--provider", required=True, choices=["openai", "gemini"])
        add_common(sp)
        if name == "submit":
            sp.add_argument("--limit", type=int, default=None)
            sp.add_argument("--dry-run", action="store_true")
    add_common(sub.add_parser("merge"))
    args = ap.parse_args()

    configure(args.prompt, args.input, args.outdir)

    if args.mode == "merge":
        return cmd_merge(args)
    if args.mode == "submit":
        qs = load_questions(args.limit)
        (submit_openai if args.provider == "openai" else submit_gemini)(qs, args.dry_run)
    elif args.mode == "poll":
        (poll_openai if args.provider == "openai" else poll_gemini)()
    elif args.mode == "collect":
        (collect_openai if args.provider == "openai" else collect_gemini)()


if __name__ == "__main__":
    main()
