#!/usr/bin/env python3
"""
annotate_spans.py
=================
Frontier-LLM span annotation for IslamicEval.

Reads a JSON file of model answers ([{id, prompt, answer, model}, ...]) and, for
each answer, asks a frontier model (from MODEL_REGISTRY) to mark the Qur'an /
Hadith citation spans. Output is written incrementally and is resumable.

There are NO gold labels and NO scoring here -- this generates annotations for
later human verification.

The task is defined by the TASK CONFIG block below (system prompt + JSON schema +
user-prompt builder). Edit that block to change the prompt / task.

Usage:
    python annotate_spans.py --model gemini-3.1-pro-preview --input answers.json --output spans.json
    python annotate_spans.py --model gpt-5 --input answers.json --output spans.json --fix-indices

Requires the relevant provider API key in .env (see MODEL_REGISTRY).
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Each entry: "safe_name" → (provider, model_id, api_key_env_var)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # OpenAI models
    "gpt-5.4":       ("openai",    "gpt-5.4",                   "OPENAI_API_KEY"),
    "gpt-5.4-mini":   ("openai",    "gpt-5-mini",               "OPENAI_API_KEY"),
    "gpt-5.4-nano":         ("openai",    "gpt-5.4-nano",                   "OPENAI_API_KEY"),
    "gpt-5":               ("openai",    "gpt-5",                         "OPENAI_API_KEY"),
    # Anthropic Claude models
    "claude-opus-4-6": ("anthropic", "claude-opus-4-6", "ANTHROPIC_API_KEY"),
    "claude-sonnet-4-6":   ("anthropic", "claude-sonnet-4-6",   "ANTHROPIC_API_KEY"),
    "claude-haiku-4-5":   ("anthropic", "claude-haiku-4-5",   "ANTHROPIC_API_KEY"),
    # Google Gemini models
    "gemini-3.1-pro-preview": ("gemini", "gemini-3.1-pro-preview", "GEMINI_API_KEY"),
    "gemini-3-flash-preview":   ("gemini", "gemini-3-flash-preview",   "GEMINI_API_KEY"),
    # Together AI (Qwen3 large — thinking disabled via enable_thinking=False)
    "Qwen3-235B":  ("together",    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "TOGETHER_API_KEY"),
    "Qwen3-80B":   ("together",    "Qwen/Qwen3-Next-80B-A3B-Instruct",        "TOGETHER_API_KEY"),
    # OpenRouter (Qwen3 small — thinking disabled via thinking.enabled=False)
    "Qwen3-32B":   ("openrouter", "qwen/qwen3-32b",  "OPENROUTER_API_KEY"),
    "Qwen3-14B":   ("openrouter", "qwen/qwen3-14b",  "OPENROUTER_API_KEY"),
    "Qwen3-8B":    ("openrouter", "qwen/qwen3-8b",   "OPENROUTER_API_KEY"),
}

# Sleep time between API calls per provider (to respect rate limits)
# Gemini Tier 1 (paid) has high RPM — 1s is sufficient for all Gemini models
SLEEP_TIMES = {
    "openai":      0.5,
    "anthropic":   1.0,
    "gemini":      1.0,
    "together":    1.0,
    "openrouter":  1.0,
}

# No longer needed — was for Gemini 2.5 Pro free tier (2 RPM). Tier 1 is much higher.
SLEEP_TIME_GEMINI_PRO = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT INITIALISATION
# Clients are created lazily on first use, based on available environment keys.
# ─────────────────────────────────────────────────────────────────────────────
_clients: Dict[str, Any] = {}  # provider → client object


def get_client(provider: str) -> Optional[Any]:
    """Return (and cache) the API client for a given provider.

    Returns None if the required environment variable is missing.
    """
    if provider in _clients:
        return _clients[provider]

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key)
        _clients[provider] = client
        return client

    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        _clients[provider] = client
        return client

    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        # New google-genai SDK (pip install google-genai)
        from google import genai
        client = genai.Client(api_key=key)
        _clients[provider] = client
        return client

    elif provider == "together":
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
        _clients[provider] = client
        return client

    elif provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        _clients[provider] = client
        return client

    return None


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED API CALL with retry + exponential back-off
# ─────────────────────────────────────────────────────────────────────────────

def call_api(model_name: str, system_msg: str, user_prompt: str,
             max_retries: int = 3, schema: Optional[Dict] = None):
    """Call a model and return (text, usage) where usage = {"input_tokens": N, "output_tokens": N}.

    Handles all four providers uniformly.  On failure retries with exponential
    back-off (2 s, 4 s, 8 s).  Returns (None, None) on permanent failure.

    Args:
        model_name: Key into MODEL_REGISTRY (e.g. "gpt-4.1-mini").
        system_msg: System / persona instruction.
        user_prompt: The task prompt sent as the user message.
        max_retries: Maximum number of retry attempts.

    Returns:
        Raw model response text, or None on failure.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    provider, model_id, _ = MODEL_REGISTRY[model_name]
    client = get_client(provider)
    if client is None:
        raise RuntimeError(f"No API key for provider '{provider}' (model={model_name})")

    # Determine sleep time for this provider/model
    sleep_time = SLEEP_TIME_GEMINI_PRO if "pro" in model_name.lower() else SLEEP_TIMES.get(provider, 1.0)

    for attempt in range(max_retries):
        try:
            text, usage = _do_api_call(provider, client, model_id, system_msg, user_prompt, schema=schema)
            time.sleep(sleep_time)  # rate-limit pause after each successful call
            return text, usage

        except Exception as exc:
            wait = 2 ** (attempt + 1)  # 2 s, 4 s, 8 s
            print(f"    [WARN] API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                print(f"    [WARN] Retrying in {wait} s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] All {max_retries} attempts failed for {model_name}.")
                return None, None


def _strip_additional_properties(schema: Dict) -> Dict:
    """Recursively remove 'additionalProperties' from a JSON schema dict.

    Gemini's response_schema does not support this OpenAI-specific field
    and will return a 400 INVALID_ARGUMENT error if it is present.
    """
    if not isinstance(schema, dict):
        return schema
    return {
        k: _strip_additional_properties(v)
        for k, v in schema.items()
        if k != "additionalProperties"
    }


def _do_api_call(provider: str, client: Any, model_id: str,
                 system_msg: str, user_prompt: str,
                 schema: Optional[Dict] = None):
    """Low-level API dispatch.  Raises on error.  Returns (text, usage).

    usage = {"input_tokens": int, "output_tokens": int}

    Args:
        schema: Optional JSON schema dict.  When provided:
                  - OpenAI: enforced via response_format (structured outputs)
                  - Gemini: enforced via response_schema in generation_config
                  - Anthropic/Together: ignored (schema is described in the system prompt)
    """

    if provider in ("openai", "together", "openrouter"):
        # OpenAI SDK (also used for Together AI via OpenAI-compatible endpoint).
        # o-series and gpt-5+ use max_completion_tokens instead of max_tokens/temperature.
        COMPLETION_TOKENS_PREFIXES = ("o1", "o3", "o4", "o5", "gpt-5")
        is_reasoning = any(model_id.startswith(p) for p in COMPLETION_TOKENS_PREFIXES)

        call_kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_prompt},
            ],
        )
        if is_reasoning:
            call_kwargs["max_completion_tokens"] = 4096
            # reasoning models ignore temperature; omit it to avoid API errors
        else:
            call_kwargs["max_tokens"] = 4096
            call_kwargs["temperature"] = 0.1

        # Structured output: only for OpenAI (OpenRouter may not support strict json_schema)
        if schema and provider == "openai":
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name":   "span_detection",
                    "strict": True,
                    "schema": schema,
                },
            }

        if model_id.startswith("gpt-5.4-pro"):
            # print("im here")
            # response = client.responses.create(**call_kwargs)
            response = client.responses.create(
                model=model_id,
                instructions=system_msg,
                input=user_prompt,
                max_output_tokens=4096,
                text={
                    "format": {
                        "type": "json_schema",
                        "name":   "span_detection",
                        "strict": True,
                        "schema": schema,
                    }
                }
            )
            usage = {"input_tokens": getattr(response.usage, "input_tokens", None),
                     "output_tokens": getattr(response.usage, "output_tokens", None)}
            return response.output_text, usage


        else:
            # Disable Qwen3 thinking mode so the answer goes into message.content.
            # Together AI uses enable_thinking=False; OpenRouter uses thinking.enabled=False.
            if provider == "together":
                call_kwargs["extra_body"] = {"enable_thinking": False}
            elif provider == "openrouter":
                call_kwargs["extra_body"] = {"thinking": {"enabled": False}}
            response = client.chat.completions.create(**call_kwargs)
            usage = {"input_tokens": response.usage.prompt_tokens,
                     "output_tokens": response.usage.completion_tokens}
            return response.choices[0].message.content, usage

    elif provider == "anthropic":
        # Anthropic SDK — system message is a separate top-level parameter.
        # No native JSON schema enforcement; the system prompt describes the format.
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1,
        )
        usage = {"input_tokens": response.usage.input_tokens,
                 "output_tokens": response.usage.output_tokens}
        return response.content[0].text, usage

    elif provider == "gemini":
        # New google-genai SDK: client was created in get_client()
        from google.genai import types as genai_types

        # Build config — add JSON schema enforcement if provided
        config_kwargs = {
            "temperature": 0.1,
            "system_instruction": system_msg,
        }
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            # Gemini does not support "additionalProperties" — strip it recursively
            config_kwargs["response_schema"]    = _strip_additional_properties(schema)

        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        um = response.usage_metadata
        usage = {"input_tokens": um.prompt_token_count,
                 "output_tokens": um.candidates_token_count}
        return response.text, usage

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — robust extraction from messy LLM output
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> Optional[Dict]:
    """Parse JSON from a model response that may contain markdown fences or prose.

    Strategy:
      1. Strip ```json ... ``` (or ``` ... ```) fences.
      2. Try json.loads() on the full stripped string.
      3. If that fails, regex-search for the first {...} block and try again.
      4. If still failing, log and return None.
    """
    if not raw:
        return None

    # Step 1: remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Step 2: try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: search for the first { ... } block (DOTALL so it spans newlines)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Failure
    print(f"    [WARN] Could not parse JSON from response:\n{raw[:300]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER INDEX FIXING for Subtask 1A
# LLMs are notoriously bad at counting characters in Arabic text.
# ─────────────────────────────────────────────────────────────────────────────

def fix_span_indices(spans: List[Dict], response_text: str) -> List[Dict]:
    """Verify and correct character indices for each predicted span.

    EMPTY SPANS:
    If the model returns {"spans": []} it means no citations were found.
    This is valid — the function will simply return [] immediately.

    WHY THIS EXISTS:
    LLMs are notoriously bad at counting characters in Arabic text, especially
    when diacritics (tashkeel) are present. A model might correctly identify
    the quoted text (e.g. "لا تنكح المرأة على عمتها") but give wrong start/end
    indices. This function trusts the TEXT the model returned and recomputes
    the correct indices by searching for that text in the response.

    WHAT IT DOES (step by step):
      1. For each span the model returned {type, text, start, end}:
         a. Try response_text[start:end] — if it exactly equals span["text"],
            the indices are correct and we keep them as-is.
         b. If NOT, the model's indices are wrong. We search for span["text"]
            as a substring inside response_text to find the real position.
         c. If the text appears more than once (e.g. a repeated quote), we pick
            the occurrence whose start index is closest to what the model guessed,
            assuming the model was roughly in the right area even if off by a few.
         d. If the text isn't found anywhere in response_text, the model hallucinated
            text that isn't actually in the input — we discard the span entirely.

    EXAMPLE (from A-Q01):
      Model raw response said: start=31, end=72
      response_text[31:72]  →  does NOT match the span text
      Search for span text  →  found at position 37
      end = 37 + len(text)  →  76
      Stored prediction: start=37, end=76  ✓  (matches gold)

    Args:
        spans:         List of span dicts from the model (may have wrong indices).
        response_text: The original Arabic response text the model was analyzing.

    Returns:
        A new list of spans with verified/corrected indices (exclusive end convention,
        i.e. response_text[start:end] == text for every returned span).
    """
    # Empty spans list is valid — model found no citations. Return immediately.
    if not spans:
        return []

    fixed = []
    for span in spans:
        text  = span.get("text", "")
        start = span.get("start", 0)
        end   = span.get("end", 0)
        stype = span.get("type", "q")

        if not text:
            continue

        # ── Step 1: Check if the model's indices are already correct ─────────
        # We asked for exclusive end [start, end), so response_text[start:end]
        # should equal the span text directly.
        predicted_slice = response_text[start:end]

        if predicted_slice == text:
            # Indices are correct — store as-is.
            fixed.append({"type": stype, "text": text, "start": start, "end": end})
            continue

        # ── Step 2: Indices are wrong — find the text's real position ────────
        # Collect every position where this exact text appears in response_text.
        pos = 0
        occurrences = []
        while True:
            idx = response_text.find(text, pos)
            if idx == -1:
                break
            occurrences.append(idx)
            pos = idx + 1  # advance past this hit to find the next one

        if not occurrences:
            # Text not found anywhere — model hallucinated or garbled the text.
            # Discard this span rather than store wrong indices.
            print(f"    [WARN] Span text not found in response, discarding: '{text[:40]}...'")
            continue

        # ── Step 3: Pick the best occurrence ─────────────────────────────────
        # If the text appears multiple times, choose the occurrence whose start
        # is closest to what the model predicted (model was roughly right area).
        best = min(occurrences, key=lambda i: abs(i - start))

        # Recompute end as exclusive: start + length of the text
        fixed.append({"type": stype, "text": text,
                      "start": best, "end": best + len(text)})

    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1A — ZERO-SHOT SPAN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# System prompt adapted from BurhanAI's DEVELOPER_MESSAGE_NO_TOOLS.
# Rule 2 has been reworded: text is provided directly (no file-reading tools).
# Indices are [start, end) exclusive-end, matching gold annotation convention.
SYSTEM_1A = """\
Task 1A: Identify every INTENDED or CLAIMED Quranic ayah, Prophetic hadith Matn, Prophetic hadith Isnad, and Source Attribution span inside the given Arabic response text and return spans ONLY.

Scoring: character-level F1 over classes {Neither, Ayah, Hadith}. Your spans MUST match exact substring indices in the provided raw text. Do not normalize or rewrite the text.

Important policy: extract what the writer CLAIMS or INTENDS to be an ayah, hadith Matn, hadith Isnad, or Source Attribution, EVEN IF IT IS FABRICATED, MISQUOTED, OR PARAPHRASED. Do not verify against sources and do not skip doubtful spans. If the text implies a citation (e.g., قال الله تعالى / يقول الله / كما قال تعالى / قال رسول الله / في الحديث / روي عن النبي), select the span exactly as it appears.

Extraction rules (apply strictly):
1. Span content
   - For Ayah and Hadith Matn spans, select the text that the writer presents as the ayah/hadith content.
   - For Hadith Isnad spans, select the text that the writer presents as the isnad (narrator chain).
   - For Source Attribution spans, select the text that the writer presents as the source attribution (e.g., (البخاري), (مسلم), (سورة النساء , 9)).
   - Select only the minimal contiguous text of the ayah, hadith, isnad, or source attribution present in the response.
   - Exclude: surrounding quotes, brackets, punctuation, emojis, tatweel (ـ), ellipses, decorative marks, verse numbers.
   - If paraphrased but clearly intended, select the paraphrase exactly as written.
2. Boundaries and indices
   - The text is provided directly in the user message. Work with that exact string.
   - Treat indices as [start, end) exclusive-end. Compute start by locating the first character of the chosen span in the raw text and end = start + len(span_text).
   - Trim ONLY edge characters if present at the span edges before finalizing indices: whitespace, newlines, tabs, RTL controls, quotes «»"' and brackets ()[]{} <> and punctuation ، ؛ : . , ! ? … and tatweel ـ.
   - After computing, assert that raw_text[start:end] == span_text. If not, re-check by sliding inward by up to 2 chars to remove stray quotes/newlines, then re-assert. If still failing, recompute start by searching for the exact span_text in the raw string and use that index.
3. Ayah vs Hadith decision (based on local cues)
   - Ayah cues within ±64 chars: قال الله تعالى، يقول الله، الآية، كما قال تعالى، سورة، (4:3) etc.
   - Hadith cues within ±64 chars: قال رسول الله، النبي ﷺ، في الحديث، روي عن، أخرجه البخاري/مسلم/الترمذي…
   - If both cues appear, classify by the MATN itself (what the span text is), not the surrounding narration.
   - Narration words like "عن"، "حدثنا"، "قال" تخص الراوي ليست جزءاً من المتن، ولا تحدد النوع وحدها.
   - Narration chains (isnad) usually are before the matn and contain narrator names and verbs of narration. If the text explicitly presents a chain of narrators, extract it as an "isnad" span (type "h_isnad") separate from the matn.
   - Source attributions (e.g. "(البخاري)", "(سورة النساء , 9)") should be extracted as "source" spans (type "h_source") separate from the matn.
4. Coverage and duplicates (very important)
   - Extract EVERY distinct occurrence (by position) of each ayah/hadith phrase in the response. Do NOT collapse identical phrases; each occurrence must be a separate span if it appears separately.
   - Scan the entire text: when you identify a phrase span, search the raw text for other exact occurrences of the same phrase and include them too, provided local cues still indicate a citation.
5. Avoid overly short or generic fragments
   - Reject single words or very short tokens unless they are clearly iconic and explicitly cited with strong cues and quotes. As a rule of thumb, require ≥ 10 Arabic letters for a span unless enclosed within quotes with explicit cue.
   - Do not extract formulaic phrases such as "بسم الله الرحمن الرحيم" unless the text explicitly presents them as a citation.
6. Selecting the correct boundaries in long narratives
   - Prefer the smallest exact quoted statement that constitutes the ayah/hadith, not entire narrative paragraphs or stories. If a long paragraph includes one or more quoted statements, extract each quoted statement separately.
   - If the span contains an embedded ayah within a hadith narration, extract only the ayah text as an "Aya" span; do not label the whole narration as "Aya".
7. Overlaps
   - Do not produce nested duplicates or overlapping shards of the same citation. Separate spans are allowed only when they represent distinct occurrences.

   
   
8. Empty output
   - If no citations are found, return an empty spans array: {"spans": []}
   - This is valid and expected when the response mentions Islamic topics without quoting specific text.

Return ONLY valid JSON with this structure:
{"spans": [{"type": "Quran", "text": "...", "start": 0, "end": 10}, ...]}
Use "Quran" for Quran (Ayah) and "Hadith_Matn" for Hadith_Matn, "Hadith_Isnad" for Hadith_Isnad, and "Source" for Quran or Hadith source. start/end are [start, end) exclusive-end indices.

---

## Examples

### Example 1 — No citations (return empty spans)

The text mentions Quran/Hadith topics but never quotes a specific ayah or hadith.
Paraphrases, topic references, and vague allusions are NOT extracted.

Text:
<<<
نعم، هناك آيات كثيرة في القرآن تتحدث عن الأصوات والأصوات التي يمكن سماعها يوم القيامة مثل الرعود والصواعق والأبواق وغيرها. كما يحذر النبي محمد (عليه الصلاة والسلام) أتباعه من الاستماع إلى "صوت" الشيطان واتباع توجيهاته.
>>>

Output:
{"spans": []}

(No specific ayah or hadith is quoted. "صوت" in quotes is not a religious citation.)

---

### Example 2 — Single Ayah, cue via prophetic speech attribution

The cue "قال عيسى عليه السلام" signals a Quranic verse being cited.
Extract only the Quran inside the inner quotes; exclude the outer quote and the reference "[مريم: 30]" is a separate span.
Note: diacritics (tashkeel) are individual characters and shift all indices — Python's str.find() handles this correctly.

Text:
<<<
"
قال عيسى عليه السلام: "قَالَ إِنِّي عَبْدُ اللَّهِ آتَانِيَ الْكِتَابَ وَجَعَلَنِي نَبِيًّا" [مريم: 30].
>>>

Output:
{"spans": [{"type": "Quran", "text": "قَالَ إِنِّي عَبْدُ اللَّهِ آتَانِيَ الْكِتَابَ وَجَعَلَنِي نَبِيًّا", "start": 25, "end": 93}, {"type": "Source", "text": "مريم: 30", "start": 96, "end": 108}]}

(Ayah cue: "قال عيسى عليه السلام". The inner quote at position 24 is excluded; span starts at position 25 on the first letter ق. Trailing " [مريم: 30]." excluded.)

---

### Example 3 — Both Ayah and Hadith in the same response

When a response contains multiple citation types, extract each span independently.
Each gets its own entry; intro phrases and source references are excluded.

Text:
<<<
قال الله تعالى في القرآن "وما كان لبشر أن يكلمه الله إلا وحيا أو من وراء حجاب أو يرسل رسولا فيوحي إليه ما يشاء إنه علي حكيم" (سورة الشورى 42). وهذا يعني أنه ليس للأنبياء القدرة على معرفة الأشياء التي تقع خارج وحي الله المباشر لهم. كما ورد في الحديث الشريف "لا يعلم الغيب إلا الله ومن أطلعه عليه".
>>>

Output:
{"spans": [{"type": "Quran", "text": "وما كان لبشر أن يكلمه الله إلا وحيا أو من وراء حجاب أو يرسل رسولا فيوحي إليه ما يشاء إنه علي حكيم", "start": 26, "end": 123}, {"type": "Hadith_Matn", "text": "لا يعلم الغيب إلا الله ومن أطلعه عليه", "start": 257, "end": 294}, {"type": "Source", "text": "سورة الشورى 42", "start": 126, "end": 140}]}

(Ayah cue: "قال الله تعالى في القرآن" → type "Quran". Hadith cue: "ورد في الحديث الشريف" → type "Hadith_Matn". "(سورة الشورى 42)" and "(ابن أبي الدنيا)" are source references — excluded.)\
"""

# JSON schema for structured output (enforced by OpenAI and Gemini APIs).
# Anthropic and Together receive the schema description via the system prompt instead.
SCHEMA_1A = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type":  {"type": "string", "enum": ["Quran", "Hadith_Matn","Hadith_Isnad", "Source" ]},
                    "text":  {"type": "string"},
                    "start": {"type": "integer"},
                    "end":   {"type": "integer"},
                },
                "required": ["type", "text", "start", "end"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["spans"],
    "additionalProperties": False,
}


def build_prompt_1a(response_text: str) -> str:
    """User message for 1A — just the raw text.  All instructions are in the system prompt."""
    return f"Text to analyze:\n<<<\n{response_text}\n>>>"

# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1B — ZERO-SHOT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_1B = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorisation of:
  • The Holy Quran in the Uthmani script (الرسم العثماني) with full tashkeel (تشكيل)
  • The six canonical Hadith collections (الكتب الستة):
      Sahih al-Bukhari, Sahih Muslim, Sunan Abu Dawud,
      Jami' al-Tirmidhi, Sunan al-Nasa'i, Sunan Ibn Majah

Your sole task is: given a list of Arabic quotations extracted from an AI-generated response about Islam, classify each quotation as "correct" or "incorrect" based on whether it faithfully matches the canonical source.

CLASSIFICATION CRITERIA
────────────────────────
Use exactly one of these four labels for each quotation:

  CorrectAyah   — the quotation faithfully matches the exact wording of a Quranic ayah
                  in the Uthmani Mushaf. Minor diacritic variation is acceptable.

  WrongAyah     — claimed to be (or appears to be) a Quran quotation, but contains
                  errors: words added, removed, substituted, reordered, or is fabricated.

  CorrectHadith — the quotation faithfully matches the exact matn of a hadith in one
                  of the six canonical books. Minor diacritic variation is acceptable.

  WrongHadith   — claimed to be (or appears to be) a hadith, but the text is fabricated,
                  paraphrased, garbled, or cannot be verified in the six canonical books.

OUTPUT FORMAT
─────────────
Return ONLY valid JSON — no explanation, no commentary:
{"validations": [{"index": 1, "label": "CorrectAyah"}, {"index": 2, "label": "WrongAyah"}, ...]}

One entry per quotation, in the same order as the input list.
The label field must be exactly one of: CorrectAyah, WrongAyah, CorrectHadith, WrongHadith.

──────────────────────────────────────────────────────────────────────────────
FEW-SHOT EXAMPLES
──────────────────────────────────────────────────────────────────────────────

Example input (3 quotations):
1. [Quran] "وَلَنَبْلُوَنَّكُمْ بِشَيْءٍ مِنَ الْخَوْفِ وَالْجُوعِ وَنَقْصٍ مِنَ الْأَمْوَالِ وَالْأَنْفُسِ وَالثَّمَرَاتِ وَبَشِّرِ الصَّابِرِينَ"
2. [Quran] "وَإِن يَصْبِرْ عَلَيْكَ إِنَّهُ كَآفٍ إِنَّهُ غَفُورٌ ذُو عِزٍ وَرَحِيمٌ"
3. [Hadith] "إن الله كتب الإيمان في قلوبكم، فاعملوا بما كتب في قلوبكم، فإن الله لا يقبل من عمل لا يُعمل له"

Example output:
{"validations": [{"index": 1, "label": "CorrectAyah"}, {"index": 2, "label": "WrongAyah"}, {"index": 3, "label": "WrongHadith"}]}

Reasoning (NOT included in your output):
  1 — CorrectAyah: exact match with Surah Al-Baqarah 2:155.
  2 — WrongAyah: this text does not appear anywhere in the Quran; fabricated.
  3 — WrongHadith: this text is not found in any of the six canonical hadith collections.

──────────────────────────────────────────────────────────────────────────────
"""


# -----------------------------------------------------------------------------
# TASK CONFIG -- edit this block to change the annotation task / prompt.
#   TASK_NAME         : label stored in the output file
#   SYSTEM_PROMPT     : system instruction sent to the model
#   SCHEMA            : JSON schema enforced for OpenAI/Gemini (None to disable)
#   build_user_prompt(item) -> str : builds the user message from one answer item
# SYSTEM_1B (above) is kept available if you want a validation-style task instead.
# -----------------------------------------------------------------------------
TASK_NAME     = "span_detection"
SYSTEM_PROMPT = SYSTEM_1A
SCHEMA        = SCHEMA_1A

def build_user_prompt(item: Dict) -> str:
    return build_prompt_1a(item["answer"])


# -----------------------------------------------------------------------------
# INCREMENTAL SAVE + RESUME
# -----------------------------------------------------------------------------
def load_resume_state(output_path: str):
    """Return (per_sample, completed_ids).

    completed_ids is None when the file is already marked complete (nothing to do).
    """
    if not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"  [WARN] could not read {output_path} for resume: {e}")
        return [], set()
    per_sample = data.get("per_sample", [])
    if data.get("status") == "complete":
        print(f"  [SKIP] {output_path} already complete -- delete it to re-run.")
        return per_sample, None
    if per_sample:
        print(f"  [RESUME] {len(per_sample)} already done -- skipping those.")
    return per_sample, {s["sample_id"] for s in per_sample}


def save_incremental(output_path: str, model_name: str,
                     per_sample: List[Dict], n_total: int) -> None:
    """Write progress after each sample so a crash loses at most one item."""
    n_done = len(per_sample)
    snapshot = {
        "task":        TASK_NAME,
        "model":       model_name,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "status":      "complete" if n_done >= n_total else "in_progress",
        "n_total":     n_total,
        "n_completed": n_done,
        "n_failed":    sum(1 for s in per_sample if s.get("error")),
        "per_sample":  per_sample,
    }
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
def run(model_name: str, input_path: str, output_path: str,
        fix_indices: bool = False, max_samples: Optional[int] = None) -> None:
    with open(input_path, encoding="utf-8") as fh:
        answers = json.load(fh)
    if not isinstance(answers, list):
        answers = [answers]
    if max_samples:
        answers = answers[:max_samples]
    n_total = len(answers)

    per_sample, completed = load_resume_state(output_path)
    if completed is None:
        return
    todo = [a for a in answers if a["id"] not in completed]
    print(f"\n=== {TASK_NAME} | {model_name} | {len(todo)} of {n_total} answers ===")

    for i, item in enumerate(todo, 1):
        sid = item["id"]
        response_text = item.get("answer", "")
        print(f"  [{i}/{len(todo)}] {model_name} | {sid} ... ", end="", flush=True)

        raw, usage = call_api(model_name, SYSTEM_PROMPT, build_user_prompt(item), schema=SCHEMA)

        if raw is None:
            print("FAILED")
            per_sample.append({
                "sample_id":    sid,
                "prompt":       item.get("prompt", ""),
                "answer":       response_text,
                "answer_model": item.get("model", ""),
                "prediction":   None,
                "raw_response": None,
                "usage":        None,
                "error":        "API call failed",
            })
            save_incremental(output_path, model_name, per_sample, n_total)
            continue

        parsed = parse_json_response(raw)
        spans  = parsed.get("spans", []) if parsed else []
        if fix_indices:
            spans = fix_span_indices(spans, response_text)

        print(f"OK ({len(spans)} spans)")
        per_sample.append({
            "sample_id":    sid,
            "prompt":       item.get("prompt", ""),
            "answer":       response_text,
            "answer_model": item.get("model", ""),
            "prediction":   {"spans": spans},
            "raw_response": raw,
            "usage":        usage,
        })
        save_incremental(output_path, model_name, per_sample, n_total)

    print(f"  -> wrote {output_path}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Frontier-LLM span annotation (IslamicEval)")
    ap.add_argument("--model",  required=True,
                    help="annotator model, one of: " + ", ".join(MODEL_REGISTRY.keys()))
    ap.add_argument("--input",  required=True,
                    help="model-answers JSON: [{id, prompt, answer, model}, ...]")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--fix-indices", action="store_true",
                    help="recompute span start/end in Python from the returned text (default: off)")
    ap.add_argument("--max-samples", type=int, default=None, help="limit to first N answers (testing)")
    args = ap.parse_args()

    if args.model not in MODEL_REGISTRY:
        sys.exit(f"Unknown model '{args.model}'. Choices: {list(MODEL_REGISTRY.keys())}")
    _, _, key_env = MODEL_REGISTRY[args.model]
    if not os.environ.get(key_env):
        sys.exit(f"{key_env} is not set in the environment (.env) for model {args.model}")

    run(args.model, args.input, args.output,
        fix_indices=args.fix_indices, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
