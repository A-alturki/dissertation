#!/usr/bin/env python3
"""
correction_pipeline.py
======================
Two-stage correction pipeline for IslamicEval incorrect citation spans.

  STAGE 1 — GATE (3 models): each span → {verdict: correctable|خطأ, confidence}.
            A span is GATE-correctable by MAJORITY VOTE (>=2 of 3).
            Gate models: gpt-5.4-mini, gpt-5.4, gemini-3-flash-preview.

  STAGE 2 — CORRECTION (2 models): only the majority-correctable spans →
            every canonical text the span distorts (candidates + matched_run).
            Correction models: gpt-5.4, gemini-3-flash-preview.

Then reports gate agreement and correction agreement (do the 2 models propose the
same correction, by rasm-robust text match).

Input: a CSV with id,model,span,ref,correctness (e.g. sample_correction_final_classified.csv;
extra cols like error_type are carried through for analysis). Reuses annotate_spans
plumbing (call_api, parse_json_response, MODEL_REGISTRY); PRICES table defined locally.
Resumable per (uid, stage, model); parallel; atomic incremental save.

Usage:
    python correction_pipeline.py --input ../outputs/correction/sample_correction_final_classified.csv
    python correction_pipeline.py --stage gate   --limit 3      # gate only, smoke test
"""
import os, csv, sys, json, re, argparse, hashlib, threading, difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections import Counter
from typing import Dict, List, Optional

from annotate_spans import MODEL_REGISTRY, call_api, parse_json_response
from citation_lookup import _TABLE                    # for rasm-robust text match

# (input, output) USD per 1M tokens — placeholders, mirror annotate_typology.PRICES.
PRICES = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5":   (1.25, 10.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gemini-3-flash-preview": (1.5, 9.0),
    "gemini-3.5-flash": (1.5, 9.0),         # placeholder — update to real rate
    "gemini-3.1-pro-preview": (2.0, 12.0),
}

GATE_MODELS    = ["gpt-5.4", "gemini-3.5-flash"]
CORRECT_MODELS = ["gpt-5.4", "gemini-3.1-pro-preview"]
# Both stages use medium reasoning + a 16k token budget.
GATE_REASONING = "medium"
GATE_MAX_TOKENS = 16384
CORRECT_REASONING = "medium"
CORRECT_MAX_TOKENS = 16384
REF_LABEL = {"quran": "Quranic ayah (آية قرآنية)", "hadith": "Prophetic hadith (حديث نبوي)"}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  (.format() collapses the doubled JSON braces; no other placeholders)
# ─────────────────────────────────────────────────────────────────────────────
GATE_PROMPT = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorization of the Holy
Quran (Uthmani script) and the six canonical Hadith collections (الكتب الستة):
Sahih al-Bukhari, Sahih Muslim, Sunan Abu Dawud, Jami' al-Tirmidhi,
Sunan al-Nasa'i, Sunan Ibn Majah.

You are given ONE Arabic span that an AI model presented as a Quranic ayah or a
Prophetic hadith, but which is INCORRECT. Your ONLY job is to decide whether this
span is a recoverable distortion of a REAL canonical text, or whether it is
baseless (خطأ). You do NOT need to identify or write out the correct text — only
judge whether one plausibly exists.

────────────────────────────────────────────────────────────────────────
DECISION
────────────────────────────────────────────────────────────────────────
"correctable" — the span reads as a corrupted, misdiacritized, misquoted, or
  lightly paraphrased version of a SPECIFIC real ayah or hadith: a recognizable
  contiguous phrase of it traces back to one or more identifiable canonical text (after
  ignoring tashkeel and letter-form variants أ إ آ → ا ; ى → ي ; ة → ه).

"خطأ" — the span is baseless. Rule خطأ when:
  • It shares only the TOPIC/MEANING with real texts (theft, marriage, charity)
    but reproduces no actual phrase of any one text.
  • It shares only ISOLATED words scattered across different texts — one word
    from here, one from there — with no contiguous phrase from a single source.
  • It is fluent and Islamic-sounding but reconstructs no real text.
  • It is garbled / stitched from incoherent fragments.

Be STRICT. Sounding Islamic, or being on a real Islamic topic, is NOT enough.
If you cannot point to ONE specific text the span is a distortion of, output خطأ.

────────────────────────────────────────────────────────────────────────
OUTPUT — valid JSON only, FIELDS IN THIS EXACT ORDER:
{{
  "rationale": "<brief: does a contiguous phrase trace to one specific real text,
                or is the link only topic / scattered words / fabrication>",
  "verdict": "correctable" | "خطأ",
  "confidence": "low" | "medium" | "high"
}}
  confidence = how sure you are of the verdict you gave:
    high   — clearly a distortion of one identifiable text, OR clearly baseless.
    medium — likely, but the corruption is heavy or the wording is partly generic.
    low    — genuinely unsure (borderline between a real distortion and a
             topic-only / fabricated span).

────────────────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────────────────
# Quran — long verbatim run, only diacritics differ → correctable, high
Span: "ولا تعزموا عقد النكاح حتى يبلغ الكتاب أجله"
{{"rationale": "A long contiguous phrase 'ولا تعزموا عقدة النكاح حتى يبلغ الكتاب أجله' traces verbatim to one specific verse (البقرة 235); the span differs only in diacritics and 'عقد' vs 'عقدة'.", "verdict": "correctable", "confidence": "high"}}

# Hadith — verbatim opening of a specific hadith → correctable, high
Span: "تزوجوا الودود الولود"
{{"rationale": "The span is a heavily distorted and misquoted version of Al-Baqarah: 185 ('فَمَن شَهِدَ مِنكُمُ الشَّهْرَ فَلْيَصُمْهُ'), where 'شهد' was corrupted to 'شك', 'الشهر' to 'في هذا الشهر', and 'فليصمه' to 'فليصم', with elements conflated from the adjacent verse 184 ('وعلى الذين يطيقونه').", "verdict": "correctable", "confidence": "medium"}} 

# Quran — 
Span: "فمن شَكَّ منكم في هذا الشهر فليصم من يطيق"
{{"rationale": "'الحج' and 'فروج' occur in different verses but no contiguous phrase traces to a single one. Topic plus scattered words.", "verdict": "خطأ", "confidence": "high"}}

# Hadith — fluent, on-topic, names a real mosque, but reconstructs nothing → خطأ, high
Span: "التراويح في الحج مثل الصلاة في المسجد، فمن تراوح في الحاجة في المسجد النبوي الشريف فإنه يلعب من الصلاة"
{{"rationale": "Fluent and Islamic-sounding, references real concepts (التراويح، المسجد النبوي), but no contiguous phrase reproduces any single hadith; it is fabricated.", "verdict": "خطأ", "confidence": "high"}}

# Garbled / stitched fragments → خطأ, high
Span: "وَمَنْ لَا يُحِيطُ بِمَالِهِ إِلَّا بِإِذْنِهِ يُحِيطُ بِمَالِهِ وَسَعَ كُرْسِيُّ السَّلَامُ..."
{{"rationale": "Incoherent fragments stitched together; traces to no single real text.", "verdict": "خطأ", "confidence": "high"}}
"""

CORRECTION_PROMPT = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorization of:
  • The Holy Quran in Uthmani script (الرسم العثماني) with full tashkeel.
  • The six canonical Hadith collections (الكتب الستة): Sahih al-Bukhari,
    Sahih Muslim, Sunan Abu Dawud, Jami' al-Tirmidhi, Sunan al-Nasa'i,
    Sunan Ibn Majah.

You are given ONE Arabic span that an AI model presented as a Quranic ayah or a
Prophetic hadith, but which is INCORRECT — misquoted, misdiacritized, paraphrased,
or garbled. A prior step has already judged this span to be a recoverable
distortion of a real canonical text. Your job: return EVERY canonical text the
span genuinely distorts.

────────────────────────────────────────────────────────────────────────
THE CORE RULE — WHAT COUNTS AS A CORRECTION
────────────────────────────────────────────────────────────────────────
Include a candidate ONLY IF, after ignoring tashkeel and normalizing letters
(أ إ آ ا → ا ; ى → ي ; ة → ه ; ؤ ئ → hamza), a MEANINGFUL CONTIGUOUS SEQUENCE OF
WORDS from the span appears in that candidate — a recognizable phrase, not
isolated words.

Do NOT include a candidate that shares only the TOPIC, only the MEANING, or only
ISOLATED scattered words. A thematically related text the span does not actually
quote is a SERIOUS ERROR — leave it out.

HOW MANY CANDIDATES:
  • Return as many as genuinely satisfy the rule — no fixed number. This may be
    one. It may be several (e.g. when the same wording is attested in more than
    one of the six books, OR the span sits between close variants). Often a span
    distorts exactly ONE specific text — in that case return ONLY that one.
  • Do NOT pad the list with near-topic texts to reach a count. Quality over
    quantity: every candidate must independently pass the contiguous-phrase rule.
  • If, on inspection, NOTHING actually passes the rule (the prior step may have
    erred), return an empty list with verdict "خطأ".

────────────────────────────────────────────────────────────────────────
OUTPUT — valid JSON only, FIELDS IN THIS EXACT ORDER:
{{
  "rationale": "<brief: which text(s) the span distorts and the contiguous phrase
                that matches each>",
  "verdict": "ok" | "خطأ",
  "candidates": [
    {{"text": "<exact canonical text>",
      "source": "سورة <name> <n>"  OR  "<collection> <n>",
      "matched_run": "<longest contiguous phrase shared with the span>",
      "confidence": "low" | "medium" | "high"}}
  ]
}}
  • Quran "text": full Uthmani ayah(s) with tashkeel. Hadith "text": matn only,
    no isnad, no "رواه...".
  • Per-candidate confidence, anchored to the contiguous run:
      high   — long contiguous run; clearly this exact text, differing only in
               diacritics / spelling / a single inflection.
      medium — a real multi-word phrase matches, but with paraphrase or a fused
               extra clause.
      low    — only a short phrase matches; plausible but uncertain.
  • State "matched_run" before "confidence"; the tier must reflect that run.
  • Rank candidates by confidence, highest first.
  • If verdict is "خطأ", candidates MUST be [].

────────────────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────────────────
# Single clear correction
Span: "رَبِّنَا هَبْ لَنَا مِنَ الصَّالِحِينَ"
{{"rationale": "Distorts الصافات 100; 'هب ... من الصالحين' matches, only the pronoun differs (لنا vs لي). No other verse carries this phrase.",
  "verdict": "ok",
  "candidates": [
    {{"text": "رَبِّ هَبْ لِي مِنَ الصَّالِحِينَ", "source": "سورة الصافات 100",
      "matched_run": "هب من الصالحين", "confidence": "high"}}
]}}

# Single correction with a fused extra clause
Span: "أفضل الصدقة سقي الماء والحج المبرور"
{{"rationale": "Verbatim spine 'أفضل الصدقة سقي الماء' matches Ibn Majah 3684; trailing 'والحج المبرور' is fused on. One specific source.",
  "verdict": "ok",
  "candidates": [
    {{"text": "أَفْضَلُ الصَّدَقَةِ سَقْيُ الْمَاءِ", "source": "سنن ابن ماجه 3684",
      "matched_run": "أفضل الصدقة سقي الماء", "confidence": "high"}}
]}}

# Multiple legitimate candidates — same wording attested in more than one book
Span: "تزوجوا الودود الولود فإني مكاثر بكم الأمم يوم القيامة"
{{"rationale": "The phrase 'تزوجوا الودود الولود فإني مكاثر بكم' is attested with minor wording differences in two of the six books; both are genuine contiguous matches.",
  "verdict": "ok",
  "candidates": [
    {{"text": "تَزَوَّجُوا الْوَدُودَ الْوَلُودَ فَإِنِّي مُكَاثِرٌ بِكُمُ الأُمَمَ", "source": "سنن أبي داود 2050",
      "matched_run": "تزوجوا الودود الولود فإني مكاثر بكم الأمم", "confidence": "high"}},
    {{"text": "تَزَوَّجُوا الْوَدُودَ الْوَلُودَ فَإِنِّي مُكَاثِرٌ بِكُمُ الأَنْبِيَاءَ يَوْمَ الْقِيَامَةِ", "source": "سنن النسائي 3227",
      "matched_run": "تزوجوا الودود الولود فإني مكاثر بكم", "confidence": "medium"}}
]}}

# Gate passed it, but on inspection nothing actually matches → veto to خطأ
Span: "السرقة حرام، وإن كانت لطعام وحاجة"
{{"rationale": "No contiguous phrase reproduces any single hadith; only scattered topic words. Overriding to خطأ.",
  "verdict": "خطأ", "candidates": []}}
"""

CONF = {"type": "string", "enum": ["low", "medium", "high"]}
SCHEMA_GATE = {
    "type": "object",
    "properties": {
        "rationale":  {"type": "string"},
        "verdict":    {"type": "string", "enum": ["correctable", "خطأ"]},
        "confidence": CONF,
    },
    "required": ["rationale", "verdict", "confidence"],
    "additionalProperties": False,
}
SCHEMA_CORRECTION = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "verdict":   {"type": "string", "enum": ["ok", "خطأ"]},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":        {"type": "string"},
                    "source":      {"type": "string"},
                    "matched_run": {"type": "string"},
                    "confidence":  CONF,
                },
                "required": ["text", "source", "matched_run", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "verdict", "candidates"],
    "additionalProperties": False,
}


ONE_SHOT_CORRECTION_PROMPT = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorization of:
  • The Holy Quran in Uthmani script (الرسم العثماني) with full tashkeel.
  • The six canonical Hadith collections (الكتب الستة): Sahih al-Bukhari,
    Sahih Muslim, Sunan Abu Dawud, Jami' al-Tirmidhi, Sunan al-Nasa'i,
    Sunan Ibn Majah.

You are given ONE Arabic span that an AI model presented as a Quranic ayah or a
Prophetic hadith, but which is INCORRECT — misquoted, misdiacritized, paraphrased,
garbled, or entirely fabricated. Do two things in one step:
  1. Decide whether the span is a recoverable distortion of a REAL canonical
     text, or whether it is baseless (خطأ).
  2. If recoverable, return EVERY canonical text the span genuinely distorts.

────────────────────────────────────────────────────────────────────────
THE CORE RULE — WHAT COUNTS AS A CORRECTION
────────────────────────────────────────────────────────────────────────
A candidate is a valid correction ONLY IF, after ignoring tashkeel and
normalizing letters (أ إ آ ا → ا ; ى → ي ; ة → ه ; ؤ ئ → hamza), a MEANINGFUL
CONTIGUOUS SEQUENCE OF WORDS from the span appears in that candidate — a
recognizable phrase, not isolated words.

Output "خطأ" (no candidates) when:
  • The span shares only the TOPIC/MEANING with real texts (theft, marriage,
    charity) but reproduces no actual phrase of any one text.
  • The span shares only isolated words scattered across different texts — one
    word from here, one from there — with no contiguous phrase from a single
    source.
  • The span is fluent and Islamic-sounding but reconstructs no real text.
  • The span is garbled / stitched from incoherent fragments.

Be STRICT. Sounding Islamic, or being on a real Islamic topic, is NOT enough.
A thematically related text the span does not actually quote is a SERIOUS ERROR.
If you cannot point to a SPECIFIC text the span is a contiguous distortion of,
output خطأ.

────────────────────────────────────────────────────────────────────────
HOW MANY CANDIDATES
────────────────────────────────────────────────────────────────────────
  • Return as many as genuinely satisfy the rule — no fixed number. Often a span
    distorts exactly ONE specific text; return only that one. Sometimes the same
    wording is attested in more than one of the six books, or the span sits
    between close variants — then return each genuine match.
  • Do NOT pad the list with near-topic texts to reach a count. Every candidate
    must independently pass the contiguous-phrase rule.

────────────────────────────────────────────────────────────────────────
OUTPUT — valid JSON only, FIELDS IN THIS EXACT ORDER:
{{
  "rationale": "<brief: which text(s) the span distorts and the contiguous phrase
                matching each — or why it is خطأ (topic-only / scattered words /
                fabricated / garbled)>",
  "verdict": "ok" | "خطأ",
  "candidates": [
    {{"text": "<exact canonical text>",
      "source": "سورة <name> <n>"  OR  "<collection> <n>",
      "matched_run": "<longest contiguous phrase shared with the span>",
      "confidence": "low" | "medium" | "high"}}
  ]
}}
  • Quran "text": full Uthmani ayah(s) with tashkeel. Hadith "text": matn only,
    no isnad, no "رواه...".
  • Per-candidate confidence, anchored to the contiguous run:
      high   — long contiguous run; clearly this exact text, differing only in
               diacritics / spelling / a single inflection.
      medium — a real multi-word phrase matches, but with heavier corruption,
               paraphrase, or a fused extra clause.
      low    — only a short phrase matches; plausible but uncertain.
  • State "matched_run" before "confidence"; the tier must reflect that run.
  • Rank candidates by confidence, highest first.
  • When verdict is "خطأ", candidates MUST be [].

────────────────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────────────────
# Quran — long verbatim run, only diacritics + one letter differ → ok, high
Span: "ولا تعزموا عقد النكاح حتى يبلغ الكتاب أجله"
{{"rationale": "Long contiguous phrase 'ولا تعزموا عقدة النكاح حتى يبلغ الكتاب أجله' traces verbatim to البقرة 235; the span differs only in diacritics and 'عقد' vs 'عقدة'.",
  "verdict": "ok",
  "candidates": [
    {{"text": "وَلَا تَعْزِمُوا عُقْدَةَ النِّكَاحِ حَتَّىٰ يَبْلُغَ الْكِتَابُ أَجَلَهُ ۚ وَاعْلَمُوا أَنَّ اللَّهَ يَعْلَمُ مَا فِي أَنفُسِكُمْ فَاحْذَرُوهُ ۚ وَاعْلَمُوا أَنَّ اللَّهَ غَفُورٌ حَلِيمٌ",
      "source": "سورة البقرة 235",
      "matched_run": "ولا تعزموا عقدة النكاح حتى يبلغ الكتاب أجله", "confidence": "high"}}
]}}

# Hadith — same wording attested in two books → ok, two candidates
Span: "تزوجوا الودود الولود"
{{"rationale": "'تزوجوا الودود الولود' is the verbatim opening of one specific hadith, attested in two of the six books.",
  "verdict": "ok",
  "candidates": [
    {{"text": "تَزَوَّجُوا الْوَدُودَ الْوَلُودَ فَإِنِّي مُكَاثِرٌ بِكُمُ الأُمَمَ", "source": "سنن أبي داود 2050",
      "matched_run": "تزوجوا الودود الولود", "confidence": "high"}},
    {{"text": "تَزَوَّجُوا الْوَدُودَ الْوَلُودَ فَإِنِّي مُكَاثِرٌ بِكُمُ الأُمَمَ", "source": "سنن النسائي 3227",
      "matched_run": "تزوجوا الودود الولود", "confidence": "high"}}
]}}

# Hadith — verbatim spine + a fused extra clause → ok, high
Span: "أفضل الصدقة سقي الماء والحج المبرور"
{{"rationale": "Verbatim spine 'أفضل الصدقة سقي الماء' matches Ibn Majah 3684; trailing 'والحج المبرور' is fused on. One specific source.",
  "verdict": "ok",
  "candidates": [
    {{"text": "أَفْضَلُ الصَّدَقَةِ سَقْيُ الْمَاءِ", "source": "سنن ابن ماجه 3684",
      "matched_run": "أفضل الصدقة سقي الماء", "confidence": "high"}}
]}}

# Quran — heavy corruption, recoverable with effort → ok, medium
Span: "فمن شَكَّ منكم في هذا الشهر فليصم من يطيق"
{{"rationale": "Heavily distorted البقرة 185: 'شهد' corrupted to 'شك', 'الشهر' to 'في هذا الشهر', 'فليصمه' to 'فليصم', with 'من يطيق' conflated from the adjacent verse 184. The core phrase is recoverable but corruption is heavy.",
  "verdict": "ok",
  "candidates": [
    {{"text": "فَمَن شَهِدَ مِنكُمُ الشَّهْرَ فَلْيَصُمْهُ", "source": "سورة البقرة 185",
      "matched_run": "فمن . منكم . الشهر فليصم", "confidence": "medium"}}
]}}

# Quran — topic + scattered single words only → خطأ
Span: "وَالْحَجُّ أَشَدُّ الْفُرُوجِ"
{{"rationale": "'الحج' occurs in البقرة 197 and 'فروج' in المؤمنون 5, but no contiguous phrase from the span exists in any single verse. Topic plus scattered words.",
  "verdict": "خطأ", "candidates": []}}

# Hadith — fluent, on-topic, names a real mosque, but reconstructs nothing → خطأ
Span: "التراويح في الحج مثل الصلاة في المسجد، فمن تراوح في الحاجة في المسجد النبوي الشريف فإنه يلعب من الصلاة"
{{"rationale": "Fluent and Islamic-sounding, references real concepts (التراويح، المسجد النبوي), but no contiguous phrase reproduces any single hadith; it is fabricated.",
  "verdict": "خطأ", "candidates": []}}

# Garbled / stitched fragments → خطأ
Span: "وَمَنْ لَا يُحِيطُ بِمَالِهِ إِلَّا بِإِذْنِهِ يُحِيطُ بِمَالِهِ وَسَعَ كُرْسِيُّ السَّلَامُ..."
{{"rationale": "Incoherent fragments stitched together; traces to no single real text.",
  "verdict": "خطأ", "candidates": []}}
"""


GEMINI_CORRECTION_PROMPT_ALTERNATIVE = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorization of:
  • The Holy Quran in Uthmani script (الرسم العثماني) with full tashkeel.
  • The six canonical Hadith collections (الكتب الستة): Sahih al-Bukhari,
    Sahih Muslim, Sunan Abu Dawud, Jami' al-Tirmidhi, Sunan al-Nasa'i,
    Sunan Ibn Majah.
 
You are given ONE Arabic span that an AI model presented as a Quranic ayah or a
Prophetic hadith, but which is INCORRECT — misquoted, misdiacritized, paraphrased,
garbled, or entirely fabricated. Do two things in one step:
  1. Decide whether the span is a recoverable distortion of a REAL canonical
     text, or whether it is baseless (خطأ).
  2. If recoverable, return EVERY canonical text the span genuinely distorts.
 
────────────────────────────────────────────────────────────────────────
THE CORE RULE — WHAT COUNTS AS A CORRECTION
────────────────────────────────────────────────────────────────────────
A candidate is a valid correction if you can identify ONE SPECIFIC canonical
text that the span is a distortion of. Distortion spans a spectrum — recover the
correction across ALL of it, not only near-verbatim cases:
 
  • VERBATIM / LIGHT: after ignoring tashkeel and normalizing letters
    (أ إ آ ا → ا ; ى → ي ; ة → ه ; ؤ ئ → hamza), a meaningful contiguous
    sequence of words from the span appears in the candidate.
  • HEAVY PARAPHRASE / SEMANTIC: the span carries no long contiguous run, but it
    is a recognizable REWORDING of ONE specific text — same verse or hadith,
    rephrased, reordered, or synonym-swapped, such that a knowledgeable reader
    identifies exactly which text it is trying (and failing) to quote. RECOVER
    THESE. Do not decline a span merely because its wording diverges heavily from
    the source; if you can name the single specific text it distorts, correct it.
 
Bias toward recovery: if a specific real text is identifiable behind the span,
returning it is preferred over declining. Decline only when NO single specific
text underlies the span (see خطأ conditions).
 
The identification must rest on DISTINCTIVE CONTENT (matn / verse wording).
Shared isnad frames and honorifics (صلى الله عليه وسلم ، رسول الله ، عن فلان ،
قال ، رضي الله عنه) are NOT content — a correspondence resting only on such
formulaic words is NOT a valid match.
 
Output "خطأ" (no candidates) when the span corresponds to NO SINGLE SPECIFIC
text — including:
  • It shares only the TOPIC/MEANING with real texts but is not a rewording of
    any ONE particular text — a generic statement on the theme.
  • Its content is scattered across DIFFERENT texts, with no single text it
    distorts as a whole.
  • The only overlap is boilerplate isnad/honorific wording.
  • It is fluent and Islamic-sounding but reconstructs no real text (fabricated).
  • It is garbled / stitched from incoherent fragments.
  • It is mislabeled: a distortion of a HADITH presented as an ayah (or vice
    versa) — if it is not a distortion of a text of the claimed type, output خطأ.
 
Loosening to semantic matches is NOT license to map any on-topic span to a
convenient text. The test is always: is this span a distortion of ONE SPECIFIC
identifiable text? If yes → correct it at the appropriate confidence. If it is
only thematically related, fabricated, or scattered → خطأ.
 
────────────────────────────────────────────────────────────────────────
HOW MANY CANDIDATES
────────────────────────────────────────────────────────────────────────
  • Return as many as genuinely satisfy the rule — no fixed number. A span may
    distort a SINGLE text, a run of consecutive ayahs, or a matn attested as
    several close variants across the six books (return each genuine variant).
  • Do NOT pad with near-topic texts. Every candidate must independently pass the
    identification rule.
 
────────────────────────────────────────────────────────────────────────
OUTPUT — valid JSON only, FIELDS IN THIS EXACT ORDER:
{{
  "rationale": "<brief: which specific text(s) the span distorts and the
                correspondence — the contiguous phrase for verbatim matches, or
                the specific rewording relationship for paraphrase matches — or
                why it is خطأ (topic-only / scattered / boilerplate / fabricated
                / garbled / mislabeled)>",
  "verdict": "ok" | "خطأ",
  "candidates": [
    {{"text": "<exact canonical text>",
      "source": "سورة <name> <n>"  OR  "<collection>",
      "matched_run": "<longest contiguous substantive phrase shared with the
                       span; for a paraphrase match with no contiguous run, give
                       the key corresponding wording instead>",
      "confidence": "paraphrase" | "low" | "medium" | "high"}}
  ]
}}
  • Quran "text": full Uthmani ayah(s) with tashkeel. Hadith "text": matn only,
    no isnad, no "رواه…". For Hadith, identification is by the MATN; a reference
    number is not required and not important.
  • Confidence:
      high       — long contiguous verbatim run; clearly this exact text,
                   differing only in diacritics / spelling / a single inflection.
      medium     — a real multi-word content phrase matches, with heavier
                   corruption, partial paraphrase, or a fused extra clause.
      low        — only a short content phrase matches verbatim; plausible.
      paraphrase — NO reliable contiguous verbatim run, but the span is
                   identifiable as a semantic rewording of ONE specific text.
                   The recovered-by-meaning tier: correct but lower trust; the
                   rationale must name the specific target and the correspondence.
  • The tier must reflect actual match strength and must NOT be inflated by shared
    isnad/honorific words.
  • Rank candidates high → paraphrase. When verdict is "خطأ", candidates = [].
 
────────────────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────────────────
# Quran — long verbatim run, only diacritics differ → ok, high
Span: "ولا تعزموا عقد النكاح حتى يبلغ الكتاب أجله"
{{"rationale": "Contiguous phrase 'ولا تعزموا عقدة النكاح حتى يبلغ الكتاب أجله' is البقرة 235; differs only in diacritics and 'عقد' vs 'عقدة'.",
  "verdict": "ok",
  "candidates": [
    {{"text": "وَلَا تَعْزِمُوا عُقْدَةَ النِّكَاحِ حَتَّىٰ يَبْلُغَ الْكِتَابُ أَجَلَهُ ۚ وَاعْلَمُوا أَنَّ اللَّهَ يَعْلَمُ مَا فِي أَنفُسِكُمْ فَاحْذَرُوهُ ۚ وَاعْلَمُوا أَنَّ اللَّهَ غَفُورٌ حَلِيمٌ",
      "source": "سورة البقرة 235",
      "matched_run": "ولا تعزموا عقدة النكاح حتى يبلغ الكتاب أجله", "confidence": "high"}}
]}}
 
# Quran — heavy paraphrase, no contiguous run, clearly ONE verse → ok, paraphrase
Span: "ولولا تدافع الرسل والمؤمنون لما ثبتت السماوات والأرض"
{{"rationale": "No verbatim run, but a recognizable rewording of البقرة 251: 'دفع الله الناس بعضهم ببعض لفسدت الأرض' rephrased as 'تدافع الرسل والمؤمنون … ثبتت السماوات والأرض'. Same specific verse, recovered by meaning.",
  "verdict": "ok",
  "candidates": [
    {{"text": "فَهَزَمُوهُم بِإِذْنِ اللَّهِ وَقَتَلَ دَاوُودُ جَالُوتَ وَآتَاهُ اللَّهُ الْمُلْكَ وَالْحِكْمَةَ وَعَلَّمَهُ مِمَّا يَشَاءُ ۗ وَلَوْلَا دَفْعُ اللَّهِ النَّاسَ بَعْضَهُم بِبَعْضٍ لَّفَسَدَتِ الْأَرْضُ وَلَٰكِنَّ اللَّهَ ذُو فَضْلٍ عَلَى الْعَالَمِينَ",
      "source": "سورة البقرة 251",
      "matched_run": "دفع الله الناس بعضهم ببعض لفسدت الأرض (rephrased as تدافع الرسل والمؤمنون … السماوات والأرض)",
      "confidence": "paraphrase"}}
]}}
 
# Hadith — same matn attested in two books → ok, two candidates (matn identifies it)
Span: "تزوجوا الودود الولود"
{{"rationale": "'تزوجوا الودود الولود' is the verbatim opening matn of one specific hadith, attested with the same wording in two of the six books.",
  "verdict": "ok",
  "candidates": [
    {{"text": "تَزَوَّجُوا الْوَدُودَ الْوَلُودَ فَإِنِّي مُكَاثِرٌ بِكُمُ الأُمَمَ", "source": "سنن أبي داود",
      "matched_run": "تزوجوا الودود الولود", "confidence": "high"}},
    {{"text": "تَزَوَّجُوا الْوَدُودَ الْوَلُودَ فَإِنِّي مُكَاثِرٌ بِكُمُ الأُمَمَ", "source": "سنن النسائي",
      "matched_run": "تزوجوا الودود الولود", "confidence": "high"}}
]}}
 
# Hadith — verbatim matn spine + a fused extra clause → ok, high
Span: "أفضل الصدقة سقي الماء والحج المبرور"
{{"rationale": "Verbatim matn spine 'أفضل الصدقة سقي الماء' matches one specific hadith; 'والحج المبرور' is fused on.",
  "verdict": "ok",
  "candidates": [
    {{"text": "أَفْضَلُ الصَّدَقَةِ سَقْيُ الْمَاءِ", "source": "سنن ابن ماجه",
      "matched_run": "أفضل الصدقة سقي الماء", "confidence": "high"}}
]}}
 
# Hadith — overlap is ONLY isnad/honorific boilerplate → خطأ
Span: "قال رسول الله صلى الله عليه وسلم في أمر عظيم من أمور الدين"
{{"rationale": "Only the isnad/honorific frame is shared; no distinctive matn phrase reproduces any single hadith, and it is not a rewording of one. Boilerplate-only.",
  "verdict": "خطأ", "candidates": []}}
 
# Quran — topic + scattered single words only → خطأ
Span: "وَالْحَجُّ أَشَدُّ الْفُرُوجِ"
{{"rationale": "'الحج' occurs in البقرة 197 and 'فروج' in المؤمنون 5, but the span is not a rewording of any ONE verse — scattered words across different verses.",
  "verdict": "خطأ", "candidates": []}}
 
# Mislabeled — presented as an ayah but is a distortion of a hadith → خطأ (ayah channel)
Span: "ولا تقولوا لما أصابكم لو كنا فاعلين لكنا خيرا من الذين فعلوا"
{{"rationale": "Not a Quranic verse; a distortion of the hadith 'وإن أصابك شيء فلا تقل لو أني فعلت … ولكن قل قدر الله وما شاء فعل'. As an ayah it corresponds to no specific verse.",
  "verdict": "خطأ", "candidates": []}}
 
# Garbled / stitched fragments → خطأ
Span: "وَمَنْ لَا يُحِيطُ بِمَالِهِ إِلَّا بِإِذْنِهِ يُحِيطُ بِمَالِهِ وَسَعَ كُرْسِيُّ السَّلَامُ…"
{{"rationale": "Incoherent fragments stitched together; corresponds to no single real text.",
  "verdict": "خطأ", "candidates": []}}
"""


GEMINI_CORRECTION_PROMPT_V3  = """\
You are a hafiz and Islamic scholar with complete memorization of the Holy Quran (Uthmani script)
and the six canonical Hadith collections (البخاري، مسلم، أبو داود، الترمذي، النسائي، ابن ماجه).
 
You receive ONE Arabic span that was presented as a Quranic ayah or a Prophetic hadith but is
INCORRECT — misquoted, misdiacritized, paraphrased, garbled, fused, or fabricated. Your job: name
the real canonical text(s) the span is a distortion of, and return them.
 
════════════════════════════════════════════════════════════════
GOVERNING PRINCIPLE
════════════════════════════════════════════════════════════════
If you can name a SPECIFIC verse or hadith that the span distorts, you MUST return it as a
candidate. Recognition IS the correction. Do not decline a text you can identify.
 
A span can be wrong in many ways and still be identifiable. NONE of these justify خطأ on their own —
they are corrections to make, not reasons to give up:
  • The span MIXES two or more real texts → return EACH text it distorts (one candidate per text).
  • The span is GARBLED or heavily paraphrased but a specific text is recognizable → return it.
  • The span FUSES the ending of one verse onto another → return the text(s) it draws from.
  • The span is MISLABELED (a real verse presented as hadith, or vice versa) → still return the
    real text you identified; note the true type in the rationale.
 
Return "خطأ" ONLY when you cannot name any specific text — i.e. the span is a generic statement on
a theme that reworps no particular verse/hadith, or is fabricated from nothing. If your reasoning
mentions a specific ayah or hadith by name, that is proof it is NOT خطأ — return that text.
 
Matching rests on DISTINCTIVE CONTENT. A match on only isnad/honorific boilerplate
(قال رسول الله، صلى الله عليه وسلم، عن فلان) is not a match. For hadith, identify by the MATN; the
reference number is not important.
 
════════════════════════════════════════════════════════════════
OUTPUT — valid JSON only, fields in this order:
{{
  "rationale": "<which specific text(s) the span distorts, and how (mixed / garbled / fused /
                mislabeled / paraphrased) — or, if خطأ, why no specific text can be named>",
  "verdict": "ok" | "خطأ",
  "candidates": [
    {{"text": "<exact canonical text: full Uthmani ayah with tashkeel, or hadith matn only>",
      "source": "سورة <name> <n>" OR "<collection>",
      "confidence": "high" | "medium" | "low" | "paraphrase"}}
  ]
}}
 
confidence:
  high       — long verbatim run; clearly this text, differing only in diacritics/spelling.
  medium     — a real content phrase matches, with heavier corruption or a fused clause.
  low        — only a short content phrase matches.
  paraphrase — no verbatim run, but the span is a recognizable rewording of ONE specific text.
Return one candidate per distinct text (so a mixed span yields multiple). Rank high → paraphrase.
When verdict is "خطأ", candidates must be [].
 
════════════════════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════════════════════
# Mixes two real verses → return BOTH (not خطأ)
Span: "ولتسبحوا بحمده وتسبحوا في الصباح والعشي"
{{"rationale": "Distorts two verses on glorifying God morning and evening: غافر 55 'وسبح بحمد ربك بالعشي والإبكار' and الفتح 9 'وتسبحوه بكرة وأصيلا'. Return both.",
  "verdict": "ok",
  "candidates": [
    {{"text": "فَاصْبِرْ إِنَّ وَعْدَ اللَّهِ حَقٌّ وَاسْتَغْفِرْ لِذَنبِكَ وَسَبِّحْ بِحَمْدِ رَبِّكَ بِالْعَشِيِّ وَالْإِبْكَارِ", "source": "سورة غافر 55", "confidence": "paraphrase"}},
    {{"text": "لِّتُؤْمِنُوا بِاللَّهِ وَرَسُولِهِ وَتُعَزِّرُوهُ وَتُوَقِّرُوهُ وَتُسَبِّحُوهُ بُكْرَةً وَأَصِيلًا", "source": "سورة الفتح 9", "confidence": "paraphrase"}}
]}}
 
# Mislabeled (verse presented as hadith) → still return the verse (not خطأ)
Span: "لا يكلف الله نفسا إلا وهي طاقة"
{{"rationale": "A distortion of البقرة 286 'لا يكلف الله نفسا إلا وسعها' (cf. الطلاق 7). Presented as hadith, but it is a Quranic verse — return the verse and note the mislabeling.",
  "verdict": "ok",
  "candidates": [
    {{"text": "لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا ۚ لَهَا مَا كَسَبَتْ وَعَلَيْهَا مَا اكْتَسَبَتْ", "source": "سورة البقرة 286", "confidence": "high"}}
]}}
 
# Garbled but one verse recognizable → return it (not خطأ)
Span: "وأحللنا لكم ليلة الفطر وأحرمّا لكم منافعكم"
{{"rationale": "Garbled, but mimics البقرة 187 'أحل لكم ليلة الصيام الرفث إلى نسائكم' with 'الفطر' swapped for 'الصيام' and an incoherent tail. One recognizable verse.",
  "verdict": "ok",
  "candidates": [
    {{"text": "أُحِلَّ لَكُمْ لَيْلَةَ الصِّيَامِ الرَّفَثُ إِلَىٰ نِسَائِكُمْ ۚ هُنَّ لِبَاسٌ لَّكُمْ وَأَنتُمْ لِبَاسٌ لَّهُنَّ", "source": "سورة البقرة 187", "confidence": "medium"}}
]}}
 
# Fuses two verses → return both texts it draws from
Span: "ولا تقتلوا أنفسكم إن الله لا يحب المسرفين"
{{"rationale": "Fuses النساء 29 'ولا تقتلوا أنفسكم إن الله كان بكم رحيما' with the ending 'لا يحب المسرفين' from الأعراف 31. Return both sources.",
  "verdict": "ok",
  "candidates": [
    {{"text": "وَلَا تَقْتُلُوا أَنفُسَكُمْ ۚ إِنَّ اللَّهَ كَانَ بِكُمْ رَحِيمًا", "source": "سورة النساء 29", "confidence": "high"}},
    {{"text": "وَكُلُوا وَاشْرَبُوا وَلَا تُسْرِفُوا ۚ إِنَّهُ لَا يُحِبُّ الْمُسْرِفِينَ", "source": "سورة الأعراف 31", "confidence": "medium"}}
]}}
 
# Genuinely unidentifiable → خطأ
Span: "التراويح في الحج مثل الصلاة في المسجد النبوي فمن تراوح فيه يلعب من الصلاة"
{{"rationale": "Fluent and on-topic but reworps no specific verse or hadith; no single text is nameable. Fabricated.",
  "verdict": "خطأ", "candidates": []}}
"""


def gate_user(span, ref):
    return (f"The following span was presented as a {REF_LABEL.get(ref, ref)} but is "
            f"INCORRECT. Judge whether it is a recoverable distortion of a real canonical "
            f"text, or baseless (خطأ).\n\nSpan:\n<<<\n{span}\n>>>")

def correct_user(span, ref):
    return (f"The following span was presented as a {REF_LABEL.get(ref, ref)} but is "
            f"INCORRECT; a prior step judged it correctable. Return every canonical text it "
            f"genuinely distorts (or خطأ if none truly does).\n\nSpan:\n<<<\n{span}\n>>>")

def oneshot_user(span, ref):
    return (f"The following span was presented as a {REF_LABEL.get(ref, ref)} but is "
            f"INCORRECT. Decide if it is a recoverable distortion of a real canonical text "
            f"and, if so, return every canonical text it genuinely distorts (or خطأ).\n\n"
            f"Span:\n<<<\n{span}\n>>>")


# ─────────────────────────────────────────────────────────────────────────────
# rasm-robust text match (for correction agreement)
# ─────────────────────────────────────────────────────────────────────────────
_T2 = {**_TABLE, 0x0621: None}
_LET = re.compile("[^" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A) + r"\s]")
def text_sim(a, b):
    aw = _LET.sub(" ", (a or "").translate(_T2)).split()
    bw = _LET.sub(" ", (b or "").translate(_T2)).split()
    if not aw or not bw:
        return 0.0
    return difflib.SequenceMatcher(None, aw, bw, autojunk=False).ratio()


# ─────────────────────────────────────────────────────────────────────────────
# IO / resume
# ─────────────────────────────────────────────────────────────────────────────
def uid_for(i, r):
    h = hashlib.sha1(f"{r['id']}|{r['model']}|{r['ref']}|{r['span']}".encode()).hexdigest()[:8]
    return f"{i:05d}__{r['id']}__{r['model']}__{h}"

def load_spans(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = []
    for i, r in enumerate(rows):
        if r["ref"] not in REF_LABEL:
            continue
        out.append({"uid": uid_for(i, r), "id": r["id"], "answer_model": r["model"],
                    "ref": r["ref"], "span": r["span"], "error_type": r.get("error_type", "")})
    return out

def load_state(path):
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    return {s["uid"]: s for s in data.get("per_sample", [])}

def save(path, recs):
    per = list(recs.values())
    tin = tout = 0.0
    by_model = {}
    for s in per:
        for stg in ("gate", "correction", "oneshot"):
            for m, d in (s.get(stg) or {}).items():
                u = (d or {}).get("usage") or {}
                ti, to = u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0
                tin += ti; tout += to
                bm = by_model.setdefault(m, [0, 0]); bm[0] += ti; bm[1] += to
    cost = {}
    grand = 0.0
    for m, (ti, to) in by_model.items():
        pin, pout = PRICES.get(m, (None, None))
        c = round(ti/1e6*pin + to/1e6*pout, 4) if pin is not None else None
        cost[m] = c
        if c: grand += c
    snap = {"task": "correction_pipeline", "timestamp": datetime.now(timezone.utc).isoformat(),
            "n": len(per), "gate_models": GATE_MODELS, "correct_models": CORRECT_MODELS,
            "input_tokens": tin, "output_tokens": tout, "cost_by_model": cost,
            "est_cost_usd": round(grand, 4), "per_sample": per}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(snap, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# workers
# ─────────────────────────────────────────────────────────────────────────────
def call_stage(model, sysmsg, user, schema, reasoning_effort=None, max_completion_tokens=None):
    raw, usage = call_api(model, sysmsg, user, schema=schema,
                          reasoning_effort=reasoning_effort,
                          max_completion_tokens=max_completion_tokens)
    if raw is None:
        return {"prediction": None, "raw_response": None, "usage": None, "error": "API call failed"}
    return {"prediction": parse_json_response(raw), "raw_response": raw, "usage": usage}

def majority_gate(gate: Dict) -> Optional[str]:
    """Gate decision from the model votes. 'correctable' if correctable votes outnumber
    خطأ; 'خطأ' if the reverse; on a TIE (e.g. 2 models split) pass through as
    'correctable' (the correction stage re-vetoes). None if no valid votes."""
    votes = [((gate.get(m) or {}).get("prediction") or {}).get("verdict") for m in GATE_MODELS]
    votes = [v for v in votes if v in ("correctable", "خطأ")]
    if not votes:
        return None
    c, k = votes.count("correctable"), votes.count("خطأ")
    return "خطأ" if k > c else "correctable"


def run(input_path, output_path, stage="all", limit=None, workers=8, retry_errors=False):
    spans = load_spans(input_path)
    if limit:
        spans = spans[:limit]
    recs = load_state(output_path)
    for sp in spans:
        recs.setdefault(sp["uid"], {**sp, "gate": {}, "correction": {}, "oneshot": {}})
        recs[sp["uid"]].setdefault("oneshot", {})
    if retry_errors:
        # drop any stored model record that errored or has no verdict, so the resume
        # task-builder (which skips uids already present) will re-issue just those calls.
        dropped = 0
        for rec in recs.values():
            for stg in ("gate", "correction", "oneshot"):
                for m in list((rec.get(stg) or {})):
                    r = rec[stg][m]
                    if r is None or r.get("error") or not ((r.get("prediction") or {}).get("verdict")):
                        del rec[stg][m]; dropped += 1
        print(f"--retry-errors: cleared {dropped} errored/empty record(s) for re-issue")

    lock = threading.Lock()
    GATE_SYS = GATE_PROMPT.format()
    COR_SYS  = CORRECTION_PROMPT.format()
    ONESHOT_SYS = ONE_SHOT_CORRECTION_PROMPT.format()

    # ── STAGE 1: GATE ──
    if stage in ("all", "gate"):
        tasks = [(sp, m) for sp in spans for m in GATE_MODELS if m not in recs[sp["uid"]]["gate"]]
        print(f"\n=== GATE | {len(tasks)} calls ({len(spans)} spans × {len(GATE_MODELS)} models) ===")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_stage, m, GATE_SYS, gate_user(sp["span"], sp["ref"]), SCHEMA_GATE,
                              GATE_REASONING, GATE_MAX_TOKENS): (sp, m)
                    for sp, m in tasks}
            for fut in as_completed(futs):
                sp, m = futs[fut]; rec = fut.result()
                with lock:
                    recs[sp["uid"]]["gate"][m] = rec
                    done += 1
                    v = "ERR" if rec.get("error") else (rec.get("prediction") or {}).get("verdict", "?")
                    print(f"  [{done}/{len(tasks)}] gate {m:22} {sp['id']} -> {v}")
                    if done % 15 == 0:
                        save(output_path, recs)
        save(output_path, recs)

    # majority per span
    for s in recs.values():
        s["gate_majority"] = majority_gate(s.get("gate", {}))

    # ── STAGE 2: CORRECTION (majority-correctable only) ──
    if stage in ("all", "correct"):
        corr_spans = [sp for sp in spans if recs[sp["uid"]].get("gate_majority") == "correctable"]
        tasks = [(sp, m) for sp in corr_spans for m in CORRECT_MODELS if m not in recs[sp["uid"]]["correction"]]
        print(f"\n=== CORRECTION | {len(corr_spans)} correctable spans × {len(CORRECT_MODELS)} models = {len(tasks)} calls ===")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_stage, m, COR_SYS, correct_user(sp["span"], sp["ref"]), SCHEMA_CORRECTION,
                              CORRECT_REASONING, CORRECT_MAX_TOKENS): (sp, m)
                    for sp, m in tasks}
            for fut in as_completed(futs):
                sp, m = futs[fut]; rec = fut.result()
                with lock:
                    recs[sp["uid"]]["correction"][m] = rec
                    done += 1
                    p = rec.get("prediction") or {}
                    tag = "ERR" if rec.get("error") else f"{p.get('verdict','?')},{len(p.get('candidates',[]) or [])}c"
                    print(f"  [{done}/{len(tasks)}] corr {m:22} {sp['id']} -> {tag}")
                    if done % 15 == 0:
                        save(output_path, recs)
        save(output_path, recs)

    # ── ONE-SHOT: gate+correct in a single call, on ALL spans (no prior gate) ──
    if stage in ("oneshot",):
        tasks = [(sp, m) for sp in spans for m in CORRECT_MODELS if m not in recs[sp["uid"]]["oneshot"]]
        print(f"\n=== ONE-SHOT | {len(spans)} spans × {len(CORRECT_MODELS)} models = {len(tasks)} calls ===")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_stage, m, ONESHOT_SYS, oneshot_user(sp["span"], sp["ref"]), SCHEMA_CORRECTION,
                              CORRECT_REASONING, CORRECT_MAX_TOKENS): (sp, m)
                    for sp, m in tasks}
            for fut in as_completed(futs):
                sp, m = futs[fut]; rec = fut.result()
                with lock:
                    recs[sp["uid"]]["oneshot"][m] = rec
                    done += 1
                    p = rec.get("prediction") or {}
                    tag = "ERR" if rec.get("error") else f"{p.get('verdict','?')},{len(p.get('candidates',[]) or [])}c"
                    print(f"  [{done}/{len(tasks)}] oneshot {m:22} {sp['id']} -> {tag}")
                    if done % 15 == 0:
                        save(output_path, recs)
        save(output_path, recs)

    analyze(recs, stage=stage)
    print(f"\n  -> wrote {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# analysis
# ─────────────────────────────────────────────────────────────────────────────
def _cands(rec):
    """Candidate texts from a correction/oneshot record when verdict==ok, else []."""
    p = (rec or {}).get("prediction") or {}
    if p.get("verdict") != "ok":
        return []
    return [c.get("text", "") for c in (p.get("candidates") or [])]

def _two_model_agree(per, stage, cm, sim_th):
    """Verdict + correction-text agreement between the 2 models of a stage."""
    both = [s for s in per if all((s.get(stage) or {}).get(m) and not s[stage][m].get("error") for m in cm)]
    av = at = 0
    for s in both:
        va = ((s[stage][cm[0]]["prediction"] or {}).get("verdict"))
        vb = ((s[stage][cm[1]]["prediction"] or {}).get("verdict"))
        av += (va == vb)
        A, B = _cands(s[stage][cm[0]]), _cands(s[stage][cm[1]])
        if A and B and any(text_sim(a, b) >= sim_th for a in A for b in B):
            at += 1
    return both, av, at

def analyze(recs, stage="all", sim_th=0.8):
    per = list(recs.values())
    print("\n================ ANALYSIS ================")
    cm = CORRECT_MODELS

    if stage in ("all", "gate", "correct"):
        def gv(s, m): return ((s["gate"].get(m) or {}).get("prediction") or {}).get("verdict")
        print("GATE verdicts by model:")
        for m in GATE_MODELS:
            print(f"  {m:22} {dict(Counter(gv(s, m) for s in per))}")
        print(f"GATE decision: {dict(Counter(s.get('gate_majority') for s in per))}")
        unan = sum(1 for s in per if all(s['gate'].get(m) for m in GATE_MODELS)
                   and len({gv(s, m) for m in GATE_MODELS}) == 1)
        print(f"GATE unanimous (all {len(GATE_MODELS)} agree): {unan}/{len(per)}")
        if any(s.get("error_type") for s in per):
            print("GATE decision vs manual error_type:")
            for (et, gm), n in sorted(Counter((s.get("error_type"), s.get("gate_majority")) for s in per).items()):
                print(f"    {str(et):16} -> {str(gm):12} {n}")
        both, av, at = _two_model_agree(per, "correction", cm, sim_th)
        n = len(both)
        print(f"\nCORRECTION ({cm[0]} vs {cm[1]}) on {n} both-answered spans:")
        if n:
            print(f"  verdict agreement (ok/خطأ): {av}/{n} = {100*av/n:.0f}%")
            print(f"  agree on a correction text (sim>={sim_th}): {at}/{n} = {100*at/n:.0f}%")

    if stage == "oneshot":
        def ov(s, m): return ((s["oneshot"].get(m) or {}).get("prediction") or {}).get("verdict")
        print("ONE-SHOT verdicts by model:")
        for m in cm:
            print(f"  {m:22} {dict(Counter(ov(s, m) for s in per))}")
        if any(s.get("error_type") for s in per):
            print("ONE-SHOT verdict (union ok if any model ok) vs manual error_type:")
            def uni(s): return "ok" if any(ov(s, m) == "ok" for m in cm) else "خطأ"
            for (et, gm), n in sorted(Counter((s.get("error_type"), uni(s)) for s in per).items()):
                print(f"    {str(et):16} -> {str(gm):12} {n}")
        both, av, at = _two_model_agree(per, "oneshot", cm, sim_th)
        n = len(both)
        print(f"\nONE-SHOT ({cm[0]} vs {cm[1]}) on {n} both-answered spans:")
        if n:
            print(f"  verdict agreement (ok/خطأ): {av}/{n} = {100*av/n:.0f}%")
            print(f"  agree on a correction text (sim>={sim_th}): {at}/{n} = {100*at/n:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Two-stage gate+correction pipeline")
    ap.add_argument("--input", default="../outputs/correction/sample_correction_final_classified.csv")
    ap.add_argument("--output", default="../outputs/correction/pipeline_results.json")
    ap.add_argument("--stage", choices=["all", "gate", "correct", "oneshot"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-issue calls that errored / returned no verdict in a prior run")
    args = ap.parse_args()
    for m in GATE_MODELS + CORRECT_MODELS:
        if m not in MODEL_REGISTRY:
            sys.exit(f"Unknown model {m}")
        _, _, key = MODEL_REGISTRY[m]
        if not os.environ.get(key):
            sys.exit(f"{key} not set for {m}")
    run(args.input, args.output, stage=args.stage, limit=args.limit, workers=args.workers,
        retry_errors=args.retry_errors)


if __name__ == "__main__":
    main()
