"""
Stage 1: Islamic Prompt Classification
Classifies raw Fanar chat prompts as Islamic (1) or not (0).
A prompt is classified as 1 if it is about Islam AND can be answered
by citing Quran verses or Hadith narrations as primary evidence.

Usage:
    python classify_prompts.py --input ../data/raw/prompts.csv --output ../data/classified/classified.csv
"""

import os, json, time, argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a binary classifier for an Islamic question-answering system.\n\n"
    "Output 1 if the question is about Islam AND can be answered by citing Quran "
    "verses or Hadith narrations as primary supporting evidence. This includes:\n"
    "  - Islamic rulings on any topic (halal/haram, worship, ethics, transactions) "
    "— the Quran and Hadith establish the principles even if scholars elaborate later\n"
    "  - Questions about Quran content, structure, verses, or meaning\n"
    "  - Events and people from the Prophet's lifetime, including his companions "
    "(sahaba) — their stories are recorded in Hadith\n"
    "  - Islamic theology, belief, and practice\n"
    "  - Contemporary scenarios or ethical dilemmas asking for Islamic guidance\n\n"
    "Output 0 if ANY of the following apply:\n"
    "  - Not about Islam\n"
    "  - A conversational fragment that requires prior context to understand\n"
    "  - Asks specifically about post-prophetic political history as a historical "
    "fact (e.g. caliphate-era assassinations, who conquered where) with no Islamic "
    "guidance dimension\n"
    "  - Asks for a specific named madhab scholar's ruling or reasoning "
    "(e.g. 'why did Imam X / Sahnun / Ibn Hanbal rule that...')\n"
    "  - Asks about Islamic academic disciplines as a subject: usul al-fiqh "
    "methodology, hadith classification sciences (mawquf, mursal, etc.)\n"
    "  - Arabic grammar or linguistic analysis of Quranic text\n"
    "  - Quiz-style historical trivia (who/when/where) with no guidance dimension\n\n"
    "When uncertain, output 0.\n"
    "Reply with exactly one character - 0 or 1 - nothing else."
)

# Token IDs for "0" and "1" in o200k_base (gpt-4.x tokenizer)
LOGIT_BIAS = {15: 100, 16: 100}


def classify(text: str, model: str = "gpt-4.1") -> int:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": str(text)},
                ],
                max_tokens=1,
                temperature=0,
                logit_bias=LOGIT_BIAS,
            )
            tok = resp.choices[0].message.content.strip()
            return int(tok) if tok in ("0", "1") else -1
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return -1


def main():
    parser = argparse.ArgumentParser(description="Classify Islamic prompts (Stage 1)")
    parser.add_argument("--input",  default="../data/raw/prompts.csv", help="Input CSV with 'prompt' column")
    parser.add_argument("--output", default="../data/classified/classified.csv")
    parser.add_argument("--model",  default="gpt-4.1")
    parser.add_argument("--col",    default="label", help="Output column name")
    args = parser.parse_args()

    df = pd.read_csv(args.input).dropna(subset=["prompt"]).reset_index(drop=True)
    print(f"Loaded {len(df)} prompts from {args.input}")

    if args.col in df.columns and df[args.col].notna().all():
        print(f"Column '{args.col}' already complete, skipping.")
        return

    results = []
    for prompt in tqdm(df["prompt"], desc=f"classifying ({args.model})"):
        results.append(classify(prompt, model=args.model))

    df[args.col] = results
    n_errors  = (df[args.col] == -1).sum()
    n_islamic = (df[args.col] == 1).sum()
    print(f"Done. Islamic: {n_islamic}  Non-Islamic: {(df[args.col] == 0).sum()}  Errors: {n_errors}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")

    # Also export Islamic-only as JSON
    json_out = args.output.replace(".csv", "_islamic.json")
    islamic_df = df[df[args.col] == 1][["prompt"]].reset_index(drop=True)
    records = [
        {"id": str(i + 1).zfill(4), "prompt": row["prompt"]}
        for i, row in islamic_df.iterrows()
    ]
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Islamic-only JSON -> {json_out}  ({len(records)} prompts)")


if __name__ == "__main__":
    main()
