"""
Stage 3: Frontier LLM Annotation Pipeline
Uses an ensemble of frontier LLMs (GPT-4o, Claude Opus) to annotate citation
spans in model-generated answers with fine-grained hallucination labels.

Only samples with 100% cross-annotator agreement are kept as gold labels.
Disagreements are saved separately for expert adjudication.

Labels:
  - correct                  (Quran + Hadith)
  - misattributed_source     (Quran + Hadith)
  - fabricated               (Quran + Hadith)
  - paraphrased_as_verbatim  (Quran only)
  - variant_attested_matn    (Hadith only)
  - fabricated_isnad         (Hadith only)

Usage:
    python annotate.py --input ../outputs/answers/allam-7b.json
"""

import os, json, time, argparse
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()
openai_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ANNOTATION_PROMPT = """You are an expert Islamic scholar annotating LLM-generated answers for hallucinations.

Given a question and an LLM answer, identify all citation spans (Quran verses or Hadith narrations) and classify each with one of these labels:
- correct: accurately cited and properly attributed
- misattributed_source: content is real but attributed to wrong surah / hadith collection
- fabricated: invented citation not found in authentic sources
- paraphrased_as_verbatim: Quranic text paraphrased but presented as exact recitation (Quran only)
- variant_attested_matn: Hadith matn has variant wordings in authentic collections (Hadith only)
- fabricated_isnad: invented chain of narration (Hadith only)

Respond ONLY with valid JSON in this exact format:
{
  "spans": [
    {
      "text": "the citation as it appears in the answer",
      "source_type": "quran" or "hadith",
      "label": "<one of the labels above>",
      "reasoning": "one sentence explanation"
    }
  ]
}
If there are no citation spans, return {"spans": []}."""


def annotate_gpt4o(question: str, answer: str) -> dict:
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ANNOTATION_PROMPT},
            {"role": "user",   "content": f"Question: {question}\n\nAnswer: {answer}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def annotate_claude(question: str, answer: str) -> dict:
    resp = anthropic_client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        system=ANNOTATION_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}],
    )
    return json.loads(resp.content[0].text)


def spans_agree(ann_a: dict, ann_b: dict) -> bool:
    a = {s["text"]: s["label"] for s in ann_a.get("spans", [])}
    b = {s["text"]: s["label"] for s in ann_b.get("spans", [])}
    return a == b


def main():
    parser = argparse.ArgumentParser(description="Annotate model answers (Stage 3)")
    parser.add_argument("--input",      required=True, help="Path to model answers JSON")
    parser.add_argument("--output-dir", default="../outputs/annotations/")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        answers = json.load(f)

    model_name = os.path.basename(args.input).replace(".json", "")
    agreed, disagreed, errors = [], [], []

    for item in tqdm(answers, desc=f"annotating {model_name}"):
        try:
            ann_gpt4o  = annotate_gpt4o(item["prompt"], item["answer"])
            time.sleep(0.5)
            ann_claude = annotate_claude(item["prompt"], item["answer"])

            if spans_agree(ann_gpt4o, ann_claude):
                agreed.append({**item, "annotations": ann_gpt4o["spans"]})
            else:
                disagreed.append({
                    **item,
                    "gpt4o_annotations":  ann_gpt4o["spans"],
                    "claude_annotations": ann_claude["spans"],
                })
        except Exception as e:
            print(f"Error on id={item.get('id', '?')}: {e}")
            errors.append({**item, "error": str(e)})

    os.makedirs(args.output_dir, exist_ok=True)
    for suffix, data in [("agreed", agreed), ("disagreed", disagreed), ("errors", errors)]:
        if data:
            path = os.path.join(args.output_dir, f"{model_name}_{suffix}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"{suffix:12s}: {len(data):4d}  -> {path}")

    total = len(answers)
    print(f"\nAgreement rate: {len(agreed)}/{total} = {len(agreed)/total:.1%}")


if __name__ == "__main__":
    main()
