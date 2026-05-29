"""
Stage 2: LLM Answer Generation (HuggingFace Transformers)
Generates answers with Quran/Hadith citations for Islamic prompts.
Uses 4-bit quantization (AWQ/NF4) to fit 7-13B models on 2080 Ti (11GB VRAM).

Usage:
    python inference_hugging_face.py --model allam-7b --input ../data/classified/rag_questions.json
    python inference_hugging_face.py --model qwen3-8b --no-quantize   # for A6000 (48GB)
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

SYSTEM_PROMPT = (
    "أنت مساعد إسلامي متخصص. أجب على السؤال بشكل دقيق ومختصر، "
    "مستنداً إلى القرآن الكريم والأحاديث النبوية الشريفة.\n"
    "عند الاستشهاد بآية قرآنية، اذكر اسم السورة ورقم الآية.\n"
    "عند الاستشهاد بحديث، اذكر المصدر (البخاري، مسلم، إلخ) إن أمكن."
)

MODELS = {
    "allam-7b":     "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "jais-13b":     "inceptionai/jais-13b-chat",
    "qwen3-8b":     "Qwen/Qwen3-8B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-3-4b":   "google/gemma-3-4b-it",
    "mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "acegpt-8b":    "FreedomIntelligence/AceGPT-v2-8B-Chat",
    "silma-9b":     "silma-ai/SILMA-9B-Instruct-v1.0",
}


def load_model(model_id: str, quantize: bool = True):
    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def generate_answer(prompt: str, tokenizer, model, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            # determenisitc decoding here
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = output[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate answers with LLMs (Stage 2)")
    parser.add_argument("--model",       required=True, choices=list(MODELS.keys()))
    parser.add_argument("--input",       default="../data/classified/rag_questions.json")
    parser.add_argument("--output-dir",  default="../outputs/answers/")
    parser.add_argument("--max-tokens",  type=int, default=512)
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization (for A6000)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        prompts = json.load(f)

    model_id = MODELS[args.model]
    print(f"Loading {args.model} ({model_id})  quantize={not args.no_quantize}")
    tokenizer, model = load_model(model_id, quantize=not args.no_quantize)
    print("Model loaded.")

    results = []
    for item in tqdm(prompts, desc=args.model):
        answer = generate_answer(item["prompt"], tokenizer, model, args.max_tokens)
        results.append({
            "id":     item["id"],
            "prompt": item["prompt"],
            "answer": answer,
            "model":  args.model,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} answers -> {out_path}")


if __name__ == "__main__":
    main()
