"""
Stage 2: LLM Answer Generation (vLLM)
High-throughput batch inference for large-scale answer generation on the cluster.
Preferred over inference.py when running all 658+ prompts against multiple models.

Usage:
    python inference_vllm.py --model qwen3-8b
    python inference_vllm.py --model llama-3.3-70b --tensor-parallel 4
"""

import os, json, argparse
from vllm import LLM, SamplingParams

NO_SYSTEM_ROLE = {"jais-13b", "jais-70b", "acegpt-8b"}

# Models with reasoning/thinking mode — disable it for clean, comparable outputs.
NO_THINKING = {"fanar-2-27b", "deepseek-r1-llama-8b", "deepseek-r1-qwen-32b",
               "deepseek-r1-llama-70b"}

# Gemma-3 is multimodal (vision+text). vLLM profiles the vision encoder at
# startup even for text-only inference, which OOMs on small MIG slices.
# Passing limit_mm_per_prompt={"image": 0} skips that profiling.
# All other models in MODELS are text-only — don't pass this param to them.
MULTIMODAL_MODELS = {"gemma-3-4b", "gemma-3-12b", "gemma-3-27b",
                     "llama-4-scout", "llama-4-maverick"}

SYSTEM_PROMPT = (
    "أنت مساعد إسلامي متخصص. أجب على السؤال بشكل دقيق ومختصر، "
    "مستنداً إلى القرآن الكريم والأحاديث النبوية الشريفة.\n"
    "عند الاستشهاد بآية قرآنية، اذكر اسم السورة ورقم الآية.\n"
    "عند الاستشهاد بحديث، اذكر المصدر (البخاري، مسلم، إلخ) إن أمكن."
)

MODELS = {
    # ==================== Arabic-centric ====================
    "allam-7b":              "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "jais-13b":              "inceptionai/jais-13b-chat",
    "acegpt-8b":             "FreedomIntelligence/AceGPT-v2-8B-Chat",
    "silma-9b":              "silma-ai/SILMA-9B-Instruct-v1.0",
    "fanar-1-9b":            "QCRI/Fanar-1-9B-Instruct",
    "fanar-2-27b":           "QCRI/Fanar-2-27B-Instruct",

    # ==================== Qwen family ======================
    "qwen3-0.6b":            "Qwen/Qwen3-0.6B",
    "qwen3-1.7b":            "Qwen/Qwen3-1.7B",
    "qwen3-4b":              "Qwen/Qwen3-4B",
    "qwen3-8b":              "Qwen/Qwen3-8B",
    "qwen3-14b":             "Qwen/Qwen3-14B",
    "qwen3-30b-a3b":         "Qwen/Qwen3-30B-A3B",           # MoE, 3B active

    # ==================== Llama family (gated) ==============
    "llama-3.2-3b":          "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b":          "meta-llama/Llama-3.1-8B-Instruct",
    "llama-4-scout":         "meta-llama/Llama-4-Scout-17B-16E-Instruct",  # MoE

    # ==================== Gemma family (gated) ==============
    "gemma-3-4b":            "google/gemma-3-4b-it",
    "gemma-3-12b":           "google/gemma-3-12b-it",
    "gemma-3-27b":           "google/gemma-3-27b-it",

    # ==================== Mistral family ====================
    "mistral-7b":            "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-small-24b":     "mistralai/Mistral-Small-Instruct-2409",
    "mixtral-8x7b":          "mistralai/Mixtral-8x7B-Instruct-v0.1",  # MoE

    # ==================== DeepSeek family ===================
    "deepseek-r1-llama-8b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",

    # ==================== Other =============================
    "phi-4-14b":             "microsoft/phi-4",
    "glm-4-9b":              "THUDM/glm-4-9b-chat",
    "command-r-7b":          "CohereForAI/c4ai-command-r7b-12-2024",

    # ==========================================================
    # HEAVY COMPUTE — need A100 80GB+ or multi-GPU
    # ==========================================================

    # 1x A100 80GB
    "qwen3-32b":             "Qwen/Qwen3-32B",
    "deepseek-r1-qwen-32b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "llama-3.3-70b":         "meta-llama/Llama-3.3-70B-Instruct",        # tight, may need quantization
    "jais-70b":              "inceptionai/jais-adapted-70b-chat",

    # 2x A100 80GB
    "deepseek-r1-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",

    # 4x A100 80GB
    "qwen3-235b":            "Qwen/Qwen3-235B-A22B",                     # MoE, 22B active
    "llama-4-maverick":      "meta-llama/Llama-4-Maverick-17B-128E-Instruct",  # MoE
    "deepseek-v3":           "deepseek-ai/DeepSeek-V3-0324",             # 685B MoE, 37B active
}


def main():
    parser = argparse.ArgumentParser(description="Generate answers with vLLM (Stage 2)")
    parser.add_argument("--model",           required=True, choices=list(MODELS.keys()))
    parser.add_argument("--input",           default="../data/classified/rag_questions.json")
    parser.add_argument("--output-dir",      default="../outputs/answers/")
    parser.add_argument("--max-tokens",      type=int, default=512)
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (for 70B+ models)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        prompts = json.load(f)

    model_id = MODELS[args.model]
    print(f"Loading {args.model} with vLLM ({model_id})  tp={args.tensor_parallel}")
    llm_kwargs = dict(
        model=model_id,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    if args.model in MULTIMODAL_MODELS:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    # We use deterministic decoding for evaluation here but we can change it later.
    sampling  = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    def build_messages(item):
        if args.model in NO_SYSTEM_ROLE:
            return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{item['prompt']}"}]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item["prompt"]},
        ]

    extra_template_kwargs = {"no_thinking": True} if args.model in NO_THINKING else {}

    conversations = []
    for item in prompts:
        try:
            text = tokenizer.apply_chat_template(
                build_messages(item), tokenize=False, add_generation_prompt=True,
                **extra_template_kwargs
            )
        except Exception as e:
            print(f"Template error on id={item['id']}: {e} — skipping")
            text = None
        conversations.append(text)

    valid_indices = [i for i, c in enumerate(conversations) if c is not None]
    valid_convs   = [conversations[i] for i in valid_indices]
    valid_prompts = [prompts[i] for i in valid_indices]
    skipped = len(prompts) - len(valid_indices)
    if skipped:
        print(f"Warning: {skipped} prompts skipped due to template errors.")

    print(f"Generating {len(valid_convs)} answers...")
    outputs = llm.generate(valid_convs, sampling)

    results = [
        {
            "id":     item["id"],
            "prompt": item["prompt"],
            "answer": out.outputs[0].text.strip(),
            "model":  args.model,
        }
        for item, out in zip(valid_prompts, outputs)
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} answers -> {out_path}")


if __name__ == "__main__":
    main()
