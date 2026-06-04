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

THINKING_KWARGS = {
    "fanar-2-27b":           {"no_thinking": True},
    "qwen3-0.6b":            {"enable_thinking": False},
    "qwen3-1.7b":            {"enable_thinking": False},
    "qwen3-4b":              {"enable_thinking": False},
    "qwen3-8b":              {"enable_thinking": False},
    "qwen3-14b":             {"enable_thinking": False},
    "qwen3-30b-a3b":         {"enable_thinking": False},
    "qwen3-32b":             {"enable_thinking": False},
}

STRIP_THINKING = {"deepseek-r1-llama-8b", "deepseek-r1-qwen-32b", "deepseek-r1-llama-70b"}

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

# Sampling fallbacks for models whose generation_config doesn't enable sampling.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.9


def resolve_sampling(model_id: str, cli_temperature, cli_top_p):
    """Return (temperature, top_p) to sample with.

    We always sample. If the model's own generation_config enables sampling
    (do_sample=True), use its temperature/top_p; otherwise fall back to the
    DEFAULT_* values. A CLI flag, if given, overrides either source.
    """
    temperature, top_p = cli_temperature, cli_top_p
    if temperature is None or top_p is None:
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(model_id)
            samples = bool(getattr(gc, "do_sample", False))
        except Exception:
            gc, samples = None, False
        if temperature is None:
            temperature = gc.temperature if (samples and gc.temperature) else DEFAULT_TEMPERATURE
        if top_p is None:
            top_p = gc.top_p if (samples and gc.top_p) else DEFAULT_TOP_P
    return temperature, top_p

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

islamic_eval_models = ["allam-7b", "jais-13b", "acegpt-8b", "silma-9b", "fanar-1-9b", "qwen3-8b", "gemma-3-12b", "mistral-7b", "deepseek-r1-llama-8b", "llama-3.1-8b"]

# Named groups you can pass to --model to run several models back-to-back.
MODEL_GROUPS = {
    "islamic-eval": islamic_eval_models,
    "all":          list(MODELS.keys()),
}


def run_group(args, model_names):
    """Run several models sequentially, each in its own fresh subprocess so every
    model gets a clean GPU (vLLM doesn't reliably free VRAM within one process)."""
    import subprocess, sys
    print(f"Running {len(model_names)} models: {model_names}")
    failed = []
    for i, m in enumerate(model_names, 1):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--model", m,
               "--input", args.input,
               "--output-dir", args.output_dir,
               "--max-tokens", str(args.max_tokens),
               "--tensor-parallel", str(args.tensor_parallel)]
        if args.temperature is not None: cmd += ["--temperature", str(args.temperature)]
        if args.top_p       is not None: cmd += ["--top-p",       str(args.top_p)]
        print(f"\n{'='*60}\n[{i}/{len(model_names)}] {m}\n{'='*60}")
        if subprocess.run(cmd).returncode != 0:
            print(f"[WARN] {m} failed — continuing with the rest.")
            failed.append(m)
    print(f"\nDone. {len(model_names) - len(failed)}/{len(model_names)} succeeded."
          + (f"  Failed: {failed}" if failed else ""))


def main():
    parser = argparse.ArgumentParser(description="Generate answers with vLLM (Stage 2)")
    parser.add_argument("--model",           required=True,
                        help="a model key, or a group: " + " | ".join(MODEL_GROUPS))
    parser.add_argument("--input",           default="../data/classified/rag_questions.json")
    parser.add_argument("--output-dir",      default="../outputs/answers/")
    parser.add_argument("--max-tokens",      type=int, default=512)
    parser.add_argument("--temperature",     type=float, default=None,
                        help="Override sampling temperature (default: the model's own; pass 0 for greedy)")
    parser.add_argument("--top-p",           type=float, default=None,
                        help="Override nucleus sampling top-p (default: the model's own)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (for 70B+ models)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        prompts = json.load(f)

    model_id = MODELS[args.model]
    print(f"Loading {args.model} with vLLM ({model_id})  tp={args.tensor_parallel}")
    temperature, top_p = resolve_sampling(model_id, args.temperature, args.top_p)
    print(f"Sampling on — temperature={temperature}, top_p={top_p}")

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

    sampling = SamplingParams(temperature=temperature, top_p=top_p,
                              max_tokens=args.max_tokens)

    def build_messages(item):
        if args.model in NO_SYSTEM_ROLE:
            return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{item['prompt']}"}]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item["prompt"]},
        ]

    extra_template_kwargs = THINKING_KWARGS.get(args.model, {})

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

    import re
    def clean(text: str) -> str:
        if args.model in STRIP_THINKING:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    results = [
        {
            "id":     item["id"],
            "prompt": item["prompt"],
            "answer": clean(out.outputs[0].text),
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
