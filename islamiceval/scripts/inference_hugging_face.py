"""
Stage 2: LLM Answer Generation (HuggingFace Transformers)
Generates answers with Quran/Hadith citations for Islamic prompts.
Uses 4-bit quantization (AWQ/NF4) to fit 7-13B models on 2080 Ti (11GB VRAM).

Usage:
    python inference_hugging_face.py --model allam-7b --input ../data/classified/rag_questions.json
    python inference_hugging_face.py --model qwen3-8b --no-quantize   # for A6000 (48GB)
"""

import os, json, argparse
# Reduce CUDA fragmentation (helps avoid OOM on variable-length batches). Must be
# set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

# Models whose chat templates don't accept a system role — system prompt is
# folded into the first user message instead.
NO_SYSTEM_ROLE = {"jais-13b", "jais-70b", "acegpt-8b"}

# Per-model extra kwargs passed to apply_chat_template to disable thinking output.
# DeepSeek-R1 models have no template flag — <think> blocks are stripped in post-processing.
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

# Models whose <think> blocks must be stripped in post-processing (no template flag).
STRIP_THINKING = {"deepseek-r1-llama-8b", "deepseek-r1-qwen-32b", "deepseek-r1-llama-70b"}

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

# used same models as islamiceval + mistral/acegpt/silma 
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
    "llama-3.3-70b":         "meta-llama/Llama-3.3-70B-Instruct",       
    "jais-70b":              "inceptionai/jais-adapted-70b-chat",

    # 2x A100 80GB
    "deepseek-r1-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",

    # 4x A100 80GB
    "qwen3-235b":            "Qwen/Qwen3-235B-A22B",                     # MoE, 22B active
    "llama-4-maverick":      "meta-llama/Llama-4-Maverick-17B-128E-Instruct",  # MoE
    "deepseek-v3":           "deepseek-ai/DeepSeek-V3-0324",             # 685B MoE, 37B active
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
    # Batched decoder generation needs LEFT padding and a pad token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def build_messages(prompt: str, model_key: str) -> list:
    if model_key in NO_SYSTEM_ROLE:
        # Fold system prompt into the user turn for models without system role support
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]


def generate_with_retry(prompts, tokenizer, model, model_key, max_new_tokens, temperature, top_p):
    """Run a batch; on CUDA OOM, free cache and recursively halve it so one long
    batch can't poison many answers. Only a lone prompt that still OOMs is recorded
    as an error."""
    try:
        return generate_batch(prompts, tokenizer, model, model_key, max_new_tokens, temperature, top_p)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            print(f"\n  [OOM] single prompt too large — recording ERROR")
            return ["ERROR: CUDA OOM on single prompt"]
        mid = len(prompts) // 2
        print(f"\n  [OOM] splitting batch of {len(prompts)} -> {mid}+{len(prompts)-mid}")
        return (generate_with_retry(prompts[:mid], tokenizer, model, model_key, max_new_tokens, temperature, top_p)
                + generate_with_retry(prompts[mid:], tokenizer, model, model_key, max_new_tokens, temperature, top_p))


def generate_batch(prompts, tokenizer, model, model_key: str, max_new_tokens: int = 512,
                   temperature: float = None, top_p: float = None) -> list:
    """Generate answers for a batch of prompts in one model.generate() call."""
    extra = THINKING_KWARGS.get(model_key, {})
    texts = [
        tokenizer.apply_chat_template(build_messages(p, model_key), tokenize=False,
                                      add_generation_prompt=True, **extra)
        for p in prompts
    ]
    # Template strings already include special tokens -> add_special_tokens=False.
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=4096, add_special_tokens=False).to(model.device)

    do_sample = temperature > 0     # temperature=0 -> greedy
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = output[:, inputs["input_ids"].shape[1]:]   # left-padded -> same input width for all
    answers = tokenizer.batch_decode(gen, skip_special_tokens=True)
    if model_key in STRIP_THINKING:
        import re
        answers = [re.sub(r"<think>.*?</think>", "", a, flags=re.DOTALL).strip() for a in answers]
    else:
        answers = [a.strip() for a in answers]
    return answers


def load_prompts(path):
    """Load prompts as a list of {'id', 'prompt'}.
    .xlsx -> reads the 'qid' and 'prompt' columns; .json -> list of {'id','prompt'}.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        from openpyxl import load_workbook
        rows = load_workbook(path, read_only=True, data_only=True).active.iter_rows(values_only=True)
        header = list(next(rows))
        qi, pi = header.index("qid"), header.index("prompt")
        return [{"id": r[qi], "prompt": r[pi]} for r in rows if r[pi]]
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported input '{ext}' (use .xlsx or .json)")


def main():
    parser = argparse.ArgumentParser(description="Generate answers with LLMs (Stage 2)")
    parser.add_argument("--model",       required=True, choices=list(MODELS.keys()))
    parser.add_argument("--input",       default="../data/classified/rag_questions.json",
                        help="prompts file: .xlsx (qid+prompt) or .json")
    parser.add_argument("--output-dir",  default="../outputs/answers/")
    parser.add_argument("--max-tokens",  type=int, default=512)
    parser.add_argument("--batch-size",  type=int, default=16,
                        help="Prompts per generate() call (raise if GPU memory allows)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override sampling temperature (default: the model's own; pass 0 for greedy)")
    parser.add_argument("--top-p",       type=float, default=None,
                        help="Override nucleus sampling top-p (default: the model's own)")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization (for A6000)")
    args = parser.parse_args()

    prompts = load_prompts(args.input)
    print(f"Loaded {len(prompts)} prompts from {args.input}")

    model_id = MODELS[args.model]
    print(f"Loading {args.model} ({model_id})  quantize={not args.no_quantize}")
    tokenizer, model = load_model(model_id, quantize=not args.no_quantize)
    print("Model loaded.")

    temperature, top_p = resolve_sampling(model_id, args.temperature, args.top_p)
    print(f"Sampling on — temperature={temperature}, top_p={top_p}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.model}.json")

    # Resume: skip prompts already processed
    done_ids = set()
    results = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        before = len(results)
        results = [r for r in results if not str(r["answer"]).startswith("ERROR")]  # retry failed ones
        done_ids = {r["id"] for r in results}
        dropped = before - len(results)
        print(f"Resuming — {len(done_ids)} already done, {len(prompts) - len(done_ids)} remaining"
              + (f" (regenerating {dropped} previous ERROR answers)" if dropped else "") + ".")

    def save():
        # atomic write so a crash mid-save can't corrupt the checkpoint
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)

    todo = [p for p in prompts if p["id"] not in done_ids]
    bs = args.batch_size
    save_every = 128                 # checkpoint cadence, independent of batch size
    last_saved = len(results)
    for i in tqdm(range(0, len(todo), bs), desc=args.model):
        chunk = todo[i:i + bs]
        try:
            answers = generate_with_retry([c["prompt"] for c in chunk], tokenizer, model,
                                          args.model, args.max_tokens, temperature, top_p)
        except Exception as e:
            print(f"\nError on batch at {i}: {e}")
            answers = [f"ERROR: {e}"] * len(chunk)
        for c, a in zip(chunk, answers):
            results.append({"id": c["id"], "prompt": c["prompt"], "answer": a, "model": args.model})
        if len(results) - last_saved >= save_every:
            save()
            last_saved = len(results)
    save()                            # final flush (remaining < save_every)

    print(f"Saved {len(results)} answers -> {out_path}")


if __name__ == "__main__":
    main()
