# IslamicEval 2026 — Project Overview

## What This Project Is

IslamicEval 2026 is a shared task on **fine-grained, source-aware hallucination detection in Arabic Islamic content**. It is the successor to IslamicEval 2025, the first shared task specifically targeting hallucination in LLM-generated Islamic text.

We (Tueki and Rahaf) are **co-organizers** of this shared task, supervised by Prof. Walid Magdy at the University of Edinburgh. Our responsibilities span the full lifecycle: designing the task, creating the dataset, building the annotation pipeline, running the shared task for participants, and conducting our own empirical analyses.

This project also forms the basis of Tueki's MSc dissertation, which focuses on a comprehensive empirical analysis of open-source LLMs on the IslamicEval 2026 benchmark.

---

## Why This Matters

LLMs are increasingly used for Arabic content, including religious topics. The Quran and Hadith carry sacred status in Islam — even minor misquotation, misattribution, or fabrication is unacceptable. LLM hallucinations in this domain can lead to:

- **Fabricated Hadiths** — entirely invented sayings attributed to the Prophet
- **Misattributed verses** — real content attributed to the wrong surah or source
- **Paraphrased-as-verbatim** — rephrased Quranic text presented as exact recitation
- **Fabricated isnads** — invented chains of narration for Hadiths
- **Irrelevant citations** — authentic but unrelated verses cited as answers

IslamicEval 2025 was the first effort to evaluate this, but used only binary correct/incorrect labels. IslamicEval 2026 introduces a **fine-grained, source-specific typology** to capture the diversity of hallucination types, plus a new **relevance subtask**.

---

## What Changed from 2025 to 2026

| Aspect | IslamicEval 2025 | IslamicEval 2026 |
|--------|-----------------|-----------------|
| Labels | Binary (correct/incorrect) | Fine-grained typology (7 categories) |
| Source distinction | Single label for both Quran and Hadith | Source-specific categories (Quran-only, Hadith-only, shared) |
| Relevance | Not evaluated | New subtask (Subtask C) |
| Evaluation | Cascading (span errors propagate) | Decoupled (gold spans provided for B and C) |
| Dataset size | ~1,500 annotated answers (251 questions × 6 models) | Target 10,000+ prompts with train/dev/test splits |
| Data source | Qur'an QA 2023 questions | Real Fanar user interactions |
| Annotation | Fully manual by Islamic studies experts | LLM-assisted pipeline with expert validation |
| Participant constraint | None | ≤13B parameters (encourages fine-tuning over brute-force scale) |

---

## Task Design

### Subtask A: Span Detection

Identify spans of text that the model presents as Quranic or Hadith citations within an LLM answer. Same as 2025.

**Metric:** Character-level macro-averaged F1 (classifying each character as Quran / Hadith / neither).

### Subtask B: Hallucination Identification

Given gold citation spans, assign each span a label from the typology:

| Category | Quran | Hadith | Scope |
|----------|-------|--------|-------|
| Correct | ✓ | ✓ | Shared |
| Misattributed source | ✓ | ✓ | Shared |
| Fabricated | ✓ | ✓ | Shared |
| Paraphrased-as-verbatim | ✓ | — | Quran-specific |
| Variant-attested matn | — | ✓ | Hadith-specific |
| Fabricated isnad | — | ✓ | Hadith-specific |

**Metric:** Macro-averaged F1 over typology. Scores reported separately for Quran and Hadith; ranking = average of both.

### Subtask C: Answer Relevance

Given a question and answer with gold citation spans, classify each span as **relevant** or **non-relevant** (binary, derived from a 4-tier annotation rubric: direct answer, indirect answer, relevant but no answer, non-relevant).

**Metric:** Macro-averaged F1, with per-question F1 averaged across all questions.

---

## Data Pipeline

### Source Data

31,000 chat prompts from the **Fanar** team (Fanar LLM user interactions). Of these:
- **500** already identified as Islamic by Fanar's RAG routing system (questions routed to their Islamic RAG pipeline using Quran and Hadith for grounding)
- **~30,500** remaining prompts being classified for Islamic relevance using LLM classification (GPT-4o-mini or Gemma)

The prompts are stratified by difficulty and topical area: general Islamic, tafsir, fiqh/fatwa-style, Hadith-focused, Quran-focused.

### Pipeline Stages

```
[Stage 1] Islamic Prompt Classification
    Input:  31k raw Fanar chat prompts
    Method: GPT-4o-mini / Gemma binary classification ("Islamic question that
            can be answered with Quran/Hadith references" vs. not)
    Output: Filtered set of Islamic prompts (target: thousands)
    Owner:  Rahaf (primary), Tueki (support)

[Stage 2] LLM Answer Generation
    Input:  Islamic prompts from Stage 1
    Method: Prompt diverse non-frontier LLMs (≤13B) to generate answers
            with Quran/Hadith citations
    Models: ALLaM-7B, Jais-13B, Qwen3-8B, Llama-3.1-8B, Gemma-3-4B,
            Mistral-7B, AceGPT-v2-8B, SILMA-9B
    Output: JSON files with (prompt, model_answer, metadata) per model
    Infra:  University of Edinburgh cluster (SLURM, 2080 Ti / A6000 GPUs)
    Owner:  Tueki (primary)

[Stage 3] Frontier LLM Annotation Pipeline
    Input:  Model answers from Stage 2
    Method: Ensemble of top frontier LLMs annotate each citation span with
            fine-grained labels. Only samples with 100% agreement across
            frontier annotators are used. Disagreement cases go to Islamic
            studies expert adjudication.
    Basis:  Frontier LLMs achieve >90% on IslamicMMLU (Quran, Hadith, Fiqh),
            validating their competence as annotators when paired with
            expert validation.
    Output: Gold-labeled dataset with fine-grained hallucination annotations
    Owner:  Joint (Tueki + Rahaf + organizer team)

[Stage 4] Quality Validation
    Input:  Annotated dataset from Stage 3
    Method: Manual expert review of a stratified sample to estimate
            annotation quality and calibrate agreement targets.
            200-item guideline-validation pilot.
    Output: Verified dataset ready for shared task release
    Owner:  Joint (organizer team)

[Stage 5] Shared Task Operations
    - Host on CodaBench/CodaLab
    - Separate leaderboards per subtask
    - Data, guidelines, scripts on GitHub + HuggingFace
    - Training-period webinar, office hours, mailing list
    Owner:  Joint (organizer team)

[Stage 6] Tueki's Empirical Analysis (Dissertation)
    Input:  Finalized benchmark dataset
    Method: Benchmark 20+ open-source/open-weight LLMs of varying sizes
    Output: Comprehensive analysis (see Dissertation section below)
    Owner:  Tueki
```

---

## Tueki's Dissertation Contribution

### Scope

A comprehensive empirical analysis of open-source/open-weight LLMs on the IslamicEval 2026 benchmark. This is independent from the shared task participants' submissions — Tueki evaluates a curated set of 20+ models as a systematic benchmarking study.

### Analysis Dimensions

- **Per-subtask performance** — which models excel at span detection vs. hallucination classification vs. relevance
- **Per-error-class breakdown** — which hallucination types are hardest to detect (fabrication vs. misattribution vs. paraphrased-as-verbatim, etc.)
- **Model size scaling** — within-family comparisons (e.g., Qwen3 1.7B → 4B → 8B → 14B → 32B)
- **Model family comparison** — Arabic-centric (ALLaM, Jais, AceGPT, SILMA) vs. multilingual (Qwen, Llama, Gemma, Mistral)
- **Citation source tendencies** — do certain models tend to cite Quran more than Hadith? Do they fabricate more in one domain?
- **Dense vs. MoE architectures** — performance differences between standard dense models and mixture-of-experts
- **Quran vs. Hadith performance gap** — systematic differences in model reliability across the two source types

### Candidate Model List (~25 models)

**Arabic-centric:** ALLaM-7B, Jais-13B-Chat, Jais-30B, AceGPT-v2-8B, SILMA-9B

**Qwen family (size scaling):** Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B

**Llama family:** Llama-3.2-3B, Llama-3.1-8B, Llama-3.3-70B

**Gemma family:** Gemma-3-4B, Gemma-3-12B, Gemma-3-27B

**Mistral family:** Mistral-7B, Mistral-Small-24B, Mixtral-8x7B (MoE)

**Other:** Phi-4-14B, DeepSeek-V3, Command-R-7B

### Relationship to IPP

The Individual Project Proposal (submitted earlier) served as a **feasibility study**:
- Identified annotation quality issues in IslamicEval 2025 gold labels
- Tested frontier LLMs as potential gold-truth annotators
- Benchmarked models on Arabic Islamic content to determine which are suitable
- Established that the proposed methodology is viable

The dissertation builds on these findings but is entirely focused on IslamicEval 2026.

---

## Technical Infrastructure

### Compute

- **University of Edinburgh SLURM cluster**
  - RTX 2080 Ti (11GB VRAM) — available on most nodes, 8 GPUs each
  - RTX A6000 (48GB VRAM) — landonia11, 8 GPUs
  - MIG-partitioned GPUs on saxa (18GB / 71GB slices)
  - Partitions: Teaching (default), ICF-Free, ICF-Research, Interactive
- **Prof. Walid Magdy's cluster** — access pending, expected to have stronger GPUs for large-scale runs

### APIs (for frontier model annotation and classification)

- OpenAI (university key) — GPT-4o-mini for classification, GPT-4o for annotation
- Anthropic — Claude for annotation pipeline
- Google — Gemini for annotation pipeline
- Together AI / OpenRouter — for accessing open models via API when needed

### Inference Stack

- **HuggingFace Transformers** — primary library for model loading and inference
- **vLLM** — high-throughput inference engine for large-scale batch runs
- **Quantization** — 4-bit AWQ/GPTQ for running 7B+ models on 2080 Ti (11GB)

### Key Tools

- **SLURM** — job scheduling on the university cluster
- **Overleaf** (XeLaTeX) — dissertation and paper writing
- **Notion** — project management and implementation notes
- **CodaBench/CodaLab** — shared task hosting platform
- **GitHub + HuggingFace** — data and code distribution

---

## Timeline

| Date | Milestone |
|------|-----------|
| Jun 1, 2026 | Website live, dev data + scripts + baselines released |
| Jun 15, 2026 | Training phase opens; webinar; office hours |
| Jul 30, 2026 | Registration deadline; blind test released |
| Jul 15, 2026 | Final results released |
| Aug 22, 2026 | System description papers due |
| Sep 1, 2026 | Shared task overview paper |

Tueki's dissertation follows the shared task timeline, with the empirical analysis running in parallel with and after the shared task evaluation period.

---

## Team

| Person | Role | Affiliation |
|--------|------|-------------|
| Tueki (s2870640) | Co-organizer, dissertation author (LLM analysis) | University of Edinburgh |
| Rahaf Alharbi | Co-organizer (data creation stream) | University of Edinburgh |
| Walid Magdy | Supervisor, task organizer | University of Edinburgh |
| Rana Malhas | Task organizer | Qatar University |
| Watheq Mansour | Task organizer | University of Queensland |
| Hamdy Mubarak | Task organizer | QCRI, HBKU |
| Kareem Darwish | Task organizer | QCRI, HBKU |
| Tamer Elsayed | Task organizer | Qatar University |

---

## Key References

- Mubarak et al., 2025 — IslamicEval 2025 shared task overview paper
- Abdelaal et al., 2026 — IslamicMMLU benchmark (validates frontier LLM competence on Islamic content)
- Fawzi et al., 2025/2026 — Hadith on social media; religious misinformation characterization
- Fanar Team et al., 2025 — Fanar Arabic-centric multimodal AI platform (source of prompts)
- Huang et al., 2025 — Survey on hallucination in LLMs

---

## Repository Structure

```
islamiceval/
├── README.md                   # This file
├── data/
│   ├── raw/                    # Raw Fanar prompts
│   ├── classified/             # Islamic-classified prompts
│   ├── prompts_100.json        # Small test set
│   └── prompts_1000.json       # Medium sample
├── scripts/
│   ├── classify_prompts.py     # Stage 1: Islamic prompt classification
│   ├── inference.py            # Stage 2: LLM answer generation (HuggingFace)
│   ├── inference_vllm.py       # Stage 2: LLM answer generation (vLLM)
│   ├── annotate.py             # Stage 3: Frontier LLM annotation pipeline
│   └── analyze.py              # Stage 6: Empirical analysis
├── jobs/
│   └── run_inference.sh        # SLURM job scripts
├── outputs/
│   ├── answers/                # Model-generated answers (Stage 2)
│   ├── annotations/            # Gold annotations (Stage 3)
│   └── analysis/               # Analysis results (Stage 6)
├── prompts/
│   └── templates/              # Prompt templates for each stage
└── logs/
    └── {job_id}.out            # SLURM job logs
```