# IslamicEval 2026

Fine-grained, source-aware hallucination detection in Arabic Islamic content.

See [CLAUDE.md](../CLAUDE.md) for full project documentation, pipeline design, and instructions.

## Quick Start

```bash
# Stage 1: Classify raw prompts
python scripts/classify_prompts.py --input data/raw/prompts.csv --output data/classified/classified.csv

# Stage 2: Generate answers (vLLM, recommended)
sbatch jobs/run_inference.sh allam-7b

# Or locally with HuggingFace
python scripts/inference.py --model allam-7b

# Stage 3: Annotate answers
python scripts/annotate.py --input outputs/answers/allam-7b.json

# Stage 6: Run empirical analysis
python scripts/analyze.py
```

## Structure

```
islamiceval/
├── data/
│   ├── raw/                    Raw Fanar prompts (31k)
│   ├── classified/             Islamic-classified prompts
│   ├── prompts_100.json        100-prompt test set
│   └── prompts_1000.json       Full classified set
├── scripts/
│   ├── classify_prompts.py     Stage 1
│   ├── inference.py            Stage 2 (HuggingFace)
│   ├── inference_vllm.py       Stage 2 (vLLM, preferred)
│   ├── annotate.py             Stage 3
│   └── analyze.py              Stage 6
├── jobs/
│   └── run_inference.sh        SLURM job script
├── outputs/
│   ├── answers/                Model-generated answers
│   ├── annotations/            Gold annotations
│   └── analysis/               Metrics and plots
├── prompts/
│   └── templates/              Prompt templates for each stage
└── logs/                       SLURM job logs
```
