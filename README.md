# The Cost of Asking

Code and data for *The Cost of Asking: A Fine-Grained Taxonomy of Hallucination in
LLM-Generated Islamic Citations*.

The paper evaluates Quranic and Hadith citations produced by 20 LLMs along two axes:
**textual fidelity** (is the quoted text correct, corrupted, or fabricated?) and
**attributional accuracy** (does the named source point to the right place?). Crossing
them gives eight terminal classes. Models are compared under two prompts — one that asks
explicitly for citations, one that never mentions them — and the change between the two is
decomposed into **Selection**, **Quality** and **Volume**.

## Quick start

```bash
pip install -r islamiceval/requirements.txt
cd islamiceval/cost_of_asking_paper

python cost_of_asking_3class.py --axis both   # decomposition figures + per-model tables
python build_8class_table.py                  # the eight-class tables
python barbell_correct_rate.py                # per-model before/after chart
```

Everything lands in `cost_of_asking_paper/outputs/`.

## Data

`cost_of_asking_paper/data_release/` holds exactly the annotated data behind the tables
and figures, one JSON object per line:

| file | responses | citations | models |
|---|---:|---:|---:|
| `explicit_12models.jsonl` | 4,537 | 16,529 | 12 |
| `implicit_12models.jsonl` | 5,845 | 2,827 | 12 |
| `implicit_8models.jsonl` | 3,814 | 2,514 | 8 |

All three cover the same 500 questions. Each record carries `question`,
`generated_answer`, and `annotations[]` whose `segments[]` give character offsets into the
answer plus the fidelity and attribution labels. `manifest.json` has per-model counts and a
SHA-256 per file; `data_release/README.md` documents the record shape.

**Scope.** The explicit condition is restricted to the public train+dev splits, because
that condition is the released annotated data and has a held-out test split. The implicit
condition is complete: it was never released and has no split, so nothing in it is held
back. `build_data_release.py` rebuilds these three files and verifies they reproduce the
published tables cell for cell.

## Layout

```
islamiceval/
  cost_of_asking_paper/   paper code, data release, generated figures and tables
  scripts/                generation, cleaning, annotation and scoring pipeline
  data/                   corpora and the IslamicEval 2026 release
  outputs/                model answers and intermediate artefacts
  tools/  jobs/  docs/    annotation tooling, cluster jobs, notes
analysis/
  attribution_errors/     qualitative analysis of attribution errors
paper/                    the submission PDF
```
