# Data release

Exactly the annotated data behind the paper's tables and figures.

| file | responses | citations | models |
|---|---:|---:|---|
| `explicit_12models.jsonl` | 4,537 | 16,529 | 12 |
| `implicit_12models.jsonl` | 5,845 | 2,827 | 12 |
| `implicit_8models.jsonl` | 3,814 | 2,514 | 8 |

All three cover the same 500 questions. Record shape is one JSON object per
line, identical to the source corpora: `id`, `cohort`, `question_id`,
`source_model`, `source_model_raw`, `question`, `generated_answer`, and
`annotations[]` with `label` (Ayah/Hadith), `segments[]` (`type`, `span_start`,
`span_end`, `label`, `label_binary`), `hadith_grade`, `confidence` and `reason`.
Character offsets index into `generated_answer`.

## Scope

The explicit file is restricted to the public train+dev splits: that condition
is the released annotated data and has a held-out test split. The implicit
files are complete -- that condition was never released and has no split, so
nothing in it is held out.

`manifest.json` carries per-model counts and a SHA-256 for each file.
