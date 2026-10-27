# new_paper — the annotated corpora in fine-grained form

Each corpus that was annotated for **both** Stage 1 (character spans) and Stage 2
(correctness labels), in the released response-level shape but carrying the
**three-class** text label instead of the binary one.

Built by `annotators/final_dataset_build/build_new_paper.py` from
`annotators/final_dataset_build/originals_readonly/` (read-only). Nothing is
re-run; this is a join of existing Stage-1 and Stage-2 output.

## Files

| File | Responses | Citations | Models | Source pair |
|---|---:|---:|---:|---|
| `fanar_main.jsonl` | 4,118 | 12,782 | 10 | `shared_task_clean` + `MERGED_final.citations` |
| `fanar_frontier.jsonl` | 831 | 5,002 | 2 | `frontier_no_rejected` + `frontier_MERGED.citations` |
| `ansari_1k.jsonl` | 1,000 | 2,909 | 10 | `ansari_stage1` + `ansari_stage2` |
| `ansari_2k.jsonl` | 2,000 | 6,844 | 12 | `subtasks2k_stage1` + `subtasks2k_stage2` |
| `a2a_12models.jsonl` | 5,845 | 2,831 | 12 | `a2a_stage1` + `a2a_stage2` |
| `a2a_8models.jsonl` | 3,814 | 2,515 | 8 | `a2a8_stage1` + `a2a8_stage2` |
| **Total** | **17,608** | **32,883** | | 68,595 segments |

`task1new` is **not** here: it has Stage-1 spans but was never passed through Stage 2.

## Record shape

One line per response. `question_id` and `source_model` are explicit, and `id` is
`{question_id}__{source_model}`, so any two files join directly.

```jsonc
{
  "id": "Q04538__acegpt-8b",
  "cohort": "fanar_main",
  "question_id": "Q04538",
  "source_model": "acegpt-8b",      // normalised
  "source_model_raw": "acegpt-8b",  // as the source file recorded it
  "question": "…",
  "generated_answer": "…",          // all offsets index into this
  "annotations": [
    {
      "annotation_id": 1,
      "label": "Hadith",            // "Ayah" | "Hadith"
      "segments": [
        {"type": "matn", "span_start": 337, "span_end": 413,
         "label": "correct", "label_binary": "correct"},
        {"type": "claimed_source", "span_start": 416, "span_end": 428,
         "label": "correct"}
      ],
      "hadith_grade": "sahih",      // recorded by the judge, never used for the label
      "confidence": "High",
      "reason": "The matn is the exact wording of …"
    }
  ]
}
```

* **Segment types** — `Ayah` (Quranic citations), `matn` / `isnad` (Hadith),
  `claimed_source` (both). A segment is present only where the citation supplies it.
* **Text segments** (`Ayah`, `matn`) carry the fine-grained label
  `correct` / `corrupted` / `fabricated`, plus `label_binary` collapsing
  corrupted+fabricated to `incorrect` — so the file is directly comparable to the
  binary release without recomputing anything.
* **Attribution segments** (`isnad`, `claimed_source`) are binary:
  `correct` / `incorrect` / `none` / `N/A`. `N/A` is the deterministic gate — when
  the text is `fabricated` there is nothing to attribute.
* **Offsets** are 0-based, start-inclusive, end-exclusive, and all verified in range
  (0 out-of-range across 68,595 segments). `generated_answer[span_start:span_end]`
  reproduces the cited text exactly.
* **Responses that cite nothing are kept**, with `annotations: []`.

## Judge failures removed

Every text segment carries one of `correct` / `corrupted` / `fabricated`; there are
no placeholder labels in these files. Seven citations whose Stage-2 label was
unusable were dropped at build time:

| Corpus | Dropped |
|---|---|
| `fanar_main` | 1 citation (`MAX_TOKENS`) |
| `a2a_12models` | 1 citation (`PARSE_ERROR`) |
| `a2a_8models` | 2 citations (`EMPTY`), 1 citation (`PARSE_ERROR`) |
| `ansari_1k`, `ansari_2k` | 1 citation each, no Stage-2 row |

In two `a2a_8models` cases that citation was the response's only one, so the whole
response was dropped as well (3,816 → 3,814) rather than leaving a record that
reads as a citation-free answer. Every other response is unchanged.

## Model slugs

`source_model` is normalised and `source_model_raw` preserves what the source file
recorded. One model had two spellings depending on the cohort that produced the row:

| raw | normalised | rows |
|---|---|---:|
| `gemini-3.1-pro` | `gemini-pro-3.1` | 500 (`a2a_12models`) + 191 (`ansari_2k`) |

That matches the form used throughout the thesis, so `gemini-pro-3.1` now pools to
1,086 responses across three corpora instead of splitting across two names. `gpt-5.4`
was already consistent. 20 distinct slugs remain, down from 21.

