# IslamicEval 2026 — Dataset Codebook

## Response-level fields (each line in `train.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique response identifier (e.g. `R000123`).|
| `question_id` | string | Identifier of the source question. Multiple responses may share a `question_id` (different models answered the same question). |
| `question` | string | The Arabic question posed to the model. |
| `generated_answer` | string | The model's full Arabic response. All span offsets index into this string. |
| `annotations` | list | One entry per citation in the answer. No citations is a single `no_annotation` entry. |

## Annotation-level fields (each item in `annotations`)

| Field | Type | Description |
|---|---|---|
| `annotation_id` | integer | Citation index within the response (1-based). |
| `label` | string | Citation type: `Ayah` (Qur'an), `Hadith`, or `no_annotation` (response cites nothing). |
| `segments` | list | The parts of the citation, each with a span and a label. Empty for `no_annotation`. |
| `correction` | list \| string | Present only when the citation's text segment (Ayah/matn) is `incorrect`. Either a list of one or more acceptable canonical corrections — the full verse for Qur'an, the matn for Hadith — or the string `"خطأ"` when the span cannot be grounded to any authentic source. Absent/`null` for correct citations. |

## Segment-level fields (each item in `segments`)

| Field | Type | Description |
|---|---|---|
| `type` | string | One of `Ayah`, `matn`, `isnad`, `claimed_source`. |
| `span_start` | integer | Character offset into `generated_answer` where the segment begins (inclusive). |
| `span_end` | integer | Character offset where the segment ends (exclusive). `generated_answer[span_start:span_end]` recovers the text. |
| `label` | string | `correct`, `incorrect`, or `N/A`. |

## Segment types

| Type | Appears in | Description |
|---|---|---|
| `Ayah` | Qur'an citations | The quoted Qur'anic verse text. |
| `matn` | Hadith citations | The text (body) of the Hadith. |
| `isnad` | Hadith citations | The chain of narration, when the response provides one. |
| `claimed_source` | both | The attribution stated in the response (a surah:ayah reference, or a Hadith collection). |

## Label semantics

| Label | Meaning |
|---|---|
| `correct` | The segment faithfully matches the authentic source. |
| `incorrect` | The text (Ayah/matn) is incorrect; or the isnad/claimed_source is wrong (only judged when the text is correct). |
| `N/A` | The isnad or claimed_source of a citation whose text is incorrect. An incorrect citation has no verifiable chain or source, so it is not judged. These segments are excluded from labelling evaluation. |

## Gating rule (how labels interact)

| Text segment (Ayah/matn) | isnad | claimed_source |
|---|---|---|
| `correct` | judged (`correct`/`incorrect`) | judged (`correct`/`incorrect`) |
| `incorrect` | `N/A` | `N/A` |

## Correction values (correction subtask)

The `correction` slot holds the authentic form of an incorrect citation. It supports subtask 3, where a system proposes the corrected text for a mis-cited verse or Hadith.

| Case | `correction` value |
|---|---|
| correctable | A list of one or more accepted corrections. Qur'an → the full verse(s); Hadith → the matn. |
| Not correctable | The string `"خطأ"` — a fabrication or unrecoverable garble with no authentic source. |

- **Multiple accepted corrections.** A single incorrect citation often has more than one valid correction: a Hadith matn attested across collections with slightly different wording, or a Qur'anic quotation matching more than one verse (or the same verse quoted at more than one length). The gold records the full set; a prediction counts correct if it matches **any one** of them.
- **A system submits exactly one correction per segment**, not the full set.
- In the gold TSV, the accepted corrections for a segment are joined by ` ||| ` in a single `Correction` cell.

## Task outputs (TSV columns)

| Subtask | Columns |
|---|---|
| Task 1 | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Span_Start`, `Span_End` |
| Task 2 | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Label` |
| Task 3 | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Correction` |

No-citation responses appear in Subtask 1 as a single row `Response_ID  1  NoAnnotation  -  -`, and contribute no rows to Subtask 2 or Subtask 3. Subtask 3 has one row only per **incorrect** `Ayah`/`matn` segment; `isnad` and `claimed_source` are never corrected.
