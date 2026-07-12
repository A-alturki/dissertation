# Correction Subtask — Statistics, Definitions, and Computation

Reference for the methods & results chapters. Every metric below is defined operationally (how it
was computed) alongside what it measures and how to interpret it.

## Dataset and unit of analysis
- 100 spans, balanced 50 Quran + 50 Hadith, each a reference span an LLM presented as scripture.
- Unit of analysis = the **span**. Predictions from any model are joined to the dataset by
  NFC-normalized span text (tashkeel stripped, alef/yaa/taa-marbuta unified).
- Three annotators judged each original correction correct/wrong; on "wrong" they supplied a manual
  correction. Two original correctors: GPT and Gemini.

## Normalization (applied at every comparison boundary)
NFC → strip tashkeel/tatweel → normalize alef (أإآا→ا), yaa (ى→ي), taa-marbuta (ة→ه) → strip
punctuation and bracketed refs → collapse whitespace. This makes textual comparison robust to
orthographic variation that is not semantically meaningful.

## Candidate matching (what counts as "the same correction")
- **Quran**: two candidates match if their normalized ayah text is equal, OR they cite the same
  surah with ayah number within ±1 (tolerates "right verse, wrong number" — a real annotator error).
- **Hadith**: matched on the MATN via content-token Jaccard ≥ 0.6, after removing isnad/honorific
  boilerplate; the reference NUMBER is ignored (same matn appears under different numbers across the
  six books). Interpretation caveat: Jaccard penalizes the same hadith quoted at different lengths,
  so all hadith content-agreement numbers are CONSERVATIVE LOWER BOUNDS.

## Two-tier gold (the reference answers)
Because "no correction" (خطأ) is itself a valid answer, the gold is defined at two levels:
- **Tier 1 — majority gold** (one answer per span): خطأ if ≥2 annotators judged the span needs no
  correction; otherwise the set of correction spans that ≥2 annotators agree on.
- **Tier 2 — gold candidate set** (the pool of all acceptable answers): every annotator-sanctioned
  correction (manual corrections + model corrections an annotator marked correct) PLUS خطأ as a
  member whenever any annotator endorsed a decline.
- Endorsement principle: when annotators mark a model correct they write no manual correction, so
  the model's own output (or خطأ) becomes the agreed answer. This is credited as a match.

## Metric definitions (how each number is computed)

### Annotator agreement (pairwise, averaged over the 3 annotator pairs per span)
- **Correctable agreement** — fraction of annotator pairs agreeing on whether the span needs a fix.
  Value: 0.826 all / 0.840 Quran / 0.811 Hadith. Cohen's κ (chance-corrected): 0.600 / 0.645 /
  0.632 (substantial). Interpretation: annotators agree strongly on WHETHER to correct.
- **Full-solution agreement** — fraction of pairs whose correction sets are identical (0.550 /
  0.667 / 0.433). Much lower — they agree far less on the EXACT correction.
- **≥1-shared-candidate** — fraction of pairs sharing at least one correction (0.663 / 0.700 /
  0.627). Sits between the two above.
Note: Cohen's κ is reported ONLY for the correctable decision. It is ill-defined for solution-set
overlap (a pairwise set relation, not an independent per-rater label); the correct chance-corrected
tool there is Krippendorff's α with MASI distance.

### Unanimous consensus (per-span, all 3 agree)
- Correctable: 73/100 spans have full 3-way agreement on whether a fix is needed.
- Full solution: 45/100 spans have full 3-way agreement on the exact correction.
Distinction from pairwise: pairwise asks "do two random annotators agree" (averaged); unanimous asks
"do all three agree" (harder, hence lower). Unanimous numbers define the clean subset used to
evaluate models.

### Span partition by majority gold
37 خطأ-majority (≥2 said no correction) / 27 correction-majority (≥2 agree on a span) / 36
no-majority-answer (annotators too split for any majority). The 36 are excluded from Tier-1 metrics
(no majority to score against) but remain in Tier-2 coverage — state this explicitly; it is ~1/3 of
the data.

### Model accuracy (each computed against the gold above)
- **verdict-maj** — fraction of a model's corrections the annotator MAJORITY judged correct. Needs
  human verdicts, so exists ONLY for the annotated originals (GPT/Gemini). GPT 0.68, Gemini 0.68.
- **majority-gold** — fraction where the model's answer matches the Tier-1 majority answer (a correct
  decline on a خطأ-majority span, or a majority correction span). Comparable across ALL models.
- **decision-acc** — fraction where the model's propose-vs-decline decision matches the majority
  correctable/خطأ decision (ignores which specific correction).
- **≥1-candidate** — fraction of gold-bearing spans where the model matches ≥1 gold candidate
  (including matching خطأ by declining).
- **recall** — over gold-bearing spans, the mean fraction of the span's gold candidates the model
  covers. Rewards returning ALL valid corrections (matching 3 of 3 beats 1 of 3). خطأ counts as one
  candidate.
- **precision** — of the model's own outputs, the fraction that hit a gold candidate (a correct
  decline scores precision 1). Penalizes spurious/padded candidates.
- **false-corrections** — count of خطأ-majority spans where the model wrongly proposed a correction.
- **approx-correct** — verdict-free stand-in for verdict-maj (matches majority gold); used to compare
  new models that have no human verdicts.

### Final model numbers (against two-tier gold, snapped outputs)
GPT     — majority-gold 0.844, decision 0.875, ≥1-cand 0.91, recall 0.666, precision 0.968.
Gemini  — majority-gold 0.703, decision 0.750, ≥1-cand 0.91, recall 0.633, precision 0.968.
(Quran/Hadith splits and the new-prompt runs Pro/Flash are in unified_results.json.)
Reading: models agree with humans strongly on the correctable decision and precision, less on full
recall; Hadith is uniformly lower than Quran (variant multiplicity + numbering), and its content
figures are lower bounds by construction.

## Snapping + expansion (post-processing applied to raw model output)
- Snapping = replace each candidate with its closest corpus match; discard if none clears threshold
  (→ خطأ if no candidate survives). Effect: normalizes near-canonical text so it matches gold, and
  prunes junk. Raises PRECISION (GPT all 0.892→0.968; Quran→1.000).
- Expansion (Hadith only) = after snapping, add every corpus variant with ≥90% overlap or where the
  snapped matn is a substring. Effect: supplies attested variants the model did not emit. Raises
  RECALL (GPT Hadith candidates 29→41). Quran unaffected (closed text).
- Net: strictly beneficial here — both recall and precision rise, no metric regresses. Caveat: gold
  was built from snapped data, so part of the precision gain is representational alignment between
  snapped candidates and canonical gold, not pure model improvement.

## Reproduction
`python3 unified_eval.py --config models_config_template.json` → unified_results.json + report.
Add any model as one config line (annotated slot, or a prediction .jsonl path). Per-span extracted
candidates + gold (with khata flag) are dumped to extracted_candidates.json for auditing.
