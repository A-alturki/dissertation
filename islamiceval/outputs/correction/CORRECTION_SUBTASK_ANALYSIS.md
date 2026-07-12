# IslamicEval 2026 — Correction Sub-task: Complete Analysis

*Recomputed 2026-07-10 against the current spans (`keep100`), the curator-verified gold (`gold_100_final.json`), and the locked grounding configuration. Supersedes all earlier ad-hoc statistics. Reproducible via `correction_model_comparison/analyze_correction_report.py` and `score_manual_gold.py` (+ `norm.py`, `parse2.py`, `keep100.json`) → `report_stats.json`, `manual_gold_results.json`; expansion study via `scripts/score_expansion_labels.py`.*

> Tip: import this file into Notion via **Import → Markdown & CSV** for clean table rendering.

## 0. Scope and set-up

**Evaluation set.** 100 annotated items (50 Quran + 50 Hadith), each a reference span an LLM presented as scripture but that was labelled **incorrect**. Every item is judged by **three annotators** (الأمير أبو عوف, Amir, هشام) for **two correctors** — **GPT-5.4** (first) and **Gemini-3.1-pro** (second) — each of which either proposes a correction or **declines** (`خطأ — لا تصحيح مقترح`). Annotators judge each corrector correct/wrong and, when wrong, supply their own correction.

**Gold.** The gold is the **curator-verified** set built in the manual gold tool: per span, the full set of accepted corrections, **each with a backer count** (how many of the 3 annotators support it), with **خطأ (baseless)** a first-class candidate. *Majority* = a candidate backed by ≥2 annotators. This replaced an earlier annotator-verdict-derived gold that had a per-corrector verdict-attribution bug. Of the 100 spans: 52 have a majority correction, 37 a majority خطأ, 50 offer خطأ as an option.

**Candidate matching (used everywhere).** Two candidate *units* (scripture text + its citation) match if their normalised text is identical, **or** boilerplate-stripped content-token Jaccard ≥ 0.6, **or** (Quran only) same surah with ayah number within ±1. Hadith is matched on the **matn**, not the reference number. Normalisation at every boundary: NFC, strip tashkeel/tatweel, unify alef/yaa/taa-marbuta, drop punctuation, collapse whitespace.

> **Conservatism note.** The Hadith Jaccard-≥0.6 rule penalises the same hadith quoted at very different lengths, so **all Hadith content-agreement and recall figures are lower bounds.**

## 1. Annotator agreement

Three measures at three granularities. **all-3-agree** = fraction of spans all three annotators give the same label; **pairwise** = mean raw agreement over the 3 annotator pairs; **Fleiss κ** for the binary "correctable" label.

| Measure | All | Quran | Hadith |
|---|:-:|:-:|:-:|
| Correctable — all-3 agree | 0.788 (78/99) | 0.800 (40/50) | 0.776 (38/49) |
| Correctable — pairwise | 0.859 | 0.867 | 0.851 |
| Correctable — Fleiss κ | 0.688 | 0.641 | 0.701 |
| Full solution — all-3 agree | 0.455 | 0.540 | 0.367 |
| Full solution — pairwise | 0.564 | 0.653 | 0.473 |
| ≥1 shared candidate — pairwise | 0.708 | 0.780 | 0.635 |

**Reading.** Annotators agree **strongly on *whether* a span needs fixing** (pairwise ~0.86, Fleiss ~0.69 = substantial), much **less on the *exact* correction set** (full-solution pairwise ~0.56), with **Quran ahead of Hadith on content throughout** — Quran corrections converge on one canonical verse, whereas Hadith admits many attested matn variants (and the Jaccard rule under-counts them).

## 2. Inter-annotator and annotator–gold reliability (Cohen's κ)

| Comparison | الأمير أبو عوف | Amir | هشام |
|---|:-:|:-:|:-:|
| Annotator vs majority gold (correctable) | 0.824 | 0.840 | 0.870 |

| Annotator pair (correctable) | κ |
|---|:-:|
| الأمير ↔ Amir | 0.657 |
| الأمير ↔ هشام | 0.696 |
| Amir ↔ هشام | 0.708 |

Every annotator aligns with the majority gold at κ ≥ 0.82 (near-perfect); pairwise annotator agreement is uniformly substantial. There is **no unreliable annotator.**

> κ on solution-*content* is not reported as Cohen's κ — candidate overlap is a pairwise set relation, not an independent per-rater label, so Cohen's chance model is undefined. A set-aware coefficient such as Krippendorff's α with MASI distance is the correct tool there and can be added if needed.

## 3. Where all three annotators agree (unanimous subsets)

| Scope | n | unanimous correctable | unanimous full solution |
|---|:-:|:-:|:-:|
| All | 100 | 78 (0.788) | 45 (0.455) |
| Quran | 50 | 40 (0.800) | 27 (0.540) |
| Hadith | 50 | 38 (0.776) | 18 (0.367) |

Unanimity on *correctable* is common (~78%); unanimity on the *exact solution* is rarer (~46%) and lowest for Hadith — the multi-variant nature of matn again. Every full-solution agreement is also a correctable agreement (a logical necessity).

## 4. Corrector model comparison vs the verified gold (headline result)

Five correctors scored on the identical 100 spans against the curator gold. **GPT-5.4** and **Gemini-3.1-pro** are the deployed one-shot correctors; **Pro** (gemini-3.1-pro) and **Flash** (gemini-3.5-flash) are a revised-prompt run; **GPT-v3** is GPT-5.4 under a "recovery-oriented" prompt. GPT/Gemini are scored on their human-shown snapped text; Pro/Flash/GPT-v3 on raw candidates — see §6 for the uniform before/after that removes this asymmetry.

**Metric definitions.** `majority-gold` = matches a ≥2-backed answer (a majority correction, or خطأ if خطأ is majority). `decision` = correctable-vs-خطأ call only. `≥1-cand` = matched ≥1 accepted answer (any count). `recall` = share of *all* accepted variants covered. `precision` = fraction of the model's own output that is accepted (a correct decline scores 1). `false-corr x/y` = proposed a fix on *y* spans whose majority is خطأ-only.

**All (100 spans)**

| Model | majority-gold | decision | ≥1-cand | recall | precision | false-corr |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | **0.899** | 0.921 | 0.910 | 0.696 | **0.959** | **1/37** |
| Gemini (orig) | 0.798 | 0.832 | 0.850 | 0.638 | 0.960 | 0/37 |
| Pro (new) | 0.865 | **0.933** | 0.870 | 0.645 | 0.821 | 4/37 |
| Flash (new) | 0.753 | 0.865 | 0.810 | 0.588 | 0.718 | 8/37 |
| GPT-v3 | **0.573** | 0.719 | 0.630 | 0.445 | **0.524** | **24/37** |

**Quran (50 spans)**

| Model | majority-gold | decision | ≥1-cand | recall | precision | false-corr |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | 0.936 | 0.936 | 0.980 | 0.806 | 0.990 | 1/14 |
| Gemini (orig) | 0.851 | 0.851 | 0.900 | 0.709 | 1.000 | 0/14 |
| Pro (new) | 0.894 | 0.915 | 0.920 | 0.761 | 0.875 | 2/14 |
| Flash (new) | 0.766 | 0.872 | 0.820 | 0.659 | 0.722 | 3/14 |
| GPT-v3 | 0.660 | 0.723 | 0.700 | 0.561 | 0.558 | 12/14 |

**Hadith (50 spans)**

| Model | majority-gold | decision | ≥1-cand | recall | precision | false-corr |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | 0.857 | 0.905 | 0.840 | 0.586 | 0.926 | 0/23 |
| Gemini (orig) | 0.738 | 0.810 | 0.800 | 0.568 | 0.919 | 0/23 |
| Pro (new) | 0.833 | 0.952 | 0.820 | 0.529 | 0.770 | 2/23 |
| Flash (new) | 0.738 | 0.857 | 0.800 | 0.518 | 0.713 | 5/23 |
| GPT-v3 | 0.476 | 0.714 | 0.560 | 0.329 | 0.490 | 12/23 |

**Coverage tiers (All scope): covered-all / partial / none**

| Model | covered all | partial | none |
|---|:-:|:-:|:-:|
| GPT (orig) | 54 | 37 | 9 |
| Gemini (orig) | 47 | 38 | 15 |
| Pro (new) | 49 | 38 | 13 |
| Flash (new) | 43 | 38 | 19 |
| GPT-v3 | 31 | 32 | 37 |

**Findings.**

- **GPT-5.4 (original one-shot) is the best corrector** — top majority-gold (0.899), top precision (0.959), only **1 of 37** false corrections.
- **The recovery-oriented prompt collapses** — majority-gold 0.573, precision 0.524, and **24 of 37** false corrections: its extra "recoveries" land on FABRICATED/GARBLED spans annotators reject ~2:1. Aggression manifests as **over-correction, not recall.** (This is what selected the original prompt for the RELEASE run.)
- **Quran ≫ Hadith for every model** (GPT 0.936 vs 0.857) — hadith matn-variant retrieval is the hard problem, consistent with the whole benchmark.

## 5. Model accuracy on the unanimous-annotator subset

Restricting to spans where **all three annotators agree**: each model's decision on the 78 unanimous-correctable spans, and full-solution match on the 45 unanimous-solution spans.

| Model | all-3 correctable → decision matches | all-3 full-solution → matches |
|---|:-:|:-:|
| GPT (orig) | **0.923** (72/78) | **1.000** (45/45) |
| Gemini (orig) | 0.782 | 0.978 |
| Pro (new) | 0.974 | 0.956 |
| Flash (new) | 0.936 | 0.911 |
| GPT-v3 | 0.705 | 0.489 |

Where humans are unanimous, GPT/Gemini/Pro/Flash reproduce the agreed solution ~91–100% of the time; the propose/decline **decision** is where models separate (GPT 0.923 vs Gemini 0.782), and GPT-v3 under-performs on both axes (its over-correction shows up as 0.489 solution match).

## 6. Effect of snapping + substring-matching + expansion (before vs after, all models uniformly)

To isolate the grounding post-processing, **every** model's raw candidates are scored against the same gold **before** and **after** the locked pipeline (jaccard-snap anchor → char-snap fallback → substring → Jaccard expansion). This removes the §4 asymmetry (there GPT/Gemini used their pre-snapped human-shown text; here all five are snapped uniformly from raw).

**All scope — raw → snapped+expanded**

| Model | majority-gold | recall | precision | ≥1-cand | # candidates |
|---|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | 0.843 → **0.865** | 0.642 → **0.672** | 0.888 → **0.934** | 0.86 → 0.88 | 69 → 66 |
| Gemini (orig) | 0.798 → 0.798 | 0.625 → 0.628 | 0.943 → **0.949** | 0.84 → 0.85 | 49 → 47 |
| Pro (new) | 0.865 → 0.865 | 0.645 → **0.658** | 0.821 → **0.859** | 0.87 → 0.87 | 90 → 86 |
| Flash (new) | 0.753 → **0.787** | 0.588 → **0.637** | 0.718 → **0.784** | 0.81 → **0.86** | 110 → 104 |
| GPT-v3 | 0.427 → **0.494** | 0.338 → **0.378** | 0.445 → **0.526** | 0.49 → **0.55** | 152 → 146 |

**Quran only — raw → snapped+expanded**

| Model | majority-gold | recall | precision |
|---|:-:|:-:|:-:|
| GPT (orig) | 0.872 → 0.915 | 0.739 → 0.786 | 0.908 → 0.956 |
| Gemini (orig) | 0.851 → 0.851 | 0.709 → 0.709 | 1.000 → 1.000 |
| Pro (new) | 0.894 → 0.894 | 0.761 → 0.761 | 0.875 → 0.875 |
| Flash (new) | 0.766 → 0.851 | 0.659 → 0.726 | 0.722 → 0.806 |
| GPT-v3 | 0.617 → 0.660 | 0.521 → 0.564 | 0.539 → 0.589 |

**Hadith only — raw → snapped+expanded**

| Model | majority-gold | recall | precision |
|---|:-:|:-:|:-:|
| GPT (orig) | 0.810 → 0.810 | 0.544 → 0.559 | 0.867 → 0.909 |
| Gemini (orig) | 0.738 → 0.738 | 0.541 → 0.547 | 0.884 → 0.895 |
| Pro (new) | 0.833 → 0.833 | 0.529 → 0.555 | 0.770 → 0.844 |
| Flash (new) | 0.738 → 0.714 | 0.518 → 0.548 | 0.713 → 0.762 |
| GPT-v3 | 0.214 → 0.310 | 0.156 → 0.193 | 0.329 → 0.446 |

**What each operation contributes (holds for all five models):**

- **Snapping raises precision** — the larger, less obvious win. Snapping (a) normalises a model's near-canonical text to the exact corpus form, so candidates just below the match threshold now match, and (b) discards candidates that snap to nothing (spurious near-misses → خطأ). It helps **both** sources: Quran precision rises even though the text is closed (e.g. Flash Quran majority-gold 0.766→0.851, GPT Quran precision 0.908→0.956) because it filters non-verses — it is a **quality filter**, not just expansion.
- **Expansion raises recall** — concentrated in **Hadith** (Quran is untouched: a closed text where models already emit the canonical verse). Adding attested matn variants covers more of the gold's variant pool (e.g. Pro Hadith recall 0.529→0.555, precision 0.770→0.844). This is the designed effect and directly addresses "models don't emit every attested wording."
- **Net:** every model improves; the effect is largest where a model was noisiest (Flash, GPT-v3) — post-processing partly launders weaker raw output. GPT-v3 still ends far behind (0.494), so grounding does **not** rescue its over-correction.

> The §4 headline uses GPT/Gemini's **human-shown** snapped text (GPT majority-gold 0.899), marginally higher than the uniform snap-from-raw here (0.865); the gap is the human curation of the exact snapped candidate. Both tell the same story.

## 7. Hadith-expansion similarity-metric study

**Problem.** The legacy expander used a character-sequence ratio, which both **over-expands** generic-frame matn ("the Prophet forbade…", "a man came to the Prophet…" partially match 100+ unrelated hadiths) and **misses** variants whose clauses are reordered (sequence metrics are order-sensitive; matn variants reorder).

**Blind pairwise study.** Same/different matn pairs were labelled blind. The researcher hand-labelled **40**; Gemini (web) labelled **100** (incl. those 40); the study was then **doubled to 200** pairs (a fresh disagreement-weighted 100, labelled by Gemini, appended to the first 100). **Annotator validation (researcher vs Gemini on the shared 40): 92% agreement, Cohen's κ = 0.825** (almost-perfect; only 3 disagreements, all Gemini marginally lenient — the tuned variant-vs-topic prompt fixed an earlier leniency, κ 0.40 → 0.825). Gemini is thus a trustworthy second annotator.

**2×2 metric ablation (sequence-vs-set × raw-vs-frame-stripped), ROC-AUC** (threshold-free separation):

| Metric family | raw (frame kept) | content (frame-stripped) |
|---|:-:|:-:|
| SEQUENCE (char-ratio) | 0.807 | 0.835 |
| SET (Jaccard) | 0.874 | **0.905** |

*(Earlier 100-label version: SEQUENCE 0.763 / 0.830, SET 0.854 / **0.912** — same ordering.)* Two design choices **stack**: **SET ≫ SEQUENCE** (~+0.07–0.09 AUC — the dominant effect, because matn variants reorder and a set metric is order-blind) and **frame-stripping helps** (~+0.03–0.06). **Winner: Jaccard on frame-stripped content tokens, AUC ≈ 0.905–0.912.** Head-to-head on disagreement pairs: Jaccard beats char-ratio 12/20 (200-set) and 10/15 (100-set).

**Operating threshold (winner's precision/recall on 200 pairs; 139 same / 61 different):**

| Jaccard threshold | precision | recall | wrong-merges (FP) |
|---|:-:|:-:|:-:|
| 0.25 | 0.93 | 0.76 | 8 |
| 0.30 | 0.97 | 0.56 | 2 |
| 0.35 | 0.97 | 0.45 | 2 |
| 0.40 | 0.98 | 0.39 | 1 |
| 0.45 | 1.00 | 0.36 | 0 |
| 0.50 | 1.00 | 0.32 | 0 |

Precision is ≥0.97 for any threshold ≥0.30; raising the threshold costs chiefly **recall**. The AUC is threshold-free, so the metric choice is robust regardless of operating point.

**Where the real levers are (dev-set diagnostics).** The **anchor**, not the expansion threshold, is decisive: several spans fail to ground purely because character-based *snapping* misses a real hadith whose corpus form is longer/reordered — one dev span is a perfect content-set match (Jaccard 1.00) yet char-snap scores 0.86 and drops it; a Jaccard snap recovers ~6 of 11 such spans. Over-expansion, conversely, is **threshold-independent**: the extreme cases are either genuine multi-attestation (a short prohibition hadith really appears 100+ times) or generic story-frames, both clearing any reasonable threshold — so a **cap/skip** on variant count is the right control, not a higher threshold. On dev @ Jaccard 0.35 uncapped: 75/99 grounding spans yield ≤5 variants (28 exactly one), 24 yield >5, 10 yield >10 (median 3).

## 8. RELEASE deployment (grounding configuration, locked 2026-07-08)

Order = **snap → dedup → expand**. **Hadith anchor:** jaccard-snap primary (`SNAP_JAC_TH 0.40`, highest frame-stripped content-Jaccard matn) → char-snap fallback (`SNAP_CHAR_FALLBACK_TH 0.85`) when nothing clears 0.40. **Expand:** `EXPAND_METRIC=jaccard`, `EXPAND_JAC_TH 0.35`; **cap replaced by a per-correction skip** — an anchor that explodes to ≥15 distinct variants (generic matn) keeps only the snapped anchor (`EXPAND_SKIP_GE 15`). Corrections are **snapped-anchor-first**. **Quran** → full ayah(s). A correction that **grounds to nothing is left null** (not asserted baseless). **Dev under the locked config: 327/943 grounded** (jaccard+char-0.85 = strict superset of char-only 318, recovering 9 length-mismatch variant hadiths, losing none). The chosen corrector (GPT-5.4 one-shot) has been run on **train (7,628 raw predictions)**; the shipped deliverables live in `data/islamiceval2026_RELEASE_correction/`. For the **relevance** sub-task (which requires a correct citation) the **highest-probability** canonical verse/hadith is substituted per span.

## 9. Summary of headline numbers (for citation)

- **Annotators** agree strongly on *whether* to correct (Fleiss κ ≈ 0.69; pairwise ≈ 0.86) and moderately on the *exact* correction (full-solution pairwise ≈ 0.56, lower for Hadith); every annotator aligns with the gold at κ ≥ 0.82.
- **Best corrector: GPT-5.4 one-shot** — majority-gold **0.899**, precision **0.959**, 1/37 false corrections; leads Gemini-3.1-pro (0.798) and the revised-prompt runs.
- **Recovery prompt over-corrects** — 0.573 majority-gold, 24/37 false corrections.
- **Quran ≫ Hadith** for all models (~0.94 vs ~0.86 for GPT).
- **Grounding helps every model**: snapping raises precision (both sources; a quality filter), expansion raises Hadith recall.
- **Expansion metric:** Jaccard on frame-stripped content tokens, **ROC-AUC ≈ 0.905–0.912**, operating threshold ~0.35–0.40; validated by a second annotator at **κ = 0.825**.
