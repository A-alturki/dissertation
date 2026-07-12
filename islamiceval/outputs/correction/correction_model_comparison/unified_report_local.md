# Unified Correction-Subtask Evaluation

All models scored on the same 100 spans (50 Quran + 50 Hadith). Two-tier gold: **majority gold** is the answer 2+ annotators agree on — either **خطأ** (2+ said no correction) or the set of correction spans 2+ annotators share; the **gold candidate set** (Tier 2) is every annotator-sanctioned correction span, used for coverage/recall. `verdict_maj` is the true human majority-correct rate (annotated models only); `majority-gold` = matches the majority answer (correct decline on a خطأ span, or a majority correction span); `decision-acc` = correctable-vs-خطأ decision only; `approx-correct` mirrors majority-gold and is comparable across ALL models.


## All 

| Model | verdict-maj | majority-gold | approx-correct | decision-acc | ≥1-cand | recall | precision | false-corr |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | 0.680 | 0.844 | 0.844 | 0.875 | 0.910 | 0.666 | 0.968 | 1/37 |
| Gemini (orig) | 0.680 | 0.703 | 0.703 | 0.750 | 0.910 | 0.633 | 0.968 | 0/37 |
| Pro (new) | — | 0.797 | 0.797 | 0.906 | 0.870 | 0.614 | 0.818 | 4/37 |
| Flash (new) | — | 0.703 | 0.703 | 0.812 | 0.820 | 0.558 | 0.717 | 8/37 |
| GPT-v3 (new) | — | 0.453 | 0.453 | 0.609 | 0.630 | 0.412 | 0.522 | 24/37 |

## Quran 

| Model | verdict-maj | majority-gold | approx-correct | decision-acc | ≥1-cand | recall | precision | false-corr |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | 0.800 | 0.857 | 0.857 | 0.857 | 0.980 | 0.763 | 1.000 | 1/14 |
| Gemini (orig) | 0.800 | 0.667 | 0.667 | 0.667 | 0.960 | 0.683 | 1.000 | 0/14 |
| Pro (new) | — | 0.762 | 0.762 | 0.809 | 0.900 | 0.701 | 0.858 | 2/14 |
| Flash (new) | — | 0.667 | 0.667 | 0.714 | 0.820 | 0.605 | 0.711 | 3/14 |
| GPT-v3 (new) | — | 0.333 | 0.333 | 0.381 | 0.680 | 0.496 | 0.548 | 12/14 |

## Hadith 

| Model | verdict-maj | majority-gold | approx-correct | decision-acc | ≥1-cand | recall | precision | false-corr |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT (orig) | 0.560 | 0.837 | 0.837 | 0.884 | 0.840 | 0.569 | 0.933 | 0/23 |
| Gemini (orig) | 0.560 | 0.721 | 0.721 | 0.791 | 0.860 | 0.583 | 0.935 | 0/23 |
| Pro (new) | — | 0.814 | 0.814 | 0.954 | 0.840 | 0.527 | 0.780 | 2/23 |
| Flash (new) | — | 0.721 | 0.721 | 0.861 | 0.820 | 0.512 | 0.723 | 5/23 |
| GPT-v3 (new) | — | 0.512 | 0.512 | 0.721 | 0.580 | 0.329 | 0.497 | 12/23 |

## Coverage tiers (gold-bearing spans, All scope)

| Model | covered all | partial | none |
|-------|:-:|:-:|:-:|
| GPT (orig) | 49 | 42 | 9 |
| Gemini (orig) | 45 | 46 | 9 |
| Pro (new) | 45 | 42 | 13 |
| Flash (new) | 39 | 43 | 18 |
| GPT-v3 (new) | 26 | 37 | 37 |

## Model accuracy on unanimous annotator decisions

| Model | all-3 correctable → decision matches (n) | all-3 full solution → matches (n) |
|-------|:-:|:-:|
| GPT (orig) | 0.918 (73) | 1.000 (45) |
| Gemini (orig) | 0.836 (73) | 0.978 (45) |
| Pro (new) | 0.973 (73) | 0.956 (45) |
| Flash (new) | 0.931 (73) | 0.844 (45) |
| GPT-v3 (new) | 0.781 (73) | 0.622 (45) |

*verdict-maj is blank for prediction-file models (no human verdicts on their outputs); approx-correct is the comparable substitute. Reproduce: `python3 unified_eval.py` (optionally `--config models.json`).*
