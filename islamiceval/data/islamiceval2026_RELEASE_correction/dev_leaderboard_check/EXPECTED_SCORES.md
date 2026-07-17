# Dev-set leaderboard sanity check — expected Task-3 scores

Gold: `dev_corrected_correction.tsv` — 943 rows (476 Ayah, 467 matn).

Scored locally with the OFFICIAL `task3_scoring.py`. CodaBench runs the same scorer,
so the leaderboard should show these exact numbers (assuming the server dev gold == this file).


## dev_pred_perfect.tsv
```json
{
  "accuracy": 1.0,
  "accuracy_Ayah": 1.0,
  "accuracy_matn": 1.0
}
```

## dev_pred_allwrong.tsv
```json
{
  "accuracy": 0.0,
  "accuracy_Ayah": 0.0,
  "accuracy_matn": 0.0
}
```

## dev_pred_half.tsv
```json
{
  "accuracy": 0.5005302226935313,
  "accuracy_Ayah": 0.4957983193277311,
  "accuracy_matn": 0.5053533190578159
}
```

## Independent cross-check (half)
- correct rows: 472/943 = 0.500530
- Ayah correct: 236/476 = 0.495798
- matn correct: 236/467 = 0.505353