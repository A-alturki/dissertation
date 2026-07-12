These numbers came from three inputs feeding one script:

The script: unified_eval.py (reads norm.py, parse2.py, and keep100.json as helpers).

The data files it read:

annotators_corrected_enriched_judged.json — the annotated originals. This produces the GPT (orig) and Gemini (orig) rows and builds the two-tier gold (majority gold + candidate set) that every model is scored against.
predictions_pro.jsonl — the Pro (new) row.
predictions_flash.jsonl — the Flash (new) row.
The mapping lives in the MODELS registry inside unified_eval.py (or in models_config_template.json if you run with --config):

json
{"models": [
  {"name": "GPT (orig)",    "kind": "annotated",  "slot": "first"},
  {"name": "Gemini (orig)", "kind": "annotated",  "slot": "second"},
  {"name": "Pro (new)",     "kind": "prediction", "path": ".../predictions_pro.jsonl"},
  {"name": "Flash (new)",   "kind": "prediction", "path": ".../predictions_flash.jsonl"}
]}
To add your new run: upload its .jsonl prediction file and I'll add one line to the config pointing at it, then re-run — it'll appear as an extra row in the same table, scored identically against the same gold. The only requirement is that it's the same format as predictions_pro.jsonl (a request with the span in <<< >>>, and a response whose model output is JSON with a candidates array), and covers the same 100 spans so the join works.

Go ahead and upload the new prediction file. One question so I label it right: what is this new run — a different prompt on the same model, a different model, or a snapped-vs-unsnapped variant? That determines what the comparison is actually isolating, and I'll name the row accordingly.


