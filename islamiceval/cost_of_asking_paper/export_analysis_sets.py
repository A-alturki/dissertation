"""
export_analysis_sets.py -- materialise the exact response sets the tables are
built from, so the filtering is auditable rather than implicit in the code.

Writes to new_paper/analysis_sets/:
    explicit_train_dev.jsonl   fanar_main + fanar_frontier, test responses removed
    implicit_all.jsonl         a2a_12models + a2a_8models, complete (no split exists)
    manifest.json              rules, per-model counts, SHA-256 of each file

The filters are imported from build_8class_table.py rather than reimplemented,
so the exported sets cannot drift from the tables. After writing, the script
re-tallies the eight classes straight from the written files and asserts they
equal the published CSVs cell for cell; it exits non-zero if anything differs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import build_8class_table as B

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis_sets"
KEEP = ("train", "dev")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep_pairs = B.public_pairs(KEEP)              # explicit responses in train+dev

    manifest = {
        "scope": "explicit: public train+dev; implicit: complete",
        "rules": {
            "explicit": "keep a response iff its (question_id, generated_answer) "
                        "appears in the release train.jsonl or dev.jsonl",
            "implicit": "no filter -- the complete condition is used. It was never "
                        "released and has no held-out split: the test split holds out "
                        "annotated EXPLICIT responses, and the implicit answer to the "
                        "same question is a different response from a different prompt "
                        "with its own annotations. The 500 questions are public anyway.",
        },
        "files": {},
    }

    for cond, out_name in (("explicit", "explicit_train_dev.jsonl"),
                           ("implicit", "implicit_all.jsonl")):
        path = OUT / out_name
        resp = cites = 0
        per_model = {}
        with open(path, "w", encoding="utf-8") as w:
            for fn in B.CONDITION_FILES[cond]:
                with open(HERE / fn, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        model = B.normalize_model(row["source_model"])
                        pair = (model, row["question_id"])
                        if cond == "explicit" and pair not in keep_pairs:
                            continue
                        n = sum(1 for a in row["annotations"]
                                if B.classify(a) is not None)
                        resp += 1
                        cites += n
                        d = per_model.setdefault(model, {"responses": 0, "citations": 0})
                        d["responses"] += 1
                        d["citations"] += n
                        w.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"][out_name] = {
            "condition": cond,
            "source_files": B.CONDITION_FILES[cond],
            "responses": resp,
            "classifiable_citations": cites,
            "models": len(per_model),
            "per_model": dict(sorted(per_model.items())),
            "sha256": sha256(path),
        }
        print(f"{out_name}: {resp:,} responses, {cites:,} citations, {len(per_model)} models")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT / 'manifest.json'}")

    # ---- self-check: rebuild the eight-class counts from the WRITTEN files ----
    print("\nSelf-check: re-tallying from the exported files...")
    ok = True
    for cond, out_name in (("explicit", "explicit_train_dev.jsonl"),
                           ("implicit", "implicit_all.jsonl")):
        counts, _ = B.tally([OUT / out_name])          # no filters: already applied
        published = {r["model"]: r for r in csv.DictReader(
            open(HERE / "outputs" / f"table_8class_{cond}.csv", encoding="utf-8"))}
        for model, leaves in sorted(counts.items()):
            row = published.get(model)
            if row is None:
                print(f"  MISSING from published CSV: {model}")
                ok = False
                continue
            for k in B.LEAF_KEYS:
                if int(row[f"n_{k}"]) != leaves[k]:
                    print(f"  MISMATCH {cond} {model} {k}: "
                          f"exported {leaves[k]} vs published {row[f'n_{k}']}")
                    ok = False
        print(f"  {cond}: {len(counts)} models checked")
    print("\nSELF-CHECK:", "PASS -- exported sets reproduce the tables exactly"
          if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
