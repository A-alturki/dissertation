"""
build_data_release.py -- assemble the shareable release: exactly the data behind
the paper's tables and figures, in three files.

    data_release/explicit_12models.jsonl   12 models, explicit prompt, train+dev
    data_release/implicit_12models.jsonl   the same 12 models, implicit prompt, complete
    data_release/implicit_8models.jsonl    8 newer open-weight models, implicit, complete

Record shape is identical to the new_paper/ files it is built from, so anything
that reads those reads these.

Why the explicit file is filtered and the implicit ones are not: the explicit
condition IS the publicly released annotated data and carries a held-out test
split, so only train+dev is shared. The implicit condition was never released
and has no split -- the test split holds out annotated EXPLICIT responses, and
an implicit answer to the same question is a different response from a different
prompt with its own annotations -- so all of it is shared.

The script re-tallies the eight taxonomy classes from the files it has just
written and asserts they equal the published table CSVs cell for cell. It exits
non-zero if anything differs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import build_8class_table as B

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_release"
KEEP = ("train", "dev")

SPEC = [
    ("explicit_12models.jsonl", ["fanar_main.jsonl", "fanar_frontier.jsonl"], True,
     "12 models, explicit prompt, restricted to the public train+dev splits"),
    ("implicit_12models.jsonl", ["a2a_12models.jsonl"], False,
     "the same 12 models, implicit prompt, complete (no split exists)"),
    ("implicit_8models.jsonl", ["a2a_8models.jsonl"], False,
     "8 newer open-weight models, implicit prompt only, complete"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep_pairs = B.public_pairs(KEEP)
    manifest = {"files": {}, "totals": {}}

    for name, sources, filtered, desc in SPEC:
        path = OUT / name
        resp = cites = 0
        per_model = {}
        with open(path, "w", encoding="utf-8") as w:
            for src in sources:
                with open(HERE / src, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        model = B.normalize_model(row["source_model"])
                        if filtered and (model, row["question_id"]) not in keep_pairs:
                            continue
                        n = sum(1 for a in row["annotations"] if B.classify(a) is not None)
                        resp += 1
                        cites += n
                        d = per_model.setdefault(model, {"responses": 0, "citations": 0})
                        d["responses"] += 1
                        d["citations"] += n
                        w.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"][name] = {
            "description": desc,
            "built_from": sources,
            "filtered_to_train_dev": filtered,
            "responses": resp,
            "classifiable_citations": cites,
            "models": sorted(per_model),
            "per_model": dict(sorted(per_model.items())),
            "sha256": sha256(path),
        }
        print(f"{name:26s} {resp:6,d} responses  {cites:6,d} citations  {len(per_model):2d} models")

    manifest["totals"] = {
        "responses": sum(v["responses"] for v in manifest["files"].values()),
        "classifiable_citations": sum(v["classifiable_citations"] for v in manifest["files"].values()),
        "questions": 500,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- self-check against the published tables ----------------
    print("\nSelf-check against the published table CSVs...")
    ok = True
    groups = {"explicit": ["explicit_12models.jsonl"],
              "implicit": ["implicit_12models.jsonl", "implicit_8models.jsonl"]}
    for cond, files in groups.items():
        counts, _ = B.tally([OUT / f for f in files])
        pub = {r["model"]: r for r in csv.DictReader(
            open(HERE / "outputs" / f"table_8class_{cond}.csv", encoding="utf-8"))}
        for model, leaves in sorted(counts.items()):
            row = pub.get(model)
            if row is None:
                print(f"  {cond}: {model} missing from published CSV")
                ok = False
                continue
            for k in B.LEAF_KEYS:
                if int(row[f"n_{k}"]) != leaves[k]:
                    print(f"  MISMATCH {cond} {model} {k}: "
                          f"release {leaves[k]} vs published {row[f'n_{k}']}")
                    ok = False
        print(f"  {cond}: {len(counts)} models, all classes checked")

    readme = OUT / "README.md"
    readme.write_text(
        "# Data release\n\n"
        "Exactly the annotated data behind the paper's tables and figures.\n\n"
        "| file | responses | citations | models |\n|---|---:|---:|---|\n"
        + "".join(f"| `{n}` | {manifest['files'][n]['responses']:,} | "
                 f"{manifest['files'][n]['classifiable_citations']:,} | "
                 f"{len(manifest['files'][n]['models'])} |\n" for n, _, _, _ in SPEC)
        + "\nAll three cover the same 500 questions. Record shape is one JSON object per\n"
          "line, identical to the source corpora: `id`, `cohort`, `question_id`,\n"
          "`source_model`, `source_model_raw`, `question`, `generated_answer`, and\n"
          "`annotations[]` with `label` (Ayah/Hadith), `segments[]` (`type`, `span_start`,\n"
          "`span_end`, `label`, `label_binary`), `hadith_grade`, `confidence` and `reason`.\n"
          "Character offsets index into `generated_answer`.\n\n"
          "## Scope\n\n"
          "The explicit file is restricted to the public train+dev splits: that condition\n"
          "is the released annotated data and has a held-out test split. The implicit\n"
          "files are complete -- that condition was never released and has no split, so\n"
          "nothing in it is held out.\n\n"
          "`manifest.json` carries per-model counts and a SHA-256 for each file.\n",
        encoding="utf-8")

    print(f"\nwrote {OUT / 'manifest.json'} and {readme}")
    print("SELF-CHECK:", "PASS -- release reproduces the published tables exactly"
          if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
