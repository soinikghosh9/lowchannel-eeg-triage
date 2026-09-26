"""Build the subject table and report what the corpus can and cannot support.

    python experiments/e00_spine.py
Out: outputs/cache/spine.csv, outputs/results/e00_spine.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import paths, spine  # noqa: E402

#: Counts as published by the parent project, which this study opens read-only. A mismatch here
#: means the upstream corpus changed underneath us and every downstream number is suspect.
EXPECTED_PARENT = {"CAUEEG": 1120, "ds004504_raw": 88, "GENEEG": 63, "AlzEEG": 183,
                   "MCI_Dataset": 166, "Mendeley": 35, "MISP": 29, "P-ADIC": 28}

#: Subjects e09_extend_corpus.py recovers from the parent taxonomy's excluded set. They land in
#: the same cohort, so the check has to allow for them or it fires on our own additions.
EXPECTED_RECOVERED = {"CAUEEG": 152}

EXPECTED = {c: n + EXPECTED_RECOVERED.get(c, 0) for c, n in EXPECTED_PARENT.items()}


def main():
    paths.ensure_dirs()
    df = spine.build()
    df.to_csv(paths.SPINE, index=False)

    counts = df["cohort"].value_counts().to_dict()
    mismatch = {c: (counts.get(c, 0), n) for c, n in EXPECTED.items() if counts.get(c, 0) != n}

    with_age = df[df["age"].notna()]
    report = {
        "n_recordings": len(df),
        "by_cohort": counts,
        "by_label": df["label"].value_counts().to_dict(),
        "expected_mismatch": mismatch,
        "age_available": {c: int(g["age"].notna().sum()) for c, g in df.groupby("cohort")},
        "n_with_age": len(with_age),
        "age_by_label": {k: {"n": len(g), "mean": round(float(g["age"].mean()), 1),
                             "sd": round(float(g["age"].std()), 1)}
                         for k, g in with_age.groupby("label")},
    }

    for tag in ("eoad", "load", "vd", "sivd", "smi", "tga", "mci_amnestic", "mci_vascular", "ad"):
        m = spine.has_subtype(df, tag)
        if m.sum():
            sub = df[m]
            report.setdefault("subtypes", {})[tag] = {
                "n": int(m.sum()),
                "age_mean": round(float(sub["age"].mean()), 1),
                "age_sd": round(float(sub["age"].std()), 1)}

    (paths.RESULTS / "e00_spine.json").write_text(json.dumps(report, indent=2))

    print(f"{len(df)} recordings, {len(with_age)} with age\n")
    print(f"{'cohort':16s}{'n':>6s}{'age':>6s}")
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{c:16s}{n:6d}{report['age_available'][c]:6d}")
    print("\nage by label:")
    for k, v in sorted(report["age_by_label"].items()):
        print(f"  {k:5s} n={v['n']:5d}  {v['mean']:.1f} +- {v['sd']:.1f}")
    print("\nsubtypes with age:")
    for k, v in sorted(report.get("subtypes", {}).items(), key=lambda x: -x[1]["n"]):
        print(f"  {k:14s} n={v['n']:4d}  {v['age_mean']:.1f} +- {v['age_sd']:.1f}")
    if mismatch:
        # The guard's whole purpose is that a corpus which changed underneath us invalidates
        # every downstream number. Printing a warning and writing the mismatch into the
        # results JSON -- where it sat unnoticed through a full manuscript cycle -- is not
        # enforcement. Halt instead, so run_queue.py stops the pipeline.
        raise SystemExit(
            "cohort count mismatch (got, expected): " + repr(mismatch) + " -- the upstream "
            "corpus differs from what this study was built against, so every downstream "
            "number is suspect. Reconcile EXPECTED_PARENT / EXPECTED_RECOVERED in this file "
            "against the corpus before continuing."
        )
    print()
    print("all cohort counts match the parent project")


if __name__ == "__main__":
    main()
