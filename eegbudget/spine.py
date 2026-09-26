"""The subject table every experiment reads.

One row per recording in the harmonised cache, carrying the label, the path to the signal, and --
where the source cohort published it -- age and diagnostic subtype. Built once by
``experiments/e00_spine.py`` and cached, so no experiment re-parses the corpus.

Age is available for CAUEEG (annotation.json, all 1,379 subjects) and ds004504 (participants.tsv).
It is not available for GENEEG, MISP, P-ADIC, AlzEEG, Mendeley or MCI_Dataset, and MISP's
spreadsheet carries only a status column. Cohorts without age cannot enter the normative arms;
they are kept in the table so that budget sweeps can still use them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths

#: Source label vocabularies disagree between cohorts. Normalise once, here, and nowhere else.
LABELS = {"CN": "CN", "CONTROL": "CN", "Healthy": "CN", "healthy": "CN", "normal": "CN",
          "Normal": "CN", "C": "CN", "HC": "CN",
          "AD": "AD", "A": "AD",
          "FTD": "FTD", "F": "FTD",
          "MCI": "MCI", "DEM": "AD", "Dementia": "AD", "dementia": "AD",
          # Recovered by e09_extend_corpus.py; the parent taxonomy discards both.
          "VAD": "VAD", "TGA": "TGA"}

IMPAIRED = {"AD", "FTD", "MCI", "VAD"}

#: Labels admitted to the screening contrast. Transient global amnesia is amnestic but not
#: neurodegenerative, so it belongs in neither arm: counting it as impaired would reward a
#: screener for flagging a self-limiting condition, and counting it as control would punish one.
#: It is analysed only as a false-referral subgroup, like subjective memory impairment.
SCREENING = {"CN", "AD", "FTD", "MCI", "VAD"}


def _caueeg_meta():
    """serial -> (age, semicolon-joined symptom tags)."""
    if not paths.CAUEEG_ANNOTATION.exists():
        return {}
    data = json.loads(paths.CAUEEG_ANNOTATION.read_text())["data"]
    return {r["serial"]: (float(r["age"]), ";".join(r.get("symptom") or [])) for r in data}


def _ds004504_meta():
    """participant_id -> (age, '')."""
    if not paths.DS004504_PARTICIPANTS.exists():
        return {}
    df = pd.read_csv(paths.DS004504_PARTICIPANTS, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return {str(r["participant_id"]).strip(): (float(r["Age"]), "") for _, r in df.iterrows()}


def build():
    """Scan the harmonised cache and join demographics. Returns a DataFrame."""
    paths.require_source()
    cau, ds = _caueeg_meta(), _ds004504_meta()

    dirs = [d for d in (paths.HARMONISED, paths.HARMONISED_EXTRA) if d.is_dir()]
    rows = []
    for js in sorted(j for d in dirs for j in d.glob("*.json")):
        npy = js.with_suffix(".npy")
        if not npy.exists():
            continue
        d = json.loads(js.read_text())
        subject, cohort = d["subject"], d["cohort"]
        label = LABELS.get(d["group"])
        if label is None:
            raise ValueError(f"{subject}: unmapped label {d['group']!r}")

        age, subtype = np.nan, ""
        if cohort == "CAUEEG":
            age, subtype = cau.get(subject.rsplit("_", 1)[-1], (np.nan, ""))
        elif cohort == "ds004504_raw":
            age, subtype = ds.get(subject, (np.nan, ""))

        rows.append({"subject": subject, "cohort": cohort, "label": label,
                     "age": age, "subtype": subtype,
                     "duration_s": d.get("qc", {}).get("duration_s", np.nan),
                     "n_channels": len(d["channels"]), "npy": str(npy)})

    df = pd.DataFrame(rows)
    df["impaired"] = df["label"].isin(IMPAIRED).astype(int)
    df["in_screening"] = df["label"].isin(SCREENING).astype(int)
    return df.sort_values(["cohort", "subject"]).reset_index(drop=True)


def load():
    if not paths.SPINE.exists():
        raise SystemExit("spine not built -- run: python experiments/e00_spine.py")
    df = pd.read_csv(paths.SPINE)
    # Derived columns are recomputed rather than trusted, so a spine written before a column
    # existed still works. Both follow from `label` alone, so there is nothing to get stale.
    df["impaired"] = df["label"].isin(IMPAIRED).astype(int)
    df["in_screening"] = df["label"].isin(SCREENING).astype(int)
    return df


def has_subtype(df, tag):
    """Boolean mask for a CAUEEG symptom tag, matching whole tags only.

    Substring matching would be wrong here: 'ad' occurs inside 'mci_ad' and 'ad_vd_mixed', and
    'mci' inside every MCI subtype, so a naive `str.contains` silently merges distinct groups.
    """
    return df["subtype"].fillna("").apply(lambda s: tag in s.split(";"))
