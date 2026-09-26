"""Widen the external test from one cohort to three, with the pipeline the paper already runs.

The review asked whether the placement result survives outside the development cohort, and noted
that one 88-participant external set is too small to say. Two further dementia-spectrum cohorts sit
in the same harmonised cache, in the same 19-channel 256 Hz format, with age published:

    ds004504   Greece, 88     AD 36, FTD 23, CN 29                  (already the paper's test set)
    BrainLat   Argentina and Chile, 80   AD 30, bvFTD 18, CN 32     (Prado et al. 2023)
    P-ADIC     Israel, two centres       AD 43, CN aged >= 55       (Shor et al. 2021)

P-ADIC controls are restricted to age >= 55, the rule the companion study fixed before this one,
because its control file includes young adults that would make any contrast trivially easy. P-ADIC
records eyes open and closed without marking which, a protocol difference kept as a caveat.

Nothing here refits or touches the paper's cache: features land in a separate file, and ds004504 is
re-extracted and compared against ``features.npz`` so the two caches are known to agree.

    python experiments/e31_external_extract.py
Out: outputs/cache/features_external.npz, outputs/cache/spine_external.csv,
     outputs/results/e31_external_extract.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import budgets, paths, spine  # noqa: E402
from eegbudget.degrade import apply_budget  # noqa: E402
from eegbudget.features import NAMES, subject_features  # noqa: E402
from eegbudget.montages import ALL, apply_montage, reorder_to_canon  # noqa: E402

BRAINLAT_DEMO = paths.DATASETS / "BrainLat Data" / "brainlat"
PADIC_MAT = {"AD": paths.DATASETS / "P-ADIC" / "p-adic alz" / "alz_c1_new.mat",
             "CN": paths.DATASETS / "P-ADIC" / "p-adic ctrl" / "controls_c1_new.mat"}
PADIC_CONTROL_MIN_AGE = 55.0

#: The montage axis at a clinical amplifier, plus the digitisation the paper recommends for the
#: device (128 Hz, 8 bits) at 19 and 4 electrodes.
CONDITIONS = ([{"montage": m, "rate": 256, "bits": None, "preproc": "full"}
               for m in ("b19", "b8", "b4", "b2", "b1", "b1_ap", "r_frontal", "r_temporal",
                         "r_central", "r_posterior", "muse", "ganglion", "insight", "frontal1",
                         "b4_inherit", "b8_inherit")]
              + [{"montage": m, "rate": 128, "bits": 8, "preproc": "full"} for m in ("b19", "b4")])


def _brainlat_rows():
    demo = pd.concat([pd.read_csv(BRAINLAT_DEMO / f"Demographics_{g}_EEG_data.csv")
                      for g in ("AD", "HC", "bvFTD")], ignore_index=True)
    demo = demo[demo["id EEG"].astype(str).str.startswith("sub-")]
    meta = {r["id EEG"]: r for _, r in demo.iterrows()}
    rows = []
    for js in sorted(paths.HARMONISED.glob("BRAINLAT_*.json")):
        d = json.loads(js.read_text())
        sid = d["subject"].split("_", 2)[-1]
        m = meta.get(sid)
        rows.append({"subject": d["subject"], "cohort": "BrainLat",
                     "label": spine.LABELS[d["group"]],
                     "age": float(m["Age"]) if m is not None else np.nan,
                     "sex": {0: "F", 1: "M"}.get(int(m["sex"])) if m is not None else None,
                     "education": float(m["years_education"]) if m is not None else np.nan,
                     "site": str(m["path"]).split("/")[-1] if m is not None else "",
                     "duration_s": d["qc"]["duration_s"], "npy": str(js.with_suffix(".npy"))})
    return rows


def _padic_rows():
    """Ages come from the release's own struct, read with the parent project's loader."""
    from harmonisation.padic_group_loader import PADICGroupDataset
    info = {}
    for g, mat in PADIC_MAT.items():
        ds = PADICGroupDataset(mat, g, f"PADIC_{g}")
        info.update({sid: ds.info_map[sid] for sid in ds.subject_ids})
    rows = []
    for js in sorted(paths.HARMONISED.glob("PADIC_*.json")):
        d = json.loads(js.read_text())
        if d["group"] not in PADIC_MAT:
            continue                      # MCI is not part of the external dementia test
        i = info.get(d["subject"])
        rows.append({"subject": d["subject"], "cohort": "P-ADIC",
                     "label": spine.LABELS[d["group"]],
                     "age": float(i["age"]) if i and i["age"] else np.nan,
                     "sex": i["sex"] if i else None, "education": np.nan, "site": "IL",
                     "duration_s": d["qc"]["duration_s"], "npy": str(js.with_suffix(".npy"))})
    return rows


def brainlat_site_offsets():
    """How far BrainLat's stand-in electrodes sit from the 10-20 sites they replace, in mm.

    BrainLat records on a 128-electrode Biosemi cap; the harmoniser takes the cap electrode nearest
    each 10-20 site (template positions, as the parent loader does). The offset is a placement
    perturbation built into that cohort, and the placement analysis should say how large it is.
    """
    import mne
    bio = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    std = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    alias = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    cap = np.array(list(bio.values()))
    from eegbudget.montages import CANON
    return {s: float(np.linalg.norm(cap - std[alias.get(s, s)], axis=1).min() * 1000) for s in CANON}


def build_spine():
    ds = spine.load()
    ds = ds[ds["cohort"] == "ds004504_raw"].copy()
    ds["cohort"], ds["sex"], ds["education"], ds["site"] = "ds004504", None, np.nan, "GR"
    rows = ds[["subject", "cohort", "label", "age", "sex", "education", "site",
               "duration_s", "npy"]].to_dict("records")
    rows += _brainlat_rows() + _padic_rows()
    df = pd.DataFrame(rows)
    df["impaired"] = df["label"].isin(spine.IMPAIRED).astype(int)
    drop = (df["cohort"] == "P-ADIC") & (df["label"] == "CN") & ~(df["age"] >= PADIC_CONTROL_MIN_AGE)
    df["excluded"] = np.where(drop, "P-ADIC control under 55", "")
    return df


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")
    df = build_spine()
    df.to_csv(paths.CACHE / "spine_external.csv", index=False)
    use = df[df["excluded"] == ""].reset_index(drop=True)
    print(use.groupby(["cohort", "label"]).agg(n=("subject", "size"), age=("age", "mean")).round(1))

    keys = [budgets.key(c) for c in CONDITIONS]
    X = np.full((len(CONDITIONS), len(use), len(NAMES)), np.nan)
    n_win = np.zeros(len(use))
    t0 = time.time()
    for i, row in use.iterrows():
        meta = json.loads(Path(row["npy"]).with_suffix(".json").read_text())
        x = reorder_to_canon(np.load(row["npy"]).astype(np.float64), meta["channels"])
        w, _ = budgets.select_windows(x, subject=row["subject"])
        n_win[i] = len(w)
        if len(w) == 0:
            continue
        for ci, c in enumerate(CONDITIONS):
            m = ALL[c["montage"]]
            deg, sf = apply_budget(apply_montage(w, m), budgets.SF_NATIVE, c["rate"], c["bits"])
            X[ci, i] = subject_features(deg, sf, m)
        if (i + 1) % 50 == 0 or i + 1 == len(use):
            print(f"  {i + 1}/{len(use)}  {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(paths.CACHE / "features_external.npz", X=X, keys=np.array(keys),
                        names=np.array(NAMES), subjects=use["subject"].to_numpy(dtype=object),
                        n_windows=n_win)

    # The re-extracted ds004504 rows must reproduce the paper's cache exactly.
    ref = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    rk, rs = [str(k) for k in ref["keys"]], {s: j for j, s in enumerate(ref["subjects"])}
    dsi = [i for i, c in enumerate(use["cohort"]) if c == "ds004504"]
    worst = 0.0
    for ci, k in enumerate(keys):
        if k not in rk:
            continue
        a = X[ci][dsi]
        b = ref["X"][rk.index(k)][[rs[s] for s in use["subject"].iloc[dsi]]]
        both = np.isfinite(a) & np.isfinite(b)
        worst = max(worst, float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0)
        assert (np.isfinite(a) == np.isfinite(b)).all(), f"{k}: finite masks differ"

    counts = {c: {l: int(n) for l, n in g["label"].value_counts().items()}
              for c, g in use.groupby("cohort")}
    ages = {c: {l: round(float(v), 1) for l, v in g.groupby("label")["age"].mean().items()}
            for c, g in use.groupby("cohort")}
    report = {"n": int(len(use)), "counts": counts, "mean_age": ages,
              "missing_age": {c: int(g["age"].isna().sum()) for c, g in use.groupby("cohort")},
              "padic_controls_excluded_under_55": int((df["excluded"] != "").sum()),
              "brainlat_sites": {k: int(v) for k, v in
                                 use[use.cohort == "BrainLat"]["site"].value_counts().items()},
              "brainlat_site_offset_mm": brainlat_site_offsets(),
              "conditions": keys, "median_windows": float(np.median(n_win)),
              "min_windows": float(n_win.min()),
              "ds004504_max_abs_diff_vs_features_npz": worst,
              "elapsed_s": round(time.time() - t0, 1)}
    (paths.RESULTS / "e31_external_extract.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("counts", "mean_age", "missing_age",
                                              "padic_controls_excluded_under_55", "brainlat_sites",
                                              "median_windows", "min_windows",
                                              "ds004504_max_abs_diff_vs_features_npz")}, indent=1))


if __name__ == "__main__":
    main()
