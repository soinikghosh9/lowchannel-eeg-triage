"""Where the ageing signal lives, and where the disease signal lives.

Diffuse frontal slowing is a feature of normal ageing; temporo-parietal disruption of the alpha
rhythm is the reported Alzheimer's signature. If that distinction holds in these recordings, the
two should separate under age adjustment: a frontal montage should lose most of its discrimination
when the case-control age gap is removed, and a posterior montage should keep more of it.

The consequence is a device-placement recommendation derived from physiology rather than from an
accuracy search -- which is the difference between a result and a leaderboard entry. It also has a
direct clinical reading: the regions whose deviation survives age adjustment are the ones a
clinician should be shown.

    python experiments/e05_regional.py
Out: outputs/results/e05_regional.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import normative as nm  # noqa: E402
from eegbudget import paths, spine  # noqa: E402
from eegbudget.features import NAMES  # noqa: E402
from eegbudget.montages import ALL  # noqa: E402

COHORT = "CAUEEG"
REGIONS = ["r_frontal", "r_temporal", "r_central", "r_posterior"]


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}

    df = spine.load()
    rows = df[(df["cohort"] == COHORT) & df["age"].notna()
              & (df["in_screening"] == 1)].reset_index(drop=True)
    ix = np.array([sub_index[s] for s in rows["subject"]])
    y = rows["impaired"].to_numpy()
    age = rows["age"].to_numpy(float)
    ctrl = y == 0
    keep_ix = ev.match_on_age(age, y)

    out = {"cohort": COHORT, "n": int(len(y)), "regions": {}}
    print(f"{'region':14s}{'el':>4s}{'EEG':>7s}{'matched':>9s}{'rho':>7s}   top age-independent features")
    print("-" * 92)

    for r in REGIONS:
        k = f"{r}|256|f32|full"
        if k not in keys:
            continue
        X = d["X"][keys.index(k)][ix]
        if not np.isfinite(X).any():
            continue

        s, keep = ev.cv_scores(X, y)
        a_eeg = ev.auc(y, s)
        s_m, _ = ev.cv_scores(X[keep_ix], y[keep_ix])
        a_m = ev.auc(y[keep_ix], s_m)
        rho = ev.substitution_ratio(a_eeg, a_m)

        # Per-feature: how much discrimination survives age matching. This is the clinician-facing
        # part -- it names which measurements in this region still mean something once age is off
        # the table, and those are the ones worth putting on a report.
        per_feat = {}
        model = nm.fit(X[ctrl], age[ctrl])
        Z = nm.deviation(model, X, age)
        for j, nm_ in enumerate(NAMES):
            if not np.isfinite(X[:, j]).any():
                continue
            raw = ev.auc(y, X[:, j])
            dev = ev.auc(y, np.abs(Z[:, j]))
            per_feat[nm_] = {"auc_raw": raw, "auc_deviation": dev,
                             "auc_matched": ev.auc(y[keep_ix], X[keep_ix, j])}

        def _lift(v):
            a = v["auc_matched"]
            return abs(a - .5) if a is not None and np.isfinite(a) else -1.0

        surviving = sorted(per_feat.items(), key=lambda kv: -_lift(kv[1]))[:3]
        out["regions"][r] = {"n_electrodes": len(ALL[r]["channels"]),
                             "channels": ALL[r]["channels"],
                             "auc_eeg": a_eeg, "auc_matched": a_m, "rho": rho,
                             "per_feature": per_feat}
        top = ", ".join(f"{n}({v['auc_matched']:.2f})" for n, v in surviving)
        print(f"{r:14s}{len(ALL[r]['channels']):4d}{a_eeg:7.3f}{a_m:9.3f}{rho:7.2f}   {top}")

    (paths.RESULTS / "e05_regional.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {paths.RESULTS / 'e05_regional.json'}")


if __name__ == "__main__":
    main()
