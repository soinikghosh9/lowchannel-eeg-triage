"""Each external cohort's increment over age, measured inside that cohort at its own size.

e35 carries a CAUEEG rule to each external cohort. That answers whether the rule travels, but its
training set is always CAUEEG's 783 dementia-contrast recordings, so it cannot be read against the
cohort-size curve of e20, where the model is both fitted and evaluated on n recordings. This fits
and evaluates inside each external cohort, as e20 does inside CAUEEG subsamples of the same size,
so the two can share an axis.

    ds004504, BrainLat   dementia (Alzheimer's and frontotemporal) against controls
    P-ADIC               Alzheimer's against controls (the only diagnosis it carries)

Age is refitted inside the cohort by the same cross-validation, so ds004504's inverted age
structure is learned rather than transported: the increment here is over the best age model the
cohort itself supports. EEG is the nineteen-electrode montage, with the seventeen features and the
prespecified three-feature slowing index, as in e20. 5-fold cross-validation repeated 20 times,
as for e33's within-cohort reading; paired DeLong interval on the averaged out-of-fold scores.

    python experiments/e37_external_within_increment.py
Out: outputs/results/e37_external_within_increment.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths  # noqa: E402
from eegbudget.easycog import SLOWING_TRIAD  # noqa: E402
from eegbudget.features import NAMES  # noqa: E402

COHORTS = {"ds004504": {"AD", "FTD"}, "BrainLat": {"AD", "FTD"}, "P-ADIC": {"AD"}}
KEY = "b19|256|f32|full"
REPEATS = 20
ESTIMATORS = {"eeg17": NAMES, "slow3": SLOWING_TRIAD}


def main():
    ext = np.load(paths.CACHE / "features_external.npz", allow_pickle=True)
    keys = [str(k) for k in ext["keys"]]
    X_all = ext["X"][keys.index(KEY)]
    eidx = {s: i for i, s in enumerate(ext["subjects"])}
    edf = pd.read_csv(paths.CACHE / "spine_external.csv")
    edf = edf[(edf.excluded.fillna("") == "") & edf.age.notna() & edf.subject.isin(eidx)]

    out = {"design": {"montage": KEY, "folds": 5, "repeats": REPEATS,
                      "age": "refitted within cohort by the same cross-validation"},
           "cohorts": {}}
    for cohort, pos in COHORTS.items():
        rows = edf[(edf.cohort == cohort) & edf.label.isin(pos | {"CN"})]
        ti = np.array([eidx[s] for s in rows.subject])
        y = rows.label.isin(pos).to_numpy().astype(int)
        age = rows.age.to_numpy(float)[:, None]
        s_age = ev.cv_scores(age, y, folds=5, repeats=REPEATS)[0]
        rec = {"n": int(len(y)), "n_case": int(y.sum()), "n_control": int((1 - y).sum()),
               "auc_age_raw": float(ev.auc(y, age[:, 0])), "auc_age_cv": float(ev.auc(y, s_age)),
               "estimators": {}}
        for est, names in ESTIMATORS.items():
            X = X_all[ti][:, [NAMES.index(n) for n in names]]
            s_eeg = ev.cv_scores(X, y, folds=5, repeats=REPEATS)[0]
            s_both = ev.cv_scores(np.column_stack([X, age]), y, folds=5, repeats=REPEATS)[0]
            inc = ev.delong_test(s_both, s_age, y)
            rec["estimators"][est] = {"auc_eeg": float(ev.auc(y, s_eeg)),
                                      "auc_eeg_plus_age": float(ev.auc(y, s_both)),
                                      "inc_margin": float(inc["diff"]),
                                      "inc_ci": [float(v) for v in inc["ci"]],
                                      "inc_p": float(inc["p"])}
        out["cohorts"][cohort] = rec
        e17 = rec["estimators"]["eeg17"]
        e3 = rec["estimators"]["slow3"]
        print(f"{cohort:9s} n={rec['n']:3d} ({rec['n_case']} cases)  age raw {rec['auc_age_raw']:.3f}"
              f" cv {rec['auc_age_cv']:.3f}  |  17 feat: EEG+age {e17['auc_eeg_plus_age']:.3f}"
              f" inc {e17['inc_margin']:+.3f} [{e17['inc_ci'][0]:+.3f}, {e17['inc_ci'][1]:+.3f}]"
              f"  |  3 feat: inc {e3['inc_margin']:+.3f} [{e3['inc_ci'][0]:+.3f}, {e3['inc_ci'][1]:+.3f}]")
    path = paths.RESULTS / "e37_external_within_increment.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
