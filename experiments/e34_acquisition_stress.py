"""How much of the recording's value survives the faults of a real low-cost acquisition (e32).

Two regimes, because they fail differently:

    device-trained   fit and test on degraded recordings: is the information still there?
    clinic-trained   fit on the clean clinical recordings, test on degraded ones: can a rule
                     developed in the clinic be carried onto a worse device unchanged?

Each is scored at 19 and at 4 electrodes on the screening (primary) and dementia contrasts, as
EEG-only AUC and as the increment over age, with the paper's 5x5 recording-stratified
cross-validation. Differences against the clean reference are paired on identical recordings
(DeLong). This is a robustness sweep, reported descriptively; no condition enters a test family.

    python experiments/e34_acquisition_stress.py
Out: outputs/results/e34_acquisition_stress.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"}, "dementia": {"AD", "FTD", "VAD"}}


def cv_shift(X_fit, X_score, y, folds=5, repeats=5, seed=ev.SEED):
    """Out-of-fold scores from a model fitted on X_fit's training rows and applied to X_score."""
    keep = np.isfinite(X_fit).all(0) & np.isfinite(X_score).all(0)
    A, B = X_fit[:, keep], X_score[:, keep]
    acc = np.zeros(len(y))
    for r in range(repeats):
        oof = np.empty(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(A, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(A[tr], y[tr])
            oof[te] = m.decision_function(B[te])
        acc += rankdata(oof) / len(oof)
    return acc / repeats


def main():
    warnings.simplefilter("ignore")
    z = np.load(paths.CACHE / "features_stress.npz", allow_pickle=True)
    X, conds, monts = z["X"], [str(c) for c in z["conditions"]], [str(m) for m in z["montages"]]
    spec = json.loads(str(z["spec"]))
    subj = [str(s) for s in z["subjects"]]
    df = spine.load().set_index("subject").loc[subj].reset_index()
    ref = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    rkeys, rsub = [str(k) for k in ref["keys"]], {str(s): i for i, s in enumerate(ref["subjects"])}

    out = {"conditions": spec, "tasks": {}}
    for task, pos in TASKS.items():
        m_task = df["label"].isin(pos | {"CN"}).to_numpy()
        y = df.loc[m_task, "label"].isin(pos).to_numpy().astype(int)
        age = df.loc[m_task, "age"].to_numpy(float)
        auc_age = ev.auc(y, age)
        rec_t = {"n": int(len(y)), "auc_age": auc_age, "montages": {}}
        for mi, mont in enumerate(monts):
            Xm = X[:, mi][:, m_task]
            clean = Xm[conds.index("clean")]
            s_clean = ev.cv_scores(clean, y)[0]
            b_clean = ev.cv_scores(np.column_stack([clean, age]), y)[0]
            # The paper's cache selects windows on all 19 channels; the device-side rule here selects
            # on the montage's own. Recorded so the reference is known to be the same instrument.
            paper = ref["X"][rkeys.index(f"{mont}|256|f32|full")][
                [rsub[s] for s in np.array(subj)[m_task]]]
            rec_m = {"paper_cache_auc_eeg": ev.auc(y, ev.cv_scores(paper, y)[0]), "conditions": {}}
            for ci, cond in enumerate(conds):
                Xc = Xm[ci]
                row = {}
                for regime in ("device", "clinic"):
                    if regime == "device":
                        s = ev.cv_scores(Xc, y)[0]
                        b = ev.cv_scores(np.column_stack([Xc, age]), y)[0]
                    else:
                        s = cv_shift(clean, Xc, y)
                        b = cv_shift(np.column_stack([clean, age]), np.column_stack([Xc, age]), y)
                    inc = ev.delong_test(b, age, y)
                    dd = ev.delong_test(s, s_clean, y)
                    di = ev.delong_test(b, b_clean, y)
                    row[regime] = {"auc_eeg": ev.auc(y, s), "auc_eeg_ci": list(ev.bootstrap_ci(y, s)),
                                   "auc_eeg_plus_age": ev.auc(y, b),
                                   "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"],
                                   "d_auc_eeg_vs_clean": dd["diff"], "d_auc_eeg_ci": dd["ci"],
                                   "d_inc_vs_clean": di["diff"], "d_inc_ci": di["ci"]}
                rec_m["conditions"][cond] = row
            rec_t["montages"][mont] = rec_m
        out["tasks"][task] = rec_t

        print(f"\n=== {task}: n={len(y)}, age {auc_age:.3f}")
        print(f"  {'condition':15s}" + "".join(f"{m + ' ' + r:>28s}" for m in monts
                                               for r in ("device", "clinic")))
        for cond in conds:
            line = f"  {cond:15s}"
            for m in monts:
                for r in ("device", "clinic"):
                    v = rec_t["montages"][m]["conditions"][cond][r]
                    line += f"   AUC {v['auc_eeg']:.3f} inc {v['inc_margin']:+.3f}{'*' if v['inc_ci'][0] > 0 else ' '}"
            print(line)
        for m in monts:
            print(f"  paper-cache EEG AUC {m}: {rec_t['montages'][m]['paper_cache_auc_eeg']:.3f}")

    def clean_json(o):
        if isinstance(o, dict):
            return {k: clean_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [clean_json(v) for v in list(o)]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    dst = paths.RESULTS / "e34_acquisition_stress.json"
    dst.write_text(json.dumps(clean_json(out), indent=1), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
