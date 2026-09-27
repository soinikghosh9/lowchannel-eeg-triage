"""The preprocessing axis: what a device gives up when it cannot run ICA.

Every other result in this project sits on ICLabel artefact removal, which needs many channels,
compute and an operator to review components. A four-electrode wearable has none of those, so the
question is not academic: if the increment survives without artefact removal, the acquisition
recommendation stands as written; if it does not, preprocessing is the binding constraint and the
electrode count matters less than it appears.

The comparison is paired and complete-case: a participant without a no-ICA counterpart is dropped
from every arm, so all arms are scored on the same people. Since the no-ICA arm was re-extracted for
the whole cohort, none is dropped, and the ICA column reproduces the headline AUCs. All three CAUEEG
contrasts are scored, as in the margin table.

One detail makes this a fair test of preprocessing alone. The no-ICA recordings reuse the sample
offsets the ICA arm selected rather than re-running artefact rejection. With artefact removal
switched off more windows exceed the peak-to-peak threshold, so re-selecting would confound the
preprocessing change with a change in which seconds of signal were analysed.

    python experiments/e27_preprocessing_budget.py
Out: outputs/results/e27_preprocessing_budget.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

COHORT = "CAUEEG"
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"}, "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}
MONTAGES = ["b19", "b4"]
ARMS = ["full", "noica", "emgfree"]


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT, "arms": ARMS, "montages": MONTAGES,
           "note": ("Complete-case paired comparison. Participants without a no-ICA counterpart "
                    "are dropped from every arm so that all arms are scored on identical people."),
           "tasks": {}}

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix_all = np.array([sub_index[s] for s in rows["subject"]])

        # A participant is covered when the no-ICA arm produced any feature for them.
        covered = np.isfinite(d["X"][keys.index(f"b19|256|f32|noica")][ix_all]).any(1)
        ix = ix_all[covered]
        y = rows.loc[covered, "label"].isin(pos).to_numpy().astype(int)
        age = rows.loc[covered, "age"].to_numpy(float)
        dropped = int((~covered).sum())

        # Age is scored as its own value (as in e13), not through a per-fold logistic fit, so the
        # age baseline here is the rank AUC that the rest of the paper reports.
        auc_age = ev.auc(y, age)
        rec = {"n_analysed": int(covered.sum()), "n_dropped": dropped,
               "n_case": int(y.sum()), "auc_age": auc_age, "montages": {}}
        print(f"\n=== {task}: {covered.sum()} analysed, {dropped} dropped for lack of a "
              f"no-ICA counterpart, age AUC {auc_age:.3f}")

        for m in MONTAGES:
            # One feature mask across all arms: a column usable in one arm but not another would
            # otherwise let the comparison turn on which features were available, not on ICA.
            mats = {a: d["X"][keys.index(f"{m}|256|f32|{a}")][ix] for a in ARMS}
            keep = np.ones(mats["full"].shape[1], bool)
            for a in ARMS:
                keep &= np.isfinite(mats[a]).all(0)
            if not keep.any():
                continue
            arm_out, scores = {}, {}
            for a in ARMS:
                X = mats[a][:, keep]
                s_eeg = ev.cv_scores(X, y)[0]
                s_both = ev.cv_scores(np.column_stack([X, age]), y)[0]
                inc = ev.delong_test(s_both, age, y)
                arm_out[a] = {"auc_eeg": ev.auc(y, s_eeg),
                              "auc_eeg_plus_age": ev.auc(y, s_both),
                              "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"]}
                scores[a] = s_eeg
            # The question is whether dropping ICA costs discrimination, so the test is paired
            # against the ICA arm on identical participants.
            for a in ARMS:
                if a == "full":
                    continue
                t = ev.delong_test(scores[a], scores["full"], y)
                arm_out[a]["vs_full"] = {"diff": t["diff"], "ci": t["ci"], "p": t["p"]}
            arm_out["n_features"] = int(keep.sum())
            rec["montages"][m] = arm_out

            f = arm_out["full"]
            n_ = arm_out["noica"]
            print(f"  {m}: {int(keep.sum())} features | ICA {f['auc_eeg']:.3f} "
                  f"(inc {f['inc_margin']:+.3f}) | no ICA {n_['auc_eeg']:.3f} "
                  f"(inc {n_['inc_margin']:+.3f}) | diff {n_['vs_full']['diff']:+.3f} "
                  f"[{n_['vs_full']['ci'][0]:+.3f}, {n_['vs_full']['ci'][1]:+.3f}] "
                  f"P={n_['vs_full']['p']:.3f}")

        out["tasks"][task] = rec

    dst = paths.RESULTS / "e27_preprocessing_budget.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
