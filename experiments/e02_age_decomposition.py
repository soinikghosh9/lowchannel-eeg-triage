"""How much of each budget's discrimination is age rather than EEG.

For every acquisition budget, discrimination is scored four ways: from age alone, from EEG alone,
from both, and from EEG after the case-control age gap is removed. The gap is removed twice by
different means -- nearest-neighbour matching, which discards subjects, and quadratic
residualisation, which keeps them -- and a claim is made only where the two agree.

Two quantities are reported. The substitution ratio is the share of above-chance discrimination
that disappears once age is equalised. The **age margin** is the plainer and, as it turns out, more
decisive one: how much the EEG instrument beats what the patient's date of birth already supplies.
A negative margin means the recording is worth less than the birth certificate.

The full budget sweep runs on the screening contrast. The ladder is additionally scored on
dementia and on mild cognitive impairment separately, because pooling them hides the fact that the
two behave very differently against the same baseline.

    python experiments/e02_age_decomposition.py
Out: outputs/results/e02_age_decomposition.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402
from eegbudget.montages import ALL  # noqa: E402

COHORT = "CAUEEG"
LADDER = ["b19", "b8", "b4", "b2", "b1", "b1_ap"]

#: Pre-specified primary comparisons. Everything else in the sweep is exploratory and is reported
#: without inferential claims, which is the honest way to run 40 conditions across three tasks
#: without spending the alpha on whichever one happened to look best.
PRIMARY = {"b19|256|f32|full", "b4|256|f32|full"}

#: label sets defining each contrast, as (positive labels, name)
TASKS = {
    "screening": {"AD", "FTD", "MCI", "VAD"},
    "dementia": {"AD", "FTD", "VAD"},
    "mci": {"MCI"},
}


def rho_ci(y, s_eeg, y_m, s_m, n=1500, seed=0):
    """Bootstrap interval for the substitution ratio.

    The ratio is a function of two AUCs estimated on overlapping samples, so it has no closed-form
    interval. Both are resampled jointly within class and the ratio recomputed, which propagates
    the uncertainty in each rather than reporting a point estimate as if it were exact.
    """
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    jp, jneg = np.where(y_m == 1)[0], np.where(y_m == 0)[0]
    vals = []
    for _ in range(n):
        a = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
        b = np.concatenate([rng.choice(jp, len(jp), True), rng.choice(jneg, len(jneg), True)])
        r = ev.substitution_ratio(ev.auc(y[a], s_eeg[a]), ev.auc(y_m[b], s_m[b]))
        if np.isfinite(r):
            vals.append(r)
    if len(vals) < 50:
        return [np.nan, np.nan]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def arms(X, y, age, keep_ix, primary=False):
    """The four scorings of one feature matrix, plus the paired test against age."""
    s_eeg, keep = ev.cv_scores(X, y)
    a_eeg = ev.auc(y, s_eeg)
    s_both, _ = ev.cv_scores(np.column_stack([X, age]), y)
    s_m, _ = ev.cv_scores(X[keep_ix], y[keep_ix])
    # Residualisation is part of the estimator, so it must be fitted inside each training fold.
    # The previous full-sample transform leaked the case/control age relationship into the test
    # fold and made the residualised rho look more stable than it was.
    s_r, _ = ev.cv_residualised_scores(X, y, age, controls_only=True)

    rec = {
        "n_features_used": int(keep.sum()),
        "auc_eeg": a_eeg,
        "auc_eeg_ci": list(ev.bootstrap_ci(y, s_eeg, n=800)),
        "auc_eeg_plus_age": ev.auc(y, s_both),
        "auc_matched": ev.auc(y[keep_ix], s_m),
        "auc_residualised": ev.auc(y, s_r),
        # The margin is a difference between two AUCs on identical subjects, so it is tested as
        # one -- separate intervals on each cannot answer whether they differ.
        "vs_age": ev.delong_test(s_eeg, age, y),
    }
    if primary:
        # The pre-specified primary comparison also gets a distribution-free check, and the
        # substitution ratio gets an interval rather than a bare point estimate.
        rec["vs_age_bootstrap"] = ev.paired_bootstrap_diff(s_eeg, age, y, n=3000)
        rec["rho_matched_ci"] = rho_ci(y, s_eeg, y[keep_ix], s_m)
    return rec


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    X_all, keys = d["X"], [str(k) for k in d["keys"]]
    df = spine.load()
    sub_index = {s: i for i, s in enumerate(d["subjects"])}

    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]
    out = {"cohort": COHORT, "tasks": {}}

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        if y.sum() < 30 or (y == 0).sum() < 30:
            continue

        auc_age = ev.auc(y, age)
        keep_ix = ev.match_on_age(age, y)
        sweep = list(keys) if task == "screening" else [f"{m}|256|f32|full" for m in LADDER]

        rec = {"n": int(len(y)), "n_control": int((y == 0).sum()), "n_positive": int(y.sum()),
               "age_only": {"auc": auc_age, "ci": list(ev.bootstrap_ci(y, age))},
               "matched": {"n": int(len(keep_ix)),
                           "residual_age_auc": ev.auc(y[keep_ix], age[keep_ix])},
               "budgets": {}}

        print(f"\n=== {task}: n={len(y)} ({(y==0).sum()} CN, {y.sum()} case) "
              f"| AGE ALONE {auc_age:.3f} | matched n={len(keep_ix)}")
        print(f"{'budget':12s}{'el':>4s}{'EEG':>7s}{'margin':>8s}{'+age':>7s}"
              f"{'matched':>9s}{'resid':>7s}{'rho_m':>7s}{'rho_r':>7s}{'p vs age':>10s}")

        for k in sweep:
            if k not in keys:
                continue
            X = X_all[keys.index(k)][ix]
            if not np.isfinite(X).any():
                continue
            a = arms(X, y, age, keep_ix, primary=(k in PRIMARY))
            a["montage"] = k.split("|")[0]
            a["n_electrodes"] = len(ALL[a["montage"]]["channels"])
            a["age_margin"] = a["auc_eeg"] - auc_age
            a["rho_matched"] = ev.substitution_ratio(a["auc_eeg"], a["auc_matched"])
            a["rho_residualised"] = ev.substitution_ratio(a["auc_eeg"], a["auc_residualised"])
            rec["budgets"][k] = a

            if k.endswith("256|f32|full") and a["montage"] in LADDER:
                print(f"{a['montage']:12s}{a['n_electrodes']:4d}{a['auc_eeg']:7.3f}"
                      f"{a['age_margin']:+8.3f}{a['auc_eeg_plus_age']:7.3f}"
                      f"{a['auc_matched']:9.3f}{a['auc_residualised']:7.3f}"
                      f"{a['rho_matched']:7.2f}{a['rho_residualised']:7.2f}"
                      f"{a['vs_age']['p']:10.1e}")
        out["tasks"][task] = rec

    (paths.RESULTS / "e02_age_decomposition.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {paths.RESULTS / 'e02_age_decomposition.json'}")


if __name__ == "__main__":
    main()
