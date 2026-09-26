"""Operating-point metrics a screening study has to report, and the external cohort.

AUC summarises ranking across all thresholds. A screening service runs at one threshold, in a
population whose prevalence is far lower than any referred clinical sample, and what it experiences
is the positive predictive value: of everyone the instrument flags, how many turn out to have the
condition. At 10% prevalence a rule with 80% specificity and 65% sensitivity flags three people
without the condition for every one with it. That number belongs beside the AUC.

This file computes, for age alone, EEG alone and EEG+age:

    sensitivity, specificity      at a threshold fixed to 80% specificity in cross-validated controls
    PPV, NPV                      transformed to assumed service prevalences by Bayes' rule
    balanced accuracy             prevalence-independent summary of the same operating point
    average precision             ranking under class imbalance, at the sample's own prevalence
    Brier score                   accuracy of the probabilities themselves
    calibration slope, intercept  whether those probabilities mean what they say

and repeats the primary contrast on the external Greek cohort, where cases are younger than
controls, so that the age baseline there is not the one the training cohort supplies.

    python experiments/e23_operating_metrics.py
Out: outputs/results/e23_operating_metrics.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import normative as nm  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

COHORT, TARGET = "CAUEEG", "ds004504_raw"
BUDGETS = {"b19": "b19|256|f32|full", "b4": "b4|256|f32|full"}
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}
SPEC_TARGET = 0.80
PREVALENCES = [0.05, 0.10, 0.20]


def _rank_to_prob(score):
    """Map a rank-fraction score onto (0, 1) so it can be read at a probability threshold.

    ``cv_normative_scores`` returns a rank fraction, which is all an AUC needs. A decision curve
    needs a risk. This is a monotone rescaling, so it changes no ranking and no AUC; it makes the
    arm comparable at an operating point without pretending the values are calibrated risks.
    """
    s = np.asarray(score, float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)


def _ppv_npv(sens, spec, prev):
    """Predictive values at an assumed prevalence, from sensitivity and specificity."""
    tp, fp = sens * prev, (1 - spec) * (1 - prev)
    tn, fn = spec * (1 - prev), (1 - sens) * prev
    return (float(tp / (tp + fp)) if tp + fp > 0 else np.nan,
            float(tn / (tn + fn)) if tn + fn > 0 else np.nan)


def _calibration(y, p):
    """Brier score, plus calibration intercept and slope on the logit scale.

    A slope below 1 means the probabilities are too extreme for the data, which is the usual
    consequence of fitting and scoring the same sample size. An intercept away from 0 means the
    average predicted risk does not match the observed rate.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    lo = np.log(p / (1 - p))[:, None]
    try:
        slope = float(LogisticRegression(penalty=None, max_iter=1000).fit(lo, y).coef_[0][0])
        icpt = float(LogisticRegression(penalty=None, max_iter=1000)
                     .fit(np.zeros_like(lo), y).intercept_[0] - np.mean(lo))
    except Exception:
        slope, icpt = np.nan, np.nan
    return {"brier": brier, "calibration_slope": slope, "calibration_intercept": icpt}


def _metrics(y, prob, label):
    """Every operating-point quantity for one score, at one fixed specificity."""
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    prob = np.asarray(prob, float)
    thr = float(np.quantile(prob[y == 0], SPEC_TARGET))
    flag = prob > thr
    sens = float(flag[y == 1].mean())
    spec = float((~flag[y == 0]).mean())
    # Likelihood ratios do not depend on prevalence, so they are the part of an operating point
    # that transports between a clinic and a screening service; predictive values are the part
    # that does not. Reporting both is what lets a reader move the result to their own setting.
    lr_pos = float(sens / (1 - spec)) if spec < 1 else float("inf")
    lr_neg = float((1 - sens) / spec) if spec > 0 else float("inf")
    rec = {"arm": label, "auc": ev.auc(y, prob), "auc_ci": list(ev.bootstrap_ci(y, prob)),
           "threshold": thr,
           "sensitivity": sens, "specificity": spec,
           "balanced_accuracy": float((sens + spec) / 2),
           "youden_j": float(sens + spec - 1),
           "lr_positive": lr_pos, "lr_negative": lr_neg,
           "average_precision": float(average_precision_score(y, prob)),
           "sample_prevalence": float(y.mean()),
           "predictive_values": {}}
    rec.update(_calibration(y, prob))
    for prev in PREVALENCES:
        ppv, npv = _ppv_npv(sens, spec, prev)
        # Referrals generated per 1,000 assessed, which is the capacity question a service asks.
        flagged = sens * prev + (1 - spec) * (1 - prev)
        # F1 and the number needed to screen are both prevalence-dependent, so they are computed
        # inside this loop rather than reported once for the sample's own case mix.
        f1 = float(2 * ppv * sens / (ppv + sens)) if (ppv + sens) > 0 else 0.0
        nns = float(1.0 / (sens * prev)) if sens * prev > 0 else float("inf")
        rec["predictive_values"][f"{prev:.2f}"] = {
            "ppv": ppv, "npv": npv, "f1": f1,
            "number_needed_to_screen": nns,
            "flagged_per_1000": float(1000 * flagged),
            "true_cases_among_flagged_per_1000": float(1000 * sens * prev)}
    return rec


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT, "spec_target": SPEC_TARGET, "prevalences": PREVALENCES,
           "tasks": {}}

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        rec = {"n": int(len(y)), "budgets": {}}
        print(f"\n=== {task}: n={len(y)}, sample prevalence {y.mean():.2f}")

        for tag, key in BUDGETS.items():
            if key not in keys:
                continue
            X = d["X"][keys.index(key)][ix]
            if not np.isfinite(X).any():
                continue
            arms = {}
            p_age, _ = ev.cv_probabilities(age[:, None], y)
            p_eeg, _ = ev.cv_probabilities(X, y)
            p_both, _ = ev.cv_probabilities(np.column_stack([X, age]), y)
            # The proposed remedy: rather than adding age as a predictor, score each feature as a
            # deviation from what an age-conditioned reference fitted on controls expects at that
            # age, and classify the deviations. The age relationship is then a property of the
            # reference rather than a fitted coefficient, so it does not have to be re-learned or
            # carried when the population's age structure changes.
            s_norm, _ = ev.cv_normative_scores(X, y, age)
            p_norm = _rank_to_prob(s_norm)
            for label, p in (("age", p_age), ("eeg", p_eeg), ("eeg+age", p_both),
                             ("normative", p_norm)):
                arms[label] = _metrics(y, p, label)
            rec["budgets"][tag] = arms

            print(f"  {tag}  {'arm':10s}{'AUC':>7s}{'sens':>7s}{'spec':>7s}{'LR+':>6s}"
                  f"{'LR-':>6s}{'AP':>7s}{'Brier':>8s}{'PPV@10%':>9s}{'NNS@10%':>9s}")
            for label, a in arms.items():
                pv = a["predictive_values"]["0.10"]
                print(f"       {label:10s}{a['auc']:7.3f}{a['sensitivity']:7.2f}"
                      f"{a['specificity']:7.2f}{a['lr_positive']:6.2f}{a['lr_negative']:6.2f}"
                      f"{a['average_precision']:7.3f}{a['brier']:8.3f}"
                      f"{pv['ppv']:9.2f}{pv['number_needed_to_screen']:9.1f}")
        out["tasks"][task] = rec

    # ---- external cohort -----------------------------------------------------------------
    # The Greek cohort is the only external data here and it inverts the age structure: its cases
    # are younger than its controls, so a model that leans on age transports badly while one that
    # does not can still rank. Both are reported.
    tgt = df[(df["cohort"] == TARGET) & df["age"].notna()
             & (df["in_screening"] == 1)].reset_index(drop=True)
    if len(tgt):
        src = base[base["label"].isin(TASKS["dementia"] | {"CN"})].reset_index(drop=True)
        si = np.array([sub_index[s] for s in src["subject"]])
        ti = np.array([sub_index[s] for s in tgt["subject"]])
        ys = src["label"].isin(TASKS["dementia"]).to_numpy().astype(int)
        yt = tgt["impaired"].to_numpy().astype(int)
        as_, at = src["age"].to_numpy(float), tgt["age"].to_numpy(float)

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        ext = {"cohort": TARGET, "n": int(len(yt)),
               "n_case": int(yt.sum()), "n_control": int((yt == 0).sum()),
               "age_gap": float(at[yt == 1].mean() - at[yt == 0].mean()),
               "prevalence": float(yt.mean()), "budgets": {}}
        for tag, key in BUDGETS.items():
            if key not in keys:
                continue
            Xs, Xt = d["X"][keys.index(key)][si], d["X"][keys.index(key)][ti]
            keep = np.isfinite(Xs).all(0) & np.isfinite(Xt).all(0)
            if not keep.any():
                continue
            arms = {}
            for label, Fs, Ft in (("eeg", Xs[:, keep], Xt[:, keep]),
                                  ("eeg+age", np.column_stack([Xs[:, keep], as_]),
                                   np.column_stack([Xt[:, keep], at]))):
                base_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                model = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5, ensemble=False)
                model.fit(Fs, ys)
                arms[label] = _metrics(yt, model.predict_proba(Ft)[:, 1], label)
            # Age is scored two ways at the target: carrying the source's direction (older implies
            # impaired) and direction-free. The cohorts disagree on that direction, so quoting one
            # number would decide the comparison by assumption.
            # The same remedy carried across sites: the normative reference is fitted on source
            # controls only, then applied to target features at target ages. Nothing is refitted.
            ctrl_s = ys == 0
            model = nm.fit(Xs[:, keep][ctrl_s], as_[ctrl_s])
            Zs = nm.deviation(model, Xs[:, keep], as_)
            Zt = nm.deviation(model, Xt[:, keep], at)
            keep_z = np.isfinite(Zs).all(0) & np.isfinite(Zt).all(0)
            if keep_z.any():
                head = CalibratedClassifierCV(
                    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                    method="sigmoid", cv=5, ensemble=False)
                head.fit(Zs[:, keep_z], ys)
                arms["normative"] = _metrics(yt, head.predict_proba(Zt[:, keep_z])[:, 1],
                                             "normative")
            arms["age_source_direction"] = _metrics(yt, at, "age_source_direction")
            arms["age_direction_free"] = _metrics(yt, -at, "age_direction_free")
            ext["budgets"][tag] = arms

            print(f"\n=== external {TARGET} ({tag}): n={len(yt)}, "
                  f"age gap {ext['age_gap']:+.1f} y, prevalence {yt.mean():.2f}")
            for label, a in arms.items():
                print(f"  {label:22s} AUC {a['auc']:.3f}  sens {a['sensitivity']:.2f}  "
                      f"spec {a['specificity']:.2f}  AP {a['average_precision']:.3f}")
        out["external"] = ext

    path = paths.RESULTS / "e23_operating_metrics.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
