"""The external operating point, with the threshold carried from the source and nothing refit.

Why this exists. ``e23`` reports the external cohort at a threshold set to the 80th percentile of
the *target's own* controls. That is the right way to compare discrimination shapes, and it is the
wrong way to report a transfer: it uses target labels, so every arm comes out at specificity 0.79
by construction and the sensitivity column is an upper bound no clinic could realise. ``e04``
fixes the threshold on the source correctly, but it trains on the screening contrast and reads
specificity on the 14 controls left over from the anchoring comparison, so it cannot be quoted
beside the dementia-contrast AUCs the manuscript reports.

This experiment closes that gap and nothing else. One source contrast (dementia, matching the
manuscript), one threshold per arm fixed on source out-of-fold controls, applied once to the
target with no refitting and no target labels. Because no arm here anchors on target controls,
all 29 target controls are held out and specificity is read on every one of them -- granularity
1/29 = 0.034, against the 1/14 = 0.071 that ``e04``'s anchoring comparison forced.

The deployment procedure is the one a clinic would follow:

    threshold   80th percentile of source out-of-fold probabilities among source controls
    model       refit on all source subjects, sigmoid-calibrated on source cross-validation
    target      scored once; specificity and sensitivity read at the carried threshold

    python experiments/e24_external_operating_point.py
Out: outputs/results/e24_external_operating_point.json
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

SOURCE, TARGET = "CAUEEG", "ds004504_raw"
BUDGETS = {"b19": "b19|256|f32|full", "b4": "b4|256|f32|full"}
#: The manuscript's external test is the dementia contrast, so that is the primary source model.
#: The screening contrast is run alongside because an earlier experiment used it, and whether the
#: transferred threshold holds turns out to depend on which of the two trained the rule -- a
#: dependence worth measuring rather than choosing between.
CONTRASTS = {"dementia": {"AD", "FTD", "VAD"},
             "screening": {"AD", "FTD", "VAD", "MCI"}}
SPEC_TARGET = 0.80


def _fit_calibrated(X, y):
    """Source rule: standardise, L2 logistic at C=1, sigmoid-calibrated on source folds."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5, ensemble=False)
    model.fit(X, y)
    return model


def _oof_probabilities(X, y):
    """Source out-of-fold probabilities on the same estimator, for setting the threshold.

    Using in-sample probabilities from the deployed model would put the threshold at an optimistic
    quantile and hand the transferred arm a specificity it has not earned.
    """
    p, _ = ev.cv_probabilities(X, y)
    return p


def _read_at(threshold, p_target, y_target):
    """Sensitivity and specificity at a threshold that was decided before the target was seen."""
    flag = p_target > threshold
    sens = float(flag[y_target == 1].mean())
    spec = float((~flag[y_target == 0]).mean())
    n_ctrl = int((y_target == 0).sum())
    return {"threshold": float(threshold),
            "sensitivity": sens,
            "specificity": spec,
            "youden_j": float(sens + spec - 1),
            "n_target_controls": n_ctrl,
            "specificity_granularity": float(1.0 / n_ctrl) if n_ctrl else float("nan"),
            "spec_shortfall": float(SPEC_TARGET - spec)}


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()

    tgt = df[(df["cohort"] == TARGET) & df["age"].notna()
             & (df["in_screening"] == 1)].reset_index(drop=True)
    ti = np.array([sub_index[s] for s in tgt["subject"]])
    yt = tgt["impaired"].to_numpy().astype(int)
    at = tgt["age"].to_numpy(float)

    print(f"target {TARGET}: n={len(yt)} ({(yt == 0).sum()} control, {yt.sum()} case), "
          f"age gap {at[yt == 1].mean() - at[yt == 0].mean():+.1f} y\n")

    out = {"source": SOURCE, "target": TARGET,
           "spec_target": SPEC_TARGET,
           "n_target": int(len(yt)),
           "n_target_controls": int((yt == 0).sum()),
           "age_gap_target": float(at[yt == 1].mean() - at[yt == 0].mean()),
           "protocol": ("threshold = 80th percentile of source out-of-fold probabilities among "
                        "source controls; model refit on all source subjects; applied once to the "
                        "target with no refitting and no target labels"),
           "source_contrasts": {}}

    for contrast, positive in CONTRASTS.items():
        src = df[(df["cohort"] == SOURCE) & df["age"].notna() & (df["in_screening"] == 1)
               & df["label"].isin(positive | {"CN"})].reset_index(drop=True)
        si = np.array([sub_index[s] for s in src["subject"]])
        ys = src["label"].isin(positive).to_numpy().astype(int)
        as_ = src["age"].to_numpy(float)
        print(f"### source contrast {contrast}: n={len(ys)} "
              f"({(ys == 0).sum()} control, {ys.sum()} case)")
        out["source_contrasts"][contrast] = {
            "positive_labels": sorted(positive), "n_source": int(len(ys)),
            "n_source_control": int((ys == 0).sum()), "budgets": {}}
        budgets_out = out["source_contrasts"][contrast]["budgets"]

        for tag, key in BUDGETS.items():
          if key not in keys:
              continue
          Xs_all, Xt_all = d["X"][keys.index(key)][si], d["X"][keys.index(key)][ti]
          # A column finite at the source but not at the target would make every target prediction
          # NaN, so the head is only ever built from measurements the destination can supply.
          keep = np.isfinite(Xs_all).all(0) & np.isfinite(Xt_all).all(0)
          if not keep.any():
              continue
          Xs, Xt = Xs_all[:, keep], Xt_all[:, keep]

          arms = {}

          # --- the three feature arms -------------------------------------------------------
          designs = {
              "eeg": (Xs, Xt),
              "eeg+age": (np.column_stack([Xs, as_]), np.column_stack([Xt, at])),
          }
          for label, (Fs, Ft) in designs.items():
              thr = float(np.quantile(_oof_probabilities(Fs, ys)[ys == 0], SPEC_TARGET))
              model = _fit_calibrated(Fs, ys)
              p_t = model.predict_proba(Ft)[:, 1]
              rec = _read_at(thr, p_t, yt)
              rec["auc"] = ev.auc(yt, p_t)
              rec["auc_ci"] = list(ev.bootstrap_ci(yt, p_t))
              rec["n_features"] = int(Fs.shape[1])
              arms[label] = rec

          # --- the remedy: an age-conditioned reference fitted on source controls only --------
          ref = nm.fit(Xs[ys == 0], as_[ys == 0])
          Zs, Zt = nm.deviation(ref, Xs, as_), nm.deviation(ref, Xt, at)
          keep_z = np.isfinite(Zs).all(0) & np.isfinite(Zt).all(0)
          if keep_z.any():
              thr = float(np.quantile(_oof_probabilities(Zs[:, keep_z], ys)[ys == 0], SPEC_TARGET))
              model = _fit_calibrated(Zs[:, keep_z], ys)
              p_t = model.predict_proba(Zt[:, keep_z])[:, 1]
              rec = _read_at(thr, p_t, yt)
              rec["auc"] = ev.auc(yt, p_t)
              rec["auc_ci"] = list(ev.bootstrap_ci(yt, p_t))
              rec["n_features"] = int(keep_z.sum())
              arms["normative"] = rec

          # --- age, carried with the direction the source learned ----------------------------
          # There is no honest direction-free operating point: choosing the direction that flatters
          # the target needs target labels, which is the thing this experiment exists to avoid. The
          # transported rule is the only one a clinic could actually run, so it is the one reported.
          thr = float(np.quantile(_oof_probabilities(as_[:, None], ys)[ys == 0], SPEC_TARGET))
          model = _fit_calibrated(as_[:, None], ys)
          p_t = model.predict_proba(at[:, None])[:, 1]
          rec = _read_at(thr, p_t, yt)
          rec["auc"] = ev.auc(yt, p_t)
          rec["n_features"] = 1
          arms["age_transported"] = rec

          budgets_out[tag] = arms

          print(f"=== {tag}: threshold fixed on source, read on all {(yt == 0).sum()} "
                f"target controls")
          print(f"    {'arm':18s}{'AUC':>7s}{'sens':>7s}{'spec':>7s}{'shortfall':>11s}")
          for label, a in arms.items():
              print(f"    {label:18s}{a['auc']:7.3f}{a['sensitivity']:7.2f}"
                    f"{a['specificity']:7.2f}{a['spec_shortfall']:+11.2f}")
          print()

    dst = paths.RESULTS / "e24_external_operating_point.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
