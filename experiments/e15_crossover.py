"""The two-cohort crossover, with the two things the first version got wrong put right.

``e12`` reported a swing of +0.496 AUC in the age margin between the training cohort and the
external clinic. Two problems make that number larger than the evidence supports, and both are
fixed here rather than argued away.

**The contrast was not held fixed.** The within-cohort margin came from the screening contrast,
whose positives include mild cognitive impairment. The external site publishes no MCI: its
positives are Alzheimer's and frontotemporal dementia. So the swing mixed a change of cohort with
a change of clinical question -- and this paper's own first result is that the clinical question
moves the margin. Here both ends of the comparison use the dementia contrast.

**The external age baseline was read one-directionally.** Age scored 0.403 at the external clinic,
below chance, because its cases are *younger* than its controls and the rank AUC assumes the
opposite. A clinic that fitted age on its own patients would not score 0.403; it would score what
a fitted rule scores. Three baselines are therefore reported: the rule transported from the source
(older implies impaired), the sign-agnostic rank baseline max(a, 1-a), and age refitted through the
identical cross-validation pipeline at the target. The swing is quoted against the most
conservative of them, and the others are printed so the choice is visible.

    python experiments/e15_crossover.py
Out: outputs/results/e15_crossover.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

BUDGET = "b19|256|f32|full"
SOURCE, TARGET = "CAUEEG", "ds004504_raw"
#: Held fixed at both ends. The target publishes no MCI, so the dementia contrast is the only one
#: both sites can express, and mixing it with screening is what inflated the original swing.
CONTRAST = {"AD", "FTD", "VAD"}


def arrays(df, d, sub_index, cohort, pos):
    rows = df[(df["cohort"] == cohort) & df["age"].notna() & (df["in_screening"] == 1)]
    rows = rows[rows["label"].isin(pos | {"CN"})].reset_index(drop=True)
    ix = np.array([sub_index[s] for s in rows["subject"]])
    X = d["X"][[str(k) for k in d["keys"]].index(BUDGET)][ix]
    return X, rows["label"].isin(pos).to_numpy().astype(int), rows["age"].to_numpy(float)


def age_baselines(y, age):
    """Three defensible readings of what age alone is worth in a cohort."""
    a = ev.auc(y, age)
    s_cv, _ = ev.cv_scores(age.reshape(-1, 1), y)
    return {"transported": a,                       # older implies impaired, carried from source
            "sign_agnostic": float(max(a, 1 - a)),  # age is informative, direction unknown
            "refit_cv": ev.auc(y, s_cv)}            # age fitted on this cohort, cross-validated


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()

    Xs, ys, ages = arrays(df, d, sub_index, SOURCE, CONTRAST)
    Xt, yt, aget = arrays(df, d, sub_index, TARGET, CONTRAST)
    print(f"contrast held fixed at both ends: {sorted(CONTRAST)} vs CN")
    print(f"  source {SOURCE}: n={len(ys)} ({(ys==0).sum()} CN / {ys.sum()} case), "
          f"age gap {ages[ys==1].mean()-ages[ys==0].mean():+.1f} y")
    print(f"  target {TARGET}: n={len(yt)} ({(yt==0).sum()} CN / {yt.sum()} case), "
          f"age gap {aget[yt==1].mean()-aget[yt==0].mean():+.1f} y")

    # Source: out-of-fold. Target: fitted on the whole source, which is the deployment situation.
    # These are intentionally different objects: source OOF scores estimate generalisation,
    # whereas target scores use the deployable final source fit. Never describe this as identical
    # weights across the two reported score vectors.
    s_src, keep = ev.cv_scores(Xs, ys)
    usable = keep & np.isfinite(Xt).all(axis=0)
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    head.fit(Xs[:, usable], ys)
    s_tgt = head.decision_function(Xt[:, usable])

    bs, bt = age_baselines(ys, ages), age_baselines(yt, aget)
    print("\nage alone, read three ways")
    for k in bs:
        print(f"  {k:15s} source {bs[k]:.3f}   target {bt[k]:.3f}")

    auc_src, auc_tgt = ev.auc(ys, s_src), ev.auc(yt, s_tgt)
    m_src = ev.delong_test(s_src, ages, ys)

    out = {"budget": BUDGET, "contrast": sorted(CONTRAST),
           "weight_protocol": {
               "source": "5x5-fold out-of-fold scores",
               "target": "final model fit on all source subjects and applied once",
               "same_weights_claim_supported": False,
           },
           "source": {"cohort": SOURCE, "n": int(len(ys)), "auc_eeg": auc_src,
                      "age_gap": float(ages[ys == 1].mean() - ages[ys == 0].mean()),
                      "age_baselines": bs, "margin_transported": m_src["diff"],
                      "margin_ci": m_src["ci"], "margin_p": m_src["p"], "margin_se": m_src["se"]},
           "target": {"cohort": TARGET, "n": int(len(yt)), "auc_eeg": auc_tgt,
                      "age_gap": float(aget[yt == 1].mean() - aget[yt == 0].mean()),
                      "age_baselines": bt},
           "swings": {}}

    print(f"\nEEG: source {auc_src:.3f} (out-of-fold)   target {auc_tgt:.3f} (transported weights)")
    print("\nmargin and swing under each reading of the target's age baseline")
    for k in bt:
        # Paired at the target against the corresponding age score, so the interval is a paired one
        # wherever the baseline is an actual score vector.
        if k == "transported":
            t = ev.delong_test(s_tgt, aget, yt)
        elif k == "refit_cv":
            s_cv, _ = ev.cv_scores(aget.reshape(-1, 1), yt)
            t = ev.delong_test(s_tgt, s_cv, yt)
        else:
            flip = aget if ev.auc(yt, aget) >= 0.5 else -aget
            t = ev.delong_test(s_tgt, flip, yt)
        se = float(np.sqrt(m_src["se"] ** 2 + t["se"] ** 2))
        swing = t["diff"] - m_src["diff"]
        out["swings"][k] = {
            "target_margin": t["diff"], "target_margin_ci": t["ci"], "target_margin_p": t["p"],
            "swing": swing, "swing_ci": [swing - 1.96 * se, swing + 1.96 * se],
            "swing_p": float(2 * norm.sf(abs(swing / se))) if se > 0 else np.nan}
        v = out["swings"][k]
        print(f"  {k:15s} target margin {t['diff']:+.3f} "
              f"[{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}]   swing {swing:+.3f} "
              f"[{v['swing_ci'][0]:+.3f},{v['swing_ci'][1]:+.3f}]  p={v['swing_p']:.1e}")

    # The conservative headline: the largest defensible target baseline, hence the smallest swing.
    worst = min(out["swings"], key=lambda k: out["swings"][k]["swing"])
    out["headline"] = {"baseline_used": worst, **out["swings"][worst]}
    print(f"\nheadline uses the most conservative baseline ({worst}): "
          f"swing {out['headline']['swing']:+.3f} "
          f"[{out['headline']['swing_ci'][0]:+.3f},{out['headline']['swing_ci'][1]:+.3f}]")

    # Same question, incremental rather than substitution: does the recording still add at the
    # target once age is in the model? At the source it does; the target's reversed age gradient
    # is what makes this worth checking.
    s_both_src, _ = ev.cv_scores(np.column_stack([Xs, ages]), ys)
    head2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    head2.fit(np.column_stack([Xs[:, usable], ages]), ys)
    s_both_tgt = head2.decision_function(np.column_stack([Xt[:, usable], aget]))
    out["incremental"] = {
        "source": ev.delong_test(s_both_src, ages, ys),
        "target_transported_ageterm": ev.delong_test(s_both_tgt, aget, yt),
        "target_eeg_only_vs_ageplus": ev.delong_test(s_tgt, s_both_tgt, yt)}
    print(f"\nincremental at source  {out['incremental']['source']['diff']:+.3f} "
          f"(p={out['incremental']['source']['p']:.1e})")
    print(f"at target, carrying the fitted age term over: EEG+age "
          f"{ev.auc(yt, s_both_tgt):.3f} vs EEG alone {auc_tgt:.3f} -- "
          f"the age term learned at the source is worth "
          f"{ev.auc(yt, s_both_tgt) - auc_tgt:+.3f} where the age gradient reverses")

    (paths.RESULTS / "e15_crossover.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {paths.RESULTS / 'e15_crossover.json'}")


if __name__ == "__main__":
    main()
