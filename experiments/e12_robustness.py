"""Robustness checks on the two-cohort claim.

The headline is a crossover: age beats the instrument in one cohort and loses to it in another.
That is a strong claim resting on two cohorts, so it needs the obvious attacks run against it
before a reviewer runs them.

    1. Is the external reversal an artefact of frontotemporal dementia? FTD patients are the
       youngest group at the external site, so their presence could manufacture the reversal on
       its own. Rerun with FTD excluded.
    2. Is the age baseline advantaged by not being cross-validated? Age has no fitted parameters,
       so in-sample and out-of-sample coincide, but a reviewer will ask. Refit age through the
       identical cross-validation pipeline and confirm the number does not move.
    3. Is the crossover itself statistically distinguishable from noise? Bootstrap the difference
       of margins across the two cohorts, which is the quantity the paper actually claims.
    4. What do the age distributions look like? Report them rather than asserting "matched".

    python experiments/e12_robustness.py
Out: outputs/results/e12_robustness.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

BUDGET = "b19|256|f32|full"
SOURCE, TARGET = "CAUEEG", "ds004504_raw"


def cohort_arrays(df, d, sub_index, cohort, pos):
    rows = df[(df["cohort"] == cohort) & df["age"].notna() & (df["in_screening"] == 1)]
    rows = rows[rows["label"].isin(pos | {"CN"})].reset_index(drop=True)
    ix = np.array([sub_index[s] for s in rows["subject"]])
    X = d["X"][[str(k) for k in d["keys"]].index(BUDGET)][ix]
    y = rows["label"].isin(pos).to_numpy().astype(int)
    return X, y, rows["age"].to_numpy(float), rows


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    out = {"budget": BUDGET, "checks": {}}

    # ---- 4. age distributions, stated rather than asserted
    dist = {}
    for coh in (SOURCE, TARGET):
        r = df[(df["cohort"] == coh) & df["age"].notna() & (df["in_screening"] == 1)]
        dist[coh] = {lab: {"n": int(len(g)), "age_mean": round(float(g["age"].mean()), 1),
                           "age_sd": round(float(g["age"].std()), 1)}
                     for lab, g in r.groupby("label")}
    out["checks"]["age_distributions"] = dist
    print("age by label")
    for coh, v in dist.items():
        print(f"  {coh}")
        for lab, s in sorted(v.items()):
            print(f"    {lab:5s} n={s['n']:5d}  {s['age_mean']:.1f} +- {s['age_sd']:.1f}")

    # ---- 1. does excluding FTD remove the external reversal?
    print("\nexternal age baseline, with and without frontotemporal dementia")
    ftd = {}
    for name, pos in (("AD+FTD", {"AD", "FTD", "VAD"}), ("AD only", {"AD", "VAD"})):
        _, y, age, _ = cohort_arrays(df, d, sub_index, TARGET, pos)
        a = ev.auc(y, age)
        lo, hi = ev.bootstrap_ci(y, age, n=3000)
        ftd[name] = {"n": int(len(y)), "n_pos": int(y.sum()), "age_only_auc": a, "ci": [lo, hi]}
        print(f"  {name:8s} n={len(y):3d}  age AUC {a:.3f} [{lo:.3f},{hi:.3f}]")
    out["checks"]["ftd_exclusion"] = ftd

    # ---- 2. age through the identical CV pipeline
    print("\nage baseline, raw ranking versus refitted through the CV pipeline")
    par = {}
    for coh in (SOURCE, TARGET):
        _, y, age, _ = cohort_arrays(df, d, sub_index, coh, {"AD", "FTD", "MCI", "VAD"})
        s_cv, _ = ev.cv_scores(age.reshape(-1, 1), y)
        par[coh] = {"age_rank_auc": ev.auc(y, age), "age_cv_auc": ev.auc(y, s_cv)}
        print(f"  {coh:14s} rank {par[coh]['age_rank_auc']:.3f}   "
              f"cross-validated {par[coh]['age_cv_auc']:.3f}")
    out["checks"]["age_cv_parity"] = par

    # ---- 3. is the crossover itself significant?
    print("\ncrossover: margin (EEG - age) in each cohort, and their difference")
    pos = {"AD", "FTD", "MCI", "VAD"}
    Xs, ys, ages, _ = cohort_arrays(df, d, sub_index, SOURCE, pos)
    Xt, yt, aget, _ = cohort_arrays(df, d, sub_index, TARGET, pos)
    s_src, keep = ev.cv_scores(Xs, ys)

    # Target scores come from a model fitted on the whole source cohort, which is the deployment
    # situation: the receiving clinic contributes nothing to training.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    usable = keep & np.isfinite(Xt).all(axis=0)
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    head.fit(Xs[:, usable], ys)
    s_tgt = head.decision_function(Xt[:, usable])

    m_src = ev.delong_test(s_src, ages, ys)
    m_tgt = ev.delong_test(s_tgt, aget, yt)
    # The cohorts are independent samples, so the variance of the difference of margins is the
    # sum of their variances.
    se = float(np.sqrt(m_src.get("se", np.nan) ** 2 + m_tgt.get("se", np.nan) ** 2))
    swing = m_tgt["diff"] - m_src["diff"]
    from scipy.stats import norm
    out["checks"]["crossover"] = {
        "source": m_src, "target": m_tgt, "swing": swing,
        "swing_ci": [swing - 1.96 * se, swing + 1.96 * se],
        "swing_p": float(2 * norm.sf(abs(swing / se))) if se > 0 else np.nan,
        "n_features_used": int(usable.sum())}
    print(f"  {SOURCE:14s} margin {m_src['diff']:+.3f} "
          f"[{m_src['ci'][0]:+.3f},{m_src['ci'][1]:+.3f}]  p={m_src['p']:.2e}")
    print(f"  {TARGET:14s} margin {m_tgt['diff']:+.3f} "
          f"[{m_tgt['ci'][0]:+.3f},{m_tgt['ci'][1]:+.3f}]  p={m_tgt['p']:.2e}")
    print(f"  swing          {swing:+.3f} "
          f"[{swing-1.96*se:+.3f},{swing+1.96*se:+.3f}]  "
          f"p={out['checks']['crossover']['swing_p']:.2e}")

    (paths.RESULTS / "e12_robustness.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {paths.RESULTS / 'e12_robustness.json'}")


if __name__ == "__main__":
    main()
