"""What a low-cost cognitive-assessment benchmark looks like once age is on the table.

EasyCog is the newest public dataset for low-burden cognitive assessment: forehead and ear EEG,
three minutes of eyes-closed rest plus a passive-video block, and clinician-administered MoCA and
MMSE. Its published benchmark predicts those two scales and reports mean absolute error and
Pearson correlation on unseen subjects. It does not report what chronological age alone achieves
on the same task, and neither does any of the eleven baselines in its table.

That omission is this study's thesis, so this file tests it rather than asserting it. Four
analyses, in decreasing order of how much the paper leans on them:

**A. The age-baseline audit.** MoCA and MMSE regressed from age alone, from EEG alone, and from
both, under the benchmark's own evaluation design -- 10-fold cross-subject with folds balanced on
the benchmark's MoCA bands. The comparison against their published table is a comparison against a
protocol, not a head-to-head: this is the 69-participant public release and their table is the
101-participant cohort, so the claim available here is "age alone reaches numbers their published
methods do not", never "age beats method X by Y".

**B. The estimator-complexity boundary.** The same features under two estimators: all seventeen,
and a three-feature slowing index declared in advance in :mod:`eegbudget.easycog`. At this sample
size the difference between them is the difference between an increment and a deficit, which is a
deployment rule rather than a modelling detail.

**C. Placement.** CAUEEG says four temporal electrodes beat seven frontal ones. EasyCog is four
forehead electrodes plus twelve around-the-ear ones, so the prediction is testable on real
hardware. Run under both referencing schemes, because re-referencing within a tight four-electrode
forehead block removes most of what that block measures and would manufacture the answer.

**D. Recruitment.** EasyCog records the setting each participant was recruited in. Nursing-home
participants are older and more impaired than outpatients, which is the same confound the CAUEEG
control-composition experiment isolates, in a second cohort with a different mechanism.

    python experiments/e19_easycog_audit.py
Out: outputs/results/e19_easycog_audit.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import easycog as ec  # noqa: E402
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths  # noqa: E402
from eegbudget.features import NAMES  # noqa: E402

ROOT = Path(os.environ.get("EASYCOG_ROOT", str(ec.DEFAULT_ROOT)))
CACHE = paths.CACHE / "easycog_subjects.csv"

#: Benchmark design, copied from the EasyCog paper so the comparison is against their protocol.
FOLDS, REPEATS = 10, 10

#: Permutations behind the out-of-fold AUC null in the discrimination arm.
N_PERM_NULL = 200

#: Events per variable below which a fitted arm is reported as underpowered rather than as a
#: result.
#:
#: This is not a stylistic threshold. On the MoCA<19-against-normal contrast the seventeen-feature
#: model reaches an apparent AUC of 0.694 and an out-of-fold AUC of 0.115, no single feature
#: separates the classes by more than 0.10 AUC, and the out-of-fold AUC degrades monotonically as
#: regularisation is relaxed (0.343 at C=0.001 down to 0.115 at C=1) and falls further under
#: leave-one-out (0.085). Ten controls split five ways leaves eight controls to fit seventeen
#: coefficients: the fitted direction is essentially arbitrary and anti-generalises. Reporting a
#: margin from that configuration reports an artefact.
#:
#: Ten events per variable is the long-standing rule of thumb for logistic prediction models, and
#: it is the quantity the EasyCog arm actually turns on -- so it is computed, printed, and used to
#: gate what counts as a finding.
MIN_EPV = 10.0


def _epv(n_smaller_class, n_features):
    """Events per variable: the smaller class divided by the number of fitted coefficients."""
    return float(n_smaller_class) / max(1, int(n_features))

#: MoCA bands as the EasyCog paper defines them: >=26 normal, 19-25 MCI, <=18 moderate/severe.
#: Used both to stratify folds, as they do, and to define the balanced diagnostic contrasts.
MOCA_EDGES = [-1, 18.5, 25.5, 31]

#: Published EasyCog benchmark results (Hu et al. 2026, Table 5), n=101, 10-fold cross-subject.
#: The test column is the generalisation column -- last-epoch weights, early-stopped on a separate
#: unseen validation group -- and is the one comparable with a cross-validated baseline that
#: performs no model selection at all. The validation column is model-selected and is quoted only
#: so that the comparison is not silently made against the weaker of their two numbers.
PUBLISHED = {
    "source": "Hu et al. 2026, EasyCog dataset paper, Table 5 (n=101)",
    "test_best": {"method": "CogAssess", "moca_mae": 6.975, "moca_pcc": 0.213,
                  "mmse_mae": 5.659, "mmse_pcc": 0.310},
    "validation_best": {"method": "BL-G (eye tracking)", "moca_mae": 4.792, "moca_pcc": 0.596,
                        "mmse_mae": 4.238, "mmse_pcc": 0.559},
    "test_range_mae": {"moca": [6.183, 8.352], "mmse": [5.536, 7.360]},
    "test_range_pcc": {"moca": [-0.106, 0.218], "mmse": [-0.081, 0.310]},
    "caveat": ("The published table is the 101-participant cohort; this study analyses the "
               "69-participant public release. Protocols are matched (10-fold cross-subject, "
               "folds balanced on MoCA band) but the samples are not identical, so no "
               "method-versus-method margin is claimed."),
}


def _feature_matrix(T, montage, names=None):
    cols = [f"{montage}__{n}" for n in (names or NAMES)]
    return T[cols].to_numpy(float)


def _arms(T, montage_all, montage_ear, montage_fh):
    """Feature matrices for every estimator arm, keyed by a readable name."""
    age = T["age"].to_numpy(float)[:, None]
    return {
        "age": age,
        "eeg17_all": _feature_matrix(T, montage_all),
        "eeg17_all+age": np.column_stack([_feature_matrix(T, montage_all), age]),
        "eeg17_ear": _feature_matrix(T, montage_ear),
        "eeg17_ear+age": np.column_stack([_feature_matrix(T, montage_ear), age]),
        "eeg17_forehead": _feature_matrix(T, montage_fh),
        "eeg17_forehead+age": np.column_stack([_feature_matrix(T, montage_fh), age]),
        "slow3_all": _feature_matrix(T, montage_all, ec.SLOWING_TRIAD),
        "slow3_all+age": np.column_stack([_feature_matrix(T, montage_all, ec.SLOWING_TRIAD), age]),
        "slow3_ear": _feature_matrix(T, montage_ear, ec.SLOWING_TRIAD),
        "slow3_ear+age": np.column_stack([_feature_matrix(T, montage_ear, ec.SLOWING_TRIAD), age]),
        "slow3_forehead": _feature_matrix(T, montage_fh, ec.SLOWING_TRIAD),
        "slow3_forehead+age": np.column_stack([_feature_matrix(T, montage_fh, ec.SLOWING_TRIAD),
                                               age]),
    }


# -------------------------------------------------------------------------------------------
# A. the age-baseline audit
# -------------------------------------------------------------------------------------------
def regression_audit(T, band):
    out = {"design": {"folds": FOLDS, "repeats": REPEATS, "stratified_on": "MoCA band",
                      "n": int(len(T))},
           "published": PUBLISHED, "targets": {}}
    arms = _arms(T, "ec16", "ec_ear", "ec_forehead")

    for target in ("moca", "mmse"):
        y = T[target].to_numpy(float)
        rec = {"n": int(len(y)), "sd": float(y.std(ddof=1)), "arms": {}}

        # A model that ignores every predictor. Any arm not beating this is not measuring
        # anything, and on a small clinical cohort that is a live possibility rather than a
        # formality.
        mean_pred = np.empty(len(y))
        from sklearn.model_selection import StratifiedKFold
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(y[:, None], band):
            mean_pred[te] = y[tr].mean()
        rec["arms"]["mean_only"] = ev.regression_metrics(y, mean_pred)

        preds = {}
        for name, X in arms.items():
            pred, keep = ev.cv_regression_scores(X, y, strat=band, folds=FOLDS, repeats=REPEATS)
            preds[name] = pred
            rec["arms"][name] = ev.regression_metrics(y, pred)
            rec["arms"][name]["n_features"] = int(keep.sum())
            # For a continuous outcome the analogous quantity is subjects per coefficient. The
            # ridge penalty is chosen inside each training fold, so this degrades far more
            # gracefully than the unpenalised logistic fits above, but a seventeen-feature arm on
            # 69 subjects is still four subjects per coefficient and is flagged as such.
            rec["arms"][name]["subjects_per_variable"] = round(_epv(len(y), keep.sum()), 2)
            rec["arms"][name]["adequately_powered"] = bool(
                _epv(len(y), keep.sum()) >= MIN_EPV)

        # Every EEG arm is compared against age on identical subjects: the increment is the
        # quantity a clinic is buying, and separate intervals on two MAEs cannot supply it.
        rec["vs_age"] = {name: ev.paired_metric_diff(y, preds[name], preds["age"])
                         for name in preds if name != "age"}
        # Placement on the continuous outcome, paired on identical subjects. The dichotomised
        # contrasts have 10 subjects in their smaller class; this one has all 69 and does not
        # throw away the gradation the scale carries, so it is where the ear/forehead question is
        # actually decidable.
        rec["placement_regression"] = {
            est: ev.paired_metric_diff(y, preds[f"{est}_ear"], preds[f"{est}_forehead"])
            for est in ("eeg17", "slow3")}
        out["targets"][target] = rec
    return out


# -------------------------------------------------------------------------------------------
# B/C. discrimination, estimator complexity, placement
# -------------------------------------------------------------------------------------------
def _contrast(T, name):
    """Return (mask, y, description) for one diagnostic contrast."""
    dis = T["disease"].to_numpy(str)
    moca = T["moca"].to_numpy(float)
    if name == "ad_vd_vs_control":
        m = np.isin(dis, ["AD", "VD", "VaD", "VAD", "Control"])
        return m, np.isin(dis[m], ["AD", "VD", "VaD", "VAD"]).astype(int), \
            "AD or vascular dementia against controls; PD and neurosyphilis excluded"
    if name == "moca_severe_vs_normal":
        m = (moca < 19) | (moca >= 26)
        return m, (moca[m] < 19).astype(int), \
            "moderate/severe impairment (MoCA<19) against normal cognition (MoCA>=26)"
    if name == "moca_severe_vs_rest":
        m = np.ones(len(T), bool)
        return m, (moca < 19).astype(int), \
            "moderate/severe impairment (MoCA<19) against everyone else"
    raise ValueError(name)


def discrimination(T, contrast):
    mask, y, desc = _contrast(T, contrast)
    S = T.loc[mask].reset_index(drop=True)
    age = S["age"].to_numpy(float)
    counts = np.bincount(y, minlength=2)
    rec = {"description": desc, "n": int(len(y)),
           "n_control": int(counts[0]), "n_case": int(counts[1]),
           "age_gap": float(age[y == 1].mean() - age[y == 0].mean()),
           "auc_age": ev.auc(y, age), "estimators": {}}
    if counts.min() < 5:
        rec["status"] = "insufficient_class_count"
        return rec

    kw = dict(folds=int(min(5, counts.min())), repeats=REPEATS, seed=19)
    for est, names in (("eeg17", NAMES), ("slow3", ec.SLOWING_TRIAD)):
        X = _feature_matrix(S, "ec16", names)
        s_eeg, keep = ev.cv_scores(X, y, **kw)
        s_both, _ = ev.cv_scores(np.column_stack([X, age]), y, **kw)
        sub = ev.delong_test(s_eeg, age, y)
        inc = ev.delong_test(s_both, age, y)
        # Where the cross-validated AUC lands under permuted labels. If that null sits below 0.5
        # then a sub-chance observed AUC is the estimator failing at this sample size, not the
        # recording carrying inverted information, and the two must not be confused.
        rng = np.random.default_rng(19)
        null = np.empty(N_PERM_NULL)
        for i in range(N_PERM_NULL):
            yp = y[rng.permutation(len(y))]
            sp, _ = ev.cv_scores(X, yp, **kw)
            null[i] = ev.auc(yp, sp)
        epv = _epv(counts.min(), keep.sum())
        rec["estimators"][est] = {
            "n_features": int(keep.sum()),
            "events_per_variable": round(epv, 2),
            "adequately_powered": bool(epv >= MIN_EPV),
            "auc_eeg": ev.auc(y, s_eeg), "auc_eeg_plus_age": ev.auc(y, s_both),
            "auc_eeg_ci": list(ev.bootstrap_ci(y, s_eeg, n=2000, seed=1902)),
            "sub_margin": sub["diff"], "sub_ci": sub["ci"], "sub_p": sub["p"],
            "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"],
            "null_auc_mean": float(null.mean()), "null_auc_sd": float(null.std()),
            "null_auc_q": [float(q) for q in np.percentile(null, [2.5, 97.5])],
            "p_perm_auc": float((1 + (null >= ev.auc(y, s_eeg)).sum()) / (N_PERM_NULL + 1))}
    return rec


def placement(T, contrast):
    """Ear against forehead on identical subjects, under both referencing schemes."""
    mask, y, desc = _contrast(T, contrast)
    S = T.loc[mask].reset_index(drop=True)
    counts = np.bincount(y, minlength=2)
    out = {"description": desc, "n": int(len(y)), "arms": {}}
    if counts.min() < 5:
        out["status"] = "insufficient_class_count"
        return out

    for ref, (m_ear, m_fh) in (("native", ("ec_ear", "ec_forehead")),
                               ("subset_car", ("ec_ear_car", "ec_forehead_car"))):
        # A subject whose forehead block is degenerate cannot be scored by the forehead arm, and
        # scoring the ear arm on subjects the forehead arm never saw is not a paired comparison.
        ok = (S[f"usable_{m_ear}"] & S[f"usable_{m_fh}"]).to_numpy()
        yy = y[ok]
        if np.bincount(yy, minlength=2).min() < 5:
            out["arms"][ref] = {"status": "insufficient after excluding unusable blocks"}
            continue
        kw = dict(folds=int(min(5, np.bincount(yy).min())), repeats=REPEATS, seed=19)
        s_ear, _ = ev.cv_scores(_feature_matrix(S.loc[ok], m_ear), yy, **kw)
        s_fh, _ = ev.cv_scores(_feature_matrix(S.loc[ok], m_fh), yy, **kw)
        t = ev.delong_test(s_ear, s_fh, yy)
        out["arms"][ref] = {
            "n": int(ok.sum()), "n_excluded_unusable": int((~ok).sum()),
            "n_electrodes_ear": len(ec.MONTAGES[m_ear]["channels"]),
            "n_electrodes_forehead": len(ec.MONTAGES[m_fh]["channels"]),
            "auc_ear": ev.auc(yy, s_ear), "auc_forehead": ev.auc(yy, s_fh),
            "diff": t["diff"], "ci": t["ci"], "p": t["p"]}
    native = out["arms"].get("native", {})
    car = out["arms"].get("subset_car", {})
    if "diff" in native and "diff" in car:
        out["referencing_dependence"] = {
            "native_diff": native["diff"], "subset_car_diff": car["diff"],
            "agree_in_sign": bool(np.sign(native["diff"]) == np.sign(car["diff"])),
            "note": ("A placement claim that holds under one referencing and not the other is a "
                     "claim about the referencing. Re-referencing within the four-electrode "
                     "forehead block removes most of what those four closely spaced electrodes "
                     "share -- the same cancellation that makes a symmetric electrode pair blind "
                     "to a symmetric rhythm on the CAUEEG ladder -- while the same operation on "
                     "twelve spatially spread ear electrodes removes the shared physical "
                     "reference and helps. Only a difference present under the recording's own "
                     "referencing is evidence about where electrodes sit.")}
    return out


# -------------------------------------------------------------------------------------------
# D. recruitment
# -------------------------------------------------------------------------------------------
def recruitment(T):
    """The baseline and the increment within recruitment strata.

    EasyCog's recruitment variable is the setting, not a diagnosis, so this is the same
    manipulation as the CAUEEG control-composition experiment reached by a different route. It is
    underpowered at 69 subjects and is reported as a directional replication, not a claim.
    """
    env = T["environment"].to_numpy(str)
    moca = T["moca"].to_numpy(float)
    strata = {"all_settings": np.ones(len(T), bool),
              "inpatient_only": env == "Inpatient",
              "hospital_only": np.isin(env, ["Inpatient", "Outpatient"]),
              "nursing_home_only": np.char.startswith(env, "Nursing")}
    out = {"description": ("MoCA<19 against the rest, within recruitment settings. Nursing-home "
                           "participants are both older and more impaired, so the age baseline "
                           "and the increment are expected to move with the setting."),
           "by_setting": {}, "composition": {}}
    for name, m in strata.items():
        if m.sum() < 20:
            out["by_setting"][name] = {"status": "too few subjects", "n": int(m.sum())}
            continue
        S = T.loc[m].reset_index(drop=True)
        y = (moca[m] < 19).astype(int)
        age = S["age"].to_numpy(float)
        counts = np.bincount(y, minlength=2)
        if counts.min() < 5:
            out["by_setting"][name] = {"status": "class too small", "n": int(m.sum())}
            continue
        X = _feature_matrix(S, "ec_ear", ec.SLOWING_TRIAD)
        kw = dict(folds=int(min(5, counts.min())), repeats=REPEATS, seed=19)
        s_eeg, _ = ev.cv_scores(X, y, **kw)
        s_both, _ = ev.cv_scores(np.column_stack([X, age]), y, **kw)
        inc = ev.delong_test(s_both, age, y)
        out["by_setting"][name] = {
            "n": int(len(y)), "n_control": int(counts[0]), "n_case": int(counts[1]),
            "mean_age": float(age.mean()), "mean_moca": float(S["moca"].mean()),
            "auc_age": ev.auc(y, age), "auc_eeg": ev.auc(y, s_eeg),
            "auc_eeg_plus_age": ev.auc(y, s_both),
            "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"]}
    for name, m in strata.items():
        if m.sum():
            out["composition"][name] = {
                "n": int(m.sum()), "mean_age": float(T.loc[m, "age"].mean()),
                "mean_moca": float(T.loc[m, "moca"].mean()),
                "share_severe": float((T.loc[m, "moca"] < 19).mean())}
    return out


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")

    T, info = ec.load_resting_subjects(ROOT)
    T.to_csv(CACHE, index=False)
    band = pd.cut(T["moca"], MOCA_EDGES, labels=[0, 1, 2]).astype(int).to_numpy()

    print(f"EasyCog: {len(T)} participants, {info['resting_sessions_loaded']} resting sessions, "
          f"median {info['median_windows_per_session']:.0f} clean windows/session")
    bv = info["block_verification"]
    print(f"block verification: consistent={bv['consistent']} over {bv['n_used']} recordings")
    for k, v in bv["blocks"].items():
        print(f"   {k:10s} within {v['mean_within']:.3f}  between {v['mean_between']:.3f}")
    print(f"diagnoses: {T['disease'].value_counts().to_dict()}")
    print(f"MoCA bands (severe/MCI/normal): {np.bincount(band).tolist()}")

    out = {"dataset": "EasyCog", "manifest": info,
           "moca_bands": {"edges": MOCA_EDGES,
                          "counts": {k: int(v) for k, v in
                                     zip(["severe", "mci", "normal"], np.bincount(band))}}}

    print("\n=== A. cognitive-score regression, benchmark protocol ===")
    out["regression_audit"] = regression_audit(T, band)
    for target, rec in out["regression_audit"]["targets"].items():
        print(f"\n  {target.upper()}  (sd {rec['sd']:.2f})")
        print(f"    {'arm':18s}{'MAE':>7s}{'[95% CI]':>16s}{'PCC':>8s}{'[95% CI]':>16s}")
        for name, r in rec["arms"].items():
            mark = "" if r.get("adequately_powered", True) else "  (low SPV)"
            print(f"    {name:20s}{r['mae']:7.3f}  [{r['mae_ci'][0]:5.2f},{r['mae_ci'][1]:5.2f}]"
                  f"{r['pcc']:8.3f}  [{r['pcc_ci'][0]:5.2f},{r['pcc_ci'][1]:5.2f}]{mark}")
        for est, d in rec["placement_regression"].items():
            print(f"    placement {est}: ear - forehead  dPCC {d['d_pcc']:+.3f} "
                  f"[{d['d_pcc_ci'][0]:+.3f},{d['d_pcc_ci'][1]:+.3f}] p={d['p_pcc']:.3f}   "
                  f"dMAE {d['d_mae']:+.3f} [{d['d_mae_ci'][0]:+.3f},{d['d_mae_ci'][1]:+.3f}] "
                  f"p={d['p_mae']:.3f}")
        pub = PUBLISHED["test_best"]
        print(f"    {'-- published best (test, n=101)':<18s} MAE {pub[f'{target}_mae']:.3f}  "
              f"PCC {pub[f'{target}_pcc']:.3f}   [{PUBLISHED['test_best']['method']}]")

    print("\n=== B. discrimination and estimator complexity ===")
    out["discrimination"] = {}
    for c in ("ad_vd_vs_control", "moca_severe_vs_normal", "moca_severe_vs_rest"):
        rec = discrimination(T, c)
        out["discrimination"][c] = rec
        print(f"\n  {c}: n={rec['n']} ({rec['n_control']}/{rec['n_case']}), "
              f"age gap {rec['age_gap']:+.1f} y, age AUC {rec['auc_age']:.3f}")
        for est, r in rec.get("estimators", {}).items():
            mark = "" if r["adequately_powered"] else "   [UNDERPOWERED, not a finding]"
            print(f"    {est:7s} ({r['n_features']:2d} feat, EPV {r['events_per_variable']:.1f})"
                  f"{mark}")
            print(f"    {'':7s} EEG {r['auc_eeg']:.3f}  "
                  f"EEG+age {r['auc_eeg_plus_age']:.3f}  "
                  f"sub {r['sub_margin']:+.3f} p={r['sub_p']:.3f}  "
                  f"inc {r['inc_margin']:+.3f} [{r['inc_ci'][0]:+.3f},{r['inc_ci'][1]:+.3f}] "
                  f"p={r['inc_p']:.3f}")
            print(f"            permuted-label OOF AUC null: "
                  f"{r['null_auc_mean']:.3f} +- {r['null_auc_sd']:.3f} "
                  f"[{r['null_auc_q'][0]:.3f},{r['null_auc_q'][1]:.3f}]  "
                  f"p_perm={r['p_perm_auc']:.3f}")

    print("\n=== C. placement: ear against forehead ===")
    out["placement"] = {}
    for c in ("moca_severe_vs_rest", "moca_severe_vs_normal", "ad_vd_vs_control"):
        rec = placement(T, c)
        out["placement"][c] = rec
        print(f"\n  {c}")
        for ref, r in rec.get("arms", {}).items():
            if "status" in r:
                print(f"    {ref:11s} {r['status']}")
                continue
            print(f"    {ref:11s} ear(12) {r['auc_ear']:.3f}  forehead(4) {r['auc_forehead']:.3f}"
                  f"  diff {r['diff']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.3f}"
                  f"  (n={r['n']}, {r['n_excluded_unusable']} excluded)")

    print("\n=== D. recruitment setting ===")
    out["recruitment"] = recruitment(T)
    for name, r in out["recruitment"]["by_setting"].items():
        if "status" in r:
            print(f"  {name:20s} {r['status']} (n={r.get('n', 0)})")
            continue
        print(f"  {name:20s} n={r['n']:3d} ({r['n_control']}/{r['n_case']})  "
              f"age {r['mean_age']:.1f}  MoCA {r['mean_moca']:.1f}  "
              f"age AUC {r['auc_age']:.3f}  EEG {r['auc_eeg']:.3f}  "
              f"inc {r['inc_margin']:+.3f} p={r['inc_p']:.3f}")

    path = paths.RESULTS / "e19_easycog_audit.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
