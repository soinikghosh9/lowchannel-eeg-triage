"""Who the screener misses, and how that changes with the acquisition budget.

A confound argument stated in AUC convinces methodologists. A subgroup failure stated in patients
convinces clinicians, and CAUEEG's diagnostic vocabulary supplies the subgroups without any new
labelling. Early-onset Alzheimer's is the decisive one: the disease is real and the patients are
seventeen years younger than the late-onset group, so an instrument leaning on age must miss them.

Scores are out-of-fold throughout, and the operating point is fixed at 80% specificity in the
control group, which is where a triage instrument would sit. Sensitivity is then read within each
subgroup with a Wilson interval, because at n=41 a normal approximation is not honest.

Two models are compared at each budget: the raw-feature classifier and the same features scored as
deviations from an age-conditioned normative model fitted on controls alone.

    python experiments/e03_subgroup_safety.py
Out: outputs/results/e03_subgroup_safety.json
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

COHORT = "CAUEEG"
SPEC = 0.80
BUDGETS = ["b19|256|f32|full", "b8|256|f32|full", "b4|256|f32|full", "b1_ap|256|f32|full"]
#: Impaired subgroups (sensitivity) and, separately, groups whose flagging is a false referral.
CASE_TAGS = ["eoad", "load", "ad", "vd", "sivd", "mci_amnestic", "mci_vascular"]
#: Flagging either of these is a false referral, for different reasons: subjective memory
#: impairment is the worried well, and transient global amnesia is amnestic but self-limiting and
#: not neurodegenerative. Both consume specialist capacity a low-resource system does not have.
CONTROL_TAGS = ["smi", "tga"]


def _score_heldout(X, y, X_ex, s_oof):
    """Score subjects excluded from the contrast on the same scale as the out-of-fold scores.

    ``cv_scores`` returns a rank fraction in [0, 1], so a raw decision value from a refitted model
    is not comparable with the threshold read off it. Fit once on the whole contrast, then place
    each held-out subject's decision value into the ranking of the contrast's own decision values
    and convert to the same fraction. This keeps the operating point meaningful for a group the
    threshold was never fitted to.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    head.fit(X, y)
    ref = head.decision_function(X)
    order = np.argsort(ref)
    ref_sorted, s_sorted = ref[order], np.sort(s_oof)
    return np.interp(head.decision_function(X_ex), ref_sorted, s_sorted)


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

    # Transient global amnesia belongs in neither arm of the contrast, so it is excluded from
    # training and from the threshold -- but the design says it is *scored* as a false-referral
    # subgroup, and an earlier version simply dropped it, leaving that claim unimplemented. These
    # subjects are held out entirely and scored by a model that never saw them.
    extra = df[(df["cohort"] == COHORT) & df["age"].notna()
               & (df["in_screening"] == 0)].reset_index(drop=True)
    ex_ix = np.array([sub_index[s] for s in extra["subject"]]) if len(extra) else None
    ex_age = extra["age"].to_numpy(float) if len(extra) else None

    out = {"cohort": COHORT, "specificity": SPEC, "budgets": {}}
    for k in BUDGETS:
        if k not in keys:
            continue
        X = d["X"][keys.index(k)][ix]
        if not np.isfinite(X).any():
            continue

        model = nm.fit(X[ctrl], age[ctrl])
        Z = nm.deviation(model, X, age)

        X_ex = d["X"][keys.index(k)][ex_ix] if ex_ix is not None else None
        Z_ex = nm.deviation(model, X_ex, ex_age) if X_ex is not None else None

        arms = {}
        for arm, feats, feats_ex in (("raw", X, X_ex), ("normative", Z, Z_ex)):
            if arm == "normative":
                # Fit the age-conditioned reference inside each fold; fitting it on all controls
                # before CV leaks the target-fold age distribution into the score.
                s, keep = ev.cv_normative_scores(X, y, age)
            else:
                s, keep = ev.cv_scores(feats, y)
            sens_all, thr = ev.sensitivity_at(y, s, SPEC)
            rec = {"auc": ev.auc(y, s), "sensitivity_overall": sens_all, "subgroups": {}}
            for tag in CASE_TAGS + CONTROL_TAGS:
                m = spine.has_subtype(rows, tag).to_numpy()
                if m.sum() < 5:
                    continue
                flagged = int((s[m] > thr).sum())
                lo, hi = ev.binom_ci(flagged, int(m.sum()))
                rec["subgroups"][tag] = {
                    "n": int(m.sum()), "age_mean": round(float(age[m].mean()), 1),
                    "flagged_rate": flagged / int(m.sum()), "ci": [lo, hi],
                    "kind": "sensitivity" if tag in CASE_TAGS else "false_referral"}
            # Held-out groups: fitted on the contrast, applied to subjects excluded from it. The
            # score scale must match the one the threshold was read on, so the held-out scores are
            # inserted into the out-of-fold ranking rather than ranked among themselves.
            if feats_ex is not None and len(feats_ex):
                s_ex = _score_heldout(feats[:, keep], y, feats_ex[:, keep], s)
                for tag in CONTROL_TAGS:
                    m = spine.has_subtype(extra, tag).to_numpy()
                    if m.sum() < 5:
                        continue
                    flagged = int((s_ex[m] > thr).sum())
                    lo, hi = ev.binom_ci(flagged, int(m.sum()))
                    rec["subgroups"][tag] = {
                        "n": int(m.sum()), "age_mean": round(float(ex_age[m].mean()), 1),
                        "flagged_rate": flagged / int(m.sum()), "ci": [lo, hi],
                        "kind": "false_referral", "held_out": True}
            arms[arm] = rec
        out["budgets"][k] = arms

        print(f"\n{k}")
        print(f"  {'subgroup':16s}{'n':>5s}{'age':>7s}{'raw':>8s}{'normative':>11s}")
        for tag in CASE_TAGS + CONTROL_TAGS:
            r = arms["raw"]["subgroups"].get(tag)
            if not r:
                continue
            n_ = arms["normative"]["subgroups"][tag]
            print(f"  {tag:16s}{r['n']:5d}{r['age_mean']:7.1f}{r['flagged_rate']:8.2f}"
                  f"{n_['flagged_rate']:11.2f}   ({r['kind']})")
        e_r = arms["raw"]["subgroups"].get("eoad")
        l_r = arms["raw"]["subgroups"].get("load")
        if e_r and l_r:
            print(f"  EOAD-LOAD sensitivity gap: raw {l_r['flagged_rate']-e_r['flagged_rate']:+.2f}, "
                  f"normative {arms['normative']['subgroups']['load']['flagged_rate']-arms['normative']['subgroups']['eoad']['flagged_rate']:+.2f}")

    (paths.RESULTS / "e03_subgroup_safety.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {paths.RESULTS / 'e03_subgroup_safety.json'}")


if __name__ == "__main__":
    main()
