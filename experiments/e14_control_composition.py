"""The age baseline is set by who counts as a control -- shown inside one cohort.

The paper's second claim is that the demographic baseline a screener is measured against is a
property of recruitment rather than of method. So far that claim has rested on a crossover between
two cohorts, one of which contributes 88 subjects. A reviewer is right to say that two sites
differ in amplifier, montage, population and diagnostic workup all at once, so the crossover shows
*that* the baseline moves without isolating *what* moves it.

This experiment isolates one cause, inside a single cohort, with everything else held fixed --
same recordings, same pipeline, same features, same folds. The training cohort's control arm is
not healthy volunteers. Of its 459 controls, 186 are patients who attended the memory clinic with
subjective memory impairment and were found to have none: the worried well, who are referred
because something prompted them to come. Splitting the control arm on that single recruitment
variable moves the age gap, the age baseline and the margin, without changing a single recording.

Three control arms are compared:

    all       the 459 controls as the cohort ships them
    clean     the 273 controls with no memory complaint
    smi       the 186 subjective-memory-impairment controls alone

and, because 'clean' is also smaller, a size-matched control arm of 273 controls drawn at random
from all 459 separates composition from sample size.

    python experiments/e14_control_composition.py
Out: outputs/results/e14_control_composition.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

COHORT = "CAUEEG"
BUDGET = "b19|256|f32|full"
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}
N_SIZE_MATCHED = 30      # random draws for the size-matched control arm


def scored(X, y, age):
    """Exploratory retrained analysis for one case-control configuration."""
    s_eeg, _ = ev.cv_scores(X, y)
    s_both, _ = ev.cv_scores(np.column_stack([X, age]), y)
    sub = ev.delong_test(s_eeg, age, y)
    inc = ev.delong_test(s_both, age, y)
    return {"n": int(len(y)), "n_control": int((y == 0).sum()), "n_case": int(y.sum()),
            "age_gap": float(age[y == 1].mean() - age[y == 0].mean()),
            "age_control": float(age[y == 0].mean()), "age_case": float(age[y == 1].mean()),
            "auc_age": ev.auc(y, age), "auc_eeg": ev.auc(y, s_eeg),
            "auc_eeg_plus_age": ev.auc(y, s_both),
            "sub_margin": sub["diff"], "sub_ci": sub["ci"], "sub_p": sub["p"],
            "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"]}


def scored_fixed(scores_eeg, scores_both, y, age):
    """Score a recruitment-defined subset with one fixed out-of-fold scoring rule.

    This is the primary composition estimand: the recordings, labels used to fit the rule, and
    folds are held fixed while only the control definition changes. The retrained result remains
    available as a separate sensitivity analysis, but must not be described as the same model.
    """
    sub = ev.delong_test(scores_eeg, age, y)
    inc = ev.delong_test(scores_both, age, y)
    return {"n": int(len(y)), "n_control": int((y == 0).sum()), "n_case": int(y.sum()),
            "age_gap": float(age[y == 1].mean() - age[y == 0].mean()),
            "age_control": float(age[y == 0].mean()), "age_case": float(age[y == 1].mean()),
            "auc_age": ev.auc(y, age), "auc_eeg": ev.auc(y, scores_eeg),
            "auc_eeg_plus_age": ev.auc(y, scores_both),
            "sub_margin": sub["diff"], "sub_ci": sub["ci"], "sub_p": sub["p"],
            "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"]}


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna()
              & (df["in_screening"] == 1)].reset_index(drop=True)
    Xall = d["X"][keys.index(BUDGET)][np.array([sub_index[s] for s in base["subject"]])]

    is_ctrl = (base["label"] == "CN").to_numpy()
    is_smi = spine.has_subtype(base, "smi").to_numpy()
    ctrl_arms = {"all": is_ctrl,
                 "clean": is_ctrl & ~is_smi,
                 "smi": is_ctrl & is_smi}

    print(f"control arms: " + "  ".join(f"{k}={int(v.sum())}" for k, v in ctrl_arms.items()))
    age_all = base["age"].to_numpy(float)
    for k, v in ctrl_arms.items():
        print(f"  {k:6s} n={int(v.sum()):4d}  mean age {age_all[v].mean():.1f}")

    out = {"cohort": COHORT, "budget": BUDGET,
           "control_arms": {k: {"n": int(v.sum()), "age_mean": float(age_all[v].mean())}
                            for k, v in ctrl_arms.items()},
           "tasks": {}}

    for task, pos in TASKS.items():
        case = base["label"].isin(pos).to_numpy()
        rec = {"arms": {}, "retrained": {}}
        all_task = case | is_ctrl
        all_ix = np.flatnonzero(all_task)
        y_all = case[all_task].astype(int)
        s_fixed, _ = ev.cv_scores(Xall[all_task], y_all)
        s_fixed_both, _ = ev.cv_scores(np.column_stack([Xall[all_task], age_all[all_task]]), y_all)
        print(f"\n=== {task}")
        print(f"{'controls':10s}{'n':>6s}{'gap':>7s}{'age':>7s}{'EEG':>7s}"
              f"{'EEG+age':>9s}{'sub':>8s}{'p_sub':>9s}{'inc':>8s}{'p_inc':>9s}")

        for arm, ctrl in ctrl_arms.items():
            m = case | ctrl
            loc = np.flatnonzero(np.isin(all_ix, np.flatnonzero(m)))
            # `loc` is the subset of the fixed model's rows corresponding to this arm.
            r = scored_fixed(s_fixed[loc], s_fixed_both[loc], case[m].astype(int), age_all[m])
            rec["arms"][arm] = r
            rec["retrained"][arm] = scored(Xall[m], case[m].astype(int), age_all[m])
            print(f"{arm:10s}{r['n']:6d}{r['age_gap']:+7.1f}{r['auc_age']:7.3f}"
                  f"{r['auc_eeg']:7.3f}{r['auc_eeg_plus_age']:9.3f}{r['sub_margin']:+8.3f}"
                  f"{r['sub_p']:9.1e}{r['inc_margin']:+8.3f}{r['inc_p']:9.1e}")

        # Size-matched control arm: 'clean' is both cleaner and smaller, and a smaller control
        # group changes AUC on its own. Draw the same number of controls at random from all 459
        # so that only composition differs.
        n_clean = int(ctrl_arms["clean"].sum())
        ctrl_ix = np.where(is_ctrl)[0]
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(N_SIZE_MATCHED):
            pick = rng.choice(ctrl_ix, n_clean, replace=False)
            sel = np.zeros(len(base), bool)
            sel[pick] = True
            m = case | sel
            loc = np.flatnonzero(np.isin(all_ix, np.flatnonzero(m)))
            draws.append(scored_fixed(s_fixed[loc], s_fixed_both[loc],
                                      case[m].astype(int), age_all[m]))
        rec["size_matched"] = {
            "n_draws": N_SIZE_MATCHED, "n_control": n_clean,
            **{f"{k}_mean": float(np.mean([dd[k] for dd in draws]))
               for k in ("age_gap", "auc_age", "auc_eeg", "auc_eeg_plus_age",
                         "sub_margin", "inc_margin")},
            **{f"{k}_sd": float(np.std([dd[k] for dd in draws]))
               for k in ("auc_age", "sub_margin", "inc_margin")}}
        sm = rec["size_matched"]
        print(f"{'size-mat':10s}{'':6s}{sm['age_gap_mean']:+7.1f}{sm['auc_age_mean']:7.3f}"
              f"{sm['auc_eeg_mean']:7.3f}{sm['auc_eeg_plus_age_mean']:9.3f}"
              f"{sm['sub_margin_mean']:+8.3f}{'':9s}{sm['inc_margin_mean']:+8.3f}"
              f"   (+/-{sm['auc_age_sd']:.3f} on age over {N_SIZE_MATCHED} draws)")

        a, c = rec["arms"]["all"], rec["arms"]["clean"]
        rec["clean_minus_all"] = {
            "d_age_baseline": c["auc_age"] - a["auc_age"],
            "d_sub_margin": c["sub_margin"] - a["sub_margin"],
            "d_inc_margin": c["inc_margin"] - a["inc_margin"],
            "d_age_gap": c["age_gap"] - a["age_gap"],
            "size_matched_d_age_baseline": c["auc_age"] - sm["auc_age_mean"]}
        print(f"  clean - all:  age baseline {rec['clean_minus_all']['d_age_baseline']:+.3f} "
              f"(size-matched control {rec['clean_minus_all']['size_matched_d_age_baseline']:+.3f}), "
              f"sub {rec['clean_minus_all']['d_sub_margin']:+.3f}, "
              f"inc {rec['clean_minus_all']['d_inc_margin']:+.3f}")
        out["tasks"][task] = rec

    (paths.RESULTS / "e14_control_composition.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nwrote {paths.RESULTS / 'e14_control_composition.json'}")


if __name__ == "__main__":
    main()
