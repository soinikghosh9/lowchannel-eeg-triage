"""What the recording *adds* to age, as distinct from whether it *beats* age.

The study's headline quantity has so far been the substitution margin,
``AUC(EEG) - AUC(age)``: does the recording outperform the birth certificate used alone. That is
a real question, and its answer is largely negative. But it is not the question the title asks.
A clinic never has to choose between the recording and the date of birth -- it holds the date of
birth already and is deciding whether to buy the recording. The quantity that decides that is
incremental:

    Delta_inc(b) = AUC(EEG + age, b) - AUC(age)

and it can be positive while the substitution margin is negative, which is exactly what happens
here. Reporting only the substitution margin understates the recording; reporting only the
incremental margin lets a screener take credit for demographics it did not measure. Both are
reported, on identical subjects, with the same paired test.

Three further things this file fixes about the inference:

* **Multiplicity.** The primary family is declared up front -- three clinical contrasts crossed
  with the two margins at the pre-specified full montage -- and Holm-adjusted. Everything else is
  exploratory and is written out without a claim attached.
* **Fold noise.** 5x5 cross-validation is averaged before scoring elsewhere, which hides how much
  of a 0.02 margin is the fold assignment. Each repeat is scored separately and the spread
  reported.
* **Placement.** The paper claims four temporal electrodes beat seven frontal ones. That is a
  difference between two correlated AUCs on identical subjects and had no test; it gets one.

    python experiments/e13_incremental.py
Out: outputs/results/e13_incremental.json
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
FULL, FOUR = "b19|256|f32|full", "b4|256|f32|full"
LADDER = ["b19", "b8", "b4", "b2", "b1_ap"]
REGIONS = ["r_frontal", "r_temporal", "r_central", "r_posterior"]
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}

#: The pre-specified primary family: both margins, three contrasts, at the full clinical montage.
#: Six tests, Holm-adjusted together. Everything else in the sweep is exploratory.
PRIMARY_BUDGET = FULL

#: Permutations behind the primary family's distribution-free p-values. DeLong's variance assumes
#: a fixed scoring rule; ours is refit in every fold and averaged over repeats, so its p-values are
#: anti-conservative by an unknown amount. Disclosing that is not the same as fixing it, and the
#: primary family is six declared tests, so the null is built by rerunning the whole estimator on
#: permuted labels.
N_PERM = 200

#: Bootstrap draws behind the substitution margin's distribution-free interval.
N_BOOT = 4000

#: Deep-learning results are read from e11 rather than recomputed: they need the raw signal cache
#: and a GPU, and the point here is only that the substitution margin depends on the estimator
#: while the increment does not. Absent e11, the block is simply omitted.
DEEP_RESULT = "e11_deep_arm.json"


def margins(X, y, age):
    """Substitution and incremental margins for one feature matrix, with fold-level spread."""
    per_eeg, s_eeg, keep = ev.cv_scores_repeats(X, y)
    per_both, s_both, _ = ev.cv_scores_repeats(np.column_stack([X, age]), y)
    a_age = ev.auc(y, age)

    sub = ev.delong_test(s_eeg, age, y)
    inc = ev.delong_test(s_both, age, y)
    return {
        "n_features_used": int(keep.sum()),
        "auc_age": a_age,
        "auc_eeg": ev.auc(y, s_eeg),
        "auc_eeg_plus_age": ev.auc(y, s_both),
        # substitution: is the recording a better score than the calendar
        "sub_margin": sub["diff"], "sub_ci": sub["ci"], "sub_p": sub["p"],
        # incremental: does adding the recording to the calendar buy anything
        "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"],
        # how much of each margin is fold-assignment noise
        "sub_margin_per_repeat": [ev.auc(y, s) - a_age for s in per_eeg],
        "inc_margin_per_repeat": [ev.auc(y, s) - a_age for s in per_both],
    }


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT, "primary_budget": PRIMARY_BUDGET, "tasks": {}}
    primary_p, primary_key = [], []

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)

        rec = {"n": int(len(y)), "n_control": int((y == 0).sum()), "n_positive": int(y.sum()),
               "age_gap": float(age[y == 1].mean() - age[y == 0].mean()), "budgets": {}}
        print(f"\n=== {task}: n={len(y)}  age gap {rec['age_gap']:+.1f} y")
        print(f"{'budget':14s}{'age':>7s}{'EEG':>7s}{'EEG+age':>9s}"
              f"{'sub':>8s}{'p_sub':>9s}{'inc':>8s}{'p_inc':>9s}{'fold sd':>9s}")

        for m in LADDER + (REGIONS if task == "screening" else []):
            k = f"{m}|256|f32|full"
            if k not in keys:
                continue
            X = d["X"][keys.index(k)][ix]
            if not np.isfinite(X).any():
                continue
            r = margins(X, y, age)
            r["n_electrodes"] = len(ALL[m]["channels"])
            rec["budgets"][k] = r
            if k == PRIMARY_BUDGET:
                primary_p += [r["sub_p"], r["inc_p"]]
                primary_key += [f"{task}|sub", f"{task}|inc"]
            sd = float(np.std(r["inc_margin_per_repeat"]))
            print(f"{m:14s}{r['auc_age']:7.3f}{r['auc_eeg']:7.3f}{r['auc_eeg_plus_age']:9.3f}"
                  f"{r['sub_margin']:+8.3f}{r['sub_p']:9.1e}{r['inc_margin']:+8.3f}"
                  f"{r['inc_p']:9.1e}{sd:9.4f}")
        out["tasks"][task] = rec

    # -- Holm across the declared primary family
    adj = ev.holm(primary_p)
    out["primary_family"] = {k: {"p": float(p), "p_holm": float(a)}
                             for k, p, a in zip(primary_key, primary_p, adj)}

    # -- distribution-free checks on the same family.
    #
    # The two margins need different nulls and it is worth being explicit about why. For the
    # increment, the question is whether adding the recording to age buys anything, so the null is
    # "the recording is noise" and a permutation that keeps each label with its own age builds it
    # exactly. For substitution, the question is whether the margin is zero, and that same
    # permutation would put the null at 0.5 - AUC(age) -- about -0.29 here -- so a test against it
    # would be asking whether the recording beats nothing. That one gets a paired bootstrap over
    # subjects instead, which tests the margin against zero and assumes nothing about the scoring
    # rule either.
    print(f"\ndistribution-free checks ({N_PERM} permutations / {N_BOOT} bootstrap draws)",
          flush=True)
    alt_p, alt_key = [], []
    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        X = d["X"][keys.index(PRIMARY_BUDGET)][ix]

        r = ev.permutation_increment_p(X, y, age, n_perm=N_PERM)
        rec = out["primary_family"][f"{task}|inc"]
        rec["p_alt"] = r["p_perm"]
        rec["p_alt_method"] = "permutation, EEG-is-noise null"
        rec["p_alt_floor"] = r["p_perm_floor"]
        rec["perm_null"] = {"mean": r["null_mean"], "sd": r["null_sd"], "q": r["null_q"]}
        alt_p.append(r["p_perm"])
        alt_key.append(f"{task}|inc")
        print(f"  {task:10s} inc  margin {r['margin']:+.3f}  "
              f"null {r['null_mean']:+.3f} +-{r['null_sd']:.3f}  "
              f"p_perm={r['p_perm']:.4f} (floor {r['p_perm_floor']:.4f})", flush=True)

        s_eeg, _ = ev.cv_scores(X, y)
        b = ev.paired_bootstrap_diff(s_eeg, age, y, n=N_BOOT)
        rec = out["primary_family"][f"{task}|sub"]
        rec["p_alt"] = b["p_two_sided"]
        rec["p_alt_method"] = "paired bootstrap over subjects, null margin = 0"
        rec["bootstrap_ci"] = b["ci"]
        alt_p.append(b["p_two_sided"])
        alt_key.append(f"{task}|sub")
        print(f"  {task:10s} sub  margin {b['diff_mean']:+.3f}  "
              f"CI [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]  p_boot={b['p_two_sided']:.4f}",
              flush=True)

    for k, a in zip(alt_key, ev.holm(alt_p)):
        out["primary_family"][k]["p_alt_holm"] = float(a)

    print(f"\nprimary family ({len(primary_p)} tests at {PRIMARY_BUDGET})")
    print(f"  {'test':16s}{'DeLong p':>11s}{'Holm':>9s}{'alt p':>9s}{'alt Holm':>10s}  method")
    for k in primary_key:
        r = out["primary_family"][k]
        agree = "" if (r["p_holm"] < 0.05) == (r.get("p_alt_holm", 1.0) < 0.05) else "   DISAGREE"
        print(f"  {k:16s}{r['p']:11.2e}{r['p_holm']:9.4f}"
              f"{r.get('p_alt', float('nan')):9.4f}{r.get('p_alt_holm', float('nan')):10.4f}"
              f"  {r.get('p_alt_method', '')[:34]}{agree}")

    # -- placement: a claim about where electrodes sit needs a paired test, not two point estimates
    rows = base[base["label"].isin(TASKS["screening"] | {"CN"})].reset_index(drop=True)
    ix = np.array([sub_index[s] for s in rows["subject"]])
    y = rows["label"].isin(TASKS["screening"]).to_numpy().astype(int)
    scores = {}
    for m in REGIONS + ["b4"]:
        k = f"{m}|256|f32|full"
        if k in keys:
            s, _ = ev.cv_scores(d["X"][keys.index(k)][ix], y)
            scores[m] = s
    place = {}
    for a, b in [("r_temporal", "r_frontal"), ("r_temporal", "r_central"),
                 ("r_temporal", "r_posterior"), ("r_temporal", "b4")]:
        if a in scores and b in scores:
            t = ev.delong_test(scores[a], scores[b], y)
            place[f"{a}_vs_{b}"] = {"diff": t["diff"], "ci": t["ci"], "p": t["p"],
                                    "n_el_a": len(ALL[a]["channels"]),
                                    "n_el_b": len(ALL[b]["channels"])}
    padj = ev.holm([v["p"] for v in place.values()])
    for (k, v), a in zip(place.items(), padj):
        v["p_holm"] = float(a)
    out["placement"] = place
    print("\nplacement, paired on identical subjects (Holm-adjusted within this family)")
    for k, v in place.items():
        print(f"  {k:28s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
              f"p={v['p']:.1e}  Holm={v['p_holm']:.4f}")

    # -- estimator axis: the substitution margin is a property of the estimator, the increment is
    #    not. Stating that requires more than one estimator, and the strongest one already exists.
    deep_path = paths.RESULTS / DEEP_RESULT
    if deep_path.exists():
        deep = json.loads(deep_path.read_text())
        best = {}
        for name, r in deep.get("runs", {}).items():
            if r.get("variant") != "none" or r.get("montage") != "b19":
                continue          # the EEG-only arm, so it is comparable with the feature model
            t = r["task"]
            if t not in best or r["auc_within"] > best[t]["auc_eeg"]:
                best[t] = {"model": r["model"], "auc_eeg": r["auc_within"],
                           "auc_age": r["age_only_within"],
                           "sub_margin": r["age_margin_within"], "n": r.get("n")}
        out["estimator_axis"] = {
            "source": DEEP_RESULT,
            "note": ("Deep results are the best EEG-only architecture per contrast at the full "
                     "montage, trained under the same subject-level protocol. Read beside the "
                     "feature model, they show the substitution margin changing sign with the "
                     "estimator while the increment stays positive -- which is the reason the "
                     "study reports both margins rather than one."),
            "feature_model": {t: {"auc_eeg": out["tasks"][t]["budgets"][FULL]["auc_eeg"],
                                  "auc_age": out["tasks"][t]["budgets"][FULL]["auc_age"],
                                  "sub_margin": out["tasks"][t]["budgets"][FULL]["sub_margin"],
                                  "inc_margin": out["tasks"][t]["budgets"][FULL]["inc_margin"]}
                              for t in out["tasks"] if FULL in out["tasks"][t]["budgets"]},
            "deep_model": best}
        print("\nestimator axis at the full montage (EEG-only arms)")
        print(f"  {'contrast':12s}{'age':>7s}{'17 feat':>9s}{'deep':>9s}"
              f"{'sub(feat)':>11s}{'sub(deep)':>11s}{'inc(feat)':>11s}")
        for t, f in out["estimator_axis"]["feature_model"].items():
            b = best.get(t)
            print(f"  {t:12s}{f['auc_age']:7.3f}{f['auc_eeg']:9.3f}"
                  f"{(b['auc_eeg'] if b else float('nan')):9.3f}"
                  f"{f['sub_margin']:+11.3f}"
                  f"{(b['sub_margin'] if b else float('nan')):+11.3f}"
                  f"{f['inc_margin']:+11.3f}")
    else:
        print(f"\n{deep_path.name} absent -- estimator axis skipped")

    (paths.RESULTS / "e13_incremental.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {paths.RESULTS / 'e13_incremental.json'}")


if __name__ == "__main__":
    main()
