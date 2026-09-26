"""Does adding the recording to age change who a clinic should refer?

Every number the study reports so far is a discrimination measure. Discrimination says how well a
score ranks patients; it does not say whether acting on the score beats the two things a triage
service can already do without it -- refer everyone who walks in, or refer nobody. A +0.041 AUC
increment is a real quantity and it is also not, on its own, an argument for buying an amplifier.

Decision-curve analysis supplies the missing step. At a threshold probability ``pt`` -- the risk of
impairment above which a clinician judges specialist referral worthwhile -- net benefit is

    NB = TP/n - (FP/n) * pt/(1 - pt)

the share of true referrals gained after charging each false referral at the exchange rate the
clinician's own threshold implies. A screener earns its place over the range of ``pt`` where its
curve sits above both references and nowhere else.

Two things this file insists on:

**Prevalence.** The CAUEEG screening contrast is 62% cases because it is a memory-clinic cohort.
No community triage service looks like that, and an unweighted curve would credit referral at every
threshold. Curves are therefore recomputed at several plausible service prevalences, with the
case-control sample re-weighted to each.

**The comparison is against age, not against chance.** The reference curves are treat-all,
treat-none, and *age alone* -- because age alone is free, and a screener that does not beat it has
not earned the recording.

    python experiments/e21_clinical_utility.py
Out: outputs/results/e21_clinical_utility.json
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
BUDGETS = {"full montage": "b19|256|f32|full", "four electrodes": "b4|256|f32|full"}
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}

#: Service prevalences the curves are recomputed at. 0.62 is the cohort as sampled and is shown
#: only so the re-weighting is visible; the lower values bracket what a primary-care or
#: community-outreach service would actually see.
PREVALENCES = [0.10, 0.20, 0.40, None]

#: Threshold probabilities swept. Below about 0.05 a service is referring almost everyone and the
#: curve is uninformative; above about 0.5 it is referring almost nobody.
THRESHOLDS = np.round(np.arange(0.05, 0.51, 0.01), 3)


def _curves(y, probs, prevalence):
    """Net-benefit curves for every arm at one assumed prevalence."""
    out = {}
    for name, p in probs.items():
        nb = ev.net_benefit(y, p, thresholds=THRESHOLDS, prevalence=prevalence)
        out[name] = nb["curve"]
        out["_prevalence"] = nb["prevalence_assumed"]
    return out


def _summarise(curves):
    """Where age+EEG beats age, and by how much, in referrals-per-thousand-assessed."""
    age = {r["threshold"]: r["net_benefit"] for r in curves["age"]}
    both = {r["threshold"]: r["net_benefit"] for r in curves["eeg+age"]}
    allref = {r["threshold"]: r["net_benefit_treat_all"] for r in curves["age"]}
    rows = []
    for t in sorted(age):
        d = both[t] - age[t]
        rows.append({"threshold": t, "d_net_benefit": d,
                     # Net benefit is in units of true positives per patient assessed, so the
                     # difference times 1000 is the extra correct referrals per thousand
                     # assessments at no increase in false referrals.
                     "extra_true_referrals_per_1000": 1000 * d,
                     "beats_age": bool(d > 0),
                     "beats_treat_all": bool(both[t] > allref[t] and both[t] > 0)})
    useful = [r for r in rows if r["beats_age"] and r["beats_treat_all"]]
    return {"rows": rows,
            "range_where_eeg_helps": ([useful[0]["threshold"], useful[-1]["threshold"]]
                                      if useful else None),
            "max_gain_per_1000": (max(r["extra_true_referrals_per_1000"] for r in useful)
                                  if useful else 0.0),
            "threshold_at_max": (max(useful, key=lambda r: r["d_net_benefit"])["threshold"]
                                 if useful else None)}


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT, "thresholds": THRESHOLDS.tolist(),
           "prevalences": PREVALENCES, "tasks": {}}

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        rec = {"n": int(len(y)), "observed_prevalence": float(y.mean()), "budgets": {}}
        print(f"\n=== {task}: n={len(y)}, cohort prevalence {y.mean():.2f}")

        for label, key in BUDGETS.items():
            if key not in keys:
                continue
            X = d["X"][keys.index(key)][ix]
            if not np.isfinite(X).any():
                continue
            # Probabilities, not rank scores: the threshold a decision curve sweeps is a risk.
            p_age, _ = ev.cv_probabilities(age[:, None], y)
            p_both, _ = ev.cv_probabilities(np.column_stack([X, age]), y)
            p_eeg, _ = ev.cv_probabilities(X, y)
            probs = {"age": p_age, "eeg": p_eeg, "eeg+age": p_both}

            by_prev = {}
            for prev in PREVALENCES:
                curves = _curves(y, probs, prev)
                summ = _summarise(curves)
                tag = "as sampled" if prev is None else f"{prev:.2f}"
                by_prev[tag] = {"curves": curves, "summary": summ}
                rng = summ["range_where_eeg_helps"]
                print(f"  {label:16s} prevalence {tag:11s} "
                      f"EEG+age beats age and treat-all over pt "
                      f"{('%.2f-%.2f' % (rng[0], rng[1])) if rng else 'nowhere':11s}"
                      f"  max +{summ['max_gain_per_1000']:.1f} correct referrals / 1000")
            rec["budgets"][label] = by_prev
        out["tasks"][task] = rec

    path = paths.RESULTS / "e21_clinical_utility.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
