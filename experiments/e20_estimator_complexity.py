"""Is a low-cost cohort's null result about the sensor, or about how many people it has?

The EasyCog arm returns no incremental value over age. Two explanations fit that, and they imply
opposite things for anyone deploying a screener in a low-resource clinic:

  1. forehead-and-ear EEG at 125 Hz does not carry age-independent cognitive information, or
  2. sixty-nine participants cannot support the estimator that was pointed at them.

They are separable, because CAUEEG has 1,200 subjects and the same three features. Subsample it
down through EasyCog's sample size and watch what happens to an estimator that is known to work at
full size. If a seventeen-coefficient model on CAUEEG collapses -- and inverts -- once it reaches
sixty-nine subjects, then EasyCog's null is a statement about sample size that would have looked
identical on a cohort where the signal is definitely present, and the deployment rule is about
estimator complexity rather than about hardware.

The x-axis worth reading is not n but events per variable: the smaller class divided by the number
of fitted coefficients. That places both cohorts, both estimators and every subsample on one axis,
and it is the quantity a clinic can compute before it starts.

Two estimators, matched to the EasyCog analysis:

    eeg17   all seventeen features
    slow3   theta/alpha ratio, relative theta, posterior dominant rhythm frequency

    python experiments/e20_estimator_complexity.py
Out: outputs/results/e20_estimator_complexity.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402
from eegbudget.easycog import SLOWING_TRIAD  # noqa: E402
from eegbudget.features import NAMES  # noqa: E402

COHORT = "CAUEEG"
BUDGET = "b19|256|f32|full"
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"}}

#: Subsample sizes. The low end brackets EasyCog (69 participants, 34 in its smaller class); 69, 79
#: and 89 are the sizes of EasyCog, BrainLat (79 with age) and P-ADIC, so each external cohort's
#: measured increment can be read against CAUEEG draws of its own size. The high end is the full
#: contrast.
SIZES = [40, 60, 69, 79, 89, 120, 200, 350, 600, 1000, None]
N_DRAWS = 25
FOLDS, REPEATS = 5, 3
MIN_EPV = 10.0

ESTIMATORS = {"eeg17": NAMES, "slow3": SLOWING_TRIAD}


def _subsample(y, n, rng):
    """Draw n subjects preserving the contrast's class balance, or all of them when n is None."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = max(2, int(round(n * len(pos) / len(y))))
    n_neg = max(2, n - n_pos)
    n_pos, n_neg = min(n_pos, len(pos)), min(n_neg, len(neg))
    return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                   rng.choice(neg, n_neg, replace=False)]))


def _one_draw(X, y, age, folds):
    """AUCs and the incremental margin for one subsample."""
    s_eeg, keep = ev.cv_scores(X, y, folds=folds, repeats=REPEATS)
    s_both, _ = ev.cv_scores(np.column_stack([X, age]), y, folds=folds, repeats=REPEATS)
    a_age = ev.auc(y, age)
    return {"auc_age": a_age, "auc_eeg": ev.auc(y, s_eeg),
            "auc_both": ev.auc(y, s_both),
            "sub": ev.auc(y, s_eeg) - a_age, "inc": ev.auc(y, s_both) - a_age,
            "n_features": int(keep.sum())}


def sweep(X_full, y_full, age_full, task):
    rows = []
    for est, names in ESTIMATORS.items():
        cols = [NAMES.index(n) for n in names]
        Xe = X_full[:, cols]
        seen_n = set()
        for size in SIZES:
            if size is not None and size >= len(y_full):
                continue          # would clamp to the full cohort, which `None` already covers
            # A stable seed per (task, estimator, size). Python's hash() of a string changes from
            # process to process, so it cannot seed a result that must reproduce.
            rng = np.random.default_rng(zlib.crc32(f"{task}|{est}|{size or 0}".encode()))
            draws, skipped = [], 0
            for _ in range(N_DRAWS if size is not None else 1):
                ix = _subsample(y_full, size, rng)
                y, age, X = y_full[ix], age_full[ix], Xe[ix]
                counts = np.bincount(y, minlength=2)
                if counts.min() < FOLDS:
                    skipped += 1
                    continue
                draws.append(_one_draw(X, y, age, folds=FOLDS))
            if not draws:
                continue
            n_used = int(len(_subsample(y_full, size, np.random.default_rng(0))))
            if n_used in seen_n:
                continue
            seen_n.add(n_used)
            smaller = int(min(np.bincount(y_full[_subsample(y_full, size,
                                                            np.random.default_rng(0))],
                                          minlength=2)))
            epv = smaller / len(names)
            agg = {"task": task, "estimator": est, "n": n_used,
                   "n_smaller_class": smaller, "n_features": len(names),
                   "events_per_variable": round(epv, 2),
                   "adequately_powered": bool(epv >= MIN_EPV),
                   "n_draws": len(draws), "n_skipped": skipped}
            for k in ("auc_age", "auc_eeg", "auc_both", "sub", "inc"):
                v = np.array([d[k] for d in draws], float)
                agg[f"{k}_mean"] = float(v.mean())
                agg[f"{k}_sd"] = float(v.std())
                agg[f"{k}_q"] = [float(np.percentile(v, 10)), float(np.percentile(v, 90))]
            # The share of draws where the out-of-fold EEG score ranks worse than chance. This is
            # the failure EasyCog exhibits, so it is measured rather than described.
            agg["frac_subchance"] = float(np.mean([d["auc_eeg"] < 0.5 for d in draws]))
            # The share of draws in which buying the recording improved on age. This is the
            # decision a clinic faces, and it is a far higher bar than beating chance.
            agg["frac_inc_positive"] = float(np.mean([d["inc"] > 0 for d in draws]))
            rows.append(agg)
            print(f"  {est:6s} n={n_used:5d}  EPV {epv:6.2f}  "
                  f"age {agg['auc_age_mean']:.3f}  EEG {agg['auc_eeg_mean']:.3f}"
                  f" +-{agg['auc_eeg_sd']:.3f}  inc {agg['inc_mean']:+.3f}"
                  f" [{agg['inc_q'][0]:+.3f},{agg['inc_q'][1]:+.3f}]"
                  f"  inc>0 in {agg['frac_inc_positive']*100:3.0f}%"
                  f"  sub-chance in {agg['frac_subchance']*100:3.0f}%", flush=True)
    return rows


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")
    t0 = time.time()

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT, "budget": BUDGET, "min_epv": MIN_EPV,
           "design": {"sizes": SIZES, "n_draws": N_DRAWS, "folds": FOLDS, "repeats": REPEATS},
           "estimators": {k: list(v) for k, v in ESTIMATORS.items()},
           "rows": []}

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        X = d["X"][keys.index(BUDGET)][ix]
        print(f"\n=== {task}: n={len(y)} ({(y == 0).sum()} control, {y.sum()} case)")
        out["rows"] += sweep(X, y, age, task)

    # The crossing point each estimator needs before its out-of-fold score is reliably above
    # chance -- the number a clinic can check against its own recruitment before choosing a model.
    out["thresholds"] = {}
    for task in TASKS:
        for est in ESTIMATORS:
            r = sorted([x for x in out["rows"] if x["task"] == task and x["estimator"] == est],
                       key=lambda x: x["n"])
            above = [x for x in r if x["frac_subchance"] <= 0.05]
            # "Reliably positive" is read as a lower 10th percentile above zero, i.e. the
            # increment survives an unlucky draw rather than merely averaging positive.
            positive = [x for x in r if x["inc_q"][0] > 0]
            out["thresholds"][f"{task}|{est}"] = {
                "smallest_n_reliably_above_chance": above[0]["n"] if above else None,
                "epv_above_chance": above[0]["events_per_variable"] if above else None,
                "smallest_n_increment_reliably_positive": positive[0]["n"] if positive else None,
                "epv_increment_positive": (positive[0]["events_per_variable"]
                                           if positive else None),
                "inc_at_full_cohort": r[-1]["inc_mean"]}

    path = paths.RESULTS / "e20_estimator_complexity.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("\nsample size needed, by estimator:")
    print(f"  {'task|estimator':22s}{'above chance':>14s}{'increment > 0':>16s}"
          f"{'inc at full n':>15s}")
    for k, v in out["thresholds"].items():
        a = v["smallest_n_reliably_above_chance"]
        b = v["smallest_n_increment_reliably_positive"]
        print(f"  {k:22s}{('n=%d' % a) if a else 'never':>14s}"
              f"{(('n=%d' % b) if b else 'never in range'):>16s}"
              f"{v['inc_at_full_cohort']:+15.3f}")
    print(f"\nwrote {path}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
