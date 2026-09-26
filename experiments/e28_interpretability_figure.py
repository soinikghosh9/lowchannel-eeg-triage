"""The output layer: what a clinician sees when age is used as context rather than as a predictor.

A screening instrument that emits one probability cannot be argued with. The same age-conditioned
reference the paper evaluates for transport has a second use: it turns each measurement into a
deviation from what is expected for a healthy person of this patient's age, which is a quantity a
neurologist can check against the record in front of them.

This draws that view compactly enough for the manuscript body: the ranked per-feature deviations,
and the single most deviant measure placed on its lifespan trajectory so the reader can see the
patient against age-matched controls rather than against a decision boundary.

The case is chosen the same way the full report chooses one, which is deliberately unflattering to
the age baseline: an impaired patient in the youngest fifth of impaired participants, that is, the
patient an instrument leaning on age is most likely to miss.

    python experiments/e28_interpretability_figure.py
Out: outputs/figures/fig3_interpretability.{pdf,png}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eegbudget import normative as nm  # noqa: E402
from eegbudget import paths, spine, viz  # noqa: E402
from eegbudget.features import NAMES  # noqa: E402

COHORT = "CAUEEG"
BUDGET = "b19|256|f32|full"
SPEC = 0.80


def main():
    paths.ensure_dirs()
    viz.use_style()
    warnings.simplefilter("ignore")

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    rows = df[(df["cohort"] == COHORT) & df["age"].notna()
              & (df["in_screening"] == 1)].reset_index(drop=True)
    ix = np.array([sub_index[s] for s in rows["subject"]])
    X = d["X"][keys.index(BUDGET)][ix]
    y = rows["impaired"].to_numpy()
    age = rows["age"].to_numpy(float)
    ctrl = y == 0

    model = nm.fit(X[ctrl], age[ctrl])
    Z = nm.deviation(model, X, age)
    score = nm.abnormality(Z)
    thr = float(np.nanquantile(score[ctrl], SPEC))

    cand = np.where((y == 1) & (age < np.nanpercentile(age[y == 1], 20)))[0]
    i = int(cand[np.argmax(score[cand])]) if len(cand) else int(np.nanargmax(score))
    z = Z[i]
    print(f"{rows.loc[i, 'subject']}  age {age[i]:.0f}  label {rows.loc[i, 'label']}  "
          f"abnormality {score[i]:.2f} (referral threshold {thr:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(5.15, 1.94), gridspec_kw={"width_ratios": [1.10, 1.0]})

    viz.deviation_bars(axes[0], z, NAMES, None, top=9)
    axes[0].set_title("a  Abnormal for this age", loc="left", fontweight="bold")

    # The most deviant measure, drawn against the control trajectory it is being judged by.
    j = int(np.nanargmax(np.abs(z)))
    grid = np.linspace(np.nanmin(age), np.nanmax(age), 120)
    mu, sd = nm.predict_moments(model, grid)
    viz.lifespan(axes[1], age[ctrl], X[ctrl, j], mu[:, j], sd[:, j], grid,
                 patient=(age[i], X[i, j]), title=None,
                 ylabel=viz.DISPLAY.get(NAMES[j], NAMES[j].replace("_", " ")))
    axes[1].set_title("b  The largest deviation, against controls", loc="left",
                      fontweight="bold")

    fig.tight_layout(w_pad=0.9)
    viz.save(fig, paths.FIGURES / "fig3_interpretability")

    # The manuscript quotes this case, so its numbers travel with the artefacts like any other.
    import json
    top = np.argsort(np.where(np.isfinite(z), np.abs(z), -np.inf))[::-1][:3]
    rec = {"subject": str(rows.loc[i, "subject"]), "age": float(age[i]),
           "label": str(rows.loc[i, "label"]), "budget": BUDGET,
           "abnormality": float(score[i]), "referral_threshold": thr,
           "n_controls_fitting_reference": int(ctrl.sum()),
           "most_deviant": [{"feature": NAMES[int(t)], "z": float(z[int(t)])} for t in top],
           "note": ("An impaired patient in the youngest fifth of impaired participants, that is "
                    "the case an instrument leaning on age is most likely to miss.")}
    (paths.RESULTS / "e28_interpretability.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    print(f"wrote {paths.FIGURES / 'fig3_interpretability'}.pdf and the case record")


if __name__ == "__main__":
    main()
