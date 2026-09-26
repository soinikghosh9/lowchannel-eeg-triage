"""What a screening device should be built like: where the electrodes go, and how cheap the rest can be.

The feature cache holds 42 acquisition conditions on three orthogonal axes -- electrode montage,
digitisation (sampling rate x ADC depth), and preprocessing -- and the manuscript previously
reported six of them. This experiment reads all of them on one contrast family and answers the
design question the sweep was built for: given a fixed budget, what actually decides whether a
low-cost device can screen.

Three things are measured and they are deliberately kept apart:

    placement      16 montages at a clinical amplifier, including four commodity headset layouts
    digitisation   3 sampling rates x 4 bit depths, at 19 and at 4 electrodes
    mechanism      per-feature discrimination by scalp region, which says *why* placement matters

The commodity rows are the ones most easily over-read. They are the nearest 10-20 sites to each
headset's electrodes, recorded on clinical hardware -- so they compare electrode *layouts*, not
devices, and they carry none of the dry-electrode impedance, amplifier noise or motion artefact a
real headset would. That distinction is the difference between a defensible design claim and one a
reviewer who owns the device can falsify.

Paired tests are Holm-corrected over the placement family, which is prespecified here rather than
chosen after looking: the ladder-versus-clinical equivalences, the temporal-versus-frontal
contrasts, the commodity layouts against the best small montage, and the re-referencing control.

    python experiments/e25_acquisition_design.py
Out: outputs/results/e25_acquisition_design.json
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
from eegbudget.montages import ALL  # noqa: E402

COHORT = "CAUEEG"
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"}, "dementia": {"AD", "FTD", "VAD"}}

#: Axis A, in the order the paper reports them: the ladder, the four regions, the commodity
#: layouts, and the two optimistic re-referencing controls.
MONTAGES = ["b19", "b8", "b4", "b2", "b1_ap",
            "r_temporal", "r_posterior", "r_central", "r_frontal",
            "muse", "ganglion", "insight", "frontal1",
            "b4_inherit", "b8_inherit"]
RATES = [256, 128, 64]
BITS = ["f32", "16", "12", "8"]
DIGITISATION_MONTAGES = ["b19", "b4"]

#: Prespecified paired comparisons. Two ask whether a small montage loses anything against the
#: clinical one (equivalence), four ask whether placement matters at matched or favourable cost, three
#: place the commodity layouts, and one checks that subset re-referencing is not inflating the
#: ladder.
PAIRS = [
    ("b2", "b19", "two temporo-parietal vs nineteen"),
    ("b4", "b19", "four temporo-parietal/occipital vs nineteen"),
    ("r_temporal", "r_frontal", "four temporal vs seven frontal"),
    ("b2", "r_frontal", "two temporo-parietal vs seven frontal"),
    ("b2", "frontal1", "two temporo-parietal vs one frontal bipolar"),
    ("r_posterior", "r_frontal", "five posterior vs seven frontal"),
    ("b2", "muse", "two temporo-parietal vs Muse layout"),
    ("b2", "ganglion", "two temporo-parietal vs Ganglion layout"),
    ("b2", "insight", "two temporo-parietal vs Insight layout"),
    ("b4", "b4_inherit", "subset re-reference vs inherited 19-channel reference"),
]


def _rows(d, keys, ix, y, age, montages=None, digitisation=False):
    """Score one contrast across the requested conditions, and cache the scores for pairing."""
    # Age is a single monotonic predictor, so it is scored as its own value (as in e13), not
    # through a per-fold logistic fit: fitting only adds cross-validation noise. This is the rank
    # AUC (0.786 on screening), the same age baseline the increment is measured against below.
    auc_age = ev.auc(y, age)
    out, scores = {}, {}
    conditions = []
    if montages:
        conditions += [(m, f"{m}|256|f32|full") for m in montages]
    if digitisation:
        for m in DIGITISATION_MONTAGES:
            for rate in RATES:
                for b in BITS:
                    conditions.append((f"{m}|{rate}|{b}", f"{m}|{rate}|{b}|full"))
    for tag, key in conditions:
        if key not in keys:
            continue
        X = d["X"][keys.index(key)][ix]
        if not np.isfinite(X).any():
            continue
        s_eeg = ev.cv_scores(X, y)[0]
        s_both = ev.cv_scores(np.column_stack([X, age]), y)[0]
        inc = ev.delong_test(s_both, age, y)
        rec = {"key": key, "n_features": int(np.isfinite(X).all(0).sum()),
               "auc_eeg": ev.auc(y, s_eeg), "auc_eeg_plus_age": ev.auc(y, s_both),
               # A bar without an interval invites the reader to rank montages that are not
               # separated. The interval is a stratified subject bootstrap on the same scores.
               "auc_eeg_ci": list(ev.bootstrap_ci(y, s_eeg)),
               "inc_margin": inc["diff"], "inc_ci": inc["ci"], "inc_p": inc["p"]}
        if tag in ALL:
            rec["n_electrodes"] = len(ALL[tag]["channels"])
            rec["channels"] = list(ALL[tag]["channels"])
            rec["family"] = ALL[tag].get("family", "ladder")
        out[tag] = rec
        scores[tag] = s_eeg
    return auc_age, out, scores


def main():
    paths.ensure_dirs()
    warnings.simplefilter("ignore")

    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT,
           "note": ("Commodity rows are nearest-site approximations of each headset's electrode "
                    "layout on clinical hardware. They compare placement, not devices, and carry no "
                    "dry-electrode impedance, amplifier noise or motion artefact."),
           "tasks": {}}

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        print(f"\n=== {task}: n={len(y)}, {int((y == 0).sum())} control")

        auc_age, placement, scores = _rows(d, keys, ix, y, age, montages=MONTAGES)
        _, digit, _ = _rows(d, keys, ix, y, age, digitisation=True)

        # Paired placement tests on identical subjects, corrected together.
        tests, ps = {}, []
        for a, b, label in PAIRS:
            if a not in scores or b not in scores:
                continue
            t = ev.delong_test(scores[a], scores[b], y)
            tests[f"{a}_vs_{b}"] = {"label": label, "diff": t["diff"], "ci": t["ci"], "p": t["p"]}
            ps.append(t["p"])
        for (k, rec), ph in zip(tests.items(), ev.holm(ps)):
            rec["p_holm"] = float(ph)

        # The two spans the design argument turns on, computed rather than asserted: how far
        # discrimination moves across placements, against how far it moves across the whole
        # digitisation grid at a fixed montage.
        place_aucs = [v["auc_eeg"] for v in placement.values() if v.get("family") != "control"]
        spans = {"placement": float(max(place_aucs) - min(place_aucs))}
        for m in DIGITISATION_MONTAGES:
            vals = [v["auc_eeg"] for k, v in digit.items() if k.startswith(f"{m}|")]
            if vals:
                spans[f"digitisation_{m}"] = float(max(vals) - min(vals))
        if spans.get("placement") and spans.get("digitisation_b19"):
            spans["ratio_b19"] = spans["placement"] / spans["digitisation_b19"]

        out["tasks"][task] = {"n": int(len(y)), "n_control": int((y == 0).sum()),
                              "auc_age": auc_age, "placement": placement,
                              "digitisation": digit, "tests": tests, "spans": spans}

        best = max(placement.items(), key=lambda kv: kv[1]["auc_eeg"])
        worst = min(placement.items(), key=lambda kv: kv[1]["auc_eeg"])
        print(f"  age {auc_age:.3f} | best {best[0]} {best[1]['auc_eeg']:.3f} "
              f"({best[1].get('n_electrodes')} el) | worst {worst[0]} {worst[1]['auc_eeg']:.3f} "
              f"({worst[1].get('n_electrodes')} el)")
        print(f"  spans: placement {spans['placement']:.3f}, "
              f"digitisation(b19) {spans.get('digitisation_b19', float('nan')):.3f}, "
              f"ratio {spans.get('ratio_b19', float('nan')):.1f}x")

    # ---- mechanism: which feature is measurable where -------------------------------------
    # A ranking of montages is an empirical fact about one cohort. A ranking that follows the
    # generators of the features involved is a design principle, and travels.
    e05 = paths.RESULTS / "e05_regional.json"
    if e05.exists():
        reg = json.loads(e05.read_text(encoding="utf-8"))["regions"]
        order = ["r_temporal", "r_posterior", "r_central", "r_frontal"]
        feats = list(next(iter(reg.values()))["per_feature"].keys())
        profile = {}
        for f in feats:
            vals = {r.replace("r_", ""): reg[r]["per_feature"][f]["auc_raw"]
                    for r in order if r in reg}
            # Distance from 0.5 is the signal: several markers are inverted predictors, and a
            # raw AUC of 0.26 is as informative as one of 0.74.
            strength = {k: abs(v - 0.5) for k, v in vals.items()}
            best_region = max(strength, key=strength.get)
            profile[f] = {"auc_by_region": vals, "strength_by_region": strength,
                          "peak_region": best_region, "peak_strength": strength[best_region]}
        ranked = sorted(profile.items(), key=lambda kv: -kv[1]["peak_strength"])
        out["mechanism"] = {
            "regions": {r.replace("r_", ""): {"n_electrodes": reg[r]["n_electrodes"],
                                              "auc_eeg": reg[r]["auc_eeg"]}
                        for r in order if r in reg},
            "per_feature": profile,
            "top_features": [k for k, _ in ranked[:6]],
            "n_top_peaking_posterior": int(sum(
                1 for _, v in ranked[:6] if v["peak_region"] in ("temporal", "posterior"))),
        }
        print(f"\nmechanism: {out['mechanism']['n_top_peaking_posterior']} of the 6 strongest "
              f"features peak over temporal or posterior sites")

    dst = paths.RESULTS / "e25_acquisition_design.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
