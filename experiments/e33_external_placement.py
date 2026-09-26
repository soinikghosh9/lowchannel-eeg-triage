"""Does the placement result travel? The prespecified placement family, on three external cohorts.

The review asked for the placement ordering to be re-tested outside CAUEEG. Three cohorts now allow
it (e31): ds004504 (Greece), BrainLat (Argentina, Chile) and P-ADIC (Israel), 257 recordings.

Two readings, kept apart because they answer different questions:

    transfer    one rule per montage, fitted on CAUEEG's dementia contrast (the manuscript's
                external source model) and applied unchanged: does the montage ranking a clinic
                would inherit from the development cohort hold at a new site?
    within      5-fold cross-validation inside each external cohort, 20 repeats: where does the
                information sit in that cohort's own recordings, whatever CAUEEG learned?

Primary: the dementia contrast (every case against controls), the ten prespecified placement
comparisons of e25, per cohort and pooled across cohorts by fixed-effect inverse-variance weighting
of DeLong differences, Holm-adjusted over the ten.

Secondary, disease-specific (motivated by the ds004504 result, so reported as exploratory): the
same montages against controls for Alzheimer's disease and for behavioural-variant frontotemporal
dementia separately, the region-by-diagnosis interaction on the two cohorts that carry both
diagnoses, the regional topography of each diagnosis's markers, and which montage best separates
Alzheimer's disease from frontotemporal dementia -- the differential a placement choice could
narrow.

    python experiments/e33_external_placement.py
Out: outputs/results/e33_external_placement.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import paths, spine  # noqa: E402
from eegbudget.features import NAMES  # noqa: E402
from eegbudget.montages import ALL  # noqa: E402

COHORTS = ["ds004504", "BrainLat", "P-ADIC"]
MONTAGES = ["b19", "b8", "b4", "b2", "b1_ap", "r_temporal", "r_posterior", "r_central",
            "r_frontal", "muse", "ganglion", "insight", "frontal1", "b4_inherit"]
PAIRS = [  # e25's prespecified family, unchanged
    ("b2", "b19"), ("b4", "b19"), ("r_temporal", "r_frontal"), ("b2", "r_frontal"),
    ("b2", "frontal1"), ("r_posterior", "r_frontal"), ("b2", "muse"), ("b2", "ganglion"),
    ("b2", "insight"), ("b4", "b4_inherit")]
#: The regional contrasts the disease-specific reading turns on.
REGION_PAIRS = [("r_temporal", "r_frontal"), ("r_posterior", "r_frontal"), ("b4", "r_frontal")]
#: Whether layouts that add frontal sites keep the full montage's discrimination: the question
#: the frontotemporal result raises, asked directly rather than by differencing two tests.
FULL_PAIRS = [("insight", "b19"), ("ganglion", "b19"), ("muse", "b19"), ("r_frontal", "b19"),
              ("b8", "b19")]
#: AD against FTD: does a temporo-parietal array separate the two dementias better than a frontal one?
DIFF_PAIRS = [("b2", "r_frontal"), ("b2", "b19"), ("r_temporal", "r_frontal"),
              ("r_posterior", "r_frontal"), ("insight", "b19")]
REGIONS = ["r_frontal", "r_central", "r_temporal", "r_posterior"]
CONTRASTS = {"dementia": {"AD", "FTD"}, "AD": {"AD"}, "FTD": {"FTD"}}
SOURCE_POS = {"AD", "FTD", "VAD"}
REPEATS = 20


def _load():
    src = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    ext = np.load(paths.CACHE / "features_external.npz", allow_pickle=True)
    sdf = spine.load()
    sidx = {s: i for i, s in enumerate(src["subjects"])}
    s_rows = sdf[(sdf.cohort == "CAUEEG") & sdf.label.isin(SOURCE_POS | {"CN"})]
    si = np.array([sidx[s] for s in s_rows.subject])
    ys = s_rows.label.isin(SOURCE_POS).to_numpy().astype(int)
    edf = pd.read_csv(paths.CACHE / "spine_external.csv")
    edf = edf[edf["excluded"].fillna("") == ""]
    eidx = {s: i for i, s in enumerate(ext["subjects"])}
    edf = edf[edf.subject.isin(eidx)].reset_index(drop=True)
    return src, ext, si, ys, edf, eidx


def _X(cache, montage, rows):
    keys = [str(k) for k in cache["keys"]]
    return cache["X"][keys.index(f"{montage}|256|f32|full")][rows]


def _transfer_scores(Xs, ys, Xt):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    keep = np.isfinite(Xs).all(0) & np.isfinite(Xt).all(0)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xs[:, keep], ys)
    return m.decision_function(Xt[:, keep])


def _within_scores(Xt, yt):
    keep = np.isfinite(Xt).all(0)
    return ev.cv_scores(Xt[:, keep], yt, folds=5, repeats=REPEATS)[0]


def _pool(diffs, ses):
    """Fixed-effect inverse-variance pool of per-cohort AUC differences."""
    d, s = np.asarray(diffs, float), np.asarray(ses, float)
    ok = np.isfinite(d) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return {"diff": np.nan, "ci": [np.nan, np.nan], "p": np.nan}
    w = 1 / s[ok] ** 2
    est = float((w * d[ok]).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    het = float((w * (d[ok] - est) ** 2).sum())          # Cochran's Q
    return {"diff": est, "se": se, "ci": [est - 1.96 * se, est + 1.96 * se],
            "p": float(2 * norm.sf(abs(est / se))), "Q": het, "k": int(ok.sum())}


def _score_all(src, ext, si, ys, edf, eidx, contrast):
    """{cohort: {'y', 'transfer': {m: s}, 'within': {m: s}}} for one contrast."""
    pos = CONTRASTS[contrast]
    out = {}
    for c in COHORTS:
        rows = edf[(edf.cohort == c) & edf.label.isin(pos | {"CN"})]
        if rows.label.isin(pos).sum() < 5:
            continue
        ti = np.array([eidx[s] for s in rows.subject])
        yt = rows.label.isin(pos).to_numpy().astype(int)
        rec = {"y": yt, "subjects": rows.subject.tolist(), "transfer": {}, "within": {}}
        for m in MONTAGES:
            Xs, Xt = _X(src, m, si), _X(ext, m, ti)
            rec["transfer"][m] = _transfer_scores(Xs, ys, Xt)
            rec["within"][m] = _within_scores(Xt, yt)
        out[c] = rec
    return out


def _family(scored, reading, pairs):
    per, pooled = {}, {}
    for a, b in pairs:
        k = f"{a}_vs_{b}"
        per[k], diffs, ses = {}, [], []
        for c, rec in scored.items():
            t = ev.delong_test(rec[reading][a], rec[reading][b], rec["y"])
            per[k][c] = {"diff": t["diff"], "ci": t["ci"], "p": t["p"], "se": t.get("se")}
            diffs.append(t["diff"])
            ses.append(t.get("se", np.nan))
        pooled[k] = _pool(diffs, ses)
    for (k, rec), ph in zip(pooled.items(), ev.holm([v["p"] for v in pooled.values()])):
        rec["p_holm"] = float(ph)
    return per, pooled


def _aucs(scored, reading):
    out = {}
    for c, rec in scored.items():
        out[c] = {m: {"auc": ev.auc(rec["y"], s), "ci": list(ev.bootstrap_ci(rec["y"], s))}
                  for m, s in rec[reading].items()}
    return out


def _interaction(scored_ad, scored_ftd, reading, a, b, n_boot=4000, seed=0):
    """[AUC_a - AUC_b](AD) - [AUC_a - AUC_b](FTD), shared controls resampled jointly."""
    per = {}
    for c in scored_ad:
        if c not in scored_ftd:
            continue
        A, F = scored_ad[c], scored_ftd[c]
        subj_a, subj_f = np.array(A["subjects"]), np.array(F["subjects"])
        ctrl = sorted(set(subj_a[A["y"] == 0]) & set(subj_f[F["y"] == 0]))
        ia = {s: i for i, s in enumerate(subj_a)}
        jf = {s: i for i, s in enumerate(subj_f)}
        ca, cf = np.array([ia[s] for s in ctrl]), np.array([jf[s] for s in ctrl])
        ad, fd = np.where(A["y"] == 1)[0], np.where(F["y"] == 1)[0]

        def stat(cix, aix, fix):
            ya = np.r_[np.ones(len(aix)), np.zeros(len(cix))]
            yf = np.r_[np.ones(len(fix)), np.zeros(len(cix))]
            da = ev.auc(ya, np.r_[A[reading][a][aix], A[reading][a][ca[cix]]]) - \
                ev.auc(ya, np.r_[A[reading][b][aix], A[reading][b][ca[cix]]])
            df = ev.auc(yf, np.r_[F[reading][a][fix], F[reading][a][cf[cix]]]) - \
                ev.auc(yf, np.r_[F[reading][b][fix], F[reading][b][cf[cix]]])
            return da, df

        da, df = stat(np.arange(len(ctrl)), ad, fd)
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            cix = rng.integers(0, len(ctrl), len(ctrl))
            x, y = stat(cix, rng.choice(ad, len(ad)), rng.choice(fd, len(fd)))
            boot.append(x - y)
        boot = np.array(boot)
        se = float(np.std(boot))
        per[c] = {"diff_AD": da, "diff_FTD": df, "interaction": da - df, "se": se,
                  "ci": list(np.percentile(boot, [2.5, 97.5])),
                  "p": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))}
    pooled = _pool([v["interaction"] for v in per.values()], [v["se"] for v in per.values()])
    return {"per_cohort": per, "pooled": pooled}


def _topography(ext, edf, eidx):
    """Per marker and region, |AUC - 0.5| for each diagnosis against controls, averaged over cohorts."""
    out = {}
    for dx in ("AD", "FTD"):
        prof = {}
        for f in NAMES[:15]:
            fi = NAMES.index(f)
            vals = {}
            for r in REGIONS:
                per = []
                for c in COHORTS:
                    rows = edf[(edf.cohort == c) & edf.label.isin({dx, "CN"})]
                    if (rows.label == dx).sum() < 5:
                        continue
                    ti = np.array([eidx[s] for s in rows.subject])
                    x = _X(ext, r, ti)[:, fi]
                    y = (rows.label == dx).to_numpy().astype(int)
                    ok = np.isfinite(x)
                    per.append(abs(ev.auc(y[ok], x[ok]) - 0.5))
                vals[r.replace("r_", "")] = float(np.mean(per))
            prof[f] = vals
        ranked = sorted(prof, key=lambda f: -max(prof[f].values()))
        region_mean = {r.replace("r_", ""): float(np.mean([prof[f][r.replace("r_", "")]
                                                           for f in ranked[:6]]))
                       for r in REGIONS}
        out[dx] = {"per_feature": prof, "top6": ranked[:6],
                   "top6_peak_region": {f: max(prof[f], key=prof[f].get) for f in ranked[:6]},
                   "top6_mean_strength_by_region": region_mean}
    return out


def _differential(ext, edf, eidx):
    """AD against FTD, within cohort, per montage; pooled over the cohorts that carry both."""
    per = {}
    for c in ("ds004504", "BrainLat"):
        rows = edf[(edf.cohort == c) & edf.label.isin({"AD", "FTD"})]
        ti = np.array([eidx[s] for s in rows.subject])
        y = (rows.label == "FTD").to_numpy().astype(int)
        per[c] = {"y": y, "within": {m: _within_scores(_X(ext, m, ti), y) for m in MONTAGES}}
    aucs = _aucs(per, "within")
    pooled = {}
    for m in MONTAGES:
        a = [aucs[c][m]["auc"] for c in per]
        pooled[m] = {"auc_mean": float(np.mean(a)), "per_cohort": dict(zip(per, a))}
    tests, pooled_tests = _family(per, "within", DIFF_PAIRS)
    return {"aucs": aucs, "pooled_mean_auc": pooled, "tests": tests, "tests_pooled": pooled_tests,
            "n": {c: {"AD": int((v["y"] == 0).sum()), "FTD": int(v["y"].sum())}
                  for c, v in per.items()}}


def main():
    warnings.simplefilter("ignore")
    src, ext, si, ys, edf, eidx = _load()
    out = {"source": "CAUEEG dementia contrast (AD, FTD, VAD vs CN)",
           "cohorts": {c: {l: int(n) for l, n in g.label.value_counts().items()}
                       for c, g in edf.groupby("cohort")},
           "contrasts": {}}
    scored = {}
    for contrast in CONTRASTS:
        sc = _score_all(src, ext, si, ys, edf, eidx, contrast)
        scored[contrast] = sc
        rec = {"n": {c: {"case": int(v["y"].sum()), "control": int((v["y"] == 0).sum())}
                     for c, v in sc.items()}}
        for reading in ("transfer", "within"):
            per, pooled = _family(sc, reading, PAIRS)
            rper, rpooled = _family(sc, reading, REGION_PAIRS)
            fper, fpooled = _family(sc, reading, FULL_PAIRS)
            # Every montage against the full one, for the appendix forest plot (descriptive; the
            # Holm values there are over this larger set and are not the prespecified family's).
            _, vs_full = _family(sc, reading, [(m, "b19") for m in MONTAGES
                                               if m not in ("b19", "b4_inherit")])
            rec[reading] = {"auc": _aucs(sc, reading), "tests_per_cohort": per,
                            "tests_pooled": pooled, "region_tests_per_cohort": rper,
                            "region_tests_pooled": rpooled, "full_tests_per_cohort": fper,
                            "full_tests_pooled": fpooled, "vs_full_pooled": vs_full}
        out["contrasts"][contrast] = rec
        print(f"\n=== {contrast}: " + ", ".join(f"{c} {v}" for c, v in rec["n"].items()))
        for reading in ("transfer", "within"):
            a = rec[reading]["auc"]
            print(f"  [{reading}] " + " | ".join(
                f"{m}: " + "/".join(f"{a[c][m]['auc']:.3f}" for c in a) for m in
                ("b19", "b4", "b2", "r_temporal", "r_posterior", "r_frontal", "frontal1", "insight")))
            for k, v in rec[reading]["tests_pooled"].items():
                print(f"    pooled {k:24s} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
                      f"  p={v['p']:.4f}  holm={v['p_holm']:.4f}  Q={v.get('Q', np.nan):.2f}")

    out["interaction"] = {reading: {f"{a}_vs_{b}": _interaction(scored["AD"], scored["FTD"],
                                                                 reading, a, b)
                                    for a, b in REGION_PAIRS}
                          for reading in ("transfer", "within")}
    out["topography"] = _topography(ext, edf, eidx)
    out["differential_AD_vs_FTD"] = _differential(ext, edf, eidx)

    print("\n=== region x diagnosis interaction ([a-b](AD) - [a-b](FTD))")
    for reading, d in out["interaction"].items():
        for k, v in d.items():
            p = v["pooled"]
            print(f"  [{reading}] {k:24s} pooled {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"
                  f" p={p['p']:.4f} | " + " ".join(f"{c}: AD {x['diff_AD']:+.3f} FTD {x['diff_FTD']:+.3f}"
                                                  for c, x in v["per_cohort"].items()))
    print("\n=== topography, top-6 marker strength by region")
    for dx, t in out["topography"].items():
        print(f"  {dx}: " + ", ".join(f"{r} {v:.3f}" for r, v in
                                      t["top6_mean_strength_by_region"].items())
              + f" | peaks {t['top6_peak_region']}")
    print("\n=== AD vs FTD differential, within-cohort AUC")
    for m, v in out["differential_AD_vs_FTD"]["pooled_mean_auc"].items():
        print(f"  {m:12s} mean {v['auc_mean']:.3f}  " + " ".join(f"{c} {a:.3f}" for c, a in
                                                            v["per_cohort"].items()))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    dst = paths.RESULTS / "e33_external_placement.json"
    dst.write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
