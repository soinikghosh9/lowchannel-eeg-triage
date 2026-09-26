"""The external operating point of e24, carried to all three external cohorts.

e24 transfers one CAUEEG rule per arm to ds004504, whose cases are younger than its controls, and
finds the transported age rule below chance. Whether that is a property of ds004504 or of age as a
transported baseline can only be told from cohorts with a different age structure. BrainLat and
P-ADIC supply two, with cases older than controls as in CAUEEG.

The protocol is e24's, imported rather than re-implemented: one rule per arm fitted on CAUEEG's
dementia contrast, threshold at the 80th percentile of source out-of-fold control probabilities,
applied once, no target label used. ds004504 must reproduce e24 exactly. Added here: the increment
of EEG+age over transported age on identical target recordings, the quantity the paper is built on,
and a fixed-effect pool of it across the three sites.

    python experiments/e35_external_transfer_all.py
Out: outputs/results/e35_external_transfer_all.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eegbudget import evaluate as ev  # noqa: E402
from eegbudget import normative as nm  # noqa: E402
from eegbudget import paths, spine  # noqa: E402

_spec = importlib.util.spec_from_file_location("e24", ROOT / "experiments" / "e24_external_operating_point.py")
e24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e24)

TARGETS = ["ds004504", "BrainLat", "P-ADIC"]
BUDGETS = {"b19": "b19|256|f32|full", "b4": "b4|256|f32|full"}
POS = {"AD", "FTD", "VAD"}


def main():
    warnings.simplefilter("ignore")
    src = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    ext = np.load(paths.CACHE / "features_external.npz", allow_pickle=True)
    skeys, ekeys = [str(k) for k in src["keys"]], [str(k) for k in ext["keys"]]
    sidx = {s: i for i, s in enumerate(src["subjects"])}
    eidx = {s: i for i, s in enumerate(ext["subjects"])}
    sdf = spine.load()
    s_rows = sdf[(sdf.cohort == "CAUEEG") & sdf.age.notna() & (sdf.in_screening == 1)
                 & sdf.label.isin(POS | {"CN"})]
    si = np.array([sidx[s] for s in s_rows.subject])
    ys = s_rows.label.isin(POS).to_numpy().astype(int)
    as_ = s_rows.age.to_numpy(float)
    edf = pd.read_csv(paths.CACHE / "spine_external.csv")
    edf = edf[(edf.excluded.fillna("") == "") & edf.age.notna() & edf.subject.isin(eidx)]

    out = {"protocol": "e24, dementia source contrast", "targets": {}, "pooled_increment": {}}
    for tag, key in BUDGETS.items():
        Xs_all = src["X"][skeys.index(key)][si]
        incs = []
        for tgt in TARGETS:
            rows = edf[(edf.cohort == tgt) & edf.label.isin({"AD", "FTD", "CN"})]
            ti = np.array([eidx[s] for s in rows.subject])
            yt = rows.label.isin({"AD", "FTD"}).to_numpy().astype(int)
            at = rows.age.to_numpy(float)
            Xt_all = ext["X"][ekeys.index(key)][ti]
            keep = np.isfinite(Xs_all).all(0) & np.isfinite(Xt_all).all(0)
            Xs, Xt = Xs_all[:, keep], Xt_all[:, keep]
            arms, probs = {}, {}
            designs = {"eeg": (Xs, Xt),
                       "eeg+age": (np.column_stack([Xs, as_]), np.column_stack([Xt, at]))}
            for label, (Fs, Ft) in designs.items():
                thr = float(np.quantile(e24._oof_probabilities(Fs, ys)[ys == 0], e24.SPEC_TARGET))
                p = e24._fit_calibrated(Fs, ys).predict_proba(Ft)[:, 1]
                arms[label] = e24._read_at(thr, p, yt)
                probs[label] = p
            ref = nm.fit(Xs[ys == 0], as_[ys == 0])
            Zs, Zt = nm.deviation(ref, Xs, as_), nm.deviation(ref, Xt, at)
            kz = np.isfinite(Zs).all(0) & np.isfinite(Zt).all(0)
            thr = float(np.quantile(e24._oof_probabilities(Zs[:, kz], ys)[ys == 0], e24.SPEC_TARGET))
            p = e24._fit_calibrated(Zs[:, kz], ys).predict_proba(Zt[:, kz])[:, 1]
            arms["normative"], probs["normative"] = e24._read_at(thr, p, yt), p
            thr = float(np.quantile(e24._oof_probabilities(as_[:, None], ys)[ys == 0], e24.SPEC_TARGET))
            p = e24._fit_calibrated(as_[:, None], ys).predict_proba(at[:, None])[:, 1]
            arms["age_transported"], probs["age_transported"] = e24._read_at(thr, p, yt), p
            for label, a in arms.items():
                a["auc"] = ev.auc(yt, probs[label])
                a["auc_ci"] = list(ev.bootstrap_ci(yt, probs[label]))
            inc = ev.delong_test(probs["eeg+age"], probs["age_transported"], yt)
            sub = ev.delong_test(probs["eeg"], probs["age_transported"], yt)
            incs.append((inc["diff"], inc["se"]))
            out["targets"].setdefault(tgt, {"n": int(len(yt)), "n_control": int((yt == 0).sum()),
                                            "age_gap": float(at[yt == 1].mean() - at[yt == 0].mean())})
            out["targets"][tgt][tag] = {"arms": arms,
                                        "increment_over_transported_age": {k: inc[k] for k in ("diff", "ci", "p")},
                                        "substitution_over_transported_age": {k: sub[k] for k in ("diff", "ci", "p")}}
            print(f"{tag} {tgt:9s} n={len(yt)} gap {out['targets'][tgt]['age_gap']:+.1f}y | " +
                  " ".join(f"{k} {a['auc']:.3f}/sens {a['sensitivity']:.2f}/spec {a['specificity']:.2f}"
                           for k, a in arms.items()) +
                  f" | inc {inc['diff']:+.3f} [{inc['ci'][0]:+.3f},{inc['ci'][1]:+.3f}]")
        d, s = np.array(incs).T
        w = 1 / s ** 2
        est, se = float((w * d).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
        out["pooled_increment"][tag] = {"diff": est, "ci": [est - 1.96 * se, est + 1.96 * se],
                                        "p": float(2 * norm.sf(abs(est / se))),
                                        "Q": float((w * (d - est) ** 2).sum())}
        # Without ds004504, whose inverted age structure is the known outlier.
        w2, d2 = w[1:], d[1:]
        e2, s2 = float((w2 * d2).sum() / w2.sum()), float(np.sqrt(1 / w2.sum()))
        out["pooled_increment"][tag + "_excl_ds004504"] = {"diff": e2, "ci": [e2 - 1.96 * s2, e2 + 1.96 * s2],
                                                           "p": float(2 * norm.sf(abs(e2 / s2)))}
        print(f"  pooled increment {tag}: {est:+.3f} [{est - 1.96 * se:+.3f}, {est + 1.96 * se:+.3f}] "
              f"| excl ds004504 {e2:+.3f} [{e2 - 1.96 * s2:+.3f}, {e2 + 1.96 * s2:+.3f}]")

    e24_ref = json.loads((paths.RESULTS / "e24_external_operating_point.json").read_text())
    ref = e24_ref["source_contrasts"]["dementia"]["budgets"]["b19"]
    mine = out["targets"]["ds004504"]["b19"]["arms"]
    out["reproduces_e24"] = all(abs(ref[k]["auc"] - mine[k]["auc"]) < 1e-9 for k in ref)
    print("ds004504 reproduces e24:", out["reproduces_e24"])

    def cj(o):
        if isinstance(o, dict):
            return {k: cj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [cj(v) for v in list(o)]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer, np.bool_)):
            return o.item()
        return o

    (paths.RESULTS / "e35_external_transfer_all.json").write_text(json.dumps(cj(out), indent=1))


if __name__ == "__main__":
    main()
