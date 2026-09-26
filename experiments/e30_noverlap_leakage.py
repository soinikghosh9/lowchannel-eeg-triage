"""C1 leakage check: patient-disjoint evaluation using CAUEEG's official no-overlap partition,
labelled with THIS paper's contrasts (not the CAUEEG benchmark task labels).

CAUEEG's public annotation carries only a per-recording ``serial`` (no patient id), but the release
ships patient-disjoint splits (``dementia-no-overlap.json``) whose val/test are pruned so no patient
spans two splits. We keep only that train/test PARTITION and re-label each serial with the paper's
own labels (impaired vs control), then train on the no-overlap train serials and test on the
held-out ones. This is leakage-free at the patient level. If the EEG+age AUC here matches the
recording-level cross-validated AUC, repeated recordings from one patient are not inflating it.

    python experiments/e30_noverlap_leakage.py
Out: outputs/results/e30_noverlap_leakage.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import paths, spine, evaluate as ev  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PRIMARY = "b19|256|f32|full"
SPLIT = paths.DATASETS / "caueeg-dataset" / "dementia-no-overlap.json"
#: The release's second patient-disjoint partition, drawn for its normal-vs-abnormal task. Its
#: held-out patients differ from the first split's, so it is a second, independent check.
SPLIT2 = paths.DATASETS / "caueeg-dataset" / "abnormal-no-overlap.json"
DEMENTIA = {"AD", "FTD", "VAD"}


def _fit_scores(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def _fit_auc(Xtr, ytr, Xte, yte):
    return ev.auc(yte, _fit_scores(Xtr, ytr, Xte))


def _inc_ci(yte, s_age, s_both, n=3000, seed=0):
    """Paired bootstrap over held-out recordings for the increment AUC(both) - AUC(age)."""
    rng = np.random.default_rng(seed)
    yte = np.asarray(yte)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[idx])) < 2:
            continue
        out.append(ev.auc(yte[idx], s_both[idx]) - ev.auc(yte[idx], s_age[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


POSITIVE = {"screening": {"AD", "FTD", "VAD", "MCI"}, "dementia": DEMENTIA, "MCI": {"MCI"}}


def _rows(df, contrast):
    d = df.reset_index()
    return d[(d["cohort"] == "CAUEEG") & d["label"].isin(POSITIVE[contrast] | {"CN"})]


def _cv_on_subset(Xall, subj, df, contrast, held):
    rows = _rows(df, contrast)
    si = {s: i for i, s in enumerate(subj)}
    ix = np.array([si[s] for s in rows["subject"]])
    y = rows["label"].isin(POSITIVE[contrast]).to_numpy().astype(int)
    age = rows["age"].to_numpy(float)
    s_eeg = ev.cv_scores(Xall[ix], y)[0]
    s_both = ev.cv_scores(np.column_stack([Xall[ix], age]), y)[0]
    m = rows["subject"].str.split("_").str[-1].isin(held).to_numpy()
    return {"n": int(m.sum()), "auc_eeg": round(ev.auc(y[m], s_eeg[m]), 3),
            "auc_age": round(ev.auc(y[m], age[m]), 3),
            "auc_eeg_plus_age": round(ev.auc(y[m], s_both[m]), 3),
            "inc_margin": round(ev.auc(y[m], s_both[m]) - ev.auc(y[m], age[m]), 3)}


def _dedup(Xall, subj, df, contrast, repeats, n_random=5, seed=1):
    rows = _rows(df, contrast).reset_index(drop=True)
    si = {s: i for i, s in enumerate(subj)}
    keep = ~rows["subject"].str.split("_").str[-1].isin(repeats).to_numpy()

    def run(mask):
        r = rows[mask]
        ix = np.array([si[s] for s in r["subject"]])
        y = r["label"].isin(POSITIVE[contrast]).to_numpy().astype(int)
        age = r["age"].to_numpy(float)
        s_eeg = ev.cv_scores(Xall[ix], y)[0]
        s_both = ev.cv_scores(np.column_stack([Xall[ix], age]), y)[0]
        t = ev.delong_test(s_both, age, y)
        return {"n": int(len(y)), "auc_eeg": round(ev.auc(y, s_eeg), 3),
                "auc_age": round(ev.auc(y, age), 3), "auc_eeg_plus_age": round(ev.auc(y, s_both), 3),
                "inc_margin": round(t["diff"], 3), "inc_ci": [round(float(v), 3) for v in t["ci"]]}

    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        m = np.ones(len(rows), bool)
        m[rng.choice(len(rows), int((~keep).sum()), replace=False)] = False
        rand.append(run(m)["inc_margin"])
    return {"removed": int((~keep).sum()), "all": run(np.ones(len(rows), bool)),
            "dedup": run(keep), "random_removal_inc_mean": round(float(np.mean(rand)), 3)}


def _partition(split):
    d = json.loads(split.read_text(encoding="utf-8"))
    train_ser = {str(e["serial"]).zfill(5) for e in d["train_split"]}
    test_ser = {str(e["serial"]).zfill(5) for e in d["validation_split"] + d["test_split"]}
    return train_ser, test_ser


def main():
    train_ser, test_ser = _partition(SPLIT)

    cache = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    ki = list(cache["keys"]).index(PRIMARY)
    Xall = cache["X"][ki]
    subj = [str(s) for s in cache["subjects"]]
    df = spine.load().set_index("subject")

    # serial -> (features, age, label) for CAUEEG
    rec = {}
    for r, sub in enumerate(subj):
        if not sub.startswith("CAUEEG"):
            continue
        ser = sub.split("_")[-1]
        if sub in df.index and np.isfinite(Xall[r]).all():
            rec[ser] = (Xall[r], float(df.loc[sub, "age"]), str(df.loc[sub, "label"]))

    def design(serials, contrast):
        Xe, ag, y = [], [], []
        for ser in sorted(serials):
            if ser not in rec:
                continue
            f, a, lab = rec[ser]
            if contrast == "screening":
                if lab == "TGA":
                    continue
                yy = int(lab != "CN")
            else:  # dementia: AD/FTD/VAD vs CN only
                if lab not in DEMENTIA and lab != "CN":
                    continue
                yy = int(lab in DEMENTIA)
            Xe.append(f); ag.append(a); y.append(yy)
        return np.array(Xe), np.array(ag).reshape(-1, 1), np.array(y)

    def evaluate(train_ser, test_ser, contrast):
        Xtr, atr, ytr = design(train_ser, contrast)
        Xte, ate, yte = design(test_ser, contrast)
        s_age = _fit_scores(atr, ytr, ate)
        s_both = _fit_scores(np.hstack([Xtr, atr]), ytr, np.hstack([Xte, ate]))
        auc_age, auc_both = ev.auc(yte, s_age), ev.auc(yte, s_both)
        return {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                "n_test_pos": int(yte.sum()), "n_test_neg": int((yte == 0).sum()),
                "auc_age": round(auc_age, 3), "auc_eeg": round(_fit_auc(Xtr, ytr, Xte, yte), 3),
                "auc_eeg_plus_age": round(auc_both, 3), "inc_margin": round(auc_both - auc_age, 3),
                "inc_ci": _inc_ci(yte, s_age, s_both)}

    out = {}
    cv = {"screening": (0.786, 0.827, 0.041), "dementia": (0.822, 0.889, 0.067)}
    for contrast in ("screening", "dementia"):
        Xtr, atr, ytr = design(train_ser, contrast)
        Xte, ate, yte = design(test_ser, contrast)
        s_age = _fit_scores(atr, ytr, ate)
        s_both = _fit_scores(np.hstack([Xtr, atr]), ytr, np.hstack([Xte, ate]))
        auc_age, auc_both = ev.auc(yte, s_age), ev.auc(yte, s_both)
        auc_eeg = _fit_auc(Xtr, ytr, Xte, yte)
        out[contrast] = {
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "n_test_pos": int(yte.sum()), "n_test_neg": int((yte == 0).sum()),
            "auc_age": round(auc_age, 3), "auc_eeg": round(auc_eeg, 3),
            "auc_eeg_plus_age": round(auc_both, 3), "inc_margin": round(auc_both - auc_age, 3),
            "inc_ci": _inc_ci(yte, s_age, s_both),
            "cv_age": cv[contrast][0], "cv_eeg_plus_age": cv[contrast][1], "cv_inc": cv[contrast][2],
        }
    # Second partition. Its held-out recordings overlap the first split's only by chance, so
    # agreement between the two is not one split agreeing with itself.
    train2, test2 = _partition(SPLIT2)
    out["second_split"] = {"split": SPLIT2.name,
                           "held_out_overlap_with_first": len(test2 & test_ser),
                           "n_held_out_serials": len(test2)}
    for contrast in ("screening", "dementia"):
        out["second_split"][contrast] = evaluate(train2, test2, contrast)
    # The same held-out recordings, scored by the recording-level cross-validation the paper
    # reports. If patient-disjoint training costs nothing, the two agree on identical recordings,
    # and any gap to the full-cohort figure belongs to the held-out sample, not to leakage.
    out["same_recordings_under_cv"] = {}
    for tag, held in (("first_split", test_ser), ("second_split", test2)):
        out["same_recordings_under_cv"][tag] = {c: _cv_on_subset(Xall, subj, df, c, held)
                                                for c in ("screening", "dementia")}

    # De-duplication. A recording pruned from a no-overlap held-out set is one whose patient has
    # another recording elsewhere: the release identifies it as a repeat visit. Dropping every
    # identified repeat from both partitions and re-running the paper's 5x5 cross-validation
    # removes the leakage those pairs could carry; a random removal of the same size separates
    # that from the loss of sample.
    repeats = set()
    for full, no in (("dementia.json", SPLIT), ("abnormal.json", SPLIT2)):
        _, held_full = _partition(SPLIT.parent / full)
        _, held_no = _partition(no)
        repeats |= held_full - held_no
    out["dedup"] = {"n_identified_repeats": len(repeats)}
    for contrast in ("screening", "dementia", "MCI"):
        out["dedup"][contrast] = _dedup(Xall, subj, df, contrast, repeats)
    (paths.RESULTS / "e30_noverlap_leakage.json").write_text(json.dumps(out, indent=2))
    print("patient-disjoint (no-overlap partition, paper labels) vs recording-level CV")
    for c, r in ((k, out[k]) for k in ("screening", "dementia")):
        print(f"  {c:10s} n_test={r['n_test']} ({r['n_test_pos']}+/{r['n_test_neg']}-)  "
              f"age {r['auc_age']} (cv {r['cv_age']})  eeg+age {r['auc_eeg_plus_age']} "
              f"(cv {r['cv_eeg_plus_age']})  inc {r['inc_margin']:+} (cv {r['cv_inc']:+})"
              )
    s2 = out["second_split"]
    print(f"second split ({s2['split']}, held-out overlap with first: "
          f"{s2['held_out_overlap_with_first']}/{s2['n_held_out_serials']})")
    for c in ("screening", "dementia"):
        r = s2[c]
        print(f"  {c:10s} n_test={r['n_test']} ({r['n_test_pos']}+/{r['n_test_neg']}-)  age "
              f"{r['auc_age']}  eeg {r['auc_eeg']}  eeg+age {r['auc_eeg_plus_age']}  inc "
              f"{r['inc_margin']:+} {r['inc_ci']}")


if __name__ == "__main__":
    main()
