"""Why the two substitution-ratio estimators disagree, and what is left of the ratio.

The paper reports rho = [AUC - AUC_age-equalised] / [AUC - 0.5] twice, by matching and by
residualisation, and the two disagree by roughly a factor of two (0.29 against 0.51 at the full
montage). The paper's response was to read the ratio as an interval. That is honest but incurious:
both estimators have a named bias, both biases run the same way, and both are fixable.

    matching        equalises age but also shrinks the training set, from 1,200 subjects to 660.
                    A smaller training set lowers AUC on its own, so part of the drop attributed
                    to age is really a drop attributed to sample size. Isolated here by a control
                    arm that discards the same number of subjects *at random*.

    residualisation retains everyone but fits the age trend on cases and controls pooled. Cases
                    are both older and abnormal, so the pooled fit absorbs disease into the age
                    term and over-removes. Isolated here by refitting the trend on controls alone,
                    which is what the normative model already does.

If the corrected estimators converge, the ratio is worth quoting. If they do not, the paper should
quote the two margins, which need no such correction, and say so.

    python experiments/e16_rho_check.py
Out: outputs/results/e16_rho_check.json
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
BUDGETS = ["b19|256|f32|full", "b4|256|f32|full"]
TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}
N_SIZE_DRAWS = 25


def main():
    paths.ensure_dirs()
    d = np.load(paths.CACHE / "features.npz", allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    sub_index = {s: i for i, s in enumerate(d["subjects"])}
    df = spine.load()
    base = df[(df["cohort"] == COHORT) & df["age"].notna() & (df["in_screening"] == 1)]

    out = {"cohort": COHORT, "tasks": {}}
    print(f"{'task':11s}{'budget':6s}{'AUC':>7s}{'match':>8s}{'sizeonly':>10s}"
          f"{'resid_p':>9s}{'resid_c':>9s} | {'rho_m':>6s}{'rho_size':>9s}"
          f"{'rho_m_adj':>10s}{'rho_rp':>8s}{'rho_rc':>8s}")

    for task, pos in TASKS.items():
        rows = base[base["label"].isin(pos | {"CN"})].reset_index(drop=True)
        ix = np.array([sub_index[s] for s in rows["subject"]])
        y = rows["label"].isin(pos).to_numpy().astype(int)
        age = rows["age"].to_numpy(float)
        keep_ix = ev.match_on_age(age, y)
        rec = {"n": int(len(y)), "n_matched": int(len(keep_ix)), "budgets": {}}

        rng = np.random.default_rng(0)
        n_pos_m = int(y[keep_ix].sum())
        n_neg_m = int((y[keep_ix] == 0).sum())
        pos_ix, neg_ix = np.where(y == 1)[0], np.where(y == 0)[0]

        for k in BUDGETS:
            if k not in keys:
                continue
            X = d["X"][keys.index(k)][ix]
            s, _ = ev.cv_scores(X, y)
            a = ev.auc(y, s)

            s_m, _ = ev.cv_scores(X[keep_ix], y[keep_ix])
            a_m = ev.auc(y[keep_ix], s_m)

            # Size-only control: throw away exactly as many subjects, chosen at random, keeping
            # the class balance the matched sample ended up with.
            sizes = []
            for _ in range(N_SIZE_DRAWS):
                sel = np.concatenate([rng.choice(pos_ix, n_pos_m, replace=False),
                                      rng.choice(neg_ix, n_neg_m, replace=False)])
                s_s, _ = ev.cv_scores(X[sel], y[sel])
                sizes.append(ev.auc(y[sel], s_s))
            a_size = float(np.mean(sizes))

            s_rp, _ = ev.cv_scores(ev.residualise_on_age(X, age), y)
            s_rc, _ = ev.cv_scores(ev.residualise_on_controls(X, age, y), y)
            a_rp, a_rc = ev.auc(y, s_rp), ev.auc(y, s_rc)

            # Matching's ratio, corrected for the sample-size penalty it also incurs.
            rho_m = ev.substitution_ratio(a, a_m)
            rho_size = ev.substitution_ratio(a, a_size)
            rho_m_adj = ev.substitution_ratio(a_size, a_m) if np.isfinite(a_size) else np.nan

            rec["budgets"][k] = {
                "auc": a, "auc_matched": a_m, "auc_size_matched": a_size,
                "auc_resid_pooled": a_rp, "auc_resid_controls": a_rc,
                "rho_matched": rho_m, "rho_size_only": rho_size,
                "rho_matched_size_adjusted": rho_m_adj,
                "rho_resid_pooled": ev.substitution_ratio(a, a_rp),
                "rho_resid_controls": ev.substitution_ratio(a, a_rc)}
            r = rec["budgets"][k]
            print(f"{task:11s}{k.split('|')[0]:6s}{a:7.3f}{a_m:8.3f}{a_size:10.3f}"
                  f"{a_rp:9.3f}{a_rc:9.3f} | {rho_m:6.2f}{rho_size:9.2f}"
                  f"{rho_m_adj:10.2f}{r['rho_resid_pooled']:8.2f}{r['rho_resid_controls']:8.2f}")
        out["tasks"][task] = rec

    (paths.RESULTS / "e16_rho_check.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {paths.RESULTS / 'e16_rho_check.json'}")
    print("\nrho_m           matching, as the paper reports it")
    print("rho_size        what discarding the same number of subjects at random costs on its own")
    print("rho_m_adj       matching's ratio measured against the size-matched rather than the "
          "full-sample AUC")
    print("rho_rp/rho_rc   residualisation, age trend fitted pooled / on controls alone")


if __name__ == "__main__":
    main()
