"""Extract features for every subject under every acquisition budget.

One pass over the harmonised cache: each recording is loaded once, windowed once, then degraded
and measured under each condition. Conditions whose cache is absent are skipped and reported, so
the 'noica' arm simply does not appear until that cache is generated.

    python experiments/e01_extract.py                  # full sweep
    python experiments/e01_extract.py --limit 40       # smoke test
    python experiments/e01_extract.py --cohorts CAUEEG,ds004504_raw
Out: outputs/cache/features.npz, outputs/results/e01_extract.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import budgets, paths, spine  # noqa: E402
from eegbudget.degrade import apply_budget  # noqa: E402
from eegbudget.features import NAMES, subject_features  # noqa: E402
from eegbudget.montages import ALL, apply_montage, reorder_to_canon  # noqa: E402


def load_windows(row, cache_dir):
    """Return the analysed windows for a subject, or None if this cache lacks them.

    `cache_dir` of None means the primary cache, whose exact path the spine already records --
    which may be the parent corpus or the recovered-cohort directory beside it.

    A sidecar carrying `match_offsets` pins the windows to the ones the ICA arm analysed. That
    matters for the no-ICA condition: with artefact removal switched off more windows exceed the
    peak-to-peak threshold, so re-running selection would confound the preprocessing comparison
    with a change in which seconds of signal were measured.
    """
    p = Path(row["npy"]) if cache_dir is None else cache_dir / Path(row["npy"]).name
    if not p.exists():
        return None, 0
    meta = json.loads(p.with_suffix(".json").read_text())
    x = reorder_to_canon(np.load(p).astype(np.float64), meta["channels"])
    offsets = meta.get("match_offsets")
    if offsets:
        w = budgets.WINDOW_SAMPLES
        keep = [s for s in offsets if s + w <= x.shape[-1]]
        if not keep:
            return None, 0
        return np.stack([x[:, s:s + w] for s in keep]), 0
    return budgets.select_windows(x, subject=row["subject"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohorts", default="")
    args = ap.parse_args()

    paths.ensure_dirs()
    df = spine.load()
    if args.cohorts:
        df = df[df["cohort"].isin(args.cohorts.split(","))].reset_index(drop=True)
    if args.limit:
        df = df.groupby("cohort", group_keys=False).head(max(1, args.limit // df["cohort"].nunique()))
        df = df.reset_index(drop=True)

    grid = budgets.grid()
    # None marks the primary cache: the spine already records each subject's exact path, which
    # may be the parent corpus or the recovered-cohort directory beside it.
    caches = {"harmonised": None, "harmonised_noica": paths.HARMONISED_NOICA}
    available = [c for c in grid
                 if caches[budgets.cache_for(c)] is None
                 or caches[budgets.cache_for(c)].is_dir()]
    skipped = sorted({budgets.cache_for(c) for c in grid if c not in available})

    keys = [budgets.key(c) for c in available]
    X = np.full((len(available), len(df), len(NAMES)), np.nan)
    rejected = np.zeros(len(df))
    n_win = np.zeros(len(df))

    t0 = time.time()
    for i, row in df.iterrows():
        by_cache = {}
        for ci, cond in enumerate(available):
            cache = caches[budgets.cache_for(cond)]
            ck = str(cache)
            if ck not in by_cache:
                w, rej = load_windows(row, cache)
                by_cache[ck] = w
                if cache is None and w is not None:
                    rejected[i], n_win[i] = rej, len(w)
            w = by_cache[ck]
            if w is None or len(w) == 0:
                continue
            m = ALL[cond["montage"]]
            sig = apply_montage(w, m)
            deg, sf = apply_budget(sig, budgets.SF_NATIVE, cond["rate"], cond["bits"],
                                   budgets.band_for(cond))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X[ci, i] = subject_features(deg, sf, m)

        if (i + 1) % 100 == 0 or i + 1 == len(df):
            el = time.time() - t0
            print(f"  {i+1:5d}/{len(df)}  {el:6.1f}s  ({el/(i+1)*1000:.0f} ms/subject)", flush=True)

    np.savez_compressed(paths.CACHE / "features.npz", X=X, keys=np.array(keys),
                        names=np.array(NAMES), subjects=df["subject"].to_numpy(),
                        rejected=rejected, n_windows=n_win)

    # Usable features per condition, counted within the analysis cohort. Counting subjects with
    # no NaN at all is the wrong diagnostic: a montage lacking frontal sites has a structurally
    # undefined anterior-posterior gradient, so every subject carries one NaN by design and the
    # subject count reads zero while nothing is wrong.
    cau = np.array([s.startswith("CAUEEG") for s in df["subject"]])
    finite = {k: int(np.isfinite(X[i][cau]).all(axis=0).sum()) if cau.any() else None
              for i, k in enumerate(keys)}
    report = {"n_subjects": len(df), "n_conditions": len(available),
              "elapsed_s": round(time.time() - t0, 1),
              "skipped_caches": skipped,
              "conditions": [budgets.describe(c) for c in available],
              "usable_features_caueeg": finite,
              "median_windows": float(np.median(n_win)),
              "median_rejected": float(np.median(rejected))}
    (paths.RESULTS / "e01_extract.json").write_text(json.dumps(report, indent=2))

    print(f"\n{len(available)} conditions x {len(df)} subjects in {report['elapsed_s']}s")
    if skipped:
        print(f"skipped (cache absent): {skipped}")
    print(f"median windows/subject {report['median_windows']:.0f}, "
          f"median rejected {report['median_rejected']:.0f}")


if __name__ == "__main__":
    main()
