"""Extend the harmonised corpus with cohorts and preprocessing arms the parent project omitted.

Two jobs, both writing **into this repository only**. The parent project's
``outputs/harmonised/`` is opened read-only and never modified, because the manuscript in progress
depends on it. Its own harmoniser hard-codes that directory as its output, so running it with
``--no-ica`` would write unclean recordings into the clean cache; this driver reuses the same
``harmonise()`` function and loaders but controls where the result lands.

    vascular  CAUEEG subjects the parent taxonomy excludes -- vascular dementia and transient
              global amnesia. `assign_class` maps both to 'EXCL', so the loader never lists them;
              we extend its subject map rather than edit its code. Harmonised WITH ICA, exactly
              like the main corpus, so the two are comparable.

    noica     CAUEEG and ds004504 harmonised WITHOUT artefact removal, for the preprocessing axis.
              A four-electrode device cannot run ICA: it needs channels, compute and an operator.

Both store full recordings, in the same format as the parent cache, so nothing downstream has to
special-case them. The noica sidecar additionally records the **sample offsets the ICA arm
selected**, and the extractor uses those rather than re-running rejection: with artefact removal
switched off more windows exceed the peak-to-peak threshold, so re-selecting would confound the
preprocessing comparison with a change in which seconds of signal were analysed.

    python experiments/e09_extend_corpus.py --job vascular
    python experiments/e09_extend_corpus.py --job noica
Out: outputs/harmonised_extra/ , outputs/harmonised_noica/
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import budgets, paths  # noqa: E402
from eegbudget.montages import CANON, reorder_to_canon  # noqa: E402

#: CAUEEG symptom tags the parent taxonomy discards, and the label we give them.
EXTRA_CLASSES = {"VAD": {"vd", "sivd", "ad_vd_mixed"}, "TGA": {"tga"}}

WIN = budgets.WINDOW_SAMPLES
QUOTA = budgets.QUOTA


def _parent():
    """Import the parent project's loaders and pipeline without importing this repo's src."""
    import mne
    mne.set_log_level("ERROR")
    from harmonisation import scaling
    from harmonisation.caueeg_loader import CAUEEGDataset, assign_class
    from harmonisation.bids_loader import BIDSDataset
    from harmonisation.harmonise import harmonise
    scaling.set_mode("none")     # the cache stores unscaled volts; scaling is a load-time choice
    return CAUEEGDataset, assign_class, BIDSDataset, harmonise


def caueeg_with_excluded(CAUEEGDataset, assign_class):
    """Build the loader, then add back the subjects its taxonomy drops.

    `load_raw` needs only `info_map[sid]['serial']`, so extending that map is enough and the
    parent loader is left untouched on disk.
    """
    ds = CAUEEGDataset(paths.DATASETS / "caueeg-dataset", verbose=False)
    ann = json.loads((paths.CAUEEG_ANNOTATION).read_text(encoding="utf-8"))["data"]
    added = {}
    for rec in ann:
        if assign_class(rec.get("symptom")) != "EXCL":
            continue
        tags = set(rec.get("symptom") or [])
        cls = next((c for c, t in EXTRA_CLASSES.items() if tags & t), None)
        if cls is None:
            continue
        serial = str(rec["serial"])
        if not (ds.edf_dir / f"{serial}.edf").exists():
            continue
        sid = f"CAUEEG_{cls}_{serial}"
        ds.info_map[sid] = {"serial": serial, "class": cls, "label": -1,
                            "age": float(rec.get("age", -1))}
        added[sid] = cls
    return ds, added


def ica_arm_offsets(subject):
    """Reproduce the offsets the ICA arm used, by re-running selection on its cached recording.

    The ICA arm lives in two directories: the parent cache, and the recovered-cohort directory
    this repository writes for the subjects the parent taxonomy drops. Looking only in the first
    silently skipped every vascular dementia and transient global amnesia recording, which is why
    the no-ICA arm covered 1,120 of 1,200 screening participants rather than all of them.
    """
    p = next((c / f"{subject}.npy"
              for c in (paths.HARMONISED, paths.HARMONISED_EXTRA)
              if (c / f"{subject}.npy").exists()), None)
    if p is None:
        return None
    meta = json.loads(p.with_suffix(".json").read_text())
    x = reorder_to_canon(np.load(p).astype(np.float64), meta["channels"])
    n = x.shape[-1] // WIN
    starts = np.arange(n) * WIN
    ptp = np.array([np.ptp(x[:, s:s + WIN], axis=-1).max() for s in starts])
    ok = starts[ptp <= budgets.PTP_REJECT_V]
    if len(ok) == 0:
        ok = starts[np.argsort(ptp)[:QUOTA]]
    if len(ok) > QUOTA:
        import hashlib
        seed = int(hashlib.md5(subject.encode()).hexdigest()[:8], 16)
        ok = np.sort(np.random.default_rng(seed).choice(ok, QUOTA, replace=False))
    return list(ok)


def save(out_dir, sid, cohort, group, x, qc, offsets=None):
    """Store the full recording in the parent cache's format, plus any matched offsets."""
    np.save(out_dir / f"{sid}.npy", x.astype(np.float32))
    meta = {"subject": sid, "cohort": cohort, "group": group, "channels": CANON, "qc": qc}
    if offsets is not None:
        meta["match_offsets"] = [int(s) for s in offsets]
    (out_dir / f"{sid}.json").write_text(json.dumps(meta, indent=1, default=float))


def run(job, limit=0):
    paths.ensure_dirs()
    CAUEEGDataset, assign_class, BIDSDataset, harmonise = _parent()

    if job == "vascular":
        out_dir, run_ica, cohort = paths.HARMONISED_EXTRA, True, "CAUEEG"
        ds, added = caueeg_with_excluded(CAUEEGDataset, assign_class)
        work = [(sid, cohort, ds, cls, None) for sid, cls in added.items()]
        print(f"vascular/TGA recovery: {len(work)} subjects "
              f"({sum(c == 'VAD' for c in added.values())} VAD, "
              f"{sum(c == 'TGA' for c in added.values())} TGA)")
    else:
        out_dir, run_ica = paths.HARMONISED_NOICA, False
        # The plain loader drops the subjects the parent taxonomy classes as EXCL, so building the
        # work list from it left vascular dementia and transient global amnesia out of the no-ICA
        # arm entirely. The recovered map is the same one the vascular job uses.
        ds_c, added_c = caueeg_with_excluded(CAUEEGDataset, assign_class)
        ds_b = BIDSDataset(paths.DATASETS / "openneuro_ds004504_raw", dataset_name="ds004504",
                           use_pkl_cache=False, verbose=False)
        caueeg_ids = list(dict.fromkeys(list(ds_c.get_subject_ids()) + list(added_c)))
        work = ([(s, "CAUEEG", ds_c, added_c.get(s), "match") for s in caueeg_ids]
                + [(s, "ds004504_raw", ds_b, None, "match") for s in ds_b.get_subject_ids()])
        print(f"no-ICA arm: {len(work)} subjects (CAUEEG {len(caueeg_ids)}, of which "
              f"{len(added_c)} recovered, plus ds004504)")

    out_dir.mkdir(parents=True, exist_ok=True)
    if limit:
        work = work[:limit]

    done = skipped = failed = 0
    t0 = time.time()
    for k, (sid, cohort, ds, cls, mode) in enumerate(work, 1):
        npy, side = out_dir / f"{sid}.npy", out_dir / f"{sid}.json"
        if npy.exists() and side.exists():
            skipped += 1
            continue
        try:
            offsets = ica_arm_offsets(sid) if mode == "match" else None
            if mode == "match" and offsets is None:
                skipped += 1          # not in the ICA arm, so it has no counterpart to match
                continue
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = ds.load_raw(sid)
                raw, qc = harmonise(raw, run_ica=run_ica)
            x = reorder_to_canon(raw.get_data().astype(np.float64), list(raw.ch_names))
            group = cls if cls else ds.get_subject_info(sid).get("group", "?")
            save(out_dir, sid, cohort, group, x, qc, offsets)
            done += 1
        except Exception as e:                                          # noqa: BLE001
            failed += 1
            print(f"  [fail] {sid}: {type(e).__name__}: {str(e)[:110]}", flush=True)
        if k % 25 == 0 or k == len(work):
            el = time.time() - t0
            print(f"  [{k}/{len(work)}] done {done} skipped {skipped} failed {failed} "
                  f"| {el/60:.1f} min", flush=True)

    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 1e9
    report = {"job": job, "out_dir": str(out_dir), "done": done, "skipped": skipped,
              "failed": failed, "elapsed_min": round((time.time() - t0) / 60, 1),
              "size_gb": round(size_gb, 2), "run_ica": run_ica}
    (paths.RESULTS / f"e09_{job}.json").write_text(json.dumps(report, indent=2))
    print(f"\n{job}: {done} written, {skipped} skipped, {failed} failed, {size_gb:.2f} GB")
    return 0 if failed <= max(2, 0.02 * len(work)) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["vascular", "noica"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    free_gb = None
    try:
        import shutil
        free_gb = shutil.disk_usage(paths.ROOT).free / 1e9
        print(f"free space on target volume: {free_gb:.1f} GB")
    except Exception:                                                   # noqa: BLE001
        pass
    if free_gb is not None and free_gb < 6:
        raise SystemExit(f"refusing to start: only {free_gb:.1f} GB free")
    sys.exit(run(args.job, args.limit))


if __name__ == "__main__":
    main()
