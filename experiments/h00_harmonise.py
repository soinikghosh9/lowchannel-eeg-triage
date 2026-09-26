"""Harmonise the four EEG cohorts the paper uses into one cache.

Every recording goes through the same pipeline (``harmonisation/harmonise.py``): mains detection and
notch, 0.5-45 Hz zero-phase band-pass, common-average reference, extended-infomax ICA with ICLabel
(eye, heart, line and channel components removed, seed fixed), resampling to 256 Hz. The cache stores
the 19 10-20 channels unscaled, in volts, one ``.npy`` per recording with a ``.json`` sidecar holding
the channel list, the diagnostic group and the QC record.

    CAUEEG     caueeg-dataset/                  Kim et al. 2023 (on request from its maintainers)
    ds004504   openneuro_ds004504_raw/          Miltiadous et al. 2023, OpenNeuro, raw (not derivatives)
    BrainLat   BrainLat Data/                   Prado et al. 2023, Synapse (128-electrode cap; the
                                                cap electrode nearest each 10-20 site is kept)
    P-ADIC     P-ADIC/p-adic alz, p-adic ctrl/  Shor et al. 2021, Dryad (AD and controls)

CAUEEG's vascular-dementia and transient-global-amnesia recordings, and the no-ICA arm, are made by
``e09_extend_corpus.py``. Resumable: a recording whose ``.npy`` and sidecar exist is skipped.

    python experiments/h00_harmonise.py [--cohorts CAUEEG,ds004504,BrainLat,P-ADIC] [--limit N]
                                        [--out DIR]
Out: <EEG_DATA_ROOT>/harmonised/<subject>.{npy,json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import paths  # noqa: E402

PADIC_FILES = {"AD": ("p-adic alz", "alz_c1_new.mat"), "CN": ("p-adic ctrl", "controls_c1_new.mat")}


def _items(cohort, limit):
    """Yield (subject id as cached, group, loader callable) for one cohort."""
    D = paths.DATASETS
    if cohort == "CAUEEG":
        from harmonisation.caueeg_loader import CAUEEGDataset
        ds = CAUEEGDataset(D / "caueeg-dataset", verbose=False)
        sids = list(ds.get_subject_ids())[:limit or None]
        return "CAUEEG", [(s, ds.get_subject_info(s).get("group", "?"), lambda s=s: ds.load_raw(s))
                          for s in sids]
    if cohort == "ds004504":
        from harmonisation.bids_loader import BIDSDataset
        # Raw files and no pickle cache: the derivatives are already cleaned by their curators, and
        # cleaning twice would make this cohort differ from every other.
        ds = BIDSDataset(D / "openneuro_ds004504_raw", dataset_name="ds004504",
                         use_pkl_cache=False, verbose=False)
        sids = list(ds.get_subject_ids())[:limit or None]
        return "ds004504_raw", [(s, ds.get_subject_info(s).get("group", "?"),
                                 lambda s=s: ds.load_raw(s)) for s in sids]
    if cohort == "BrainLat":
        from harmonisation.brainlat_loader import BrainLatLoader
        L = BrainLatLoader(D / "BrainLat Data")
        out = []
        for grp in ("AD", "CN", "FTD"):
            for sub in L.get_subject_ids(group=grp)[:limit or None]:
                out.append((f"BRAINLAT_{grp}_{sub}", grp, lambda sub=sub: L.load_raw(sub)))
        return "BrainLat", out
    if cohort == "P-ADIC":
        from harmonisation.padic_group_loader import PADICGroupDataset
        out = []
        for grp, (folder, name) in PADIC_FILES.items():
            ds = PADICGroupDataset(D / "P-ADIC" / folder / name, grp, f"PADIC_{grp}")
            for sid in ds.get_subject_ids()[:limit or None]:
                out.append((sid, grp, lambda sid=sid, ds=ds: ds.load_raw(sid)))
        return "P-ADIC", out
    raise SystemExit(f"unknown cohort {cohort!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="CAUEEG,ds004504,BrainLat,P-ADIC")
    ap.add_argument("--limit", type=int, default=0, help="recordings per cohort (and group)")
    ap.add_argument("--out", default="", help="cache directory (default: the configured one)")
    args = ap.parse_args()

    import mne
    mne.set_log_level("ERROR")
    from harmonisation import scaling
    from harmonisation.harmonise import harmonise
    scaling.set_mode("none")                 # the cache stores unscaled volts

    out = Path(args.out) if args.out else paths.HARMONISED
    out.mkdir(parents=True, exist_ok=True)
    for cohort in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        name, items = _items(cohort, args.limit)
        print(f"\n{cohort}: {len(items)} recordings -> {out}", flush=True)
        t0, done, skipped, failed = time.time(), 0, 0, 0
        for k, (sid, grp, load) in enumerate(items, 1):
            npy, side = out / f"{sid}.npy", out / f"{sid}.json"
            if npy.exists() and side.exists():
                skipped += 1
                continue
            try:
                raw, qc = harmonise(load(), run_ica=True)
                np.save(npy, raw.get_data().astype(np.float32))
                side.write_text(json.dumps({"subject": sid, "cohort": name, "group": grp,
                                            "channels": raw.ch_names, "qc": qc},
                                           indent=1, default=float))
                done += 1
            except Exception as e:                                       # noqa: BLE001
                failed += 1
                print(f"  [fail] {sid}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            if k % 25 == 0 or k == len(items):
                print(f"  {k}/{len(items)} done {done} skipped {skipped} failed {failed} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
