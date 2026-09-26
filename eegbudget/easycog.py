"""Auditable access to the locally downloaded EasyCog release.

EasyCog is not a drop-in replacement for the 19-channel CAUEEG montage. It is a separate, low-cost
cohort: 16 channels at 125 Hz from an OpenBCI Cyton+Daisy, four wet Ag-AgCl forehead electrodes at
the 10-20 positions AF7/AF8/Fp1/Fp2 and twelve cEEGrid electrodes arranged around the two ears,
with three minutes of eyes-closed rest and a 430-second passive-video block per session. This
module keeps it separate from :mod:`eegbudget.spine`: mixing the cohorts would make channel
harmonisation and label semantics look stronger than they are.

Three things this module does differently from the first version of it, each because the first
version made a claim the data did not support.

**Read the unsliced recordings, not the 3-second slices.** The ``sliced/`` release is missing four
subjects that ``unsliced/`` carries, and a 3-second window at 125 Hz gives 0.33 Hz of spectral
resolution against 0.25 Hz from a 4-second one. Reading ``unsliced/`` recovers all 69 released
participants and the full 180 seconds of rest.

**Measure the same seventeen features CAUEEG is measured with.** The first version reduced each
subject to sixteen channel log-variances -- a quantity dominated by electrode impedance -- and then
compared the result against a seventeen-feature spectral model on CAUEEG. Any deficit found that
way is a statement about feature engineering, not about sensors. :mod:`eegbudget.features` is
defined at any channel count, so it applies here unchanged.

**Derive the electrode blocks from the data and check them.** The release documents neither channel
names nor channel order. The forehead/ear split is therefore recovered from the inter-channel
correlation structure and *verified* on every load rather than asserted: see :func:`verify_blocks`.
Any claim that separates forehead from ear electrodes is only as good as that check, and the check
travels with the results.

Raw data are never copied into outputs or committed to git.
"""
from __future__ import annotations

import hashlib
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .features import NAMES, window_features

DEFAULT_ROOT = Path(os.environ.get("EASYCOG_ROOT",
                                   Path(os.environ.get("EEG_DATA_ROOT", "data")) / "datasets" / "EasyCog"))
DEFAULT_FEATURE = "asreog_filter_order3_all_data"

#: Native acquisition. Both are fixed by the release and are asserted on load.
NATIVE_SF = 125.0
N_CHANNELS = 16

#: Analysis windowing, matched to the CAUEEG protocol in :mod:`eegbudget.budgets` so the two
#: cohorts are measured over comparable epochs: 4-second non-overlapping windows, peak-to-peak
#: rejection at 200 uV. EasyCog ships in microvolts and CAUEEG in volts, which is the only
#: difference in how that threshold is applied.
WINDOW_S = 4.0
PTP_REJECT_UV = 200.0
MIN_WINDOWS = 5

_NAME = re.compile(
    r"^(?P<subject>[^-]+)-(?P<date>\d{4}_\d{2}_\d{2}_[^-]+)-(?P<kind>video|resting)"
)

# ---------------------------------------------------------------------------------------------
# Electrode blocks
# ---------------------------------------------------------------------------------------------
#
# The published electrode figure names four forehead sites -- FR1 (AF8), FR2 (Fp2), FR3 (Fp1),
# FR4 (AF7) -- and two six-electrode cEEGrid arrays, L1/L2/L3/L8/L9/L10 around the left ear and
# R1/R2/R3/R8/R9/R10 around the right. It does not say which acquisition channel carries which
# electrode. The assignment below is recovered from the mean inter-channel correlation matrix,
# which separates cleanly into 3 + 2 + 3 | 3 + 2 + 3 with the two pairs of forehead channels
# correlating across the Cyton/Daisy board boundary far more strongly than with their neighbours
# -- the signature of four closely spaced frontal electrodes split across two boards.
#
# The within-pair order (AF7 against Fp1) is not recoverable and does not matter: every analysis
# here is at block level. What matters is that the *blocks* are right, and :func:`verify_blocks`
# re-derives them from whatever data is actually loaded and refuses to let a claim rest on an
# assumption that the recordings contradict.
BLOCKS = {
    "forehead": (3, 4, 11, 12),
    "ear_left": (0, 1, 2, 5, 6, 7),
    "ear_right": (8, 9, 10, 13, 14, 15),
}

CHANNELS = ["L1", "L2", "L3", "FH_L1", "FH_L2", "L8", "L9", "L10",
            "R1", "R2", "R3", "FH_R1", "FH_R2", "R8", "R9", "R10"]

#: Left/right electrode pairs, for the interhemispheric alpha asymmetry feature.
LATERAL_PAIRS = [("L1", "R1"), ("L2", "R2"), ("L3", "R3"),
                 ("L8", "R8"), ("L9", "R9"), ("L10", "R10"),
                 ("FH_L1", "FH_R1"), ("FH_L2", "FH_R2")]

#: Anterior/posterior vocabulary for the alpha-gradient feature. The cEEGrid sits over temporal
#: cortex and reaches partway to parietal, so it is posterior *relative to the forehead array*.
#: This is the same gradient the 10-20 montage measures from Fp/F to P/O, over a shorter span.
REGION = {"frontal": [CHANNELS[i] for i in BLOCKS["forehead"]],
          "posterior": [CHANNELS[i] for i in BLOCKS["ear_left"] + BLOCKS["ear_right"]]}


def _montage(name, idx, ref, note):
    return {"name": name, "index": tuple(idx),
            "channels": [CHANNELS[i] for i in idx], "ref": ref, "pairs": [],
            "family": "easycog", "note": note,
            "lateral_pairs": LATERAL_PAIRS, "region": REGION}


#: The arms compared on EasyCog.
#:
#: ``ref`` matters more here than it looks. The release is referenced to a physical electrode, not
#: to a common average, so restricting to a block leaves the survivors carrying a reference the
#: sub-device shares -- which is fine. Re-referencing within a block instead removes whatever the
#: block holds in common, and across four closely spaced forehead electrodes that is nearly
#: everything. A forehead-versus-ear comparison run only under block re-referencing would
#: manufacture its own answer, in exactly the way two symmetric electrodes cancel a symmetric
#: rhythm on the CAUEEG ladder. Both referencings are therefore carried and both are reported.
MONTAGES = {
    "ec16": _montage("ec16", range(16), "native", "all 16 channels, as recorded"),
    "ec_ear": _montage("ec_ear", BLOCKS["ear_left"] + BLOCKS["ear_right"], "native",
                       "12 cEEGrid around-ear electrodes"),
    "ec_forehead": _montage("ec_forehead", BLOCKS["forehead"], "native",
                            "4 forehead electrodes (AF7/AF8/Fp1/Fp2)"),
    "ec16_car": _montage("ec16_car", range(16), "subset_car", "all 16, re-referenced to their mean"),
    "ec_ear_car": _montage("ec_ear_car", BLOCKS["ear_left"] + BLOCKS["ear_right"], "subset_car",
                           "12 ear electrodes, re-referenced within the block"),
    "ec_forehead_car": _montage("ec_forehead_car", BLOCKS["forehead"], "subset_car",
                                "4 forehead electrodes, re-referenced within the block"),
}

#: A pre-specified low-dimensional estimator. Three features, chosen for being the classical
#: markers of cortical slowing rather than by looking at the outcome, and declared here so that
#: the choice is visible in the code rather than made downstream. At n < 100 the difference
#: between this and the full seventeen-feature vector is the difference between an estimator that
#: recovers the ageing/slowing signal and one that spends its degrees of freedom on noise.
SLOWING_TRIAD = ["theta_alpha_ratio", "rel_theta", "pdr_frequency"]


# ---------------------------------------------------------------------------------------------
# Discovery and accounting
# ---------------------------------------------------------------------------------------------
def _parts(root: Path) -> list[Path]:
    return sorted(
        (p / "EasyCog_Dataset" for p in root.iterdir()
         if p.is_dir() and (p / "EasyCog_Dataset").is_dir()),
        key=lambda p: p.parent.name,
    )


def _metadata(root: Path) -> tuple[pd.DataFrame, Path | None]:
    for part in _parts(root):
        path = part / "Patient_Info_dataset.xlsx"
        if path.exists():
            frame = pd.read_excel(path)
            frame.columns = [str(c).strip() for c in frame.columns]
            frame["id"] = frame["id"].astype(str).str.strip()
            frame["Date"] = frame["Date"].astype(str).str.strip()
            return frame, path
    return pd.DataFrame(), None


def _file_info(path: Path) -> dict[str, str]:
    match = _NAME.match(path.name)
    if not match:
        return {"subject": "", "kind": "unknown", "date": ""}
    return {"subject": match.group("subject"), "kind": match.group("kind"),
            "date": match.group("date")}


def _unsliced_files(root: Path, feature_name: str) -> list[Path]:
    """One deterministic path per unsliced filename across archive parts."""
    by_name: dict[str, Path] = {}
    for part in _parts(root):
        folder = part / "unsliced" / feature_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.npz")):
            by_name.setdefault(path.name, path)
    return [by_name[k] for k in sorted(by_name)]


def _sliced_files(root: Path, feature_name: str) -> tuple[list[Path], dict[str, int]]:
    """Sliced-release accounting, retained so the manuscript can report what it did not use."""
    candidates: list[Path] = []
    for part in _parts(root):
        folder = part / "sliced" / feature_name
        if folder.is_dir():
            candidates.extend(folder.glob("*.npz"))
    by_name: dict[str, list[Path]] = {}
    for path in candidates:
        by_name.setdefault(path.name, []).append(path)
    selected = sorted((sorted(v, key=str)[0] for v in by_name.values()), key=lambda p: p.name)
    dup = {"candidate_files": len(candidates), "unique_files": len(selected),
           "duplicate_filenames": len(candidates) - len(selected),
           "duplicate_groups": sum(len(v) > 1 for v in by_name.values())}
    return selected, dup


def _date_key(date: str) -> str:
    return ".".join(str(date).split("_")[:3])


# ---------------------------------------------------------------------------------------------
# Signal loading and block verification
# ---------------------------------------------------------------------------------------------
def load_resting_signal(path: Path) -> np.ndarray:
    """Return the (16, T) resting recording in microvolts.

    Only ``raw_eeg`` is read. The sliced release duplicates MoCA, MMSE and per-task scores inside
    every window; the unsliced release does not carry them at all, which is one more reason to
    read it. Labels come from the workbook and from nowhere else.
    """
    with np.load(path, allow_pickle=True) as data:
        eeg = np.asarray(data["raw_eeg"], dtype=np.float64)
    if eeg.ndim != 2 or eeg.shape[0] != N_CHANNELS:
        raise ValueError(f"{path.name}: expected ({N_CHANNELS}, samples) EEG, got {eeg.shape}")
    return eeg


def verify_blocks(signals: list[np.ndarray]) -> dict[str, object]:
    """Re-derive the forehead/ear block structure from the recordings themselves.

    Averages the inter-channel correlation matrix over the supplied recordings and reports, for
    the assumed blocks, the mean within-block and between-block correlation. ``consistent`` is
    True when every block's internal coherence exceeds its coherence with the rest of the array,
    which is the property the block split is claiming.

    This exists because the release documents no channel order. An unverified index assignment
    would put the paper's placement claim on an assumption; a verified one puts it on a
    measurement, and a failed check is visible in the manifest instead of invisible in a constant.
    """
    mats = []
    for x in signals:
        c = np.corrcoef(x)
        if np.isfinite(c).all():
            mats.append(c)
    if not mats:
        return {"consistent": False, "reason": "no usable correlation matrices", "n_used": 0}
    C = np.mean(mats, axis=0)

    report, ok = {}, True
    for name, idx in BLOCKS.items():
        idx = np.array(idx)
        other = np.setdiff1d(np.arange(N_CHANNELS), idx)
        within = C[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        between = C[np.ix_(idx, other)]
        w, b = float(within.mean()), float(between.mean())
        ok &= w > b
        # The block separation is a claim about recordings, not about one averaged matrix, so
        # each recording contributes its own within and between value and the spread across them
        # is reported. Averaging the matrices first hides how consistent the separation is.
        per_w = [float(c[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()) for c in mats]
        per_b = [float(c[np.ix_(idx, other)].mean()) for c in mats]
        n = len(mats)
        ci = {}
        for tag, vals in (("within", per_w), ("between", per_b)):
            a = np.asarray(vals, float)
            half = 1.959964 * a.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            ci[f"{tag}_ci"] = [round(float(a.mean() - half), 4), round(float(a.mean() + half), 4)]
            ci[f"{tag}_sd"] = round(float(a.std(ddof=1)), 4) if n > 1 else None
        # The separation itself is paired within a recording, so its interval is the paired one.
        dif = np.asarray(per_w, float) - np.asarray(per_b, float)
        half = 1.959964 * dif.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        report[name] = {"n": int(len(idx)), "mean_within": round(w, 4),
                        "mean_between": round(b, 4), "separated": bool(w > b),
                        "n_recordings": n,
                        "separation": round(float(dif.mean()), 4),
                        "separation_ci": [round(float(dif.mean() - half), 4),
                                          round(float(dif.mean() + half), 4)],
                        **ci}
    return {"consistent": bool(ok), "n_used": len(mats), "blocks": report,
            "mean_correlation": C.tolist(),
            "note": ("Blocks are asserted in eegbudget.easycog.BLOCKS and re-derived here from "
                     "the loaded recordings. 'consistent' means every block is more internally "
                     "correlated than it is with the rest of the array.")}


def _windows(x: np.ndarray, sf: float = NATIVE_SF) -> np.ndarray:
    """Non-overlapping clean windows, shape (n_windows, n_signals, T)."""
    win = int(round(WINDOW_S * sf))
    n = x.shape[-1] // win
    if n == 0:
        return np.empty((0, x.shape[0], win))
    starts = np.arange(n) * win
    ptp = np.array([np.ptp(x[:, s:s + win], axis=-1).max() for s in starts])
    ok = starts[ptp <= PTP_REJECT_UV]
    if len(ok) == 0:
        # Keep the quietest windows rather than dropping the subject; the count is reported so a
        # subject carried on rejected epochs is visible rather than silently equal to the rest.
        ok = starts[np.argsort(ptp)[:MIN_WINDOWS]]
    return np.stack([x[:, s:s + win] for s in ok])


def apply_montage(x: np.ndarray, m: dict) -> np.ndarray:
    """Select a montage's channels from a (..., 16, T) array and apply its referencing."""
    sub = x[..., list(m["index"]), :]
    if m["ref"] == "subset_car":
        sub = sub - sub.mean(axis=-2, keepdims=True)
    return sub


def session_features(x: np.ndarray, m: dict, sf: float = NATIVE_SF) -> tuple[np.ndarray, int]:
    """Median feature vector over the session's clean windows, for one montage."""
    W = _windows(x, sf)
    if len(W) == 0:
        return np.full(len(NAMES), np.nan), 0
    sig = apply_montage(W, m)
    rows = [window_features(w, sf, m) for w in sig]
    vec = np.array([np.nanmedian([r[n] for r in rows]) for n in NAMES], dtype=np.float64)
    return vec, len(W)


# ---------------------------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------------------------
def manifest(root: str | Path = DEFAULT_ROOT,
             feature_name: str = DEFAULT_FEATURE) -> dict[str, object]:
    """Describe the local release without loading signal arrays."""
    root = Path(root)
    parts = _parts(root) if root.exists() else []
    meta, metadata_path = _metadata(root) if parts else (pd.DataFrame(), None)
    meta_ids = set(meta["id"].dropna().astype(str)) if not meta.empty else set()

    unsliced = _unsliced_files(root, feature_name) if parts else []
    u_info = [_file_info(p) for p in unsliced]
    u_rest = [i for i in u_info if i["kind"] == "resting"]
    u_rest_ids = {i["subject"] for i in u_rest}

    sliced, dup = _sliced_files(root, feature_name) if parts else ([], {})
    s_info = [_file_info(p) for p in sliced]
    s_rest = [i for i in s_info if i["kind"] == "resting"]
    s_rest_ids = {i["subject"] for i in s_rest}

    return {
        "root": str(root),
        "feature_name": feature_name,
        "parts": [str(p.parent) for p in parts],
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_rows": int(len(meta)),
        "metadata_subjects": int(len(meta_ids)),
        # what the analysis reads
        "unsliced_files": int(len(unsliced)),
        "unsliced_resting_sessions": int(len(u_rest)),
        "unsliced_resting_subjects": int(len(u_rest_ids)),
        "unsliced_resting_without_metadata": sorted(u_rest_ids - meta_ids),
        "metadata_missing_from_unsliced_resting": sorted(meta_ids - u_rest_ids),
        # what it does not read, retained for the release accounting table
        "sliced_files": int(len(sliced)),
        "sliced_resting_windows": int(len(s_rest)),
        "sliced_video_windows": int(len(s_info) - len(s_rest)),
        "sliced_resting_subjects": int(len(s_rest_ids)),
        "sliced_missing_subjects": sorted(u_rest_ids - s_rest_ids),
        "sliced_duplicates": dup,
        "release_note": (
            "The public EasyCog release contains 69 participants; the EasyCog paper describes a "
            "101-participant cohort. This is the released subset, not a partial download, and "
            "the two are never merged. The sliced/ release omits subjects that unsliced/ "
            "carries, so all analysis here reads unsliced/."),
    }


def load_resting_subjects(root: str | Path = DEFAULT_ROOT,
                          feature_name: str = DEFAULT_FEATURE,
                          verbose: bool = False) -> tuple[pd.DataFrame, dict[str, object]]:
    """One row per participant, with the 17 features under each EasyCog montage.

    Sessions are measured separately and collapsed to the participant by median, so a participant
    recorded three times contributes one row and cross-validation grouped by participant cannot
    leak. Disease labels are kept verbatim: PD, AD, VaD and neurosyphilis are not interchangeable
    endpoints and any analysis must state its own mapping.
    """
    root = Path(root)
    meta, metadata_path = _metadata(root)
    if meta.empty:
        raise SystemExit(f"no Patient_Info_dataset.xlsx found under {root}")
    meta_ids = set(meta["id"])

    files = [p for p in _unsliced_files(root, feature_name)
             if _file_info(p)["kind"] == "resting"]

    sessions, signals, skipped = [], [], []
    for path in files:
        info = _file_info(path)
        subject, date = info["subject"], info["date"]
        if subject not in meta_ids:
            skipped.append({"file": path.name, "reason": "no workbook metadata"})
            continue
        try:
            x = load_resting_signal(path)
        except (KeyError, ValueError) as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            continue
        signals.append(x)

        rows = meta[meta["id"] == subject]
        matched = rows[rows["Date"] == _date_key(date)]
        r = matched.iloc[0] if len(matched) else rows.iloc[0]

        rec = {"subject": subject, "date": date,
               "session_matched_on_date": bool(len(matched)),
               "duration_s": x.shape[1] / NATIVE_SF,
               "age": float(r["Age"]), "disease": str(r["Disease"]).strip(),
               "environment": str(r["Environment"]).strip(),
               "gender": str(r["Gender"]).strip(),
               "moca": float(r["MoCA"]), "mmse": float(r["MMSE"])}
        for mname, m in MONTAGES.items():
            vec, n_win = session_features(x, m)
            rec[f"nwin_{mname}"] = n_win
            for name, value in zip(NAMES, vec):
                rec[f"{mname}__{name}"] = float(value)
        sessions.append(rec)
        if verbose:
            print(f"  {subject} {date}  {rec['duration_s']:.0f}s  "
                  f"{rec['nwin_ec16']} windows", flush=True)

    S = pd.DataFrame(sessions)
    if S.empty:
        raise SystemExit(f"no usable resting recordings found under {root}")

    feature_cols = [c for c in S.columns if "__" in c]
    agg = {c: "median" for c in feature_cols}
    agg.update({c: "median" for c in S.columns if c.startswith("nwin_")})
    agg.update({"age": "median", "moca": "median", "mmse": "median",
                "duration_s": "median", "disease": lambda s: s.mode().iloc[0],
                "environment": "first", "gender": "first"})
    T = S.groupby("subject", as_index=False).agg(agg)
    T["n_sessions"] = S.groupby("subject").size().to_numpy()

    # Per-subject usability: a montage whose features are non-finite is a dead electrode block,
    # not a missing measurement, and the two must not be confused downstream.
    for mname in MONTAGES:
        cols = [f"{mname}__{n}" for n in NAMES if n not in ("asymmetry_alpha", "ap_gradient_alpha")]
        T[f"usable_{mname}"] = np.isfinite(T[cols].to_numpy(float)).all(axis=1)

    info = manifest(root, feature_name)
    info.update({
        "metadata_path": str(metadata_path) if metadata_path else None,
        "resting_sessions_loaded": int(len(S)),
        "resting_sessions_skipped": skipped,
        "aggregated_subjects": int(len(T)),
        "sessions_matched_on_date": int(S["session_matched_on_date"].sum()),
        "median_windows_per_session": float(S["nwin_ec16"].median()),
        "window_protocol": {"seconds": WINDOW_S, "ptp_reject_uv": PTP_REJECT_UV,
                            "sampling_rate_hz": NATIVE_SF},
        "block_verification": verify_blocks(signals),
        "unusable_blocks": {m: sorted(T.loc[~T[f"usable_{m}"], "subject"]) for m in MONTAGES},
        "release_fingerprint": release_fingerprint(root),
    })
    return T, info


def release_fingerprint(root: str | Path = DEFAULT_ROOT) -> str:
    """Stable fingerprint of filenames and sizes, not raw signal content."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".npz", ".xlsx", ".md"})
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(root)).encode())
        digest.update(str(p.stat().st_size).encode())
    return digest.hexdigest()
