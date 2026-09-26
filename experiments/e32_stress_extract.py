"""Stress the four-electrode design with the acquisition faults a clinic-grade recording never has.

Every other condition in this repository is measured on ICA-cleaned clinical EEG with wet
electrodes placed by a technician. The review's objection is that a low-cost device in a
low-resource clinic records something worse: a higher noise floor, mains pick-up, movement, a lost
contact, electrodes put on by someone untrained, and less time. None of that can be measured
without the device. What can be measured is how fast the recorded information degrades when each
fault is injected at a stated severity into the same recordings, and whether the four-electrode
montage degrades faster than the clinical one. That is a bound on robustness, not a validation.

Faults, each injected at the electrode, before the montage is formed, with a seed fixed per
recording and condition:

    noise{1,2,5,10}   1/f-shaped electrode/amplifier noise, band-limited to the device's 0.5-45 Hz
                      pass-band, at the stated RMS (uV) per electrode. Signal RMS after
                      re-referencing is ~5.6 uV at 19 electrodes and ~3.9 uV at four.
    mains             differential 50 Hz interference (10 uV, electrode gain 0.5-1.5, with a 150 Hz
                      harmonic), and no notch or low-pass: the device that skips its filter.
    motion{2,6}       movement transients per minute, half single-electrode pops (50-150 uV,
                      exponential decay) and half whole-head movement (20-100 uV, 0.5-3 Hz bursts).
    contact_bad       one electrode of the montage loses contact and records mostly noise
                      (0.2 x signal + 20 uV), undetected.
    contact_drop      the same electrode detected and dropped; the montage re-references without it.
    displace{50,100}  untrained placement: every electrode moved a fraction of the way (0.5 or 1.0)
                      toward a random 10-20 neighbour, approximated by linear interpolation.
    short{12,6,3}     only 48, 24 or 12 s of clean signal instead of 96.
    field_moderate    noise 2 uV + motion 2/min + displacement 0.5 + 60 s + 128 Hz, 8-bit.
    field_severe      noise 5 uV + motion 6/min + displacement 1.0 + 24 s + 128 Hz, 8-bit.

Windows are selected on the montage's own signals, as a device would reject them, with the same
200 uV peak-to-peak rule and subject-seeded draw the rest of the study uses; 'clean' applies that
device-side selection to the unaltered recording and is the reference every fault is compared to.

    python experiments/e32_stress_extract.py [--limit N] [--workers 12]
Out: outputs/cache/features_stress.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import budgets, paths, spine  # noqa: E402
from eegbudget.degrade import apply_budget  # noqa: E402
from eegbudget.features import NAMES, subject_features  # noqa: E402
from eegbudget.montages import ALL, CANON, IX, apply_montage, reorder_to_canon  # noqa: E402

SF = budgets.SF_NATIVE
UV = 1e-6
MONTAGES = ("b19", "b4")

CONDITIONS = {
    "clean": {},
    "noise1": {"noise_uv": 1.0}, "noise2": {"noise_uv": 2.0},
    "noise5": {"noise_uv": 5.0}, "noise10": {"noise_uv": 10.0},
    "mains": {"mains_uv": 10.0},
    "motion2": {"motion_per_min": 2.0}, "motion6": {"motion_per_min": 6.0},
    "contact_bad": {"contact": "bad"}, "contact_drop": {"contact": "drop"},
    "displace50": {"displace": 0.5}, "displace100": {"displace": 1.0},
    "short12": {"quota": 12}, "short6": {"quota": 6}, "short3": {"quota": 3},
    "field_moderate": {"noise_uv": 2.0, "motion_per_min": 2.0, "displace": 0.5, "quota": 15,
                       "rate": 128, "bits": 8},
    "field_severe": {"noise_uv": 5.0, "motion_per_min": 6.0, "displace": 1.0, "quota": 6,
                     "rate": 128, "bits": 8},
}

#: 10-20 adjacency: the sites an electrode lands nearest when it is put on in the wrong place.
NEIGHBOURS = {
    "Fp1": ["Fp2", "F7", "F3"], "Fp2": ["Fp1", "F4", "F8"],
    "F7": ["Fp1", "F3", "T3"], "F3": ["Fp1", "F7", "Fz", "C3"], "Fz": ["F3", "F4", "Cz"],
    "F4": ["Fp2", "Fz", "F8", "C4"], "F8": ["Fp2", "F4", "T4"],
    "T3": ["F7", "C3", "T5"], "C3": ["F3", "T3", "Cz", "P3"], "Cz": ["Fz", "C3", "C4", "Pz"],
    "C4": ["F4", "Cz", "T4", "P4"], "T4": ["F8", "C4", "T6"],
    "T5": ["T3", "P3", "O1"], "P3": ["C3", "T5", "Pz", "O1"], "Pz": ["Cz", "P3", "P4"],
    "P4": ["C4", "Pz", "T6", "O2"], "T6": ["T4", "P4", "O2"],
    "O1": ["T5", "P3", "O2"], "O2": ["T6", "P4", "O1"],
}
PASS_BAND = (0.5, 45.0)


def _rng(subject, tag):
    return np.random.default_rng(int(hashlib.md5(f"{subject}|{tag}".encode()).hexdigest()[:8], 16))


def _band_noise(rng, shape, rms_v, exponent=1.0):
    """Gaussian noise with a 1/f**exponent power spectrum inside the pass-band, scaled to rms_v."""
    n = shape[-1]
    spec = rng.standard_normal(shape[:-1] + (n // 2 + 1,)) \
        + 1j * rng.standard_normal(shape[:-1] + (n // 2 + 1,))
    f = np.fft.rfftfreq(n, 1.0 / SF)
    gain = np.zeros_like(f)
    band = (f >= PASS_BAND[0]) & (f <= PASS_BAND[1])
    gain[band] = f[band] ** (-exponent / 2.0)
    x = np.fft.irfft(spec * gain, n=n, axis=-1)
    x /= x.std(axis=-1, keepdims=True) + 1e-30
    return x * rms_v


def _band_limit(x):
    """The device's own zero-phase 0.5-45 Hz filter, applied to an injected artefact."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / SF)
    X[..., (f < PASS_BAND[0]) | (f > PASS_BAND[1])] = 0
    return np.fft.irfft(X, n=x.shape[-1], axis=-1)


def _mains(rng, n_ch, n, amp_uv):
    t = np.arange(n) / SF
    gain = rng.uniform(0.5, 1.5, (n_ch, 1)) * amp_uv * UV
    phase = rng.uniform(-0.3, 0.3, (n_ch, 1))
    return gain * (np.sin(2 * np.pi * 50 * t + phase) + 0.3 * np.sin(2 * np.pi * 150 * t + 3 * phase))


def _motion(rng, n_ch, n, per_min):
    art = np.zeros((n_ch, n))
    n_ev = rng.poisson(per_min * n / SF / 60.0)
    for _ in range(n_ev):
        start = int(rng.integers(0, max(1, n - 2 * SF)))
        if rng.random() < 0.5:            # electrode pop: a step that decays
            L = SF
            tt = np.arange(L) / SF
            w = rng.choice([-1, 1]) * rng.uniform(50, 150) * UV * np.exp(-tt / rng.uniform(0.1, 0.5))
            ch = int(rng.integers(0, n_ch))
            seg = min(L, n - start)
            art[ch, start:start + seg] += w[:seg]
        else:                             # head movement: a slow burst on every electrode
            L = int(rng.uniform(0.5, 2.0) * SF)
            tt = np.arange(L) / SF
            w = np.hanning(L) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * tt + rng.uniform(0, 6.28))
            g = rng.uniform(0.3, 1.0, (n_ch, 1)) * rng.uniform(20, 100) * UV * rng.choice([-1, 1])
            seg = min(L, n - start)
            art[:, start:start + seg] += g * w[:seg]
    return _band_limit(art)


def _device_montage(base, sites):
    """A subset-referenced montage over the named sites (what a device with a dropped lead records)."""
    m = dict(base)
    m["channels"] = sorted(sites, key=IX.get)
    m["ref"] = "subset_car"
    m["pairs"] = []
    return m


def degrade(x, subject, cond_name, cond, montage_name):
    """Return (windows, sampling rate, montage used) for one recording, condition and montage."""
    base = ALL[montage_name]
    x = x.copy()
    n = x.shape[-1]
    # Electrode-level additive faults: the same draw for every montage, as the scalp is shared.
    if "noise_uv" in cond:
        x += _band_noise(_rng(subject, f"{cond_name}|noise"), x.shape, cond["noise_uv"] * UV)
    if "mains_uv" in cond:
        x += _mains(_rng(subject, f"{cond_name}|mains"), x.shape[0], n, cond["mains_uv"])
    if "motion_per_min" in cond:
        x += _motion(_rng(subject, f"{cond_name}|motion"), x.shape[0], n, cond["motion_per_min"])

    sites = list(base["channels"])
    if "displace" in cond:
        rng = _rng(subject, f"{cond_name}|displace")
        a = cond["displace"]
        src = x.copy()
        for s in CANON:                   # every site draws, so b4 and b19 share each electrode's move
            j = rng.choice(NEIGHBOURS[s])
            if s in sites:
                x[IX[s]] = (1 - a) * src[IX[s]] + a * src[IX[j]]
    m = base
    if "contact" in cond:
        lost = _rng(subject, f"contact|{montage_name}").choice(sites)
        if cond["contact"] == "bad":
            x[IX[lost]] = 0.2 * x[IX[lost]] + _band_noise(_rng(subject, "contact|noise"),
                                                         (n,), 20 * UV)
        else:
            m = _device_montage(base, [s for s in sites if s != lost])

    sig = apply_montage(x, m)
    w, _ = budgets.select_windows(sig, quota=cond.get("quota", budgets.QUOTA), subject=subject)
    if len(w) == 0:
        return None, SF, m
    deg, sf = apply_budget(w, SF, cond.get("rate", SF), cond.get("bits"))
    return deg, sf, m


def _work(args):
    i, subject, npy = args
    warnings.simplefilter("ignore")
    meta = json.loads(Path(npy).with_suffix(".json").read_text())
    x = reorder_to_canon(np.load(npy).astype(np.float64), meta["channels"])
    out = np.full((len(CONDITIONS), len(MONTAGES), len(NAMES)), np.nan)
    for ci, (cn, c) in enumerate(CONDITIONS.items()):
        for mi, mn in enumerate(MONTAGES):
            w, sf, m = degrade(x, subject, cn, c, mn)
            if w is not None:
                out[ci, mi] = subject_features(w, sf, m)
    return i, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    paths.ensure_dirs()

    df = spine.load()
    df = df[(df["cohort"] == "CAUEEG") & (df["in_screening"] == 1)].reset_index(drop=True)
    if args.limit:
        df = df.sample(args.limit, random_state=0).reset_index(drop=True)
    X = np.full((len(CONDITIONS), len(MONTAGES), len(df), len(NAMES)), np.nan)
    jobs = [(i, r["subject"], r["npy"]) for i, r in df.iterrows()]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (i, out) in enumerate(ex.map(_work, jobs, chunksize=4)):
            X[:, :, i] = out
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    dst = paths.CACHE / ("features_stress.npz" if not args.limit else "features_stress_smoke.npz")
    np.savez_compressed(dst, X=X, conditions=np.array(list(CONDITIONS)),
                        montages=np.array(MONTAGES), names=np.array(NAMES),
                        subjects=df["subject"].to_numpy(dtype=object),
                        spec=json.dumps(CONDITIONS))
    print(f"wrote {dst} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
