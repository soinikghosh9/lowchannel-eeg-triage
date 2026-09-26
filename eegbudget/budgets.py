"""The acquisition budgets swept, and the windowing protocol shared by all of them.

A budget is a montage crossed with a digitisation setting and a preprocessing arm. The full cross
product is not swept: montage effects are measured at native digitisation, and digitisation
effects at two montages -- the clinical reference and the four-electrode device that a deployment
would actually use. That keeps the grid at 45 conditions rather than several hundred without
losing any comparison the paper makes.

Windowing follows the parent project so that budgets are the only thing that differs from its
published corpus: 4-second windows, no overlap, a fixed quota per subject chosen by a generator
seeded from the subject identifier, and rejection of windows exceeding 200 uV peak-to-peak
applied to the unscaled signal in volts, which is the only order in which that threshold means
what it says.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .degrade import PREPROC
from .montages import ALL

WINDOW_SAMPLES = 1024      # 4 s at 256 Hz
QUOTA = 24                 # 96 s of analysed signal per subject
SF_NATIVE = 256
PTP_REJECT_V = 200e-6


def select_windows(x, quota=QUOTA, subject="", win=WINDOW_SAMPLES):
    """Deterministically choose up to `quota` clean windows from a (19, T) recording.

    Returns the stacked windows and the number rejected, which is itself an artefact-burden
    measure and is recorded because it can separate diagnostic classes on its own.
    """
    n = x.shape[-1] // win
    if n == 0:
        return np.empty((0, x.shape[0], win), x.dtype), 0
    starts = np.arange(n) * win
    ptp = np.array([np.ptp(x[:, s:s + win], axis=-1).max() for s in starts])
    ok = starts[ptp <= PTP_REJECT_V]
    rejected = n - len(ok)
    if len(ok) == 0:
        ok, rejected = starts[np.argsort(ptp)[:quota]], n - min(n, quota)
    seed = int(hashlib.md5(subject.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if len(ok) > quota:
        ok = np.sort(rng.choice(ok, quota, replace=False))
    return np.stack([x[:, s:s + win] for s in ok]), rejected


def grid():
    """The conditions swept in E1. Each is a dict the extractor can execute directly."""
    out = []

    # Axis A -- electrode budget at a clinical amplifier.
    for name in ("b19", "b8", "b4", "b2", "b1", "b1_ap",
                 "r_frontal", "r_temporal", "r_central", "r_posterior",
                 "muse", "ganglion", "insight", "frontal1",
                 "b4_inherit", "b8_inherit"):
        out.append({"montage": name, "rate": 256, "bits": None, "preproc": "full", "axis": "electrodes"})

    # Axis B -- digitisation, at the clinical reference and at the deployable device.
    for m in ("b19", "b4"):
        for rate in (256, 128, 64):
            for bits in (None, 16, 12, 8):
                if rate == 256 and bits is None:
                    continue                      # already in axis A
                out.append({"montage": m, "rate": rate, "bits": bits,
                            "preproc": "full", "axis": "digitisation"})

    # Axis C -- preprocessing budget.
    for m in ("b19", "b4"):
        for p in ("emgfree", "noica"):
            out.append({"montage": m, "rate": 256, "bits": None, "preproc": p,
                        "axis": "preprocessing"})
    return out


def key(c):
    return f"{c['montage']}|{c['rate']}|{c['bits'] or 'f32'}|{c['preproc']}"


def describe(c):
    m = ALL[c["montage"]]
    n_el = len(m["channels"])
    bits = "float32" if c["bits"] is None else f"{c['bits']}-bit"
    return f"{c['montage']} ({n_el} el), {c['rate']} Hz, {bits}, {c['preproc']}"


def cache_for(c):
    """Which harmonised cache a condition reads. 'noica' needs a separately generated arm."""
    return PREPROC[c["preproc"]]["cache"]


def band_for(c):
    return PREPROC[c["preproc"]]["band"]
