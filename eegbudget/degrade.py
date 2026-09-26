"""Digitisation and preprocessing budgets.

The cache arrives at 256 Hz, float32, in volts, already notched and band-passed 0.5-45 Hz. A
portable device is cheaper along two further axes, and both are simulated here rather than argued.

Sampling rate is reduced by integer decimation with the anti-alias filter that real hardware
would also need. Dropping to 64 Hz puts Nyquist at 32 Hz, so the gamma band is not merely noisy
but absent -- which is the honest consequence of a cheap converter, not a modelling artefact.

Amplitude is quantised against a fixed input range, as a real ADC does. Cheap converters pair a
low bit depth with a wide input range to survive electrode drift, so resolution is set by
full_scale / 2**bits and not by the signal's own amplitude.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

#: Typical differential input range of a portable amplifier, in volts. Wide enough to survive
#: electrode drift, which is why low-bit-depth converters lose so much on a microvolt signal.
FULL_SCALE_V = 400e-6


def resample_to(x, sf_in, sf_out):
    """Decimate to a lower sampling rate. Requires sf_in to be an integer multiple of sf_out."""
    if sf_out == sf_in:
        return x
    if sf_in % sf_out:
        raise ValueError(f"{sf_in} Hz is not an integer multiple of {sf_out} Hz")
    return resample_poly(x, 1, sf_in // sf_out, axis=-1)


def quantise(x, bits, full_scale=FULL_SCALE_V):
    """Simulate a bipolar ADC of the given depth over +/- full_scale/2."""
    if bits is None:
        return x
    lsb = full_scale / (2 ** bits)
    return np.clip(np.round(x / lsb) * lsb, -full_scale / 2, full_scale / 2)


def restrict_band(x, sf, lo, hi):
    """Zero-phase band restriction, for the EMG-free preprocessing arm."""
    nyq = sf / 2
    hi = min(hi, nyq * 0.99)
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


#: The digitisation grid swept in E1. None = no quantisation, the clinical-amplifier reference.
RATES = [256, 128, 64]
DEPTHS = [None, 16, 12, 8]

#: Preprocessing arms. 'full' is the cached corpus; 'emgfree' additionally restricts to a band
#: where scalp EMG cannot reach; 'noica' requires a separately generated cache and is skipped
#: when that cache is absent.
PREPROC = {
    "full": {"band": None, "cache": "harmonised"},
    "emgfree": {"band": (2.0, 20.0), "cache": "harmonised"},
    "noica": {"band": None, "cache": "harmonised_noica"},
}


def apply_budget(x, sf_in=256, rate=256, bits=None, band=None):
    """Apply a digitisation budget to (..., n_signals, T). Returns (array, sampling rate)."""
    if band is not None:
        x = restrict_band(x, sf_in, *band)
    x = resample_to(x, sf_in, rate)
    return quantise(x, bits), rate
