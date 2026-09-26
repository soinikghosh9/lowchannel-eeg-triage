"""Clinically named features that survive an electrode budget.

Every quantity here is one a neurophysiologist already reads, and every quantity is defined for
any number of signals, so the same feature vector is extractable at 19 electrodes and at 1. That
is what makes budgets comparable: the model changes, the measurement does not.

Two features are spatial and therefore conditional. Interhemispheric alpha asymmetry needs a
left-right pair; the anterior-posterior alpha gradient needs both an anterior and a posterior
site. A montage lacking them returns NaN, and the caller drops the column for that budget rather
than imputing it -- a four-electrode occipital device genuinely cannot measure a frontal gradient,
and pretending otherwise would flatter the low budgets.

Band definitions are clipped to Nyquist. At 64 Hz the gamma band does not exist, and the affected
features are returned as NaN rather than computed over a truncated range.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .montages import IX, LATERAL_PAIRS, REGION

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
PDR_BAND = (6.0, 13.0)
APERIODIC_FIT = (2.0, 24.0)   # available at every sampling rate in the sweep


def _psd(x, sf):
    nper = int(min(x.shape[-1], 2 * sf))
    f, p = welch(x, fs=sf, nperseg=nper, noverlap=nper // 2, axis=-1)
    return f, p


def _band_power(f, p, lo, hi, nyq):
    if hi > nyq:
        return np.full(p.shape[:-1], np.nan)
    m = (f >= lo) & (f < hi)
    return np.trapezoid(p[..., m], f[m], axis=-1) if m.sum() > 1 else np.full(p.shape[:-1], np.nan)


def _spectral_shape(f, p):
    """Entropy, 95% edge frequency and median frequency of the normalised spectrum."""
    tot = p.sum(axis=-1, keepdims=True)
    q = p / np.where(tot > 0, tot, np.nan)
    ent = -(q * np.log(np.where(q > 0, q, 1.0))).sum(axis=-1) / np.log(q.shape[-1])
    c = np.cumsum(q, axis=-1)
    edge = f[np.argmax(c >= 0.95, axis=-1)]
    med = f[np.argmax(c >= 0.50, axis=-1)]
    return ent, edge, med


def _pdr(f, p):
    """Posterior dominant rhythm: peak frequency in 6-13 Hz and its share of that band."""
    m = (f >= PDR_BAND[0]) & (f <= PDR_BAND[1])
    if m.sum() < 3:
        n = p.shape[:-1]
        return np.full(n, np.nan), np.full(n, np.nan)
    sub = p[..., m]
    k = np.argmax(sub, axis=-1)
    peak_f = f[m][k]
    peak_share = np.take_along_axis(sub, k[..., None], -1)[..., 0] / sub.sum(-1)
    return peak_f, peak_share


def _aperiodic_exponent(f, p):
    """Slope of log power against log frequency. Negated, so larger = steeper 1/f."""
    m = (f >= APERIODIC_FIT[0]) & (f <= APERIODIC_FIT[1]) & (f > 0)
    if m.sum() < 5:
        return np.full(p.shape[:-1], np.nan)
    lf = np.log10(f[m])
    lp = np.log10(np.maximum(p[..., m], 1e-30))
    lf_c = lf - lf.mean()
    slope = ((lp - lp.mean(-1, keepdims=True)) * lf_c).sum(-1) / (lf_c ** 2).sum()
    return -slope


def _hjorth(x):
    d1 = np.diff(x, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = x.var(-1)
    v1 = d1.var(-1)
    v2 = d2.var(-1)
    mob = np.sqrt(np.divide(v1, v0, out=np.full_like(v0, np.nan), where=v0 > 0))
    mob1 = np.sqrt(np.divide(v2, v1, out=np.full_like(v1, np.nan), where=v1 > 0))
    return mob, np.divide(mob1, mob, out=np.full_like(mob, np.nan), where=mob > 0)


def _alpha_by_site(f, p, sites):
    """Alpha power keyed by electrode name, for the two spatial features."""
    nyq = f[-1]
    a = _band_power(f, p, *BANDS["alpha"], nyq)
    return {s: a[i] for i, s in enumerate(sites)}


def window_features(w, sf, montage):
    """Features for one window, shape (n_signals, T). Returns a name -> float dict."""
    f, p = _psd(w, sf)
    nyq = sf / 2.0
    out = {}

    bp = {b: _band_power(f, p, lo, hi, nyq) for b, (lo, hi) in BANDS.items()}
    total = np.nansum(np.stack([v for v in bp.values()]), axis=0)
    for b, v in bp.items():
        out[f"rel_{b}"] = np.nanmean(np.divide(v, total, out=np.full_like(v, np.nan), where=total > 0))

    a, t = bp["alpha"], bp["theta"]
    out["theta_alpha_ratio"] = np.nanmean(np.divide(t, a, out=np.full_like(a, np.nan), where=a > 0))
    slow_num = bp["delta"] + bp["theta"]
    slow_den = bp["alpha"] + bp["beta"]
    out["slowing_ratio"] = np.nanmean(
        np.divide(slow_num, slow_den, out=np.full_like(slow_den, np.nan), where=slow_den > 0))

    ent, edge, med = _spectral_shape(f, p)
    out["spectral_entropy"] = float(np.nanmean(ent))
    out["spectral_edge95"] = float(np.nanmean(edge))
    out["spectral_median"] = float(np.nanmean(med))

    pf, ps = _pdr(f, p)
    out["pdr_frequency"] = float(np.nanmean(pf))
    out["pdr_prominence"] = float(np.nanmean(ps))
    out["aperiodic_exponent"] = float(np.nanmean(_aperiodic_exponent(f, p)))

    mob, cpx = _hjorth(w)
    out["hjorth_mobility"] = float(np.nanmean(mob))
    out["hjorth_complexity"] = float(np.nanmean(cpx))

    # Spatial features: defined only where the montage carries the sites they name.
    #
    # The site vocabulary is the montage's own. A 10-20 montage inherits the module-level
    # LATERAL_PAIRS and REGION; a montage from a different device -- EasyCog's four forehead and
    # twelve around-the-ear electrodes, say -- carries its own `lateral_pairs` and `region` maps
    # and gets the same two spatial features computed over its own geometry. Without this the
    # only way to measure a non-10-20 device is a separate feature function, and then the two
    # cohorts are no longer being measured with the same instrument.
    out["asymmetry_alpha"] = np.nan
    out["ap_gradient_alpha"] = np.nan
    if montage["ref"] != "bipolar":
        sites = montage["channels"]
        lateral = montage.get("lateral_pairs", LATERAL_PAIRS)
        region = montage.get("region", REGION)
        alpha = _alpha_by_site(f, p, sites)
        pairs = [(l, r) for l, r in lateral if l in alpha and r in alpha]
        if pairs:
            vals = [(alpha[l] - alpha[r]) / (alpha[l] + alpha[r])
                    for l, r in pairs if (alpha[l] + alpha[r]) > 0]
            if vals:
                out["asymmetry_alpha"] = float(np.mean(np.abs(vals)))
        ant = [alpha[s] for s in region["frontal"] if s in alpha]
        post = [alpha[s] for s in region["posterior"] if s in alpha]
        if ant and post and (np.mean(ant) + np.mean(post)) > 0:
            out["ap_gradient_alpha"] = float(
                (np.mean(post) - np.mean(ant)) / (np.mean(post) + np.mean(ant)))

    return out


NAMES = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
         "theta_alpha_ratio", "slowing_ratio", "spectral_entropy", "spectral_edge95",
         "spectral_median", "pdr_frequency", "pdr_prominence", "aperiodic_exponent",
         "hjorth_mobility", "hjorth_complexity", "asymmetry_alpha", "ap_gradient_alpha"]


def subject_features(windows, sf, montage):
    """Median across windows, shape (n_windows, n_signals, T) -> vector over NAMES."""
    rows = [window_features(w, sf, montage) for w in windows]
    return np.array([np.nanmedian([r[n] for r in rows]) for n in NAMES], dtype=np.float64)
