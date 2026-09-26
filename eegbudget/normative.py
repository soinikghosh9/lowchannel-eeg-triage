"""Age-conditioned normative model, and the control-anchoring step that deploys it.

A normative model asks what a feature should be for a person of this age, and scores the patient
by how far they sit from it. Fitted on controls alone, it never sees a diagnostic label, so the
reference cannot inherit the case-control age gap that the raw features do.

The location-scale form is deliberately plain: a quadratic mean in age and a log-linear spread.
Over a 23-96 year span this is enough to carry the ageing trend without the fit wandering in the
sparse tails, and it has few enough parameters to be stable on 459 controls. A richer basis is
easy to substitute and would change the numbers slightly; it would not change the argument.

The reason this matters for deployment is the anchoring step. A new clinic differs from the
training corpus in amplifier, electrode, montage and population. Anchoring corrects for that using
only recordings from people the clinic already knows to be healthy -- no labelled patients, no
retraining, no gradient. A district hospital cannot assemble a labelled dementia cohort. It can
record twenty healthy volunteers.
"""
from __future__ import annotations

import numpy as np

MIN_CONTROLS = 20
FLOOR = 1e-9

#: The spread model is fitted on log|residual|, so exp(fit) recovers the geometric mean of the
#: absolute residual, not the standard deviation. For r ~ N(0, s^2),
#: E[log|r|] = log s - (euler_gamma + log 2)/2, so this constant converts the fit back to s.
#: Omitting it inflates every deviation by exp(0.635) = 1.89 and the z scores are not unit-scaled.
_LOG_ABS_TO_SD = (np.euler_gamma + np.log(2)) / 2


def _design(age):
    age = np.asarray(age, float)
    return np.column_stack([np.ones_like(age), age, age ** 2])


def fit(X, age, min_n=30):
    """Fit per-feature age trajectories on control recordings only.

    Returns a dict with the mean coefficients, the log-spread coefficients, and the age range
    the fit is supported on. Features with too few finite observations are marked unusable.
    """
    X = np.asarray(X, float)
    age = np.asarray(age, float)
    n_feat = X.shape[1]

    beta = np.full((n_feat, 3), np.nan)
    gamma = np.full((n_feat, 2), np.nan)
    usable = np.zeros(n_feat, bool)

    for j in range(n_feat):
        ok = np.isfinite(X[:, j]) & np.isfinite(age)
        if ok.sum() < min_n:
            continue
        A = _design(age[ok])
        b, *_ = np.linalg.lstsq(A, X[ok, j], rcond=None)
        resid = X[ok, j] - A @ b
        # log-linear spread, fitted on log|residual| so sigma stays positive by construction
        L = np.column_stack([np.ones(ok.sum()), age[ok]])
        g, *_ = np.linalg.lstsq(L, np.log(np.abs(resid) + FLOOR), rcond=None)
        beta[j], gamma[j], usable[j] = b, g, True

    a = age[np.isfinite(age)]
    return {"beta": beta, "gamma": gamma, "usable": usable,
            "age_range": (float(np.min(a)), float(np.max(a))),
            "n_controls": int(np.isfinite(age).sum())}


def predict_moments(model, age):
    """Expected value and spread of every feature at the given ages."""
    A = _design(age)
    L = np.column_stack([np.ones(len(A)), np.asarray(age, float)])
    mu = A @ model["beta"].T
    sigma = np.exp(L @ model["gamma"].T + _LOG_ABS_TO_SD)
    return mu, np.maximum(sigma, FLOOR)


def deviation(model, X, age, anchor=None):
    """Signed z-deviation from the age-expected value, optionally re-anchored to a site.

    `anchor` is the output of :func:`fit_anchor` and shifts and scales the reference so that the
    target site's own controls are centred. Without it the raw training-corpus reference is used,
    which is the direct-transfer arm.
    """
    mu, sigma = predict_moments(model, age)
    z = (np.asarray(X, float) - mu) / sigma
    if anchor is not None:
        z = (z - anchor["shift"]) / anchor["scale"]
    z[:, ~model["usable"]] = np.nan
    return z


def fit_anchor(model, X_ctrl, age_ctrl):
    """Estimate a site correction from that site's controls alone.

    Robust statistics throughout: a clinic contributing twenty volunteers will contribute at
    least one bad recording, and a mean would follow it.
    """
    z = deviation(model, X_ctrl, age_ctrl)
    # Columns the montage cannot measure are all-NaN by construction, and nanmedian warns rather
    # than returning NaN quietly. Mask them out first: an unusable feature has no site correction
    # to estimate, and the identity is the right answer for it.
    usable = np.isfinite(z).any(axis=0)
    shift = np.full(z.shape[1], np.nan)
    mad = np.full(z.shape[1], np.nan)
    if usable.any():
        shift[usable] = np.nanmedian(z[:, usable], axis=0)
        mad[usable] = np.nanmedian(np.abs(z[:, usable] - shift[usable]), axis=0) * 1.4826
    scale = np.where(np.isfinite(mad) & (mad > FLOOR), mad, 1.0)
    return {"shift": np.nan_to_num(shift), "scale": scale, "n": len(X_ctrl)}


def abnormality(z, weights=None):
    """Collapse a deviation vector to one number a clinician can act on.

    The root-mean-square deviation across features, which is direction-free: a feature abnormal
    in either direction contributes. Where a montage cannot measure a feature the column is NaN
    and is simply not counted, so budgets remain comparable.
    """
    z = np.asarray(z, float)
    w = np.ones(z.shape[1]) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(z)
    num = np.nansum((z ** 2) * w, axis=1)
    den = (ok * w).sum(axis=1)
    return np.sqrt(np.divide(num, den, out=np.full(len(z), np.nan), where=den > 0))
