"""Discrimination, age matching, and the demographic substitution ratio.

The quantity this study is built around is not AUC but how much of AUC is demographics. Define,
for an acquisition budget b,

    rho(b) = [AUC_total(b) - AUC_matched(b)] / [AUC_total(b) - 0.5]

the share of above-chance discrimination that disappears when the case and control age
distributions are equalised. rho = 0 means the measurement carries the decision; rho = 1 means the
calendar does. The prediction under test is that rho rises as b falls -- that stripping the
acquisition budget does not degrade the instrument evenly, but shifts it onto the confound.

Two independent estimates of the same quantity are provided, because neither is clean alone.
Matching discards subjects and can bias the retained sample; residualising keeps everyone but
assumes the age relationship is well specified. They are reported side by side and a claim is
made only where they agree.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def auc(y, s):
    """Mann-Whitney AUC. NaN when either class is empty."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootstrap_ci(y, s, n=2000, seed=SEED):
    """Stratified bootstrap interval for AUC, resampling subjects within class."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(ip) < 2 or len(ineg) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        ix = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
        vals[i] = auc(y[ix], s[ix])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def _splitter(X, y, groups, folds, random_state):
    """Return a row- or group-wise stratified splitter."""
    if groups is None:
        return StratifiedKFold(folds, shuffle=True, random_state=random_state), None
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("groups must have one entry per row")
    if len(np.unique(groups)) < folds:
        raise ValueError(f"need at least {folds} unique groups, got {len(np.unique(groups))}")
    return StratifiedGroupKFold(folds, shuffle=True, random_state=random_state), groups


def cv_scores(X, y, groups=None, folds=5, repeats=5, seed=SEED, rank_normalize=True):
    """Out-of-fold decision scores from L2 logistic regression on standardised features.

    Columns that are entirely non-finite are dropped -- that is how a montage which cannot
    measure a feature is handled, rather than by imputing a value it never observed.
    """
    X = np.asarray(X, float)
    keep = np.isfinite(X).all(axis=0)
    if keep.sum() == 0:
        return np.full(len(y), np.nan), keep
    X = X[:, keep]
    y = np.asarray(y)

    acc = np.zeros(len(y))
    for r in range(repeats):
        skf, split_groups = _splitter(X, y, groups, folds, seed + r)
        oof = np.empty(len(y))
        for tr, te in skf.split(X, y, split_groups):
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=1.0))
            model.fit(X[tr], y[tr])
            oof[te] = model.decision_function(X[te])
        # rank-normalise per repeat so scores from different folds are commensurable
        acc += rankdata(oof) / len(oof) if rank_normalize else oof
    return acc / repeats, keep


def match_on_age(age, y, caliper=3.0, seed=SEED):
    """Nearest-neighbour age matching without replacement. Returns the retained index.

    Each case is paired with the nearest unused control within `caliper` years. Cases with no
    partner are dropped, so the retained sample is smaller and its composition must be reported.
    """
    age = np.asarray(age, float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    cases = np.where((y == 1) & np.isfinite(age))[0]
    ctrls = list(np.where((y == 0) & np.isfinite(age))[0])
    rng.shuffle(cases)

    keep = []
    for c in cases:
        if not ctrls:
            break
        d = np.abs(age[ctrls] - age[c])
        j = int(np.argmin(d))
        if d[j] <= caliper:
            keep += [c, ctrls.pop(j)]
    return np.array(sorted(keep), dtype=int)


def residualise_on_age(X, age):
    """Remove a quadratic age trend from every feature, fitted on the whole sample.

    This helper is descriptive only. Use :func:`cv_residualised_scores` for inference.
    """
    X = np.asarray(X, float)
    age = np.asarray(age, float)
    ok = np.isfinite(age)
    A = np.column_stack([np.ones(ok.sum()), age[ok], age[ok] ** 2])
    out = X.copy()
    for j in range(X.shape[1]):
        col = X[ok, j]
        good = np.isfinite(col)
        if good.sum() < 10:
            continue
        beta, *_ = np.linalg.lstsq(A[good], col[good], rcond=None)
        pred = A @ beta
        out[ok, j] = col - pred
    return out


def _midrank(x):
    """Midranks, which is what makes DeLong exact under ties.

    The tie scan must advance even when the comparison is false for a value compared with itself.
    NaN is exactly that value: ``nan == nan`` is False, so a naive ``while Z[j] == Z[i]`` leaves
    ``j`` equal to ``i``, the assignment writes an empty slice, ``i = j`` does not move, and the
    loop spins forever. Callers strip non-finite values, but this stays robust regardless -- a
    ranking helper that can hang is not worth the two characters it saves.
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i + 1
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def delong(scores, y):
    """Fast DeLong (Sun & Xu 2014): AUCs and their covariance for correlated ROC curves.

    Parameters
    ----------
    scores : array (k, n)
        k score vectors over the same n subjects.
    y : array (n,)
        Binary labels.

    Returns
    -------
    aucs : array (k,)
    cov : array (k, k)
    """
    scores = np.atleast_2d(np.asarray(scores, float))
    y = np.asarray(y)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]

    tx = np.stack([_midrank(pos[r]) for r in range(k)])
    ty = np.stack([_midrank(neg[r]) for r in range(k)])
    tz = np.stack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # structural components over positives
    v10 = 1.0 - (tz[:, m:] - ty) / m    # over negatives
    sx = np.cov(v01, ddof=1).reshape(k, k)
    sy = np.cov(v10, ddof=1).reshape(k, k)
    return aucs, sx / m + sy / n


def delong_test(score_a, score_b, y):
    """Two-sided test that two correlated AUCs differ, on the same subjects.

    This is the comparison the whole study turns on -- an EEG model against chronological age,
    measured on identical patients -- and an unpaired interval on each AUC separately does not
    answer it. Returns the difference, its 95% interval and the p-value.
    """
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    y = np.asarray(y)
    # A subject the model could not score cannot be ranked, so it leaves the comparison entirely
    # rather than being imputed. Both curves must be read on the same subjects for the paired
    # test to mean anything, so the mask is the intersection.
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return {"auc_a": np.nan, "auc_b": np.nan, "diff": np.nan, "ci": [np.nan, np.nan],
                "se": np.nan, "z": np.nan, "p": np.nan, "n": int(ok.sum())}
    a, b, y = a[ok], b[ok], y[ok]
    aucs, cov = delong(np.vstack([a, b]), y)
    L = np.array([1.0, -1.0])
    var = float(L @ cov @ L)
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        # Two predictors that rank every pair identically have exactly zero variance in their
        # difference. That is not a failure to compute a p-value, it is the answer: there is no
        # evidence of a difference because there is no difference.
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
                "ci": [0.0, 0.0], "se": 0.0, "z": 0.0,
                "p": 1.0 if abs(diff) < 1e-12 else 0.0, "degenerate": True}
    se = np.sqrt(var)
    z = diff / se
    from scipy.stats import norm
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "ci": [diff - 1.96 * se, diff + 1.96 * se], "se": se,
            "z": float(z), "p": float(2 * norm.sf(abs(z))), "n": int(len(y))}


def paired_bootstrap_diff(score_a, score_b, y, n=4000, seed=SEED):
    """Non-parametric check on the same difference, resampling subjects within class.

    Reported beside DeLong rather than instead of it: DeLong is the standard and assumes
    asymptotic normality, the bootstrap assumes neither, and agreement between them is worth
    more than either alone.
    """
    y = np.asarray(y)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        ix = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
        d[i] = auc(y[ix], a[ix]) - auc(y[ix], b[ix])
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"diff_mean": float(np.nanmean(d)), "ci": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def substitution_ratio(auc_total, auc_matched):
    """Share of above-chance discrimination attributable to age structure."""
    lift = auc_total - 0.5
    if not np.isfinite(lift) or lift <= 1e-9:
        return np.nan
    return float((auc_total - auc_matched) / lift)


def sensitivity_at(y, s, spec_target=0.80):
    """Sensitivity at the threshold giving the requested specificity in this sample."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    neg = s[y == 0]
    if len(neg) == 0:
        return np.nan, np.nan
    thr = np.quantile(neg, spec_target)
    return float((s[y == 1] > thr).mean()), float(thr)


def binom_ci(k, n):
    """Wilson interval, which behaves at the small counts the subtype arms produce."""
    if n == 0:
        return (np.nan, np.nan)
    z, p = 1.96, k / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((c - half) / d, (c + half) / d)


def cv_scores_repeats(X, y, groups=None, folds=5, repeats=5, seed=SEED, rank_normalize=True):
    """Out-of-fold scores kept separately for each repeat, plus their average.

    ``cv_scores`` averages the repeats before returning, which hides how much of the reported
    margin is resampling noise in the fold assignment. Several claims in this study are
    differences of 0.02--0.07 AUC, so the between-repeat spread is worth reporting beside the
    point estimate rather than being averaged out of existence.

    Returns ``(per_repeat, mean, keep)`` where ``per_repeat`` has shape ``(repeats, n)``.
    """
    X = np.asarray(X, float)
    keep = np.isfinite(X).all(axis=0)
    if keep.sum() == 0:
        return np.full((repeats, len(y)), np.nan), np.full(len(y), np.nan), keep
    X = X[:, keep]
    y = np.asarray(y)

    per = np.empty((repeats, len(y)))
    for r in range(repeats):
        skf, split_groups = _splitter(X, y, groups, folds, seed + r)
        oof = np.empty(len(y))
        for tr, te in skf.split(X, y, split_groups):
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=1.0))
            model.fit(X[tr], y[tr])
            oof[te] = model.decision_function(X[te])
        per[r] = rankdata(oof) / len(oof) if rank_normalize else oof
    return per, per.mean(axis=0), keep


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the order given.

    The primary comparisons form a family: three clinical contrasts crossed with the estimators
    reported inferentially. Reporting each at 0.05 uncorrected and then describing the pattern
    across them is the classic way to turn a marginal result into a headline, so the family is
    declared and corrected.
    """
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def residualise_on_controls(X, age, y):
    """Remove an age trend fitted on the control arm alone.

    ``residualise_on_age`` fits the trend on cases and controls pooled. Because cases are both
    older and abnormal, that pooled fit absorbs part of the disease effect into the age term and
    over-removes, which inflates every substitution ratio derived from it. Fitting on controls is
    the same choice the normative model makes and is the one that estimates ageing rather than
    ageing-plus-disease.
    """
    X = np.asarray(X, float)
    age = np.asarray(age, float)
    y = np.asarray(y)
    ctrl = (y == 0) & np.isfinite(age)
    A_ctrl = np.column_stack([np.ones(ctrl.sum()), age[ctrl], age[ctrl] ** 2])
    ok = np.isfinite(age)
    A_all = np.column_stack([np.ones(ok.sum()), age[ok], age[ok] ** 2])
    out = X.copy()
    for j in range(X.shape[1]):
        col_c = X[ctrl, j]
        good = np.isfinite(col_c)
        if good.sum() < 10:
            continue
        beta, *_ = np.linalg.lstsq(A_ctrl[good], col_c[good], rcond=None)
        out[ok, j] = X[ok, j] - A_all @ beta
    return out


def _age_design(age):
    age = np.asarray(age, float)
    return np.column_stack([np.ones_like(age), age, age ** 2])


def _fit_age_residualiser(X, age, fit_ix, y=None, controls_only=True):
    """Fit a quadratic age trend using a training fold only."""
    X, age = np.asarray(X, float), np.asarray(age, float)
    fit_ix = np.asarray(fit_ix, dtype=int)
    if controls_only and y is not None:
        fit_ix = fit_ix[np.asarray(y)[fit_ix] == 0]
    beta = np.full((X.shape[1], 3), np.nan)
    for j in range(X.shape[1]):
        ok = np.isfinite(age[fit_ix]) & np.isfinite(X[fit_ix, j])
        if ok.sum() < 5:
            continue
        beta[j], *_ = np.linalg.lstsq(_age_design(age[fit_ix][ok]), X[fit_ix, j][ok], rcond=None)
    return beta


def _apply_age_residualiser(X, age, beta):
    X, age = np.asarray(X, float), np.asarray(age, float)
    out = X.copy()
    ok = np.isfinite(age)
    if ok.any():
        out[ok] -= _age_design(age[ok]) @ beta.T
    out[:, ~np.isfinite(beta).all(axis=1)] = np.nan
    return out


def cv_residualised_scores(X, y, age, groups=None, folds=5, repeats=5, seed=SEED,
                           controls_only=True, rank_normalize=True):
    """Cross-fitted age-residualised EEG scores.

    The residualiser is refit inside each training fold and, by default, uses training controls
    only. This prevents the full-sample age/disease relationship from leaking into the test fold.
    """
    X, y, age = np.asarray(X, float), np.asarray(y), np.asarray(age, float)
    if not np.isfinite(age).all():
        raise ValueError("age must be finite for cross-fitted age adjustment")
    keep = np.isfinite(X).all(axis=0)
    if keep.sum() == 0:
        return np.full(len(y), np.nan), keep
    X = X[:, keep]
    acc = np.zeros(len(y))
    for r in range(repeats):
        splitter, split_groups = _splitter(X, y, groups, folds, seed + r)
        oof = np.empty(len(y))
        for tr, te in splitter.split(X, y, split_groups):
            beta = _fit_age_residualiser(X, age, tr, y, controls_only)
            ztr = _apply_age_residualiser(X[tr], age[tr], beta)
            zte = _apply_age_residualiser(X[te], age[te], beta)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            model.fit(ztr, y[tr])
            oof[te] = model.decision_function(zte)
        acc += rankdata(oof) / len(oof) if rank_normalize else oof
    return acc / repeats, keep


def cv_normative_scores(X, y, age, groups=None, folds=5, repeats=5, seed=SEED,
                        rank_normalize=True):
    """Cross-fitted classifier on a control-only age-conditioned normative representation."""
    from . import normative as nm

    X, y, age = np.asarray(X, float), np.asarray(y), np.asarray(age, float)
    if not np.isfinite(age).all():
        raise ValueError("age must be finite for cross-fitted normative scoring")
    keep = np.isfinite(X).all(axis=0)
    if keep.sum() == 0:
        return np.full(len(y), np.nan), keep
    X = X[:, keep]
    acc = np.zeros(len(y))
    for r in range(repeats):
        splitter, split_groups = _splitter(X, y, groups, folds, seed + r)
        oof = np.empty(len(y))
        for tr, te in splitter.split(X, y, split_groups):
            ctrl = y[tr] == 0
            model_norm = nm.fit(X[tr][ctrl], age[tr][ctrl])
            ztr = nm.deviation(model_norm, X[tr], age[tr])
            zte = nm.deviation(model_norm, X[te], age[te])
            head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            head.fit(ztr, y[tr])
            oof[te] = head.decision_function(zte)
        acc += rankdata(oof) / len(oof) if rank_normalize else oof
    return acc / repeats, keep


# ------------------------------------------------------------------------------------------------
# Permutation inference
# ------------------------------------------------------------------------------------------------
def permutation_increment_p(X, y, age, n_perm=200, folds=5, repeats=5, seed=SEED, groups=None):
    """Permutation p-value for the incremental margin, under an explicit "EEG is noise" null.

    DeLong's variance is derived for a fixed scoring rule. Ours is refit in every fold and then
    averaged over repeats, so the pseudo-observations are neither independent nor identically
    scaled and the resulting p-value is anti-conservative by an unknown amount. This builds the
    null by rerunning the entire estimator -- scaling, fitting, out-of-fold scoring -- on permuted
    data, so the reference distribution absorbs every optimism the pipeline contains.

    **What the null is.** The subject index is permuted, which carries each label together with its
    age and breaks only the correspondence between a subject and their recording. So the null world
    is one where age predicts exactly as well as it really does and the EEG features are noise. The
    statistic AUC(EEG+age) - AUC(age) then measures what a clinic gains by adding an uninformative
    recording to the demographics it already has, which is exactly the null the incremental
    question needs. The null is *not* centred on zero: appending a noise block to a working
    predictor costs a little, so the null sits slightly below zero and the observed increment is
    read against that.

    This construction is wrong for the substitution margin. AUC(EEG) - AUC(age) under this null
    sits near 0.5 - AUC(age), which for CAUEEG screening is -0.286, so a test against it asks
    whether the recording beats nothing rather than whether it beats the calendar. Use
    :func:`paired_bootstrap_diff` for that comparison, which resamples subjects and tests the
    margin against zero without assuming anything about the scoring rule.

    Returns the observed margin, a one-sided p-value against the null, and the null distribution.
    Resolution is bounded below by ``1 / (n_perm + 1)``; a p reported at that floor means "no
    permutation reached the observed value", not "p = 0".
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    age = np.asarray(age, float)

    def margin(yy, aa):
        s, _ = cv_scores(np.column_stack([X, aa]), yy, groups=groups, folds=folds,
                         repeats=repeats, seed=seed)
        return auc(yy, s) - auc(yy, aa)

    observed = margin(y, age)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(y))
        null[i] = margin(y[perm], age[perm])
    p_one = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {"margin": float(observed), "p_perm": float(min(1.0, p_one)),
            "p_perm_floor": float(1.0 / (n_perm + 1)),
            "n_perm": int(n_perm), "null_mean": float(np.mean(null)),
            "null_sd": float(np.std(null)),
            "null_q": [float(q) for q in np.percentile(null, [2.5, 50, 97.5])],
            "null_hypothesis": "EEG features are uninformative; age retains its true association"}


# ------------------------------------------------------------------------------------------------
# Continuous outcomes
# ------------------------------------------------------------------------------------------------
def cv_regression_scores(X, y, strat=None, folds=10, repeats=10, seed=SEED, alphas=None):
    """Out-of-fold ridge predictions for a continuous outcome, averaged over repeats.

    Cognitive scales are the outcome low-cost assessment devices are actually benchmarked on.
    Dichotomising one at a cut-point discards most of its information and, on a clinical cohort,
    usually produces an unusable class balance as well. The estimator is a ridge with the penalty
    chosen inside each training fold and standardisation inside the fold -- the regression
    counterpart of ``cv_scores``.

    ``strat`` supplies fold-stratification labels (band membership on the outcome), mirroring how
    the EasyCog benchmark balances its own folds, so the comparison is against their split rather
    than an easier one.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    keep = np.isfinite(X).all(axis=0)
    if keep.sum() == 0:
        return np.full(len(y), np.nan), keep
    X = X[:, keep]
    alphas = np.logspace(-2, 3, 20) if alphas is None else alphas

    acc = np.zeros((repeats, len(y)))
    for r in range(repeats):
        if strat is None:
            splitter, split_args = KFold(folds, shuffle=True, random_state=seed + r), (X,)
        else:
            splitter = StratifiedKFold(folds, shuffle=True, random_state=seed + r)
            split_args = (X, np.asarray(strat))
        for tr, te in splitter.split(*split_args):
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(X[tr], y[tr])
            acc[r, te] = model.predict(X[te])
    return acc.mean(axis=0), keep


def regression_metrics(y, pred, n_boot=2000, seed=SEED):
    """MAE and Pearson correlation with bootstrap intervals over subjects."""
    from scipy.stats import pearsonr

    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    rng = np.random.default_rng(seed)
    mae_b, pcc_b = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        mae_b[i] = np.abs(pred[ix] - y[ix]).mean()
        pcc_b[i] = pearsonr(pred[ix], y[ix])[0] if np.std(pred[ix]) > 0 else np.nan

    def q(v):
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {"n": int(len(y)),
            "mae": float(np.abs(pred - y).mean()), "mae_ci": q(mae_b),
            "pcc": float(pearsonr(pred, y)[0]), "pcc_ci": q(pcc_b),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}


def paired_metric_diff(y, pred_a, pred_b, n_boot=2000, seed=SEED):
    """Paired bootstrap on the MAE and PCC difference between two prediction vectors.

    Two separate intervals on two MAEs cannot answer whether one model beats the other on the same
    subjects. This resamples subjects once and recomputes both.
    """
    from scipy.stats import pearsonr

    y = np.asarray(y, float)
    a, b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    rng = np.random.default_rng(seed)
    dm, dp = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        dm[i] = np.abs(a[ix] - y[ix]).mean() - np.abs(b[ix] - y[ix]).mean()
        dp[i] = pearsonr(a[ix], y[ix])[0] - pearsonr(b[ix], y[ix])[0]

    def q(v):
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {"d_mae": float(np.abs(a - y).mean() - np.abs(b - y).mean()), "d_mae_ci": q(dm),
            "d_pcc": float(pearsonr(a, y)[0] - pearsonr(b, y)[0]), "d_pcc_ci": q(dp),
            "p_mae": float(2 * min((dm <= 0).mean(), (dm >= 0).mean())),
            "p_pcc": float(2 * min((dp <= 0).mean(), (dp >= 0).mean()))}


# ------------------------------------------------------------------------------------------------
# Clinical utility
# ------------------------------------------------------------------------------------------------
def cv_probabilities(X, y, groups=None, folds=5, repeats=5, seed=SEED):
    """Out-of-fold predicted probabilities, averaged over repeats.

    ``cv_scores`` returns a rank fraction, which is all discrimination needs and all that a paired
    AUC test can use. Decision-curve analysis needs a probability, because the threshold it sweeps
    is a risk at which a clinician would act. Averaging probabilities across repeats is coherent
    in a way that averaging decision values is not, so this is a separate function rather than a
    flag on the other one.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    keep = np.isfinite(X).all(axis=0)
    if keep.sum() == 0:
        return np.full(len(y), np.nan), keep
    X = X[:, keep]

    acc = np.zeros(len(y))
    for r in range(repeats):
        splitter, split_groups = _splitter(X, y, groups, folds, seed + r)
        oof = np.empty(len(y))
        for tr, te in splitter.split(X, y, split_groups):
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=1.0))
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
        acc += oof
    return acc / repeats, keep


def net_benefit(y, prob, thresholds=None, prevalence=None):
    """Decision-curve net benefit for a probability, against treat-all and treat-none.

    Discrimination says how well a score ranks patients. It does not say whether acting on the
    score beats the two things a service can already do without it: refer everyone, or refer
    nobody. At a threshold probability ``pt`` -- the risk above which a clinician judges referral
    worthwhile -- net benefit is

        NB = TP/n - (FP/n) * pt/(1 - pt)

    the share of true referrals gained after charging each false referral at the exchange rate the
    clinician's own threshold implies. A screener is worth deploying over the range of ``pt`` where
    its curve sits above both references, and nowhere else.

    ``prevalence`` re-weights a case-control sample to a service's actual case mix. This matters
    here: the CAUEEG screening contrast is 62% cases, and no triage clinic looks like that, so an
    unweighted curve would overstate the benefit of referring at every threshold.
    """
    y = np.asarray(y)
    prob = np.asarray(prob, float)
    ok = np.isfinite(prob)
    y, prob = y[ok], prob[ok]
    n = len(y)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.51, 0.01), 3)

    obs_prev = float((y == 1).mean())
    prev = obs_prev if prevalence is None else float(prevalence)
    # Weight cases and controls so the sample behaves like a population with prevalence `prev`.
    w_pos = (prev / obs_prev) if obs_prev > 0 else 0.0
    w_neg = ((1 - prev) / (1 - obs_prev)) if obs_prev < 1 else 0.0
    n_eff = w_pos * (y == 1).sum() + w_neg * (y == 0).sum()

    rows = []
    for pt in thresholds:
        flag = prob >= pt
        tp = float(((y == 1) & flag).sum()) * w_pos
        fp = float(((y == 0) & flag).sum()) * w_neg
        odds = pt / (1 - pt)
        rows.append({
            "threshold": float(pt),
            "net_benefit": float(tp / n_eff - (fp / n_eff) * odds),
            "net_benefit_treat_all": float(prev - (1 - prev) * odds),
            "net_benefit_treat_none": 0.0,
            "flagged_fraction": float((w_pos * ((y == 1) & flag).sum()
                                       + w_neg * ((y == 0) & flag).sum()) / n_eff)})
    return {"n": int(n), "prevalence_assumed": prev, "prevalence_observed": obs_prev,
            "curve": rows}
