"""The manuscript's two figures.

Figure 1 carries the argument in three panels: what the recording adds once the estimator is
allowed to vary, what a published low-cost benchmark looks like when age is put on its own axis,
and how large a cohort the increment needs before it can be seen at all.

Figure 2 answers the question discrimination cannot: at a service's actual case mix, does adding
the recording to age change who gets referred.

    python experiments/e17_paper_figure.py
Out: outputs/figures/fig1_results.{pdf,png}, fig2_utility.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from eegbudget import paths, viz  # noqa: E402

TASKS = ["screening", "dementia", "mci"]
NICE = {"screening": "Screening", "dementia": "Dementia", "mci": "MCI"}
C_SUB, C_INC, C_DEEP = viz.SERIES[0], viz.SERIES[2], viz.SERIES[1]


def load(name):
    p = paths.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------
def _err(values, cis):
    """Asymmetric error-bar array, with NaN where an interval is unavailable."""
    lo = [v - c[0] if c else np.nan for v, c in zip(values, cis)]
    hi = [c[1] - v if c else np.nan for v, c in zip(values, cis)]
    return np.abs(np.array([lo, hi], float))


def panel_margins(ax, e13, e11):
    """(a) The substitution margin moves with the estimator; the increment does not.

    Every bar carries its 95% interval. Without them the panel reads as three clean sign flips,
    when in fact only the dementia deep margin is separated from zero -- and the whole argument
    of the paper is about how much confidence a margin of this size supports.
    """
    ea = e13.get("estimator_axis", {})
    feat, deep = ea.get("feature_model", {}), ea.get("deep_model", {})
    full = e13.get("primary_budget", "b19|256|f32|full")
    x = np.arange(len(TASKS))
    w = 0.26
    sub_f = [feat.get(t, {}).get("sub_margin", np.nan) for t in TASKS]
    sub_d = [deep.get(t, {}).get("sub_margin", np.nan) for t in TASKS]
    inc_f = [feat.get(t, {}).get("inc_margin", np.nan) for t in TASKS]

    # Feature-model intervals come from the same file's per-task record; the deep intervals come
    # from the deep arm's own DeLong test against age, which is where those margins were computed.
    budgets = {t: e13.get("tasks", {}).get(t, {}).get("budgets", {}).get(full, {}) for t in TASKS}
    ci_sub_f = [budgets[t].get("sub_ci") for t in TASKS]
    ci_inc_f = [budgets[t].get("inc_ci") for t in TASKS]
    ci_sub_d = []
    for t in TASKS:
        key = f"b19|{t}|{deep.get(t, {}).get('model', '')}|none"
        ci_sub_d.append((e11 or {}).get("runs", {}).get(key, {}).get("vs_age_within", {}).get("ci"))

    bars = ((x - w, sub_f, ci_sub_f, C_SUB, "substitute, 17 features"),
            (x, sub_d, ci_sub_d, C_DEEP, "substitute, deep network"),
            (x + w, inc_f, ci_inc_f, C_INC, "add to age, 17 features"))
    for xs, vals, cis, colour, label in bars:
        ax.bar(xs, vals, w, color=colour, label=label)
        if any(c for c in cis):
            ax.errorbar(xs, vals, yerr=_err(vals, cis), fmt="none",
                        ecolor=viz.INK, elinewidth=0.9, capsize=2)
    ax.axhline(0, color=viz.INK, lw=0.9)
    # Labels sit outside the interval, not the bar, or the whisker overprints the number. The
    # middle bar's label is pushed further out: at this bar width the three labels in a group are
    # wider than the group, so adjacent ones collide unless they are staggered.
    for i, (xs, vals, cis, _, _) in enumerate(bars):
        pad = 11 if i == 1 else 3
        for xi, v, c in zip(xs, vals, cis):
            if not np.isfinite(v):
                continue
            anchor = (c[1] if v >= 0 else c[0]) if c else v
            ax.annotate(f"{v:+.3f}", (xi, anchor), ha="center",
                        va="bottom" if v >= 0 else "top",
                        xytext=(0, pad if v >= 0 else -pad), textcoords="offset points",
                        fontsize=6.2, color=viz.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[t] for t in TASKS], fontsize=7.0)
    ax.set_ylabel("AUC margin against age")
    ax.set_title("a  What the estimator decides", loc="left", fontweight="bold")
    # Every task has a bar or a CI whisker reaching well above zero, so no pocket inside the axes
    # is reliably clear. The key goes above the plot instead -- attached to the figure, not to this
    # axes, because a three-column row is wider than one panel and anchoring it here pushed it off
    # the canvas, which made the tight bounding box wider than the column and shrank every label.
    ax.figure.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.4,
                     loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                     columnspacing=1.4, handlelength=1.4)
    viz.hgrid(ax)
    ax.margins(y=0.22)


def panel_easycog(ax, e19):
    """(b) A published benchmark against its own missing age baseline."""
    rec = e19["regression_audit"]["targets"]["moca"]
    pub = e19["regression_audit"]["published"]
    # Every bar is an all-channel arm, so the panel varies the feature set and nothing else. An
    # earlier version drew the ear-block variant beside the all-channel 17-feature arm, which
    # compared two things at once. The mean predictor is omitted: it is constant within a fold, so
    # its "correlation" is fold-to-fold variation rather than signal, and the table prints a dash
    # for that cell -- drawing a bar there contradicted the table it accompanies.
    arms = [("eeg17_all+age", "17 EEG features\n+ age", C_SUB),
            ("age", "age alone", viz.MUTED),
            ("slow3_all+age", "3 EEG features\n+ age", C_INC)]
    vals = [rec["arms"][k]["pcc"] for k, _, _ in arms]
    los = [rec["arms"][k]["pcc_ci"][0] for k, _, _ in arms]
    his = [rec["arms"][k]["pcc_ci"][1] for k, _, _ in arms]
    y = np.arange(len(arms))
    ax.barh(y, vals, color=[c for _, _, c in arms], height=0.6)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor=viz.INK, elinewidth=0.9, capsize=2)
    ax.axvline(pub["test_best"]["moca_pcc"], color=viz.INK, ls="--", lw=1.0)
    # Annotate inside the axes: the reference line is the point of the panel, and an
    # annotation anchored above the top bar lands outside the frame and is clipped away.
    ax.annotate(f"best published method ({pub['test_best']['moca_pcc']:.2f})",
                (pub["test_best"]["moca_pcc"], 0.45),
                xytext=(5, 0), textcoords="offset points", fontsize=6.3,
                color=viz.INK, va="center", ha="left")
    ax.axvline(0, color=viz.FAINT, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in arms], fontsize=7)
    ax.set_xlabel("correlation with MoCA, held-out subjects")
    ax.set_title("b  EasyCog, with age on the axis", loc="left", fontweight="bold")
    ax.set_xlim(-0.35, 0.75)


def panel_complexity(ax, e20, e19):
    """(c) How many participants the increment needs, and where each cohort falls on that axis."""
    rows = [r for r in e20["rows"] if r["task"] == "screening"]
    for est, colour, label in (("eeg17", C_SUB, "17 features"),
                               ("slow3", C_INC, "3 features")):
        r = sorted([x for x in rows if x["estimator"] == est], key=lambda x: x["n"])
        n = [x["n"] for x in r]
        mid = [x["inc_mean"] for x in r]
        lo = [x["inc_q"][0] for x in r]
        hi = [x["inc_q"][1] for x in r]
        ax.plot(n, mid, "-o", color=colour, ms=3, lw=1.3, label=label)
        ax.fill_between(n, lo, hi, color=colour, alpha=0.15, lw=0)

    # Every external cohort, not one of them: EasyCog (69) and the three dementia test sets.
    sizes = {"EasyCog": int(e19["manifest"]["aggregated_subjects"])}
    e31 = load("e31_external_extract.json") or {}
    for c, g in e31.get("counts", {}).items():
        sizes[c] = int(sum(g.values()))
    lo_ext, hi_ext = min(sizes.values()), max(sizes.values())
    n_full = max(x["n"] for x in rows)
    ax.set_xscale("log")
    ax.set_xlim(40, 2000)
    ylo, ydata = ax.get_ylim()
    yhi = ydata + 0.45 * (ydata - ylo)          # headroom above the data for the labels
    ax.set_ylim(ylo, yhi)
    ax.axhline(0, color=viz.INK, lw=0.9, zorder=1)
    # The external cohorts sit within a few recordings of each other on a log axis, so they are
    # one band with one horizontal label rather than four lines whose labels would collide.
    ax.axvspan(lo_ext, hi_ext, color=viz.SERIES[4], alpha=0.18, lw=0, zorder=0)
    ax.text(np.sqrt(lo_ext * hi_ext), ydata + 0.08 * (ydata - ylo),
            f"external cohorts\n{lo_ext}–{hi_ext}", ha="center", va="bottom", fontsize=6.0,
            color=viz.INK, linespacing=1.1, zorder=4)
    # The size at which every draw is positive, and the full development cohort.
    ax.axvline(350, color=viz.INK, ls="--", lw=0.9, zorder=1)
    ax.text(350 * 0.94, ydata + 0.08 * (ydata - ylo), "≈350\nneeded", ha="right", va="bottom",
            fontsize=6.0, color=viz.INK, linespacing=1.1, zorder=4)
    # The curve's last point is the full cohort, so it is labelled rather than given a line.
    ax.text(n_full, ydata + 0.08 * (ydata - ylo), f"CAUEEG\n{n_full:,}", ha="center",
            va="bottom", fontsize=6.0, color=viz.MUTED, linespacing=1.1, zorder=4)
    ax.set_xticks([40, 100, 300, 1000])
    ax.set_xticklabels(["40", "100", "300", "1000"])
    ax.set_xlabel("recordings drawn")
    ax.set_ylabel("incremental margin")
    ax.set_title("c  Cohort size", loc="left", fontweight="bold")
    # Lower right, where the curves have converged and nothing else is drawn.
    ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.0,
              handletextpad=0.4, borderaxespad=0.3)
    viz.hgrid(ax)


#: Names for the montages the placement panel shows, in the order it shows them.
PLACE_LABEL = {
    "r_temporal": "temporal (4)", "b4": "temporo-occipital (4)",
    "b2": "temporo-parietal (2)", "b19": "full 10–20 (19)", "b8": "bilateral (8)",
    "muse": "Muse layout (4)", "insight": "Insight layout (5)",
    "r_posterior": "posterior (5)", "ganglion": "Ganglion layout (4)",
    "r_central": "central (3)", "r_frontal": "frontal (7)",
    "b1_ap": "fronto-occipital (2)", "frontal1": "frontal bipolar (2)",
}


#: Montages that carry a frontal-pole electrode (Fp1 or Fp2), the grouping the disease-specific
#: result turns on. The full montage carries them too, but it is the reference and is drawn apart.
#: The bilateral array's F3/F4 are not frontal-pole sites.
HAS_FP = {"muse", "insight", "ganglion", "b1_ap", "r_frontal", "frontal1"}
#: Placement colours. Kept apart from the contrast colours of the lower row (screening blue,
#: dementia orange, MCI teal), so no hue means two things in one figure.
C_NOFP, C_FP, C_FULL = "#4A5A66", "#87688F", "#B8BEC2"


def _fp_colour(key):
    return C_FULL if key == "b19" else (C_FP if key in HAS_FP else C_NOFP)


def _region_shades(e25):
    """Rank-based YlGnBu colour per region, identical to panel (b)'s key."""
    mech = e25.get("mechanism", {})
    strength = {r: float(np.mean([mech["per_feature"][f]["strength_by_region"][r]
                                  for f in mech["top_features"]]))
                for r in ("temporal", "posterior", "central", "frontal")}
    ranked = sorted(strength, key=strength.get)
    cmap = plt.get_cmap("YlGnBu")
    shades = {r: cmap(0.16 + 0.74 * i / (len(ranked) - 1)) for i, r in enumerate(ranked)}
    return shades, strength


def _ink_or_white(color):
    """Dark ink on a light fill, white on a dark one, by luminance."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return viz.INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def panel_placement(ax, e25, task="screening"):
    """(a) Discrimination by electrode placement in CAUEEG, at a fixed amplifier.

    Ordered by AUC rather than by electrode count, because the ordering is the finding. Bars are
    coloured by whether the montage carries a frontal-pole electrode, the grouping panel (b) turns
    on. Returns the montage order, bottom to top, so panel (b) can share the rows.
    """
    rec = e25["tasks"][task]["placement"]
    items = [(k, v) for k, v in rec.items() if v.get("family") != "control" and k in PLACE_LABEL]
    items.sort(key=lambda kv: kv[1]["auc_eeg"])
    y = np.arange(len(items))
    vals = [v["auc_eeg"] for _, v in items]
    cols = [_fp_colour(k) for k, _ in items]
    ax.barh(y, vals, color=cols, height=0.62)
    age = e25["tasks"][task]["auc_age"]
    ax.axvline(age, color=viz.INK, ls="--", lw=0.9)
    # Values just past the end of each bar, in muted ink: a label inside a bar this thin overruns
    # it, and the longest bar (0.751) leaves its label clear of the dashed age line.
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(2, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=5.4, color=viz.MUTED, zorder=4)
    ax.annotate(f"age\n{age:.3f}", (age, len(items) - 1), xytext=(2, 1),
                textcoords="offset points", ha="left", va="bottom", fontsize=5.6,
                color=viz.INK, linespacing=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([PLACE_LABEL[k] for k, _ in items], fontsize=6.0)
    ax.set_xlim(0.62, 0.81)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80])
    ax.set_xticklabels(["0.65", "0.70", "0.75", "0.80"], fontsize=6.0)
    ax.set_xlabel("AUC, EEG alone", fontsize=6.6)
    ax.set_title("a  Placement, CAUEEG", loc="left", fontweight="bold", fontsize=7.6)
    return [k for k, _ in items]


def panel_by_disease(ax_ad, ax_ftd, e33, order):
    """(b) Each montage against the full one on the external cohorts, by diagnosis.

    Alzheimer's disease in transfer (the CAUEEG rule applied unchanged to three cohorts) and
    frontotemporal dementia within cohort (CAUEEG holds 14 frontotemporal cases, too few to learn
    from), pooled across cohorts by fixed-effect weighting. Rows follow panel (a).
    """
    for ax, con, reading, title in ((ax_ad, "AD", "transfer", "b  AD, transfer"),
                                    (ax_ftd, "FTD", "within", "FTD, within cohort")):
        pooled = e33["contrasts"][con][reading]["vs_full_pooled"]
        for yi, key in enumerate(order):
            if key == "b19":
                ax.plot(0, yi, "D", ms=2.8, color=C_FULL, mec=viz.INK, mew=0.4, zorder=3)
                continue
            v = pooled[f"{key}_vs_b19"]
            col = _fp_colour(key)
            ax.plot(v["ci"], [yi, yi], color=col, lw=1.1, solid_capstyle="round", zorder=2)
            ax.plot(v["diff"], yi, "o", ms=3.2, color=col, mec="white", mew=0.4, zorder=3)
        ax.axvline(0, color=viz.INK, lw=0.7, zorder=1)
        ax.set_xlim(-0.28, 0.13)
        ax.set_xticks([-0.2, -0.1, 0, 0.1])
        ax.set_xticklabels(["−0.2", "−0.1", "0", "0.1"], fontsize=6.0)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.xaxis.grid(True, color=viz.FAINT, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.4)
    ax_ad.text(0.0, -0.20, "AUC difference from the full montage, external cohorts",
               transform=ax_ad.transAxes, fontsize=6.6, ha="left", va="top", color=viz.INK)


#: The increment over age across every check, top to bottom: (label, contrast, source).
def _increment_rows(e13, e30, e34, e35):
    t = e13["tasks"]
    full = "b19|256|f32|full"
    four = e34["tasks"]["screening"]["montages"]["b4"]["conditions"]
    pooled = e35["pooled_increment"]

    def rec(d, key="inc_margin", ci="inc_ci"):
        return d[key], d[ci]

    return [
        ("CAUEEG screening", "screening", rec(t["screening"]["budgets"][full])),
        ("CAUEEG dementia", "dementia", rec(t["dementia"]["budgets"][full])),
        ("CAUEEG MCI", "mci", rec(t["mci"]["budgets"][full])),
        ("no repeat visits", "screening", rec(e30["dedup"]["screening"]["dedup"])),
        ("4 el., clean", "screening", rec(four["clean"]["device"])),
        ("4 el., moderate field", "screening", rec(four["field_moderate"]["device"])),
        ("4 el., severe field", "screening", rec(four["field_severe"]["device"])),
        ("4 el., lost contact", "screening", rec(four["contact_bad"]["device"])),
        ("4 el., mains, clinic rule", "screening", rec(four["mains"]["clinic"])),
        ("external, 19 el.", "dementia", (pooled["b19_excl_ds004504"]["diff"],
                                          pooled["b19_excl_ds004504"]["ci"])),
        ("external, 4 el.", "dementia", (pooled["b4_excl_ds004504"]["diff"],
                                         pooled["b4_excl_ds004504"]["ci"])),
    ]


C_TASK = {"screening": C_SUB, "dementia": C_INC, "mci": C_DEEP}


def panel_increment(ax, e13, e30, e34, e35):
    """(c) What the recording adds beyond age, in every setting the paper tests."""
    rows = _increment_rows(e13, e30, e34, e35)
    y = np.arange(len(rows))[::-1]
    for yi, (label, task, (est, ci)) in zip(y, rows):
        col = C_TASK[task]
        ax.plot(ci, [yi, yi], color=col, lw=1.1, solid_capstyle="round", zorder=2)
        ax.plot(est, yi, "o", ms=3.2, color=col, mec="white", mew=0.4, zorder=3)
    ax.axvline(0, color=viz.INK, lw=0.7, zorder=1)
    # Group separators: CAUEEG and its leakage check, the simulated device, the external cohorts.
    for yb in (y[3] - 0.5, y[8] - 0.5):
        ax.axhline(yb, color=viz.FAINT, lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    # Each row label takes its contrast's colour, so the panel needs no legend over the intervals.
    for tick, (_, task, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(C_TASK[task])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.09, 0.16)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
    ax.set_xticklabels(["−0.05", "0", "0.05", "0.10", "0.15"], fontsize=6.0)
    ax.xaxis.grid(True, color=viz.FAINT, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("AUC(EEG + age) − AUC(age)", fontsize=6.6)
    ax.set_title("c  Increment over age", loc="left", fontweight="bold", fontsize=7.6)


def panel_cohort_size(ax, e20, e37):
    """(d) How many recordings the increment needs, and what cohorts of 79-89 measure.

    Curves: CAUEEG subsampled to n and cross-validated inside the subsample, 10th-90th
    percentile of 25 draws. Points: each external cohort fitted and evaluated inside itself
    (e37), on the dementia contrast, with its 95% interval -- the same reading as the curve.
    """
    from matplotlib.lines import Line2D

    rows = [r for r in e20["rows"] if r["estimator"] == "eeg17"]
    need = {}
    for task, colour in (("screening", C_SUB), ("dementia", C_INC)):
        r = sorted([x for x in rows if x["task"] == task], key=lambda x: x["n"])
        n = [x["n"] for x in r]
        ax.plot(n, [x["inc_mean"] for x in r], "-o", color=colour, lw=1.1, ms=1.8, zorder=3)
        ax.fill_between(n, [x["inc_q"][0] for x in r], [x["inc_q"][1] for x in r],
                        color=colour, alpha=0.16, lw=0, zorder=1)
        need[task] = e20["thresholds"][f"{task}|eeg17"]["smallest_n_increment_reliably_positive"]
    ax.set_xscale("log")
    ax.set_xlim(35, 1500)
    ylo, yhi = -0.17, 0.23
    ax.set_ylim(ylo, yhi)
    ax.axhline(0, color=viz.INK, lw=0.7, zorder=2)
    for task, colour in (("screening", C_SUB), ("dementia", C_INC)):
        # Stops short of the top edge, which carries the ds004504 marker and its label.
        ax.axvline(need[task], ymax=0.86, color=colour, ls="--", lw=0.8, zorder=2)
        ax.text(need[task] * 1.05, ylo + 0.008, f"n = {need[task]}", color=colour, fontsize=5.6,
                ha="left", va="bottom")
    # External cohorts: points with 95% intervals; ds004504, whose within-cohort age model is
    # weak because its cases are younger, lies above the frame and is marked at the edge.
    ext_keys = []
    for cohort, marker in (("BrainLat", "s"), ("P-ADIC", "^")):
        c = e37["cohorts"][cohort]
        v = c["estimators"]["eeg17"]
        x = c["n"]
        ax.plot([x, x], v["inc_ci"], color=C_INC, lw=0.9, zorder=4)
        ax.plot(x, v["inc_margin"], marker, ms=3.4, color="white", mec=C_INC, mew=1.0, zorder=5)
        ext_keys.append(Line2D([0], [0], marker=marker, ls="", ms=3.4, color="white", mec=C_INC,
                               mew=1.0, label=f"{cohort} ({x})"))
    # ds004504 lies above the frame (its within-cohort age model is weak, the cases being younger):
    # an arrowhead at the top edge, labelled beside it.
    ds = e37["cohorts"]["ds004504"]
    ax.plot(ds["n"], yhi - 0.006, "^", ms=3.4, color=C_INC, clip_on=False, zorder=5)
    ax.text(ds["n"] * 1.12, yhi - 0.006,
            f"ds004504 {ds['estimators']['eeg17']['inc_margin']:+.2f}",
            fontsize=5.6, color=viz.INK, ha="left", va="center")
    # The curves are named at n = 200, above and below their bands, where the panel is empty; the
    # legend holds only the cohorts.
    at = {t: next(x for x in rows if x["task"] == t and x["n"] == 200) for t in need}
    ax.text(200, at["dementia"]["inc_q"][1] + 0.008, "dementia", color=C_INC, fontsize=5.8,
            ha="center", va="bottom")
    ax.text(200, at["screening"]["inc_q"][0] - 0.012, "screening", color=C_SUB, fontsize=5.8,
            ha="center", va="top")
    ax.legend(handles=ext_keys, loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False,
              fontsize=5.6, handlelength=1.0, handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.3)
    ax.set_xticks([40, 100, 300, 1000])
    ax.set_xticklabels(["40", "100", "300", "1000"], fontsize=6.0)
    ax.tick_params(axis="y", labelsize=6.0)
    ax.set_xlabel("recordings, fitted and tested within cohort", fontsize=6.6)
    ax.set_ylabel("increment over age", fontsize=6.6)
    ax.set_title("d  Cohort size", loc="left", fontweight="bold", fontsize=7.6)
    viz.hgrid(ax)


def panel_topography(ax, e25):
    """(b) Where the discriminating markers are strongest, on the scalp.

    A standard top-down topomap (nose up). Regions are shaded by rank, so frontal and central --
    0.004 apart -- stay distinguishable while light-to-dark still reads as increasing strength. The
    key below names each colour and gives its value, strongest region first.
    """
    from matplotlib.patches import Circle, Rectangle
    from eegbudget.montages import REGION

    mech = e25.get("mechanism")
    if not mech:
        return
    # Distance from chance, averaged over the six strongest markers, because several are inverted
    # predictors and a raw AUC would rank those as uninformative.
    strength = {}
    for r in ("temporal", "posterior", "central", "frontal"):
        vals = [mech["per_feature"][f]["strength_by_region"][r] for f in mech["top_features"]]
        strength[r] = float(np.mean(vals))
    site_region = {ch: r for r, chans in REGION.items() for ch in chans}

    ranked = sorted(strength, key=strength.get)                 # weak -> strong
    cmap = plt.get_cmap("YlGnBu")
    shade = {r: cmap(0.16 + 0.74 * i / (len(ranked) - 1)) for i, r in enumerate(ranked)}

    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-2.70, 1.42)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.16, fill=False, lw=1.0, edgecolor=viz.MUTED, zorder=2))
    ax.plot([-0.16, 0, 0.16], [1.14, 1.38, 1.14], color=viz.MUTED, lw=1.0, zorder=2)
    for name, (px, py) in viz.POS.items():
        r = site_region.get(name)
        ax.add_patch(Circle((px * 0.93, py * 0.93), 0.135, facecolor=shade.get(r, viz.FAINT),
                            edgecolor=viz.INK, lw=0.5, zorder=3))

    # A single-column key below the head, strongest region first: each swatch beside its name and
    # value. One column fits the panel width where two did not, and the ranking reads top to bottom.
    sw, lx = 0.19, -0.72
    for i, r in enumerate(ranked[::-1]):
        cy = -1.46 - 0.33 * i
        ax.add_patch(Rectangle((lx, cy - sw / 2), sw, sw, facecolor=shade[r], edgecolor=viz.INK,
                               lw=0.4, zorder=3))
        ax.text(lx + sw + 0.12, cy, f"{r} {strength[r]:.02f}", ha="left", va="center",
                fontsize=6.2, color=viz.INK)
    ax.set_title("b  Marker strength", loc="center", fontweight="bold")


def panel_mechanism(ax, e25):
    """(b) Why placement matters: where each marker is actually measurable.

    Plotted as distance from chance, not raw AUC, because several markers are inverted
    predictors -- a posterior dominant rhythm that slows gives an AUC of 0.26, which is as
    informative as one of 0.74 and would otherwise appear at the bottom of the panel.
    """
    mech = e25.get("mechanism")
    if not mech:
        return
    order = ["temporal", "posterior", "central", "frontal"]
    x = np.arange(len(order))
    nice = {"pdr_frequency": "PDR frequency", "rel_theta": "relative theta",
            "aperiodic_exponent": "aperiodic exponent", "rel_beta": "relative beta",
            "theta_alpha_ratio": "theta/alpha", "slowing_ratio": "slowing ratio"}
    for i, f in enumerate(mech["top_features"]):
        s = mech["per_feature"][f]["strength_by_region"]
        ax.plot(x, [s[r] for r in order], "-o", ms=2.6, lw=1.2,
                color=viz.SERIES[i % len(viz.SERIES)], label=nice.get(f, f))
    ax.set_xticks(x)
    ax.set_xticklabels([r[:4] for r in order])
    ax.set_xlabel("scalp region")
    ax.set_ylabel("|AUC $-$ 0.5|, single feature")
    ax.set_title("b  Why: where the markers live", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", ncol=1, handlelength=1.2)
    viz.hgrid(ax)


#: The style block's text measure. Body figures are drawn at exactly this width so that nothing is
#: scaled on inclusion and a label set at 7pt reaches the page at 7pt.
BODY_W = 5.5


def figure_one(e25, e20, e13, e30, e33, e34, e35, e37):
    """The paper's results figure: placement in CAUEEG and by disease on the external cohorts
    (top), the increment over age in every setting tested, and the cohort size it needs (bottom).

    Drawn at the text width, so every label reaches the page at the size set here.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(BODY_W, 3.34))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.30, 1.0], hspace=0.74,
                  left=0.185, right=0.985, top=0.94, bottom=0.115)
    top = gs[0].subgridspec(1, 3, width_ratios=[1.30, 0.80, 0.80], wspace=0.10)
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.34)
    ax_a = fig.add_subplot(top[0])
    ax_ad, ax_ftd = fig.add_subplot(top[1]), fig.add_subplot(top[2])
    ax_c, ax_d = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])

    order = panel_placement(ax_a, e25)
    panel_by_disease(ax_ad, ax_ftd, e33, order)
    panel_increment(ax_c, e13, e30, e34, e35)
    panel_cohort_size(ax_d, e20, e37)

    place_keys = [Line2D([0], [0], marker="s", ls="", ms=4.5, color=C_NOFP,
                         label="no frontal-pole electrode"),
                  Line2D([0], [0], marker="s", ls="", ms=4.5, color=C_FP,
                         label="carries Fp1 or Fp2"),
                  Line2D([0], [0], marker="D", ls="", ms=3.4, color=C_FULL, mec=viz.INK, mew=0.4,
                         label="full montage")]
    fig.legend(handles=place_keys, ncol=3, loc="upper right", bbox_to_anchor=(0.99, 0.497),
               frameon=False, fontsize=6.0, handletextpad=0.3, columnspacing=1.2)
    viz.save(fig, paths.FIGURES / "fig1_results")


def _head(ax, strength, title, lo, hi):
    """A top-down head with each 10-20 site shaded by its region's marker strength."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle
    from eegbudget.montages import REGION

    cmap, norm = plt.get_cmap("YlGnBu"), Normalize(lo, hi)
    site_region = {ch: r for r, chans in REGION.items() for ch in chans}
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-2.35, 1.45)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.16, fill=False, lw=1.0, edgecolor=viz.MUTED, zorder=2))
    ax.plot([-0.16, 0, 0.16], [1.14, 1.38, 1.14], color=viz.MUTED, lw=1.0, zorder=2)
    for name, (px, py) in viz.POS.items():
        r = site_region.get(name)
        ax.add_patch(Circle((px * 0.93, py * 0.93), 0.135,
                            facecolor=cmap(norm(strength[r])) if r else viz.FAINT,
                            edgecolor=viz.INK, lw=0.5, zorder=3))
    for i, r in enumerate(sorted(strength, key=strength.get, reverse=True)):
        ax.text(0, -1.45 - 0.24 * i, f"{r} {strength[r]:.3f}", ha="center", va="center",
                fontsize=6.4, color=viz.INK)
    ax.set_title(title, fontsize=7.4, fontweight="bold")


def figure_topography(e25, e33):
    """Where the six strongest markers discriminate, by scalp region, in CAUEEG and in each
    external diagnosis. One colour scale across the three heads, so strengths compare directly.
    """
    mech = e25["mechanism"]
    heads = [("CAUEEG, screening",
              {r: float(np.mean([mech["per_feature"][f]["strength_by_region"][r]
                                 for f in mech["top_features"]]))
               for r in ("temporal", "posterior", "central", "frontal")}),
             ("External, AD", e33["topography"]["AD"]["top6_mean_strength_by_region"]),
             ("External, FTD",
              e33["topography"]["FTD"]["top6_mean_strength_by_region"])]
    vals = [v for _, s in heads for v in s.values()]
    fig, axes = plt.subplots(1, 3, figsize=(BODY_W * 0.86, 2.05))
    for ax, (title, s) in zip(axes, heads):
        _head(ax, s, title, min(vals), max(vals))
    fig.tight_layout(w_pad=0.6)
    viz.save(fig, paths.FIGURES / "figA10_topography")


def figure_samplesize(e20, e19):
    """The cohort-size curve, on its own so it can be read."""
    fig, ax = plt.subplots(figsize=(BODY_W * 0.62, 2.25))
    panel_complexity(ax, e20, e19)
    fig.tight_layout()
    viz.save(fig, paths.FIGURES / "figA6_samplesize")


def figure_margins(e13, e11, e19):
    """The two panels the acquisition figure displaced, kept for the appendix."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1))
    panel_margins(axes[0], e13, e11)
    panel_easycog(axes[1], e19)
    fig.tight_layout(w_pad=2.2, rect=(0, 0, 1, 0.86))
    viz.save(fig, paths.FIGURES / "figA5_margins")


# ------------------------------------------------------------------------------------------
def figure_two(e21):
    """Net benefit against age at two plausible service prevalences."""
    # Not sharey: net benefit scales with prevalence, so a shared axis sized for the 10% panel
    # runs the 20% curves off the top of the frame. Each panel gets the range its own curves need
    # and the vertical scale is stated in the caption instead.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
    for ax, prev in zip(axes, ("0.10", "0.20")):
        top = 0.0
        # All three contrasts, because the appendix quotes a net-benefit figure for each and a
        # figure that showed only two invited the reader to assume MCI behaved like screening.
        for task, colour in (("dementia", C_INC), ("screening", C_SUB), ("mci", C_DEEP)):
            rec = e21["tasks"][task]["budgets"]["full montage"][prev]["curves"]
            t = [r["threshold"] for r in rec["age"]]
            for key, style, lw, lab in (("eeg+age", "-", 1.6, "EEG + age"),
                                        ("age", "--", 1.1, "age alone")):
                v = [r["net_benefit"] for r in rec[key]]
                top = max(top, max(v))
                ax.plot(t, v, style, color=colour, lw=lw, label=f"{NICE[task]}: {lab}")
        ref = e21["tasks"]["dementia"]["budgets"]["full montage"][prev]["curves"]["age"]
        ax.plot([r["threshold"] for r in ref], [r["net_benefit_treat_all"] for r in ref],
                ":", color=viz.MUTED, lw=1.0, label="refer everyone")
        ax.axhline(0, color=viz.INK, lw=0.9)
        ax.set_xlabel("threshold probability for referral")
        ax.set_title(f"service prevalence {float(prev):.0%}", loc="left", fontweight="bold")
        ax.set_ylim(-0.02, top * 1.12)
        ax.set_ylabel("net benefit")
        viz.hgrid(ax)
    # Colour carries the contrast, line style the arm, so the key splits cleanly into two rows
    # below the panels -- clear of the curves and the zero line, which the per-panel legend was not.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    contrast_keys = [Patch(facecolor=C_INC, edgecolor="none", label="dementia"),
                     Patch(facecolor=C_SUB, edgecolor="none", label="screening"),
                     Patch(facecolor=C_DEEP, edgecolor="none", label="MCI")]
    arm_keys = [Line2D([0], [0], color=viz.INK, lw=1.6, ls="-", label="EEG + age"),
                Line2D([0], [0], color=viz.INK, lw=1.1, ls="--", label="age alone"),
                Line2D([0], [0], color=viz.MUTED, lw=1.0, ls=":", label="refer everyone")]
    # matplotlib fills legend columns top-to-bottom, so interleave to get contrasts on the top row
    # and line styles on the bottom row rather than one of each stacked per column.
    ordered = [contrast_keys[0], arm_keys[0], contrast_keys[1], arm_keys[1],
               contrast_keys[2], arm_keys[2]]
    fig.legend(handles=ordered, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=6.3,
               columnspacing=1.8, handlelength=1.7, handletextpad=0.6)
    fig.tight_layout(w_pad=1.6, rect=(0, 0.17, 1, 1))
    viz.save(fig, paths.FIGURES / "fig2_utility")


def main():
    viz.use_style()
    paths.ensure_dirs()
    e13 = load("e13_incremental.json")
    e19 = load("e19_easycog_audit.json")
    e20 = load("e20_estimator_complexity.json")
    e21 = load("e21_clinical_utility.json")
    e11 = load("e11_deep_arm.json")
    e25 = load("e25_acquisition_design.json")
    e30, e33 = load("e30_noverlap_leakage.json"), load("e33_external_placement.json")
    e34, e35 = load("e34_acquisition_stress.json"), load("e35_external_transfer_all.json")
    e37 = load("e37_external_within_increment.json")
    needed = (("e13", e13), ("e19", e19), ("e20", e20), ("e21", e21), ("e25", e25), ("e30", e30),
              ("e33", e33), ("e34", e34), ("e35", e35), ("e37", e37))
    missing = [n for n, v in needed if not v]
    if missing:
        raise SystemExit(f"missing artefacts: {missing}")
    figure_one(e25, e20, e13, e30, e33, e34, e35, e37)
    figure_topography(e25, e33)
    figure_margins(e13, e11, e19)
    figure_samplesize(e20, e19)
    figure_two(e21)
    print(f"wrote {paths.FIGURES / 'fig1_results'}.pdf, figA10_topography.pdf, "
          f"{paths.FIGURES / 'fig2_utility'}.pdf and {paths.FIGURES / 'figA5_margins'}.pdf")


if __name__ == "__main__":
    main()
