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


#: Montages over a single lobe take that lobe's colour from panel (b); the rest are neutral grey.
MONTAGE_REGION = {"r_temporal": "temporal", "b2": "temporal", "r_posterior": "posterior",
                  "r_central": "central", "r_frontal": "frontal", "frontal1": "frontal"}


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
    """(a) Discrimination by electrode placement, at a fixed amplifier.

    Ordered by AUC rather than by electrode count, because the ordering is the finding: the
    montages carrying the fewest electrodes sit at the top and a seven-electrode frontal array
    sits at the bottom. Commodity layouts are drawn in a second colour -- they are nearest-site
    approximations of headset geometry on clinical hardware, so they belong on the axis but must
    not read as measurements of those devices.
    """
    rec = e25["tasks"][task]["placement"]
    items = [(k, v) for k, v in rec.items() if v.get("family") != "control" and k in PLACE_LABEL]
    items.sort(key=lambda kv: kv[1]["auc_eeg"])
    y = np.arange(len(items))
    vals = [v["auc_eeg"] for _, v in items]
    shades, _ = _region_shades(e25)
    grey = viz.SERIES[5]
    cols = [shades[MONTAGE_REGION[k]] if k in MONTAGE_REGION else grey for k, _ in items]
    ax.barh(y, vals, color=cols, height=0.68)
    age = e25["tasks"][task]["auc_age"]
    ax.axvline(age, color=viz.INK, ls="--", lw=1.0)
    # The reference line spans the full height, so any in-axes legend crosses it and the shortest
    # bars. Both annotations therefore sit outside the bars: the line is labelled above the panel,
    # and the colour key goes below the x-axis.
    # Values sit inside the right end of each bar, in white, so no label reaches the dashed age
    # line. The bars are all long enough (shortest 0.675) to hold three decimals.
    for yi, v, c in zip(y, vals, cols):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(-3, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=5.4, color=_ink_or_white(c), zorder=4)
    # Age reference above the top bar, ending to the left of its line.
    ax.annotate(f"age {age:.3f}", (age, len(items) - 1), xytext=(-3, 7),
                textcoords="offset points", ha="right", va="bottom", fontsize=6.2,
                color=viz.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([PLACE_LABEL[k] for k, _ in items], fontsize=6.2)
    ax.set_xlim(0.62, 0.80)
    ax.set_ylim(-0.7, len(items) + 0.5)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80])
    ax.set_xlabel("AUC, EEG alone")
    ax.set_title("a  Placement", loc="left", fontweight="bold")


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


def figure_one(e25, e20, e19):
    """Placement, its mechanism, and the cohort size the whole argument needs.

    Drawn slightly wider than the text block so three panels fit without shrinking the montage
    labels below legibility; the modest reduction on inclusion still leaves every label above 5pt,
    and the printed height is no greater than the two-panel version it replaces.
    """
    fig, axes = plt.subplots(1, 3, figsize=(5.95, 2.10),
                             gridspec_kw={"width_ratios": [1.30, 1.02, 1.78]})
    panel_placement(axes[0], e25)
    panel_topography(axes[1], e25)
    panel_complexity(axes[2], e20, e19)
    fig.tight_layout(w_pad=0.9)
    viz.save(fig, paths.FIGURES / "fig1_results")


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
    missing = [n for n, v in (("e13", e13), ("e19", e19), ("e20", e20), ("e21", e21)) if not v]
    if missing:
        raise SystemExit(f"missing artefacts: {missing}")
    if not e25:
        raise SystemExit("missing artefacts: ['e25']")
    figure_one(e25, e20, e19)
    figure_margins(e13, e11, e19)
    figure_samplesize(e20, e19)
    figure_two(e21)
    print(f"wrote {paths.FIGURES / 'fig1_results'}.pdf, "
          f"{paths.FIGURES / 'fig2_utility'}.pdf and {paths.FIGURES / 'figA5_margins'}.pdf")


if __name__ == "__main__":
    main()
