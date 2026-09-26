"""Figure system: muted light-theme scientific plots, and clinician-facing decision views.

Two audiences, one visual language. Reviewers need the budget frontier and the substitution
curve; clinicians need to see why a particular patient was flagged. Both are drawn from the same
palette so a figure lifted from the paper into a clinic slide does not change appearance.

Design rules held throughout: light ground, desaturated ink-forward palette, no chartjunk, no
gridlines competing with data, semantic colour reserved for normal/borderline/abnormal and never
reused as a categorical hue. Deviation is drawn on a diverging scale centred at zero because zero
is the meaningful midpoint -- an age-expected value, not an arbitrary baseline.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
from pathlib import Path

INK = "#22282D"
MUTED = "#6B7780"
FAINT = "#C8CFD3"
GROUND = "#FFFFFF"
PANEL = "#F4F6F5"

#: Categorical hues: desaturated, distinguishable in greyscale by lightness ordering.
SERIES = ["#3E6B8A", "#4F8A80", "#B0764F", "#7D8F63", "#87688F", "#9AA0A4"]

#: Semantic, used only for clinical status. Never as a categorical hue.
NORMAL, BORDERLINE, ABNORMAL = "#5F8A6B", "#C29049", "#A85549"

#: Diverging scale for deviation maps, centred on the age-expected value.
DEVIATION_CMAP = LinearSegmentedColormap.from_list(
    "deviation", ["#3E6B8A", "#8FA9BC", "#EDEFEE", "#D2A98F", "#A85549"])

#: Approximate 10-20 positions on a unit head, nose up.
POS = {"Fp1": (-.31, .95), "Fp2": (.31, .95),
       "F7": (-.81, .59), "F3": (-.42, .55), "Fz": (0, .5), "F4": (.42, .55), "F8": (.81, .59),
       "T3": (-.93, 0), "C3": (-.47, 0), "Cz": (0, 0), "C4": (.47, 0), "T4": (.93, 0),
       "T5": (-.81, -.59), "P3": (-.42, -.55), "Pz": (0, -.5), "P4": (.42, -.55), "T6": (.81, -.59),
       "O1": (-.31, -.95), "O2": (.31, -.95)}


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": GROUND, "axes.facecolor": GROUND, "savefig.facecolor": GROUND,
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.edgecolor": FAINT, "axes.linewidth": .7, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.width": .7, "ytick.major.width": .7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": "#E8EBEA", "grid.linewidth": .7, "axes.grid": False,
        "legend.frameon": False, "legend.fontsize": 7.5, "legend.handlelength": 1.4,
        "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
        # A tight bounding box can still clip an axis label by a fraction of a point when the
        # label is wider than the axes it belongs to; a little padding costs nothing.
        "savefig.pad_inches": 0.05,
        # Embed real fonts in the PDF rather than outlining glyphs to curves: type 42 keeps the
        # axis labels selectable and searchable in the submitted paper, and keeps the file small.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "pdf.compression": 6,
    })


def hgrid(ax):
    """Horizontal reference lines only, behind the data."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)


def head_outline(ax, lw=1.0):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=MUTED, lw=lw, zorder=1))
    ax.add_patch(Polygon([(-.10, .995), (0, 1.14), (.10, .995)], closed=True,
                         fill=False, ec=MUTED, lw=lw, zorder=1))
    # Ears sit clear of T3/T4 (radius .93 + marker .105), so a carried temporal electrode never
    # merges with the outline.
    for s in (-1, 1):
        ax.add_patch(Circle((s * 1.18, 0), .07, fill=False, ec=MUTED, lw=lw, zorder=1))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.30, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")


def montage_head(ax, channels, title=None, active=None, vmax=None, label="carried"):
    """Draw the 10-20 layout, marking which electrodes a montage carries.

    `active` optionally maps a channel to a value, drawn on the deviation scale; otherwise
    carried electrodes are filled and absent ones drawn as faint outlines.

    `label` controls which sites get a name. Nineteen names on a head the width of a journal
    column collide with each other, with the electrodes of the row below, and with the head
    outline, and the collisions get worse the smaller the panel. ``"carried"`` names only the
    electrodes the device actually has -- which is the panel's subject -- and leaves the absent
    sites as unlabelled ghosts. ``"all"`` restores the exhaustive labelling and ``"none"``
    suppresses it entirely. Whatever is drawn gets a white halo so a name is legible where it
    crosses the head outline.
    """
    head_outline(ax)
    halo = [pe.withStroke(linewidth=1.8, foreground=GROUND)]
    for ch, (x, y) in POS.items():
        on = ch in channels
        if active and ch in active:
            v = active[ch]
            lim = vmax or max(1e-9, np.nanmax(np.abs(list(active.values()))))
            ax.add_patch(Circle((x, y), .115, facecolor=DEVIATION_CMAP(0.5 + 0.5 * v / lim),
                                ec=INK, lw=.6, zorder=3))
        else:
            ax.add_patch(Circle((x, y), .105, facecolor=INK if on else "none",
                                ec=INK if on else FAINT, lw=.8, zorder=3))
        show = label == "all" or (label == "carried" and (on or (active and ch in active)))
        if show:
            # Names go below their electrode except on the frontal pole, where below is directly
            # on top of the next row down. Fp1/Fp2 are the only sites with a neighbour that close,
            # and flipping them above the head clears the one collision the layout produces.
            above = y > .7
            ax.text(x, y + (.19 if above else -.19), ch, ha="center",
                    va="bottom" if above else "top", fontsize=6.2,
                    color=INK if on else FAINT, zorder=5, path_effects=halo)
    if title:
        ax.set_title(title)


#: Report names for the qEEG markers, for the clinician-facing deviation panel. The feature keys
#: are machine names and "ap gradient alpha" is not what a report would call it.
DISPLAY = {
    "theta_alpha_ratio": "theta/alpha ratio", "slowing_ratio": "slowing ratio",
    "rel_delta": "relative delta", "rel_theta": "relative theta", "rel_alpha": "relative alpha",
    "rel_beta": "relative beta", "rel_gamma": "relative gamma",
    "pdr_frequency": "PDR frequency", "pdr_prominence": "PDR prominence",
    "aperiodic_exponent": "aperiodic exponent", "aperiodic_offset": "aperiodic offset",
    "spectral_entropy": "spectral entropy", "spectral_edge": "spectral edge",
    "median_frequency": "median frequency", "hjorth_mobility": "Hjorth mobility",
    "hjorth_complexity": "Hjorth complexity", "alpha_asymmetry": "alpha asymmetry",
    "ap_gradient_alpha": "anterior-posterior alpha",
}


def deviation_bars(ax, z, names, title=None, flag=2.0, top=None):
    """Per-feature deviation from the age-expected value -- the clinician's read.

    Bars are signed so direction is visible (slowing shows as a positive theta deviation, not as
    an unsigned magnitude), and the |z| > flag band is shaded so abnormality is read positionally
    rather than by decoding a colour. `top` keeps only the largest deviations, for a panel that has
    to fit a page rather than a screen.
    """
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    order = np.argsort(np.where(ok, np.abs(z), -np.inf))[::-1]
    order = [i for i in order if ok[i]]
    if top is not None:
        order = order[:top]
    vals = z[order]
    lbl = [DISPLAY.get(names[i], names[i].replace("_", " ")) for i in order]
    y = np.arange(len(order))

    lim = max(3.0, np.abs(vals).max() * 1.15)
    ax.axvspan(flag, lim, color=ABNORMAL, alpha=.06, lw=0)
    ax.axvspan(-lim, -flag, color=ABNORMAL, alpha=.06, lw=0)
    ax.axvline(0, color=MUTED, lw=.9, zorder=2)

    cols = [ABNORMAL if abs(v) >= flag else (BORDERLINE if abs(v) >= 1.0 else SERIES[0])
            for v in vals]
    ax.barh(y, vals, color=cols, height=.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(lbl)
    ax.invert_yaxis()
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("deviation from age-expected value (z)")
    if title:
        ax.set_title(title)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def frontier(ax, x, series, xlabel="electrodes", ylabel="AUC", logx=True):
    """Budget frontier: one line per arm, annotated at the last point rather than in a legend box."""
    for i, (label, (vals, lo, hi)) in enumerate(series.items()):
        c = SERIES[i % len(SERIES)]
        vals = np.asarray(vals, float)
        if lo is not None:
            ax.fill_between(x, lo, hi, color=c, alpha=.13, lw=0)
        ax.plot(x, vals, "-o", color=c, label=label, zorder=3)
        j = int(np.max(np.where(np.isfinite(vals))))
        ax.annotate(label, (x[j], vals[j]), textcoords="offset points", xytext=(7, 0),
                    color=c, fontsize=7.5, va="center", weight="semibold")
    ax.axhline(.5, color=FAINT, lw=.9, ls=(0, (3, 3)), zorder=1)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    hgrid(ax)


def lifespan(ax, ages, values, mu, sigma, age_grid, patient=None, title=None, ylabel=""):
    """A feature against age, with the normative band -- the growth-chart view of a brain measure.

    Clinicians read paediatric growth charts fluently, and this is the same object: a population
    trajectory, a normal range, and one patient's position in it. It answers 'is this abnormal
    *for this person's age*', which is the question a raw number cannot.
    """
    ax.scatter(ages, values, s=5, color=FAINT, alpha=.75, lw=0, zorder=2)
    for k, a in ((2, .13), (1, .20)):
        ax.fill_between(age_grid, mu - k * sigma, mu + k * sigma, color=SERIES[0], alpha=a, lw=0,
                        zorder=1)
    ax.plot(age_grid, mu, color=SERIES[0], lw=1.4, zorder=3)
    if patient is not None:
        pa, pv = patient
        ax.scatter([pa], [pv], s=52, marker="D", facecolor=ABNORMAL, ec="white", lw=1.1, zorder=5)
        ax.annotate("this patient", (pa, pv), textcoords="offset points", xytext=(9, -3),
                    color=ABNORMAL, fontsize=7.0, weight="semibold", zorder=5,
                    va="top")
    ax.set_xlabel("age (years)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    hgrid(ax)


def triage_bar(ax, score, thr, lo=None, hi=None):
    """Where one patient's abnormality score sits relative to the referral threshold."""
    lo = lo if lo is not None else 0.0
    hi = hi if hi is not None else max(score, thr) * 1.35
    ax.axvspan(lo, thr, color=NORMAL, alpha=.13, lw=0)
    ax.axvspan(thr, hi, color=ABNORMAL, alpha=.13, lw=0)
    ax.axvline(thr, color=INK, lw=1.0, zorder=3)
    ax.scatter([score], [0], s=90, marker="D", facecolor=ABNORMAL if score > thr else NORMAL,
               ec="white", lw=1.2, zorder=4)
    ax.annotate("refer" if score > thr else "no referral", (score, 0),
                textcoords="offset points", xytext=(0, 15), ha="center",
                color=ABNORMAL if score > thr else NORMAL, fontsize=8, weight="semibold")
    ax.annotate("referral threshold", (thr, 0), textcoords="offset points", xytext=(4, -18),
                fontsize=7, color=MUTED)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-.5, .6)
    ax.set_yticks([])
    ax.set_xlabel("overall deviation from age-expected EEG")
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, path, close=True):
    """Write the figure as vector PDF and as PNG, and return the PDF.

    The PDF is what the paper includes: a raster figure in a print-resolution submission shows
    its pixels wherever a reviewer zooms, and axis text stops being selectable or searchable. The
    PNG is kept beside it for the README and for quick visual checks, so both stay in step.

    ``bbox_inches='tight'`` comes from the rcParams and is what keeps rotated tick labels and
    outboard annotations inside the canvas instead of clipped at the figure edge.
    """
    path = Path(path)
    pdf = path.with_suffix(".pdf")
    for target in (pdf, path.with_suffix(".png")):
        fig.savefig(target)
    if close:
        plt.close(fig)
    return pdf
