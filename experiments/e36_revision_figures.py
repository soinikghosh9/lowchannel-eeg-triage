"""Appendix figures for the camera-ready additions: placement by disease, and the stress test.

    figA8_external_placement   every montage against the full one on the external cohorts, pooled
                               by fixed-effect weighting: Alzheimer's disease in transfer (three
                               cohorts), frontotemporal dementia within cohort (two). Read with A.12.
    figA9_stress               what each injected field fault costs: EEG-only AUC change against the
                               clean recording, and the increment over age, for a device-trained rule
                               at 19 and 4 electrodes and a clinic-trained rule at 4. Read with A.13.

Drawn from the result files alone (e33, e34), as vector PDF with editable text.

    python experiments/e36_revision_figures.py
Out: outputs/figures/figA8_external_placement.{pdf,png}, figA9_stress.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from eegbudget import paths, viz  # noqa: E402

#: Montage order and names as in Figure 2a, with the region colour that figure uses.
TP, POST, CEN, FRONT, OTHER = "#1F2F86", "#2F86C0", "#6CC4B8", "#DCEFA6", viz.SERIES[5]
MONTAGES = [("r_temporal", "temporal (4)", TP), ("b4", "temporo-occipital (4)", OTHER),
            ("b2", "temporo-parietal (2)", TP), ("b8", "bilateral (8)", OTHER),
            ("muse", "Muse layout (4)", OTHER), ("insight", "Insight layout (5)", OTHER),
            ("r_posterior", "posterior (5)", POST), ("ganglion", "Ganglion layout (4)", OTHER),
            ("b1_ap", "fronto-occipital (2)", OTHER), ("r_central", "central (3)", CEN),
            ("r_frontal", "frontal (7)", FRONT), ("frontal1", "frontal bipolar (2)", FRONT)]
#: Montages that carry a frontal-pole electrode (Fp1 or Fp2). The bilateral array's F3/F4 are not
#: frontal-pole sites, and it behaves like the montages without one.
HAS_FRONTAL = {"muse", "insight", "ganglion", "b1_ap", "r_frontal", "frontal1"}
#: The frontal-pole colours of Figure 2's placement row, so the grouping reads the same in both.
C_NOFP, C_FP = "#4A5A66", "#87688F"


def _load(name):
    return json.loads((paths.RESULTS / name).read_text(encoding="utf-8"))


def _forest(ax, rows, title, xlim):
    y = np.arange(len(rows))[::-1]
    for yi, (label, est, lo, hi, frontal) in zip(y, rows):
        col = C_FP if frontal else C_NOFP
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(est, yi, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3)
    ax.axvline(0, color=viz.INK, lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("AUC difference from full montage")
    ax.set_title(title, loc="left", fontweight="bold", pad=13)   # room for the subtitle below it
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def figure_external(e33):
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.9), sharey=True)
    for ax, (con, reading, title, sub) in zip(axes, (
            ("AD", "transfer", "a  Alzheimer's disease", "CAUEEG rule transferred, 3 cohorts pooled"),
            ("FTD", "within", "b  Frontotemporal dementia", "within-cohort CV, 2 cohorts pooled"))):
        ax.text(0.0, 1.005, sub, transform=ax.transAxes, fontsize=6.6, color=viz.MUTED,
                ha="left", va="bottom")
        pooled = e33["contrasts"][con][reading]["vs_full_pooled"]
        rows = []
        for key, label, _ in MONTAGES:
            v = pooled[f"{key}_vs_b19"]
            rows.append((label, v["diff"], v["ci"][0], v["ci"][1], key in HAS_FRONTAL))
        _forest(ax, rows, title, (-0.30, 0.16))
    axes[1].tick_params(labelleft=False)
    keys = [Line2D([0], [0], marker="o", color=C_NOFP, lw=1.4, ms=4,
                   label="no frontal-pole electrode"),
            Line2D([0], [0], marker="o", color=C_FP, lw=1.4, ms=4,
                   label="carries Fp1 or Fp2")]
    fig.legend(handles=keys, ncol=2, loc="lower center", bbox_to_anchor=(0.55, 0.0),
               frameon=False, fontsize=7.2)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    viz.save(fig, paths.FIGURES / "figA8_external_placement")


FAULTS = [("noise1", "electrode noise 1 µV"), ("noise2", "noise 2 µV"), ("noise5", "noise 5 µV"),
          ("noise10", "noise 10 µV"), ("mains", "mains 50 Hz, unfiltered"),
          ("motion2", "movement 2/min"), ("motion6", "movement 6/min"),
          ("contact_bad", "lost contact, undetected"), ("contact_drop", "lost contact, dropped"),
          ("displace50", "misplacement, half-site"), ("displace100", "misplacement, full site"),
          ("short12", "48 s of signal"), ("short6", "24 s"), ("short3", "12 s"),
          ("field_moderate", "field, moderate"), ("field_severe", "field, severe")]
ARMS = [("b19", "device", "19 electrodes, device-trained", viz.SERIES[5], -0.22),
        ("b4", "device", "4 electrodes, device-trained", viz.SERIES[0], 0.0),
        ("b4", "clinic", "4 electrodes, clinic-trained", viz.SERIES[2], 0.22)]


def figure_stress(e34):
    t = e34["tasks"]["screening"]["montages"]
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.9), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(FAULTS))[::-1]
    for mont, reg, label, col, off in ARMS:
        for yi, (key, _) in zip(y, FAULTS):
            v = t[mont]["conditions"][key][reg]
            axes[0].plot(v["d_auc_eeg_ci"], [yi + off] * 2, color=col, lw=1.2, zorder=2)
            axes[0].plot(v["d_auc_eeg_vs_clean"], yi + off, "o", ms=3.4, color=col,
                         mec="white", mew=0.5, zorder=3)
            axes[1].plot(v["inc_ci"], [yi + off] * 2, color=col, lw=1.2, zorder=2)
            axes[1].plot(v["inc_margin"], yi + off, "o", ms=3.4, color=col,
                         mec="white", mew=0.5, zorder=3)
    clean = t["b4"]["conditions"]["clean"]["device"]["inc_margin"]
    axes[1].axvline(clean, color=viz.MUTED, lw=0.8, ls=":", zorder=1)
    axes[1].text(clean, len(FAULTS) - 0.35, " clean", fontsize=6.6, color=viz.MUTED,
                 va="bottom", ha="left")
    for ax, title, xlabel in ((axes[0], "a  EEG-only AUC", "change from the clean recording"),
                              (axes[1], "b  Increment over age", "AUC(EEG + age) − AUC(age)")):
        ax.axvline(0, color=viz.INK, lw=0.8, zorder=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(FAULTS) - 0.1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f for _, f in FAULTS])
    axes[0].set_xlim(-0.22, 0.04)
    axes[1].set_xlim(-0.09, 0.07)
    axes[1].tick_params(labelleft=False)
    keys = [Line2D([0], [0], marker="o", color=c, lw=1.2, ms=3.6, label=lab)
            for _, _, lab, c, _ in ARMS]
    fig.legend(handles=keys, ncol=3, loc="lower center", bbox_to_anchor=(0.55, 0.0),
               frameon=False, fontsize=7.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    viz.save(fig, paths.FIGURES / "figA9_stress")


def main():
    viz.use_style()
    paths.ensure_dirs()
    figure_external(_load("e33_external_placement.json"))
    figure_stress(_load("e34_acquisition_stress.json"))
    print(f"wrote {paths.FIGURES / 'figA8_external_placement'}.pdf, "
          f"{paths.FIGURES / 'figA9_stress'}.pdf")


if __name__ == "__main__":
    main()
