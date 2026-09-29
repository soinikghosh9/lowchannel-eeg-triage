"""Appendix figure: the montage decomposition, drawn as devices.

Each head shows which 10-20 electrodes a simulated montage carries (filled) and which it does not
(faint outline). Three groups: the nested resource ladder that halves the array while keeping
temporo-parietal sites, the four single-lobe regional arrays, and the commodity headset layouts
mapped to their nearest 10-20 sites. Bipolar montages, which yield one signal from two electrodes,
carry a link between the pair.

The same heads, as the five-rung ladder strip below the boxes of Figure 1, are written for
tools/patch_framework_figure.py to place into the diagrams.net figure.

    python experiments/e29_montage_map.py
Out: outputs/figures/figA7_montages.{pdf,png}, fig1_ladder.{pdf,svg,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eegbudget import paths, viz  # noqa: E402
from eegbudget.montages import ALL  # noqa: E402

# Each row is a group of montages; the label is what the row shows.
#: Titles match Figure 2a and Figure A8, so a montage has one name throughout the paper.
ROWS = [
    ("Resource\nladder", [("b19", "full 10–20 (19)"), ("b8", "bilateral (8)"),
                          ("b4", "temporo-occipital (4)"), ("b2", "temporo-parietal (2)"),
                          ("b1_ap", "fronto-occipital (2)")]),
    ("Regional\narrays", [("r_temporal", "temporal (4)"), ("r_posterior", "posterior (5)"),
                          ("r_central", "central (3)"), ("r_frontal", "frontal (7)")]),
    ("Commodity\nlayouts", [("muse", "Muse layout (4)"), ("ganglion", "Ganglion layout (4)"),
                            ("insight", "Insight layout (5)"), ("frontal1", "frontal bipolar (2)")]),
]


def main():
    viz.use_style()
    paths.ensure_dirs()
    ncol = max(len(items) for _, items in ROWS)
    fig, axes = plt.subplots(len(ROWS), ncol, figsize=(5.5, 3.35))

    for ri, (group, items) in enumerate(ROWS):
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci >= len(items):
                ax.axis("off")
                continue
            name, label = items[ci]
            _draw_head(ax, name)
            ax.set_title(label, fontsize=6.4, pad=2)
        # Row-group label to the left of the first head; clip off so it can sit outside the axes.
        axes[ri][0].text(-1.85, 0.06, group, rotation=90, va="center", ha="center",
                         fontsize=7.2, fontweight="semibold", color=viz.INK, clip_on=False)

    fig.tight_layout(h_pad=0.6, w_pad=0.3, rect=(0.04, 0, 1, 1))
    # After the layout, so the key's text cannot change the spacing of the heads.
    _key(axes[1][ncol - 1])
    viz.save(fig, paths.FIGURES / "figA7_montages")
    print(f"wrote {paths.FIGURES / 'figA7_montages'}.pdf")
    ladder_strip()


#: The resource ladder as Figure 1 shows it, below the Input and Sensor boxes: name, then what the
#: montage carries.
LADDER = [("b19", "full 10–20", "19 electrodes"), ("b8", "bilateral", "8 electrodes"),
          ("b4", "temporo-occipital", "4 electrodes"), ("b2", "temporo-parietal", "2 electrodes"),
          ("b1_ap", "fronto-occipital", "2 electrodes → 1 signal")]
#: The strip fills the image cell of the Figure 1 diagram (303.8 x 61.9 pt in the exported PDF).
#: The right 16 pt stay empty: the "4 Arms" box of the diagram overlaps that edge.
STRIP_PT = (303.8, 61.9)
STRIP_RIGHT_PT = 16.0


def _key(ax):
    """What the symbols mean, drawn in an empty grid cell so the figure reads without the caption."""
    from matplotlib.patches import Circle, Polygon
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    rows = [0.90, 0.72, 0.54, 0.36, 0.18]
    x0, tx = 0.10, 0.24
    ax.add_patch(Circle((x0, rows[0]), 0.045, facecolor=viz.INK, ec=viz.INK, lw=0.8))
    ax.add_patch(Circle((x0, rows[1]), 0.045, facecolor="none", ec=viz.FAINT, lw=0.8))
    ax.plot([x0 - 0.07, x0 + 0.07], [rows[2] - 0.02, rows[2] + 0.02], color=viz.SERIES[2], lw=1.1)
    ax.add_patch(Polygon([(x0 - 0.045, rows[3] - 0.035), (x0, rows[3] + 0.05),
                          (x0 + 0.045, rows[3] - 0.035)], closed=True, fill=False,
                         ec=viz.MUTED, lw=1.0))
    ax.add_patch(Circle((x0, rows[4]), 0.03, fill=False, ec=viz.MUTED, lw=1.0))
    for y, text in zip(rows, ("recorded", "not recorded", "bipolar pair (1 signal)",
                              "nose (view from above)", "ear")):
        ax.text(tx, y, text, fontsize=6.0, va="center", ha="left", color=viz.INK)
    ax.text(0.0, 1.06, "(n), electrodes recorded", fontsize=6.0, va="bottom", ha="left",
            color=viz.INK, fontweight="semibold")


def _draw_head(ax, name):
    m = ALL[name]
    viz.montage_head(ax, set(m["channels"]), label="none")
    # A bipolar montage is one derived signal from two electrodes; draw the link so the head does
    # not read as two independent channels. The link bows gently toward the head centre, so a long
    # one (Fp1-O1) clears the F3/P3 markers a straight chord would graze.
    if m["ref"] == "bipolar":
        for a, b in m["pairs"]:
            p0, p2 = np.array(viz.POS[a]), np.array(viz.POS[b])
            mid = (p0 + p2) / 2
            n = np.array([-(p2 - p0)[1], (p2 - p0)[0]])
            n /= np.linalg.norm(n)
            if n @ mid > 0:
                n = -n
            ctrl = mid + 2 * 0.09 * n        # peak offset 0.09 head radii
            t = np.linspace(0, 1, 40)[:, None]
            curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p2
            ax.plot(curve[:, 0], curve[:, 1], color=viz.SERIES[2], lw=1.1, zorder=4)


def ladder_strip():
    """The five-rung ladder for Figure 1, transparent, as PDF (placed into the exported figure) and
    SVG (embedded in the diagrams.net source)."""
    w_pt, h_pt = STRIP_PT
    fig = plt.figure(figsize=(w_pt / 72, h_pt / 72))
    fig.patch.set_alpha(0)
    usable = (w_pt - STRIP_RIGHT_PT) / w_pt
    slot = usable / len(LADDER)
    head_h = 0.77                                     # of the strip height
    head_w = head_h * h_pt / w_pt * (2.6 / 2.72)      # the head axes' data aspect
    for i, (name, title, carries) in enumerate(LADDER):
        cx = slot * (i + 0.5)
        ax = fig.add_axes((cx - head_w / 2, 0.0, head_w, head_h))
        ax.patch.set_alpha(0)
        _draw_head(ax, name)
        fig.text(cx, 0.985, title, ha="center", va="top", fontsize=5.8, fontweight="semibold",
                 color=viz.INK)
        fig.text(cx, 0.855, carries, ha="center", va="top", fontsize=5.2, color=viz.MUTED)
    out = paths.FIGURES / "fig1_ladder"
    # The full canvas, not the style's tight crop: the strip must fill the diagram cell exactly.
    with plt.rc_context({"savefig.bbox": "standard", "svg.fonttype": "path"}):
        for ext, dpi in ((".pdf", None), (".svg", None), (".png", 400)):
            fig.savefig(out.with_suffix(ext), transparent=True, **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
    print(f"wrote {out}.pdf, .svg")


if __name__ == "__main__":
    main()
