"""Electrode budgets and the canonical channel order.

Two facts govern this module.

**Channel order is not consistent in the source cache.** 1,529 recordings carry the CAUEEG
ordering and 183 an alphabetical one, so every array is reindexed to CANON on load and a montage
is always a set of names, never a set of positions.

**The cache is already common-average referenced across all 19 electrodes.** Dropping columns
therefore does not simulate a low-channel device: the survivors still carry a reference built from
electrodes the device does not have. Re-referencing to the subset removes it exactly --

    x_car[i] - mean_S(x_car) = (x[i] - mean_19) - (mean_S(x) - mean_19) = x[i] - mean_S(x)

-- so subset re-referencing and bipolar derivations are both recoverable from the cache. A true
single-ended recording against a physical reference is not, because the original reference signal
was destroyed by the average; single-channel devices are simulated as bipolar pairs, which is what
such hardware measures anyway.
"""
from __future__ import annotations

import numpy as np

CANON = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
         "T3", "C3", "Cz", "C4", "T4",
         "T5", "P3", "Pz", "P4", "T6",
         "O1", "O2"]
IX = {c: i for i, c in enumerate(CANON)}

REGION = {"frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
          "temporal": ["T3", "T4", "T5", "T6"],
          "central": ["C3", "Cz", "C4"],
          "posterior": ["P3", "Pz", "P4", "O1", "O2"]}

LATERAL_PAIRS = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T3", "T4"),
                 ("C3", "C4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2")]


def _m(name, channels, ref="subset_car", pairs=(), family="anatomical", note=""):
    bad = [c for c in list(channels) + [c for p in pairs for c in p] if c not in IX]
    if bad:
        raise ValueError(f"{name}: sites not in the 10-20 cache: {sorted(set(bad))}")
    return {"name": name, "channels": sorted(set(channels), key=IX.get), "ref": ref,
            "pairs": list(pairs), "family": family, "note": note}


def n_electrodes(m):
    """Electrodes the device must physically carry -- the quantity that sets its cost."""
    return len(m["channels"])


def n_signals(m):
    return len(m["pairs"]) if m["ref"] == "bipolar" else len(m["channels"])


def apply_montage(x, m):
    """Derive a montage's signals from a canonical 19-channel array, shape (..., 19, T)."""
    if x.shape[-2] != len(CANON):
        raise ValueError(f"expected {len(CANON)} channels, got {x.shape[-2]}")
    if m["ref"] == "bipolar":
        a = [IX[p[0]] for p in m["pairs"]]
        b = [IX[p[1]] for p in m["pairs"]]
        return x[..., a, :] - x[..., b, :]
    sub = x[..., [IX[c] for c in m["channels"]], :]
    if m["ref"] == "subset_car":
        sub = sub - sub.mean(axis=-2, keepdims=True)
    return sub


def reorder_to_canon(x, channels):
    """Reindex a loaded recording from its own channel order into CANON."""
    missing = [c for c in CANON if c not in channels]
    if missing:
        raise ValueError(f"recording is missing canonical sites: {missing}")
    return x[[channels.index(c) for c in CANON], :]


# The budget ladder. Nested and bilaterally symmetric at every step, so it describes
# progressively cheaper versions of one device rather than five unrelated devices. It keeps
# whole-head coverage while it can, then concentrates on temporo-parietal sites -- where the
# Alzheimer's signature is reported -- rather than thinning uniformly.
LADDER = [
    _m("b19", CANON, note="full clinical 10-20 montage"),
    _m("b8", ["F3", "F4", "C3", "C4", "T5", "T6", "O1", "O2"],
       note="bilateral frontal, central, temporo-parietal, occipital"),
    _m("b4", ["T5", "T6", "O1", "O2"], note="temporo-parietal and occipital"),
    _m("b2", ["T5", "T6"], note="bilateral temporo-parietal only"),
    _m("b1", ["T5", "T6"], "bipolar", [("T5", "T6")],
       note="one derived signal from two electrodes, placed symmetrically"),
    # A symmetric pair is blind to bilaterally symmetric rhythms: re-referencing or subtracting
    # two mirror sites cancels exactly the posterior alpha the measurement is after. An
    # anterior-posterior derivation instead spans the gradient, so it is the fair one-signal
    # device and b1 must not be reported without it.
    _m("b1_ap", ["Fp1", "O1"], "bipolar", [("Fp1", "O1")],
       note="one derived signal spanning the anterior-posterior axis"),
]

REGIONAL = [_m(f"r_{k}", v) for k, v in REGION.items()]

# Commodity headsets mapped to the nearest standard site in the cache. These are APPROXIMATIONS:
# TP9/TP10 sit near T5/T6 and AF7/AF8 near Fp1/Fp2. A real headset also differs in electrode
# material, contact impedance and amplifier noise floor, none of which this captures.
COMMODITY = [
    _m("muse", ["T5", "Fp1", "Fp2", "T6"], family="commodity",
       note="Muse S (TP9/AF7/AF8/TP10) approximated by T5/Fp1/Fp2/T6"),
    _m("ganglion", ["Fp1", "Fp2", "T3", "T4"], family="commodity",
       note="OpenBCI Ganglion, 4 user-placed electrodes, frontotemporal layout"),
    _m("insight", ["Fp1", "Fp2", "T3", "T4", "Pz"], family="commodity",
       note="Emotiv Insight (AF3/AF4/T7/T8/Pz) approximated by Fp1/Fp2/T3/T4/Pz"),
    _m("frontal1", ["Fp1", "Fp2"], "bipolar", [("Fp1", "Fp2")], family="commodity",
       note="single-channel frontal wearable, bipolar Fp1-Fp2"),
]

# Optimistic controls: the same sites keeping the 19-channel reference the device cannot have.
# The gap against the matching ladder rung is the error introduced by naive channel subsetting.
CONTROLS = [
    _m("b4_inherit", ["T5", "T6", "O1", "O2"], "inherit", family="control",
       note="b4 sites retaining the 19-channel average reference; not physically realisable"),
    _m("b8_inherit", ["F3", "F4", "C3", "C4", "T5", "T6", "O1", "O2"], "inherit", family="control",
       note="b8 sites retaining the 19-channel average reference"),
]

ALL = {m["name"]: m for m in LADDER + REGIONAL + COMMODITY + CONTROLS}
