"""Constants the harmonisation loaders share with the analysis: the 10-20 order and the target rate.

Vendored from the authors' parent dementia-EEG project so this repository is self-contained. Only
the constants the loaders read are kept.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Root holding ``datasets/`` (the raw releases) and ``harmonised/`` (the cleaned cache).
DATA_ROOT = Path(os.environ.get("EEG_DATA_ROOT", REPO / "data")) / "datasets"
OUTPUT_ROOT = REPO / "outputs"

SAMPLING_RATE = 256  # Hz
N_CHANNELS = 19

#: The 19-channel 10-20 order every loader returns.
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
