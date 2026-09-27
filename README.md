# Low-channel EEG cognitive triage: placement, age referencing and output design

Code for *Decomposing a clinical EEG montage for low-channel cognitive triage: placement, age
referencing and output design* (Ghosh, Chaurasia, Sharma and Sharma; NeurIPS 2026 Workshop: Global
South AI).

A 19-channel clinical EEG montage is re-derived, in silico, as the low-channel devices a
low-resource clinic could afford (16 montages, 22 sampling-rate by bit-depth settings, 4
preprocessing arms, and injected field faults). Every condition is scored by one fixed classifier and
benchmarked against the patient's **age**, reporting the incremental margin
`AUC(EEG + age) − AUC(age)` rather than discrimination against chance.

## Repository layout

```
harmonisation/      one preprocessing pipeline for every cohort (notch, band-pass, CAR,
                    ICA + ICLabel, resampling) and the dataset loaders
eegbudget/          montages, digitisation budgets, the 17 qEEG features, the age-conditioned
                    (normative) reference, cross-validation and paired inference
experiments/        h00 harmonise -> e00 spine -> e01 features -> e02...e37 analyses and figures;
                    e07 verifies every number printed in the paper against outputs/results
tools/              Figure 1 patch (text, montage strip)
outputs/results/    the aggregate result files behind every reported number
outputs/figures/    the paper's figures (Figure 1 source: fig1_framework.drawio)
run_queue.py        runs the whole pipeline in order, resumably
```

## Data

The datasets are not redistributed. Obtain each from its maintainers under its own terms and place
it under `<EEG_DATA_ROOT>/datasets/` (default `./data/datasets/`):

| Dataset | Role | Source | Folder expected |
|---|---|---|---|
| CAUEEG (Kim et al., 2023) | development, 1,272 recordings | <https://github.com/ipis-mjkim/caueeg-dataset> (on request) | `caueeg-dataset/` with `annotation.json`, `signal/edf/`, the four split files |
| ds004504 (Miltiadous et al., 2023) | external test, 88 | <https://openneuro.org/datasets/ds004504> (raw, not derivatives) | `openneuro_ds004504_raw/` |
| BrainLat (Prado et al., 2023) | external test, 80 | <https://doi.org/10.7303/syn51549340> (registered access, Synapse) | `BrainLat Data/` (resting `.set`/`.fdt` and the `Demographics_*_EEG_data.csv` files) |
| P-ADIC (Shor et al., 2021) | external test, 89 | <https://doi.org/10.5061/dryad.8gtht76pw> | `P-ADIC/p-adic alz/alz_c1_new.mat`, `P-ADIC/p-adic ctrl/controls_c1_new.mat` |
| EasyCog (Hu et al., 2026) | wearable check, 69 | <https://github.com/EasyCog/EasyCog-Benchmark> | `EasyCog/` (the unzipped release; or set `EASYCOG_ROOT`) |

Per-recording features derived from the access-controlled releases are not included; the aggregate
result files are, so every number in the paper can be checked without the data (below).

## Setup

Python 3.12. Results were produced with numpy 2.4, scipy 1.17, scikit-learn 1.9, mne 1.12 and
mne-icalabel 0.9 (torch backend).

```bash
pip install -r requirements.txt
export EEG_DATA_ROOT=/path/to/data        # holds datasets/ ; harmonised/ is written beside it
```

## Reproduce

```bash
python run_queue.py                        # every stage in order; resumable; --status to inspect
```

or stage by stage:

```bash
python experiments/h00_harmonise.py        # ICA-cleaned 19-channel cache (hours; resumable)
python experiments/e09_extend_corpus.py --job vascular
python experiments/e09_extend_corpus.py --job noica
python experiments/e00_spine.py            # subject table with ages and labels
python experiments/e01_extract.py          # 42 acquisition conditions x 17 features (~45 min)
python experiments/e13_incremental.py      # the margins of Table 1
python experiments/e25_acquisition_design.py
python experiments/e31_external_extract.py && python experiments/e33_external_placement.py
python experiments/e32_stress_extract.py && python experiments/e34_acquisition_stress.py
python experiments/e35_external_transfer_all.py
python experiments/e37_external_within_increment.py
python experiments/e07_numbers.py          # verify every number the paper reports
```

`e11_deep_arm.py` (deep comparators) additionally needs `torch` and `braindecode` and a GPU is
advisable.

### Check the paper's numbers without the data

```bash
python experiments/e07_numbers.py
```

recomputes every quantity the paper states from `outputs/results/*.json` and compares it with the
value the paper prints, recorded in `outputs/results/paper_values.json` (956 quantities).

## Notes

- Figure 1 was drawn in diagrams.net; its source is `outputs/figures/fig1_framework.drawio`
  (open it in diagrams.net; the montage strip in it is drawn by `e29_montage_map.py`).
- Every other figure is drawn by a script as vector PDF with editable text: the results figure
  (`fig1_results.pdf`: placement, placement by disease, the increment across every check, and
  cohort size), the marker topography (`figA10_topography.pdf`) and net benefit
  (`fig2_utility.pdf`) by `e17_paper_figure.py`; the per-patient deviation (`fig3_interpretability.pdf`) by
  `e28_interpretability_figure.py`; the montage map (`figA7_montages.pdf`) by `e29_montage_map.py`;
  placement by disease on the external cohorts (`figA8_external_placement.pdf`) and the acquisition
  stress test (`figA9_stress.pdf`) by `e36_revision_figures.py`.
- The serial of the single CAUEEG patient shown in Figure 3 is withheld from
  `outputs/results/e28_interpretability.json`.
- Low-channel devices are simulated by subsetting and re-referencing clinical recordings, and field
  faults are injected, not measured on hardware; see the paper's limitations.

## Citation

```bibtex
@inproceedings{ghosh2026lowchannel,
  title     = {Decomposing a clinical {EEG} montage for low-channel cognitive triage:
               placement, age referencing and output design},
  author    = {Ghosh, Soinik and Chaurasia, Rameshwar Nath and Sharma, Shiru and Sharma, Neeraj},
  booktitle = {NeurIPS 2026 Workshop on Global South AI},
  year      = {2026}
}
```

Please also cite the datasets you use (see the table above and `paper/references.bib`).

## License

Code: MIT (see `LICENSE`). The datasets remain under their own licences and access terms.
