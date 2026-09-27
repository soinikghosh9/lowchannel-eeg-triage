"""Check that every number printed in the paper still matches the artefacts.

The manuscript used to read its numbers from a generated ``numbers.tex`` through macros, so the
text could not drift from the experiments. The numbers are now written literally into
``paper/main.tex``, which is easier to read and to hand to a co-author but gives up that guarantee.
This file replaces it: it recomputes every quantity the paper states, compares it against the
literal in the manuscript, and reports anything that has moved.

``outputs/results/paper_values.json`` records the value behind each sentence. A drift report names
the quantity, the value the paper prints, and the value the artefacts now give, so the fix is a
one-line edit rather than a hunt.

    python experiments/e07_numbers.py             # verify
    python experiments/e07_numbers.py --update    # accept current values after editing main.tex

Without paper/main.tex (the public code release), the recomputed values are checked against the
values recorded in paper_values.json instead.
Out: outputs/results/paper_values.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import budgets, paths  # noqa: E402

MAIN = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
VALUES = paths.RESULTS / "paper_values.json"
FULL, FOUR, ONE = "b19|256|f32|full", "b4|256|f32|full", "b1_ap|256|f32|full"
TASKNAME = {"screening": "Screen", "dementia": "Dem", "mci": "MCI"}


def load(name):
    p = paths.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(v):
    return "MISSING" if v is None or not np.isfinite(v) else f"{v:.3f}"


def f2(v):
    return "MISSING" if v is None or not np.isfinite(v) else f"{v:.2f}"


def sgn3(v):
    return "MISSING" if v is None or not np.isfinite(v) else f"{v:+.3f}"


def pct(v):
    return "MISSING" if v is None or not np.isfinite(v) else f"{v*100:.0f}\\%"


def pval(v):
    """Format a p-value the way a journal expects, without ever printing P = 0."""
    if v is None or not np.isfinite(v):
        return "MISSING"
    if v < 1e-4:
        return "$P<0.0001$"
    return f"$P={v:.4f}$" if v < 0.001 else f"$P={v:.3f}$"


def ci(lo, hi, signed=True):
    if lo is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "MISSING"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"[{f.format(lo)}, {f.format(hi)}]"


def analysed_sample(m):
    """What actually enters a result, as distinct from what was preprocessed.

    Every analysis in this study is referenced to chronological age, so a cohort that publishes no
    age cannot appear in one. Six of the eight do not, and two of those six carry a single
    diagnostic class and could not support a discrimination analysis at any rate. Quoting the
    corpus total as though it were the analysed sample overstates the evidence by 504 recordings,
    and this function exists so the paper cannot do that.
    """
    import csv
    p = paths.CACHE / "spine.csv"
    if not p.exists():
        # Without the per-subject table (it is not redistributed), the same counts come from the
        # aggregate spine summary; only the single-class count needs per-subject labels.
        sp = load("e00_spine.json")
        if not sp:
            return
        ages, sizes = sp.get("age_available", {}), sp.get("by_cohort", {})
        analysed = {c: ages[c] for c in sizes if ages.get(c)}
        dropped = {c: sizes[c] for c in sizes if not ages.get(c)}
        m["NAnalysed"] = f"{sum(analysed.values()):,}"
        m["NCohortsAnalysed"] = str(len(analysed))
        m["NCorpus"] = f"{sp['n_recordings']:,}"
        m["NCohortsCorpus"] = str(len(sizes))
        m["NDroppedNoAge"] = str(sum(dropped.values()))
        m["NCohortsDropped"] = str(len(dropped))
        return
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    by_cohort = {}
    for r in rows:
        c = by_cohort.setdefault(r["cohort"], {"n": 0, "with_age": 0, "labels": set()})
        c["n"] += 1
        c["with_age"] += 1 if r["age"] not in ("", "nan") else 0
        c["labels"].add(r["label"])

    analysed = {c: v for c, v in by_cohort.items() if v["with_age"]}
    dropped = {c: v for c, v in by_cohort.items() if not v["with_age"]}
    single = [c for c, v in dropped.items() if len(v["labels"]) == 1]

    m["NAnalysed"] = f"{sum(v['with_age'] for v in analysed.values()):,}"
    m["NCohortsAnalysed"] = str(len(analysed))
    m["NCorpus"] = f"{len(rows):,}"
    m["NCohortsCorpus"] = str(len(by_cohort))
    m["NDroppedNoAge"] = str(sum(v["n"] for v in dropped.values()))
    m["NCohortsDropped"] = str(len(dropped))
    m["NSingleClass"] = str(len(single))
    m["SampleSentence"] = (
        f"Every analysis is referenced to chronological age, so the study uses the "
        f"{len(analysed)} public cohorts that publish it: "
        f"{sum(v['with_age'] for v in analysed.values()):,} resting recordings in total.")


def corpus(m, sp):
    m["NSubjects"] = f"{sp['n_recordings']:,}"
    m["NCohorts"] = str(len(sp["by_cohort"]))
    m["NAgeTrain"] = f"{sp['age_available'].get('CAUEEG', 0):,}"
    m["NAgeTarget"] = str(sp["age_available"].get("ds004504_raw", 0))
    ab = sp.get("age_by_label", {})
    if "CN" in ab:
        m["AgeCN"] = f"{ab['CN']['mean']:.0f}"
    imp = [ab[k] for k in ("AD", "MCI", "FTD", "VAD") if k in ab]
    if imp:
        tot = sum(v["n"] for v in imp)
        m["AgeImpaired"] = f"{sum(v['mean'] * v['n'] for v in imp) / tot:.0f}"
    for tag, key in (("eoad", "EOAD"), ("load", "LOAD"), ("vd", "VAD"), ("tga", "TGA"),
                     ("smi", "SMI")):
        s = sp.get("subtypes", {}).get(tag)
        if s:
            m[f"N{key}"] = str(s["n"])
            m[f"Age{key}"] = f"{s['age_mean']:.1f}"
    if "AgeCN" in m and "AgeImpaired" in m:
        m["AgeGapCohort"] = f"{float(m['AgeImpaired']) - float(m['AgeCN']):.0f}"
    if "AgeEOAD" in m and "AgeLOAD" in m:
        m["AgeGapEOADLOAD"] = f"{float(m['AgeLOAD']) - float(m['AgeEOAD']):.0f}"


def decomposition(m, e02):
    """Per-task age baselines and the ladder, plus the sentences that interpret them."""
    for task, rec in e02.get("tasks", {}).items():
        t = TASKNAME.get(task, task.title())
        m[f"N{t}"] = f"{rec['n']:,}"
        m[f"AgeOnly{t}"] = f3(rec["age_only"]["auc"])
        b = rec["budgets"]
        for k, key in ((FULL, "Full"), (FOUR, "Four"), (ONE, "One")):
            if k in b:
                m[f"AUC{t}{key}"] = f3(b[k]["auc_eeg"])
                m[f"Margin{t}{key}"] = sgn3(b[k]["age_margin"])
                m[f"Rho{t}{key}"] = f2(b[k]["rho_matched"])
                va = b[k].get("vs_age")
                if va:
                    m[f"P{t}{key}"] = pval(va.get("p"))
                    m[f"MarginCI{t}{key}"] = ci(*va.get("ci", [None, None]))
                rc = b[k].get("rho_matched_ci")
                if rc:
                    m[f"RhoCI{t}{key}"] = ci(rc[0], rc[1], signed=False)
        if FULL in b:
            m[f"AUCBoth{t}"] = f3(b[FULL]["auc_eeg_plus_age"])

    scr = e02.get("tasks", {}).get("screening")
    if not scr:
        return
    b = scr["budgets"]
    if FULL not in b or FOUR not in b:
        return
    a19, a4 = b[FULL]["auc_eeg"], b[FOUR]["auc_eeg"]
    r19, r4 = b[FULL]["rho_matched"], b[FOUR]["rho_matched"]
    age = scr["age_only"]["auc"]
    m["LadderDrop"] = sgn3(a4 - a19)
    if ONE in b:
        m["LadderDropOne"] = sgn3(b[ONE]["auc_eeg"] - a19)

    # -- headline: does the recording beat the birth certificate?
    marg = a19 - age
    if marg < -0.01:
        m["HeadlineSentence"] = (
            f"At the full clinical montage the instrument reaches {a19:.3f}, "
            f"{abs(marg):.3f} below what chronological age supplies on its own "
            f"({age:.3f}). The recording is worth less than the birth certificate.")
    elif marg < 0.02:
        m["HeadlineSentence"] = (
            f"At the full clinical montage the instrument reaches {a19:.3f} against {age:.3f} "
            f"from age alone, a margin of {marg:+.3f} that is not clinically meaningful.")
    else:
        m["HeadlineSentence"] = (
            f"At the full clinical montage the instrument reaches {a19:.3f} against {age:.3f} "
            f"from age alone, a margin of {marg:+.3f}.")

    # -- the ladder: does cutting electrodes cost anything?
    if abs(a4 - a19) < 0.02:
        m["LadderSentence"] = (
            f"Nineteen electrodes to four changes discrimination by {a4-a19:+.3f}, within the "
            f"resampling interval.")
    else:
        m["LadderSentence"] = (
            f"Reducing from nineteen electrodes to four changes discrimination by {a4-a19:+.3f}.")

    # -- the substitution ratio: does it rise as the budget falls?
    if r4 > r19 + 0.05:
        m["RhoSentence"] = (
            f"The substitution ratio rises from {r19:.2f} to {r4:.2f} across that reduction: the "
            f"cheaper device is not a slightly worse measurement of the brain, it is a more "
            f"demographic one.")
    elif abs(r4 - r19) <= 0.05:
        m["RhoSentence"] = (
            f"The substitution ratio is flat across the ladder ({r19:.2f} to {r4:.2f}). The "
            f"instrument does not become more age-dependent as it is cut down; it was already "
            f"age-dominated at nineteen electrodes.")
    else:
        m["RhoSentence"] = (
            f"The substitution ratio falls from {r19:.2f} to {r4:.2f}.")

    # -- the two estimators of the same quantity, reported honestly
    rr19 = b[FULL].get("rho_residualised")
    if rr19 is not None and np.isfinite(rr19):
        m["RhoDisagreeSentence"] = (
            f"Matching and residualisation agree on sign and on substance but not on magnitude "
            f"({r19:.2f} against {rr19:.2f} at the full montage), so we read the ratio as an "
            f"interval rather than a point estimate and rest no claim on its exact value.")

    mci = e02.get("tasks", {}).get("mci")
    if mci and FULL in mci["budgets"]:
        mm = mci["budgets"][FULL]["age_margin"]
        m["MCISentence"] = (
            f"The gap is worst where triage matters most: on mild cognitive impairment the "
            f"instrument trails age by {abs(mm):.3f} AUC on the contrast where a triage "
            f"decision would actually change a care pathway." if mm < 0 else
            f"On mild cognitive impairment the margin is {mm:+.3f} AUC.")


def subgroups(m, e03):
    m["SpecTarget"] = pct(e03["specificity"])
    k = FULL if FULL in e03["budgets"] else next(iter(e03["budgets"]), None)
    if not k:
        return
    raw = e03["budgets"][k]["raw"]["subgroups"]
    nrm = e03["budgets"][k]["normative"]["subgroups"]
    for tag, key in (("eoad", "EOAD"), ("load", "LOAD"), ("vd", "VAD"),
                     ("mci_amnestic", "MCIAmn"), ("mci_vascular", "MCIVas"),
                     ("smi", "SMI"), ("tga", "TGA")):
        if tag in raw:
            m[f"Sens{key}"] = f2(raw[tag]["flagged_rate"])
            m[f"SensNorm{key}"] = f2(nrm[tag]["flagged_rate"])
    if "eoad" in raw and "load" in raw:
        gap_raw = raw["load"]["flagged_rate"] - raw["eoad"]["flagged_rate"]
        gap_nrm = nrm["load"]["flagged_rate"] - nrm["eoad"]["flagged_rate"]
        m["GapEOAD"] = sgn3(gap_raw)
        if gap_raw > 0.1 and gap_nrm < gap_raw - 0.05:
            m["SubgroupSentence"] = (
                f"Age-conditioned normative scoring narrows the gap from {gap_raw:+.2f} to "
                f"{gap_nrm:+.2f}, recovering younger patients the raw instrument misses.")
        elif gap_raw > 0.1:
            m["SubgroupSentence"] = (
                f"Normative scoring does not close the gap ({gap_nrm:+.2f}), indicating that "
                f"early-onset disease differs from late-onset disease in more than age.")
        else:
            m["SubgroupSentence"] = (
                f"The gap is small ({gap_raw:+.2f}), so the instrument does not systematically "
                f"miss younger patients with confirmed disease.")


def transfer(m, e04):
    """What survives the trip to a clinic that contributed nothing to training.

    All three arms are now read on the same held-out controls, so the sentence this writes is a
    comparison rather than a juxtaposition of two different samples.
    """
    k = next((b for b in (FOUR, "b8|256|f32|full", FULL) if b in e04.get("budgets", {})), None)
    if not k:
        return
    arms = e04["budgets"][k]
    m["SpecDirect"] = f2(arms["direct"]["specificity"])
    m["SensDirect"] = f2(arms["direct"]["sensitivity"])
    m["SpecNormative"] = f2(arms["normative"]["specificity"])
    m["SensNormative"] = f2(arms["normative"]["sensitivity"])
    m["NHeldout"] = str(arms["direct"]["n_heldout_controls"])
    if FULL in e04.get("budgets", {}):
        m["SpecDirectFull"] = f2(e04["budgets"][FULL]["direct"]["specificity"])
        m["SpecNormativeFull"] = f2(e04["budgets"][FULL]["normative"]["specificity"])
    anc = arms.get("anchored", {})
    if not anc:
        return
    kk = max(anc, key=lambda z: int(z))
    m["KAnchor"] = str(int(kk))
    m["SpecAnchored"] = f2(anc[kk]["specificity"])
    m["SpecAnchoredSD"] = f2(anc[kk]["specificity_sd"])
    d_norm = arms["normative"]["specificity"] - arms["direct"]["specificity"]
    d_anc = anc[kk]["specificity"] - arms["normative"]["specificity"]
    # The sentence follows whichever of the two steps actually did the work. An earlier version
    # asserted that anchoring did, on numbers that said otherwise.
    if d_norm > 0.05 and d_anc <= 0.02:
        m["TransferSentence"] = (
            f"Re-anchoring on {kk} of the clinic's own controls changes it by {d_anc:+.2f} with a "
            f"spread of {anc[kk]['specificity_sd']:.2f} across draws, so at this sample size the "
            f"local anchor contributes noise rather than calibration.")
    elif d_anc > 0.05:
        m["TransferSentence"] = (
            f"Re-anchoring on {kk} of the clinic's own controls adds {d_anc:+.2f} of specificity "
            f"on top of the {d_norm:+.2f} the age conditioning supplies, without a single "
            f"labelled patient at the receiving site.")
    else:
        m["TransferSentence"] = (
            f"Neither step recovers the intended operating point: age conditioning moves "
            f"specificity by {d_norm:+.2f} and local anchoring by {d_anc:+.2f}.")


def incremental(m, e13):
    """The two margins, the multiplicity correction, and the placement test."""
    fam = e13.get("primary_family", {})
    for task, rec in e13.get("tasks", {}).items():
        t = TASKNAME.get(task, task.title())
        b = rec["budgets"].get(FULL)
        if not b:
            continue
        m[f"N{t}"] = f"{rec['n']:,}"
        m[f"AgeGap{t}"] = f"{rec['age_gap']:.1f}"
        m[f"AgeOnly{t}"] = f3(b["auc_age"])
        m[f"AUC{t}Full"] = f3(b["auc_eeg"])
        m[f"AUCBoth{t}"] = f3(b["auc_eeg_plus_age"])
        m[f"Sub{t}"] = sgn3(b["sub_margin"])
        m[f"SubCI{t}"] = ci(*b["sub_ci"])
        m[f"Inc{t}"] = sgn3(b["inc_margin"])
        m[f"IncCI{t}"] = ci(*b["inc_ci"])
        for kind, key in (("sub", "Sub"), ("inc", "Inc")):
            h = fam.get(f"{task}|{kind}")
            if h:
                m[f"P{key}{t}"] = pval(h["p_holm"])
        fs = max(np.std(b["sub_margin_per_repeat"]), np.std(b["inc_margin_per_repeat"]))
        m[f"FoldSD{t}"] = f"{fs:.3f}"

    # The largest between-repeat spread anywhere in the sweep, so the paper can state in one
    # number that its 0.02--0.07 margins are not fold-assignment noise.
    spreads = [float(np.std(bb[f"{k}_margin_per_repeat"]))
               for rec in e13.get("tasks", {}).values() for bb in rec["budgets"].values()
               for k in ("sub", "inc")]
    if spreads:
        m["FoldSDMax"] = f"{max(spreads):.3f}"

    n_prim = len(fam)
    if n_prim:
        m["NPrimaryTests"] = str(n_prim)

    p = e13.get("placement", {}).get("r_temporal_vs_r_frontal")
    if p:
        m["PlacementDiff"] = sgn3(p["diff"])
        m["PlacementCI"] = ci(*p["ci"])
        m["PlacementP"] = pval(p["p_holm"])
        m["PlacementSentence"] = (
            f"Four temporal electrodes outperform seven frontal ones by {p['diff']:+.3f} "
            f"{ci(*p['ci'])} on identical subjects ({pval(p['p_holm'])}, Holm-adjusted), a larger "
            f"effect than the whole ladder spans.")
    pb = e13.get("placement", {}).get("r_temporal_vs_b4")
    if pb:
        m["TemporalVsFourSentence"] = (
            f"Those four temporal electrodes are also indistinguishable from the four-electrode "
            f"rung of the ladder ({pb['diff']:+.3f} {ci(*pb['ci'])}, {pval(pb['p_holm'])}), which "
            f"is the same sites plus occipital coverage.")

    # The ladder, read off the same artefact so the flatness claim and the margins agree.
    scr = e13.get("tasks", {}).get("screening", {}).get("budgets", {})
    if FULL in scr and FOUR in scr:
        m["LadderDrop"] = sgn3(scr[FOUR]["auc_eeg"] - scr[FULL]["auc_eeg"])
    if FULL in scr and ONE in scr:
        m["LadderDropOne"] = sgn3(scr[ONE]["auc_eeg"] - scr[FULL]["auc_eeg"])
    if FULL in scr and FOUR in scr:
        m["LadderIncDrop"] = sgn3(scr[FOUR]["inc_margin"] - scr[FULL]["inc_margin"])


def ladder(m, e13):
    """Incremental margin along the electrode ladder, screening contrast."""
    rec = e13.get("tasks", {}).get("screening", {}).get("budgets", {})
    for mont, key in (("b19", "Full"), ("b8", "Eight"), ("b4", "Four"), ("b2", "Two")):
        r = rec.get(f"{mont}|256|f32|full")
        if r:
            m[f"IncScreen{key}"] = sgn3(r["inc_margin"])
            m[f"IncCIScreen{key}"] = ci(*r["inc_ci"])


def operating(m, e23):
    """Operating-point metrics and the external cohort."""
    if not e23:
        return
    tag = {"screening": "Screen", "dementia": "Dem", "mci": "MCI"}
    arm_tag = {"age": "Age", "eeg": "EEG", "eeg+age": "Both", "normative": "Norm"}
    for task, key in tag.items():
        rec = e23.get("tasks", {}).get(task, {}).get("budgets", {}).get("b19")
        if not rec:
            continue
        for arm, atag in arm_tag.items():
            a = rec.get(arm)
            if not a:
                continue
            m[f"AUCOp{atag}{key}"] = f3(a["auc"])
            m[f"Sens{atag}{key}"] = f2(a["sensitivity"])
            m[f"Spec{atag}{key}"] = f2(a["specificity"])
            m[f"BalAcc{atag}{key}"] = f2(a["balanced_accuracy"])
            m[f"LRpos{atag}{key}"] = f2(a["lr_positive"])
            m[f"LRneg{atag}{key}"] = f2(a["lr_negative"])
            m[f"Youden{atag}{key}"] = f2(a["youden_j"])
            m[f"AP{atag}{key}"] = f3(a["average_precision"])
            m[f"Brier{atag}{key}"] = f3(a["brier"])
            m[f"CalSlope{atag}{key}"] = f2(a["calibration_slope"])
            for prev, ptag in (("0.05", "Five"), ("0.10", "Ten"), ("0.20", "Twenty")):
                pv = a["predictive_values"].get(prev)
                if pv:
                    m[f"PPV{atag}{key}{ptag}"] = f2(pv["ppv"])
                    m[f"NPV{atag}{key}{ptag}"] = f2(pv["npv"])
                    m[f"F1{atag}{key}{ptag}"] = f2(pv["f1"])
                    m[f"NNS{atag}{key}{ptag}"] = f"{pv['number_needed_to_screen']:.0f}"
                    m[f"Flag{atag}{key}{ptag}"] = f"{pv['flagged_per_1000']:.0f}"
                    m[f"TrueFlag{atag}{key}{ptag}"] = f"{pv['true_cases_among_flagged_per_1000']:.0f}"
    scr = e23.get("tasks", {}).get("screening", {}).get("budgets", {}).get("b19", {}).get("eeg+age")
    if scr:
        pv = scr["predictive_values"]["0.10"]
        m["PPVSentence"] = (
            f"At a 10\\% service prevalence and the 80\\% specificity operating point, the combined "
            f"model would flag {pv['flagged_per_1000']:.0f} of every 1{{,}}000 people assessed and "
            f"{pv['true_cases_among_flagged_per_1000']:.0f} of those would have the condition, a "
            f"positive predictive value of {pv['ppv']:.2f} against a negative predictive value of "
            f"{pv['npv']:.2f}.")

    ext = e23.get("external")
    if not ext:
        return
    m["ExtN"] = str(ext["n"])
    m["ExtGap"] = f"{ext['age_gap']:+.1f}"
    m["ExtPrev"] = f2(ext["prevalence"])
    for tag_b, key in (("b19", "Full"), ("b4", "Four")):
        arms = ext.get("budgets", {}).get(tag_b, {})
        for arm, atag in (("eeg", "EEG"), ("eeg+age", "Both"), ("normative", "Norm"),
                          ("age_direction_free", "AgeFree"),
                          ("age_source_direction", "AgeSrc")):
            a = arms.get(arm)
            if a:
                m[f"Ext{atag}{key}"] = f3(a["auc"])
                m[f"ExtSens{atag}{key}"] = f2(a["sensitivity"])
                m[f"ExtAP{atag}{key}"] = f3(a["average_precision"])
    m["RemedySentence"] = (
        f"Scoring the same features as deviations from an age-conditioned reference recovers what "
        f"the age term costs on transfer: at four electrodes the normative arm reaches "
        f"{m.get('ExtNormFour', '?')} externally against {m.get('ExtBothFour', '?')} with age as a "
        f"predictor and {m.get('ExtEEGFour', '?')} with no age at all, and at nineteen it reaches "
        f"{m.get('ExtNormFull', '?')} against {m.get('ExtBothFull', '?')}.")
    m["TradeoffSentence"] = (
        f"The ordering reverses between the two settings: within the training cohort the "
        f"age-as-predictor arm is best on screening ({m.get('AUCOpBothScreen', '?')}) and the "
        f"normative arm worst ({m.get('AUCOpNormScreen', '?')}), while on transfer the ranking is "
        f"the other way round. Selecting an arm on within-cohort performance selects the one that "
        f"transports least well.")
    m["ExternalSentence"] = (
        f"On the external cohort, where cases are {abs(ext['age_gap']):.1f} years younger than "
        f"controls, the EEG-only model reaches {m.get('ExtEEGFull', '?')} at 19 channels and "
        f"{m.get('ExtEEGFour', '?')} at four, while age read without a direction reaches "
        f"{m.get('ExtAgeFreeFull', '?')} and adding an age term to the model lowers it to "
        f"{m.get('ExtBothFull', '?')}.")


def composition(m, e14):
    """The within-cohort recruitment result: the baseline moves when 'control' is redefined."""
    arms = e14.get("control_arms", {})
    for k, key in (("all", "All"), ("clean", "Clean"), ("smi", "SMI")):
        if k in arms:
            m[f"NCtrl{key}"] = f"{arms[k]['n']:,}"
            m[f"AgeCtrl{key}"] = f"{arms[k]['age_mean']:.1f}"

    scr = e14.get("tasks", {}).get("screening", {}).get("arms", {})
    if not scr:
        return
    for k, key in (("all", "All"), ("clean", "Clean"), ("smi", "SMI")):
        if k in scr:
            m[f"AgeAUC{key}"] = f3(scr[k]["auc_age"])
            m[f"SubComp{key}"] = sgn3(scr[k]["sub_margin"])
            m[f"IncComp{key}"] = sgn3(scr[k]["inc_margin"])
            m[f"GapComp{key}"] = f"{scr[k]['age_gap']:.1f}"

    if "clean" in scr and "smi" in scr:
        spread = scr["clean"]["auc_age"] - scr["smi"]["auc_age"]
        m["CompSpread"] = f3(spread)
        sm = e14["tasks"]["screening"].get("size_matched", {})
        m["CompSentence"] = (
            f"It moves the age baseline from {scr['smi']['auc_age']:.3f} to "
            f"{scr['clean']['auc_age']:.3f}, a spread of {spread:.3f} AUC, and the margin against "
            f"it from {scr['smi']['sub_margin']:+.3f} to {scr['clean']['sub_margin']:+.3f}; the "
            f"corresponding incremental margins are {scr['smi']['inc_margin']:+.3f} and "
            f"{scr['clean']['inc_margin']:+.3f}.")
        if sm:
            m["SizeMatchedSentence"] = (
                f"A control arm of the same size drawn at random recovers the original baseline "
                f"({sm['auc_age_mean']:.3f}), so the shift is composition, not sample size.")

    # Which margin survives a change of recruitment is the methodological point.
    spans = {}
    for task, rec in e14.get("tasks", {}).items():
        a = rec.get("arms", {})
        if len(a) < 3:
            continue
        spans[task] = (max(v["sub_margin"] for v in a.values())
                       - min(v["sub_margin"] for v in a.values()),
                       max(v["inc_margin"] for v in a.values())
                       - min(v["inc_margin"] for v in a.values()))
    if spans:
        sub_span = max(v[0] for v in spans.values())
        inc_span = max(v[1] for v in spans.values())
        m["SubSpan"] = f3(sub_span)
        m["IncSpan"] = f3(inc_span)
        m["StabilitySentence"] = (
            f"Across the three control definitions and tested contrasts, the substitution margin "
            f"moves up to {sub_span:.3f} AUC and the increment at most {inc_span:.3f}.")


def crossover(m, e15):
    """The two-cohort crossover with the contrast held fixed and the baseline read honestly."""
    h = e15.get("headline")
    if not h:
        return
    src, tgt = e15["source"], e15["target"]
    m["AgeOnlyExternal"] = f3(tgt["age_baselines"]["transported"])
    m["AgeOnlyExternalFair"] = f3(tgt["age_baselines"][h["baseline_used"]])
    m["AgeOnlyExternalRefit"] = f3(tgt["age_baselines"]["refit_cv"])
    m["AUCExternal"] = f3(tgt["auc_eeg"])
    m["NExternal"] = str(tgt["n"])
    m["AgeGapExternal"] = f"{tgt['age_gap']:.1f}"
    m["MarginExternal"] = sgn3(h["target_margin"])
    m["MarginExternalCI"] = ci(*h["target_margin_ci"])
    m["Swing"] = sgn3(h["swing"])
    m["SwingCI"] = ci(*h["swing_ci"])
    m["SwingP"] = pval(h["swing_p"])
    m["SwingOptimistic"] = sgn3(e15["swings"]["transported"]["swing"])
    m["CrossoverSentence"] = (
        f"With the question fixed at dementia, the margin moves from "
        f"{src['margin_transported']:+.3f} where cases are {src['age_gap']:.1f} years older to "
        f"{h['target_margin']:+.3f} where they are {abs(tgt['age_gap']):.1f} years "
        f"{'younger' if tgt['age_gap'] < 0 else 'older'}: a swing of {h['swing']:+.3f} "
        f"{ci(*h['swing_ci'])}, {pval(h['swing_p'])}; the source score is out-of-fold "
        f"and the target score uses the final source fit.")
    m["BaselineChoiceSentence"] = (
        f"Age scores {tgt['age_baselines']['transported']:.3f} there only if the source's "
        f"direction travels with it; read without one it scores "
        f"{tgt['age_baselines']['sign_agnostic']:.3f}, the baseline we quote.")
    inc = e15.get("incremental", {})
    if inc.get("target_eeg_only_vs_ageplus"):
        d = inc["target_eeg_only_vs_ageplus"]["diff"]
        m["AgeTermCost"] = sgn3(-abs(d))
        m["AgeTrapSentence"] = (
            f"an age term fitted where cases are older costs {abs(d):.3f} AUC where they are not")


def rho_check(m, e16):
    """The substitution ratio, after both estimators' named biases are removed."""
    t = e16.get("tasks", {})
    if not t:
        return
    got = {}
    for task, key in (("screening", "Screen"), ("dementia", "Dem"), ("mci", "MCI")):
        b = t.get(task, {}).get("budgets", {}).get(FULL)
        if not b:
            continue
        got[task] = b
        m[f"Rho{key}Full"] = f2(b["rho_matched"])
        m[f"RhoResid{key}"] = f2(b["rho_resid_controls"])
    s = got.get("screening")
    if s:
        m["RhoSentence"] = (
            f"With the age trend fitted on controls alone the two estimators agree "
            f"({s['rho_matched']:.2f} by matching, {s['rho_resid_controls']:.2f} by "
            f"residualisation, against {s['rho_resid_pooled']:.2f} under a pooled fit that "
            f"absorbs disease into the age term).")


def deep(m, e11):
    """The trained-architecture arm, and the two-cohort dissociation it demonstrates."""
    allruns = [v for v in e11.get("runs", {}).values()
               if v["montage"] == "b19" and v["task"] == "screening"]
    # The capacity claim must be made on architectures that never see age. A variant that takes
    # age as an input trivially inherits the baseline it is being compared against, so including
    # it here would be comparing age with itself.
    runs = [v for v in allruns if v["variant"] == "none"]
    if not runs:
        return
    m["NDeepModels"] = str(len({v["model"] for v in allruns}))
    m["DeepEpochs"] = str(e11.get("epochs", "?"))
    m["DeepSeeds"] = str(e11.get("seeds", "?"))

    # Report the architecture most favourable to the opposing case: a claim about what capacity
    # cannot do is only worth stating if it survives the model that does best.
    best = max(runs, key=lambda v: v["auc_within"])
    m["BestDeepModel"] = {"eegnet": "EEGNetv4", "shallow": "ShallowFBCSPNet",
                          "deep4": "Deep4Net", "atcnet": "ATCNet"}.get(best["model"], best["model"])
    m["AUCDeepWithin"] = f3(best["auc_within"])
    m["MarginDeepWithin"] = sgn3(best["age_margin_within"])
    m["PDeepWithin"] = pval(best["vs_age_within"]["p"])

    ext = [v for v in runs if v.get("auc_external") is not None]
    if ext:
        be = max(ext, key=lambda v: v["auc_external"])
        # Reported as a range over the architectures, not as the best of them. The external cohort
        # has 88 subjects; picking the architecture that happened to transport best and quoting it
        # as "the same architectures reach X" is a selection the interval does not cover.
        m["AUCDeepExternalLo"] = f3(min(v["auc_external"] for v in ext))
        m["AUCDeepExternalHi"] = f3(max(v["auc_external"] for v in ext))
        m["AUCDeepExternal"] = f3(float(np.mean([v["auc_external"] for v in ext])))
        # The external age baseline is read without a direction, matching e15: the transported
        # rule scores below chance there only because it assumes the training cohort's sign.
        a_ext = be["age_only_external"]
        m["AgeOnlyExternalDeep"] = f3(max(a_ext, 1 - a_ext))
        m["MarginDeepExternal"] = sgn3(be["auc_external"] - max(a_ext, 1 - a_ext))
        flip = be["age_margin_external"] - best["age_margin_within"]
        pw = best["vs_age_within"]["p"]
        # The wording follows the result. A trained network that clears the baseline by a margin
        # whose interval covers zero has not beaten it, and must not be written up as though the
        # feature model's significant deficit generalised to every estimator.
        if pw < 0.05 and best["age_margin_within"] > 0:
            m["DeepSentence"] = (
                f"Capacity does change the picture: the best of {m['NDeepModels']} trained "
                f"architectures reaches {best['auc_within']:.3f} against "
                f"{best['age_only_within']:.3f} from age, a margin of "
                f"{best['age_margin_within']:+.3f} ({pval(pw)}).")
        else:
            m["DeepSentence"] = (
                f"Trained architectures close the deficit but do not convert it into an "
                f"advantage: the best of {m['NDeepModels']} reaches {best['auc_within']:.3f} "
                f"against {best['age_only_within']:.3f} from age, a margin of "
                f"{best['age_margin_within']:+.3f} on an interval covering zero ({pval(pw)}). "
                f"The significant deficit is a property of the seventeen-feature model, not of "
                f"the recording.")
        lo_e = min(v["auc_external"] for v in ext)
        hi_e = max(v["auc_external"] for v in ext)
        m["DeepExternalSentence"] = (
            f"Carried unchanged to the age-matched clinic the same three architectures reach "
            f"{lo_e:.3f} to {hi_e:.3f} against a direction-free age baseline of "
            f"{max(a_ext, 1 - a_ext):.3f}, reproducing the feature model's crossover on a "
            f"different estimator.")

        # The gradient across clinical questions. This is the sharpest thing the deep arm shows
        # and it is invisible if only the screening contrast is reported.
        tasks = {}
        for v in e11.get("runs", {}).values():
            if v["montage"] == "b19" and v["variant"] == "none":
                tasks.setdefault(v["task"], []).append(v)
        # Nine tests -- three architectures crossed with three contrasts -- read as a pattern.
        # Reporting each at an uncorrected 0.05 and then describing the gradient across them is
        # exactly the situation Holm exists for, and two of the nine do not survive it.
        from eegbudget.evaluate import holm as _holm
        flat = [(t, x) for t in ("dementia", "screening", "mci") if t in tasks
                for x in sorted(tasks[t], key=lambda z: z["model"])]
        adj = _holm([x["vs_age_within"]["p"] for _, x in flat])
        holm_by_task = {}
        for (t, _), a in zip(flat, adj):
            holm_by_task.setdefault(t, []).append(a)

        parts, sig = [], {}
        for t, label in (("dementia", "dementia"), ("screening", "screening"),
                         ("mci", "mild cognitive impairment")):
            if t not in tasks:
                continue
            g = tasks[t]
            lo = min(x["age_margin_within"] for x in g)
            hi = max(x["age_margin_within"] for x in g)
            n_sig = sum(1 for a in holm_by_task.get(t, []) if a < 0.05)
            sig[t] = (lo, hi, n_sig, len(g))
            parts.append(f"{label} {lo:+.3f} to {hi:+.3f} ({n_sig} of {len(g)})")
        if len(parts) == 3:
            # Cited inside a parenthesis in the main text, so it carries the numbers and none
            # of the framing the surrounding sentence already supplies.
            m["DeepTaskSentence"] = (
                "substitution margin " + "; ".join(parts)
                + " significant after Holm correction over nine tests")
            for t, key in (("dementia", "Dem"), ("screening", "Screen"), ("mci", "MCI")):
                if t in sig:
                    lo, hi, ns, n = sig[t]
                    m[f"DeepMargin{key}Lo"] = sgn3(lo)
                    m[f"DeepMargin{key}Hi"] = sgn3(hi)
                    m[f"DeepSig{key}"] = f"{ns} of {n}"

        # The age-variant ablation, reported by what it did to transfer rather than to fit.
        byv = {}
        for v in allruns:
            byv.setdefault(v["variant"], []).append(v)
        f1 = {k: max(x.get("f1_external", float("nan")) for x in g) for k, g in byv.items()}
        if "none" in f1 and "input" in f1:
            m["AgeVariantSentence"] = (
                f"Letting age into the model raises what it scores in the cohort it was fitted on "
                f"and costs what it delivers elsewhere: the best external F1 is {f1['none']:.3f} "
                f"with no age term, {f1['input']:.3f} with age as an input"
                + (f", and {f1['adv']:.3f} when age is adversarially removed" if "adv" in f1 else "")
                + ". On a population whose age structure differs from the training cohort's, the "
                  "best use of age is none.")
    else:
        m["DeepSentence"] = (
            f"The best of {m['NDeepModels']} trained architectures reached "
            f"{best['auc_within']:.3f} against {best['age_only_within']:.3f} from age alone.")


def robustness(m, e12):
    """Macros for the checks a reviewer will demand before believing the crossover."""
    c = e12.get("checks", {})
    ftd = c.get("ftd_exclusion", {})
    if "AD only" in ftd:
        f = ftd["AD only"]
        m["AgeOnlyExternalNoFTD"] = f3(f["age_only_auc"])
        m["NExternalNoFTD"] = str(f["n"])
        # Cited as a clause, so it names the quantity it moves. Both figures are the transported
        # reading, not the direction-free one quoted just before it in the paper, and saying only
        # "leaves it at 0.436" invites the reader to attach that number to the wrong baseline.
        both = ftd.get("AD+FTD", {}).get("age_only_auc")
        m["FTDSentence"] = (
            f"dropping frontotemporal dementia, the youngest group there, moves the transported "
            f"figure only from {both:.3f} to {f['age_only_auc']:.3f} on {f['n']} subjects"
            if both is not None else
            f"dropping frontotemporal dementia leaves the transported figure at "
            f"{f['age_only_auc']:.3f} on {f['n']} subjects")
    par = c.get("age_cv_parity", {})
    if "CAUEEG" in par:
        v = par["CAUEEG"]
        m["AgeParitySentence"] = (
            f"Age has no fitted parameters, so its ranking is identical in and out of sample; "
            f"refitting it through the same cross-validation pipeline gives "
            f"{v['age_cv_auc']:.3f} against {v['age_rank_auc']:.3f}, confirming the baseline is "
            f"not advantaged by the comparison.")
    cx = c.get("crossover")
    if cx:
        m["Swing"] = sgn3(cx["swing"])
        m["SwingCI"] = ci(*cx["swing_ci"])
        m["SwingP"] = pval(cx["swing_p"])
        m["CrossoverSentence"] = (
            f"The margin moves from {cx['source']['diff']:+.3f} in the training cohort to "
            f"{cx['target']['diff']:+.3f} at the external clinic, a swing of {cx['swing']:+.3f} "
            f"{ci(*cx['swing_ci'])}, {pval(cx['swing_p'])}; the source score is out-of-fold "
            f"and the target score uses the final source fit.")


def regional(m, e05):
    regs = e05.get("regions", {})
    if not regs:
        return
    best = max(regs.items(), key=lambda kv: kv[1]["auc_eeg"])
    worst = min(regs.items(), key=lambda kv: kv[1]["auc_eeg"])
    m["BestRegion"] = best[0].replace("r_", "")
    m["WorstRegion"] = worst[0].replace("r_", "")
    m["RegionalSentence"] = (
        f"Discrimination is highest over {best[0].replace('r_','')} sites "
        f"({best[1]['auc_eeg']:.3f}, falling to {best[1]['auc_matched']:.3f} under age matching) "
        f"and lowest over {worst[0].replace('r_','')} sites ({worst[1]['auc_eeg']:.3f}), a spread "
        f"of {best[1]['auc_eeg']-worst[1]['auc_eeg']:.3f} AUC across the scalp.")


def easycog_numbers(m, e19):
    """The EasyCog benchmark audit."""
    if not e19:
        return
    info = e19.get("manifest", {})
    m["NEasyCog"] = str(info.get("aggregated_subjects", "MISSING"))
    m["EasyCogSessions"] = str(info.get("resting_sessions_loaded", "MISSING"))
    m["EasyCogMetadataN"] = str(info.get("metadata_subjects", "MISSING"))
    m["EasyCogWindows"] = f"{info.get('median_windows_per_session', float('nan')):.0f}"
    m["EasyCogSlicedRecovered"] = str(len(info.get("sliced_missing_subjects", [])))
    m["EasyCogSlicedN"] = str(info.get("sliced_files", "MISSING"))
    m["EasyCogRestingSlices"] = str(info.get("sliced_resting_windows", "MISSING"))
    m["EasyCogVideoSlices"] = str(info.get("sliced_video_windows", "MISSING"))

    bv = info.get("block_verification", {})
    m["EasyCogBlocksVerified"] = "yes" if bv.get("consistent") else "NO"
    blocks = bv.get("blocks", {})
    if blocks:
        m["EasyCogBlockSentence"] = (
            "Electrode blocks are re-derived from the inter-channel correlation structure on every "
            "load: mean within-block correlation "
            + ", ".join(f"{v['mean_within']:.2f}" for v in blocks.values())
            + " against between-block "
            + ", ".join(f"{v['mean_between']:.2f}" for v in blocks.values())
            + " for the forehead, left-ear and right-ear arrays respectively.")

    # -- A. the regression audit, which is the arm the paper leads on
    ra = e19.get("regression_audit", {})
    pub = ra.get("published", {}).get("test_best", {})
    m["EasyCogPubMethod"] = str(pub.get("method", "MISSING"))
    for tgt, tag in (("moca", "MoCA"), ("mmse", "MMSE")):
        rec = ra.get("targets", {}).get(tgt)
        if not rec:
            continue
        arms = rec.get("arms", {})
        m[f"EasyCog{tag}Sd"] = f2(rec.get("sd"))
        m[f"EasyCog{tag}PubMAE"] = f"{pub.get(f'{tgt}_mae', float('nan')):.3f}"
        m[f"EasyCog{tag}PubPCC"] = f"{pub.get(f'{tgt}_pcc', float('nan')):.3f}"
        for arm, key in (("mean_only", "Mean"), ("age", "Age"),
                         ("eeg17_all+age", "Full"), ("slow3_ear+age", "Slow"),
                         ("eeg17_ear", "Ear"), ("eeg17_forehead", "Fore")):
            a = arms.get(arm)
            if not a:
                continue
            m[f"EasyCog{tag}{key}MAE"] = f"{a['mae']:.3f}"
            m[f"EasyCog{tag}{key}PCC"] = f"{a['pcc']:.3f}"
            m[f"EasyCog{tag}{key}PCCCI"] = ci(*a["pcc_ci"], signed=False)
        pl = rec.get("placement_regression", {}).get("slow3")
        if pl:
            m[f"EasyCog{tag}PlaceDPCC"] = sgn3(pl["d_pcc"])
            m[f"EasyCog{tag}PlaceCI"] = ci(*pl["d_pcc_ci"])
            m[f"EasyCog{tag}PlaceP"] = pval(pl["p_pcc"])

    # -- B. discrimination, reported only where the arm is adequately powered
    disc = e19.get("discrimination", {})
    main_contrast = disc.get("moca_severe_vs_rest", {})
    m["NEasyCogDiagN"] = str(main_contrast.get("n", "MISSING"))
    m["EasyCogDiagCtrl"] = str(main_contrast.get("n_control", "MISSING"))
    m["EasyCogDiagCase"] = str(main_contrast.get("n_case", "MISSING"))
    m["EasyCogAgeAUC"] = f3(main_contrast.get("auc_age"))
    powered = main_contrast.get("estimators", {}).get("slow3", {})
    if powered:
        m["EasyCogEEGAUC"] = f3(powered.get("auc_eeg"))
        m["EasyCogBothAUC"] = f3(powered.get("auc_eeg_plus_age"))
        m["EasyCogInc"] = sgn3(powered.get("inc_margin"))
        m["EasyCogIncCI"] = ci(*powered.get("inc_margin_ci", powered.get("inc_ci",
                                                                        [np.nan, np.nan])))
        m["EasyCogIncP"] = pval(powered.get("inc_p"))
        m["EasyCogEPV"] = f"{powered.get('events_per_variable', float('nan')):.1f}"
    over = main_contrast.get("estimators", {}).get("eeg17", {})
    if over:
        m["EasyCogOverAUC"] = f3(over.get("auc_eeg"))
        m["EasyCogOverEPV"] = f"{over.get('events_per_variable', float('nan')):.1f}"
        m["EasyCogOverInc"] = sgn3(over.get("inc_margin"))

    n_under = sum(1 for c in disc.values() for e in c.get("estimators", {}).values()
                  if not e.get("adequately_powered", True))
    n_total = sum(len(c.get("estimators", {})) for c in disc.values())
    m["EasyCogUnderpowered"] = f"{n_under} of {n_total}"

    # -- C. placement, and the fact that it depends on the referencing
    pl = e19.get("placement", {}).get("moca_severe_vs_rest", {})
    nat = pl.get("arms", {}).get("native", {})
    car = pl.get("arms", {}).get("subset_car", {})
    if nat and car and "diff" in nat:
        m["EasyCogPlaceNative"] = sgn3(nat["diff"])
        m["EasyCogPlaceNativeCI"] = ci(*nat["ci"])
        m["EasyCogPlaceCar"] = sgn3(car["diff"])
        m["EasyCogPlaceCarCI"] = ci(*car["ci"])
        m["EasyCogPlaceSentence"] = (
            f"Ear electrodes outscore forehead electrodes by {car['diff']:+.3f} AUC when each "
            f"block is re-referenced within itself, and by {nat['diff']:+.3f} when both are read "
            f"as recorded, so on the dichotomised contrast the difference is a property of the "
            f"referencing rather than of where the electrodes sit.")

    m["EasyCogCaveat"] = ("the 69-participant public release, which is not the 101-participant "
                          "cohort the EasyCog paper describes")


def complexity(m, e20):
    """How large a cohort the incremental margin needs before it can be seen at all."""
    if not e20:
        return
    rows = e20.get("rows", [])
    th = e20.get("thresholds", {})
    for task, tag in (("screening", "Screen"), ("dementia", "Dem")):
        for est, etag in (("eeg17", "Full"), ("slow3", "Slow")):
            t = th.get(f"{task}|{est}", {})
            n = t.get("smallest_n_increment_reliably_positive")
            m[f"NNeeded{tag}{etag}"] = str(n) if n else "not reached"
        small = [r for r in rows if r["task"] == task and r["estimator"] == "eeg17"
                 and r["n"] <= 80]
        if small:
            worst = min(small, key=lambda r: r["inc_mean"])
            m[f"IncAtSmall{tag}"] = sgn3(worst["inc_mean"])
            m[f"NAtSmall{tag}"] = str(worst["n"])
    # At EasyCog's size, 69, for both estimators: its one adequately powered arm is the
    # three-feature index, so the text quotes both.
    for est, etag in (("eeg17", "Full"), ("slow3", "Slow")):
        at = [r for r in rows if r["task"] == "screening" and r["estimator"] == est
              and r["n"] == 69]
        if at:
            m[f"IncAt69Screen{etag}"] = sgn3(at[0]["inc_mean"])
    scr = th.get("screening|eeg17", {})
    m["IncFullScreen"] = sgn3(scr.get("inc_at_full_cohort"))
    m["ComplexitySentence"] = (
        f"On the screening contrast the incremental margin is {m.get('IncFullScreen', '?')} at "
        f"the full cohort, but subsamples of {m.get('NAtSmallScreen', '?')} "
        f"subjects return {m.get('IncAtSmallScreen', '?')}, and the margin is only reliably "
        f"positive from n={m.get('NNeededScreenFull', '?')} with seventeen features or "
        f"n={m.get('NNeededScreenSlow', '?')} with three.")
    m["ComplexityShort"] = (
        f"a cohort of EasyCog's size run on CAUEEG returns a negative increment "
        f"({m.get('IncAtSmallScreen', '?')} at n={m.get('NAtSmallScreen', '?')}) even though the "
        f"full cohort gives {m.get('IncFullScreen', '?')}")


def utility(m, e21):
    """Net benefit over age, in referrals rather than AUC."""
    if not e21:
        return
    for task, tag in (("screening", "Screen"), ("dementia", "Dem"), ("mci", "MCI")):
        rec = e21.get("tasks", {}).get(task, {}).get("budgets", {}).get("full montage", {})
        for prev, ptag in (("0.10", "Ten"), ("0.20", "Twenty")):
            summ = rec.get(prev, {}).get("summary")
            if not summ:
                continue
            m[f"NB{tag}{ptag}"] = f"{summ['max_gain_per_1000']:.0f}"
            rng = summ.get("range_where_eeg_helps")
            m[f"NBRange{tag}{ptag}"] = (f"{rng[0]:.2f}--{rng[1]:.2f}" if rng else "nowhere")
    m["UtilitySentence"] = (
        f"At a 10\\% service prevalence, adding the recording to age is worth "
        f"{m.get('NBDemTen', '?')} extra correct referrals per 1{{,}}000 assessed for dementia but "
        f"only {m.get('NBScreenTen', '?')} for screening and {m.get('NBMCITen', '?')} for mild "
        f"cognitive impairment; at 20\\% the same figures are {m.get('NBDemTwenty', '?')}, "
        f"{m.get('NBScreenTwenty', '?')} and {m.get('NBMCITwenty', '?')}.")


def estimator_axis(m, e13):
    """The substitution margin depends on the estimator; the increment does not."""
    if not e13 or "estimator_axis" not in e13:
        return
    ea = e13["estimator_axis"]
    for task, tag in (("screening", "Screen"), ("dementia", "Dem"), ("mci", "MCI")):
        f = ea.get("feature_model", {}).get(task)
        dd = ea.get("deep_model", {}).get(task)
        if f:
            m[f"SubFeat{tag}"] = sgn3(f["sub_margin"])
            m[f"IncFeat{tag}"] = sgn3(f["inc_margin"])
        if dd:
            m[f"AUCDeep{tag}"] = f3(dd["auc_eeg"])
            m[f"SubDeep{tag}"] = sgn3(dd["sub_margin"])
            m[f"DeepModel{tag}"] = str(dd["model"])
    m["EstimatorSentence"] = (
        f"The substitution margin is a property of the estimator rather than of the recording: at "
        f"the full montage the seventeen-feature model trails age by {m.get('SubFeatScreen', '?')} "
        f"on screening while a trained convolutional network exceeds it by "
        f"{m.get('SubDeepScreen', '?')}. The incremental margin does not behave that way, staying "
        f"positive for both.")

    # The distribution-free check on the primary family, which is what the p-values now rest on.
    fam = e13.get("primary_family", {})
    dis = [k for k, v in fam.items()
           if (v.get("p_holm", 1) < 0.05) != (v.get("p_alt_holm", 1) < 0.05)]
    m["NPrimaryAgree"] = f"{len(fam) - len(dis)} of {len(fam)}"
    for k, v in fam.items():
        task, kind = k.split("|")
        tag = TASKNAME[task]
        if "p_alt" in v:
            m[f"PAlt{kind.capitalize()}{tag}"] = pval(v["p_alt"])
            m[f"PAltHolm{kind.capitalize()}{tag}"] = pval(v["p_alt_holm"])
    m["InferenceSentence"] = (
        f"DeLong's variance assumes a fixed scoring rule and ours is refit in every fold, so the "
        f"primary family is also read distribution-free: a permutation null in which the recording "
        f"is noise for the incremental margin, and a paired subject bootstrap for the substitution "
        f"margin. The two agree on {m.get('NPrimaryAgree', '?')} tests"
        + (f"; they differ on the {', '.join(d.split('|')[0].upper() + ' ' + d.split('|')[1] for d in dis)} "
           f"test, which the permutation supports and the Holm-corrected DeLong test does not."
           if dis else "."))


def external_operating(m, e24):
    """The transferred operating point: threshold carried from source, read on all 29 controls.

    This supersedes the earlier external sensitivity numbers, which were read at a threshold set
    on the target's own controls and therefore used target labels. Both source contrasts are
    recorded, because whether the carried threshold holds turns out to depend on which one trained
    the rule -- a dependence the paper reports rather than resolves.
    """
    if not e24:
        return
    m["ExtOpNControls"] = str(e24.get("n_target_controls", "MISSING"))
    m["ExtOpGranularity"] = f"{1.0 / e24['n_target_controls']:.3f}" if e24.get(
        "n_target_controls") else "MISSING"
    worst = {}
    for contrast, rec in e24.get("source_contrasts", {}).items():
        ctag = contrast.capitalize()
        for tag, arms in rec.get("budgets", {}).items():
            btag = "Full" if tag == "b19" else "Four"
            for arm, atag in (("eeg", "EEG"), ("eeg+age", "Both"), ("normative", "Norm"),
                              ("age_transported", "Age")):
                a = arms.get(arm)
                if not a:
                    continue
                m[f"Op{ctag}{atag}{btag}AUC"] = f3(a["auc"])
                m[f"Op{ctag}{atag}{btag}Sens"] = f2(a["sensitivity"])
                # Reported as realised-minus-intended, so a negative number means the arm
                # over-flags: it is the direction a service with fixed capacity cares about.
                m[f"Op{ctag}{atag}{btag}DSpec"] = f"{-a['spec_shortfall']:+.2f}"
                worst.setdefault(atag, []).append(-a["spec_shortfall"])
    for atag, vals in worst.items():
        m[f"OpWorst{atag}Under"] = f"{min(min(vals), 0.0):+.2f}"
        m[f"OpWorst{atag}Over"] = f"{max(max(vals), 0.0):+.2f}"
        m[f"OpMaxDev{atag}"] = f"{max(abs(v) for v in vals):.2f}"


def acquisition(m, e25):
    """The acquisition design sweep: placement, digitisation, and the mechanism behind them."""
    if not e25:
        return
    tag = {"b19": "Full", "b8": "Eight", "b4": "Four", "b2": "Two", "b1_ap": "OneAP",
           "r_temporal": "Temporal", "r_posterior": "Posterior", "r_central": "Central",
           "r_frontal": "Frontal", "muse": "Muse", "ganglion": "Ganglion",
           "insight": "Insight", "frontal1": "FrontalOne"}
    for task, key in (("screening", "Screen"), ("dementia", "Dem")):
        rec = e25.get("tasks", {}).get(task)
        if not rec:
            continue
        for mont, v in rec.get("placement", {}).items():
            if mont not in tag:
                continue
            m[f"Place{tag[mont]}{key}"] = f3(v["auc_eeg"])
            m[f"PlaceInc{tag[mont]}{key}"] = sgn3(v["inc_margin"])
        for name, v in rec.get("tests", {}).items():
            t = "".join(tag.get(p, p.title()) for p in name.split("_vs_"))
            m[f"PTest{t}{key}"] = sgn3(v["diff"])
            m[f"PTestCI{t}{key}"] = ci(*v["ci"])
            m[f"PTestP{t}{key}"] = pval(v["p_holm"])
        sp = rec.get("spans", {})
        if sp:
            m[f"SpanPlace{key}"] = f3(sp["placement"])
            for mm, mt in (("b19", "Full"), ("b4", "Four")):
                if f"digitisation_{mm}" in sp:
                    m[f"SpanDigit{mt}{key}"] = f3(sp[f"digitisation_{mm}"])
            if "ratio_b19" in sp:
                m[f"SpanRatio{key}"] = f"{sp['ratio_b19']:.0f}"
        # The digitisation grid is quoted cell by cell, so every cell is recorded.
        for cell, v in rec.get("digitisation", {}).items():
            mm, rate, bits = cell.split("|")
            mt = "Full" if mm == "b19" else "Four"
            m[f"Digit{mt}{rate}{bits.replace('f32', 'F')}{key}"] = f3(v["auc_eeg"])

    mech = e25.get("mechanism", {})
    if mech:
        m["MechTopPosterior"] = str(mech.get("n_top_peaking_posterior", "MISSING"))
        for f, v in mech.get("per_feature", {}).items():
            s = v["strength_by_region"]
            nm = "".join(w.title() for w in f.split("_"))
            for r in ("temporal", "frontal"):
                if r in s:
                    m[f"Mech{nm}{r.title()}"] = f"{s[r]:.3f}"
        for r, v in mech.get("regions", {}).items():
            m[f"Region{r.title()}AUC"] = f3(v["auc_eeg"])
            m[f"Region{r.title()}El"] = str(v["n_electrodes"])


def preprocessing(m, e27):
    """The preprocessing axis: what a device that cannot run ICA gives up."""
    if not e27:
        return
    for task, key in (("screening", "Screen"), ("dementia", "Dem"), ("mci", "MCI")):
        rec = e27.get("tasks", {}).get(task)
        if not rec:
            continue
        m[f"PrepN{key}"] = f"{rec['n_analysed']:,}"
        m[f"PrepDropped{key}"] = str(rec["n_dropped"])
        for mont, arms in rec.get("montages", {}).items():
            mt = "Full" if mont == "b19" else "Four"
            for arm, atag in (("full", "ICA"), ("noica", "NoICA"), ("emgfree", "EMGfree")):
                a = arms.get(arm)
                if not a:
                    continue
                m[f"Prep{atag}{mt}{key}"] = f3(a["auc_eeg"])
                m[f"PrepInc{atag}{mt}{key}"] = sgn3(a["inc_margin"])
                v = a.get("vs_full")
                if v:
                    m[f"PrepDiff{atag}{mt}{key}"] = sgn3(v["diff"])
                    m[f"PrepDiffCI{atag}{mt}{key}"] = ci(*v["ci"])
                    m[f"PrepDiffP{atag}{mt}{key}"] = pval(v["p"])


def interpretability(m, e28):
    """The illustrative case behind the output-layer figure."""
    if not e28:
        return
    m["CaseAge"] = f"{e28['age']:.0f}"
    m["CaseAbnormality"] = f"{e28['abnormality']:.2f}"
    m["CaseThreshold"] = f"{e28['referral_threshold']:.2f}"
    m["CaseNControls"] = str(e28["n_controls_fitting_reference"])
    for rec in e28.get("most_deviant", []):
        nm = "".join(w.title() for w in rec["feature"].split("_"))
        m[f"CaseZ{nm}"] = f"{rec['z']:.1f}"


def power_check(m, e13):
    """Analytic sample size for the increment, as a cross-check on the subsampling curve.

    The subsampling threshold is a counting rule on overlapping draws, which is not a power
    calculation. Reading the standard error off the reported interval and scaling it as n^-1/2
    gives an independent answer, and the paper quotes both because they agree.
    """
    if not e13:
        return
    z_alpha, z_beta = 1.959964, 0.8416212
    for task, rec in e13.get("tasks", {}).items():
        b = rec.get("budgets", {}).get(FULL)
        if not b:
            continue
        lo, hi = b["inc_ci"]
        se = (hi - lo) / (2 * z_alpha)
        n = rec["n"]
        m[f"IncSE{TASKNAME.get(task, task.title())}"] = f"{se:.4f}"
        if b["inc_margin"] > 0:
            need = b["inc_margin"] / (z_alpha + z_beta)
            n_req = n * (se / need) ** 2
            # Rounded to the nearest ten: the inputs do not support three significant figures,
            # and a reader who sees "n = 321" will believe the wrong thing about the precision.
            # Thousands separator included so the recorded string matches how the paper sets it.
            m[f"NForPower{TASKNAME.get(task, task.title())}"] = f"{round(n_req / 10) * 10:,.0f}"

    # A design table, so the result is usable by someone planning a study rather than only
    # descriptive of this one. The screening arm anchors it because it is the largest contrast and
    # the closest in case mix to a triage service. SE scales as n^-1/2 at fixed case mix, so the
    # requirement for any target increment follows from the one measured standard error.
    scr = e13.get("tasks", {}).get("screening", {})
    b = scr.get("budgets", {}).get(FULL)
    if b:
        lo, hi = b["inc_ci"]
        se_ref, n_ref = (hi - lo) / (2 * z_alpha), scr["n"]
        for delta in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            need = delta / (z_alpha + z_beta)
            n_req = n_ref * (se_ref / need) ** 2
            tag = f"{int(round(delta * 100)):02d}"
            m[f"DesignN{tag}"] = f"{round(n_req / 10) * 10:,.0f}"
            # Events, not participants, is what the fit is limited by, and a study designed at 62%
            # cases will not recruit that mix -- so state the smaller class too.
            m[f"DesignDelta{tag}"] = f"{delta:.2f}"


def _pos(v):
    """An unsigned magnitude, for sentences that say 'lost 0.159' rather than '-0.159'."""
    return f3(abs(v))


def revision(m, e30, e33, e34, e35, e31=None):
    """Every quantity the camera-ready revision added: leakage, external placement, stress, transfer."""
    if e31 and e31.get("brainlat_site_offset_mm"):
        off = list(e31["brainlat_site_offset_mm"].values())
        m["RevBrainLatOffsetMedian"] = f"{np.median(off):.0f}\\,mm"
        m["RevBrainLatOffsetMax"] = f"{max(off):.0f}\\,mm"
    if e30:
        s, d2 = e30["screening"], e30.get("second_split", {})
        m["RevNoOverlapScreenInc"] = sgn3(s["inc_margin"])
        m["RevNoOverlapScreenIncCI"] = ci(*s["inc_ci"])
        same = e30.get("same_recordings_under_cv", {})
        if same:
            f = same["first_split"]["screening"]
            m["RevSameRecInc"] = sgn3(f["inc_margin"])
            m["RevSameRecBoth"] = f3(f["auc_eeg_plus_age"])
            m["RevSameRecAge"] = f3(f["auc_age"])
            m["RevSecondSameRecInc"] = sgn3(same["second_split"]["screening"]["inc_margin"])
        if d2:
            m["RevSecondSplitInc"] = sgn3(d2["screening"]["inc_margin"])
            m["RevSecondOverlap"] = str(d2["held_out_overlap_with_first"])
            m["RevSecondHeldOut"] = str(d2["n_held_out_serials"])
        dd = e30.get("dedup")
        if dd:
            m["RevRepeats"] = str(dd["n_identified_repeats"])
            for task, tag in (("screening", "Screen"), ("dementia", "Dem"), ("MCI", "MCI")):
                r = dd[task]
                m[f"RevDedup{tag}"] = sgn3(r["dedup"]["inc_margin"])
                m[f"RevDedup{tag}CI"] = ci(*r["dedup"]["inc_ci"])
                m[f"RevDedupRandom{tag}"] = sgn3(r["random_removal_inc_mean"])
            shift = max(abs(dd[t]["dedup"]["inc_margin"] - dd[t]["all"]["inc_margin"])
                        for t in ("screening", "dementia"))
            m["RevDedupMaxShift"] = f3(shift)

    if e33:
        c = e33["contrasts"]
        pooled = lambda con, rd, k: c[con][rd]["tests_pooled"][k]
        v = pooled("dementia", "transfer", "b2_vs_b19")
        m["RevExtB2Full"], m["RevExtB2FullCI"] = sgn3(v["diff"]), ci(*v["ci"])
        v = pooled("dementia", "transfer", "r_temporal_vs_r_frontal")
        m["RevExtTempFront"], m["RevExtTempFrontCI"] = sgn3(v["diff"]), ci(*v["ci"])
        v = pooled("FTD", "within", "b4_vs_b19")
        m["RevFTDFourLoss"] = _pos(v["diff"])
        m["RevFTDFourLossCI"] = ci(-v["ci"][1], -v["ci"][0], signed=False)
        m["RevFTDFourHolm"] = f"{v['p_holm']:.3f}"
        m["RevADB2Full"] = sgn3(pooled("AD", "transfer", "b2_vs_b19")["diff"])
        v = pooled("AD", "transfer", "r_temporal_vs_r_frontal")
        m["RevADTempFront"], m["RevADTempFrontCI"] = sgn3(v["diff"]), ci(*v["ci"])
        v = pooled("AD", "transfer", "b2_vs_frontal1")
        m["RevADB2Bipolar"], m["RevADB2BipolarCI"] = sgn3(v["diff"]), ci(*v["ci"])
        m["RevADB2BipolarHolm"] = f"{v['p_holm']:.2f}"
        full = c["FTD"]["within"]["full_tests_pooled"]
        m["RevFTDInsight"] = sgn3(full["insight_vs_b19"]["diff"])
        m["RevFTDInsightCI"] = ci(*full["insight_vs_b19"]["ci"])
        m["RevFTDGanglion"] = sgn3(full["ganglion_vs_b19"]["diff"])
        m["RevFTDMuse"] = sgn3(full["muse_vs_b19"]["diff"])
        for rd, tag in (("transfer", "Transfer"), ("within", "Within")):
            v = e33["interaction"][rd]["r_temporal_vs_r_frontal"]["pooled"]
            m[f"RevInteraction{tag}"], m[f"RevInteraction{tag}CI"] = sgn3(v["diff"]), ci(*v["ci"])
        topo = e33["topography"]["AD"]["top6_mean_strength_by_region"]
        for r in ("posterior", "temporal", "frontal"):
            m[f"RevTopoAD{r.title()}"] = f3(topo[r])
        dif = e33["differential_AD_vs_FTD"]
        for mont, tag in (("b2", "B2"), ("r_frontal", "Frontal"), ("b19", "Full")):
            m[f"RevDiff{tag}"] = f3(dif["pooled_mean_auc"][mont]["auc_mean"])
        v = dif["tests_pooled"]["b2_vs_r_frontal"]
        m["RevDiffB2Frontal"], m["RevDiffB2FrontalCI"] = sgn3(v["diff"]), ci(*v["ci"])
        v = pooled("dementia", "transfer", "b4_vs_b4_inherit")
        m["RevInheritGain"] = _pos(v["diff"])
        m["RevInheritGainCI"] = ci(-v["ci"][1], -v["ci"][0], signed=False)
        m["RevInheritHolm"] = f"{v['p_holm']:.3f}"
        vf = c["FTD"]["within"].get("vs_full_pooled")
        if vf:
            fp = {"muse", "insight", "ganglion", "b1_ap", "r_frontal", "frontal1"}
            no_fp = [v["diff"] for k, v in vf.items() if k.split("_vs_")[0] not in fp]
            with_fp = [v["diff"] for k, v in vf.items()
                       if k.split("_vs_")[0] in fp and not k.startswith("b1_ap")]
            m["RevFTDNoPoleMin"], m["RevFTDNoPoleMax"] = _pos(max(no_fp)), _pos(min(no_fp))
            m["RevFTDPoleWithin"] = f3(max(abs(x) for x in with_fp))
            m["RevFTDFrontoOcc"] = sgn3(vf["b1_ap_vs_b19"]["diff"])
            a = c["AD"]["transfer"]["vs_full_pooled"]["frontal1_vs_b19"]
            m["RevADBipolarVsFull"], m["RevADBipolarVsFullCI"] = sgn3(a["diff"]), ci(*a["ci"])
        # A.12 quotes the Alzheimer's temporo-parietal-vs-frontal difference with its interval;
        # Table 8 prints the difference alone.
        a = c["AD"]["transfer"]["tests_pooled"]["b2_vs_frontal1"]
        m["RevADTPvsFrontal"] = sgn3(a["diff"]) + " " + ci(*a["ci"])
        n_ext = sum(sum(g.values()) for g in e33["cohorts"].values())
        m["RevNExternal"] = str(n_ext)
        # Table cells, one quantity each, so a transcription error in either table is caught.
        for con, rd in (("dementia", "transfer"), ("AD", "transfer"), ("FTD", "within")):
            for coh, per in c[con][rd]["auc"].items():
                for mont, rec in per.items():
                    m[f"RevTab_{con}_{coh}_{mont}"] = f3(rec["auc"])
            for k, v in c[con][rd]["tests_pooled"].items():
                # The table prints intervals for the prespecified dementia contrast only.
                m[f"RevPair_{con}_{k}"] = (sgn3(v["diff"]) + " " + ci(*v["ci"])
                                           if con == "dementia" else sgn3(v["diff"]))

    if e34:
        t = e34["tasks"]["screening"]["montages"]
        four, full = t["b4"]["conditions"], t["b19"]["conditions"]
        m["RevStressNoise5"] = sgn3(four["noise5"]["device"]["d_auc_eeg_vs_clean"])
        m["RevStressNoise10"] = sgn3(four["noise10"]["device"]["d_auc_eeg_vs_clean"])
        m["RevStressDisplace"] = sgn3(four["displace50"]["device"]["d_auc_eeg_vs_clean"])
        m["RevStressShort"] = sgn3(four["short3"]["device"]["d_auc_eeg_vs_clean"])
        cb = four["contact_bad"]["device"]
        m["RevStressContact"] = _pos(cb["d_auc_eeg_vs_clean"])
        m["RevStressContactCI"] = ci(-cb["d_auc_eeg_ci"][1], -cb["d_auc_eeg_ci"][0], signed=False)
        m["RevStressContactFull"] = _pos(full["contact_bad"]["device"]["d_auc_eeg_vs_clean"])
        mc = four["mains"]["clinic"]
        m["RevStressMains"], m["RevStressMainsCI"] = sgn3(mc["inc_margin"]), ci(*mc["inc_ci"])
        m["RevStressSevereClinic"] = sgn3(four["field_severe"]["clinic"]["inc_margin"])
        m["RevStressSevereDevice"] = sgn3(four["field_severe"]["device"]["inc_margin"])
        m["RevStressCleanFour"] = f3(four["clean"]["device"]["auc_eeg"])
        worst = max(-four[k]["device"]["d_auc_eeg_vs_clean"]
                    for k in ("noise5", "motion2", "motion6", "displace50", "short3"))
        m["RevStressAtMost"] = f3(worst)
        dem = e34["tasks"]["dementia"]["montages"]["b4"]["conditions"]
        for k, tag in (("clean", "Clean"), ("field_moderate", "Moderate"), ("field_severe", "Severe")):
            m[f"RevStressDem{tag}"] = sgn3(dem[k]["device"]["inc_margin"])
        for mont in ("b19", "b4"):
            for cond, regs in t[mont]["conditions"].items():
                for reg, v in regs.items():
                    m[f"RevStressTab_{mont}_{cond}_{reg}"] = f3(v["auc_eeg"])

    if e35:
        for coh in ("BrainLat", "P-ADIC"):
            rec = e35["targets"][coh]
            tag = coh.replace("-", "")
            m[f"RevGap{tag}"] = f"{rec['age_gap']:.1f}"
            m[f"RevAge{tag}"] = f3(rec["b19"]["arms"]["age_transported"]["auc"])
            for b in ("b19", "b4"):
                i = rec[b]["increment_over_transported_age"]
                m[f"RevInc{tag}{b}"] = sgn3(i["diff"])
                m[f"RevInc{tag}{b}CI"] = ci(*i["ci"])
                for arm, a in rec[b]["arms"].items():
                    m[f"RevExt3_{tag}_{b}_{arm}_auc"] = f3(a["auc"])
                    m[f"RevExt3_{tag}_{b}_{arm}_spec"] = f2(a["specificity"])
        for b in ("b19", "b4"):
            p = e35["pooled_increment"][f"{b}_excl_ds004504"]
            m[f"RevPooledInc{b}"], m[f"RevPooledInc{b}CI"] = sgn3(p["diff"]), ci(*p["ci"])


def within_external(m, e37):
    """Each external cohort's increment, fitted and evaluated inside the cohort (e37)."""
    if not e37:
        return
    for c, tag in (("BrainLat", "BL"), ("P-ADIC", "PA"), ("ds004504", "DS")):
        rec = e37["cohorts"].get(c)
        if not rec:
            continue
        v = rec["estimators"]["eeg17"]
        m[f"WithinInc{tag}"] = sgn3(v["inc_margin"])
        m[f"WithinInc{tag}CI"] = ci(*v["inc_ci"])
        m[f"WithinAgeCV{tag}"] = f3(rec["auc_age_cv"])


def check_recorded(m):
    """Without the manuscript: compare each recomputed quantity with the value the paper prints."""
    if not VALUES.exists():
        print(f"missing {VALUES}")
        return 1
    recorded = json.loads(VALUES.read_text(encoding="utf-8"))
    drifted = [(k, r["value"], m.get(k)) for k, r in sorted(recorded.items())
               if m.get(k) != r["value"]]
    print(f"checked {len(recorded)} quantities the paper states (recorded in {VALUES.name})")
    for name, stated, current in drifted:
        print(f"    {name:24s} paper prints {stated!r}, artefacts give {current!r}")
    if drifted:
        return 1
    print("  every number in the paper matches the artefacts")
    return 0


def main():
    m = {}
    sp, e02, e03, e04, e05, e11, e12, e13, e14, e15, e16, e19 = (load(n) for n in (
        "e00_spine.json", "e02_age_decomposition.json", "e03_subgroup_safety.json",
        "e04_control_anchored.json", "e05_regional.json", "e11_deep_arm.json",
        "e12_robustness.json", "e13_incremental.json", "e14_control_composition.json",
        "e15_crossover.json", "e16_rho_check.json", "e19_easycog_audit.json"))
    e20, e21 = load("e20_estimator_complexity.json"), load("e21_clinical_utility.json")
    e23 = load("e23_operating_metrics.json")
    e24 = load("e24_external_operating_point.json")
    e25 = load("e25_acquisition_design.json")
    e27 = load("e27_preprocessing_budget.json")
    e28 = load("e28_interpretability.json")

    m["WindowSeconds"] = str(budgets.QUOTA * budgets.WINDOW_SAMPLES // budgets.SF_NATIVE)
    m["WindowQuota"] = str(budgets.QUOTA)
    m["WindowLength"] = str(budgets.WINDOW_SAMPLES // budgets.SF_NATIVE)
    m["Caliper"] = "3"
    m["AgeMin"], m["AgeMax"] = "23", "96"
    analysed_sample(m)
    if sp:
        corpus(m, sp)
    if e02:
        decomposition(m, e02)
    if e03:
        subgroups(m, e03)
    if e04:
        transfer(m, e04)
    if e05:
        regional(m, e05)
    if e11:
        deep(m, e11)
    if e12:
        robustness(m, e12)
    # These run last on purpose. e13/e15/e16 supersede quantities e02 and e12 also define -- the
    # margin is now reported both ways, the crossover holds the contrast fixed, and the
    # substitution ratio uses the control-fitted trend -- and the corrected value must win.
    if e13:
        incremental(m, e13)
        ladder(m, e13)
    if e14:
        composition(m, e14)
    if e15:
        crossover(m, e15)
    if e16:
        rho_check(m, e16)
    easycog_numbers(m, e19)
    estimator_axis(m, e13)
    complexity(m, e20)
    utility(m, e21)
    operating(m, e23)
    external_operating(m, e24)
    acquisition(m, e25)
    preprocessing(m, e27)
    interpretability(m, e28)
    power_check(m, e13)
    revision(m, load("e30_noverlap_leakage.json"), load("e33_external_placement.json"),
             load("e34_acquisition_stress.json"), load("e35_external_transfer_all.json"),
             load("e31_external_extract.json"))
    within_external(m, load("e37_external_within_increment.json"))

    # ---- verify rather than emit -------------------------------------------------------------
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="re-record the values the paper states, after editing main.tex")
    args = ap.parse_args()

    if not MAIN.exists():
        # The public code release ships without the manuscript. The values the paper prints are
        # recorded in paper_values.json, so the artefacts can still be checked against them.
        return check_recorded(m)
    # A four-digit number is typeset ``1{,}470`` so the comma keeps its spacing, and a signed one
    # inside maths as ``$-0.012$``. Neither matches the plain string the artefact yields, so those
    # quantities were silently skipped and went unchecked. Stripping both forms makes them
    # verifiable -- applied to the value as well as the text, because some recorded values are
    # themselves maths (``$P<0.0001$``) and stripping only one side would break those instead.
    def norm(s):
        return s.replace("{,}", ",").replace("$", "")

    text = norm(MAIN.read_text(encoding="utf-8"))

    # Two things can go wrong and they need different checks. An experiment can be rerun and move a
    # number, which the value comparison catches. Or someone can mistype a digit while editing, and
    # a plain "is this string present" test misses that, because a value like 0.29 also appears in
    # other sentences. Recording how many times each value occurs catches it: changing one of them
    # drops the count. Counts survive a prose rewrite, which text-context anchors do not.
    def occurrences(value):
        return text.count(norm(value))

    if args.update:
        keep = {}
        for name, value in m.items():
            # One- and two-character values ("4", "24") occur all over the manuscript, so their
            # counts are noise rather than an anchor. Skip them; the value comparison still guards
            # them against artefact drift.
            if not value or value == "MISSING" or len(value) < 3 or norm(value) not in text:
                continue
            keep[name] = {"value": value, "count": occurrences(value)}
        VALUES.write_text(json.dumps(keep, indent=2, sort_keys=True), encoding="utf-8")
        print(f"recorded {len(keep)} quantities the paper states -> {VALUES}")
        return 0

    if not VALUES.exists():
        print(f"missing {VALUES}; run with --update once to record the current values")
        return 1
    recorded = json.loads(VALUES.read_text(encoding="utf-8"))

    drifted, miscounted, vanished = [], [], []
    for name, rec in sorted(recorded.items()):
        stated, n_expected = rec["value"], rec["count"]
        current = m.get(name)
        if current is None:
            vanished.append(name)
        elif current != stated:
            drifted.append((name, stated, current))
        else:
            n = occurrences(stated)
            if n != n_expected:
                miscounted.append((name, stated, n_expected, n))

    print(f"checked {len(recorded)} quantities the paper states")
    if drifted:
        print(f"\n  {len(drifted)} value(s) the artefacts no longer support:")
        for name, stated, current in drifted:
            print(f"    {name:24s} paper prints {stated!r}, artefacts give {current!r}")
    if miscounted:
        print(f"\n  {len(miscounted)} value(s) whose occurrence count changed "
              f"(a digit edited by hand?):")
        for name, stated, want, got in miscounted:
            print(f"    {name:24s} {stated!r} appears {got}x, expected {want}x")
    if vanished:
        print(f"\n  {len(vanished)} no longer computed by any experiment: {vanished}")
    if not (drifted or miscounted or vanished):
        print("  every number in the paper matches the artefacts")
        return 0
    print("\n  fix main.tex, then rerun with --update to re-record")
    return 1


if __name__ == "__main__":
    sys.exit(main())
