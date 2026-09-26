"""Run the whole pipeline in order, resumably, and refuse to run twice at once.

Stages are ordered by dependency: corpus extension, then the spine that indexes it, then feature
extraction, then the analyses that read the features, then figures and the generated paper macros.

Three properties this needs and did not have:

**Single instance.** A lock file records the running process id. A second invocation exits rather
than racing the first over the same output files -- two queues writing the same artefacts is silent
corruption, not a doubled speed.

**Resumable.** Completed stages are recorded in `outputs/logs/queue_state.json` and skipped on a
later invocation, so a run interrupted by a reboot, a power cut or a closed terminal continues
where it stopped instead of repeating hours of work.

**Detached.** Nothing here depends on the terminal that started it. Launch with `run_pipeline.cmd`
and the queue survives the shell, the network and the session that spawned it.

    python run_queue.py                          # run whatever is not yet done
    python run_queue.py --redo e02,e03           # force specific stages to run again
    python run_queue.py --only e11a              # run one stage
    python run_queue.py --assume-done e00,e01    # mark stages complete without running them
    python run_queue.py --status                 # print what is done and what remains
Out: outputs/logs/<stage>.log, outputs/logs/queue_state.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOGS = ROOT / "outputs" / "logs"
STATE = LOGS / "queue_state.json"
LOCK = LOGS / "queue.lock"
CONSOLE = LOGS / "queue_console.log"


def say(msg):
    """Print and persist in one place.

    The queue used to be tee'd through a shell pipe. That pipe was one of the ways a second
    interpreter got involved, so the queue now writes its own transcript and the launcher does
    nothing but start it.
    """
    print(msg, flush=True)
    with CONSOLE.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")

#: name, command, description, optional, artefact that proves it ran
STAGES = [
    ("h00", ["experiments/h00_harmonise.py"],
     "harmonise CAUEEG, ds004504, BrainLat and P-ADIC into the cache", False, None),
    ("e09v", ["experiments/e09_extend_corpus.py", "--job", "vascular"],
     "recover vascular dementia and TGA subjects", False, "results/e09_vascular.json"),
    ("e09n", ["experiments/e09_extend_corpus.py", "--job", "noica"],
     "generate the no-ICA preprocessing arm", False, "results/e09_noica.json"),
    ("e00", ["experiments/e00_spine.py"], "build the subject table", False, "cache/spine.csv"),
    ("e01", ["experiments/e01_extract.py"], "features under every acquisition budget", False,
     "cache/features.npz"),
    ("e02", ["experiments/e02_age_decomposition.py"], "age decomposition per task and budget",
     False, "results/e02_age_decomposition.json"),
    ("e03", ["experiments/e03_subgroup_safety.py"], "subgroup sensitivity", False,
     "results/e03_subgroup_safety.json"),
    ("e04", ["experiments/e04_control_anchored.py"], "control-anchored transfer", False,
     "results/e04_control_anchored.json"),
    ("e05", ["experiments/e05_regional.py"], "regional dissociation", False,
     "results/e05_regional.json"),
    ("e12", ["experiments/e12_robustness.py"], "robustness checks on the crossover", False,
     "results/e12_robustness.json"),
    # These four supersede quantities e02 and e12 also produce, and e07 applies them last so the
    # corrected value is the one the paper cites: the incremental margin beside the substitution
    # margin, the recruitment manipulation the crossover claim now rests on, the crossover with
    # the clinical contrast held fixed, and the two debiased substitution-ratio estimators.
    ("e13", ["experiments/e13_incremental.py"],
     "incremental vs substitution margin, Holm-corrected, plus the placement test", False,
     "results/e13_incremental.json"),
    ("e14", ["experiments/e14_control_composition.py"],
     "the age baseline moved by redefining 'control' within one cohort", False,
     "results/e14_control_composition.json"),
    ("e15", ["experiments/e15_crossover.py"],
     "two-cohort crossover, contrast held fixed and baseline read three ways", False,
     "results/e15_crossover.json"),
    ("e16", ["experiments/e16_rho_check.py"],
     "substitution ratio: sample-size and pooled-fit biases isolated", False,
     "results/e16_rho_check.json"),
    ("e19", ["experiments/e19_easycog_audit.py"],
     "EasyCog: age-baseline audit, estimator complexity, placement, recruitment", True,
     "results/e19_easycog_audit.json"),
    ("e20", ["experiments/e20_estimator_complexity.py"],
     "how many subjects the incremental margin needs before it is estimable", False,
     "results/e20_estimator_complexity.json"),
    ("e21", ["experiments/e21_clinical_utility.py"],
     "decision-curve net benefit against age at service prevalence", False,
     "results/e21_clinical_utility.json"),
    # The deep arm is the long pole and is split so the load-bearing configuration finishes first.
    # All three write into one merged artefact after every configuration, so an interrupted run
    # still leaves usable results and a resumed one does not lose them.
    ("e11a", ["experiments/e11_deep_arm.py", "--montages", "b19", "--tasks", "screening",
              "--variants", "none,input,normative,normative_anchored,adv",
              "--models", "eegnet,shallow,deep4", "--epochs", "15", "--seeds", "3"],
     "deep arm: age-variant ablation on the primary contrast (GPU, hours)", True,
     "results/e11_deep_arm.json"),
    ("e11b", ["experiments/e11_deep_arm.py", "--montages", "b19",
              "--tasks", "dementia,mci", "--variants", "none",
              "--models", "eegnet,shallow,deep4", "--epochs", "15", "--seeds", "3"],
     "deep arm: remaining tasks (GPU, hours)", True, None),
    ("e11c", ["experiments/e11_deep_arm.py", "--montages", "b4", "--tasks", "screening",
              "--variants", "none", "--models", "eegnet,shallow,deep4",
              "--epochs", "15", "--seeds", "3"],
     "deep arm: four electrodes (GPU, ~1h)", True, None),
    ("e23", ["experiments/e23_operating_metrics.py"],
     "operating-point metrics, predictive values, and the external cohort", False,
     "results/e23_operating_metrics.json"),
    ("e24", ["experiments/e24_external_operating_point.py"],
     "external operating point at a threshold carried from the source", False,
     "results/e24_external_operating_point.json"),
    ("e25", ["experiments/e25_acquisition_design.py"],
     "acquisition design: placement, digitisation, and the mechanism behind them", False,
     "results/e25_acquisition_design.json"),
    ("e27", ["experiments/e27_preprocessing_budget.py"],
     "preprocessing axis: what a device gives up when it cannot run ICA", False,
     "results/e27_preprocessing_budget.json"),
    ("e30", ["experiments/e30_noverlap_leakage.py"],
     "patient-level leakage: official no-overlap splits and repeat-visit de-duplication", False,
     "results/e30_noverlap_leakage.json"),
    ("e31", ["experiments/e31_external_extract.py"],
     "external cohorts (ds004504, BrainLat, P-ADIC): spine and montage features", False,
     "cache/features_external.npz"),
    ("e32", ["experiments/e32_stress_extract.py"],
     "acquisition stress test: inject field faults and re-extract (~15 min)", False,
     "cache/features_stress.npz"),
    ("e33", ["experiments/e33_external_placement.py"],
     "placement on three external cohorts, and by disease", False,
     "results/e33_external_placement.json"),
    ("e34", ["experiments/e34_acquisition_stress.py"],
     "acquisition stress test: device- and clinic-trained scoring", False,
     "results/e34_acquisition_stress.json"),
    ("e35", ["experiments/e35_external_transfer_all.py"],
     "external operating point carried to all three external cohorts", False,
     "results/e35_external_transfer_all.json"),
    # Figure 1 is a diagrams.net drawing (outputs/figures/fig1_framework.drawio); e29 also
    # draws the montage strip it embeds.
    ("e29", ["experiments/e29_montage_map.py"],
     "montage decomposition drawn as devices", False, "figures/figA7_montages.pdf"),
    ("e36", ["experiments/e36_revision_figures.py"],
     "appendix figures: placement by disease, acquisition stress test", False,
     "figures/figA8_external_placement.pdf"),
    ("e17", ["experiments/e17_paper_figure.py"], "the paper's results figure", False,
     "figures/fig1_results.pdf"),
    ("e07", ["experiments/e07_numbers.py"],
     "verify every number printed in the paper against the artefacts", False, None),
]


def alive(pid):
    """Is a process with this id still running? Windows has no signal-0 trick, so ask tasklist."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
        return str(pid) in out
    except Exception:
        return False


def take_lock():
    """Claim the queue, atomically.

    Checking whether the lock exists and then creating it is two operations, and two processes
    starting together can both pass the check before either writes -- which is exactly how this
    pipeline came to run twice over the same output files. ``O_CREAT | O_EXCL`` makes the test and
    the claim one indivisible operation, so precisely one process can win no matter how many start
    at the same instant.
    """
    for attempt in (1, 2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            return True
        except FileExistsError:
            try:
                old = int(LOCK.read_text().split()[0])
            except (ValueError, IndexError, OSError):
                old = None
            if old and old != os.getpid() and alive(old):
                say(f"another queue is already running as pid {old}; exiting rather than racing "
                    f"it over the same output files. Watch it with: "
                    f"python run_queue.py --status")
                return False
            if attempt == 1:
                say(f"clearing stale lock from pid {old}")
                LOCK.unlink(missing_ok=True)
    return False


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {"completed": {}, "history": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", default="", help="stages to run again even if complete")
    ap.add_argument("--only", default="", help="run just these stages")
    ap.add_argument("--assume-done", default="", help="mark complete without running")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    st = load_state()
    names = [s[0] for s in STAGES]

    for n in (x.strip() for x in args.assume_done.split(",") if x.strip()):
        st["completed"][n] = {"assumed": True, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for n in (x.strip() for x in args.redo.split(",") if x.strip()):
        st["completed"].pop(n, None)
    if args.assume_done or args.redo:
        save_state(st)

    if args.status:
        print(f"{'stage':7s}{'status':16s}description")
        for name, _, desc, _, art in STAGES:
            done = name in st["completed"]
            mark = "assumed" if done and st["completed"][name].get("assumed") else \
                   ("done" if done else "pending")
            print(f"{name:7s}{mark:16s}{desc}")
        return 0

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only - set(names):
        raise SystemExit(f"unknown stage(s) {only - set(names)}; have {names}")

    if not take_lock():
        return 1

    try:
        t_all = time.time()
        say(f"--- queue started {time.strftime('%Y-%m-%d %H:%M:%S')} as pid {os.getpid()} ---")
        for name, cmd, desc, optional, art in STAGES:
            if only and name not in only:
                continue
            if name in st["completed"] and name not in only:
                say(f"[done] {name}  {desc}")
                continue

            log = LOGS / f"{name}.log"
            say(f"[run ] {name}  {desc}")
            t0 = time.time()
            with log.open("w", encoding="utf-8") as fh:
                # -u matters more than it looks. Without it a stage's stdout is block-buffered
                # into the log file, so a long stage appears to hang with an empty log while it is
                # in fact making progress -- which is exactly how an earlier run went unwatched.
                rc = subprocess.call([PY, "-u"] + cmd, cwd=ROOT,
                                     stdout=fh, stderr=subprocess.STDOUT)
            el = time.time() - t0
            lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            tail = lines[-1] if lines else ""

            # An exit code of zero is not proof on its own: a stage that was meant to write an
            # artefact and did not has failed regardless of what it returned.
            missing = art is not None and not (ROOT / "outputs" / art).exists()
            status = "ok" if rc == 0 and not missing else (
                "no-artefact" if rc == 0 else ("optional-fail" if optional else "FAIL"))

            say(f"[{status:>13s}] {name}  {el/60:.1f} min  | {tail[:90]}")
            st["history"].append({"stage": name, "rc": rc, "minutes": round(el / 60, 1),
                                  "status": status, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
            if rc == 0 and not missing:
                st["completed"][name] = {"minutes": round(el / 60, 1),
                                         "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_state(st)

            if rc != 0 and not optional:
                say(f"queue halted at {name} (exit {rc}); see {log}. "
                    f"Rerun to resume -- completed stages are skipped.")
                return rc
            if missing and not optional:
                say(f"queue halted: {name} exited 0 but did not write outputs/{art}")
                return 1

        say(f"queue complete in {(time.time()-t_all)/60:.1f} min")
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
