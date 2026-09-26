"""Deep architectures under the same evaluation, trained here rather than borrowed.

The obvious objection to a paper built on seventeen hand-crafted spectral features is that the
features are the limitation: a learned representation might recover age-independent signal the
spectral family cannot express. This trains reference architectures from scratch, in this
repository, under exactly the protocol the feature arm uses, so the objection is answered with
evidence rather than conceded in a limitations paragraph.

Reference implementations, not reimplementations. The architectures come from braindecode, the
field's maintained source for them. A deep baseline that loses because it was rebuilt badly proves
nothing, and a reviewer has no way to tell a faithful reimplementation from a weakened one.

Both cohorts are scored, because the paper's claim is about recruitment rather than about method:
the training cohort carries a large case-control age gap, and the external clinic is age-matched.
A model that beats age at one and not the other is evidence that the baseline is a property of the
cohort, not of the architecture.

    python experiments/e11_deep_arm.py                       # full run
    python experiments/e11_deep_arm.py --models eegnet --tasks screening --epochs 3
Out: outputs/results/e11_deep_arm.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegbudget import budgets, evaluate as ev, paths, spine  # noqa: E402
from eegbudget.montages import ALL, apply_montage, reorder_to_canon  # noqa: E402

TASKS = {"screening": {"AD", "FTD", "MCI", "VAD"},
         "dementia": {"AD", "FTD", "VAD"},
         "mci": {"MCI"}}
SOURCE, TARGET = "CAUEEG", "ds004504_raw"
SPEC = 0.80          # operating point the transferred threshold is fixed at
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(name, n_chans, n_times):
    from braindecode.models import ATCNet, Deep4Net, EEGNetv4, ShallowFBCSPNet
    kw = dict(n_chans=n_chans, n_outputs=2, n_times=n_times)
    if name == "eegnet":
        return EEGNetv4(**kw)
    if name == "shallow":
        return ShallowFBCSPNet(**kw, final_conv_length="auto")
    if name == "deep4":
        return Deep4Net(**kw, final_conv_length="auto")
    if name == "atcnet":
        return ATCNet(**kw)
    raise ValueError(name)


class _GradRev(torch.autograd.Function):
    """Identity forwards, sign-flipped backwards -- the standard adversarial invariance trick."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class AgeModel(nn.Module):
    """One backbone, five ways of letting age into the decision.

    ``none``        the control: EEG only.
    ``input``       age concatenated to the representation. The obvious thing to do, and the one
                    that transports worst, because it hard-codes "older implies impaired" -- a
                    relationship that reverses at the external clinic.
    ``normative``   age generates a location and scale for the representation and the model scores
                    the deviation. Age cannot move the decision on its own: a subject whose
                    representation equals the age-expected one scores the bias term regardless of
                    how old they are. The pathway is grounded by an auxiliary loss fitted *on
                    controls only*, so it learns what is typical rather than what discriminates.
    ``normative_anchored``
                    the same, with the deviation space re-centred on the target clinic's own
                    healthy volunteers before scoring. Added because plain ``normative`` transports
                    badly: the learned healthy-ageing trajectory belongs to the source cohort's
                    amplifier, montage and population, and is simply the wrong reference elsewhere.
    ``adv``         a gradient-reversed age regressor, training the representation to be
                    age-uninformative. Costs in-distribution accuracy and buys ranking on transfer.

    The empirical ordering is not the intuitive one, and the module exists to measure it rather
    than to assume it. Where the target cohort's age relationship reverses, the best use of age
    may be none at all.
    """

    NORMATIVE = ("normative", "normative_anchored")

    def __init__(self, backbone, feat_dim, variant="none", hidden=32):
        super().__init__()
        self.backbone = backbone
        self.variant = variant
        self.lam = 0.0
        self.feat_dim = feat_dim
        self.head = nn.Linear(feat_dim + (1 if variant == "input" else 0), 2)
        if variant in self.NORMATIVE:
            self.age_mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                         nn.Linear(hidden, 2 * feat_dim))
        if variant == "adv":
            self.age_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 1))
        # Site correction, identity until a target clinic's controls set it.
        self.register_buffer("anchor_shift", torch.zeros(feat_dim))
        self.register_buffer("anchor_scale", torch.ones(feat_dim))

    def deviation(self, x, age):
        h = self.backbone(x).flatten(1)
        mu, log_s = self.age_mlp(age).chunk(2, dim=1)
        z = (h - mu) * torch.exp(-log_s.clamp(-3.0, 3.0))
        return h, mu, z

    def set_anchor(self, z_ctrl):
        """Re-centre the deviation space on a target site's healthy volunteers.

        The normative pathway learns the source cohort's healthy-ageing trajectory. Carried to a
        clinic with a different amplifier, montage and population, that trajectory is simply the
        wrong reference, and the deviations it produces are offset wholesale. Median and MAD over
        the target's own controls remove the offset without any labelled patient -- the same
        correction the feature-space normative model applies, moved inside the network.
        """
        med = z_ctrl.median(dim=0).values
        mad = (z_ctrl - med).abs().median(dim=0).values * 1.4826
        self.anchor_shift.copy_(med)
        self.anchor_scale.copy_(torch.where(mad > 1e-6, mad, torch.ones_like(mad)))

    def forward(self, x, age):
        aux = {}
        if self.variant == "input":
            h = self.backbone(x).flatten(1)
            return self.head(torch.cat([h, age], 1)), aux
        if self.variant in self.NORMATIVE:
            h, mu, z = self.deviation(x, age)
            z = (z - self.anchor_shift) / self.anchor_scale
            aux["mu"], aux["h"], aux["z"] = mu, h, z
            return self.head(z), aux
        h = self.backbone(x).flatten(1)
        if self.variant == "adv":
            aux["age_hat"] = self.age_head(_GradRev.apply(h, self.lam))
        return self.head(h), aux


def make_model(name, variant, n_chans, n_times, device):
    """Backbone with its classifier removed, wrapped in the age variant."""
    bb = build(name, n_chans, n_times)
    bb.final_layer = nn.Identity()
    with torch.no_grad():
        feat_dim = bb(torch.zeros(2, n_chans, n_times)).flatten(1).shape[1]
    return AgeModel(bb, feat_dim, variant).to(device)


def load_windows(df, montage):
    """Stack every subject's analysed windows for one montage. Returns X, subject index, counts."""
    m = ALL[montage]
    xs, owner = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        p = Path(row["npy"])
        meta = json.loads(p.with_suffix(".json").read_text())
        x = reorder_to_canon(np.load(p).astype(np.float32), meta["channels"])
        offs = meta.get("match_offsets")
        if offs:
            w = budgets.WINDOW_SAMPLES
            wins = np.stack([x[:, s:s + w] for s in offs if s + w <= x.shape[-1]])
        else:
            wins, _ = budgets.select_windows(x, subject=row["subject"])
        if wins is None or len(wins) == 0:
            continue
        sig = apply_montage(wins, m)
        # Per-window per-channel standardisation: the amplitude scale differs by cohort and
        # amplifier, and is not what we want the network to key on.
        sig = (sig - sig.mean(-1, keepdims=True)) / (sig.std(-1, keepdims=True) + 1e-8)
        xs.append(sig.astype(np.float32))
        owner.append(np.full(len(sig), i))
    return np.concatenate(xs), np.concatenate(owner)


#: Age is standardised on the training cohort's own distribution, and the same constants are
#: applied at the target site. Re-standardising at the target would hide the very shift under study.
AGE_MU, AGE_SD = 70.0, 10.0

LAMBDA_NORM = 0.1    # weight on the control-grounded normative reconstruction
LAMBDA_ADV = 0.3     # peak weight on the reversed age regressor


def train_eval(name, Xtr, ytr, atr, Xte, ate, epochs, batch, lr, variant="none", seed=0,
               anchor=None):
    """Train one model and return window posteriors for the held-out set.

    `anchor` is an optional (X, age) pair of recordings from people the *target* site already
    knows to be healthy. It is used only to re-centre the deviation space before scoring, never
    for training, and carries no diagnostic label.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(name, variant, Xtr.shape[1], Xtr.shape[2], DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(Xtr) // batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    # Class weights: the window count per class follows the subject count, which is unbalanced
    # on the dementia contrast.
    w = torch.tensor([len(ytr) / max(1, (ytr == c).sum()) for c in (0, 1)],
                     dtype=torch.float32, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum())

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    atr_t = torch.from_numpy(((atr - AGE_MU) / AGE_SD).astype(np.float32)).unsqueeze(1)

    model.train()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t))
        for s in range(0, len(perm), batch):
            ix = perm[s:s + batch]
            xb = Xtr_t[ix].to(DEV)
            yb = ytr_t[ix].to(DEV)
            ab = atr_t[ix].to(DEV)
            # Ramp the adversary in: reversing gradients from step zero destabilises the encoder
            # before it has learned anything worth making invariant.
            model.lam = LAMBDA_ADV * min(1.0, step / max(1, 0.4 * steps))
            opt.zero_grad(set_to_none=True)
            logits, aux = model(xb, ab)
            loss = lossf(logits, yb)
            if variant in AgeModel.NORMATIVE and "mu" in aux:
                # Ground the normative pathway on controls only, so mu(age) learns what is
                # typical for an age rather than what separates the classes.
                ctrl = yb == 0
                if ctrl.any():
                    loss = loss + LAMBDA_NORM * nn.functional.mse_loss(
                        aux["mu"][ctrl], aux["h"][ctrl].detach())
            if variant == "adv" and "age_hat" in aux:
                loss = loss + nn.functional.mse_loss(aux["age_hat"], ab)
            loss.backward()
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            step += 1

    model.eval()
    if anchor is not None and variant in AgeModel.NORMATIVE:
        Xa, aa = anchor
        aa_t = torch.from_numpy(((aa - AGE_MU) / AGE_SD).astype(np.float32)).unsqueeze(1)
        zs = []
        with torch.no_grad():
            for s in range(0, len(Xa), 256):
                _, _, z = model.deviation(torch.from_numpy(Xa[s:s + 256]).to(DEV),
                                          aa_t[s:s + 256].to(DEV))
                zs.append(z)
        if zs:
            model.set_anchor(torch.cat(zs))

    ate_t = torch.from_numpy(((ate - AGE_MU) / AGE_SD).astype(np.float32)).unsqueeze(1)
    out = []
    with torch.no_grad():
        for s in range(0, len(Xte), 256):
            xb = torch.from_numpy(Xte[s:s + 256]).to(DEV)
            ab = ate_t[s:s + 256].to(DEV)
            logits, _ = model(xb, ab)
            out.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def subject_scores(win_scores, owner, n_subjects):
    """Subject score is the mean of its window posteriors, matching the feature arm."""
    s = np.full(n_subjects, np.nan)
    for i in range(n_subjects):
        m = owner == i
        if m.any():
            s[i] = win_scores[m].mean()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="eegnet,shallow,deep4")
    ap.add_argument("--tasks", default="screening,dementia,mci")
    ap.add_argument("--montages", default="b19,b4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--variants", default="none",
                    help="none,input,normative,adv -- how age enters the decision")
    ap.add_argument("--seeds", type=int, default=1,
                    help="independent restarts; scores are averaged and per-seed AUCs kept")
    args = ap.parse_args()

    paths.ensure_dirs()
    df = spine.load()
    src = df[(df["cohort"] == SOURCE) & df["age"].notna() & (df["in_screening"] == 1)]
    tgt = df[(df["cohort"] == TARGET) & df["age"].notna() & (df["in_screening"] == 1)]

    # Merge into any previous run rather than replacing it: the montages are executed as separate
    # invocations so the load-bearing arm finishes first, and a fresh dict would erase the earlier
    # one. Keys are montage|task|model, so a repeated configuration overwrites only itself.
    res = paths.RESULTS / "e11_deep_arm.json"
    out = {"device": DEV, "epochs": args.epochs, "folds": args.folds,
           "seeds": args.seeds, "spec_target": SPEC,
           "source": SOURCE, "target": TARGET, "runs": {}}
    if res.exists():
        try:
            prev = json.loads(res.read_text())
            out["runs"].update(prev.get("runs", {}))
            print(f"merging with {len(out['runs'])} run(s) already on disk")
        except json.JSONDecodeError:
            pass

    for montage in args.montages.split(","):
        print(f"\n########## montage {montage} ##########", flush=True)
        t0 = time.time()
        Xs_all, owner_s = load_windows(src, montage)
        Xt_all, owner_t = load_windows(tgt, montage)
        print(f"  windows: source {Xs_all.shape}, target {Xt_all.shape} "
              f"({time.time()-t0:.0f}s to load)", flush=True)

        for task in args.tasks.split(","):
            pos = TASKS[task]
            s_keep = src["label"].isin(pos | {"CN"}).to_numpy()
            t_keep = tgt["label"].isin(pos | {"CN"}).to_numpy()
            if t_keep.sum() < 20:
                print(f"  [skip] {task}: target has {t_keep.sum()} subjects")
                t_keep = None

            ys = src["label"].isin(pos).to_numpy().astype(int)[s_keep]
            ages = src["age"].to_numpy(float)[s_keep]
            sidx = np.where(s_keep)[0]
            remap = {o: n for n, o in enumerate(sidx)}
            wm = np.isin(owner_s, sidx)
            Xs = Xs_all[wm]
            ow = np.array([remap[o] for o in owner_s[wm]])
            yw = ys[ow]

            auc_age = ev.auc(ys, ages)
            keep_ix = ev.match_on_age(ages, ys)

            aw = ages[ow]                       # window-level age, for the age-aware variants

            # Target-side arrays, prepared once per task rather than per model.
            tgt_pack = None
            if t_keep is not None:
                yt = tgt["label"].isin(pos).to_numpy().astype(int)[t_keep]
                aget = tgt["age"].to_numpy(float)[t_keep]
                tidx = np.where(t_keep)[0]
                tmap = {o: n for n, o in enumerate(tidx)}
                wmt = np.isin(owner_t, tidx)
                ow_t = np.array([tmap[o] for o in owner_t[wmt]])
                tgt_pack = (Xt_all[wmt], aget[ow_t], ow_t, yt, aget)

            for name in args.models.split(","):
                for variant in args.variants.split(","):
                    t1 = time.time()
                    per_seed = []
                    for seed in range(args.seeds):
                        oof = np.full(len(ys), np.nan)
                        skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=seed)
                        for tr, te in skf.split(np.zeros(len(yw)), yw, groups=ow):
                            p = train_eval(name, Xs[tr], yw[tr], aw[tr], Xs[te], aw[te],
                                           args.epochs, args.batch, args.lr, variant, seed)
                            sc = subject_scores(p, ow[te], len(ys))
                            oof = np.where(np.isnan(oof), sc, oof)
                        per_seed.append(oof)

                    oof = np.nanmean(np.stack(per_seed), axis=0)
                    a = ev.auc(ys, oof)
                    a_m = ev.auc(ys[keep_ix], oof[keep_ix])
                    # The comparison the study turns on, tested on identical subjects rather than
                    # by eyeballing two separate intervals.
                    vs_age = ev.delong_test(oof, ages, ys)
                    rec = {"montage": montage, "task": task, "model": name, "variant": variant,
                           "n": int(len(ys)), "seeds": args.seeds,
                           "auc_within": a,
                           "auc_within_ci": list(ev.bootstrap_ci(ys, oof, n=600)),
                           "auc_within_per_seed": [ev.auc(ys, s) for s in per_seed],
                           "age_only_within": auc_age,
                           "age_margin_within": a - auc_age,
                           "vs_age_within": vs_age,
                           "auc_matched": a_m,
                           "rho_matched": ev.substitution_ratio(a, a_m)}

                    if tgt_pack is not None:
                        Xte_t, awt, ow_t, yt, aget = tgt_pack
                        # Half the target's controls calibrate the reference, the other half
                        # measure specificity. Anchoring on the same subjects the estimate is
                        # read from would flatter the result.
                        ctrl_ix = np.where(yt == 0)[0]
                        n_anchor = len(ctrl_ix) // 2
                        anchor_subj = ctrl_ix[:n_anchor]
                        held = np.setdiff1d(ctrl_ix, anchor_subj)
                        anchor = None
                        if variant == "normative_anchored":
                            am = np.isin(ow_t, anchor_subj)
                            anchor = (Xte_t[am], awt[am])
                        st_seeds = []
                        for seed in range(args.seeds):
                            pt = train_eval(name, Xs, yw, aw, Xte_t, awt,
                                            args.epochs, args.batch, args.lr, variant, seed,
                                            anchor=anchor)
                            st_seeds.append(subject_scores(pt, ow_t, len(yt)))
                        st = np.nanmean(np.stack(st_seeds), axis=0)
                        at = ev.auc(yt, st)
                        age_t = ev.auc(yt, aget)
                        # A clinic experiences a threshold, not a ranking. Fix it on the source
                        # cohort at the intended specificity and read F1 at the target.
                        thr = float(np.nanquantile(oof[ys == 0], SPEC))
                        ev_ix = np.concatenate([np.where(yt == 1)[0], held])
                        pred = (st[ev_ix] > thr).astype(int)
                        ye = yt[ev_ix]
                        tp = int(((pred == 1) & (ye == 1)).sum())
                        fp = int(((pred == 1) & (ye == 0)).sum())
                        fn = int(((pred == 0) & (ye == 1)).sum())
                        prec = tp / max(1, tp + fp)
                        recl = tp / max(1, tp + fn)
                        rec.update({
                            "n_target": int(len(yt)), "auc_external": at,
                            "age_only_external": age_t,
                            "age_margin_external": at - age_t,
                            "vs_age_external": ev.delong_test(st, aget, yt),
                            "f1_external": 2 * prec * recl / max(1e-9, prec + recl),
                            "sensitivity_external": recl,
                            "n_anchor_controls": int(n_anchor),
                            "specificity_external": float((st[held] <= thr).mean())})

                    key = f"{montage}|{task}|{name}|{variant}"
                    out["runs"][key] = rec
                    ext = (f" | ext {rec.get('auc_external', float('nan')):.3f} "
                           f"(margin {rec.get('age_margin_external', float('nan')):+.3f}, "
                           f"F1 {rec.get('f1_external', float('nan')):.3f})"
                           if tgt_pack is not None else "")
                    print(f"  {key:36s} within {a:.3f} (margin {a-auc_age:+.3f}, "
                          f"p={vs_age['p']:.1e}){ext}  [{(time.time()-t1)/60:.1f} min]", flush=True)
                    res.write_text(json.dumps(out, indent=2))

        del Xs_all, Xt_all

    print(f"\nwrote {res} with {len(out['runs'])} runs")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
