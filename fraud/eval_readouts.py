"""Analyses of the saved FINAL outputs: no model runs, no new scoring → fraud/results/eval_readouts/ (revision Stage 3).

    python -m fraud.eval_readouts     # about 15-30 min on the CPU, most of it the bootstraps and the price grid

Reads only the saved FINAL traces of Phase 23 (fraud/runs/20261104-090649_eval_*_final*.jsonl) and of the GBDT read-out
(20261119-231754), and dev scores (Phase 21's saved LM scores; LR refitted and tab-all reloaded exactly as Phase 23 and the
read-out did) for the price grid's lambda. Results: fraud/results/eval_readouts/<eval id>.json + .md.

REGISTRATION (revision Stage 3; committed before it ran). Every number here is labelled "read-out", except the registered
percentile intervals it recomputes (labelled "held-out (pre-registered)", the read-out setup's "read-out") and the seed noise
("dev"). The registered percentile intervals stay the headline; nothing here replaces them.
- 3.1 MDE for every FINAL comparison: 2.8 x paired SD / sqrt(20) (80% power, two-sided 5%, normal approximation), from the
  FINAL per-window paired differences.
- 3.2 Robustness, every comparison: the registered percentile interval (10,000 resamples, seed 20261102); a moving-block
  bootstrap over windows in time order, blocks of 2 and of 4 (Kunsch 1989; 10,000 resamples, same seed); and BCa (Efron 1987;
  10,000 resamples, jackknife acceleration).
- 3.3 ROC-AUC and PR-AUC (average precision) per setup, with a 95% window-bootstrap percentile interval (10,000 resamples).
- 3.4 Why LM alone ranks like LR but costs more: per setup, a reliability table of the rule's p (Platt-scaled score, 10 equal-
  count bins: mean p against observed fraud rate); cost split by action (fraud let through, good customers declined, review
  fees) and by amount band (<50, 50-200, 200-1,000, >=1,000 USD); and LM alone minus LR with the largest 1 and 2 frauds of the
  20 windows (by amount) removed from both.
- 3.5 Price grid: decline friction r in {5, 10, 20}%, review fee c_r in {5, 10, 25} USD, customer price in {0, 10, 30} USD, 27
  cells. The customer price is a charged cost per good customer declined (costs.Costs.decline_flat, Phase 16's stress), which the
  rule therefore also prices; it is a price assumption, not the decision-only customer line. Review share stays 5%, friction
  minimum 1 USD. Per cell and setup: p as fitted on dev30 (it doesn't depend on prices), lambda re-picked on dev under the cell's
  prices (LR on dev30, the others on all 118 dev windows, as registered), applied to the saved FINAL scores. Setups: LR, LM + tree,
  LM alone, GBDT alone (the customer-line setup is left out: the grid's customer price replaces its knob). Per cell: each setup's
  mean cost, the cheapest, and whether the cheapest is distinguishable from LR by the registered rule. Check: the cell (10%, 10,
  0) must reproduce every setup's saved FINAL cost exactly.
- 3.6 Disparity check on proxies (not a fairness audit): the rate of good customers declined (declined legit / legit) and of
  frauds missed (approved fraud / fraud), with 95% window-bootstrap percentile intervals (10,000 resamples), by card network
  (card4), card type (card6), device type (DeviceType, none recorded as its own group) and purchaser email domain (the 5 most
  common on FINAL, "none", the rest pooled), for every setup.
- 3.7 Seed noise: Phase 21's two-seed dev comparison (lm-seed1 vs lm-phase12, fraud/results/eval_lm_nuc/20261104-090217.json),
  copied with its source, label dev.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, eval_final as ef, eval_gbdt_readout as gr, eval_ieee as ev, eval_ieee_amount as am, eval_lm_nuc as nuc, eval_tabular as et, final_test as ft, history, ieee_data, tabular as tb
from .runs import read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_readouts"
SEED, REPS = ev.SEED, 10_000
P23, GB = ef.SETUPS, gr.SETUP
NAMES = {ef.BASELINE: "LR", ef.SIGNAL_AMT: "LM + tree", ef.SIGNAL_CUST: "LM + tree + customer line", ef.RICH_AMT: "LM alone", GB: "GBDT alone", ev.APPROVE_ALL: "approve all"}
SCORED = (ef.BASELINE, ef.SIGNAL_AMT, ef.SIGNAL_CUST, ef.RICH_AMT, GB)
COMPARISONS = {  # (a, b, label): Phase 23's registered ones, then the read-out's
    "p23_primary": (ef.SIGNAL_AMT, ef.BASELINE, "held-out (pre-registered)"), "p23_second": (ef.SIGNAL_CUST, ef.SIGNAL_AMT, "held-out (pre-registered)"),
    "p23_third": (ef.RICH_AMT, ef.BASELINE, "held-out (pre-registered)"), "p23_cust_vs_lr": (ef.SIGNAL_CUST, ef.BASELINE, "held-out (pre-registered)"),
    "p23_rich_vs_signal": (ef.RICH_AMT, ef.SIGNAL_AMT, "held-out (pre-registered)"), "p23_signal_vs_all": (ef.SIGNAL_AMT, ev.APPROVE_ALL, "held-out (pre-registered)"),
    "ro_primary": (ef.SIGNAL_AMT, GB, "read-out"), "ro_secondary": (GB, ef.BASELINE, "read-out"), "ro_rich_vs_gbdt": (ef.RICH_AMT, GB, "read-out")}
BANDS = (0, 50, 200, 1000, np.inf)
GRID = {"r": (0.05, 0.10, 0.20), "c_r": (5.0, 10.0, 25.0), "customer": (0.0, 10.0, 30.0)}
GRID_SETUPS = (ef.BASELINE, ef.SIGNAL_AMT, ef.RICH_AMT, GB)
DEV_SCORE = {ef.BASELINE: "tab-logreg", ef.SIGNAL_AMT: "lm-signal", ef.RICH_AMT: "lm-rich", GB: "tab-all"}


# ---------------------------------------------------------------- loading

def traces() -> dict[str, list[list[dict]]]:
    t = ieee_data.load()
    w = ft.final_windows(len(t))
    out = {s: tb.read_setup(gr.FINAL_EVAL, s, w) for s in P23}
    out[GB] = tb.read_setup(_readout_id(), GB, w)
    return out


def _readout_id() -> str:
    found = sorted({p.name.split("_")[0] for p in ev.RUNS.glob(f"*_eval_{GB}_final*.jsonl")})
    if len(found) != 1:
        raise SystemExit(f"expected one GBDT read-out, found {found}")
    return found[0]


def policy_p(pol: dict, q: np.ndarray) -> np.ndarray:
    """The rule's p for scores q, from the policy a trace's header saved (tabular.Policy.header)."""
    from .baselines import LogReg

    m = LogReg()
    m.w, m.mu, m.sd = np.array(pol["fit_weights"]), np.array(pol["fit_mean"]), np.array(pol["fit_sd"])
    return m.predict(am.logit(q)[:, None])


def tx(tr: list[list[dict]]) -> list[list[dict]]:
    return [[r for r in recs if r.get("type") == "tx"] for recs in tr]


def arr(tr, key) -> list[np.ndarray]:
    return [np.array([r[key] for r in w]) for w in tx(tr)]


# ---------------------------------------------------------------- bootstraps

def percentile(d: np.ndarray, seed: int = SEED, reps: int = REPS) -> list[float]:
    return list(ev.boot_ci(d, seed, reps))


def block(d: np.ndarray, b: int, seed: int = SEED, reps: int = REPS) -> list[float]:
    """Moving-block bootstrap of the mean: ceil(n/b) blocks of b consecutive windows, cut to n."""
    n = len(d)
    starts = np.arange(n - b + 1)
    k = -(-n // b)
    rng = np.random.default_rng(seed)
    s = rng.choice(starts, (reps, k))
    idx = (s[:, :, None] + np.arange(b)).reshape(reps, -1)[:, :n]
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def bca(d: np.ndarray, seed: int = SEED, reps: int = REPS) -> list[float]:
    from statistics import NormalDist

    nd = NormalDist()
    n = len(d)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, n, (reps, n))].mean(1)
    theta = d.mean()
    z0 = nd.inv_cdf(min(max((m < theta).mean(), 1 / reps), 1 - 1 / reps))
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    u = jack.mean() - jack
    a = (u ** 3).sum() / (6 * ((u ** 2).sum() ** 1.5)) if (u ** 2).sum() else 0.0
    out = []
    for alpha in (0.025, 0.975):
        z = z0 + nd.inv_cdf(alpha)
        out.append(float(np.percentile(m, 100 * nd.cdf(z0 + z / (1 - a * z)))))
    return out


def mde(d: np.ndarray) -> dict:
    sd = float(d.std(ddof=1))
    return {"paired_sd": sd, "mde": 2.8 * sd / np.sqrt(len(d))}


def ap(score: np.ndarray, y: np.ndarray) -> float:
    """Average precision (PR-AUC), ties broken by order of a stable sort on -score."""
    o = np.argsort(-score, kind="stable")
    y = y[o]
    hits = np.cumsum(y)
    return float((hits[y == 1] / (np.flatnonzero(y == 1) + 1)).mean()) if y.sum() else float("nan")


def ranking_ci(qs: list[np.ndarray], ys: list[np.ndarray], seed: int = SEED, reps: int = REPS) -> dict:
    rng = np.random.default_rng(seed)
    n = len(qs)
    a, p = [], []
    for k in rng.integers(0, n, (reps, n)):
        q, y = np.concatenate([qs[j] for j in k]), np.concatenate([ys[j] for j in k])
        a.append(ev.auc(q, y))
        p.append(ap(q, y))
    q, y = np.concatenate(qs), np.concatenate(ys)
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"roc_auc": ev.auc(q, y), "roc_auc_ci": pc(a), "pr_auc": ap(q, y), "pr_auc_ci": pc(p), "base_rate": float(y.mean()), "label": "read-out"}


def ratio_ci(num: np.ndarray, den: np.ndarray, seed: int = SEED, reps: int = REPS) -> dict:
    """A rate pooled over windows (sum num / sum den), with a window-bootstrap interval. num, den: per window."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(num), (reps, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num[k].sum(1) / den[k].sum(1)
    r = r[np.isfinite(r)]
    est = float(num.sum() / den.sum()) if den.sum() else None
    return {"rate": est, "ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))] if len(r) else None, "n": int(den.sum())}


# ---------------------------------------------------------------- sections

def section_comparisons(by: dict) -> dict:
    out = {}
    for k, (a, b, label) in COMPARISONS.items():
        d = by[a] - by[b]
        reg = ev.compare(by, a, b)
        out[k] = {"a": a, "b": b, "label": label, "mean_diff_a_minus_b": float(d.mean()), "registered_percentile": percentile(d), "registered_check": reg["mean_diff_ci95"],
                  "block2": block(d, 2), "block4": block(d, 4), "bca": bca(d), **mde(d), "mde_label": "read-out", "robustness_label": "read-out"}
    return out


def section_diagnostic(tr: dict, t) -> dict:
    out = {}
    for s in SCORED:
        recs = [r for w in tx(tr[s]) for r in w]
        pol = tr[s][0][0].get("policy")
        q = np.array([r["q"] for r in recs])
        y = np.array([r["truth"] for r in recs])
        amt = np.array([r["tx"]["amount"] for r in recs])
        act = np.array([r["action"] for r in recs])
        cost = np.array([r["cost"] for r in recs])
        p = policy_p(pol, q) if pol else None
        rel = []
        if p is not None:
            for b in np.array_split(np.argsort(p, kind="stable"), 10):
                rel.append({"mean_p": float(p[b].mean()), "fraud_rate": float(y[b].mean()), "n": int(len(b))})
        bands = []
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            m = (amt >= lo) & (amt < hi)
            bands.append({"band": [lo, None if hi == np.inf else hi], "rows": int(m.sum()), "frauds": int(y[m].sum()), "cost_per_window": float(cost[m].sum() / 20),
                          "fraud_let_through": float(cost[m & (act == "approve") & (y == 1)].sum() / 20), "good_declined": float(cost[m & (act == "decline") & (y == 0)].sum() / 20),
                          "review_fees": float(cost[m & (act == "review")].sum() / 20)})
        out[s] = {"reliability": rel, "by_action_per_window": {"fraud_let_through": float(cost[(act == "approve") & (y == 1)].sum() / 20), "good_declined": float(cost[(act == "decline") & (y == 0)].sum() / 20),
                                                               "review_fees": float(cost[act == "review"].sum() / 20), "fraud_declined": float(cost[(act == "decline") & (y == 1)].sum() / 20)},
                  "by_amount_band": bands, "label": "read-out"}
    # LM alone minus LR without the largest frauds
    a, b = tx(tr[ef.RICH_AMT]), tx(tr[ef.BASELINE])
    frauds = sorted(((r["tx"]["amount"], wi, ri) for wi, w in enumerate(a) for ri, r in enumerate(w) if r["truth"]), reverse=True)
    drop = {}
    for k in (0, 1, 2):
        gone = {(wi, ri) for _, wi, ri in frauds[:k]}
        ca = np.array([sum(r["cost"] for ri, r in enumerate(w) if (wi, ri) not in gone) for wi, w in enumerate(a)])
        cb = np.array([sum(r["cost"] for ri, r in enumerate(w) if (wi, ri) not in gone) for wi, w in enumerate(b)])
        d = ca - cb
        drop[str(k)] = {"removed_amounts": [float(x[0]) for x in frauds[:k]], "lm_alone_action": [a[wi][ri]["action"] for _, wi, ri in frauds[:k]],
                        "lr_action": [b[wi][ri]["action"] for _, wi, ri in frauds[:k]], "mean_diff": float(d.mean()), "ci": percentile(d), "label": "read-out"}
    out["lm_alone_minus_lr_without_largest_frauds"] = drop
    # Added after the first read (REVISION_LOG.md, Stage 3): the largest frauds by amount were reviewed by both setups, so removing
    # them can't move the gap. Also remove the frauds whose cost differs most between the two (LM alone's cost minus LR's).
    gaps = sorted(((r["cost"] - b[wi][ri]["cost"], wi, ri) for wi, w in enumerate(a) for ri, r in enumerate(w) if r["truth"]), reverse=True)
    by_gap = {}
    for k in (0, 1, 2):
        gone = {(wi, ri) for _, wi, ri in gaps[:k]}
        d = np.array([sum(r["cost"] - b[wi][ri]["cost"] for ri, r in enumerate(w) if (wi, ri) not in gone) for wi, w in enumerate(a)])
        by_gap[str(k)] = {"removed": [{"amount": float(a[wi][ri]["tx"]["amount"]), "gap": float(g), "lm_alone_action": a[wi][ri]["action"], "lr_action": b[wi][ri]["action"]} for g, wi, ri in gaps[:k]],
                          "mean_diff": float(d.mean()), "ci": percentile(d), "label": "read-out"}
    out["lm_alone_minus_lr_without_largest_gap_frauds"] = by_gap
    return out


def dev_scores(t, log=print) -> dict:
    n = len(t)
    ref = nuc.references(t, log)
    return {ef.BASELINE: ref["tab-logreg"], GB: ref["tab-all"], ef.SIGNAL_AMT: nuc.saved("lm-signal", n)[0], ef.RICH_AMT: nuc.saved("lm-rich", n)[0]}


def section_grid(tr: dict, t, log=print) -> dict:
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    q_dev = dev_scores(t, log)
    fits = {s: tb.fit_policy(q_dev[s], t, dev) for s in GRID_SETUPS}
    win = ft.final_windows(n)
    q_fin = {s: arr(tr[s], "q") for s in GRID_SETUPS}
    cells, checks = [], {}
    for r in GRID["r"]:
        for cr in GRID["c_r"]:
            for cu in GRID["customer"]:
                c = costs.Costs(friction_rate=r, friction_min=ev.COSTS.friction_min, review_fee=cr, review_share=ev.COSTS.review_share, decline_flat=cu)
                by, lams = {}, {}
                for s in GRID_SETUPS:
                    pol = fits[s]
                    ws = dev if s == ef.BASELINE else dev + held
                    lam = am.pick_lambda([pol.p(q_dev[s][w.rows]) for w in ws], [t.is_fraud[w.rows].astype(int) for w in ws], [t.amount[w.rows] for w in ws], c)[0]
                    lams[s] = lam
                    per = []
                    for w, q in zip(win, q_fin[s]):
                        wanted, fallback = am.decide(pol.p(q), t.amount[w.rows], lam, c)
                        per.append(costs.score(c, wanted, fallback, t.is_fraud[w.rows].astype(bool), t.amount[w.rows]).summary["cost"])
                    by[s] = np.array(per)
                best = min(GRID_SETUPS, key=lambda s: by[s].sum())
                cmp = ev.compare(by, best, ef.BASELINE) if best != ef.BASELINE else None
                cell = {"r": r, "c_r": cr, "customer": cu, "lambda": lams, "mean_cost": {s: float(by[s].mean()) for s in GRID_SETUPS}, "cheapest": best,
                        "cheapest_vs_lr": cmp, "distinguishable_from_lr": bool(cmp and cmp["distinguishable"]), "label": "read-out"}
                cells.append(cell)
                if (r, cr, cu) == (0.10, 10.0, 0.0):
                    checks = {s: {"grid": float(by[s].mean()), "saved": float(tb.window_costs(tr[s]).mean())} for s in GRID_SETUPS}
                log(f"grid r {r:.2f} c_r {cr:g} cust {cu:g}: cheapest {NAMES[best]}" + (f", vs LR {cmp['mean_diff']:+,.0f} {'distinguishable' if cmp['distinguishable'] else 'n.d.'}" if cmp else ""))
    bad = {s: v for s, v in checks.items() if abs(v["grid"] - v["saved"]) > 1e-6}
    if bad:
        raise SystemExit(f"the base cell doesn't reproduce the saved FINAL costs: {bad}")
    return {"cells": cells, "base_cell_check": checks, "note": "customer price charged per good customer declined (Costs.decline_flat) and priced by the rule; lambda re-picked on dev per cell"}


def section_disparity(tr: dict, t) -> dict:
    groups = {"card4": "card4", "card6": "card6", "DeviceType": "DeviceType", "P_emaildomain": "P_emaildomain"}
    rows = [np.array([r["row"] for r in w]) for w in tx(tr[ef.BASELINE])]
    out = {}
    for gname, col in groups.items():
        vals = [np.array([t.text(col, int(i)) or "none" for i in w]) for w in rows]
        allv = np.concatenate(vals)
        if gname == "P_emaildomain":
            u, c = np.unique(allv[allv != "none"], return_counts=True)
            top = set(u[np.argsort(-c)[:5]])
            vals = [np.array([v if v in top or v == "none" else "other" for v in w]) for w in vals]
            allv = np.concatenate(vals)
        res = {}
        for g in sorted(set(allv)):
            res[g] = {}
            for s in SCORED:
                act, y = arr(tr[s], "action"), arr(tr[s], "truth")
                m = [v == g for v in vals]
                good_dec = np.array([((a == "decline") & (yy == 0) & mm).sum() for a, yy, mm in zip(act, y, m)])
                good = np.array([((yy == 0) & mm).sum() for yy, mm in zip(y, m)])
                missed = np.array([((a == "approve") & (yy == 1) & mm).sum() for a, yy, mm in zip(act, y, m)])
                fraud = np.array([((yy == 1) & mm).sum() for yy, mm in zip(y, m)])
                res[g][s] = {"good_declined": ratio_ci(good_dec, good), "fraud_missed": ratio_ci(missed, fraud), "label": "read-out"}
        out[gname] = res
    return out


def section_seed() -> dict:
    src = ROOT / "results" / "eval_lm_nuc" / "20261104-090217.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    pair = next(p for p in d["verdict"]["among_lm"] if {p["a"], p["b"]} == {"lm-seed1-amt", "lm-phase12-amt"})
    return {"source": "fraud/results/eval_lm_nuc/20261104-090217.json → verdict.among_lm", "comparison": pair, "label": "dev",
            "noise_floor_usd_per_window": 100, "note": "two seeds of the same recipe differ by about 100 USD/window on dev; differences of that size are read as noise"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log = lambda m: print(m, flush=True)
    t = ieee_data.load()
    tr = traces()
    by = {s: tb.window_costs(v) for s, v in tr.items()}
    out = {"eval": stamp(), "script": "fraud/eval_readouts.py", "label": "read-out", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {"phase23": gr.FINAL_EVAL, "gbdt_readout": _readout_id()},
           "names": NAMES, "mean_cost": {s: float(v.mean()) for s, v in by.items()}}
    log("comparisons")
    out["comparisons"] = section_comparisons(by)
    log("ranking")
    out["ranking"] = {s: ranking_ci(arr(tr[s], "q"), arr(tr[s], "truth")) for s in SCORED}
    log("diagnostic")
    out["diagnostic"] = section_diagnostic(tr, t)
    log("disparity")
    out["disparity"] = section_disparity(tr, t)
    out["seed_noise"] = section_seed()
    log("price grid")
    out["price_grid"] = section_grid(tr, t, log)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{out['eval']}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS / f"{out['eval']}.md").write_text(markdown(out), encoding="utf-8")
    log(markdown(out))


def markdown(r: dict) -> str:
    n = r["names"]
    ci = lambda c: f"{c[0]:,.0f} to {c[1]:,.0f}"
    L = [f"# FINAL read-outs {r['eval']}", "", "Saved FINAL outputs only; nothing was scored. Labels as in the JSON; the registered percentile intervals stay the headline.", "",
         "## Comparisons (USD per window, a − b)", "", "| comparison | label | mean | percentile (registered) | block 2 | block 4 | BCa | paired SD | MDE |", "|---|---|---:|---|---|---|---|---:|---:|"]
    for k, c in r["comparisons"].items():
        L.append(f"| {n[c['a']]} − {n[c['b']]} | {c['label']} | {c['mean_diff_a_minus_b']:+,.0f} | {ci(c['registered_percentile'])} | {ci(c['block2'])} | {ci(c['block4'])} | {ci(c['bca'])} | {c['paired_sd']:,.0f} | {c['mde']:,.0f} |")
    L += ["", "## Ranking (read-out)", "", "| setup | ROC-AUC | 95% CI | PR-AUC | 95% CI |", "|---|---:|---|---:|---|"]
    for s, x in r["ranking"].items():
        L.append(f"| {n[s]} | {x['roc_auc']:.3f} | {x['roc_auc_ci'][0]:.3f}–{x['roc_auc_ci'][1]:.3f} | {x['pr_auc']:.3f} | {x['pr_auc_ci'][0]:.3f}–{x['pr_auc_ci'][1]:.3f} |")
    L += ["", "## Cost by action, USD per window (read-out)", "", "| setup | fraud let through | good declined | review fees |", "|---|---:|---:|---:|"]
    for s in SCORED:
        x = r["diagnostic"][s]["by_action_per_window"]
        L.append(f"| {n[s]} | {x['fraud_let_through']:,.0f} | {x['good_declined']:,.0f} | {x['review_fees']:,.0f} |")
    L += ["", "## LM alone − LR without the largest frauds (read-out)", ""]
    for k, x in r["diagnostic"]["lm_alone_minus_lr_without_largest_frauds"].items():
        L.append(f"- {k} removed {x['removed_amounts']} (LM alone {x['lm_alone_action']}, LR {x['lr_action']}): {x['mean_diff']:+,.0f}, {ci(x['ci'])}")
    L += ["", "## LM alone − LR without the frauds it priced worst (read-out)", ""]
    for k, x in r["diagnostic"]["lm_alone_minus_lr_without_largest_gap_frauds"].items():
        L.append(f"- {k} removed {[(e['amount'], e['lm_alone_action'], e['lr_action']) for e in x['removed']]}: {x['mean_diff']:+,.0f}, {ci(x['ci'])}")
    g = r["price_grid"]["cells"]
    L += ["", "## Price grid (read-out)", "", "| r | review fee | customer | cheapest | vs LR | distinguishable |", "|---:|---:|---:|---|---:|---|"]
    for c in g:
        v = c["cheapest_vs_lr"]
        L.append(f"| {c['r']:.0%} | {c['c_r']:g} | {c['customer']:g} | {n[c['cheapest']]} | {(v['mean_diff'] if v else 0):+,.0f} | {'yes' if c['distinguishable_from_lr'] else 'no'} |")
    s = r["seed_noise"]["comparison"]
    L += ["", f"## Seed noise (dev)", "", f"{s['better']} vs {s['other']}: {s['mean_diff']:+,.0f} USD/window, {ci(s['mean_diff_ci95'])}. Noise floor used: about ±100."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
