"""Option-order bias of the trained IEEE-CIS setups, on dev only → fraud/results/eval_order_bias/ (revision Stage 2.1).

    python -m fraud.eval_order_bias --score lm-rich      # both orders, all three option scores, dev30 + 88 held-out chains (~85 min per order)
    python -m fraud.eval_order_bias --score lm-signal
    python -m fraud.eval_order_bias --report             # flip rate, cost per order, and the averaged orders if the rule below calls for it

Scores: fraud/data/ieee-cis/lm_scores/order/<setup>-<order>.npz (git-ignored). Results: fraud/results/eval_order_bias/<eval id>.json + .md.
Traces: fraud/runs/<eval id>_eval_<setup>-<order>-amt_hdev<k>.jsonl.

PRE-REGISTRATION (revision Stage 2.1; committed before any row was scored here).
- Setups: lm-rich (LM alone, adapter 20261103-230750_lora-d1-ieee-full-rich) and lm-signal (LM + tree score, adapter
  20261104-011148_lora-d1-ieee-full-signal), the trained adapters as they are. No retraining.
- Orders: `rev` (A decline, B review, C approve), the order both setups were scored in for every reported number, and `fwd`
  (A approve, B review, C decline), the new one. Training showed each example in a seeded random order. The revision prompt's
  "reversed" is the order these setups already use, so the new order scored is fwd (REVISION_LOG.md, Stage 0).
- Each order scores every dev30 and held-out-dev row once, in batches of 16 as Phase 21, keeping all three option scores.
  Check: rev here must reproduce Phase 21's saved decline score (fraud/data/ieee-cis/lm_scores/<setup>.npz), max gap reported.
- Per order and setup: p fitted on dev30 and lambda on all 118 dev windows, as Phase 21 did for the LM setups; Phase 14's rule;
  read on the 88 held-out chains. Costs as every IEEE-CIS phase since 14.
- Reported: the top-answer flip rate, the share of held-out rows whose argmax option differs between orders; the rule cost of
  each order; the paired window-bootstrap 95% percentile interval (10,000 resamples, seed 20261102) of fwd minus rev; AUC.
- Decision rule (from the revision prompt): if the flip rate is above 10% or the fwd-rev cost interval excludes zero, also
  score the two orders averaged: the mean of each option's score over both orders, decline score as q, refitted the same way,
  and compare it with rev. No new model pass is needed for that.
- Labels: every number here is "dev". The held-out chains are not clean for the LM setups (lambda was picked with them in),
  as the paper already says.
- Stated before running (Inferred): the top answer is "decline" for nearly every row in both orders (as on PaySim), so the
  flip rate is small, but the decline score moves with order, and the cost gap is within about +-150 USD/window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_lm_nuc as nuc, eval_tabular as et, history, ieee_brain, ieee_data, tabular as tb
from .brain import OPTIONS
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_order_bias"
SCORES = nuc.SCORES / "order"
LABEL = "dev"
ADAPTERS = {"lm-rich": ROOT / "models" / "20261103-230750_lora-d1-ieee-full-rich", "lm-signal": ROOT / "models" / "20261104-011148_lora-d1-ieee-full-signal"}
ORDERS = ("rev", "fwd")
FLIP_BAR = 0.10


def score_all(model, t, rows: np.ndarray, variant: str, hist: dict, batch: int = nuc.BATCH, log=print) -> np.ndarray:
    """[rows, 3] option scores in OPTIONS order (approve, review, decline), whatever order they were shown in."""
    import torch

    opts = list(nuc.order_of(variant))
    back = [opts.index(o) for o in OPTIONS]
    seqs = []
    for i in rows:
        d = ieee_brain.decision(t, int(i), hist)
        seqs.append(model.encode(f"{d.context}\n\n{d.state}", d.instruction, opts))
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out = np.empty((len(rows), 3))
    t0 = time.perf_counter()
    for k in range(0, len(order), batch):
        part = order[k:k + batch]
        with torch.no_grad():
            p = torch.softmax(model.logits([seqs[j] for j in part], len(opts)), dim=-1).cpu().numpy()
        out[part] = p[:, back]
        if log and (k // batch) % 500 == 0:
            log(f"{variant}: {k + len(part):,} of {len(order):,} rows, {time.perf_counter() - t0:.0f} s")
    return out


def score(setup: str, log=print) -> None:
    t = ieee_data.load()
    n = len(t)
    rows = np.unique(np.concatenate([w.rows for w in tb.dev30(n) + tb.heldout_dev(n)]))
    hist = history.load()
    if setup == "lm-signal":
        from .lm_signal import load as load_signal

        hist["signal"] = load_signal()
    model = nuc.load_model(ADAPTERS[setup])
    SCORES.mkdir(parents=True, exist_ok=True)
    for variant in ORDERS:
        path = SCORES / f"{setup}-{variant}.npz"
        if path.exists():
            log(f"{path.name} exists, skipped")
            continue
        t0 = time.perf_counter()
        p = score_all(model, t, rows, variant, hist, log=log)
        info = {"setup": setup, "order": variant, "adapter": ADAPTERS[setup].name, "rows": int(len(rows)), "score_s": round(time.perf_counter() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        np.savez(path, rows=rows, p=p, info=np.array(json.dumps(info)))
        log(json.dumps(info))


def load(setup: str, variant: str, n: int) -> np.ndarray:
    with np.load(SCORES / f"{setup}-{variant}.npz") as z:
        p = np.full((n, 3), np.nan)
        p[z["rows"]] = z["p"]
        return p


def fit(q: np.ndarray, t, dev, held) -> tb.Policy:
    pol = tb.fit_policy(q, t, dev)
    pol.lam = et.lambda_on(pol, q, t, dev + held)
    return pol


def report(log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    hrows = np.concatenate([w.rows for w in held])
    y = t.is_fraud[hrows].astype(int)
    eval_id, out = stamp(), {}
    for setup in ADAPTERS:
        p = {v: load(setup, v, n) for v in ORDERS}
        was = nuc.saved(setup, n)[0]
        both = np.unique(np.concatenate([w.rows for w in dev + held]))
        repro = {"max_gap": float(np.abs(p["rev"][both, 2] - was[both]).max()), "mean_gap": float(np.abs(p["rev"][both, 2] - was[both]).mean())}
        top = {v: p[v][hrows].argmax(1) for v in ORDERS}
        flip = float((top["rev"] != top["fwd"]).mean())
        variants = dict(p)
        cost, fits, aucs = {}, {}, {}

        def run(name, probs):
            q = np.clip(probs[:, 2], et.TREE_SCORE_FLOOR, 1 - et.TREE_SCORE_FLOOR)
            pol = fit(q, t, dev, held)
            s = f"{setup}-{name}-amt"
            tb.write_setup(eval_id, t, held, s, q, pol, "rule", "lm")
            cost[name] = tb.window_costs(tb.read_setup(eval_id, s, held))
            fits[name] = {"lambda": pol.lam}
            aucs[name] = ev.auc(q[hrows], y)

        for v in ORDERS:
            run(v, variants[v])
        gap = ev.paired(cost["fwd"], cost["rev"])
        averaged = flip > FLIP_BAR or gap["mean_diff_ci95"][0] > 0 or gap["mean_diff_ci95"][1] < 0
        res = {"flip_rate": flip, "top_answer_share": {v: {o: float((top[v] == k).mean()) for k, o in enumerate(ev.ACTIONS)} for v in ORDERS},
               "rev_reproduces_phase21": repro, "fwd_minus_rev": gap, "averaged_run": averaged}
        if averaged:
            run("avg", (p["rev"] + p["fwd"]) / 2)
            res["avg_minus_rev"] = ev.paired(cost["avg"], cost["rev"])
        res["orders"] = {k: {"mean_cost": float(c.mean()), "auc": aucs[k], **fits[k], "label": LABEL} for k, c in cost.items()}
        res["decline_score_rank_corr"] = nuc.rank_corr(p["rev"][hrows, 2], p["fwd"][hrows, 2])
        out[setup] = res
        log(f"{setup}: flip {flip:.4f}, " + ", ".join(f"{k} {c.mean():,.0f}" for k, c in cost.items()) + f", fwd-rev {gap['mean_diff']:+,.0f} {gap['mean_diff_ci95']}")
    r = {"eval": eval_id, "script": "fraud/eval_order_bias.py", "label": LABEL, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": len(held), "flip_bar": FLIP_BAR, "setups": out}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    L = [f"# Option-order bias, dev {eval_id}", "", f"88 held-out dev chains. Label: dev. Flip bar {FLIP_BAR:.0%}. Model scores are uncalibrated.", "",
         "| setup | flip rate | rev cost | fwd cost | fwd - rev | 95% CI | avg cost | avg - rev | 95% CI |", "|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for s, x in out.items():
        g, a = x["fwd_minus_rev"], x.get("avg_minus_rev")
        L.append(f"| {s} | {x['flip_rate']:.2%} | {x['orders']['rev']['mean_cost']:,.0f} | {x['orders']['fwd']['mean_cost']:,.0f} | {g['mean_diff']:+,.0f} | {g['mean_diff_ci95'][0]:,.0f} to {g['mean_diff_ci95'][1]:,.0f} | "
                 + (f"{x['orders']['avg']['mean_cost']:,.0f} | {a['mean_diff']:+,.0f} | {a['mean_diff_ci95'][0]:,.0f} to {a['mean_diff_ci95'][1]:,.0f} |" if a else "not run | | |"))
    (RESULTS / f"{eval_id}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))
    return r


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--score", choices=tuple(ADAPTERS))
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.score:
        score(a.score, log)
    else:
        report(log)


if __name__ == "__main__":
    main()
