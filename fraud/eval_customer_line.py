"""The customer line as a trade-off curve, on dev only → fraud/results/eval_customer_line/ (revision Stage 2.4).

    python -m fraud.eval_customer_line     # a few minutes on the CPU; no model pass, existing dev scores only

PRE-REGISTRATION (revision Stage 2.4; committed before it ran).
- What the customer line is: Phase 18's flat price added to the expected cost of declining a payment, in the decision only.
  It is never charged. At a flat 10 USD it is FINAL's final-signal-cust.
- Setups: LR (tab-logreg, Phase 20's logreg on all columns, refitted on fit rows as every phase does) and LM + tree (lm-signal,
  Phase 21's saved dev scores). Both use the policy they carry on FINAL: p fitted on dev30; lambda from dev30 for LR, and
  from all 118 dev windows for LM + tree. The policy is fixed across prices. Only the flat price moves, as on FINAL, where the
  10 USD line kept lm-signal's fit.
- Prices: 0 to 60 USD, step 5 (13 points). Read on the 88 held-out dev chains.
- Per point: frauds stopped (declined or reviewed), good customers declined, total charged cost per window, and the paired
  window-bootstrap 95% interval (10,000 resamples, seed 20261102) of the cost against the same setup at 0 USD.
- Label: dev. LM + tree's held-out read is flattered (lambda was picked with these chains in).
- Stated before running (Inferred): as the price rises, good customers declined fall steeply, frauds stopped fall slowly,
  and cost rises a little. A trade-off, not a free gain.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, eval_ieee as ev, eval_lm_nuc as nuc, eval_tabular as et, history, ieee_data, rule, tabular as tb
from .runs import stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_customer_line"
LABEL = "dev"
PRICES = tuple(range(0, 61, 5))


def scores(t, log=print) -> dict:
    n = len(t)
    windows = tb.dev30(n) + tb.heldout_dev(n)
    kind, parts, drop = et.SETS["tab-logreg"]
    lr = et.score(*et.fit("tab-logreg", kind, parts, drop, t, history.load(), log), t, windows)
    return {"logreg": lr, "lm-signal": nuc.saved("lm-signal", n)[0]}


def curve(q: np.ndarray, pol: tb.Policy, t, held) -> dict:
    out = {}
    base = None
    for flat in PRICES:
        per, stopped, declined, frauds = [], 0, 0, 0
        for w in held:
            truth = t.is_fraud[w.rows].astype(bool)
            wanted, fallback = rule.decide_paced(pol.p(q[w.rows]), t.amount[w.rows], pol.lam, ev.COSTS, 0.0, float(flat))
            s = costs.score(ev.COSTS, wanted, fallback, truth, t.amount[w.rows])
            act = np.asarray(s.final)
            per.append(s.summary["cost"])
            stopped += int((truth & (act != "approve")).sum())
            declined += int((~truth & (act == "decline")).sum())
            frauds += int(truth.sum())
        per = np.array(per)
        base = per if base is None else base
        out[str(flat)] = {"price": flat, "mean_cost": float(per.mean()), "frauds_stopped": stopped, "frauds": frauds, "good_customers_declined": declined,
                          "vs_price_0": ev.paired(per, base) if flat else None, "label": LABEL}
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log = lambda m: print(m, flush=True)
    t = ieee_data.load()
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    res = {}
    for name, q in scores(t, log).items():
        pol = tb.fit_policy(q, t, dev)
        if name != "logreg":
            pol.lam = et.lambda_on(pol, q, t, dev + held)
        res[name] = {"lambda": pol.lam, "curve": curve(q, pol, t, held)}
        log(f"{name}: lambda {pol.lam}")
    eval_id = stamp()
    r = {"eval": eval_id, "script": "fraud/eval_customer_line.py", "label": LABEL, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": len(held), "prices": list(PRICES), "setups": res}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    L = [f"# Customer line trade-off, dev {eval_id}", "", "88 held-out dev chains. Label: dev. The price is in the decision only, never charged.", "",
         "| setup | price | mean cost | vs 0 USD | 95% CI | frauds stopped | good declined |", "|---|---:|---:|---:|---|---:|---:|"]
    for name, x in res.items():
        for k, c in x["curve"].items():
            v = c["vs_price_0"]
            L.append(f"| {name} | {c['price']} | {c['mean_cost']:,.0f} | " + (f"{v['mean_diff']:+,.0f} | {v['mean_diff_ci95'][0]:,.0f} to {v['mean_diff_ci95'][1]:,.0f}" if v else "| ") + f" | {c['frauds_stopped']} of {c['frauds']} | {c['good_customers_declined']} |")
    (RESULTS / f"{eval_id}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))


if __name__ == "__main__":
    main()
