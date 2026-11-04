"""A read-out, not a test: the customer line on final-rich-amt's recorded FINAL scores → fraud/results/eval_final/rich_cust_readout.json.

    python -m fraud.dev.final_rich_cust <eval id>

Owner, 2026-11-04, after the final test: final-signal-cust is "really not a decision model doing its grunt work", and yes, price the customer line on the d1-Omni-2B setup that reads no tabular score, as a read-out.
No model runs and no row gets a new score. FINAL has been used, this setup was not among its registered candidates, and it was asked for after every FINAL result was seen. So nothing here is a pre-registered answer, and it can't be called distinguishable in the registered sense. CIs are shown by the same bootstrap, for scale.
Rule: final-rich-amt's own fit and lambda, from dev rows, with Phase 18's flat 10 USD in the expected cost of declining, never in what it's charged. Nothing is tuned.
Replayed here with fraud/dev/verify_tabular.py's own arithmetic, apart from the app's rule and pricing code.
Stated before running (Inferred), from dev's 3,459 → 945 and +114: good customers declined fall to about 30% of 1,081, about 300. Frauds stopped fall by about 15%, about 320 of 630. Cost moves within about 150 of final-rich-amt's 2,427, probably up, so it stays above the baseline's 2,106.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .. import costs, eval_final as ef, eval_ieee as ev, final_test as ft, ieee_data, rule, tabular as tb
from . import verify_tabular as vt

SETUP = "final-rich-cust"
TOLD = 10.0


def tx(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("type") == "tx"]


def replay(recs: list[dict], policy: dict) -> tuple[float, list[str]]:
    """Cost and actions by verify_tabular's arithmetic, from the trace alone."""
    head, rows = recs[0], tx(recs)
    c = head["costs"]
    q, a = np.array([r["q"] for r in rows]), np.array([r["tx"]["amount"] for r in rows])
    wanted, fallback = vt.decide_rule(q, a, policy, c)
    left, total, actions = int(np.floor(c["review_share"] * len(q))), 0.0, []
    for w, f, r, amount in zip(wanted, fallback, rows, a):
        act = w
        if act == "review":
            if left > 0:
                left -= 1
            else:
                act = f
        actions.append(act)
        if act == "review":
            total += c["review_fee"]
        elif act == "approve":
            total += amount if r["truth"] else 0.0
        elif not r["truth"]:
            total += max(c["friction_min"], c["friction_rate"] * amount)
    return total, actions


def counts(actions: list[list[str]], truth: list[np.ndarray], amount: list[np.ndarray]) -> dict:
    a, y, m = np.concatenate(actions), np.concatenate(truth).astype(bool), np.concatenate(amount)
    return {"frauds": int(y.sum()), "frauds_stopped": int((y & (a != "approve")).sum()), "fraud_usd_stopped": float(m[y & (a != "approve")].sum()), "fraud_usd_let_through": float(m[y & (a == "approve")].sum()),
            "good_customers_declined": int((~y & (a == "decline")).sum()), "good_declined_or_reviewed": int((~y & (a != "approve")).sum()), "reviews": int((a == "review").sum())}


def main() -> None:
    eval_id = sys.argv[1]
    t = ieee_data.load()
    windows = ft.final_windows(len(t))
    tr = {s: tb.read_setup(eval_id, s, windows) for s in ef.SETUPS}
    head = tr[ef.RICH_AMT][0][0]["policy"]
    pol = ef.policies(t)["lm-rich"]
    if pol.lam != head["lambda"] or not np.allclose(pol.m.w, head["fit_weights"]):
        raise SystemExit("the fit from dev rows is not the one the final test used")
    policy = {**head, "flat": ef.FLAT}
    scored, gap, differ = [], 0.0, 0
    for w, recs in zip(windows, tr[ef.RICH_AMT]):
        rows = tx(recs)
        if [r["row"] for r in rows] != w.rows.tolist():
            raise SystemExit(f"{w.id}: the trace is not this window's rows")
        q = np.array([r["q"] for r in rows])
        wanted, fallback = rule.decide_paced(pol.p(q), t.amount[w.rows], pol.lam, ev.COSTS, 0.0, ef.FLAT)
        s = costs.score(ev.COSTS, wanted, fallback, t.is_fraud[w.rows].astype(bool), t.amount[w.rows])
        scored.append(s)
        total, actions = replay(recs, policy)
        gap = max(gap, abs(total - s.summary["cost"]))
        differ += sum(x != y for x, y in zip(actions, s.final))
    truth = [t.is_fraud[w.rows] for w in windows]
    amount = [t.amount[w.rows] for w in windows]
    by = {s: tb.window_costs(tr[s]) for s in ef.SETUPS}
    by[SETUP] = tb.per_window(scored)
    table = {SETUP: {"mean_cost": float(by[SETUP].mean()), "cost_ci95": list(ev.boot_ci(by[SETUP])), **counts([np.array(s.final) for s in scored], truth, amount)}}
    for s in (ef.RICH_AMT, ef.BASELINE, ef.SIGNAL_CUST):
        table[s] = {"mean_cost": float(by[s].mean()), "cost_ci95": list(ev.boot_ci(by[s])), **counts([np.array([r["action"] for r in tx(recs)]) for recs in tr[s]], truth, amount)}
    told = {s: by[s] + TOLD * np.array([np.sum(~truth[k].astype(bool) & (np.array(a) == "decline")) for k, a in enumerate(acts)]) for s, acts in {SETUP: [x.final for x in scored], ef.BASELINE: [[r["action"] for r in tx(recs)] for recs in tr[ef.BASELINE]]}.items()}
    k = int(np.argmax(by[SETUP]))
    out = {"note": "A read-out after every FINAL result was seen, not a registered answer.", "eval": eval_id, "table": table,
           "against": [ev.compare(by, SETUP, b) for b in (ef.BASELINE, ef.RICH_AMT, ef.SIGNAL_CUST, ev.APPROVE_ALL)] + [ev.compare(by, ef.RICH_AMT, ev.APPROVE_ALL)],
           "windows": [float(v) for v in by[SETUP]], "told_10_usd": {"mean": {s: float(v.mean()) for s, v in told.items()}, "against_baseline": float((told[SETUP] - told[ef.BASELINE]).mean()), "ci": list(ev.boot_ci(told[SETUP] - told[ef.BASELINE]))},
           "dearest_window": {"window": windows[k].id, "cost": float(by[SETUP][k]), "baseline": float(by[ef.BASELINE][k])}, "replay": {"rows_decided_differently": differ, "largest_cost_gap": gap}}
    path = ef.RESULTS / "rich_cust_readout.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
