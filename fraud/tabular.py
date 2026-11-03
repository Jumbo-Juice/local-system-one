"""What Phases 17-20 share: score rows with a tabular model, fit Phase 14's rule and lines on dev, write traces, read them back.

The rule is Phase 14's (fraud/eval_ieee_amount.py). Windows are fraud/final_test.py's. Nothing here reads a label of a row outside the windows it is given, and `guard` refuses FINAL rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import costs, eval_ieee as ev, eval_ieee_amount as am, final_test as ft, ieee_data
from .baselines import LogReg
from .runs import read_run


def dev30(n: int) -> list[ev.Win]:
    """Phase 14's dev windows: what its rule and lines were fitted on."""
    return am.old_dev(n) + am.extra_dev(n)


def heldout_dev(n: int) -> list[ev.Win]:
    """Every dev row outside dev30, in arrival order, cut into chains of 1,000. Never used to fit a rule or a line, so a second read with many more windows."""
    lo, hi = ieee_data.split_rows(n, "dev")
    used = np.concatenate([w.rows for w in dev30(n)])
    rest = np.setdiff1d(np.arange(lo, hi), used)
    return [ev.Win(f"hdev{k}", "dev", rest[k * ev.WINDOW_SIZE:(k + 1) * ev.WINDOW_SIZE], ft.SEED) for k in range(len(rest) // ev.WINDOW_SIZE)]


@dataclass
class Policy:
    m: LogReg
    lam: float
    review: float | None
    decline: float | None

    def header(self) -> dict:
        return {"kind": "amount-aware", "lambda": self.lam, "fit_weights": self.m.w.tolist(), "fit_mean": self.m.mu.tolist(), "fit_sd": self.m.sd.tolist()}

    def p(self, q: np.ndarray) -> np.ndarray:
        return self.m.predict(am.logit(q)[:, None])


def fit_policy(q: np.ndarray, t, windows: list[ev.Win], c: costs.Costs = ev.COSTS) -> Policy:
    """As Phase 14: p from a logistic fit of the score's logit, lambda and the two lines from the lowest total cost, all on these windows. q holds a score per row of t."""
    scores = [q[w.rows] for w in windows]
    truth = [t.is_fraud[w.rows].astype(int) for w in windows]
    amount = [t.amount[w.rows] for w in windows]
    x = [am.logit(s)[:, None] for s in scores]
    m = LogReg().fit(np.concatenate(x), np.concatenate(truth).astype(float))
    lam = am.pick_lambda([m.predict(k) for k in x], truth, amount, c)[0]
    tr, td, _ = ev.pick_thresholds(scores, truth, amount, c)
    return Policy(m, lam, tr, td)


def scores_for(model, features, t, windows: list[ev.Win]) -> np.ndarray:
    """A score per row of t, NaN outside these windows. Scored in blocks to keep memory down."""
    q = np.full(len(t), np.nan)
    rows = np.unique(np.concatenate([w.rows for w in windows]))
    for k in range(0, len(rows), 20000):
        part = rows[k:k + 20000]
        q[part] = model.predict(features(t, part))
    return q


def rule_cost(policy: Policy, q: np.ndarray, t, w: ev.Win, c: costs.Costs = ev.COSTS) -> costs.Scored:
    wanted, fallback = am.decide(policy.p(q[w.rows]), t.amount[w.rows], policy.lam, c)
    return costs.score(c, wanted, fallback, t.is_fraud[w.rows].astype(bool), t.amount[w.rows])


def lines_cost(policy: Policy, q: np.ndarray, t, w: ev.Win, c: costs.Costs = ev.COSTS) -> costs.Scored:
    wanted = [ev.act(float(v), policy.review, policy.decline) for v in q[w.rows]]
    return costs.score(c, wanted, ["approve"] * len(wanted), t.is_fraud[w.rows].astype(bool), t.amount[w.rows])


def write_setup(eval_id: str, t, windows: list[ev.Win], setup: str, q: np.ndarray, policy: Policy, kind: str) -> None:
    """A trace per window, kind "rule" or "lines". The trace holds the score, so the decisions can be replayed from it."""
    ft.guard(len(t), np.concatenate([w.rows for w in windows]))
    for w in windows:
        outs = [{"by": "logreg", "q": float(v), "ms": 0.0} for v in q[w.rows]]
        if kind == "rule":
            am.write_trace(eval_id, t, w, setup, outs, None, am.decide(policy.p(q[w.rows]), t.amount[w.rows], policy.lam, ev.COSTS), policy.header(), {})
        else:
            am.write_trace(eval_id, t, w, setup, outs, (policy.review, policy.decline), None, None, {})


def read_setup(eval_id: str, setup: str, windows: list[ev.Win]) -> list[list[dict]]:
    return [read_run(ev.trace_file(eval_id, "eval", setup, w.id)) for w in windows]


def customers(traces: list[list[dict]]) -> dict:
    """Counts a business reads, from the traces."""
    tx = [r for recs in traces for r in recs if r.get("type") == "tx"]
    fraud = [r for r in tx if r["truth"]]
    return {"frauds": len(fraud), "frauds_stopped": sum(r["action"] != "approve" for r in fraud), "fraud_usd": float(sum(r["tx"]["amount"] for r in fraud)),
            "fraud_usd_stopped": float(sum(r["tx"]["amount"] for r in fraud if r["action"] != "approve")), "good_customers_declined": sum(r["action"] == "decline" and not r["truth"] for r in tx),
            "good_reviewed": sum(r["action"] == "review" and not r["truth"] for r in tx)}


def row(traces: list[list[dict]], scored: bool = True) -> dict:
    v = ev.view(traces)
    return {**ev.setup_row(traces, v, scored), **am.split_money(traces), **customers(traces)}


def window_costs(traces: list[list[dict]]) -> np.ndarray:
    return np.array([r[-1]["summary"]["cost"] for r in traces])


def per_window(scored: list[costs.Scored]) -> np.ndarray:
    return np.array([s.summary["cost"] for s in scored])
