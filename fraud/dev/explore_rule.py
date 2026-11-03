"""Phase 18's look before registering, dev rows only: what calibration, pacing and a price on declines do. Fit on the 30 dev windows, read on the 88 held-out dev chains.

    python -m fraud.dev.explore_rule      # prints a table; no test row is read
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import costs, eval_ieee as ev, eval_ieee_amount as am, ieee_data, rule, tabular as tb
from ..baselines import LogReg

OUT = Path(__file__).resolve().parents[1] / "results" / "eval_rule" / "dev_look.json"
C = ev.COSTS
RECENT = 5000


def evaluate(ps, windows, t, lam, gamma=0.0, flat=0.0):
    cost, good, stopped, reviews = [], 0, 0, 0
    for p, w in zip(ps, windows):
        a, y = t.amount[w.rows], t.is_fraud[w.rows].astype(bool)
        s = costs.score(C, *rule.decide_paced(p, a, lam, C, gamma, flat), y, a).summary
        cost.append(s["cost"]),
        good += s["confusion"]["legit"]["decline"]
        stopped += s["confusion"]["fraud"]["decline"] + s["confusion"]["fraud"]["review"]
        reviews += s["reviews_done"]
    return np.array(cost), good, stopped, reviews


def main() -> None:
    t = ieee_data.load()
    n = len(t)
    fit_rows = np.arange(ieee_data.split_bounds(n)[0])
    lr = LogReg().fit(ev.features_all(t, fit_rows), t.is_fraud[fit_rows].astype(float))
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    lo, hi = ieee_data.split_rows(n, "dev")
    span = ev.Win("span", "dev", np.arange(lo - RECENT, hi))
    q = tb.scores_for(lr, ev.features_all, t, [span])
    # Phase 14's fit on dev30
    pol = tb.fit_policy(q, t, dev)
    p_of = lambda w, m=pol.m: m.predict(am.logit(q[w.rows])[:, None])
    y_dev = np.concatenate([t.is_fraud[w.rows] for w in dev]).astype(float)
    y_held = np.concatenate([t.is_fraud[w.rows] for w in held]).astype(float)
    base_prior = float(y_dev.mean())
    pd, ph = [p_of(w) for w in dev], [p_of(w) for w in held]
    out = {"dev_fraud_rate": base_prior, "held_fraud_rate": float(y_held.mean()), "sum_p_held": float(np.concatenate(ph).sum()), "frauds_held": int(y_held.sum()), "phase14_lambda": pol.lam}
    print(json.dumps(out))
    rows = {}

    def add(name, ps_dev, ps_held, gamma=0.0, flat=0.0, lam=None):
        if lam is None:
            grid = [(k, evaluate(ps_dev, dev, t, k, gamma, flat)[0].sum()) for k in am.LAMBDAS]
            lam = min(grid, key=lambda g: g[1])[0]
        c, good, stopped, reviews = evaluate(ps_held, held, t, lam, gamma, flat)
        rows[name] = {"lam": lam, "gamma": gamma, "flat": flat, "mean": float(c.mean()), "good": good, "stopped": stopped, "reviews": reviews, "per": c}
        b = rows.get("phase14")
        d = (c - b["per"]) if b else c * 0
        ci = ev.boot_ci(d) if b else (0, 0)
        print(f"{name:32} lam {lam:<5g} g {gamma:<3g} f {flat:<4g} mean {c.mean():7.0f} diff {d.mean():6.0f} ({ci[0]:6.0f},{ci[1]:6.0f}) good {good:5d} stopped {stopped:5d} reviews {reviews:5d}", flush=True)

    add("phase14", pd, ph, lam=pol.lam)
    # Calibration
    iso = rule.Isotonic().fit(np.concatenate(pd), y_dev)
    add("isotonic", [iso.predict(p) for p in pd], [iso.predict(p) for p in ph])

    def amount_fit():
        x = lambda w: np.c_[am.logit(q[w.rows]), np.log1p(t.amount[w.rows])]
        m = LogReg().fit(np.concatenate([x(w) for w in dev]), y_dev)
        return [m.predict(x(w)) for w in dev], [m.predict(x(w)) for w in held]

    add("logit + log amount fit", *amount_fit())

    def recent(w):
        first = int(w.rows[0])
        return rule.recent_prior(pol.p(q[first - RECENT:first]), base_prior)

    pri_d, pri_h = [recent(w) for w in dev], [recent(w) for w in held]
    print("recent prior, dev mean", round(float(np.mean(pri_d)), 4), "held mean", round(float(np.mean(pri_h)), 4), "held truth", round(float(y_held.mean()), 4), "held corr", round(float(np.corrcoef(pri_h, [t.is_fraud[w.rows].mean() for w in held])[0, 1]), 3))
    add("prior shift", [rule.shift_p(p, base_prior, k) for p, k in zip(pd, pri_d)], [rule.shift_p(p, base_prior, k) for p, k in zip(ph, pri_h)])
    # Pacing: lambda and gamma picked on dev30
    for g in (0.5, 1.0, 2.0, 4.0):
        add(f"paced g={g}", pd, ph, gamma=g)
    # A price on declining, customers against dollars
    for f in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        add(f"flat f={f}", pd, ph, flat=f)
    tr = tb.per_window([tb.lines_cost(pol, q, t, w) for w in held])
    good_lines = sum(tb.lines_cost(pol, q, t, w).summary["confusion"]["legit"]["decline"] for w in held)
    print("two lines on held-out: mean", round(float(tr.mean())), "good declined", good_lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out["rows"] = {k: {x: (v[x] if x != "per" else None) for x in v if x != "per"} for k, v in rows.items()}
    out["lines"] = {"mean": float(tr.mean()), "good": good_lines}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
