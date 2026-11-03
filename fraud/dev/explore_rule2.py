"""Phase 18's second look before registering, dev rows only: how the changes combine. Fit on the 30 dev windows, read on the 88 held-out chains.

    python -m fraud.dev.explore_rule2     # prints a table; no test row is read
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import eval_ieee as ev, eval_ieee_amount as am, ieee_data, rule, tabular as tb
from ..baselines import LogReg
from .explore_rule import C, evaluate

OUT = Path(__file__).resolve().parents[1] / "results" / "eval_rule" / "dev_look_combined.json"
GAMMAS = (0.0, 0.5, 1.0, 2.0)
FLATS = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)


def main() -> None:
    t = ieee_data.load()
    n = len(t)
    fit_rows = np.arange(ieee_data.split_bounds(n)[0])
    lr = LogReg().fit(ev.features_all(t, fit_rows), t.is_fraud[fit_rows].astype(float))
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    q = tb.scores_for(lr, ev.features_all, t, dev + held)
    pol = tb.fit_policy(q, t, dev)
    y_dev = np.concatenate([t.is_fraud[w.rows] for w in dev]).astype(float)
    p_of = lambda w: pol.m.predict(am.logit(q[w.rows])[:, None])
    pd, ph = [p_of(w) for w in dev], [p_of(w) for w in held]
    iso = rule.Isotonic().fit(np.concatenate(pd), y_dev)
    x = lambda w: np.c_[am.logit(q[w.rows]), np.log1p(t.amount[w.rows])]
    amt = LogReg().fit(np.concatenate([x(w) for w in dev]), y_dev)
    cals = {"platt": (pd, ph), "isotonic": ([iso.predict(p) for p in pd], [iso.predict(p) for p in ph]), "amount": ([amt.predict(x(w)) for w in dev], [amt.predict(x(w)) for w in held])}
    for k, (a, b) in cals.items():
        print(f"{k}: sum p dev {np.concatenate(a).sum():.0f} of {int(y_dev.sum())}; held {np.concatenate(b).sum():.0f} of {int(sum(t.is_fraud[w.rows].sum() for w in held))}")
    good_lines_dev = sum(tb.lines_cost(pol, q, t, w).summary["confusion"]["legit"]["decline"] for w in dev)
    good_lines_held = sum(tb.lines_cost(pol, q, t, w).summary["confusion"]["legit"]["decline"] for w in held)
    print("lines: good declined dev", good_lines_dev, "held", good_lines_held)
    base = evaluate(ph, held, t, pol.lam)[0]
    rows = []
    for cal, (a, b) in cals.items():
        for g in GAMMAS:
            for f in FLATS:
                best = min(((evaluate(a, dev, t, k, g, f)) + (k,) for k in am.LAMBDAS if k <= 50), key=lambda r: r[0].sum())
                c, good, stopped, reviews = evaluate(b, held, t, best[4], g, f)
                d = c - base
                ci = ev.boot_ci(d)
                rows.append({"cal": cal, "gamma": g, "flat": f, "lam": best[4], "dev_mean": float(best[0].mean()), "dev_good": best[1], "held_mean": float(c.mean()), "held_good": good, "held_stopped": stopped, "held_reviews": reviews, "diff": float(d.mean()), "ci": list(ci)})
        print(cal, "done", flush=True)
    rows.sort(key=lambda r: r["dev_mean"])
    print("Lowest dev cost, then held-out:")
    for r in rows[:12]:
        print(r)
    print("Lowest dev cost among those declining no more good customers on dev than the lines:")
    ok = [r for r in rows if r["dev_good"] <= good_lines_dev]
    for r in ok[:8]:
        print(r)
    print("Lowest dev cost among those declining no more than half the lines' good customers on dev:")
    for r in [r for r in rows if r["dev_good"] <= good_lines_dev / 2][:6]:
        print(r)
    OUT.write_text(json.dumps({"lines_good_dev": good_lines_dev, "lines_good_held": good_lines_held, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
