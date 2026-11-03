"""Phase 19's look before registering, dev rows only: which feature sets lower cost. Fit on fit rows, rule and lines on the 30 dev windows, read on the 88 held-out chains.

    python -m fraud.dev.explore_features     # prints a table; no test row is read
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import eval_ieee as ev, feature_sets as fs, history as h, ieee_data, tabular as tb
from ..baselines import LogReg

OUT = Path(__file__).resolve().parents[1] / "results" / "eval_features" / "dev_look.json"
SETS = {
    "all": ((), ""),
    "all+steady": (("steady",), ""),
    "all+history": (("history",), ""),
    "all+steady+extra": (("steady", "extra"), ""),
    "all+steady+extra+identity": (("steady", "extra", "identity"), ""),
    "all+steady+extra+identity+inter": (("steady", "extra", "identity", "inter"), ""),
    "all+history+extra+identity+inter": (("history", "extra", "identity", "inter"), ""),
    "all+steady+extra+identity+inter+uid": (("steady", "extra", "identity", "inter", "uid"), ""),
    "noCDV": ((), "CDV"),
    "noCDV+steady+extra+identity+inter": (("steady", "extra", "identity", "inter"), "CDV"),
    "noCDV+history+extra+identity+inter": (("history", "extra", "identity", "inter"), "CDV"),
    "noC+steady+extra+identity+inter": (("steady", "extra", "identity", "inter"), "C"),
}


def main() -> None:
    t = ieee_data.load()
    n = len(t)
    a = ieee_data.split_bounds(n)[0]
    fit_rows = np.arange(a)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    y_held = np.concatenate([t.is_fraud[w.rows] for w in held]).astype(int)
    cache = h.load()
    rows, base = {}, None
    for name, (parts, drop) in SETS.items():
        t0 = time.perf_counter()
        f = fs.FeatureSet(parts, drop, cache)
        m = LogReg().fit(f(t, fit_rows), t.is_fraud[fit_rows].astype(float))
        q = tb.scores_for(m, f, t, dev + held)
        pol = tb.fit_policy(q, t, dev)
        s = [tb.rule_cost(pol, q, t, w) for w in held]
        per = tb.per_window(s)
        good = sum(x.summary["confusion"]["legit"]["decline"] for x in s)
        stopped = sum(x.summary["confusion"]["fraud"]["decline"] + x.summary["confusion"]["fraud"]["review"] for x in s)
        if base is None:
            base = per
        diff, ci = float((per - base).mean()), ev.boot_ci(per - base)
        sum_p = float(sum(pol.p(q[w.rows]).sum() for w in held))
        rows[name] = {"features": int(len(m.w) - 1), "lam": pol.lam, "auc": ev.auc(np.concatenate([q[w.rows] for w in held]), y_held), "mean": float(per.mean()), "diff": diff, "ci": list(ci), "good": good, "stopped": stopped, "sum_p": sum_p, "frauds": int(y_held.sum()),
                      "fit_s": round(time.perf_counter() - t0, 1)}
        print(f"{name:40} f {rows[name]['features']:4d} lam {pol.lam:<4g} auc {rows[name]['auc']:.4f} mean {per.mean():6.0f} diff {diff:6.0f} ({ci[0]:6.0f},{ci[1]:6.0f}) good {good:5d} stopped {stopped:5d} sum p {sum_p:5.0f}/{int(y_held.sum())} {rows[name]['fit_s']:.0f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
