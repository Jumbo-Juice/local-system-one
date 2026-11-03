"""Phase 20's look before registering, dev rows only: boosted trees by feature set and size. Fit on fit rows, rule and lines on the 30 dev windows, read on the 88 held-out chains.

    python -m fraud.dev.explore_gbdt           # prints a table and writes dev_look.json as it goes; no test row is read
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import eval_ieee as ev, feature_sets as fs, history as h, ieee_data, tabular as tb
from ..gbdt import GBDT, Params

OUT = Path(__file__).resolve().parents[1] / "results" / "eval_tabular" / "dev_look.json"
SETS = {"all": ((), ""), "rich": (("steady", "extra", "identity"), ""), "rich+uid": (("steady", "extra", "identity", "uid"), ""), "noCDV+rich": (("steady", "extra", "identity"), "CDV"), "noC+rich": (("steady", "extra", "identity"), "C"), "noV+rich": (("steady", "extra", "identity"), "V")}
TREES = (100, 200, 300, 400, 600, 800)


def read(name: str, q_of, t, dev, held, y_held, base) -> list[dict]:
    rows = []
    for k in TREES:
        q = q_of(k)
        pol = tb.fit_policy(q, t, dev)
        s = [tb.rule_cost(pol, q, t, w) for w in held]
        per = tb.per_window(s)
        good = sum(x.summary["confusion"]["legit"]["decline"] for x in s)
        stopped = sum(x.summary["confusion"]["fraud"]["decline"] + x.summary["confusion"]["fraud"]["review"] for x in s)
        d = per - base
        ci = ev.boot_ci(d)
        row = {"set": name, "trees": k, "lam": pol.lam, "auc": ev.auc(np.concatenate([q[w.rows] for w in held]), y_held), "mean": float(per.mean()), "diff": float(d.mean()), "ci": list(ci), "good": good, "stopped": stopped,
               "lines_mean": float(tb.per_window([tb.lines_cost(pol, q, t, w) for w in held]).mean()), "sum_p": float(sum(pol.p(q[w.rows]).sum() for w in held)), "frauds": int(y_held.sum())}
        print(f"{name:12} trees {k:4d} lam {pol.lam:<4g} auc {row['auc']:.4f} mean {row['mean']:6.0f} diff {row['diff']:6.0f} ({ci[0]:6.0f},{ci[1]:6.0f}) good {good:5d} stopped {stopped:5d} lines {row['lines_mean']:6.0f} sum p {row['sum_p']:5.0f}/{row['frauds']}", flush=True)
        rows.append(row)
    return rows


def main() -> None:
    names = sys.argv[1:] or list(SETS)
    t = ieee_data.load()
    n = len(t)
    a = ieee_data.split_bounds(n)[0]
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    y_held = np.concatenate([t.is_fraud[w.rows] for w in held]).astype(int)
    from .. import eval_column_audit as au
    from ..baselines import LogReg

    base_f = au.features_without("")
    lr = LogReg().fit(base_f(t, np.arange(a)), t.is_fraud[:a].astype(float))
    qb = tb.scores_for(lr, base_f, t, dev + held)
    pol = tb.fit_policy(qb, t, dev)
    base = tb.per_window([tb.rule_cost(pol, qb, t, w) for w in held])
    print(f"logreg all columns, rule, held-out: {base.mean():.0f}", flush=True)
    cache = h.load()
    all_rows = np.unique(np.concatenate([w.rows for w in dev + held]))
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names:
        parts, drop = SETS[name]
        t0 = time.perf_counter()
        f = fs.FeatureSet(parts, drop, cache)
        x = f(t, np.arange(a)).astype(np.float32)
        m = GBDT(Params(trees=max(TREES), depth=6, lr=0.05)).fit(x, t.is_fraud[:a].astype(np.float64))
        del x
        xv = f(t, all_rows).astype(np.float32)
        q = np.full(n, np.nan)

        def q_of(k):
            q[all_rows] = np.clip(m.predict(xv, k), 1e-6, 1 - 1e-6)
            return q

        out[name] = {"rows": read(name, q_of, t, dev, held, y_held, base), "fit_s": round(time.perf_counter() - t0, 1), "features": x.shape[1] if False else xv.shape[1]}
        print(f"{name} done in {out[name]['fit_s']:.0f} s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
