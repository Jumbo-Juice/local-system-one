"""Does a stronger tabular model lower the cost? Numpy boosted trees on IEEE-CIS, 10 new windows → fraud/results/eval_tabular/.

    python -m fraud.history                          # the feature cache, if it isn't there yet
    python -m fraud.eval_tabular                     # fit, dev, 10 new windows, report; about an hour
    python -m fraud.eval_tabular --report <eval id>  # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_eval_<setup>_tabular<k>.jsonl. Models: fraud/models/tabular_<name>.pkl (git-ignored). Results: fraud/results/eval_tabular/<eval id>.json + .md, and the dev look dev_look.json.

PRE-REGISTRATION (Phase 20; committed before any setup here scored a new window).
Owner, 2026-11-03: "stronger tabular model. A numpy-only option needs no approval." Then, while this phase was being prepared: the final setup must be a language decision model at heart. So these trees are not a candidate to deploy on their own. They are the reference that prices that choice, and a possible signal for the language model (Phase 21).
- Model: fraud/gbdt.py, histogram boosting on the logistic loss, numpy only: 800 trees, depth 6, learning rate 0.05, row sample 0.5, column sample 0.3, seed 0. Fit rows only. Card and address ids stay out, as in every setup since Phase 11.
- Observed before registering, dev rows only (dev_look.json): fit on fit rows, rule and lines on Phase 14's 30 dev windows, read on 88 held-out chains. Logreg on all columns: AUC 0.888, 2,125, 4,003 good customers declined. Trees at 800: all columns AUC 0.914, 1,830, -295, CI -411 to -173, 3,003 declined; rich (steady, extra, identity) AUC 0.916; without C, D, V plus rich AUC 0.863, 2,494; without C plus rich AUC 0.902, 2,147. Which number of trees and which feature set won moved by less than its interval, so nothing finer than these sets is chosen.
  One thing did not hold still: rich at 800 trees picked lambda 1 on the 30 windows and cost 2,099, where 600 trees picked 5 and cost 1,888. So lambda on 30 windows is unstable for trees.
- Changed from Phase 14 because of that, tree setups only (Implementation choice, from dev): lambda is picked on all 118 dev windows, from am.LAMBDAS. p's fit and the two lines stay on the 30. Held-out dev is then no longer held out for lambda, so its read is flattered and reported as such. Logreg setups keep Phase 14's procedure, so the baseline doesn't move.
- Setups: tab-logreg (ieee-logreg-all); tab-all (trees, all columns); tab-rich (trees, all columns plus steady, extra and identity); tab-logreg-nocdv; tab-nocdv (trees, without C, D, V, plus steady, extra and identity). Each with the rule and with two lines. tab-all-cust: tab-all's scores and rule with Phase 18's flat 10 USD in the expected cost of declining, never in what it's charged. ieee-approve-all.
- New windows: chains tabular0-tabular9 from fraud/final_test.py, 10,000 rows, the last of the 38. Each runs once per setup. Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed.
- Primary question: tab-all against tab-logreg, both with the rule, on total cost. Distinguishable only if the 95% bootstrap CI of the mean paired difference (windows resampled, seed 20261102) excludes 0.
- Second: tab-rich against tab-all. Third: tab-nocdv against tab-logreg-nocdv, what trees give back if the doubt is real. Fourth: tab-all-cust against tab-all, on cost and on good customers declined. Not corrected for multiple comparisons beyond the primary. All reported in customers as well as dollars.
- Limits stated now: 10 windows, heavy tail, and these chains are the last unused rows before FINAL. These numbers don't decide what's deployed: the owner's constraint does. Trees read C, whose timing Phase 17 could not clear. Scoring speed isn't compared with a language model's here.
- A rerun, stated before any result of this phase was read: the first run, eval 20261103-231949, stopped on seeing its held-out dev read of 2,682 for tab-all, against 1,830 in the dev look. Cause (Observed): scoring passed float64 to trees whose bin edges came from float32, so values on an edge moved bin and every row's score changed (AUC 0.934 to 0.887 on 20 chains). It wrote traces for tab-logreg and tab-all on the new windows. Nobody read them, and they were deleted. The fix is in fraud/gbdt.py with a test. Same setups, windows and questions; the fitted tab-all model is kept. This reruns on the same chains because no result prompted the change: a bug seen on dev did.
- Stated before running (Inferred): tab-all has a lower mean than tab-logreg by less than dev's 295, and may be distinguishable on 10 windows. tab-rich about ties tab-all. tab-nocdv is lower than tab-logreg-nocdv by about 250. tab-all-cust costs a little more than tab-all and declines about a third of the good customers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_ieee_amount as am, feature_sets as fs, final_test as ft, history as h, ieee_data, rule, tabular as tb
from .baselines import LogReg
from .gbdt import GBDT, Params
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_tabular"
MODELS = ROOT / "models"
RICH = ("steady", "extra", "identity")
PARAMS = Params(trees=800, depth=6, lr=0.05)
SETS = {"tab-logreg": ("logreg", (), ""), "tab-all": ("trees", (), ""), "tab-rich": ("trees", RICH, ""), "tab-logreg-nocdv": ("logreg", (), "CDV"), "tab-nocdv": ("trees", RICH, "CDV")}
CUST, FLAT = "tab-all-cust", 10.0
TREE_SCORE_FLOOR = 1e-6


def fit(name: str, kind: str, parts, drop: str, t, cache, log=print):
    """(predict on rows, the feature set). A tree model is saved, so later phases can score with it without refitting."""
    a = ieee_data.split_bounds(len(t))[0]
    f = fs.FeatureSet(parts, drop, cache)
    t0 = time.perf_counter()
    saved = MODELS / f"tabular_{name}.pkl"
    if kind == "logreg":
        m = LogReg().fit(f(t, np.arange(a)), t.is_fraud[:a].astype(float))
    elif saved.exists():  # the same seed gives the same model, so a finished fit needn't run again
        m = pickle.loads(saved.read_bytes())
    else:
        m = GBDT(PARAMS).fit(f(t, np.arange(a)).astype(np.float32), t.is_fraud[:a].astype(np.float64))
        MODELS.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(pickle.dumps(m))
    log(f"{name} fitted in {time.perf_counter() - t0:.0f} s")
    return m, f


def score(m, f, t, windows) -> np.ndarray:
    return np.clip(tb.scores_for(m, f, t, windows), TREE_SCORE_FLOOR, 1 - TREE_SCORE_FLOOR)


def lambda_on(pol: tb.Policy, q: np.ndarray, t, windows) -> float:
    """Lambda from these windows, p's fit unchanged."""
    return am.pick_lambda([pol.p(q[w.rows]) for w in windows], [t.is_fraud[w.rows].astype(int) for w in windows], [t.amount[w.rows] for w in windows], ev.COSTS)[0]


def run(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    dev, held, new = tb.dev30(n), tb.heldout_dev(n), ft.phase_windows(n, "tabular")
    ft.guard(n, np.concatenate([w.rows for w in new]))
    y_held = np.concatenate([t.is_fraud[w.rows] for w in held]).astype(int)
    cache = h.load()
    info, heldout = {}, {}
    for name, (kind, parts, drop) in SETS.items():
        m, f = fit(name, kind, parts, drop, t, cache, log)
        q = score(m, f, t, dev + held + new)
        pol = tb.fit_policy(q, t, dev)
        if kind == "trees":
            pol.lam = lambda_on(pol, q, t, dev + held)
        tb.write_setup(eval_id, t, new, name + tb.RULE, q, pol, "rule")
        tb.write_setup(eval_id, t, new, name + tb.LINES, q, pol, "lines")
        s = [tb.rule_cost(pol, q, t, w) for w in held]
        heldout[name] = {"rule": tb.per_window(s), "lines": tb.per_window([tb.lines_cost(pol, q, t, w) for w in held]), "auc": ev.auc(np.concatenate([q[w.rows] for w in held]), y_held),
                         "good_customers_declined": sum(x.summary["confusion"]["legit"]["decline"] for x in s), "frauds_stopped": sum(x.summary["confusion"]["fraud"]["decline"] + x.summary["confusion"]["fraud"]["review"] for x in s),
                         "sum_p": float(sum(pol.p(q[w.rows]).sum() for w in held)), "frauds": int(y_held.sum())}
        info[name] = {"kind": kind, "lambda": pol.lam, "lines": [pol.review, pol.decline], "sum_p_new": float(sum(pol.p(q[w.rows]).sum() for w in new))}
        log(f"{name}: {info[name]}, held-out rule {heldout[name]['rule'].mean():,.0f}, AUC {heldout[name]['auc']:.4f}")
        if name == "tab-all":
            head = {**pol.header(), "flat": FLAT}
            for w in new:
                outs = [{"by": "logreg", "q": float(v), "ms": 0.0} for v in q[w.rows]]
                am.write_trace(eval_id, t, w, CUST, outs, None, rule.decide_paced(pol.p(q[w.rows]), t.amount[w.rows], pol.lam, ev.COSTS, 0.0, FLAT), head, {})
    for w in new:
        am.write_trace(eval_id, t, w, ev.APPROVE_ALL, [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows], (None, None), None, None, {})
    return report(eval_id, info, tb.held_table(heldout, "tab-logreg"), log)


def report(eval_id: str, info=None, held=None, log=print) -> dict:
    t = ieee_data.load()
    new = ft.phase_windows(len(t), "tabular")
    path = RESULTS / f"{eval_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    setups = [k + b for k in SETS for b in (tb.RULE, tb.LINES)] + [CUST, ev.APPROVE_ALL]
    tr = {s: tb.read_setup(eval_id, s, new) for s in setups}
    by = {s: tb.window_costs(tr[s]) for s in setups}
    amt = lambda k: k + tb.RULE
    verdict = {"primary": ev.compare(by, amt("tab-all"), amt("tab-logreg")), "second": ev.compare(by, amt("tab-rich"), amt("tab-all")), "third": ev.compare(by, amt("tab-nocdv"), amt("tab-logreg-nocdv")), "fourth": ev.compare(by, CUST, amt("tab-all")),
               "reported": [ev.compare(by, amt("tab-all"), "tab-all" + tb.LINES), ev.compare(by, amt("tab-nocdv"), ev.APPROVE_ALL), ev.compare(by, CUST, amt("tab-logreg"))]}
    y = np.concatenate(ev.view(tr[amt("tab-logreg")])["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_tabular.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in new}, "test_rows": int(len(y)), "test_fraud": int(y.sum()),
           "info": info or old.get("info", {}), "setups": {s: tb.row(tr[s], s != ev.APPROVE_ALL) for s in setups}, "verdict": verdict, "heldout": held or old.get("heldout", {})}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = tb.report_table(out, "Tabular eval", "fraud/eval_tabular.py")
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", metavar="EVAL_ID", help="rebuild the results files from this eval's traces")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.report:
        report(a.report, log=log)
    else:
        run(stamp(), log)


if __name__ == "__main__":
    main()
