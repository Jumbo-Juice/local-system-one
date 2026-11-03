"""Do per-card and per-address history, velocity and more categories lower logreg's cost? IEEE-CIS, 10 new windows → fraud/results/eval_features/.

    python -m fraud.history                          # the feature cache (fraud/data/ieee-cis/history.npz), no labels
    python -m fraud.eval_features                    # fit, dev, 10 new windows, report
    python -m fraud.eval_features --report <eval id> # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_eval_<setup>_features<k>.jsonl. Results: fraud/results/eval_features/<eval id>.json + .md, and the dev look dev_look.json.

PRE-REGISTRATION (Phase 19; committed before any setup here scored a new window).
Owner, 2026-11-03: "better logreg features, with no look-ahead: per-card and per-address history, velocity, time since last payment, amount relative to the card's past, interactions, more categories. The signal for row i may use only earlier rows."
- Features, fraud/history.py and fraud/feature_sets.py. No label is an input to any of them (a test proves it). A later row can't change an earlier row's history (a test proves it). Rows of FINAL windows are read as earlier rows' history for later rows; none of their labels are read, and they aren't scored. Top-category lists come from fit rows only.
- Observed before registering, dev rows only (dev_look.json): fit on fit rows, rule and lines on Phase 14's 30 dev windows, read on 88 held-out dev chains. All columns: AUC 0.8882, 2,125, 4,003 good customers declined. Adding lifetime counts (`history`) lowers neither: +23. Adding those without them (`steady`), cents and calendar shape (`extra`) and identity columns (`identity`): AUC 0.8893, -54, CI -116 to 0. Adding product interactions and the D1-keyed history (`uid`) too: AUC 0.8903, -81, CI -176 to 8.
  Without C, D and V: 2,752, +626. With steady, extra, identity and interactions in their place: AUC 0.846, 2,555, +430, CI 291 to 570. Lifetime counts are dropped because their level keeps rising with the row number, and the dev look agrees (Inferred).
- Setups, each logistic regression refitted on fit rows, rule and lines as Phase 14 on its 30 dev windows. Names: feat-all (ieee-logreg-all, the base); feat-rich (all + steady + extra + identity); feat-uid (feat-rich + interactions + uid, which leans on D1); feat-nocdv (without C, D, V); feat-nocdv-rich (without C, D, V, with steady + extra + identity + interactions). ieee-approve-all.
- New windows: chains features0-features9 from fraud/final_test.py, 10,000 rows. Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed.
- Primary question: feat-rich against feat-all, both with the rule, on total cost. Distinguishable only if the 95% bootstrap CI of the mean paired difference (windows resampled, seed 20261102) excludes 0.
- Second: feat-uid against feat-all. Third: feat-nocdv-rich against feat-nocdv, what history gives back. Fourth: feat-nocdv-rich against feat-all, the doubt that remains. Not corrected for multiple comparisons beyond the primary. Also in customers: frauds stopped, fraud dollars stopped, good customers declined.
- Limits stated now: per-card history is only as good as the key, since card numbers are not in the data. History starts at the start of the file, so early rows have little of it. 10 windows is few. id_ columns are also anonymised and no audit was run on them.
- Stated before running (Inferred): feat-rich has a lower mean than feat-all by less than dev's 54 USD, not distinguishable. feat-uid the same. feat-nocdv-rich gets back about a third of what dropping C, D, V costs, and may not be distinguishable on 10 windows. Logistic regression gains little from these columns because they need interactions it doesn't have, so a tree model may do much more with them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, feature_sets as fs, final_test as ft, history as h, ieee_data, tabular as tb
from .baselines import LogReg

RESULTS = Path(__file__).resolve().parent / "results" / "eval_features"
SETS = {"feat-all": ((), ""), "feat-rich": (("steady", "extra", "identity"), ""), "feat-uid": (("steady", "extra", "identity", "inter", "uid"), ""), "feat-nocdv": ((), "CDV"),
        "feat-nocdv-rich": (("steady", "extra", "identity", "inter"), "CDV")}


def scorers(t) -> dict:
    fit_rows = np.arange(ieee_data.split_bounds(len(t))[0])
    cache = h.load()

    def make(name, parts, drop):
        def score(windows):
            t0 = time.perf_counter()
            f = fs.FeatureSet(parts, drop, cache)
            m = LogReg().fit(f(t, fit_rows), t.is_fraud[fit_rows].astype(float))
            print(f"{name} fitted in {time.perf_counter() - t0:.0f} s, {len(m.w) - 1} features", flush=True)
            return tb.scores_for(m, f, t, windows)
        return score

    return {k: make(k, *v) for k, v in SETS.items()}


def run(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    new = ft.phase_windows(len(t), "features")
    info, heldout = tb.run_scores(eval_id, t, scorers(t), new, log)
    return report(eval_id, info, tb.held_table(heldout, "feat-all"), log)


def report(eval_id: str, info=None, held=None, log=print) -> dict:
    t = ieee_data.load()
    new = ft.phase_windows(len(t), "features")
    path = RESULTS / f"{eval_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    setups = [k + b for k in SETS for b in (tb.RULE, tb.LINES)] + [ev.APPROVE_ALL]
    tr = {s: tb.read_setup(eval_id, s, new) for s in setups}
    by = {s: tb.window_costs(tr[s]) for s in setups}
    amt = lambda k: k + tb.RULE
    verdict = {"primary": ev.compare(by, amt("feat-rich"), amt("feat-all")), "second": ev.compare(by, amt("feat-uid"), amt("feat-all")), "third": ev.compare(by, amt("feat-nocdv-rich"), amt("feat-nocdv")), "fourth": ev.compare(by, amt("feat-nocdv-rich"), amt("feat-all")),
               "reported": [ev.compare(by, amt("feat-all"), "feat-all" + tb.LINES), ev.compare(by, amt("feat-nocdv-rich"), ev.APPROVE_ALL)]}
    y = np.concatenate(ev.view(tr[amt("feat-all")])["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_features.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in new}, "test_rows": int(len(y)), "test_fraud": int(y.sum()),
           "info": info or old.get("info", {}), "setups": {s: tb.row(tr[s], s != ev.APPROVE_ALL) for s in setups}, "verdict": verdict, "heldout": held or old.get("heldout", {})}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = tb.report_table(out, "Feature eval", "fraud/eval_features.py")
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def main() -> None:
    from .runs import stamp

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
