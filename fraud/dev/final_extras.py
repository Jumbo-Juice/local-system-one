"""Read-outs of the final test beyond its pre-registration, from its traces. No model, no new score. Nothing here chooses anything → fraud/results/eval_final/extras.json.

    python -m fraud.dev.final_extras <eval id>

- How far lm-signal follows the score it reads, on FINAL rows: rank correlation only, no label. The signal alone wasn't a registered setup, so its cost isn't computed.
- Each setup if a declined good customer truly cost 10 USD more, Phase 18's told cost: arithmetic on the traces.
- Customers touched in any way: a good customer declined or reviewed.
- The dearest window of the recommended setup, and its largest fraud let through.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .. import eval_final as ef, eval_ieee as ev, final_test as ft, ieee_data, lm_signal, tabular as tb

FLAT = 10.0


def tx(recs):
    return [r for r in recs if r.get("type") == "tx"]


def main() -> None:
    eval_id = sys.argv[1]
    t = ieee_data.load()
    windows = ft.final_windows(len(t))
    sig = lm_signal.load()
    tr = {s: tb.read_setup(eval_id, s, windows) for s in (ef.BASELINE, ef.SIGNAL_AMT, ef.SIGNAL_CUST, ef.RICH_AMT)}
    rows = np.concatenate([w.rows for w in windows])
    q = np.concatenate(ev.view(tr[ef.SIGNAL_AMT])["scores"])
    rank = lambda a: np.argsort(np.argsort(a))
    out = {"signal_rank_corr": float(np.corrcoef(rank(q), rank(sig[rows]))[0, 1])}
    told, good_touched = {}, {}
    for s, traces in tr.items():
        told[s] = np.array([r[-1]["summary"]["cost"] + FLAT * sum(x["action"] == "decline" and not x["truth"] for x in tx(r)) for r in traces])
        good_touched[s] = sum(x["action"] != "approve" and not x["truth"] for r in traces for x in tx(r))
    out["told_mean"] = {s: float(v.mean()) for s, v in told.items()}
    out["told_vs_baseline"] = {s: {"mean": float((told[s] - told[ef.BASELINE]).mean()), "ci": list(ev.boot_ci(told[s] - told[ef.BASELINE]))} for s in told if s != ef.BASELINE}
    out["told_cust_vs_amt"] = {"mean": float((told[ef.SIGNAL_CUST] - told[ef.SIGNAL_AMT]).mean()), "ci": list(ev.boot_ci(told[ef.SIGNAL_CUST] - told[ef.SIGNAL_AMT]))}
    out["good_touched"] = good_touched
    costs = tb.window_costs(tr[ef.SIGNAL_CUST])
    k = int(np.argmax(costs))
    missed = sorted((x["tx"]["amount"] for x in tx(tr[ef.SIGNAL_CUST][k]) if x["truth"] and x["action"] == "approve"), reverse=True)
    out["dearest_window"] = {"window": windows[k].id, "cost": float(costs[k]), "baseline": float(tb.window_costs(tr[ef.BASELINE])[k]), "largest_frauds_let_through": missed[:5]}
    d = costs - tb.window_costs(tr[ef.BASELINE])
    out["cust_vs_baseline_without_that_window"] = {"mean": float(np.delete(d, k).mean()), "ci": list(ev.boot_ci(np.delete(d, k)))}
    path = ef.RESULTS / "extras.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
