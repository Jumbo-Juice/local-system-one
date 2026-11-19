"""Boosted trees alone, with the rule, on the 20 FINAL windows: one labelled read-out → fraud/results/eval_gbdt_readout/.

    python -m fraud.eval_gbdt_readout                     # dev fits, then FINAL once; refuses a second run
    python -m fraud.eval_gbdt_readout --report <eval id>  # rebuild the results files from the traces

REGISTRATION: fraud/prereg/final_gbdt_readout.md, committed with this script before it scored any FINAL row.
Why it exists: the revision's Stage 0 audit found no IEEE-CIS test window left unscored (REVISION_LOG.md), so no clean held-out test is possible. The revision prompt's fallback is
this read-out. It is not a fresh test: the FINAL windows were read once already (Phase 23), and every number here is labelled "read-out".

Setup final-tab-all-amt: Phase 20's tab-all (fraud/gbdt.py, 800 trees, fraud/models/tabular_tab-all.pkl, loaded, never refitted), Phase 14's rule.
p's fit on the 30 dev windows, lambda on all 118 dev windows, exactly as Phase 21 (fraud/eval_lm_nuc.py) fitted tab-all. No FINAL label is read before the decisions are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_final as ef, eval_ieee as ev, eval_ieee_amount as am, eval_tabular as et, final_test as ft, history, ieee_data, tabular as tb
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_gbdt_readout"
LABEL = "read-out"
SETUP = "final-tab-all-amt"
PHASE21 = ROOT / "results" / "eval_lm_nuc" / "20261104-090217.json"
MODEL_SHA256 = "92640f230ce032c4faa6bf8589a2bd524f8a73487c89f83af884504a0e50adb1"  # fraud/models/tabular_tab-all.pkl, as registered
FINAL_EVAL = "20261104-090649"  # Phase 23: the saved traces of the other setups on the same windows
COMPARE = {"primary": (ef.SIGNAL_AMT, SETUP), "secondary": (SETUP, ef.BASELINE), "reported": [(ef.RICH_AMT, SETUP)]}


def refuse_second_run(eval_id: str) -> None:
    other = [p.name for p in ev.RUNS.glob(f"*_eval_{SETUP}_final*.jsonl") if not p.name.startswith(f"{eval_id}_")]
    if other:
        raise SystemExit(f"this read-out has already run ({other[0]}). It runs once; use --report with that eval's id.")


def policy(m, f, t, log=print) -> tb.Policy:
    """tab-all's fit from dev rows only, checked against what Phase 21 recorded."""
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    q = et.score(m, f, t, dev + held)
    pol = tb.fit_policy(q, t, dev)
    pol.lam = et.lambda_on(pol, q, t, dev + held)
    p21 = json.loads(PHASE21.read_text(encoding="utf-8"))
    was, row = p21["fits"]["tab-all"], p21["setups"]["tab-all-amt"]
    rows = np.concatenate([w.rows for w in held])
    cost = float(tb.per_window([tb.rule_cost(pol, q, t, w) for w in held]).mean())
    auc = ev.auc(q[rows], t.is_fraud[rows].astype(int))
    log(f"dev reproduction, 88 held-out chains: lambda {pol.lam} (Phase 21 {was['lambda']}), {cost:,.2f} USD/window (Phase 21 {row['mean_cost']:,.2f}), AUC {auc:.4f} (Phase 21 {row['auc']:.4f})")
    if pol.lam != was["lambda"] or not np.allclose([pol.review, pol.decline], was["lines"]) or abs(cost - row["mean_cost"]) > 0.5 or abs(auc - row["auc"]) > 1e-4:
        raise SystemExit("the dev fit doesn't reproduce Phase 21; stopping before any FINAL row is scored")
    return pol


def run(eval_id: str, log=print) -> dict:
    refuse_second_run(eval_id)
    t = ieee_data.load()
    windows = ft.final_windows(len(t))
    registered = json.loads((ROOT / "results" / "eval_final" / f"{FINAL_EVAL}.json").read_text(encoding="utf-8"))["windows"]
    if {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in windows} != registered:
        raise SystemExit("the FINAL windows differ from Phase 23's")
    if hashlib.sha256((et.MODELS / "tabular_tab-all.pkl").read_bytes()).hexdigest() != MODEL_SHA256:
        raise SystemExit("tabular_tab-all.pkl isn't the registered model")
    kind, parts, drop = et.SETS["tab-all"]
    m, f = et.fit("tab-all", kind, parts, drop, t, history.load(), log)
    pol = policy(m, f, t, log)
    q = et.score(m, f, t, windows)
    for w in windows:
        outs = [{"by": "trees", "q": float(v), "ms": 0.0} for v in q[w.rows]]
        am.write_trace(eval_id, t, w, SETUP, outs, None, am.decide(pol.p(q[w.rows]), t.amount[w.rows], pol.lam, ev.COSTS), pol.header(), {})
    log(f"wrote {len(windows)} traces, eval {eval_id}")
    return report(eval_id, log)


def report(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    windows = ft.final_windows(len(t))
    tr = {SETUP: tb.read_setup(eval_id, SETUP, windows)} | {s: tb.read_setup(FINAL_EVAL, s, windows) for s in ef.SETUPS}
    by = {s: tb.window_costs(v) for s, v in tr.items()}
    verdict = {"primary": ev.compare(by, *COMPARE["primary"]), "secondary": ev.compare(by, *COMPARE["secondary"]),
               "reported": [ev.compare(by, a, b) for a, b in COMPARE["reported"]]}
    out = {"eval": eval_id, "script": "fraud/eval_gbdt_readout.py", "label": LABEL, "prereg": "fraud/prereg/final_gbdt_readout.md", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "others_from": f"fraud/results/eval_final/{FINAL_EVAL}.json", "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in windows},
           "setups": {s: {**tb.row(v, s != ev.APPROVE_ALL), "label": LABEL if s == SETUP else "held-out (pre-registered)"} for s, v in tr.items()},
           "mde_realised": {"label": LABEL, "formula": "2.8 * paired SD / sqrt(20): 80% power, two-sided 5%, normal approximation; SD from these FINAL windows (the registered MDE used dev SDs)",
                            "pairs": {f"{a} - {b}": {"sd": float((by[a] - by[b]).std(ddof=1)), "mde": float(2.8 * (by[a] - by[b]).std(ddof=1) / np.sqrt(len(windows)))}
                                      for a, b in (COMPARE["primary"], COMPARE["secondary"], *COMPARE["reported"])}},
           "test_rows": int(sum(len(w.rows) for w in windows)), "test_fraud": int(sum(t.is_fraud[w.rows].sum() for w in windows)), "verdict": verdict}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = tb.report_table({**out, "heldout": {}}, "GBDT alone on FINAL (read-out)", "fraud/eval_gbdt_readout.py")
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", metavar="EVAL_ID")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.report:
        report(a.report, log)
    else:
        run(stamp(), log)


if __name__ == "__main__":
    main()
