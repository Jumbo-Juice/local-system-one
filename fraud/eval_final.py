"""The final test: 20 FINAL windows, once → fraud/results/eval_final/.

    python -m fraud.eval_final                   # about 45 min on this GPU. Refuses if a FINAL trace from another eval exists
    python -m fraud.eval_final --resume <id>     # the same eval, after an interruption; finished windows aren't scored again
    python -m fraud.eval_final --report <id>     # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_eval_<setup>_final<k>.jsonl. Results: fraud/results/eval_final/<eval id>.json + .md. Check apart from the report code: python -m fraud.dev.verify_tabular <results json>.

PRE-REGISTRATION (Phase 23; committed before any setup here scored a FINAL row, and before fraud.lm_signal --final added their signal). No FINAL label has been read.
Owner, 2026-11-03: "Pre-register at most 3 candidates plus logreg with the Phase 14 rule as baseline, run FINAL windows once." The final setup must be a language decision model at heart, and may read a tabular score as a signal.
- Windows: the 20 FINAL windows of fraud/final_test.py (seed 20261105, reserved in the first commit of this work), 20 blocks of 1,000 contiguous rows. Once. No result of this run prompts a rerun on them; a bug that stops the run and is found on anything but a result is fixed and noted here, with --resume.
- Baseline, final-logreg-amt: logreg on all columns (Phase 20's tab-logreg, fit rows only), Phase 14's rule: p's fit and lambda on Phase 14's 30 dev windows.
- Candidates, chosen by the rule stated in fraud/eval_combos.py (results fraud/results/eval_combos/20261104-080932.*):
  final-signal-amt: lm-signal, d1-Omni-2B fine-tuned in order rev on the rich prompt plus the tabular score (adapter 20261104-011148_lora-d1-ieee-full-signal), Phase 14's rule. p's fit on the 30 dev windows, lambda on all 118, from the dev scores of Phase 21.
  final-signal-cust: the same scores and fit, with Phase 18's flat 10 USD in the expected cost of declining, never in what it's charged.
  final-rich-amt: lm-rich, the same recipe without the tabular score (adapter 20261103-230750_lora-d1-ieee-full-rich), Phase 14's rule, fit as above. The language model with no trees in it.
- ieee-approve-all, which needs no model, is reported beside them. Trees alone aren't run here: the limit is 3 candidates, and the owner's constraint rules them out for deployment. What the language model costs against trees therefore has no FINAL number.
- Scoring: lone prompts on this GPU, nothing else running on it, a row's time recorded in its trace. p90 must stay at or under 250 ms. A setup over it is reported as failing the bar.
- Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% of a window reviewed (eval_ieee.COSTS). Distinguishable only if the 95% bootstrap CI of the mean paired difference (windows resampled, seed 20261102) excludes 0.
- Primary question: final-signal-amt against final-logreg-amt, on total cost.
- Second: final-signal-cust against final-signal-amt, on cost and on good customers declined.
- Third: final-rich-amt against final-logreg-amt, on cost. Reported, not questions: final-signal-cust against final-logreg-amt, final-rich-amt against final-signal-amt, final-signal-amt against ieee-approve-all. Not corrected for multiple comparisons beyond the primary. All in customers as well as dollars.
- What gets recommended (fixed here, Implementation choice). Among the three: final-signal-cust, unless the second question shows it distinguishably dearer than final-signal-amt, then final-signal-amt. The reason is the customers: on dev the line cut good customers declined by about 74% for 63 USD per window. If the primary shows the language model distinguishably dearer than the baseline, that is reported as the price of the owner's constraint and the recommendation stands, because the constraint isn't this test's to change.
- Limits stated now: 20 windows and a heavy tail. On dev (Phase 21), lm-signal only partly differs from the score it reads: rank correlation 0.984, 2,233 of 88,000 decisions differ, 67 USD per window dearer than the score alone, not distinguishable. So final-signal-amt doesn't show what a language model adds on its own; final-rich-amt is there for that. The lm-signal adapter was trained on a score from 4 time-block models and reads one from a model fitted on all fit rows. C columns stay in, and Phase 17 couldn't clear their timing. Latency here isn't comparable with the laptop's.
- Stated before running (Inferred): every setup is dearer in absolute terms than on dev, as in Phases 18 and 20, and the trees' dev edge shrinks on test, so lm-signal's does too. final-signal-amt is lower than the baseline by about 100, not distinguishable. final-signal-cust costs within noise of final-signal-amt and declines about a quarter of the good customers. final-rich-amt about ties the baseline, within about 150. All p90 under 100 ms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_combos as ec, eval_ieee as ev, eval_ieee_amount as am, eval_lm_nuc as nuc, eval_tabular as et, final_test as ft, history, ieee_brain, ieee_data, lm_signal, rule, tabular as tb
from .brain import OPTIONS
from .eval_ft import release
from .runs import finished, read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_final"
BASELINE = "final-logreg-amt"
SIGNAL_AMT, SIGNAL_CUST, RICH_AMT = "final-signal-amt", "final-signal-cust", "final-rich-amt"
SIGNAL_ADAPTER = ROOT / "models" / "20261104-011148_lora-d1-ieee-full-signal"
RICH_ADAPTER = ROOT / "models" / "20261103-230750_lora-d1-ieee-full-rich"
FLAT = ec.FLAT
SETUPS = (BASELINE, SIGNAL_AMT, SIGNAL_CUST, RICH_AMT, ev.APPROVE_ALL)


def refuse_second_run(eval_id: str, runs: Path = ev.RUNS) -> None:
    """FINAL runs once."""
    other = [p.name for p in runs.glob("*_eval_*_final*.jsonl") if not p.name.startswith(f"{eval_id}_")]
    if other:
        raise SystemExit(f"FINAL has already run ({other[0]}, and {len(other) - 1} more). It runs once; use --resume with that eval's id.")


def policies(t, log=print) -> dict:
    """Every fit from dev rows, before any FINAL row is scored."""
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    out = {}
    q = nuc.references(t, log)["tab-logreg"]
    out["logreg"] = tb.fit_policy(q, t, dev)
    for k in ("lm-signal", "lm-rich"):
        q = nuc.saved(k, n)[0]
        pol = tb.fit_policy(q, t, dev)
        pol.lam = et.lambda_on(pol, q, t, dev + held)
        out[k] = pol
    return out


class LoneModel:
    """A language-model setup on lone prompts, as a deployment would read it."""

    def __init__(self, adapter: Path, hist: dict):
        man = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
        self.model = nuc.load_model(adapter)
        self.hist = hist
        self.info = {**self.model.info(), "adapter": {"dir": adapter.name, "sha256": man["adapter_sha256"], "rows_sha256": man["rows_sha256"], "steps": man["steps_done"]}}

    def score(self, t, i: int) -> dict:
        body = ieee_brain.request(t, i, nuc.VARIANT, self.hist)
        t0 = time.perf_counter()
        r = self.model.post(body)
        ms = (time.perf_counter() - t0) * 1000
        p = [float(r["answers"][nuc.VARIANT]["probabilities"][o]) for o in OPTIONS]
        return {"by": "model", "q": p[2], "scores": dict(zip(ev.ACTIONS, p)), "orders": [dict(zip(ev.ACTIONS, p))], "prompt_tokens": int(r["usage"]["input_tokens"]), "ms": ms}


def run_language(eval_id: str, t, windows: list[ev.Win], setup: str, adapter: Path, hist: dict, pol: tb.Policy, log=print) -> None:
    model = None
    for w in windows:
        path = ev.trace_file(eval_id, "eval", setup, w.id)
        if finished(path):
            continue
        if model is None:
            model = LoneModel(adapter, hist)
        t0 = time.perf_counter()
        outs = [model.score(t, int(i)) for i in w.rows]
        q = np.array([o["q"] for o in outs])
        recs = am.write_trace(eval_id, t, w, setup, outs, None, am.decide(pol.p(q), t.amount[w.rows], pol.lam, ev.COSTS), pol.header(), model.info)
        log(f"{setup} {w.id}: done in {time.perf_counter() - t0:.0f} s, p90 {recs[-1]['summary']['latency_ms']['p90']:.0f} ms")
    del model
    release()


def derive_customer_line(eval_id: str, t, windows: list[ev.Win], pol: tb.Policy) -> None:
    head = {**pol.header(), "flat": FLAT}
    for w in windows:
        recs = read_run(ev.trace_file(eval_id, "eval", SIGNAL_AMT, w.id))
        outs = am.outs_of(recs)
        q = np.array([o["q"] for o in outs])
        am.write_trace(eval_id, t, w, SIGNAL_CUST, outs, None, rule.decide_paced(pol.p(q), t.amount[w.rows], pol.lam, ev.COSTS, 0.0, FLAT), head, recs[0]["backend"])


def run(eval_id: str, log=print) -> dict:
    refuse_second_run(eval_id)
    t = ieee_data.load()
    n = len(t)
    windows = ft.final_windows(n)
    pols = policies(t, log)
    log("fits, from dev rows: " + json.dumps({k: {"lambda": p.lam} for k, p in pols.items()}))
    log(f"eval {eval_id}")
    hist = history.load()
    lr_kind, lr_parts, lr_drop = et.SETS["tab-logreg"]
    m, f = et.fit("tab-logreg", lr_kind, lr_parts, lr_drop, t, hist, log)
    for w in windows:
        q = m.predict(f(t, w.rows))
        am.write_trace(eval_id, t, w, BASELINE, [{"by": "logreg", "q": float(v), "ms": 0.0} for v in q], None, am.decide(pols["logreg"].p(q), t.amount[w.rows], pols["logreg"].lam, ev.COSTS), pols["logreg"].header(), {})
        am.write_trace(eval_id, t, w, ev.APPROVE_ALL, [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows], (None, None), None, None, {})
    run_language(eval_id, t, windows, RICH_AMT, RICH_ADAPTER, hist, pols["lm-rich"], log)
    sig = lm_signal.load()
    rows = np.concatenate([w.rows for w in windows])
    if np.isnan(sig[rows]).any():
        lm_signal.add_final(log)
        sig = lm_signal.load()
    run_language(eval_id, t, windows, SIGNAL_AMT, SIGNAL_ADAPTER, {**hist, "signal": sig}, pols["lm-signal"], log)
    derive_customer_line(eval_id, t, windows, pols["lm-signal"])
    return report(eval_id, log)


def report(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    windows = ft.final_windows(len(t))
    tr = {s: tb.read_setup(eval_id, s, windows) for s in SETUPS}
    by = {s: tb.window_costs(tr[s]) for s in SETUPS}
    verdict = {"primary": ev.compare(by, SIGNAL_AMT, BASELINE), "second": ev.compare(by, SIGNAL_CUST, SIGNAL_AMT), "third": ev.compare(by, RICH_AMT, BASELINE),
               "reported": [ev.compare(by, SIGNAL_CUST, BASELINE), ev.compare(by, RICH_AMT, SIGNAL_AMT), ev.compare(by, SIGNAL_AMT, ev.APPROVE_ALL)]}
    y = np.concatenate(ev.view(tr[BASELINE])["truth"])
    setups = {s: tb.row(tr[s], s != ev.APPROVE_ALL) for s in SETUPS}
    second_dearer = verdict["second"]["distinguishable"] and verdict["second"]["better"] == SIGNAL_AMT
    out = {"eval": eval_id, "script": "fraud/eval_final.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "complete": True, "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in windows}, "test_rows": int(len(y)), "test_fraud": int(y.sum()),
           "setups": setups, "verdict": verdict, "recommended": SIGNAL_AMT if second_dearer else SIGNAL_CUST, "latency_bar_ms": nuc.P90_BAR_MS, "latency_pass": {s: setups[s]["latency_ms"]["p90"] <= nuc.P90_BAR_MS for s in (SIGNAL_AMT, RICH_AMT)}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = tb.report_table({**out, "heldout": {}}, "Final test", "fraud/eval_final.py") + f"\nBy the pre-registered rule, recommended: **{out['recommended']}**.\n"
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", metavar="EVAL_ID")
    ap.add_argument("--report", metavar="EVAL_ID")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.report:
        report(a.report, log)
    else:
        run(a.resume or stamp(), log)


if __name__ == "__main__":
    main()
