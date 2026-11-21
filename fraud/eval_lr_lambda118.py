"""LR with lambda picked on all 118 dev windows, applied to its saved FINAL scores: one labelled read-out → fraud/results/eval_lr_lambda118/.

    python -m fraud.eval_lr_lambda118     # a few minutes on the CPU (LR is refitted on fit rows for its dev scores); no model touches FINAL

REGISTRATION: fraud/prereg/final_lr_lambda118_readout.md, committed with this script before it ran (referee revision, 2026-11-21).
Why: the baseline's lambda comes from the 30 dev windows, every other setup's from all 118 (REVISION_LOG.md Stage 0, section 3).
This measures what that asymmetry does on FINAL. Nothing is re-scored: LR's FINAL scores come from its Phase 23 traces, and the
decisions are recomputed with the same rule and costs, as eval_readouts.section_grid does. Every number is labelled "read-out".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, eval_final as ef, eval_gbdt_readout as gr, eval_ieee as ev, eval_ieee_amount as am, eval_readouts as ro, final_test as ft, ieee_data, tabular as tb
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_lr_lambda118"
LABEL = "read-out"
LR118 = "final-logreg-amt-lam118"
OTHERS = (ef.SIGNAL_AMT, ef.SIGNAL_CUST, ef.RICH_AMT, gr.SETUP)


def window_costs(pol: tb.Policy, lam: float, q_fin: list[np.ndarray], windows, t) -> np.ndarray:
    out = []
    for w, q in zip(windows, q_fin):
        wanted, fallback = am.decide(pol.p(q), t.amount[w.rows], lam, ev.COSTS)
        out.append(costs.score(ev.COSTS, wanted, fallback, t.is_fraud[w.rows].astype(bool), t.amount[w.rows]).summary["cost"])
    return np.array(out)


def diff(by: dict, a: str, b: str) -> dict:
    d = by[a] - by[b]
    ci = list(ev.boot_ci(d))
    sd = float(d.std(ddof=1))
    return {"a": a, "b": b, "mean_diff_a_minus_b": float(d.mean()), "ci95": ci, "distinguishable": bool(ci[0] > 0 or ci[1] < 0),
            "a_cheaper_windows": int((d < 0).sum()), "paired_sd": sd, "mde": 2.8 * sd / np.sqrt(len(d)), "label": LABEL}


def run(log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    windows = ft.final_windows(n)
    tr = ro.traces()
    by = {s: tb.window_costs(v) for s, v in tr.items()}

    dev, held = tb.dev30(n), tb.heldout_dev(n)
    q_dev = ro.dev_scores(t, log)[ef.BASELINE]
    pol = tb.fit_policy(q_dev, t, dev)
    lam30 = pol.lam
    saved_lam = tr[ef.BASELINE][0][0]["policy"]["lambda"]
    if lam30 != saved_lam:
        raise SystemExit(f"check 1 failed: lambda on dev30 is {lam30}, the saved LR trace has {saved_lam}")
    lam118 = am.pick_lambda([pol.p(q_dev[w.rows]) for w in dev + held], [t.is_fraud[w.rows].astype(int) for w in dev + held], [t.amount[w.rows] for w in dev + held], ev.COSTS)[0]

    q_fin = ro.arr(tr[ef.BASELINE], "q")
    again = window_costs(pol, lam30, q_fin, windows, t)
    gap = float(np.abs(again - by[ef.BASELINE]).max())
    if gap > 1e-6:
        raise SystemExit(f"check 2 failed: lambda {lam30} doesn't reproduce the saved LR costs (max gap {gap})")
    by[LR118] = window_costs(pol, lam118, q_fin, windows, t)
    log(f"lambda dev30 {lam30}, all 118 {lam118}; LR mean {by[ef.BASELINE].mean():,.2f} -> {by[LR118].mean():,.2f} USD/window")

    out = {"eval": stamp(), "script": "fraud/eval_lr_lambda118.py", "label": LABEL, "prereg": "fraud/prereg/final_lr_lambda118_readout.md",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {"phase23": gr.FINAL_EVAL, "gbdt_readout": ro._readout_id()}, "costs": ev.COSTS.as_dict(),
           "lambda": {"dev30": lam30, "all_118": lam118}, "checks": {"lambda_dev30_matches_trace": True, "reproduces_saved_lr_max_gap": gap},
           "mean_cost": {s: float(v.mean()) for s, v in by.items()},
           "primary": diff(by, LR118, ef.BASELINE),
           "against_lr118": {s: diff(by, s, LR118) for s in OTHERS},
           "against_lr_as_registered": {s: diff(by, s, ef.BASELINE) for s in OTHERS}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{out['eval']}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = report_md(out)
    (RESULTS / f"{out['eval']}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def report_md(r: dict) -> str:
    ci = lambda c: f"{c[0]:,.0f} to {c[1]:,.0f}"
    dist = lambda x: "distinguishable" if x["distinguishable"] else "not distinguishable"
    p = r["primary"]
    L = [f"# LR with lambda on all 118 dev windows, on FINAL (read-out)", "",
         f"Eval `{r['eval']}` (`python -m fraud.eval_lr_lambda118`). Registration: `{r['prereg']}`. Every number is a read-out (Observed): the FINAL windows were read before.", "",
         f"- lambda: {r['lambda']['dev30']} on dev30 (the baseline), {r['lambda']['all_118']} on all 118 dev windows.",
         f"- Check: lambda {r['lambda']['dev30']} reproduces the saved LR costs (max gap {r['checks']['reproduces_saved_lr_max_gap']:.2g} USD).",
         f"- Primary, LR (118) − LR (dev30): {p['mean_diff_a_minus_b']:+,.1f} USD/window ({ci(p['ci95'])}), {dist(p)}; paired SD {p['paired_sd']:,.0f}, MDE {p['mde']:,.0f}.", "",
         "| setup | − LR as registered | 95% interval | verdict | − LR, lambda on 118 | 95% interval | verdict |", "|---|---:|---:|---|---:|---:|---|"]
    for s in OTHERS:
        a, b = r["against_lr_as_registered"][s], r["against_lr118"][s]
        L.append(f"| {ro.NAMES[s]} | {a['mean_diff_a_minus_b']:+,.0f} | {ci(a['ci95'])} | {dist(a)} | {b['mean_diff_a_minus_b']:+,.0f} | {ci(b['ci95'])} | {dist(b)} |")
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
