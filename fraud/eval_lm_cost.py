"""Does training on cost lower the language model's cost? d1-Omni-2B with no tabular score, held-out dev only → fraud/results/eval_lm_cost/.

    python -m fraud.finetune_ieee --train --full --rich --weighted                                 # the adapter, about 2 hours
    python -m fraud.eval_lm_nuc --score lm-weighted fraud/models/<time>_lora-d1-ieee-full-rich-weighted   # dev scores, about 85 min
    python -m fraud.eval_lm_cost                                                                   # the comparison, a few minutes

Traces: fraud/runs/<eval id>_eval_<setup>_hdev<k>.jsonl. Results: fraud/results/eval_lm_cost/<eval id>.json + .md.

PRE-REGISTRATION (Phase 25; committed before the adapter was trained).
Owner, 2026-11-04: focus on IEEE-CIS cost while staying on a decision model. After the final test, the owner judged the setup reading a tree score "really not a decision model doing its grunt work". So this phase has no tabular score in the prompt.
- Why: until now every row's loss counted the same, so a 5 USD fraud taught as much as a 1,500 USD one, while the cost is in dollars. The top 10% of training frauds hold 41% of their dollars (Observed).
- Setup lm-weighted: lm-rich's recipe exactly: the same 47,952 fit rows, prompt, seed 0, order `rev`, LoRA, schedule and one pass. One change: each row's loss counts by what a mistake on it costs (fraud.finetune_ieee.cost_weights). A fraud approved costs its amount, a legit row declined its friction: 10% of the amount, at least 1 USD.
- Fixed from the training rows before any model ran (Implementation choice): capped at each class's 99th percentile, because uncapped one row could count up to 40 times the mean; scaled to mean 1 within each class, so the 1:3 class balance doesn't move and only attention within a class does. Observed on those rows: largest weight 7.3 for fraud, 8.3 for legit; effective rows 40% and 37%. No other weighting will be tried under this registration.
- Dev only. All 38 test chains and FINAL are spent, and nothing here scores a test row. Fit rule and lines on Phase 14's 30 dev windows, lambda on all 118 as in Phases 21-22, read on the 88 held-out chains, so the read is flattered. Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed. Scored in batches of 16, latency on lone prompts, bar p90 <= 250 ms.
- Primary question: lm-weighted against lm-rich, both with the rule, on total cost. Distinguishable only if the 95% bootstrap CI of the mean paired difference (windows resampled, seed 20261102) excludes 0.
- Second: lm-weighted against logreg with Phase 14's rule. Third: lm-weighted against trees, the reference. Reported, not questions: lm-weighted with Phase 18's customer line against lm-rich with it, and against logreg.
- Reported to show the mechanism: frauds of 500 USD or more stopped, fraud dollars stopped, good customers declined, AUC.
- What follows from it: this phase doesn't produce a test number and doesn't choose what's deployed. If the primary is distinguishable, weighting joins the recipe for the owner's next options. If not, that's reported and the recipe stays. Whether to read FINAL again is the owner's decision, and it would be labelled a second look.
- Limits stated now: dev isn't test, and lm-rich's tie with logreg on dev did not hold on FINAL. One adapter: a second seed moved cost by about 100 on dev (Phase 21), so a primary difference near that is weak even if it excludes 0.
- Stated before running (Inferred): lm-weighted is lower than lm-rich by about 100, not distinguishable. AUC falls by up to 0.01, because attention moves away from small frauds. Fraud dollars stopped rise, frauds of 500 USD or more stopped rise, and frauds stopped overall fall a little. It beats logreg by about 100, not distinguishable, and stays dearer than the trees.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_lm_nuc as nuc, eval_tabular as et, ieee_data, rule, tabular as tb, eval_ieee_amount as am
from .runs import stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_lm_cost"
WEIGHTED, RICH, LOGREG, TREES = "lm-weighted", "lm-rich", "tab-logreg", "tab-all"
FLAT, CUST = 10.0, "-cust"
BIG_USD = 500.0


def big_frauds(traces: list[list[dict]], at_least: float = BIG_USD) -> dict:
    """Frauds of at least this amount, and how many were stopped, from the traces."""
    fraud = [r for recs in traces for r in recs if r.get("type") == "tx" and r["truth"] and r["tx"]["amount"] >= at_least]
    return {"big_frauds": len(fraud), "big_frauds_stopped": sum(r["action"] != "approve" for r in fraud)}


def run(log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    info = {}
    scores = {}
    for k in (WEIGHTED, RICH):
        scores[k], info[k] = nuc.saved(k, n)
    refs = nuc.references(t, log)
    scores.update({k: refs[k] for k in (LOGREG, TREES)})
    eval_id, fits = stamp(), {}
    for k, q in scores.items():
        pol = tb.fit_policy(q, t, dev)
        if k != LOGREG:
            pol.lam = et.lambda_on(pol, q, t, dev + held)
        by = "lm" if k in info else "logreg"
        tb.write_setup(eval_id, t, held, k + tb.RULE, q, pol, "rule", by)
        tb.write_setup(eval_id, t, held, k + tb.LINES, q, pol, "lines", by)
        if k in info:
            head = {**pol.header(), "flat": FLAT}
            for w in held:
                outs = [{"by": by, "q": float(v), "ms": 0.0} for v in q[w.rows]]
                am.write_trace(eval_id, t, w, k + CUST, outs, None, rule.decide_paced(pol.p(q[w.rows]), t.amount[w.rows], pol.lam, ev.COSTS, 0.0, FLAT), head, {})
        fits[k] = {"lambda": pol.lam, "lines": [pol.review, pol.decline], "sum_p": float(sum(pol.p(q[w.rows]).sum() for w in held))}
        log(f"{k}: {fits[k]}")
    names = [k + b for k in scores for b in (tb.RULE, tb.LINES)] + [k + CUST for k in info]
    tr = {s: tb.read_setup(eval_id, s, held) for s in names}
    by = {s: tb.window_costs(tr[s]) for s in names}
    amt = lambda k: k + tb.RULE
    verdict = {"primary": ev.compare(by, amt(WEIGHTED), amt(RICH)), "second": ev.compare(by, amt(WEIGHTED), amt(LOGREG)), "third": ev.compare(by, amt(WEIGHTED), amt(TREES)),
               "reported": [ev.compare(by, WEIGHTED + CUST, RICH + CUST), ev.compare(by, WEIGHTED + CUST, amt(LOGREG))]}
    q_w, q_r = (scores[k][np.concatenate([w.rows for w in held])] for k in (WEIGHTED, RICH))
    out = {"eval": eval_id, "script": "fraud/eval_lm_cost.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in held},
           "fits": fits, "lm": info, "setups": {s: {**tb.row(tr[s]), **big_frauds(tr[s])} for s in names}, "verdict": verdict, "rank_corr_with_lm_rich": nuc.rank_corr(q_w, q_r), "latency_pass": info[WEIGHTED]["latency_ms"]["p90"] <= nuc.P90_BAR_MS}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = report_md(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def report_md(r: dict) -> str:
    L = [f"# Training on cost, held-out dev, {r['eval']}", "", f"Pre-registration: `fraud/eval_lm_cost.py`. {len(r['windows'])} held-out dev chains. Dev only. Costs: {r['costs']}. All numbers Observed. Model scores are uncalibrated. Lambda from all 118 dev windows except logreg's, so these reads are flattered.", "",
         f"| setup | mean cost | 95% CI | AUC | frauds stopped | frauds of {BIG_USD:.0f} USD or more stopped | fraud USD stopped | good customers declined | reviews |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {x['auc']:.4f} | {x['frauds_stopped']} of {x['frauds']} | {x['big_frauds_stopped']} of {x['big_frauds']} | {x['fraud_usd_stopped']:,.0f} | {x['good_customers_declined']} | {x['reviews']} |")
    m = r["lm"][WEIGHTED]["latency_ms"]
    L += ["", f"lm-weighted latency on lone prompts: p50 {m['p50']:.0f} ms, p90 {m['p90']:.0f} ms, p99 {m['p99']:.0f} ms, bar 250 ms. Rank correlation with lm-rich: {r['rank_corr_with_lm_rich']:.4f}.", "", "## Answers", ""]

    def line(p):
        return f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer; mean paired difference {p['mean_diff']:,.0f} USD per window, 95% CI {p['mean_diff_ci95'][0]:,.0f} to {p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}"

    for k, v in r["verdict"].items():
        L += [f"- {k.capitalize()}: {line(p)}." for p in (v if isinstance(v, list) else [v])]
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
