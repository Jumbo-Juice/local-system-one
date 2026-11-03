"""A better use of the amount-aware rule: calibration, a review price that follows the budget, a decline line that counts customers → fraud/results/eval_rule/.

    python -m fraud.eval_rule                     # fit on dev, score 10 new windows, report
    python -m fraud.eval_rule --report <eval id>  # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_eval_<setup>_rule<k>.jsonl. Results: fraud/results/eval_rule/<eval id>.json + .md, and the dev looks dev_look*.json.

PRE-REGISTRATION (Phase 18; committed before any setup here scored a new window).
Owner, 2026-11-03: "a calibration that fixes the p overshoot (numpy only), lambda or budget use aware of how many reviews remain in the window, a decline line that counts customers as well as dollars."
- Scores, all from this machine, one scorer: ieee-logreg-all, refitted on fit rows, which must reproduce Phase 14's dev scores. New windows: chains rule0-rule9 from fraud/final_test.py, 10,000 rows. Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed. A flat price changes what a setup decides, never what it's charged.
- Observed before registering, dev rows only (dev_look.json, dev_look_combined.json): each fitted on Phase 14's 30 dev windows and read on 88 held-out dev chains, base 2,125 per window, 4,003 good customers declined, two lines 3,085 and 2,619. Sum of p 3,672 against 3,409 frauds on held-out, and isotonic (3,680) and a fit with the amount (3,674) don't change that, because the level comes from dev's higher fraud rate. An EM prior from the recent 5,000 rows was worse (+143, CI 33 to 258) and is dropped.
  Pacing lambda and gamma on dev: gamma 1 lam 5, -92, CI -167 to -21. A flat price on declining, lambda 10: 1 USD -29 and 3,095 customers; 2 USD -41 and 2,539; 5 USD -28 and 1,635; 10 USD +3 and 933; 20 USD +132 and 380. Combined, amount fit gamma 1 flat 3 lam 5: -94, CI -168 to -24, 1,961 customers. These read held-out after seeing dev; the combined one was picked from among those with both held-out and dev read, so its dev gain is a little flattered.
- Setups on the new windows, each from the same scores:
  - rule-base: Phase 14's rule, its logistic fit, lambda 10.
  - rule-longrun: the same fit moved by rule.shift_p from dev30's fraud rate to the rate over all fit and dev rows, lambda re-picked on dev30. The calibration. Moves the level only, keeps the ranking.
  - rule-paced: Phase 14's fit, lambda_i = 5 * pace ** -1 (fraud/rule.py), where pace is the reviews left per row left against the budget's rate.
  - rule-combined: a logistic fit of the score's logit and log amount on dev30, paced with gamma 1, lambda 5, and a flat price of 3 USD in the expected cost of declining.
  - rule-customers: Phase 14's fit, lambda 10, a flat price of 10 USD in the expected cost of declining. For a business that counts customers.
  - rule-lines: two lines on the score, re-picked on dev30. ieee-approve-all.
- Primary question: rule-combined against rule-base on total cost. Distinguishable only if the 95% bootstrap CI of the mean paired difference (windows resampled, seed 20261102) excludes 0.
- Second: rule-paced against rule-base. Third: rule-longrun against rule-base, and the sum of p against frauds for each. Fourth: rule-customers against rule-base, on cost, and on good customers declined. Not corrected for multiple comparisons beyond the primary.
- Also reported, no bar: frauds stopped, fraud USD stopped, good customers declined, reviews; every setup re-priced at a true flat price per good customer of 0, 10, 20, 30 and 50 USD, its decisions unchanged. The same setups on held-out dev, biased as said above.
- Limits stated now: 10 windows give wide intervals. pace needs a window's size up front, which a real day's traffic only estimates. The rule still reads every column, so Phase 17's doubt applies. A flat price here is a knob, not a measured cost.
- Stated before running (Inferred): rule-combined has a lower mean than rule-base by less than dev's 94 USD, and this isn't distinguishable on 10 windows, but it declines about half the good customers. rule-paced is lower, not distinguishable. rule-longrun closes much of the overshoot when the window's fraud rate is low, and moves cost little. rule-customers costs about the same as rule-base, within noise, and declines about a quarter of the good customers, stopping about 15% fewer frauds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import costs, eval_ieee as ev, eval_ieee_amount as am, final_test as ft, ieee_data, rule, tabular as tb
from .baselines import LogReg
from .runs import read_run, stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_rule"
PHASE14 = "20261103-164830"
BASE, LONGRUN, PACED, COMBINED, CUSTOMERS, LINES = "rule-base", "rule-longrun", "rule-paced", "rule-combined", "rule-customers", "rule-lines"
RULES = (BASE, LONGRUN, PACED, COMBINED, CUSTOMERS)
SETUPS = (*RULES, LINES, ev.APPROVE_ALL)
FLAT_PRICES = (0.0, 10.0, 20.0, 30.0, 50.0)


def x_logit(q: np.ndarray, amount: np.ndarray) -> np.ndarray:
    return am.logit(q)[:, None]


def x_amount(q: np.ndarray, amount: np.ndarray) -> np.ndarray:
    return np.c_[am.logit(q), np.log1p(amount)]


@dataclass
class Spec:
    """A rule: how p is made, and the rule's three knobs. Everything a trace header needs to replay it."""
    fit: LogReg
    lam: float
    gamma: float = 0.0
    flat: float = 0.0
    with_amount: bool = False
    shift: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    def p(self, q: np.ndarray, amount: np.ndarray) -> np.ndarray:
        out = self.fit.predict((x_amount if self.with_amount else x_logit)(q, amount))
        return rule.shift_p(out, *self.shift) if self.shift else out

    def decide(self, q: np.ndarray, amount: np.ndarray, c: costs.Costs = ev.COSTS):
        return rule.decide_paced(self.p(q, amount), amount, self.lam, c, self.gamma, self.flat)

    def header(self) -> dict:
        h = {"kind": "amount-aware", "lambda": self.lam, "fit_weights": self.fit.w.tolist(), "fit_mean": self.fit.mu.tolist(), "fit_sd": self.fit.sd.tolist(), "fit_features": ["logit", "log_amount"] if self.with_amount else ["logit"]}
        h.update({k: v for k, v in (("gamma", self.gamma), ("flat", self.flat), ("shift", list(self.shift) if self.shift else None)) if v})
        return h


def total_cost(spec: Spec, q: np.ndarray, t, windows: list[ev.Win]) -> float:
    total = 0.0
    for w in windows:
        a = t.amount[w.rows]
        total += costs.score(ev.COSTS, *spec.decide(q[w.rows], a), t.is_fraud[w.rows].astype(bool), a).summary["cost"]
    return total


def specs(q: np.ndarray, t, dev: list[ev.Win]) -> tuple[dict[str, Spec], tb.Policy]:
    """Every rule, fitted on the dev windows only, as registered."""
    n = len(t)
    y = np.concatenate([t.is_fraud[w.rows] for w in dev]).astype(float)
    pol = tb.fit_policy(q, t, dev)
    x = np.concatenate([x_amount(q[w.rows], t.amount[w.rows]) for w in dev])
    amount_fit = LogReg().fit(x, y)
    dev_rate = float(y.mean())
    long_rate = float(t.is_fraud[:ieee_data.split_bounds(n)[1]].mean())
    longrun = Spec(pol.m, 0.0, shift=(dev_rate, long_rate))
    longrun.lam = min(am.LAMBDAS, key=lambda k: total_cost(Spec(pol.m, k, shift=(dev_rate, long_rate)), q, t, dev))
    return {BASE: Spec(pol.m, pol.lam), LONGRUN: longrun, PACED: Spec(pol.m, 5.0, gamma=1.0), COMBINED: Spec(amount_fit, 5.0, gamma=1.0, flat=3.0, with_amount=True),
            CUSTOMERS: Spec(pol.m, pol.lam, flat=10.0)}, pol


def sum_p(spec: Spec, q: np.ndarray, t, windows: list[ev.Win]) -> float:
    return float(sum(spec.p(q[w.rows], t.amount[w.rows]).sum() for w in windows))


def run(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    fit_rows = np.arange(ieee_data.split_bounds(n)[0])
    dev, held, new = tb.dev30(n), tb.heldout_dev(n), ft.phase_windows(n, "rule")
    ft.guard(n, np.concatenate([w.rows for w in new]))
    t0 = time.perf_counter()
    lr = LogReg().fit(ev.features_all(t, fit_rows), t.is_fraud[fit_rows].astype(float))
    q = tb.scores_for(lr, ev.features_all, t, dev + held + new)
    gap = max(float(np.abs(q[w.rows] - ev.view([read_run(ev.trace_file(PHASE14, "dev", am.LR, w.id))])["scores"][0]).max()) for w in dev)
    if gap > 1e-6:
        raise SystemExit(f"logreg differs from Phase 14's by {gap}; nothing else was run")
    log(f"logreg fitted in {time.perf_counter() - t0:.0f} s, matches Phase 14's dev scores, largest gap {gap:.1e}")
    sp, pol = specs(q, t, dev)
    held_out = {}
    y_held = float(sum(t.is_fraud[w.rows].sum() for w in held))
    for name, s in sp.items():
        log(f"{name}: lambda {s.lam}, gamma {s.gamma}, flat {s.flat}, shift {s.shift}")
        for w in new:
            outs = [{"by": "logreg", "q": float(v), "ms": 0.0} for v in q[w.rows]]
            am.write_trace(eval_id, t, w, name, outs, None, s.decide(q[w.rows], t.amount[w.rows]), s.header(), {})
        per, good, stopped = [], 0, 0
        for w in held:
            a = t.amount[w.rows]
            r = costs.score(ev.COSTS, *s.decide(q[w.rows], a), t.is_fraud[w.rows].astype(bool), a).summary
            per.append(r["cost"]),
            good += r["confusion"]["legit"]["decline"]
            stopped += r["confusion"]["fraud"]["decline"] + r["confusion"]["fraud"]["review"]
        held_out[name] = {"mean": float(np.mean(per)), "good_customers_declined": good, "frauds_stopped": stopped, "sum_p": sum_p(s, q, t, held), "frauds": y_held, "per": per}
    tb.write_setup(eval_id, t, new, LINES, q, pol, "lines")
    for w in new:
        am.write_trace(eval_id, t, w, ev.APPROVE_ALL, [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows], (None, None), None, None, {})
    base = np.array(held_out[BASE]["per"])
    for name, h in held_out.items():
        d = np.array(h.pop("per")) - base
        h["diff_to_base"], h["diff_ci"] = float(d.mean()), list(ev.boot_ci(d))
    sums = {name: sum_p(s, q, t, new) for name, s in sp.items()}
    return report(eval_id, {"sum_p_new": sums, "held_out": held_out, "params": {name: s.header() for name, s in sp.items()}}, log)


def report(eval_id: str, info=None, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    new = ft.phase_windows(n, "rule")
    path = RESULTS / f"{eval_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info = info or old.get("info", {})
    tr = {s: tb.read_setup(eval_id, s, new) for s in SETUPS}
    rows = {s: tb.row(tr[s], s != ev.APPROVE_ALL) for s in SETUPS}
    by = {s: tb.window_costs(tr[s]) for s in SETUPS}
    verdict = {"primary": ev.compare(by, COMBINED, BASE), "second": ev.compare(by, PACED, BASE), "third": ev.compare(by, LONGRUN, BASE), "fourth": ev.compare(by, CUSTOMERS, BASE),
               "reported": [ev.compare(by, COMBINED, LINES), ev.compare(by, CUSTOMERS, LINES), ev.compare(by, BASE, LINES), ev.compare(by, COMBINED, ev.APPROVE_ALL)]}
    flat = {}
    for s in (*RULES, LINES):
        flat[s] = {}
        for f in FLAT_PRICES:
            c = costs.Costs(ev.COSTS.friction_rate, ev.COSTS.friction_min, ev.COSTS.review_fee, ev.COSTS.review_share, f)
            per = []
            for recs in tr[s]:
                x = [r for r in recs if r.get("type") == "tx"]
                per.append(costs.score(c, [r["wanted"] for r in x], [r["fallback"] for r in x], np.array([r["truth"] for r in x]).astype(bool), np.array([r["tx"]["amount"] for r in x])).summary["cost"])
            flat[s][f"{f:g}"] = {"mean": float(np.mean(per)), "per": per}
    for s in (*RULES, LINES):
        base = np.array(flat[BASE]["0"]["per"])
        for f, v in flat[s].items():
            d = np.array(v["per"]) - np.array(flat[BASE][f]["per"])
            v["diff_to_base"], v["ci"] = float(d.mean()), list(ev.boot_ci(d))
            v.pop("per") if s != BASE else None
    for v in flat[BASE].values():
        v.pop("per", None)
    y = np.concatenate(ev.view(tr[BASE])["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_rule.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in new},
           "test_rows": int(len(y)), "test_fraud": int(y.sum()), "info": info, "setups": rows, "verdict": verdict, "true_flat_price": flat}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def markdown(r: dict) -> str:
    L = [f"# Rule eval {r['eval']}", "", f"Pre-registration: `fraud/eval_rule.py`. {r['test_rows']:,} new test rows, {r['test_fraud']} fraud, {len(r['windows'])} windows. Costs: {r['costs']}. All numbers Observed. Model scores are uncalibrated.", "",
         "| setup | mean cost | 95% CI | frauds stopped | fraud USD stopped | good declined | reviews | fraud let through | good declined USD | review fees | sum of p |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sums = r["info"].get("sum_p_new", {})
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        sp = f"{sums[s]:,.0f}" if s in sums else ""
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {x['frauds_stopped']} of {x['frauds']} | {x['fraud_usd_stopped']:,.0f} | {x['good_customers_declined']} | {x['reviews']} | {x['fraud_let_through']:,.0f} | {x['good_declined']:,.0f} | {x['review_fees']:,.0f} | {sp} |")

    def line(p):
        return f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer; mean paired difference {p['mean_diff']:,.0f} USD per window, 95% CI {p['mean_diff_ci95'][0]:,.0f} to {p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}"

    v = r["verdict"]
    L += ["", "## Answers", "", f"- Primary: {line(v['primary'])}.", f"- Second: {line(v['second'])}.", f"- Third: {line(v['third'])}.", f"- Fourth: {line(v['fourth'])}.", *[f"- Reported: {line(p)}." for p in v["reported"]]]
    L += ["", "## Re-priced at a true flat price per good customer declined, decisions unchanged, mean per window", "", "| setup | " + " | ".join(f"{f:g} USD" for f in FLAT_PRICES) + " |", "|---|" + "---:|" * len(FLAT_PRICES)]
    for s, x in r["true_flat_price"].items():
        L.append(f"| {s} | " + " | ".join(f"{x[f'{f:g}']['mean']:,.0f}" for f in FLAT_PRICES) + " |")
    if r["info"].get("held_out"):
        L += ["", "## Held-out dev, 88 chains, no bar; partly used to pick these setups", "", "| setup | mean | against rule-base | 95% CI | good declined | frauds stopped | sum of p | frauds |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        for s, x in r["info"]["held_out"].items():
            L.append(f"| {s} | {x['mean']:,.0f} | {x['diff_to_base']:,.0f} | {x['diff_ci'][0]:,.0f} to {x['diff_ci'][1]:,.0f} | {x['good_customers_declined']} | {x['frauds_stopped']} | {x['sum_p']:,.0f} | {x['frauds']:,.0f} |")
    return "\n".join(L) + "\n"


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
