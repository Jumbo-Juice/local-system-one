"""Where does the amount-aware rule stop winning? Phase 14's recorded scores under other cost models → fraud/results/eval_cost_stress/.

    python -m fraud.eval_cost_stress            # re-decide and re-price from the traces of eval 20261103-164830; no model runs

Results: fraud/results/eval_cost_stress/<run id>.json + .md. No new window and no new trace.

PRE-REGISTRATION (Phase 16; committed before anything was priced at a new cost).
Owner, 2026-11-03: "the rule and thresholds under other friction rates and minimums, and with a flat price per good customer declined, from recorded scores. Find where the rule stops
winning. This is what I most want to know." Why: the rule optimises the cost model it is scored on (Limit in fraud/RESULTS-IEEE-AMOUNT.md), and the true cost of a wrong decline isn't known.
- Scores, all from the laptop's traces, so no two machines are mixed: ieee-logreg-all and ieee-ft-ieee-full on Phase 14's 30 dev windows and 20 test windows amt0-amt19. These test windows have been read. Nothing here is
  chosen on them: every line or lambda comes from dev. They stay usable because no setup is being changed by what they show here.
- Two views of each cost set, both reported, because a business may or may not know its true friction:
  - "As deployed": Phase 14's decisions unchanged (rule at lambda 10 assuming 10%, thresholds 0.0586 / 0.1141), only the prices differ.
  - "Re-picked": each setup knows the costs. Lines re-picked on the 30 dev windows (ev.pick_thresholds); the rule uses the new costs in its expected-cost sums and lambda is re-picked on dev. The logistic fit of p is kept: it doesn't use costs.
- Cost sets, each changing one thing from Phase 14's (friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed):
  friction rate in FRICTION_RATES; friction minimum in FRICTION_MINS; flat price per good customer declined in FLATS, on top of friction; flat price alone with no percentage in FLATS_ONLY. Then, past the owner's list, because the same doubt applies:
  review fee in FEES and review share in SHARES. Then one grid of friction rate by flat price, GRID_RATES x GRID_FLATS, as deployed and re-picked, for logreg only.
- The question: at each cost set, logreg with the rule against logreg on thresholds, ev.compare (95% window bootstrap of the mean paired difference, seed 20261102). "Stops winning" is stated two ways: the first set where the rule's mean is no longer lower, and the first where the
  interval no longer excludes 0. Reported for logreg (primary) and the adapter. Many sets are compared, not corrected for multiple comparisons, so these are a map and not separate findings.
- Reported with each: frauds stopped, good customers declined, fraud amount stopped.
- Stated before running (Inferred), from Phase 14's table. As deployed, the rule declines 982 good customers worth 89,680 USD and thresholds 655 worth 131,400 USD. So:
  (1) a higher friction RATE should favour the rule more, because its declined dollars are fewer. A lower one should narrow the gap, but the rule should still win on fraud let through.
  (2) a higher friction MINIMUM and a flat price hurt the rule, because it declines 327 more customers. As deployed the 16,504 USD saving over 20 windows should be gone near 16,504 / 327 = about 50 USD a customer, and stop being distinguishable well before that, perhaps near 20-30.
  (3) re-picked, the rule should win further out than as deployed, and possibly everywhere, as it can raise its decline line when declines get dearer.
  (4) a dearer review should narrow the gap but the rule should keep winning, because it uses fewer reviews (702 against 944) on bigger payments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, eval_ieee as ev, eval_ieee_amount as am, ieee_data
from .runs import read_run, stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_cost_stress"
EVAL = "20261103-164830"
SCORERS = {"logreg": ("lr", am.LR, am.LR_AMT), "adapter": ("ft", am.FT, am.FT_AMT)}
BASE = ev.COSTS

FRICTION_RATES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)
FRICTION_MINS = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0)
FLATS = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0)
FLATS_ONLY = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0)
FEES = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0)
SHARES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.20)
GRID_RATES = (0.01, 0.05, 0.10, 0.20, 0.50)
GRID_FLATS = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def cost_sets() -> list[tuple[str, str, float, costs.Costs]]:
    """(family, label, value, costs). Each family changes one thing from Phase 14's costs."""
    b = BASE
    out = [("friction rate", f"{v:g}", v, costs.Costs(v, b.friction_min, b.review_fee, b.review_share)) for v in FRICTION_RATES]
    out += [("friction minimum", f"{v:g}", v, costs.Costs(b.friction_rate, v, b.review_fee, b.review_share)) for v in FRICTION_MINS]
    out += [("flat per good customer declined", f"{v:g}", v, costs.Costs(b.friction_rate, b.friction_min, b.review_fee, b.review_share, v)) for v in FLATS]
    out += [("flat only, no percentage", f"{v:g}", v, costs.Costs(0.0, 0.0, b.review_fee, b.review_share, v)) for v in FLATS_ONLY]
    out += [("review fee", f"{v:g}", v, costs.Costs(b.friction_rate, b.friction_min, v, b.review_share)) for v in FEES]
    out += [("review share", f"{v:g}", v, costs.Costs(b.friction_rate, b.friction_min, b.review_fee, v)) for v in SHARES]
    return out


def grid_sets() -> list[tuple[float, float, costs.Costs]]:
    return [(r, f, costs.Costs(r, BASE.friction_min, BASE.review_fee, BASE.review_share, f)) for r in GRID_RATES for f in GRID_FLATS]


def counts(c: costs.Costs, wanted, fallback, y, a) -> dict:
    """One setup's money and customers over the windows, scored only by costs.score."""
    per, stopped, stopped_usd, good = [], 0, 0.0, 0
    for w, f, yy, aa in zip(wanted, fallback, y, a):
        s = costs.score(c, w, f, yy.astype(bool), aa)
        per.append(s.summary["cost"])
        m = s.summary["confusion"]
        stopped += m["fraud"]["review"] + m["fraud"]["decline"]
        good += m["legit"]["decline"]
        stopped_usd += s.summary["fraud_amount_stopped"]
    return {"per": np.array(per), "frauds_stopped": stopped, "fraud_usd_stopped": stopped_usd, "good_declined": good}


class Scorer:
    """One scorer's recorded dev and test scores, its fit of p, and Phase 14's decisions from them."""

    def __init__(self, key: str, setup: str, setup_amt: str, d: dict, test_view: dict, tests_amt: list[list[dict]], tests_thr: list[list[dict]]):
        self.key, self.setup, self.setup_amt, self.d, self.v = key, setup, setup_amt, d, test_view
        self.m, self.lam = am.fit_rule(d, setup_amt)
        self.p_test = [self.m.predict(am.logit(q)[:, None]) for q in test_view["scores"]]
        self.p_dev = [self.m.predict(k) for k in am.features(d, setup_amt)]
        pick = lambda recs, f: [[r[f] for r in x if r.get("type") == "tx"] for x in recs]
        self.dep_rule = pick(tests_amt, "wanted"), pick(tests_amt, "fallback")
        self.dep_thr = pick(tests_thr, "wanted"), pick(tests_thr, "fallback")

    def deployed(self, c: costs.Costs) -> tuple[dict, dict]:
        y, a = self.v["truth"], self.v["amount"]
        return counts(c, *self.dep_rule, y, a), counts(c, *self.dep_thr, y, a)

    def repicked(self, c: costs.Costs) -> tuple[dict, dict, dict]:
        y, a = self.v["truth"], self.v["amount"]
        lam = am.pick_lambda(self.p_dev, self.d["truth"], self.d["amount"], c)[0]
        pairs = [am.decide(p, aa, lam, c) for p, aa in zip(self.p_test, a)]
        rule = counts(c, [x[0] for x in pairs], [x[1] for x in pairs], y, a)
        tr, td, _ = ev.pick_thresholds(self.d[self.key], self.d["truth"], self.d["amount"], c)
        wanted = [[ev.act(float(v), tr, td) for v in q] for q in self.v["scores"]]
        thr = counts(c, wanted, [["approve"] * len(q) for q in wanted], y, a)
        return rule, thr, {"lambda": lam, "thresholds": [tr, td]}


def line(rule: dict, thr: dict, extra: dict | None = None) -> dict:
    p = ev.compare({"rule": rule["per"], "thr": thr["per"]}, "rule", "thr")
    diff = float((rule["per"] - thr["per"]).mean())
    lo, hi = ev.boot_ci(rule["per"] - thr["per"])
    out = {"rule_cost": float(rule["per"].mean()), "thresholds_cost": float(thr["per"].mean()), "diff": diff, "ci": [lo, hi], "rule_cheaper_windows": int((rule["per"] < thr["per"]).sum()),
           "rule_wins": diff < 0, "distinguishable": bool(hi < 0 or lo > 0), "better": p["better"],
           "rule_good_declined": rule["good_declined"], "thresholds_good_declined": thr["good_declined"], "rule_frauds_stopped": rule["frauds_stopped"], "thresholds_frauds_stopped": thr["frauds_stopped"],
           "rule_fraud_usd_stopped": rule["fraud_usd_stopped"], "thresholds_fraud_usd_stopped": thr["fraud_usd_stopped"]}
    return {**out, **(extra or {})}


def stops(rows: list[dict], view: str) -> dict:
    """In a family's order: the first value where the rule stops winning on the mean, and where it stops being distinguishably cheaper."""
    no_mean = next((r["value"] for r in rows if not r[view]["rule_wins"]), None)
    no_ci = next((r["value"] for r in rows if not (r[view]["rule_wins"] and r[view]["distinguishable"])), None)
    return {"first_not_lower": no_mean, "first_not_distinguishably_lower": no_ci}


def run(run_id: str, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    test = am.test_windows(n)
    d = am.dev_scores(EVAL, n)
    scorers = {}
    for name, (key, setup, setup_amt) in SCORERS.items():
        thr = [read_run(ev.trace_file(EVAL, "eval", setup, w.id)) for w in test]
        amt = [read_run(ev.trace_file(EVAL, "eval", setup_amt, w.id)) for w in test]
        scorers[name] = Scorer(key, setup, setup_amt, d, ev.view(thr), amt, thr)
    out = {"run": run_id, "script": "fraud/eval_cost_stress.py", "scores_from_eval": EVAL, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "base_costs": BASE.as_dict(),
           "fit": {k: {"lambda": s.lam} for k, s in scorers.items()}, "families": {}, "stops": {}, "grid": []}
    t0 = time.perf_counter()
    sets = cost_sets()
    for k, (family, label, value, c) in enumerate(sets):
        row = {"value": value, "label": label, "costs": c.as_dict()}
        for name, s in scorers.items():
            rule, thr = s.deployed(c)
            row[f"{name}_deployed"] = line(rule, thr)
            rule, thr, extra = s.repicked(c)
            row[f"{name}_repicked"] = line(rule, thr, extra)
        out["families"].setdefault(family, []).append(row)
        log(f"[{k + 1}/{len(sets)}] {family} {label}: logreg deployed {row['logreg_deployed']['diff']:,.0f} repicked {row['logreg_repicked']['diff']:,.0f} ({time.perf_counter() - t0:.0f} s)")
    for family, rows in out["families"].items():
        ordered = sorted(rows, key=lambda r: r["value"])
        out["stops"][family] = {f"{name}_{view}": {"below_base": stops(sorted([r for r in ordered if r["value"] <= base_value(family)], key=lambda r: -r["value"]), f"{name}_{view}"),
                                                   "above_base": stops([r for r in ordered if r["value"] >= base_value(family)], f"{name}_{view}")}
                                for name in scorers for view in ("deployed", "repicked")}
    s = scorers["logreg"]
    for r, f, c in grid_sets():
        rule, thr = s.deployed(c)
        rp, tp, extra = s.repicked(c)
        out["grid"].append({"rate": r, "flat": f, "deployed": line(rule, thr), "repicked": line(rp, tp, extra)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{run_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{run_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def base_value(family: str) -> float:
    return {"friction rate": BASE.friction_rate, "friction minimum": BASE.friction_min, "flat per good customer declined": 0.0, "flat only, no percentage": 1.0,
            "review fee": BASE.review_fee, "review share": BASE.review_share}[family]


def _cell(x: dict) -> str:
    mark = "**" if x["distinguishable"] and x["rule_wins"] else ""
    return f"{mark}{x['diff']:,.0f}{mark} ({x['ci'][0]:,.0f} to {x['ci'][1]:,.0f})"


def markdown(r: dict) -> str:
    L = [f"# Cost-model stress {r['run']}", "", f"Scores from eval `{r['scores_from_eval']}`, 20 test windows, recorded. Mean cost per window of the rule minus thresholds, with its 95% CI. Negative means the rule is cheaper; bold means distinguishably cheaper. All numbers Observed.", ""]
    for family, rows in r["families"].items():
        L += [f"## {family}", "", "| value | logreg, as deployed | good declined rule / lines | logreg, re-picked | good declined rule / lines | lambda | adapter, as deployed | adapter, re-picked |", "|---:|---|---|---|---|---:|---|---|"]
        for x in rows:
            a, b, c, e = x["logreg_deployed"], x["logreg_repicked"], x["adapter_deployed"], x["adapter_repicked"]
            L.append(f"| {x['label']} | {_cell(a)} | {a['rule_good_declined']} / {a['thresholds_good_declined']} | {_cell(b)} | {b['rule_good_declined']} / {b['thresholds_good_declined']} | {b['lambda']} | {_cell(c)} | {_cell(e)} |")
        L.append("")
    L += ["## Friction rate by flat price, logreg", "", "| rate | flat | as deployed | re-picked |", "|---:|---:|---|---|"]
    for g in r["grid"]:
        L.append(f"| {g['rate']:g} | {g['flat']:g} | {_cell(g['deployed'])} | {_cell(g['repicked'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    run(stamp(), lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
