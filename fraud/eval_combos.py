"""Combinations of what survived, held-out dev only: which language-model setups go to the final test → fraud/results/eval_combos/.

    python -m fraud.eval_combos    # from Phase 21's saved scores; no model, no test row; a few minutes

Traces: fraud/runs/<eval id>_eval_<setup>_hdev<k>.jsonl. Results: fraud/results/eval_combos/<eval id>.json + .md.

STATED BEFORE RUNNING (Phase 22; written and committed before lm-signal's and lm-frozen's scores existed. Observed by then, on held-out dev: tab-all 1,830, logreg 2,125, lm-rich 2,112, lm-seed1 2,186, lm-phase12 2,288).
One interim run came before the commit, on those three adapters, as a check of this code: 20261104-064340. It chose lm-rich and lm-rich with the customer line, and lm-rich+trees came out at 1,868 with a language-model share of 0.25. Nothing below was changed after it, and it could not show lm-signal, which is what the choice most depends on.
Owner, 2026-11-03: "combinations of survivors, dev only", and the final setup is a language decision model at heart. No new window is used, and none is left.
- Scorers: lm-rich; lm-signal; lm-two-seeds (mean logit of lm-phase12 and lm-seed1); lm-three (mean logit of lm-phase12, lm-seed1 and lm-rich); lm-rich-signal (mean logit of lm-rich and lm-signal); lm-rich+trees and lm-signal+trees (a logistic fit on Phase 14's 30 dev windows of the language model's logit and the tabular signal's logit). Each with Phase 14's rule, and each with Phase 18's customer line (flat 10 USD in the expected cost of declining, never in what it's charged).
- Rule: p's fit and two lines on Phase 14's 30 dev windows, lambda on all 118, as for trees and Phase 21. So the read on the 88 held-out chains is flattered. Logreg keeps Phase 14's procedure.
- Latency of a mean of N adapters is the sum of its members' p90 (Inferred: they would run one after another), and must stay under 250 ms.
- "At heart": a fitted combination counts as a language-model setup only if the language model carries at least half of the standardised weight in the fit (Implementation choice). A setup under that line is reported but can't go on.
- Choosing at most 3 for the final test (fixed here, Implementation choice):
  1. The lowest held-out rule cost among language-model setups under the latency bar.
  2. The same scorer with the customer line, the steadiest finding so far (Phases 18 and 20).
  3. The lowest rule cost among setups that read no tabular score, so the final test also prices a language model with no trees in it. If choice 1 already reads none, then the lowest rule cost among those that do.
  The baseline is logreg with Phase 14's rule. The trees are not candidates, by the owner's constraint.
- Stated before running (Inferred): lm-signal is cheapest, close to the trees. Means of adapters gain a little on their members, about 0.005-0.01 AUC. A fit with trees leans on the trees and fails the "at heart" line. The customer line cuts good customers declined by more than half, at no distinguishable cost. All of this is dev, whose ordering has failed to carry over in Phases 14, 18 and 20.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_ieee_amount as am, eval_lm_nuc as nuc, eval_tabular as et, ieee_data, rule, tabular as tb
from .baselines import LogReg
from .runs import stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_combos"
FLAT = 10.0
CUST = "-cust"
HEART_SHARE = 0.5
BASE = "tab-logreg"
TREES = "tab-all"


@dataclass(frozen=True)
class Combo:
    name: str
    members: tuple[str, ...]
    fitted: bool = False  # a logistic fit of the members' logits; otherwise their mean logit

    def adapters(self) -> tuple[str, ...]:
        return tuple(m for m in self.members if m in nuc.LM)


COMBOS = (Combo("lm-rich", ("lm-rich",)), Combo("lm-signal", ("lm-signal",)), Combo("lm-two-seeds", ("lm-phase12", "lm-seed1")), Combo("lm-three", ("lm-phase12", "lm-seed1", "lm-rich")), Combo("lm-rich-signal", ("lm-rich", "lm-signal")),
          Combo("lm-rich+trees", ("lm-rich", nuc.SIGNAL), True), Combo("lm-signal+trees", ("lm-signal", nuc.SIGNAL), True))
READS_TABULAR = {"lm-signal", "lm-rich-signal", "lm-rich+trees", "lm-signal+trees"}


def mean_logit(qs: list[np.ndarray]) -> np.ndarray:
    z = np.mean([am.logit(q) for q in qs], axis=0)
    return 1 / (1 + np.exp(-z))


def fit_blend(qs: list[np.ndarray], t, windows: list[ev.Win]) -> tuple[np.ndarray, list[float]]:
    """(the fitted score per row, each member's share of the standardised weight). NaN outside the scored rows stays NaN."""
    x = np.column_stack([am.logit(q) for q in qs])
    rows = np.concatenate([w.rows for w in windows])
    m = LogReg().fit(x[rows], t.is_fraud[rows].astype(float))
    w = np.abs(m.w[1:])
    q = np.full(len(t), np.nan)
    ok = ~np.isnan(qs[0])
    q[ok] = m.predict(x[ok])
    return q, (w / w.sum()).tolist()


def latency_p90(c: Combo, info: dict) -> float:
    return sum(info[m]["latency_ms"]["p90"] for m in c.adapters())


def is_language_model(shares: list[float] | None) -> bool:
    return shares is None or shares[0] >= HEART_SHARE


def choose(costs: dict[str, float], latency: dict[str, float], shares: dict[str, list[float] | None], reads_tabular=READS_TABULAR) -> list[str]:
    """The stated choice: the cheapest language-model setup; it with the customer line; the cheapest on the other side of "reads a tabular score"."""
    ok = [k for k in costs if latency[k] <= nuc.P90_BAR_MS and is_language_model(shares[k])]
    if not ok:
        return []
    first = min(ok, key=costs.get)
    other = [k for k in ok if (k in reads_tabular) != (first in reads_tabular)]
    return [first + tb.RULE, first + CUST] + ([min(other, key=costs.get) + tb.RULE] if other else [])


def run(log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    scores, info = {}, {}
    for k in nuc.LM:
        if (nuc.SCORES / f"{k}.npz").exists():
            scores[k], info[k] = nuc.saved(k, n)
    scores.update(nuc.references(t, log))
    eval_id, fits, setups_of = stamp(), {}, {}
    combos = [c for c in COMBOS if all(m in scores for m in c.members)]
    for k in (BASE, TREES):
        q = scores[k]
        pol = tb.fit_policy(q, t, dev)
        if k != BASE:
            pol.lam = et.lambda_on(pol, q, t, dev + held)
        tb.write_setup(eval_id, t, held, k + tb.RULE, q, pol, "rule")
        fits[k] = {"lambda": pol.lam, "shares": None, "latency_p90": 0.0}
    for c in combos:
        qs = [scores[m] for m in c.members]
        shares = None
        if c.fitted:
            q, shares = fit_blend(qs, t, dev)
        else:
            q = qs[0] if len(qs) == 1 else mean_logit(qs)
        pol = tb.fit_policy(q, t, dev)
        pol.lam = et.lambda_on(pol, q, t, dev + held)
        tb.write_setup(eval_id, t, held, c.name + tb.RULE, q, pol, "rule", "lm")
        head = {**pol.header(), "flat": FLAT}
        for w in held:
            outs = [{"by": "lm", "q": float(v), "ms": 0.0} for v in q[w.rows]]
            am.write_trace(eval_id, t, w, c.name + CUST, outs, None, rule.decide_paced(pol.p(q[w.rows]), t.amount[w.rows], pol.lam, ev.COSTS, 0.0, FLAT), head, {})
        fits[c.name] = {"lambda": pol.lam, "shares": shares, "latency_p90": latency_p90(c, info), "members": list(c.members), "fitted": c.fitted}
        log(f"{c.name}: {fits[c.name]}")
    names = [k + tb.RULE for k in (BASE, TREES)] + [c.name + b for c in combos for b in (tb.RULE, CUST)]
    tr = {s: tb.read_setup(eval_id, s, held) for s in names}
    by = {s: tb.window_costs(tr[s]) for s in names}
    setups = {s: tb.row(tr[s]) for s in names}
    base = BASE + tb.RULE
    verdict = {"against_logreg": [ev.compare(by, c.name + tb.RULE, base) for c in combos], "against_trees": [ev.compare(by, c.name + tb.RULE, TREES + tb.RULE) for c in combos], "customer_line": [ev.compare(by, c.name + CUST, c.name + tb.RULE) for c in combos]}
    names_c = [c.name for c in combos]
    chosen = choose({k: setups[k + tb.RULE]["mean_cost"] for k in names_c}, {k: fits[k]["latency_p90"] for k in names_c}, {k: fits[k]["shares"] for k in names_c})
    out = {"eval": eval_id, "script": "fraud/eval_combos.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in held},
           "fits": fits, "setups": setups, "verdict": verdict, "chosen": chosen, "scores_used": list(info)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = report_md(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def report_md(r: dict) -> str:
    L = [f"# Combinations, held-out dev, {r['eval']}", "", f"Stated before running: `fraud/eval_combos.py`. {len(r['windows'])} held-out dev chains. Dev only. Costs: {r['costs']}. All numbers Observed, except the latency of a mean, which is the sum of its members' p90 (Inferred). Model scores are uncalibrated. Lambda from all 118 dev windows except logreg's, so these reads are flattered.", "",
         "| setup | mean cost | 95% CI | AUC | frauds stopped | fraud USD stopped | good customers declined | reviews | lambda | p90 ms | language model's share |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        k = s.removesuffix(tb.RULE).removesuffix(CUST)
        f = r["fits"][k]
        share = f"{f['shares'][0]:.2f}" if f["shares"] else "–"
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {x['auc']:.4f} | {x['frauds_stopped']} of {x['frauds']} | {x['fraud_usd_stopped']:,.0f} | {x['good_customers_declined']} | {x['reviews']} | {f['lambda']:g} | {f['latency_p90']:.0f} | {share} |")

    def line(p):
        return f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer; mean paired difference {p['mean_diff']:,.0f} USD per window, 95% CI {p['mean_diff_ci95'][0]:,.0f} to {p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}"

    L += ["", "## Paired", ""]
    for k, v in r["verdict"].items():
        L += [f"- {k}: {line(p)}." for p in v]
    L += ["", f"Chosen for the final test by the stated rule: {', '.join(r['chosen']) or 'none'}."]
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
