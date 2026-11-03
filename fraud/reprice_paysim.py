"""PaySim's older results at a 10 USD review, from the recorded scores → fraud/results/reprice_paysim/.

    python -m fraud.reprice_paysim --gate     # only check that the $50 numbers reproduce; prices nothing at 10
    python -m fraud.reprice_paysim            # the gate, then both views at 10 USD, then the report

No model runs and no trace is written or changed. Results: fraud/results/reprice_paysim/<id>.json + .md.

PRE-REGISTRATION (Phase 13; committed before anything was priced at 10 USD).
Owner, 2026-11-03: re-price the older PaySim results at 10 USD. (Owner, 2026-11-02: a review costs 10 for all, the default for every review.)
- What is re-priced: the test windows of Phases 4, 5, 7 and 9 (evals 20261003-052158, 20261005-022333, 20261021-155222, 20261102-162345), and the dev results of
  Phases 6 and 8 (20261021-020632, 20261030-122539), which never reached test. Not re-priced: Phase 10 (eval_chain, scored by F1, no USD) and reviewer_check (exploratory, dev).
- Costs: fraud.costs.DEFAULT. Only the review fee changes, 50 → 10. Friction stays 10% of the amount, at least 10, and the budget stays 5%. The committed results stay as they are,
  at fraud.costs.PHASES_1_10; these are new files beside them. Bars recorded in those evals stand as recorded; this reports what they would read at 10 USD.
- Two views, both reported:
  - Same decisions: every recorded decision stands and only the price changes. Pure arithmetic.
  - Re-picked (the headline where it differs, as Phase 11's --reprice): review and decline thresholds re-picked on dev0-dev9 at 10 USD with each setup's own grid and tie rule, from the
    recorded dev scores; test rows re-decided from the recorded test scores. Setups: logreg (dev traces of eval 20261021-155222, same fit, grid of fraud/eval.py), hybrid-ft, hybrid-d3ft,
    hybrid-d1ft (their own dev traces and chosen order, grid of fraud/eval_ft.py). The rest have no thresholds, so both views are the same: rules, random, approve-all and hybrid-1.5b (argmax).
- Gate, before any price at 10: at 50 USD this code must reproduce every committed test-window cost to 0.01 USD, every recorded test action, the committed thresholds of hybrid-ft, hybrid-d3ft,
  hybrid-d1ft and of the dev-only models, and their committed best dev cost. Natural-rate cost must be within 0.5% (trace weights are rounded). If the gate fails nothing is priced.
- Reported, no new bar: mean and median cost, natural-rate cost, recall, precision, alert rate, reviews, where the money goes; the bars at 10 USD; paired window counts, sign tests and the
  mean paired difference with its 95% window bootstrap (distinguishable if it excludes 0, the rule since Phase 11).
- Limits stated now: these test windows have all been read, but nothing is tuned on them, since thresholds come from dev only and the price is the owner's. There are no new windows, because
  new ones would need the models re-run. Reviews are still assumed to find the truth, at a flat assumed fee. PaySim's windows are enriched to about 5% fraud.
- Stated before running (Inferred): hybrid-ft, hybrid-d3ft, hybrid-d1ft and rules pay almost only review fees, so they fall to about a fifth (about 170 per window). logreg and hybrid-1.5b lose
  mostly to fraud or friction, so they barely move. Bars and the order of setups stay the same. Thresholds move little, since hybrid-ft's review line is already the lowest on its grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, eval as ev, eval_ft as ft, eval_ieee as ie
from .eval_d3 import logreg_scores_view, variant_view
from .runs import read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "reprice_paysim"
PHASE4, PHASE5, PHASE6, PHASE7, PHASE8, PHASE9 = ("20261003-052158", "20261005-022333", "20261021-020632", "20261021-155222",
                                                  "20261030-122539", "20261102-162345")
RESULT_FILES = {PHASE4: "eval", PHASE5: "eval", PHASE6: "eval_d3", PHASE7: "eval_d3ft", PHASE8: "eval_lfm", PHASE9: "eval_d1ft"}
TEST_SETS = {
    "test100-119": {"windows_of": PHASE4, "setups": {s: PHASE4 for s in ("hybrid-1.5b", "rules", "logreg", "random", "approve-all")}},
    "test200-233": {"windows_of": PHASE5, "setups": {"hybrid-d1ft": PHASE9, "hybrid-d3ft": PHASE7, "hybrid-ft": PHASE5, "rules": PHASE5, "logreg": PHASE5,
                                                     "hybrid-1.5b": PHASE5, "random": PHASE5, "approve-all": PHASE5}},
}
LOGREG_DEV = PHASE7
TUNED = {"hybrid-ft": PHASE5, "hybrid-d3ft": PHASE7, "hybrid-d1ft": PHASE9}
DEV_ONLY = {"d3-nano": (PHASE6, "hybrid-d3nano"), "d3-mini": (PHASE6, "hybrid-d3mini"), "d1-omni-2b": (PHASE8, "hybrid-d1omni2b")}
FROZEN_DEV = (PHASE5, "hybrid-1.5b")
BARS = {
    "Phase 4 bar 1": ("test100-119", "hybrid-1.5b", "rules", 16, "strict"),
    "Phase 5 bar A": ("test200-233", "hybrid-ft", "logreg", 10, "at most"),
    "Phase 7 bar A": ("test200-233", "hybrid-d3ft", "logreg", 10, "at most"),
    "Phase 7 bar C": ("test200-233", "hybrid-d3ft", "hybrid-ft", 10, "at most"),
    "Phase 9 bar A": ("test200-233", "hybrid-d1ft", "logreg", 10, "at most"),
    "Phase 9 bar C": ("test200-233", "hybrid-d1ft", "hybrid-ft", 10, "at most"),
}
PAIRS = {
    "test100-119": (("hybrid-1.5b", "rules"), ("hybrid-1.5b", "logreg"), ("logreg", "rules")),
    "test200-233": (("hybrid-d1ft", "logreg"), ("hybrid-d1ft", "rules"), ("hybrid-d1ft", "hybrid-ft"), ("hybrid-d3ft", "logreg"), ("hybrid-d3ft", "hybrid-ft"),
                    ("hybrid-ft", "logreg"), ("hybrid-ft", "rules"), ("logreg", "rules"), ("rules", "approve-all")),
}
TOL_USD, TOL_NATURAL = 0.01, 0.005


def committed(eval_id: str) -> dict:
    return json.loads((ROOT / "results" / RESULT_FILES[eval_id] / f"{eval_id}.json").read_text(encoding="utf-8"))


def tx_rows(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("type") == "tx"]


def score_of(r: dict) -> float:
    return r["scores"]["fraud"] if r["by"] == "logreg" else r["scores"]["decline"]


def redecide(recs: list[dict], c: costs.Costs, thresholds: tuple[float, float] | None = None) -> costs.Scored:
    """One window at costs c. Without thresholds every recorded decision stands; with them, the model or logreg rows are decided again from their recorded scores."""
    tx = tx_rows(recs)
    if thresholds is None:
        wanted = [r["wanted"] for r in tx]
    else:
        tr, td = thresholds
        wanted = [("decline" if score_of(r) >= td else "review" if score_of(r) >= tr else "approve") if r["by"] in ("model", "logreg") else r["wanted"] for r in tx]
    return costs.score(c, wanted, [r["fallback"] for r in tx], np.array([bool(r["truth"]) for r in tx]), np.array([float(r["tx"]["amount"]) for r in tx]),
                       np.array([float(r["weight"]) for r in tx]))


def pick(view: dict, trs, tds, c: costs.Costs) -> tuple[float, float, float]:
    """(review, decline, total dev cost): the lowest total over the dev windows, first in grid order on a tie (as fraud/eval.py and fraud/eval_ft.py)."""
    best = None
    for tr in trs:
        for td in tds:
            if td < tr:
                continue
            total = 0.0
            for q, y, a, f, b in zip(view["scores"], view["truth"], view["amount"], view["filt"], view["fallbacks"]):
                wanted = ["approve" if fk else "decline" if qk >= td else "review" if qk >= tr else "approve" for qk, fk in zip(q, f)]
                fb = ["approve" if fk else bk for bk, fk in zip(b, f)]
                total += costs.score(c, wanted, fb, y, a).summary["cost"]
            if best is None or total < best[2]:
                best = (tr, td, total)
    return best


def split(s: costs.Scored, recs: list[dict]) -> dict:
    """Where one window's money went, total USD."""
    tx = tx_rows(recs)
    y = np.array([bool(r["truth"]) for r in tx])
    act = np.array(s.final)
    return {"fraud_let_through": float(s.cost[(act == "approve") & y].sum()), "good_declined": float(s.cost[(act == "decline") & ~y].sum()),
            "review_fees": float(s.cost[act == "review"].sum())}


# ---------------------------------------------------------------- loading

def traces_of(eval_id: str, stage: str, setup: str, wids: list[str]) -> list[list[dict]]:
    return [read_run(ft.trace_file(eval_id, stage, setup, w)) for w in wids]


def dev_ids() -> list[str]:
    return [f"dev{k}" for k in ev.DEV_SEEDS]


def logreg_view() -> dict:
    return logreg_scores_view(traces_of(LOGREG_DEV, "dev", "logreg", dev_ids()))


def tuned_view(setup: str) -> dict:
    return variant_view(traces_of(TUNED[setup], "dev", setup, dev_ids()), committed(TUNED[setup])["dev"].get("variant", "both"))


def picked(setup: str, c: costs.Costs) -> tuple[float, float, float] | None:
    if setup == "logreg":
        return pick(logreg_view(), ev.T_REVIEW, ev.T_DECLINE, c)
    return pick(tuned_view(setup), ft.GRID, ft.GRID, c) if setup in TUNED else None


def dev_only_rows() -> dict:
    """name → (dev view, committed thresholds, committed best dev cost)."""
    out = {}
    for name, (eval_id, setup) in DEV_ONLY.items():
        d = committed(eval_id)["dev"][name]
        out[name] = (variant_view(traces_of(eval_id, "dev", setup, dev_ids()), d["variant"]), d["thresholds"], d["grid_best"][0][2])
    f = committed(PHASE8)["dev"]["frozen_1.5b"]
    # eval_lfm.dev_cost keeps a mean per dev window here, so it is turned back into a total.
    out["frozen-1.5b"] = (variant_view(traces_of(FROZEN_DEV[0], "dev", FROZEN_DEV[1], dev_ids()), "both"), f["thresholds"], f["dev_cost_thresholds"] * len(ev.DEV_SEEDS))
    return out


# ---------------------------------------------------------------- the gate

def gate(log=print) -> list[str]:
    """Everything that must reproduce at 50 USD. Returns the failures."""
    bad, n = [], 0
    old = costs.PHASES_1_10
    picks = {}
    for setup, eval_id in TUNED.items():
        tr, td, _ = picks[setup] = picked(setup, old)
        th = committed(eval_id)["dev"]["thresholds"]
        if (tr, td) != (th["review"], th["decline"]):
            bad.append(f"{setup}: thresholds {tr}, {td} against committed {th}")
    picks["logreg"] = picked("logreg", old)
    for name, (view, th, cost) in dev_only_rows().items():
        tr, td, total = pick(view, ft.GRID, ft.GRID, old)
        if (tr, td) != (th["review"], th["decline"]) or abs(total - cost) > TOL_USD:
            bad.append(f"{name}: dev {tr}, {td}, {total:.2f} against committed {th}, {cost:.2f}")
    for ts in TEST_SETS.values():
        wids = committed(ts["windows_of"])["test_windows"]
        for setup, eval_id in ts["setups"].items():
            per = committed(eval_id)["windows"]
            per = per.get(setup, per)  # Phases 7 and 9 keep one setup, keyed by window
            for w, recs in zip(wids, traces_of(eval_id, "eval", setup, wids)):
                p = picks.get(setup)
                s = redecide(recs, old, p[:2] if p else None)
                sm = recs[-1]["summary"]
                n += 1
                if abs(s.summary["cost"] - per[w]["cost"]) > TOL_USD:
                    bad.append(f"{eval_id} {setup} {w}: cost {s.summary['cost']:.2f} against committed {per[w]['cost']:.2f}")
                if s.final != [r["action"] for r in tx_rows(recs)]:
                    bad.append(f"{eval_id} {setup} {w}: actions differ")
                if sm.get("cost_natural") and abs(s.summary["cost_natural"] - sm["cost_natural"]) > TOL_NATURAL * sm["cost_natural"]:
                    bad.append(f"{eval_id} {setup} {w}: natural {s.summary['cost_natural']:.2f} against {sm['cost_natural']:.2f}")
    log(f"gate: {n} test windows at 50 USD, {len(bad)} failures" + "".join(f"\n  {b}" for b in bad[:20]))
    return bad


# ---------------------------------------------------------------- the report

def view_row(scored: list[costs.Scored], traces: list[list[dict]]) -> dict:
    c = np.array([s.summary["cost"] for s in scored])
    m = [split(s, recs) for s, recs in zip(scored, traces)]
    return {"mean_cost": float(c.mean()), "median_cost": float(np.median(c)), "total_cost": float(c.sum()),
            "cost_natural_mean": float(np.mean([s.summary["cost_natural"] for s in scored])),
            "recall_mean": float(np.mean([s.summary["recall"] for s in scored])), "precision_mean": float(np.mean([s.summary["precision"] for s in scored])),
            "alert_rate_mean": float(np.mean([s.summary["alert_rate"] for s in scored])), "reviews": int(sum(s.summary["reviews_done"] for s in scored)),
            **{k: float(sum(x[k] for x in m)) for k in ("fraud_let_through", "good_declined", "review_fees")}}


def bar_value(b: tuple, by: dict[str, dict[str, np.ndarray]]) -> int:
    ts, a, other, _, kind = b
    x, y = by[ts][a], by[ts][other]
    return int((x < y).sum() if kind == "strict" else (x <= y).sum())


def run(rid: str, log=print) -> dict:
    new = costs.DEFAULT
    old = costs.PHASES_1_10
    picks = {s: picked(s, new) for s in (*TUNED, "logreg")}
    picks_old = {s: picked(s, old) for s in picks}
    out = {"id": rid, "script": "fraud/reprice_paysim.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": new.as_dict(), "costs_committed": old.as_dict(),
           "thresholds": {s: {"at_50": {"review": picks_old[s][0], "decline": picks_old[s][1]}, "at_10": {"review": p[0], "decline": p[1]}} for s, p in picks.items()},
           "test": {}, "bars": {}, "dev_only": {}}
    by = {"at_50": {}, "same": {}, "repicked": {}}
    for name, ts in TEST_SETS.items():
        wids = committed(ts["windows_of"])["test_windows"]
        rows = {}
        for k in by:
            by[k][name] = {}
        for setup, eval_id in ts["setups"].items():
            traces = traces_of(eval_id, "eval", setup, wids)
            at_50 = [redecide(r, old) for r in traces]
            same = [redecide(r, new) for r in traces]
            again = [redecide(r, new, picks[setup][:2]) for r in traces] if setup in picks else same
            rows[setup] = {"eval": eval_id, "at_50": view_row(at_50, traces), "same": view_row(same, traces), "repicked": view_row(again, traces)}
            for k, v in (("at_50", at_50), ("same", same), ("repicked", again)):
                by[k][name][setup] = np.array([s.summary["cost"] for s in v])
        out["test"][name] = {"windows": wids, "setups": rows,
                             "pairs": {k: [ie.compare(by[k][name], a, b) for a, b in PAIRS[name]] for k in ("same", "repicked")}}
    for label, b in BARS.items():
        out["bars"][label] = {"rule": f"{b[1]} {'cheaper than' if b[4] == 'strict' else 'cost <='} {b[2]} in >= {b[3]} of 20 windows", "need": b[3],
                              **{k: bar_value(b, by[k]) for k in by}}
    for name, (view, th, cost) in dev_only_rows().items():
        tr, td, total = pick(view, ft.GRID, ft.GRID, new)
        out["dev_only"][name] = {"at_50": {"review": th["review"], "decline": th["decline"], "dev_total": cost}, "at_10": {"review": tr, "decline": td, "dev_total": total}}
    p = picks["logreg"]
    out["dev_only"]["logreg"] = {"at_50": {"review": picks_old["logreg"][0], "decline": picks_old["logreg"][1], "dev_total": picks_old["logreg"][2]},
                                 "at_10": {"review": p[0], "decline": p[1], "dev_total": p[2]}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{rid}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{rid}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def markdown(r: dict) -> str:
    L = [f"# PaySim results at a 10 USD review {r['id']}", "",
         f"Pre-registration: `fraud/reprice_paysim.py`. From the recorded scores; no model ran and no trace changed. Costs in USD: {r['costs']}. All numbers Observed. Model scores are uncalibrated.", ""]
    for name, ts in r["test"].items():
        L += [f"## {name}", "",
              "| setup | mean at 50 | same decisions at 10 | re-picked at 10 | natural-rate, re-picked | recall | precision | alert rate | reviews | fraud let through | good declined | review fees |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s, x in sorted(ts["setups"].items(), key=lambda kv: kv[1]["repicked"]["mean_cost"]):
            p = x["repicked"]
            L.append(f"| {s} | {x['at_50']['mean_cost']:,.0f} | {x['same']['mean_cost']:,.0f} | {p['mean_cost']:,.0f} | {p['cost_natural_mean']:,.0f} | {p['recall_mean']:.3f} | "
                     f"{p['precision_mean']:.3f} | {p['alert_rate_mean']:.3f} | {p['reviews']} | {p['fraud_let_through']:,.0f} | {p['good_declined']:,.0f} | {p['review_fees']:,.0f} |")
        L += ["", "| view | lower cost | against | cheaper / dearer / ties | sign test p | mean difference | 95% CI | |", "|---|---|---|---:|---:|---:|---|---|"]
        for k in ("repicked", "same"):
            for p in ts["pairs"][k]:
                lo, hi = p["mean_diff_ci95"]
                L.append(f"| {k} | {p['better'] or 'tie'} | {p['other']} | {p['cheaper']} / {p['dearer']} / {p['ties']} | {p['sign_p']:.3g} | {p['mean_diff']:,.0f} | {lo:,.0f} to {hi:,.0f} | "
                         f"{'distinguishable' if p['distinguishable'] else 'not distinguishable'} |")
        L.append("")
    L += ["## Bars at 10 USD (the recorded verdicts stand; this is what they would read)", "", "| bar | rule | need | at 50 | same decisions | re-picked |", "|---|---|---:|---:|---:|---:|"]
    for k, b in r["bars"].items():
        L.append(f"| {k} | {b['rule']} | {b['need']} | {b['at_50']} | {b['same']} | {b['repicked']} |")
    L += ["", "## Thresholds (review, decline)", "", "| setup | at 50 | at 10 |", "|---|---|---|"]
    for s, t in r["thresholds"].items():
        L.append(f"| {s} | {t['at_50']['review']}, {t['at_50']['decline']} | {t['at_10']['review']}, {t['at_10']['decline']} |")
    L += ["", "## Dev only, dev0-dev9 totals (Phases 6 and 8; never reached test)", "", "| model | thresholds at 50 | total at 50 | thresholds at 10 | total at 10 |", "|---|---|---:|---|---:|"]
    for s, d in sorted(r["dev_only"].items(), key=lambda kv: kv[1]["at_10"]["dev_total"]):
        a, b = d["at_50"], d["at_10"]
        L.append(f"| {s} | {a['review']}, {a['decline']} | {a['dev_total']:,.0f} | {b['review']}, {b['decline']} | {b['dev_total']:,.0f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="only check that the committed 50 USD numbers reproduce")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if gate(log):
        raise SystemExit("the gate failed: nothing priced at 10 USD")
    if not args.gate:
        run(stamp(), log)


if __name__ == "__main__":
    main()
