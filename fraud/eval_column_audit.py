"""What would it cost if the C, D or V columns aren't known at decision time? Logreg by column group → fraud/results/eval_column_audit/.

    python -m fraud.eval_column_audit --leak             # do the columns look like they know the future? Feature columns only, no labels
    python -m fraud.eval_column_audit                    # fit, dev, 8 new windows, report
    python -m fraud.eval_column_audit --report <eval id> # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_eval_<setup>_audit<k>.jsonl. Results: fraud/results/eval_column_audit/<eval id>.json + .md, leak.json.

PRE-REGISTRATION (Phase 17; committed before any setup here scored a new window).
Owner, 2026-11-03: "logreg without those columns, and by column group, to bound what the 'known at decision time' doubt could cost. Use dev for choices and new windows for the result."
- Why: Phase 14's cheapest setup reads every column. From memory of the data page, not verified: C = counts, D = time deltas, V = Vesta-engineered. If some are not known when a payment arrives, its numbers are optimistic. This can't be settled from the data. It can be bounded.
- Setups, all numpy logistic regression refitted on fit rows, each with Phase 14's rule and with two score lines. Names say what is left out of ieee-logreg-all: audit-all (nothing, must match Phase 14's scores), audit-noV, audit-noC, audit-noD, audit-noCD, audit-noCV, audit-noDV, audit-noCDV, audit-noM (match flags, not asked but the same doubt).
  Dropping a group also drops its missing flags. card1, card2, addr1, addr2 stay out, as in ieee-logreg-all. Amount, hour, card3, card5, dist1, dist2 and the categories stay in.
- Choices, dev only: as Phase 14, on its 30 dev windows, a logistic fit of the score's logit, lambda from am.LAMBDAS, lines from ev.ALERT_SHARES. Nothing else is chosen.
- New windows: 8 chains from fraud/final_test.py, audit0-audit7, 8,000 rows. Each runs once per setup. Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed.
- Primary question: audit-noCDV with the rule against audit-all with the rule. The cost of losing all three groups, which bounds the doubt from above within this model family. Distinguishable only if the 95% bootstrap CI of the mean paired difference (windows resampled, seed 20261102) excludes 0.
- Second: audit-noV, audit-noC and audit-noD with the rule, each against audit-all with the rule. Third: does the rule still beat lines when all three are gone? audit-noCDV, rule against lines. Fourth: audit-noCDV with the rule against approve-all.
- A second read with far more windows, no bar: the 88 chains of dev rows outside the 30 windows, which fitted nothing here. 8 windows is few, and these are not new test rows, so it can't replace the primary answer. It says whether 8 windows mislead.
- Both reported in customers as well as dollars: frauds stopped, fraud amount stopped, good customers declined.
- Leak look, --leak, no labels, not a test: by a same-card key (card1, card2, card3, card5, card6, addr1) over fit and dev rows, each column's rank regressed on the rank of the card's earlier rows, of its later rows, and of time. A column that knows the future should lean on later rows. Also whether C1 ever falls within a card, and whether D1 grows one day per day. It can raise suspicion or lower it, not prove anything.
- Limits stated now: 8 windows give wide intervals, so "not distinguishable" here is weak. A stronger model might lose less or more from the same columns. These rows' chains span more time than contiguous windows. Reviews are still assumed to find the truth at a flat assumed fee.
- Stated before running (Inferred): audit-noCDV costs distinguishably more than audit-all, by 20-50%, and still beats approve-all and keeps the rule's advantage over lines. Dropping V alone costs little because C and D overlap with V. Dropping C hurts more than D. The leak look finds C and D leaning on past rows, and some V columns unclear.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_ieee_amount as am, final_test as ft, ieee_data, tabular as tb
from .baselines import LogReg
from .runs import read_run, stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_column_audit"
PHASE14 = "20261103-164830"
DROPS = {"all": "", "noV": "V", "noC": "C", "noD": "D", "noCD": "CD", "noCV": "CV", "noDV": "DV", "noCDV": "CDV", "noM": "M"}
BY_RULE, BY_LINES = "-amt", "-lines"


def setup_name(name: str, kind: str) -> str:
    return f"audit-{name}{kind}"


def features_without(drop: str):
    """ev.features_all without these column groups (letters of C, D, V, M), and their missing flags."""
    pattern = re.compile(f"^[{drop}]\\d+$") if drop else None
    gone = (lambda k: bool(pattern and pattern.match(k)))

    def features(t, idx: np.ndarray) -> np.ndarray:
        keep = [j for j, k in enumerate(t.num_names) if k not in ev.ID_LIKE and not gone(k)]
        flags = [j for j in keep if not t.num_names[j].startswith("V")]
        x = t.num[idx][:, keep].astype(np.float64)
        return np.c_[np.log1p(t.amount[idx]), ev._hours(t, idx), *[ev._one_hot(t, k, idx) for k in t.cat_names if not gone(k)], ev._signed_log(x), np.isnan(t.num[idx][:, flags])]

    return features


def run(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    fit_rows = np.arange(ieee_data.split_bounds(n)[0])
    dev, held, new = tb.dev30(n), tb.heldout_dev(n), ft.phase_windows(n, "audit")
    all_rows = dev + held + new
    info, heldout = {}, {}
    for name, drop in DROPS.items():
        t0 = time.perf_counter()
        f = features_without(drop)
        m = LogReg().fit(f(t, fit_rows), t.is_fraud[fit_rows].astype(float))
        q = tb.scores_for(m, f, t, all_rows)
        if name == "all":  # must match Phase 14's scores
            gap = max(float(np.abs(q[w.rows] - ev.view([read_run(ev.trace_file(PHASE14, "dev", am.LR, w.id))])["scores"][0]).max()) for w in dev)
            if gap > 1e-6:
                raise SystemExit(f"audit-all differs from Phase 14's logreg by {gap}; nothing else was run")
            log(f"audit-all matches Phase 14's dev scores, largest gap {gap:.1e}")
        pol = tb.fit_policy(q, t, dev)
        tb.write_setup(eval_id, t, new, setup_name(name, BY_RULE), q, pol, "rule")
        tb.write_setup(eval_id, t, new, setup_name(name, BY_LINES), q, pol, "lines")
        heldout[name] = {"rule": tb.per_window([tb.rule_cost(pol, q, t, w) for w in held]), "lines": tb.per_window([tb.lines_cost(pol, q, t, w) for w in held]),
                         "auc": ev.auc(np.concatenate([q[w.rows] for w in held]), np.concatenate([t.is_fraud[w.rows] for w in held]).astype(int))}
        info[name] = {"features": int(len(m.w) - 1), "lambda": pol.lam, "lines": [pol.review, pol.decline], "fit_s": round(time.perf_counter() - t0, 1)}
        log(f"{name}: {info[name]}")
    for w in new:
        am.write_trace(eval_id, t, w, ev.APPROVE_ALL, [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows], (None, None), None, None, {})
    return report(eval_id, info, heldout, log)


def report(eval_id: str, info=None, heldout=None, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    new = ft.phase_windows(n, "audit")
    path = RESULTS / f"{eval_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info = info or old.get("info", {})
    setups = [setup_name(k, b) for k in DROPS for b in (BY_RULE, BY_LINES)]
    tr = {s: tb.read_setup(eval_id, s, new) for s in setups + [ev.APPROVE_ALL]}
    rows = {s: tb.row(tr[s], s != ev.APPROVE_ALL) for s in tr}
    by = {s: tb.window_costs(tr[s]) for s in tr}
    all_amt = setup_name("all", BY_RULE)
    cmp = lambda a, b: ev.compare(by, a, b)
    verdict = {"primary": cmp(setup_name("noCDV", BY_RULE), all_amt),
               "second": [cmp(setup_name(k, BY_RULE), all_amt) for k in ("noV", "noC", "noD")],
               "third": cmp(setup_name("noCDV", BY_RULE), setup_name("noCDV", BY_LINES)),
               "fourth": cmp(setup_name("noCDV", BY_RULE), ev.APPROVE_ALL),
               "reported": [cmp(setup_name(k, BY_RULE), all_amt) for k in ("noCD", "noCV", "noDV", "noM")] + [cmp(all_amt, setup_name("all", BY_LINES))]}
    if heldout:
        base = heldout["all"]["rule"]
        held = {k: {"rule_mean": float(v["rule"].mean()), "lines_mean": float(v["lines"].mean()), "auc": v["auc"], "diff_to_all": float((v["rule"] - base).mean()),
                    "diff_ci": list(ev.boot_ci(v["rule"] - base)), "rule_minus_lines": float((v["rule"] - v["lines"]).mean()), "rule_minus_lines_ci": list(ev.boot_ci(v["rule"] - v["lines"])), "windows": len(base)} for k, v in heldout.items()}
    else:
        held = old.get("heldout", {})
    y = np.concatenate(ev.view(tr[all_amt])["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_column_audit.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": {w.id: [int(w.rows[0]), int(w.rows[-1])] for w in new},
           "test_rows": int(len(y)), "test_fraud": int(y.sum()), "info": info, "setups": rows, "verdict": verdict, "heldout": held}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def markdown(r: dict) -> str:
    L = [f"# Column audit {r['eval']}", "", f"Pre-registration: `fraud/eval_column_audit.py`. {r['test_rows']:,} new test rows, {r['test_fraud']} fraud, {len(r['windows'])} windows. Costs: {r['costs']}. All numbers Observed. Model scores are uncalibrated.", "",
         "| setup | mean cost | 95% CI | AUC | frauds stopped | fraud USD stopped | good declined | reviews | fraud let through | good declined USD | review fees |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        a = f"{x['auc']:.4f}" if x["auc"] is not None else "n/a"
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {a} | {x['frauds_stopped']} of {x['frauds']} | {x['fraud_usd_stopped']:,.0f} | {x['good_customers_declined']} | {x['reviews']} | {x['fraud_let_through']:,.0f} | {x['good_declined']:,.0f} | {x['review_fees']:,.0f} |")

    def line(p):
        return f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer; mean paired difference {p['mean_diff']:,.0f} USD per window, 95% CI {p['mean_diff_ci95'][0]:,.0f} to {p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}"

    v = r["verdict"]
    L += ["", "## Answers", "", f"- Primary: {line(v['primary'])}.", *[f"- Second: {line(p)}." for p in v["second"]], f"- Third: {line(v['third'])}.", f"- Fourth: {line(v['fourth'])}.", *[f"- Reported: {line(p)}." for p in v["reported"]]]
    if r["heldout"]:
        L += ["", "## Held-out dev, no bar", "", "| columns | rule mean | lines mean | AUC | rule minus audit-all's rule | 95% CI | rule minus lines | 95% CI |", "|---|---:|---:|---:|---:|---|---:|---|"]
        for k, x in r["heldout"].items():
            L.append(f"| {k} | {x['rule_mean']:,.0f} | {x['lines_mean']:,.0f} | {x['auc']:.4f} | {x['diff_to_all']:,.0f} | {x['diff_ci'][0]:,.0f} to {x['diff_ci'][1]:,.0f} | {x['rule_minus_lines']:,.0f} | {x['rule_minus_lines_ci'][0]:,.0f} to {x['rule_minus_lines_ci'][1]:,.0f} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the leak look

def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x)) / max(1, len(x) - 1)
    return r


def card_counts(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(earlier rows, later rows) of the same key, for rows in order."""
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    first = np.r_[0, np.cumsum(cnt)[:-1]]
    earlier = np.empty(len(inv), np.int64)
    earlier[order] = np.arange(len(inv)) - first[inv[order]]
    return earlier, cnt[inv] - earlier - 1


def card_key(t, rows: np.ndarray) -> np.ndarray:
    cols = [t.col(k)[rows] for k in ("card1", "card2", "card3", "card5", "addr1")] + [t.cat[rows, t.cat_names.index("card6")].astype(float)]
    return np.nan_to_num(np.stack(cols, 1), nan=-1.0)


def leak(log=print) -> dict:
    """No label is read."""
    t = ieee_data.load()
    rows = np.arange(ieee_data.split_bounds(len(t))[1])
    key = card_key(t, rows)
    earlier, later = card_counts(key)
    pe, pl, pt = _rank(earlier + rows * 1e-9), _rank(later + rows * 1e-9), rows / rows[-1]
    out = {}
    for j, name in enumerate(t.num_names):
        if name in ev.ID_LIKE:
            continue
        v = t.num[rows, j].astype(np.float64)
        ok = ~np.isnan(v)
        if ok.sum() < 1000 or np.nanstd(v) == 0:
            continue
        y = _rank(v[ok])
        x = np.c_[np.ones(ok.sum()), pe[ok], pl[ok], pt[ok]]
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        out[name] = {"earlier": float(b[1]), "later": float(b[2]), "time": float(b[3]), "rows": int(ok.sum())}
    groups = {"C": [k for k in out if re.match(r"^C\d+$", k)], "D": [k for k in out if re.match(r"^D\d+$", k)], "V": [k for k in out if re.match(r"^V\d+$", k)], "other": ["card3", "card5", "dist1", "dist2"]}
    summary = {}
    for g, names in groups.items():
        names = [k for k in names if k in out]
        e, l = np.array([abs(out[k]["earlier"]) for k in names]), np.array([abs(out[k]["later"]) for k in names])
        worst = sorted(names, key=lambda k: abs(out[k]["later"]) - abs(out[k]["earlier"]), reverse=True)[:8]
        summary[g] = {"columns": len(names), "median_abs_earlier": float(np.median(e)), "median_abs_later": float(np.median(l)), "share_later_above_earlier": float((l > e).mean()), "share_later_above_0.1": float((l > 0.1).mean()),
                      "most_later_leaning": {k: out[k] for k in worst}}
    # Do C1 and D1 behave like point-in-time values within a card?
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    same = inv[order][1:] == inv[order][:-1]
    prev, cur = order[:-1][same], order[1:][same]
    c1, d1, days = t.col("C1")[rows], t.col("D1")[rows], t.dt[rows] / 86400
    ok = ~np.isnan(c1[prev]) & ~np.isnan(c1[cur])
    gap = days[cur] - days[prev]
    d_ok = ~np.isnan(d1[prev]) & ~np.isnan(d1[cur]) & (gap > 1)
    delta = (d1[cur] - d1[prev])[d_ok] - gap[d_ok]
    res = {"rows": int(len(rows)), "card_key_groups": int(inv.max() + 1), "summary": summary, "columns": out,
           "C1_falls_within_card_share": float((c1[cur][ok] < c1[prev][ok]).mean()), "D1_step_within_1_day_of_elapsed_share": float((np.abs(delta) <= 1).mean()), "pairs": int(same.sum())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leak.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in res.items() if k != "columns"}, indent=1))
    return res


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leak", action="store_true", help="the leak look, no labels, then stop")
    ap.add_argument("--report", metavar="EVAL_ID", help="rebuild the results files from this eval's traces")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.leak:
        leak(log)
    elif a.report:
        report(a.report, log=log)
    else:
        run(stamp(), log)


if __name__ == "__main__":
    main()
