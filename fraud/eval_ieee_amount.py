"""Do decisions that weigh the amount lower the cost? IEEE-CIS, new test windows → fraud/results/eval_ieee_amount/.

    python -m fraud.eval_ieee_amount --windows                  # the new windows; no model, no labels
    python -m fraud.eval_ieee_amount --adapter <dir>            # score new dev and test windows, derive the setups, report
    python -m fraud.eval_ieee_amount --adapter <dir> --resume <eval id>  # finish an interrupted eval (finished windows are kept)
    python -m fraud.eval_ieee_amount --report <eval id>         # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_dev_<setup>_{dev,xdev}<k>.jsonl, <eval id>_eval_<setup>_amt<k>.jsonl. Results: fraud/results/eval_ieee_amount/<eval id>.json + .md.

PRE-REGISTRATION (Phase 14; committed before any model or policy touched a new window).
Owner, 2026-11-03: yes to amount-aware decisions. Goal: lower cost. Nothing is retrained.
- Why: a score threshold gives a 5 USD and a 5,000 USD payment the same decision, though approved fraud costs its amount and a review a flat fee. Review budgets are also spent first come,
  first served (Observed, Phase 12: 958-968 of 1,000 reviews used).
- New test windows: seed 20261104, the test split cut into 20 equal blocks, one start per block, redrawn until the window shares no row with Phase 11's, Phase 12's or an earlier new window.
  Same stretch of time as those, so not a later period. Each runs once per setup. Dev windows: Phase 11's 10 and 20 new ones by the same draw in the dev split (30 in all).
- Scorers, frozen: ieee-logreg-all (refitted on fit rows, must reproduce Phase 11's dev scores) and Phase 12's adapter ieee-ft-ieee-full, order `rev` as Phase 12 chose, reusing its dev0-dev9 scores.
- Setups on the new test windows (each from the same recorded scores):
  - ieee-logreg-all, ieee-ft-ieee-full: score thresholds as they are, Phase 12's values. Past the budget a review becomes an approve.
  - ieee-logreg-all-dev30, ieee-ft-full-dev30: the same rule, thresholds re-picked on the 30 dev windows.
  - ieee-logreg-all-amt, ieee-ft-full-amt: the amount-aware rule below, fitted on the 30 dev windows.
  - ieee-blend-amt: the same rule on a blend of the two scores (both logits fitted together on the 30 dev windows). ieee-approve-all.
- The amount-aware rule, using only the current row: p from a logistic fit of the score's logit on dev (used only to weigh costs; not shown or claimed as a probability). Expected cost of approve p * amount, of decline
  (1 - p) * friction, of review fee + lambda. Take the lowest; ties to approve, then decline. Past the budget a review becomes the cheaper of approve and decline. Lambda, the price of using up a review, is picked
  from LAMBDAS by lowest total dev cost, first on a tie.
- Costs: Phase 11's. Friction 10%, at least 1 USD; review 10 USD; at most 5% of a window reviewed.
- Primary question: ieee-logreg-all-amt against ieee-logreg-all, today's cheapest, on total cost. Distinguishable only if the 95% bootstrap interval (windows resampled, seed 20261102) of the mean paired difference excludes 0.
- Second question: ieee-ft-full-amt against ieee-ft-ieee-full, same rule. Third: ieee-blend-amt against ieee-logreg-all-amt. Fourth, which is cheapest overall: ieee-ft-full-amt against ieee-logreg-all-amt.
- Also reported, no bar: dev30 against as-they-are (what more dev windows alone do), where the money goes, reviews and their amounts, recall, precision, AUC, cost at reviews of 1, 5 and 20 USD with lambda or thresholds re-picked on
  dev; latency bar p90 <= 250 ms for the adapter. Not corrected for multiple comparisons beyond the primary.
- Observed before registering, dev only, recorded scores, two folds over Phase 11's 10 dev windows (fitted on one half, scored on the other): logreg 34,011 → 22,941 USD (lower in 8 of 10 windows), adapter 37,757 → 22,772 (8 of 10), blend 24,105.
- Limits stated now: a process that dies resumes with finished windows kept. Scores are the same as Phase 12's, so whether C, D and V columns are known at decision time is still uncheckable, and logreg-all relies on them.
  The past-budget fallback differs between the two rules, on purpose: it is part of the new rule. Reviews are still assumed to find the truth at a flat assumed fee.
- Stated before running (Inferred): both amount-aware setups are cheaper than their score-only versions, by less than the held-out dev gap, probably distinguishable for at least one. The blend does not beat logreg's amount-aware setup. More dev windows alone change little.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, eval_ieee as ev, eval_ieee_full as fu, ieee_data
from .baselines import LogReg
from .eval_ft import BAR_P90_MS, release
from .runs import RUNS, finished, read_run, stamp, to_jsonl

RESULTS = Path(__file__).resolve().parent / "results" / "eval_ieee_amount"
SEED = 20261104
PHASE11, PHASE12 = fu.OLD_EVAL, "20261103-142622"
VARIANT = "rev"  # Phase 12's debias rule chose it from dev
LR, FT = ev.LOGREG_ALL, fu.FT_FULL
LR30, FT30 = "ieee-logreg-all-dev30", "ieee-ft-full-dev30"
LR_AMT, FT_AMT, BLEND = "ieee-logreg-all-amt", "ieee-ft-full-amt", "ieee-blend-amt"
SETUPS = (LR, FT, LR30, FT30, LR_AMT, FT_AMT, BLEND, ev.APPROVE_ALL)
LAMBDAS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
EXTRA_DEV, TEST_COUNT = 20, 20


# ---------------------------------------------------------------- windows

def fresh_windows(n: int, split: str, count: int, taken: list[int], prefix: str, size: int = ev.WINDOW_SIZE, seed: int = SEED) -> list[ev.Win]:
    """count windows of contiguous rows in a split, one per equal block, sharing no row with the starts in taken or with each other."""
    lo, hi = ieee_data.split_rows(n, split)
    edges = np.linspace(lo, hi, count + 1).astype(np.int64)
    rng = np.random.default_rng([seed, 0 if split == "dev" else 1])
    taken, out = list(taken), []
    for k in range(count):
        while True:
            start = int(rng.integers(edges[k], edges[k + 1] - size + 1))
            if all(abs(start - s) >= size for s in taken):
                break
        taken.append(start)
        out.append(ev.Win(f"{prefix}{k}", split, np.arange(start, start + size), seed))
    return out


def old_dev(n: int) -> list[ev.Win]:
    return ev.make_windows(n, "dev", ev.DEV_WINDOWS)


def extra_dev(n: int) -> list[ev.Win]:
    return fresh_windows(n, "dev", EXTRA_DEV, [int(w.rows[0]) for w in old_dev(n)], "xdev")


def test_windows(n: int) -> list[ev.Win]:
    taken = [int(w.rows[0]) for w in ev.make_windows(n, "test", ev.TEST_WINDOWS) + fu.new_windows(n)]
    return fresh_windows(n, "test", TEST_COUNT, taken, "amt")


# ---------------------------------------------------------------- the amount-aware rule

def logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def decide(p: np.ndarray, amount: np.ndarray, lam: float, c: costs.Costs) -> tuple[list[str], list[str]]:
    """(wanted, fallback) per row: the lowest expected cost, a review priced at its fee plus lam."""
    approve = p * amount
    decline = (1 - p) * np.maximum(c.friction_min, c.friction_rate * amount)
    fallback = np.where(approve <= decline, "approve", "decline")
    wanted = np.where(c.review_fee + lam < np.minimum(approve, decline), "review", fallback)
    return wanted.tolist(), fallback.tolist()


def pick_lambda(ps, truth, amount, c: costs.Costs) -> tuple[float, list]:
    """The lowest total dev cost over LAMBDAS, first on a tie."""
    grid = []
    for lam in LAMBDAS:
        total = 0.0
        for p, y, a in zip(ps, truth, amount):
            wanted, fallback = decide(p, a, lam, c)
            total += costs.score(c, wanted, fallback, y.astype(bool), a).summary["cost"]
        grid.append((lam, total))
    return min(grid, key=lambda g: g[1])[0], grid


# ---------------------------------------------------------------- scores

def outs_of(recs: list[dict]) -> list[dict]:
    keep = ("by", "q", "ms", "scores", "orders", "prompt_tokens")
    return [{k: r[k] for k in keep if k in r} for r in recs if r.get("type") == "tx"]


def dev_scores(eval_id: str, n: int) -> dict:
    """Per dev window, in one order: logreg and adapter scores, truth, amount."""
    old, new = old_dev(n), extra_dev(n)
    lr = ev.view([read_run(ev.trace_file(eval_id, "dev", LR, w.id)) for w in old + new])
    a = ev.view([read_run(ev.trace_file(PHASE12, "dev", FT, w.id)) for w in old], VARIANT)
    b = ev.view([read_run(ev.trace_file(eval_id, "dev", FT, w.id)) for w in new])
    if not all((x == y).all() for x, y in zip(lr["amount"], a["amount"] + b["amount"])):
        raise ValueError("the logreg and adapter dev traces are not the same rows")
    return {"lr": lr["scores"], "ft": a["scores"] + b["scores"], "truth": lr["truth"], "amount": lr["amount"]}


def features(d: dict, setup: str) -> list[np.ndarray]:
    """What the logistic fit reads per window: the score's logit, or both logits for the blend."""
    if setup == BLEND:
        return [np.c_[logit(x), logit(y)] for x, y in zip(d["lr"], d["ft"])]
    return [logit(x)[:, None] for x in d["lr" if setup == LR_AMT else "ft"]]


def fit_rule(d: dict, setup: str, c: costs.Costs = ev.COSTS) -> tuple[LogReg, float]:
    x = features(d, setup)
    m = LogReg().fit(np.concatenate(x), np.concatenate(d["truth"]).astype(float))
    return m, pick_lambda([m.predict(k) for k in x], d["truth"], d["amount"], c)[0]


# ---------------------------------------------------------------- the run

def s10_thresholds(setup: str) -> tuple[float | None, float | None]:
    th = read_run(ev.trace_file(PHASE12, "eval", setup, "new0"))[0]["thresholds"]
    return th["review"], th["decline"]


def check_logreg(eval_id: str, n: int) -> None:
    """Refitted logreg must give Phase 11's dev scores."""
    for w in old_dev(n):
        a = ev.view([read_run(ev.trace_file(eval_id, "dev", LR, w.id))])["scores"][0]
        b = ev.view([read_run(ev.trace_file(PHASE11, "dev", LR, w.id))])["scores"][0]
        if np.abs(a - b).max() > 1e-6:
            raise SystemExit(f"logreg on {w.id} does not reproduce Phase 11's scores; nothing else was run")


def write_trace(eval_id: str, t, w: ev.Win, setup: str, outs: list[dict], thresholds, decided, policy: dict | None, backend: dict) -> list[dict]:
    recs = ev.trace(t, w, setup, outs, thresholds, backend, eval_id, decided)
    if policy:
        recs[0]["policy"] = policy
    ev.trace_file(eval_id, "eval", setup, w.id).write_text(to_jsonl(recs), encoding="utf-8")
    return recs


def derive(eval_id: str, t, log=print) -> None:
    """Every other setup from the recorded scores of the two base setups. No model."""
    n = len(t)
    test = test_windows(n)
    d = dev_scores(eval_id, n)
    base = {s: {w.id: read_run(ev.trace_file(eval_id, "eval", s, w.id)) for w in test} for s in (LR, FT)}
    for setup, src, key in ((LR30, LR, "lr"), (FT30, FT, "ft")):
        tr, td, _ = ev.pick_thresholds(d[key], d["truth"], d["amount"], ev.COSTS)
        log(f"{setup}: thresholds review {tr}, decline {td}")
        for w in test:
            recs = base[src][w.id]
            write_trace(eval_id, t, w, setup, outs_of(recs), (tr, td), None, None, recs[0]["backend"])
    for setup in (LR_AMT, FT_AMT, BLEND):
        m, lam = fit_rule(d, setup)
        policy = {"kind": "amount-aware", "lambda": lam, "fit_weights": m.w.tolist(), "fit_mean": m.mu.tolist(), "fit_sd": m.sd.tolist()}
        log(f"{setup}: lambda {lam}")
        for w in test:
            lr_q, ft_q = (np.array([o["q"] for o in outs_of(base[s][w.id])]) for s in (LR, FT))
            x = {LR_AMT: logit(lr_q)[:, None], FT_AMT: logit(ft_q)[:, None], BLEND: np.c_[logit(lr_q), logit(ft_q)]}[setup]
            p = m.predict(x)
            recs = base[LR if setup == LR_AMT else FT][w.id]
            outs = outs_of(recs)
            if setup == BLEND:
                for o, v in zip(outs, p):
                    o["q"] = float(v)  # the blend's own ranking score; uncalibrated in the console's sense, as every score here
            write_trace(eval_id, t, w, setup, outs, None, decide(p, t.amount[w.rows], lam, ev.COSTS), policy, recs[0]["backend"])


def run(eval_id: str, adapter: Path, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    old, new, test = old_dev(n), extra_dev(n), test_windows(n)
    man = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
    head = read_run(ev.trace_file(PHASE12, "eval", FT, "new0"))[0]
    if man["adapter_sha256"] != head["backend"]["adapter"]["sha256"]:
        raise SystemExit("this is not Phase 12's adapter")
    need = [ev.trace_file(eval_id, "dev", LR, w.id) for w in old + new] + [ev.trace_file(eval_id, "eval", LR, w.id) for w in test]
    lr_outs = {}
    if not all(finished(p) for p in need):
        t0 = time.perf_counter()
        lr = ev.fit_logreg(t, ev.features_all)
        log(f"{LR} fitted in {time.perf_counter() - t0:.0f} s")
        lr_outs = {w.id: [{"by": "logreg", "q": float(p), "ms": 0.0} for p in lr.predict(ev.features_all(t, w.rows))] for w in old + new + test}
        del lr
    ev.run_setup(eval_id, "dev", LR, old + new, t, lambda w: lr_outs[w.id], None, {}, log)
    check_logreg(eval_id, n)
    ev.run_setup(eval_id, "eval", LR, test, t, lambda w: lr_outs[w.id], s10_thresholds(LR), {}, log)
    ev.run_setup(eval_id, "eval", ev.APPROVE_ALL, test, t, lambda w: [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows], (None, None), {}, log)

    model = None

    def ensure():
        nonlocal model
        if model is None:
            model = ev.D1Fine(adapter)
        return model

    ev.run_setup(eval_id, "dev", FT, new, t, lambda w: [ensure().score(t, int(i), VARIANT) for i in w.rows], None, lambda: ensure().info, log)
    ev.run_setup(eval_id, "eval", FT, test, t, lambda w: [ensure().score(t, int(i), VARIANT) for i in w.rows], s10_thresholds(FT), lambda: ensure().info, log)
    model = None
    release()
    derive(eval_id, t, log)
    return report(eval_id, log)


# ---------------------------------------------------------------- the report

def split_money(traces: list[list[dict]]) -> dict:
    tx = [r for recs in traces for r in recs if r.get("type") == "tx"]
    reviewed = [r["tx"]["amount"] for r in tx if r["action"] == "review"]
    return {"fraud_let_through": float(sum(r["cost"] for r in tx if r["action"] == "approve" and r["truth"])),
            "good_declined": float(sum(r["cost"] for r in tx if r["action"] == "decline" and not r["truth"])),
            "review_fees": float(sum(r["cost"] for r in tx if r["action"] == "review")),
            "reviewed_amount_median": float(np.median(reviewed)) if reviewed else None,
            "reviews_finding_fraud": sum(1 for r in tx if r["action"] == "review" and r["truth"]),
            "fraud_amount": float(sum(r["tx"]["amount"] for r in tx if r["truth"]))}


def sensitivity(eval_id: str, d: dict, test_tr: dict, sets: dict) -> dict:
    """From the same scores, the lines or lambda re-picked on dev per cost set; the amount-aware logistic fits don't use the fee and are kept."""
    out = {}
    base = {s: ev.view(test_tr[s]) for s in (LR, FT)}
    y, a = base[LR]["truth"], base[LR]["amount"]
    x_test = {LR_AMT: [logit(q)[:, None] for q in base[LR]["scores"]], FT_AMT: [logit(q)[:, None] for q in base[FT]["scores"]],
              BLEND: [np.c_[logit(p), logit(q)] for p, q in zip(base[LR]["scores"], base[FT]["scores"])]}

    def by_lines(q, tr, td, c):
        return float(np.mean([costs.score(c, [ev.act(float(v), tr, td) for v in s], ["approve"] * len(s), yy, aa).summary["cost"] for s, yy, aa in zip(q, y, a)]))

    for name, c in sets.items():
        row = {ev.APPROVE_ALL: by_lines([np.zeros(len(k)) for k in y], None, None, c)}
        ten = ev.DEV_WINDOWS
        for s, setup, setup30 in (("lr", LR, LR30), ("ft", FT, FT30)):
            tr, td, _ = ev.pick_thresholds(d[s][:ten], d["truth"][:ten], d["amount"][:ten], c)
            row[setup] = by_lines(base[setup]["scores"], tr, td, c)
            tr, td, _ = ev.pick_thresholds(d[s], d["truth"], d["amount"], c)
            row[setup30] = by_lines(base[setup]["scores"], tr, td, c)
        for setup in (LR_AMT, FT_AMT, BLEND):
            m, lam = fit_rule(d, setup, c)
            vals = []
            for x, yy, aa in zip(x_test[setup], y, a):
                wanted, fallback = decide(m.predict(x), aa, lam, c)
                vals.append(costs.score(c, wanted, fallback, yy, aa).summary["cost"])
            row[setup] = float(np.mean(vals))
        out[name] = {"costs": c.as_dict(), "mean_cost": row}
    return out


def verdict(by: dict[str, np.ndarray]) -> dict:
    return {"primary": ev.compare(by, LR_AMT, LR), "second": ev.compare(by, FT_AMT, FT), "third": ev.compare(by, BLEND, LR_AMT), "fourth": ev.compare(by, FT_AMT, LR_AMT),
            "reported": [ev.compare(by, LR30, LR), ev.compare(by, FT30, FT), ev.compare(by, LR_AMT, ev.APPROVE_ALL), ev.compare(by, FT_AMT, ev.APPROVE_ALL)]}


def report(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    test = test_windows(n)
    test_tr = {s: [read_run(ev.trace_file(eval_id, "eval", s, w.id)) for w in test] for s in SETUPS}
    views = {s: ev.view(test_tr[s]) for s in SETUPS}
    rows = {s: {**ev.setup_row(test_tr[s], views[s], s != ev.APPROVE_ALL), **split_money(test_tr[s]), "policy": test_tr[s][0][0].get("policy")} for s in SETUPS}
    by = {s: np.array([r[-1]["summary"]["cost"] for r in test_tr[s]]) for s in SETUPS}
    d = dev_scores(eval_id, n)
    y_all = np.concatenate(views[LR]["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_ieee_amount.py", "dataset": "IEEE-CIS train files, time split", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "costs": ev.COSTS.as_dict(), "seed": SEED, "phase12_eval": PHASE12, "variant": VARIANT, "lambdas": list(LAMBDAS), "backend": test_tr[FT][0][0]["backend"].get("adapter"),
           "windows": {w.id: w.header()["steps"] for w in test}, "dev_windows": {w.id: w.header()["steps"] for w in old_dev(n) + extra_dev(n)},
           "test_rows": int(len(y_all)), "test_fraud": int(y_all.sum()), "setups": rows, "verdict": verdict(by),
           "sensitivity": sensitivity(eval_id, d, test_tr, fu.SENSITIVITY), "bar_latency": rows[FT]["latency_ms"]["p90"] <= BAR_P90_MS}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def markdown(r: dict) -> str:
    v = r["verdict"]
    L = [f"# IEEE-CIS amount-aware eval {r['eval']}", "",
         f"Pre-registration: `fraud/eval_ieee_amount.py`. {r['test_rows']:,} new test rows, {r['test_fraud']} fraud, natural rate, {len(r['windows'])} windows. Costs in USD: {r['costs']}. All numbers Observed. Model scores are uncalibrated.", "",
         "| setup | mean cost | 95% CI | AUC | recall | precision | alert rate | reviews | declines | fraud let through | good declined | review fees | median amount reviewed | rule |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        a = f"{x['auc']:.4f}" if x["auc"] is not None else "n/a"
        med = f"{x['reviewed_amount_median']:,.0f}" if x["reviewed_amount_median"] is not None else "n/a"
        th = x["thresholds"] or {}
        rule = f"lambda {x['policy']['lambda']}" if x["policy"] else f"{fu._th(th.get('review'))}, {fu._th(th.get('decline'))}"
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {a} | {x['recall']:.3f} | {x['precision']:.3f} | {x['alert_rate']:.3f} | {x['reviews']} | {x['declines']} | "
                 f"{x['fraud_let_through']:,.0f} | {x['good_declined']:,.0f} | {x['review_fees']:,.0f} | {med} | {rule} |")

    def line(p):
        return (f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer, {p['ties']} ties; sign test p {p['sign_p']:.3g}; mean paired difference {p['mean_diff']:,.0f} USD per window, "
                f"95% CI {p['mean_diff_ci95'][0]:,.0f} to {p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}")

    L += ["", "## Answers", "", f"- Primary: {line(v['primary'])}.", f"- Second: {line(v['second'])}.", f"- Third: {line(v['third'])}.", f"- Fourth: {line(v['fourth'])}.",
          *[f"- {line(p)}." for p in v["reported"]], f"- Latency bar, p90 <= {BAR_P90_MS:.0f} ms, {FT}: {'PASS' if r['bar_latency'] else 'FAIL'} ({r['setups'][FT]['latency_ms']['p90']:.0f} ms).",
          "", "## Cost sensitivity (no bar; lines or lambda re-picked on dev per row)", "", "| cost set | " + " | ".join(SETUPS) + " |", "|---|" + "---:|" * len(SETUPS)]
    for name, x in r["sensitivity"].items():
        L.append(f"| {name} | " + " | ".join(f"{x['mean_cost'][s]:,.0f}" for s in SETUPS) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", action="store_true", help="print the new windows, then stop")
    ap.add_argument("--resume", metavar="EVAL_ID", help="continue this eval: skip its finished windows")
    ap.add_argument("--report", metavar="EVAL_ID", help="only rebuild the results files from this eval's traces")
    ap.add_argument("--adapter", type=Path, help="Phase 12's adapter folder (python -m fraud.finetune_ieee --train --full)")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if args.windows:
        n = len(ieee_data.load())
        for w in extra_dev(n) + test_windows(n):
            log(f"{w.id}: rows {w.rows[0]}–{w.rows[-1]}")
    elif args.report:
        report(args.report, log)
    else:
        if args.adapter is None or not (args.adapter / "manifest.json").exists():
            ap.error("--adapter must be Phase 12's folder, with manifest.json and adapter/")
        RUNS.mkdir(parents=True, exist_ok=True)
        run(args.resume or stamp(), args.adapter.resolve(), log)


if __name__ == "__main__":
    main()
