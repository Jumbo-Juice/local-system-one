"""No human review: approve the transfer queue, stop the money at a tightened cash-out rule (Phase 10).

    python -m fraud.eval_chain --hours    # how many untouched hours there are; reads steps only, no outcome
    python -m fraud.eval_chain --train    # the training hours: checks the code against the exploratory numbers
    python -m fraud.eval_chain            # the pre-registered test, once → fraud/results/eval_chain/<id>.json|.md

PRE-REGISTRATION (written and committed before this ran on any test-split hour; fraud/ROADMAP.md Phase 10).
Owner's request of 2026-11-02: minimise review, forget cost, focus on F1. The rule was found on training steps
1–300 (PROGRESS 2026-11-02), where it scored chain F1 0.98 and row F1 0.67, in-sample.
Question: on hours nothing has touched, does the tightened rule with no reviews reach a better chain F1 than
today's rules with a human on the whole queue?
- Hours: every hour of the test split (301–743) in no window of fraud.eval.TEST_SEEDS or fraud.eval_ft.new_windows,
  the windows every earlier test ran on. Every row of those hours, at the natural fraud rate, no sampling. Run once.
- Setups, all rule tables with no model:
  - human: rules.RULES as they are; a review finds the truth (fraud blocked, legit approved).
  - approve: rules.RULES with the review lines approving instead; no reviews.
  - tight: approve, and the cash-out line also needs an earlier TRANSFER of exactly this amount in this hour. Earlier
    rows only, so no look-ahead; a cash-out that fails it falls through to "otherwise" (approve). No reviews.
  - decline: rules.RULES with the review lines declining; no reviews.
- A chain is one theft: a fraud TRANSFER paired with the first unpaired later fraud CASH_OUT of exactly its amount in the
  same hour; any other fraud row is a chain of its own. A chain is caught if any of its rows is declined or reviewed.
- Chain F1: TP = chains caught, FN = chains not caught, FP = legit rows declined. Assumes money can still be stopped at
  the cash-out (Implementation choice; on real rails it might not).
- Row F1: each row its own positive, reported for every setup, no bar: the tight setup's recall cannot pass about 0.5
  there, because its fraud transfers are approved. The human setup's row F1 is expected to beat it.
- Bar 1: tight chain F1 > human chain F1 on the same hours.
- Bar 2: tight chain F1 >= 0.90: the in-sample 0.98 is expected to shrink, and below this the rule is not worth having.
  Both bars must pass. Reported, no bar: precision, recall, FP and FN counts, review counts, 95% bootstrap intervals over
  hours (2,000 resamples, seed 0) for chain F1 per setup.
- Limits, stated before running (Inferred): PaySim's cash-out of the exact amount within the hour looks like a property of
  its generator, so a pass says the rule fits PaySim, not that real fraud behaves this way. Cost is not scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import rules, windows
from .data import TYPE_ID
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_chain"
SETUPS = ("human", "approve", "tight", "decline")
BAR_F1 = 0.90
BOOTSTRAPS, BOOT_SEED = 2000, 0


def rule_lines(t, s) -> np.ndarray:
    """rules.rule_for for every row, as an array."""
    typ, same = t.type, np.asarray(s["same_amount_step_before"]) > 0
    transfer, cash = typ == TYPE_ID["TRANSFER"], typ == TYPE_ID["CASH_OUT"]
    round_, new = np.asarray(s["round_amount"]).astype(bool), np.asarray(s["dest_in_before"]) == 0
    free = np.isin(typ, list(rules.FRAUD_FREE))
    return np.select([free, cash & same, transfer & round_, transfer & new, transfer & same], [0, 1, 2, 3, 4], default=5)


def prior_transfer(t) -> np.ndarray:
    """For each row: an earlier TRANSFER of exactly this amount in this hour. Earlier rows only."""
    out, seen, hour = np.zeros(len(t), bool), set(), None
    wanted = np.flatnonzero(np.isin(t.type, [TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"]]))
    for i in wanted:
        if t.step[i] != hour:
            seen, hour = set(), t.step[i]
        if t.type[i] == TYPE_ID["TRANSFER"]:
            seen.add(float(t.amount[i]))
        else:
            out[i] = float(t.amount[i]) in seen
    return out


def actions(setup: str, lines: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(declined, reviewed) row masks of one setup."""
    queue = np.isin(lines, [3, 4])
    declined = np.isin(lines, [1, 2])
    if setup == "human":
        return declined, queue
    if setup == "approve":
        return declined, np.zeros(len(lines), bool)
    if setup == "tight":
        return declined & ~((lines == 1) & ~prior), np.zeros(len(lines), bool)
    if setup == "decline":
        return declined | queue, np.zeros(len(lines), bool)
    raise ValueError(setup)


def chains(t, rows: np.ndarray) -> np.ndarray:
    """A chain number per fraud row of ``rows`` (file order), -1 on legit rows: see the pre-registration."""
    out = np.full(len(rows), -1, dtype=np.int64)
    fraud = np.flatnonzero(t.is_fraud[rows])
    open_, hour, n = {}, None, 0
    for j in fraud:
        i = rows[j]
        if t.step[i] != hour:
            open_, hour = {}, t.step[i]
        key = float(t.amount[i])
        if t.type[i] == TYPE_ID["CASH_OUT"] and open_.get(key):
            out[j] = open_[key].pop(0)
            continue
        out[j] = n
        if t.type[i] == TYPE_ID["TRANSFER"]:
            open_.setdefault(key, []).append(n)
        n += 1
    return out


def f1(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def score(y: np.ndarray, chain: np.ndarray, declined: np.ndarray, reviewed: np.ndarray) -> dict:
    """Row and chain scores. A review finds the truth: it catches fraud and never declines legit."""
    caught = declined | reviewed
    fp = int((declined & ~y).sum())
    row = f1(int((caught & y).sum()), fp, int((~caught & y).sum()))
    ids = chain[y]
    hit = np.zeros(ids.max() + 1 if len(ids) else 0, bool)
    np.logical_or.at(hit, ids, caught[y])
    return {"row": row, "chain": f1(int(hit.sum()), fp, int((~hit).sum())), "reviews": int(reviewed.sum())}


def chain_f1_by_hours(step, y, chain, declined, reviewed, picks) -> list[float]:
    """Chain F1 of bootstrap resamples: per-hour counts summed over each resample."""
    hours, at = np.unique(step, return_inverse=True)
    fp = np.bincount(at, declined & ~y, minlength=len(hours))
    caught = declined | reviewed
    ids = chain[y]
    n = ids.max() + 1 if len(ids) else 0
    hit = np.zeros(n, bool)
    np.logical_or.at(hit, ids, caught[y])
    first = np.zeros(n, dtype=np.int64)
    first[ids[::-1]] = at[y][::-1]
    tp = np.bincount(first, hit, minlength=len(hours))
    fn = np.bincount(first, ~hit, minlength=len(hours))
    return [f1(tp[p].sum(), fp[p].sum(), fn[p].sum())["f1"] for p in picks]


def untouched_hours(t) -> list[int]:
    from .eval import TEST_SEEDS
    from .eval_ft import new_windows

    used = set()
    for w in [windows.make(t, "test", k, wid=f"test{k}") for k in TEST_SEEDS] + new_windows(t):
        used |= set(range(w.steps[0], w.steps[1] + 1))
    return [h for h in range(windows.TEST[0], windows.TEST[1] + 1) if h not in used]


def evaluate(t, s, hours: list[int], boot: bool = True) -> dict:
    mask = np.isin(t.step, hours)
    rows = np.flatnonzero(mask)
    lines, prior = rule_lines(t, s)[rows], prior_transfer(t)[rows]
    y, chain, step = t.is_fraud[rows].astype(bool), chains(t, rows), t.step[rows]
    out = {"hours": len(hours), "rows": len(rows), "fraud_rows": int(y.sum()), "chains": int(chain.max() + 1), "setups": {}}
    rng = np.random.default_rng(BOOT_SEED)
    n_hours = len(np.unique(step))
    picks = [rng.integers(0, n_hours, n_hours) for _ in range(BOOTSTRAPS)] if boot else []
    for setup in SETUPS:
        declined, reviewed = actions(setup, lines, prior)
        r = score(y, chain, declined, reviewed)
        if boot:
            lo, hi = np.percentile(chain_f1_by_hours(step, y, chain, declined, reviewed, picks), [2.5, 97.5])
            r["chain"]["f1_95"] = [float(lo), float(hi)]
        out["setups"][setup] = r
    tight, human = out["setups"]["tight"]["chain"]["f1"], out["setups"]["human"]["chain"]["f1"]
    out["bars"] = {"bar1": {"rule": "tight chain F1 > human chain F1", "pass": tight > human},
                   "bar2": {"rule": f"tight chain F1 >= {BAR_F1}", "pass": tight >= BAR_F1}}
    return out


def markdown(r: dict, title: str) -> str:
    L = [f"# {title}", "", "Pre-registration: the `fraud/eval_chain.py` docstring. All numbers are Observed.", "",
         f"{r['hours']} hours, {r['rows']:,} rows at the natural rate, {r['fraud_rows']:,} fraud rows, {r['chains']:,} chains.", "",
         "| setup | reviews | chain F1 (95% over hours) | chain P | chain R | chain FP | chain FN | row F1 | row P | row R |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in r["setups"].items():
        c, w = v["chain"], v["row"]
        ci = f" ({c['f1_95'][0]:.3f}–{c['f1_95'][1]:.3f})" if "f1_95" in c else ""
        L.append(f"| {k} | {v['reviews']:,} | {c['f1']:.4f}{ci} | {c['precision']:.4f} | {c['recall']:.4f} | {c['fp']:,} | "
                 f"{c['fn']:,} | {w['f1']:.4f} | {w['precision']:.4f} | {w['recall']:.4f} |")
    L += [""] + [f"- {b['rule']}: **{'PASS' if b['pass'] else 'FAIL'}**" for b in r["bars"].values()]
    return "\n".join(L) + "\n"


def main() -> None:
    from .capture import load_all

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", action="store_true", help="count the untouched hours (no outcome is read)")
    ap.add_argument("--train", action="store_true", help="the training hours, not a result")
    args = ap.parse_args()
    t, s = load_all()
    if args.hours:
        h = untouched_hours(t)
        print(f"{len(h)} untouched hours of {windows.TEST[1] - windows.TEST[0] + 1} in the test split: {h[0]}…{h[-1]}")
        return
    if args.train:
        print(markdown(evaluate(t, s, list(range(windows.TRAIN[0], windows.TRAIN[1] + 1)), boot=False), "Training hours 1–300 (in-sample)"))
        return
    id_ = stamp()
    r = {"eval": id_, "script": "fraud/eval_chain.py", "hour_list": untouched_hours(t)}
    r.update(evaluate(t, s, r["hour_list"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{id_}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    md = markdown(r, f"Chain eval {id_}: no review, tightened cash-out rule")
    (RESULTS / f"{id_}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
