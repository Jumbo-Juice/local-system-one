"""Could a second model replace part of the human review? A no-training check on the dev windows.

    python -m fraud.reviewer_check                  # the scorers already traced on dev
    python -m fraud.reviewer_check --d1ft <eval id> # plus the fine-tuned d1-Omni-2B of that eval
    python -m fraud.reviewer_check --probe          # the queue on the training steps: what separates it?

Exploratory, dev rows only, nothing pre-registered, no test window is read: it decides whether a reviewer
is worth building. The queue is what the hand rules send to review (rules.rule_for: TRANSFER to a new
receiver, or of an amount moved this hour), so no fine-tune's training rows shape it. Each scorer's existing
score on those rows (recorded traces, no inference) gives a reviewer that may block a transaction or pass it
to a human. It never approves. Cost: a human review 50; a correct block 0; a wrong block
costs.unit_cost's friction.

- blocks_before_wrong: frauds scored above the highest-scored legit row; those blocks are free of mistakes.
- cross-fold: a threshold picked on dev0–4 is scored on dev5–9 and the reverse, so its saving is out of sample.
- natural: the same with each row's trace weight, which restores the natural fraud rate of the window's span.
  Dev windows are enriched to about 5% fraud, so the unweighted queue holds far more fraud than real traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import costs, rules, windows
from .data import TYPE_ID
from .eval_ft import auc, trace_file
from .runs import read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "reviewer_check"
DEV_SEEDS = tuple(range(0, 10))
FOLDS = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
# name: (eval, setup, score key, order index or None for the stored average)
SCORERS = {
    "logreg": ("20261030-122539", "logreg", "fraud", None),
    "d1-omni frozen": ("20261030-122539", "hybrid-d1omni2b", "decline", None),
    "d3-mini frozen": ("20261021-020632", "hybrid-d3mini", "decline", None),
    "hybrid-ft (1.5B)": ("20261005-022333", "hybrid-ft", "decline", None),
    "hybrid-d3ft": ("20261021-155222", "hybrid-d3ft", "decline", None),
}


def friction(amount: np.ndarray) -> np.ndarray:
    return np.array([costs.unit_cost(costs.DEFAULT, "decline", False, float(a)) for a in amount])


def queue_cost(q, y, a, w, threshold: float) -> float:
    """Total cost of the queue when scores >= threshold are blocked and the rest go to a human."""
    block = q >= threshold
    return float((w * np.where(block, np.where(y == 1, 0.0, friction(a)), costs.DEFAULT.review_fee)).sum())


def best_threshold(q, y, a, w) -> float:
    """The cheapest threshold over the observed scores; infinity (block nothing) unless blocking saves."""
    best, cost = np.inf, queue_cost(q, y, a, w, np.inf)
    for th in sorted(set(q.tolist()), reverse=True):  # highest first: an equal cost keeps the stricter one
        c = queue_cost(q, y, a, w, th)
        if c < cost:
            best, cost = th, c
    return best


def cross_fold(q, y, a, w, k) -> dict:
    """Pick on one fold, score on the other, both ways."""
    total, wrong, blocked, frauds_blocked, thresholds = 0.0, 0, 0, 0, []
    for pick, score in (FOLDS, FOLDS[::-1]):
        p, m = np.isin(k, pick), np.isin(k, score)
        th = best_threshold(q[p], y[p], a[p], w[p])
        thresholds.append(None if np.isinf(th) else float(th))
        total += queue_cost(q[m], y[m], a[m], w[m], th)
        hit = m & (q >= th)
        blocked += int(hit.sum())
        frauds_blocked += int(y[hit].sum())
        wrong += int((y[hit] == 0).sum())
    return {"cost": total, "blocked": blocked, "frauds_blocked": frauds_blocked, "wrong_blocks": wrong, "thresholds": thresholds}


def evaluate(q, y, a, w, k) -> dict:
    n = len(DEV_SEEDS)
    human = queue_cost(q, y, a, np.ones_like(q), np.inf) / n
    human_nat = queue_cost(q, y, a, w, np.inf) / n
    legit_top = q[y == 0].max()
    one = np.ones_like(q)
    plain, natural = cross_fold(q, y, a, one, k), cross_fold(q, y, a, w, k)
    in_sample = queue_cost(q, y, a, one, best_threshold(q, y, a, one)) / n
    return {"auc": auc(q, y), "blocks_before_wrong": int((q[y == 1] > legit_top).sum()),
            "human_cost_per_window": human, "in_sample_cost_per_window": in_sample,
            "cross_fold": {**plain, "cost_per_window": plain["cost"] / n,
                           "saving_share": 1 - plain["cost"] / (human * n)},
            "natural": {"human_cost_per_window": human_nat, **natural, "cost_per_window": natural["cost"] / n,
                        "saving_share": 1 - natural["cost"] / (human_nat * n)}}


def read_queue(name: str, spec, t, s) -> dict:
    """Scorer `name`'s queue rows: {(window, row): (score, truth, amount, weight)}."""
    eval_id, setup, key, _ = spec
    out = {}
    for k in DEV_SEEDS:
        for r in read_run(trace_file(eval_id, "dev", setup, f"dev{k}")):
            if r.get("type") == "tx" and rules.RULES[rules.rule_for(t, s, r["row"])][1] == "review":
                out[(k, r["row"])] = (r["scores"][key], r["truth"], r["tx"]["amount"], r.get("weight", 1.0))
    return out


def probe(t, s) -> dict:
    """The rules' queue on the training steps (no test or dev choice rests on it): each signal's AUC at decision
    time, and whether a CASH_OUT of exactly this amount follows in the same hour. The second is not a signal for
    this row (no look-ahead, fraud/CLAUDE.md): it measures what would be known after the decision."""
    train = np.flatnonzero((t.step >= windows.TRAIN[0]) & (t.step <= windows.TRAIN[1]))
    tr = train[t.type[train] == TYPE_ID["TRANSFER"]]
    queue = np.array([i for i in tr if rules.RULES[rules.rule_for(t, s, int(i))][1] == "review"])
    y = t.is_fraud[queue].astype(int)
    aucs = {"amount": auc(t.amount[queue].astype(float), y)}
    for name in s:
        v = np.asarray(s[name])[queue].astype(float)
        aucs[name] = auc(v, y) if v.std() > 0 else None
    cash = {}
    for i in np.flatnonzero(t.type == TYPE_ID["CASH_OUT"]):
        cash.setdefault(float(t.amount[i]), []).append(int(i))
    after = np.array([any(c > i and t.step[c] == t.step[i] for c in cash.get(float(t.amount[i]), [])) for i in queue])
    return {"rows": len(queue), "fraud": int(y.sum()), "legit": int((y == 0).sum()), "auc": aucs,
            "cash_out_after_same_hour": {"fraud": int(after[y == 1].sum()), "legit": int(after[y == 0].sum())}}


def probe_markdown(r: dict, id_: str) -> str:
    a = r["cash_out_after_same_hour"]
    L = [f"# Review queue on the training steps {id_}", "",
         "Exploratory, training steps 1–300. Docstring: `fraud/reviewer_check.py`. All numbers are Observed.", "",
         f"Queue: {r['rows']:,} rows, {r['fraud']:,} fraud ({r['fraud'] / r['rows']:.2%}), {r['legit']:,} legit.", "",
         "| known at decision time | AUC |", "|---|---:|"]
    L += [f"| {k} | {'constant' if v is None else f'{v:.4f}'} |" for k, v in r["auc"].items()]
    L += ["", "| known only afterwards | fraud | legit |", "|---|---:|---:|",
          f"| a CASH_OUT of exactly this amount follows in the same hour | {a['fraud']:,} of {r['fraud']:,} "
          f"({a['fraud'] / r['fraud']:.2%}) | {a['legit']:,} of {r['legit']:,} ({a['legit'] / r['legit']:.3%}) |"]
    return "\n".join(L) + "\n"


def markdown(out: dict) -> str:
    q = out["queue"]
    L = [f"# Reviewer check {out['id']}", "",
         "Exploratory, dev windows only (dev0–dev9), recorded scores, no training, no test window. Docstring: "
         "`fraud/reviewer_check.py`. All numbers are Observed; model scores are uncalibrated.", "",
         f"Queue (the rules' review rows): {q['rows']} rows, {q['fraud']} fraud, {q['legit']} legit; all human: "
         f"{q['human_cost_per_window']:,.0f} per window. Fraud share at the natural rate: {q['natural_fraud_share']:.2%}.", "",
         "| scorer | AUC on the queue | frauds blocked before the first wrong block | cost per window, in-sample | "
         "cost per window, cross-fold | saving | blocks / wrong | natural: saving | natural: blocks / wrong |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in out["scorers"].items():
        c, n = r["cross_fold"], r["natural"]
        L.append(f"| {name} | {r['auc']:.4f} | {r['blocks_before_wrong']} of {q['fraud']} | {r['in_sample_cost_per_window']:,.0f} | "
                 f"{c['cost_per_window']:,.0f} | {c['saving_share']:+.0%} | {c['blocked']} / {c['wrong_blocks']} | "
                 f"{n['saving_share']:+.0%} | {n['blocked']} / {n['wrong_blocks']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    from .capture import load_all

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d1ft", metavar="EVAL_ID", help="add the fine-tuned d1-Omni-2B of this eval")
    ap.add_argument("--probe", action="store_true", help="the queue on the training steps")
    args = ap.parse_args()
    if args.probe:
        t, s = load_all()
        id_, r = stamp(), None
        r = probe(t, s)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{id_}_probe.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        md = probe_markdown(r, id_)
        (RESULTS / f"{id_}_probe.md").write_text(md, encoding="utf-8")
        print(md)
        return
    scorers = dict(SCORERS)
    if args.d1ft:
        scorers["d1-omni fine-tuned"] = (args.d1ft, "hybrid-d1ft", "decline", None)
    t, s = load_all()
    rows = {name: read_queue(name, spec, t, s) for name, spec in scorers.items()}
    keys = sorted(next(iter(rows.values())))
    if any(sorted(v) != keys for v in rows.values()):
        raise RuntimeError("the scorers' queues differ")
    base = next(iter(rows.values()))
    y = np.array([base[k][1] for k in keys])
    a = np.array([base[k][2] for k in keys])
    w = np.array([base[k][3] for k in keys], dtype=float)
    k = np.array([key[0] for key in keys])
    out = {"id": stamp(), "script": "fraud/reviewer_check.py", "scorers": {},
           "queue": {"rows": len(keys), "fraud": int(y.sum()), "legit": int((y == 0).sum()),
                     "human_cost_per_window": len(keys) * costs.DEFAULT.review_fee / len(DEV_SEEDS),
                     "natural_fraud_share": float((w * y).sum() / w.sum())}}
    for name, v in rows.items():
        out["scorers"][name] = evaluate(np.array([v[key][0] for key in keys], dtype=float), y, a, w, k)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{out['id']}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{out['id']}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
