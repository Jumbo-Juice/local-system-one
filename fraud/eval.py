"""The pre-registered fraud eval: 20 held-out test windows × 6 setups → fraud/results/eval/.

    python -m fraud.eval                          # everything (about 40 min on the Lenovo)
    python -m fraud.eval --setups rules logreg    # a subset (no model)
    python -m fraud.eval --resume <eval id>       # finish an interrupted eval (same id, skips finished traces)
    python -m fraud.eval --report <eval id>       # rebuild the results files from that eval's traces

Each trace is fraud/runs/<eval id>_eval_<setup>_<window>.jsonl. The results are
fraud/results/eval/<eval id>.json (read by the console's eval page) and
fraud/results/eval/<eval id>.md.

PRE-REGISTRATION (written and committed before any test window ran; see fraud/PROGRESS.md)
- Test windows: test100 … test119 (windows.make(t, "test", seed), 500 rows, 25 fraud target),
  steps 301–743. Nothing else runs on them: prompts, rules and thresholds were set on dev0–dev9.
- Setups: hybrid-1.5b (config/default.toml, order_debias on, LM head in bfloat16),
  hybrid-3b (config/lenovo-3b.toml, order_debias on, LM head in bfloat16), rules, logreg, random (seed = the window seed), approve-all.
- Costs: fraud.costs.DEFAULT (friction 10% of the amount, min 10; review 50; budget 5%).
- Logistic regression: fitted on every TRANSFER and CASH_OUT of steps 1–300. Thresholds from the
  grid below, minimising total cost over dev0–dev9 (ties: first in grid order).
- Bar 1 (pass/fail): hybrid-1.5b cost < rules cost in >= 16 of 20 test windows.
- Bar 2 (pass/fail): p90 of hybrid-1.5b per-decision latency over all its model decisions in the
  20 test windows <= 250 ms.
- Reported, no bar: every setup's cost, natural-rate cost, recall, precision, alert rate; the
  hybrid vs logreg and hybrid-3b vs rules window counts; ECE of the model's top score against
  "the right action" (legit → approve, fraud → review or decline), 10 equal-width bins; mean
  outside_mass; a two-sided sign test p-value for each paired count.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from . import baselines, brain, costs, rules, windows
from .capture import HEAD_DTYPE, chain_for, load_all, run_window
from .data import TYPE_ID
from .runs import RUNS, finished, model_tag, read_run, stamp, to_jsonl

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval"
REPO = ROOT.parent

DEV_SEEDS = tuple(range(0, 10))
TEST_SEEDS = tuple(range(100, 120))
SETUPS = {
    "hybrid-1.5b": REPO / "config" / "default.toml",
    "hybrid-3b": REPO / "config" / "lenovo-3b.toml",
    "rules": None,
    "logreg": None,
    "random": None,
    "approve-all": None,
}
BAR_WINDOWS = 16
BAR_P90_MS = 250.0
T_REVIEW = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
T_DECLINE = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


# ---------------------------------------------------------------- logistic regression

def fit_logreg(t, s) -> baselines.LogReg:
    lo, hi = windows.TRAIN
    m = (t.step >= lo) & (t.step <= hi) & ((t.type == TYPE_ID["TRANSFER"]) | (t.type == TYPE_ID["CASH_OUT"]))
    idx = np.flatnonzero(m)
    return baselines.LogReg().fit(baselines.features(t, s, idx), t.is_fraud[idx].astype(float))


def pick_thresholds(t, s, model: baselines.LogReg) -> tuple[float, float, list]:
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    probs = [model.predict(baselines.features(t, s, w.rows)) for w in dev]
    grid = []
    for tr in T_REVIEW:
        for td in T_DECLINE:
            if td < tr:
                continue
            total = 0.0
            for w, p in zip(dev, probs):
                wanted = ["decline" if q >= td else "review" if q >= tr else "approve" for q in p]
                fb = ["decline" if q >= 0.5 else "approve" for q in p]
                filt = np.isin(t.type[w.rows], list(rules.FRAUD_FREE))
                wanted = ["approve" if f else a for a, f in zip(wanted, filt)]
                fb = ["approve" if f else a for a, f in zip(fb, filt)]
                total += costs.score(costs.DEFAULT, wanted, fb, t.is_fraud[w.rows], t.amount[w.rows]).summary["cost"]
            grid.append((tr, td, total))
    best = min(grid, key=lambda g: g[2])  # min keeps the first of equal costs
    return best[0], best[1], grid


def logreg_decider(t, s) -> baselines.LogRegDecider:
    model = fit_logreg(t, s)
    tr, td, _ = pick_thresholds(t, s, model)
    return baselines.LogRegDecider(model, tr, td)


# ---------------------------------------------------------------- statistics

def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    if not len(conf):
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0 else (conf >= a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def model_calibration(traces: list[list[dict]]) -> dict:
    conf, right, outside = [], [], []
    for recs in traces:
        for r in recs:
            if r.get("type") == "tx" and r.get("by") == "model":
                top = max(r["scores"], key=r["scores"].get)
                conf.append(r["scores"][top])
                right.append((top == "approve") == (r["truth"] == 0))
                outside.append(r["outside_mass"])
    return {"decisions": len(conf), "ece": ece(conf, right), "mean_top_score": float(np.mean(conf)) if conf else None,
            "top_right_rate": float(np.mean(right)) if right else None,
            "outside_mass_mean": float(np.mean(outside)) if outside else None}


# ---------------------------------------------------------------- the run

def trace_file(eval_id: str, setup: str, wid: str) -> Path:
    return RUNS / f"{eval_id}_eval_{setup}_{wid}.jsonl"


def run(eval_id: str, setups: list[str]) -> None:
    from system_one import load_config, make_engine

    t, s = load_all()
    test = [windows.make(t, "test", k, wid=f"test{k}") for k in TEST_SEEDS]
    lr = logreg_decider(t, s) if "logreg" in setups else None
    if lr:
        print(f"logreg thresholds: review >= {lr.t_review}, decline >= {lr.t_decline}", flush=True)
    for setup in setups:
        engine, backend = None, {}
        for w in test:
            path = trace_file(eval_id, setup, w.id)
            if finished(path):
                continue
            if setup.startswith("hybrid") and engine is None:
                cfg = load_config(SETUPS[setup])
                cfg["engine"]["order_debias"] = True
                cfg["backend"]["head_dtype"] = HEAD_DTYPE
                engine = make_engine(cfg)
                backend = engine.backend.info()
                if backend.get("kind") != "mock":
                    brain.warm_up(engine)
            kind = "hybrid" if setup.startswith("hybrid") else setup
            t0 = time.perf_counter()
            recs = run_window(t, s, w, chain_for(kind, engine, lr, w.seed), costs.DEFAULT, setup, backend)
            recs[0]["eval"] = eval_id
            RUNS.mkdir(parents=True, exist_ok=True)
            path.write_text(to_jsonl(recs), encoding="utf-8")
            summ = recs[-1]["summary"]
            print(f"{setup:12} {w.id:8} cost {summ['cost']:>14,.0f}  recall {summ['recall']:.2f}  "
                  f"p90 {summ['latency_ms']['p90']:6.1f} ms  ({time.perf_counter() - t0:5.1f} s)", flush=True)
        if engine is not None:
            del engine
            gc.collect()
            try:
                import torch
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except ImportError:
                pass
    report(eval_id)


def report(eval_id: str) -> dict:
    per: dict[str, dict[str, dict]] = {}
    traces: dict[str, list] = {}
    for setup in SETUPS:
        for k in TEST_SEEDS:
            path = trace_file(eval_id, setup, f"test{k}")
            if finished(path):
                recs = read_run(path)
                per.setdefault(setup, {})[f"test{k}"] = recs[-1]["summary"]
                traces.setdefault(setup, []).append(recs)
    wins = lambda a, b: [w for w in per.get(a, {}) if w in per.get(b, {}) and per[a][w]["cost"] < per[b][w]["cost"]]
    loss = lambda a, b: [w for w in per.get(a, {}) if w in per.get(b, {}) and per[a][w]["cost"] > per[b][w]["cost"]]

    def paired(a, b):
        n = len([w for w in per.get(a, {}) if w in per.get(b, {})])
        wn, ls = len(wins(a, b)), len(loss(a, b))
        return {"windows": n, "a_cheaper": wn, "b_cheaper": ls, "ties": n - wn - ls, "sign_test_p": sign_test(wn, ls)}

    def model_ms(setup):
        return [r["ms"] for recs in traces.get(setup, []) for r in recs if r.get("type") == "tx" and r["by"] == "model"]

    totals = {}
    for setup, ws in per.items():
        vals = list(ws.values())
        totals[setup] = {
            "windows": len(vals),
            "cost_total": float(sum(v["cost"] for v in vals)),
            "cost_mean": float(np.mean([v["cost"] for v in vals])),
            "cost_natural_mean": float(np.mean([v["cost_natural"] for v in vals])),
            "recall_mean": float(np.mean([v["recall"] for v in vals])),
            "precision_mean": float(np.mean([v["precision"] for v in vals])),
            "alert_rate_mean": float(np.mean([v["alert_rate"] for v in vals])),
            "fraud_amount_stopped_share": float(sum(v["fraud_amount_stopped"] for v in vals) / max(1e-9, sum(v["fraud_amount"] for v in vals))),
        }
        ms = model_ms(setup)
        if ms:
            totals[setup]["latency_ms"] = {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
                                           "max": float(max(ms)), "decisions": len(ms)}
            totals[setup]["calibration"] = model_calibration(traces[setup])
    b1 = paired("hybrid-1.5b", "rules")
    ms = model_ms("hybrid-1.5b")
    p90 = float(np.percentile(ms, 90)) if ms else None
    complete = len(per.get("hybrid-1.5b", {})) == len(TEST_SEEDS) and len(per.get("rules", {})) == len(TEST_SEEDS)
    out = {
        "eval": eval_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complete": complete,
        "costs": costs.DEFAULT.as_dict(),
        "test_windows": [f"test{k}" for k in TEST_SEEDS],
        "bars": {
            "bar1": {"rule": f"hybrid-1.5b cheaper than rules in >= {BAR_WINDOWS} of {len(TEST_SEEDS)} windows",
                     "value": b1["a_cheaper"], "pass": complete and b1["a_cheaper"] >= BAR_WINDOWS},
            "bar2": {"rule": f"hybrid-1.5b p90 latency <= {BAR_P90_MS:.0f} ms", "value": p90,
                     "pass": p90 is not None and p90 <= BAR_P90_MS},
        },
        "paired": {
            "hybrid-1.5b vs rules": b1,
            "hybrid-1.5b vs logreg": paired("hybrid-1.5b", "logreg"),
            "hybrid-3b vs rules": paired("hybrid-3b", "rules"),
            "hybrid-1.5b vs random": paired("hybrid-1.5b", "random"),
            "logreg vs rules": paired("logreg", "rules"),
        },
        "totals": totals,
        "windows": per,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS / f"{eval_id}.md").write_text(markdown(out), encoding="utf-8")
    print(markdown(out))
    return out


def markdown(r: dict) -> str:
    b = r["bars"]
    lines = [f"# Fraud eval {r['eval']}", "", f"Created {r['created']}. Complete: {r['complete']}. "
             "Pre-registration: the docstring of `fraud/eval.py`. All numbers are Observed.", "",
             "## Bars", "",
             f"- Bar 1: {b['bar1']['rule']}: **{b['bar1']['value']}** → **{'PASS' if b['bar1']['pass'] else 'FAIL'}**",
             f"- Bar 2: {b['bar2']['rule']}: **{b['bar2']['value'] if b['bar2']['value'] is None else round(b['bar2']['value'], 1)} ms** → "
             f"**{'PASS' if b['bar2']['pass'] else 'FAIL'}**", "",
             "## Setups (means over the test windows)", "",
             "| setup | windows | cost | cost (natural rate) | recall | precision | alert rate | fraud $ stopped | p50 / p90 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in r["totals"].items():
        lat = v.get("latency_ms")
        lines.append(f"| {k} | {v['windows']} | {v['cost_mean']:,.0f} | {v['cost_natural_mean']:,.0f} | {v['recall_mean']:.2f} | "
                     f"{v['precision_mean']:.2f} | {v['alert_rate_mean']:.3f} | {v['fraud_amount_stopped_share']:.1%} | "
                     f"{f'{lat['p50']:.0f} / {lat['p90']:.0f}' if lat else '–'} |")
    lines += ["", "## Paired window counts (cheaper = lower cost in that window)", "",
              "| pair | windows | first cheaper | second cheaper | ties | sign test p |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in r["paired"].items():
        lines.append(f"| {k} | {v['windows']} | {v['a_cheaper']} | {v['b_cheaper']} | {v['ties']} | {v['sign_test_p']:.3g} |")
    cal = {k: v["calibration"] for k, v in r["totals"].items() if "calibration" in v}
    if cal:
        lines += ["", "## Model scores (uncalibrated)", "", "| setup | decisions | top score mean | top right | ECE | outside_mass |",
                  "|---|---:|---:|---:|---:|---:|"]
        for k, c in cal.items():
            lines.append(f"| {k} | {c['decisions']} | {c['mean_top_score']:.3f} | {c['top_right_rate']:.3f} | {c['ece']:.3f} | "
                         f"{c['outside_mass_mean']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    ap.add_argument("--resume", metavar="EVAL_ID", help="continue this eval: skip its finished traces")
    ap.add_argument("--report", metavar="EVAL_ID", help="only rebuild the results files from this eval's traces")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    run(args.resume or stamp(), args.setups)


if __name__ == "__main__":
    main()
