"""The pre-registered eval of the fine-tuned brain: 20 NEW test windows × 6 setups → fraud/results/eval/.

    python -m fraud.eval_ft --adapter fraud/models/<id>_lora          # dev gate, thresholds, test windows
    python -m fraud.eval_ft --adapter <dir> --resume <eval id>         # finish an interrupted eval
    python -m fraud.eval_ft --report <eval id>                         # rebuild the results files
    python -m fraud.eval_ft --windows                                  # print the new windows only (no model)

Traces: fraud/runs/<eval id>_dev_<setup>_dev<k>.jsonl (dev windows, model setups only) and
fraud/runs/<eval id>_eval_<setup>_test<k>.jsonl. Results: fraud/results/eval/<eval id>.json + .md
(same schema as fraud/eval.py plus "dev" and "adapter", read by the console's eval page).

PRE-REGISTRATION (written and committed before the adapter was trained; fraud/ROADMAP.md Phase 5)
Question: can a LoRA fine-tuned Qwen2.5-1.5B, behind the same type filter and reading the same
prompt, decide as cheaply as the numpy logistic regression? Owner's choices, 2026-10-05.
- Adapter: the one run of `python -m fraud.finetune` with the configuration in its docstring
  (steps 1–300, TRANSFER/CASH_OUT, minus every hour of dev0–dev9; all 3,021 fraud + 3× legit,
  seed 0; LoRA r16 on q/k/v/o; 1 epoch). Its sha256 is recorded in the results.
- New test windows: windows.make(t, "test", seed) for seed = 200, 201, … (500 rows, 25 fraud
  target, steps 301–743). A seed is skipped if any of its hours lies in a window of test100–test119
  or in an already accepted new window. The first 20 accepted are the test set (ids test<seed>).
  Each runs once per setup. Nothing is tuned on them.
- Setups: hybrid-ft (type filter + the fine-tuned model, prompt v4, order debias on, LM head in
  float32, adapter merged; action from the dev thresholds below), hybrid-1.5b (the frozen Phase 3
  hybrid, argmax), logreg (fitted and thresholded exactly as in fraud/eval.py, on dev0–dev9), rules,
  random (seed = window seed), approve-all. Costs: fraud.costs.DEFAULT.
- Dev gate: hybrid-1.5b and hybrid-ft (argmax) run on dev0–dev9. AUC of the order-averaged
  "decline" score over the model rows (ties count half). The test windows run only if
  AUC(hybrid-ft) > AUC(hybrid-1.5b) on those rows. Otherwise the eval stops, the result is reported,
  and the owner decides whether one more training configuration may be tried (dev only).
- Thresholds (Implementation choice): review if decline score >= t_review, decline if >= t_decline,
  from GRID × GRID with t_review <= t_decline, minimising total cost over dev0–dev9 (unweighted,
  the type filter approving its rows; ties: first in grid order). When the review budget is spent,
  a review falls back to the larger of the approve/decline scores (as in brain.ModelDecider).
- Bar A (pass/fail): hybrid-ft cost <= logreg cost in >= 10 of the 20 new windows.
- Bar B (pass/fail): p90 of hybrid-ft per-decision latency over its model decisions <= 250 ms.
- Reported, no bar: every setup's cost, natural-rate cost, recall, precision, alert rate; paired
  window counts with two-sided sign tests (ties dropped) for hybrid-ft vs logreg, rules,
  hybrid-1.5b and random, and logreg vs rules; dev AUCs; the chosen thresholds; ECE and
  outside_mass of the model scores (uncalibrated). Stated before the run: hybrid-ft cheaper than
  rules in >= 16 of 20 (the old bar 1) is expected to FAIL; rules-only paid only review fees in
  19 of 20 old test windows (fraud/docs/results.md).
- Known asymmetry (stated, not corrected): logreg's features include four account-count signals
  the prompt does not show (orig_out/orig_in/dest_out/dest_transfer_in before).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import brain, costs, rules, windows
from .runs import RUNS, finished, read_run, stamp, to_jsonl

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval"
REPO = ROOT.parent

OLD_TEST_SEEDS = tuple(range(100, 120))  # = fraud.eval.TEST_SEEDS
DEV_SEEDS = tuple(range(0, 10))           # = fraud.eval.DEV_SEEDS
NEW_SEED_FROM, N_NEW, MAX_TRIES = 200, 20, 2000
SETUPS = ("hybrid-ft", "hybrid-1.5b", "logreg", "rules", "random", "approve-all")
MODEL_SETUPS = ("hybrid-1.5b", "hybrid-ft")
GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
BAR_A_WINDOWS = 10
BAR_P90_MS = 250.0


# ---------------------------------------------------------------- windows

def new_windows(t, size: int = 500, fraud_target: int = 25) -> list[windows.Window]:
    """The first N_NEW test-split windows from seed NEW_SEED_FROM on that share no hour with
    test100–test119 or with each other."""
    used: set[int] = set()
    for k in OLD_TEST_SEEDS:
        a, b = windows.make(t, "test", k, size=size, fraud_target=fraud_target).steps
        used |= set(range(a, b + 1))
    out = []
    for seed in range(NEW_SEED_FROM, NEW_SEED_FROM + MAX_TRIES):
        w = windows.make(t, "test", seed, size=size, fraud_target=fraud_target, wid=f"test{seed}")
        hrs = set(range(w.steps[0], w.steps[1] + 1))
        if hrs & used:
            continue
        out.append(w)
        used |= hrs
        if len(out) == N_NEW:
            return out
    raise ValueError(f"only {len(out)} non-overlapping windows in {MAX_TRIES} seeds")


# ---------------------------------------------------------------- dev: AUC and thresholds

def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def pick_thresholds(scores, truth, amount, filt, fallbacks=None) -> tuple[float, float, list]:
    """Per dev window: decline score per row, truth, amount, type-filter mask, optional fallbacks."""
    grid = []
    for tr in GRID:
        for td in GRID:
            if td < tr:
                continue
            total = 0.0
            for k, (q, y, a, f) in enumerate(zip(scores, truth, amount, filt)):
                wanted = ["approve" if fk else "decline" if qk >= td else "review" if qk >= tr else "approve"
                          for qk, fk in zip(q, f)]
                fb = (["approve" if fk else b for b, fk in zip(fallbacks[k], f)] if fallbacks is not None
                      else ["approve" if fk else "decline" if qk >= 0.5 else "approve" for qk, fk in zip(q, f)])
                total += costs.score(costs.DEFAULT, wanted, fb, y, a).summary["cost"]
            grid.append((tr, td, total))
    best = min(grid, key=lambda g: g[2])  # min keeps the first of equal costs
    return best[0], best[1], grid


def dev_view(traces: list[list[dict]]) -> dict:
    """Arrays from dev traces: per window decline scores, fallbacks, truth, amount, filter mask."""
    v = {"scores": [], "fallbacks": [], "truth": [], "amount": [], "filt": []}
    for recs in traces:
        tx = [r for r in recs if r.get("type") == "tx"]
        v["scores"].append(np.array([r["scores"]["decline"] if r["by"] == "model" else 0.0 for r in tx]))
        v["fallbacks"].append([r["fallback"] for r in tx])
        v["truth"].append(np.array([r["truth"] for r in tx]))
        v["amount"].append(np.array([r["tx"]["amount"] for r in tx]))
        v["filt"].append(np.array([r["by"] == "filter" for r in tx]))
    return v


def dev_auc(v: dict) -> tuple[float, int, int]:
    m = ~np.concatenate(v["filt"])
    q, y = np.concatenate(v["scores"])[m], np.concatenate(v["truth"])[m]
    return auc(q, y), int(m.sum()), int(y.sum())


# ---------------------------------------------------------------- the run

def trace_file(eval_id: str, stage: str, setup: str, wid: str) -> Path:
    return RUNS / f"{eval_id}_{stage}_{setup}_{wid}.jsonl"


def make_model_engine(setup: str, adapter: Path | None):
    from system_one import load_config, make_engine

    from .capture import HEAD_DTYPE

    cfg = load_config(REPO / "config" / "default.toml")
    cfg["engine"]["order_debias"] = True
    cfg["backend"]["head_dtype"] = HEAD_DTYPE
    engine = make_engine(cfg)
    info = engine.backend.info()
    if setup == "hybrid-ft":
        man = brain.load_adapter(engine, adapter)
        info["adapter"] = {"dir": adapter.name, "sha256": man["adapter_sha256"], "rows_sha256": man["rows_sha256"],
                           "steps": man["steps"], "lora": man["lora"]}
    if info.get("kind") != "mock":
        brain.warm_up(engine)
    return engine, info


def release() -> None:
    """Return freed model memory to the device. Call after the last reference to an engine is gone
    (two 1.5B engines do not fit the iGPU at once)."""
    gc.collect()
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except ImportError:
        pass


def run_traces(eval_id, stage, setup, ws, chain_of, t, s, backend, log, header=None) -> list[list[dict]]:
    from .capture import run_window

    out = []
    for w in ws:
        path = trace_file(eval_id, stage, setup, w.id)
        if not finished(path):
            t0 = time.perf_counter()
            recs = run_window(t, s, w, chain_of(w), costs.DEFAULT, setup, backend)
            recs[0]["eval"] = eval_id
            recs[0].update(header or {})
            RUNS.mkdir(parents=True, exist_ok=True)
            path.write_text(to_jsonl(recs), encoding="utf-8")
            sm = recs[-1]["summary"]
            log(f"{stage:4} {setup:12} {w.id:8} cost {sm['cost']:>14,.0f}  recall {sm['recall']:.2f}  "
                f"p90 {sm['latency_ms']['p90']:6.1f} ms  ({time.perf_counter() - t0:5.1f} s)")
        out.append(read_run(path))
    return out


def run(eval_id: str, adapter: Path, setups: list[str], log=print) -> dict:
    from . import eval as ev
    from .capture import chain_for, load_all

    t, s = load_all()
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    test = new_windows(t)
    log("new test windows: " + ", ".join(f"{w.id} {w.steps[0]}-{w.steps[1]}" for w in test))

    # Dev gate (both model setups, argmax actions; the traces keep the scores).
    dev_traces, infos = {}, {}
    for setup in MODEL_SETUPS:
        engine = None
        if not all(finished(trace_file(eval_id, "dev", setup, w.id)) for w in dev):
            engine, infos[setup] = make_model_engine(setup, adapter)
        dev_traces[setup] = run_traces(eval_id, "dev", setup, dev,
                                       lambda w: [rules.TypeFilter(), brain.ModelDecider(engine)],
                                       t, s, infos.get(setup, {}), log)
        engine = None
        release()
    views = {k: dev_view(v) for k, v in dev_traces.items()}
    aucs = {k: dev_auc(v) for k, v in views.items()}
    log("dev AUC (decline score, model rows): " + ", ".join(f"{k} {a:.4f} ({n} rows, {f} fraud)"
                                                            for k, (a, n, f) in aucs.items()))
    gate = aucs["hybrid-ft"][0] > aucs["hybrid-1.5b"][0]
    v = views["hybrid-ft"]
    tr, td, grid = pick_thresholds(v["scores"], v["truth"], v["amount"], v["filt"], v["fallbacks"])
    log(f"gate {'PASS' if gate else 'FAIL'}; hybrid-ft thresholds: review >= {tr}, decline >= {td}")
    dev_info = {"auc": {k: {"auc": a, "rows": n, "fraud": f} for k, (a, n, f) in aucs.items()},
                "gate": {"rule": "AUC(hybrid-ft) > AUC(hybrid-1.5b) on dev0-dev9 model rows", "pass": gate},
                "thresholds": {"review": tr, "decline": td},
                "grid_best": sorted(grid, key=lambda g: g[2])[:5]}
    if not gate:
        return report(eval_id, adapter, dev_info, log)

    lr = ev.logreg_decider(t, s) if "logreg" in setups else None
    for setup in setups:
        engine, info = None, {}
        if setup in MODEL_SETUPS and not all(finished(trace_file(eval_id, "eval", setup, w.id)) for w in test):
            engine, info = make_model_engine(setup, adapter)
        if setup == "hybrid-ft":
            chain_of = lambda w: [rules.TypeFilter(), brain.ThresholdDecider(engine, tr, td)]
        elif setup == "hybrid-1.5b":
            chain_of = lambda w: [rules.TypeFilter(), brain.ModelDecider(engine)]
        else:
            chain_of = lambda w, setup=setup: chain_for(setup, None, lr, w.seed)
        run_traces(eval_id, "eval", setup, test, chain_of, t, s, info, log,
                   {"thresholds": {"review": tr, "decline": td}} if setup == "hybrid-ft" else None)
        engine = None
        release()
    return report(eval_id, adapter, dev_info, log)


# ---------------------------------------------------------------- the report

def report(eval_id: str, adapter: Path | None, dev_info: dict | None = None, log=print) -> dict:
    from . import eval as ev

    from .capture import load_all

    path = RESULTS / f"{eval_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    dev_info = dev_info or old.get("dev")
    t, _ = load_all()
    test_ids = [w.id for w in new_windows(t)]
    per: dict[str, dict[str, dict]] = {}
    traces: dict[str, list] = {}
    for setup in SETUPS:
        for wid in test_ids:
            p = trace_file(eval_id, "eval", setup, wid)
            if finished(p):
                recs = read_run(p)
                per.setdefault(setup, {})[wid] = recs[-1]["summary"]
                traces.setdefault(setup, []).append(recs)

    def paired(a, b):
        ws = [w for w in per.get(a, {}) if w in per.get(b, {})]
        wn = sum(per[a][w]["cost"] < per[b][w]["cost"] for w in ws)
        ls = sum(per[a][w]["cost"] > per[b][w]["cost"] for w in ws)
        return {"windows": len(ws), "a_cheaper": wn, "b_cheaper": ls, "ties": len(ws) - wn - ls,
                "sign_test_p": ev.sign_test(wn, ls)}

    def model_ms(setup):
        return [r["ms"] for recs in traces.get(setup, []) for r in recs if r.get("type") == "tx" and r["by"] == "model"]

    totals = {}
    for setup, ws in per.items():
        vals = list(ws.values())
        totals[setup] = {
            "windows": len(vals),
            "cost_total": float(sum(v["cost"] for v in vals)),
            "cost_mean": float(np.mean([v["cost"] for v in vals])),
            "cost_median": float(np.median([v["cost"] for v in vals])),
            "cost_natural_mean": float(np.mean([v["cost_natural"] for v in vals])),
            "recall_mean": float(np.mean([v["recall"] for v in vals])),
            "precision_mean": float(np.mean([v["precision"] for v in vals])),
            "alert_rate_mean": float(np.mean([v["alert_rate"] for v in vals])),
            "fraud_amount_stopped_share": float(sum(v["fraud_amount_stopped"] for v in vals)
                                                / max(1e-9, sum(v["fraud_amount"] for v in vals))),
        }
        ms = model_ms(setup)
        if ms:
            totals[setup]["latency_ms"] = {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
                                           "max": float(max(ms)), "decisions": len(ms)}
            totals[setup]["calibration"] = ev.model_calibration(traces[setup])
    a = paired("hybrid-ft", "logreg")
    not_worse = a["a_cheaper"] + a["ties"]
    ms = model_ms("hybrid-ft")
    p90 = float(np.percentile(ms, 90)) if ms else None
    complete = all(len(per.get(k, {})) == N_NEW for k in ("hybrid-ft", "logreg"))
    out = {
        "eval": eval_id,
        "script": "fraud/eval_ft.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complete": complete,
        "costs": costs.DEFAULT.as_dict(),
        "test_windows": test_ids,
        "dev": dev_info,
        "adapter": (adapter.name if adapter else old.get("adapter")),
        "bars": {
            "barA": {"rule": f"hybrid-ft cost <= logreg cost in >= {BAR_A_WINDOWS} of {N_NEW} new windows",
                     "value": not_worse if complete else None, "pass": complete and not_worse >= BAR_A_WINDOWS},
            "barB": {"rule": f"hybrid-ft p90 latency <= {BAR_P90_MS:.0f} ms", "value": p90,
                     "pass": p90 is not None and p90 <= BAR_P90_MS},
        },
        "paired": {
            "hybrid-ft vs logreg": a,
            "hybrid-ft vs rules": paired("hybrid-ft", "rules"),
            "hybrid-ft vs hybrid-1.5b": paired("hybrid-ft", "hybrid-1.5b"),
            "hybrid-ft vs random": paired("hybrid-ft", "random"),
            "logreg vs rules": paired("logreg", "rules"),
        },
        "totals": totals,
        "windows": per,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS / f"{eval_id}.md").write_text(markdown(out), encoding="utf-8")
    log(markdown(out))
    return out


def markdown(r: dict) -> str:
    lines = [f"# Fraud eval {r['eval']} (fine-tuned brain)", "",
             f"Created {r['created']}. Complete: {r['complete']}. Adapter: `{r['adapter']}`. "
             "Pre-registration: the docstring of `fraud/eval_ft.py`. All numbers are Observed.", ""]
    d = r.get("dev")
    if d:
        lines += ["## Dev gate (dev0–dev9)", ""]
        for k, v in d["auc"].items():
            lines.append(f"- AUC of the decline score, {k}: **{v['auc']:.4f}** ({v['rows']} model rows, {v['fraud']} fraud)")
        lines += [f"- Gate: {d['gate']['rule']} → **{'PASS' if d['gate']['pass'] else 'FAIL'}**",
                  f"- hybrid-ft thresholds: review ≥ {d['thresholds']['review']}, decline ≥ {d['thresholds']['decline']}", ""]
    lines += ["## Bars", ""]
    for k, b in r["bars"].items():
        val = "–" if b["value"] is None else (f"{b['value']:.1f} ms" if isinstance(b["value"], float) else b["value"])
        lines.append(f"- {k}: {b['rule']}: **{val}** → **{'PASS' if b['pass'] else 'FAIL'}**")
    if r["totals"]:
        lines += ["", "## Setups (over the new test windows)", "",
                  "| setup | windows | mean cost | median cost | cost (natural rate) | recall | precision | alert rate | fraud $ stopped | p50 / p90 ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in r["totals"].items():
            lat = v.get("latency_ms")
            lines.append(f"| {k} | {v['windows']} | {v['cost_mean']:,.0f} | {v['cost_median']:,.0f} | {v['cost_natural_mean']:,.0f} | "
                         f"{v['recall_mean']:.2f} | {v['precision_mean']:.2f} | {v['alert_rate_mean']:.3f} | "
                         f"{v['fraud_amount_stopped_share']:.1%} | {f'{lat['p50']:.0f} / {lat['p90']:.0f}' if lat else '–'} |")
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, help="the adapter folder written by python -m fraud.finetune")
    ap.add_argument("--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    ap.add_argument("--resume", metavar="EVAL_ID", help="continue this eval: skip its finished traces")
    ap.add_argument("--report", metavar="EVAL_ID", help="only rebuild the results files from this eval's traces")
    ap.add_argument("--windows", action="store_true", help="print the new test windows and exit")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if args.windows:
        from .capture import load_all
        t, _ = load_all()
        for w in new_windows(t):
            log(f"{w.id:8} steps {w.steps[0]}-{w.steps[1]}  rows {len(w.rows)}  fraud {w.fraud}")
        return
    if args.report:
        report(args.report, args.adapter, log=log)
        return
    if args.adapter is None or not (args.adapter / "manifest.json").exists():
        ap.error("--adapter must be a folder with manifest.json and adapter/ (python -m fraud.finetune)")
    run(args.resume or stamp(), args.adapter.resolve(), args.setups, log)


if __name__ == "__main__":
    main()
