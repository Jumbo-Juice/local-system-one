"""Does training on every fit fraud lower the cost? IEEE-CIS, new test windows → fraud/results/eval_ieee_full/.

    python -m fraud.eval_ieee_full --windows            # the new windows; no model, no labels
    python -m fraud.eval_ieee_full --adapter <dir>      # the eval: dev, thresholds, new test windows, report
    python -m fraud.eval_ieee_full --adapter <dir> --resume <eval id>  # finish an interrupted eval (finished windows are kept)
    python -m fraud.eval_ieee_full --report <eval id>   # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_dev_<setup>_dev<k>.jsonl, <eval id>_eval_<setup>_new<k>.jsonl.
Results: fraud/results/eval_ieee_full/<eval id>.json + .md.

PRE-REGISTRATION (Phase 12; committed before the new adapter was trained and before any model scored a new window).
Owner, 2026-11-03: yes to training on all the fit frauds. Phase 11's adapter used 3,021 of the 11,988, and "perfect" means lower cost.
- The adapter: one run of `fraud.finetune_ieee --train --full`. d1-Omni-2B from its frozen weights, the same prompt, LoRA and optimiser. Every fit
  fraud plus three legit rows for each, seed 0, one pass: 47,952 rows, 5,994 batches. Fit rows only. So more rows and a schedule sized to them are what
  changes. A safety stop at 10 h saves what it has and is reported. If the process dies, it is restarted from zero and that is reported.
- New test windows. Phase 11's 20 test windows have been read, and fraud/CLAUDE.md gives a change prompted by a test result new, untouched windows. Seed
  20261103: the test split cut into 20 equal blocks, one start per block, redrawn until the window shares no row with any Phase 11 window or earlier new window.
  They come from the same rows in time as Phase 11's, so they are not a later period. Each runs once per setup.
- Setups on the new windows:
  - ieee-ft-ieee-full, the new adapter. It reads Phase 11's 10 dev windows in both orders. Phase 9's debias rule picks the variant. Thresholds from dev.
  - ieee-ft-ieee-d1omni, Phase 11's adapter, with the option order and thresholds it was given in eval 20261102-183431. Nothing re-picked.
  - ieee-logreg-all, ieee-logreg-same, refitted on fit rows, with the thresholds of that eval. ieee-approve-all.
- Costs: Phase 11's. Friction 10%, at least 1 USD; review 10 USD; at most 5% of a window reviewed; past the budget a review becomes an approve.
- Primary question: on the same 20 new windows, which of ieee-ft-ieee-full and ieee-ft-ieee-d1omni has the lower total cost? Distinguishable only if the
  95% bootstrap interval (windows resampled, seed 20261102) of the mean paired difference excludes 0.
- Second question, the one that decides what to use: is ieee-ft-ieee-full cheaper than ieee-logreg-all, by the same rule? Reported whatever the answer.
- Also reported, no bar: both adapters against logreg-same, logreg-all and approve-all; Phase 11's adapter is re-tested here on windows nobody has read.
  AUC, recall, precision, alert rate, latency, cost sensitivity at review 1, 5 and 20 USD with thresholds re-picked on dev from the same scores.
- Latency bar: p90 per decision <= 250 ms on test, per model (as Phase 11).
- Limit stated now: the two adapters' thresholds come from the same dev windows, but the new adapter's are new and Phase 11's were already known. Dev is not test
  for either, so this should not favour one. Whether C, D and V columns are known at decision time is still uncheckable.
- Stated before running (Inferred): the new adapter beats the old on AUC and probably on cost, but the difference may not be distinguishable on 20 windows.
  It probably still does not beat logistic regression on all columns: four times the rows does not put the V columns in the prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, ieee_data
from .eval_ft import BAR_P90_MS, release
from .runs import finished, read_run, stamp

RESULTS = Path(__file__).resolve().parent / "results" / "eval_ieee_full"
OLD_EVAL = "20261102-183431"
OLD_ADAPTER = "20261102-210541_lora-d1-ieee"
SEED = 20261103
FT_FULL = "ieee-ft-ieee-full"
MODEL_SETUPS = (FT_FULL, ev.FT_IEEE)
SETUPS = (FT_FULL, ev.FT_IEEE, ev.LOGREG_SAME, ev.LOGREG_ALL, ev.APPROVE_ALL)
SENSITIVITY = {k: v for k, v in ev.SENSITIVITY.items() if k.startswith("review")}


def new_windows(n: int, count: int | None = None, size: int | None = None, seed: int = SEED) -> list[ev.Win]:
    """count windows of contiguous test rows that share no row with Phase 11's test windows or each other."""
    count = ev.TEST_WINDOWS if count is None else count
    size = ev.WINDOW_SIZE if size is None else size
    taken = [int(w.rows[0]) for w in ev.make_windows(n, "test", ev.TEST_WINDOWS, size)]
    lo, hi = ieee_data.split_rows(n, "test")
    edges = np.linspace(lo, hi, count + 1).astype(np.int64)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        while True:
            start = int(rng.integers(edges[k], edges[k + 1] - size + 1))
            if all(abs(start - s) >= size for s in taken):
                break
        taken.append(start)
        out.append(ev.Win(f"new{k}", "test", np.arange(start, start + size), seed))
    return out


def dev_windows(n: int) -> list[ev.Win]:
    return ev.make_windows(n, "dev", ev.DEV_WINDOWS)


def old_settings(setup: str, n: int) -> tuple[str, tuple]:
    """(option order, thresholds) a setup was given in Phase 11's eval. Read from its traces, not picked again."""
    head = read_run(ev.trace_file(OLD_EVAL, "eval", setup, ev.make_windows(n, "test", ev.TEST_WINDOWS)[0].id))[0]
    th = head["thresholds"]
    variant = "both"
    if setup == ev.FT_IEEE:
        dev = [read_run(ev.trace_file(OLD_EVAL, "dev", setup, w.id)) for w in dev_windows(n)]
        variant = ev.variant_of(setup, dev)[0]
    return variant, (th["review"], th["decline"])


def run(eval_id: str, adapter: Path, log=print) -> dict:
    t = ieee_data.load()
    dev, test = dev_windows(len(t)), new_windows(len(t))

    for setup, features in ((ev.LOGREG_SAME, ev.features_same), (ev.LOGREG_ALL, ev.features_all)):
        if not all(finished(ev.trace_file(eval_id, "eval", setup, w.id)) for w in test):
            lr = ev.fit_logreg(t, features)
            outs = {w.id: [{"by": "logreg", "q": float(p), "ms": 0.0} for p in lr.predict(features(t, w.rows))] for w in test}
            del lr
            ev.run_setup(eval_id, "eval", setup, test, t, lambda w: outs[w.id], old_settings(setup, len(t))[1], {}, log)
    ev.run_setup(eval_id, "eval", ev.APPROVE_ALL, test, t, lambda w: [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows],
                 (None, None), {}, log)

    for setup, folder in ((FT_FULL, adapter), (ev.FT_IEEE, ev.MODELS / OLD_ADAPTER)):
        model = None

        def ensure():
            nonlocal model
            if model is None:
                model = ev.D1Fine(folder)
            return model

        if setup == FT_FULL:
            traces = ev.run_setup(eval_id, "dev", setup, dev, t, lambda w: [ensure().score(t, int(i), "both") for i in w.rows], None,
                                  lambda: ensure().info, log)
            variant = ev.variant_of(setup, traces)[0]
            v = ev.view(traces, variant)
            tr, td, _ = ev.pick_thresholds(v["scores"], v["truth"], v["amount"], ev.COSTS)
            thresholds = (tr, td)
        else:
            variant, thresholds = old_settings(setup, len(t))
        log(f"{setup}: variant {variant}, thresholds {thresholds}")
        ev.run_setup(eval_id, "eval", setup, test, t, lambda w: [ensure().score(t, int(i), variant) for i in w.rows], thresholds,
                     lambda: ensure().info, log)
        model = None
        release()
    return report(eval_id, log)


def verdict(costs_by: dict[str, np.ndarray]) -> dict:
    return {"full_vs_old": ev.compare(costs_by, FT_FULL, ev.FT_IEEE), "full_vs_logreg_all": ev.compare(costs_by, FT_FULL, ev.LOGREG_ALL),
            "reported": [ev.compare(costs_by, FT_FULL, ev.LOGREG_SAME), ev.compare(costs_by, FT_FULL, ev.APPROVE_ALL),
                         *[ev.compare(costs_by, ev.FT_IEEE, b) for b in (ev.LOGREG_SAME, ev.LOGREG_ALL, ev.APPROVE_ALL)]]}


def report(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    dev, test = dev_windows(len(t)), new_windows(len(t))
    test_tr = {s: [read_run(ev.trace_file(eval_id, "eval", s, w.id)) for w in test] for s in SETUPS}
    dev_tr = {s: [read_run(ev.trace_file(eval_id if s == FT_FULL else OLD_EVAL, "dev", s, w.id)) for w in dev]
              for s in SETUPS if s != ev.APPROVE_ALL}
    views = {s: ev.view(test_tr[s]) for s in SETUPS}
    win_cost = {s: np.array([r[-1]["summary"]["cost"] for r in test_tr[s]]) for s in SETUPS}
    rows = {s: ev.setup_row(test_tr[s], views[s], s != ev.APPROVE_ALL) for s in SETUPS}
    dev_info = {s: {"auc": ev.pooled_auc(ev.view(dev_tr[s]))} for s in dev_tr}
    for s in MODEL_SETUPS:
        dev_info[s]["variant"], dev_info[s]["debias_gain"], dev_info[s]["auc_orders"] = ev.variant_of(s, dev_tr[s])
    sens = ev.sensitivity(SETUPS, dev_tr, views, {s: dev_info[s]["variant"] for s in MODEL_SETUPS}, SENSITIVITY)
    y_all = np.concatenate(views[ev.FT_IEEE]["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_ieee_full.py", "dataset": "IEEE-CIS train files, time split",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "seed": SEED, "old_eval": OLD_EVAL,
           "backends": {s: test_tr[s][0][0]["backend"].get("adapter") for s in MODEL_SETUPS},
           "windows": {w.id: w.header()["steps"] for w in test}, "test_rows": int(len(y_all)), "test_fraud": int(y_all.sum()),
           "setups": rows, "dev": dev_info, "sensitivity": sens, "verdict": verdict(win_cost),
           "bar_latency": {s: rows[s]["latency_ms"]["p90"] <= BAR_P90_MS for s in MODEL_SETUPS}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def _th(x) -> str:
    return "never" if x is None else f"{x:.4g}"


def markdown(r: dict) -> str:
    v = r["verdict"]
    L = [f"# IEEE-CIS full-training eval {r['eval']}", "",
         f"Pre-registration: `fraud/eval_ieee_full.py`. {r['test_rows']:,} new test rows, {r['test_fraud']} fraud, natural rate, "
         f"{len(r['windows'])} windows. Costs in USD: {r['costs']}. All numbers Observed. Model scores are uncalibrated.", "",
         "| setup | mean cost | 95% CI | AUC | 95% CI | recall | precision | alert rate | reviews | declines | p90 ms | thresholds (review, decline) |",
         "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        a = f"{x['auc']:.4f}" if x["auc"] is not None else "n/a"
        ac = f"{x['auc_ci95'][0]:.3f}–{x['auc_ci95'][1]:.3f}" if x["auc_ci95"] else "n/a"
        lat = f"{x['latency_ms']['p90']:.0f}" if x["latency_ms"] else "n/a"
        th = x["thresholds"] or {}
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {a} | {ac} | {x['recall']:.3f} | "
                 f"{x['precision']:.3f} | {x['alert_rate']:.3f} | {x['reviews']} | {x['declines']} | {lat} | "
                 f"{_th(th.get('review'))}, {_th(th.get('decline'))} |")

    def line(p):
        return (f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer, {p['ties']} ties; sign test p "
                f"{p['sign_p']:.3g}; mean paired difference {p['mean_diff']:,.0f} USD per window, 95% CI {p['mean_diff_ci95'][0]:,.0f} to "
                f"{p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}")

    L += ["", "## Answers", "",
          f"- Full adapter vs Phase 11's adapter: {line(v['full_vs_old'])}.",
          f"- Full adapter vs logistic regression on all columns: {line(v['full_vs_logreg_all'])}.",
          *[f"- {line(p)}." for p in v["reported"]],
          "- Latency bar, p90 <= 250 ms: " + ", ".join(f"{s} {'PASS' if ok else 'FAIL'}" for s, ok in r["bar_latency"].items()) + ".",
          "", "## Dev", "", "| setup | dev AUC | orders |", "|---|---:|---|"]
    for s, d in r["dev"].items():
        orders = f"{', '.join(f'{k} {x:.4f}' for k, x in d['auc_orders'].items())} → {d['variant']}" if "auc_orders" in d else ""
        L.append(f"| {s} | {d['auc']:.4f} | {orders} |")
    L += ["", "## Cost sensitivity (no bar; thresholds re-picked on dev per row)", "",
          "| cost set | " + " | ".join(SETUPS) + " |", "|---|" + "---:|" * len(SETUPS)]
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
    ap.add_argument("--adapter", type=Path, help="the folder written by python -m fraud.finetune_ieee --train --full")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if args.windows:
        for w in new_windows(len(ieee_data.load())):
            log(f"{w.id}: rows {w.rows[0]}–{w.rows[-1]}")
    elif args.report:
        report(args.report, log)
    else:
        if args.adapter is None or not (args.adapter / "manifest.json").exists():
            ap.error("--adapter must be a folder with manifest.json and adapter/ (python -m fraud.finetune_ieee --train --full)")
        run(args.resume or stamp(), args.adapter.resolve(), log)


if __name__ == "__main__":
    main()
