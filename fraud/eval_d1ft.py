"""The pre-registered eval of the fine-tuned d1-Omni-2B: a reproduction check, dev, a gate, then the 20
Phase 5 test windows → fraud/results/eval_d1ft/.

    python -m fraud.eval_d1ft --repro                               # frozen d1-Omni in torch vs the Phase 8 traces
    python -m fraud.eval_d1ft --adapter fraud/models/<id>_lora-d1   # dev, debias rule, gate, test windows
    python -m fraud.eval_d1ft --adapter <dir> --resume <eval id>    # finish an interrupted eval
    python -m fraud.eval_d1ft --report <eval id>                    # rebuild the results files

Traces: fraud/runs/<eval id>_repro_hybrid-d1torch_dev<k>.jsonl, <eval id>_dev_{hybrid-d1ft,logreg}_dev<k>.jsonl,
<eval id>_eval_hybrid-d1ft_test<k>.jsonl. Results: fraud/results/eval_d1ft/<eval id>.json + .md
(<eval id>_repro.json + .md for the reproduction check).

PRE-REGISTRATION (written and committed before the adapter was trained; fraud/ROADMAP.md Phase 9).
Owner's request of 2026-11-02: "train d1 omni for an hour". The rest follows Phase 7, the closest
earlier fine-tune (Implementation choice).
Question: fine-tuned for one hour, does d1-Omni-2B rank better than its frozen self (Phase 8 dev AUC
0.9847) and the frozen 1.5B (0.9028), and does it decide as cheaply as logistic regression and the
Phase 5 fine-tune?
- Model: LiquidAI/d1-Omni-2B @ 431f1d667d (BF16 text layers; sha256 pinned in fraud/d1torch.py), run
  by fraud/d1torch.py, our own torch code, on the Arc 140V (XPU). The repo's remote code is not run.
- Reproduction check (before any training): the frozen model through fraud/d1torch.py, each option
  order its own lone prompt, as llama-server reads them (corrected before the check ran: PROGRESS
  2026-11-02), on every model row of dev0–dev9 (2,194 rows), against the Phase 8 llama-server traces
  (eval 20261030-122539, GGUF Q8_0). It passes if (R1) the prompt token count equals the trace's on
  every row, (R2) per order, |AUC − Phase 8 AUC| <= 0.01 (0.9830 fwd, 0.9847 rev), and (R3) per order,
  the median |decline score − Phase 8 decline score| <= 0.02. If it fails, nothing is trained until
  the owner has seen why.
- Adapter: the one training run of `python -m fraud.finetune_d1 --plan <smoke dir>` with the
  configuration in its docstring (Phase 5's rows, row hash 983ed4fa…; LoRA r16 on every layer; N
  batches from the hour; hard stop inside one hour). Its sha256 and N are recorded in the results.
- Setup hybrid-d1ft: rules.TypeFilter, then d3.D3Decider on the merged model, each order read as its
  own lone prompt, the function trained and the one Phase 8 read. Decline score = the softmax share
  of "yes: decline it" at calibration.json's temperature, which the adapter was trained under.
- Dev: dev0–dev9, both option orders per transaction; argmax of the average is the traced action.
  Debias rule (Phase 6, fraud.eval_d3.choose_variant): keep both orders only if their AUC beats the
  better single order by >= 0.005, else that order.
- Reported: the chosen variant's dev AUC vs frozen d1-Omni 0.9847 (Phase 8) and frozen 1.5B 0.9028.
- Gate: dev AUC of the chosen variant >= logreg's dev AUC on the same model rows (0.9916 in Phases
  5–8). If it fails, the eval stops, the result is reported, and the owner decides.
- Thresholds (as fraud/eval_ft.py): review / decline on the chosen variant's decline score, from
  GRID × GRID minimising total dev cost; fallback = the larger of its approve / decline scores.
- Test (if the gate passes): the 20 Phase 5 windows (fraud.eval_ft.new_windows: test200, …), once
  each, frozen variant and thresholds. Nothing is tuned on them.
- Bar A: hybrid-d1ft cost <= logreg cost in >= 10 of the 20 windows.
- Bar B: p90 of hybrid-d1ft's per-decision latency over its model decisions <= 250 ms.
- Bar C: hybrid-d1ft cost <= hybrid-ft cost in >= 10 of the 20 windows.
  A and C use the committed per-window costs of eval 20261005-022333, as in Phase 7.
- Reported, no bar: cost, recall, precision, alert rate per setup; paired window counts with
  two-sided sign tests vs logreg, rules, hybrid-ft, hybrid-1.5b and random; pooled test AUC vs
  hybrid-ft on the same rows; dev AUCs per order; thresholds; training N and time; ECE (uncalibrated
  scores).
- Stated before running (Inferred): Phase 7's d3-nano passed the gate with 90% of an epoch in about
  100 min. One hour here is likely near half an epoch, so it may learn less. Bar B is at risk: this
  is unoptimized torch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import d3, rules, windows
from .eval_d3 import choose_variant, flip_rate, logreg_scores_view, variant_view
from .eval_d3ft import REPRO_AUC_TOL, REPRO_MEDIAN_TOL, repro_compare, report as base_report, warmed
from .eval_ft import dev_auc, new_windows, pick_thresholds, run_traces, trace_file
from .runs import finished, read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_d1ft"
PHASE8_EVAL = "20261030-122539"
PHASE8_SETUP = "hybrid-d1omni2b"
FROZEN_AUC, FROZEN_15B_AUC = 0.9847, 0.9028
DEV_SEEDS = tuple(range(0, 10))  # = fraud.eval.DEV_SEEDS
SETUP = "hybrid-d1ft"
REPRO_SETUP = "hybrid-d1torch"
TITLE = "fine-tuned d1-Omni-2B"


def report(eval_id, adapter, dev_info=None, log=print):
    return base_report(eval_id, adapter, dev_info, log, setup=SETUP, results=RESULTS,
                       script="fraud/eval_d1ft.py", title=TITLE)


def repro(eval_id: str, log=print) -> dict:
    from .capture import load_all
    from .d1torch import D1Torch

    t, s = load_all()
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    model = warmed(D1Torch.load())
    ours = run_traces(eval_id, "repro", REPRO_SETUP, dev,
                      lambda w: [rules.TypeFilter(), d3.D3Decider(model.post, "both")],
                      t, s, {**model.info(), "orders": "both"}, log)
    ref = [read_run(trace_file(PHASE8_EVAL, "dev", PHASE8_SETUP, w.id)) for w in dev]
    out = {"eval": eval_id, "script": "fraud/eval_d1ft.py", "stage": "repro", "phase8_eval": PHASE8_EVAL,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **repro_compare(ours, ref)}
    lat = [r["ms"] for recs in ours for r in recs if r.get("type") == "tx" and r["by"] == "model"]
    out["latency_ms_both"] = {"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}_repro.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = repro_markdown(out)
    (RESULTS / f"{eval_id}_repro.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def repro_markdown(r: dict) -> str:
    L = [f"# d1-Omni-2B reproduction check {r['eval']}", "",
         f"Frozen d1-Omni-2B through `fraud/d1torch.py` (BF16, XPU) vs the llama-server Q8_0 traces of eval "
         f"{r['phase8_eval']}, dev0–dev9 model rows ({r['rows']}). Pre-registration: `fraud/eval_d1ft.py`. "
         "All numbers are Observed.", ""]
    if not r["same_rows"]:
        return "\n".join(L + ["**FAIL**: the model rows differ from the Phase 8 traces."]) + "\n"
    L += [f"- R1 prompt tokens equal on {r['tokens_equal']} of {r['rows']} rows → **{'PASS' if r['R1'] else 'FAIL'}**",
          "", "| order | AUC ours | AUC Phase 8 | median abs diff | p90 abs diff | max abs diff |",
          "|---|---:|---:|---:|---:|---:|"]
    for o in ("fwd", "rev"):
        v = r[o]
        L.append(f"| {o} | {v['auc']:.4f} | {v['auc_phase6']:.4f} | {v['median_abs_diff']:.4f} | "
                 f"{v['p90_abs_diff']:.4f} | {v['max_abs_diff']:.4f} |")
    L += ["", f"- R2 |AUC diff| <= {REPRO_AUC_TOL} per order → **{'PASS' if r['R2'] else 'FAIL'}**",
          f"- R3 median |decline score diff| <= {REPRO_MEDIAN_TOL} per order → **{'PASS' if r['R3'] else 'FAIL'}**",
          f"- Latency (both orders, one batch): p50 {r['latency_ms_both']['p50']:.0f} ms, "
          f"p90 {r['latency_ms_both']['p90']:.0f} ms", "",
          f"**Reproduction check: {'PASS' if r['pass'] else 'FAIL'}**"]
    return "\n".join(L) + "\n"


def run(eval_id: str, adapter: Path, log=print) -> dict:
    from . import eval as ev
    from .capture import load_all
    from .d1torch import D1Torch

    t, s = load_all()
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    test = new_windows(t)
    man = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
    lr = ev.logreg_decider(t, s)
    lr_traces = run_traces(eval_id, "dev", "logreg", dev, lambda w: [rules.TypeFilter(), lr], t, s, {}, log)
    lr_auc = dev_auc(logreg_scores_view(lr_traces))
    log(f"dev AUC logreg {lr_auc[0]:.4f} ({lr_auc[1]} rows, {lr_auc[2]} fraud)")

    model, info = None, {}
    paths = [trace_file(eval_id, "dev", SETUP, w.id) for w in dev] + [trace_file(eval_id, "eval", SETUP, w.id) for w in test]
    if not all(finished(p) for p in paths):
        model = warmed(D1Torch.load().merge_adapter(adapter / "adapter"))
        info = {**model.info(), "adapter": {"dir": adapter.name, "sha256": man["adapter_sha256"],
                                            "rows_sha256": man["rows_sha256"], "steps": man["steps_done"]}}
    post = model.post if model else None
    traces = run_traces(eval_id, "dev", SETUP, dev, lambda w: [rules.TypeFilter(), d3.D3Decider(post, "both")],
                        t, s, {**info, "orders": "both"}, log)
    aucs = {k: dev_auc(variant_view(traces, k))[0] for k in d3.ORDERS}
    variant, gain = choose_variant(aucs)
    gate = aucs[variant] >= lr_auc[0]
    vv = variant_view(traces, variant)
    tr, td, grid = pick_thresholds(vv["scores"], vv["truth"], vv["amount"], vv["filt"], vv["fallbacks"])
    log(f"dev AUC both {aucs['both']:.4f} fwd {aucs['fwd']:.4f} rev {aucs['rev']:.4f} -> {variant} (gain {gain:+.4f}); "
        f"frozen d1-Omni {FROZEN_AUC}, frozen 1.5B {FROZEN_15B_AUC}; gate {'PASS' if gate else 'FAIL'} "
        f"({aucs[variant]:.4f} vs logreg {lr_auc[0]:.4f}); thresholds review >= {tr}, decline >= {td}")
    dev_info = {"logreg": {"auc": lr_auc[0], "rows": lr_auc[1], "fraud": lr_auc[2]},
                "auc": aucs, "rows": dev_auc(vv)[1], "fraud": dev_auc(vv)[2], "debias_gain": gain, "variant": variant,
                "flip_rate": flip_rate(traces), "frozen_d1_auc": FROZEN_AUC, "frozen_15b_auc": FROZEN_15B_AUC,
                "gate": {"rule": "AUC(chosen variant) >= AUC(logreg) on dev0-dev9 model rows", "pass": gate},
                "thresholds": {"review": tr, "decline": td}, "grid_best": sorted(grid, key=lambda g: g[2])[:5]}
    if gate:
        run_traces(eval_id, "eval", SETUP, test, lambda w: [rules.TypeFilter(), d3.D3Decider(post, variant, (tr, td))],
                   t, s, {**info, "orders": variant}, log, {"thresholds": {"review": tr, "decline": td}})
    return report(eval_id, adapter, dev_info, log)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repro", action="store_true", help="the reproduction check (frozen model, dev0-dev9)")
    ap.add_argument("--adapter", type=Path, help="the folder written by python -m fraud.finetune_d1 --plan")
    ap.add_argument("--resume", metavar="EVAL_ID", help="continue this eval (or check): skip its finished traces")
    ap.add_argument("--report", metavar="EVAL_ID", help="only rebuild the results files from this eval's traces")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if args.repro:
        repro(args.resume or stamp(), log)
        return
    if args.report:
        report(args.report, args.adapter, log=log)
        return
    if args.adapter is None or not (args.adapter / "manifest.json").exists():
        ap.error("--adapter must be a folder with manifest.json and adapter/ (python -m fraud.finetune_d1 --plan)")
    run(args.resume or stamp(), args.adapter.resolve(), log)


if __name__ == "__main__":
    main()
