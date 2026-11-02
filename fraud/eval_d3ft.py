"""The pre-registered eval of the fine-tuned d3-nano (nano-ft): a reproduction check, dev, a gate, then
the 20 Phase 5 test windows → fraud/results/eval_d3ft/.

    python -m fraud.eval_d3ft --repro                          # frozen nano in torch vs the Phase 6 traces
    python -m fraud.eval_d3ft --adapter fraud/models/<id>_lora-d3   # dev, debias rule, gate, test windows
    python -m fraud.eval_d3ft --adapter <dir> --resume <eval id>    # finish an interrupted eval
    python -m fraud.eval_d3ft --report <eval id>                    # rebuild the results files

Traces: fraud/runs/<eval id>_repro_hybrid-d3torch_dev<k>.jsonl, <eval id>_dev_{hybrid-d3ft,logreg}_dev<k>.jsonl,
<eval id>_eval_hybrid-d3ft_test<k>.jsonl. Results: fraud/results/eval_d3ft/<eval id>.json + .md
(<eval id>_repro.json + .md for the reproduction check).

PRE-REGISTRATION (written and committed before the adapter was trained; fraud/ROADMAP.md Phase 7).
Question: fine-tuned like the Phase 5 Qwen2.5-1.5B, does d3-nano decide as cheaply as logistic
regression and as the Phase 5 fine-tune (hybrid-ft)? Owner's choices from a grilling session,
2026-10-21.
- Model: vllm-sr/d3-nano @ 462efc3662 (BF16 backbone, FP32 readout; sha256 pinned in
  fraud/d3torch.py), run by fraud/d3torch.py in our own torch code on the Arc 140V (XPU). The
  Gated DeltaNet layers use transformers' PyTorch reference: flash-linear-attention 0.5.2 was
  installed and removed because its Triton kernels do not run here (Observed 2026-10-21: no C
  compiler for triton-xpu, then "PyTorch was compiled without CUDA support").
- Reproduction check (before any training): the frozen model through fraud/d3torch.py, both
  option orders, on every model row of dev0–dev9 (2,194 rows), against the Phase 6 llama-server
  traces (eval 20261021-020632, GGUF Q8_0). It passes if (R1) the prompt token count (both orders)
  equals the trace's on every row, (R2) per order, |AUC − Phase 6 AUC| <= 0.01 (0.6886 fwd,
  0.8748 rev), and (R3) per order, the median |decline score − Phase 6 decline score| <= 0.02.
  If it fails, nothing is trained until the owner has seen why.
- Adapter: the one training run of `python -m fraud.finetune_d3 --plan <smoke dir>` with the
  configuration in its docstring (Phase 5's rows, row hash 983ed4fa…; LoRA r16 on every layer;
  N batches from the cap rule; hard stop at 2 h). Its sha256 and N are recorded in the results.
  If the rule gives N < 755, the run does not happen and the owner decides.
- Setup hybrid-d3ft: rules.TypeFilter, then d3.D3Decider on the merged model (prompt v4 as one
  "d3" choice question). Decline score = the readout's softmax share of "yes: decline it".
- Dev: dev0–dev9, both option orders per transaction (one forward pass of two prompts); argmax of
  the average is the traced action. Debias rule (Phase 6, fraud.eval_d3.choose_variant): keep
  both orders only if their AUC beats the better single order by >= 0.005, else that order.
- Gate: dev AUC of the chosen variant >= logreg's dev AUC on the same model rows (logreg fitted
  and run as in fraud/eval.py; 0.9916 in Phases 5 and 6). AUC: ties count half. If it fails, the
  eval stops, the result is reported, and the owner decides about any further configuration
  (dev only).
- Thresholds (as fraud/eval_ft.py): review / decline on the chosen variant's decline score, from
  GRID × GRID minimising total dev cost; fallback = the larger of its approve / decline scores.
- Test (if the gate passes): the 20 Phase 5 windows (fraud.eval_ft.new_windows: test200, …),
  once each, frozen variant and thresholds. Nothing is tuned on them.
- Bar A: hybrid-d3ft cost <= logreg cost in >= 10 of the 20 windows.
- Bar B: p90 of hybrid-d3ft's per-decision latency over its model decisions <= 250 ms.
- Bar C: hybrid-d3ft cost <= hybrid-ft cost in >= 10 of the 20 windows.
  A and C use the committed per-window costs of eval 20261005-022333 (fraud/results/eval/),
  where the other setups ran on the same windows (Implementation choice: they are not re-run).
- Reported, no bar: cost, recall, precision, alert rate per setup; paired window counts with
  two-sided sign tests (ties dropped) vs logreg, rules, hybrid-ft, hybrid-1.5b and random; pooled
  test AUC of hybrid-d3ft vs hybrid-ft on the same rows (Phase 5 traces, when on disk); dev AUCs
  per order; thresholds; training N, time and stopped_early; ECE (uncalibrated scores).
- Stated before running (Inferred): rules-only paid only review fees in most Phase 5 windows and
  hybrid-ft tied it in 11 of 20, so Bar C is decided mostly by ties; a fine-tune that learns the
  rule table passes A and C. Bar B is at risk: the DeltaNet layers run the PyTorch reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, d3, rules, windows
from .eval_d3 import choose_variant, flip_rate, logreg_scores_view, pooled_auc, variant_view
from .eval_ft import N_NEW, auc, dev_auc, new_windows, pick_thresholds, run_traces, trace_file
from .runs import finished, read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_d3ft"
PHASE5_EVAL = "20261005-022333"
PHASE6_EVAL = "20261021-020632"
DEV_SEEDS = tuple(range(0, 10))  # = fraud.eval.DEV_SEEDS
SETUP = "hybrid-d3ft"
REPRO_SETUP = "hybrid-d3torch"
REPRO_AUC_TOL, REPRO_MEDIAN_TOL = 0.01, 0.02
BAR_WINDOWS, BAR_P90_MS = 10, 250.0


# ---------------------------------------------------------------- the reproduction check

def repro_compare(ours: list[list[dict]], ref: list[list[dict]]) -> dict:
    """R1–R3 of the pre-registration: our frozen-model traces vs the Phase 6 traces, row by row."""
    mine = {(k, r["row"]): r for k, recs in enumerate(ours) for r in recs if r.get("type") == "tx" and r["by"] == "model"}
    theirs = {(k, r["row"]): r for k, recs in enumerate(ref) for r in recs if r.get("type") == "tx" and r["by"] == "model"}
    keys = sorted(theirs)
    out = {"rows": len(keys), "same_rows": sorted(mine) == keys}
    if not out["same_rows"]:
        out["pass"] = False
        return out
    y = np.array([theirs[k]["truth"] for k in keys])
    out["tokens_equal"] = int(sum(mine[k]["prompt_tokens"] == theirs[k]["prompt_tokens"] for k in keys))
    r1 = out["tokens_equal"] == len(keys)
    r2 = r3 = True
    for j, order in enumerate(("fwd", "rev")):
        a = np.array([mine[k]["orders"][j]["decline"] for k in keys])
        b = np.array([theirs[k]["orders"][j]["decline"] for k in keys])
        d = np.abs(a - b)
        out[order] = {"auc": auc(a, y), "auc_phase6": auc(b, y), "median_abs_diff": float(np.median(d)),
                      "p90_abs_diff": float(np.percentile(d, 90)), "max_abs_diff": float(d.max())}
        r2 &= abs(out[order]["auc"] - out[order]["auc_phase6"]) <= REPRO_AUC_TOL
        r3 &= out[order]["median_abs_diff"] <= REPRO_MEDIAN_TOL
    out.update({"R1": r1, "R2": r2, "R3": r3, "pass": bool(r1 and r2 and r3)})
    return out


def warmed(model) -> object:
    filler = {"state": "TRANSFER of 1,000.00, hour 3:00.", "questions": {"q": {
        "type": "choice", "instructions": "Warm-up.", "criteria": {"a": None, "b": None, "c": None}}}}
    for _ in range(3):  # first XPU passes compile kernels; keep them out of the timing
        model.post(filler)
    return model


def repro(eval_id: str, log=print) -> dict:
    from .capture import load_all
    from .d3torch import D3Torch

    t, s = load_all()
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    model = None
    if not all(finished(trace_file(eval_id, "repro", REPRO_SETUP, w.id)) for w in dev):
        model = warmed(D3Torch.load())
    post = model.post if model else None
    info = model.info() if model else {}
    ours = run_traces(eval_id, "repro", REPRO_SETUP, dev, lambda w: [rules.TypeFilter(), d3.D3Decider(post, "both")],
                      t, s, {**info, "orders": "both"}, log)
    ref = [read_run(trace_file(PHASE6_EVAL, "dev", "hybrid-d3nano", w.id)) for w in dev]
    out = {"eval": eval_id, "script": "fraud/eval_d3ft.py", "stage": "repro", "phase6_eval": PHASE6_EVAL,
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
    L = [f"# d3-nano reproduction check {r['eval']}", "",
         f"Frozen d3-nano through `fraud/d3torch.py` (BF16, XPU) vs the llama-server Q8_0 traces of eval "
         f"{r['phase6_eval']}, dev0–dev9 model rows ({r['rows']}). Pre-registration: `fraud/eval_d3ft.py`. "
         "All numbers are Observed.", ""]
    if not r["same_rows"]:
        return "\n".join(L + ["**FAIL**: the model rows differ from the Phase 6 traces."]) + "\n"
    L += [f"- R1 prompt tokens equal on {r['tokens_equal']} of {r['rows']} rows → **{'PASS' if r['R1'] else 'FAIL'}**",
          "", "| order | AUC ours | AUC Phase 6 | median abs diff | p90 abs diff | max abs diff |",
          "|---|---:|---:|---:|---:|---:|"]
    for o in ("fwd", "rev"):
        v = r[o]
        L.append(f"| {o} | {v['auc']:.4f} | {v['auc_phase6']:.4f} | {v['median_abs_diff']:.4f} | "
                 f"{v['p90_abs_diff']:.4f} | {v['max_abs_diff']:.4f} |")
    L += ["", f"- R2 |AUC diff| <= {REPRO_AUC_TOL} per order → **{'PASS' if r['R2'] else 'FAIL'}**",
          f"- R3 median |decline score diff| <= {REPRO_MEDIAN_TOL} per order → **{'PASS' if r['R3'] else 'FAIL'}**",
          f"- Latency (both orders, one pass): p50 {r['latency_ms_both']['p50']:.0f} ms, "
          f"p90 {r['latency_ms_both']['p90']:.0f} ms", "",
          f"**Reproduction check: {'PASS' if r['pass'] else 'FAIL'}**"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the eval

def bars(mine: dict, phase5: dict, p90: float | None, setup: str = SETUP) -> dict:
    """Bars A–C from per-window costs: ours, and Phase 5's committed {setup: {window: summary}}."""
    def not_worse(other):
        ws = [w for w in mine if w in phase5.get(other, {})]
        return sum(mine[w]["cost"] <= phase5[other][w]["cost"] for w in ws), len(ws)

    complete = len(mine) == N_NEW
    a, _ = not_worse("logreg")
    c, _ = not_worse("hybrid-ft")
    return {
        "barA": {"rule": f"{setup} cost <= logreg cost in >= {BAR_WINDOWS} of {N_NEW} windows",
                 "value": a if complete else None, "pass": complete and a >= BAR_WINDOWS},
        "barB": {"rule": f"{setup} p90 latency <= {BAR_P90_MS:.0f} ms", "value": p90,
                 "pass": p90 is not None and p90 <= BAR_P90_MS},
        "barC": {"rule": f"{setup} cost <= hybrid-ft cost in >= {BAR_WINDOWS} of {N_NEW} windows",
                 "value": c if complete else None, "pass": complete and c >= BAR_WINDOWS},
    }


def run(eval_id: str, adapter: Path, log=print) -> dict:
    from . import eval as ev
    from .capture import load_all
    from .d3torch import D3Torch

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
        model = warmed(D3Torch.load().merge_adapter(adapter / "adapter"))
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
        f"gate {'PASS' if gate else 'FAIL'} ({aucs[variant]:.4f} vs logreg {lr_auc[0]:.4f}); "
        f"thresholds review >= {tr}, decline >= {td}")
    dev_info = {"logreg": {"auc": lr_auc[0], "rows": lr_auc[1], "fraud": lr_auc[2]},
                "auc": aucs, "rows": dev_auc(vv)[1], "fraud": dev_auc(vv)[2], "debias_gain": gain, "variant": variant,
                "flip_rate": flip_rate(traces),
                "gate": {"rule": "AUC(chosen variant) >= AUC(logreg) on dev0-dev9 model rows", "pass": gate},
                "thresholds": {"review": tr, "decline": td}, "grid_best": sorted(grid, key=lambda g: g[2])[:5]}
    if gate:
        run_traces(eval_id, "eval", SETUP, test, lambda w: [rules.TypeFilter(), d3.D3Decider(post, variant, (tr, td))],
                   t, s, {**info, "orders": variant}, log, {"thresholds": {"review": tr, "decline": td}})
    return report(eval_id, adapter, dev_info, log)


def report(eval_id: str, adapter: Path | None, dev_info: dict | None = None, log=print, setup: str = SETUP,
           results: Path = RESULTS, script: str = "fraud/eval_d3ft.py", title: str = "fine-tuned d3-nano") -> dict:
    from . import eval as ev
    from .capture import load_all

    path = results / f"{eval_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    dev_info = dev_info or old.get("dev")
    training = old.get("training")
    if adapter is not None:
        man = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
        training = {k: man.get(k) for k in ("steps", "steps_done", "stopped_early", "train_s", "process_s",
                                            "step_s_mean", "peak_mem_gib", "loss_first100", "loss_last100",
                                            "adapter_sha256", "rows_sha256", "lora", "smoke")}
        training["adapter"] = adapter.name
    t, _ = load_all()
    test_ids = [w.id for w in new_windows(t)]
    phase5 = json.loads((ROOT / "results" / "eval" / f"{PHASE5_EVAL}.json").read_text(encoding="utf-8"))["windows"]
    mine, traces = {}, []
    for wid in test_ids:
        p = trace_file(eval_id, "eval", setup, wid)
        if finished(p):
            recs = read_run(p)
            for r in recs:
                r["window"] = wid
            mine[wid] = recs[-1]["summary"]
            traces.append(recs)
    ms = [r["ms"] for recs in traces for r in recs if r.get("type") == "tx" and r["by"] == "model"]
    p90 = float(np.percentile(ms, 90)) if ms else None
    per = {setup: mine, **phase5}

    def paired(a, b):
        ws = [w for w in per.get(a, {}) if w in per.get(b, {})]
        wn = sum(per[a][w]["cost"] < per[b][w]["cost"] for w in ws)
        ls = sum(per[a][w]["cost"] > per[b][w]["cost"] for w in ws)
        return {"windows": len(ws), "a_cheaper": wn, "b_cheaper": ls, "ties": len(ws) - wn - ls,
                "sign_test_p": ev.sign_test(wn, ls)}

    totals = {}
    for name, ws in per.items():
        vals = [ws[w] for w in test_ids if w in ws]
        if not vals:
            continue
        totals[name] = {"windows": len(vals), "cost_mean": float(np.mean([v["cost"] for v in vals])),
                         "cost_median": float(np.median([v["cost"] for v in vals])),
                         "recall_mean": float(np.mean([v["recall"] for v in vals])),
                         "precision_mean": float(np.mean([v["precision"] for v in vals])),
                         "alert_rate_mean": float(np.mean([v["alert_rate"] for v in vals]))}
        if "latency_ms" in vals[0]:
            totals[name]["latency_p90_mean"] = float(np.mean([v["latency_ms"]["p90"] for v in vals]))
    if ms:
        totals[setup]["latency_ms"] = {"p50": float(np.percentile(ms, 50)), "p90": p90, "decisions": len(ms)}
        totals[setup]["calibration"] = ev.model_calibration(traces)
    pooled = None
    ft_paths = [trace_file(PHASE5_EVAL, "eval", "hybrid-ft", w) for w in test_ids]
    if traces and all(finished(p) for p in ft_paths):
        ft = []
        for wid, p in zip(test_ids, ft_paths):
            recs = read_run(p)
            for r in recs:
                r["window"] = wid
            ft.append(recs)
        rows = {(r["window"], r["row"]) for recs in traces for r in recs if r.get("type") == "tx" and r["by"] == "model"}
        a, b = pooled_auc(traces, "model", "decline"), pooled_auc(ft, "model", "decline", rows)
        pooled = {setup: {"auc": a[0], "rows": a[1], "fraud": a[2]}, "hybrid-ft": {"auc": b[0], "rows": b[1]}}
    out = {
        "eval": eval_id, "script": script, "setup": setup, "title": title, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complete": len(mine) == N_NEW, "phase5_eval": PHASE5_EVAL, "costs": costs.PHASES_1_10.as_dict(),
        "test_windows": test_ids, "dev": dev_info, "training": training,
        "bars": bars(mine, phase5, p90, setup),
        "paired": {f"{setup} vs {o}": paired(setup, o) for o in ("logreg", "rules", "hybrid-ft", "hybrid-1.5b", "random")},
        "pooled_test_auc": pooled, "totals": totals, "windows": mine,
    }
    results.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (results / f"{eval_id}.md").write_text(markdown(out), encoding="utf-8")
    log(markdown(out))
    return out


def markdown(r: dict) -> str:
    setup = r.get("setup", SETUP)
    L = [f"# Fraud eval {r['eval']} ({r.get('title', 'fine-tuned d3-nano')})", "",
         f"Created {r['created']}. Complete: {r['complete']}. Pre-registration: the docstring of `{r.get('script', 'fraud/eval_d3ft.py')}`. "
         f"Other setups: the committed per-window results of eval {r['phase5_eval']}. All numbers are Observed; "
         "model scores are uncalibrated.", ""]
    tr = r.get("training")
    if tr:
        L += ["## Training", "",
              f"- Adapter `{tr['adapter']}` (sha256 `{str(tr['adapter_sha256'])[:8]}…`), rows `{str(tr['rows_sha256'])[:8]}…`",
              f"- Batches: {tr['steps_done']} of {tr['steps']} planned{' (stopped early)' if tr['stopped_early'] else ''}; "
              f"training {tr['train_s'] / 60:.1f} min, process {tr['process_s'] / 60:.1f} min, "
              f"{tr['step_s_mean']:.2f} s/batch, peak {tr['peak_mem_gib']:.2f} GiB",
              f"- Loss: first 100 batches {tr['loss_first100']:.4f}, last 100 {tr['loss_last100']:.4f}", ""]
    d = r.get("dev")
    if d:
        L += ["## Dev gate (dev0–dev9 model rows)", "",
              f"- AUC of the decline score: both {d['auc']['both']:.4f}, fwd {d['auc']['fwd']:.4f}, rev {d['auc']['rev']:.4f} "
              f"({d['rows']} rows, {d['fraud']} fraud); debias gain {d['debias_gain']:+.4f} → variant **{d['variant']}**; "
              f"argmax flip rate {d['flip_rate']:.3f}",
              f"- logreg AUC {d['logreg']['auc']:.4f}. Gate: {d['gate']['rule']} → **{'PASS' if d['gate']['pass'] else 'FAIL'}**",
              f"- Thresholds: review ≥ {d['thresholds']['review']}, decline ≥ {d['thresholds']['decline']}", ""]
    L += ["## Bars", ""]
    if not r["windows"]:
        L.append("The test windows were not run (the dev gate failed).")
    for k, b in r["bars"].items():
        val = "–" if b["value"] is None else (f"{b['value']:.1f} ms" if isinstance(b["value"], float) else b["value"])
        L.append(f"- {k}: {b['rule']}: **{val}** → **{'PASS' if b['pass'] else 'FAIL'}**")
    if r["windows"]:
        L += ["", "## Setups (over the 20 test windows)", "",
              "| setup | windows | mean cost | median cost | recall | precision | alert rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in r["totals"].items():
            L.append(f"| {k} | {v['windows']} | {v['cost_mean']:,.0f} | {v['cost_median']:,.0f} | {v['recall_mean']:.2f} | "
                     f"{v['precision_mean']:.2f} | {v['alert_rate_mean']:.3f} |")
        lat = r["totals"][setup].get("latency_ms")
        if lat:
            L += ["", f"Latency of {setup} over {lat['decisions']} model decisions: p50 {lat['p50']:.0f} ms, p90 {lat['p90']:.0f} ms."]
        L += ["", "## Paired window counts (cheaper = lower cost in that window)", "",
              "| pair | windows | first cheaper | second cheaper | ties | sign test p |", "|---|---:|---:|---:|---:|---:|"]
        for k, v in r["paired"].items():
            L.append(f"| {k} | {v['windows']} | {v['a_cheaper']} | {v['b_cheaper']} | {v['ties']} | {v['sign_test_p']:.3g} |")
        if r.get("pooled_test_auc"):
            p = r["pooled_test_auc"]
            L += ["", f"Pooled test AUC over {setup}'s model rows ({p[setup]['rows']}): {setup} {p[setup]['auc']:.4f}, "
                      f"hybrid-ft {p['hybrid-ft']['auc']:.4f} (same rows, reported)."]
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repro", action="store_true", help="the reproduction check (frozen model, dev0-dev9)")
    ap.add_argument("--adapter", type=Path, help="the folder written by python -m fraud.finetune_d3 --plan")
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
        ap.error("--adapter must be a folder with manifest.json and adapter/ (python -m fraud.finetune_d3 --plan)")
    run(args.resume or stamp(), args.adapter.resolve(), log)


if __name__ == "__main__":
    main()
