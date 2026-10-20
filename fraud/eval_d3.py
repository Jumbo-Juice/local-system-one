"""The Decision 3.0 eval: d3-nano and d3-mini on dev, a gate, then the 20 original test windows.

    python -m fraud.eval_d3 --smoke d3-nano          # 5 dev rows: the endpoint, both orders, timing
    python -m fraud.eval_d3                          # dev runs, debias rule, gate, test (passers only)
    python -m fraud.eval_d3 --resume <eval id>       # finish an interrupted eval
    python -m fraud.eval_d3 --report <eval id>       # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_dev_<setup>_dev<k>.jsonl and fraud/runs/<eval id>_eval_<setup>_test<k>.jsonl,
setup = hybrid-d3nano / hybrid-d3mini (and logreg on dev). Results: fraud/results/eval_d3/<eval id>.json + .md.

PRE-REGISTRATION (written and committed with the PROGRESS pre-statement of 2026-10-21, before any
D3 run; fraud/ROADMAP.md Phase 6). Owner's choices from a grilling session on 2026-10-21.
Question: does an off-the-shelf decision model rank fraud better than the frozen 1.5B (dev AUC
0.9028), with no fine-tuning?
- Models: vllm-sr Decision 3.0 d3-nano (2.21B) and d3-mini (4.54B), GGUF Q8_0 from
  ggml-org/vllm-d3-{nano,mini}-GGUF, sha256 pinned in fraud/d3.py and checked at start. llama.cpp
  llama-server build 11748 (Vulkan, Arc 140V), /v1/systemone, -np 1, one request at a time.
- Setup hybrid-d3<size>: rules.TypeFilter, then d3.D3Decider reading prompt v4 unchanged as one
  "choice" question (state = CONTEXT + describe(), instructions = QUESTION, criteria = OPTIONS).
  Decline score = P("yes: decline it").
- Dev: dev0–dev9, both option orders per transaction (one request, two questions), argmax of the
  average as the traced action. Scores of each order are kept in the trace.
- Debias rule (per model): AUC of the order-averaged decline score over the dev model rows,
  minus the larger of the two single-order AUCs. If >= DEBIAS_MIN_GAIN (0.005) the model keeps
  both orders; otherwise it uses its better single order. Argmax flip rate between orders is
  reported, not used. A timing pass on dev0 with one order is reported (latency is not a bar).
- Gate (per model): dev AUC of the chosen variant >= logreg's dev AUC on the same model rows
  (logreg fitted and thresholded exactly as in fraud/eval.py; 0.9916 in Phase 5). AUC: ties count
  half (fraud.eval_ft.auc). A model that fails stops at dev.
- Thresholds (as fraud/eval_ft.py): review / decline on the chosen variant's decline score, from
  GRID × GRID minimising total dev cost; fallback = the larger of its approve / decline scores.
- Test (models that pass only): test100–test119 (fraud.eval.TEST_SEEDS), once each, frozen
  variant and thresholds.
- Bar (pass/fail): pooled AUC of the decline score over the test model rows > hybrid-1.5b's pooled
  AUC on the same rows, recomputed from the traces of eval 20261003-052158.
- Reported, no bar: logreg's pooled AUC on the same rows (its traces of 20261003-052158); cost per
  window, recall, alert rate against rules, logreg and hybrid-1.5b of that eval; latency p50/p90;
  prompt tokens. Model scores are uncalibrated.
- Stated before running (Inferred): the gate is at logreg level, so both models may stop at dev.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, d3, rules, windows
from .eval_ft import GRID, auc, dev_auc, dev_view, pick_thresholds, run_traces, trace_file
from .runs import RUNS, finished, read_run, stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_d3"
OLD_EVAL = "20261003-052158"
DEV_SEEDS = tuple(range(0, 10))        # = fraud.eval.DEV_SEEDS
TEST_SEEDS = tuple(range(100, 120))    # = fraud.eval.TEST_SEEDS
MODELS = ("d3-nano", "d3-mini")
DEBIAS_MIN_GAIN = 0.005


def setup_of(model: str) -> str:
    return "hybrid-" + model.replace("-", "")


# ---------------------------------------------------------------- dev analysis

def variant_view(traces: list[list[dict]], variant: str) -> dict:
    """dev_view with the decline score (and fallback) of one variant: both / fwd / rev."""
    v = dev_view(traces)
    if variant == "both":
        return v
    k = ("fwd", "rev").index(variant)
    v["scores"], v["fallbacks"] = [], []
    for recs in traces:
        tx = [r for r in recs if r.get("type") == "tx"]
        v["scores"].append(np.array([r["orders"][k]["decline"] if r["by"] == "model" else 0.0 for r in tx]))
        v["fallbacks"].append([("approve" if r["orders"][k]["approve"] >= r["orders"][k]["decline"] else "decline")
                               if r["by"] == "model" else r["fallback"] for r in tx])
    return v


def flip_rate(traces: list[list[dict]]) -> float:
    rows = [r for recs in traces for r in recs if r.get("type") == "tx" and r["by"] == "model"]
    argmax = lambda o: max(o, key=o.get)
    return float(np.mean([argmax(r["orders"][0]) != argmax(r["orders"][1]) for r in rows])) if rows else 0.0


def choose_variant(aucs: dict) -> tuple[str, float]:
    """The debias rule: both orders only if they beat the better single order by DEBIAS_MIN_GAIN."""
    single = max(("fwd", "rev"), key=lambda k: aucs[k])  # fwd on a tie
    gain = aucs["both"] - aucs[single]
    return ("both" if gain >= DEBIAS_MIN_GAIN else single), gain


def logreg_scores_view(traces: list[list[dict]]) -> dict:
    v = dev_view(traces)
    v["scores"] = [np.array([r["scores"]["fraud"] if r["by"] == "logreg" else 0.0
                             for r in recs if r.get("type") == "tx"]) for recs in traces]
    v["filt"] = [np.array([r["by"] == "filter" for r in recs if r.get("type") == "tx"]) for recs in traces]
    return v


# ---------------------------------------------------------------- test analysis

def pooled_auc(traces: list[list[dict]], by: str, key: str, rows: set | None = None) -> tuple[float, int, int]:
    q, y = [], []
    for recs in traces:
        for r in recs:
            if r.get("type") == "tx" and r["by"] == by and (rows is None or (r["window"], r["row"]) in rows):
                q.append(r["scores"][key])
                y.append(r["truth"])
    q, y = np.array(q), np.array(y)
    return auc(q, y), len(y), int(y.sum())


def tagged(eval_id: str, stage: str, setup: str, seeds, prefix: str) -> list[list[dict]]:
    out = []
    for k in seeds:
        recs = read_run(trace_file(eval_id, stage, setup, f"{prefix}{k}"))
        for r in recs:
            r["window"] = f"{prefix}{k}"
        out.append(recs)
    return out


# ---------------------------------------------------------------- the run

def make_server(model: str, log) -> tuple[d3.D3Server, dict]:
    server = d3.D3Server(model)
    log(f"{model}: {server.info()['server']}; log {server.log_path}")
    filler = {"state": "TRANSFER of 1,000.00, hour 3:00.", "questions": {"q": {
        "type": "choice", "instructions": "Warm-up.", "criteria": {"a": None, "b": None}}}}
    for _ in range(3):  # Vulkan pipelines compile on the first passes; keep them out of the timing
        server.post(filler)
    return server, server.info()


def run(eval_id: str, models: list[str], log=print) -> dict:
    from . import eval as ev
    from .capture import load_all

    t, s = load_all()
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    lr = ev.logreg_decider(t, s)
    lr_traces = run_traces(eval_id, "dev", "logreg", dev, lambda w: [rules.TypeFilter(), lr], t, s, {}, log)
    lr_auc = dev_auc(logreg_scores_view(lr_traces))
    log(f"dev AUC logreg {lr_auc[0]:.4f} ({lr_auc[1]} rows, {lr_auc[2]} fraud)")

    dev_info = {"logreg": {"auc": lr_auc[0], "rows": lr_auc[1], "fraud": lr_auc[2]}}
    for model in models:
        setup, server, info = setup_of(model), None, {}
        needs = [trace_file(eval_id, "dev", setup, w.id) for w in dev] + [trace_file(eval_id, "dev", setup + "-fwd", "dev0")]
        if not all(finished(p) for p in needs):
            server, info = make_server(model, log)
        post = server.post if server else None
        traces = run_traces(eval_id, "dev", setup, dev, lambda w: [rules.TypeFilter(), d3.D3Decider(post, "both")],
                            t, s, {**info, "orders": "both"}, log)
        run_traces(eval_id, "dev", setup + "-fwd", dev[:1], lambda w: [rules.TypeFilter(), d3.D3Decider(post, "fwd")],
                   t, s, {**info, "orders": "fwd", "timing_only": True}, log)
        aucs = {k: dev_auc(variant_view(traces, k))[0] for k in d3.ORDERS}
        variant, gain = choose_variant(aucs)
        chosen = aucs[variant]
        gate = chosen >= lr_auc[0]
        vv = variant_view(traces, variant)
        tr, td, grid = pick_thresholds(vv["scores"], vv["truth"], vv["amount"], vv["filt"], vv["fallbacks"])
        timing = read_run(trace_file(eval_id, "dev", setup + "-fwd", "dev0"))[-1]["summary"]["latency_ms"]
        dev_info[model] = {"auc": aucs, "rows": dev_auc(vv)[1], "fraud": dev_auc(vv)[2],
                           "debias_gain": gain, "variant": variant, "flip_rate": flip_rate(traces),
                           "gate": {"rule": "AUC(chosen variant) >= AUC(logreg) on dev0-dev9 model rows", "pass": gate},
                           "thresholds": {"review": tr, "decline": td}, "grid_best": sorted(grid, key=lambda g: g[2])[:5],
                           "latency_both": [r for r in traces[0] if r.get("type") == "end"][0]["summary"]["latency_ms"],
                           "latency_single_dev0": timing}
        log(f"{model}: dev AUC both {aucs['both']:.4f} fwd {aucs['fwd']:.4f} rev {aucs['rev']:.4f} -> {variant} "
            f"(gain {gain:+.4f}); gate {'PASS' if gate else 'FAIL'} ({chosen:.4f} vs logreg {lr_auc[0]:.4f}); "
            f"thresholds review >= {tr}, decline >= {td}")
        if gate:
            test = [windows.make(t, "test", k, wid=f"test{k}") for k in TEST_SEEDS]
            if server is None and not all(finished(trace_file(eval_id, "eval", setup, w.id)) for w in test):
                server, info = make_server(model, log)
            post = server.post if server else None
            run_traces(eval_id, "eval", setup, test,
                       lambda w: [rules.TypeFilter(), d3.D3Decider(post, variant, (tr, td))], t, s,
                       {**info, "orders": variant}, log, {"thresholds": {"review": tr, "decline": td}})
        if server:
            server.close()
    return report(eval_id, dev_info, log)


# ---------------------------------------------------------------- results

def report(eval_id: str, dev_info: dict, log=print) -> dict:
    out = {"eval": eval_id, "script": "fraud/eval_d3.py", "old_eval": OLD_EVAL, "dev": dev_info,
           "debias_min_gain": DEBIAS_MIN_GAIN, "grid": list(GRID), "test": {}}
    old = {k: tagged(OLD_EVAL, "eval", k, TEST_SEEDS, "test") for k in ("hybrid-1.5b", "logreg", "rules")}
    for model in MODELS:
        setup = setup_of(model)
        if not all(finished(trace_file(eval_id, "eval", setup, f"test{k}")) for k in TEST_SEEDS):
            continue
        mine = tagged(eval_id, "eval", setup, TEST_SEEDS, "test")
        rows = {(r["window"], r["row"]) for recs in mine for r in recs if r.get("type") == "tx" and r["by"] == "model"}
        a = pooled_auc(mine, "model", "decline")
        b = pooled_auc(old["hybrid-1.5b"], "model", "decline", rows)
        lg = pooled_auc(old["logreg"], "logreg", "fraud", rows)
        per = []
        for k, recs in zip(TEST_SEEDS, mine):
            row = {"window": f"test{k}", setup: recs[-1]["summary"]["cost"]}
            for o, otr in old.items():
                row[o] = otr[TEST_SEEDS.index(k)][-1]["summary"]["cost"]
            row["recall"] = recs[-1]["summary"]["recall"]
            row["p90_ms"] = recs[-1]["summary"]["latency_ms"]["p90"]
            per.append(row)
        ms = [r["ms"] for recs in mine for r in recs if r.get("type") == "tx" and r["by"] == "model"]
        out["test"][model] = {
            "auc": {"value": a[0], "rows": a[1], "fraud": a[2]},
            "auc_hybrid_1.5b": {"value": b[0], "rows": b[1]}, "auc_logreg": {"value": lg[0], "rows": lg[1]},
            "bar": {"rule": "pooled AUC > hybrid-1.5b pooled AUC on the same rows", "pass": a[0] > b[0]},
            "windows": per, "mean_cost": {k: float(np.mean([r[k] for r in per])) for k in (setup, *old)},
            "latency_ms": {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90))}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS / f"{eval_id}.md").write_text(markdown(out), encoding="utf-8")
    log(f"results: {RESULTS / (eval_id + '.md')}")
    return out


def markdown(out: dict) -> str:
    lg = out["dev"]["logreg"]
    L = [f"# Decision 3.0 eval {out['eval']}", "",
         "Pre-registration: the `fraud/eval_d3.py` docstring. Model scores are uncalibrated.", "",
         "## Dev (dev0–dev9 model rows)", "",
         f"logreg AUC {lg['auc']:.4f} ({lg['rows']} rows, {lg['fraud']} fraud): the gate.", "",
         "| model | AUC both | AUC fwd | AUC rev | debias gain | variant | flip rate | gate | thresholds | dev0 p90 both / one order (ms) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        if m not in out["dev"]:
            continue
        d = out["dev"][m]
        L.append(f"| {m} | {d['auc']['both']:.4f} | {d['auc']['fwd']:.4f} | {d['auc']['rev']:.4f} | "
                 f"{d['debias_gain']:+.4f} | {d['variant']} | {d['flip_rate']:.3f} | "
                 f"{'PASS' if d['gate']['pass'] else 'FAIL'} | {d['thresholds']['review']} / {d['thresholds']['decline']} | "
                 f"{d['latency_both']['p90']:.0f} / {d['latency_single_dev0']['p90']:.0f} |")
    L += ["", "## Test (test100–test119)", ""]
    if not out["test"]:
        L.append("No model passed the dev gate; the test windows were not run.")
    for m, r in out["test"].items():
        L += [f"### {m}", "",
              f"- Pooled AUC {r['auc']['value']:.4f} vs hybrid-1.5b {r['auc_hybrid_1.5b']['value']:.4f} "
              f"({r['auc']['rows']} rows): bar **{'PASS' if r['bar']['pass'] else 'FAIL'}**. "
              f"logreg {r['auc_logreg']['value']:.4f} (reported).",
              f"- Mean cost per window: " + ", ".join(f"{k} {v:,.0f}" for k, v in r["mean_cost"].items()),
              f"- Latency p50 {r['latency_ms']['p50']:.0f} ms, p90 {r['latency_ms']['p90']:.0f} ms", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- smoke

def smoke(model: str, n: int = 5) -> None:
    """A few dev0 model rows: both orders, each order alone (must match), timing. Not a result."""
    from .capture import load_all

    t, s = load_all()
    w = windows.make(t, "dev", 0)
    rows = [int(i) for i in w.rows if t.type[i] not in rules.FRAUD_FREE][:n]
    server, _ = make_server(model, print)
    try:
        for i in rows:
            t0 = time.perf_counter()
            both = server.post(d3.request(t, s, i, "both"))
            ms = (time.perf_counter() - t0) * 1000
            fwd = server.post(d3.request(t, s, i, "fwd"))["answers"]["fwd"]["probabilities"]
            print(f"row {i} fraud={int(t.is_fraud[i])} {ms:6.0f} ms  tokens {both['usage']['input_tokens']}")
            for q in ("fwd", "rev"):
                print(f"  {q}: " + ", ".join(f"{k[:6]} {v:.4f}" for k, v in both["answers"][q]["probabilities"].items()))
            same = all(abs(fwd[k] - both["answers"]["fwd"]["probabilities"][k]) < 1e-4 for k in fwd)
            print(f"  fwd alone matches fwd in the pair: {same}")
    finally:
        server.close()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--resume", help="eval id to finish")
    ap.add_argument("--report", help="eval id: rebuild the results files from its traces and saved dev info")
    ap.add_argument("--smoke", choices=MODELS, help="a few dev rows through the endpoint (no files written)")
    args = ap.parse_args()
    if args.smoke:
        smoke(args.smoke)
    elif args.report:
        prev = json.loads((RESULTS / f"{args.report}.json").read_text(encoding="utf-8"))
        report(args.report, prev["dev"])
    else:
        run(args.resume or stamp(), args.models)


if __name__ == "__main__":
    main()
