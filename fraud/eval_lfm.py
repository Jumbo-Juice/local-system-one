"""The LFM3 eval: LiquidAI d1-Omni-2B and LFM3-1.7B, frozen, on dev, a gate, then the 20 original test windows.

    python -m fraud.eval_lfm --smoke d1-omni-2b      # 5 dev rows: the endpoint / engine, both orders, timing
    python -m fraud.eval_lfm                         # dev runs, debias rule, gate, test (passers only)
    python -m fraud.eval_lfm --resume <eval id>      # finish an interrupted eval
    python -m fraud.eval_lfm --report <eval id>      # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_dev_<setup>_dev<k>.jsonl and fraud/runs/<eval id>_eval_<setup>_test<k>.jsonl,
setup = hybrid-d1omni2b / hybrid-lfm317b (and logreg on dev). Results: fraud/results/eval_lfm/<eval id>.json + .md.

PRE-REGISTRATION (written and committed with the PROGRESS pre-statement of 2026-10-30, before any
LFM run; fraud/ROADMAP.md Phase 8). Owner's request of 2026-10-30: "run the same test [as Phase 6].
don't finetune. tell me if they're better than base qwen 1.5b + filter."
Question: does either LFM3-family model, frozen, rank fraud better than the frozen Qwen2.5-1.5B behind
the same type filter?
- Models (both frozen, GGUF Q8_0, sha256 pinned here / in fraud/d3.py and checked at start):
  - d1-omni-2b: LiquidAI/d1-Omni-2B-GGUF d1-Omni-2B-Q8_0.gguf, a decision model (built on
    LFM3-Omni-2B) answering typed questions on /v1/systemone, as the D3 models in Phase 6. Text only.
  - lfm3-1.7b: LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf, a chat model. It has no /v1/systemone
    readout, so it runs through the engine as the 1.5B does (next-token scores of the option labels,
    prompt v4 via its own chat template, thinking off), on the engine's llama.cpp backend with the
    Hugging Face tokenizer of LiquidAI/LFM3-1.7B (transformers 5.17 has no lfm3 model class, and the
    repo's remote code is not run).
  - Runtime: llama.cpp llama-server, Vulkan; both models need b11960 or newer (Documented, their
    model cards). Pre-registered as a project-local b11960 zip; DEVIATION (owner's choice,
    2026-10-30, before any LFM run): the winget install upgraded from b11748 to b11977 instead,
    because running a downloaded binary was blocked. One request at a time.
- Setups hybrid-d1omni2b / hybrid-lfm317b: rules.TypeFilter, then the model on prompt v4 unchanged.
  d1-omni-2b: d3.D3Decider (one "choice" question per order, decline score = P("yes: decline it")).
  lfm3-1.7b: brain.ModelDecider with engine order_debias (both orders in one batch).
- Dev: dev0–dev9, both option orders per transaction, argmax of the average as the traced action;
  each order's scores are kept in the trace. A timing pass on dev0 with one (forward) order is
  reported (latency is not a bar).
- Debias rule (per model, as Phase 6): AUC of the order-averaged decline score over the dev model
  rows minus the better single-order AUC; keep both orders if >= 0.005, else the better single order.
- Headline comparison (the owner's question, reported as better / not better, dev only): the chosen
  variant's dev AUC vs the frozen 1.5B's dev AUC on the same model rows, recomputed from the
  hybrid-1.5b dev traces of eval 20261005-022333 (0.9028, both orders, as that setup runs).
  Also reported: AUC per type (TRANSFER, CASH_OUT), and dev cost with dev-picked thresholds for
  both the model and the 1.5B (in-sample, so optimistic for both).
- Gate (per model, as Phase 6): chosen variant's dev AUC >= logreg's dev AUC on the same rows
  (logreg fitted and thresholded as in fraud/eval.py; 0.9916 in Phases 5 and 6). Fail: stop at dev.
- Thresholds (as fraud/eval_ft.py): review / decline from GRID x GRID minimising total dev cost.
- Test (models that pass only): test100–test119, once each, frozen variant and thresholds.
  Bar: pooled AUC over the test model rows > hybrid-1.5b's pooled AUC on the same rows (traces of
  eval 20261003-052158). Reported, no bar: logreg's pooled AUC, cost per window vs rules, logreg,
  hybrid-1.5b; recall; latency p50/p90.
- Stated before running (Inferred): the gate is at logreg level, so both models may stop at dev, as
  both D3 models did. Model scores are uncalibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import brain, costs, d3, rules, windows
from .eval_d3 import (DEBIAS_MIN_GAIN, DEV_SEEDS, OLD_EVAL, TEST_SEEDS, choose_variant, flip_rate,
                      logreg_scores_view, order, pooled_auc, tagged, variant_view)
from .eval_ft import GRID, auc, dev_auc, pick_thresholds, release, run_traces, trace_file
from .runs import finished, read_run, stamp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results" / "eval_lfm"
REF_EVAL = "20261005-022333"   # Phase 5: the frozen 1.5B's dev traces (setup hybrid-1.5b)
LFM3 = {"gguf": "lfm/LFM3-1.7B-Q8_0.gguf", "sha256": "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
        "tokenizer": "LiquidAI/LFM3-1.7B"}
MODELS = ("d1-omni-2b", "lfm3-1.7b")
SETUPS = {"d1-omni-2b": "hybrid-d1omni2b", "lfm3-1.7b": "hybrid-lfm317b"}


def server_exe() -> str:
    """llama-server as Phase 6 finds it (LLAMA_SERVER, PATH, winget); b11960 or newer (owner: winget b11977)."""
    return d3.find_server()


# ---------------------------------------------------------------- the two model kinds

class Model:
    """One loaded model: ``decider(variant, thresholds)`` returns a fresh decider; ``close()`` frees it."""

    def __init__(self, name: str, log):
        self.name, self.engine, self.server = name, None, None
        if name == "d1-omni-2b":
            self.server = d3.D3Server(name, server=server_exe())
            log(f"{name}: {self.server.info()['server']}; log {self.server.log_path}")
            filler = {"state": "TRANSFER of 1,000.00, hour 3:00.", "questions": {"q": {
                "type": "choice", "instructions": "Warm-up.", "criteria": {"a": None, "b": None}}}}
            for _ in range(3):  # Vulkan pipelines compile on the first passes; keep them out of the timing
                self.server.post(filler)
            self.info = self.server.info()
        else:
            self.engine, self.info = lfm3_engine()
            log(f"{name}: {self.info['server_build']}; log {self.engine.backend.log_path}")

    def decider(self, variant: str, thresholds=None):
        if self.server:
            return d3.D3Decider(self.server.post, variant, thresholds)
        return VariantDecider(self.engine, variant, thresholds)

    def close(self) -> None:
        if self.server:
            self.server.close()
        if self.engine:
            self.engine.backend.close()
        self.engine = self.server = None
        release()


def lfm3_engine():
    from system_one import load_config, make_engine

    gguf = ROOT / "models" / LFM3["gguf"]
    if not gguf.exists():
        raise FileNotFoundError(f"{gguf} missing; download it (RUN-GUIDE.md)")
    if d3.sha256(gguf) != LFM3["sha256"]:
        raise RuntimeError(f"{gguf.name}: sha256 differs from the pinned one")
    cfg = load_config(REPO / "config" / "default.toml")
    cfg["backend"] = {"kind": "llamacpp", "model": str(gguf), "tokenizer": LFM3["tokenizer"], "server": server_exe(),
                      "ctx_size": 8192, "parallel": 2, "n_probs": 100}
    cfg["engine"]["order_debias"] = True
    engine = make_engine(cfg)
    info = engine.backend.info()
    info.update({"device": "vulkan", "device_name": "Intel Arc 140V (llama.cpp Vulkan)", "sha256": LFM3["sha256"]})
    brain.warm_up(engine)
    return engine, info


class VariantDecider(brain.ModelDecider):
    """brain.ModelDecider reading one option-order variant: both (the engine's average), fwd (the
    engine asks the forward order only) or rev (both asked, the reversed reading kept). With
    thresholds, the action comes from that variant's decline score (as brain.ThresholdDecider)."""

    def __init__(self, engine, variant: str = "both", thresholds=None):
        super().__init__(engine)
        self.variant, self.thresholds = variant, thresholds

    def decide(self, t, s, i):
        self.engine.order_debias = self.variant != "fwd"
        out = super().decide(t, s, i)
        if self.variant == "rev":
            p = order(out, 1)
            out["scores"], out["fallback"] = p, "approve" if p["approve"] >= p["decline"] else "decline"
            out["action"] = max(p, key=p.get)
        if self.thresholds:
            tr, td = self.thresholds
            q = out["scores"]["decline"]
            out["action"] = "decline" if q >= td else "review" if q >= tr else "approve"
            out["thresholds"] = {"review": tr, "decline": td}
        return out


# ---------------------------------------------------------------- dev analysis

def type_aucs(traces, view) -> dict:
    """AUC of a view's decline score per transaction type, over the model rows."""
    types = np.concatenate([[r["tx"]["type"] for r in recs if r.get("type") == "tx"] for recs in traces])
    m = ~np.concatenate(view["filt"])
    q, y = np.concatenate(view["scores"]), np.concatenate(view["truth"])
    return {k: auc(q[m & (types == k)], y[m & (types == k)]) for k in ("TRANSFER", "CASH_OUT")}


def dev_cost(view, tr, td) -> float:
    total = 0.0
    for q, y, a, f, fb in zip(view["scores"], view["truth"], view["amount"], view["filt"], view["fallbacks"]):
        wanted = ["approve" if fk else "decline" if qk >= td else "review" if qk >= tr else "approve" for qk, fk in zip(q, f)]
        fbs = ["approve" if fk else b for b, fk in zip(fb, f)]
        total += costs.score(costs.DEFAULT, wanted, fbs, y, a).summary["cost"]
    return total / len(view["scores"])


def reference_15b() -> dict:
    """The frozen 1.5B on dev (Phase 5 traces): AUC by variant and type, dev cost (argmax and thresholds)."""
    traces = tagged(REF_EVAL, "dev", "hybrid-1.5b", DEV_SEEDS, "dev")
    aucs = {k: dev_auc(variant_view(traces, k))[0] for k in d3.ORDERS}
    v = variant_view(traces, "both")
    tr, td, _ = pick_thresholds(v["scores"], v["truth"], v["amount"], v["filt"], v["fallbacks"])
    return {"eval": REF_EVAL, "auc": aucs, "rows": dev_auc(v)[1], "type_auc": type_aucs(traces, v),
            "thresholds": {"review": tr, "decline": td}, "dev_cost_thresholds": dev_cost(v, tr, td),
            "dev_cost_argmax": float(np.mean([recs[-1]["summary"]["cost"] for recs in traces]))}


# ---------------------------------------------------------------- the run

def run(eval_id: str, models: list[str], log=print) -> dict:
    from . import eval as ev
    from .capture import load_all

    t, s = load_all()
    dev = [windows.make(t, "dev", k) for k in DEV_SEEDS]
    lr = ev.logreg_decider(t, s)
    lr_traces = run_traces(eval_id, "dev", "logreg", dev, lambda w: [rules.TypeFilter(), lr], t, s, {}, log)
    lr_auc = dev_auc(logreg_scores_view(lr_traces))
    log(f"dev AUC logreg {lr_auc[0]:.4f} ({lr_auc[1]} rows, {lr_auc[2]} fraud)")
    ref = reference_15b()
    log(f"dev AUC frozen 1.5B (eval {REF_EVAL}) {ref['auc']['both']:.4f}")

    dev_info = {"logreg": {"auc": lr_auc[0], "rows": lr_auc[1], "fraud": lr_auc[2]}, "frozen_1.5b": ref}
    for name in models:
        setup, m, info = SETUPS[name], None, {}
        needs = [trace_file(eval_id, "dev", setup, w.id) for w in dev] + [trace_file(eval_id, "dev", setup + "-fwd", "dev0")]
        if not all(finished(p) for p in needs):
            m = Model(name, log)
            info = m.info
        dec = lambda variant, th=None: m.decider(variant, th) if m else None
        traces = run_traces(eval_id, "dev", setup, dev, lambda w: [rules.TypeFilter(), dec("both")],
                            t, s, {**info, "orders": "both"}, log)
        run_traces(eval_id, "dev", setup + "-fwd", dev[:1], lambda w: [rules.TypeFilter(), dec("fwd")],
                   t, s, {**info, "orders": "fwd", "timing_only": True}, log)
        aucs = {k: dev_auc(variant_view(traces, k))[0] for k in d3.ORDERS}
        variant, gain = choose_variant(aucs)
        chosen = aucs[variant]
        gate = chosen >= lr_auc[0]
        vv = variant_view(traces, variant)
        tr, td, grid = pick_thresholds(vv["scores"], vv["truth"], vv["amount"], vv["filt"], vv["fallbacks"])
        timing = read_run(trace_file(eval_id, "dev", setup + "-fwd", "dev0"))[-1]["summary"]["latency_ms"]
        dev_info[name] = {"auc": aucs, "rows": dev_auc(vv)[1], "fraud": dev_auc(vv)[2],
                          "debias_gain": gain, "variant": variant, "flip_rate": flip_rate(traces),
                          "type_auc": type_aucs(traces, vv),
                          "vs_frozen_1.5b": {"rule": "AUC(chosen variant) > AUC(frozen 1.5B, both orders) on the same dev rows",
                                             "delta": chosen - ref["auc"]["both"], "better": chosen > ref["auc"]["both"]},
                          "gate": {"rule": "AUC(chosen variant) >= AUC(logreg) on dev0-dev9 model rows", "pass": gate},
                          "thresholds": {"review": tr, "decline": td}, "grid_best": sorted(grid, key=lambda g: g[2])[:5],
                          "dev_cost_thresholds": dev_cost(vv, tr, td),
                          "latency_both": traces[0][-1]["summary"]["latency_ms"], "latency_single_dev0": timing}
        log(f"{name}: dev AUC both {aucs['both']:.4f} fwd {aucs['fwd']:.4f} rev {aucs['rev']:.4f} -> {variant} "
            f"(gain {gain:+.4f}); vs frozen 1.5B {chosen - ref['auc']['both']:+.4f}; gate {'PASS' if gate else 'FAIL'} "
            f"({chosen:.4f} vs logreg {lr_auc[0]:.4f}); thresholds review >= {tr}, decline >= {td}")
        if gate:
            test = [windows.make(t, "test", k, wid=f"test{k}") for k in TEST_SEEDS]
            if m is None and not all(finished(trace_file(eval_id, "eval", setup, w.id)) for w in test):
                m = Model(name, log)
                info = m.info
            run_traces(eval_id, "eval", setup, test, lambda w: [rules.TypeFilter(), dec(variant, (tr, td))], t, s,
                       {**info, "orders": variant}, log, {"thresholds": {"review": tr, "decline": td}})
        if m:
            m.close()
    return report(eval_id, dev_info, log)


# ---------------------------------------------------------------- results

def report(eval_id: str, dev_info: dict, log=print) -> dict:
    out = {"eval": eval_id, "script": "fraud/eval_lfm.py", "old_eval": OLD_EVAL, "dev": dev_info,
           "debias_min_gain": DEBIAS_MIN_GAIN, "grid": list(GRID), "test": {}}
    old = {k: tagged(OLD_EVAL, "eval", k, TEST_SEEDS, "test") for k in ("hybrid-1.5b", "logreg", "rules")}
    for name in MODELS:
        setup = SETUPS[name]
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
        out["test"][name] = {
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
    lg, ref = out["dev"]["logreg"], out["dev"]["frozen_1.5b"]
    L = [f"# LFM3 eval {out['eval']}", "",
         "Pre-registration: the `fraud/eval_lfm.py` docstring. Model scores are uncalibrated.", "",
         "## Dev (dev0–dev9 model rows)", "",
         f"logreg AUC {lg['auc']:.4f} ({lg['rows']} rows, {lg['fraud']} fraud): the gate. "
         f"Frozen 1.5B (eval {ref['eval']}, both orders): {ref['auc']['both']:.4f}.", "",
         "| model | AUC both | AUC fwd | AUC rev | debias gain | variant | vs frozen 1.5B | AUC TRANSFER | AUC CASH_OUT | gate | thresholds | dev cost / window | dev0 p90 both / one order (ms) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| frozen 1.5B | {ref['auc']['both']:.4f} | {ref['auc']['fwd']:.4f} | {ref['auc']['rev']:.4f} | | both | | "
         f"{ref['type_auc']['TRANSFER']:.4f} | {ref['type_auc']['CASH_OUT']:.4f} | | "
         f"{ref['thresholds']['review']} / {ref['thresholds']['decline']} | {ref['dev_cost_thresholds']:,.0f} | |"]
    for m in MODELS:
        if m not in out["dev"]:
            continue
        d = out["dev"][m]
        L.append(f"| {m} | {d['auc']['both']:.4f} | {d['auc']['fwd']:.4f} | {d['auc']['rev']:.4f} | "
                 f"{d['debias_gain']:+.4f} | {d['variant']} | {d['vs_frozen_1.5b']['delta']:+.4f} "
                 f"({'better' if d['vs_frozen_1.5b']['better'] else 'not better'}) | "
                 f"{d['type_auc']['TRANSFER']:.4f} | {d['type_auc']['CASH_OUT']:.4f} | "
                 f"{'PASS' if d['gate']['pass'] else 'FAIL'} | {d['thresholds']['review']} / {d['thresholds']['decline']} | "
                 f"{d['dev_cost_thresholds']:,.0f} | {d['latency_both']['p90']:.0f} / {d['latency_single_dev0']['p90']:.0f} |")
    L += ["", "Dev costs use thresholds picked on the same dev windows (in-sample). "
          f"The frozen 1.5B with argmax actions: {ref['dev_cost_argmax']:,.0f} per window.",
          "", "## Test (test100–test119)", ""]
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

def smoke(name: str, n: int = 5) -> None:
    """A few dev0 model rows: both orders and each order alone, timing. Not a result."""
    from .capture import load_all

    t, s = load_all()
    w = windows.make(t, "dev", 0)
    rows = [int(i) for i in w.rows if t.type[i] not in rules.FRAUD_FREE][:n]
    m = Model(name, print)
    try:
        for i in rows:
            t0 = time.perf_counter()
            both = m.decider("both").decide(t, s, i)
            ms = (time.perf_counter() - t0) * 1000
            fwd = m.decider("fwd").decide(t, s, i)
            print(f"row {i} fraud={int(t.is_fraud[i])} {ms:6.0f} ms  tokens {both['prompt_tokens']}")
            for k, q in enumerate(("fwd", "rev")):
                print(f"  {q}: " + ", ".join(f"{a[:6]} {v:.4f}" for a, v in order(both, k).items()))
            same = all(abs(fwd["scores"][a] - order(both, 0)[a]) < 1e-3 for a in brain.ACTIONS)
            print(f"  fwd alone matches fwd in the pair: {same}")
        if m.engine:
            print("clipped reads:", m.engine.backend.clipped, "of", m.engine.backend.reads)
    finally:
        m.close()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--resume", help="eval id to finish")
    ap.add_argument("--report", help="eval id: rebuild the results files from its traces and saved dev info")
    ap.add_argument("--smoke", choices=MODELS, help="a few dev rows through the model (no files written)")
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
