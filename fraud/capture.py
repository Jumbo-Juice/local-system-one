"""Decide one window under one setup and write the trace to fraud/runs/.

    python -m fraud.capture --setup hybrid --window dev0                            # Qwen2.5-1.5B
    python -m fraud.capture --setup hybrid --window dev0 --config config/lenovo-3b.toml
    python -m fraud.capture --setup hybrid --window dev0 --config config/mock.toml  # no model
    python -m fraud.capture --setup rules --window dev3                             # rules only (no model)

Setups: hybrid (type filter + model), rules, logreg (type filter + logistic regression), random
(type filter + random), approve-all. Windows: dev<seed> (training steps) or test<seed> (held-out
steps; the eval's job, so --test-ok is required here). Watch the trace with python -m fraud.viewer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import baselines, brain, costs, data, rules, signals, windows
from .runs import model_tag, stamp, to_jsonl, write_run

# Implementation choice (fraud/PROGRESS.md → Phase 2): on 2026-10-03 the iGPU reported 5.42 GiB, not the
# 8,097 MB in docs/machine.md, and the 1.5B with its float32 LM head ran out of memory.
HEAD_DTYPE = "model"  # the model dtype (bfloat16)
SETUPS = ("hybrid", "rules", "logreg", "random", "approve-all")
SIGNALS_SHOWN = [k for k in signals.NAMES if k != "hour"]


def chain_for(setup: str, engine=None, logreg: baselines.LogRegDecider | None = None, seed: int = 0) -> list:
    if setup == "hybrid":
        if engine is None:
            raise ValueError("hybrid needs an engine")
        return [rules.TypeFilter(), brain.ModelDecider(engine)]
    if setup == "rules":
        return [rules.RulesOnly()]
    if setup == "logreg":
        if logreg is None:
            raise ValueError("logreg needs a fitted model")
        return [rules.TypeFilter(), logreg]
    if setup == "random":
        return [rules.TypeFilter(), baselines.RandomDecider(seed)]
    if setup == "approve-all":
        return [baselines.ApproveAll()]
    raise ValueError(f"unknown setup {setup!r}")


def parse_window(spec: str) -> tuple[str, int]:
    for split in ("dev", "test"):
        if spec.startswith(split) and spec[len(split):].isdigit():
            return split, int(spec[len(split):])
    raise ValueError(f"window must look like dev3 or test12, not {spec!r}")


def tx_view(t: data.Transactions, s: dict, i: int) -> tuple[dict, dict]:
    tx = {"type": data.TYPES[t.type[i]], "amount": round(float(t.amount[i]), 2),
          "to": "merchant" if t.dest_merchant[i] else "customer", "hour": int(s["hour"][i])}
    return tx, {k: int(s[k][i]) for k in SIGNALS_SHOWN}


def run_window(t, s, w: windows.Window, chain: list, c: costs.Costs, setup: str, backend: dict | None = None,
               on_tx=None) -> list[dict]:
    decided = []
    for k, i in enumerate(w.rows):
        for d in chain:
            out = d.decide(t, s, int(i))
            if out is not None:
                break
        decided.append(out)
        if on_tx:
            on_tx(k, int(i), out)
    fraud = t.is_fraud[w.rows].astype(bool)
    amount = t.amount[w.rows]
    scored = costs.score(c, [d["action"] for d in decided], [d["fallback"] for d in decided], fraud, amount, w.weight)
    ticks = []
    for k, (i, d) in enumerate(zip(w.rows, decided)):
        tx, sig = tx_view(t, s, int(i))
        rec = {"type": "tx", "i": k, "row": int(i), "step": int(t.step[i]), "tx": tx, "signals": sig,
               "by": d["by"], "wanted": d["action"], "action": scored.final[k], "fallback": d["fallback"],
               "ms": round(d["ms"], 3), "truth": int(fraud[k]), "cost": round(float(scored.cost[k]), 2),
               "budget_left": scored.budget_left[k], "weight": round(float(w.weight[k]), 4)}
        for key in ("rule", "scores", "orders", "outside_mass", "prompt_tokens"):
            if key in d:
                rec[key] = d[key]
        ticks.append(rec)
    model_ms = [r["ms"] for r in ticks if r["by"] == "model"]
    all_ms = [r["ms"] for r in ticks]
    summary = dict(scored.summary)
    summary.update({
        "setup": setup,
        "total_ms": round(float(sum(all_ms)), 1),
        "decisions_by": {b: sum(r["by"] == b for r in ticks) for b in sorted({r["by"] for r in ticks})},
        "model_decisions": len(model_ms),
        "latency_ms": {"p50": brain.percentile(model_ms, 50), "p90": brain.percentile(model_ms, 90),
                       "max": max(model_ms, default=0.0), "mean": float(np.mean(model_ms)) if model_ms else 0.0},
        "outside_mass_mean": float(np.mean([r["outside_mass"] for r in ticks if "outside_mass" in r] or [0.0])),
    })
    header = {"type": "header", "app": "fraud", "setup": setup, "window": w.header(), "backend": backend or {},
              "costs": c.as_dict(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "signals": SIGNALS_SHOWN,
              "actions": list(costs.ACTIONS), "rules": [list(r) for r in rules.RULES],
              "fraud_free_types": [data.TYPES[k] for k in sorted(rules.FRAUD_FREE)]}
    for d in chain:
        if isinstance(d, brain.ModelDecider):
            first = next((int(i) for i in w.rows if t.type[i] not in rules.FRAUD_FREE), int(w.rows[0]))
            header.update({"options": list(brain.OPTIONS), "question": brain.QUESTION, "context": brain.CONTEXT,
                           "example_prompt": d.prompt(t, s, first), "order_debias": bool(d.engine.order_debias)})
    return [header] + ticks + [{"type": "end", "summary": summary}]


def load_all():
    t = data.load()
    return t, signals.load(t)


def main() -> None:
    from system_one import load_config, make_engine

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=SETUPS, default="hybrid")
    ap.add_argument("--window", default="dev0", help="dev<seed> or test<seed>")
    ap.add_argument("--config", default=None, help="model config for the hybrid (default config/default.toml)")
    ap.add_argument("--order-debias", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--head-dtype", default=HEAD_DTYPE,
                    help=f"LM head dtype (default {HEAD_DTYPE}: the float32 head did not fit the iGPU; fraud/PROGRESS.md)")
    ap.add_argument("--test-ok", action="store_true", help="allow a test window (normally only the eval runs them)")
    ap.add_argument("--out", default=None, help="write the trace here instead of fraud/runs/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    split, seed = parse_window(args.window)
    if split == "test" and not args.test_ok:
        ap.error("test windows belong to the pre-registered eval (python -m fraud.eval); pass --test-ok to override")
    t, s = load_all()
    w = windows.make(t, split, seed, wid=args.window)
    engine, backend, lr = None, {}, None
    if args.setup == "hybrid":
        cfg = load_config(args.config)
        cfg["engine"]["order_debias"] = args.order_debias
        if cfg["backend"].get("kind") == "hf":
            cfg["backend"]["head_dtype"] = args.head_dtype
        engine = make_engine(cfg)
        backend = engine.backend.info()
        if backend.get("kind") != "mock":
            brain.warm_up(engine)
    if args.setup == "logreg":
        from .eval import logreg_decider
        lr = logreg_decider(t, s)

    def show(k, i, out):
        if args.quiet or out["by"] == "filter":
            return
        print(f"{k:3} {data.TYPES[t.type[i]]:9} {t.amount[i]:>14,.2f} -> {out['action']:8} by {out['by']:6} "
              f"{out['ms']:7.1f} ms  truth {'FRAUD' if t.is_fraud[i] else 'legit'}", flush=True)

    started = stamp()
    records = run_window(t, s, w, chain_for(args.setup, engine, lr, seed), costs.DEFAULT, args.setup, backend, show)
    tag = f"_{model_tag(backend)}" if args.setup == "hybrid" else ""
    label = f"{args.setup}{tag}_{args.window}"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_jsonl(records), encoding="utf-8")
    else:
        path = write_run(label, records, when=started)
    summ = records[-1]["summary"]
    print(json.dumps({k: summ[k] for k in ("cost", "cost_natural", "recall", "precision", "alert_rate",
                                           "reviews_done", "latency_ms", "total_ms")}, indent=1))
    print("wrote", os.path.relpath(path))
    print("watch it: python -m fraud.viewer", os.path.relpath(path))


if __name__ == "__main__":
    main()
