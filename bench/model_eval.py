"""Accuracy of candidate models and answer templates on bench/eval_set.py.

Usage: python -m bench.model_eval --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
Writes raw per-item results to bench/results/model_eval_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

from system_one import Engine
from system_one.backends.hf import HFBackend

from .eval_set import SELECTION_CATEGORIES, eval_items

VARIANTS = [
    ("label", '{"choice": "<label>"}'),
    ("label", "Answer: <label>"),
    ("text", '{"choice": "<label>"}'),
]
OUT = Path(__file__).parent / "results"


def run(model: str, device: str, dtype: str, max_batch: int = 32) -> dict:
    backend = HFBackend(model, device=device, dtype=dtype, max_batch=max_batch)
    items = eval_items()
    report = {"backend": backend.info(), "variants": []}
    for answer, template in VARIANTS:
        engine = Engine(backend, answer=answer, answer_template=template)
        engine.decide_batch([items[0].decision])  # warm-up (kernel compilation)
        t = time.perf_counter()
        results = engine.decide_batch([it.decision for it in items])
        seconds = time.perf_counter() - t
        per_cat = defaultdict(list)
        rows = []
        for it, r in zip(items, results):
            ok = r.choice in it.correct
            per_cat[it.category].append(ok)
            rows.append({
                "category": it.category, "instruction": it.decision.instruction,
                "options": list(it.decision.options), "correct": sorted(it.correct),
                "choice": r.choice, "ok": ok, "probs": r.probs, "outside_mass": r.outside_mass,
                "method": r.method, "top_token": r.top_token, "prompt_tokens": r.prompt_tokens,
            })
        summary = {
            "answer": answer, "template": template,
            "accuracy": sum(r["ok"] for r in rows) / len(rows),
            "accuracy_by_category": {k: sum(v) / len(v) for k, v in per_cat.items()},
            "mean_chosen_prob": sum(max(r["probs"]) for r in rows) / len(rows),
            "mean_outside_mass": sum(r["outside_mass"] for r in rows) / len(rows),
            "first_option_rate": sum(r["choice"] == r["options"][0] for r in rows) / len(rows),
            "multi_token_items": sum(r["method"] == "multi_token" for r in rows),
            "batch_seconds": seconds, "items": len(rows),
        }
        summary["passes_selection_rule"] = all(
            summary["accuracy_by_category"][c] >= 0.8 for c in SELECTION_CATEGORIES
        )
        report["variants"].append({"summary": summary, "rows": rows})
        cats = " ".join(f"{k}={v:.0%}" for k, v in summary["accuracy_by_category"].items())
        print(f"{model:32} {answer:5} {template:24} acc={summary['accuracy']:.0%} [{cats}] "
              f"p={summary['mean_chosen_prob']:.2f} outside={summary['mean_outside_mass']:.3f} "
              f"firstopt={summary['first_option_rate']:.0%} multi={summary['multi_token_items']} "
              f"batch={seconds:.2f}s pass={summary['passes_selection_rule']}", flush=True)
    del backend
    gc.collect()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-batch", type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    reports = [run(m, args.device, args.dtype, args.max_batch) for m in args.models]
    path = OUT / f"model_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(reports, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
