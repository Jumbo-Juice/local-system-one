"""Latency and throughput of single-token decisions: batched vs sequential vs text generation.

Usage:
    python -m bench.bench --config config/default.toml
    python -m bench.bench --config config/default.toml --batches 1 8 --lengths 64 --repeats 3

For every (prompt length, batch size) the same B distinct decisions are timed:
  batched     one decide_batch() call (one forward pass per max_batch chunk)
  sequential  decide_sequential(): one forward pass per decision
Each mode gets one untimed warm-up call per configuration, then --repeats timed calls.
Optionally (--gen-baseline) a greedy text-generation baseline answers the same decisions
by generating the whole JSON answer token by token.
Raw timings go to bench/results/bench/bench_<model>_<timestamp>.json plus a .md summary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

from system_one import Decision, load_config, make_engine

from .hwinfo import host_info

OUT = Path(__file__).parent / "results" / "bench"
MOVES = ("move north", "move south", "move east", "move west", "stay")
DIRS = ("north", "south", "east", "west")
SENTENCES = (
    "Agent {i} stands at ({x},{y}).",
    "A gem lies {n} cells to the {d}.",
    "Food is {n} cells to the {d}.",
    "A hazard patrols {n} cells to the {d}.",
    "Energy is {e} out of 100 and health is {h} out of 100.",
    "The cell to the {d} is a wall.",
    "Another agent is {n} cells to the {d}.",
)


def make_decisions(engine, n: int, state_tokens: int, seed: int = 0) -> list[Decision]:
    """n distinct decisions whose state is about ``state_tokens`` tokens long."""
    out = []
    for i in range(n):
        rng = random.Random(seed * 100_003 + i)
        parts = [f"Agent {i} stands at ({rng.randrange(20)},{rng.randrange(15)})."]
        while len(engine.backend.encode(" ".join(parts))) < state_tokens:
            parts.append(rng.choice(SENTENCES).format(
                i=i, x=rng.randrange(20), y=rng.randrange(15), n=rng.randrange(1, 12),
                d=rng.choice(DIRS), e=rng.randrange(100), h=rng.randrange(100)))
        out.append(Decision("Which move brings you closer to the nearest gem?", MOVES, state=" ".join(parts)))
    return out


def _stats(times: list[float], n: int) -> dict:
    q = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0]] * 3
    med = statistics.median(times)
    return {
        "runs_s": times, "median_s": med, "min_s": min(times), "max_s": max(times),
        "p25_s": q[0], "p75_s": q[2],
        "per_decision_ms": 1000 * med / n, "per_decision_ms_min": 1000 * min(times) / n,
        "per_decision_ms_max": 1000 * max(times) / n, "decisions_per_s": n / med,
    }


def _time(fn, repeats: int) -> list[float]:
    fn()  # warm-up for this shape
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return times


_JSON_RE = re.compile(r'"choice"\s*:\s*"([A-Z]{1,2})"')
_BARE_RE = re.compile(r'\s*"?([A-Z]{1,2})\b')
_ASK = "Reply with the label of one option."
_ASK_JSON = 'Reply with JSON only, in the form {"choice": "<label>"}.'


def generation_baseline(engine, decisions: list[Decision], batch: int, repeats: int, style: str) -> dict:
    """Normal autoregressive answer, no prefill. style "label": same prompt as the engine, the
    model writes the label and stops. style "json": the model writes {"choice": "X"} token by token."""
    be = engine.backend
    prompts, labels = [], []
    for d in decisions:
        lab = engine.labels_for(len(d.options))
        text = engine.render(d, lab)
        text = text[: len(text) - len(engine.prefill)]  # drop the prefill: the model starts the reply
        if style == "json":
            text = text.replace(_ASK, _ASK_JSON)
        prompts.append(be.encode(text))
        labels.append(lab)
    chunks = [prompts[i:i + batch] for i in range(0, len(prompts), batch)]
    outputs: list[list[int]] = []

    def run():
        outputs.clear()
        for c in chunks:
            outputs.extend(be.generate(c, max_new_tokens=16))

    times = _time(run, repeats)
    texts = [be.decode(o) for o in outputs]
    parsed = []
    for t, lab, d in zip(texts, labels, decisions):
        m = _JSON_RE.search(t) or _BARE_RE.match(t)
        parsed.append(d.options[lab.index(m.group(1))] if m and m.group(1) in lab else None)
    stats = _stats(times, len(decisions))
    stats.update({
        "style": style, "new_tokens_mean": statistics.mean(len(o) for o in outputs),
        "parse_failures": sum(p is None for p in parsed), "answers": parsed, "sample_text": texts[:3],
    })
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024],
                    help="approximate state lengths in tokens (the prompt adds ~90 more)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--gen-baseline", action="store_true")
    ap.add_argument("--gen-batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--gen-decisions", type=int, default=16)
    ap.add_argument("--gen-styles", nargs="+", default=["label", "json"])
    ap.add_argument("--no-sweep", action="store_true", help="skip the batched/sequential sweep")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = make_engine(cfg)
    info = {"host": host_info(), "backend": engine.backend.info(), "config": cfg,
            "engine": {"answer": engine.answer, "prefill": engine.prefill, "suffix": engine.suffix},
            "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(info["host"]), "\n", json.dumps(info["backend"]), flush=True)
    engine.decide_batch(make_decisions(engine, 2, 32, seed=99))  # global warm-up (kernel compilation)

    rows = []
    for length in ([] if args.no_sweep else args.lengths):
        for b in args.batches:
            ds = make_decisions(engine, b, length, seed=length)
            tokens = [len(engine.prepare(d).ids) for d in ds]
            base = {"state_tokens": length, "prompt_tokens_mean": statistics.mean(tokens),
                    "prompt_tokens_max": max(tokens), "batch": b}
            for mode, fn in (("batched", lambda: engine.decide_batch(ds)),
                             ("sequential", lambda: engine.decide_sequential(ds))):
                try:
                    row = {**base, "mode": mode, **_stats(_time(fn, args.repeats), b)}
                except RuntimeError as exc:  # e.g. device out of memory
                    row = {**base, "mode": mode, "error": str(exc)[:300]}
                rows.append(row)
                if "error" in row:
                    print(f"len~{row['prompt_tokens_mean']:.0f} B={b:3} {mode:10} ERROR {row['error'][:80]}", flush=True)
                else:
                    print(f"len~{row['prompt_tokens_mean']:.0f} B={b:3} {mode:10} median {1000*row['median_s']:8.1f} ms"
                          f"  per-decision {row['per_decision_ms']:7.1f} ms"
                          f" [{row['per_decision_ms_min']:.1f}-{row['per_decision_ms_max']:.1f}]"
                          f"  {row['decisions_per_s']:6.1f} dec/s", flush=True)

    gen_rows = []
    if args.gen_baseline:
        ds = make_decisions(engine, args.gen_decisions, args.gen_length, seed=7)
        single = [r.choice for r in engine.decide_batch(ds)]
        for style, gb in [(st, gb) for st in args.gen_styles for gb in args.gen_batches]:
            try:
                g = generation_baseline(engine, ds, gb, max(1, args.repeats // 2), style)
            except NotImplementedError as exc:
                print("generation baseline skipped:", exc)
                break
            g.update({"batch": gb, "decisions": len(ds), "state_tokens": args.gen_length,
                      "agreement_with_single_token": sum(a == s for a, s in zip(g["answers"], single)) / len(ds)})
            gen_rows.append(g)
            print(f"generate {style:5} B={gb:3} per-decision {g['per_decision_ms']:7.1f} ms [{g['per_decision_ms_min']:.1f}-"
                  f"{g['per_decision_ms_max']:.1f}] new tokens {g['new_tokens_mean']:.1f} parse failures "
                  f"{g['parse_failures']} agreement {g['agreement_with_single_token']:.0%}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", str(info["backend"].get("model", "model")).split("/")[-1])
    stem = f"bench_{model}_{info['backend'].get('dtype', '')}_{time.strftime('%Y%m%d_%H%M%S')}{args.tag}"
    (OUT / f"{stem}.json").write_text(json.dumps({**info, "rows": rows, "generation": gen_rows}, indent=1), encoding="utf-8")
    (OUT / f"{stem}.md").write_text(summary_markdown(info, rows, gen_rows), encoding="utf-8")
    print("wrote", OUT / f"{stem}.json")


def summary_markdown(info: dict, rows: list[dict], gen_rows: list[dict]) -> str:
    h, b = info["host"], info["backend"]
    lines = [
        f"# Benchmark: {b.get('model')} ({b.get('dtype')}) on {b.get('device_name', b.get('device'))}",
        "",
        f"Host: {h['cpu']}, {h['ram_gb']} GB RAM, {h['os']}. Runtime: torch {b.get('torch')}, "
        f"transformers {b.get('transformers')}. Quantisation: {b.get('quantisation')}. Started {info['started']}.",
        "",
        "Per-decision latency = call time / batch size. Median over repeats; range = min–max.",
        "",
        "| prompt tokens (mean) | batch | mode | call median ms | per-decision ms | range | decisions/s |",
        "|---:|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['prompt_tokens_mean']:.0f} | {r['batch']} | {r['mode']} | error | | | |")
            continue
        lines.append(f"| {r['prompt_tokens_mean']:.0f} | {r['batch']} | {r['mode']} | {1000*r['median_s']:.1f} | "
                     f"{r['per_decision_ms']:.1f} | {r['per_decision_ms_min']:.1f}–{r['per_decision_ms_max']:.1f} | "
                     f"{r['decisions_per_s']:.1f} |")
    if gen_rows:
        lines += ["", "Text-generation baseline (greedy, no prefill, max 16 new tokens). style label: the model "
                  "writes the label and stops; style json: it writes {\"choice\": \"X\"}.", "",
                  "| style | batch | per-decision ms | range | new tokens | parse failures | agreement with single-token |",
                  "|---|---:|---:|---|---:|---:|---:|"]
        for g in gen_rows:
            lines.append(f"| {g['style']} | {g['batch']} | {g['per_decision_ms']:.1f} | {g['per_decision_ms_min']:.1f}–"
                         f"{g['per_decision_ms_max']:.1f} | {g['new_tokens_mean']:.1f} | {g['parse_failures']}/"
                         f"{g['decisions']} | {g['agreement_with_single_token']:.0%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
