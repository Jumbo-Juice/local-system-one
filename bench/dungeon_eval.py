"""Closed-loop evaluation of the single-agent dungeon demo.

Pre-registered before the first evaluation run (committed with this file):
- Seeds 0-7. Seed 100 was used during development and is excluded.
- Rules as in demo/dungeon/world.py (fixed before any model run): tick limit 400.
- Plan budget 1 per tick, tournament groups of 8.
- Setups:
    3b-closer    Qwen2.5-3B (config/lenovo-3b.toml), move outcomes worded "closer/farther" (demo default)
    3b-steps     Qwen2.5-3B, the first demo's wording "target: N steps"
    1.5b-closer  Qwen2.5-1.5B (config/default.toml), "closer/farther"
    random       mock backend (random decisions), "closer/farther": the floor
- Reported for every run, none dropped: outcome (escaped / died from enemy or starvation /
  out of time), tick of escape, key picked up, rooms seen, gems, hits, stuck ticks (streaks of
  6+ ticks without getting closer to the same target), forward-pass latency.
- The replay shown in the docs is seed 0 with 3b-closer, whatever its outcome.

Added after the pre-registered results (post hoc; docs/research.md -> Dungeon):
    3b-enemy-aware  3b-closer plus enemy_aware wording and safe spots (DungeonBrain). Evaluated on
                    seeds 0-15: 0-7 to compare with the runs above, 8-15 (never run before) to check
                    it is not fitted to seeds 0-7; 3b-closer is also run on 8-15 for comparison.

Usage:
    python -m bench.dungeon_eval                       # all setups, seeds 0-7
    python -m bench.dungeon_eval --setups random --seeds 0 1
    python -m bench.dungeon_eval --resume 20260926-163151   # resume that eval (its id is in the trace names)
Traces go to the dungeon pool: runs/dungeon/<eval id>_eval_<setup>_seed<N>.jsonl (python -m viewer).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from demo import runs as pool
from demo.common import warm_up
from demo.dungeon.capture import capture
from system_one import load_config, make_engine
from system_one.config import REPO_ROOT

from .hwinfo import host_info

RESULTS = Path(__file__).parent / "results" / "dungeon_eval"
PRE_REGISTERED = {"label_style": "closer", "enemy_aware": False}
SETUPS = {  # name -> (config, DungeonBrain options)
    "3b-closer": ("config/lenovo-3b.toml", PRE_REGISTERED),
    "3b-steps": ("config/lenovo-3b.toml", {**PRE_REGISTERED, "label_style": "steps"}),
    "1.5b-closer": ("config/default.toml", PRE_REGISTERED),
    "random": ("config/mock.toml", PRE_REGISTERED),
    "3b-enemy-aware": ("config/lenovo-3b.toml", {**PRE_REGISTERED, "enemy_aware": True}),
}


def empty_device_cache() -> None:
    """Call after dropping every reference to the engine (the caller's included)."""
    gc.collect()
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def aggregate(runs: list[dict]) -> dict:
    s = [r["summary"] for r in runs]
    esc = [x["escape_tick"] for x in s if x["outcome"] == "escaped"]
    mean = lambda k: round(statistics.mean(x[k] for x in s), 1)  # noqa: E731
    return {
        "runs": len(s), "escaped": len(esc),
        "died_enemy": sum(x["outcome"] == "died" and x["cause"] == "enemy" for x in s),
        "died_starvation": sum(x["outcome"] == "died" and x["cause"] == "starvation" for x in s),
        "out_of_time": sum(x["outcome"] == "timeout" for x in s),
        "key_picked": sum(x["key_tick"] is not None for x in s),
        "escape_tick_median": statistics.median(esc) if esc else None,
        "ticks_mean": mean("ticks"), "rooms_seen_mean": mean("rooms_seen"), "gems_mean": mean("gems"),
        "hits_mean": mean("hits"), "food_mean": mean("food_eaten"), "stuck_ticks_mean": mean("stuck_ticks"),
        "stuck_ticks_max": max(x["stuck_ticks"] for x in s),
        "forward_ms_median": round(statistics.median(x["forward_ms_median"] for x in s), 1),
        "forward_ms_p90_median": round(statistics.median(x["forward_ms_p90"] for x in s), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", default=["3b-closer", "3b-steps", "1.5b-closer", "random"], choices=list(SETUPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--resume", default=None, metavar="EVAL_ID",
                    help="reuse the finished runs of this eval (the time at the start of its trace names)")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    eval_id = args.resume or pool.stamp()
    trace = lambda name, seed: pool.trace_path("dungeon", f"eval_{name}_seed{seed}", eval_id)  # noqa: E731
    if args.resume and not any(pool.pool("dungeon").glob(f"{eval_id}_eval_*.jsonl")):
        ap.error(f"no traces of eval {eval_id} in {pool.pool('dungeon')}")
    runs, backends = [], {}

    def report(name: str, seed: int, summary: dict, wall: float | None, reused: bool) -> None:
        runs.append({"setup": name, "seed": seed, "wall_s": wall, "reused_trace": reused, "summary": summary})
        print(json.dumps({"setup": name, "seed": seed, "reused": reused, **{k: summary[k] for k in (
            "outcome", "cause", "ticks", "key_tick", "escape_tick", "rooms_seen", "gems", "hits", "stuck_ticks",
            "forward_ms_median")}}), flush=True)

    todo: dict[str, list[tuple[str, int]]] = {}
    for name in args.setups:
        for seed in args.seeds:
            path = trace(name, seed)
            end = pool.finished(path)
            if end:  # finished earlier: runs are deterministic, reuse it
                backends[name] = json.loads(path.open(encoding="utf-8").readline())["backend"]
                report(name, seed, end["summary"], None, True)
            else:
                todo.setdefault(SETUPS[name][0], []).append((name, seed))
    for config, jobs in todo.items():  # load each model once
        cfg = load_config(REPO_ROOT / config)
        engine = make_engine(cfg)
        if cfg["backend"].get("kind") != "mock":
            warm_up(engine)
        for name, seed in jobs:
            backends[name] = engine.backend.info()
            t0 = time.perf_counter()
            records = capture(engine, seed, verbose=False, **SETUPS[name][1])
            trace(name, seed).write_text(pool.to_jsonl(records), encoding="utf-8")
            report(name, seed, records[-1]["summary"], round(time.perf_counter() - t0, 1), False)
        engine = None  # drop the last reference so the weights can be freed before the next model
        empty_device_cache()
    runs.sort(key=lambda r: (args.setups.index(r["setup"]), r["seed"]))
    table = {name: aggregate([r for r in runs if r["setup"] == name]) for name in args.setups}
    for name, agg in table.items():
        print(name, json.dumps(agg))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"dungeon_eval_{stamp}.json"
    path.write_text(json.dumps({"host": host_info(), "backends": backends, "args": vars(args), "setups": SETUPS,
                                "eval_id": eval_id, "traces": f"runs/dungeon/{eval_id}_eval_*.jsonl",
                                "aggregate": table, "runs": runs}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
