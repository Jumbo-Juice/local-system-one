"""Closed-loop evaluation of the single-agent dungeon demo (demo/dungeon/, rebuilt on 2026-09-30).

The first dungeon and its evaluation are gone (0 of 48 model runs escaped; results in
docs/research.md -> Dungeon demo and bench/results/dungeon_eval/dungeon_eval_20260926_*.json,
code at commit 52bb540).

Fixed before the first model run of the new dungeon (committed with this file):
- Rules as frozen in demo/dungeon/world.py: tick limit 400. Chosen with bots only, on dev seeds
  1000-1059 (bench/dungeon_calibration.py): reference bot 59/60, careless bot 52/60, random 0/60.
- Plan budget 1 per tick, tournament groups of 8. One agent per run.
- Setups:
    1.5b           Qwen2.5-1.5B (config/default.toml), order averaging as configured (on)
    3b             Qwen2.5-3B (config/lenovo-3b.toml), order averaging as configured (off: listed order)
    random         mock backend (a fixed function of the prompt) through the same brain
    bot-reference  the non-model reference bot (demo/dungeon/bots.py): the ceiling, NOT a model
    bot-careless   the reference bot ignoring the ghouls: a careless player, NOT a model
    bot-random     uniformly random moves and dashes: the floor
- Reported for every run, none dropped: outcome (escaped / died / out of time), escape tick, key
  picked up, rooms seen, gems, hits taken, moves into a ghoul's reach while a safe move existed,
  hits after a move labelled safe, dashes, stuck ticks, loop ticks, forward-pass latency.
- Development seeds for the wording: 1000-1009. Seeds 0-4 are the 20-run check's
  (bench/escape_check.py); the default here is seeds 0-9.

Usage:
    python -m bench.dungeon_eval                           # all setups, seeds 0-9
    python -m bench.dungeon_eval --setups bot-reference bot-careless bot-random --seeds 1000 1001
    python -m bench.dungeon_eval --resume 20260930-120000   # resume that eval (its id is in the trace names)
Traces go to the dungeon pool: runs/dungeon/<eval id>_eval_<setup>_seed<N>.jsonl (python -m viewer).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from demo import runs as pool
from demo.common import warm_up
from demo.dungeon import bots
from demo.dungeon.capture import capture, order_debias
from demo.dungeon.world import Dungeon, Rules
from system_one import load_config, make_engine
from system_one.config import REPO_ROOT

from .hwinfo import host_info

RESULTS = Path(__file__).parent / "results" / "dungeon_eval"
SETUPS = {  # name -> (config, or None for a bot; bot name)
    "1.5b": ("config/default.toml", None),
    "3b": ("config/lenovo-3b.toml", None),
    "random": ("config/mock.toml", None),
    "bot-reference": (None, "reference"),
    "bot-careless": (None, "careless"),
    "bot-random": (None, "random"),
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


def bot_run(seed: int, which: str, rules: Rules | None = None) -> dict:
    """A bot run, summarised with the same keys the model runs report where they apply."""
    d = Dungeon(seed, rules)
    policy = {"reference": bots.reference, "careless": lambda d: bots.reference(d, see_threats=False),
              "random": bots.random_policy(random.Random(seed))}[which]
    hits, key_tick = 0, None
    while d.outcome is None:
        ev = d.step(policy(d))
        hits += sum(e["kind"] == "hit" for e in ev)
        key_tick = key_tick if key_tick is not None or not d.has_key else d.tick - 1
    return {"outcome": d.outcome, "cause": d.cause, "ticks": d.tick, "key_tick": key_tick,
            "escape_tick": d.tick - 1 if d.outcome == "escaped" else None, "rooms_seen": len(d.seen),
            "gems": d.collected, "hits": hits, "dashes": d.dashes}


def aggregate(runs: list[dict]) -> dict:
    s = [r["summary"] for r in runs]
    esc = [x["escape_tick"] for x in s if x["outcome"] == "escaped"]
    mean = lambda k: round(statistics.mean(x[k] for x in s), 1) if s and all(k in x for x in s) else None  # noqa: E731
    out = {
        "runs": len(s), "escaped": len(esc), "died": sum(x["outcome"] == "died" for x in s),
        "out_of_time": sum(x["outcome"] == "timeout" for x in s),
        "key_picked": sum(x["key_tick"] is not None for x in s),
        "escape_tick_median": statistics.median(esc) if esc else None,
        "ticks_mean": mean("ticks"), "rooms_seen_mean": mean("rooms_seen"), "gems_mean": mean("gems"),
        "hits_mean": mean("hits"), "dashes_mean": mean("dashes"),
    }
    if s and all("forward_ms_median" in x for x in s):
        out.update({
            "avoidable_risky_moves_mean": mean("avoidable_risky_moves"),
            "hits_after_safe_move": sum(x["hits_after_safe_move"] for x in s),
            "stuck_ticks_mean": mean("stuck_ticks"), "stuck_ticks_max": max(x["stuck_ticks"] for x in s),
            "loop_ticks_mean": mean("loop_ticks"), "loop_ticks_max": max(x["loop_ticks"] for x in s),
            "forward_ms_median": round(statistics.median(x["forward_ms_median"] for x in s), 1),
            "forward_ms_p90_median": round(statistics.median(x["forward_ms_p90"] for x in s), 1),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
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
        keys = ("outcome", "cause", "ticks", "key_tick", "escape_tick", "rooms_seen", "gems", "hits", "dashes",
                "stuck_ticks", "loop_ticks", "forward_ms_median")
        print(json.dumps({"setup": name, "seed": seed, "reused": reused, **{k: summary.get(k) for k in keys}}), flush=True)

    todo: dict[str, list[tuple[str, int]]] = {}
    for name in args.setups:
        config, which = SETUPS[name]
        for seed in args.seeds:
            if config is None:
                report(name, seed, bot_run(seed, which), None, False)
                continue
            path = trace(name, seed)
            end = pool.finished(path)
            if end:  # finished earlier: runs are deterministic, reuse it
                backends[name] = json.loads(path.open(encoding="utf-8").readline())["backend"]
                report(name, seed, end["summary"], None, True)
            else:
                todo.setdefault(config, []).append((name, seed))
    for config, jobs in todo.items():  # load each model once
        cfg = load_config(REPO_ROOT / config)
        engine = make_engine(cfg)
        if cfg["backend"].get("kind") != "mock":
            warm_up(engine)
        for name, seed in jobs:
            backends[name] = engine.backend.info()
            t0 = time.perf_counter()
            records = capture(engine, seed, verbose=False, order_debias=order_debias(cfg))
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
    path.write_text(json.dumps({"host": host_info(), "backends": backends, "args": vars(args), "rules": asdict(Rules()),
                                "setups": SETUPS, "eval_id": eval_id, "traces": f"runs/dungeon/{eval_id}_eval_*.jsonl",
                                "aggregate": table, "runs": runs}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
