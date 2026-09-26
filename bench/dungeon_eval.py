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

Usage:
    python -m bench.dungeon_eval                       # all setups, seeds 0-7
    python -m bench.dungeon_eval --setups random --seeds 0 1
Traces go to demo/output/eval_<time>/ (rebuild a replay page with demo.dungeon.capture --rebuild).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from demo.dungeon.capture import OUT as DEMO_OUT
from demo.dungeon.capture import capture, to_jsonl, warm_up
from system_one import load_config, make_engine
from system_one.config import REPO_ROOT

from .hwinfo import host_info

RESULTS = Path(__file__).parent / "results"
SETUPS = {
    "3b-closer": ("config/lenovo-3b.toml", "closer"),
    "3b-steps": ("config/lenovo-3b.toml", "steps"),
    "1.5b-closer": ("config/default.toml", "closer"),
    "random": ("config/mock.toml", "closer"),
}


def free(engine) -> None:
    del engine
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
    ap.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    traces = DEMO_OUT / f"eval_{stamp}"
    traces.mkdir(parents=True, exist_ok=True)
    runs, backends = [], {}
    by_config: dict[str, list[str]] = {}
    for name in args.setups:  # load each model once
        by_config.setdefault(SETUPS[name][0], []).append(name)
    for config, names in by_config.items():
        cfg = load_config(REPO_ROOT / config)
        engine = make_engine(cfg)
        if cfg["backend"].get("kind") != "mock":
            warm_up(engine)
        for name in names:
            backends[name] = engine.backend.info()
            for seed in args.seeds:
                t0 = time.perf_counter()
                records = capture(engine, seed, verbose=False, label_style=SETUPS[name][1])
                (traces / f"{name}_seed{seed}.jsonl").write_text(to_jsonl(records), encoding="utf-8")
                summary = records[-1]["summary"]
                runs.append({"setup": name, "seed": seed, "wall_s": round(time.perf_counter() - t0, 1), "summary": summary})
                print(json.dumps({"setup": name, "seed": seed, **{k: summary[k] for k in (
                    "outcome", "cause", "ticks", "key_tick", "escape_tick", "rooms_seen", "gems", "hits", "stuck_ticks",
                    "forward_ms_median")}}), flush=True)
        free(engine)
    table = {name: aggregate([r for r in runs if r["setup"] == name]) for name in args.setups}
    for name, agg in table.items():
        print(name, json.dumps(agg))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"dungeon_eval_{stamp}.json"
    path.write_text(json.dumps({"host": host_info(), "backends": backends, "args": vars(args), "setups": SETUPS,
                                "traces": str(traces), "aggregate": table, "runs": runs}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
