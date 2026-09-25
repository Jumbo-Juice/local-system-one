"""Tiered goals vs flat control in the demo world (same seeds), plus a random-policy floor.

Usage: python -m bench.demo_compare --config config/default.toml --ticks 100 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from demo.brain import Controller
from demo.sim import run_headless
from demo.world import World
from system_one import Decision, Engine, load_config, make_engine
from system_one.backends.mock import MockBackend

from .hwinfo import host_info

OUT = Path(__file__).parent / "results"
KEYS = ("gems", "food_eaten", "hazard_hits", "deaths", "forward_ms_median", "forward_ms_p90",
        "decisions_per_tick_mean", "tournament_decisions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--group-size", type=int, default=8)
    args = ap.parse_args()
    engine = make_engine(load_config(args.config))
    engine.decide_batch([Decision("warm-up", ("a", "b"))])
    runs = []
    for label, eng, goals in (("model, tiered goals", engine, True), ("model, flat", engine, False),
                              ("random (mock), tiered goals", Engine(MockBackend()), True)):
        for seed in args.seeds:
            ctrl = Controller(World(seed=seed), eng, use_goals=goals, group_size=args.group_size)
            s = run_headless(ctrl, args.ticks, verbose=False)
            runs.append({"setup": label, "seed": seed, **{k: s[k] for k in KEYS}, "action_counts": s["action_counts"]})
            print(f"{label:28} seed {seed}: " + " ".join(f"{k}={s[k]:.1f}" if isinstance(s[k], float) else f"{k}={s[k]}"
                                                        for k in KEYS), flush=True)
    summary = {}
    for label in dict.fromkeys(r["setup"] for r in runs):
        rs = [r for r in runs if r["setup"] == label]
        summary[label] = {k: statistics.mean(r[k] for r in rs) for k in KEYS}
        print(f"MEAN {label:28} " + " ".join(f"{k}={v:.1f}" for k, v in summary[label].items()))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"demo_compare_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"host": host_info(), "backend": engine.backend.info(), "args": vars(args),
                                "runs": runs, "summary": summary}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
