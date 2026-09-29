"""Does the agent survive? Old strategy prompt vs "aware" (consequences + re-plan on condition change).

Usage: python -m bench.survival_compare --config config/lenovo-3b.toml --ticks 250 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from demo.grid.brain import Controller
from demo.grid.world import World
from system_one import Decision, load_config, make_engine

from .hwinfo import host_info

OUT = Path(__file__).parent / "results" / "survival_compare"


SETUPS = {
    "old prompts": {"aware": False, "label_moves": False},
    "aware (consequences, re-plan triggers, no stale distance)": {"aware": True, "label_moves": False},
    "aware + move options labelled with outcomes": {"aware": True, "label_moves": True},
}


def run(engine, setup: str, seed: int, ticks: int, plan_budget: int) -> dict:
    ctrl = Controller(World(seed=seed, n_agents=1), engine, group_size=8, plan_budget=plan_budget,
                      **SETUPS[setup])
    agent, brain = ctrl.world.agents[0], ctrl.brains[0]
    energy, strategy_when_low, food, gems, fw = [], Counter(), 0, 0, []
    stuck, streak, best = 0, 0, None  # stuck: ticks in streaks of 6+ without getting closer to the target
    for _ in range(ticks):
        cell = brain.target_cell()
        d = ctrl.world.distances(agent.pos).get(cell) if cell else None
        rep = ctrl.tick()
        if d is not None and cell == brain.target_cell():
            nd = ctrl.world.distances(agent.pos).get(cell, d)
            if best is None or nd < best:
                best, streak = nd, 0
            else:
                streak += 1
                if streak == 6:
                    stuck += 6
                elif streak > 6:
                    stuck += 1
        else:
            best, streak = None, 0
        energy.append(agent.energy)
        food += rep.events["food"]
        if agent.energy < 35:
            strategy_when_low[brain.stack.current.get("strategy", "-")] += 1
        if rep.decisions:
            fw.append(1000 * rep.forward_s)
    return {"setup": setup, "seed": seed, "stuck_ticks": stuck, "deaths": agent.deaths, "starved": agent.starved, "gems": agent.score,
            "food_eaten": food, "min_energy": min(energy), "ticks_at_zero_energy": sum(e == 0 for e in energy),
            "ticks_low_energy": sum(e < 35 for e in energy), "strategy_while_low": dict(strategy_when_low),
            "forward_ms_median": statistics.median(fw)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ticks", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--plan-budget", type=int, default=1)
    args = ap.parse_args()
    engine = make_engine(load_config(args.config))
    warm = [Decision("warm-up", tuple("abcdefgh"), state="Gems, food and hazards nearby. " * (10 + 8 * i)) for i in range(2)]
    engine.decide_batch(warm)
    engine.decide_batch(warm)
    runs = []
    for setup in SETUPS:
        for seed in args.seeds:
            r = run(engine, setup, seed, args.ticks, args.plan_budget)
            runs.append(r)
            print(json.dumps(r), flush=True)
    for setup in SETUPS:
        rs = [r for r in runs if r["setup"] == setup]
        keys = ("deaths", "starved", "stuck_ticks", "gems", "food_eaten", "min_energy", "ticks_at_zero_energy",
                "forward_ms_median")
        print(f"MEAN {setup}: " + " ".join(f"{k}={statistics.mean(r[k] for r in rs):.1f}" for k in keys))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"survival_compare_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"host": host_info(), "backend": engine.backend.info(), "args": vars(args),
                                "runs": runs}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
