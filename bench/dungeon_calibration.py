"""Choose the new dungeon's ghoul rules with bots only (no model), on dev seeds 1000-1059.

How the rules were chosen (2026-09-30; not pre-registered: exploratory bot runs, then this grid to
record them). Aim: the careful reference bot escapes nearly every dev seed (the shooter's did 60/60)
and random moves none. The first draft (ghouls moving 2 ticks in 3, chasing anywhere in their room,
never tiring) trapped the careful bot: a ghoul shadowing it along a doorway from inside its room
never gave way. Slower ghouls did not fix that; ghouls that tire after some ticks of chasing and walk
back to their post did. 12 ticks was the first value tried that passed; 8 and 20 behave alike.

Result (bench/results/dungeon_calibration/dungeon_calibration_20260930_094509.json), dev seeds
1000-1059: draft 28/60 reference, 50/60 careless, 0/60 random; tire after 12 (frozen in
demo/dungeon/world.py) 59/60, 52/60, 0/60.

    python -m bench.dungeon_calibration
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from demo.dungeon import bots
from demo.dungeon.world import Dungeon, Rules

RESULTS = Path(__file__).parent / "results" / "dungeon_calibration"
DRAFT = replace(Rules(), ghoul_chase=10 ** 6)  # the first draft: ghouls never tire
GRID = {
    "draft": DRAFT,
    "slower ghouls (1 tick in 2)": replace(DRAFT, ghoul_move_every=2),
    "tire after 12 ticks (chosen)": Rules(),
    "tire after 8 ticks": replace(Rules(), ghoul_chase=8),
    "tire after 20 ticks": replace(Rules(), ghoul_chase=20),
    "slower and tire after 12": replace(Rules(), ghoul_move_every=2),
}


def play(rules: Rules, seed: int, who: str) -> dict:
    d = Dungeon(seed, rules)
    policy = {"reference": bots.reference, "careless": lambda d: bots.reference(d, see_threats=False),
              "random": bots.random_policy(random.Random(seed))}[who]
    hits = 0
    while d.outcome is None:
        hits += sum(e["kind"] == "hit" for e in d.step(policy(d)))
    return {"seed": seed, "outcome": d.outcome, "ticks": d.tick, "hits": hits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1060)))
    args = ap.parse_args()
    table = {}
    for name, rules in GRID.items():
        row = {}
        for who in ("reference", "careless", "random"):
            runs = [play(rules, s, who) for s in args.seeds]
            esc = [r["ticks"] for r in runs if r["outcome"] == "escaped"]
            row[who] = {"escaped": len(esc), "died": sum(r["outcome"] == "died" for r in runs),
                        "out_of_time": sum(r["outcome"] == "timeout" for r in runs),
                        "escape_tick_median": statistics.median(esc) if esc else None,
                        "escape_tick_max": max(esc, default=None),
                        "hits_mean": round(statistics.mean(r["hits"] for r in runs), 2)}
        table[name] = {"rules": asdict(rules), **row}
        print(f"{name:32} " + "  ".join(f"{who} {row[who]['escaped']}/{len(args.seeds)}" for who in row), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"dungeon_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"seeds": args.seeds, "grid": table}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
