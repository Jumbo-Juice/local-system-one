"""Rule calibration for the shooter demo, with bots only (no model).

The rules in demo/shooter/world.py were chosen with this script before any model run, on dev
seeds 1000-1059 (never used for the model evaluation). Bots (demo/shooter/bots.py):
    reference  BFS + aim rule, dodges every bullet it can see coming: the ceiling
    dodge50    the reference bot, but it ignores threats on a random half of the ticks
    nododge    the reference bot that never dodges (it still picks safe-looking firing spots)
    random     random move, random shot: the floor
Decision rule (implementation choice): keep the variant in which the reference bot always wins
and a bot that never dodges wins about half the time. That was "dmg15", now the default Rules.

    python -m bench.shooter_calibration                 # the variant grid on the dev seeds
    python -m bench.shooter_calibration --seeds 0 1 2   # the frozen rules on other seeds
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from demo.shooter import bots
from demo.shooter.world import Dungeon, Rules

RESULTS = Path(__file__).parent / "results"
VARIANTS = {  # name -> Rules overrides against the first draft (enemy bullets 20, brutes 25)
    "draft": dict(enemy_shot_damage=20, brute_damage=25),
    "dmg15": {},  # the frozen default
    "period6": dict(enemy_shot_damage=20, brute_damage=25, gunner_period=6),
    "dmg15_p6": dict(gunner_period=6),
    "potions5": dict(enemy_shot_damage=20, brute_damage=25, potions=5),
    "hp2": dict(enemy_shot_damage=20, brute_damage=25, gunner_hp=2, brute_hp=3),
}


def dodging(p: float):
    """The reference bot that sees threats on a seeded random share ``p`` of the ticks."""
    def policy(d: Dungeon):
        if random.Random(d.seed * 1000 + d.tick).random() < p:
            return bots.reference(d)
        return bots.reference(d, see_threats=False)
    return policy


POLICIES = {
    "reference": lambda seed: bots.reference,
    "dodge50": lambda seed: dodging(0.5),
    "nododge": lambda seed: dodging(0.0),
    "random": lambda seed: bots.random_policy(random.Random(seed)),
}


def run(rules: Rules, seeds, policy: str) -> dict:
    outs = []
    for s in seeds:
        d = Dungeon(s, rules)
        pol, hits = POLICIES[policy](s), 0
        while d.outcome is None:
            hits += sum(e["kind"] == "hit" for e in d.step(*pol(d)))
        outs.append({"seed": s, "outcome": d.outcome, "cause": d.cause, "ticks": d.tick, "hits": hits,
                     "rooms_seen": len(d.seen), "kills": d.kills, "health": d.health})
    esc = [o["ticks"] for o in outs if o["outcome"] == "escaped"]
    return {"escaped": len(esc), "runs": len(outs), "escape_tick_median": statistics.median(esc) if esc else None,
            "died": sum(o["outcome"] == "died" for o in outs), "timeout": sum(o["outcome"] == "timeout" for o in outs),
            "hits_mean": round(statistics.mean(o["hits"] for o in outs), 2), "runs_detail": outs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: dev seeds 1000-1059, all variants")
    args = ap.parse_args()
    seeds = args.seeds if args.seeds else list(range(1000, 1060))
    variants = VARIANTS if not args.seeds else {"dmg15": {}}
    table = {}
    for name, over in variants.items():
        rules = Rules(**over)
        table[name] = {p: run(rules, seeds, p) for p in POLICIES}
        print(name.ljust(9), " | ".join(f"{p} {r['escaped']}/{r['runs']} (hits {r['hits_mean']})"
                                        for p, r in table[name].items()), flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"shooter_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"seeds": seeds, "variants": {k: VARIANTS[k] for k in variants}, "table": table},
                               indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
