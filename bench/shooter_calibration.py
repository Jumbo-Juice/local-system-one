"""Rule calibration for the shooter demo, with bots only (no model).

The rules in demo/shooter/world.py were chosen with this script before any model run, on dev
seeds 1000-1059 (never used for the model evaluation). Bots (demo/shooter/bots.py):
    reference  BFS + aim rule, dodges every bullet it can see coming: the ceiling
    dodge50    the reference bot, but it ignores threats on a random half of the ticks
    nododge    the reference bot that never dodges (it still picks safe-looking firing spots)
    random     random move, random shot: the floor
Decision rule (implementation choice): keep the variant in which the reference bot always wins
and a bot that never dodges wins about half the time. That was "dmg15", now the default Rules.

Ammo (--ammo; added after the model evaluations on seeds 0-39, which ran without ammo). The
combat rules above stay frozen; the grid varies the bullets at the start, the ammo boxes and the
reload time. Bots:
    reference  as above, and it reloads when no enemy is in sight and fetches known ammo boxes
               that fit its reserve
    noammo     the reference bot that ignores ammo: it reloads only an empty gun and never walks
               to a box on purpose (it still picks up boxes it happens to step on)
    nododge    the reference bot that never dodges (it manages ammo)
    random     random move, random shot or reload: the floor
Decision rule, fixed before the first ammo run: among the variants in which the reference bot
escapes all 60 dev seeds and never runs out of bullets, keep the one in which the bot that ignores
ammo escapes closest to half of the seeds (30); on a tie, the one with more bullets at the start.
Round 1 (bench/results/shooter_calibration_ammo_20260928_122708.json): no variant met the rule. The
reference bot ran dry in every variant, even with 36 bullets at the start ("ample": 60/60 escaped,
dry on one seed, where no box lay in the four rooms before the three-enemy key room). Round 2 adds
more generous variants, written before running it, under the same rule; if none meets it, the
variant with the fewest dry reference runs, then the rule's criterion, then more bullets.

    python -m bench.shooter_calibration                 # the combat grid on the dev seeds (no ammo)
    python -m bench.shooter_calibration --ammo          # the ammo grid on the dev seeds
    python -m bench.shooter_calibration --seeds 0 1 2   # the frozen rules on other seeds (with ammo)
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
VARIANTS = {  # name -> Rules overrides against the first draft (enemy bullets 20, brutes 25); no ammo
    "draft": dict(ammo=False, enemy_shot_damage=20, brute_damage=25),
    "dmg15": dict(ammo=False),  # the frozen combat rules
    "period6": dict(ammo=False, enemy_shot_damage=20, brute_damage=25, gunner_period=6),
    "dmg15_p6": dict(ammo=False, gunner_period=6),
    "potions5": dict(ammo=False, enemy_shot_damage=20, brute_damage=25, potions=5),
    "hp2": dict(ammo=False, enemy_shot_damage=20, brute_damage=25, gunner_hp=2, brute_hp=3),
}
AMMO_VARIANTS = {  # name -> ammo settings on the frozen combat rules (magazine 6 unless stated)
    "ample": dict(reserve=30, ammo_boxes=4, ammo_box=10, max_reserve=30, reload_ticks=3),  # 36 at the start
    "mid": dict(reserve=24, ammo_boxes=4, ammo_box=10, max_reserve=30, reload_ticks=3),  # 30
    "tight": dict(reserve=18, ammo_boxes=4, ammo_box=10, max_reserve=30, reload_ticks=3),  # 24
    "scarce": dict(reserve=12, ammo_boxes=5, ammo_box=10, max_reserve=30, reload_ticks=3),  # 18
    "tight_r2": dict(reserve=18, ammo_boxes=4, ammo_box=10, max_reserve=30, reload_ticks=2),  # 24, faster reload
    "tight_m8": dict(magazine=8, reserve=16, ammo_boxes=4, ammo_box=10, max_reserve=30, reload_ticks=3),  # 24
    # round 2
    "ample_b5": dict(reserve=30, ammo_boxes=5, ammo_box=10, max_reserve=30, reload_ticks=3),  # 36, one more box
    "ample_b6": dict(reserve=30, ammo_boxes=6, ammo_box=10, max_reserve=30, reload_ticks=3),  # 36, two more boxes
    "plenty": dict(reserve=36, ammo_boxes=4, ammo_box=10, max_reserve=36, reload_ticks=3),  # 42
    "plenty_b5": dict(reserve=36, ammo_boxes=5, ammo_box=10, max_reserve=36, reload_ticks=3),  # 42, one more box
}


def dodging(p: float, manage_ammo: bool = True):
    """The reference bot that sees threats on a seeded random share ``p`` of the ticks."""
    def policy(d: Dungeon):
        if random.Random(d.seed * 1000 + d.tick).random() < p:
            return bots.reference(d, manage_ammo=manage_ammo)
        return bots.reference(d, see_threats=False, manage_ammo=manage_ammo)
    return policy


POLICIES = {
    "reference": lambda seed: bots.reference,
    "dodge50": lambda seed: dodging(0.5),
    "nododge": lambda seed: dodging(0.0),
    "random": lambda seed: bots.random_policy(random.Random(seed)),
}
AMMO_POLICIES = {
    "reference": lambda seed: bots.reference,
    "noammo": lambda seed: dodging(1.0, manage_ammo=False),
    "nododge": lambda seed: dodging(0.0),
    "random": lambda seed: bots.random_policy(random.Random(seed)),
}


def run(rules: Rules, seeds, policy: str, policies=POLICIES) -> dict:
    outs = []
    for s in seeds:
        d = Dungeon(s, rules)
        pol, hits, dry = policies[policy](s), 0, 0
        while d.outcome is None:
            hits += sum(e["kind"] == "hit" for e in d.step(*pol(d)))
            dry += rules.ammo and d.ammo_total() == 0
        o = {"seed": s, "outcome": d.outcome, "cause": d.cause, "ticks": d.tick, "hits": hits,
             "rooms_seen": len(d.seen), "kills": d.kills, "health": d.health}
        if rules.ammo:
            o.update(shots=d.shots, reloads=d.reloads, ammo_picked=d.ammo_picked, ticks_dry=dry,
                     ammo_left=d.ammo_total())
        outs.append(o)
    esc = [o["ticks"] for o in outs if o["outcome"] == "escaped"]
    out = {"escaped": len(esc), "runs": len(outs), "escape_tick_median": statistics.median(esc) if esc else None,
           "died": sum(o["outcome"] == "died" for o in outs), "timeout": sum(o["outcome"] == "timeout" for o in outs),
           "hits_mean": round(statistics.mean(o["hits"] for o in outs), 2)}
    if rules.ammo:
        out.update(ran_dry=sum(o["ticks_dry"] > 0 for o in outs),
                   shots_mean=round(statistics.mean(o["shots"] for o in outs), 1),
                   ammo_picked_mean=round(statistics.mean(o["ammo_picked"] for o in outs), 1))
    out["runs_detail"] = outs
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: dev seeds 1000-1059, all variants")
    ap.add_argument("--ammo", action="store_true", help="the ammo grid instead of the combat grid")
    args = ap.parse_args()
    seeds = args.seeds if args.seeds else list(range(1000, 1060))
    if args.ammo:
        grid, policies = AMMO_VARIANTS, AMMO_POLICIES
    else:
        grid, policies = VARIANTS, POLICIES
    variants = grid if not args.seeds else {"frozen": {}}
    if args.seeds:
        policies = AMMO_POLICIES
    table = {}
    for name, over in variants.items():
        rules = Rules(**over)
        table[name] = {p: run(rules, seeds, p, policies) for p in policies}
        print(name.ljust(9), " | ".join(
            f"{p} {r['escaped']}/{r['runs']} (hits {r['hits_mean']}" + (f", dry {r['ran_dry']}" if "ran_dry" in r else "") + ")"
            for p, r in table[name].items()), flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"shooter_calibration_{'ammo_' if args.ammo else ''}{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"seeds": seeds, "variants": {k: grid.get(k, {}) for k in variants}, "table": table},
                               indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
