"""Closed-loop evaluation of the single-agent shooter demo (demo/shooter/).

Pre-registered before the first evaluation run (committed with this file):
- Seeds 0-9. Never run before with any model or bot. Excluded: dev seeds 1000-1059 (rules were
  calibrated on them with bots) and 1000-1001 (prompt wording was developed on them with both models).
- Rules as frozen in demo/shooter/world.py: tick limit 400. Plan budget 1 per tick, groups of 8.
- One agent per run. Setups:
    1.5b          Qwen2.5-1.5B (config/default.toml), order averaging on (the demo default)
    3b            Qwen2.5-3B (config/lenovo-3b.toml), order averaging on (the demo default)
    1.5b-listed   the 1.5B with order averaging off (options read once, in the listed order)
    3b-listed     the 3B with order averaging off
    random        mock backend (random decisions) through the same brain: the floor
    bot-reference the non-model reference bot (demo/shooter/bots.py): the ceiling, NOT a model
    bot-nododge   the reference bot that never dodges: a careless player, NOT a model
- Reported for every run, none dropped: outcome (escaped / died to a gunner or brute / out of
  time), escape tick, key picked up, rooms seen and cleared, kills, shots and hits on target,
  hits taken, moves into danger while a safe move existed, held fire, stuck ticks, forward-pass
  latency (median and p90 per run).
- Primary result: escapes out of 10 for 1.5b and 3b. The -listed setups measure what order
  averaging does; they are not the demo.
- The replays shown in the docs are seed 0 with 1.5b and with 3b, whatever their outcome.

Pre-registered results (bench/results/shooter_eval_20260927_002315.json): escaped 1.5b 7/10,
3b 2/10, 1.5b-listed 0/10, 3b-listed 7/10, random 0/10, bot-reference 10/10, bot-nododge 5/10.

Added after those results (post hoc), fixed before running: a replication on seeds 10-19 (never
run) with 1.5b, 3b and 3b-listed, because 3b vs 3b-listed (2 vs 7 of 10) was the one surprising
difference (two-sided Fisher p = 0.07). Decision rule: switch the 3B demo to the listed order
only if 3b-listed also escapes more often than 3b on seeds 10-19; otherwise keep order
averaging for both models. Nothing else changes.
    python -m bench.shooter_eval --setups 1.5b 3b 3b-listed --seeds 10 11 12 13 14 15 16 17 18 19
Replication result (bench/results/shooter_eval_20260927_005732.json): escaped 1.5b 7/10, 3b 6/10,
3b-listed 7/10. By the rule the 3B demo now reads options in the listed order
([shooter] order_debias = false in config/lenovo-3b.toml); the setups above are unchanged.

Post-hoc fixes, found in the traces of both runs above, and a check fixed before running it.
The fixes change what the model reads, so the demo is no longer the version evaluated above:
  1. A risky stay on the target read "reach the target"; it now reads "on the target".
  2. A sleeping brute did not count as a threat, but stepping into its room wakes it before
     enemies act: all 4 hits taken after a move labelled safe were such steps. It now counts
     (labels, "closer" routes, firing spots) when the move enters its room (bots.brute_reach).
  3. Awake enemies of a room counted only while the agent was inside it or its doorway. Brutes
     that guarded a doorway from inside left the agent in the corridor with no fight on offer
     (1.5b seed 3: 389 stuck ticks). In a corridor out of a seen room with awake enemies, the
     agent now fights them, from firing spots in the room, its doorways or its corridors.
  The same functions drive the bots. While making the fixes they were run on dev seeds 1000-1059
  (where fix 3 first made the reference bot flip-flop, then fight rooms it had not seen; both
  corrected) and on seeds 0-29: reference bot 60/60 and 30/30 before and after the fixes.
Check: seeds 30-39 (never run with any model or bot; seeds 20-29 were used by the bots above),
setups 1.5b and 3b-listed (the two demos as configured) plus bot-reference and bot-nododge; every
run reported as above, plus hits_after_safe_move (a wrong "safe" label; expected 0 from brutes).
The fixes stay whatever the result, because they correct what the text tells the model; the
escapes are reported next to the 14/20 each demo had on seeds 0-19 before the fixes, with no
change in between.
    python -m bench.shooter_eval --setups 1.5b 3b-listed bot-reference bot-nododge --seeds 30 31 32 33 34 35 36 37 38 39

Everything above ran without ammo, before it existed; rerun it with --game classic.

Usage:
    python -m bench.shooter_eval                          # all setups, seeds 0-9, with ammo
    python -m bench.shooter_eval --game classic           # the same without ammo (as evaluated above)
    python -m bench.shooter_eval --setups random bot-reference --seeds 0 1
    python -m bench.shooter_eval --traces demo/output/shooter_eval_<time>   # resume
Traces go to demo/output/shooter_eval_<time>/ (rebuild a page with demo.shooter.capture --rebuild).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import statistics
import time
from pathlib import Path

from demo.dungeon.capture import OUT as DEMO_OUT
from demo.shooter import bots
from demo.shooter.capture import capture, to_jsonl, warm_up
from demo.shooter.world import CLASSIC, Dungeon, Rules
from system_one import load_config, make_engine
from system_one.config import REPO_ROOT

from .dungeon_eval import empty_device_cache
from .hwinfo import host_info

RESULTS = Path(__file__).parent / "results"
SETUPS = {  # name -> (config or None for a bot, Runner options or bot name)
    "1.5b": ("config/default.toml", {"order_debias": True}),
    "3b": ("config/lenovo-3b.toml", {"order_debias": True}),
    "1.5b-listed": ("config/default.toml", {"order_debias": False}),
    "3b-listed": ("config/lenovo-3b.toml", {"order_debias": False}),
    "random": ("config/mock.toml", {"order_debias": True}),
    "bot-reference": (None, "reference"),
    "bot-nododge": (None, "nododge"),
    "bot-noammo": (None, "noammo"),  # ammo: reloads only an empty gun, never walks to a box
}
GAMES = {"ammo": Rules(), "classic": CLASSIC}


def bot_run(seed: int, which: str, rules: Rules | None = None) -> dict:
    """A bot run, summarised with the same keys the model runs report where they apply."""
    d = Dungeon(seed, rules)
    hits = dry = boxes = 0
    key_tick = escape_tick = None
    while d.outcome is None:
        ev = d.step(*bots.reference(d, see_threats=(which != "nododge"), manage_ammo=(which != "noammo")))
        hits += sum(e["kind"] == "hit" for e in ev)
        boxes += sum(e["kind"] == "ammo" for e in ev)
        dry += d.rules.ammo and d.ammo_total() == 0
        key_tick = key_tick if key_tick is not None or not d.has_key else d.tick - 1
        escape_tick = d.tick - 1 if d.outcome == "escaped" else escape_tick
    out = {"outcome": d.outcome, "cause": d.cause, "ticks": d.tick, "key_tick": key_tick, "escape_tick": escape_tick,
           "rooms_seen": len(d.seen), "rooms_cleared": len(d.cleared), "kills": d.kills, "shots": d.shots,
           "hits": hits}
    if d.rules.ammo:
        out.update(reloads=d.reloads, ammo_boxes_picked=boxes, ammo_picked=d.ammo_picked, ticks_out_of_ammo=dry,
                   ammo_left=d.ammo_total())
    return out


def aggregate(runs: list[dict]) -> dict:
    s = [r["summary"] for r in runs]
    esc = [x["escape_tick"] for x in s if x["outcome"] == "escaped"]
    mean = lambda k: round(statistics.mean(x[k] for x in s), 1) if all(k in x for x in s) else None  # noqa: E731
    out = {
        "runs": len(s), "escaped": len(esc),
        "died_gunner": sum(x["outcome"] == "died" and x["cause"] == "gunner" for x in s),
        "died_brute": sum(x["outcome"] == "died" and x["cause"] == "brute" for x in s),
        "out_of_time": sum(x["outcome"] == "timeout" for x in s),
        "key_picked": sum(x["key_tick"] is not None for x in s),
        "escape_tick_median": statistics.median(esc) if esc else None,
        "ticks_mean": mean("ticks"), "rooms_seen_mean": mean("rooms_seen"), "rooms_cleared_mean": mean("rooms_cleared"),
        "kills_mean": mean("kills"), "shots_mean": mean("shots"), "hits_mean": mean("hits"),
    }
    if all("ticks_out_of_ammo" in x for x in s):
        out.update({
            "ran_out_of_ammo": sum(x["ticks_out_of_ammo"] > 0 for x in s),
            "not_escaped_after_running_out": sum(x["ticks_out_of_ammo"] > 0 and x["outcome"] != "escaped" for x in s),
            "reloads_mean": mean("reloads"), "ammo_boxes_picked_mean": mean("ammo_boxes_picked"),
            "ammo_left_mean": mean("ammo_left"),
        })
        if all("reloads_chosen" in x for x in s):
            chosen, offered = sum(x["reloads_chosen"] for x in s), sum(x["reload_offered"] for x in s)
            out.update({
                "reloads_chosen_mean": mean("reloads_chosen"),
                "reload_chosen_share": round(chosen / offered, 3) if offered else None,
                "reloads_chosen_in_fight": sum(x["reloads_chosen_in_fight"] for x in s),
                "ammo_goal_ticks_mean": mean("ammo_goal_ticks"),
            })
    if all("forward_ms_median" in x for x in s):
        shots = sum(x["shots"] for x in s)
        out.update({
            "shots_on_target_share": round(sum(x["shots_on_target"] for x in s) / shots, 3) if shots else None,
            "held_fire_share": round(sum(x["held_fire"] for x in s) / max(1, sum(x["shoot_decisions"] for x in s)), 3),
            "avoidable_risky_moves_mean": mean("avoidable_risky_moves"),
            "hits_after_safe_move": (sum(x["hits_after_safe_move"] for x in s)
                                     if all("hits_after_safe_move" in x for x in s) else None),
            "stuck_ticks_mean": mean("stuck_ticks"), "stuck_ticks_max": max(x["stuck_ticks"] for x in s),
            "forward_ms_median": round(statistics.median(x["forward_ms_median"] for x in s), 1),
            "forward_ms_p90_median": round(statistics.median(x["forward_ms_p90"] for x in s), 1),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--game", choices=list(GAMES), default="ammo",
                    help="ammo (default) or classic: without ammo, the game evaluated on seeds 0-39")
    ap.add_argument("--traces", default=None,
                    help="write traces here and reuse finished runs already in it (resume); default: a new directory")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    traces = Path(args.traces) if args.traces else DEMO_OUT / f"shooter_eval_{args.game}_{stamp}"
    traces.mkdir(parents=True, exist_ok=True)
    runs, backends = [], {}

    def report(name: str, seed: int, summary: dict, wall: float | None, reused: bool) -> None:
        runs.append({"setup": name, "seed": seed, "wall_s": wall, "reused_trace": reused, "summary": summary})
        keys = ("outcome", "cause", "ticks", "key_tick", "escape_tick", "rooms_seen", "rooms_cleared", "kills", "hits",
                "stuck_ticks", "forward_ms_median", "reloads", "ammo_boxes_picked", "ticks_out_of_ammo")
        print(json.dumps({"setup": name, "seed": seed, "reused": reused, **{k: summary.get(k) for k in keys}}), flush=True)

    todo: dict[str, list[tuple[str, int]]] = {}
    for name in args.setups:
        config, opts = SETUPS[name]
        for seed in args.seeds:
            if config is None:
                report(name, seed, bot_run(seed, opts, GAMES[args.game]), None, False)
                continue
            path = traces / f"{name}_seed{seed}.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            end = json.loads(lines[-1]) if lines else {}
            if end.get("type") == "end":  # finished earlier: runs are deterministic, reuse it
                backends[name] = json.loads(lines[0])["backend"]
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
            records = capture(engine, seed, GAMES[args.game], verbose=False, **SETUPS[name][1])
            (traces / f"{name}_seed{seed}.jsonl").write_text(to_jsonl(records), encoding="utf-8")
            report(name, seed, records[-1]["summary"], round(time.perf_counter() - t0, 1), False)
        engine = None  # drop the last reference so the weights can be freed before the next model
        empty_device_cache()
    runs.sort(key=lambda r: (args.setups.index(r["setup"]), r["seed"]))
    table = {name: aggregate([r for r in runs if r["setup"] == name]) for name in args.setups}
    for name, agg in table.items():
        print(name, json.dumps(agg))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"shooter_eval_{args.game}_{stamp}.json"
    path.write_text(json.dumps({"host": host_info(), "backends": backends, "args": vars(args),
                                "rules": asdict(GAMES[args.game]),
                                "setups": {k: [v[0], v[1]] for k, v in SETUPS.items()},
                                "traces": str(traces), "aggregate": table, "runs": runs}, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
