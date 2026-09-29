"""Capture one shooter run into the shooter pool.

    python -m demo.shooter.capture --config config/default.toml --seed 0     # Qwen2.5-1.5B
    python -m demo.shooter.capture --config config/lenovo-3b.toml --seed 0   # Qwen2.5-3B
    python -m demo.shooter.capture --config config/mock.toml --seed 0        # no model: random choices
    python -m demo.shooter.capture --classic --seed 0                        # without ammo (as evaluated on seeds 0-39)

Writes runs/shooter/<time>_<model>_seed<N>.jsonl (a header line, one line per tick, an end line).
Watch it with the Master Viewer: python -m viewer <that file>, or python -m viewer for the pools.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from system_one import load_config, make_engine

from ..common import warm_up
from ..runs import model_tag, to_jsonl, write_run
from .brain import Runner
from .world import CLASSIC, Dungeon, Rules


def order_debias(cfg: dict, flag: bool | None = None) -> bool:
    """Order averaging for this model: the command-line flag, else ``[shooter] order_debias`` in the
    config, else on. The 1.5B needs it; the 3B escaped more often without it (docs/research.md ->
    Shooter demo: pre-registered seeds 0-9 and the replication on seeds 10-19)."""
    return flag if flag is not None else bool(cfg.get("shooter", {}).get("order_debias", True))


def fire_head(cfg: dict, flag: bool | None = None, classic: bool = False) -> bool:
    """One "shoot" option plus an aim head, instead of one shoot option per enemy: the command-line
    flag, else off for the classic game (evaluated with one option per enemy), else ``[shooter]
    fire_head`` in the config, else off. On for the 1.5B, which otherwise held fire (docs/research.md)."""
    if flag is not None:
        return flag
    return False if classic else bool(cfg.get("shooter", {}).get("fire_head", False))


def capture(engine, seed: int, rules: Rules | None = None, group_size: int = 8, plan_budget: int | None = 1,
            verbose: bool = True, **brain_options) -> list[dict]:
    runner = Runner(Dungeon(seed, rules), engine, group_size=group_size, plan_budget=plan_budget, **brain_options)
    records = [runner.header()]

    def show(rec: dict) -> None:
        if not verbose:
            return
        plans = " ".join(f"{x['tier']}={x['options'][x['choice']][:44]!r}({x['probs'][x['choice']]:.2f})"
                         for x in rec["decisions"] if x["kind"] == "plan")
        ev = ",".join(e["kind"] for e in rec["events"] if e["kind"] not in ("impact",))
        a = rec["world"]["agent"]
        shot = "reload" if rec.get("reload") else "" if rec["shoot"] is None else f"shoot#{rec['shoot']}"
        ammo = f"ammo {a['loaded']}+{a['reserve']:<2} " if "loaded" in a else ""
        print(f"tick {rec['tick']:3} {rec['batch']['forward_ms']:6.0f} ms  hp {a['health']:3} {ammo}"
              f"{rec['move']:<10} {shot:<8} {plans} {ev}", flush=True)

    end = runner.run(on_tick=show)
    return records + runner.records + [end]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the trace to this file instead of the shooter pool")
    ap.add_argument("--max-ticks", type=int, default=None, help="override the tick limit (default 400)")
    ap.add_argument("--classic", action="store_true", help="the game without ammo, as evaluated on seeds 0-39")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--plan-budget", type=int, default=1, help="planning decisions per tick; -1 = unlimited")
    ap.add_argument("--order-debias", action=argparse.BooleanOptionalAction, default=None,
                    help="read each decision in two option orders and average (default: [shooter] order_debias "
                         "in the config, else on)")
    ap.add_argument("--fire-head", action=argparse.BooleanOptionalAction, default=None,
                    help="one 'shoot' option and an aim head that picks the enemy (default: [shooter] fire_head in "
                         "the config; always off with --classic)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine = make_engine(cfg)
    if cfg["backend"].get("kind") != "mock":
        warm_up(engine)
    rules = CLASSIC if args.classic else Rules()
    if args.max_ticks:
        rules = replace(rules, max_ticks=args.max_ticks)
    records = capture(engine, args.seed, rules, args.group_size, args.plan_budget if args.plan_budget >= 0 else None,
                      verbose=not args.quiet, order_debias=order_debias(cfg, args.order_debias),
                      fire_head=fire_head(cfg, args.fire_head, args.classic))
    label = f"{'classic_' if args.classic else ''}{model_tag(engine.backend.info())}_seed{args.seed}"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_jsonl(records), encoding="utf-8")
    else:
        path = write_run("shooter", label, records)
    print(json.dumps(records[-1]["summary"], indent=1))
    shown = os.path.relpath(path)
    print("wrote", shown, f"({path.stat().st_size / 1e6:.2f} MB)")
    print("watch it: python -m viewer", shown)


if __name__ == "__main__":
    main()
