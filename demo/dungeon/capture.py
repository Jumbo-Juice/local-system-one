"""Capture one dungeon run into the dungeon pool.

    python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
    python -m demo.dungeon.capture --config config/mock.toml --seed 0     # no model: random choices

Writes runs/dungeon/<time>_<model>_seed<N>.jsonl (a header line, one line per tick, an end line).
Watch it with the Master Viewer: python -m viewer <that file>, or python -m viewer for the pools.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from system_one import load_config, make_engine

from ..common import warm_up
from ..runs import model_tag, stamp, to_jsonl, write_run
from .brain import Runner
from .world import Dungeon, Rules


def capture(engine, seed: int, rules: Rules | None = None, group_size: int = 8, plan_budget: int | None = 1,
            verbose: bool = True, **brain_options) -> list[dict]:
    runner = Runner(Dungeon(seed, rules), engine, group_size=group_size, plan_budget=plan_budget, **brain_options)
    records = [runner.header()]

    def show(rec: dict) -> None:
        if not verbose:
            return
        plans = " ".join(f"{x['tier']}={x['options'][x['choice']][:40]!r}({x['probs'][x['choice']]:.2f})"
                         for x in rec["decisions"] if x["kind"] == "plan")
        ev = ",".join(e["kind"] for e in rec["events"])
        a = rec["world"]["agent"]
        print(f"tick {rec['tick']:3} {rec['batch']['forward_ms']:6.0f} ms  hp {a['health']:3} en {a['energy']:3} "
              f"{rec['move']:<10} {plans} {ev}", flush=True)

    end = runner.run(on_tick=show)
    records += runner.records + [end]
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the trace to this file instead of the dungeon pool")
    ap.add_argument("--max-ticks", type=int, default=None, help="override the tick limit (default 400)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--plan-budget", type=int, default=1, help="planning decisions per tick; -1 = unlimited")
    ap.add_argument("--label-style", choices=("closer", "steps"), default="closer",
                    help="move-outcome wording (see DungeonBrain)")
    ap.add_argument("--enemy-aware", action=argparse.BooleanOptionalAction, default=False,
                    help="enemy consequences in move labels and safer flee targets (see DungeonBrain)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine = make_engine(cfg)
    if cfg["backend"].get("kind") != "mock":
        warm_up(engine)
    rules = Rules(max_ticks=args.max_ticks) if args.max_ticks else None
    started = stamp()  # the run's name carries its start time
    records = capture(engine, args.seed, rules, args.group_size, args.plan_budget if args.plan_budget >= 0 else None,
                      verbose=not args.quiet, label_style=args.label_style,
                      enemy_aware=args.enemy_aware)
    variant = ("enemy-aware_" if args.enemy_aware else "") + ("steps_" if args.label_style == "steps" else "")
    label = f"{variant}{model_tag(engine.backend.info())}_seed{args.seed}"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_jsonl(records), encoding="utf-8")
    else:
        path = write_run("dungeon", label, records, when=started)
    print(json.dumps(records[-1]["summary"], indent=1))
    shown = os.path.relpath(path)
    print("wrote", shown, f"({path.stat().st_size / 1e6:.2f} MB)")
    print("watch it: python -m viewer", shown)


if __name__ == "__main__":
    main()
