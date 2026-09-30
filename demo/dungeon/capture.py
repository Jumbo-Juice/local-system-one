"""Capture one dungeon run into the dungeon pool.

    python -m demo.dungeon.capture --config config/default.toml --seed 0     # Qwen2.5-1.5B
    python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0   # Qwen2.5-3B
    python -m demo.dungeon.capture --config config/mock.toml --seed 0        # no model: random choices

Writes runs/dungeon/<time>_<model>_seed<N>.jsonl (a header line, one line per tick, an end line).
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
from ..runs import model_tag, stamp, to_jsonl, write_run
from .brain import Runner
from .world import Dungeon, Rules


def order_debias(cfg: dict, flag: bool | None = None) -> bool:
    """Order averaging for this model: the command-line flag, else ``[dungeon] order_debias`` in the
    config, else ``[shooter] order_debias`` (the setting each model was evaluated with there), else on."""
    if flag is not None:
        return flag
    for section in ("dungeon", "shooter"):
        if "order_debias" in cfg.get(section, {}):
            return bool(cfg[section]["order_debias"])
    return True


def capture(engine, seed: int, rules: Rules | None = None, group_size: int = 8, plan_budget: int | None = 1,
            verbose: bool = True, **brain_options) -> list[dict]:
    runner = Runner(Dungeon(seed, rules), engine, group_size=group_size, plan_budget=plan_budget, **brain_options)
    records = [runner.header()]

    def show(rec: dict) -> None:
        if not verbose:
            return
        plans = " ".join(f"{x['tier']}={x['options'][x['choice']][:44]!r}({x['probs'][x['choice']]:.2f})"
                         for x in rec["decisions"] if x["kind"] == "plan")
        ev = ",".join(e["kind"] for e in rec["events"])
        a = rec["world"]["agent"]
        print(f"tick {rec['tick']:3} {rec['batch']['forward_ms']:6.0f} ms  hp {a['health']:3} "
              f"{rec['move']:<10} {plans} {ev}", flush=True)

    end = runner.run(on_tick=show)
    return records + runner.records + [end]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the trace to this file instead of the dungeon pool")
    ap.add_argument("--max-ticks", type=int, default=None, help="override the tick limit (default 400)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--plan-budget", type=int, default=1, help="planning decisions per tick; -1 = unlimited")
    ap.add_argument("--order-debias", action=argparse.BooleanOptionalAction, default=None,
                    help="read each decision in two option orders and average (default: [dungeon] order_debias "
                         "in the config, else [shooter] order_debias, else on)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine = make_engine(cfg)
    if cfg["backend"].get("kind") != "mock":
        warm_up(engine)
    rules = replace(Rules(), max_ticks=args.max_ticks) if args.max_ticks else Rules()
    started = stamp()  # the run's name carries its start time
    records = capture(engine, args.seed, rules, args.group_size, args.plan_budget if args.plan_budget >= 0 else None,
                      verbose=not args.quiet, order_debias=order_debias(cfg, args.order_debias))
    label = f"{model_tag(engine.backend.info())}_seed{args.seed}"
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
