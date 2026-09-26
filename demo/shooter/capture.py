"""Capture one shooter run and write a replay page.

    python -m demo.shooter.capture --config config/default.toml --seed 0     # Qwen2.5-1.5B
    python -m demo.shooter.capture --config config/lenovo-3b.toml --seed 0   # Qwen2.5-3B
    python -m demo.shooter.capture --config config/mock.toml --seed 0        # no model: random choices
    python -m demo.shooter.capture --rebuild demo/output/<run>/trace.jsonl   # new viewer, same trace

Writes <out>/trace.jsonl (a header line, one line per tick, an end line) and <out>/replay.html:
demo/shooter/viewer.html with the trace embedded, so it opens from disk with no server. The
viewer draws the recorded decisions; it does not recompute or re-decide anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from system_one import load_config, make_engine

from ..dungeon.capture import OUT, build_replay, to_jsonl, warm_up
from .brain import Runner
from .world import Dungeon, Rules

VIEWER = Path(__file__).with_name("viewer.html")


def order_debias(cfg: dict, flag: bool | None = None) -> bool:
    """Order averaging for this model: the command-line flag, else ``[shooter] order_debias`` in the
    config, else on. The 1.5B needs it; the 3B escaped more often without it (docs/research.md ->
    Shooter demo: pre-registered seeds 0-9 and the replication on seeds 10-19)."""
    return flag if flag is not None else bool(cfg.get("shooter", {}).get("order_debias", True))


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
        shot = "" if rec["shoot"] is None else f"shoot#{rec['shoot']}"
        print(f"tick {rec['tick']:3} {rec['batch']['forward_ms']:6.0f} ms  hp {a['health']:3} "
              f"{rec['move']:<10} {shot:<8} {plans} {ev}", flush=True)

    end = runner.run(on_tick=show)
    return records + runner.records + [end]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output directory (default: demo/output/shooter_<model>_seed<N>)")
    ap.add_argument("--max-ticks", type=int, default=None, help="override the tick limit (default 400)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--plan-budget", type=int, default=1, help="planning decisions per tick; -1 = unlimited")
    ap.add_argument("--order-debias", action=argparse.BooleanOptionalAction, default=None,
                    help="read each decision in two option orders and average (default: [shooter] order_debias "
                         "in the config, else on)")
    ap.add_argument("--rebuild", default=None, help="only rebuild replay.html next to this trace.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.rebuild:
        trace = Path(args.rebuild)
        page = trace.with_name("replay.html") if trace.name == "trace.jsonl" else trace.with_suffix(".html")
        print("wrote", build_replay(trace.read_text(encoding="utf-8"), page, VIEWER))
        return
    cfg = load_config(args.config)
    engine = make_engine(cfg)
    if cfg["backend"].get("kind") != "mock":
        warm_up(engine)
    rules = Rules(max_ticks=args.max_ticks) if args.max_ticks else None
    records = capture(engine, args.seed, rules, args.group_size, args.plan_budget if args.plan_budget >= 0 else None,
                      verbose=not args.quiet, order_debias=order_debias(cfg, args.order_debias))
    model = str(engine.backend.info().get("model", engine.backend.info().get("kind"))).split("/")[-1]
    out = Path(args.out) if args.out else OUT / f"shooter_{model}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    text = to_jsonl(records)
    (out / "trace.jsonl").write_text(text, encoding="utf-8")
    if VIEWER.exists():
        build_replay(text, out / "replay.html", VIEWER)
        print("wrote", out / "replay.html")
    print(json.dumps(records[-1]["summary"], indent=1))
    print("wrote", out / "trace.jsonl", f"({len(text) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
