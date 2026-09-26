"""Capture one dungeon run and write a replay page.

    python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
    python -m demo.dungeon.capture --config config/mock.toml --seed 0     # no model: random choices
    python -m demo.dungeon.capture --rebuild demo/output/<run>/trace.jsonl  # new viewer, same trace

Writes <out>/trace.jsonl (a header line, one line per tick, an end line) and <out>/replay.html:
demo/dungeon/viewer.html with the trace embedded, so it opens from disk with no server. The
viewer draws the recorded decisions; it does not recompute or re-decide anything.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from system_one import Decision, load_config, make_engine

from .brain import Runner
from .world import Dungeon, Rules

VIEWER = Path(__file__).with_name("viewer.html")
OUT = Path(__file__).resolve().parent.parent / "output"
_SLOT = re.compile(r'<script id="trace" type="application/x-ndjson">.*?</script>', re.S)


def warm_up(engine) -> None:
    """The first XPU pass compiles kernels (~3 s) and the first long padded batch allocates
    buffers. Doing both here keeps them out of the run (Observed; docs/research.md -> Smoother ticks)."""
    engine.decide_batch([Decision("warm-up", ("a", "b"))])
    warm = [Decision("warm-up", tuple("abcdefgh"), state="Rooms, doors, gems and enemies. " * (10 + 8 * i))
            for i in range(2)]
    engine.decide_batch(warm)
    engine.decide_batch(warm)  # the second call also compiles the prefix-cached path


def to_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in records)


def build_replay(trace_text: str, out: Path, template: Path = VIEWER) -> Path:
    """Embed a JSONL trace in the viewer. '<' only occurs inside JSON strings, so escaping it as
    \\u003c keeps the JSON identical and stops '</script>' in a prompt from ending the tag."""
    html = template.read_text(encoding="utf-8")
    if not _SLOT.search(html):
        raise ValueError(f"{template} has no trace slot")
    body = trace_text.replace("<", "\\u003c")
    out.write_text(_SLOT.sub(lambda _: f'<script id="trace" type="application/x-ndjson">{body}</script>', html, count=1),
                   encoding="utf-8")
    return out


def capture(engine, seed: int, rules: Rules | None = None, group_size: int = 8, plan_budget: int | None = 1,
            verbose: bool = True) -> list[dict]:
    runner = Runner(Dungeon(seed, rules), engine, group_size=group_size, plan_budget=plan_budget)
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
    ap.add_argument("--out", default=None, help="output directory (default: demo/output/dungeon_<model>_seed<N>)")
    ap.add_argument("--max-ticks", type=int, default=None, help="override the tick limit (default 400)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--plan-budget", type=int, default=1, help="planning decisions per tick; -1 = unlimited")
    ap.add_argument("--rebuild", default=None, help="only rebuild replay.html next to this trace.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.rebuild:
        trace = Path(args.rebuild)
        print("wrote", build_replay(trace.read_text(encoding="utf-8"), trace.with_name("replay.html")))
        return
    cfg = load_config(args.config)
    engine = make_engine(cfg)
    if cfg["backend"].get("kind") != "mock":
        warm_up(engine)
    rules = Rules(max_ticks=args.max_ticks) if args.max_ticks else None
    records = capture(engine, args.seed, rules, args.group_size, args.plan_budget if args.plan_budget >= 0 else None,
                      verbose=not args.quiet)
    model = str(engine.backend.info().get("model", engine.backend.info().get("kind"))).split("/")[-1]
    out = Path(args.out) if args.out else OUT / f"dungeon_{model}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    text = to_jsonl(records)
    (out / "trace.jsonl").write_text(text, encoding="utf-8")
    build_replay(text, out / "replay.html")
    print(json.dumps(records[-1]["summary"], indent=1))
    print("wrote", out / "trace.jsonl", f"({len(text) / 1e6:.2f} MB)")
    print("wrote", out / "replay.html")


if __name__ == "__main__":
    main()
