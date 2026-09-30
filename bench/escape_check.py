"""The 20-run escape check across both game demos: each model plays each game on five seeds.

Asked for on 2026-09-30, after the dungeon was rebuilt and both games got the dash: "rerun 20
iterations; done when 80% of runs escape". Fixed in this file before its first run:
- Runs: {dungeon, shooter} x {Qwen2.5-1.5B (config/default.toml), Qwen2.5-3B (config/lenovo-3b.toml)}
  x seeds 0-4 = 20 runs. Each model plays with the settings of its config ([shooter] order_debias
  and fire_head, [dungeon] order_debias); the shooter plays its default game (ammo and the dash).
- Done when at least 16 of the 20 runs escape (80%).
- Seeds 0-4 are development data for this goal: when the check fails, the traces are read, the
  games' texts or rules are fixed, and the check runs again. Every attempt is reported in
  docs/research.md -> Dash and loops, failed ones included.
- Once seeds 0-4 pass, the same check runs once on seeds 50-54 (never run with any model or bot in
  either game) with no change in between, and is reported whatever the result.

Usage:
    python -m bench.escape_check                       # seeds 0-4
    python -m bench.escape_check --seeds 50 51 52 53 54
    python -m bench.escape_check --resume 20260930-120000
Traces go to each game's pool: runs/<game>/<check id>_check_<model>_seed<N>.jsonl (python -m viewer).
Results: bench/results/escape_check/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from demo import runs as pool
from demo.common import warm_up
from demo.dungeon import capture as dungeon
from demo.shooter import capture as shooter
from system_one import load_config, make_engine
from system_one.config import REPO_ROOT

from .dungeon_eval import empty_device_cache
from .hwinfo import host_info

RESULTS = Path(__file__).parent / "results" / "escape_check"
MODELS = {"1.5b": "config/default.toml", "3b": "config/lenovo-3b.toml"}
GAMES = ("dungeon", "shooter")
GOAL = 0.8


def play(game: str, engine, cfg: dict, seed: int) -> list[dict]:
    if game == "dungeon":
        return dungeon.capture(engine, seed, verbose=False, order_debias=dungeon.order_debias(cfg))
    return shooter.capture(engine, seed, shooter.GAMES["dash"], verbose=False,
                           order_debias=shooter.order_debias(cfg), fire_head=shooter.fire_head(cfg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--games", nargs="+", default=list(GAMES), choices=list(GAMES))
    ap.add_argument("--resume", default=None, metavar="CHECK_ID",
                    help="reuse the finished runs of this check (the time at the start of its trace names)")
    args = ap.parse_args()
    check_id = args.resume or pool.stamp()
    trace = lambda game, model, seed: pool.trace_path(game, f"check_{model}_seed{seed}", check_id)  # noqa: E731
    runs = []

    def report(game: str, model: str, seed: int, summary: dict, wall: float | None, reused: bool) -> None:
        runs.append({"game": game, "model": model, "seed": seed, "wall_s": wall, "reused_trace": reused,
                     "summary": summary})
        keys = ("outcome", "cause", "ticks", "key_tick", "rooms_seen", "hits", "dashes", "stuck_ticks", "loop_ticks",
                "forward_ms_median")
        print(json.dumps({"game": game, "model": model, "seed": seed, "reused": reused,
                          **{k: summary.get(k) for k in keys}}), flush=True)

    for model in args.models:
        jobs = []
        for game in args.games:
            for seed in args.seeds:
                end = pool.finished(trace(game, model, seed))
                if end:  # finished earlier: runs are deterministic, reuse it
                    report(game, model, seed, end["summary"], None, True)
                else:
                    jobs.append((game, seed))
        if not jobs:
            continue
        cfg = load_config(REPO_ROOT / MODELS[model])
        engine = make_engine(cfg)
        warm_up(engine)
        for game, seed in jobs:
            t0 = time.perf_counter()
            records = play(game, engine, cfg, seed)
            trace(game, model, seed).write_text(pool.to_jsonl(records), encoding="utf-8")
            report(game, model, seed, records[-1]["summary"], round(time.perf_counter() - t0, 1), False)
        engine = None  # drop the last reference so the weights can be freed before the next model
        empty_device_cache()
    runs.sort(key=lambda r: (r["game"], r["model"], r["seed"]))
    escaped = sum(r["summary"]["outcome"] == "escaped" for r in runs)
    cells = {f"{g} {m}": sum(r["summary"]["outcome"] == "escaped" for r in runs if r["game"] == g and r["model"] == m)
             for g in args.games for m in args.models}
    verdict = {"escaped": escaped, "runs": len(runs), "share": round(escaped / len(runs), 3) if runs else None,
               "goal": GOAL, "done": bool(runs) and escaped >= GOAL * len(runs), "per_game_and_model": cells}
    print(json.dumps(verdict))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"escape_check_{check_id}.json"
    path.write_text(json.dumps({"host": host_info(), "args": vars(args), "check_id": check_id,
                                "traces": f"runs/<game>/{check_id}_check_*.jsonl", "verdict": verdict, "runs": runs},
                               indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
