"""Do move decisions actually approach the target? Probe how the target is shown to the move tier.

The demo gives the move tier its goal context. The target's option text carries a distance measured
when it was chosen ("food at (7,0), 1 step away"). That distance goes stale as the agent moves.
Here it is made stale on purpose (true distance +-2), as in real play.
Metric (fixed before running): share of decisions whose move shortens the walking distance (BFS,
around walls) to the target. Usage: python -m bench.action_probe --config config/lenovo-3b.toml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from demo.brain import annotate_moves
from demo.world import MOVES, STEP, World
from system_one import Decision, load_config, make_engine

OUT = Path(__file__).parent / "results"
INSTR = "Which move brings you closer to your current target? Do not move into a wall or a hazard."


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"


def cases(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        w = World(seed=rng.randrange(1000), n_agents=1, n_hazards=0)
        a = w.agents[0]
        dist = w.distances(a.pos)
        target = rng.choice([c for c, d in dist.items() if 1 <= d <= 8])
        stated = max(1, dist[target] + rng.choice([-2, -1, 0, 1, 2]))
        out.append((w, target, stated))
    return out


def state_text(w: World, target) -> str:
    from demo.brain import Brain

    b = Brain(w, w.agents[0])
    b.stack.apply(b.stack.tier("target"), f"food at ({target[0]},{target[1]})", 0)
    b.refresh()
    return b.action_state()


VARIANTS = {
    "v0 context: target with stale distance": lambda t, n: f"Strategic goal: find food\nCurrent target: food at ({t[0]},{t[1]}), {_steps(n)} away",
    "v1 context: target without distance": lambda t, n: f"Strategic goal: find food\nCurrent target: food at ({t[0]},{t[1]})",
    "v2 context: target kind only": lambda t, n: "Strategic goal: find food\nCurrent target: food",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    engine = make_engine(load_config(args.config))
    report = {"backend": engine.backend.info(), "variants": {}}
    runs = [(name, ctx, False) for name, ctx in VARIANTS.items()]
    runs.append(("v3 = v2 + options labelled with outcomes", VARIANTS["v2 context: target kind only"], True))
    data = cases(args.n)
    for name, ctx, labelled in runs:
        decisions = []
        for w, target, stated in data:
            opts = annotate_moves(w, w.agents[0], target) if labelled else MOVES
            decisions.append(Decision(INSTR, tuple(opts), state=state_text(w, target), context=ctx(target, stated)))
        results = engine.decide_batch(decisions)
        good, rows = 0, []
        for (w, target, stated), r in zip(data, results):
            move = r.choice.split(" (")[0]
            ax, ay = w.agents[0].pos
            dx, dy = STEP[move]
            nxt = (ax + dx, ay + dy) if w.passable((ax + dx, ay + dy)) else (ax, ay)
            before, after = w.distances((ax, ay))[target], w.distances(nxt).get(target, 99)
            good += after < before
            rows.append({"pos": (ax, ay), "target": target, "stated": stated, "choice": r.choice,
                         "before": before, "after": after, "p": r.prob})
        report["variants"][name] = {"approach_rate": good / len(rows), "rows": rows}
        print(f"{name:42} moves that approach the target: {good}/{len(rows)} = {good / len(rows):.0%}", flush=True)
    OUT.mkdir(exist_ok=True)
    model = str(report["backend"].get("model", "")).split("/")[-1]
    path = OUT / f"action_probe_{model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
