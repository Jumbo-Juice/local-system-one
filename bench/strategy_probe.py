"""Does the strategy tier react to low energy? Probe prompt variants on fixed demo-style states.

Truth (fixed before running): hazard adjacent -> avoid hazards; else energy < 35 -> find food;
else collect gems. Usage: python -m bench.strategy_probe --config config/lenovo-3b.toml
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from demo.grid.brain import GOALS
from system_one import Decision, load_config, make_engine

OUT = Path(__file__).parent / "results" / "strategy_probe"
ENERGIES = (95, 70, 50, 36, 34, 25, 15, 5, 0)
LOW = 35


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"


def state_v0(energy: int, hazard_adjacent: bool, food: int) -> str:
    e = f"LOW ({energy}/100)" if energy < LOW else f"ok ({energy}/100)"
    hz = "ADJACENT" if hazard_adjacent else "far"
    return f"Energy: {e}. Health: fine (100/100). Food: nearest {_steps(food)} away. " \
           f"Gems: 24 visible, nearest 2 steps away. Hazard: {hz}."


def state_v1(energy: int, hazard_adjacent: bool, food: int) -> str:
    if energy == 0:
        e = "EMPTY (0/100): the agent is starving and loses 5 health every tick"
    elif energy < LOW:
        e = f"LOW ({energy}/100): about {energy} ticks until starving, then 5 health lost per tick"
    else:
        e = f"ok ({energy}/100): about {energy} ticks left"
    hz = "ADJACENT: it hits for 25 health if you stay" if hazard_adjacent else "far"
    return f"Energy: {e}. Health: fine (100/100). Food: nearest {_steps(food)} away. " \
           f"Gems: 24 visible, nearest 2 steps away. Hazard: {hz}."


INSTR_V0 = "Which goal should the agent pursue now?"
INSTR_V2 = ("Which goal should the agent pursue now? Survive first: avoid an adjacent hazard, find food "
            "when energy is low; otherwise collect gems.")
VARIANTS = {"v0 current": (state_v0, INSTR_V0), "v1 consequences": (state_v1, INSTR_V0),
            "v2 consequences + priority": (state_v1, INSTR_V2)}


def items():
    out = []
    for e in ENERGIES:
        for hz in (False, True):
            for food in (4, 10):
                truth = "avoid hazards" if hz else ("find food" if e < LOW else "collect gems")
                out.append((e, hz, food, truth))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    engine = make_engine(load_config(args.config))
    report = {"backend": engine.backend.info(), "variants": {}}
    for name, (state_fn, instr) in VARIANTS.items():
        its = items()
        results = engine.decide_batch([Decision(instr, GOALS, state=state_fn(e, hz, f)) for e, hz, f, _ in its])
        rows = [{"energy": e, "hazard_adjacent": hz, "food": f, "truth": t, "choice": r.choice,
                 "p_food": r.probs[GOALS.index("find food")]} for (e, hz, f, t), r in zip(its, results)]
        by = {k: [r for r in rows if r["truth"] == k] for k in ("find food", "collect gems", "avoid hazards")}
        acc = {k: sum(r["choice"] == r["truth"] for r in v) / len(v) for k, v in by.items()}
        overall = sum(r["choice"] == r["truth"] for r in rows) / len(rows)
        report["variants"][name] = {"accuracy": overall, "by_truth": acc, "rows": rows}
        curve = " ".join(f"{e}:{next(r['p_food'] for r in rows if r['energy'] == e and not r['hazard_adjacent']):.2f}"
                         for e in ENERGIES)
        print(f"{name:28} acc={overall:.0%} food-cases={acc['find food']:.0%} gems-cases={acc['collect gems']:.0%} "
              f"hazard-cases={acc['avoid hazards']:.0%} | P(find food) by energy, hazard far: {curve} | "
              f"choices {dict(Counter(r['choice'] for r in rows))}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model = str(report["backend"].get("model", "")).split("/")[-1]
    path = OUT / f"strategy_probe_{model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
