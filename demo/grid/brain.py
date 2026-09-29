"""Turns the world into text decisions for each agent and applies the answers.

Tiers per agent (tiered goals, see system_one/goals.py):
  strategy  every ~12 ticks  collect gems / find food / avoid hazards / explore
  target    every ~6 ticks   a concrete cell that serves the strategy (can be a large set:
                             every gem on the map -> tournament sampling)
  action    every tick       move north/south/east/west/stay towards the target

Implementation choices: the state names conditions in words ("Energy: LOW"). It gives the
target as a relative offset ("3 cells east and 2 cells north"), because the eval showed the
small model cannot do the arithmetic itself (docs/research.md -> Observed). "Flat" mode drops
the goal tiers so the two can be compared.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from system_one.goals import GoalStack, Tier, step_all

from ..common import _steps, base_move
from .world import MOVES, STEP, Agent, Cell, World

GOALS = ("collect gems", "find food", "avoid hazards", "explore")
REGIONS = ("north-west", "north", "north-east", "south-west", "south", "south-east")
_CELL = re.compile(r"\((\d+),(\d+)\)")


def annotate_moves(world: World, agent: Agent, target: Cell | None) -> list[str]:
    """Move options labelled with their outcome: walking distance to the target after the move,
    or what is in the way. The model still chooses. Precedent: S4's Doom demo gave its model an
    A*-computed waypoint bearing. Observed need: with plain options the 3B oscillated next to food
    it was told was "1 cell west" until it starved.
    """
    dist = world.distances(target) if target else {}
    others = {o.pos for o in world.agents if o is not agent}
    out = []
    for move in MOVES:
        dx, dy = STEP[move]
        c = (agent.pos[0] + dx, agent.pos[1] + dy)
        if not world.passable(c):
            label = "wall"
        elif move != "stay" and c in others:
            label = "blocked by another agent"
        elif c in world.hazards:
            label = "HAZARD: -25 health"
        elif target is None:
            label = "free"
        elif c == target:
            label = "on the target" if move == "stay" else "reach the target"
        elif c in dist:
            label = f"target: {_steps(dist[c])}"
        else:
            label = "target unreachable"
        out.append(f"{move} ({label})")
    return out


def _offset(dx: int, dy_north: int) -> str:
    parts = []
    if dx:
        parts.append(f"{abs(dx)} cells {'east' if dx > 0 else 'west'}")
    if dy_north:
        parts.append(f"{abs(dy_north)} cells {'north' if dy_north > 0 else 'south'}")
    return " and ".join(parts) or "0 cells away"


class Brain:
    """The goal stack and state text of one agent."""

    def __init__(self, world: World, agent: Agent, use_goals: bool = True,
                 strategy_every: int = 12, target_every: int = 6, aware: bool = True,
                 label_moves: bool = True):
        self.world, self.agent, self.use_goals, self.aware = world, agent, use_goals, aware
        self.label_moves = label_moves
        self._bands: tuple | None = None
        if use_goals:
            tiers = [
                Tier("strategy", "Which goal should the agent pursue now?", GOALS,
                     every=strategy_every, title="Strategic goal"),
                Tier("target", "Which target best serves the strategic goal?",
                     lambda goals: self.target_options(goals.get("strategy", "explore")),
                     every=target_every, title="Current target",
                     # The option's "N steps away" was measured when it was chosen and goes stale;
                     # Observed: the stale number made the 3B step away from food 1 cell west.
                     describe=(lambda c: c.split(", ")[0]) if aware else None),
                Tier("action", "Which move brings you closer to your current target? "
                     "Do not move into a wall or a hazard.", lambda goals: self.move_options(), every=1),
            ]
        else:
            tiers = [Tier("action", "Which move is best right now? Collect gems, eat food when "
                          "energy is low, and stay away from hazards.", MOVES, every=1)]
        self.stack = GoalStack(tiers)
        self._dist: dict[Cell, int] = {}

    # -- world -> text ----------------------------------------------------------------

    def _condition_bands(self) -> tuple:
        a = self.agent
        energy = 0 if a.energy == 0 else 1 if a.energy < 5 else 2 if a.energy < 15 else 3 if a.energy < 25             else 4 if a.energy < 35 else 5
        hz = self._nearest(self.world.hazards)
        return energy, a.health < 40, bool(hz and hz[1] <= 1)

    def refresh(self) -> None:
        """Recompute distances; drop a target that was reached or no longer exists.

        With ``aware``, a change in the agent's condition (energy band, low health, adjacent
        hazard) makes the strategy tier re-decide on this tick instead of waiting for its period.
        """
        self._dist = self.world.distances(self.agent.pos)
        if self.aware and self.use_goals:
            bands = self._condition_bands()
            if self._bands is not None and bands != self._bands:
                self.stack.invalidate("strategy")
            self._bands = bands
        cell = self.target_cell()
        if cell is None or "target" not in self.stack.current:
            return
        gone = self.stack.current["target"].startswith(("gem", "food")) and \
            cell not in self.world.gems and cell not in self.world.food
        if cell == self.agent.pos or gone:
            self.stack.invalidate("target")

    def move_options(self) -> list[str]:
        if self.aware and self.label_moves:
            return annotate_moves(self.world, self.agent, self.target_cell())
        return list(MOVES)

    def target_cell(self) -> Cell | None:
        m = _CELL.search(self.stack.current.get("target", ""))
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _nearest(self, cells) -> tuple[Cell, int] | None:
        reach = [(c, self._dist[c]) for c in cells if c in self._dist]
        return min(reach, key=lambda x: x[1]) if reach else None

    def _status(self, detailed: bool = False) -> str:
        """``detailed`` spells out consequences. Observed: without them the 3B chose "collect gems"
        in all 36 probe states, even at energy 0 (bench/strategy_probe.py)."""
        a = self.agent
        if detailed and a.energy == 0:
            energy = "EMPTY (0/100): the agent is starving and loses 5 health every tick"
        elif detailed and a.energy < 35:
            energy = f"LOW ({a.energy}/100): about {a.energy} ticks until starving, then 5 health lost per tick"
        elif detailed:
            energy = f"ok ({a.energy}/100): about {a.energy} ticks left"
        else:
            energy = f"LOW ({a.energy}/100)" if a.energy < 35 else f"ok ({a.energy}/100)"
        health = f"LOW ({a.health}/100)" if a.health < 40 else f"fine ({a.health}/100)"
        return f"Energy: {energy}. Health: {health}."

    def strategy_state(self) -> str:
        food, gem, hz = (self._nearest(self.world.food), self._nearest(self.world.gems),
                         self._nearest(self.world.hazards))
        food_t = f"nearest {_steps(food[1])} away" if food else "none reachable"
        gem_t = f"{len(self.world.gems)} visible, nearest {_steps(gem[1])} away" if gem else "none visible"
        if hz and hz[1] <= 1:
            hz_t = "ADJACENT: it hits for 25 health if it reaches you" if self.aware else "ADJACENT"
        elif hz and hz[1] <= 3:
            hz_t = f"close, {_steps(hz[1])} away"
        else:
            hz_t = "far"
        return f"{self._status(detailed=self.aware)} Food: {food_t}. Gems: {gem_t}. Hazard: {hz_t}."

    def _neighbours(self) -> str:
        return "Neighbouring cells: " + ", ".join(f"{d} {v}" for d, v in self.world.neighbours(self.agent).items()) + "."

    def _rel(self, cell: Cell) -> str:
        x, y = self.agent.pos
        return _offset(cell[0] - x, y - cell[1])

    def action_state(self) -> str:
        x, y = self.agent.pos
        head = f"You are at ({x},{y}). North is up."
        if not self.use_goals:
            lines = [head]
            for name, cells in (("gem", self.world.gems), ("food", self.world.food), ("hazard", self.world.hazards)):
                n = self._nearest(cells)
                lines.append(f"Nearest {name}: " + (f"{self._rel(n[0])} of you." if n else "none."))
            return " ".join(lines + [self._neighbours(), self._status()])
        cell = self.target_cell()
        if cell is None:
            where = "You have no target yet."
        elif cell == self.agent.pos:
            where = "You are on your target."
        else:
            where = f"Your target is {self._rel(cell)} of you."
        return " ".join([head, where, self._neighbours(), self._status()])

    def state_for(self, tier: Tier) -> str:
        return self.strategy_state() if tier.name in ("strategy", "target") else self.action_state()

    def target_options(self, strategy: str) -> list[str]:
        """Candidate targets for a strategy, in map order (not sorted by distance)."""
        d = self._dist or self.world.distances(self.agent.pos)
        w = self.world
        if strategy in ("collect gems", "find food"):
            kind, cells = ("gem", w.gems) if strategy == "collect gems" else ("food", w.food)
            opts = [f"{kind} at ({x},{y}), {_steps(d[(x, y)])} away" for x, y in sorted(cells, key=lambda c: (c[1], c[0]))
                    if (x, y) in d]
            if opts:
                return opts
        if strategy == "avoid hazards" and w.hazards:
            spots = []
            for c, steps in d.items():
                if 2 <= steps <= 6:
                    h = min(abs(c[0] - hx) + abs(c[1] - hy) for hx, hy in w.hazards)
                    spots.append((h, steps, c))
            spots.sort(key=lambda s: (-s[0], s[1], s[2]))
            chosen = sorted(spots[:6], key=lambda s: (s[2][1], s[2][0]))
            if chosen:
                return [f"safe spot at ({c[0]},{c[1]}), {_steps(steps)} away, {_steps(h)} from the nearest hazard"
                        for h, steps, c in chosen]
        out = []
        for i, name in enumerate(REGIONS):
            cx = (i % 3) * w.width // 3 + w.width // 6
            cy = (i // 3) * w.height // 2 + w.height // 4
            cell = min(d, key=lambda c: abs(c[0] - cx) + abs(c[1] - cy))
            out.append(f"{name} area around ({cell[0]},{cell[1]}), {_steps(d[cell])} away")
        return out


@dataclass
class TickReport:
    tick: int
    decisions: int
    forward_s: float
    total_s: float
    prompt_tokens: int
    events: dict
    updates: list = field(default_factory=list)  # (agent id, tier name, DecisionResult, tournament or None)


class Controller:
    """Runs the world: each tick = ONE batched engine call for every agent's due decisions."""

    def __init__(self, world: World, engine, use_goals: bool = True, group_size: int = 8,
                 plan_budget: int | None = None, **brain_options):
        self.world, self.engine, self.group_size, self.plan_budget = world, engine, group_size, plan_budget
        self.brains = [Brain(world, a, use_goals, **brain_options) for a in world.agents]

    def tick(self) -> TickReport:
        t0 = time.perf_counter()
        for b in self.brains:
            b.refresh()
        agents = [(b.stack, b.state_for) for b in self.brains]
        updates = step_all(self.engine, agents, self.world.tick, group_size=self.group_size,
                           plan_budget=self.plan_budget)
        stats = dict(self.engine.last_stats) if updates else {}
        moves = {b.agent.id: base_move(b.stack.current.get("action", "stay")) for b in self.brains}
        events = self.world.step(moves)
        return TickReport(
            tick=self.world.tick - 1, decisions=len(updates), forward_s=stats.get("forward_s", 0.0),
            total_s=time.perf_counter() - t0, prompt_tokens=stats.get("prompt_tokens", 0), events=events,
            updates=[(self.brains[i].agent.id, tier.name, r, tour) for i, tier, r, tour in updates],
        )
