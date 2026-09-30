"""Turns the dungeon into text decisions for one agent, and runs it tick by tick.

Tiers (tiered goals, system_one/goals.py), all due decisions in ONE batched forward pass per tick:
  strategy  when the agent's situation changes, or it stalls: a goal from GOALS. Only goals that
            are possible now are offered (no "get the key" before it has been seen). Once the agent
            carries the key and knows the way to the exit, exploring and gems are no longer offered:
            the shooter's 3B stayed on an explore target with the key in hand until time ran out
            (seed 37, docs/research.md).
  target    when the goal changes, the target is reached, gone or stalls: a concrete cell.
  move      control head, every tick: step or dash north/south/east/west, or stay, each labelled
            with its outcome.

Built on 2026-09-30 with the shooter's lessons from the start (docs/lessons-learned.md):
- "closer/farther" is measured along routes that avoid the cells a ghoul can hit next tick, so the
  move the model likes best (closer) is also the safe one; a move into a ghoul's reach is labelled
  with the damage only.
- No trap options: walls and ghouls are not offered as moves, a one-cell dash is not offered,
  "stay" is left out while a safe move gets closer to the target, and a move into a ghoul's reach is
  left out while any option is safe (in the shooter's first dash runs every hit the 3B took came
  from such a move while a safe one was on offer).
- Targets and the strategy are held until reached, gone or changed by the situation, not re-decided
  every few ticks (re-deciding from alternating positions made the shooter's 3B alternate between
  two targets; seed 1). If the agent gets no closer for STUCK ticks, the target is re-decided (and
  every second time the strategy); after LOOP ticks without a new cell or any progress, both are.

The model makes every choice. The code only computes what the text says: walking distances over
the cells the agent knows, where the ghouls can reach. A tier with a single possible option is
committed without a model call.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter

from system_one.goals import ONLY_OPTION, GoalStack, Tier, step_all

from ..common import LOOP, IdleTracker, _steps, base_move, is_dash, streak_ticks
from .bots import frontier, threat_cells
from .world import DASHES, DIRS, MOVES, STEP, Cell, Dungeon

STANDING_ORDER = ("Find the key, then leave through the exit alive. Ghouls guard the rooms and cannot be fought: "
                  "keep out of their reach. Gems are a bonus.")
GOALS = ("explore", "get the key", "go to the exit", "collect gems", "drink a health potion")
MOVE_QUESTION = "Which move is best? Get closer to your target, but never step next to a ghoul."
_CELL = re.compile(r"\((\d+),(\d+)\)")


def _offset(dx: int, dy_north: int) -> str:
    """'3 cells east and 1 cell north' (relative position; the model cannot do the arithmetic)."""
    parts = [f"{abs(v)} {'cell' if abs(v) == 1 else 'cells'} {pos if v > 0 else neg}"
             for v, pos, neg in ((dx, "east", "west"), (dy_north, "north", "south")) if v]
    return " and ".join(parts) or "here"


def cell_of(text: str | None) -> Cell | None:
    m = _CELL.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def drop_distance(option: str) -> str:
    """How lower tiers see a target: the part before the first comma (distances go stale)."""
    return option.split(", ")[0]


def _ticks(n: int) -> str:
    return f"{n} tick" if n == 1 else f"{n} ticks"


def _hits(n: int) -> str:
    return f"{n} hit" if n == 1 else f"{n} hits"


class DungeonBrain:
    """Goal stack and state texts of the one agent."""

    def __init__(self, dungeon: Dungeon, strategy_every: int = 12, target_every: int = 6):
        self.d = dungeon
        self.stack = GoalStack([
            Tier("strategy", "Which goal should the agent pursue now?", lambda goals: self.goal_options(),
                 every=strategy_every, title="Strategic goal"),
            Tier("target", "Which target best serves the strategic goal?",
                 lambda goals: self.target_options(goals.get("strategy", "explore")),
                 every=target_every, title="Current target", describe=drop_distance),
            Tier("move", MOVE_QUESTION, lambda goals: self.move_options(), every=1),
        ])
        for name in ("strategy", "target"):  # held: re-decided when needed, not on the period
            self.stack.hold(name)
        self._situation: tuple | None = None
        self.refresh()

    # -- per-tick bookkeeping ------------------------------------------------------------

    def refresh(self) -> None:
        """Recompute what the agent knows. Re-plan the strategy when the situation changes or its
        goal became impossible; re-plan the target when it was reached or is gone."""
        d = self.d
        self._known = d.known_cells()
        self._dist = d.distances(d.pos, within=self._known)
        self._threat = threat_cells(d)
        situation = self._situation_now()
        if self._situation is not None and situation != self._situation:
            self.stack.invalidate("strategy")
        self._situation = situation
        strategy = self.stack.current.get("strategy")
        if strategy is not None and strategy not in self.goal_options():
            self.stack.invalidate("strategy")
        target = self.stack.current.get("target")
        if target is not None and self._target_done(target):
            self.stack.invalidate("target")

    def replan(self, strategy: bool = False) -> None:
        """Re-decide the target (and the strategy) next tick: the agent stopped making progress."""
        self.stack.invalidate("target")
        if strategy:
            self.stack.invalidate("strategy")

    def _situation_now(self) -> tuple:
        d = self.d
        hits = min(3, math.ceil(d.health / d.rules.ghoul_damage))
        return (hits, d.has_key, len(d.seen), d.key is not None and d.key in self._dist, d.exit in self._dist,
                bool(self.chasers()))

    def chasers(self) -> list:
        """Awake ghouls that can hit the agent soon: in the room (or doorway) it stands in, not tired."""
        rid = self.d.area.get(self.d.pos)
        return [g for g in self.d.known_ghouls() if g.home == rid and g.awake and not g.tired]

    def _target_done(self, target: str) -> bool:
        d, cell = self.d, cell_of(target)
        if cell is None or cell == d.pos:
            return True
        if target.startswith("gem"):
            return cell not in d.gems
        if target.startswith("health potion"):
            return cell not in d.potions
        if target.startswith("the key"):
            return d.key != cell
        if target.startswith("unexplored room"):
            return d.area.get(cell) in d.seen
        return False

    def target_cell(self) -> Cell | None:
        return cell_of(self.stack.current.get("target"))

    # -- options -----------------------------------------------------------------------

    def _reachable(self, cells) -> list[Cell]:
        return sorted((c for c in cells if c in self._dist), key=lambda c: (c[1], c[0]))  # map order

    def goal_options(self) -> list[str]:
        """The goals that are possible now, in the fixed order of GOALS."""
        d = self.d
        leaving = d.has_key and d.exit in self._dist  # the key in hand and the way out known
        ok = {
            "explore": bool(frontier(d, self._dist)) and not leaving,
            "get the key": d.key is not None and d.key in self._dist,
            "go to the exit": leaving,
            "collect gems": bool(self._reachable(d.gems)) and not leaving,
            "drink a health potion": bool(self._reachable(d.potions)) and d.health < d.rules.health,
        }
        return [g for g in GOALS if ok[g]]

    def place(self, c: Cell) -> str:
        room = self.d.room_at(c)
        if room is None:
            return "a corridor"
        return f"the {room.name} doorway" if self.d.is_door(c) else f"the {room.name}"

    def target_options(self, strategy: str) -> list[str]:
        """Candidate targets for a goal, in map order (not sorted by distance)."""
        d, dist = self.d, self._dist
        if strategy == "explore":
            return [f"unexplored room behind the doorway at ({x},{y}), {_steps(dist[(x, y)])} away"
                    for x, y in sorted(frontier(d, dist), key=lambda c: (c[1], c[0]))]
        if strategy == "get the key" and d.key in dist:
            (x, y) = d.key
            return [f"the key at ({x},{y}) in {self.place(d.key)}, {_steps(dist[d.key])} away"]
        if strategy == "go to the exit" and d.exit in dist:
            (x, y) = d.exit
            return [f"the exit at ({x},{y}) in {self.place(d.exit)}, {_steps(dist[d.exit])} away"]
        if strategy == "collect gems":
            return [f"gem at ({x},{y}) in {self.place((x, y))}, {_steps(dist[(x, y)])} away"
                    for x, y in self._reachable(d.gems)]
        if strategy == "drink a health potion":
            return [f"health potion at ({x},{y}) in {self.place((x, y))}, {_steps(dist[(x, y)])} away"
                    for x, y in self._reachable(d.potions)]
        return []

    def move_options(self) -> list[str]:
        """Each move (and dash, when ready) labelled with its outcome. The model still chooses."""
        d, target = self.d, self.target_cell()
        tdist = self._route_dist()
        here = tdist.get(d.pos)
        out, progress = [], False
        for move in MOVES[:4] + (DASHES if d.can_dash() else ()):
            if is_dash(move):
                path = d.dash_cells(move)
                if len(path) < 2:
                    continue  # a one-cell dash is a step that wastes the charge
            else:
                dx, dy = STEP[move]
                path = [(d.pos[0] + dx, d.pos[1] + dy)]
                if not d.passable(path[0]) or d.ghoul_at(path[0]) is not None:
                    continue  # walls and ghouls are not offered (they would be trap options)
            label, good = self._label(path, target, tdist, here)
            progress |= good
            out.append(f"{move} ({len(path)} cells; {label})" if is_dash(move) else f"{move} ({label})")
        if not progress or target is None or target == d.pos:
            out.append(f"stay ({self._label([], target, tdist, here)[0]})")
        safe = [o for o in out if "(safe" in o or "cells; safe" in o or "EXIT: you escape" in o]
        return safe or out  # risky options only when nothing is safe

    def _label(self, path: list[Cell], target: Cell | None, tdist: dict[Cell, int], here: int | None) -> tuple[str, bool]:
        """The outcome of landing on path[-1] after passing the cells of ``path`` (empty: staying),
        and whether it is safe progress (closer to the target, or onto it)."""
        d = self.d
        c = path[-1] if path else d.pos
        notes = []
        risky = c in self._threat
        if risky:
            notes.append(f"GHOUL: -{d.rules.ghoul_damage} health")
        if d.exit in path:
            notes.append("EXIT: you escape" if d.has_key else "the exit, locked without the key")
        elif d.key is not None and d.key in path:
            notes.append("pick up the key")
        if any(x in d.potions for x in path):
            notes.append(f"potion: +{d.rules.potion_health} health")
        gems = sum(x in d.gems for x in path)
        if gems:
            notes.append("gem" if gems == 1 else f"{gems} gems")
        good = False
        if target is not None and (c == target or not risky):
            if c == target:
                notes.append("reach the target" if path else "on the target")
                good = bool(path) and not risky
            elif target in path:
                notes.append("passes over the target")
                good = not risky
            elif c in tdist and here is not None:
                n = tdist[c]
                rel = "closer" if n < here else "farther" if n > here else "no closer"
                notes.append(f"{rel}: {_steps(n)} to the target")
                good = n < here
            elif c in tdist:
                notes.append(f"target: {_steps(tdist[c])}")
            else:
                notes.append("no safe route to the target")
        if not risky:
            notes.insert(0, "safe")
        return "; ".join(notes), good

    def _route_dist(self) -> dict[Cell, int]:
        """Walking distance to the target over known cells that no ghoul can hit next tick."""
        d, target = self.d, self.target_cell()
        if target is None:
            return {}
        blocked = self._threat | {g.pos for g in d.ghouls}
        return d.distances(target, within=(self._known - blocked) | {target, d.pos})

    def route(self) -> list[Cell]:
        """The safe route to the target, first step first (ties in DIRS order: deterministic)."""
        dist = self._route_dist()
        c = self.d.pos
        if c not in dist:
            return []
        path = []
        while dist[c] > 0:
            c = next(n for n in ((c[0] + dx, c[1] + dy) for dx, dy in DIRS.values()) if dist.get(n) == dist[c] - 1)
            path.append(c)
        return path

    # -- world -> text -----------------------------------------------------------------

    def status(self) -> str:
        d, h = self.d, self.d.health
        hits = math.ceil(h / d.rules.ghoul_damage)
        word = "CRITICAL" if hits <= 2 else "hurt" if h < 60 else "fine"
        return f"Health: {word} ({h}/{d.rules.health}): {_hits(hits)} by a ghoul would kill you."

    def _nearest(self, cells) -> tuple[int, int] | None:
        steps = [self._dist[c] for c in cells if c in self._dist]
        return (len(steps), min(steps)) if steps else None

    def _ghoul_state(self, g) -> str:
        if not g.awake:
            return "asleep"
        if g.rest:
            return "backing off"
        if g.tired:
            return f"tired, going back to its post for {_ticks(g.tired)}"
        return "awake, chasing you" if g in self.chasers() else "awake"

    def _ghouls_line(self, limit: int = 3) -> str:
        d = self.d
        x, y = d.pos
        rid = d.area.get(d.pos)
        near = sorted((g for g in d.known_ghouls() if g.home == rid or abs(g.pos[0] - x) + abs(g.pos[1] - y) <= 6),
                      key=lambda g: (abs(g.pos[0] - x) + abs(g.pos[1] - y), g.id))
        if not near:
            return "Ghouls: none near you."
        parts = [f"ghoul #{g.id} {_offset(g.pos[0] - x, y - g.pos[1])} ({self._ghoul_state(g)})" for g in near[:limit]]
        return "Ghouls near you: " + "; ".join(parts) + "."

    def strategy_state(self) -> str:
        d = self.d
        lines = [f"Standing order: {STANDING_ORDER}",
                 f"You are in {self.place(d.pos)}. Rooms explored: {len(d.seen)} of {len(d.rooms)}.", self.status()]
        if d.has_key:
            lines.append("The key: you carry it, so the exit will open for you.")
        elif d.key in self._dist:
            lines.append(f"The key: seen in {self.place(d.key)}, {_steps(self._dist[d.key])} away. "
                         "The exit stays locked until you carry it.")
        else:
            lines.append("The key: not found yet. The exit stays locked until you carry it.")
        if d.exit in self._dist:
            lines.append(f"The exit: in {self.place(d.exit)}, {_steps(self._dist[d.exit])} away"
                         + ("." if d.has_key else ", locked."))
        else:
            lines.append("The exit: not found yet.")
        g = self._nearest(d.gems)
        lines.append((f"Gems: {g[0]} known, nearest {_steps(g[1])} away." if g else "Gems: none known.")
                     + f" You have collected {d.collected}.")
        p = self._nearest(d.potions)
        lines.append(f"Health potions: {p[0]} known, nearest {_steps(p[1])} away; each gives "
                     f"+{d.rules.potion_health} health." if p else "Health potions: none known.")
        lines.append(self._ghouls_line())
        fr = frontier(d, self._dist)
        if fr:
            lines.append(f"Unexplored rooms: {len(fr)} reachable, nearest doorway {_steps(min(self._dist[c] for c in fr))} away.")
        else:
            lines.append("Unexplored rooms: none reachable." if len(d.seen) < len(d.rooms) else "Unexplored rooms: none left.")
        return " ".join(lines)

    def move_state(self) -> str:
        d = self.d
        x, y = d.pos
        cell, path = self.target_cell(), self.route()
        if cell is None:
            where = "You have no target yet."
        elif cell == d.pos:
            where = "You are on your target."
        elif not path:
            where = f"Your target is {_offset(cell[0] - x, y - cell[1])} of you; no safe route to it now."
        else:
            first = (path[0][0] - x, path[0][1] - y)
            run = 1
            while run < len(path) and (path[run][0] - path[run - 1][0], path[run][1] - path[run - 1][1]) == first:
                run += 1
            name = next(k for k, v in DIRS.items() if v == first)
            turn = "" if run == len(path) else ", then turns"
            where = (f"Your target is {_offset(cell[0] - x, y - cell[1])} of you, {_steps(len(path))} along the safe "
                     f"route. The route goes {_steps(run)} {name}{turn}.")
        nb = []
        for name, (dx, dy) in DIRS.items():
            c = (x + dx, y + dy)
            nb.append(f"{name} {'wall' if not d.passable(c) else 'ghoul' if d.ghoul_at(c) else 'free'}")
        R = d.rules
        dash = (f"Dash: ready. A dash moves you up to {R.dash_cells} cells in a straight line in one tick; then it "
                f"recharges for {_ticks(R.dash_recharge)}." if d.can_dash() else
                f"Dash: recharging, ready in {_ticks(d.dash_charge)}.") if R.dash else ""
        return " ".join(p for p in [f"You are at ({x},{y}) in {self.place(d.pos)}. North is up.", where,
                                    "Neighbouring cells: " + ", ".join(nb) + ".", dash, self._ghouls_line(), self.status()] if p)

    def state_for(self, tier: Tier) -> str:
        return self.move_state() if tier.name == "move" else self.strategy_state()


class Runner:
    """Runs one episode. Each tick is ONE batched engine call for every due decision, and
    returns a trace record: the world the decisions saw, every decision with its question,
    options and probabilities, the batch latency, and what happened."""

    STUCK = 6  # ticks without getting closer to an unchanged target that count as stuck
    PROGRESS = ("room_seen", "key", "gem", "potion", "escaped")

    def __init__(self, dungeon: Dungeon, engine, group_size: int = 8, plan_budget: int | None = 1,
                 order_debias: bool = True, **brain_options):
        """``order_debias``: every decision is read in two option orders and averaged (Engine)."""
        self.d, self.engine, self.group_size, self.plan_budget = dungeon, engine, group_size, plan_budget
        self.order_debias = order_debias
        self.brain = DungeonBrain(dungeon, **brain_options)
        self.records: list[dict] = []
        self._t0: float | None = None
        self._best: int | None = None
        self._streak = 0
        self._idle = IdleTracker(dungeon.pos, self.PROGRESS)

    def header(self) -> dict:
        eng = self.engine
        example = eng.render(self.brain.stack.decision(self.brain.stack.tier("move"), "<state>"), ["A", "B", "C", "D", "E"])
        return {
            "type": "header", "version": 2, "scenario": "dungeon", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "standing_order": STANDING_ORDER, "goals": list(GOALS), "map": self.d.layout(),
            "tiers": [{"name": t.name, "question": t.instruction, "every": t.every, "title": t.title,
                       "kind": "control" if t.every == 1 else "plan"} for t in self.brain.stack.tiers],
            "group_size": self.group_size, "plan_budget": self.plan_budget, "order_debias": self.order_debias,
            "backend": eng.backend.info(),
            "brain": {"hold_navigation": True, "stuck_replan": self.STUCK, "loop": LOOP},
            "engine": {"answer": eng.answer, "prompt_order": eng.prompt_order, "system_prompt": eng.system_prompt,
                       "prefill": eng.prefill, "suffix": eng.suffix},
            "example_prompt": example,
        }

    def tick(self) -> dict:
        d, stack = self.d, self.brain.stack
        start = time.perf_counter()
        if self._t0 is None:
            self._t0 = start
        self.brain.refresh()
        world, tick = d.snapshot(), d.tick
        target_before = self.brain.target_cell()
        before, self.engine.order_debias = getattr(self.engine, "order_debias", False), self.order_debias
        try:
            updates = step_all(self.engine, [(stack, self.brain.state_for)], tick, group_size=self.group_size,
                               plan_budget=self.plan_budget, skip_single=True)
        finally:
            self.engine.order_debias = before
        model = [u for u in updates if u[2].method != ONLY_OPTION]
        stats = dict(self.engine.last_stats) if model else {}
        move = base_move(stack.current.get("move", "stay"))
        target = self.brain.target_cell()
        dist_before = self._dist_to(target_before)
        events = d.step(move)
        self._track_progress(target_before, dist_before, target)
        idle = self._idle.update(d.pos, events)
        if self._streak and self._streak % self.STUCK == 0:  # stalled: re-plan, and the strategy too every 2nd time
            self.brain.replan(strategy=self._streak % (2 * self.STUCK) == 0)
        if idle and idle % LOOP == 0:  # going in circles: re-plan both
            self.brain.replan(strategy=True)
        rec = {
            "type": "tick", "tick": tick, "t": round(start - self._t0, 4), "move": move, "world": world,
            "target": list(target) if target else None, "events": events, "stuck_streak": self._streak, "idle": idle,
            "batch": {"decisions": len(model), "forward_ms": round(1000 * stats.get("forward_s", 0.0), 1),
                      "prompt_tokens": stats.get("prompt_tokens", 0),
                      "tick_ms": round(1000 * (time.perf_counter() - start), 1)},
            "decisions": [self._decision(tier, r, tour) for _, tier, r, tour in updates],
            "goals": {t.name: {"choice": stack.current.get(t.name), "p": round(stack.probability.get(t.name, 0.0), 5),
                               "since": stack.decided_at(t.name), "waiting": stack.waiting(t.name)}
                      for t in stack.tiers},
            "tournaments": {name: dict(zip(("round", "resolved", "contested"), t.progress()),
                                       alive=len(t.alive), options=len(t.options))
                            for name, t in stack.tournaments.items()},
        }
        self.records.append(rec)
        return rec

    def _dist_to(self, cell: Cell | None) -> int | None:
        if cell is None:
            return None
        return self.d.distances(self.d.pos, within=self.d.known_cells() | {cell}).get(cell)

    def _track_progress(self, cell: Cell | None, before: int | None, target_now: Cell | None) -> None:
        """Stuck = ticks in streaks of STUCK+ ticks without getting closer to an unchanged target."""
        if before is None or cell != target_now or before == 0:
            self._best, self._streak = None, 0
            return
        now = self._dist_to(cell)
        now = before if now is None else now
        if self._best is None or now < self._best:
            self._best, self._streak = now, 0
        else:
            self._streak += 1

    @staticmethod
    def _decision(tier, r, tour) -> dict:
        dec = r.decision
        out = {
            "tier": tier.name, "kind": "control" if tier.every == 1 else "plan", "question": dec.instruction,
            "state": dec.state, "context": dec.context, "options": list(dec.options), "labels": list(r.labels),
            "probs": [round(p, 5) for p in r.probs], "choice": r.index, "outside": round(r.outside_mass, 5),
            "method": r.method, "top_token": r.top_token, "prompt_tokens": r.prompt_tokens,
        }
        if getattr(r, "orders", None):
            out["orders"] = [[round(p, 5) for p in ps] for ps in r.orders]
        if tour is not None:
            out["tournament"] = {"options": len(tour.options), "alive": len(tour.alive), "done": tour.done,
                                 "winner": tour.winner if tour.done else None}
        return out

    def run(self, max_ticks: int | None = None, on_tick=None) -> dict:
        while self.d.outcome is None and (max_ticks is None or self.d.tick < max_ticks):
            rec = self.tick()
            if on_tick:
                on_tick(rec)
        return self.end()

    def end(self) -> dict:
        return {"type": "end", "tick": self.d.tick, "outcome": self.d.outcome, "cause": self.d.cause,
                "world": self.d.snapshot(), "summary": summarise(self.records, self.d)}


def _risky(option: str) -> bool:
    return "GHOUL" in option


def summarise(records: list[dict], d: Dungeon) -> dict:
    """Closed-loop metrics of one run."""
    events = [(r["tick"], e) for r in records for e in r["events"]]

    def first(kind, **match):
        return next((t for t, e in events if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())), None)

    fw = [r["batch"]["forward_ms"] for r in records if r["batch"]["decisions"]]
    ticks = [r["batch"]["tick_ms"] for r in records]
    decisions = [x for r in records for x in r["decisions"]]
    moves = [x for x in decisions if x["tier"] == "move" and x["method"] != ONLY_OPTION]
    avoidable = [x for x in moves if _risky(x["options"][x["choice"]]) and any(not _risky(o) for o in x["options"])]
    safe_hits = 0  # hits on a tick whose chosen move was labelled safe: each one is a label that was wrong
    for r in records:
        mv = next((x for x in r["decisions"] if x["tier"] == "move"), None)
        if mv and not _risky(mv["options"][mv["choice"]]):
            safe_hits += sum(1 for e in r["events"] if e["kind"] == "hit")
    return {
        "outcome": d.outcome, "cause": d.cause, "ticks": len(records), "gems": d.collected,
        "hits": sum(1 for _, e in events if e["kind"] == "hit"), "hits_after_safe_move": safe_hits,
        "move_decisions": len(moves), "avoidable_risky_moves": len(avoidable),
        "potions_drunk": sum(1 for _, e in events if e["kind"] == "potion"),
        "dashes": d.dashes, "dashes_offered": sum(1 for x in moves if any(is_dash(o) for o in x["options"])),
        "bumps": sum(1 for _, e in events if e["kind"] == "bump"),
        "rooms_seen": len(d.seen), "key_seen_tick": first("room_seen", room=d.key_room),
        "key_tick": first("key"), "exit_seen_tick": first("room_seen", room=d.exit_room), "escape_tick": first("escaped"),
        "min_health": min([r["world"]["agent"]["health"] for r in records] + [d.health]),
        "stuck_ticks": streak_ticks((r["stuck_streak"] for r in records), Runner.STUCK),
        "loop_ticks": streak_ticks((r["idle"] for r in records), LOOP),
        "longest_idle": max((r["idle"] for r in records), default=0),
        "strategy_ticks": dict(Counter(r["goals"]["strategy"]["choice"] or "-" for r in records)),
        "model_decisions": sum(1 for x in decisions if x["method"] != ONLY_OPTION),
        "planning_decisions": sum(1 for x in decisions if x["kind"] == "plan" and x["method"] != ONLY_OPTION),
        "only_option_commits": sum(1 for x in decisions if x["method"] == ONLY_OPTION),
        "tournament_decisions": sum(1 for x in decisions if "tournament" in x),
        "forward_ms_median": round(statistics.median(fw), 1) if fw else 0.0,
        "forward_ms_p90": round(statistics.quantiles(fw, n=10)[-1], 1) if len(fw) >= 10 else max(fw, default=0.0),
        "forward_ms_max": max(fw, default=0.0),
        "tick_ms_median": round(statistics.median(ticks), 1) if ticks else 0.0,
    }
