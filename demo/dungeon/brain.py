"""Turns the dungeon into text decisions for one agent, and runs it tick by tick.

Tiers (tiered goals, see system_one/goals.py):
  strategy  every 12 ticks, or at once when the agent's condition changes: a goal from GOALS.
            Only goals that are possible now are offered (no "get the key" before the key
            has been seen; no "go to the exit" without the key).
  target    every 6 ticks, or when the target is reached or gone: a concrete cell for the goal.
            All known gems can be more than 8 options: then it is a tournament.
  action    every tick: move north/south/east/west/stay, each labelled with its outcome.

The model makes every choice. The code only computes what the text says: walking distances
over the cells the agent knows, what is in the way, and the consequences of the agent's
condition. These are the lessons of the first demo (docs/research.md -> Agent awareness).
A tier with a single possible option is committed without a model call.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter

from system_one.goals import ONLY_OPTION, GoalStack, Tier, step_all

from ..brain import _steps, base_move
from .world import DIRS, MOVES, STEP, Cell, Dungeon

STANDING_ORDER = "Find the key, then leave through the exit alive. Gems are a bonus. Eat and heal when needed."
GOALS = ("explore", "get the key", "go to the exit", "collect gems", "eat food", "drink a health potion",
         "flee the enemy")
ITEM_GOALS = {"collect gems": ("gem", "gems"), "eat food": ("food", "food"),
              "drink a health potion": ("health potion", "potions")}
_CELL = re.compile(r"\((\d+),(\d+)\)")


def _offset(dx: int, dy_north: int) -> str:
    """'3 cells east and 1 cell north' (relative position; the model cannot do the arithmetic)."""
    parts = [f"{abs(v)} {'cell' if abs(v) == 1 else 'cells'} {pos if v > 0 else neg}"
             for v, pos, neg in ((dx, "east", "west"), (dy_north, "north", "south")) if v]
    return " and ".join(parts) or "0 cells away"


def cell_of(text: str | None) -> Cell | None:
    m = _CELL.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def drop_distance(option: str) -> str:
    """How lower tiers see a target: 'gem at (5,6) in the hall, 4 steps away' -> 'gem at (5,6) in the hall'.
    The distance was measured when the target was chosen and goes stale (Observed in the first demo)."""
    return option.split(", ")[0]


def _ticks(n: int) -> str:
    return f"{n} tick" if n == 1 else f"{n} ticks"


class DungeonBrain:
    """Goal stack and state text of the one agent."""

    def __init__(self, dungeon: Dungeon, strategy_every: int = 12, target_every: int = 6,
                 label_style: str = "closer", enemy_aware: bool = False):
        """``label_style`` words the move outcomes: "closer" ('closer: 2 steps to the target') or
        "steps" ('target: 2 steps', the first demo's wording). Observed on the dev seed: with
        "steps" the 3B walked away from its target in one doorway state and looped until it starved.

        ``enemy_aware`` (added after the pre-registered evaluation; docs/research.md -> Dungeon):
        a move into an enemy says the agent stays in place and is hit; a cell next to an enemy says
        it can hit; safe spots are only cells the agent reaches before any visible enemy, by a path
        that never passes next to one. Off = the wording the pre-registered evaluation used."""
        if label_style not in ("closer", "steps"):
            raise ValueError("label_style must be 'closer' or 'steps'")
        self.d, self.label_style, self.enemy_aware = dungeon, label_style, enemy_aware
        self.stack = GoalStack([
            Tier("strategy", "Which goal should the agent pursue now?", lambda goals: self.goal_options(),
                 every=strategy_every, title="Strategic goal"),
            Tier("target", "Which target best serves the strategic goal?",
                 lambda goals: self.target_options(goals.get("strategy", "explore")),
                 every=target_every, title="Current target", describe=drop_distance),
            Tier("action", "Which move brings you closer to your current target? "
                 "Do not move into a wall or an enemy.", lambda goals: self.move_options(), every=1),
        ])
        self._bands: tuple | None = None
        self.refresh()

    # -- per-tick bookkeeping ------------------------------------------------------------

    def refresh(self) -> None:
        """Recompute what the agent knows. Re-plan the strategy when the agent's condition
        changes or its goal became impossible; re-plan the target when it was reached or is gone."""
        d = self.d
        self._known = d.known_cells()
        self._dist = d.distances(d.pos, within=self._known)
        self._enemies = d.visible_enemies()
        bands = self._condition()
        if self._bands is not None and bands != self._bands:
            self.stack.invalidate("strategy")
        self._bands = bands
        strategy = self.stack.current.get("strategy")
        if strategy is not None and strategy not in self.goal_options():
            self.stack.invalidate("strategy")
        target = self.stack.current.get("target")
        if target is not None and self._target_done(target):
            self.stack.invalidate("target")

    def _condition(self) -> tuple:
        d = self.d
        e = d.energy
        energy = 0 if e == 0 else 1 if e < 5 else 2 if e < 15 else 3 if e < 25 else 4 if e < 35 else 5
        hits = min(3, math.ceil(d.health / d.rules.enemy_damage))
        return energy, hits, frozenset(e.id for e, _ in self._enemies), d.has_key, len(d.seen)

    def _target_done(self, target: str) -> bool:
        d, cell = self.d, cell_of(target)
        if cell is None or cell == d.pos:
            return True
        if target.startswith("gem"):
            return cell not in d.gems
        if target.startswith("food"):
            return cell not in d.food
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

    def frontier(self) -> list[tuple[Cell, int]]:
        """(door, steps) for each unseen room behind a known door; the nearest door per room."""
        d, best = self.d, {}
        for cor in d.corridors:
            for rid, door, other in ((cor.rooms[0], cor.doors[0], cor.rooms[1]), (cor.rooms[1], cor.doors[1], cor.rooms[0])):
                if rid not in d.seen and other in d.seen and door in self._dist:
                    if rid not in best or self._dist[door] < best[rid][1]:
                        best[rid] = (door, self._dist[door])
        return sorted(best.values(), key=lambda x: (x[0][1], x[0][0]))

    def goal_options(self) -> list[str]:
        """The goals that are possible now, in the fixed order of GOALS."""
        d = self.d
        ok = {
            "explore": bool(self.frontier()),
            "get the key": d.key is not None and d.key in self._dist,
            "go to the exit": d.has_key and d.exit in self._dist,
            "collect gems": bool(self._reachable(d.gems)),
            "eat food": bool(self._reachable(d.food)) and d.energy < 100,
            "drink a health potion": bool(self._reachable(d.potions)) and d.health < 100,
            "flee the enemy": bool(self._enemies),
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
            return [f"unexplored room behind the door at ({x},{y}), {_steps(s)} away" for (x, y), s in self.frontier()]
        if strategy == "get the key" and d.key in dist:
            (x, y) = d.key
            return [f"the key at ({x},{y}) in {self.place(d.key)}, {_steps(dist[d.key])} away"]
        if strategy == "go to the exit" and d.exit in dist:
            (x, y) = d.exit
            return [f"the exit at ({x},{y}) in {self.place(d.exit)}, {_steps(dist[d.exit])} away"]
        if strategy in ITEM_GOALS:
            noun, attr = ITEM_GOALS[strategy]
            return [f"{noun} at ({x},{y}) in {self.place((x, y))}, {_steps(dist[(x, y)])} away"
                    for x, y in self._reachable(getattr(d, attr))]
        if strategy == "flee the enemy":
            return self.safe_spots()
        return []

    def safe_spots(self, n: int = 5) -> list[str]:
        """Known cells 2-8 steps away, far (walking) from the visible enemies.

        With ``enemy_aware``: only cells the agent reaches first, by a path that never passes next
        to an enemy, ranked by how many steps ahead of the nearest enemy the agent arrives.
        Without it (the pre-registered version) a spot could lie beyond the enemy."""
        if not self._enemies:
            return []
        from_enemy: dict[Cell, int] = {}
        for e, _ in self._enemies:
            for c, s in self.d.distances(e.pos).items():
                from_enemy[c] = min(from_enemy.get(c, 10 ** 6), s)
        if self.enemy_aware:
            danger = {(e.pos[0] + dx, e.pos[1] + dy) for e, _ in self._enemies
                      for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))}
            mine = self.d.distances(self.d.pos, within=(self._known - danger) | {self.d.pos})
            cand = [(from_enemy.get(c, 99) - s, from_enemy.get(c, 99), s, c) for c, s in mine.items()
                    if 2 <= s <= 8 and from_enemy.get(c, 99) > s]
            spots = sorted(cand, key=lambda x: (-x[0], -x[1], x[2], x[3]))[:n]
            return [f"safe spot at ({c[0]},{c[1]}) in {self.place(c)}, {_steps(s)} away; "
                    f"the enemy is {_steps(h)} from it" for _, h, s, c in sorted(spots, key=lambda x: (x[3][1], x[3][0]))]
        spots = sorted(((from_enemy.get(c, 99), s, c) for c, s in self._dist.items() if 2 <= s <= 8),
                       key=lambda x: (-x[0], x[1], x[2]))[:n]
        return [f"safe spot at ({c[0]},{c[1]}) in {self.place(c)}, {_steps(s)} away, {_steps(h)} from the enemy"
                for h, s, c in sorted(spots, key=lambda x: (x[2][1], x[2][0]))]

    def move_options(self) -> list[str]:
        """Each move labelled with its outcome. The model still chooses."""
        d, target = self.d, self.target_cell()
        tdist = d.distances(target, within=self._known | {target}) if target else {}
        here = tdist.get(d.pos)
        nearby = [(e.pos, e.stunned) for e, _ in self._enemies]
        out = []
        for move in MOVES:
            dx, dy = STEP[move]
            c = (d.pos[0] + dx, d.pos[1] + dy)
            enemy = d.enemy_at(c) if move != "stay" else None
            if not d.passable(c):
                out.append(f"{move} (wall)")
                continue
            if enemy is not None and self.enemy_aware:
                out.append(f"{move} (blocked by a stunned enemy: you stay here; it can hit you again in "
                           f"{_ticks(enemy.stunned + 1)})" if enemy.stunned else
                           f"{move} (ENEMY: you stay here and lose {d.rules.enemy_damage} health)")
                continue
            if enemy is not None:
                out.append(f"{move} (blocked by a stunned enemy; it recovers in {_ticks(enemy.stunned)})"
                           if enemy.stunned else f"{move} (ENEMY: -{d.rules.enemy_damage} health)")
                continue
            notes = []
            if move != "stay":
                if c == d.exit:
                    notes.append("EXIT: you escape" if d.has_key else "the exit, locked without the key")
                elif c == d.key:
                    notes.append("pick up the key")
                elif c in d.gems:
                    notes.append("gem")
                elif c in d.food:
                    notes.append(f"food: +{d.rules.food_energy} energy")
                elif c in d.potions:
                    notes.append(f"potion: +{d.rules.potion_health} health")
            beside = [st for p, st in nearby if abs(c[0] - p[0]) + abs(c[1] - p[1]) == 1]
            if beside and self.enemy_aware:
                notes.append(f"next to an enemy: it can hit you for {d.rules.enemy_damage} health" if min(beside) == 0 else
                             f"next to a stunned enemy: it can hit you again in {_ticks(min(beside) + 1)}")
            elif beside:
                notes.append("next to an enemy" if min(beside) == 0 else
                             f"next to a stunned enemy that recovers in {_ticks(min(beside))}")
            if target is not None:
                if c == target:
                    notes.append("on the target" if move == "stay" else "reach the target")
                elif c in tdist and (self.label_style == "steps" or here is None):
                    notes.append(f"target: {_steps(tdist[c])}")
                elif c in tdist:
                    n = tdist[c]
                    rel = "closer" if n < here else "farther" if n > here else "no closer"
                    notes.append(f"{rel}: {_steps(n)} to the target")
                else:
                    notes.append("target unreachable")
            out.append(f"{move} ({'; '.join(notes) or 'free'})")
        return out

    # -- world -> text -----------------------------------------------------------------

    def status(self, detailed: bool) -> str:
        """``detailed`` spells out consequences; Observed in the first demo: without them the 3B
        ignored "Energy: LOW" even at 0 (bench/strategy_probe.py)."""
        d, e, h = self.d, self.d.energy, self.d.health
        hits = math.ceil(h / d.rules.enemy_damage)
        if not detailed:
            return (f"Energy: {'LOW' if e < 35 else 'ok'} ({e}/100). "
                    f"Health: {'LOW' if hits <= 1 else 'ok'} ({h}/100).")
        if e == 0:
            energy = f"EMPTY (0/100): you are starving and lose {d.rules.starve_damage} health every tick"
        elif e < 35:
            energy = f"LOW ({e}/100): about {e} ticks until starving, then {d.rules.starve_damage} health lost per tick"
        else:
            energy = f"ok ({e}/100): about {e} ticks of energy left"
        word = "CRITICAL" if hits <= 1 else "hurt" if hits == 2 else "fine"
        health = f"{word} ({h}/100): {hits} more enemy {'hit' if hits == 1 else 'hits'} would kill you"
        return f"Energy: {energy}. Health: {health}."

    def _nearest(self, cells) -> tuple[int, int] | None:
        steps = [self._dist[c] for c in cells if c in self._dist]
        return (len(steps), min(steps)) if steps else None

    def strategy_state(self) -> str:
        d = self.d
        lines = [f"Standing order: {STANDING_ORDER}",
                 f"You are in {self.place(d.pos)}. Rooms explored: {len(d.seen)} of {len(d.rooms)}.",
                 self.status(detailed=True)]
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
        f = self._nearest(d.food)
        lines.append(f"Food: {f[0]} known, nearest {_steps(f[1])} away; each gives +{d.rules.food_energy} energy."
                     if f else "Food: none known.")
        p = self._nearest(d.potions)
        lines.append(f"Health potions: {p[0]} known, nearest {_steps(p[1])} away; each gives "
                     f"+{d.rules.potion_health} health." if p else "Health potions: none known.")
        lines.append(self._enemy_text())
        fr = self.frontier()
        if fr:
            lines.append(f"Unexplored rooms: {len(fr)} reachable, nearest door {_steps(min(s for _, s in fr))} away.")
        else:
            lines.append("Unexplored rooms: none reachable." if len(d.seen) < len(d.rooms) else "Unexplored rooms: none left.")
        return " ".join(lines)

    def _enemy_text(self) -> str:
        if not self._enemies:
            return "Enemies: none in sight."
        e, s = self._enemies[0]
        dmg, radius = self.d.rules.enemy_damage, self.d.rules.chase_radius
        who = "Enemy" if len(self._enemies) == 1 else f"{len(self._enemies)} enemies in sight. Nearest enemy"
        if e.stunned:
            what = f"{_steps(s)} away, stunned for {_ticks(e.stunned)} more"
        elif s == 1:
            what = f"ADJACENT: it hits you for {dmg} health"
        elif e.mode == "chase":
            what = f"{_steps(s)} away and chasing you; each hit costs {dmg} health"
        else:
            what = f"{_steps(s)} away, not chasing you yet (enemies chase within {radius} steps)"
        return f"{who}: {what}."

    def action_state(self) -> str:
        d = self.d
        x, y = d.pos
        cell = self.target_cell()
        if cell is None:
            where = "You have no target yet."
        elif cell == d.pos:
            where = "You are on your target."
        else:
            where = f"Your target is {_offset(cell[0] - x, y - cell[1])} of you."
        nb = []
        for name, (dx, dy) in DIRS.items():
            c = (x + dx, y + dy)
            nb.append(f"{name} {'wall' if not d.passable(c) else 'enemy' if d.enemy_at(c) else 'free'}")
        parts = [f"You are at ({x},{y}) in {self.place(d.pos)}. North is up.", where,
                 "Neighbouring cells: " + ", ".join(nb) + "."]
        for e, _ in self._enemies[:2]:
            parts.append(f"An enemy is {_offset(e.pos[0] - x, y - e.pos[1])} of you.")
        parts.append(self.status(detailed=False))
        return " ".join(parts)

    def state_for(self, tier: Tier) -> str:
        return self.action_state() if tier.name == "action" else self.strategy_state()


class Runner:
    """Runs one episode. Each tick is ONE batched engine call for every due decision, and
    returns a trace record: the world the decisions saw, every decision with its question,
    options and probabilities, the batch latency, and what happened."""

    STUCK = 6  # a streak of this many ticks without getting closer to the same target counts as stuck

    def __init__(self, dungeon: Dungeon, engine, group_size: int = 8, plan_budget: int | None = 1, **brain_options):
        self.d, self.engine, self.group_size, self.plan_budget = dungeon, engine, group_size, plan_budget
        self.brain = DungeonBrain(dungeon, **brain_options)
        self.records: list[dict] = []
        self._t0: float | None = None
        self._best: int | None = None
        self._streak = 0

    def header(self) -> dict:
        eng = self.engine
        example = eng.render(self.brain.stack.decision(self.brain.stack.tiers[-1], "<state>"), ["A", "B", "C", "D", "E"])
        return {
            "type": "header", "version": 1, "scenario": "dungeon", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "standing_order": STANDING_ORDER, "goals": list(GOALS), "map": self.d.layout(),
            "tiers": [{"name": t.name, "question": t.instruction, "every": t.every, "title": t.title,
                       "kind": "control" if t.every == 1 else "plan"} for t in self.brain.stack.tiers],
            "group_size": self.group_size, "plan_budget": self.plan_budget, "backend": eng.backend.info(),
            "brain": {"label_style": self.brain.label_style, "enemy_aware": self.brain.enemy_aware,
                      "strategy_every": self.brain.stack.tier("strategy").every,
                      "target_every": self.brain.stack.tier("target").every},
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
        rounds = {id(t): len(t.rounds) + 1 for t in stack.tournaments.values()}
        updates = step_all(self.engine, [(stack, self.brain.state_for)], tick, group_size=self.group_size,
                           plan_budget=self.plan_budget, skip_single=True)
        model = [u for u in updates if u[2].method != ONLY_OPTION]
        stats = dict(self.engine.last_stats) if model else {}
        move = base_move(stack.current.get("action", "stay"))
        target = self.brain.target_cell()
        dist_before = self._dist_to(target_before)
        events = d.step(move)
        self._track_progress(target_before, dist_before, target)
        rec = {
            "type": "tick", "tick": tick, "t": round(start - self._t0, 4), "move": move, "world": world,
            "target": list(target) if target else None, "events": events, "stuck_streak": self._streak,
            "batch": {"decisions": len(model), "forward_ms": round(1000 * stats.get("forward_s", 0.0), 1),
                      "prompt_tokens": stats.get("prompt_tokens", 0),
                      "tick_ms": round(1000 * (time.perf_counter() - start), 1)},
            "decisions": [self._decision(tier, r, tour, rounds) for _, tier, r, tour in updates],
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
        """Stuck = ticks in streaks of STUCK+ ticks without getting closer to an unchanged target.
        Same definition as bench/survival_compare.py."""
        if before is None or cell != target_now:
            self._best, self._streak = None, 0
            return
        now = self._dist_to(cell)
        now = before if now is None else now
        if self._best is None or now < self._best:
            self._best, self._streak = now, 0
        else:
            self._streak += 1

    @staticmethod
    def _decision(tier, r, tour, rounds) -> dict:
        dec = r.decision
        out = {
            "tier": tier.name, "kind": "control" if tier.every == 1 else "plan", "question": dec.instruction,
            "state": dec.state, "context": dec.context, "options": list(dec.options), "labels": list(r.labels),
            "probs": [round(p, 5) for p in r.probs], "choice": r.index, "outside": round(r.outside_mass, 5),
            "method": r.method, "top_token": r.top_token, "prompt_tokens": r.prompt_tokens,
        }
        if tour is not None:
            out["tournament"] = {"round": rounds.get(id(tour), 1), "options": len(tour.options),
                                 "alive": len(tour.alive), "done": tour.done,
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


def summarise(records: list[dict], d: Dungeon) -> dict:
    """Closed-loop metrics of one run."""
    events = [(r["tick"], e) for r in records for e in r["events"]]

    def first(kind, **match):
        return next((t for t, e in events if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())), None)

    fw = [r["batch"]["forward_ms"] for r in records if r["batch"]["decisions"]]
    ticks = [r["batch"]["tick_ms"] for r in records]
    stuck = 0
    for r in records:
        s = r["stuck_streak"]
        stuck += 0 if s < Runner.STUCK else Runner.STUCK if s == Runner.STUCK else 1
    decisions = [x for r in records for x in r["decisions"]]
    return {
        "outcome": d.outcome, "cause": d.cause, "ticks": len(records), "gems": d.collected,
        "hits": sum(1 for _, e in events if e["kind"] == "hit"),
        "food_eaten": sum(1 for _, e in events if e["kind"] == "food"),
        "potions_drunk": sum(1 for _, e in events if e["kind"] == "potion"),
        "bumps": sum(1 for _, e in events if e["kind"] == "bump"),
        "rooms_seen": len(d.seen), "key_seen_tick": first("room_seen", room=d.key_room) if d.key_room != d.start_room else 0,
        "key_tick": first("key"), "exit_seen_tick": first("room_seen", room=d.exit_room), "escape_tick": first("escaped"),
        "min_energy": min((r["world"]["agent"]["energy"] for r in records), default=d.energy),
        "min_health": min([r["world"]["agent"]["health"] for r in records] + [d.health]),
        "stuck_ticks": stuck,
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
