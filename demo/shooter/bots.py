"""Two policies without a model, to calibrate the rules: NOT the demo's agent.

- ``reference``: a hand-written bot (breadth-first search plus a simple aim rule) that sees only
  what the agent's prompts describe: known cells, visible enemies, flying bullets. Its win rate is
  the ceiling that shows the level can be beaten. It was used to tune ``Rules`` before any model run.
- ``random_policy``: a uniformly random move and a random choice among the shots on offer
  (holding fire included): the floor.
"""

from __future__ import annotations

import math
import random
from collections import deque

from .world import DIRS, MOVES, STEP, Cell, Dungeon


def shot_options(d: Dungeon) -> list[int]:
    """Enemies the agent can shoot now: visible, with a clear line, and the gun ready."""
    return [e.id for e in d.visible_enemies()] if d.cooldown == 0 else []


def fighting(d: Dungeon) -> list:
    """Awake enemies in the room the agent is in (or sealed in), or that can see it."""
    room = d.inside(d.pos) or d.room_at(d.pos)
    return [e for e in d.living() if e.awake and ((room is not None and e.home == room.id) or d.sealed == e.home)]


def threat_cells(d: Dungeon) -> set[Cell]:
    """Cells an enemy can hurt next tick: bullet paths, and cells next to an active brute."""
    out = set(d.danger())
    for e in d.living():
        if e.kind == "brute" and e.awake and e.timer == 0:
            out |= {(e.pos[0] + dx, e.pos[1] + dy) for dx, dy in DIRS.values()} | {e.pos}
    return out


def frontier(d: Dungeon, dist: dict[Cell, int]) -> list[Cell]:
    """The middle doorway cell of each unseen room behind a known corridor."""
    out = []
    for cor in d.corridors:
        for rid, door, other in ((cor.rooms[0], cor.doors[0], cor.rooms[1]), (cor.rooms[1], cor.doors[1], cor.rooms[0])):
            mid = door[len(door) // 2]
            if rid not in d.seen and other in d.seen and mid in dist:
                out.append(mid)
    return out


def _path_step(d: Dungeon, goal: Cell, avoid: set[Cell], known: set[Cell]) -> str | None:
    """The first move of a shortest path to ``goal`` over known cells that avoids ``avoid``."""
    blocked = avoid | {e.pos for e in d.living()}
    blocked.discard(goal)
    allowed = (known | {goal}) - blocked
    dist = d.distances(goal, within=allowed | {d.pos})
    here = dist.get(d.pos)
    if here is None:
        return None
    for move in MOVES[:4]:
        dx, dy = STEP[move]
        c = (d.pos[0] + dx, d.pos[1] + dy)
        if dist.get(c, 10 ** 6) < here and d.passable(c):
            return move
    return "stay"


def reference(d: Dungeon, see_threats: bool = True) -> tuple[str, int | None]:
    """``see_threats=False`` ignores bullets and brutes when choosing where to stand and how to walk."""
    known = d.known_cells()
    shots = shot_options(d)
    shoot = shots[0] if shots else None
    threat = threat_cells(d) if see_threats else set()
    dist = d.distances(d.pos, within=known)
    foes = fighting(d)
    goal: Cell | None = None
    if foes:
        room = d.rooms[foes[0].home]
        spots = []
        for c in room.cells:
            if c in threat or c not in dist or d.enemy_at(c):
                continue
            near = min(math.dist(c, e.pos) for e in foes)
            if near < 3:
                continue
            if any(d.clear_line(c, e.pos) for e in foes):
                spots.append((dist[c], -near, c))
        if spots:
            goal = min(spots)[2]
    if goal is None and not foes:
        potions = [c for c in d.potions if c in dist]
        if potions and d.health <= 60:
            goal = min(potions, key=lambda c: dist[c])
        elif d.has_key and d.exit in dist:
            goal = d.exit
        elif d.key is not None and d.key in dist:
            goal = d.key
        else:
            doors = frontier(d, dist)
            if doors:
                goal = min(doors, key=lambda c: dist[c])
    move = None
    if goal is not None and goal != d.pos:
        move = _path_step(d, goal, threat, known)
    if move is None or move == "stay":
        move = "stay"
        if d.pos in threat:  # step out of harm's way
            for m in MOVES[:4]:
                dx, dy = STEP[m]
                c = (d.pos[0] + dx, d.pos[1] + dy)
                if d.passable(c) and c not in threat and not d.enemy_at(c):
                    move = m
                    break
    return move, shoot


def random_policy(rng: random.Random):
    def policy(d: Dungeon) -> tuple[str, int | None]:
        shots = shot_options(d)
        return rng.choice(MOVES), rng.choice(shots + [None])
    return policy


def play(d: Dungeon, policy) -> Dungeon:
    while d.outcome is None:
        d.step(*policy(d))
    return d
