"""Two policies without a model, to calibrate the rules: NOT the demo's agent.

- ``reference``: a hand-written bot (breadth-first search around the ghouls' reach) that sees only
  what the agent's prompts describe: known cells and the ghouls of seen rooms. Its escape rate is
  the ceiling that shows the level can be beaten. ``Rules`` were tuned against it before any model
  run. It ignores gems and uses the dash only to get out of reach.
- ``random_policy``: a uniformly random move or dash: the floor.

The threat geometry here is shared with the brain, so the bot and the move labels agree.
"""

from __future__ import annotations

import random

from .world import DASHES, DIRS, MOVES, STEP, Cell, Dungeon, Ghoul


def ghoul_reach(d: Dungeon, g: Ghoul) -> set[Cell]:
    """Cells where ghoul ``g`` hits the agent at the end of this tick: next to it, unless it is
    asleep (a woken ghoul acts from the next tick on), backing off or tired."""
    if not g.awake or g.rest > 0 or g.tired > 0:
        return set()
    return {(g.pos[0] + dx, g.pos[1] + dy) for dx, dy in DIRS.values()}


def threat_cells(d: Dungeon) -> set[Cell]:
    """Cells a ghoul the agent knows of can hit next tick, and the ghouls' own cells."""
    out: set[Cell] = set()
    for g in d.known_ghouls():
        reach = ghoul_reach(d, g)
        if reach:
            out |= reach | {g.pos}
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
    blocked = (avoid | {g.pos for g in d.ghouls}) - {goal}
    dist = d.distances(goal, within=((known | {goal}) - blocked) | {d.pos})
    here = dist.get(d.pos)
    if here is None:
        return None
    for move in MOVES[:4]:
        dx, dy = STEP[move]
        c = (d.pos[0] + dx, d.pos[1] + dy)
        if dist.get(c, 10 ** 6) < here and d.passable(c):
            return move
    return "stay"


def reference(d: Dungeon, see_threats: bool = True) -> str:
    """The next move or dash. ``see_threats=False`` ignores the ghouls (a careless player)."""
    known = d.known_cells()
    threat = threat_cells(d) if see_threats else set()
    dist = d.distances(d.pos, within=known)
    goal: Cell | None = None
    potions = [c for c in d.potions if c in dist]
    if potions and d.health <= 40:
        goal = min(potions, key=lambda c: dist[c])
    elif d.has_key and d.exit in dist:
        goal = d.exit
    elif d.key is not None and d.key in dist:
        goal = d.key
    else:
        doors = frontier(d, dist)
        if doors:
            goal = min(doors, key=lambda c: dist[c])
    move = _path_step(d, goal, threat, known) if goal is not None and goal != d.pos else None
    if move in (None, "stay") and d.pos in threat:  # out of reach: dash if it helps, else step
        options = ([m for m in DASHES if d.can_dash()] + list(MOVES[:4]))
        for m in options:
            path = d.dash_cells(m) if m.startswith("dash") else [(d.pos[0] + STEP[m][0], d.pos[1] + STEP[m][1])]
            if path and d.passable(path[-1]) and d.ghoul_at(path[-1]) is None and path[-1] not in threat:
                return m
    return move or "stay"


def random_policy(rng: random.Random):
    def policy(d: Dungeon) -> str:
        return rng.choice(MOVES + DASHES)
    return policy


def play(d: Dungeon, policy) -> Dungeon:
    while d.outcome is None:
        d.step(policy(d))
    return d
