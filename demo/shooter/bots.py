"""Two policies without a model, to calibrate the rules: NOT the demo's agent.

- ``reference``: a hand-written bot (breadth-first search plus a simple aim rule) that sees only
  what the agent's prompts describe: known cells, visible enemies, flying bullets. Its win rate is
  the ceiling that shows the level can be beaten. It was used to tune ``Rules`` before any model run.
- ``random_policy``: a uniformly random move and a random choice among the shots on offer
  (holding fire included, and reloading when it is possible): the floor.

With ammo (``Rules.ammo``) a policy returns ``(move, shot, reload)``. The reference bot reloads
when no enemy is in sight and fetches known ammo boxes that fit its reserve; with
``manage_ammo=False`` it reloads only an empty gun and never walks to a box on purpose.
"""

from __future__ import annotations

import math
import random
from collections import deque

from .world import DIRS, MOVES, STEP, Cell, Dungeon


def shot_options(d: Dungeon) -> list[int]:
    """Enemies the agent can shoot now: visible, with a clear line, and the gun ready (and loaded)."""
    return [e.id for e in d.visible_enemies()] if d.cooldown == 0 and d.can_fire() else []


def fighting(d: Dungeon) -> list:
    """Awake enemies of the room the agent is in (or sealed in). Otherwise, in a corridor, the awake
    enemies of the room it leads to (the one with the nearest such enemy): brutes that guard a
    doorway from inside their room block the way without the agent ever being inside it (found
    after the evaluation: 1.5B, seed 3, 389 stuck ticks in the corridor, never offered a fight).
    By position, not line of sight: with line of sight the bot flip-flopped between two cells."""
    room = d.inside(d.pos) or d.room_at(d.pos)
    here = [e for e in d.living() if e.awake and ((room is not None and e.home == room.id) or d.sealed == e.home)]
    if here:
        return here
    near = [e for e in d.living() if e.awake and e.home in d.seen and d.pos in fight_cells(d, e.home)]
    if not near:
        return []
    home = min(near, key=lambda e: (math.dist(d.pos, e.pos), e.id)).home
    return [e for e in near if e.home == home]


def fight_cells(d: Dungeon, rid: int) -> set[Cell]:
    """Where the agent may stand to fight room ``rid``'s enemies: the room, its doorways, and the
    corridors out of it (so that a doorway guarded from inside can be shot through)."""
    out = set(d.rooms[rid].cells)
    for cor in d.corridors:
        if rid in cor.rooms:
            out.update(cor.cells, *cor.doors)
    return out


def brute_reach(d: Dungeon, e) -> set[Cell]:
    """Cells where brute ``e`` hits the agent at the end of this tick: next to it, unless it is
    resting. A sleeping brute counts only inside its room, because stepping in wakes the room before
    enemies act (found after the evaluation: all 4 hits taken after a move labelled safe)."""
    if e.kind != "brute" or e.timer > 0:
        return set()
    near = {(e.pos[0] + dx, e.pos[1] + dy) for dx, dy in DIRS.values()}
    return near if e.awake else {c for c in near if c in d.rooms[e.home]}


def threat_cells(d: Dungeon) -> set[Cell]:
    """Cells an enemy can hurt next tick: bullet paths, and cells a brute can hit (see brute_reach)."""
    out = set(d.danger())
    for e in d.living():
        reach = brute_reach(d, e)
        if reach:
            out |= reach | {e.pos}
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


def reference(d: Dungeon, see_threats: bool = True, manage_ammo: bool = True):
    """``see_threats=False`` ignores bullets and brutes when choosing where to stand and how to walk.
    Returns ``(move, shot)``, or ``(move, shot, reload)`` with ammo."""
    R = d.rules
    known = d.known_cells()
    shots = shot_options(d)
    shoot = shots[0] if shots else None
    reload = False
    if R.ammo and d.can_reload() and (d.loaded == 0 or (manage_ammo and not d.visible_enemies())):
        shoot, reload = None, True
    threat = threat_cells(d) if see_threats else set()
    dist = d.distances(d.pos, within=known)
    foes = fighting(d)
    boxes = [c for c in d.ammo if c in dist] if R.ammo and manage_ammo else []
    goal: Cell | None = None
    if foes and boxes and d.ammo_total() == 0:  # dry in a fight: a box within reach first
        goal = min(boxes, key=lambda c: dist[c])
    elif foes:
        spots = []
        for c in fight_cells(d, foes[0].home):
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
        elif boxes and d.box_gain() == R.ammo_box:  # the whole box fits in the reserve
            goal = min(boxes, key=lambda c: dist[c])
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
    return (move, shoot, reload) if R.ammo else (move, shoot)


def random_policy(rng: random.Random):
    def policy(d: Dungeon):
        shots = shot_options(d)
        if not d.rules.ammo:
            return rng.choice(MOVES), rng.choice(shots + [None])
        pick = rng.choice(shots + [None] + (["reload"] if d.can_reload() else []))
        return rng.choice(MOVES), (None if pick == "reload" else pick), pick == "reload"
    return policy


def play(d: Dungeon, policy) -> Dungeon:
    while d.outcome is None:
        d.step(*policy(d))
    return d
