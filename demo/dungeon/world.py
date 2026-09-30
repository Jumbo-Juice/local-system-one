"""The dungeon: a seeded grid world for one agent without a weapon. No model code here.

Rebuilt on 2026-09-30. The first dungeon (one-cell corridors, enemies that chased anywhere, hunger)
was never escaped in 48 model runs; it is gone, and its results stay in docs/research.md. Rules of
this one (implementation choices; the values in ``Rules`` were tuned only against the non-model
bots in bots.py, on dev seeds 1000-1059, and frozen before any model run):

- Nine rooms in a 3x3 layout, joined by straight corridors three cells wide. The corridors follow
  a random spanning tree, then more are added until every room has at least ``min_doors``
  doorways, plus ``extra_links`` more. So no room is a dead end, no doorway is a single cell, and
  the agent can always leave a room by another way than it came in.
- The exit is in the room farthest from the start and opens only for an agent that carries the key.
  The key lies in another room. Reaching the exit with the key: "escaped".
- Ghouls guard rooms (never the start room; two in the key room). They cannot be fought. A ghoul
  sleeps until the agent steps inside its room; then every ghoul of that room wakes (acting from the
  next tick on) and chases the agent while it is in the room or its doorways, resting one tick in
  ``ghoul_move_every``. A ghoul next to the agent hits it, then backs off for ``ghoul_rest`` ticks.
  After ``ghoul_chase`` ticks of chasing it tires and walks back to its post for ``ghoul_tired``
  ticks, ignoring the agent. Ghouls never leave their room; when the agent has left, they walk back
  to their post and fall asleep there. (Chosen with the bots: without tiring, a ghoul shadowing the
  agent along a doorway from inside its room kept the careful bot out for good; 38 of 60 dev seeds
  timed out. With it the bot escapes 59 of 60.)
- Dash: instead of a step the agent can dash up to ``dash_cells`` cells in a straight line in one
  tick. It stops before a wall or a ghoul and picks up what lies on every cell it passes. Then it
  recharges for ``dash_recharge`` ticks. The agent always outruns a ghoul; with the dash, by far.
- Gems are a bonus. Health potions heal when stepped on. There is no hunger.
- The agent knows only the rooms it has seen (room-level fog of war): the rooms it stood in, their
  doorways, and the corridors out of them.
- Order within a tick: the agent moves or dashes and picks up what lies on its cells; ghouls wake
  and act; the dash recharges.

North is up: "move north" decreases y.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass

from ..common import DASHES, DIRS, MOVES, STEP, dash_path, is_dash  # noqa: F401 (re-exported)

ROOM_NAMES = ("hall", "library", "armoury", "crypt", "kitchen", "chapel", "vault", "cellar", "gallery")
COLS, ROWS, SLOT_W, SLOT_H = 3, 3, 11, 9  # each room lies inside its own SLOT_W x SLOT_H slot

Cell = tuple[int, int]


@dataclass(frozen=True)
class Rules:
    max_ticks: int = 400
    health: int = 100
    door_width: int = 3
    min_doors: int = 2  # doorways every room has at least
    extra_links: int = 1  # corridors added beyond those
    # ghouls
    room_ghouls: tuple[int, int] = (0, 1)  # per ordinary room, inclusive range
    key_room_ghouls: int = 2
    exit_room_ghouls: int = 1
    ghoul_damage: int = 20
    ghoul_rest: int = 3  # ticks a ghoul backs off after a hit
    ghoul_move_every: int = 3  # a ghoul rests one tick in this many
    ghoul_chase: int = 12  # ticks of chasing (in all, since it last rested) before a ghoul tires
    ghoul_tired: int = 10  # ticks a tired ghoul walks back to its post, ignoring the agent
    # supplies
    gems: int = 6
    potions: int = 3
    potion_health: int = 40
    # dash
    dash: bool = True
    dash_cells: int = 3  # cells a dash covers at most
    dash_recharge: int = 8  # ticks after a dash until the next one


@dataclass(frozen=True)
class Room:
    id: int
    name: str
    x0: int
    y0: int
    w: int
    h: int

    @property
    def cells(self) -> list[Cell]:
        return [(x, y) for y in range(self.y0, self.y0 + self.h) for x in range(self.x0, self.x0 + self.w)]

    def __contains__(self, c) -> bool:
        return self.x0 <= c[0] < self.x0 + self.w and self.y0 <= c[1] < self.y0 + self.h

    @property
    def centre(self) -> Cell:
        return self.x0 + self.w // 2, self.y0 + self.h // 2


@dataclass(frozen=True)
class Corridor:
    rooms: tuple[int, int]
    doors: tuple[tuple[Cell, ...], tuple[Cell, ...]]  # doors[i]: the doorway cells in the wall of rooms[i]
    cells: tuple[Cell, ...]  # between the two doorways; empty when the walls touch


@dataclass
class Ghoul:
    id: int
    pos: Cell
    home: int  # room id
    post: Cell  # where it sleeps
    awake: bool = False
    rest: int = 0  # ticks of backing off left after a hit
    dazed: bool = False  # woken this tick: acts from the next tick on
    chase: int = 0  # ticks chased since it last rested (hit the agent or tired)
    tired: int = 0  # ticks left walking home, ignoring the agent


class Dungeon:
    def __init__(self, seed: int = 0, rules: Rules | None = None):
        self.seed, self.rules = seed, rules or Rules()
        self.rng = random.Random(seed)
        self.width, self.height = COLS * SLOT_W, ROWS * SLOT_H
        self.tick = 0
        self.outcome: str | None = None  # escaped | died | timeout
        self.cause: str | None = None  # for "died": ghoul
        self.health = self.rules.health
        self.collected = 0  # gems
        self.has_key = False
        self.dash_charge = 0  # ticks until the dash is ready again
        self.dashes = 0
        self._build_rooms()
        self._build_corridors()
        self._place()
        self.seen: set[int] = set()
        self._known: set[Cell] | None = None
        self._look([])

    # -- layout -------------------------------------------------------------------

    def _build_rooms(self) -> None:
        names = list(ROOM_NAMES)
        self.rng.shuffle(names)
        self.rooms: list[Room] = []
        for r in range(ROWS):
            for c in range(COLS):
                # Interiors 6-9 x 5-7 inside an 11 x 9 slot, walls included. Neighbouring rooms then
                # overlap by at least 3 rows (or columns), so every corridor can be 3 cells wide.
                w, h = self.rng.randint(6, SLOT_W - 2), self.rng.randint(5, SLOT_H - 2)
                x0 = c * SLOT_W + self.rng.randint(1, SLOT_W - 1 - w)
                y0 = r * SLOT_H + self.rng.randint(1, SLOT_H - 1 - h)
                self.rooms.append(Room(len(self.rooms), names[len(self.rooms)], x0, y0, w, h))

    def _links(self) -> list[tuple[int, int]]:
        """A random spanning tree, then links until every room has ``min_doors``, then extras."""
        edges = [(i, i + 1) for i in range(COLS * ROWS) if i % COLS < COLS - 1]
        edges += [(i, i + COLS) for i in range(COLS * (ROWS - 1))]
        linked, links = {self.rng.randrange(COLS * ROWS)}, []
        while len(linked) < COLS * ROWS:  # Prim-style
            e = self.rng.choice([e for e in edges if (e[0] in linked) != (e[1] in linked)])
            links.append(e)
            linked |= set(e)
        degree = lambda r: sum(r in e for e in links)  # noqa: E731
        for r in self.rng.sample(range(COLS * ROWS), COLS * ROWS):
            while degree(r) < self.rules.min_doors:
                spare = [e for e in edges if r in e and e not in links]
                if not spare:
                    break
                low = min(degree(e[0] if e[1] == r else e[1]) for e in spare)  # prefer a partner short of doors
                links.append(self.rng.choice([e for e in spare if degree(e[0] if e[1] == r else e[1]) == low]))
        spare = [e for e in edges if e not in links]
        return links + self.rng.sample(spare, min(self.rules.extra_links, len(spare)))

    def _build_corridors(self) -> None:
        wd = self.rules.door_width
        self.corridors: list[Corridor] = []
        for a, b in sorted(self._links()):
            A, B = self.rooms[a], self.rooms[b]
            if b == a + 1:  # B is east of A
                y = self.rng.randint(max(A.y0, B.y0), min(A.y0 + A.h, B.y0 + B.h) - wd)
                ys = range(y, y + wd)
                da, db = tuple((A.x0 + A.w, v) for v in ys), tuple((B.x0 - 1, v) for v in ys)
                cells = tuple((x, v) for v in ys for x in range(A.x0 + A.w + 1, B.x0 - 1))
            else:  # B is south of A
                x = self.rng.randint(max(A.x0, B.x0), min(A.x0 + A.w, B.x0 + B.w) - wd)
                xs = range(x, x + wd)
                da, db = tuple((v, A.y0 + A.h) for v in xs), tuple((v, B.y0 - 1) for v in xs)
                cells = tuple((v, y) for y in range(A.y0 + A.h + 1, B.y0 - 1) for v in xs)
            self.corridors.append(Corridor((a, b), (da, db), cells))
        self.area: dict[Cell, int] = {}  # interior and doorway cells -> room id
        for room in self.rooms:
            for c in room.cells:
                self.area[c] = room.id
        self.doors: dict[int, set[Cell]] = {r.id: set() for r in self.rooms}
        for cor in self.corridors:
            for rid, door in zip(cor.rooms, cor.doors):
                for c in door:
                    self.area[c] = rid
                    self.doors[rid].add(c)
        self.open: set[Cell] = set(self.area) | {c for cor in self.corridors for c in cor.cells}

    def room_distances(self, start: int) -> dict[int, int]:
        """Rooms reachable from ``start`` -> number of corridors on the shortest route."""
        adj: dict[int, list[int]] = {r.id: [] for r in self.rooms}
        for a, b in (cor.rooms for cor in self.corridors):
            adj[a].append(b)
            adj[b].append(a)
        dist, queue = {start: 0}, deque([start])
        while queue:
            r = queue.popleft()
            for n in adj[r]:
                if n not in dist:
                    dist[n] = dist[r] + 1
                    queue.append(n)
        return dist

    def _place(self) -> None:
        rng, n, R = self.rng, len(self.rooms), self.rules
        self.start_room = rng.randrange(n)
        dist = self.room_distances(self.start_room)
        far = max(dist.values())
        self.exit_room = rng.choice([r for r, d in dist.items() if d == far])
        self.key_room = rng.choice([r for r in range(n) if r not in (self.start_room, self.exit_room)])
        taken: set[Cell] = set()

        def free_cell(room: int, margin: int = 0) -> Cell:
            rm = self.rooms[room]
            cells = [c for c in rm.cells if c not in taken
                     and rm.x0 + margin <= c[0] < rm.x0 + rm.w - margin and rm.y0 + margin <= c[1] < rm.y0 + rm.h - margin]
            c = rng.choice(cells or [c for c in rm.cells if c not in taken])
            taken.add(c)
            return c

        self.pos = free_cell(self.start_room, margin=1)
        self.exit = free_cell(self.exit_room)
        self.key: Cell | None = free_cell(self.key_room)
        self.ghouls: list[Ghoul] = []
        for rid in range(n):
            if rid == self.start_room:
                continue
            k = (R.key_room_ghouls if rid == self.key_room else R.exit_room_ghouls if rid == self.exit_room
                 else rng.randint(*R.room_ghouls))
            for _ in range(k):
                c = free_cell(rid, margin=1)
                self.ghouls.append(Ghoul(len(self.ghouls), c, rid, c))
        others = [r for r in range(n) if r != self.start_room]
        self.gems = [free_cell(rng.randrange(n)) for _ in range(R.gems)]
        self.potions = [free_cell(rng.choice(others)) for _ in range(R.potions)]

    # -- geometry -------------------------------------------------------------------

    def passable(self, c: Cell) -> bool:
        return c in self.open

    def distances(self, start: Cell, within: set[Cell] | None = None) -> dict[Cell, int]:
        """Walking distance from ``start`` to every reachable open cell (optionally only through
        ``within``, e.g. the cells the agent knows)."""
        allowed = self.open if within is None else self.open & within
        dist, queue = {start: 0}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS.values():
                n = (x + dx, y + dy)
                if n not in dist and n in allowed:
                    dist[n] = dist[(x, y)] + 1
                    queue.append(n)
        return dist

    def known_cells(self) -> set[Cell]:
        """Cells the agent knows: seen rooms and their doorways, and corridors out of seen rooms
        up to and including the far doorway."""
        if self._known is None:
            known: set[Cell] = set()
            for rid in self.seen:
                known.update(self.rooms[rid].cells)
            for cor in self.corridors:
                if cor.rooms[0] in self.seen or cor.rooms[1] in self.seen:
                    known.update(cor.cells)
                    for door in cor.doors:
                        known.update(door)
            self._known = known
        return self._known

    def room_at(self, c: Cell) -> Room | None:
        rid = self.area.get(c)
        return None if rid is None else self.rooms[rid]

    def inside(self, c: Cell) -> Room | None:
        """The room whose interior holds ``c`` (doorways do not count)."""
        room = self.room_at(c)
        return room if room is not None and c in room else None

    def is_door(self, c: Cell) -> bool:
        return c in self.area and c not in self.rooms[self.area[c]]

    def ghoul_at(self, c: Cell) -> Ghoul | None:
        return next((g for g in self.ghouls if g.pos == c), None)

    def known_ghouls(self) -> list[Ghoul]:
        """The ghouls of the rooms the agent has seen."""
        return [g for g in self.ghouls if g.home in self.seen]

    # -- dynamics -------------------------------------------------------------------

    def step(self, move: str) -> list[dict]:
        """Apply the agent's move or dash, then pickups, ghouls and the dash's recharge. Returns events."""
        if self.outcome is not None:
            raise RuntimeError(f"the run is over ({self.outcome})")
        ev: list[dict] = []
        dx, dy = STEP.get(move, (0, 0))
        to = (self.pos[0] + dx, self.pos[1] + dy)
        if is_dash(move) and self.can_dash():
            self._dash(move, ev)
        elif (dx, dy) != (0, 0):
            if is_dash(move):
                ev.append({"kind": "dash_refused", "recharge": self.dash_charge})
            elif not self.passable(to):
                ev.append({"kind": "bump", "cell": list(to)})
            elif self.ghoul_at(to) is not None:
                ev.append({"kind": "blocked", "ghoul": self.ghoul_at(to).id})
            else:
                self.pos = to
                self._pickups(ev)
        self._look(ev)
        if self.outcome is None:
            self._ghouls_act(ev)
        self.dash_charge = max(0, self.dash_charge - 1)
        self.tick += 1
        if self.outcome is None and self.tick >= self.rules.max_ticks:
            self.outcome = "timeout"
            ev.append({"kind": "timeout"})
        return ev

    def can_dash(self) -> bool:
        return self.rules.dash and self.dash_charge == 0

    def dash_cells(self, move: str) -> list[Cell]:
        """The cells a dash in that direction would pass through now (the last one is where it lands)."""
        return dash_path(self.pos, move, self.rules.dash_cells, lambda c: self.passable(c) and self.ghoul_at(c) is None)

    def _dash(self, move: str, ev: list[dict]) -> None:
        path = self.dash_cells(move)
        if not path:
            ev.append({"kind": "bump", "cell": [self.pos[0] + STEP[move][0], self.pos[1] + STEP[move][1]]})
            return
        start = self.pos
        for c in path:  # pick up what lies on every cell passed; the exit ends the run at once
            self.pos = c
            self._pickups(ev)
            self._look(ev)
            if self.outcome is not None:
                break
        self.dash_charge = self.rules.dash_recharge
        self.dashes += 1
        ev.append({"kind": "dash", "from": list(start), "to": list(self.pos), "path": [list(c) for c in path]})

    def _pickups(self, ev: list[dict]) -> None:
        c = self.pos
        if c == self.key:
            self.key, self.has_key = None, True
            ev.append({"kind": "key", "cell": list(c)})
        if c in self.gems:
            self.gems.remove(c)
            self.collected += 1
            ev.append({"kind": "gem", "cell": list(c)})
        if c in self.potions:
            self.potions.remove(c)
            gain = min(self.rules.health, self.health + self.rules.potion_health) - self.health
            self.health += gain
            ev.append({"kind": "potion", "cell": list(c), "gain": gain})
        if c == self.exit:
            if self.has_key:
                self.outcome = "escaped"
                ev.append({"kind": "escaped"})
            else:
                ev.append({"kind": "exit_locked"})

    def _look(self, ev: list[dict]) -> None:
        rid = self.area.get(self.pos)
        if rid is not None and rid not in self.seen:
            self.seen.add(rid)
            self._known = None
            ev.append({"kind": "room_seen", "room": rid, "name": self.rooms[rid].name})
        room = self.inside(self.pos)
        if room is not None:
            sleeping = [g for g in self.ghouls if g.home == room.id and not g.awake]
            if sleeping:
                for g in sleeping:
                    g.awake, g.dazed, g.chase = True, True, 0
                ev.append({"kind": "ghouls_woke", "room": room.id, "ghouls": [g.id for g in sleeping]})

    def _ghouls_act(self, ev: list[dict]) -> None:
        R = self.rules
        for g in self.ghouls:
            if not g.awake:
                continue
            if g.dazed:  # woken this tick
                g.dazed = False
                continue
            room = self.rooms[g.home]
            moves = (self.tick + g.id) % R.ghoul_move_every != 0  # rests one tick in ghoul_move_every
            if g.rest > 0:
                g.rest -= 1
                g.pos = self._away_step(g, room)
                continue
            if g.tired > 0:
                g.tired -= 1
                if moves and g.pos != g.post:
                    g.pos = self._home_step(g, room)
                continue
            if abs(g.pos[0] - self.pos[0]) + abs(g.pos[1] - self.pos[1]) == 1:
                self._hurt(g, ev)
                g.rest, g.chase = R.ghoul_rest, 0
                continue
            here = self.area.get(self.pos) == g.home
            chasing = here
            g.chase += chasing  # all chasing since it last rested counts
            if g.chase >= R.ghoul_chase:
                g.tired, g.chase = R.ghoul_tired, 0
                ev.append({"kind": "ghoul_tired", "ghoul": g.id})
                continue
            if not moves:
                continue
            if chasing:
                g.pos = self._toward_step(g, room)
            elif g.pos != g.post:
                g.pos = self._home_step(g, room)
            elif not here:
                g.awake = False
                ev.append({"kind": "ghoul_slept", "ghoul": g.id})

    def _hurt(self, g: Ghoul, ev: list[dict]) -> None:
        self.health -= self.rules.ghoul_damage
        ev.append({"kind": "hit", "ghoul": g.id, "damage": self.rules.ghoul_damage})
        if self.health <= 0 and self.outcome is None:
            self.health, self.outcome, self.cause = 0, "died", "ghoul"
            ev.append({"kind": "died", "cause": "ghoul"})

    def _free(self, c: Cell, g: Ghoul, room: Room) -> bool:
        return c in room and c != self.pos and all(o.pos != c for o in self.ghouls if o is not g)

    def _neighbours(self, g: Ghoul, room: Room) -> list[Cell]:
        return [(g.pos[0] + dx, g.pos[1] + dy) for dx, dy in DIRS.values() if self._free((g.pos[0] + dx, g.pos[1] + dy), g, room)]

    def _descend(self, g: Ghoul, room: Room, goals: list[Cell]) -> Cell:
        """One step inside the room that shortens the walk to the nearest of ``goals``."""
        if not goals:
            return g.pos
        dist: dict[Cell, int] = {}
        for goal in goals:
            for c, s in self.distances(goal, within=set(room.cells)).items():
                dist[c] = min(dist.get(c, 10 ** 6), s)
        best, here = g.pos, dist.get(g.pos, 10 ** 6)
        for c in self._neighbours(g, room):
            if dist.get(c, 10 ** 6) < here:
                best, here = c, dist[c]
        return best

    def _toward_step(self, g: Ghoul, room: Room) -> Cell:
        """Toward a cell next to the agent, inside the room (the agent may stand in a doorway)."""
        return self._descend(g, room, [c for c in ((self.pos[0] + dx, self.pos[1] + dy) for dx, dy in DIRS.values())
                                       if c in room])

    def _home_step(self, g: Ghoul, room: Room) -> Cell:
        return self._descend(g, room, [g.post])

    def _away_step(self, g: Ghoul, room: Room) -> Cell:
        opts = self._neighbours(g, room)
        return max(opts + [g.pos], key=lambda c: (abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]), c))

    # -- records ---------------------------------------------------------------------

    def layout(self) -> dict:
        """Static map for the trace header."""
        return {
            "width": self.width, "height": self.height, "seed": self.seed, "rules": asdict(self.rules),
            "rooms": [asdict(r) for r in self.rooms],
            "corridors": [{"rooms": list(c.rooms), "doors": [[list(x) for x in d] for d in c.doors],
                           "cells": [list(x) for x in c.cells]} for c in self.corridors],
            "exit": list(self.exit), "start_room": self.start_room, "key_room": self.key_room,
            "exit_room": self.exit_room,
        }

    def snapshot(self) -> dict:
        """Dynamic state for one trace record."""
        return {
            "agent": {"pos": list(self.pos), "health": self.health, "gems": self.collected, "has_key": self.has_key,
                      "dash": self.dash_charge, "dashes": self.dashes},
            "ghouls": [{"id": g.id, "pos": list(g.pos), "home": g.home, "awake": g.awake, "resting": g.rest > 0,
                        "tired": g.tired}
                       for g in self.ghouls],
            "items": {"gems": [list(c) for c in self.gems], "potions": [list(c) for c in self.potions],
                      "key": list(self.key) if self.key else None},
            "seen": sorted(self.seen),
        }
