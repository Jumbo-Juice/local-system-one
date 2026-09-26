"""The dungeon: a seeded grid world for one agent. No model code here.

Rules (implementation choices, fixed before the first model run):
- Nine rooms of random size in a 3x3 layout, joined by straight corridors along a random
  spanning tree plus one extra link, so most maps have a loop.
- The exit is in the room farthest from the start; the key is in another room. The exit opens
  only for an agent that carries the key. Reaching it with the key ends the run: "escaped".
- Enemies patrol their home room. When the agent is within 6 steps they chase it, moving on two
  of every three ticks (the agent can move every tick). An enemy next to the agent hits it for
  30 health instead of moving, then is stunned for 3 ticks. Walking into an enemy also costs a
  hit. One enemy lives in the key room.
- Energy drops by 1 per tick. At 0 the agent loses 5 health per tick. Food gives +40 energy,
  potions +40 health, both capped at 100. Items are used when stepped on and do not respawn.
- The agent knows only the rooms it has seen: the room it stands in (or in whose doorway it
  stands), plus the corridors leading out of seen rooms (room-level fog of war).
- The run ends when the agent escapes, dies, or reaches the tick limit.

North is up: "move north" decreases y.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass

MOVES = ("move north", "move south", "move east", "move west", "stay")
STEP = {"move north": (0, -1), "move south": (0, 1), "move east": (1, 0), "move west": (-1, 0), "stay": (0, 0)}
DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
ROOM_NAMES = ("hall", "library", "armoury", "crypt", "kitchen", "chapel", "vault", "cellar", "gallery")
COLS, ROWS, SLOT_W, SLOT_H = 3, 3, 8, 7  # each room lies inside its own SLOT_W x SLOT_H slot

Cell = tuple[int, int]


@dataclass(frozen=True)
class Rules:
    max_ticks: int = 400
    energy: int = 100  # at the start
    food_energy: int = 40
    potion_health: int = 40
    enemy_damage: int = 30
    starve_damage: int = 5
    stun_ticks: int = 3
    chase_radius: int = 6  # walking distance at which an enemy starts chasing
    sight: int = 7  # the agent sees an enemy on a known cell within this walking distance
    gems: int = 12
    food: int = 3
    potions: int = 2
    enemies: int = 2
    extra_links: int = 1


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
    doors: tuple[Cell, Cell]  # doors[i] is in the wall of rooms[i]
    cells: tuple[Cell, ...]  # between the two doors; empty when the walls touch


@dataclass
class Enemy:
    id: int
    pos: Cell
    home: int  # room id
    mode: str = "patrol"  # patrol | chase | return | stunned
    stunned: int = 0


class Dungeon:
    def __init__(self, seed: int = 0, rules: Rules | None = None):
        self.seed, self.rules = seed, rules or Rules()
        self.rng = random.Random(seed)
        self.width, self.height = COLS * SLOT_W, ROWS * SLOT_H
        self.tick = 0
        self.outcome: str | None = None  # escaped | died | timeout
        self.cause: str | None = None  # for "died": enemy | starvation
        self.health, self.energy = 100, self.rules.energy
        self.collected = 0  # gems
        self.has_key = False
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
                # Interior sizes 4-6 x 3-5 inside an 8 x 7 slot, walls included. Any two
                # neighbouring rooms then overlap in rows (or columns), so corridors are straight.
                w, h = self.rng.randint(4, SLOT_W - 2), self.rng.randint(3, SLOT_H - 2)
                x0 = c * SLOT_W + self.rng.randint(1, SLOT_W - 1 - w)
                y0 = r * SLOT_H + self.rng.randint(1, SLOT_H - 1 - h)
                self.rooms.append(Room(len(self.rooms), names[len(self.rooms)], x0, y0, w, h))

    def _build_corridors(self) -> None:
        edges = [(i, i + 1) for i in range(COLS * ROWS) if i % COLS < COLS - 1]
        edges += [(i, i + COLS) for i in range(COLS * (ROWS - 1))]
        linked, tree = {self.rng.randrange(COLS * ROWS)}, []
        while len(linked) < COLS * ROWS:  # random spanning tree (Prim-style)
            e = self.rng.choice([e for e in edges if (e[0] in linked) != (e[1] in linked)])
            tree.append(e)
            linked |= set(e)
        spare = [e for e in edges if e not in tree]
        links = tree + self.rng.sample(spare, min(self.rules.extra_links, len(spare)))
        self.corridors: list[Corridor] = []
        for a, b in sorted(links):
            A, B = self.rooms[a], self.rooms[b]
            if b == a + 1:  # B is east of A
                y = self.rng.randint(max(A.y0, B.y0), min(A.y0 + A.h, B.y0 + B.h) - 1)
                da, db = (A.x0 + A.w, y), (B.x0 - 1, y)
                cells = tuple((x, y) for x in range(da[0] + 1, db[0]))
            else:  # B is south of A
                x = self.rng.randint(max(A.x0, B.x0), min(A.x0 + A.w, B.x0 + B.w) - 1)
                da, db = (x, A.y0 + A.h), (x, B.y0 - 1)
                cells = tuple((x, y) for y in range(da[1] + 1, db[1]))
            self.corridors.append(Corridor((a, b), (da, db), cells))
        self.area: dict[Cell, int] = {}  # interior and door cells -> room id
        for room in self.rooms:
            for c in room.cells:
                self.area[c] = room.id
        for cor in self.corridors:
            for rid, door in zip(cor.rooms, cor.doors):
                self.area[door] = rid
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
        rng, n = self.rng, len(self.rooms)
        self.start_room = rng.randrange(n)
        dist = self.room_distances(self.start_room)
        far = max(dist.values())
        self.exit_room = rng.choice([r for r, d in dist.items() if d == far])
        self.key_room = rng.choice([r for r in range(n) if r not in (self.start_room, self.exit_room)])
        taken: set[Cell] = set()

        def free_cell(room: int) -> Cell:
            c = rng.choice([c for c in self.rooms[room].cells if c not in taken])
            taken.add(c)
            return c

        self.pos = free_cell(self.start_room)
        self.exit = free_cell(self.exit_room)
        self.key: Cell | None = free_cell(self.key_room)
        others = [r for r in range(n) if r != self.start_room]
        k = self.rules.enemies
        homes = ([self.key_room] + rng.sample([r for r in others if r != self.key_room], max(0, k - 1)))[:k]
        self.enemies = [Enemy(i, free_cell(home), home) for i, home in enumerate(homes)]
        self.gems = [free_cell(rng.randrange(n)) for _ in range(self.rules.gems)]
        self.food = [free_cell(rng.choice(others)) for _ in range(self.rules.food)]
        self.potions = [free_cell(rng.choice(others)) for _ in range(self.rules.potions)]

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
        """Cells the agent knows: seen rooms and their doors, and corridors out of seen rooms
        up to and including the far door."""
        if self._known is None:
            known: set[Cell] = set()
            for rid in self.seen:
                known.update(self.rooms[rid].cells)
            for cor in self.corridors:
                if cor.rooms[0] in self.seen or cor.rooms[1] in self.seen:
                    known.update(cor.cells)
                    known.update(cor.doors)
            self._known = known
        return self._known

    def room_at(self, c: Cell) -> Room | None:
        rid = self.area.get(c)
        return None if rid is None else self.rooms[rid]

    def is_door(self, c: Cell) -> bool:
        return c in self.area and c not in self.rooms[self.area[c]]

    def enemy_at(self, c: Cell) -> Enemy | None:
        return next((e for e in self.enemies if e.pos == c), None)

    def visible_enemies(self) -> list[tuple[Enemy, int]]:
        """Enemies the agent can see, with their walking distance, nearest first."""
        known, dist = self.known_cells(), self.distances(self.pos)
        out = [(e, dist[e.pos]) for e in self.enemies
               if e.pos in known and dist.get(e.pos, 10 ** 6) <= self.rules.sight]
        return sorted(out, key=lambda x: (x[1], x[0].id))

    # -- dynamics -------------------------------------------------------------------

    def step(self, move: str) -> list[dict]:
        """Apply the agent's move, then pickups, sight, enemies, energy. Returns events."""
        if self.outcome is not None:
            raise RuntimeError(f"the run is over ({self.outcome})")
        ev: list[dict] = []
        dx, dy = STEP.get(move, (0, 0))
        to = (self.pos[0] + dx, self.pos[1] + dy)
        if (dx, dy) != (0, 0):
            enemy = self.enemy_at(to)
            if not self.passable(to):
                ev.append({"kind": "bump", "cell": list(to)})
            elif enemy is not None:
                if enemy.stunned:
                    ev.append({"kind": "blocked", "enemy": enemy.id})
                else:  # walking into an enemy
                    self._hit(enemy, ev)
            else:
                self.pos = to
                self._pickups(ev)
        self._look(ev)
        if self.outcome is None:
            self._enemies_act(ev)
            self.energy = max(0, self.energy - 1)
            if self.energy == 0:
                self.health -= self.rules.starve_damage
                ev.append({"kind": "starving", "damage": self.rules.starve_damage})
                if self.health <= 0:
                    self.cause = self.cause or "starvation"
            if self.health <= 0:
                self.health = 0
                self.outcome = "died"
                ev.append({"kind": "died", "cause": self.cause})
        self.tick += 1
        if self.outcome is None and self.tick >= self.rules.max_ticks:
            self.outcome = "timeout"
            ev.append({"kind": "timeout"})
        return ev

    def _hit(self, enemy: Enemy, ev: list[dict]) -> None:
        self.health -= self.rules.enemy_damage
        enemy.stunned, enemy.mode = self.rules.stun_ticks, "stunned"
        ev.append({"kind": "hit", "enemy": enemy.id, "damage": self.rules.enemy_damage})
        if self.health <= 0:
            self.cause = "enemy"

    def _pickups(self, ev: list[dict]) -> None:
        c = self.pos
        if c == self.key:
            self.key, self.has_key = None, True
            ev.append({"kind": "key", "cell": list(c)})
        if c in self.gems:
            self.gems.remove(c)
            self.collected += 1
            ev.append({"kind": "gem", "cell": list(c)})
        if c in self.food:
            self.food.remove(c)
            gain = min(100, self.energy + self.rules.food_energy) - self.energy
            self.energy += gain
            ev.append({"kind": "food", "cell": list(c), "gain": gain})
        if c in self.potions:
            self.potions.remove(c)
            gain = min(100, self.health + self.rules.potion_health) - self.health
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

    def _enemies_act(self, ev: list[dict]) -> None:
        to_agent = self.distances(self.pos)
        for e in self.enemies:
            if e.stunned:
                e.stunned -= 1
                e.mode = "stunned"
                continue
            d = to_agent.get(e.pos)
            if d is not None and d <= self.rules.chase_radius:
                e.mode = "chase"
                if self.tick % 3 == 2:  # rests one tick in three
                    continue
                if d == 1:
                    self._hit(e, ev)
                    continue
                e.pos = self._step_down(e, to_agent)
            elif e.pos not in self.rooms[e.home]:
                e.mode = "return"
                if self.tick % 3 == 0:
                    e.pos = self._step_down(e, self.distances(self.rooms[e.home].centre))
            else:
                e.mode = "patrol"
                if self.tick % 3 == 0:
                    blocked = {self.pos, self.exit, self.key} | {o.pos for o in self.enemies}
                    steps = [(e.pos[0] + dx, e.pos[1] + dy) for dx, dy in DIRS.values()]
                    steps = [c for c in steps if c in self.rooms[e.home] and c not in blocked]
                    e.pos = self.rng.choice(steps + [e.pos])

    def _step_down(self, e: Enemy, dist: dict[Cell, int]) -> Cell:
        """One step that lowers ``dist`` (first in DIRS order), avoiding other enemies and the agent."""
        here = dist.get(e.pos)
        if here is None:
            return e.pos
        taken = {o.pos for o in self.enemies if o is not e} | {self.pos}
        for dx, dy in DIRS.values():
            c = (e.pos[0] + dx, e.pos[1] + dy)
            if dist.get(c) == here - 1 and c not in taken:
                return c
        return e.pos

    # -- records ---------------------------------------------------------------------

    def layout(self) -> dict:
        """Static map for the trace header."""
        return {
            "width": self.width, "height": self.height, "seed": self.seed, "rules": asdict(self.rules),
            "rooms": [asdict(r) for r in self.rooms],
            "corridors": [{"rooms": list(c.rooms), "doors": [list(d) for d in c.doors],
                           "cells": [list(x) for x in c.cells]} for c in self.corridors],
            "exit": list(self.exit), "start_room": self.start_room, "key_room": self.key_room,
            "exit_room": self.exit_room,
        }

    def snapshot(self) -> dict:
        """Dynamic state for one trace record."""
        return {
            "agent": {"pos": list(self.pos), "health": self.health, "energy": self.energy,
                      "gems": self.collected, "has_key": self.has_key},
            "enemies": [{"id": e.id, "pos": list(e.pos), "mode": e.mode, "stunned": e.stunned} for e in self.enemies],
            "items": {"gems": [list(c) for c in self.gems], "food": [list(c) for c in self.food],
                      "potions": [list(c) for c in self.potions], "key": list(self.key) if self.key else None},
            "seen": sorted(self.seen),
        }
