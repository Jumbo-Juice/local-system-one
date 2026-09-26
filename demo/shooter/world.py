"""The shooter: a seeded grid dungeon for one agent with a gun. No model code here.

A room-clearing shooter in the spirit of Enter the Gungeon, kept discrete so that every
decision can be a short list of text options. Rules (implementation choices; the values in
``Rules`` were tuned only against the non-model bots in bots.py and frozen before any model run):

- Nine rooms in a 3x3 layout, joined by straight corridors three cells wide along a random
  spanning tree plus one extra link. Doorways are three cells wide, so no single enemy can block
  one (the first dungeon's one-cell doors trapped the agent).
- The exit is in the room farthest from the start and opens only for an agent that carries the key.
  The key lies in another room, guarded by more enemies. Reaching the exit with the key: "escaped".
- Every room except the start holds enemies. They wake when the agent steps inside their room or
  shoots one of them. While the agent is inside a room with living enemies, that room's doorways are
  sealed (walls for walking and bullets) until every enemy in it is dead: fights happen in open
  rooms, not in corridors. Enemies never leave their room.
- The agent's gun fires one bullet every ``cooldown`` ticks at the enemy the agent chooses. Bullets
  fly in a straight line toward where the enemy stood when the shot was fired, ``shot_speed``
  cells per tick, and stop at the first wall or enemy. A moving enemy can be missed.
- Gunners keep their distance and fire a slow bullet at the agent every ``gunner_period`` ticks,
  aiming visibly for one tick first. Their bullets fly ``enemy_shot_speed`` cells per tick, so the
  agent can step out of the way. Brutes walk at the agent and hit it when adjacent, then back off
  for ``brute_rest`` ticks.
- Health potions heal when stepped on. There is no hunger.
- The agent knows only the rooms it has seen (room-level fog of war, as in the first dungeon).
- Order within a tick: the agent moves, then fires; bullets fly; enemies act; doorways seal or open.

North is up: "move north" decreases y. Positions of bullets are fractional cell coordinates; a
bullet is in the cell its position rounds to.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass

MOVES = ("move north", "move south", "move east", "move west", "stay")
STEP = {"move north": (0, -1), "move south": (0, 1), "move east": (1, 0), "move west": (-1, 0), "stay": (0, 0)}
DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
ROOM_NAMES = ("hall", "library", "armoury", "crypt", "kitchen", "chapel", "vault", "cellar", "gallery")
COLS, ROWS, SLOT_W, SLOT_H = 3, 3, 11, 9  # each room lies inside its own SLOT_W x SLOT_H slot
SUBSTEP = 0.25  # bullets are moved and checked in steps of this many cells

Cell = tuple[int, int]


@dataclass(frozen=True)
class Rules:
    max_ticks: int = 400
    health: int = 100
    door_width: int = 3
    extra_links: int = 1
    seal: bool = True  # doorways close while the agent is inside a room whose enemies live
    # the agent's gun
    cooldown: int = 2  # ticks between shots (1 = every tick)
    shot_speed: float = 3.0  # cells per tick
    shot_damage: int = 1
    # enemies
    room_enemies: tuple[int, int] = (1, 2)  # per ordinary room, inclusive range
    key_room_enemies: int = 3
    exit_room_enemies: int = 2
    brute_share: float = 0.3  # chance that an enemy is a brute instead of a gunner
    gunner_hp: int = 3
    gunner_period: int = 5  # ticks between a gunner's shots; it aims on the tick before firing
    gunner_range: float = 9.0  # a gunner fires only at an agent this close (straight line)
    gunner_move_every: int = 2  # a gunner moves on one tick in this many
    enemy_shot_speed: float = 1.0
    enemy_shot_damage: int = 15
    brute_hp: int = 4
    brute_damage: int = 20
    brute_rest: int = 3  # ticks a brute backs off after a hit
    brute_move_every: int = 3  # a brute rests one tick in this many
    # supplies
    potions: int = 3
    potion_health: int = 40


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
class Enemy:
    id: int
    kind: str  # gunner | brute
    pos: Cell
    home: int  # room id
    hp: int
    awake: bool = False
    timer: int = 0  # gunner: ticks until it fires; brute: ticks of rest left
    aiming: bool = False  # gunner: fires at the end of this tick

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Bullet:
    id: int
    owner: str  # agent | enemy
    x: float
    y: float
    vx: float  # cells per tick
    vy: float
    damage: int
    source: int | None = None  # enemy id for enemy bullets


def rnd(v: float) -> int:
    return math.floor(v + 0.5)


class Dungeon:
    def __init__(self, seed: int = 0, rules: Rules | None = None):
        self.seed, self.rules = seed, rules or Rules()
        self.rng = random.Random(seed)
        self.width, self.height = COLS * SLOT_W, ROWS * SLOT_H
        self.tick = 0
        self.outcome: str | None = None  # escaped | died | timeout
        self.cause: str | None = None  # for "died": gunner | brute
        self.health = self.rules.health
        self.has_key = False
        self.cooldown = 0  # ticks until the gun can fire again
        self.bullets: list[Bullet] = []
        self._bullet_ids = 0
        self.sealed: int | None = None  # the room whose doorways are closed, if any
        self.cleared: set[int] = set()
        self.shots = self.kills = 0
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
        wd = self.rules.door_width
        self.corridors: list[Corridor] = []
        for a, b in sorted(links):
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
        self.enemies: list[Enemy] = []
        for rid in range(n):
            if rid == self.start_room:
                continue
            k = (R.key_room_enemies if rid == self.key_room else R.exit_room_enemies if rid == self.exit_room
                 else rng.randint(*R.room_enemies))
            for _ in range(k):
                kind = "brute" if rng.random() < R.brute_share else "gunner"
                hp = R.brute_hp if kind == "brute" else R.gunner_hp
                timer = rng.randint(2, R.gunner_period) if kind == "gunner" else 0
                self.enemies.append(Enemy(len(self.enemies), kind, free_cell(rid, margin=1), rid, hp, timer=timer))
        others = [r for r in range(n) if r != self.start_room]
        self.potions = [free_cell(rng.choice(others)) for _ in range(R.potions)]

    # -- geometry -------------------------------------------------------------------

    def closed(self) -> set[Cell]:
        """Doorway cells that are sealed now."""
        return self.doors[self.sealed] if self.sealed is not None else set()

    def passable(self, c: Cell) -> bool:
        return c in self.open and (self.sealed is None or c not in self.doors[self.sealed])

    def distances(self, start: Cell, within: set[Cell] | None = None) -> dict[Cell, int]:
        """Walking distance from ``start`` to every reachable open cell (optionally only through
        ``within``, e.g. the cells the agent knows). Sealed doorways block."""
        allowed = (self.open if within is None else self.open & within) - self.closed()
        dist, queue = {start: 0}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS.values():
                n = (x + dx, y + dy)
                if n not in dist and n in allowed:
                    dist[n] = dist[(x, y)] + 1
                    queue.append(n)
        return dist

    def clear_line(self, a: Cell, b: Cell) -> bool:
        """True when a bullet from the centre of ``a`` to the centre of ``b`` meets no wall."""
        n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / SUBSTEP))
        for i in range(1, n):
            t = i / n
            c = (rnd(a[0] + (b[0] - a[0]) * t), rnd(a[1] + (b[1] - a[1]) * t))
            if not self.passable(c):
                return False
        return True

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

    def enemy_at(self, c: Cell) -> Enemy | None:
        return next((e for e in self.enemies if e.alive and e.pos == c), None)

    def living(self, room: int | None = None) -> list[Enemy]:
        return [e for e in self.enemies if e.alive and (room is None or e.home == room)]

    def visible_enemies(self) -> list[Enemy]:
        """Living enemies in rooms the agent has seen with a clear line from the agent, nearest first."""
        out = [e for e in self.living() if e.home in self.seen and self.clear_line(self.pos, e.pos)]
        return sorted(out, key=lambda e: (math.dist(self.pos, e.pos), e.id))

    def bullet_path(self, b: Bullet) -> list[Cell]:
        """Cells a bullet passes through during its next tick of flight, up to a wall (walls and
        sealed doorways as they are now)."""
        n = max(1, math.ceil(math.hypot(b.vx, b.vy) / SUBSTEP))
        out, x, y = [], b.x, b.y
        for _ in range(n):
            x, y = x + b.vx / n, y + b.vy / n
            c = (rnd(x), rnd(y))
            if not self.passable(c):
                break
            if not out or out[-1] != c:
                out.append(c)
        return out

    def danger(self) -> dict[Cell, int]:
        """Cells that an enemy bullet will pass through during the next tick -> its damage."""
        out: dict[Cell, int] = {}
        for b in self.bullets:
            if b.owner == "enemy":
                for c in self.bullet_path(b):
                    out[c] = out.get(c, 0) + b.damage
        return out

    # -- dynamics -------------------------------------------------------------------

    def step(self, move: str, shoot: int | None = None) -> list[dict]:
        """Apply the agent's move and shot (an enemy id, or None to hold fire), then bullets,
        enemies and doorways. Returns the events of the tick."""
        if self.outcome is not None:
            raise RuntimeError(f"the run is over ({self.outcome})")
        ev: list[dict] = []
        dx, dy = STEP.get(move, (0, 0))
        to = (self.pos[0] + dx, self.pos[1] + dy)
        if (dx, dy) != (0, 0):
            if not self.passable(to):
                ev.append({"kind": "bump", "cell": list(to)})
            elif self.enemy_at(to) is not None:
                ev.append({"kind": "blocked", "enemy": self.enemy_at(to).id})
            else:
                self.pos = to
                self._pickups(ev)
        self._look(ev)
        if self.outcome is None:
            self._fire(shoot, ev)
            self._fly(ev)
            if self.outcome is None:
                self._enemies_act(ev)
            self._doors(ev)
            self.cooldown = max(0, self.cooldown - 1)
        self.tick += 1
        if self.outcome is None and self.tick >= self.rules.max_ticks:
            self.outcome = "timeout"
            ev.append({"kind": "timeout"})
        return ev

    def _new_bullet(self, owner: str, frm: Cell, to: Cell, speed: float, damage: int, source=None) -> Bullet:
        d = math.dist(frm, to) or 1.0
        self._bullet_ids += 1
        b = Bullet(self._bullet_ids, owner, float(frm[0]), float(frm[1]), (to[0] - frm[0]) / d * speed,
                   (to[1] - frm[1]) / d * speed, damage, source)
        self.bullets.append(b)
        return b

    def _fire(self, shoot: int | None, ev: list[dict]) -> None:
        if shoot is None:
            return
        target = next((e for e in self.enemies if e.id == shoot and e.alive), None)
        if target is None or self.cooldown > 0:
            ev.append({"kind": "dry_fire", "enemy": shoot})
            return
        b = self._new_bullet("agent", self.pos, target.pos, self.rules.shot_speed, self.rules.shot_damage)
        self.cooldown = self.rules.cooldown
        self.shots += 1
        ev.append({"kind": "shot", "bullet": b.id, "enemy": target.id, "from": list(self.pos), "at": list(target.pos)})
        if not target.awake:
            self._wake(target.home)

    def _fly(self, ev: list[dict]) -> None:
        """Move every bullet one tick, in substeps; stop it at the first wall or target."""
        keep = []
        for b in self.bullets:
            n = max(1, math.ceil(math.hypot(b.vx, b.vy) / SUBSTEP))
            done = None
            for _ in range(n):
                b.x, b.y = b.x + b.vx / n, b.y + b.vy / n
                c = (rnd(b.x), rnd(b.y))
                if not self.passable(c):
                    done = {"kind": "impact", "bullet": b.id, "pos": [round(b.x, 3), round(b.y, 3)]}
                    break
                if b.owner == "agent":
                    e = self.enemy_at(c)
                    if e is not None:
                        e.hp -= b.damage
                        self._wake(e.home)
                        done = {"kind": "enemy_hit", "bullet": b.id, "enemy": e.id, "hp": max(0, e.hp),
                                "pos": [round(b.x, 3), round(b.y, 3)]}
                        if e.hp <= 0:
                            self.kills += 1
                            ev.append(done)
                            done = {"kind": "enemy_killed", "enemy": e.id, "enemy_kind": e.kind, "cell": list(e.pos)}
                        break
                elif c == self.pos:
                    self._hurt(b.damage, "gunner", ev, bullet=b.id)
                    done = {"kind": "impact", "bullet": b.id, "pos": [round(b.x, 3), round(b.y, 3)]}
                    break
            if done is None:
                keep.append(b)
            else:
                ev.append(done)
        self.bullets = keep

    def _hurt(self, damage: int, by: str, ev: list[dict], **extra) -> None:
        self.health -= damage
        ev.append({"kind": "hit", "by": by, "damage": damage, **extra})
        if self.health <= 0 and self.outcome is None:
            self.health, self.outcome, self.cause = 0, "died", by
            ev.append({"kind": "died", "cause": by})

    def _pickups(self, ev: list[dict]) -> None:
        c = self.pos
        if c == self.key:
            self.key, self.has_key = None, True
            ev.append({"kind": "key", "cell": list(c)})
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
        if room is not None and self.living(room.id) and not any(e.awake for e in self.living(room.id)):
            self._wake(room.id)

    def _wake(self, rid: int) -> None:
        for e in self.living(rid):
            e.awake = True

    def _doors(self, ev: list[dict]) -> None:
        if self.sealed is not None and not self.living(self.sealed):
            ev.append({"kind": "room_cleared", "room": self.sealed})
            self.cleared.add(self.sealed)
            self.sealed = None
            self._known = None
        room = self.inside(self.pos)
        if self.rules.seal and self.sealed is None and room is not None and self.living(room.id):
            self.sealed = room.id
            ev.append({"kind": "room_sealed", "room": room.id})
        for rid in {e.home for e in self.enemies if not e.alive} - self.cleared:
            if not self.living(rid) and rid != self.sealed:
                self.cleared.add(rid)
                ev.append({"kind": "room_cleared", "room": rid})

    def _enemies_act(self, ev: list[dict]) -> None:
        R = self.rules
        for e in self.enemies:
            if not e.alive or not e.awake:
                continue
            room = self.rooms[e.home]
            see = self.clear_line(e.pos, self.pos)
            d = math.dist(e.pos, self.pos)
            if e.kind == "gunner":
                if e.aiming:
                    e.aiming = False
                    if see and d <= R.gunner_range:
                        b = self._new_bullet("enemy", e.pos, self.pos, R.enemy_shot_speed, R.enemy_shot_damage, e.id)
                        ev.append({"kind": "enemy_shot", "bullet": b.id, "enemy": e.id, "from": list(e.pos),
                                   "at": list(self.pos)})
                    e.timer = R.gunner_period
                    continue
                e.timer = max(0, e.timer - 1)
                if e.timer <= 1 and see and d <= R.gunner_range:
                    e.aiming = True  # telegraph: it fires at the end of the next tick
                    continue
                if (self.tick + e.id) % R.gunner_move_every == 0:
                    e.pos = self._gunner_step(e, room, see, d)
            else:  # brute
                if e.timer > 0:
                    e.timer -= 1
                    e.pos = self._away_step(e, room)
                    continue
                if abs(e.pos[0] - self.pos[0]) + abs(e.pos[1] - self.pos[1]) == 1:
                    self._hurt(R.brute_damage, "brute", ev, enemy=e.id)
                    e.timer = R.brute_rest
                    continue
                if (self.tick + e.id) % R.brute_move_every != 0:
                    e.pos = self._toward_step(e, room)

    def _free(self, c: Cell, e: Enemy, room: Room) -> bool:
        return c in room and c != self.pos and all(o.pos != c for o in self.living() if o is not e)

    def _neighbours(self, e: Enemy, room: Room) -> list[Cell]:
        return [(e.pos[0] + dx, e.pos[1] + dy) for dx, dy in DIRS.values() if self._free((e.pos[0] + dx, e.pos[1] + dy), e, room)]

    def _toward_step(self, e: Enemy, room: Room) -> Cell:
        """One step that shortens the walk to a cell next to the agent, inside the room."""
        best, here = e.pos, None
        goal = [c for c in ((self.pos[0] + dx, self.pos[1] + dy) for dx, dy in DIRS.values()) if c in room]
        if not goal:
            return e.pos
        dist = {}
        for g in goal:
            for c, s in self.distances(g, within=set(room.cells)).items():
                dist[c] = min(dist.get(c, 10 ** 6), s)
        here = dist.get(e.pos, 10 ** 6)
        for c in self._neighbours(e, room):
            if dist.get(c, 10 ** 6) < here:
                best, here = c, dist[c]
        return best

    def _away_step(self, e: Enemy, room: Room) -> Cell:
        opts = self._neighbours(e, room)
        far = max(opts + [e.pos], key=lambda c: (math.dist(c, self.pos), c))
        return far

    def _gunner_step(self, e: Enemy, room: Room, see: bool, d: float) -> Cell:
        """Keep 3-6 cells from the agent with a clear line; otherwise sidestep at random."""
        opts = self._neighbours(e, room)
        if not opts:
            return e.pos
        if d < 3:
            return max(opts, key=lambda c: (math.dist(c, self.pos), c))
        if not see or d > 6:
            return min(opts, key=lambda c: (math.dist(c, self.pos), c))
        return self.rng.choice(opts + [e.pos])

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
            "agent": {"pos": list(self.pos), "health": self.health, "has_key": self.has_key,
                      "cooldown": self.cooldown, "shots": self.shots, "kills": self.kills},
            "enemies": [{"id": e.id, "kind": e.kind, "pos": list(e.pos), "hp": e.hp, "home": e.home,
                         "awake": e.awake, "aiming": e.aiming, "resting": e.kind == "brute" and e.timer > 0}
                        for e in self.enemies],
            "bullets": [{"id": b.id, "owner": b.owner, "pos": [round(b.x, 3), round(b.y, 3)],
                         "vel": [round(b.vx, 4), round(b.vy, 4)]} for b in self.bullets],
            "items": {"potions": [list(c) for c in self.potions], "key": list(self.key) if self.key else None},
            "seen": sorted(self.seen), "sealed": self.sealed, "cleared": sorted(self.cleared),
        }
