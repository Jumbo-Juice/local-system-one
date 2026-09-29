"""A small deterministic grid world for the demo. No model code here.

Agents collect gems (score), eat food (energy) and avoid roaming hazards (health).
North is up: "move north" decreases y.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

MOVES = ("move north", "move south", "move east", "move west", "stay")
STEP = {"move north": (0, -1), "move south": (0, 1), "move east": (1, 0), "move west": (-1, 0), "stay": (0, 0)}
DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
COLOURS = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c")

Cell = tuple[int, int]


@dataclass
class Agent:
    id: int
    pos: Cell
    energy: int = 100
    health: int = 100
    score: int = 0
    deaths: int = 0
    starved: int = 0  # deaths with energy at 0 (the rest: hazards)
    colour: str = "#000000"


@dataclass
class World:
    width: int = 16
    height: int = 12
    n_agents: int = 4
    n_gems: int = 24
    n_food: int = 12
    n_hazards: int = 3
    seed: int = 0
    walls: set[Cell] = field(default_factory=set)
    gems: list[Cell] = field(default_factory=list)
    food: list[Cell] = field(default_factory=list)
    hazards: list[Cell] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
    tick: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        # Two fixed wall segments with gaps, so paths are not always straight lines.
        for y in range(2, self.height - 3):
            self.walls.add((self.width // 3, y))
        for y in range(3, self.height - 1):
            self.walls.add((2 * self.width // 3, y))
        for i in range(self.n_agents):
            self.agents.append(Agent(i, self._free_cell(), colour=COLOURS[i % len(COLOURS)]))
        # Place items one at a time so each new cell sees the ones already placed.
        for items, n in ((self.hazards, self.n_hazards), (self.gems, self.n_gems), (self.food, self.n_food)):
            for _ in range(n):
                items.append(self._free_cell())

    # -- geometry -------------------------------------------------------------------

    def inside(self, c: Cell) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def passable(self, c: Cell) -> bool:
        return self.inside(c) and c not in self.walls

    def _occupied(self) -> set[Cell]:
        return set(self.walls) | set(self.gems) | set(self.food) | set(self.hazards) | {a.pos for a in self.agents}

    def _free_cell(self) -> Cell:
        taken = self._occupied()
        while True:
            c = (self.rng.randrange(self.width), self.rng.randrange(self.height))
            if c not in taken:
                return c

    def distances(self, start: Cell) -> dict[Cell, int]:
        """Shortest walking distance (BFS around walls) from ``start`` to every reachable cell."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS.values():
                n = (x + dx, y + dy)
                if n not in dist and self.passable(n):
                    dist[n] = dist[(x, y)] + 1
                    queue.append(n)
        return dist

    def neighbours(self, agent: Agent) -> dict[str, str]:
        out = {}
        others = {a.pos for a in self.agents if a is not agent}
        for name, (dx, dy) in DIRS.items():
            c = (agent.pos[0] + dx, agent.pos[1] + dy)
            if not self.passable(c):
                out[name] = "wall"
            elif c in self.hazards:
                out[name] = "hazard"
            elif c in others:
                out[name] = "agent"
            else:
                out[name] = "free"
        return out

    # -- dynamics -------------------------------------------------------------------

    def step(self, moves: dict[int, str]) -> dict:
        """Apply one move per agent, then pickups, hazards, energy and respawns."""
        events = {"gems": 0, "food": 0, "hits": 0, "deaths": 0, "starved": 0}
        for a in self.agents:
            dx, dy = STEP.get(moves.get(a.id, "stay"), (0, 0))
            target = (a.pos[0] + dx, a.pos[1] + dy)
            if self.passable(target) and all(o.pos != target for o in self.agents if o is not a):
                a.pos = target
            if a.pos in self.gems:
                self.gems.remove(a.pos)
                a.score += 1
                events["gems"] += 1
                self.gems.append(self._free_cell())
            if a.pos in self.food:
                self.food.remove(a.pos)
                a.energy = min(100, a.energy + 40)
                events["food"] += 1
                self.food.append(self._free_cell())
        if self.tick % 2 == 0:  # hazards wander every other tick
            for i, (x, y) in enumerate(self.hazards):
                dx, dy = self.rng.choice(list(DIRS.values()) + [(0, 0)])
                c = (x + dx, y + dy)
                if self.passable(c) and c not in self.hazards:
                    self.hazards[i] = c
        for a in self.agents:
            a.energy = max(0, a.energy - 1)
            if a.pos in self.hazards:
                a.health -= 25
                events["hits"] += 1
            if a.energy == 0:
                a.health -= 5
            if a.health <= 0:  # respawn with full stats; the score is kept
                a.deaths += 1
                events["deaths"] += 1
                if a.energy == 0:
                    a.starved += 1
                    events["starved"] += 1
                a.pos, a.health, a.energy = self._free_cell(), 100, 100
        self.tick += 1
        return events
