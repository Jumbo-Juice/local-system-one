"""Helpers shared by the game demos (grid, dungeon, shooter)."""

from __future__ import annotations

from typing import Callable, Iterable

from system_one import Decision

DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
MOVES = ("move north", "move south", "move east", "move west", "stay")
DASHES = ("dash north", "dash south", "dash east", "dash west")
STEP = {**{f"move {k}": v for k, v in DIRS.items()}, **{f"dash {k}": v for k, v in DIRS.items()}, "stay": (0, 0)}

LOOP = 30  # ticks in a row without a new cell or any progress that count as going in circles

Cell = tuple[int, int]


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"


def base_move(choice: str) -> str:
    """'move west (target: 2 steps)' -> 'move west'."""
    return choice.split(" (")[0]


def is_dash(move: str) -> bool:
    return move.startswith("dash ")


def dash_path(pos: Cell, move: str, cells: int, free: Callable[[Cell], bool]) -> list[Cell]:
    """The cells a dash passes through, in order, up to ``cells`` of them: it stops before the
    first cell that is not ``free`` (a wall, a closed doorway, an enemy). The last cell is where
    the agent lands; an empty list means the dash cannot move at all."""
    dx, dy = STEP[move]
    out, (x, y) = [], pos
    for _ in range(cells):
        x, y = x + dx, y + dy
        if not free((x, y)):
            break
        out.append((x, y))
    return out


class IdleTracker:
    """Ticks since the agent last did something new: stood on a cell for the first time in the
    run, or had one of the game's progress events. The stuck streak counts ticks without getting
    closer to one unchanged target, so a loop between shifting targets never registers there
    (Observed: shooter, 3B, seed 1: 570 ticks exploring the same few cells, stuck_ticks 0)."""

    def __init__(self, start: Cell, progress: Iterable[str]):
        self.visited = {tuple(start)}
        self.progress = set(progress)
        self.streak = 0

    def update(self, pos: Cell, events: list[dict]) -> int:
        new = tuple(pos) not in self.visited
        self.visited.add(tuple(pos))
        for e in events:  # a dash passes cells on its way
            for c in e.get("path", ()) if e["kind"] == "dash" else ():
                new |= tuple(c) not in self.visited
                self.visited.add(tuple(c))
        self.streak = 0 if new or any(e["kind"] in self.progress for e in events) else self.streak + 1
        return self.streak


def streak_ticks(streaks: Iterable[int], n: int) -> int:
    """Ticks that belong to streaks of at least ``n`` (each record holds its streak so far)."""
    return sum(0 if s < n else n if s == n else 1 for s in streaks)


def warm_up(engine) -> None:
    """The first XPU pass compiles kernels (~3 s) and the first long padded batch allocates
    buffers. Doing both here keeps them out of the run (Observed; docs/research.md -> Smoother ticks)."""
    engine.decide_batch([Decision("warm-up", ("a", "b"))])
    warm = [Decision("warm-up", tuple("abcdefgh"), state="Rooms, doors, gems and enemies. " * (10 + 8 * i))
            for i in range(2)]
    engine.decide_batch(warm)
    engine.decide_batch(warm)  # the second call also compiles the prefix-cached path
