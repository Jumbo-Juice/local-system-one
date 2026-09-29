"""Helpers shared by the game demos (grid, dungeon, shooter)."""

from __future__ import annotations

from system_one import Decision


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"


def base_move(choice: str) -> str:
    """'move west (target: 2 steps)' -> 'move west'."""
    return choice.split(" (")[0]


def warm_up(engine) -> None:
    """The first XPU pass compiles kernels (~3 s) and the first long padded batch allocates
    buffers. Doing both here keeps them out of the run (Observed; docs/research.md -> Smoother ticks)."""
    engine.decide_batch([Decision("warm-up", ("a", "b"))])
    warm = [Decision("warm-up", tuple("abcdefgh"), state="Rooms, doors, gems and enemies. " * (10 + 8 * i))
            for i in range(2)]
    engine.decide_batch(warm)
    engine.decide_batch(warm)  # the second call also compiles the prefix-cached path
