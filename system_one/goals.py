"""Tiered goals.

Documented [S1]: a single fast decision cannot work out a goal by itself. So slower loops
periodically choose from a FIXED list of goals, and the chosen goal goes into the prompt
of the fast loop. S1 proposes layers such as strategy (~10 s), tactic (~5 s), target
(~1 s) and inputs (~100 ms).

Implementation choices (the sources leave these open):
- Periods are counted in ticks, not seconds.
- A tier's options may be a fixed list or a function of the goals chosen above it.
- Each tier's prompt context lists the goals currently chosen by all tiers above it.
- When a tier's choice changes, every tier below it is re-decided on the next tick.
- All due decisions of all agents in a tick go into ONE batch. Each uses the goals that
  were current at the start of the tick, so a new goal reaches lower tiers one tick later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Union

from .engine import Decision, DecisionResult

Options = Union[Sequence[str], Callable[[dict], Sequence[str]]]


@dataclass(frozen=True)
class Tier:
    name: str  # key, e.g. "strategy"
    instruction: str  # the question asked at this tier
    options: Options  # fixed list, or f(current goals of higher tiers) -> list
    every: int = 1  # re-decide every N ticks
    title: str = ""  # how lower tiers see this goal, e.g. "Strategic goal"

    def __post_init__(self):
        if self.every < 1:
            raise ValueError("Tier.every must be >= 1")


class GoalStack:
    """The goal state of one agent. Tiers are ordered from slowest (top) to fastest (bottom)."""

    def __init__(self, tiers: Sequence[Tier]):
        names = [t.name for t in tiers]
        if len(set(names)) != len(names):
            raise ValueError("tier names must be unique")
        self.tiers = list(tiers)
        self.current: dict[str, str] = {}
        self.probability: dict[str, float] = {}
        self._decided_at: dict[str, int] = {}
        self._stale: set[str] = set()
        self._pending: set[str] = set()  # decided over several ticks (e.g. a tournament)

    def tier(self, name: str) -> Tier:
        return next(t for t in self.tiers if t.name == name)

    def _above(self, tier: Tier) -> list[Tier]:
        return self.tiers[: self.tiers.index(tier)]

    def options(self, tier: Tier) -> tuple[str, ...]:
        opts = tier.options(dict(self.current)) if callable(tier.options) else tier.options
        return tuple(opts)

    def context(self, tier: Tier) -> str:
        """Goals of the higher tiers, as shown to this tier's prompt."""
        return "\n".join(
            f"{t.title or t.name}: {self.current[t.name]}" for t in self._above(tier) if t.name in self.current
        )

    def due(self, tick: int) -> list[Tier]:
        out = []
        for t in self.tiers:
            if t.name in self._pending:
                continue
            last = self._decided_at.get(t.name)
            if last is None or t.name in self._stale or tick - last >= t.every:
                out.append(t)
        return out

    def decision(self, tier: Tier, state: str) -> Decision:
        return Decision(tier.instruction, self.options(tier), state=state, context=self.context(tier))

    def mark_pending(self, tier: Tier) -> None:
        """The tier is being decided over several ticks; do not schedule it again meanwhile."""
        self._pending.add(tier.name)

    def apply(self, tier: Tier, choice: str, tick: int, probability: float | None = None) -> bool:
        """Commit a tier's choice. Returns True when the goal changed."""
        changed = self.current.get(tier.name) != choice
        self.current[tier.name] = choice
        if probability is not None:
            self.probability[tier.name] = probability
        self._decided_at[tier.name] = tick
        self._stale.discard(tier.name)
        self._pending.discard(tier.name)
        if changed:
            for t in self.tiers[self.tiers.index(tier) + 1:]:
                self._stale.add(t.name)
        return changed


def step_all(engine, agents: Sequence[tuple[GoalStack, str]], tick: int) -> list[tuple[int, Tier, DecisionResult]]:
    """One tick for many agents: all due tier decisions in ONE batched engine call.

    ``agents`` is a list of (goal stack, state text). Results are applied bottom-up so
    that a higher tier that changes in this tick still marks its lower tiers stale.
    """
    jobs = []
    for i, (stack, state) in enumerate(agents):
        for tier in stack.due(tick):
            jobs.append((i, tier, stack.decision(tier, state)))
    results = engine.decide_batch([d for _, _, d in jobs]) if jobs else []
    out = [(i, tier, r) for (i, tier, _), r in zip(jobs, results)]
    for i, tier, r in sorted(out, key=lambda x: -agents[x[0]][0].tiers.index(x[1])):
        agents[i][0].apply(tier, r.choice, tick, r.prob)
    return out
