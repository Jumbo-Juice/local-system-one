"""Tournament choice sampling for choice sets too large for one decision.

Documented [S1]: feed ~100 options at a time into each choice, then run another pass on the
winners; "ordinary LLMs are way better at relative judgements than absolute ratings".
Documented [S4]: contiguous groups in the original order; regroup the winners until one
remains.

Implementation choices:
- Groups are contiguous and of near-equal size (e.g. 101 options with group size 100 ->
  groups of 51 and 50, not 100 and 1).
- A group with one option advances without a model call.
- All groups of a round are independent, so a round is ONE batched engine call.
- The final probabilities cover only the last round's options. They are not a
  distribution over all options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .engine import Decision


class Chosen(Protocol):
    index: int
    probs: list[float]


Chooser = Callable[[list[Decision]], Sequence[Chosen]]  # e.g. Engine.decide_batch


def split_groups(items: Sequence[int], group_size: int) -> list[list[int]]:
    """Contiguous, order-preserving groups of near-equal size, each at most ``group_size``."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    n = len(items)
    k = max(1, math.ceil(n / group_size))
    base, extra = divmod(n, k)
    groups, start = [], 0
    for g in range(k):
        size = base + (1 if g < extra else 0)
        groups.append(list(items[start:start + size]))
        start += size
    return groups


@dataclass
class Round:
    groups: list[list[int]]  # option indices per group
    winners: list[int]  # winning option index per group
    probs: list[list[float]]  # per group, in group order (empty for byes)


@dataclass
class Tournament:
    """A multi-round choice over many options. Drive it with pending() / submit()."""

    options: tuple[str, ...]
    group_size: int
    make_decision: Callable[[list[str]], Decision]
    rounds: list[Round] = field(default_factory=list)

    def __post_init__(self):
        self.options = tuple(self.options)
        if not self.options:
            raise ValueError("a tournament needs at least one option")
        self.alive = list(range(len(self.options)))
        self._groups: list[list[int]] | None = None
        self._results: dict[int, Chosen] = {}  # group index -> result, current round
        self._asked: list[int] = []  # group indices handed out by the last pending() call
        self.model_decisions = 0

    @property
    def done(self) -> bool:
        return len(self.alive) == 1

    def pending(self, limit: int | None = None) -> list[Decision]:
        """Decisions for the unresolved groups of the current round (groups of 2+ options).

        With ``limit``, at most that many groups are handed out; later calls return the rest.
        A round finishes once every group has a result.
        """
        if self.done:
            self._asked = []
            return []
        if self._groups is None:
            self._groups = split_groups(self.alive, self.group_size)
            self._results = {}
        todo = [k for k, g in enumerate(self._groups) if len(g) > 1 and k not in self._results]
        self._asked = todo if limit is None else todo[:limit]
        return [self.make_decision([self.options[i] for i in self._groups[k]]) for k in self._asked]

    def submit(self, results: Sequence[Chosen]) -> None:
        """Results for the decisions returned by the last pending() call, in the same order."""
        if len(results) != len(self._asked):
            raise ValueError(f"expected {len(self._asked)} results, got {len(results)}")
        for k, r in zip(self._asked, results):
            self._results[k] = r
        self.model_decisions += len(results)
        self._asked = []
        groups = self._groups
        if groups is None or not all(len(g) == 1 or k in self._results for k, g in enumerate(groups)):
            return
        winners, probs = [], []
        for k, g in enumerate(groups):
            if len(g) == 1:
                winners.append(g[0])
                probs.append([])
            else:
                r = self._results[k]
                winners.append(g[r.index])
                probs.append(list(r.probs))
        self.rounds.append(Round(groups, winners, probs))
        self.alive = winners
        self._groups = None
        self._results = {}

    def progress(self) -> tuple[int, int, int]:
        """(current round number, groups resolved in it, contested groups in it)."""
        groups = self._groups or ([] if self.done else split_groups(self.alive, self.group_size))
        contested = sum(1 for g in groups if len(g) > 1)
        return len(self.rounds) + 1, len(self._results), contested

    @property
    def winner(self) -> str:
        if not self.done:
            raise RuntimeError("tournament not finished")
        return self.options[self.alive[0]]

    @property
    def winner_index(self) -> int:
        return self.alive[0]

    def final_probs(self) -> dict[str, float]:
        """Probabilities of the last round's options (not a distribution over all options)."""
        if not self.rounds:
            return {self.options[self.alive[0]]: 1.0}
        last = self.rounds[-1]
        group, probs = next((g, p) for g, p in zip(last.groups, last.probs) if p)
        return {self.options[i]: p for i, p in zip(group, probs)}


def run_tournaments(choose: Chooser, tournaments: Sequence[Tournament]) -> int:
    """Advance several tournaments in lockstep: every round is one ``choose`` call.

    Returns the number of ``choose`` calls (rounds) used.
    """
    calls = 0
    while True:
        # An unfinished tournament always has a group of 2+ options, so the batch is never empty.
        active = [(t, t.pending()) for t in tournaments if not t.done]
        if not active:
            return calls
        results = list(choose([d for _, ds in active for d in ds]))
        calls += 1
        k = 0
        for t, ds in active:
            t.submit(results[k:k + len(ds)])
            k += len(ds)


def run_tournament(choose: Chooser, options: Sequence[str], group_size: int,
                   make_decision: Callable[[list[str]], Decision]) -> Tournament:
    t = Tournament(tuple(options), group_size, make_decision)
    run_tournaments(choose, [t])
    return t
