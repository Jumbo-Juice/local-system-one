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
- With ``group_size`` set, a tier with more options than that is decided by tournament
  sampling across ticks, inside the same batch. The tier keeps its previous goal until the
  tournament finishes. By default a whole round runs per tick.
- ``plan_budget`` caps planning work (slow tiers and their tournament groups) per agent per
  tick, so the fast tier's tick latency stays nearly constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Union

from .engine import Decision, DecisionResult
from .tournament import Tournament

Options = Union[Sequence[str], Callable[[dict], Sequence[str]]]


@dataclass(frozen=True)
class Tier:
    name: str  # key, e.g. "strategy"
    instruction: str  # the question asked at this tier
    options: Options  # fixed list, or f(current goals of higher tiers) -> list
    every: int = 1  # re-decide every N ticks
    title: str = ""  # how lower tiers see this goal, e.g. "Strategic goal"
    # How lower tiers see the chosen option text (default: verbatim). Useful to drop details that
    # go stale, e.g. a distance measured when the option was chosen.
    describe: Callable[[str], str] | None = None
    # Names of the higher tiers whose goals this tier's prompt shows (default: all). Parallel
    # control heads (e.g. move and shoot, both every tick) should not see each other's last answer.
    context_from: tuple[str, ...] | None = None

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
        self.tournaments: dict[str, Tournament] = {}  # tier name -> tournament in progress

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
            f"{t.title or t.name}: {t.describe(self.current[t.name]) if t.describe else self.current[t.name]}"
            for t in self._above(tier)
            if t.name in self.current and (tier.context_from is None or t.name in tier.context_from)
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

    def invalidate(self, name: str) -> None:
        """Re-decide this tier on the next tick (e.g. its target was reached or disappeared)."""
        self._stale.add(name)

    def waiting(self, name: str) -> bool:
        """True while a new choice for this tier is owed: invalidated, or a tournament running."""
        return name in self._stale or name in self._pending

    def decided_at(self, name: str) -> int | None:
        """Tick of the tier's last committed choice (None if never decided)."""
        return self._decided_at.get(name)

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
                # A tournament below was built from the old goal's options: drop it.
                if self.tournaments.pop(t.name, None) is not None:
                    self._pending.discard(t.name)
        return changed


State = Union[str, Callable[[Tier], str]]


ONLY_OPTION = "only_option"  # DecisionResult.method of a choice committed without a model call


def _only_option(decision: Decision) -> DecisionResult:
    return DecisionResult(decision=decision, index=0, probs=[1.0], labels=[""], outside_mass=0.0,
                          method=ONLY_OPTION, top_token=None, prompt_tokens=0)


def step_all(
    engine, agents: Sequence[tuple[GoalStack, State]], tick: int, group_size: int | None = None,
    plan_budget: int | None = None, skip_single: bool = False,
) -> list[tuple[int, Tier, DecisionResult, Tournament | None]]:
    """One tick for many agents: all due tier decisions in ONE batched engine call.

    ``agents`` is a list of (goal stack, state), where state is a string or a function
    tier -> state text. With ``group_size``, tiers with more options than that run as
    tournaments. Results are applied bottom-up, so a higher tier that changes in this tick
    still marks its lower tiers stale.

    ``plan_budget`` caps the planning decisions per agent per tick. Planning means slow tiers
    (``every > 1``) and their tournament groups. Tiers with ``every == 1`` always run.
    Deferred tiers stay due, and tournaments continue with their remaining groups next tick.
    ``None`` means unlimited (a whole tournament round per tick).

    ``skip_single`` commits a tier that has exactly one option at once, without a model call
    and without using the budget. Its result has ``method == ONLY_OPTION`` and p = 1. The
    tiers below it see the new goal in the same tick.

    Returns (agent index, tier, result, tournament or None) for every decision made.
    """
    jobs: list[tuple[int, Tier, Decision, Tournament | None]] = []
    instant: list[tuple[int, Tier, DecisionResult, None]] = []
    for i, (stack, state) in enumerate(agents):
        state_of = state if callable(state) else (lambda _tier, s=state: s)
        budget = plan_budget
        for tier in stack.tiers:  # top-down, so a strategy is decided before its target
            # Re-checked per tier: an instant commit above makes the tiers below it due.
            if tier not in stack.due(tick):
                continue
            options = stack.options(tier)
            if not options:
                continue
            if skip_single and len(options) == 1:
                r = _only_option(stack.decision(tier, state_of(tier)))
                stack.apply(tier, options[0], tick, 1.0)
                instant.append((i, tier, r, None))
                continue
            if tier.every > 1 and budget == 0:
                continue  # budget used up: stays due; built next tick with fresh context
            if group_size and len(options) > group_size:
                snapshot, context = state_of(tier), stack.context(tier)
                stack.tournaments[tier.name] = Tournament(
                    options, group_size,
                    lambda opts, t=tier, s=snapshot, c=context: Decision(t.instruction, tuple(opts), state=s, context=c),
                )
                stack.mark_pending(tier)
            elif tier.every == 1:
                jobs.append((i, tier, stack.decision(tier, state_of(tier)), None))
            elif budget is None or budget > 0:
                jobs.append((i, tier, stack.decision(tier, state_of(tier)), None))
                budget = None if budget is None else budget - 1
        for name, tour in stack.tournaments.items():
            fast = stack.tier(name).every == 1
            ds = tour.pending(None if fast else budget)
            if not fast and budget is not None:
                budget -= len(ds)
            jobs.extend((i, stack.tier(name), d, tour) for d in ds)

    results = engine.decide_batch([d for _, _, d, _ in jobs]) if jobs else []
    out = [(i, tier, r, tour) for (i, tier, _, tour), r in zip(jobs, results)]

    commits = [(i, tier, r.choice, r.prob) for i, tier, r, tour in out if tour is None]
    rounds: dict[int, tuple[int, Tier, Tournament, list]] = {}
    for i, tier, r, tour in out:
        if tour is not None:
            rounds.setdefault(id(tour), (i, tier, tour, []))[3].append(r)
    for i, tier, tour, rs in rounds.values():
        tour.submit(rs)
        if tour.done:
            del agents[i][0].tournaments[tier.name]
            commits.append((i, tier, tour.winner, tour.final_probs()[tour.winner]))
    for i, tier, choice, p in sorted(commits, key=lambda c: -agents[c[0]][0].tiers.index(c[1])):
        agents[i][0].apply(tier, choice, tick, p)
    return instant + out
