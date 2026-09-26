"""Step 6: tournament choice sampling, tested with fake scorers (no model)."""

import random
from types import SimpleNamespace

import pytest

from system_one import Decision, Engine
from system_one.backends.mock import MockBackend
from system_one.tournament import Tournament, run_tournament, run_tournaments, split_groups


def make(opts):
    return Decision("Pick the best item.", tuple(opts))


class MaxScorer:
    """Picks the option with the largest trailing number: a consistent, transitive judge."""

    def __init__(self):
        self.batches = []

    def __call__(self, decisions):
        self.batches.append(len(decisions))
        out = []
        for d in decisions:
            vals = [int(o.split()[-1]) for o in d.options]
            i = vals.index(max(vals))
            out.append(SimpleNamespace(index=i, probs=[1.0 if j == i else 0.0 for j in range(len(vals))]))
        return out


def items(n, seed=0):
    vals = list(range(n))
    random.Random(seed).shuffle(vals)
    return [f"item {v}" for v in vals]


def test_split_groups_is_contiguous_balanced_and_complete():
    assert [len(g) for g in split_groups(list(range(101)), 100)] == [51, 50]
    assert [len(g) for g in split_groups(list(range(10)), 3)] == [3, 3, 2, 2]
    groups = split_groups(list(range(57)), 10)
    assert [i for g in groups for i in g] == list(range(57))
    assert max(len(g) for g in groups) <= 10
    with pytest.raises(ValueError):
        split_groups([1, 2], 1)


def test_finds_global_best_with_consistent_scorer():
    scorer = MaxScorer()
    t = run_tournament(scorer, items(1000), group_size=100, make_decision=make)
    assert t.winner == "item 999"
    assert len(t.rounds) == 2 and scorer.batches == [10, 1]  # 10 groups, then the final
    assert t.model_decisions == 11


def test_rounds_record_groups_and_winners():
    t = run_tournament(MaxScorer(), items(30, seed=3), group_size=8, make_decision=make)
    first = t.rounds[0]
    assert [len(g) for g in first.groups] == [8, 8, 7, 7]
    for g, w in zip(first.groups, first.winners):
        assert w == max(g, key=lambda i: int(t.options[i].split()[-1]))
    final = t.final_probs()  # covers the 4 finalists only, not all 30 options
    assert len(final) == 4 and final[t.winner] == 1.0


def test_position_biased_scorer_keeps_first_of_each_group():
    first = lambda ds: [SimpleNamespace(index=0, probs=[1.0] + [0.0] * (len(d.options) - 1)) for d in ds]
    t = run_tournament(first, items(20), group_size=6, make_decision=make)
    assert t.winner == t.options[0]
    assert t.rounds[0].winners == [g[0] for g in t.rounds[0].groups]


def test_trivial_sizes_and_byes():
    scorer = MaxScorer()
    t = run_tournament(scorer, ["only 1"], group_size=5, make_decision=make)
    assert t.done and t.winner == "only 1" and scorer.batches == []
    t = run_tournament(scorer, ["a 1", "b 2", "c 3"], group_size=2, make_decision=make)
    assert t.rounds[0].probs[1] == []  # the single-option group was a bye
    assert t.winner == "c 3" and scorer.batches == [1, 1]


def test_lockstep_rounds_share_one_call():
    scorer = MaxScorer()
    ts = [Tournament(tuple(items(n, seed=n)), 10, make) for n in (95, 12, 5)]
    calls = run_tournaments(scorer, ts)
    assert calls == 2  # the 95-option tournament needs two rounds; the others ride along
    assert scorer.batches == [10 + 2 + 1, 1 + 1]
    assert [t.winner for t in ts] == ["item 94", "item 11", "item 4"]


def test_submit_checks_result_count():
    t = Tournament(tuple(items(10)), 3, make)
    t.pending()
    with pytest.raises(ValueError):
        t.submit([])


def test_matches_full_decision_when_the_judge_is_consistent():
    scorer = MaxScorer()
    for seed in range(20):
        opts = items(random.Random(seed).randrange(20, 120), seed)
        full = scorer([make(opts)])[0]
        t = run_tournament(scorer, opts, group_size=9, make_decision=make)
        assert t.winner == opts[full.index]


def test_runs_through_the_engine_with_one_pass_per_round():
    be = MockBackend()
    eng = Engine(be)
    before = be.forward_calls
    t = run_tournament(eng.decide_batch, items(60), group_size=10, make_decision=make)
    assert t.done and be.forward_calls - before == len(t.rounds) == 2
    assert sum(t.final_probs().values()) == pytest.approx(1.0)


def test_goal_tier_runs_as_tournament_one_round_per_tick():
    from system_one.goals import GoalStack, Tier, step_all

    be = MockBackend()
    eng = Engine(be)
    many = tuple(f"gem {i}" for i in range(20))
    stack = GoalStack([Tier("target", "Which gem?", many, every=100),
                       Tier("action", "Which move?", ("north", "south"), every=1)])
    other = GoalStack([Tier("action", "Which move?", ("north", "south"), every=1)])

    before = be.forward_calls
    out = step_all(eng, [(stack, "s"), (other, "s")], tick=0, group_size=6)
    assert be.forward_calls - before == 1  # round 1 (4 groups) + 2 actions share one pass
    assert sum(1 for *_, tour in out if tour is not None) == 4
    assert "target" in stack.tournaments and "target" not in stack.current
    assert "target" not in [t.name for t in stack.due(1)]  # pending, not rescheduled

    out = step_all(eng, [(stack, "s"), (other, "s")], tick=1, group_size=6)
    assert sum(1 for *_, tour in out if tour is not None) == 1  # the final round
    assert not stack.tournaments and stack.current["target"] in many
    assert 0 < stack.probability["target"] <= 1
    assert [t.name for t in stack.due(2)] == ["action"]


def test_invalidate_forces_redecision():
    from system_one.goals import GoalStack, Tier

    stack = GoalStack([Tier("target", "q", ("a", "b"), every=100)])
    stack.apply(stack.tier("target"), "a", 0)
    assert stack.due(1) == []
    stack.invalidate("target")
    assert [t.name for t in stack.due(1)] == ["target"]


def test_partial_rounds_give_the_same_winner():
    scorer = MaxScorer()
    full = run_tournament(scorer, items(50, seed=4), group_size=8, make_decision=make)
    t = Tournament(tuple(items(50, seed=4)), 8, make)
    calls = 0
    while not t.done:
        ds = t.pending(limit=1)
        assert len(ds) == 1
        t.submit(scorer(ds))
        calls += 1
    assert t.winner == full.winner == "item 49"
    assert calls == t.model_decisions == full.model_decisions  # same work, one group at a time
    assert len(t.rounds) == len(full.rounds)


def test_progress_reports_resolved_groups():
    t = Tournament(tuple(items(24)), 8, make)
    assert t.progress() == (1, 0, 3)
    t.submit(MaxScorer()(t.pending(limit=2)))
    assert t.progress() == (1, 2, 3)
    t.submit(MaxScorer()(t.pending()))
    assert t.progress() == (2, 0, 1)


def test_plan_budget_limits_planning_per_tick():
    from system_one.goals import GoalStack, Tier, step_all

    be = MockBackend()
    eng = Engine(be)
    gems = tuple(f"gem {i}" for i in range(24))
    stack = GoalStack([Tier("strategy", "Which goal?", ("gems", "food"), every=50),
                       Tier("target", "Which gem?", gems, every=50),
                       Tier("action", "Which move?", ("north", "south"), every=1)])
    per_tick = []
    for tick in range(8):
        before = be.forward_calls
        out = step_all(eng, [(stack, "s")], tick, group_size=8, plan_budget=1)
        assert be.forward_calls - before == 1
        assert sum(1 for _, tier, _, _ in out if tier.name == "action") == 1
        per_tick.append(sum(1 for _, tier, _, _ in out if tier.name != "action"))
    assert max(per_tick) == 1  # never more than one planning decision next to the move
    # tick 0: strategy; ticks 1-3: the 3 first-round groups; tick 4: the final; then nothing due
    assert per_tick == [1, 1, 1, 1, 1, 0, 0, 0]
    assert stack.current["target"] in gems and not stack.tournaments
