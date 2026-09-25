"""Step 5: tiered goals (schedule, context, dependent options, batching across agents)."""

import numpy as np
import pytest

from system_one import Engine
from system_one.backends.mock import MockBackend
from system_one.goals import GoalStack, Tier, step_all

TARGETS = {"collect gems": ("gem 1", "gem 2"), "find food": ("apple", "bread")}


def tiers():
    return [
        Tier("strategy", "Which goal?", ("collect gems", "find food"), every=6, title="Strategic goal"),
        Tier("target", "Which target?", lambda goals: TARGETS[goals.get("strategy", "collect gems")],
             every=3, title="Target"),
        Tier("action", "Which move?", ("north", "south"), every=1),
    ]


def test_schedule_by_period():
    s = GoalStack(tiers())
    due = {}
    for tick in range(7):
        names = [t.name for t in s.due(tick)]
        due[tick] = names
        for t in s.due(tick):
            s.apply(t, s.options(t)[0], tick)  # same choice every time -> nothing goes stale
    assert due[0] == ["strategy", "target", "action"]
    assert due[1] == due[2] == ["action"]
    assert due[3] == ["target", "action"]
    assert due[6] == ["strategy", "target", "action"]


def test_context_lists_only_higher_tiers():
    s = GoalStack(tiers())
    s.apply(s.tier("strategy"), "find food", 0)
    s.apply(s.tier("target"), "apple", 0)
    assert s.context(s.tier("strategy")) == ""
    assert s.context(s.tier("target")) == "Strategic goal: find food"
    assert s.context(s.tier("action")) == "Strategic goal: find food\nTarget: apple"
    d = s.decision(s.tier("action"), "state text")
    assert d.context.endswith("Target: apple") and d.state == "state text"


def test_options_can_depend_on_parent_goal():
    s = GoalStack(tiers())
    s.apply(s.tier("strategy"), "find food", 0)
    assert s.options(s.tier("target")) == ("apple", "bread")


def test_goal_change_makes_lower_tiers_due_next_tick():
    s = GoalStack(tiers())
    for t in s.due(0):
        s.apply(t, s.options(t)[0], 0)
    assert [t.name for t in s.due(1)] == ["action"]
    s.apply(s.tier("strategy"), "find food", 1)  # e.g. forced re-plan
    assert [t.name for t in s.due(2)] == ["target", "action"]


def test_pending_tier_is_not_rescheduled():
    s = GoalStack(tiers())
    s.mark_pending(s.tier("target"))
    assert "target" not in [t.name for t in s.due(0)]
    s.apply(s.tier("target"), "gem 2", 0)
    assert "target" not in [t.name for t in s.due(1)]


def test_step_all_batches_every_agent_in_one_pass():
    be = MockBackend()
    eng = Engine(be)
    agents = [(GoalStack(tiers()), f"agent {i} state") for i in range(4)]
    before = be.forward_calls
    out = step_all(eng, agents, tick=0)
    assert be.forward_calls - before == 1
    assert len(out) == 12  # 4 agents x 3 tiers due at tick 0
    for stack, _ in agents:
        assert set(stack.current) == {"strategy", "target", "action"}
        assert all(0 < p <= 1 for p in stack.probability.values())


def test_same_tick_parent_change_keeps_child_stale():
    # Scripted mock: always answer label "B". At tick 0 the target is chosen from the
    # collect-gems list; strategy becomes "find food", so target must be re-decided at tick 1.
    be = MockBackend()

    def fn(ids):
        v = np.zeros(be.vocab_size, dtype=np.float32)
        v[be.token_id("B")] = 5.0
        return v

    be.logit_fn = fn
    stack = GoalStack(tiers())
    step_all(Engine(be), [(stack, "s")], tick=0)
    assert stack.current["strategy"] == "find food" and stack.current["target"] == "gem 2"
    assert [t.name for t in stack.due(1)] == ["target", "action"]
    step_all(Engine(be), [(stack, "s")], tick=1)
    assert stack.current["target"] == "bread"


def test_invalid_period():
    with pytest.raises(ValueError):
        Tier("x", "q", ("a",), every=0)
