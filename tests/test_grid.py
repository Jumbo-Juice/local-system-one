"""Step 7: the 2D demo runs on the mock backend with one batched pass per tick."""

import re

from demo.grid.brain import GOALS, Brain, Controller
from demo.grid.sim import run_headless
from demo.grid.world import World
from system_one import Engine
from system_one.backends.mock import MockBackend


def test_world_is_deterministic_and_walls_block():
    a, b = World(seed=3), World(seed=3)
    assert a.gems == b.gems and a.walls == b.walls and [x.pos for x in a.agents] == [x.pos for x in b.agents]
    w = World(seed=3)
    agent = w.agents[0]
    wall = next(iter(w.walls))
    agent.pos = (wall[0] - 1, wall[1])
    w.gems = [g for g in w.gems if g != agent.pos]
    w.step({agent.id: "move east"})
    assert agent.pos == (wall[0] - 1, wall[1])


def test_pickups_respawn_and_hazards_hurt():
    w = World(seed=1, n_hazards=0)
    agent = w.agents[0]
    gem = w.gems[0]
    agent.pos = (gem[0], gem[1] + 1) if w.passable((gem[0], gem[1] + 1)) else (gem[0], gem[1] - 1)
    move = "move north" if agent.pos[1] > gem[1] else "move south"
    n = len(w.gems)
    w.step({agent.id: move})
    assert agent.score == 1 and len(w.gems) == n and gem not in w.gems[:-1]
    w.hazards = [agent.pos]
    health = agent.health
    w.tick = 1  # odd tick: hazards do not move
    w.step({agent.id: "stay"})
    assert agent.health == health - 25


def test_brain_texts_and_target_options():
    w = World(seed=0)
    brain = Brain(w, w.agents[0])
    brain.refresh()
    assert brain.strategy_state().startswith("Energy: ok")
    for goal in GOALS:
        opts = brain.target_options(goal)
        assert opts and all(re.search(r"\(\d+,\d+\)", o) for o in opts)
    assert len(brain.target_options("collect gems")) > 8  # large enough for a tournament
    assert "You have no target yet." in brain.action_state()
    flat = Brain(w, w.agents[1], use_goals=False)
    assert [t.name for t in flat.stack.tiers] == ["action"]
    assert "Nearest gem:" in flat.action_state()


def test_controller_one_forward_pass_per_tick_with_tournaments():
    be = MockBackend()
    ctrl = Controller(World(seed=0), Engine(be), group_size=8)
    tournament_rounds = 0
    for _ in range(40):
        before = be.forward_calls
        rep = ctrl.tick()
        assert be.forward_calls - before == 1
        assert sum(1 for u in rep.updates if u[1] == "action") == 4
        tournament_rounds += sum(1 for u in rep.updates if u[3] is not None)
    assert tournament_rounds > 0
    assert ctrl.world.tick == 40


def test_headless_summary_and_flat_mode():
    be = MockBackend()
    s = run_headless(Controller(World(seed=2), Engine(be), use_goals=False), ticks=10, verbose=False)
    assert s["decisions_per_tick_mean"] == 4 and s["use_goals"] is False and s["tournament_decisions"] == 0
    assert sum(s["action_counts"].values()) == 40


def test_world_items_never_share_a_cell():
    for seed in range(20):
        w = World(seed=seed, n_food=12, n_gems=24)
        cells = w.gems + w.food + w.hazards + [a.pos for a in w.agents]
        assert len(cells) == len(set(cells)) and not set(cells) & w.walls


def test_starvation_death_is_counted():
    w = World(seed=0, n_agents=1, n_hazards=0)
    a = w.agents[0]
    a.energy, a.health = 0, 5
    w.food.clear()
    w.step({a.id: "stay"})
    assert a.deaths == 1 and a.starved == 1


def test_condition_change_triggers_strategy_replan():
    from demo.grid.brain import Brain

    w = World(seed=0, n_agents=1, n_hazards=0)
    b = Brain(w, w.agents[0])
    b.refresh()
    for t in b.stack.due(0):
        b.stack.apply(t, b.stack.options(t)[0], 0)
    assert "strategy" not in [t.name for t in b.stack.due(1)]
    w.agents[0].energy = 30  # crosses the 35 band
    b.refresh()
    assert "strategy" in [t.name for t in b.stack.due(1)]
    b.stack.apply(b.stack.tier("strategy"), "find food", 1)
    b.refresh()  # same band: no new trigger
    assert "strategy" not in [t.name for t in b.stack.due(2)]


def test_strategy_state_spells_out_low_energy():
    from demo.grid.brain import Brain

    w = World(seed=0, n_agents=1)
    w.agents[0].energy = 20
    b = Brain(w, w.agents[0])
    b.refresh()
    assert "about 20 ticks until starving" in b.strategy_state()
    assert "until starving" not in Brain(w, w.agents[0], aware=False).strategy_state()


def test_move_labels_describe_outcomes():
    from demo.grid.brain import annotate_moves

    w = World(seed=0, n_agents=1, n_hazards=0)
    a = w.agents[0]
    a.pos = (w.width - 1, 5)  # east edge: east is a wall
    w.hazards = [(w.width - 1, 4)]  # north is a hazard
    labels = dict(o.split(" (", 1) for o in annotate_moves(w, a, (w.width - 2, 5)))
    assert labels["move east"] == "wall)" and labels["move north"] == "HAZARD: -25 health)"
    assert labels["move west"] == "reach the target)" and labels["stay"] == "target: 1 step)"


def test_target_shown_to_move_tier_without_stale_distance():
    from demo.grid.brain import Brain

    w = World(seed=0, n_agents=1)
    b = Brain(w, w.agents[0])
    b.stack.apply(b.stack.tier("strategy"), "find food", 0)
    b.stack.apply(b.stack.tier("target"), "food at (7,0), 1 step away", 0)
    ctx = b.stack.context(b.stack.tier("action"))
    assert "Current target: food at (7,0)" in ctx and "step" not in ctx


def test_labelled_action_choice_still_moves_the_agent():
    be = MockBackend()
    ctrl = Controller(World(seed=0, n_agents=1), Engine(be), group_size=8)
    start = ctrl.world.agents[0].pos
    moved = False
    for _ in range(10):
        rep = ctrl.tick()
        choice = next(r.choice for _, tier, r, _ in rep.updates if tier == "action")
        assert " (" in choice  # labelled option
        moved |= ctrl.world.agents[0].pos != start
    assert moved
