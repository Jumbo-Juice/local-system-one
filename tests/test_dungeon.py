"""The dungeon demo (rebuilt 2026-09-30): rules, ghouls, dash, bots, brain texts, runner (mock backend only)."""

import json
import random
from dataclasses import replace

from demo.dungeon import bots
from demo.dungeon.world import Dungeon, Ghoul, Rules


def quiet(seed=0, **rules):
    """A dungeon without ghouls or supplies unless asked for, so single rules can be tested."""
    base = dict(room_ghouls=(0, 0), key_room_ghouls=0, exit_room_ghouls=0, gems=0, potions=0)
    base.update(rules)
    return Dungeon(seed, Rules(**base))


def walk_to(d, cell):
    """Put the agent next to ``cell`` and return the move onto it."""
    for move, (dx, dy) in (("move east", (-1, 0)), ("move west", (1, 0)), ("move south", (0, -1)), ("move north", (0, 1))):
        here = (cell[0] + dx, cell[1] + dy)
        if d.passable(here) and d.ghoul_at(here) is None:
            d.pos = here
            return move
    raise AssertionError("no open neighbour")


def place_ghoul(d, room=None, awake=True):
    """Replace the ghouls with one ghoul at the centre of ``room`` (default: a room next to the start)."""
    room = d.rooms[room if room is not None else next(c.rooms[1] if c.rooms[0] == d.start_room else c.rooms[0]
                                                        for c in d.corridors if d.start_room in c.rooms)]
    g = Ghoul(0, room.centre, room.id, room.centre, awake=awake)
    d.ghouls = [g]
    return g, room


def test_layout_is_deterministic_connected_and_consistent():
    for seed in range(25):
        d, again = Dungeon(seed), Dungeon(seed)
        assert d.layout() == again.layout() and d.snapshot() == again.snapshot()
        assert set(d.room_distances(d.start_room)) == set(range(9))  # every room reachable
        reach = d.distances(d.pos)
        items = [d.exit, d.key, *d.gems, *d.potions, *(g.pos for g in d.ghouls)]
        assert all(c in reach for c in items)
        assert len(set(items + [d.pos])) == len(items) + 1  # nothing shares a cell
        assert all(d.inside(c) is not None for c in items)
        assert d.exit_room != d.start_room and d.key_room not in (d.start_room, d.exit_room)
        assert sum(g.home == d.key_room for g in d.ghouls) == 2 and all(g.home != d.start_room for g in d.ghouls)
        assert d.seen == {d.start_room}


def test_every_room_has_two_doorways_three_cells_wide():
    """No dead ends and no one-cell doors: the first dungeon's corridors trapped its agent."""
    for seed in range(40):
        d = Dungeon(seed)
        for room in d.rooms:
            assert sum(room.id in c.rooms for c in d.corridors) >= 2
        for c in d.corridors:
            assert all(len(door) == 3 for door in c.doors)
            assert len(c.cells) % 3 == 0


def test_rooms_open_only_through_their_doorways():
    d = Dungeon(4)
    for room in d.rooms:
        border = set()
        for x, y in room.cells:
            for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if c not in room and d.passable(c):
                    border.add(c)
        assert border and all(d.is_door(c) and d.area[c] == room.id for c in border)


def test_exit_is_locked_without_the_key_and_opens_with_it():
    d = quiet()
    ev = d.step(walk_to(d, d.exit))
    assert d.pos == d.exit and d.outcome is None and {"kind": "exit_locked"} in ev
    d.step(walk_to(d, d.key))
    assert d.has_key and d.key is None
    ev = d.step(walk_to(d, d.exit))
    assert d.outcome == "escaped" and {"kind": "escaped"} in ev


def test_gems_count_and_potions_heal_up_to_the_cap():
    d = quiet(gems=1, potions=1)
    d.step(walk_to(d, d.gems[0]))
    assert d.collected == 1 and not d.gems
    d.health = 80
    ev = d.step(walk_to(d, d.potions[0]))
    assert d.health == 100 and any(e["kind"] == "potion" and e["gain"] == 20 for e in ev)


def test_tick_limit():
    d = quiet(max_ticks=5)
    for _ in range(5):
        d.step("stay")
    assert d.outcome == "timeout"


def test_ghouls_sleep_until_the_agent_steps_inside_then_act_from_the_next_tick():
    d = quiet()
    g, room = place_ghoul(d, awake=False)
    d.pos = next(c for c in room.cells if abs(c[0] - g.pos[0]) + abs(c[1] - g.pos[1]) == 2)
    d.step("stay")  # already inside: wakes the room, but a woken ghoul does not act yet
    assert g.awake and g.pos == g.post and d.health == 100
    ev = d.step("stay")
    assert g.pos != g.post  # now it closes in
    for _ in range(4):
        ev += d.step("stay")
    assert any(e["kind"] == "hit" for e in ev) and d.health == 100 - d.rules.ghoul_damage
    assert g.rest > 0 or abs(g.pos[0] - d.pos[0]) + abs(g.pos[1] - d.pos[1]) > 1  # backs off after a hit


def test_a_ghoul_tires_then_goes_home_and_sleeps_once_the_agent_has_left():
    d = quiet(ghoul_chase=3)
    g, room = place_ghoul(d)
    d.pos = next(c for c in room.cells if abs(c[0] - g.pos[0]) + abs(c[1] - g.pos[1]) >= 4)
    ev = []
    for _ in range(4):
        ev += d.step("stay")
    assert any(e["kind"] == "ghoul_tired" for e in ev) and g.tired > 0
    assert not bots.ghoul_reach(d, g)  # a tired ghoul hits nobody
    d.pos = d.rooms[d.start_room].centre  # the agent left
    for _ in range(40):
        ev += d.step("stay")
    assert g.pos == g.post and not g.awake and any(e["kind"] == "ghoul_slept" for e in ev)


def test_ghouls_never_leave_their_room():
    d = quiet()
    g, room = place_ghoul(d)
    door = next(iter(d.doors[room.id]))
    d.pos = door
    for _ in range(30):
        d.step("stay")
        assert g.pos in room


def test_dash_moves_three_cells_and_recharges_in_eight_ticks():
    d = quiet()
    room = d.rooms[d.start_room]
    d.pos = (room.x0, room.y0)
    ev = d.step("dash east")
    assert d.pos == (room.x0 + 3, room.y0) and any(e["kind"] == "dash" for e in ev)
    for _ in range(7):
        assert not d.can_dash() and any(e["kind"] == "dash_refused" for e in d.step("dash west"))
    assert d.can_dash()
    d.pos = (room.x0, room.y0)
    d.step("dash north")  # a wall right there: nothing moves, no charge used
    assert d.pos == (room.x0, room.y0) and d.can_dash()


def test_dash_stops_before_a_ghoul_and_picks_up_what_it_passes():
    d = quiet(gems=0)
    room = d.rooms[d.start_room]
    d.pos = (room.x0, room.y0)
    d.gems = [(room.x0 + 1, room.y0)]
    d.ghouls = [Ghoul(0, (room.x0 + 3, room.y0), room.id, (room.x0 + 3, room.y0))]
    d.step("dash east")
    assert d.pos == (room.x0 + 2, room.y0) and d.collected == 1


def test_reference_bot_escapes_and_random_does_not_on_dev_seeds():
    ref = [bots.play(Dungeon(s), bots.reference).outcome for s in range(1000, 1010)]
    rnd = [bots.play(Dungeon(s), bots.random_policy(random.Random(s))).outcome for s in range(1000, 1010)]
    assert ref.count("escaped") >= 9 and rnd.count("escaped") == 0


def test_sleeping_tired_and_resting_ghouls_are_no_threat():
    d = quiet()
    g, room = place_ghoul(d, awake=False)
    d.seen.add(room.id)
    assert not bots.threat_cells(d)
    g.awake = True
    assert (g.pos[0] + 1, g.pos[1]) in bots.threat_cells(d)
    g.rest = 2
    assert not bots.threat_cells(d)


# -- brain -------------------------------------------------------------------------------------

from system_one import Engine  # noqa: E402
from system_one.backends.mock import MockBackend  # noqa: E402

from demo.dungeon.brain import GOALS, DungeonBrain, Runner, cell_of  # noqa: E402
from demo.runs import to_jsonl  # noqa: E402


def test_goals_offered_only_when_possible_and_leaving_once_the_way_out_is_known():
    d = quiet()
    b = DungeonBrain(d)
    assert b.goal_options() == ["explore"]
    d.seen = set(range(9))  # the whole map known
    d._known = None
    b.refresh()
    assert "get the key" in b.goal_options() and "go to the exit" not in b.goal_options()
    d.has_key, d.key = True, None
    b.refresh()
    assert b.goal_options() == ["go to the exit"]  # no more exploring or gems with the key in hand
    d.health = 50
    d.potions = [next(c for c in d.rooms[d.start_room].cells if c != d.pos)]
    b.refresh()
    assert b.goal_options() == ["go to the exit", "drink a health potion"]
    assert all(g in GOALS for g in b.goal_options())


def test_targets_name_cells_and_distances():
    d = quiet(gems=3)
    b = DungeonBrain(d)
    opts = b.target_options("explore")
    assert opts and all(o.startswith("unexplored room behind the doorway at (") and " away" in o for o in opts)
    assert all(cell_of(o) in b._dist for o in opts)


def test_moves_are_labelled_closer_and_stay_is_left_out():
    d = quiet()
    b = DungeonBrain(d)
    b.stack.apply(b.stack.tier("strategy"), "explore", 0)
    b.stack.apply(b.stack.tier("target"), b.target_options("explore")[0], 0)
    b.refresh()
    opts = b.move_options()
    assert any("safe; closer" in o for o in opts) and not any(o.startswith("stay") for o in opts)
    assert any(o.startswith("dash") and "cells; safe" in o for o in opts)
    s = b.move_state()
    assert "North is up" in s and "Dash: ready" in s and "safe route" in s


def test_moves_into_a_ghouls_reach_are_left_out_while_a_safe_move_exists():
    d = quiet()
    g, room = place_ghoul(d)
    d.pos = (g.pos[0] - 2, g.pos[1])
    d.seen.add(room.id)
    d._known = None
    b = DungeonBrain(d)
    assert "GHOUL: -20 health" in b._label([(d.pos[0] + 1, d.pos[1])], None, {}, None)[0]
    opts = b.move_options()
    assert opts and not any("GHOUL" in o for o in opts) and not any(o.startswith("move east") for o in opts)
    assert "ghoul #0 2 cells east (awake, chasing you)" in b.move_state()


def test_navigation_targets_are_held_and_replanned_when_stalled():
    d = quiet()
    b = DungeonBrain(d)
    b.stack.apply(b.stack.tier("strategy"), "explore", 0)
    b.stack.apply(b.stack.tier("target"), b.target_options("explore")[0], 0)
    b.refresh()
    assert {t.name for t in b.stack.due(50)} == {"move"}
    b.replan()
    assert {t.name for t in b.stack.due(51)} == {"target", "move"}


def test_runner_one_forward_pass_per_tick_and_json_records():
    be = MockBackend()
    r = Runner(Dungeon(2), Engine(be), group_size=8, plan_budget=1)
    header = r.header()
    for _ in range(60):
        before = be.forward_calls
        rec = r.tick()
        assert be.forward_calls - before <= 1
        model = [x for x in rec["decisions"] if x["method"] != "only_option"]
        assert rec["batch"]["decisions"] == len(model)
        assert sum(1 for x in model if x["kind"] == "plan") <= 1
        assert sum(1 for x in rec["decisions"] if x["tier"] == "move") == 1
        if r.d.outcome:
            break
    assert not r.engine.order_debias  # restored after each tick
    end = r.end()
    lines = [json.loads(x) for x in to_jsonl([header, *r.records, end]).splitlines()]
    assert lines[0]["scenario"] == "dungeon" and lines[0]["version"] == 2 and lines[-1]["type"] == "end"
    assert "ghouls" in lines[1]["world"] and "idle" in lines[1]
    assert {"loop_ticks", "stuck_ticks", "dashes", "hits_after_safe_move"} <= set(end["summary"])


def test_order_averaging_default_comes_from_the_config():
    from demo.dungeon.capture import order_debias
    assert order_debias({}) is True
    assert order_debias({"shooter": {"order_debias": False}}) is False
    assert order_debias({"shooter": {"order_debias": False}, "dungeon": {"order_debias": True}}) is True
    assert order_debias({"dungeon": {"order_debias": True}}, flag=False) is False


def test_rules_are_frozen_as_calibrated():
    """Changing them invalidates bench/results/dungeon_calibration/ and every dungeon result."""
    r = Rules()
    assert (r.ghoul_move_every, r.ghoul_chase, r.ghoul_tired, r.ghoul_rest, r.ghoul_damage) == (3, 12, 10, 3, 20)
    assert (r.dash_cells, r.dash_recharge, r.max_ticks, r.min_doors, r.door_width) == (3, 8, 400, 2, 3)
    assert replace(r, dash=False).dash is False
