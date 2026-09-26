"""The single-agent dungeon demo: world rules, brain texts, runner and trace, replay page."""

import json

import pytest

from demo.dungeon.world import Dungeon, Rules


def test_layout_is_deterministic_connected_and_consistent():
    for seed in range(25):
        d, again = Dungeon(seed), Dungeon(seed)
        assert d.layout() == again.layout() and d.snapshot() == again.snapshot()
        assert set(d.room_distances(d.start_room)) == set(range(9))  # every room reachable
        assert len(d.corridors) == 9  # spanning tree (8) + one extra link
        reach = d.distances(d.pos)
        items = [d.exit, d.key, *d.gems, *d.food, *d.potions, *(e.pos for e in d.enemies)]
        assert all(c in reach for c in items)
        assert len(set(items + [d.pos])) == len(items) + 1  # nothing shares a cell
        assert all(d.room_at(c) is not None and not d.is_door(c) for c in items)
        assert d.exit_room != d.start_room and d.key_room not in (d.start_room, d.exit_room)
        assert d.enemies[0].home == d.key_room and all(e.home != d.start_room for e in d.enemies)
        assert d.seen == {d.start_room}


def test_rooms_open_only_through_their_doors():
    d = Dungeon(4)
    for room in d.rooms:
        border = set()
        for x, y in room.cells:
            for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if c not in room and d.passable(c):
                    border.add(c)
        assert border and all(d.is_door(c) and d.area[c] == room.id for c in border)


def quiet(seed=0, **rules):
    """A dungeon without enemies or supplies unless asked for, so single rules can be tested."""
    base = dict(enemies=0, gems=0, food=0, potions=0)
    base.update(rules)
    return Dungeon(seed, Rules(**base))


def walk_to(d, cell):
    """Teleport the agent next to ``cell`` (on a known open cell) and return the move onto it."""
    for move, (dx, dy) in (("move east", (-1, 0)), ("move west", (1, 0)), ("move south", (0, -1)), ("move north", (0, 1))):
        here = (cell[0] + dx, cell[1] + dy)
        if d.passable(here):
            d.pos = here
            return move
    raise AssertionError("no open neighbour")


def test_exit_is_locked_without_the_key_and_opens_with_it():
    d = quiet()
    move = walk_to(d, d.exit)
    ev = d.step(move)
    assert d.pos == d.exit and d.outcome is None and {"kind": "exit_locked"} in ev
    key = d.key
    move = walk_to(d, key)
    ev = d.step(move)
    assert d.has_key and d.key is None and any(e["kind"] == "key" for e in ev)
    move = walk_to(d, d.exit)
    ev = d.step(move)
    assert d.outcome == "escaped" and ev[-1] == {"kind": "escaped"}
    with pytest.raises(RuntimeError):
        d.step("stay")


def test_walls_block_and_pickups_apply_caps():
    d = quiet(gems=1, food=1, potions=1)
    wall_move = next(m for m, (dx, dy) in (("move north", (0, -1)), ("move south", (0, 1)), ("move east", (1, 0)),
                                           ("move west", (-1, 0))) if not d.passable((d.pos[0] + dx, d.pos[1] + dy)))
    before = d.pos
    assert d.step(wall_move)[0]["kind"] == "bump" and d.pos == before
    d.step(walk_to(d, d.gems[0]))
    assert d.collected == 1 and not d.gems
    d.energy = 90
    ev = d.step(walk_to(d, d.food[0]))
    assert d.energy == 99 and any(e["kind"] == "food" and e["gain"] == 10 for e in ev)  # capped at 100, then -1
    d.health = 50
    d.step(walk_to(d, d.potions[0]))
    assert d.health == 90


def test_starvation_kills():
    d = quiet(energy=2)
    outcomes = []
    for _ in range(40):
        d.step("stay")
        outcomes.append(d.outcome)
        if d.outcome:
            break
    assert d.outcome == "died" and d.cause == "starvation" and d.health == 0
    assert d.tick == 1 + 20  # 1 tick to reach 0 energy, then 20 ticks x 5 health


def test_tick_limit():
    d = quiet(max_ticks=3)
    for _ in range(3):
        d.step("stay")
    assert d.outcome == "timeout"


def test_enemy_chases_hits_and_is_stunned():
    d = Dungeon(0, Rules(enemies=1, gems=0, food=0, potions=0))
    e = d.enemies[0]
    room = d.rooms[e.home]
    d.pos = room.cells[0]
    e.pos = next(c for c in room.cells if d.distances(d.pos)[c] == 3)
    d.seen.add(e.home)
    d._known = None
    hits = []
    for _ in range(12):
        ev = d.step("stay")
        hits += [x for x in ev if x["kind"] == "hit"]
        if hits:
            break
    assert len(hits) == 1 and d.health == 100 - d.rules.enemy_damage
    assert e.mode == "stunned" and e.stunned == d.rules.stun_ticks
    assert d.distances(d.pos)[e.pos] == 1
    for _ in range(d.rules.stun_ticks):
        assert not any(x["kind"] == "hit" for x in d.step("stay"))
    assert d.visible_enemies()[0][0] is e


def test_walking_into_an_enemy_costs_a_hit():
    d = Dungeon(0, Rules(enemies=1, gems=0, food=0, potions=0))
    e = d.enemies[0]
    move = walk_to(d, e.pos)
    ev = d.step(move)
    assert any(x["kind"] == "hit" for x in ev) and d.pos != e.pos


def test_fog_of_war_reveals_rooms_through_their_doors():
    d = quiet(3)
    known = d.known_cells()
    start = d.rooms[d.start_room]
    assert set(start.cells) <= known
    other = next(r for r in d.rooms if r.id != d.start_room)
    assert not set(other.cells) & known
    cor = next(c for c in d.corridors if d.start_room in c.rooms)
    far_room = cor.rooms[1] if cor.rooms[0] == d.start_room else cor.rooms[0]
    far_door = cor.doors[cor.rooms.index(far_room)]
    assert far_door in known and not set(d.rooms[far_room].cells) & known
    d.pos = far_door  # standing in a doorway shows the room behind it
    ev = d.step("stay")
    assert far_room in d.seen and set(d.rooms[far_room].cells) <= d.known_cells()
    assert {"kind": "room_seen", "room": far_room, "name": d.rooms[far_room].name} in ev


def test_snapshot_and_layout_are_json():
    d = Dungeon(1)
    json.dumps(d.layout())
    json.dumps(d.snapshot())


# -- brain, runner, trace ----------------------------------------------------------------

from demo.dungeon.brain import GOALS, DungeonBrain, Runner, cell_of, drop_distance  # noqa: E402
from demo.dungeon.capture import build_replay, to_jsonl  # noqa: E402
from system_one import Engine  # noqa: E402
from system_one.backends.mock import MockBackend  # noqa: E402


def test_goals_offered_only_when_possible():
    d = Dungeon(0)
    b = DungeonBrain(d)
    first = b.goal_options()
    assert "explore" in first and "get the key" not in first and "go to the exit" not in first
    assert all(g in GOALS for g in first)
    d.seen = set(range(9))  # as if explored; in play, seen rooms are joined by known corridors
    d._known = None
    b.refresh()
    assert "get the key" in b.goal_options() and "explore" not in b.goal_options()
    assert b.target_options("get the key")[0].startswith(f"the key at ({d.key[0]},{d.key[1]}) in the ")
    d.key, d.has_key = None, True
    b.refresh()
    opts = b.goal_options()
    assert "get the key" not in opts and "go to the exit" in opts
    assert "The key: you carry it" in b.strategy_state()


def test_texts_spell_out_consequences():
    d = Dungeon(0)
    d.energy, d.health = 20, 30
    b = DungeonBrain(d)
    s = b.strategy_state()
    assert s.startswith("Standing order: Find the key")
    assert "about 20 ticks until starving" in s and "CRITICAL (30/100): 1 more enemy hit would kill you" in s
    assert "The key: not found yet. The exit stays locked until you carry it." in s


def test_move_options_are_labelled_with_outcomes():
    d = Dungeon(0)
    e = d.enemies[0]
    b = DungeonBrain(d, enemy_aware=True)
    x, y = d.pos
    free = [(m, c) for m, c in (("move north", (x, y - 1)), ("move south", (x, y + 1)), ("move east", (x + 1, y)),
                                ("move west", (x - 1, y))) if d.passable(c)]
    move, cell = free[0]
    e.pos = cell
    b.refresh()
    opts = b.move_options()
    assert f"{move} (ENEMY: you stay here and lose 30 health)" in opts
    assert any(o.endswith("(wall)") for o in opts) or len(free) == 4
    assert any("next to an enemy: it can hit you for 30 health" in o for o in opts if o.startswith("stay"))
    v1 = DungeonBrain(d)  # enemy_aware off: the wording of the pre-registered evaluation
    v1.refresh()
    assert f"{move} (ENEMY: -30 health)" in v1.move_options()
    assert len(opts) == 5 and [o.split(" (")[0] for o in opts] == ["move north", "move south", "move east", "move west", "stay"]


def test_target_text_for_lower_tiers_drops_the_stale_distance():
    assert drop_distance("gem at (5,6) in the hall, 4 steps away") == "gem at (5,6) in the hall"
    assert drop_distance("safe spot at (1,2) in a corridor, 3 steps away, 7 steps from the enemy") == \
        "safe spot at (1,2) in a corridor"


def test_runner_one_forward_pass_per_tick_and_json_records():
    be = MockBackend()
    r = Runner(Dungeon(2), Engine(be), group_size=8, plan_budget=1)
    header = r.header()
    for _ in range(60):
        before = be.forward_calls
        rec = r.tick()
        assert be.forward_calls - before == 1  # the action tier always needs the model
        model = [x for x in rec["decisions"] if x["method"] != "only_option"]
        assert rec["batch"]["decisions"] == len(model)
        assert sum(1 for x in model if x["kind"] == "plan") <= 1  # plan budget
        assert sum(1 for x in rec["decisions"] if x["tier"] == "action") == 1
        for x in rec["decisions"]:
            assert abs(sum(x["probs"]) - 1) < 1e-3 and 0 <= x["choice"] < len(x["options"])
        if r.d.outcome:
            break
    end = r.end()
    text = to_jsonl([header, *r.records, end])
    lines = [json.loads(x) for x in text.splitlines()]
    assert lines[0]["type"] == "header" and lines[-1]["type"] == "end" and len(lines) == len(r.records) + 2
    assert lines[0]["map"]["width"] == 24 and "Standing order" not in lines[0]["standing_order"]
    assert end["summary"]["ticks"] == len(r.records)


def test_only_option_goals_skip_the_model():
    d = Dungeon(0)
    d.seen = set(range(9))
    d._known = None
    r = Runner(d, Engine(MockBackend()), plan_budget=1)
    r.brain.stack.apply(r.brain.stack.tier("strategy"), "get the key", 0, 0.9)
    rec = r.tick()
    target = [x for x in rec["decisions"] if x["tier"] == "target"]
    assert target and target[0]["method"] == "only_option" and target[0]["options"][0].startswith("the key at")


def test_replay_embeds_the_trace_safely(tmp_path):
    template = tmp_path / "viewer.html"
    template.write_text('<p>x</p><script id="trace" type="application/x-ndjson"></script><script>1</script>',
                        encoding="utf-8")
    text = to_jsonl([{"type": "header", "note": "</script><b>"}])
    out = build_replay(text, tmp_path / "replay.html", template)
    html = out.read_text(encoding="utf-8")
    assert "</script><b>" not in html and html.count("</script>") == 2
    inner = html.split('type="application/x-ndjson">')[1].split("</script>")[0]
    assert json.loads(inner)["note"] == "</script><b>"
    build_replay(text, out, out)  # rebuilding an already-built replay replaces the old trace
    assert out.read_text(encoding="utf-8").count("application/x-ndjson") == 1


def test_move_labels_say_closer_or_farther():
    d = quiet(0)
    b = DungeonBrain(d)
    b.stack.apply(b.stack.tier("strategy"), "explore", 0)
    b.stack.apply(b.stack.tier("target"), b.target_options("explore")[0], 0)
    b.refresh()
    opts = [o for o in b.move_options() if "to the target" in o]
    assert any("(closer:" in o or "; closer:" in o for o in opts) and any("farther:" in o or "no closer" in o for o in opts)
    steps = DungeonBrain(d, label_style="steps")
    steps.stack.current.update(b.stack.current)
    steps.refresh()
    assert any("target: " in o for o in steps.move_options()) and not any("closer" in o for o in steps.move_options())
    with pytest.raises(ValueError):
        DungeonBrain(d, label_style="other")


def test_safe_spots_are_reached_before_the_enemy():
    d = quiet(0, enemies=1)
    e = d.enemies[0]
    room = d.rooms[d.start_room]
    e.home = d.start_room
    e.pos = next(c for c in room.cells if d.distances(d.pos)[c] == 2)
    b = DungeonBrain(d, enemy_aware=True)
    b.refresh()
    spots = b.safe_spots()
    assert spots
    from_enemy = d.distances(e.pos)
    for text in spots:
        c = cell_of(text)
        assert d.distances(d.pos)[c] < from_enemy[c] and "the enemy is" in text
